"""Full single-device CSA latency, including compressor and both cache writes.

Run test/srt/kernels/csa first. Rotate four independent inputs, restoring caches
outside device-module timing so every invocation sees the same initial state.
"""

import argparse
import itertools
import json

import jax
import ml_dtypes
import numpy as np

from benchmark.kernels.csa_compressor.benchmark_compressor import make_inputs, measure
from sgl_jax.srt.kernels.csa import CSACache, csa_attention
from sgl_jax.srt.kernels.csa_attention.tune import get_csa_attention_schedule
from sgl_jax.srt.kernels.csa_compressor.tune import get_compressor_schedule
from sgl_jax.srt.layers.attention.csa_metadata import prepare_csa_metadata


def inputs(pattern, batch, sequence, hidden, heads, seed):
    values, old_meta, _, lengths = make_inputs(pattern, batch, sequence, hidden, seed)
    values = jax.tree.map(np.asarray, values)
    positions = np.asarray(old_meta.positions)
    prefixes = positions[np.asarray(old_meta.cu_q_lens)[:-1]]
    rng = np.random.default_rng(seed)
    tokens = values[0].shape[0]

    def bf16(shape):
        return rng.normal(0, 0.3, shape).astype(ml_dtypes.bfloat16)

    nope, rope, index = [v.copy() for v in values[10:]]
    for target, width, scales, record_bytes in ((nope, 448, 7, 512), (index, 128, 1, 256)):
        rows = target.reshape(-1, record_bytes)
        rows[:, :width] = (
            rng.normal(0, 0.3, (len(rows), width)).astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
        )
        rows[:, width : width + scales] = 127
    bits = bf16((rope.size // 128, 64)).view(np.uint16)
    rope.reshape(-1, 128)[:] = np.concatenate(
        ((bits >> 8).astype(np.uint8), (bits & 255).astype(np.uint8)), axis=-1
    )
    page_count = nope.shape[0]
    cp = rng.permutation(np.arange(1, page_count, dtype=np.int32)).reshape(batch, -1)
    wp = rng.permutation(np.arange(1, batch + 1, dtype=np.int32)).reshape(batch, 1)
    meta = prepare_csa_metadata(
        lengths,
        prefixes,
        np.arange(batch),
        wp,
        cp,
        num_tokens=tokens,
        window_size=128,
        window_page_size=128,
        compressed_page_size=128,
    )
    operands = (
        *values[:8],
        bf16((tokens, 64, 128)),
        rng.uniform(0, 0.1, (tokens, 64)).astype(np.float32),
        bf16((tokens, heads, 512)),
        bf16((tokens, 512)),
        np.zeros(heads, np.float32),
    )
    cache = CSACache(*values[8:10], nope, rope, index, bf16((batch + 1, 64, 2, 512)))
    return operands, cache, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pattern",
        choices=("decode_emit", "decode_update", "prefill", "ragged"),
        default="decode_emit",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=64)
    args = parser.parse_args()
    if min(args.batch, args.sequence, args.hidden, args.heads) <= 0:
        parser.error("shapes must be positive")
    buffers = [inputs(**vars(args), seed=i) for i in range(4)]
    rotation = itertools.cycle(buffers)
    device = jax.devices()[0].device_kind
    decode = args.pattern.startswith("decode")
    options = dict(
        compressor_schedule=get_compressor_schedule(args.hidden, device_kind=device),
        attention_schedule=get_csa_attention_schedule(device, decode=decode),
        scale=512**-0.5,
        window_size=128,
        top_k=512,
        kv_pages_per_block=1,
        queries_per_block=1 if decode else 32,
    )

    def call():
        operands, cache, meta = next(rotation)
        # Independent host-backed buffers keep cross-program prefetch out of timing.
        operands, cache = jax.tree.map(lambda v: jax.device_put(v.copy()), (operands, cache))
        jax.block_until_ready((operands, cache))
        return csa_attention(*operands, cache, meta, **options)

    for _ in buffers:
        output, _ = jax.block_until_ready(call())
        if not np.isfinite(np.asarray(output, np.float32)).all():
            raise AssertionError("nonfinite output before timing")
    result = measure(call)
    print(
        json.dumps(
            {
                **vars(args),
                "device": device,
                "jax": jax.__version__,
                "tokens": buffers[0][0][0].shape[0],
                "actual_context": int(np.asarray(buffers[0][2].attention.seq_lens).max()),
                "warmup": 3,
                "timed": 10,
                "buffers": 4,
                **result,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
