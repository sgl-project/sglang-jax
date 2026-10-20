"""Experimental exact two-u32-limb Pallas hash; not wired into the scheduler.

Includes JAX context/EOS construction and reports completed ids-to-CPU time.
Host gather and embedding H2D are intentionally excluded from BOTH variants.

    # Released Qwen4Exp hash, CPU vs TPU+ids D2H, decode B=256.
    python benchmark/kernels/ngram/bench_ngram_hash_limb.py --tokens 256 --batch 256
    # Released Qwen4Exp hash, CPU vs TPU+ids D2H, prefill T=8192/B=4.
    python benchmark/kernels/ngram/bench_ngram_hash_limb.py --tokens 8192 --batch 4
"""

import argparse
import functools
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.layers.ngram_embedding import build_hash_params, compute_ngram_ids


def _mul_words(token, multiplier):
    """Exact low/high words of u32 * a constant u64, modulo 2**64."""
    token = token.astype(jnp.uint32)
    a, b = token & 65535, token >> 16
    c, d = jnp.uint32(multiplier & 65535), jnp.uint32((multiplier >> 16) & 65535)
    w0 = a * c
    t = b * c + (w0 >> 16)
    w1 = a * d + (t & 65535)
    lo = (w1 << 16) | (w0 & 65535)
    hi = b * d + (t >> 16) + (w1 >> 16) + token * jnp.uint32(multiplier >> 32)
    return lo, hi


def _remainder(n, p):
    """Exact for 0 <= n < 128*p, 0 < p < 2**25.

    FP32 estimates the quotient with error < 1; two integer corrections
    handle rounding at a multiple of p. Products fit u32 under this bound.
    """
    q = (n.astype(jnp.float32) / p.astype(jnp.float32)).astype(jnp.uint32)
    product = q * p
    r = n - product
    r = jnp.where(n < product, r + p, r)
    return jnp.where(r >= p, r - p, r)


def _hash_kernel(tokens_ref, sizes_ref, offsets_ref, out_ref, *, multipliers, heads):
    lo, hi = _mul_words(tokens_ref[0, :], multipliers[0])
    lows, highs = [], []
    for shift in range(1, len(multipliers)):
        next_lo, next_hi = _mul_words(tokens_ref[shift, :], multipliers[shift])
        lo, hi = lo ^ next_lo, hi ^ next_hi
        lows.append(jnp.broadcast_to(lo[None, :], (heads, lo.shape[0])))
        highs.append(jnp.broadcast_to(hi[None, :], (heads, hi.shape[0])))
    lo, hi = jnp.concatenate(lows, axis=0), jnp.concatenate(highs, axis=0)
    p = sizes_ref[...]
    residue = jnp.zeros_like(lo)
    # Two 32-bit words, each consumed as 4 + 7 + 7 + 7 + 7 bits.
    for word in (hi, lo):
        residue = _remainder((residue << 4) | (word >> 28), p)
        for shift in (21, 14, 7, 0):
            residue = _remainder((residue << 7) | ((word >> shift) & 127), p)
    out_ref[...] = (residue + offsets_ref[...]).astype(jnp.int32)


def make_hash(params, *, decode, tile=128):
    if max(params.sizes) >= 2**25 or params.total_vocab_size >= 2**31:
        raise ValueError("Experimental limb hash requires primes < 2**25 and total rows < 2**31")
    sizes = np.asarray(params.sizes[:, None], np.uint32)
    offsets = np.asarray(params.offsets[:, None], np.uint32)

    def compute(tokens, cu, context):
        count = tokens.shape[0]
        position = jnp.arange(count)
        request = position if decode else jnp.searchsorted(cu, position, side="right") - 1
        chunk_pos = jnp.zeros_like(position) if decode else position - cu[request]
        history = [tokens]
        crossed = jnp.zeros(count, bool)
        for shift in range(1, params.ngram_size):
            if decode:
                previous = context[:, params.ngram_size - 1 - shift]
            else:
                col = jnp.clip(params.ngram_size - 1 - shift + chunk_pos, 0, params.ngram_size - 2)
                previous = jnp.where(
                    chunk_pos >= shift,
                    tokens[jnp.maximum(position - shift, 0)],
                    context[request, col],
                )
            previous = jnp.where(crossed, params.eos_token_id, previous)
            crossed |= previous == params.eos_token_id
            history.append(previous)
        packed = jnp.pad(jnp.stack(history), ((0, 0), (0, pl.cdiv(count, tile) * tile - count)))
        ids = pl.pallas_call(
            functools.partial(
                _hash_kernel,
                multipliers=tuple(map(int, params.multipliers)),
                heads=params.heads_per_ngram,
            ),
            grid=(pl.cdiv(count, tile),),
            in_specs=(
                pl.BlockSpec((params.ngram_size, tile), lambda b: (0, b)),
                pl.BlockSpec(sizes.shape, lambda b: (0, 0)),
                pl.BlockSpec(offsets.shape, lambda b: (0, 0)),
            ),
            out_specs=pl.BlockSpec((params.ngram_heads, tile), lambda b: (0, b)),
            out_shape=jax.ShapeDtypeStruct((params.ngram_heads, packed.shape[1]), jnp.int32),
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
            name="ngram_hash_two_u32_limbs",
        )(packed, jnp.asarray(sizes), jnp.asarray(offsets))
        return ids[:, :count].T

    return jax.jit(compute)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--reps", type=int, default=40)
    args = parser.parse_args()
    params = build_hash_params(
        ngram_size=3,
        heads_per_ngram=8,
        vocab_size=248320,
        ngram_vocab_size_base=20_000_000,
        eos_token_id=2,
    )
    fn = make_hash(params, decode=args.tokens == args.batch)
    rng = np.random.default_rng(3)
    lengths = np.full(args.batch, args.tokens // args.batch, np.int32)
    lengths[: args.tokens % args.batch] += 1
    cu = np.r_[np.int32(0), np.cumsum(lengths, dtype=np.int32)]
    samples = []
    for _ in range(args.reps + 8):
        tokens = rng.integers(0, 248320, args.tokens, np.int32)
        context = rng.integers(0, 248320, (args.batch, 2), np.int32)
        tokens[::17] = params.eos_token_id
        context[::3, 1] = params.eos_token_id
        tokens[-1] = 248319
        samples.append((tokens, cu, context))
    device_samples = [jax.device_put(sample) for sample in samples]
    jax.block_until_ready(device_samples)
    # Parity on every independently generated sample (including EOS barriers).
    for host, device in zip(samples, device_samples, strict=True):
        np.testing.assert_array_equal(np.asarray(fn(*device)), compute_ngram_ids(*host, params))
    print(
        f"parity: {len(samples)} samples, {args.tokens} tokens/sample, all {params.ngram_heads} heads exact",
        flush=True,
    )
    for label in (
        "cpu_numpy",
        "tpu_ready_inputs",
        "tpu_plus_ids_d2h",
        "input_h2d_tpu_ids_d2h",
    ):
        timings = []
        for i, (host, device) in enumerate(zip(samples, device_samples, strict=True)):
            start = time.perf_counter()
            if label == "cpu_numpy":
                compute_ngram_ids(*host, params)
            elif label == "tpu_ready_inputs":
                jax.block_until_ready(fn(*device))
            elif label == "tpu_plus_ids_d2h":
                np.asarray(fn(*device))
            else:
                np.asarray(fn(*jax.device_put(host)))
            if i >= 8:
                timings.append((time.perf_counter() - start) * 1000)
        print(f"{label}: median_ms={statistics.median(timings):.4f}", flush=True)


if __name__ == "__main__":
    main()
