"""Auto-tuner for RPA v3 block-config table.

Sweeps candidate (bq_sz, bkv_sz, bq_csz, bkv_csz) tuples per stage and emits
Python-literal entries to paste into
python/sgl_jax/srt/kernels/ragged_paged_attention/tuned_block_sizes_v3.py.

Usage:
    python benchmark/kernels/flash_attention/get_block_spec_config_v3.py
    python benchmark/kernels/flash_attention/get_block_spec_config_v3.py --stages d
    python benchmark/kernels/flash_attention/get_block_spec_config_v3.py --quick

Output (stdout) is grouped by device and ready to drop into the
TUNED_BLOCK_SIZES_V3 dict.
"""

import argparse
import functools
import itertools
from math import inf

import jax
import jax.numpy as jnp
import numpy as np
from utils import create_decode_uniform_data, create_prefill_uniform_data

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3 import (
    RpaCase,
    get_vmem_limit,
    ragged_paged_attention,
)
from sgl_jax.srt.kernels.ragged_paged_attention.util import get_dtype_packing
from sgl_jax.srt.kernels.utils.perf import multiple_iteration_timeit_from_trace
from sgl_jax.srt.utils.jax_utils import get_device_name


def _bq_candidates(max_q: int, stage: str) -> list[int]:
    if stage == "d":
        return [1]
    cap = min(32, max_q)
    return [b for b in (1, 2, 4, 8, 16, 32) if b <= cap]


def _bkv_candidates(page_size: int, kv_packing: int, max_kv: int) -> list[int]:
    alignment = max(page_size, kv_packing)
    raw = [256, 512, 1024, 2048, 4096, 8192, 16384]
    out = []
    for v in raw:
        v = max(alignment, (v // alignment) * alignment)
        v = min(v, max_kv)
        if v not in out:
            out.append(v)
    return out


def _csz_candidates(sz: int, alignment: int = 1) -> list[int]:
    if sz <= 1:
        return [1]
    out = set()
    v = sz
    while v >= max(alignment, 1):
        out.add(v)
        if v == 1:
            break
        v //= 2
    out.add(max(alignment, 1))
    out = {c for c in out if sz % c == 0}
    return sorted(out)


def _enumerate_block_sizes(
    stage: str,
    max_q: int,
    max_kv: int,
    page_size: int,
    kv_packing: int,
) -> list[tuple[int, int, int, int]]:
    bkv_align = max(page_size, kv_packing)
    out = []
    for bq in _bq_candidates(max_q, stage):
        for bkv in _bkv_candidates(page_size, kv_packing, max_kv):
            for bq_csz in _csz_candidates(bq, alignment=1):
                for bkv_csz in _csz_candidates(bkv, alignment=bkv_align):
                    out.append((bq, bkv, bq_csz, bkv_csz))
    return out


def _make_inputs(
    stage: str,
    page_size: int,
    max_kv_cache_tokens: int,
    max_num_tokens: int,
    q_head_num: int,
    kv_head_num: int,
    head_dim: int,
    max_context_len: int,
):
    """Returns (kwargs_for_kernel, rpa_case, chunk_prefill_size)."""
    if stage == "d":
        (q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, cu_kv_lens, _, _, _, distribution) = (
            create_decode_uniform_data(
                max_context_len,
                max_kv_cache_tokens,
                max_num_tokens,
                q_head_num,
                kv_head_num,
                head_dim,
                page_size=page_size,
            )
        )
        rpa_case = RpaCase.DECODE
        chunk_prefill_size = None
    elif stage == "m":
        (q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, cu_kv_lens, _, _, _, distribution) = (
            create_prefill_uniform_data(
                max_context_len,
                max_kv_cache_tokens,
                max_num_tokens,
                q_head_num,
                kv_head_num,
                head_dim,
                page_size=page_size,
            )
        )
        rpa_case = RpaCase.MIXED
        chunk_prefill_size = None
    elif stage == "p":
        (q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, cu_kv_lens, _, _, _, distribution) = (
            create_prefill_uniform_data(
                max_context_len,
                max_kv_cache_tokens,
                max_num_tokens,
                q_head_num,
                kv_head_num,
                head_dim,
                page_size=page_size,
            )
        )
        # Force all sequences into PREFILL bucket: distribution = [0, N, N]
        batch_size = int(distribution[-1])
        distribution = jnp.array([0, batch_size, batch_size], dtype=jnp.int32)
        rpa_case = RpaCase.PREFILL
        chunk_prefill_size = max_num_tokens
    else:
        raise ValueError(f"unknown stage {stage!r}")
    return (
        dict(
            q=q,
            k=k,
            v=v,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            cu_kv_lens=cu_kv_lens,
            distribution=distribution,
        ),
        rpa_case,
        chunk_prefill_size,
    )


def _benchmark_one(
    stage,
    page_size,
    max_kv_cache_tokens,
    max_num_tokens,
    q_head_num,
    kv_head_num,
    head_dim,
    block_sizes,
    max_context_len,
    tries: int,
):
    inputs, rpa_case, chunk_prefill_size = _make_inputs(
        stage,
        page_size,
        max_kv_cache_tokens,
        max_num_tokens,
        q_head_num,
        kv_head_num,
        head_dim,
        max_context_len,
    )
    scale = head_dim**-0.5

    block_kwarg_name = {"d": "d_block_sizes", "p": "p_block_sizes", "m": "m_block_sizes"}[stage]

    @functools.partial(
        jax.jit,
        static_argnames=["sm_scale", "chunk_prefill_size", block_kwarg_name, "vmem_limit_bytes"],
    )
    def attn(
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
        sm_scale,
        **block_kwargs,
    ):
        return ragged_paged_attention(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            distribution,
            custom_mask=None,
            causal=1,
            sm_scale=sm_scale,
            chunk_prefill_size=chunk_prefill_size,
            vmem_limit_bytes=get_vmem_limit(),
            **block_kwargs,
        )

    block_kwargs = {block_kwarg_name: block_sizes}

    bound = functools.partial(
        attn,
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["kv_cache"],
        inputs["kv_lens"],
        inputs["page_indices"],
        inputs["cu_q_lens"],
        inputs["cu_kv_lens"],
        inputs["distribution"],
        scale,
        **block_kwargs,
    )

    # Warmup (compile).
    out = bound()
    jax.block_until_ready(out)

    scope = (
        f"RPA{rpa_case.symbol}-p_{page_size}"
        f"-bq_{block_sizes[0]}_{block_sizes[2]}"
        f"-bkv_{block_sizes[1]}_{block_sizes[3]}"
    )
    times = multiple_iteration_timeit_from_trace(
        compute_func=lambda: bound(),
        data_generator=lambda: (),
        task=scope,
        tries=tries,
    )
    return float(np.mean(times)) if times else float("nan")


def sweep(
    stage: str,
    page_size: int,
    q_head_num: int,
    kv_head_num: int,
    head_dim: int,
    max_num_tokens: int,
    max_context_len: int,
    max_kv_cache_tokens: int,
    dtype=jnp.bfloat16,
    tries: int = 1,
):
    kv_packing = get_dtype_packing(dtype)
    max_kv = max_context_len
    candidates = _enumerate_block_sizes(stage, max_num_tokens, max_kv, page_size, kv_packing)

    best_time = inf
    best = None
    for bs in candidates:
        try:
            t = _benchmark_one(
                stage,
                page_size,
                max_kv_cache_tokens,
                max_num_tokens,
                q_head_num,
                kv_head_num,
                head_dim,
                bs,
                max_context_len,
                tries,
            )
        except Exception:  # noqa: BLE001
            continue
        if t < best_time:
            best_time = t
            best = bs
    return best, best_time


def _grid(args):
    page_sizes = [128, 256]
    head_dims = [128]
    head_combos = [
        (1, 1),
        (2, 1),
        (2, 2),
        (4, 1),
        (4, 2),
        (4, 4),
        (8, 1),
        (8, 2),
        (8, 4),
        (8, 8),
        (16, 1),
        (16, 2),
        (16, 4),
        (16, 8),
        (16, 16),
        (32, 1),
        (32, 2),
        (32, 4),
        (32, 8),
        (32, 16),
        (32, 32),
    ]
    if args.quick:
        page_sizes = [256]
        head_combos = [(32, 1)]
    decode_mnt = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    prefill_mnt = [512, 1024, 2048, 4096, 8192]
    if args.quick:
        decode_mnt = [64]
        prefill_mnt = [2048]
    return page_sizes, head_dims, head_combos, decode_mnt, prefill_mnt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", default="d,p,m", help="comma-separated subset of d/p/m to tune")
    parser.add_argument("--tries", type=int, default=1)
    parser.add_argument("--quick", action="store_true", help="smoke-test grid (1 combo per stage)")
    parser.add_argument("--max-context-len", type=int, default=40960)
    parser.add_argument("--max-kv-cache-tokens", type=int, default=600000)
    args = parser.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in stages:
        if s not in ("d", "p", "m"):
            raise SystemExit(f"unknown stage {s!r}")

    page_sizes, head_dims, head_combos, decode_mnt, prefill_mnt = _grid(args)
    device = get_device_name()
    print(f"# Device: {device}")
    print(f"# {jax.devices()}")
    print()

    rows = []
    for stage in stages:
        mnt_list = decode_mnt if stage == "d" else prefill_mnt
        for ps, hd in itertools.product(page_sizes, head_dims):
            for q_h, kv_h in head_combos:
                for mnt in mnt_list:
                    try:
                        best, best_t = sweep(
                            stage,
                            ps,
                            q_h,
                            kv_h,
                            hd,
                            mnt,
                            args.max_context_len,
                            args.max_kv_cache_tokens,
                            tries=args.tries,
                        )
                    except Exception as e:  # noqa: BLE001
                        print(
                            f"# SKIP stage={stage} ps={ps} q={q_h} kv={kv_h} "
                            f"hd={hd} mnt={mnt}: {e}"
                        )
                        continue
                    if best is None:
                        continue
                    key = (stage, "bfloat16", "bfloat16", q_h, kv_h, hd, ps, mnt)
                    rows.append((key, best, best_t))
                    print(f"# best for {key}: {best} ({best_t*1000:.4f} ms)")

    print()
    print(f"# --- Paste into TUNED_BLOCK_SIZES_V3[{device!r}] ---")
    for key, best, _ in rows:
        print(f"    {key}: {best},")


if __name__ == "__main__":
    main()
