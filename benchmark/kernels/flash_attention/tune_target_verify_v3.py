"""Tune RPA v3 MIXED blocks for batched speculative target verification.

This intentionally has a separate entry point from the generic MIXED tuner:
``mnt=128`` there means one 128-token prefill, while the production workload
here is 32 sequences x 4 draft tokens, each attending to a long KV prefix.

Example matching the MiMo-V2-Pro 16K-input target-verify HLO::

    python benchmark/kernels/flash_attention/tune_target_verify_v3.py \
      --batch-size 32 --draft-token-num 4 --prefix-len 16384 \
      --page-size 256 --page-indices-capacity 32768 \
      --max-kv-cache-tokens 1655296 --q-head-num 16 \
      --kv-head-num 1 --head-dim 256 --tries 5
"""

import argparse
import functools
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from utils import create_target_verify_uniform_data, create_tree_mask

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3 import (
    get_vmem_estimate_bytes,
    get_vmem_limit,
    ragged_paged_attention,
)
from sgl_jax.srt.kernels.ragged_paged_attention.util import cdiv
from sgl_jax.srt.kernels.utils.perf import multiple_iteration_timeit_from_trace
from sgl_jax.srt.utils.jax_utils import get_device_name

_DEFAULT_CANDIDATES = (
    (32, 256, 32, 256),  # current production table hit
    (1, 1024, 1, 512),
    (2, 1024, 2, 512),
    (4, 1024, 4, 512),
    (8, 1024, 8, 512),
    (4, 1024, 4, 1024),
    (1, 2048, 1, 512),
    (2, 2048, 2, 512),
    (4, 2048, 4, 512),
    (8, 2048, 8, 512),
    (4, 2048, 4, 1024),
    (4, 2048, 4, 2048),
    (4, 4096, 4, 1024),
)


def _focused_candidates(
    draft_token_num: int,
    sliding_window: int | None,
) -> list[tuple[int, int, int, int]]:
    q = draft_token_num
    q_lo = max(1, q // 2)
    q_hi = min(32, q * 2)
    if sliding_window is not None:
        return [
            (32, 256, 32, 256),
            (q_lo, 256, q_lo, 256),
            (q, 256, q, 256),
            (q_hi, 256, q_hi, 256),
            (q, 512, q, 256),
            (q, 512, q, 512),
            (q, 1024, q, 256),
            (q, 1024, q, 512),
            (q, 1024, q, 1024),
        ]
    return [
        (32, 256, 32, 256),
        (q_lo, 2048, q_lo, 2048),
        (q, 1024, q, 1024),
        (q, 2048, q, 1024),
        (q, 2048, q, 2048),
        (q_hi, 2048, q_hi, 2048),
        (q, 4096, q, 1024),
    ]


def _parse_candidate(value: str) -> tuple[int, int, int, int]:
    parts = tuple(int(item) for item in value.split(","))
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("candidate must be bq_sz,bkv_sz,bq_csz,bkv_csz")
    return parts


def _benchmark_one(
    inputs,
    block_sizes,
    head_dim: int,
    tries: int,
    sliding_window: int | None,
) -> float:
    use_mask = inputs.get("custom_mask") is not None

    @functools.partial(
        jax.jit,
        static_argnames=("sm_scale", "m_block_sizes"),
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
        custom_mask,
        sm_scale,
        m_block_sizes,
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
            custom_mask=custom_mask,
            causal=0 if use_mask else 1,
            mask_aligned_to_cu_kv=use_mask,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            chunk_prefill_size=None,
            m_block_sizes=m_block_sizes,
            vmem_limit_bytes=get_vmem_limit(),
        )

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
        inputs.get("custom_mask"),
        head_dim**-0.5,
        block_sizes,
    )
    out = bound()
    jax.block_until_ready(out)
    # This string is used as a REGEX over trace event names to pull
    # device_duration_ps, so it must match the pallas_call name built in
    # ragged_paged_attention_v3.py (`scope_name`) exactly. On a miss the
    # extractor silently falls back to host-side MARKER events, which are not
    # comparable -- two runs would then be timed by two different methods.
    # Which mode ran is already recorded in the header line and the JSONL, so
    # do not encode it here.
    scope = (
        f"RPAm-p_{inputs['page_size']}"
        f"-bq_{block_sizes[0]}_{block_sizes[2]}"
        f"-bkv_{block_sizes[1]}_{block_sizes[3]}"
    )
    times = multiple_iteration_timeit_from_trace(
        compute_func=bound,
        data_generator=lambda: (),
        task=scope,
        tries=tries,
    )
    return float(np.mean(times)) if times else math.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--draft-token-num", type=int, default=4)
    parser.add_argument("--prefix-len", type=int, default=16384)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--page-indices-capacity", type=int, default=32768)
    parser.add_argument("--max-kv-cache-tokens", type=int, default=1655296)
    parser.add_argument("--q-head-num", type=int, default=16)
    parser.add_argument("--kv-head-num", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--tries", type=int, default=5)
    parser.add_argument(
        "--custom-mask",
        action="store_true",
        help=(
            "run the tree-mask path (causal=0 + a rank-3 custom_mask) instead of "
            "causal verification. Run once with and once without to measure what "
            "the mask costs; nothing else in this repo exercises the masked path."
        ),
    )
    parser.add_argument(
        "--sliding-window",
        type=int,
        default=0,
        help="0 for full attention; positive values tune an SWA target-verify layer",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSONL path for Falcon operator-analysis",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        type=_parse_candidate,
        dest="candidates",
        help="repeatable bq_sz,bkv_sz,bq_csz,bkv_csz; defaults to focused grid",
    )
    parser.add_argument(
        "--grid",
        choices=("broad", "focused"),
        default="broad",
        help="broad reproduces the original q4 sweep; focused scales with q_len",
    )
    args = parser.parse_args()

    values = create_target_verify_uniform_data(
        batch_size=args.batch_size,
        draft_token_num=args.draft_token_num,
        prefix_len=args.prefix_len,
        page_indices_capacity=args.page_indices_capacity,
        max_kv_cache_tokens=args.max_kv_cache_tokens,
        q_head_num=args.q_head_num,
        kv_head_num=args.kv_head_num,
        head_dim=args.head_dim,
        page_size=args.page_size,
    )
    inputs = dict(
        zip(
            (
                "q",
                "k",
                "v",
                "kv_cache",
                "kv_lens",
                "page_indices",
                "cu_q_lens",
                "cu_kv_lens",
            ),
            values[:8],
            strict=True,
        )
    )
    inputs["distribution"] = values[-1]
    inputs["page_size"] = args.page_size
    _kv_len = args.prefix_len + args.draft_token_num
    inputs["custom_mask"] = (
        create_tree_mask(
            batch_size=args.batch_size,
            draft_token_num=args.draft_token_num,
            kv_len=_kv_len,
            aligned_kv_len=cdiv(_kv_len, args.page_size) * args.page_size,
        )
        if args.custom_mask
        else None
    )

    candidates = args.candidates or (
        _focused_candidates(args.draft_token_num, args.sliding_window or None)
        if args.grid == "focused"
        else list(_DEFAULT_CANDIDATES)
    )
    vmem_limit = get_vmem_limit()
    print(f"# device={get_device_name()} devices={jax.devices()}")
    print(
        "# workload="
        f"bs{args.batch_size}xq{args.draft_token_num} "
        f"prefix={args.prefix_len} page={args.page_size} "
        f"q_heads={args.q_head_num} kv_heads={args.kv_head_num} hd={args.head_dim} "
        f"sliding_window={args.sliding_window or None} "
        f"custom_mask={'on (causal=0)' if args.custom_mask else 'off (causal=1)'}"
    )
    if args.custom_mask:
        print(f"# custom_mask shape={inputs['custom_mask'].shape} dtype=int32")
    print(
        f"# shapes q={inputs['q'].shape} kv_cache={inputs['kv_cache'].shape} "
        f"page_indices={inputs['page_indices'].shape}"
    )

    rows = []
    for block_sizes in candidates:
        est = get_vmem_estimate_bytes(
            args.kv_head_num,
            args.q_head_num // args.kv_head_num,
            args.head_dim,
            block_sizes[0],
            block_sizes[1],
            jnp.bfloat16,
            jnp.bfloat16,
            use_custom_mask=inputs.get("custom_mask") is not None,
            bkv_csz=block_sizes[3],
        )
        if est > vmem_limit:
            print(
                f"# SKIP {block_sizes}: estimated VMEM {est / 2**20:.2f}MiB "
                f"> limit {vmem_limit / 2**20:.2f}MiB"
            )
            continue
        try:
            elapsed = _benchmark_one(
                inputs,
                block_sizes,
                args.head_dim,
                args.tries,
                args.sliding_window or None,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"# FAIL {block_sizes}: {type(exc).__name__}: {exc}")
            continue
        rows.append((elapsed, block_sizes, est))
        print(f"# MEASURED {block_sizes}: {elapsed:.4f}ms vmem={est / 2**20:.2f}MiB")

    if not rows:
        raise SystemExit("no candidate completed")
    rows.sort()
    best_time, best, _ = rows[0]
    baseline = next((time for time, bs, _ in rows if bs == (32, 256, 32, 256)), math.nan)
    speedup = baseline / best_time if math.isfinite(baseline) else math.nan
    print(f"RESULT best={best} time_ms={best_time:.4f}")
    print(f"RESULT baseline_ms={baseline:.4f} speedup={speedup:.3f}x")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as output:
            for elapsed, block_sizes, est in rows:
                output.write(
                    json.dumps(
                        {
                            "variant": "target_verify",
                            "batch_size": args.batch_size,
                            "draft_token_num": args.draft_token_num,
                            "prefix_len": args.prefix_len,
                            "page_size": args.page_size,
                            "q_head_num": args.q_head_num,
                            "kv_head_num": args.kv_head_num,
                            "head_dim": args.head_dim,
                            "sliding_window": args.sliding_window or None,
                            "custom_mask": bool(args.custom_mask),
                            "block_sizes": list(block_sizes),
                            "latency_ms": elapsed,
                            "estimated_vmem_mib": est / 2**20,
                            "is_baseline": block_sizes == (32, 256, 32, 256),
                            "is_best": block_sizes == best,
                        }
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
