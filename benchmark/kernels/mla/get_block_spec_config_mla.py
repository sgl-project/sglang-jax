"""Auto-tuner for the MLA v2 block-config table.

Sweeps candidate (num_kv_pages_per_block, num_queries_per_block, decode_batch_size)
per case ("decode" / "mixed") and emits paste-friendly entries for
``python/sgl_jax/srt/kernels/mla/v2/tuned_block_sizes.py``.

The MLA v2 kernel runs three pallas_calls but only uses two block-size slots:

  - slot[0] is shared by BATCHED_DECODE (batch_size=decode_batch_size,
    static_q_len=1) and DECODE-tail (batch_size=1, static_q_len=1)
  - slot[2] is MIXED (batch_size=1, static_q_len=None)
  - slot[1] (PREFILL) is currently dead code

So we tune two independent buckets: ``"decode"`` (slot[0] + decode_batch_size)
and ``"mixed"`` (slot[2]).

Usage:
    # Full default grid (Ling-1T shape: num_q_heads={8,16}, ps={128,256},
    # mnt buckets matching server precompile for tp=32 dp=4
    # max-prefill-tokens=2048 max-running-requests=512 moe-backend=fused).
    python benchmark/kernels/mla/get_block_spec_config_mla.py

    # Narrow to the user's 16k-input + chunked-prefill=2048 + 1k-decode case:
    python benchmark/kernels/mla/get_block_spec_config_mla.py \\
        --num-q-heads 8 --page-sizes 256 --kv-len 16384 \\
        --decode-mnt 16,32,64,128 --mixed-mnt 512

For multi-worker dispatch (FALCON_RANK aware), use --shard auto,N.
"""

from __future__ import annotations

import argparse
import functools
import os
from math import inf

import jax
import jax.numpy as jnp
import numpy as np
from utils import create_mla_decode_uniform_data, create_mla_mixed_uniform_data

from sgl_jax.srt.kernels.mla.v2.kernel import mla_ragged_paged_attention
from sgl_jax.srt.kernels.utils.perf import multiple_iteration_timeit_from_trace
from sgl_jax.srt.utils.common_utils import next_power_of_2
from sgl_jax.srt.utils.jax_utils import get_device_name

# -----------------------------------------------------------------------------
# Defaults (mirror the post-bucket-derivation outer-grid in the plan)
# -----------------------------------------------------------------------------

# Ling-1T launch params: tp=32 dp=4, moe-backend=fused, max-prefill-tokens=2048,
# chunked-prefill-size=2048, max-running-requests=512 → server jits at:
#   global EXTEND token_buckets = [512, 1024, 2048, 4096, 8192]
#     → per-shard mnt = [128, 256, 512, 1024, 2048]
#   global DECODE bs_buckets = [64, 128, 256, 512]
#     → per-shard mnt = [16, 32, 64, 128]
_DEFAULT_DECODE_MNT = (16, 32, 64, 128)
_DEFAULT_MIXED_MNT = (128, 256, 512, 1024, 2048)

# Ling-1T (64 total heads). attention_tp=8 → per-shard 8; attention_tp=4 → 16.
_DEFAULT_NUM_Q_HEADS = (8, 16)

# Production page_size=256; 128 included for comparison only.
_DEFAULT_PAGE_SIZES = (128, 256)

# Inner search space.
_BKV_P_CANDIDATES = (1, 2, 3, 4, 6, 8, 16, 32)
# BATCHED_DECODE and DECODE-tail both have static_q_len=1, so kernel.py:1486
# clamps bq_sz = min(num_queries_per_block, 1) = 1 regardless of what we pass.
# Sweeping bq for decode just spawns extra jit cache entries with identical
# kernels — wasteful and produces noise. Pin to [1].
_BQ_DECODE_CANDIDATES = (1,)
_BQ_MIXED_CANDIDATES = (1, 4, 8, 16, 32, 64, 128, 256)
_DBS_CANDIDATES = (1, 2, 4, 8, 16, 32)

# Hardcoded defaults to compare against (matches historical mla_backend.py
# values, also matches kernel.py:1411 fallback when a tuned entry misses).
_HEURISTIC_DECODE = (3, 1, 4)  # (bkv_p, bq, dbs)
_HEURISTIC_MIXED = (1, 16)  # (bkv_p, bq)


# -----------------------------------------------------------------------------
# Bench primitives
# -----------------------------------------------------------------------------


def _make_jitted_attn(case_label: str):
    """JIT wrapper around `mla_ragged_paged_attention`. Static argnames cover
    every knob we sweep so each candidate triggers a fresh compile."""

    # NB: do NOT add donate_argnames at this outer wrapper — the inner
    # mla_ragged_paged_attention already donates cache_kv. Adding a second
    # donate boundary deletes the captured Python ref between bench iters,
    # producing "Array has been deleted with shape=..." errors.
    @functools.partial(
        jax.jit,
        static_argnames=[
            "sm_scale",
            "num_kv_pages_per_block",
            "num_queries_per_block",
            "decode_batch_size",
            "vmem_limit_bytes",
        ],
    )
    def attn(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
        sm_scale,
        num_kv_pages_per_block,
        num_queries_per_block,
        decode_batch_size,
        vmem_limit_bytes,
    ):
        return mla_ragged_paged_attention(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            distribution,
            sm_scale=sm_scale,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
            decode_batch_size=decode_batch_size,
            vmem_limit_bytes=vmem_limit_bytes,
        )

    return attn


def _bench_one(
    case_label: str,
    inputs: dict,
    sm_scale: float,
    num_kv_pages_per_block: tuple,
    num_queries_per_block: tuple,
    decode_batch_size: int,
    vmem_limit_bytes: int,
    tries: int,
    scope: str,
) -> float:
    """Compile + warmup + measure mean latency for one config.

    Returns mean ms. Raises on compile/runtime error so caller can `try`.
    """
    attn = _make_jitted_attn(case_label)
    bound = functools.partial(
        attn,
        inputs["ql_nope"],
        inputs["q_pe"],
        inputs["new_kv_c"],
        inputs["new_k_pe"],
        inputs["cache_kv"],
        inputs["kv_lens"],
        inputs["page_indices"],
        inputs["cu_q_lens"],
        inputs["cu_kv_lens"],
        inputs["distribution"],
        sm_scale,
        num_kv_pages_per_block,
        num_queries_per_block,
        decode_batch_size,
        vmem_limit_bytes,
    )

    # Warmup (compile). If this raises (XLA MSA / VMEM overflow / etc.) the
    # caller catches and skips the candidate.
    out = bound()
    jax.block_until_ready(out)

    times = multiple_iteration_timeit_from_trace(
        compute_func=lambda: bound(),
        data_generator=lambda: (),
        task=scope,
        tries=tries,
    )
    return float(np.mean(times)) if times else float("nan")


# -----------------------------------------------------------------------------
# Sweep helpers
# -----------------------------------------------------------------------------


def _enum_decode_candidates(max_q_per_block: int):
    out = []
    for bkv_p in _BKV_P_CANDIDATES:
        for bq in _BQ_DECODE_CANDIDATES:
            if bq > max_q_per_block:
                continue
            for dbs in _DBS_CANDIDATES:
                # When dbs > mnt, the BATCHED_DECODE pallas_call has empty
                # grid (kernel.py:1632 batch_distribution = floor(N/dbs)*dbs
                # = 0 for dbs>N) and all work falls to DECODE-tail. Our
                # bench scope matches `MLA-bd-...` events, so a degenerate
                # dbs>mnt would extract a no-op event time. Skip.
                if dbs > max_q_per_block:
                    continue
                out.append((bkv_p, bq, dbs))
    return out


def _enum_mixed_candidates(max_q_per_block: int):
    out = []
    for bkv_p in _BKV_P_CANDIDATES:
        for bq in _BQ_MIXED_CANDIDATES:
            if bq > max_q_per_block:
                continue
            out.append((bkv_p, bq))
    return out


def _sweep_decode(
    *,
    max_num_tokens: int,
    num_q_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    page_size: int,
    kv_len: int,
    vmem_limit_bytes: int,
    tries: int,
    dtype,
):
    """Returns (best_config, best_t_ms, heur_config, heur_t_ms, n_attempted, n_failed)."""
    inputs = create_mla_decode_uniform_data(
        max_num_tokens=max_num_tokens,
        num_q_heads=num_q_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        page_size=page_size,
        kv_len=kv_len,
        dtype=dtype,
    )
    sm_scale = (kv_lora_rank + qk_rope_head_dim) ** -0.5

    candidates = _enum_decode_candidates(max_q_per_block=max_num_tokens)
    if _HEURISTIC_DECODE not in candidates:
        candidates = [_HEURISTIC_DECODE] + candidates

    best_t = inf
    best = None
    heur_t = inf
    n_failed = 0
    for i, (bkv_p, bq, dbs) in enumerate(candidates):
        # slot[1] (PREFILL) and slot[2] (MIXED) are unused for decode-only
        # benchmarks — fill placeholders.
        nkv = (bkv_p, 1, 1)
        nq = (bq, 1, 1)
        # IMPORTANT: scope must match the kernel's pallas_call name (set by
        # kernel.py:1582) so multiple_iteration_timeit_from_trace's regex
        # extractor pulls the actual `device_duration_ps` from the BATCHED_
        # DECODE pallas event. Otherwise it falls back to MARKER-wall-time,
        # which conflates 3 pallas_calls + jit dispatch overhead and is
        # noisy.
        # Kernel scope for BATCHED_DECODE (clamps bq_sz=1, batch_size=dbs):
        #   "MLA-bd-bq_1-bkvp_{bkv_p}-p_{page_size}-bsz_{dbs}"
        # We've already filtered dbs <= mnt so BATCHED_DECODE handles all
        # work and DECODE-tail is empty.
        scope = f"MLA-bd-bq_1-bkvp_{bkv_p}-p_{page_size}-bsz_{dbs}"
        try:
            t_ms = _bench_one(
                "decode",
                inputs,
                sm_scale,
                nkv,
                nq,
                dbs,
                vmem_limit_bytes,
                tries,
                scope,
            )
        except Exception as e:  # noqa: BLE001
            tag = f"# [{i + 1}/{len(candidates)}] decode mnt={max_num_tokens} bkv_p={bkv_p} bq={bq} dbs={dbs} FAIL: {type(e).__name__}: {e}"
            print(tag, flush=True)
            if (bkv_p, bq, dbs) == _HEURISTIC_DECODE:
                print(
                    f"# heur-FAILURE decode mnt={max_num_tokens} h={num_q_heads} "
                    f"ps={page_size}: {type(e).__name__}: {e}",
                    flush=True,
                )
            n_failed += 1
            continue
        print(
            f"# [{i + 1}/{len(candidates)}] decode mnt={max_num_tokens} "
            f"bkv_p={bkv_p} bq={bq} dbs={dbs} t={t_ms * 1000:.4f}ms",
            flush=True,
        )
        if (bkv_p, bq, dbs) == _HEURISTIC_DECODE:
            heur_t = t_ms
        if t_ms < best_t:
            best_t = t_ms
            best = (bkv_p, bq, dbs)
    return best, best_t, _HEURISTIC_DECODE, heur_t, len(candidates), n_failed


def _sweep_mixed(
    *,
    max_num_tokens: int,
    num_q_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    page_size: int,
    kv_len: int,
    vmem_limit_bytes: int,
    tries: int,
    dtype,
):
    """Returns (best_config, best_t_ms, heur_config, heur_t_ms, n_attempted, n_failed)."""
    inputs = create_mla_mixed_uniform_data(
        max_num_tokens=max_num_tokens,
        num_q_heads=num_q_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        page_size=page_size,
        kv_len=max(kv_len, max_num_tokens),
        dtype=dtype,
    )
    sm_scale = (kv_lora_rank + qk_rope_head_dim) ** -0.5

    candidates = _enum_mixed_candidates(max_q_per_block=max_num_tokens)
    if _HEURISTIC_MIXED not in candidates:
        candidates = [_HEURISTIC_MIXED] + candidates

    best_t = inf
    best = None
    heur_t = inf
    n_failed = 0
    # decode_batch_size is irrelevant for the MIXED branch (its grid is
    # empty for distribution=[0,0,N]) — pin to 1 to keep jit cache small.
    dbs_for_mixed = 1
    for i, (bkv_p, bq) in enumerate(candidates):
        # slot[0] (decode) and slot[1] (prefill) are unused — placeholders.
        nkv = (1, 1, bkv_p)
        nq = (1, 1, bq)
        # Scope matches the kernel's MIXED pallas_call name. MIXED has
        # static_q_len=None so bq_sz = num_queries_per_block[2] = our `bq`
        # (no clamping). batch_size=1 always for MIXED.
        scope = f"MLA-m-bq_{bq}-bkvp_{bkv_p}-p_{page_size}-bsz_1"
        try:
            t_ms = _bench_one(
                "mixed",
                inputs,
                sm_scale,
                nkv,
                nq,
                dbs_for_mixed,
                vmem_limit_bytes,
                tries,
                scope,
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"# [{i + 1}/{len(candidates)}] mixed mnt={max_num_tokens} "
                f"bkv_p={bkv_p} bq={bq} FAIL: {type(e).__name__}: {e}",
                flush=True,
            )
            if (bkv_p, bq) == _HEURISTIC_MIXED:
                print(
                    f"# heur-FAILURE mixed mnt={max_num_tokens} h={num_q_heads} "
                    f"ps={page_size}: {type(e).__name__}: {e}",
                    flush=True,
                )
            n_failed += 1
            continue
        print(
            f"# [{i + 1}/{len(candidates)}] mixed mnt={max_num_tokens} "
            f"bkv_p={bkv_p} bq={bq} t={t_ms * 1000:.4f}ms",
            flush=True,
        )
        if (bkv_p, bq) == _HEURISTIC_MIXED:
            heur_t = t_ms
        if t_ms < best_t:
            best_t = t_ms
            best = (bkv_p, bq)
    return best, best_t, _HEURISTIC_MIXED, heur_t, len(candidates), n_failed


# -----------------------------------------------------------------------------
# CLI plumbing
# -----------------------------------------------------------------------------


def _csv_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_shard(s: str) -> tuple[int, int]:
    if not s:
        return (0, 1)
    a, b = s.split(",")
    total = int(b)
    if a == "auto":
        rank = int(os.environ.get("FALCON_RANK", os.environ.get("FALCON_JAX_PROCESS_ID", "0")))
    else:
        rank = int(a)
    if not (0 <= rank < total):
        raise SystemExit(f"--shard rank={rank} out of [0,{total})")
    return rank, total


def _table_key(
    case_label: str,
    q_dtype_name: str,
    kv_dtype_name: str,
    num_q_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    page_size: int,
    max_num_tokens: int,
):
    """Match the normalization done by tuned_block_sizes.get_tuned_block_sizes_mla."""
    return (
        case_label,
        q_dtype_name,
        kv_dtype_name,
        next_power_of_2(num_q_heads),
        int(kv_lora_rank),
        int(qk_rope_head_dim),
        next_power_of_2(page_size),
        next_power_of_2(max_num_tokens),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        default="decode,mixed",
        help="comma-separated subset of decode/mixed",
    )
    parser.add_argument("--tries", type=int, default=5)
    parser.add_argument(
        "--num-q-heads",
        default="",
        help=f"comma list, default {','.join(map(str, _DEFAULT_NUM_Q_HEADS))}",
    )
    parser.add_argument(
        "--page-sizes",
        default="",
        help=f"comma list, default {','.join(map(str, _DEFAULT_PAGE_SIZES))}",
    )
    parser.add_argument("--kv-lora-rank", type=int, default=512)
    parser.add_argument("--qk-rope-head-dim", type=int, default=64)
    parser.add_argument(
        "--decode-mnt",
        default="",
        help=f"comma list per-shard mnt for decode case, default {','.join(map(str, _DEFAULT_DECODE_MNT))}",
    )
    parser.add_argument(
        "--mixed-mnt",
        default="",
        help=f"comma list per-shard mnt for mixed case, default {','.join(map(str, _DEFAULT_MIXED_MNT))}",
    )
    parser.add_argument(
        "--kv-len",
        type=int,
        default=16384,
        help="actual seq KV length used to size the cache (Ling 16k input case)",
    )
    parser.add_argument(
        "--vmem-limit-bytes",
        type=int,
        default=None,
        help="kernel vmem_limit_bytes; default = 90%% of hardware VMEM capacity",
    )
    parser.add_argument(
        "--shard",
        default="",
        help="'RANK,TOTAL' or 'auto,TOTAL' — slice outer grid across workers",
    )
    parser.add_argument(
        "--write-threshold-pct",
        type=float,
        default=10.0,
        help="only emit a table entry if tuned beats heuristic by ≥ this %",
    )
    args = parser.parse_args()

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    for c in cases:
        if c not in ("decode", "mixed"):
            raise SystemExit(f"unknown case {c!r}; expected 'decode' or 'mixed'")

    num_q_heads_list = (
        _csv_ints(args.num_q_heads) if args.num_q_heads else list(_DEFAULT_NUM_Q_HEADS)
    )
    page_sizes = _csv_ints(args.page_sizes) if args.page_sizes else list(_DEFAULT_PAGE_SIZES)
    decode_mnt_list = _csv_ints(args.decode_mnt) if args.decode_mnt else list(_DEFAULT_DECODE_MNT)
    mixed_mnt_list = _csv_ints(args.mixed_mnt) if args.mixed_mnt else list(_DEFAULT_MIXED_MNT)

    device = get_device_name()
    shard_rank, shard_total = _parse_shard(args.shard)
    print(f"# Device: {device}")
    print(f"# {jax.devices()}")
    print(
        f"# cases={cases} num_q_heads={num_q_heads_list} page_sizes={page_sizes} "
        f"kv_len={args.kv_len} decode_mnt={decode_mnt_list} mixed_mnt={mixed_mnt_list}"
    )
    print(f"# shard={shard_rank}/{shard_total}")
    print()

    # Build outer grid (case, num_q_heads, page_size, mnt).
    outer = []
    for case in cases:
        mnt_list = decode_mnt_list if case == "decode" else mixed_mnt_list
        for num_q_heads in num_q_heads_list:
            for page_size in page_sizes:
                for mnt in mnt_list:
                    outer.append((case, num_q_heads, page_size, mnt))
    my_work = outer[shard_rank::shard_total]
    print(f"# outer-grid total={len(outer)} mine={len(my_work)}")
    print()

    if args.vmem_limit_bytes is None:
        from jax.experimental.pallas import tpu as pltpu

        args.vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
    print(f"# vmem_limit_bytes={args.vmem_limit_bytes} ({args.vmem_limit_bytes / (1<<20):.1f} MiB)")

    dtype = jnp.bfloat16
    q_dtype_name = jnp.dtype(dtype).name

    rows = []
    for case, num_q_heads, page_size, mnt in my_work:
        if case == "decode":
            best, best_t, heur, heur_t, n_attempted, n_failed = _sweep_decode(
                max_num_tokens=mnt,
                num_q_heads=num_q_heads,
                kv_lora_rank=args.kv_lora_rank,
                qk_rope_head_dim=args.qk_rope_head_dim,
                page_size=page_size,
                kv_len=args.kv_len,
                vmem_limit_bytes=args.vmem_limit_bytes,
                tries=args.tries,
                dtype=dtype,
            )
        else:
            best, best_t, heur, heur_t, n_attempted, n_failed = _sweep_mixed(
                max_num_tokens=mnt,
                num_q_heads=num_q_heads,
                kv_lora_rank=args.kv_lora_rank,
                qk_rope_head_dim=args.qk_rope_head_dim,
                page_size=page_size,
                kv_len=args.kv_len,
                vmem_limit_bytes=args.vmem_limit_bytes,
                tries=args.tries,
                dtype=dtype,
            )
        if best is None or heur_t == inf:
            print(
                f"# DROP case={case} h={num_q_heads} ps={page_size} mnt={mnt}: "
                f"best={best} heur_t={heur_t} attempted={n_attempted} failed={n_failed}"
            )
            continue
        delta_pct = (heur_t - best_t) / heur_t * 100.0
        key = _table_key(
            case,
            q_dtype_name,
            q_dtype_name,
            num_q_heads,
            args.kv_lora_rank,
            args.qk_rope_head_dim,
            page_size,
            mnt,
        )
        rows.append((key, best, best_t, heur, heur_t, delta_pct, n_attempted, n_failed))
        win = "WIN " if delta_pct >= args.write_threshold_pct else "skip"
        print(
            f"# [{win}] {key}: heur={heur} {heur_t * 1000:.4f}ms "
            f"best={best} {best_t * 1000:.4f}ms Δ={delta_pct:+.1f}% "
            f"(tried {n_attempted}, failed {n_failed})"
        )

    print()
    print(
        f"# --- Paste into TUNED_BLOCK_SIZES_MLA[{device!r}] (≥{args.write_threshold_pct}% win only) ---"
    )
    for key, best, _, _, _, delta_pct, _, _ in rows:
        if delta_pct >= args.write_threshold_pct:
            print(f"        {key}: {best},")
    print()
    print("# --- All measured (for audit) ---")
    for key, best, best_t, heur, heur_t, delta_pct, n_attempted, n_failed in rows:
        print(
            f"# {key}: best={best} ({best_t * 1000:.4f}ms) "
            f"heur={heur} ({heur_t * 1000:.4f}ms) Δ={delta_pct:+.1f}% "
            f"(tried {n_attempted}, failed {n_failed})"
        )


if __name__ == "__main__":
    main()
