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

    # GLM-5.2 DSA DP16 C32, 128K hit + 1K extend + 1K decode. Falcon only
    # needs one v7x-8 slice (4 chips / 8 devices); attention_tp=1 means one
    # selected device is representative of every independent DP rank.
    python benchmark/kernels/mla/get_block_spec_config_mla.py \\
        --scenario glm52-dp16-128k --device-index 0

For multi-worker dispatch (FALCON_RANK aware), use --shard auto,N.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
from math import inf
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

try:
    from .utils import create_mla_decode_uniform_data, create_mla_mixed_uniform_data
except ImportError:
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

# GLM-5.2 production mapping:
#   Falcon serving allocation: 2 replicas * 8 devices = DP16
#   attention_tp = global TP16 / DP16 = 1
#   global decode BS32 / DP16 = 2 tokens per rank
#   global extend bucket 32768 / DP16 = 2048 tokens per rank
#   each rank owns 2 requests, so extend is 2 seqs * 1024 tokens
#
# A tuner run only needs Falcon's minimum v7x-8 allocation. MLA contains no
# collectives for attention_tp=1, so one selected device reproduces the local
# Pallas shape; the other seven devices do not change the tuned key.
_SCENARIOS = {
    "glm52-dp16-128k": {
        "num_q_heads": (64,),
        "page_sizes": (64,),
        "decode_mnt": (2,),
        "mixed_mnt": (2048,),
        # Extend sees the 128K cached prefix plus the 1K extension.
        "mixed_kv_len": 131072 + 1024,
        # Decode is tuned at the largest KV length reached by the 1K decode.
        "decode_kv_len": 131072 + 1024 + 1024,
        "mixed_num_seqs": 2,
    }
}

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

    Returns mean milliseconds. Raises on compile/runtime error so caller can
    ``try``.
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
                # Keep a single active Pallas call per measurement. All
                # production buckets are powers of two, so this does not
                # remove a useful production candidate and avoids having to
                # add BATCHED_DECODE and DECODE-tail event durations.
                if max_q_per_block % dbs != 0:
                    continue
                out.append((bkv_p, bq, dbs))
    return out


def _decode_trace_scope(
    *, max_num_tokens: int, bkv_p: int, page_size: int, decode_batch_size: int
) -> str:
    """Return the active decode Pallas scope for a candidate.

    The historical fallback uses ``decode_batch_size=4``. For the GLM-5.2
    DP16 local decode bucket, ``max_num_tokens=2``; therefore the batched grid
    is empty and all work runs in the ``MLA-d`` tail. The old tuner always
    selected ``MLA-bd`` and could benchmark a no-op event for this shape.

    Candidate decode batch sizes are constrained to divisors of the token
    bucket, so exactly one of BATCHED_DECODE or DECODE-tail is active.
    """
    batched_tokens = (max_num_tokens // decode_batch_size) * decode_batch_size
    tail_tokens = max_num_tokens - batched_tokens
    if batched_tokens == 0:
        return f"MLA-d-bq_1-bkvp_{bkv_p}-p_{page_size}-bsz_1"
    if tail_tokens == 0:
        return f"MLA-bd-bq_1-bkvp_{bkv_p}-p_{page_size}-bsz_{decode_batch_size}"
    raise ValueError(
        "decode tuner requires decode_batch_size to divide max_num_tokens "
        f"or exceed it, got mnt={max_num_tokens}, dbs={decode_batch_size}"
    )


def _enum_mixed_candidates(max_q_per_block: int):
    out = []
    for bkv_p in _BKV_P_CANDIDATES:
        for bq in _BQ_MIXED_CANDIDATES:
            if bq > max_q_per_block:
                continue
            out.append((bkv_p, bq))
    return out


def _candidate_failure_label(error: Exception) -> str:
    """Summarize an expected candidate rejection without dumping megabytes.

    TPU compile-time VMEM exhaustion is a normal outcome while sweeping block
    sizes, not a failed benchmark. Keeping it as a short ``SKIP_VMEM`` line
    also prevents generic artifact analyzers from treating the expected
    rejection as a workload exception.
    """
    message = str(error)
    message_lower = message.lower()
    if "resource_exhausted" in message_lower and (
        "vmem" in message_lower or "out of memory" in message_lower
    ):
        return "SKIP_VMEM"
    first_line = message.splitlines()[0] if message else "no error message"
    return f"FAIL: {type(error).__name__}: {first_line}"


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
        # Match the Pallas call that actually performs the work. This is
        # MLA-bd for divisible candidate batch sizes, but MLA-d for the dbs=4
        # fallback when the local GLM decode bucket is only two tokens.
        scope = _decode_trace_scope(
            max_num_tokens=max_num_tokens,
            bkv_p=bkv_p,
            page_size=page_size,
            decode_batch_size=dbs,
        )
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
            failure_label = _candidate_failure_label(e)
            tag = f"# [{i + 1}/{len(candidates)}] decode mnt={max_num_tokens} bkv_p={bkv_p} bq={bq} dbs={dbs} {failure_label}"
            print(tag, flush=True)
            if (bkv_p, bq, dbs) == _HEURISTIC_DECODE:
                print(
                    f"# heur-FAILURE decode mnt={max_num_tokens} h={num_q_heads} "
                    f"ps={page_size}: {failure_label}",
                    flush=True,
                )
            n_failed += 1
            continue
        print(
            f"# [{i + 1}/{len(candidates)}] decode mnt={max_num_tokens} "
            f"bkv_p={bkv_p} bq={bq} dbs={dbs} t={t_ms:.4f}ms",
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
    num_seqs: int,
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
        num_seqs=num_seqs,
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
            failure_label = _candidate_failure_label(e)
            print(
                f"# [{i + 1}/{len(candidates)}] mixed mnt={max_num_tokens} "
                f"bkv_p={bkv_p} bq={bq} {failure_label}",
                flush=True,
            )
            if (bkv_p, bq) == _HEURISTIC_MIXED:
                print(
                    f"# heur-FAILURE mixed mnt={max_num_tokens} h={num_q_heads} "
                    f"ps={page_size}: {failure_label}",
                    flush=True,
                )
            n_failed += 1
            continue
        print(
            f"# [{i + 1}/{len(candidates)}] mixed mnt={max_num_tokens} "
            f"bkv_p={bkv_p} bq={bq} t={t_ms:.4f}ms",
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
        "--scenario",
        choices=sorted(_SCENARIOS),
        default=None,
        help="reviewed workload preset; explicit shape flags override it",
    )
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
        default=None,
        help="common KV length for both cases; case-specific flags override it",
    )
    parser.add_argument("--decode-kv-len", type=int, default=None)
    parser.add_argument("--mixed-kv-len", type=int, default=None)
    parser.add_argument(
        "--mixed-num-seqs",
        type=int,
        default=None,
        help="number of uniform local extend sequences (GLM DP16 C32 uses 2)",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="local JAX device used for the single-device kernel sweep",
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
        help="only emit a table entry if tuned beats heuristic by ≥ this %%",
    )
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="optional path for one best-vs-heuristic metrics row per shape",
    )
    args = parser.parse_args()
    scenario = _SCENARIOS.get(args.scenario, {})

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    for c in cases:
        if c not in ("decode", "mixed"):
            raise SystemExit(f"unknown case {c!r}; expected 'decode' or 'mixed'")

    num_q_heads_list = (
        _csv_ints(args.num_q_heads)
        if args.num_q_heads
        else list(scenario.get("num_q_heads", _DEFAULT_NUM_Q_HEADS))
    )
    page_sizes = (
        _csv_ints(args.page_sizes)
        if args.page_sizes
        else list(scenario.get("page_sizes", _DEFAULT_PAGE_SIZES))
    )
    decode_mnt_list = (
        _csv_ints(args.decode_mnt)
        if args.decode_mnt
        else list(scenario.get("decode_mnt", _DEFAULT_DECODE_MNT))
    )
    mixed_mnt_list = (
        _csv_ints(args.mixed_mnt)
        if args.mixed_mnt
        else list(scenario.get("mixed_mnt", _DEFAULT_MIXED_MNT))
    )
    common_kv_len = args.kv_len
    decode_kv_len = (
        args.decode_kv_len
        if args.decode_kv_len is not None
        else (
            common_kv_len
            if common_kv_len is not None
            else int(scenario.get("decode_kv_len", 16384))
        )
    )
    mixed_kv_len = (
        args.mixed_kv_len
        if args.mixed_kv_len is not None
        else (
            common_kv_len if common_kv_len is not None else int(scenario.get("mixed_kv_len", 16384))
        )
    )
    mixed_num_seqs = (
        args.mixed_num_seqs
        if args.mixed_num_seqs is not None
        else int(scenario.get("mixed_num_seqs", 1))
    )

    local_devices = jax.local_devices()
    if not 0 <= args.device_index < len(local_devices):
        raise SystemExit(
            f"--device-index={args.device_index} out of range for "
            f"{len(local_devices)} local devices"
        )
    selected_device = local_devices[args.device_index]

    device = get_device_name()
    shard_rank, shard_total = _parse_shard(args.shard)
    print(f"# Device: {device}")
    print(f"# allocated local devices ({len(local_devices)}): {local_devices}")
    print(f"# selected local device: index={args.device_index} {selected_device}")
    if args.scenario == "glm52-dp16-128k":
        print("# Falcon allocation: v7x-8 = 4 chips / 8 devices / topology 2x2x1")
        print("# Production mapping: 2 replicas * 8 devices = DP16; attention_tp=1")
    print(
        f"# cases={cases} num_q_heads={num_q_heads_list} page_sizes={page_sizes} "
        f"decode_kv_len={decode_kv_len} mixed_kv_len={mixed_kv_len} "
        f"decode_mnt={decode_mnt_list} mixed_mnt={mixed_mnt_list} "
        f"mixed_num_seqs={mixed_num_seqs}"
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
    print(
        f"# vmem_limit_bytes={args.vmem_limit_bytes} ({args.vmem_limit_bytes / (1 << 20):.1f} MiB)"
    )

    dtype = jnp.bfloat16
    q_dtype_name = jnp.dtype(dtype).name

    rows = []
    with jax.default_device(selected_device):
        for case, num_q_heads, page_size, mnt in my_work:
            case_kv_len = decode_kv_len if case == "decode" else mixed_kv_len
            if case == "decode":
                best, best_t, heur, heur_t, n_attempted, n_failed = _sweep_decode(
                    max_num_tokens=mnt,
                    num_q_heads=num_q_heads,
                    kv_lora_rank=args.kv_lora_rank,
                    qk_rope_head_dim=args.qk_rope_head_dim,
                    page_size=page_size,
                    kv_len=case_kv_len,
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
                    kv_len=case_kv_len,
                    num_seqs=mixed_num_seqs,
                    vmem_limit_bytes=args.vmem_limit_bytes,
                    tries=args.tries,
                    dtype=dtype,
                )
            if best is None or heur_t == inf:
                print(
                    f"# DROP case={case} h={num_q_heads} ps={page_size} "
                    f"mnt={mnt} kv_len={case_kv_len}: best={best} "
                    f"heur_t={heur_t} attempted={n_attempted} failed={n_failed}"
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
            rows.append(
                (
                    key,
                    best,
                    best_t,
                    heur,
                    heur_t,
                    delta_pct,
                    n_attempted,
                    n_failed,
                    case_kv_len,
                )
            )
            win = "WIN " if delta_pct >= args.write_threshold_pct else "skip"
            print(
                f"# [{win}] {key}: kv_len={case_kv_len} "
                f"heur={heur} {heur_t:.4f}ms "
                f"best={best} {best_t:.4f}ms Δ={delta_pct:+.1f}% "
                f"(tried {n_attempted}, failed {n_failed})"
            )

    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_file:
            for (
                key,
                best,
                best_t,
                heur,
                heur_t,
                delta_pct,
                n_attempted,
                n_failed,
                case_kv_len,
            ) in rows:
                case_label = key[0]
                metric = {
                    "variant": "tuned",
                    "scenario": args.scenario or "custom",
                    "case": case_label,
                    "q_dtype": key[1],
                    "kv_dtype": key[2],
                    "num_q_heads": key[3],
                    "kv_lora_rank": key[4],
                    "qk_rope_head_dim": key[5],
                    "page_size": key[6],
                    "max_num_tokens": key[7],
                    "kv_len": case_kv_len,
                    "num_seqs": mixed_num_seqs if case_label == "mixed" else key[7],
                    "heuristic_config": list(heur),
                    "best_config": list(best),
                    "heuristic_latency_ms": heur_t,
                    "latency_ms": best_t,
                    "speedup_pct": delta_pct,
                    "attempted_configs": n_attempted,
                    "failed_configs": n_failed,
                    "write_threshold_pct": args.write_threshold_pct,
                    "table_entry_selected": delta_pct >= args.write_threshold_pct,
                }
                output_file.write(json.dumps(metric, sort_keys=True) + "\n")
        print(f"# wrote metrics: {output_path}")

    print()
    print(
        f"# --- Paste into TUNED_BLOCK_SIZES_MLA[{device!r}] (≥{args.write_threshold_pct}% win only) ---"
    )
    for key, best, _, _, _, delta_pct, _, _, _ in rows:
        if delta_pct >= args.write_threshold_pct:
            print(f"        {key}: {best},")
    print()
    print("# --- All measured (for audit) ---")
    for (
        key,
        best,
        best_t,
        heur,
        heur_t,
        delta_pct,
        n_attempted,
        n_failed,
        case_kv_len,
    ) in rows:
        print(
            f"# {key}: best={best} ({best_t:.4f}ms) "
            f"heur={heur} ({heur_t:.4f}ms) Δ={delta_pct:+.1f}% "
            f"kv_len={case_kv_len} (tried {n_attempted}, failed {n_failed})"
        )


if __name__ == "__main__":
    main()
