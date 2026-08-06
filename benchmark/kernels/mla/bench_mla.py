"""MLA v2 kernel benchmark + CI regression test.

Mirrors `benchmark/kernels/flash_attention/bench_flashattention.py`:

  - ``full_benchmark()`` iterates Ling-1T-relevant shapes, prints latency for
    both the hardcoded-default config (matches kernel.py:1411 fallback) and
    the table-driven config (None → triggers tuned_block_sizes_mla lookup).
    Use this manually after a tuner run to confirm wins.

  - ``TestPerformance`` is a regression test with checked-in expected
    latencies (per device). Runs in CI via ``SGLANG_JAX_IS_IN_CI=true``.

Usage:
    SGLANG_JAX_IS_IN_CI=true python benchmark/kernels/mla/bench_mla.py   # CI test
    python benchmark/kernels/mla/bench_mla.py                            # full grid
    python benchmark/kernels/mla/bench_mla.py --scenario glm52-dp16-128k
"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from utils import create_mla_decode_uniform_data, create_mla_mixed_uniform_data

from sgl_jax.srt.kernels.mla.v2.kernel import mla_ragged_paged_attention
from sgl_jax.srt.kernels.utils.perf import multiple_iteration_timeit_from_trace
from sgl_jax.srt.utils.jax_utils import get_device_name

# Hardcoded-default config (matches kernel.py:1411 fallback when no tuned
# entry exists). Bench this to anchor expected baselines.
# Tuples are (num_kv_pages_per_block, num_queries_per_block).
_DEFAULT_DECODE_BLOCK = (3, 1)
_DEFAULT_DECODE_DBS = 4
_DEFAULT_MIXED_BLOCK = (1, 16)


def benchmark_backend(
    case: str,
    max_num_tokens: int,
    num_q_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    page_size: int,
    kv_len: int,
    *,
    use_lookup: bool,
    mixed_num_seqs: int = 1,
    tries: int = 3,
    dtype=jnp.bfloat16,
) -> float:
    """Run one bench config and return mean ms.

    Args:
        case: "decode" or "mixed".
        use_lookup: if True, pass None for block params so the kernel's
            tuned-table lookup runs (table or hardcoded fallback). If False,
            pass the explicit hardcoded defaults so the bench is independent
            of table state — used for stable CI baselines.
        mixed_num_seqs: number of uniform local sequences for mixed/extend.
    """
    if case == "decode":
        inputs = create_mla_decode_uniform_data(
            max_num_tokens=max_num_tokens,
            num_q_heads=num_q_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            page_size=page_size,
            kv_len=kv_len,
            dtype=dtype,
        )
    elif case == "mixed":
        inputs = create_mla_mixed_uniform_data(
            max_num_tokens=max_num_tokens,
            num_q_heads=num_q_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            page_size=page_size,
            kv_len=max(kv_len, max_num_tokens),
            num_seqs=mixed_num_seqs,
            dtype=dtype,
        )
    else:
        raise ValueError(f"unknown case {case!r}")

    sm_scale = (kv_lora_rank + qk_rope_head_dim) ** -0.5

    if use_lookup:
        nkv = None
        nq = None
        dbs = _DEFAULT_DECODE_DBS  # kernel will override if "decode" tuned hit
    else:
        if case == "decode":
            nkv = (_DEFAULT_DECODE_BLOCK[0], 1, 1)
            nq = (_DEFAULT_DECODE_BLOCK[1], 1, 1)
        else:
            nkv = (1, 1, _DEFAULT_MIXED_BLOCK[0])
            nq = (1, 1, _DEFAULT_MIXED_BLOCK[1])
        dbs = _DEFAULT_DECODE_DBS

    # Match production vmem budget: query actual hardware capacity (× 0.9
    # headroom), same as mla_backend.py does at construction time.
    from jax.experimental.pallas import tpu as pltpu

    vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)

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
        nkv,
        nq,
        dbs,
        vmem_limit_bytes,
    )

    # Warmup (compile).
    out = bound()
    jax.block_until_ready(out)

    scope = (
        f"MLA-{case}-h{num_q_heads}-p{page_size}-mnt{max_num_tokens}"
        f"-{'lookup' if use_lookup else 'default'}"
    )
    times = multiple_iteration_timeit_from_trace(
        compute_func=lambda: bound(),
        data_generator=lambda: (),
        task=scope,
        tries=tries,
    )
    return float(np.mean(times)) if times else float("nan")


# Production deploy: tp=32 dp=4 → attention_tp=8 → per-shard num_q_heads=8.
# kv_lora_rank=512, qk_rope_head_dim=64 (Ling-1T).
_LING_HEADS = 8
_LING_LKV = 512
_LING_R = 64


def _run_benchmark(
    *,
    scenario: str,
    cases: list[tuple[str, int, int, int, int]],
    num_q_heads: int,
) -> list[dict]:
    print(f"[MLA-V2] BENCHMARK RESULTS scenario={scenario}")
    print(f"# device={get_device_name()}, devices={jax.devices()}")
    print()

    rows = []
    for case, mnt, page_size, kv_len, num_seqs in cases:
        print(
            f"# case={case} num_q_heads={num_q_heads} page_size={page_size} "
            f"mnt={mnt} kv_len={kv_len} num_seqs={num_seqs}"
        )
        try:
            t_default = benchmark_backend(
                case=case,
                max_num_tokens=mnt,
                num_q_heads=num_q_heads,
                kv_lora_rank=_LING_LKV,
                qk_rope_head_dim=_LING_R,
                page_size=page_size,
                kv_len=kv_len,
                use_lookup=False,
                mixed_num_seqs=num_seqs,
            )
            t_lookup = benchmark_backend(
                case=case,
                max_num_tokens=mnt,
                num_q_heads=num_q_heads,
                kv_lora_rank=_LING_LKV,
                qk_rope_head_dim=_LING_R,
                page_size=page_size,
                kv_len=kv_len,
                use_lookup=True,
                mixed_num_seqs=num_seqs,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue
        delta = (t_default - t_lookup) / t_default * 100.0
        print(f"  default: {t_default:.4f}ms   lookup: {t_lookup:.4f}ms   Δ={delta:+.1f}%")
        rows.append(
            {
                "scenario": scenario,
                "case": case,
                "num_q_heads": num_q_heads,
                "page_size": page_size,
                "max_num_tokens": mnt,
                "kv_len": kv_len,
                "num_seqs": num_seqs,
                "heuristic_latency_ms": t_default,
                "latency_ms": t_lookup,
                "speedup_pct": delta,
            }
        )
    return rows


def full_benchmark() -> list[dict]:
    cases = [
        # (case, mnt, page_size, kv_len, num_seqs)
        # Decode bs_buckets / dp=4 = {16, 32, 64, 128}
        ("decode", 16, 256, 16384, 16),
        ("decode", 32, 256, 16384, 32),
        ("decode", 64, 256, 16384, 64),
        ("decode", 128, 256, 16384, 128),
        # Mixed token_buckets / dp=4 = {128, 256, 512, 1024, 2048}
        ("mixed", 128, 256, 16384, 1),
        ("mixed", 256, 256, 16384, 1),
        ("mixed", 512, 256, 16384, 1),  # ← user's hot prefill chunk
        ("mixed", 1024, 256, 16384, 1),
        ("mixed", 2048, 256, 16384, 1),
    ]
    return _run_benchmark(
        scenario="ling",
        cases=cases,
        num_q_heads=_LING_HEADS,
    )


def glm52_benchmark() -> list[dict]:
    return _run_benchmark(
        scenario="glm52-dp16-128k",
        cases=[
            ("decode", 2, 64, 133120, 2),
            ("mixed", 2048, 64, 132096, 2),
        ],
        num_q_heads=64,
    )


def _import_test_utils():
    """Lazy import — sgl_jax.test.test_utils pulls in heavy serving deps."""
    from sgl_jax.test.test_utils import CustomTestCase, is_in_ci

    return CustomTestCase, is_in_ci


CustomTestCase = None  # populated in __main__


class _TestPerformanceBase:
    """CI regression test for MLA v2 hardcoded-default latency.

    Baselines are measured at the hardcoded-default config (use_lookup=False)
    so the test is decoupled from changes to the tuned table.
    """

    def test_mla_kernel_performance(self, floating_threshold: float = 0.1):
        # (case, mnt, num_q_heads, page_size, kv_len) → expected ms at default config.
        # NOTE: baselines are placeholders — populate after first stable run on
        # the target device (e.g. v7x falcon pod). See bench_flashattention.py
        # TestPerformance for the same pattern.
        test_cases_for_different_devices: dict[str, dict[tuple, float]] = {
            "TPU v6e": {},
            "TPU v7": {},
        }

        device = get_device_name()
        if device not in test_cases_for_different_devices:
            self.skipTest(f"no MLA bench baselines for device {device}")
        test_cases = test_cases_for_different_devices[device]
        if not test_cases:
            self.skipTest(
                f"MLA bench baselines empty for {device}; populate after first "
                "stable run on this device."
            )

        for case_key, baseline in test_cases.items():
            (case, mnt, num_q_heads, page_size, kv_len) = case_key
            res = benchmark_backend(
                case=case,
                max_num_tokens=mnt,
                num_q_heads=num_q_heads,
                kv_lora_rank=_LING_LKV,
                qk_rope_head_dim=_LING_R,
                page_size=page_size,
                kv_len=kv_len,
                use_lookup=False,
            )
            expected = baseline * (1 + floating_threshold)
            print(f"{case_key}, res={res:.4}ms, expected={expected:.4}ms")
            self.assertLess(
                res,
                expected,
                f"MLA v2 kernel performance regression for {case_key}",
            )


if __name__ == "__main__":
    import os

    if os.environ.get("SGLANG_JAX_IS_IN_CI", "").lower() in ("1", "true", "yes"):
        # Defer the test_utils import — pulls in fastapi / pybase64 / etc.
        CustomTestCase, _is_in_ci = _import_test_utils()

        class TestPerformance(_TestPerformanceBase, CustomTestCase):
            pass

        print("Run MLA v2 Kernel Performance Test...")
        TestPerformance().test_mla_kernel_performance()
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--scenario", choices=("ling", "glm52-dp16-128k"), default="ling")
        parser.add_argument("--output-jsonl", default=None)
        args = parser.parse_args()

        print("Run MLA v2 Full Benchmark...")
        rows = glm52_benchmark() if args.scenario == "glm52-dp16-128k" else full_benchmark()
        if args.output_jsonl:
            output_path = Path(args.output_jsonl)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as output_file:
                for row in rows:
                    output_file.write(json.dumps(row, sort_keys=True) + "\n")
            print(f"# wrote metrics: {output_path}")
