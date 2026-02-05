"""
  Usage:
  1. For test benchmark in ci
  SGLANG_JAX_IS_IN_CI=true python benchmark/kernels/flash_attention/bench_flashattention.py
  2. For generic benchmark results
  python benchmark/kernels/flash_attention/bench_flashattention.py
"""

import functools

import jax
import numpy as np
from utils import create_decode_uniform_data, create_prefill_uniform_data

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention import (
    get_kernel_scope_name,
    ragged_paged_attention,
)
from sgl_jax.srt.kernels.ragged_paged_attention.tuned_block_sizes import (
    get_tuned_block_sizes,
)
from sgl_jax.srt.kernels.utils.perf import multiple_iteration_timeit_from_trace
from sgl_jax.srt.utils.jax_utils import get_device_name
from sgl_jax.test.test_utils import CustomTestCase, is_in_ci


def benchmark_backend(
    mode,
    max_context_len,
    max_kv_cache_tokens,
    max_num_batched_tokens,
    q_head_num,
    kv_head_num,
    head_dim,
    page_size,
):
    scale = head_dim**-0.5

    if mode == "prefill":
        (
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            _,
            _,
            _,
            distribution,
        ) = create_prefill_uniform_data(
            max_context_len,
            max_kv_cache_tokens,
            max_num_batched_tokens,
            q_head_num,
            kv_head_num,
            head_dim,
            page_size=page_size,
        )
    elif mode == "decode":
        (
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            _,
            _,
            _,
            distribution,
        ) = create_decode_uniform_data(
            max_context_len,
            max_kv_cache_tokens,
            max_num_batched_tokens,
            q_head_num,
            kv_head_num,
            head_dim,
            page_size=page_size,
        )
    else:
        raise ValueError(f"Invalid mode: {mode=}")

    @functools.partial(
        jax.jit,
        static_argnames=["sm_scale"],
    )
    def jitted_attn(
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
            decode_mode=0,
            sm_scale=sm_scale,
        )

    attn = functools.partial(
        jitted_attn,
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
        scale,
    )

    # Warmup
    output = attn()
    jax.block_until_ready(output)

    # Benchmark
    best_bkv_p, best_bq_sz = get_tuned_block_sizes(
        q.dtype,
        k.dtype,
        q_head_num,
        kv_head_num,
        head_dim,
        page_size,
        max_num_batched_tokens,
        page_indices.shape[0] // kv_lens.shape[0],
        True,
    )
    times = multiple_iteration_timeit_from_trace(
        compute_func=lambda: attn(),
        data_generator=lambda: (),
        task=get_kernel_scope_name(best_bq_sz, best_bkv_p, page_size),
        tries=1,
    )
    avg_time = float(np.mean(times)) if times else float("nan")

    # cal num_q_heads_per_blk, num_kv_heads_per_blk
    return avg_time


def full_benchmark():
    bench_modes = ["prefill", "decode"]
    page_size_config = [64, 128, 256]
    max_num_batched_tokens_config_for_decode = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
    ]
    max_num_batched_tokens_config_for_prefill = [
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
    ]
    q_head_num_config = [2, 4, 8, 16, 32, 64]
    kv_head_num_config = [2, 4, 8, 16, 32, 64]
    head_dim_config = [128]
    max_kv_cache_tokens_config = [600000]
    all_combinations = []
    config_of_modes = {}
    max_context_len = 40960
    for mode in bench_modes:
        for q_head_num in q_head_num_config:
            for kv_head_num in kv_head_num_config:
                for head_dim in head_dim_config:
                    for page_size in page_size_config:
                        for max_kv_cache_tokens in max_kv_cache_tokens_config:
                            if mode == "prefill":
                                max_num_batched_tokens_config = (
                                    max_num_batched_tokens_config_for_prefill
                                )
                            elif mode == "decode":
                                max_num_batched_tokens_config = (
                                    max_num_batched_tokens_config_for_decode
                                )

                            for max_num_batched_tokens in max_num_batched_tokens_config:
                                if q_head_num < kv_head_num or q_head_num % kv_head_num != 0:
                                    continue
                                all_combinations.append(
                                    (
                                        page_size,
                                        max_kv_cache_tokens,
                                        max_num_batched_tokens,
                                        q_head_num,
                                        kv_head_num,
                                        head_dim,
                                    )
                                )
        config_of_modes[mode] = all_combinations
        all_combinations = []

    for mode, configs in config_of_modes.items():
        print(f"[{mode.upper()}] BENCHMARK RESULTS SUMMARY")
        for _, (
            page_size,
            max_kv_cache_tokens,
            max_num_batched_tokens,
            q_head_num,
            kv_head_num,
            head_dim,
        ) in enumerate(configs):
            print(
                f"Config: q_head_num={q_head_num}, kv_head_num={kv_head_num}, head_dim={head_dim=}, page_size={page_size}, max_num_batched_tokens={max_num_batched_tokens}"
            )
            try:
                flash_time = benchmark_backend(
                    mode,
                    max_context_len,
                    max_kv_cache_tokens,
                    max_num_batched_tokens,
                    q_head_num,
                    kv_head_num,
                    head_dim,
                    page_size,
                )
            except Exception as e:
                raise ValueError(f"run failed: {e=}")

            print(f"cost: {flash_time * 1000}ms")


class TestPerformance(CustomTestCase):
    def test_ragged_paged_attention_performance(self, floating_threshold: int = 0.1):
        """
        Args:
            floating_threshold: the ratio of expected results
        """
        # Key: (mode, page_size, max_num_batched_tokens, q_head_num, kv_head_num, head_dim, max_kv_cache_tokens)
        # Value: expected cost-time (baseline) in ms
        test_cases_for_different_devices = {
            "TPU v6e": {
                ("prefill", 128, 1024, 4, 1, 128, 600000): 0.01574625,
                ("prefill", 128, 1024, 4, 2, 128, 600000): 0.01525625,
                ("prefill", 128, 1024, 8, 1, 128, 600000): 0.02189,
                ("prefill", 128, 1024, 8, 4, 128, 600000): 0.0239625,
                ("prefill", 128, 4096, 4, 1, 128, 600000): 0.09134625,
                ("prefill", 128, 4096, 4, 2, 128, 600000): 0.0915075,
                ("prefill", 128, 4096, 8, 1, 128, 600000): 0.14679375,
                ("prefill", 128, 4096, 8, 4, 128, 600000): 0.162105,
                ("prefill", 256, 1024, 4, 1, 128, 600000): 0.0157125,
                ("prefill", 256, 1024, 4, 2, 128, 600000): 0.01491875,
                ("prefill", 256, 1024, 8, 1, 128, 600000): 0.02186125,
                ("prefill", 256, 1024, 8, 4, 128, 600000): 0.0237575,
                ("prefill", 256, 4096, 4, 1, 128, 600000): 0.09128125,
                ("prefill", 256, 4096, 4, 2, 128, 600000): 0.0919325,
                ("prefill", 256, 4096, 8, 1, 128, 600000): 0.14679125,
                ("prefill", 256, 4096, 8, 4, 128, 600000): 0.16232,
                ("decode", 128, 128, 4, 1, 128, 600000): 0.2329225,
                ("decode", 128, 128, 4, 2, 128, 600000): 0.29882625,
                ("decode", 128, 128, 8, 1, 128, 600000): 0.23746875,
                ("decode", 128, 128, 8, 4, 128, 600000): 0.5295875,
                ("decode", 128, 256, 4, 1, 128, 600000): 0.4627675,
                ("decode", 128, 256, 4, 2, 128, 600000): 0.5923825,
                ("decode", 128, 256, 8, 1, 128, 600000): 0.4630275,
                ("decode", 128, 256, 8, 4, 128, 600000): 1.05854875,
                ("decode", 256, 128, 4, 1, 128, 600000): 0.16898125,
                ("decode", 256, 128, 4, 2, 128, 600000): 0.25190875,
                ("decode", 256, 128, 8, 1, 128, 600000): 0.16870875,
                ("decode", 256, 128, 8, 4, 128, 600000): 0.3997425,
                ("decode", 256, 256, 4, 1, 128, 600000): 0.33464,
                ("decode", 256, 256, 4, 2, 128, 600000): 0.51019875,
                ("decode", 256, 256, 8, 1, 128, 600000): 0.33216375,
                ("decode", 256, 256, 8, 4, 128, 600000): 0.81125875,
            },
            "TPU v7": {
                ("prefill", 128, 1024, 4, 1, 128, 600000): 0.014767107,
                ("prefill", 128, 1024, 4, 2, 128, 600000): 0.015102041,
                ("prefill", 128, 1024, 8, 1, 128, 600000): 0.022470588,
                ("prefill", 128, 1024, 8, 4, 128, 600000): 0.023110444,
                ("prefill", 128, 4096, 4, 1, 128, 600000): 0.0794994,
                ("prefill", 128, 4096, 4, 2, 128, 600000): 0.084321729,
                ("prefill", 128, 4096, 8, 1, 128, 600000): 0.145336134,
                ("prefill", 128, 4096, 8, 4, 128, 600000): 0.145752701,
                ("prefill", 256, 1024, 4, 1, 128, 600000): 0.014686675,
                ("prefill", 256, 1024, 4, 2, 128, 600000): 0.015236495,
                ("prefill", 256, 1024, 8, 1, 128, 600000): 0.022472989,
                ("prefill", 256, 1024, 8, 4, 128, 600000): 0.023372149,
                ("prefill", 256, 4096, 4, 1, 128, 600000): 0.079831933,
                ("prefill", 256, 4096, 4, 2, 128, 600000): 0.084252101,
                ("prefill", 256, 4096, 8, 1, 128, 600000): 0.144997599,
                ("prefill", 256, 4096, 8, 4, 128, 600000): 0.145212485,
                ("decode", 128, 128, 4, 1, 128, 600000): 0.15942617,
                ("decode", 128, 128, 4, 2, 128, 600000): 0.226237695,
                ("decode", 128, 128, 8, 1, 128, 600000): 0.159394958,
                ("decode", 128, 128, 8, 4, 128, 600000): 0.331831933,
                ("decode", 128, 256, 4, 1, 128, 600000): 0.313444178,
                ("decode", 128, 256, 4, 2, 128, 600000): 0.452889556,
                ("decode", 128, 256, 8, 1, 128, 600000): 0.313991597,
                ("decode", 128, 256, 8, 4, 128, 600000): 0.649698679,
                ("decode", 256, 128, 4, 1, 128, 600000): 0.15329892,
                ("decode", 256, 128, 4, 2, 128, 600000): 0.207255702,
                ("decode", 256, 128, 8, 1, 128, 600000): 0.152806723,
                ("decode", 256, 128, 8, 4, 128, 600000): 0.313996399,
                ("decode", 256, 256, 4, 1, 128, 600000): 0.303966387,
                ("decode", 256, 256, 4, 2, 128, 600000): 0.415367347,
                ("decode", 256, 256, 8, 1, 128, 600000): 0.303222089,
                ("decode", 256, 256, 8, 4, 128, 600000): 0.610410564,
            },
        }
        test_cases = test_cases_for_different_devices[get_device_name()]
        max_context_len = 40960
        for case, baseline in test_cases.items():
            (
                mode,
                page_size,
                max_num_batched_tokens,
                q_head_num,
                kv_head_num,
                head_dim,
                max_kv_cache_tokens,
            ) = case
            res = benchmark_backend(
                mode,
                max_context_len,
                max_kv_cache_tokens,
                max_num_batched_tokens,
                q_head_num,
                kv_head_num,
                head_dim,
                page_size,
            )
            expected_result = baseline * (1 + floating_threshold)
            print(f"{case}, res={res:.4}ms, expected_result={expected_result:.4}ms")
            self.assertLess(
                res,
                expected_result,
                f"Run ragged_paged_attention performance test failed, {case=}",
            )


if __name__ == "__main__":
    if is_in_ci():
        print("Run Ragged Paged Attention Performance Test...")
        TestPerformance().test_ragged_paged_attention_performance()
    else:
        print("Run Ragged Paged Attention Full Benchmark...")
        full_benchmark()
