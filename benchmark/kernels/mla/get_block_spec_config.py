import functools
from math import inf
import sys

import jax
import jax.numpy as jnp
import numpy as np
from utils import create_decode_uniform_data, create_prefill_uniform_data

from sgl_jax.srt.kernels.mla.v2.kernel import mla_ragged_paged_attention
from sgl_jax.srt.kernels.utils.perf import multiple_iteration_timeit_from_trace


def is_decode_only(max_num_batched_tokens):
    return max_num_batched_tokens <= 256


def get_kernel_scope_name(num_queries_per_block, num_kv_pages_per_block, page_size):
    return f"MLA-tuning-bq_{num_queries_per_block}-bkvp_{num_kv_pages_per_block}-p_{page_size}"


def benchmark_backend(
    max_context_len,
    max_kv_cache_tokens,
    max_num_batched_tokens,
    q_head_num,
    kv_lora_rank,
    qk_rope_head_dim,
    num_kv_pages_per_block,
    num_queries_per_block,
    page_size,
    dtype=np.float16,
):
    scale = (kv_lora_rank + qk_rope_head_dim) ** -0.5

    if not is_decode_only(max_num_batched_tokens):
        (
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            num_seqs,
            seq_lens,
            _,
            distribution,
        ) = create_prefill_uniform_data(
            max_context_len,
            max_kv_cache_tokens,
            max_num_batched_tokens,
            q_head_num,
            kv_lora_rank,
            qk_rope_head_dim,
            page_size=page_size,
            dtype=dtype,
        )
    else:
        (
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            num_seqs,
            seq_lens,
            _,
            distribution,
        ) = create_decode_uniform_data(
            max_context_len,
            max_kv_cache_tokens,
            max_num_batched_tokens,
            q_head_num,
            kv_lora_rank,
            qk_rope_head_dim,
            page_size=page_size,
            dtype=dtype,
        )

    # Wrap the attention execution for JIT compilation
    @functools.partial(
        jax.jit,
        static_argnames=["sm_scale", "num_kv_pages_per_block", "num_queries_per_block"],
    )
    def jitted_attn(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
        sm_scale,
        num_kv_pages_per_block,
        num_queries_per_block,
    ):
        return mla_ragged_paged_attention(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            distribution,
            sm_scale=sm_scale,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
            decode_batch_size=1,  # Keep it simple for tuning
        )

    attn = functools.partial(
        jitted_attn,
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
        scale,
        num_kv_pages_per_block,
        num_queries_per_block,
    )

    # Warmup compiler
    output, _ = attn()
    jax.block_until_ready(output)

    # Benchmark average execution time
    times = multiple_iteration_timeit_from_trace(
        compute_func=lambda: attn(),
        data_generator=lambda: (),
        task=get_kernel_scope_name(num_queries_per_block, num_kv_pages_per_block, page_size),
        tries=3,
    )
    avg_time = float(np.mean(times)) if times else float("nan")

    return (
        avg_time,
        ql_nope.dtype,
        kv_cache.dtype,
    )


def main():
    jax.distributed.initialize()
    if jax.process_index() == 0:
        print("JAX devices:", jax.devices())
        print("Device count:", jax.device_count())
        print()

    # GLM-5.1 default shapes to tune specifically, or general sweep
    # If a specific argument is passed, tune ONLY GLM-5.1 shapes on TPU v7x
    tune_only_glm5 = len(sys.argv) > 1 and sys.argv[1] == "--glm5-only"

    if tune_only_glm5:
        print("Tuning exclusively for GLM-5.1 (TP=32, TP=16, TP=8) configurations.")
        page_size_config = [256]
        max_num_batched_tokens_config = [
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
        ]
        # Query heads per rank: 128 total heads / TP size
        q_head_num_config = [4, 8, 16] 
        kv_lora_rank_config = [512]
        qk_rope_head_dim_config = [64]
    else:
        page_size_config = [128, 256]
        max_num_batched_tokens_config = [
            1, 4, 16, 64, 256, 1024, 4096, 8192
        ]
        q_head_num_config = [2, 4, 8, 16]
        kv_lora_rank_config = [512]
        qk_rope_head_dim_config = [64]

    max_kv_cache_tokens_config = [600000]
    max_context_len = 40960
    
    all_combinations = []
    for q_head_num in q_head_num_config:
        for kv_lora_rank in kv_lora_rank_config:
            for qk_rope_head_dim in qk_rope_head_dim_config:
                for page_size in page_size_config:
                    for max_kv_cache_tokens in max_kv_cache_tokens_config:
                        for max_num_batched_tokens in max_num_batched_tokens_config:
                            all_combinations.append(
                                (
                                    page_size,
                                    max_kv_cache_tokens,
                                    max_num_batched_tokens,
                                    q_head_num,
                                    kv_lora_rank,
                                    qk_rope_head_dim,
                                )
                            )

    # Tiling search space
    num_kv_pages_per_blk_config = [1, 2, 4, 8, 16, 32]
    num_queries_per_block_config = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    prefill_only_block_spec_configs = []
    for num_kv_pages_per_blk in num_kv_pages_per_blk_config:
        for num_queries_per_block in num_queries_per_block_config:
            prefill_only_block_spec_configs.append((num_kv_pages_per_blk, num_queries_per_block))

    decode_only_block_spec_configs = []
    for num_kv_pages_per_blk in num_kv_pages_per_blk_config:
        decode_only_block_spec_configs.append((num_kv_pages_per_blk, 1))

    for i, (
        page_size,
        max_kv_cache_tokens,
        max_num_batched_tokens,
        q_head_num,
        kv_lora_rank,
        qk_rope_head_dim,
    ) in enumerate(all_combinations):
        best_output = inf
        best_config = None
        if is_decode_only(max_num_batched_tokens):
            block_spec_configs = decode_only_block_spec_configs
        else:
            block_spec_configs = prefill_only_block_spec_configs

        for j, (num_kv_pages_per_blk, num_queries_per_block) in enumerate(block_spec_configs):
            try:
                (
                    flash_time,
                    q_dtype,
                    k_dtype,
                ) = benchmark_backend(
                    max_context_len,
                    max_kv_cache_tokens,
                    max_num_batched_tokens,
                    q_head_num,
                    kv_lora_rank,
                    qk_rope_head_dim,
                    num_kv_pages_per_blk,
                    num_queries_per_block,
                    page_size,
                    dtype=jnp.bfloat16,
                )
                if flash_time < best_output:
                    best_output = flash_time
                    best_config = (num_kv_pages_per_blk, num_queries_per_block)
            except Exception as e:
                import traceback
                traceback.print_exc()
        if jax.process_index() == 0 and best_config:
            # Output in python dict format for easy copying to tuned_block_sizes.py
            print(
                f"('{q_dtype}', '{k_dtype}', {q_head_num}, {kv_lora_rank}, {qk_rope_head_dim}, {page_size}, {max_num_batched_tokens}): ({best_config[0]}, {best_config[1]}),"
            )


if __name__ == "__main__":
    main()
