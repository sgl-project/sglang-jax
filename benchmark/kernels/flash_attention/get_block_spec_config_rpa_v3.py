import functools
from math import inf

import jax
import numpy as np
from utils import create_decode_uniform_data, create_prefill_uniform_data

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3 import (
    ragged_paged_attention,
)
from sgl_jax.srt.kernels.utils.perf import multiple_iteration_timeit_from_trace


def is_decode_only(max_num_batched_tokens):
    return max_num_batched_tokens <= 256


def benchmark_backend(
    max_context_len,
    max_kv_cache_tokens,
    max_num_batched_tokens,
    q_head_num,
    kv_head_num,
    head_dim,
    bkv_sz,
    bq_sz,
    bkv_csz,
    bq_csz,
    page_size,
    static_q_len,
):
    scale = head_dim**-0.5
    m_block_config = (bq_sz, bkv_sz, bq_csz, bkv_csz)
    p_block_config = (bq_sz, bkv_sz, bq_csz, bkv_csz)
    d_block_config = (1, bkv_sz, 1, bkv_csz)

    if not is_decode_only(max_num_batched_tokens):
        (
            q,
            k,
            v,
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
            kv_head_num,
            head_dim,
            page_size=page_size,
        )

        if static_q_len is None:
            # Updating distribution for mixed kernel
            distribution = distribution.at[1].set(0)
    else:
        (
            q,
            k,
            v,
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
            kv_head_num,
            head_dim,
            page_size=page_size,
        )

        static_q_len = 1

    @functools.partial(
        jax.jit,
        static_argnames=[
            "sm_scale",
            "chunk_prefill_size",
            "m_block_config",
            "d_block_config",
            "p_block_config",
        ],
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
        chunk_prefill_size,
        m_block_config,
        p_block_config,
        d_block_config,
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
            sm_scale=sm_scale,
            chunk_prefill_size=chunk_prefill_size,
            custom_mask=None,
            attention_sink=None,
            p_block_sizes=p_block_config,
            m_block_sizes=m_block_config,
            d_block_sizes=d_block_config,
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
        static_q_len,
        p_block_config,
        m_block_config,
        d_block_config,
    )

    # Warmup
    output = attn()
    jax.block_until_ready(output)

    # Benchmark
    rpa_case = "d" if static_q_len == 1 else "m" if static_q_len is None else "p"
    scope_name = f"RPA{rpa_case}-p_{page_size}-bq_{bq_sz}_{bq_csz}-bkv_{bkv_sz}_{bkv_csz}"
    print(scope_name)
    times = multiple_iteration_timeit_from_trace(
        compute_func=lambda: attn(),
        data_generator=lambda: (),
        task=scope_name,
        tries=3,
    )

    avg_time = float(np.mean(times)) if times else float("nan")

    return (
        avg_time,
        q.dtype,
        kv_cache.dtype,
    )


def get_tuned_value(
    max_context_len,
    max_kv_cache_tokens,
    max_num_batched_tokens,
    q_head_num,
    kv_head_num,
    head_dim,
    page_size,
    static_q_len,
    block_spec_configs,
):
    best_output = inf
    best_config = None

    for i, (bkv_sz, bq_sz, bkv_csz, bq_csz) in enumerate(block_spec_configs):
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
                kv_head_num,
                head_dim,
                bkv_sz,
                bq_sz,
                bkv_csz,
                bq_csz,
                page_size,
                static_q_len,
            )

            if flash_time < best_output:
                best_output = flash_time
                best_config = (bkv_sz, bq_sz, bkv_csz, bq_csz)
        except Exception:
            pass
    if best_config:
        print(
            f"('{q_dtype}', '{k_dtype}', {q_head_num}, {kv_head_num}, {head_dim}, {page_size}, {max_num_batched_tokens}, {static_q_len}): ({best_config[0]}, {best_config[1]}, {best_config[2]}, {best_config[3]}),"
        )


def main():
    print("JAX devices:", jax.devices())
    print("Device count:", jax.device_count())
    print()

    page_size_config = [128]
    max_num_batched_tokens_config = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
    ]
    q_head_num_config = [1, 2, 4, 8, 16, 32]
    kv_head_num_config = [1, 2, 4, 8, 16, 32]
    head_dim_config = [128]
    max_kv_cache_tokens_config = [600000]
    all_combinations = []
    max_context_len = 40960
    prefill_kernel_chunk_size = [8192]

    for q_head_num in q_head_num_config:
        for kv_head_num in kv_head_num_config:
            for head_dim in head_dim_config:
                for page_size in page_size_config:
                    for max_kv_cache_tokens in max_kv_cache_tokens_config:
                        for max_num_batched_tokens in max_num_batched_tokens_config:
                            for chunk in prefill_kernel_chunk_size:
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
                                        chunk,
                                    )
                                )

    num_kv_per_blk_config = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    bkv_csz = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    num_queries_per_block_config = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    bq_csz = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    prefill_only_block_spec_configs = []
    for num_kv_per_blk in num_kv_per_blk_config:
        for num_q_per_blk in num_queries_per_block_config:
            for num_bkv_csz in bkv_csz:
                for num_bq_csz in bq_csz:
                    prefill_only_block_spec_configs.append(
                        (num_kv_per_blk, num_q_per_blk, num_bkv_csz, num_bq_csz)
                    )

    decode_only_block_spec_configs = []
    for num_kv_per_blk in num_kv_per_blk_config:
        for num_bkv_csz in bkv_csz:
            decode_only_block_spec_configs.append((num_kv_per_blk, 1, num_bkv_csz, 1))

    for i, (
        page_size,
        max_kv_cache_tokens,
        max_num_batched_tokens,
        q_head_num,
        kv_head_num,
        head_dim,
        chunk_prefill_size,
    ) in enumerate(all_combinations):
        if is_decode_only(max_num_batched_tokens):
            block_spec_configs = decode_only_block_spec_configs

            # Decode kernel tuning
            get_tuned_value(
                max_context_len,
                max_kv_cache_tokens,
                max_num_batched_tokens,
                q_head_num,
                kv_head_num,
                head_dim,
                page_size,
                1,
                block_spec_configs,
            )
        else:
            block_spec_configs = prefill_only_block_spec_configs

            # Mixed kernel tuning
            get_tuned_value(
                max_context_len,
                max_kv_cache_tokens,
                max_num_batched_tokens,
                q_head_num,
                kv_head_num,
                head_dim,
                page_size,
                None,
                block_spec_configs,
            )

            # Prefill kernel tuning
            get_tuned_value(
                max_context_len,
                max_kv_cache_tokens,
                max_num_batched_tokens,
                q_head_num,
                kv_head_num,
                head_dim,
                page_size,
                chunk_prefill_size,
                block_spec_configs,
            )


if __name__ == "__main__":
    main()
