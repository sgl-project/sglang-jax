import functools
from math import inf

import jax
import numpy as np
from utils import create_decode_uniform_data, create_prefill_uniform_data

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3 import (
    RpaCase,
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
    page_size,
    d_block_sizes,
    m_block_sizes,
    sliding_window=None,
):
    scale = head_dim**-0.5
    mode = "decode" if is_decode_only(max_num_batched_tokens) else "prefill"

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

    @functools.partial(
        jax.jit,
        static_argnames=["sm_scale", "sliding_window", "d_block_sizes", "m_block_sizes"],
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
        sliding_window=None,
        d_block_sizes=None,
        m_block_sizes=None,
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
            None,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            d_block_sizes=d_block_sizes,
            m_block_sizes=m_block_sizes,
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
        sliding_window=sliding_window,
        d_block_sizes=d_block_sizes,
        m_block_sizes=m_block_sizes,
    )

    rpa_case = RpaCase.DECODE if mode == "decode" else RpaCase.MIXED
    # Warmup
    block_sizes = d_block_sizes if mode == "decode" else m_block_sizes
    scope_name = f"RPA{rpa_case.symbol}-p_{page_size}-bq_{block_sizes[0]}_{block_sizes[2]}-bkv_{block_sizes[1]}_{block_sizes[3]}"
    if sliding_window:
        scope_name += f"-sw_{sliding_window}"
    output = attn()
    jax.block_until_ready(output)
    # Benchmark
    times = multiple_iteration_timeit_from_trace(
        compute_func=lambda: attn(),
        data_generator=lambda: (),
        task=scope_name,
        tries=3,
    )
    avg_time = float(np.mean(times)) if times else float("nan")

    # cal num_q_heads_per_blk, num_kv_heads_per_blk
    return (
        avg_time,
        q.dtype,
        kv_cache.dtype,
    )


def main():
    print("JAX devices:", jax.devices())
    print("Device count:", jax.device_count())
    print()

    # page_size_config = [128, 256]
    page_size_config = [256]
    max_num_batched_tokens_config = [
        # 1,
        # 2,
        # 4,
        # 8,
        # 16,
        # 32,
        64,
        128,
        # 256,
        # 512,
        # 1024,
        # 2048,
        # 4096,
        # 8192,
    ]
    # q_head_num_config = [1, 2, 4, 8, 16, 32]
    # kv_head_num_config = [1, 2, 4, 8, 16, 32]
    q_head_num_config = [16]
    kv_head_num_config = [1]
    # head_dim_config = [128]
    head_dim_config = [256]
    max_kv_cache_tokens_config = [600000]
    all_combinations = []
    # max_context_len = 40960
    max_context_len = 262144
    for q_head_num in q_head_num_config:
        for kv_head_num in kv_head_num_config:
            for head_dim in head_dim_config:
                for page_size in page_size_config:
                    for max_kv_cache_tokens in max_kv_cache_tokens_config:
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

    bkv_sz_configs = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    bkv_csz_configs = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    bq_sz_configs = [16, 32, 64, 128, 256, 512]
    bq_csz_configs = [16, 32, 64, 128, 256, 512]

    prefill_only_block_spec_configs = []
    for bq_sz in bq_sz_configs:
        for bkv_sz in bkv_sz_configs:
            for bq_csz in bq_csz_configs:
                for bkv_csz in bkv_csz_configs:
                    prefill_only_block_spec_configs.append((bq_sz, bkv_sz, bq_csz, bkv_csz))

    decode_only_block_spec_configs = []
    for bkv_sz in bkv_sz_configs:
        for bkv_csz in bkv_csz_configs:
            if bkv_csz <= bkv_sz:
                decode_only_block_spec_configs.append((1, bkv_sz, 1, bkv_csz))

    for i, (
        page_size,
        max_kv_cache_tokens,
        max_num_batched_tokens,
        q_head_num,
        kv_head_num,
        head_dim,
    ) in enumerate(all_combinations):
        best_output = inf
        best_config = None
        if is_decode_only(max_num_batched_tokens):
            # decode only
            block_spec_configs = decode_only_block_spec_configs
        else:
            block_spec_configs = prefill_only_block_spec_configs

        for i, block_sizes in enumerate(block_spec_configs):
            if is_decode_only(max_num_batched_tokens):
                # decode only
                d_block_sizes = block_sizes
                m_block_sizes = None
            else:
                d_block_sizes = None
                m_block_sizes = block_sizes
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
                    page_size,
                    d_block_sizes,
                    m_block_sizes,
                    # sliding_window=128,
                )
                if flash_time < best_output:
                    best_output = flash_time
                    best_config = block_sizes
            except Exception:
                pass
        if best_config:
            print(
                f"('{q_dtype}', '{k_dtype}', {q_head_num}, {kv_head_num}, {head_dim}, {page_size}, {max_num_batched_tokens}): {best_config},"
            )
            print(f"best cost: {best_output:.4}ms")


if __name__ == "__main__":
    main()
