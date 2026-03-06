import functools
from math import inf

import jax
import jax.numpy as jnp
import numpy as np
from utils import create_decode_uniform_data, create_prefill_uniform_data

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention import (
    get_kernel_scope_name,
    get_smem_estimate_bytes,
    get_vmem_estimate_bytes,
    ragged_paged_attention,
    TPU_SMEM_LIMIT_BYTES,
    TPU_VMEM_LIMIT_BYTES,
)
from sgl_jax.srt.kernels.ragged_paged_attention.util import cdiv
from sgl_jax.srt.kernels.utils.perf import multiple_iteration_timeit_from_trace
from sgl_jax.srt.utils.jax_utils import get_device_name


def is_decode_only(max_num_batched_tokens):
    return max_num_batched_tokens <= 512 # original 256


def benchmark_backend(
    max_context_len,
    max_kv_cache_tokens,
    max_num_batched_tokens,
    q_head_num,
    kv_head_num,
    head_dim,
    num_kv_pages_per_block,
    num_queries_per_block,
    page_size,
):
    scale = head_dim**-0.5

    #print(f"=========begin to create data", flush=True)

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
    #print(f"=========complete creating data", flush=True)

    @functools.partial(
        jax.jit,
        static_argnames=["sm_scale", "num_kv_pages_per_block", "num_queries_per_block"],
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
        num_kv_pages_per_block,
        num_queries_per_block,
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
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
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
        num_kv_pages_per_block,
        num_queries_per_block,
    )

    # Warmup
    output = attn()
    jax.block_until_ready(output)

    # Benchmark
    times = multiple_iteration_timeit_from_trace(
        compute_func=lambda: attn(),
        data_generator=lambda: (),
        task=get_kernel_scope_name(num_queries_per_block, num_kv_pages_per_block, page_size),
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
    
    device_name=get_device_name()

    page_size_config = [128, 256, 512]
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
    ]
    # For DeepSeek-R1-Distill-Qwen-1.5B(num_attention_heads=12, num_key_value_heads=2), run in tp=1,2,4
    # q_head_num_config = [3,6,12], kv_head_num_config=[2,1,1], max_context_len=131072 
    q_head_num_config = [3,6,12] # q=12, tp=1,2,4 -> 12,6,3 
    kv_head_num_config = [1, 2] # kv=2, tp=1,2,4 -> 2,1,1
    head_dim_config = [128]
    max_kv_cache_tokens_config = [600000] # 6696805
    all_combinations = []
    max_context_len = 131072
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
        kv_head_num,
        head_dim,
    ) in enumerate(all_combinations):
        best_output = inf
        best_config = None

        # Check SMEM budget before attempting compilation.
        # Decode: batch_size = max_num_batched_tokens (one query per sequence)
        # Prefill: batch_size = ceil(max_num_batched_tokens / 2048) if > 2048, else 1
        if is_decode_only(max_num_batched_tokens):
            est_max_num_seqs = max_num_batched_tokens
        else:
            est_max_num_seqs = cdiv(max_num_batched_tokens, 2048) if max_num_batched_tokens > 2048 else 1
        pages_per_seq = cdiv(max_context_len, page_size)
        smem_bytes = get_smem_estimate_bytes(est_max_num_seqs, pages_per_seq)
        smem_limit_bytes = TPU_SMEM_LIMIT_BYTES[device_name]
        if smem_bytes > smem_limit_bytes:
            print(
                f"SKIP page_size={page_size} max_num_batched_tokens={max_num_batched_tokens} "
                f"q={q_head_num} kv={kv_head_num}: SMEM estimate {smem_bytes} bytes > {smem_limit_bytes} "
                f"(batch_size={est_max_num_seqs}, pages_per_seq={pages_per_seq})"
            )
            continue

        if is_decode_only(max_num_batched_tokens):
            # decode only
            block_spec_configs = decode_only_block_spec_configs
        else:
            block_spec_configs = prefill_only_block_spec_configs

        for i, (num_kv_pages_per_blk, num_queries_per_block) in enumerate(block_spec_configs):
            # Check VMEM budget for this block config.
            bkv_sz = num_kv_pages_per_blk * page_size
            actual_num_q_heads_per_kv_head = q_head_num // kv_head_num
            vmem_bytes = get_vmem_estimate_bytes(
                kv_head_num,
                actual_num_q_heads_per_kv_head,
                head_dim,
                num_queries_per_block,
                bkv_sz,
                jnp.bfloat16,
                jnp.bfloat16,
            )
            vmem_limit_bytes = TPU_VMEM_LIMIT_BYTES[device_name]
            if vmem_bytes > vmem_limit_bytes:
                continue

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
                    num_kv_pages_per_blk,
                    num_queries_per_block,
                    page_size,
                )
                if flash_time < best_output:
                    best_output = flash_time
                    best_config = (num_kv_pages_per_blk, num_queries_per_block)
            except Exception:
                pass
        if best_config:
            print(
                f"('{q_dtype}', '{k_dtype}', {q_head_num}, {kv_head_num}, {head_dim}, {page_size}, {max_num_batched_tokens}): ({best_config[0]}, {best_config[1]}),"
            )


if __name__ == "__main__":
    try:
        import pathwaysutils
        pathwaysutils.initialize()
        print(f"In Pathways")
    except Exception as e:
        print("Not in Pathways")

    main()
