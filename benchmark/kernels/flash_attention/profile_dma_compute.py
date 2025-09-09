"""
简化的profiling：通过调整block参数来分析DMA vs 计算比例
"""

import time

import jax
import numpy as np
from utils import create_prefill_uniform_data

from sgl_jax.srt.layers.attention.flash_attn_kernel.flash_attention import (
    ragged_paged_attention,
)


def benchmark_separated():
    """通过不同的block配置来间接分析DMA vs 计算比例"""

    # 使用相同的数据配置
    batch_size, seq_len, num_heads, head_dim = 2, 2048, 8, 128
    page_size = 128
    max_kv_cache_tokens_num = 120000

    q, k, v, _, page_indices, cu_q_lens, cu_kv_lens, num_seqs, seq_lens, _ = (
        create_prefill_uniform_data(
            batch_size,
            seq_len,
            seq_len,
            max_kv_cache_tokens_num,
            num_heads,
            head_dim,
            page_size=page_size,
        )
    )

    def benchmark_config(name, kv_pages, q_block):
        print(f"\n=== {name} ===")
        print(f"KV pages per block: {kv_pages}, Queries per block: {q_block}")

        # 为每个配置创建单独的jit函数
        @jax.jit
        def flash_attention_config():
            return ragged_paged_attention(
                q,
                k,
                v,
                page_indices,
                cu_q_lens,
                cu_kv_lens,
                num_seqs,
                seq_lens,
                sm_scale=head_dim**-0.5,
                num_kv_pages_per_block=kv_pages,
                num_queries_per_block=q_block,
            )

        # 预热
        result = flash_attention_config()
        jax.block_until_ready(result)

        # 测试
        times = []
        for i in range(5):
            start = time.perf_counter()
            result = flash_attention_config()
            jax.block_until_ready(result)
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times) * 1000
        print(f"Average time: {avg_time:.3f} ms")
        return avg_time

    # 测试不同配置来分析瓶颈（调整为32M VMEM友好的配置）
    results = {}

    # 1. 极小块：测试DMA开销极限
    results["Tiny_blocks"] = benchmark_config(
        "极小块 (最大DMA开销)", kv_pages=1, q_block=8
    )

    # 2. DMA密集型：小块，频繁传输
    results["DMA_intensive"] = benchmark_config(
        "DMA密集型 (小块频繁传输)", kv_pages=2, q_block=16
    )

    # 3. 中等配置：平衡
    results["Balanced"] = benchmark_config("平衡配置", kv_pages=4, q_block=16)

    # 4. 计算密集型：稍大块，减少传输（调小避免OOM）
    results["Compute_intensive"] = benchmark_config(
        "计算密集型 (大块少传输)", kv_pages=8, q_block=24
    )

    print(f"\n{'='*50}")
    print("分析结果:")
    print(f"{'='*50}")

    dma_intensive = results["DMA_intensive"]
    compute_intensive = results["Compute_intensive"]
    tiny_blocks = results["Tiny_blocks"]
    balanced = results["Balanced"]

    print(f"DMA密集型:    {dma_intensive:.3f} ms")
    print(f"计算密集型:   {compute_intensive:.3f} ms")
    print(f"平衡配置:     {balanced:.3f} ms")
    print(f"极小块:       {tiny_blocks:.3f} ms")

    # 分析瓶颈
    if tiny_blocks > dma_intensive * 1.5:
        print("\n🔍 DMA开销很大：极小块比DMA密集型慢很多")
        bottleneck = "DMA setup overhead"
    elif compute_intensive < dma_intensive * 0.8:
        print("\n🔍 计算是瓶颈：大块配置显著更快")
        bottleneck = "Compute bound"
    else:
        print("\n🔍 相对平衡：不同配置性能相近")
        bottleneck = "Balanced"

    # DMA vs 计算比例估算
    dma_overhead_ratio = (dma_intensive - compute_intensive) / dma_intensive * 100
    print(f"\nDMA开销占比估算: ~{dma_overhead_ratio:.1f}%")
    print(f"主要瓶颈: {bottleneck}")

    return results, bottleneck


if __name__ == "__main__":
    benchmark_separated()
