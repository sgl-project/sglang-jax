"""
测试四缓冲vs双缓冲的性能对比
"""

import time

import jax
import numpy as np
from utils import create_prefill_uniform_data

from sgl_jax.srt.layers.attention.flash_attn_kernel.flash_attention import (
    ragged_paged_attention,
)


def benchmark_buffer_depth():
    """比较不同缓冲深度的性能"""

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

    # 最优配置 (16,48)
    optimal_kv_pages = 16
    optimal_q_block = 48

    def benchmark_config(name, vmem_limit_mb):
        print(f"\n=== {name} ===")
        print(f"VMEM limit: {vmem_limit_mb}MB")

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
                num_kv_pages_per_block=optimal_kv_pages,
                num_queries_per_block=optimal_q_block,
                vmem_limit_bytes=vmem_limit_mb * 1024 * 1024,
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

    print("🔄 测试不同VMEM配置对缓冲深度的影响")
    print("理论：更多VMEM -> 更多缓冲 -> 更好的流水线")

    results = {}

    # 测试不同VMEM大小，观察性能变化
    # 更多VMEM意味着可以支持更深的缓冲
    results["32MB_likely_2buffers"] = benchmark_config("32MB VMEM (双缓冲)", 32)

    results["64MB_likely_3buffers"] = benchmark_config("64MB VMEM (三缓冲)", 64)

    results["96MB_likely_4buffers"] = benchmark_config("96MB VMEM (四缓冲)", 96)

    results["128MB_likely_5buffers"] = benchmark_config("128MB VMEM (五缓冲)", 128)

    print(f"\n{'='*50}")
    print("缓冲深度性能分析:")
    print(f"{'='*50}")

    vmem_32 = results["32MB_likely_2buffers"]
    vmem_64 = results["64MB_likely_3buffers"]
    vmem_96 = results["96MB_likely_4buffers"]
    vmem_128 = results["128MB_likely_5buffers"]

    print(f"32MB (双缓冲):    {vmem_32:.3f} ms")
    print(f"64MB (三缓冲):    {vmem_64:.3f} ms")
    print(f"96MB (四缓冲):    {vmem_96:.3f} ms")
    print(f"128MB (五缓冲):   {vmem_128:.3f} ms")

    # 计算提升
    improvement_64 = (vmem_32 - vmem_64) / vmem_32 * 100
    improvement_96 = (vmem_32 - vmem_96) / vmem_32 * 100
    improvement_128 = (vmem_32 - vmem_128) / vmem_32 * 100

    print(f"\n相对32MB的性能提升:")
    print(f"64MB:  {improvement_64:+.1f}%")
    print(f"96MB:  {improvement_96:+.1f}%")
    print(f"128MB: {improvement_128:+.1f}%")

    # 分析结果
    best_performance = min(vmem_32, vmem_64, vmem_96, vmem_128)
    if best_performance == vmem_96:
        print("\n🎯 最佳配置: 96MB (四缓冲)")
        print("建议：使用四缓冲获得最佳性能")
    elif best_performance == vmem_64:
        print("\n🎯 最佳配置: 64MB (三缓冲)")
        print("建议：三缓冲提供最佳性价比")
    elif best_performance == vmem_128:
        print("\n🎯 最佳配置: 128MB (五缓冲)")
        print("建议：深缓冲在此配置下最优")
    else:
        print("\n📊 结果：更深缓冲无明显提升")
        print("建议：保持当前双缓冲配置")

    return results


if __name__ == "__main__":
    benchmark_buffer_depth()
