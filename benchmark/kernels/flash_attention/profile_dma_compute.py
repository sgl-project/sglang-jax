"""
分离式profiling：独立测试DMA和计算部分的耗时
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from utils import create_prefill_uniform_data


def dma_only_kernel(
    k_cache_hbm_ref,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    v_cache_hbm_ref,
    page_indices_ref,
    # Output (dummy)
    dummy_output_ref,
    # Scratch buffers
    k_bufs,  # [2, num_kv_pages_per_blk, page_size, num_kv_heads_per_blk, head_dim]
    v_bufs,
    sems,
):
    """只测试DMA部分：从HBM加载KV数据到VMEM"""
    num_kv_pages_per_blk = k_bufs.shape[1]
    page_size = k_bufs.shape[2]
    num_kv_heads_per_blk = k_bufs.shape[3]

    # 模拟DMA操作：从HBM复制数据到VMEM
    for buf_idx in range(2):  # 测试双缓冲
        for page_idx in range(num_kv_pages_per_blk):
            # K cache DMA
            async_copy_k = pltpu.make_async_copy(
                k_cache_hbm_ref.at[page_indices_ref[page_idx]],
                k_bufs.at[buf_idx, page_idx],
                sems.at[buf_idx, 0],
            )
            # V cache DMA
            async_copy_v = pltpu.make_async_copy(
                v_cache_hbm_ref.at[page_indices_ref[page_idx]],
                v_bufs.at[buf_idx, page_idx],
                sems.at[buf_idx, 1],
            )

            async_copy_k.start()
            async_copy_v.start()
            k_data = async_copy_k.wait()
            v_data = async_copy_v.wait()

    # 写入dummy输出（避免被优化掉）
    dummy_output_ref[0] = k_data[0, 0, 0]


def compute_only_kernel(
    # 预先加载到VMEM的数据
    q_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    k_vmem_ref,  # [num_kv_per_blk, num_kv_heads_per_blk, head_dim]
    v_vmem_ref,  # [num_kv_per_blk, num_kv_heads_per_blk, head_dim]
    # Output
    o_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
):
    """只测试计算部分：假设数据已在VMEM，执行attention计算"""
    num_q_per_blk, num_q_heads_per_blk, head_dim = q_ref.shape
    num_kv_per_blk, num_kv_heads_per_blk, _ = k_vmem_ref.shape

    # 简化的flash attention计算（单个head）
    for head_idx in range(min(num_q_heads_per_blk, num_kv_heads_per_blk)):
        q_head = q_ref[:, head_idx, :]  # [num_q_per_blk, head_dim]
        k_head = k_vmem_ref[
            :, head_idx % num_kv_heads_per_blk, :
        ]  # [num_kv_per_blk, head_dim]
        v_head = v_vmem_ref[
            :, head_idx % num_kv_heads_per_blk, :
        ]  # [num_kv_per_blk, head_dim]

        # QK^T
        qk = jnp.einsum(
            "qd,kd->qk", q_head.astype(jnp.float32), k_head.astype(jnp.float32)
        )
        qk = qk * (head_dim**-0.5)

        # Softmax
        qk_max = jnp.max(qk, axis=-1, keepdims=True)
        qk_exp = jnp.exp(qk - qk_max)
        qk_sum = jnp.sum(qk_exp, axis=-1, keepdims=True)
        attn = qk_exp / qk_sum

        # Attention * V
        out = jnp.einsum("qk,kd->qd", attn, v_head.astype(jnp.float32))
        o_ref = o_ref.at[:, head_idx, :].set(out.astype(q_ref.dtype))


def benchmark_separated():
    """分别benchmark DMA和计算"""
    # 使用与flash attention相同的数据
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

    # 配置参数
    num_kv_pages_per_blk = 8
    num_q_per_blk = 32
    num_kv_heads_per_blk = min(8, num_heads)
    num_q_heads_per_blk = min(16, num_heads)

    print(
        f"Testing with: pages_per_blk={num_kv_pages_per_blk}, q_per_blk={num_q_per_blk}"
    )

    # === 1. DMA Only Test ===
    def create_dma_test():
        return pl.pallas_call(
            dma_only_kernel,
            out_shape=jax.ShapeDtypeStruct((1,), jnp.float32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,  # page_indices is scalar prefetch
                in_specs=[
                    pl.BlockSpec(memory_space=pltpu.ANY),  # k_cache
                    pl.BlockSpec(memory_space=pltpu.ANY),  # v_cache
                ],
                out_specs=pl.BlockSpec((1,), lambda *_: (0,)),
                grid=(1,),
                scratch_shapes=[
                    pltpu.VMEM(
                        (
                            2,
                            num_kv_pages_per_blk,
                            page_size,
                            num_kv_heads_per_blk,
                            head_dim,
                        ),
                        k.dtype,
                    ),
                    pltpu.VMEM(
                        (
                            2,
                            num_kv_pages_per_blk,
                            page_size,
                            num_kv_heads_per_blk,
                            head_dim,
                        ),
                        v.dtype,
                    ),
                    pltpu.SemaphoreType.DMA((2, 2)),
                ],
            ),
            name="dma_only_test",
        )

    # === 2. Compute Only Test ===
    def create_compute_test():
        return pl.pallas_call(
            compute_only_kernel,
            out_shape=jax.ShapeDtypeStruct(
                (num_q_per_blk, num_q_heads_per_blk, head_dim), q.dtype
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(
                        (num_q_per_blk, num_q_heads_per_blk, head_dim),
                        lambda *_: (0, 0, 0),
                    ),
                    pl.BlockSpec(
                        (
                            num_kv_pages_per_blk * page_size,
                            num_kv_heads_per_blk,
                            head_dim,
                        ),
                        lambda *_: (0, 0, 0),
                    ),
                    pl.BlockSpec(
                        (
                            num_kv_pages_per_blk * page_size,
                            num_kv_heads_per_blk,
                            head_dim,
                        ),
                        lambda *_: (0, 0, 0),
                    ),
                ],
                out_specs=pl.BlockSpec(
                    (num_q_per_blk, num_q_heads_per_blk, head_dim), lambda *_: (0, 0, 0)
                ),
                grid=(1,),
                scratch_shapes=[],
            ),
            name="compute_only_test",
        )

    # 编译kernels
    dma_test = jax.jit(create_dma_test())
    compute_test = jax.jit(create_compute_test())

    # 准备测试数据
    sample_q = q[:num_q_per_blk]
    sample_k = k[: num_kv_pages_per_blk * page_size, :num_kv_heads_per_blk]
    sample_v = v[: num_kv_pages_per_blk * page_size, :num_kv_heads_per_blk]
    sample_page_indices = page_indices[:num_kv_pages_per_blk]

    # 预热
    print("Warming up...")
    _ = dma_test(
        k, v, sample_page_indices
    )  # 修正参数顺序：k, v是普通输入，page_indices是scalar prefetch
    _ = compute_test(sample_q, sample_k, sample_v)

    # 测试DMA
    print("\nTesting DMA only...")
    dma_times = []
    for i in range(5):
        start = time.perf_counter()
        result = dma_test(k, v, sample_page_indices)  # 修正参数顺序
        jax.block_until_ready(result)
        dma_times.append(time.perf_counter() - start)

    # 测试计算
    print("Testing Compute only...")
    compute_times = []
    for i in range(5):
        start = time.perf_counter()
        result = compute_test(sample_q, sample_k, sample_v)
        jax.block_until_ready(result)
        compute_times.append(time.perf_counter() - start)

    # 结果
    dma_avg = np.mean(dma_times) * 1000
    compute_avg = np.mean(compute_times) * 1000
    total_est = dma_avg + compute_avg

    print(f"\nResults:")
    print(f"DMA only:     {dma_avg:.3f} ms")
    print(f"Compute only: {compute_avg:.3f} ms")
    print(f"Total (est):  {total_est:.3f} ms")
    print(f"DMA ratio:    {dma_avg/total_est*100:.1f}%")
    print(f"Compute ratio:{compute_avg/total_est*100:.1f}%")


if __name__ == "__main__":
    benchmark_separated()
