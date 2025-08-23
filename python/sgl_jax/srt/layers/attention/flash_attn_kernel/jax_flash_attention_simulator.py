"""
简化的JAX版本Flash Attention模拟器

这个版本避免了复杂的动态操作，专注于模拟核心的attention计算逻辑，
便于调试page_size > 1情况下的精度问题。
"""

import functools
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


def cdiv(a, b):
    """向上整除"""
    return (a + b - 1) // b


def simple_jax_ragged_paged_attention_simulator(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    k_cache: jax.Array,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    v_cache: jax.Array,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    page_indices: jax.Array,  # i32[num_pages]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    cu_kv_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    seq_lens: jax.Array,  # i32[max_num_seqs]
    *,
    sm_scale: float = 1.0,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
    mask_value: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    debug_print: bool = True,
) -> jax.Array:
    """
    简化的JAX模拟器版本的ragged paged attention

    这个版本重建KV cache为完整格式，然后使用与ref_ragged_paged_attention类似的逻辑，
    但加入了对page alignment的模拟。
    """
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE

    # 获取基本形状信息
    num_q_tokens, num_q_heads, head_dim = q.shape
    total_pages, page_size, num_kv_heads, _ = k_cache.shape
    num_q_heads_per_kv_head = num_q_heads // num_kv_heads

    if debug_print:
        print(f"\n=== Simple JAX Simulator Debug Info ===")
        print(f"q.shape: {q.shape}")
        print(f"k_cache.shape: {k_cache.shape}")
        print(f"v_cache.shape: {v_cache.shape}")
        print(f"page_size: {page_size}")
        print(f"num_q_heads: {num_q_heads}")
        print(f"num_kv_heads: {num_kv_heads}")
        print(f"num_seqs: {num_seqs[0]}")
        print(f"cu_q_lens: {cu_q_lens}")
        print(f"cu_kv_lens: {cu_kv_lens}")
        print(f"seq_lens: {seq_lens}")
        print(f"page_indices shape: {page_indices.shape}")
        print(f"page_indices: {page_indices}")

    # 重建完整的KV缓存
    # 计算总的对齐KV长度
    max_aligned_kv_len = jnp.max(cu_kv_lens)

    # 创建重建后的K, V数组
    k_reconstructed = jnp.zeros(
        (max_aligned_kv_len, num_kv_heads, head_dim), dtype=k_cache.dtype
    )
    v_reconstructed = jnp.zeros(
        (max_aligned_kv_len, num_kv_heads, head_dim), dtype=v_cache.dtype
    )

    # 重建每个序列的KV数据
    for seq_idx in range(int(num_seqs[0])):
        kv_start = cu_kv_lens[seq_idx]
        kv_end = cu_kv_lens[seq_idx + 1]
        aligned_kv_len = kv_end - kv_start

        if debug_print:
            print(f"\nReconstructing seq {seq_idx}:")
            print(
                f"  kv_start: {kv_start}, kv_end: {kv_end}, aligned_kv_len: {aligned_kv_len}"
            )

        # 计算这个序列需要的页面范围
        start_page_idx = kv_start // page_size
        end_page_idx = cdiv(kv_end, page_size)

        if debug_print:
            print(f"  start_page_idx: {start_page_idx}, end_page_idx: {end_page_idx}")

        # 收集页面数据
        seq_k_pages = []
        seq_v_pages = []

        for page_offset in range(end_page_idx - start_page_idx):
            abs_page_idx = start_page_idx + page_offset
            if abs_page_idx < page_indices.shape[0]:
                page_idx = page_indices[abs_page_idx]
                seq_k_pages.append(k_cache[page_idx])
                seq_v_pages.append(v_cache[page_idx])
                if debug_print:
                    print(
                        f"    page_offset {page_offset}: abs_page_idx {abs_page_idx} -> page_idx {page_idx}"
                    )

        # 拼接页面
        if seq_k_pages:
            seq_k_full = jnp.concatenate(seq_k_pages, axis=0)
            seq_v_full = jnp.concatenate(seq_v_pages, axis=0)

            # 计算在页面拼接数组中的有效范围
            in_page_start = kv_start % (len(seq_k_pages) * page_size)
            in_page_end = in_page_start + aligned_kv_len  # 使用对齐长度，这是关键！
            in_page_end = min(in_page_end, seq_k_full.shape[0])

            if debug_print:
                print(f"  seq_k_full.shape: {seq_k_full.shape}")
                print(f"  in_page_start: {in_page_start}, in_page_end: {in_page_end}")

            # 提取对齐长度的数据
            seq_k = seq_k_full[in_page_start:in_page_end]
            seq_v = seq_v_full[in_page_start:in_page_end]

            # 放入重建数组的正确位置
            k_reconstructed = k_reconstructed.at[kv_start:kv_end].set(seq_k)
            v_reconstructed = v_reconstructed.at[kv_start:kv_end].set(seq_v)

    if debug_print:
        print(f"\nReconstructed KV shapes:")
        print(f"k_reconstructed.shape: {k_reconstructed.shape}")
        print(f"v_reconstructed.shape: {v_reconstructed.shape}")

    # 现在使用重建后的KV执行类似ref_ragged_paged_attention的逻辑
    outputs = []
    for i in range(int(num_seqs[0])):
        q_start = cu_q_lens[i]
        q_end = cu_q_lens[i + 1]
        q_len = q_end - q_start

        kv_start = cu_kv_lens[i]
        kv_end = cu_kv_lens[i + 1]
        aligned_kv_len = kv_end - kv_start

        # 获取实际序列长度（这是关键差异！）
        if i < seq_lens.shape[0]:
            actual_kv_len = seq_lens[i]
        else:
            actual_kv_len = aligned_kv_len
        actual_kv_len = max(0, min(actual_kv_len, aligned_kv_len))

        if debug_print:
            print(f"\nProcessing seq {i}:")
            print(f"  q_start: {q_start}, q_end: {q_end}, q_len: {q_len}")
            print(f"  kv_start: {kv_start}, kv_end: {kv_end}")
            print(f"  aligned_kv_len: {aligned_kv_len}, actual_kv_len: {actual_kv_len}")

        # 获取query
        seq_q = q[q_start:q_end]

        # 获取重建后的KV，但只取实际长度
        seq_k = k_reconstructed[kv_start : kv_start + actual_kv_len]
        seq_v = v_reconstructed[kv_start : kv_start + actual_kv_len]

        if debug_print:
            print(f"  seq_q.shape: {seq_q.shape}")
            print(f"  seq_k.shape: {seq_k.shape}")
            print(f"  seq_v.shape: {seq_v.shape}")

        # 应用K/V缩放
        if k_scale is not None:
            seq_k = seq_k.astype(jnp.float32) * k_scale
            seq_k = seq_k.astype(seq_q.dtype)
        if v_scale is not None:
            seq_v = seq_v.astype(jnp.float32) * v_scale
            seq_v = seq_v.astype(seq_q.dtype)

        # 处理GQA：重复KV heads
        seq_k = jnp.repeat(seq_k, num_q_heads_per_kv_head, axis=1)
        seq_v = jnp.repeat(seq_v, num_q_heads_per_kv_head, axis=1)

        # 计算attention
        attn = jnp.einsum(
            "qhd,khd->hqk", seq_q, seq_k, preferred_element_type=jnp.float32
        )
        attn *= sm_scale

        # 构建causal mask - 关键：与原kernel保持一致的逻辑
        q_span = (actual_kv_len - q_len) + jax.lax.broadcasted_iota(
            jnp.int32, attn.shape, 1
        )
        kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
        mask = q_span < kv_span

        if sliding_window is not None:
            mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)

        if soft_cap is not None:
            attn = soft_cap * jnp.tanh(attn / soft_cap)

        attn += jnp.where(mask, mask_value, 0.0)
        attn = jax.nn.softmax(attn, axis=-1).astype(seq_v.dtype)
        out = jnp.einsum("hqk,khd->qhd", attn, seq_v).astype(seq_q.dtype)

        outputs.append(out)

        if debug_print:
            print(f"  output.shape: {out.shape}")

    final_output = jnp.concatenate(outputs, axis=0)

    if debug_print:
        print(f"\nFinal output shape: {final_output.shape}")
        print("=== End Simple JAX Simulator Debug Info ===\n")

    return final_output


def test_simple_simulator():
    """测试简化模拟器的基本功能"""
    # 创建测试数据
    page_size = 8
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 64

    # 测试案例：3个序列，复现失败的情况
    seq_lens = jnp.array([5, 5, 5], dtype=jnp.int32)  # 实际序列长度
    aligned_seq_lens = (
        (seq_lens + page_size - 1) // page_size
    ) * page_size  # 对齐后的长度

    total_q_tokens = jnp.sum(seq_lens)
    total_aligned_kv = jnp.sum(aligned_seq_lens)
    total_pages = total_aligned_kv // page_size

    print(f"Test setup:")
    print(f"seq_lens: {seq_lens}")
    print(f"aligned_seq_lens: {aligned_seq_lens}")
    print(f"total_q_tokens: {total_q_tokens}")
    print(f"total_aligned_kv: {total_aligned_kv}")
    print(f"total_pages: {total_pages}")

    # 创建测试输入
    key = jax.random.PRNGKey(42)
    q = jax.random.normal(
        key, (total_q_tokens, num_q_heads, head_dim), dtype=jnp.bfloat16
    )
    k_cache = jax.random.normal(
        jax.random.split(key, 2)[0],
        (total_pages, page_size, num_kv_heads, head_dim),
        dtype=jnp.bfloat16,
    )
    v_cache = jax.random.normal(
        jax.random.split(key, 2)[1],
        (total_pages, page_size, num_kv_heads, head_dim),
        dtype=jnp.bfloat16,
    )

    page_indices = jnp.arange(total_pages, dtype=jnp.int32)
    cu_q_lens = jnp.concatenate([jnp.array([0]), jnp.cumsum(seq_lens)])
    cu_kv_lens = jnp.concatenate([jnp.array([0]), jnp.cumsum(aligned_seq_lens)])
    num_seqs = jnp.array([3])

    print(f"cu_q_lens: {cu_q_lens}")
    print(f"cu_kv_lens: {cu_kv_lens}")

    # 运行简化模拟器
    output = simple_jax_ragged_paged_attention_simulator(
        q,
        k_cache,
        v_cache,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        num_seqs,
        seq_lens,
        sm_scale=1.0 / jnp.sqrt(head_dim),
        debug_print=True,
    )

    print(f"Final test output shape: {output.shape}")
    print("Simple test completed!")

    return output


if __name__ == "__main__":
    test_simple_simulator()
