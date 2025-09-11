# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TPU-Friendly Ragged Paged Attention kernel.

This kernel offers a highly optimized implementation of ragged paged attention,
specifically designed for TPU and compatible with a wide range of model
specifications. It supports mixed prefill and decoding, enhancing throughput
during inference.
"""
import functools

import jax
import jax.experimental.pallas.ops.tpu.ragged_paged_attention.tuned_block_sizes as tbs
import jax.numpy as jnp
from jax import lax
from jax._src import dtypes
from jax._src.lax.lax import Precision
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.utils import cdiv

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


TUNED_BLOCK_SIZES = {
    "TPU v6": {
        # (q_dtype, kv_dtype, num_kv_heads_per_blk, head_dim, page_size)
        ("bfloat16", "bfloat16", 2, 128, 1): (64, 256),
        ("bfloat16", "bfloat16", 4, 128, 1): (64, 256),
        ("bfloat16", "bfloat16", 8, 128, 1): (64, 256),
        ("bfloat16", "bfloat16", 16, 128, 1): (64, 256),
        ("bfloat16", "bfloat16", 32, 128, 1): (64, 256),
        ("bfloat16", "bfloat16", 2, 128, 2): (64, 256),
        ("bfloat16", "bfloat16", 4, 128, 2): (64, 256),
        ("bfloat16", "bfloat16", 8, 128, 2): (64, 256),
        ("bfloat16", "bfloat16", 16, 128, 2): (64, 256),
        ("bfloat16", "bfloat16", 32, 128, 2): (64, 256),
        ("bfloat16", "bfloat16", 2, 128, 4): (64, 256),
        ("bfloat16", "bfloat16", 4, 128, 4): (64, 256),
        ("bfloat16", "bfloat16", 8, 128, 4): (64, 256),
        ("bfloat16", "bfloat16", 16, 128, 4): (64, 256),
        ("bfloat16", "bfloat16", 32, 128, 4): (64, 256),
        ("bfloat16", "bfloat16", 2, 128, 8): (64, 256),
        ("bfloat16", "bfloat16", 4, 128, 8): (64, 256),
        ("bfloat16", "bfloat16", 8, 128, 8): (64, 256),
        ("bfloat16", "bfloat16", 16, 128, 8): (64, 256),
        ("bfloat16", "bfloat16", 32, 128, 8): (64, 256),
        ("bfloat16", "bfloat16", 2, 128, 16): (64, 128),
        ("bfloat16", "bfloat16", 4, 128, 16): (64, 128),
        ("bfloat16", "bfloat16", 8, 128, 16): (64, 128),
        ("bfloat16", "bfloat16", 16, 128, 16): (64, 128),
        ("bfloat16", "bfloat16", 32, 128, 16): (64, 128),
        ("bfloat16", "bfloat16", 2, 128, 32): (64, 128),
        ("bfloat16", "bfloat16", 4, 128, 32): (64, 128),
        ("bfloat16", "bfloat16", 8, 128, 32): (32, 64),
        ("bfloat16", "bfloat16", 16, 128, 32): (32, 64),
        ("bfloat16", "bfloat16", 32, 128, 32): (32, 64),
        ("bfloat16", "bfloat16", 2, 128, 64): (32, 64),
        ("bfloat16", "bfloat16", 4, 128, 64): (32, 64),
        ("bfloat16", "bfloat16", 8, 128, 64): (64, 128),
        ("bfloat16", "bfloat16", 16, 128, 64): (24, 48),
        ("bfloat16", "bfloat16", 32, 128, 64): (24, 48),
        ("bfloat16", "bfloat16", 2, 128, 128): (16, 32),
        ("bfloat16", "bfloat16", 4, 128, 128): (16, 32),
        ("bfloat16", "bfloat16", 8, 128, 128): (8, 32),
        ("bfloat16", "bfloat16", 16, 128, 128): (8, 32),
        ("bfloat16", "bfloat16", 32, 128, 128): (8, 32),
        ("bfloat16", "bfloat16", 2, 128, 256): (32, 64),
        ("bfloat16", "bfloat16", 4, 128, 256): (16, 64),
        ("bfloat16", "bfloat16", 8, 128, 256): (32, 64),
        ("bfloat16", "bfloat16", 16, 128, 256): (16, 32),
        ("bfloat16", "bfloat16", 32, 128, 256): (16, 64),
        # go/keep-sorted end
    },
}


def next_power_of_2(x: int):
    """Finds the smallest power of 2 >= x using bit manipulation.

    Args:
      x: The input number (should be an integer).

    Returns:
      The smallest integer power of 2 that is >= x.
    """
    assert x > 0
    if x == 1:
        return 1
    return 1 << (x - 1).bit_length()


def simplify_key(key):
    """Simplify the key to reduce the number of combinations."""
    (
        q_dtype,
        kv_dtype,
        num_kv_heads_per_blk,
        head_dim,
        page_size,
    ) = key
    return (
        jnp.dtype(q_dtype).name,
        jnp.dtype(kv_dtype).name,
        next_power_of_2(num_kv_heads_per_blk),
        (head_dim + 127) // 128 * 128,
        next_power_of_2(page_size),
    )


def get_tuned_block_sizes(
    q_dtype,
    kv_dtype,
    num_kv_heads_per_blk,
    head_dim,
    page_size,
) -> tuple[int, int]:
    """Look up for the best (num_kv_pages_per_blk, num_queries_per_blk) from auto-tuned table."""
    tpu_version = tbs.get_tpu_version()
    if tpu_version < 5:
        raise NotImplementedError("TPU version must be 4 or higher.")
    key = (
        q_dtype,
        kv_dtype,
        num_kv_heads_per_blk,
        head_dim,
        page_size,
    )
    key = simplify_key(key)
    device_name = tbs.get_device_name()

    # Default block sizes.
    bkv, bq = (8, 32)
    if device_name in TUNED_BLOCK_SIZES:
        if key in TUNED_BLOCK_SIZES[device_name]:
            bkv, bq = TUNED_BLOCK_SIZES[device_name][key]
    return (bkv, bq)


class MultiPageAsyncCopyDescriptor:
    """Descriptor for async copy of multiple K/V pages from HBM."""

    def __init__(
        self,
        pages_hbm_ref,  # [total_num_pages, page_size, num_kv_heads_per_blk, head_dim]
        vmem_buf,  # [num_kv_pages_per_blk, page_size, num_kv_heads_per_blk, head_dim]
        sem,
        page_indices_ref,  # i32[num_pages]
        metadata,  # [start_page_idx, end_page_idx]
    ):
        self._vmem_buf = vmem_buf
        start_page_idx, end_page_idx = metadata
        self._async_copies = []
        for i in range(vmem_buf.shape[0]):
            page_idx = start_page_idx + i
            page_idx = jax.lax.select(page_idx < end_page_idx, page_idx, 0)
            self._async_copies.append(
                pltpu.make_async_copy(
                    pages_hbm_ref.at[page_indices_ref[page_idx]],
                    vmem_buf.at[i],
                    sem,
                )
            )

    def start(self):
        """Starts the async copies."""
        for async_copy in self._async_copies:
            async_copy.start()

    def wait(self):
        for async_copy in self._async_copies:
            async_copy.wait()
        return self._vmem_buf


# Expect to run these checks during compile time.
def static_validate_inputs(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    k_cache: jax.Array,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    v_cache: jax.Array,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    page_indices: jax.Array,  # i32[num_pages]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    cu_kv_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    seq_lens: jax.Array,  # i32[max_num_seqs]
    *,
    # These inputs are optional. If not specified, we will not validate them.
    sm_scale: float | None = None,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
    _, num_q_heads, head_dim = q.shape
    _, _, num_kv_heads, head_dim_k = k_cache.shape
    assert k_cache.dtype == v_cache.dtype
    assert k_cache.shape == v_cache.shape
    assert num_kv_heads % 2 == 0
    assert isinstance(k_scale, float) or k_scale is None
    assert isinstance(v_scale, float) or v_scale is None

    max_num_seqs = cu_kv_lens.shape[0] - 1
    num_pages = len(page_indices)
    if num_seqs.shape != (1,):
        raise ValueError(f"{num_seqs.shape=} must be (1,)")
    if head_dim_k != head_dim:
        raise ValueError(
            f"Q head_dim {head_dim} must be the same as that of K/V {head_dim_k}."
        )
    if cu_q_lens.shape != (max_num_seqs + 1,):
        raise ValueError(
            f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},)  where"
            " `max_num_seqs` is `page_indices.shape[0]`."
        )
    if (
        cu_kv_lens.dtype != jnp.int32
        or page_indices.dtype != jnp.int32
        or cu_q_lens.dtype != jnp.int32
        or seq_lens.dtype != jnp.int32
    ):
        raise ValueError(
            "The dtype of `kv_lens`, `page_indices`, `cu_q_lens`, and `seq_lens` must be"
            f" int32. Got {cu_kv_lens.dtype=}, {page_indices.dtype=},"
            f" {cu_q_lens.dtype=}, {seq_lens.dtype=}."
        )
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"{num_q_heads=} must be divisible by {num_kv_heads=}")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError(f"{sliding_window=} must be positive.")
    if soft_cap is not None and soft_cap == 0.0:
        raise ValueError(f"{soft_cap=} must not be 0.0.")
    if num_queries_per_block is not None and num_queries_per_block <= 0:
        raise ValueError(f"{num_queries_per_block=} must be positive.")
    if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
        raise ValueError(f"{vmem_limit_bytes=} must be positive.")
    del sm_scale  # No constraints on sm_scale.
    del mask_value  # No consstraints on mask_value.


def ref_ragged_paged_attention_fused(
    queries: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages_fused: jax.Array,  # [total_num_pages, page_size, num_kv_heads, head_dim * 2]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1],
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
    """
    Ragged paged attention reference implementation with fused KV cache.

    This version accepts a fused KV cache where k and v are concatenated
    along the head_dim axis, following tpu_commons v3 layout.

    Args:
        queries: Query tensor [max_num_batched_tokens, num_q_heads, head_dim]
        kv_pages_fused: Fused KV cache [total_num_pages, page_size, num_kv_heads, head_dim * 2]
                       Each kv_head contains [k_features, v_features]
        Other args: Same as original ref_ragged_paged_attention
    """
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE

    head_dim = queries.shape[-1]
    num_q_heads = queries.shape[1]
    num_kv_heads = kv_pages_fused.shape[-2]
    assert num_q_heads % num_kv_heads == 0
    num_query_per_kv = num_q_heads // num_kv_heads

    outputs = []
    for i in range(num_seqs[0]):
        q_start = cu_q_lens[i]
        q_end = cu_q_lens[i + 1]
        q_len = q_end - q_start
        kv_len = kv_lens[i]
        indices = page_indices[i]
        q = queries[q_start:q_end]

        # Key change: extract k and v from fused KV cache
        kv_fused = kv_pages_fused[indices, :, :, :].reshape(
            -1, num_kv_heads, head_dim * 2
        )[:kv_len]
        k = kv_fused[:, :, :head_dim]  # Extract k from first half
        v = kv_fused[:, :, head_dim:]  # Extract v from second half

        # Apply scaling if provided
        if k_scale is not None:
            k = k.astype(jnp.float32) * k_scale
            k = k.astype(q.dtype)
        if v_scale is not None:
            v = v.astype(jnp.float32) * v_scale
            v = v.astype(q.dtype)

        # Repeat for GQA
        k = jnp.repeat(k, num_query_per_kv, axis=1)
        v = jnp.repeat(v, num_query_per_kv, axis=1)

        # Compute attention (same as original)
        attn = jnp.einsum("qhd,khd->hqk", q, k, preferred_element_type=jnp.float32)
        attn *= sm_scale
        q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(jnp.int32, attn.shape, 1)
        kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
        mask = q_span < kv_span
        if sliding_window is not None:
            mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
        if soft_cap is not None:
            attn = soft_cap * jnp.tanh(attn / soft_cap)
        attn += jnp.where(mask, mask_value, 0.0)
        attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
        out = jnp.einsum("hqk,khd->qhd", attn, v).astype(queries.dtype)
        outputs.append(out)

    return jnp.concatenate(outputs, axis=0)


def ref_ragged_paged_attention(
    queries: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    k_pages: jax.Array,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    v_pages: jax.Array,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1],
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE
    _, _, num_kv_heads, head_dim = k_pages.shape
    num_q_heads = queries.shape[1]
    assert num_q_heads % num_kv_heads == 0
    num_query_per_kv = num_q_heads // num_kv_heads
    outputs = []
    for i in range(num_seqs[0]):
        q_start = cu_q_lens[i]
        q_end = cu_q_lens[i + 1]
        q_len = q_end - q_start
        kv_len = kv_lens[i]
        indices = page_indices[i]
        q = queries[q_start:q_end]
        k = k_pages[indices, :, :, :].reshape(-1, num_kv_heads, head_dim)[:kv_len]
        v = v_pages[indices, :, :, :].reshape(-1, num_kv_heads, head_dim)[:kv_len]
        if k_scale is not None:
            k = k.astype(jnp.float32) * k_scale
            k = k.astype(q.dtype)
        if v_scale is not None:
            v = v.astype(jnp.float32) * v_scale
            v = v.astype(q.dtype)
        k = jnp.repeat(k, num_query_per_kv, axis=1)
        v = jnp.repeat(v, num_query_per_kv, axis=1)
        attn = jnp.einsum("qhd,khd->hqk", q, k, preferred_element_type=jnp.float32)
        attn *= sm_scale
        q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(jnp.int32, attn.shape, 1)
        kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
        mask = q_span < kv_span
        if sliding_window is not None:
            mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
        if soft_cap is not None:
            attn = soft_cap * jnp.tanh(attn / soft_cap)
        attn += jnp.where(mask, mask_value, 0.0)
        attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
        out = jnp.einsum("hqk,khd->qhd", attn, v).astype(queries.dtype)
        outputs.append(out)

    return jnp.concatenate(outputs, axis=0)


def ragged_paged_attention_kernel(
    # Prefetch
    page_indices_ref,  # [num_pages]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    cu_kv_lens_ref,  # [max_num_seqs + 1]
    seq_buf_idx_ref,
    num_seqs_ref,
    seq_lens_ref,
    # Input
    q_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    kv_cache_hbm_ref,  # [total_num_pages, page_size, num_kv_heads, head_dim * 2]  # FUSED!
    # Output
    o_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    # Scratch
    kv_bufs,  # [2, num_kv_pages_per_blk, page_size, num_kv_heads_per_blk, head_dim * 2]  # FUSED!
    sems,  # [2, 2]
    l_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
    m_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
    acc_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    *,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE
    num_q_per_blk, num_q_heads_per_blk, head_dim = q_ref.shape
    num_seqs = num_seqs_ref[0]
    # kv_bufs shape: [2, num_kv_pages_per_blk, page_size, num_kv_heads_per_blk, head_dim * 2]
    _, num_kv_pages_per_blk, page_size, num_kv_heads_per_blk, fused_head_dim = (
        kv_bufs.shape
    )
    assert fused_head_dim == head_dim * 2  # Verify fused dimension
    num_kv_per_blk = num_kv_pages_per_blk * page_size
    num_q_heads_per_kv_head = num_q_heads_per_blk // num_kv_heads_per_blk
    heads_blk_idx, q_blk_idx = (
        pl.program_id(0),
        pl.program_id(1),
    )
    num_heads_blks = pl.num_programs(0)
    init_seq_idx = seq_buf_idx_ref[0]
    init_buf_idx = seq_buf_idx_ref[1]
    q_len_start = q_blk_idx * num_q_per_blk
    q_len_end = q_len_start + num_q_per_blk

    def create_fused_kv_async_copy_descriptor(
        heads_blk_idx, seq_idx, kv_blk_idx, buf_idx
    ):
        start_kv_page_idx = (
            cdiv(cu_kv_lens_ref[seq_idx], page_size) + kv_blk_idx * num_kv_pages_per_blk
        )
        end_kv_page_idx = cdiv(cu_kv_lens_ref[seq_idx + 1], page_size)
        metadata = (start_kv_page_idx, end_kv_page_idx)
        heads_start = heads_blk_idx * num_kv_heads_per_blk

        # Single async copy for fused KV cache
        async_copy_kv = MultiPageAsyncCopyDescriptor(
            kv_cache_hbm_ref.at[:, :, pl.ds(heads_start, num_kv_heads_per_blk), :],
            kv_bufs.at[buf_idx],
            sems.at[buf_idx, 0],  # Only need one semaphore for fused copy
            page_indices_ref,
            metadata,
        )
        return async_copy_kv

    def strided_load_kv(ref, start, step):
        if ref.dtype == jnp.float32:
            return ref[start::step, :]
        packing = get_dtype_packing(ref.dtype)
        assert ref.dtype == jnp.bfloat16
        assert step % packing == 0
        b_start = start // packing
        b_offset = start % packing
        b_step = step // packing
        b_ref = pltpu.bitcast(ref, jnp.int32)
        b = b_ref[b_start::b_step, :]
        bw = 32 // packing
        b = jnp.right_shift(b, bw * b_offset)
        b = jnp.left_shift(b, bw * (packing - 1))
        return pltpu.bitcast(b, jnp.float32).astype(jnp.bfloat16)

    def fold_on_2nd_minor(vec):
        assert vec.dtype == jnp.bfloat16 or vec.dtype == jnp.float32
        assert len(vec.shape) >= 2
        last_dim = vec.shape[-1]
        packing = get_dtype_packing(vec.dtype)
        if vec.shape[-2] % packing != 0:
            vec = vec.astype(jnp.float32)
        return vec.reshape(-1, last_dim)

    def batch_load_all_heads_kv(k_ref, v_ref, num_kv_heads_per_blk):
        k_heads = []
        v_heads = []

        for head_idx in range(num_kv_heads_per_blk):
            k_head = strided_load_kv(k_ref, head_idx, num_kv_heads_per_blk)
            v_head = strided_load_kv(v_ref, head_idx, num_kv_heads_per_blk)
            k_heads.append(k_head)
            v_heads.append(v_head)

        return jnp.stack(k_heads, axis=0), jnp.stack(v_heads, axis=0)

    def batch_prepare_queries(q_ref, num_kv_heads_per_blk, num_q_heads_per_kv_head):
        q_heads = []
        for kv_head_idx in range(num_kv_heads_per_blk):
            q_head_idx = kv_head_idx * num_q_heads_per_kv_head
            q = fold_on_2nd_minor(
                q_ref[:, q_head_idx : q_head_idx + num_q_heads_per_kv_head, :]
            )
            q_heads.append(q)

        return jnp.stack(q_heads, axis=0)

    @pl.when(heads_blk_idx + q_blk_idx == 0)
    def prefetch_first_kv_blk():
        async_copy_kv = create_fused_kv_async_copy_descriptor(
            heads_blk_idx, init_seq_idx, 0, init_buf_idx
        )
        async_copy_kv.start()

    def is_cur_q_blk_needed(q_states):
        done, cur_seq_idx, _ = q_states
        should_run = jnp.logical_and(
            q_len_start < cu_q_lens_ref[num_seqs], cur_seq_idx < num_seqs
        )
        return jnp.logical_and(done == 0, should_run)

    def compute_with_cur_q_blk(q_states):
        done, cur_seq_idx, cur_buf_idx = q_states
        q_start = cu_q_lens_ref[cur_seq_idx]
        q_end = cu_q_lens_ref[cur_seq_idx + 1]
        q_len = q_end - q_start
        kv_start = cu_kv_lens_ref[cur_seq_idx]
        kv_end = cu_kv_lens_ref[cur_seq_idx + 1]
        kv_len = kv_end - kv_start

        actual_kv_len = seq_lens_ref[cur_seq_idx]

        def get_next_prefetch_ids(heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx):
            next_kv_blk_idx = kv_blk_idx + 1
            is_last_kv_blk = next_kv_blk_idx * num_kv_per_blk >= kv_len
            next_kv_blk_idx = lax.select(
                is_last_kv_blk,
                0,
                next_kv_blk_idx,
            )
            is_cur_seq_end_in_cur_q_blk = q_end <= q_len_end
            next_seq_idx = lax.select(
                is_last_kv_blk,
                lax.select(is_cur_seq_end_in_cur_q_blk, cur_seq_idx + 1, cur_seq_idx),
                cur_seq_idx,
            )
            is_last_seq = next_seq_idx == num_seqs
            next_seq_idx = lax.select(
                is_last_seq,
                0,
                next_seq_idx,
            )
            next_heads_blk_idx = lax.select(
                is_last_seq,
                heads_blk_idx + 1,
                heads_blk_idx,
            )
            next_buf_idx = lax.select(cur_buf_idx == 0, 1, 0)
            return next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx

        def flash_attention(
            q_batch,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, head_dim]
            k_batch,  # [num_kv_heads_per_blk, num_kv_per_blk_batch, head_dim]
            v_batch,  # [num_kv_heads_per_blk, num_kv_per_blk_batch, head_dim]
            l_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
            m_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
            acc_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
            *,
            kv_blk_idx,
            actual_kv_len,
            sm_scale,
            mask_value,
            q_start,
            q_end,
            q_len,
            q_len_start,
            kv_len_start,
            sliding_window=None,
            soft_cap=None,
            k_scale=None,
            v_scale=None,
        ):
            num_kv_heads_per_blk_batch, num_kv_per_blk_batch, head_dim_batch = (
                k_batch.shape
            )

            def load_with_init_batch(ref, init_val):
                return jnp.where(
                    kv_blk_idx == 0, jnp.full_like(ref, init_val), ref[...]
                )

            def masked_store_batch(ref, val, start, end, group=1):
                iota = lax.broadcasted_iota(jnp.int32, ref.shape, 1) // group
                mask_bool = jnp.logical_and(iota >= start, iota < end)
                pl.store(
                    ref,
                    idx=tuple(slice(None) for _ in ref.shape),
                    val=val,
                    mask=mask_bool,
                )

            kv_len_start = kv_blk_idx * num_kv_per_blk_batch

            q_batch_f32 = q_batch.astype(jnp.float32)
            k_batch_f32 = k_batch.astype(jnp.float32)
            v_batch_f32 = v_batch.astype(jnp.float32)

            if k_scale is not None:
                k_batch_f32 = k_batch_f32 * k_scale
            if v_scale is not None:
                v_batch_f32 = v_batch_f32 * v_scale

            effective_kv_len = actual_kv_len - kv_len_start
            kv_indices = lax.broadcasted_iota(jnp.int32, k_batch_f32.shape[:-1], 1)
            kv_mask_int = (kv_indices < effective_kv_len).astype(jnp.int32)
            kv_mask_expanded = jnp.expand_dims(kv_mask_int, axis=-1)
            kv_mask_broadcast = jnp.broadcast_to(kv_mask_expanded, k_batch_f32.shape)

            k_batch_f32 = jnp.where(kv_mask_broadcast > 0, k_batch_f32, 0.0)
            v_batch_f32 = jnp.where(kv_mask_broadcast > 0, v_batch_f32, 0.0)
            qk_batch = (
                jnp.einsum(
                    "hqd,hkd->hqk",
                    q_batch_f32,
                    k_batch_f32,
                    preferred_element_type=jnp.float32,
                )
                * sm_scale
            )  # [num_kv_heads_per_blk, num_q_total, num_kv_per_blk_batch]

            store_start = jnp.maximum(q_start - q_len_start, 0)
            store_end = jnp.minimum(q_end - q_len_start, num_q_per_blk)

            row_ids = (
                (actual_kv_len - q_len)
                + q_len_start
                - q_start
                + jax.lax.broadcasted_iota(jnp.int32, qk_batch.shape, 1)
                // num_q_heads_per_kv_head
            )
            col_ids = kv_len_start + jax.lax.broadcasted_iota(
                jnp.int32, qk_batch.shape, 2
            )

            causal_mask = row_ids < col_ids
            if sliding_window is not None:
                causal_mask = jnp.logical_or(
                    causal_mask, row_ids - sliding_window >= col_ids
                )

            if soft_cap is not None:
                qk_batch = soft_cap * jnp.tanh(qk_batch / soft_cap)

            qk_batch += jnp.where(causal_mask, mask_value, 0.0)

            m_curr = jnp.max(
                qk_batch, axis=-1, keepdims=True
            )  # [num_kv_heads_per_blk, num_q_total, 1]
            s_curr = jnp.exp(qk_batch - m_curr)

            qkv_batch = jnp.einsum(
                "hqk,hkd->hqd", s_curr, v_batch_f32, preferred_element_type=jnp.float32
            )

            lm_store_shape = m_ref.shape
            m_curr_expanded = jnp.broadcast_to(m_curr, lm_store_shape)
            l_curr_expanded = jnp.broadcast_to(
                s_curr.sum(axis=-1, keepdims=True), lm_store_shape
            )

            m_prev = load_with_init_batch(m_ref, -jnp.inf)
            l_prev = load_with_init_batch(l_ref, 0.0)

            m_next = jnp.maximum(m_prev, m_curr_expanded)
            masked_store_batch(
                m_ref, m_next, store_start, store_end, num_q_heads_per_kv_head
            )

            alpha = jnp.exp(m_prev - m_next)
            beta = jnp.exp(m_curr_expanded - m_next)
            l_alpha = alpha * l_prev
            l_next = l_alpha + beta * l_curr_expanded
            l_next_safe = jnp.where(l_next == 0.0, 1.0, l_next)

            masked_store_batch(
                l_ref, l_next_safe, store_start, store_end, num_q_heads_per_kv_head
            )

            def broadcast_to_qkv_shape(arr, target_shape):
                return jnp.broadcast_to(arr, target_shape)

            o_curr = load_with_init_batch(acc_ref, 0.0)

            # o_curr: [num_q_per_blk, num_q_heads_per_blk, head_dim]
            o_curr_list = []
            for kv_head_idx in range(num_kv_heads_per_blk_batch):
                q_head_start = kv_head_idx * num_q_heads_per_kv_head
                q_head_end = q_head_start + num_q_heads_per_kv_head
                kv_head_slice = o_curr[
                    :, q_head_start:q_head_end, :
                ]  # [num_q_per_blk, num_q_heads_per_kv_head, head_dim]
                kv_head_data = kv_head_slice.reshape(
                    num_q_per_blk * num_q_heads_per_kv_head, head_dim_batch
                )
                o_curr_list.append(kv_head_data)
            o_curr_reshaped = jnp.stack(o_curr_list, axis=0)

            l_alpha_broadcast = broadcast_to_qkv_shape(l_alpha, qkv_batch.shape)
            beta_broadcast = broadcast_to_qkv_shape(beta, qkv_batch.shape)
            l_next_safe_broadcast = broadcast_to_qkv_shape(l_next_safe, qkv_batch.shape)

            out_batch = lax.div(
                l_alpha_broadcast * o_curr_reshaped + beta_broadcast * qkv_batch,
                l_next_safe_broadcast,
            )

            # [num_q_per_blk, num_q_heads_per_blk, head_dim]
            out_batch_reshaped = out_batch.reshape(
                num_kv_heads_per_blk_batch,
                num_q_per_blk,
                num_q_heads_per_kv_head,
                head_dim_batch,
            )

            out_heads_list = []
            for kv_head_idx in range(num_kv_heads_per_blk_batch):
                kv_head_output = out_batch_reshaped[
                    kv_head_idx
                ]  # [num_q_per_blk, num_q_heads_per_kv_head, head_dim]
                out_heads_list.append(kv_head_output)

            out_reshaped = jnp.concatenate(
                out_heads_list, axis=1
            )  # [num_q_per_blk, num_q_heads_per_blk, head_dim]

            def masked_store_acc(ref, val, start, end):
                # [num_q_per_blk, num_q_heads_per_blk, head_dim]
                iota = lax.broadcasted_iota(jnp.int32, ref.shape, 0)
                mask_bool = jnp.logical_and(iota >= start, iota < end)
                pl.store(
                    ref,
                    idx=tuple(slice(None) for _ in ref.shape),
                    val=val,
                    mask=mask_bool,
                )

            masked_store_acc(acc_ref, out_reshaped, store_start, store_end)

        def is_valid_kv_blk_in_cur_seq(kv_states):
            kv_blk_idx, _ = kv_states
            return kv_blk_idx * num_kv_per_blk < actual_kv_len

        def compute_with_kv_blk_in_cur_seq(kv_states):
            kv_blk_idx, cur_buf_idx = kv_states
            next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx = (
                get_next_prefetch_ids(
                    heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
                )
            )

            @pl.when(next_heads_blk_idx < num_heads_blks)
            def prefetch_next_kv_blk():
                # DMA to fixed size buffer!
                next_async_copy_kv = create_fused_kv_async_copy_descriptor(
                    next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx
                )
                next_async_copy_kv.start()

            cur_async_copy_kv = create_fused_kv_async_copy_descriptor(
                heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
            )
            # VMEM 优化：避免不必要的 reshape，4D 切片后再 reshape
            kv_buf_fused = (
                cur_async_copy_kv.wait()
            )  # [num_kv_pages_per_blk, page_size, num_kv_heads_per_blk, head_dim * 2]

            # Extract k and v directly from 4D tensor, then reshape only once
            # 这比先 reshape 再切片更高效，减少 VMEM 内存操作
            k_ref = kv_buf_fused[..., :head_dim].reshape(
                num_kv_pages_per_blk * page_size * num_kv_heads_per_blk, head_dim
            )
            v_ref = kv_buf_fused[..., head_dim:].reshape(
                num_kv_pages_per_blk * page_size * num_kv_heads_per_blk, head_dim
            )

            q_batch = batch_prepare_queries(
                q_ref, num_kv_heads_per_blk, num_q_heads_per_kv_head
            )

            k_batch, v_batch = batch_load_all_heads_kv(
                k_ref, v_ref, num_kv_heads_per_blk
            )

            flash_attention(
                q_batch,
                k_batch,
                v_batch,
                l_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
                m_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
                acc_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
                kv_blk_idx=kv_blk_idx,
                actual_kv_len=actual_kv_len,
                sm_scale=sm_scale,
                mask_value=mask_value,
                q_start=q_start,
                q_end=q_end,
                q_len=q_len,
                q_len_start=q_len_start,
                kv_len_start=kv_blk_idx * num_kv_per_blk,
                sliding_window=sliding_window,
                soft_cap=soft_cap,
                k_scale=k_scale,
                v_scale=v_scale,
            )
            return kv_blk_idx + 1, next_buf_idx

        _, next_buf_idx = lax.while_loop(
            is_valid_kv_blk_in_cur_seq,
            compute_with_kv_blk_in_cur_seq,
            (0, cur_buf_idx),  # (kv_blk_idx, buf_idx)
        )
        next_seq_idx = lax.select(q_end <= q_len_end, cur_seq_idx + 1, cur_seq_idx)
        done = lax.select(q_end < q_len_end, done, 1)
        return done, next_seq_idx, next_buf_idx

    _, seq_idx, buf_idx = lax.while_loop(
        is_cur_q_blk_needed,
        compute_with_cur_q_blk,
        (0, init_seq_idx, init_buf_idx),  # (done, seq_idx, buf_idx)
    )
    # Reset seq_idx for next kv_heads_blk if run out of seqs!
    seq_buf_idx_ref[0] = lax.select(seq_idx < num_seqs, seq_idx, 0)
    seq_buf_idx_ref[1] = buf_idx
    o_ref[...] = acc_ref[...].astype(q_ref.dtype)


def get_dtype_packing(dtype):
    bits = dtypes.bit_width(dtype)
    return 32 // bits


def get_min_heads_per_blk(num_q_heads, num_kv_heads, q_dtype, kv_dtype):
    q_packing = get_dtype_packing(q_dtype)
    kv_packing = get_dtype_packing(kv_dtype)

    def can_be_xla_fully_tiled(x, packing):
        if x % packing != 0:
            return False
        x //= packing
        return x in (1, 2, 4, 8) or x % 8 == 0

    if not can_be_xla_fully_tiled(num_kv_heads, kv_packing):
        raise ValueError(
            f"Not implemented: {num_kv_heads=} can not be XLA fully tiled."
        )
    assert (
        num_q_heads % num_kv_heads == 0
    ), f"num_q_heads is not divisible by num_kv_heads, {num_q_heads=}, {num_kv_heads=}"
    ratio = num_q_heads // num_kv_heads
    # second minor tiling is not on.
    max_kv_tiling = 8 * kv_packing
    min_kv_heads = max_kv_tiling if num_kv_heads % max_kv_tiling == 0 else num_kv_heads
    min_q_heads = min_kv_heads * ratio
    if can_be_xla_fully_tiled(min_q_heads, q_packing):
        return min_q_heads, min_kv_heads
    return num_q_heads, num_kv_heads


@functools.partial(
    jax.jit,
    static_argnames=[
        "sm_scale",
        "mask_value",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
        "sliding_window",
        "soft_cap",
        "k_scale",
        "v_scale",
    ],
)
def ragged_paged_attention(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_cache_fused: jax.Array,  # [total_num_pages, page_size, num_kv_heads, head_dim * 2]
    page_indices: jax.Array,  # i32[num_pages]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    cu_kv_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    seq_lens: jax.Array,  # i32[padded_num_seqs]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
    """Ragged paged attention that supports mixed prefill and decode.

    Args:
      q: concatenated all sequences' queries.
      k_cache, v_cache: paged KV cache. Normally in HBM.
      kv_lens: padded kv lengths. Only the first num_seqs values are valid.
      page_indices: the first index indicates which page to use in the kv cache
        for each sequence. Only the first num_seqs values are valid.
      cu_q_lens: the cumulative sum of the effective query lengths. Similar to
        kv_lens, only the first num_seqs+1 values are valid.
      num_seqs: the dynamic number of sequences.
      sm_scale: the softmax scale which will be applied to the Q@K^T.
      sliding_window: the sliding window size for the attention.
      soft_cap: the logit soft cap for the attention.
      mask_value: mask value for causal mask.
      k_scale: the scale for the key cache.
      v_scale: the scale for the value cache.
      num_kv_pages_per_block: number of kv pages to be processed in one flash
        attention block in the pallas kernel.
      num_queries_per_block: number of kv pages to be processed in one flash
        attention block in the pallas kernel.
      vmem_limit_bytes: the vmem limit for the pallas kernel.

    Returns:
      The output of the attention.
    """
    # TODO: Update static_validate_inputs for fused KV cache
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE
    num_q_tokens, num_q_heads, head_dim = q.shape
    _, page_size, num_kv_heads, fused_head_dim = kv_cache_fused.shape
    assert (
        fused_head_dim == head_dim * 2
    ), f"Expected fused head_dim={head_dim * 2}, got {fused_head_dim}"
    num_q_heads_per_blk, num_kv_heads_per_blk = get_min_heads_per_blk(
        num_q_heads, num_kv_heads, q.dtype, kv_cache_fused.dtype
    )

    num_q_per_blk = num_queries_per_block
    num_kv_pages_per_blk = num_kv_pages_per_block
    if num_q_per_blk is None or num_kv_pages_per_blk is None:
        num_kv_pages_per_blk, num_q_per_blk = get_tuned_block_sizes(
            q.dtype,
            kv_cache_fused.dtype,
            num_kv_heads,
            head_dim,
            page_size,
        )

    num_q_heads_per_kv_head = num_q_heads // num_kv_heads
    num_q_blks = cdiv(num_q_tokens, num_q_per_blk)
    assert num_q_heads_per_blk % num_q_heads_per_kv_head == 0
    num_heads_blks = num_q_heads // num_q_heads_per_blk
    grid = (num_heads_blks, num_q_blks)

    def q_index_map(heads_blk_idx, q_blk_idx, *_):
        return (q_blk_idx, heads_blk_idx, 0)

    q_block_spec = pl.BlockSpec(
        (num_q_per_blk, num_q_heads_per_blk, head_dim),
        q_index_map,
    )
    in_specs = [
        q_block_spec,
        pl.BlockSpec(memory_space=pltpu.ANY),  # fused kv_cache
    ]
    out_specs = q_block_spec
    lm_scratch = pltpu.VMEM(
        # unaligned slicing!
        (num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128),
        jnp.float32,
    )
    acc_scratch = pltpu.VMEM(
        (num_q_per_blk, num_q_heads_per_blk, head_dim),
        jnp.float32,
    )
    double_fused_kv_buf_scratch = pltpu.VMEM(
        (
            2,  # For double buffering during DMA copies.
            num_kv_pages_per_blk,
            page_size,
            num_kv_heads_per_blk,
            head_dim * 2,  # Fused K and V
        ),
        kv_cache_fused.dtype,
    )
    scratch_shapes = [
        double_fused_kv_buf_scratch,  # fused kv_bufs
        pltpu.SemaphoreType.DMA((2, 2)),  # Semaphores for double buffers.
        lm_scratch,  # l_ref
        lm_scratch,  # m_ref
        acc_scratch,
    ]
    scalar_prefetches = (
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        jnp.array((0, 0), jnp.int32),  # seq_idx, buf_idx
        num_seqs,
        seq_lens,
    )
    kernel = pl.pallas_call(
        functools.partial(
            ragged_paged_attention_kernel,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            mask_value=mask_value,
            k_scale=k_scale,
            v_scale=v_scale,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_prefetches),
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.TPUCompilerParams(
            dimension_semantics=(
                "arbitrary",
                "arbitrary",
            ),
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
        out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
        name="ragged_paged_attention_kernel",
    )

    return kernel(*scalar_prefetches, q, kv_cache_fused)
