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
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


TUNED_BLOCK_SIZES = {
    "TPU v6": {
        # (q_dtype, kv_dtype, num_kv_heads_per_blk, head_dim, page_size)
        ("bfloat16", "bfloat16", 2, 128, 64): (32, 64),
        ("bfloat16", "bfloat16", 4, 128, 64): (32, 64),
        ("bfloat16", "bfloat16", 8, 128, 64): (16, 64),
        ("bfloat16", "bfloat16", 16, 128, 64): (16, 64),
        ("bfloat16", "bfloat16", 32, 128, 64): (16, 64),
        ("bfloat16", "bfloat16", 2, 128, 128): (16, 32),
        ("bfloat16", "bfloat16", 4, 128, 128): (8, 32),
        ("bfloat16", "bfloat16", 8, 128, 128): (16, 32),
        ("bfloat16", "bfloat16", 16, 128, 128): (8, 32),
        ("bfloat16", "bfloat16", 32, 128, 128): (8, 32),
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
        k_pages_hbm_ref,  # [total_num_pages, page_size, num_kv_heads_per_blk, head_dim]
        k_vmem_buf,  # [num_kv_pages_per_blk, page_size, num_kv_heads_per_blk, head_dim]
        v_pages_hbm_ref,  # [total_num_pages, page_size, num_kv_heads_per_blk, head_dim]
        v_vmem_buf,  # [num_kv_pages_per_blk, page_size, num_kv_heads_per_blk, head_dim]
        k_sem,
        v_sem,
        page_indices_ref,  # i32[num_pages]
        metadata,  # [start_page_idx, end_page_idx]
    ):
        self._k_vmem_buf = k_vmem_buf
        self._v_vmem_buf = v_vmem_buf
        start_page_idx, end_page_idx = metadata
        self._k_async_copies = []
        self._v_async_copies = []
        for i in range(k_vmem_buf.shape[0]):
            page_idx = start_page_idx + i
            page_idx = jax.lax.select(page_idx < end_page_idx, page_idx, 0)
            self._k_async_copies.append(
                pltpu.make_async_copy(
                    k_pages_hbm_ref.at[page_indices_ref[page_idx]],
                    k_vmem_buf.at[i],
                    k_sem,
                )
            )
            self._v_async_copies.append(
                pltpu.make_async_copy(
                    v_pages_hbm_ref.at[page_indices_ref[page_idx]],
                    v_vmem_buf.at[i],
                    v_sem,
                )
            )

    def start(self):
        """Starts the async copies."""
        for async_copy in self._k_async_copies:
            async_copy.start()
        for async_copy in self._v_async_copies:
            async_copy.start()

    def wait(self):
        for async_copy in self._k_async_copies:
            async_copy.wait()
        for async_copy in self._v_async_copies:
            async_copy.wait()
        return self._k_vmem_buf, self._v_vmem_buf


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
    k_cache_hbm_ref,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    v_cache_hbm_ref,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    # Output
    o_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    # Scratch
    k_bufs,  # [2, num_kv_pages_per_blk, page_size, num_k_heads_per_blk, head_dim]
    v_bufs,  # [2, num_kv_pages_per_blk, page_size, num_v_heads_per_blk, head_dim]
    k_sems,  # [2, 2]
    v_sems,  # [2, 2]
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
    assert k_bufs.shape == v_bufs.shape
    _, num_kv_pages_per_blk, page_size, num_kv_heads_per_blk, _ = k_bufs.shape
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

    def create_kv_async_copy_descriptors(heads_blk_idx, seq_idx, kv_blk_idx, buf_idx):
        start_kv_page_idx = (
            cdiv(cu_kv_lens_ref[seq_idx], page_size) + kv_blk_idx * num_kv_pages_per_blk
        )
        end_kv_page_idx = cdiv(cu_kv_lens_ref[seq_idx + 1], page_size)
        metadata = (start_kv_page_idx, end_kv_page_idx)
        heads_start = heads_blk_idx * num_kv_heads_per_blk
        async_copy_kv = MultiPageAsyncCopyDescriptor(
            k_cache_hbm_ref.at[:, :, pl.ds(heads_start, num_kv_heads_per_blk), :],
            k_bufs.at[buf_idx],
            v_cache_hbm_ref.at[:, :, pl.ds(heads_start, num_kv_heads_per_blk), :],
            v_bufs.at[buf_idx],
            k_sems.at[buf_idx],
            v_sems.at[buf_idx],
            page_indices_ref,
            metadata,
        )
        return async_copy_kv

    def fold_on_2nd_minor(vec):
        assert vec.dtype == jnp.bfloat16 or vec.dtype == jnp.float32
        assert len(vec.shape) >= 2
        last_dim = vec.shape[-1]
        packing = get_dtype_packing(vec.dtype)
        if vec.shape[-2] % packing != 0:
            vec = vec.astype(jnp.float32)
        return vec.reshape(-1, last_dim)

    @pl.when(heads_blk_idx + q_blk_idx == 0)
    def prefetch_first_kv_blk():
        async_copy_kv = create_kv_async_copy_descriptors(
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
            q,  # [num_q_per_blk * num_q_heads_per_kv_head, head_dim]
            k,  # [num_kv_per_blk, head_dim]
            v,  # [num_kv_per_blk, head_dim]
            head_l_ref,  # [num_q_per_blk * num_q_heads_per_kv_head, 128]
            head_m_ref,  # [num_q_per_blk * num_q_heads_per_kv_head, 128]
            head_acc_ref,  # [num_q_per_blk, num_q_heads_per_kv_head, head_dim]
            *,
            kv_blk_idx,
            actual_kv_len,
        ):
            assert q.shape == (
                num_q_per_blk * num_q_heads_per_kv_head,
                head_dim,
            ), f"q.shape is not correct, {q.shape=}, expected shape=({num_q_per_blk}, {num_q_heads_per_kv_head}, {head_dim})."
            assert (
                k.shape
                == v.shape
                == (
                    num_kv_per_blk,
                    head_dim,
                )
            ), f"k.shape or v.shape is not correct, {k.shape=} {v.shape=}, expected shape=({num_kv_per_blk}, {head_dim})."
            assert k.dtype == v.dtype
            assert (
                head_m_ref.shape
                == head_l_ref.shape
                == (
                    num_q_per_blk * num_q_heads_per_kv_head,
                    128,
                )
            ), f"head_m_ref.shape or head_l_ref.shape is not correct, {head_m_ref.shape=} {head_l_ref.shape=}, expected shape=({num_q_per_blk*num_q_heads_per_kv_head}, 128)."
            assert head_acc_ref.shape == (
                num_q_per_blk,
                num_q_heads_per_kv_head,
                head_dim,
            ), f"head_acc_ref.shape is not correct, {head_acc_ref.shape=}, expected shape=({num_q_per_blk}, {num_q_heads_per_kv_head}, {head_dim})."
            kv_len_start = kv_blk_idx * num_kv_per_blk

            def masked_store(ref, val, start, end, group=1):
                iota = lax.broadcasted_iota(jnp.int32, ref.shape, 0) // group
                mask = jnp.logical_and(iota >= start, iota < end)
                pl.store(
                    ref, idx=tuple(slice(None) for _ in ref.shape), val=val, mask=mask
                )

            def load_with_init(ref, init_val):
                return jnp.where(
                    kv_blk_idx == 0, jnp.full_like(ref, init_val), ref[...]
                )

            effective_kv_len = actual_kv_len - kv_len_start
            kv_mask = lax.broadcasted_iota(jnp.int32, k.shape, 0) < effective_kv_len
            k = jnp.where(kv_mask, k.astype(jnp.float32), 0).astype(k.dtype)
            v = jnp.where(kv_mask, v.astype(jnp.float32), 0).astype(v.dtype)

            qk = (
                jnp.einsum("nd,md->nm", q, k, preferred_element_type=jnp.float32)
                * sm_scale
            )
            store_start = jnp.maximum(q_start - q_len_start, 0)
            store_end = jnp.minimum(q_end - q_len_start, num_q_per_blk)

            row_ids = (
                (actual_kv_len - q_len)
                + q_len_start
                - q_start
                + jax.lax.broadcasted_iota(
                    jnp.int32,
                    (num_q_per_blk * num_q_heads_per_kv_head, num_kv_per_blk),
                    0,
                )
                // num_q_heads_per_kv_head
            )
            col_ids = kv_len_start + jax.lax.broadcasted_iota(
                jnp.int32,
                (num_q_per_blk * num_q_heads_per_kv_head, num_kv_per_blk),
                1,
            )
            causal_mask = row_ids < col_ids
            if sliding_window is not None:
                causal_mask = jnp.logical_or(
                    causal_mask, row_ids - sliding_window >= col_ids
                )
            if soft_cap is not None:
                qk = soft_cap * jnp.tanh(qk / soft_cap)
            qk += jnp.where(causal_mask, mask_value, 0.0)
            m_curr = jnp.max(qk, axis=1, keepdims=True)
            s_curr = jnp.exp(qk - m_curr)
            qkv = jnp.dot(s_curr, v, preferred_element_type=jnp.float32)
            lm_store_shape = head_m_ref.shape
            m_curr = jnp.broadcast_to(m_curr, lm_store_shape)
            l_curr = jnp.broadcast_to(s_curr.sum(axis=1, keepdims=True), lm_store_shape)
            m_prev = load_with_init(head_m_ref, -jnp.inf)
            l_prev = load_with_init(head_l_ref, 0.0)
            m_next = jnp.maximum(m_prev, m_curr)
            masked_store(
                head_m_ref, m_next, store_start, store_end, num_q_heads_per_kv_head
            )
            alpha = jnp.exp(m_prev - m_next)
            beta = jnp.exp(m_curr - m_next)
            l_alpha = alpha * l_prev
            l_next = l_alpha + beta * l_curr
            l_next_safe = jnp.where(l_next == 0.0, 1.0, l_next)
            masked_store(
                head_l_ref,
                l_next_safe,
                store_start,
                store_end,
                num_q_heads_per_kv_head,
            )

            def broadcast_to_shape(arr, shape):
                if arr.shape == shape:
                    return arr
                assert len(arr.shape) == len(shape)
                assert arr.shape[0] == shape[0]
                assert shape[1] % arr.shape[1] == 0
                # no-op concatenation.
                return jnp.concatenate(
                    [arr for _ in range(shape[1] // arr.shape[1])], axis=1
                )

            o_curr = load_with_init(head_acc_ref, 0.0).reshape(-1, head_dim)
            l_alpha = broadcast_to_shape(l_alpha, qkv.shape)
            beta = broadcast_to_shape(beta, qkv.shape)
            l_next_safe = broadcast_to_shape(l_next_safe, qkv.shape)
            out = lax.div(
                l_alpha * o_curr + beta * qkv,
                l_next_safe,
            )
            masked_store(
                head_acc_ref,
                out.reshape(head_acc_ref.shape),
                store_start,
                store_end,
            )

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
                # TODO(jevinjiang): reuse the same buffer if it is already prefetched!
                # TODO(jevinjiang): only fetch effective dynamic size to hold kv_len and
                # DMA to fixed size buffer!
                next_async_copy_kv = create_kv_async_copy_descriptors(
                    next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx
                )
                next_async_copy_kv.start()

            cur_async_copy_kv = create_kv_async_copy_descriptors(
                heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
            )
            k_ref, v_ref = cur_async_copy_kv.wait()
            k_ref = k_ref.reshape(
                num_kv_pages_per_blk * page_size,
                num_kv_heads_per_blk,
                head_dim,
            )
            v_ref = v_ref.reshape(
                num_kv_pages_per_blk * page_size,
                num_kv_heads_per_blk,
                head_dim,
            )
            for kv_head_chunk_idx in range(0, num_kv_heads_per_blk):
                k = k_ref[:, kv_head_chunk_idx, :]
                v = v_ref[:, kv_head_chunk_idx, :]
                if k_scale is not None:
                    # NOTE: Conversion between arbitrary data types is not supported.
                    # That's why it is converted to float32 first.
                    k = k.astype(jnp.float32) * k_scale
                    k = k.astype(q_ref.dtype)
                if v_scale is not None:
                    v = v.astype(jnp.float32) * v_scale
                    v = v.astype(q_ref.dtype)
                kv_head_idx = kv_head_chunk_idx
                q_head_idx = kv_head_idx * num_q_heads_per_kv_head
                # TODO(jevinjiang): extra handling for packed type that can start at
                # unaligned position!
                q = fold_on_2nd_minor(
                    q_ref[:, q_head_idx : q_head_idx + num_q_heads_per_kv_head, :]
                )
                flash_attention(
                    q,
                    k,
                    v,
                    l_ref.at[kv_head_idx],
                    m_ref.at[kv_head_idx],
                    acc_ref.at[:, q_head_idx : q_head_idx + num_q_heads_per_kv_head, :],
                    kv_blk_idx=kv_blk_idx,
                    actual_kv_len=actual_kv_len,
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


def cdiv(a, b):
    assert b != 0, f"b is equal to 0, {b=}"
    return (a + b - 1) // b


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

    # TODO(jevinjiang): support unaligned number of heads!
    if not can_be_xla_fully_tiled(num_kv_heads, kv_packing):
        raise ValueError(
            f"Not implemented: {num_kv_heads=} can not be XLA fully tiled."
        )
    assert (
        num_q_heads % num_kv_heads == 0
    ), f"num_q_heads is not divisible by num_kv_heads, {num_q_heads=}, {num_kv_heads=}"
    ratio = num_q_heads // num_kv_heads
    # TODO(jevinjiang): we can choose smaller tiling for packed type if large
    # second minor tiling is not on.
    max_kv_tiling = 8 * kv_packing
    min_kv_heads = max_kv_tiling if num_kv_heads % max_kv_tiling == 0 else num_kv_heads
    min_q_heads = min_kv_heads * ratio
    if can_be_xla_fully_tiled(min_q_heads, q_packing):
        return min_q_heads, min_kv_heads
    return num_q_heads, num_kv_heads


def ragged_paged_attention(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    k_cache: jax.Array,  # [total_num_pages, page_size, num_k_heads, head_dim]
    v_cache: jax.Array,  # [total_num_pages, page_size, num_v_heads, head_dim]
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
    static_validate_inputs(
        q,
        k_cache,
        v_cache,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        num_seqs,
        seq_lens,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        k_scale=k_scale,
        v_scale=v_scale,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE
    num_q_tokens, num_q_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape
    num_q_heads_per_blk, num_kv_heads_per_blk = get_min_heads_per_blk(
        num_q_heads, num_kv_heads, q.dtype, k_cache.dtype
    )

    num_q_per_blk = num_queries_per_block
    num_kv_pages_per_blk = num_kv_pages_per_block
    if num_q_per_blk is None or num_kv_pages_per_blk is None:
        num_kv_pages_per_blk, num_q_per_blk = get_tuned_block_sizes(
            q.dtype,
            k_cache.dtype,
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
        pl.BlockSpec(memory_space=pltpu.ANY),
        pl.BlockSpec(memory_space=pltpu.ANY),
    ]
    out_specs = q_block_spec
    lm_scratch = pltpu.VMEM(
        # TODO(jevinjiang): use 128 instead of 1 is due to Mosaic does not support
        # unaligned slicing!
        (num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128),
        jnp.float32,
    )
    acc_scratch = pltpu.VMEM(
        (num_q_per_blk, num_q_heads_per_blk, head_dim),
        jnp.float32,
    )
    double_kv_buf_scratch = pltpu.VMEM(
        (
            2,  # For double buffering during DMA copies.
            num_kv_pages_per_blk,
            page_size,
            num_kv_heads_per_blk,
            head_dim,
        ),
        k_cache.dtype,
    )
    scratch_shapes = [
        double_kv_buf_scratch,  # k_bufs
        double_kv_buf_scratch,  # v_bufs
        pltpu.SemaphoreType.DMA((2,)),  # Semaphores for k double buffers.
        pltpu.SemaphoreType.DMA((2,)),  # Semaphores for k double buffers.
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
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                "arbitrary",
                "arbitrary",
            ),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
        name="ragged_paged_attention_kernel",
    )

    return kernel(*scalar_prefetches, q, k_cache, v_cache)
