"""TPU-Friendly Ragged Paged Attention kernel.

This kernel offers a highly optimized implementation of ragged paged attention,
specifically designed for TPU and compatible with a wide range of model
specifications. It supports mixed prefill and decoding, enhancing throughput
during inference.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.layers.attention.flash_attn_kernel.tuned_block_sizes import (
    get_tuned_block_sizes,
)
from sgl_jax.srt.layers.attention.flash_attn_kernel.util import (
    align_to,
    cdiv,
    get_dtype_packing,
)

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


def ref_ragged_paged_attention_fused(
    queries: jax.Array,  # [padded_num_tokens, num_q_heads, head_dim]
    kv_pages_fused: jax.Array,  # [total_num_pages, page_size, num_kv_heads * 2, head_dim]
    kv_lens: jax.Array,  # i32[padded_batch_size]
    page_indices: jax.Array,  # i32[(padded_batch_size * model_context_len + page_size - 1) // page_size]
    cu_q_lens: jax.Array,  # i32[padded_batch_size + 1]
    num_seqs: jax.Array,  # i32[1],
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
    xai_temperature_len: float = -1.0,
):
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE
    _, _, num_kv_heads_interleaved, head_dim = kv_pages_fused.shape
    assert num_kv_heads_interleaved % 2 == 0
    num_kv_heads = num_kv_heads_interleaved // 2
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

        kv_fused = kv_pages_fused[indices, :, :, :].reshape(
            -1, num_kv_heads_interleaved, head_dim
        )[:kv_len]

        # Head format: [K1, V1, K2, V2, ...]
        k = kv_fused[:, 0::2, :]  # indices 0, 2, 4, ...
        v = kv_fused[:, 1::2, :]  # indices 1, 3, 5, ...

        if k_scale is not None:
            k = (k.astype(jnp.float32) * k_scale).astype(q.dtype)
        if v_scale is not None:
            v = (v.astype(jnp.float32) * v_scale).astype(q.dtype)

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

        # xai_temperature_scale: ref implementation from sgl-project/sglang
        # python/sglang/srt/layers/attention/triton_ops/decode_attention.py
        if xai_temperature_len is not None:
            # xai_temperature_scale = 1.0 / jnp.log2(jnp.array(xai_temperature_len, float))
            xai_temperature_scale = 1.0 / jnp.log2(float(xai_temperature_len))
            qidx = jnp.arange(q_start, q_end) - 1
            _qtemp = jnp.log2(qidx.astype(jnp.float32)) * xai_temperature_scale
            xai_temperature_reg = jnp.where(qidx > xai_temperature_len, _qtemp, 1.0)
            attn = attn * xai_temperature_reg[:, None]

        attn += jnp.where(mask, mask_value, 0.0)
        attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
        out = jnp.einsum("hqk,khd->qhd", attn, v).astype(queries.dtype)
        outputs.append(out)

    return jnp.concatenate(outputs, axis=0)


def ref_ragged_paged_attention(
    queries: jax.Array,  # [padded_num_tokens, num_q_heads, head_dim]
    k_pages: jax.Array,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    v_pages: jax.Array,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[padded_batch_size]
    page_indices: jax.Array,  # i32[(padded_batch_size * model_context_len + page_size - 1) // page_size]
    cu_q_lens: jax.Array,  # i32[padded_batch_size + 1]
    num_seqs: jax.Array,  # i32[1],
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
    xai_temperature_len: float | None = 3,#None,
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

        # xai_temperature_scale: ref implementation from sgl-project/sglang
        # python/sglang/srt/layers/attention/triton_ops/decode_attention.py
        if xai_temperature_len is not None:
            # xai_temperature_scale = 1.0 / jnp.log2(jnp.array(xai_temperature_len, float))
            xai_temperature_scale = 1.0 / jnp.log2(float(xai_temperature_len))
            qidx = jnp.arange(q_start, q_end) - 1
            _qtemp = jnp.log2(qidx.astype(jnp.float32)) * xai_temperature_scale
            xai_temperature_reg = jnp.where(qidx > xai_temperature_len, _qtemp, 1.0)
            attn = attn * xai_temperature_reg[:, None]

        attn += jnp.where(mask, mask_value, 0.0)
        attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
        out = jnp.einsum("hqk,khd->qhd", attn, v).astype(queries.dtype)
        outputs.append(out)

    return jnp.concatenate(outputs, axis=0)


def get_smem_estimate_bytes(max_num_seqs, pages_per_seq):
    total_bits = (
        # kv_lens_ref: i32[max_num_seqs]
        align_to(max_num_seqs, 128) * 32
        +
        # page_indices_ref: i32[max_num_seqs * pages_per_seq]
        align_to(max_num_seqs * pages_per_seq, 128) * 32
        +
        # cu_q_lens_ref: i32[max_num_seqs + 1]
        align_to(max_num_seqs + 1, 128) * 32
        +
        # distribution_ref: i32[3]
        128 * 32
        +
        # sem_ids_ref: i32[3]
        128 * 32
        +
        # bo_ids_ref: i32[4]
        128 * 32
        +
        # bkv_update_ids_ref: i32[6]
        128 * 32
    )
    return cdiv(total_bits, 8)


def get_vmem_estimate_bytes(
    actual_num_kv_heads,
    actual_num_q_heads_per_kv_head,
    actual_head_dim,
    bq_sz,
    bkv_sz,
    q_dtype,
    kv_dtype,
):
    q_packing = get_dtype_packing(q_dtype)
    kv_packing = get_dtype_packing(kv_dtype)
    num_q_heads_per_kv_head = align_to(actual_num_q_heads_per_kv_head, q_packing)
    num_kv_heads_x2 = align_to(actual_num_kv_heads * 2, kv_packing)
    head_dim = align_to(actual_head_dim, 128)

    total_bits = (
        # bkv_x2_ref
        (2 * bkv_sz * num_kv_heads_x2 * head_dim) * (32 // kv_packing)
        +
        # bq_x2_ref + bo_x2_ref
        2
        * (2 * actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head * head_dim)
        * (32 // q_packing)
        +
        # l_ref + m_ref
        2 * (actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head * 128) * 32
        +
        # acc_ref
        (actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head * head_dim) * 32
    )
    return cdiv(total_bits, 8)


def get_kv_cache_shape(
    total_num_pages,
    page_size,
    actual_num_kv_heads,
    actual_head_dim,
    kv_dtype,
):
    kv_packing = get_dtype_packing(kv_dtype)
    return (
        total_num_pages,
        page_size,
        align_to(actual_num_kv_heads * 2, kv_packing) // kv_packing,
        kv_packing,
        align_to(actual_head_dim, 128),
    )


def _ragged_paged_attention_kernel(
    # Prefetch
    kv_lens_ref,  # [padded_batch_size]
    page_indices_ref,  # [(padded_batch_size * model_context_len + page_size - 1) // page_size]
    cu_q_lens_ref,  # [padded_batch_size + 1]
    cu_kv_lens_ref,  # [padded_batch_size + 1]
    distribution_ref,  # [3] (decode_end, prefill_end, mixed_end)
    sem_ids_ref,  # [3] (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
    bo_ids_ref,  # [4] (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx)
    bkv_update_ids_ref,  # [6] (bkv_sem_0_seq_idx, bkv_sem_1_seq_idx, bkv_sem_0_offset, bkv_sem_1_offset, bkv_sem_0_sz, bkv_sem_1_sz)
    # Input
    q_hbm_ref,  # [actual_num_kv_heads, padded_num_tokens, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    kv_hbm_ref,  # [padded_num_tokens, num_kv_heads_x2 // kv_packing, kv_packing, head_dim] - Fused KV with interleaved [K1,V1,K2,V2,...]
    kv_cache_fused_hbm_ref,  # [total_num_pages, page_size, num_kv_heads_interleaved // kv_packing, kv_packing, head_dim]
    # Output
    o_hbm_ref,  # [actual_num_kv_heads, max_num_tokens, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    updated_kv_cache_fused_hbm_ref,  # [total_num_pages, page_size, num_kv_heads_interleaved // kv_packing, kv_packing, head_dim]
    # Scratch
    bkv_fused_x2_ref,  # [2, bkv_sz, num_kv_heads_interleaved // kv_packing, kv_packing, head_dim]
    bq_x2_ref,  # [2, actual_num_kv_heads, bq_sz, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    bo_x2_ref,  # [2, actual_num_kv_heads, bq_sz, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    sems,  # [4, 2]
    l_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128],
    m_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128],
    acc_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, head_dim],
    *,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    xai_temperature_len: float | None = 3,#None,
    chunk_prefill_size: int | None = None,
    bkv_p,
    bq_sz,
):
    assert q_hbm_ref.shape == o_hbm_ref.shape
    assert (
        q_hbm_ref.shape[-1] == kv_cache_fused_hbm_ref.shape[-1]
    )  # head_dim should match
    (
        actual_num_kv_heads,
        max_num_tokens,
        num_q_heads_per_kv_head_per_packing,
        q_packing,
        head_dim,
    ) = q_hbm_ref.shape
    (
        total_num_pages,
        page_size,
        num_kv_heads_per_kv_packing,
        kv_packing,
        _,
    ) = kv_cache_fused_hbm_ref.shape
    max_num_seqs = kv_lens_ref.shape[0]
    num_page_indices = page_indices_ref.shape[0]
    pages_per_seq = num_page_indices // max_num_seqs
    num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_packing * q_packing
    q_dtype = q_hbm_ref.dtype
    kv_dtype = kv_cache_fused_hbm_ref.dtype
    assert o_hbm_ref.dtype == q_dtype
    assert get_dtype_packing(q_dtype) == q_packing
    assert get_dtype_packing(kv_dtype) == kv_packing
    assert head_dim % 128 == 0
    bkv_sz = bkv_p * page_size
    seq_idx = pl.program_id(0)
    num_seqs = pl.num_programs(0)
    decode_end = distribution_ref[0]
    prefill_end = distribution_ref[1]
    mixed_end = distribution_ref[2]

    q_start = cu_q_lens_ref[seq_idx]
    q_end = cu_q_lens_ref[seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens_ref[seq_idx]

    def _async_copy(src, dst, sem, wait):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False):
        sem = sems.at[0, bkv_sem_idx]
        kv_fused_vmem_ref = bkv_fused_x2_ref.at[bkv_sem_idx]

        kv_fused_cache_hbm_shape = kv_cache_fused_hbm_ref.shape
        kv_fused_cache_hbm_ref = kv_cache_fused_hbm_ref.reshape(
            kv_fused_cache_hbm_shape[0] * kv_fused_cache_hbm_shape[1],
            *kv_fused_cache_hbm_shape[2:],
        )
        kv_len = kv_lens_ref[seq_idx]
        kv_len_start = bkv_idx * bkv_sz
        kv_p_start = bkv_idx * bkv_p
        q_start = cu_q_lens_ref[seq_idx]
        q_end = cu_q_lens_ref[seq_idx + 1]
        q_len = q_end - q_start

        kv_left = kv_len - kv_len_start
        kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)
        kv_left_frm_new = kv_left - kv_left_frm_cache
        bkv_p_frm_cache = jnp.minimum(cdiv(kv_left_frm_cache, page_size), bkv_p)
        bkv_sz_frm_new = jnp.minimum(
            jnp.maximum(bkv_sz - kv_left_frm_cache, 0), kv_left_frm_new
        )

        start_kv_page_idx = cdiv(cu_kv_lens_ref[seq_idx], page_size)
        page_indices_offset = start_kv_page_idx + kv_p_start

        # Make sure the current bkv buffer is safe to overwrite.
        wait_update_kv_cache(bkv_sem_idx)

        def loop_body(i, offset):
            sz = jnp.minimum(page_size, kv_left_frm_cache - i * page_size)
            _async_copy(
                kv_fused_cache_hbm_ref.at[
                    pl.ds(page_indices_ref[page_indices_offset + i] * page_size, sz)
                ],
                kv_fused_vmem_ref.at[pl.ds(i * page_size, sz)],
                sem,
                wait,
            )
            return offset + sz

        offset = lax.fori_loop(
            0,
            bkv_p_frm_cache,
            loop_body,
            0,  # offset
            unroll=False,
        )

        # Fetch fused kv from new kv.
        @pl.when(bkv_sz_frm_new > 0)
        def _fetch_bkv_from_new_kv():
            new_kv_len_start = q_end - kv_left_frm_new
            _async_copy(
                kv_hbm_ref.at[pl.ds(new_kv_len_start, bkv_sz_frm_new)],
                kv_fused_vmem_ref.at[pl.ds(offset, bkv_sz_frm_new)],
                sem,
                wait,
            )

        return kv_len_start + offset, bkv_sz_frm_new

    def _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, *, wait=False):
        sem = sems.at[3, bkv_sem_idx]
        kv_fused_vmem_ref = bkv_fused_x2_ref.at[bkv_sem_idx]
        bkv_id = offset // bkv_sz
        kv_p_start = offset // page_size
        kv_p_end = cdiv(offset + update_sz, page_size)
        ignore = offset % page_size
        p_ignore = kv_p_start - bkv_id * bkv_p
        start_kv_page_idx = cdiv(cu_kv_lens_ref[seq_idx], page_size)
        page_indices_offset = start_kv_page_idx + kv_p_start

        # Fused KV cache HBM reference for updates
        kv_fused_cache_hbm_shape = updated_kv_cache_fused_hbm_ref.shape
        kv_fused_cache_hbm_ref = updated_kv_cache_fused_hbm_ref.reshape(
            kv_fused_cache_hbm_shape[0] * kv_fused_cache_hbm_shape[1],
            *kv_fused_cache_hbm_shape[2:],
        )

        def loop_body(i, states):
            update_sz, ignore = states
            sz = jnp.minimum(page_size - ignore, update_sz)
            _async_copy(
                kv_fused_vmem_ref.at[pl.ds((p_ignore + i) * page_size + ignore, sz)],
                kv_fused_cache_hbm_ref.at[
                    pl.ds(
                        page_indices_ref[page_indices_offset + i] * page_size + ignore,
                        sz,
                    )
                ],
                sem,
                wait,
            )
            return update_sz - sz, 0

        lax.fori_loop(
            0,
            kv_p_end - kv_p_start,
            loop_body,
            (update_sz, ignore),  # total transfer size
            unroll=False,
        )

    def _fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False):
        sem = sems.at[1, bq_sem_idx]
        vmem_ref = bq_x2_ref.at[bq_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        _async_copy(
            q_hbm_ref.at[:, pl.ds(q_len_start, sz)],
            vmem_ref.at[:, pl.ds(0, sz)],
            sem,
            wait,
        )

    def _send_bo(seq_idx, bo_idx, bo_sem_idx, *, wait=False):
        sem = sems.at[2, bo_sem_idx]
        vmem_ref = bo_x2_ref.at[bo_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bo_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        _async_copy(
            vmem_ref.at[:, pl.ds(0, sz)],
            o_hbm_ref.at[:, pl.ds(q_len_start, sz)],
            sem,
            wait,
        )

    def start_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        return _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

    def wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        return _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, wait=True)

    def start_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx)

    def wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx, wait=True)

    def start_send_bo(seq_idx, bo_idx, bo_sem_idx):
        bo_ids_ref[bo_sem_idx] = seq_idx
        bo_ids_ref[bo_sem_idx + 2] = bo_idx
        _send_bo(seq_idx, bo_idx, bo_sem_idx)

    def wait_send_bo(bo_sem_idx):
        old_seq_idx = bo_ids_ref[bo_sem_idx]
        old_bo_idx = bo_ids_ref[bo_sem_idx + 2]

        @pl.when(jnp.logical_and(0 <= old_seq_idx, old_seq_idx <= seq_idx))
        def _():
            _send_bo(old_seq_idx, old_bo_idx, bo_sem_idx, wait=True)

    def start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz):
        bkv_update_ids_ref[bkv_sem_idx] = seq_idx
        bkv_update_ids_ref[bkv_sem_idx + 2] = offset
        bkv_update_ids_ref[bkv_sem_idx + 4] = update_sz
        _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)

    def wait_update_kv_cache(bkv_sem_idx):
        update_sz = bkv_update_ids_ref[bkv_sem_idx + 4]

        @pl.when(update_sz > 0)
        def _():
            seq_idx = bkv_update_ids_ref[bkv_sem_idx]
            offset = bkv_update_ids_ref[bkv_sem_idx + 2]
            bkv_update_ids_ref[bkv_sem_idx + 4] = 0
            _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, wait=True)

    def load_bq(bq_sem_idx, kv_head_idx, *, actual_bq_sz=bq_sz):
        q_ref = (
            bq_x2_ref.bitcast(jnp.uint32)
            .at[bq_sem_idx, kv_head_idx]
            .reshape(bq_sz * num_q_heads_per_kv_head_per_packing, head_dim)
        )
        return pltpu.bitcast(
            q_ref[: actual_bq_sz * num_q_heads_per_kv_head_per_packing], q_dtype
        )

    def strided_load(ref, start, step, *, dtype=None):
        assert get_dtype_packing(ref.dtype) == 1
        assert len(ref.shape) == 2
        r, l = ref.shape  # noqa
        assert l % 128 == 0
        folds = l // 128
        ref = ref.reshape(r * folds, 128)
        start *= folds
        step *= folds
        vec = jnp.concat([ref[start + i :: step] for i in range(folds)], axis=1)
        if dtype is not None:
            vec = pltpu.bitcast(vec, dtype)
        return vec

    def strided_load_bkv_fused(bkv_sem_idx, start, step, *, bkv_bitmask):
        assert start % kv_packing == 0
        assert step % kv_packing == 0
        start //= kv_packing
        step //= kv_packing

        kv_ref = (
            bkv_fused_x2_ref.bitcast(jnp.uint32)
            .at[bkv_sem_idx]
            .reshape(bkv_sz * step, head_dim)
        )

        def _mask_kv(k, v):
            k = pltpu.bitcast(k, jnp.uint32)
            v = pltpu.bitcast(v, jnp.uint32)
            k = k & bkv_bitmask
            v = v & bkv_bitmask
            k = pltpu.bitcast(k, kv_dtype)
            v = pltpu.bitcast(v, kv_dtype)
            return (k, v)

        if kv_packing == 1:
            k = strided_load(kv_ref, start, step, dtype=kv_dtype)
            v = strided_load(kv_ref, start + 1, step, dtype=kv_dtype)
            return [_mask_kv(k, v)]

        kv = strided_load(kv_ref, start, step)
        bitwidth = 32 // kv_packing
        repack_ty = jnp.dtype(f"uint{bitwidth}")
        lst = []
        for i in range(0, kv_packing, 2):
            k = (kv >> (i * bitwidth)).astype(repack_ty)
            v = (kv >> ((i + 1) * bitwidth)).astype(repack_ty)
            lst.append(_mask_kv(k, v))
        return lst

    def broadcast_minor(src, shape):
        if src.shape == shape:
            return src
        assert src.shape[:-1] == shape[:-1]
        assert src.shape[-1] % 128 == 0
        target_minor = align_to(shape[-1], src.shape[-1])
        # no-op concatenation.
        return jnp.concatenate(
            [src for _ in range(target_minor // src.shape[-1])], axis=-1
        )[..., : shape[-1]]

    def process(static_q_len=None):
        num_bkv = cdiv(kv_len, bkv_sz)
        if static_q_len is None:
            actual_bq_sz = bq_sz
            num_bq = cdiv(q_len, actual_bq_sz)
        else:
            actual_bq_sz = min(bq_sz, static_q_len)
            num_bq = cdiv(static_q_len, actual_bq_sz)

        def get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx):
            next_bq_idx = bq_idx + 1
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
            next_bq_sem_idx = lax.select(bq_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bq_idx, next_bq_sem_idx

        def get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx):
            next_bkv_idx = bkv_idx + 1
            is_last_bkv = next_bkv_idx == num_bkv
            next_bkv_idx = lax.select(is_last_bkv, 0, next_bkv_idx)
            next_bq_idx = lax.select(is_last_bkv, bq_idx + 1, bq_idx)
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
            next_bkv_sem_idx = lax.select(bkv_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx

        def compute_with_bq(bq_idx, _):
            bq_sem_idx = sem_ids_ref[0]
            next_seq_idx, next_bq_idx, next_bq_sem_idx = get_next_bq_ids(
                seq_idx, bq_idx, bq_sem_idx
            )

            # Prefetch next bq
            @pl.when(next_seq_idx < num_seqs)
            def prefetch_next_bq():
                sem_ids_ref[0] = next_bq_sem_idx
                start_fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

            def compute_with_bkv(bkv_idx, _):
                # Create bitmask for KV.
                assert bkv_sz % kv_packing == 0
                actual_bkv_sz = jnp.minimum(bkv_sz, kv_len - bkv_idx * bkv_sz)
                bkv_shape = (bkv_sz, head_dim)
                bkv_mask = lax.broadcasted_iota(jnp.int32, bkv_shape, 0) < actual_bkv_sz
                bkv_bitmask = pltpu.bitcast(
                    lax.select(
                        bkv_mask,
                        jnp.full(bkv_shape, 0xFFFFFFFF, dtype=jnp.uint32),
                        jnp.full(bkv_shape, 0, dtype=jnp.uint32),
                    ).astype(jnp.dtype(f"uint{32 // kv_packing}")),
                    jnp.uint32,
                )

                # Get next bkv ids.
                bkv_sem_idx = sem_ids_ref[1]
                next_seq_idx, _, next_bkv_idx, next_bkv_sem_idx = get_next_bkv_ids(
                    seq_idx, bq_idx, bkv_idx, bkv_sem_idx
                )

                # Prefetch next bkv
                @pl.when(next_seq_idx < num_seqs)
                def prefetch_next_bkv():
                    sem_ids_ref[1] = next_bkv_sem_idx
                    start_fetch_bkv(next_seq_idx, next_bkv_idx, next_bkv_sem_idx)

                # Wait for cur bq if not ready yet
                @pl.when(bkv_idx == 0)
                def wait_cur_bq():
                    wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)

                # Wait for cur bkv
                offset, update_sz = wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

                # Start updating bkv to kv cache if applicable.
                # Only needed in first bq loop.
                @pl.when(jnp.logical_and(update_sz > 0, bq_idx == 0))
                def update_cur_bkv_to_cache():
                    start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)

                # Flash attention with cur bkv and bq
                # NOTE: kv_packing is divided by 2 because k and v are packed together.

                def batch_load_all_heads_kv():
                    k_heads = []
                    v_heads = []

                    for head_idx in range(actual_num_kv_heads):
                        bkv_lst = strided_load_bkv_fused(
                            bkv_sem_idx,
                            head_idx * 2,
                            actual_num_kv_heads * 2,
                            bkv_bitmask=bkv_bitmask,
                        )

                        k_head, v_head = bkv_lst[0]
                        k_heads.append(k_head)
                        v_heads.append(v_head)

                    return jnp.stack(k_heads, axis=0), jnp.stack(v_heads, axis=0)

                def batch_prepare_queries():
                    q_heads = []
                    for head_idx in range(actual_num_kv_heads):
                        bq = load_bq(bq_sem_idx, head_idx, actual_bq_sz=actual_bq_sz)
                        q_heads.append(bq)
                    q_batch = jnp.stack(q_heads, axis=0)

                    if xai_temperature_len is not None:
                        import numpy as np

                        q_len_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
                        q_batch_shape = q_batch.shape
                        print('bq_sz', bq_sz, 'actual_bq_sz', actual_bq_sz)
                        base = (q_len_start - 1) + lax.iota(jnp.int32, q_batch_shape[1])
                        offs_qidx = jnp.tile(base, (q_batch_shape[0], 1))
                        return q_batch, offs_qidx
                    return q_batch, None

                # Load batched data
                k_batch, v_batch = batch_load_all_heads_kv()
                q_batch, offs_qidx_batch = batch_prepare_queries()
                print('q_batch.shape', q_batch.shape, xai_temperature_len)
                print('offs_qidx_batch.shape', offs_qidx_batch.shape)

                def flash_attention(q_batch, k_batch, v_batch):
                    q_batch_f32 = q_batch.astype(jnp.float32)
                    k_batch_f32 = k_batch.astype(jnp.float32)
                    v_batch_f32 = v_batch.astype(jnp.float32)

                    if k_scale is not None:
                        k_batch_f32 = k_batch_f32 * k_scale
                    if v_scale is not None:
                        v_batch_f32 = v_batch_f32 * v_scale

                    s = (
                        jnp.einsum(
                            "hqd,hkd->hqk",
                            q_batch_f32,
                            k_batch_f32,
                            preferred_element_type=jnp.float32,
                        )
                        * sm_scale
                    )

                    if q_scale is not None:
                        s *= q_scale

                    q_span = (
                        kv_len
                        - q_len
                        + bq_idx * bq_sz
                        + lax.broadcasted_iota(jnp.int32, s.shape, 1)
                        // num_q_heads_per_kv_head
                    )
                    k_span = bkv_idx * bkv_sz + lax.broadcasted_iota(
                        jnp.int32, s.shape, 2
                    )
                    mask = q_span < k_span

                    if sliding_window is not None:
                        mask = jnp.logical_or(mask, q_span - sliding_window >= k_span)

                    if soft_cap is not None:
                        s = soft_cap * jnp.tanh(s / soft_cap)

                    # xai_temperature_scale: ref implementation from sgl-project/sglang
                    # python/sglang/srt/layers/attention/triton_ops/decode_attention.py
                    if xai_temperature_len is not None:
                        # xai_temperature_scale = 1.0 / jnp.log2(jnp.array(xai_temperature_len, float))
                        xai_temperature_scale = 1.0 / jnp.log2(float(xai_temperature_len))
                        _qtemp = jnp.log2(offs_qidx_batch.astype(jnp.float32)) * xai_temperature_scale
                        xai_temperature_reg = jnp.where(offs_qidx_batch > xai_temperature_len, _qtemp, 1.0)

                        s = s * xai_temperature_reg[:, :, None]

                    s += jnp.where(mask, mask_value, 0.0)

                    for head_idx in range(actual_num_kv_heads):
                        head_l_ref = l_ref.at[head_idx, : q_batch.shape[1]]
                        head_m_ref = m_ref.at[head_idx, : q_batch.shape[1]]
                        head_acc_ref = acc_ref.at[head_idx, : q_batch.shape[1]]

                        def load_with_init(ref, init_val):
                            return jnp.where(
                                bkv_idx == 0, jnp.full_like(ref, init_val), ref[...]
                            )

                        s_head = s[head_idx]
                        s_head_rowmax = jnp.max(s_head, axis=1, keepdims=True)

                        m_prev = load_with_init(head_m_ref, -jnp.inf)
                        m_curr = jnp.maximum(m_prev, s_head_rowmax)
                        head_m_ref[...] = m_curr

                        p = jnp.exp(s_head - broadcast_minor(m_curr, s_head.shape))

                        pv = jnp.einsum(
                            "qk,kd->qd",
                            p,
                            v_batch_f32[head_idx],
                            preferred_element_type=jnp.float32,
                        )

                        p_rowsum = jnp.sum(p, axis=1, keepdims=True)
                        exp_m_diff = jnp.exp(m_prev - m_curr)
                        l_prev = load_with_init(head_l_ref, 0.0)
                        l_curr = exp_m_diff * l_prev + p_rowsum
                        head_l_ref[...] = l_curr

                        o_prev = load_with_init(head_acc_ref, 0.0)
                        o_curr = broadcast_minor(exp_m_diff, o_prev.shape) * o_prev + pv
                        head_acc_ref[...] = o_curr

                flash_attention(q_batch, k_batch, v_batch)

            lax.fori_loop(0, num_bkv, compute_with_bkv, None, unroll=False)

            # Load acc and calculate final output.
            acc = acc_ref[...]
            l = broadcast_minor(l_ref[...], acc.shape)  # noqa
            out = (
                lax.div(acc, l)
                if q_dtype == jnp.float32
                else (acc * pl.reciprocal(l, approx=True)).astype(q_dtype)
            )

            # Wait for previous bo to be fully sent before storing new bo.
            bo_sem_idx = sem_ids_ref[2]
            sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)
            wait_send_bo(bo_sem_idx)

            # Store output from acc to bo.
            bo_x2_ref.at[bo_sem_idx].bitcast(jnp.int32).reshape(
                actual_num_kv_heads,
                bq_sz * num_q_heads_per_kv_head_per_packing,
                head_dim,
            )[...] = pltpu.bitcast(out, jnp.int32)

            # Send cur bo
            start_send_bo(seq_idx, bq_idx, bo_sem_idx)

        lax.fori_loop(0, num_bq, compute_with_bq, None, unroll=False)

    ### ------- Kernel start ------- ###

    @pl.when(seq_idx == 0)
    def prologue():
        start_fetch_bq(0, 0, 0)
        start_fetch_bkv(0, 0, 0)

    @pl.when(seq_idx < decode_end)
    def process_decode():
        process(static_q_len=1)

    @pl.when(jnp.logical_and(decode_end <= seq_idx, seq_idx < prefill_end))
    def process_prefill():
        process(static_q_len=chunk_prefill_size)

    @pl.when(jnp.logical_and(prefill_end <= seq_idx, seq_idx < mixed_end))
    def process_mixed():
        process()

    @pl.when(seq_idx == num_seqs - 1)
    def epilogue():
        for i in range(2):
            wait_send_bo(i)
            wait_update_kv_cache(i)

    ### ------- Kernel end ------- ###


def merge_kv(
    k: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
    v: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
):
    assert k.shape == v.shape
    assert k.dtype == v.dtype
    max_num_tokens, actual_num_kv_heads, actual_head_dim = k.shape
    kv_packing = get_dtype_packing(k.dtype)
    actual_num_kv_heads_x2 = actual_num_kv_heads * 2
    num_kv_heads_x2 = align_to(actual_num_kv_heads_x2, kv_packing)
    head_dim = align_to(actual_head_dim, 128)

    # Interleave k and v: [k1, v1, k2, v2, ...]
    kv = jnp.pad(
        jnp.concat([k, v], axis=-1).reshape(
            max_num_tokens, actual_num_kv_heads_x2, actual_head_dim
        ),
        (
            (0, 0),
            (0, num_kv_heads_x2 - actual_num_kv_heads_x2),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    ).reshape(
        max_num_tokens,
        num_kv_heads_x2 // kv_packing,
        kv_packing,
        head_dim,
    )
    return kv


def prepare_kv(
    k: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
    v: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
):
    assert k.shape == v.shape
    assert k.dtype == v.dtype
    max_num_tokens, actual_num_kv_heads, actual_head_dim = k.shape
    kv_packing = get_dtype_packing(k.dtype)
    # actual_num_kv_heads_x2 = actual_num_kv_heads * 2
    num_kv_heads = align_to(actual_num_kv_heads, kv_packing)
    head_dim = align_to(actual_head_dim, 128)
    k = jnp.pad(
        k,
        (
            (0, 0),
            (0, num_kv_heads - actual_num_kv_heads),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    ).reshape(
        max_num_tokens,
        num_kv_heads // kv_packing,
        kv_packing,
        head_dim,
    )
    v = jnp.pad(
        v,
        (
            (0, 0),
            (0, num_kv_heads - actual_num_kv_heads),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    ).reshape(
        max_num_tokens,
        num_kv_heads // kv_packing,
        kv_packing,
        head_dim,
    )
    return k, v


def prepare_inputs(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim],
    k: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
    v: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
):
    max_num_tokens, actual_num_q_heads, actual_head_dim = q.shape
    actual_num_kv_heads = k.shape[1]
    assert actual_num_q_heads % actual_num_kv_heads == 0
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    q_packing = get_dtype_packing(q.dtype)
    num_q_heads_per_kv_head = align_to(actual_num_q_heads_per_kv_head, q_packing)
    head_dim = align_to(actual_head_dim, 128)
    q = (
        jnp.pad(
            q.reshape(
                max_num_tokens,
                actual_num_kv_heads,
                actual_num_q_heads_per_kv_head,
                actual_head_dim,
            ),
            (
                (0, 0),
                (0, 0),
                (0, num_q_heads_per_kv_head - actual_num_q_heads_per_kv_head),
                (0, head_dim - actual_head_dim),
            ),
            constant_values=0,
        )
        .reshape(
            max_num_tokens,
            actual_num_kv_heads,
            num_q_heads_per_kv_head // q_packing,
            q_packing,
            head_dim,
        )
        .swapaxes(0, 1)
    )
    kv = merge_kv(k, v)
    return q, kv


def prepare_outputs(
    out,  # [actual_num_kv_heads, max_num_tokens, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    actual_num_q_heads_per_kv_head: int,
    actual_head_dim: int,
):
    (
        actual_num_kv_heads,
        max_num_tokens,
        num_q_heads_per_kv_head_per_q_packing,
        q_packing,
        head_dim,
    ) = out.shape
    actual_num_q_heads = actual_num_q_heads_per_kv_head * actual_num_kv_heads
    return (
        out.swapaxes(0, 1)
        .reshape(
            max_num_tokens,
            actual_num_kv_heads,
            num_q_heads_per_kv_head_per_q_packing * q_packing,
            head_dim,
        )[:, :, :actual_num_q_heads_per_kv_head, :actual_head_dim]
        .reshape(max_num_tokens, actual_num_q_heads, actual_head_dim)
    )


# Expect to run this validation during compile time for fused KV cache.
def static_validate_inputs_fused(
    queries: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    kv_cache_fused: jax.Array,  # [total_num_pages, page_size, actual_num_kv_heads * 2, actual_head_dim] - Head interleaving
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    cu_kv_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    # TODO: temp
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
    """Validate inputs to the RPA kernel statically with fused KV cache."""
    q, k, v = queries, keys, values
    if not (len(q.shape) == len(k.shape) == len(v.shape) == 3):
        raise ValueError(f"Expected 3D array for {q.shape=}, {k.shape=}, {v.shape=}")
    if k.shape != v.shape:
        raise ValueError(f"Expected {k.shape=} to be equal to {v.shape=}")
    if not (q.shape[0] == k.shape[0] == v.shape[0]):
        raise ValueError(
            f"Expected {q.shape[0]=} to be equal to {k.shape[0]=} and {v.shape[0]=}"
        )
    if not (q.shape[2] == k.shape[2] == v.shape[2]):
        raise ValueError(
            f"Expected {q.shape[2]=} to be equal to {k.shape[2]=} and {v.shape[2]=}"
        )

    actual_head_dim = q.shape[2]
    actual_num_q_heads = q.shape[1]
    actual_num_kv_heads = k.shape[1]

    if actual_num_q_heads % actual_num_kv_heads != 0:
        raise ValueError(
            f"Expected {actual_num_q_heads=} to be divisible by"
            f" {actual_num_kv_heads=}."
        )

    # Validate fused KV cache
    if len(kv_cache_fused.shape) != 4:
        raise ValueError(
            f"Expected 4D kv_cache_fused, got shape {kv_cache_fused.shape}"
        )

    _, page_size, cache_num_kv_heads_interleaved, head_dim = kv_cache_fused.shape

    if cache_num_kv_heads_interleaved % 2 != 0:
        raise ValueError(
            f"Head interleaving requires even number of heads, got {cache_num_kv_heads_interleaved}"
        )

    cache_num_kv_heads = cache_num_kv_heads_interleaved // 2
    if cache_num_kv_heads != actual_num_kv_heads:
        raise ValueError(
            f"Expected cache_num_kv_heads {cache_num_kv_heads} == actual_num_kv_heads {actual_num_kv_heads}"
        )

    if head_dim != align_to(actual_head_dim, 128):
        raise ValueError(
            f"Expected {head_dim=} is equal to {align_to(actual_head_dim, 128)=}"
        )

    if not (kv_cache_fused.dtype == k.dtype == v.dtype):
        raise ValueError(
            f"Expected all dtypes to match: {kv_cache_fused.dtype=}, {k.dtype=}, {v.dtype=}."
        )

    if not jnp.issubdtype(kv_cache_fused.dtype, jnp.floating):
        raise ValueError(f"Expected {kv_cache_fused.dtype=} to be a floating point.")

    if not (
        jnp.int32
        == kv_lens.dtype
        == page_indices.dtype
        == cu_q_lens.dtype
        == distribution.dtype
    ):
        raise ValueError(
            f"Expected int32 dtype for {kv_lens.dtype=}, {page_indices.dtype=},"
            f" {cu_q_lens.dtype=}, {distribution.dtype=}"
        )

    if not (len(kv_lens.shape) == len(page_indices.shape) == len(cu_q_lens.shape) == 1):
        raise ValueError(
            f"Expected 1D array for {kv_lens.shape=}, {page_indices.shape=},"
            f" {cu_q_lens.shape=}"
        )

    max_num_seqs = kv_lens.shape[0]
    if cu_q_lens.shape != (max_num_seqs + 1,):
        raise ValueError(f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},).")
    if cu_kv_lens.shape != (max_num_seqs + 1,):
        raise ValueError(f"Expected {cu_kv_lens.shape=} to be ({max_num_seqs + 1},).")
    if distribution.shape != (3,):
        raise ValueError(f"Expected {distribution.shape=} to be (3,).")

    if sliding_window is not None and sliding_window <= 0:
        raise ValueError(f"{sliding_window=} must be positive.")
    if soft_cap is not None and soft_cap == 0.0:
        raise ValueError(f"{soft_cap=} must not be 0.0.")
    if chunk_prefill_size is not None and chunk_prefill_size <= 0:
        raise ValueError(f"{chunk_prefill_size=} must be positive.")
    if num_kv_pages_per_block is not None:
        if num_kv_pages_per_block <= 0:
            raise ValueError(f"{num_kv_pages_per_block=} must be positive.")
    if num_queries_per_block is not None:
        if num_queries_per_block <= 0:
            raise ValueError(f"{num_queries_per_block=} must be positive.")
    if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
        raise ValueError(f"{vmem_limit_bytes=} must be positive.")

    del sm_scale
    del mask_value
    del q_scale
    del k_scale
    del v_scale


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "mask_value",
        "q_scale",
        "k_scale",
        "v_scale",
        "xai_temperature_len",
        "chunk_prefill_size",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
    ),
    donate_argnames=("kv_cache_fused",),
)
def ragged_paged_attention(
    queries: jax.Array,  # [padded_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [padded_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.Array,  # [padded_num_tokens, actual_num_kv_heads, actual_head_dim]
    kv_cache_fused: jax.Array,  # [total_num_pages, page_size, actual_num_kv_heads * 2, actual_head_dim]
    kv_lens: jax.Array,  # i32[padded_batch_size]
    page_indices: jax.Array,  # i32[(padded_batch_size * model_context_len + page_size - 1) // page_size]
    cu_q_lens: jax.Array,  # i32[padded_batch_size + 1]
    cu_kv_lens: jax.Array,  # i32[padded_batch_size + 1]
    distribution: jax.Array,  # i32[3]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    xai_temperature_len: float | None = None,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
    """Ragged paged attention that supports mixed prefill and decode with fused KV cache.

    Args:
      queries: concatenated all sequences' queries.
      keys: concatenated all sequences' keys (quantized).
      values: concatenated all sequences' values (quantized).
      kv_cache_fused: paged KV cache with head interleaving format [K1,V1,K2,V2,...].
      kv_lens: padded kv lengths. Only the first num_seqs values are valid.
      page_indices: flattened page indices look-up table.
      cu_q_lens: the cumulative sum of the effective query lengths. Similar to
        kv_lens, only the first num_seqs+1 values are valid.
      distribution: (i, j, k) represents that sequences[0:i] are decode-only,
        sequences[i:j] are chunked-prefill-only, and sequences[j:k] are mixed. The
        k is also the total number of sequences.
      actual_head_dim: the actual head size of the attention. Here we assume k and
        v have the same actual head size.
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
    q, k, v = queries, keys, values
    static_validate_inputs_fused(
        q,
        k,
        v,
        kv_cache_fused,
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        chunk_prefill_size=chunk_prefill_size,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
    )

    actual_num_q_heads = q.shape[1]
    actual_head_dim = q.shape[2]
    actual_num_kv_heads = k.shape[1]

    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    q, kv = prepare_inputs(q, k, v)
    kv_cache_fused_processed = prepare_kv_cache_fused(kv_cache_fused)
    (
        _,
        max_num_tokens,
        num_q_heads_per_kv_head_per_q_packing,
        q_packing,
        head_dim,
    ) = q.shape
    page_size = kv_cache_fused_processed.shape[1]
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    pages_per_seq = num_page_indices // max_num_seqs
    num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_q_packing * q_packing

    bkv_p = num_kv_pages_per_block
    bq_sz = num_queries_per_block
    if bq_sz is None or bkv_p is None:
        bkv_p, bq_sz = get_tuned_block_sizes(
            q.dtype,
            kv_cache_fused_processed.dtype,
            actual_num_q_heads,
            actual_num_kv_heads,
            head_dim,
            page_size,
            max_num_tokens,
            pages_per_seq,
        )
    kv_packing = get_dtype_packing(kv_cache_fused_processed.dtype)
    if page_size == 1:
        bkv_p = bkv_p // 2
        if bkv_p == 0:
            bkv_p = 1
    bkv_p = align_to(bkv_p, kv_packing)
    bkv_sz = bkv_p * page_size
    if vmem_limit_bytes is None:
        vmem_limit_bytes = int(
            get_vmem_estimate_bytes(
                actual_num_kv_heads,
                actual_num_q_heads_per_kv_head,
                head_dim,
                bq_sz,
                bkv_sz,
                q.dtype,
                kv_cache_fused_processed.dtype,
            )
            * 2.4
        )
    grid = (distribution[2],)

    in_specs = [
        pl.BlockSpec(memory_space=pltpu.ANY),  # q
        pl.BlockSpec(memory_space=pltpu.ANY),  # kv_fused
        pl.BlockSpec(memory_space=pltpu.ANY),  # kv_cache_fused
    ]

    out_specs = [
        pl.BlockSpec(memory_space=pltpu.ANY),  # output
        pl.BlockSpec(memory_space=pltpu.ANY),  # updated kv_cache_fused
    ]

    bkv_fused_double_buf = pltpu.VMEM(
        (2, bkv_sz, *kv_cache_fused_processed.shape[2:]),
        kv_cache_fused_processed.dtype,
    )

    bq_double_buf = pltpu.VMEM(
        (2, actual_num_kv_heads, bq_sz, *q.shape[2:]),
        q.dtype,
    )

    bo_double_buf = bq_double_buf

    l_scratch = pltpu.VMEM(
        (actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128),
        jnp.float32,
    )
    m_scratch = l_scratch

    acc_scratch = pltpu.VMEM(
        (actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, head_dim),
        jnp.float32,
    )

    scratch_shapes = [
        bkv_fused_double_buf,  # Double buffering for fused kv block with head interleaving.
        bq_double_buf,  # Double buffering for q block.
        bo_double_buf,  # Double buffering for output block.
        # Semaphores for double buffering of bkv, bq, bo and bkv_update.
        pltpu.SemaphoreType.DMA((4, 2)),
        # Intermediate buffers per kv head for flash attention.
        l_scratch,
        m_scratch,
        acc_scratch,
    ]

    scalar_prefetches = (
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
        # (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
        jnp.zeros((3,), jnp.int32),
        # (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx)
        jnp.full((4,), -1, jnp.int32),
        # (bkv_sem_0_seq_idx, bkv_sem_1_seq_idx, bkv_sem_0_offset, bkv_sem_1_offset, bkv_sem_0_sz, bkv_sem_1_sz)
        jnp.full((6,), -1, jnp.int32),
    )

    scope_name = f"RPA-bq_{bq_sz}-bkvp_{bkv_p}-p_{page_size}"
    kernel = jax.named_scope(scope_name)(
        pl.pallas_call(
            functools.partial(
                _ragged_paged_attention_kernel,
                sm_scale=sm_scale,
                sliding_window=sliding_window,
                soft_cap=soft_cap,
                mask_value=mask_value,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                xai_temperature_len=xai_temperature_len,
                chunk_prefill_size=chunk_prefill_size,
                bq_sz=bq_sz,
                bkv_p=bkv_p,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(scalar_prefetches),
                in_specs=in_specs,
                out_specs=out_specs,
                grid=grid,
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                # one, we need some extra work to support Megacore mode.
                dimension_semantics=("arbitrary",),
                vmem_limit_bytes=vmem_limit_bytes,
                disable_bounds_checks=True,
            ),
            out_shape=[
                jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
                jax.ShapeDtypeStruct(
                    shape=kv_cache_fused_processed.shape,
                    dtype=kv_cache_fused_processed.dtype,
                ),
            ],
            input_output_aliases={
                8: 0,  # q input -> q output
                10: 1,  # kv_cache_fused input -> updated kv_cache_fused output
            },
            name=scope_name,
        )
    )

    output, updated_kv_cache_fused = kernel(
        *scalar_prefetches, q, kv, kv_cache_fused_processed
    )
    return (
        prepare_outputs(output, actual_num_q_heads_per_kv_head, actual_head_dim),
        prepare_updated_kv_cache_fused(
            updated_kv_cache_fused, actual_num_kv_heads, actual_head_dim
        ),
    )


def prepare_updated_kv_cache(
    kv_cache,  # [total_num_pages, page_size, num_kv_heads // kv_packing, kv_packing, head_dim]
    actual_num_kv_heads: int,
    actual_head_dim: int,
):
    """
    return [total_num_pages, page_size , actual_num_kv_heads, actual_head_dim]
    """
    (
        total_num_pages,
        page_size,
        num_kv_heads,
        kv_packing,
        head_dim,
    ) = kv_cache.shape
    return kv_cache.reshape(
        -1,
        num_kv_heads * kv_packing,
        head_dim,
    )[:, :actual_num_kv_heads, :actual_head_dim]


def prepare_kv_cache_fused(
    kv_cache_fused: jax.Array,  # [total_num_pages, page_size, actual_num_kv_heads * 2, actual_head_dim]
):
    total_num_pages, page_size, actual_num_kv_heads_interleaved, actual_head_dim = (
        kv_cache_fused.shape
    )
    assert actual_num_kv_heads_interleaved % 2 == 0

    kv_packing = get_dtype_packing(kv_cache_fused.dtype)
    num_kv_heads_interleaved = align_to(actual_num_kv_heads_interleaved, kv_packing)
    head_dim = align_to(actual_head_dim, 128)

    kv_cache_fused_processed = jnp.pad(
        kv_cache_fused,
        (
            (0, 0),
            (0, 0),
            (0, num_kv_heads_interleaved - actual_num_kv_heads_interleaved),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    ).reshape(
        total_num_pages,
        page_size,
        num_kv_heads_interleaved // kv_packing,
        kv_packing,
        head_dim,
    )
    return kv_cache_fused_processed


def prepare_updated_kv_cache_fused(
    kv_cache_fused,  # [total_num_pages, page_size, num_kv_heads_interleaved // kv_packing, kv_packing, head_dim]
    actual_num_kv_heads: int,
    actual_head_dim: int,
):
    """Extract actual KV cache from processed fused format."""
    (
        total_num_pages,
        page_size,
        num_kv_heads_interleaved_packed,
        kv_packing,
        head_dim,
    ) = kv_cache_fused.shape

    actual_num_kv_heads_interleaved = (
        actual_num_kv_heads * 2
    )  # Head interleaving: K1,V1,K2,V2,...
    return kv_cache_fused.reshape(
        -1,
        num_kv_heads_interleaved_packed * kv_packing,
        head_dim,
    )[:, :actual_num_kv_heads_interleaved, :actual_head_dim]


def prepare_kv_cache(
    k_cache: jax.Array,  # [total_num_pages, page_size, actual_num_kv_heads, actual_head_dim],
    v_cache: jax.Array,  # [total_num_pages, page_size, actual_num_kv_heads, actual_head_dim],
):
    total_num_pages, page_size, actual_num_kv_heads, actual_head_dim = k_cache.shape
    kv_packing = get_dtype_packing(k_cache.dtype)
    actual_num_kv_heads = k_cache.shape[2]
    num_kv_heads = align_to(actual_num_kv_heads, kv_packing)
    head_dim = align_to(actual_head_dim, 128)
    k_cache_processed = jnp.pad(
        k_cache,
        ((0, 0), (0, 0), (0, 0), (0, head_dim - actual_head_dim)),
        constant_values=0,
    ).reshape(
        total_num_pages,
        page_size,
        num_kv_heads // kv_packing,
        kv_packing,
        head_dim,
    )
    v_cache_processed = jnp.pad(
        v_cache,
        ((0, 0), (0, 0), (0, 0), (0, head_dim - actual_head_dim)),
        constant_values=0,
    ).reshape(
        total_num_pages,
        page_size,
        num_kv_heads // kv_packing,
        kv_packing,
        head_dim,
    )
    return k_cache_processed, v_cache_processed
