# Adapted from https://github.com/vllm-project/tpu-inference/releases/tag/v0.11.1
# Copyright 2025 The tpu-inference Authors. All rights reserved.
"""TPU-Friendly Ragged Paged Attention kernel (v3).

This kernel offers a highly optimized implementation of ragged paged attention,
specifically designed for TPU and compatible with a wide range of model
specifications. It supports mixed prefill and decoding, enhancing throughput
during inference.

Key optimizations over v2:
- Split kernel into 3 separate pallas_calls (DECODE/PREFILL/MIXED)
- l/m/acc re-initialization per bq block
- Sliding window precise skipping via per-bq-block start/end indices
- Block size API (d_block_sizes/p_block_sizes/m_block_sizes)
- KV cache update timing moved to last bq iteration

sglang-jax specific features:
- custom_mask support for speculative decoding
- attention_sink support for streaming inference
- xai_temperature support for Grok-style models
- cu_kv_lens-based page_indices offset computation
"""
import functools
import inspect as _inspect
import logging
from enum import Enum

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.kernels.ragged_paged_attention.util import (
    align_to,
    cdiv,
    get_dtype_bitwidth,
    get_dtype_packing,
    get_tpu_version,
    next_power_of_2,
)

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
DEFAULT_VMEM_LIMIT_BYTES = 120 * 1024 * 1024  # 120MB
logger = logging.getLogger(__name__)

_COMPILER_PARAMS_SUPPORTS_SEMAPHORE = (
    "disable_semaphore_checks" in _inspect.signature(pltpu.CompilerParams).parameters
)


def _semaphore_kwargs(disable_semaphore_checks: bool) -> dict:
    if _COMPILER_PARAMS_SUPPORTS_SEMAPHORE:
        return {"disable_semaphore_checks": disable_semaphore_checks}
    return {}


class RpaCase(Enum):
    """Represents the different cases for Ragged Paged Attention.

    - DECODE: Sequences are in decode-only mode (q_len = 1).
    - PREFILL: Sequences are in prefill-only mode (q_len > 1, static).
    - MIXED: Sequences can be a mix of prefill and decode (q_len > 1, dynamic).
    """

    DECODE = 0
    PREFILL = 1
    MIXED = 2

    @property
    def symbol(self):
        return {
            RpaCase.DECODE: "d",
            RpaCase.PREFILL: "p",
            RpaCase.MIXED: "m",
        }[self]

    def get_range(self, distribution):
        assert distribution.shape == (3,)
        if self == RpaCase.DECODE:
            return 0, distribution[0]
        elif self == RpaCase.PREFILL:
            return distribution[0], distribution[1]
        elif self == RpaCase.MIXED:
            return distribution[1], distribution[2]
        else:
            raise ValueError(f"Unsupported RPA case: {self}")


def ref_ragged_paged_attention(
    queries: jax.Array,  # [padded_num_tokens, num_q_heads, head_dim]
    k_pages: jax.Array,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    v_pages: jax.Array,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[padded_batch_size]
    page_indices: jax.Array,  # i32[padded_batch_size, max_pages_per_seq]
    cu_q_lens: jax.Array,  # i32[padded_batch_size + 1]
    num_seqs: jax.Array,  # i32[1],
    *,
    custom_mask: jax.Array = None,  # [pattern_total_kv_len]
    causal: bool = True,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
    xai_temperature_len: float | None = None,
    attention_sink: jax.Array | float | None = None,
):
    """Reference implementation for ragged paged attention."""
    if not causal:
        assert (
            custom_mask is not None and custom_mask.size > jnp.cumsum(kv_lens)[-1]
        ), f"use custom_mask, custom_mask length {custom_mask.size=} must larger than total kv length {jnp.cumsum(kv_lens)[-1]=}"
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE
    _, _, num_kv_heads, head_dim = k_pages.shape
    num_q_heads = queries.shape[1]
    assert num_q_heads % num_kv_heads == 0
    num_query_per_kv = num_q_heads // num_kv_heads
    outputs = []
    mask_len_list = []
    for i in range(num_seqs[0]):
        kv_len = kv_lens[i]
        q_len = cu_q_lens[i + 1] - cu_q_lens[i]
        mask_len_list.append(q_len * kv_len)
    mask_lens = jnp.array(mask_len_list, dtype=jnp.int32)
    cu_mask_lens = jnp.concatenate([jnp.array([0], dtype=jnp.int32), jnp.cumsum(mask_lens)])

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
        if causal:
            q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(jnp.int32, attn.shape, 1)
            kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
            mask = q_span < kv_span
        else:
            mask_start = cu_mask_lens[i]
            mask_end = cu_mask_lens[i + 1]
            mask = custom_mask[mask_start:mask_end]
            mask = (
                jnp.repeat(jnp.expand_dims(mask, axis=0), num_q_heads, axis=0).reshape(
                    num_q_heads, q_len, kv_len
                )
                < 1
            )
        if sliding_window is not None:
            mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
        if soft_cap is not None:
            attn = soft_cap * jnp.tanh(attn / soft_cap)

        if xai_temperature_len is not None:
            prefix_len = kv_len - q_len
            qidx = jnp.arange(prefix_len, kv_len)
            xai_temperature_scale = 1.0 / jnp.log2(float(xai_temperature_len))
            _qtemp = jnp.log2(qidx.astype(jnp.float32)) * xai_temperature_scale
            xai_temperature_reg = jnp.where(qidx > xai_temperature_len, _qtemp, 1.0)
            attn = attn * xai_temperature_reg[None, :, None]

        attn += jnp.where(mask, mask_value, 0.0)
        if attention_sink is not None:
            sink = jnp.asarray(attention_sink, dtype=jnp.float32)
            if sink.ndim == 0:
                sink = jnp.full((num_q_heads,), sink)
            sink_logits = jnp.broadcast_to(
                sink.reshape(num_q_heads, 1, 1),
                (num_q_heads, q_len, 1),
            )
            attn = jnp.concatenate([sink_logits, attn], axis=-1)
            attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
            attn = attn[..., 1:]
        else:
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
        # cu_kv_lens_ref: i32[max_num_seqs + 1]
        align_to(max_num_seqs + 1, 128) * 32
        +
        # cu_seq_mask_lens: i32[max_num_seqs + 1]
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
    use_custom_mask=False,
    bkv_csz=None,
):
    q_packing = get_dtype_packing(q_dtype)
    kv_packing = get_dtype_packing(kv_dtype)
    out_bits = get_dtype_bitwidth(q_dtype) if q_dtype != jnp.float32 else 32
    num_q_heads_per_kv_head = align_to(actual_num_q_heads_per_kv_head, q_packing)
    bkv_stride = cdiv(actual_num_kv_heads * 2, kv_packing)
    if has_bank_conflicts(bkv_stride):
        bkv_stride += 1
    head_dim = align_to(actual_head_dim, 128)

    total_bits = (
        # bkv_x2_ref
        (2 * bkv_sz * bkv_stride * kv_packing * head_dim) * (32 // kv_packing)
        +
        # bq_x2_ref + bo_x2_ref
        2
        * (2 * actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head * head_dim)
        * (32 // q_packing)
        +
        # l_ref + m_ref (out_dtype, not float32)
        2 * (actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head * 128) * out_bits
        +
        # acc_ref (out_dtype)
        (actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head * head_dim) * out_bits
    )

    if use_custom_mask:
        # bkvmask_double_buf: (2, bq_sz, bkv_sz, head_dim) in int32
        total_bits += 2 * bq_sz * bkv_sz * head_dim * 32

    # Attention compute intermediates (f32). The for-loop over kv_heads is
    # statically unrolled by the compiler, so all heads' intermediates coexist
    # in VMEM.  Per head: QK^T scores (bq*qpkv, bkv_csz) + PV result (bq*qpkv, hd)
    # in f32, with ~4x overhead for softmax temps, pipelining, and spills.
    # Use bkv_csz (compute size) since that determines the matmul tile size.
    compute_bkv = bkv_csz if bkv_csz is not None else bkv_sz
    compute_bits = (
        actual_num_kv_heads
        * bq_sz
        * num_q_heads_per_kv_head
        * (compute_bkv + head_dim)
        * 32  # f32
        * 4  # compiler overhead factor
    )
    total_bits += compute_bits

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


def _ragged_paged_attention_kernel(*args, **kwargs):
    # distribution_ref is at index 5 (after kv_lens, page_indices, cu_q_lens,
    # cu_kv_lens, cu_seq_mask_lens).
    distribution_ref = args[5]
    start_seq_idx, end_seq_idx = kwargs["case"].get_range(distribution_ref)

    @pl.loop(start_seq_idx, end_seq_idx)
    def _(seq_idx):
        return _ragged_paged_attention_kernel_loop(
            seq_idx,
            *args,
            **kwargs,
        )


def _ragged_paged_attention_kernel_loop(
    seq_idx,
    # Prefetch (9 scalar prefetches)
    kv_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [flat page indices]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    cu_kv_lens_ref,  # [max_num_seqs + 1]
    cu_seq_mask_lens,  # [max_num_seqs + 1]
    distribution_ref,  # [3] (decode_end, prefill_end, mixed_end)
    sem_ids_ref,  # [3] (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
    bo_ids_ref,  # [4]
    bkv_update_ids_ref,  # [6]
    # Input
    q_hbm_ref,  # [actual_num_kv_heads, max_num_tokens, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    kv_hbm_ref,  # [max_num_tokens, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_cache_hbm_ref,  # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    custom_mask_ref,  # [flatten_total_kv_len, head_dim] or None
    zero_mask_ref,  # [bkv_sz, head_dim] or None
    attention_sink_ref,  # [actual_num_kv_heads, num_q_heads_per_kv_head, 128] or None
    # Output
    o_hbm_ref,  # same shape as q_hbm_ref
    updated_kv_cache_hbm_ref,  # same shape as kv_cache_hbm_ref
    # Scratch
    bkvmask_ref,  # [2, bq_sz, bkv_sz, head_dim] or None
    bkv_x2_ref,  # [2, bkv_sz, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    bq_x2_ref,  # [2, actual_num_kv_heads, bq_sz, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    bo_x2_ref,  # [2, actual_num_kv_heads, bq_sz, ...]
    sems,  # [5, 2]
    l_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128]
    m_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128]
    acc_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, head_dim]
    *,
    causal: bool = True,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    xai_temperature_len: float | None = None,
    static_q_len: int | None = None,
    bq_sz,  # bq fetch size
    bkv_sz,  # bkv prefetch size
    bq_csz,  # bq compute size
    bkv_csz,  # bkv compute size
    case: RpaCase = RpaCase.MIXED,
    skip_kv_mask: bool = False,
    tpu_version: int = 6,
    debug_mode: bool = False,
):
    assert q_hbm_ref.shape == o_hbm_ref.shape
    assert q_hbm_ref.shape[-1] == kv_cache_hbm_ref.shape[-1]

    use_causal_mask = causal
    if case == RpaCase.DECODE:
        use_causal_mask = False

    out_dtype = acc_ref.dtype
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
        num_kv_heads_x2_per_kv_packing,
        kv_packing,
        _,
    ) = kv_cache_hbm_ref.shape
    bkv_stride = bkv_x2_ref.shape[2]
    num_page_indices = page_indices_ref.shape[0]
    num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_packing * q_packing
    q_dtype = q_hbm_ref.dtype
    kv_dtype = kv_cache_hbm_ref.dtype
    assert o_hbm_ref.dtype == q_dtype
    assert get_dtype_packing(q_dtype) == q_packing
    assert get_dtype_packing(kv_dtype) == kv_packing
    assert head_dim % 128 == 0
    assert bkv_sz % page_size == 0
    assert bkv_sz % bkv_csz == 0, f"bkv_sz={bkv_sz} not divisible by bkv_csz={bkv_csz}"
    bkv_p = bkv_sz // page_size
    start_seq_idx, end_seq_idx = case.get_range(distribution_ref)

    q_start = cu_q_lens_ref[seq_idx]
    q_end = cu_q_lens_ref[seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens_ref[seq_idx]
    kv_q_gap = kv_len - q_len
    cur_seq_start_bkv_idx = 0
    next_seq_start_bkv_idx = 0

    if sliding_window is not None:
        cur_seq_start_bkv_idx = jnp.maximum(kv_q_gap - sliding_window, 0) // bkv_sz
        next_seq_idx = jnp.minimum(seq_idx + 1, end_seq_idx - 1)
        next_q_start = cu_q_lens_ref[next_seq_idx]
        next_q_end = cu_q_lens_ref[next_seq_idx + 1]
        next_q_len = next_q_end - next_q_start
        next_kv_len = kv_lens_ref[next_seq_idx]
        next_kv_q_gap = next_kv_len - next_q_len
        next_seq_start_bkv_idx = jnp.maximum(next_kv_q_gap - sliding_window, 0) // bkv_sz

    def debug_print(msg, *args):
        if debug_mode:
            pl.debug_print(msg, *args)

    def flash_attention_step1_qk_softmax(
        q,  # [actual_bq_csz * num_q_heads_per_kv_head, head_dim]
        k,  # [bkv_csz, head_dim]
        v,  # [bkv_csz, head_dim]
        l_ref,  # [actual_bq_csz * num_q_heads_per_kv_head, 128]
        m_ref,  # [actual_bq_csz * num_q_heads_per_kv_head, 128]
        *,
        processed_q_len,
        processed_kv_len,
        effective_kv_len,
        xai_temperature_reg=None,
        custom_mask_data=None,
    ):
        assert len(q.shape) == 2
        assert q.shape[0] % num_q_heads_per_kv_head == 0
        assert q.shape[1] == head_dim
        actual_bq_csz = q.shape[0] // num_q_heads_per_kv_head
        assert k.shape == (bkv_csz, head_dim)
        assert v.shape == (bkv_csz, head_dim)
        assert l_ref.shape == (actual_bq_csz * num_q_heads_per_kv_head, 128)
        assert m_ref.shape == (actual_bq_csz * num_q_heads_per_kv_head, 128)
        assert k.dtype == v.dtype

        # Follow FlashAttention-2 forward pass.
        if q_scale is not None:
            q = q / q_scale
            if jnp.issubdtype(k.dtype, jnp.floating):
                dtype_info = jnp.finfo(k.dtype)
                minval = float(dtype_info.min)
                maxval = float(dtype_info.max)
                q = jnp.clip(q, min=minval, max=maxval)
            q = q.astype(k.dtype)

        s = jnp.matmul(q, k.T, preferred_element_type=jnp.float32)

        s_scale = sm_scale
        if k_scale is not None:
            s_scale *= k_scale
        if q_scale is not None:
            s_scale *= q_scale

        s *= s_scale

        # xai temperature scaling
        if xai_temperature_reg is not None:
            s = s * xai_temperature_reg[:, None]

        if soft_cap is not None:
            s = soft_cap * jnp.tanh(s / soft_cap)

        # Use int16 for span computations when safe: non-f32 dtype on TPU v6+
        # with causal mask. Custom mask shapes can trigger a Mosaic compiler bug.
        int_ty = jnp.int32
        if get_dtype_packing(q.dtype) != 1 and tpu_version >= 6 and use_causal_mask:
            int_ty = jnp.int16
        processed_q_len_int = processed_q_len.astype(int_ty)
        processed_kv_len_int = processed_kv_len.astype(int_ty)
        effective_kv_len_int = effective_kv_len.astype(int_ty)
        q_span = processed_q_len_int + (
            lax.broadcasted_iota(jnp.int32, s.shape, 0) // num_q_heads_per_kv_head
        ).astype(int_ty)
        k_span = processed_kv_len_int + lax.broadcasted_iota(int_ty, s.shape, 1)
        v_span = processed_kv_len_int + lax.broadcasted_iota(int_ty, v.shape, 0)

        mask = None
        if use_causal_mask:
            assert not skip_kv_mask
            mask = mask_and(mask, q_span >= k_span)
        elif custom_mask_data is not None:
            # custom_mask_data: [actual_bq_csz, bkv_csz] int32, 1=keep
            custom_mask_expanded = jnp.repeat(custom_mask_data, num_q_heads_per_kv_head, axis=0)
            mask = mask_and(mask, custom_mask_expanded == 1)

        if not skip_kv_mask:
            mask = mask_and(mask, k_span < effective_kv_len_int)
            v = jnp.where(v_span < effective_kv_len_int, v, 0.0)

        if sliding_window is not None:
            mask = mask_and(mask, q_span < k_span + sliding_window)

        if mask is not None:
            s = jnp.where(mask, s, mask_value)

        s_rowmax = jnp.max(s, axis=1, keepdims=True)
        m_prev = m_ref[...].astype(jnp.float32)
        m_curr = jnp.maximum(m_prev, s_rowmax)
        m_ref[...] = m_curr.astype(out_dtype)
        p = jnp.exp(s - broadcast_minor(m_curr, s.shape))

        p_rowsum = jnp.sum(p, axis=1, keepdims=True)
        exp_m_diff = jnp.exp(m_prev - m_curr)
        l_prev = l_ref[...].astype(jnp.float32)
        l_ref[...] = (exp_m_diff * l_prev + p_rowsum).astype(out_dtype)

        return p, v, exp_m_diff

    def flash_attention_step2_pv(
        p,  # [actual_bq_csz * num_q_heads_per_kv_head, bkv_csz]
        v,  # [bkv_csz, head_dim]
        exp_m_diff,  # [actual_bq_csz * num_q_heads_per_kv_head, 128]
        o_ref,  # [actual_bq_csz * num_q_heads_per_kv_head, head_dim]
    ):
        assert len(p.shape) == 2
        assert p.shape[0] % num_q_heads_per_kv_head == 0
        assert p.shape[1] == bkv_csz
        actual_bq_csz = p.shape[0] // num_q_heads_per_kv_head
        assert v.shape == (bkv_csz, head_dim)
        assert exp_m_diff.shape == (actual_bq_csz * num_q_heads_per_kv_head, 128)
        assert o_ref.shape == (actual_bq_csz * num_q_heads_per_kv_head, head_dim)
        pv = jnp.matmul(p, v, preferred_element_type=jnp.float32)
        if v_scale is not None:
            pv *= v_scale
        o_prev = o_ref[...].astype(jnp.float32)
        o_ref[...] = (broadcast_minor(exp_m_diff, o_prev.shape) * o_prev + pv).astype(out_dtype)

    def _async_copy(src, dst, sem, wait):
        if debug_mode:
            return
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_mask(seq_idx, bq_idx, bkvmask_idx, bkvmask_sem_idx, *, wait=False):
        if custom_mask_ref is None:
            return
        sem = sems.at[4, bkvmask_sem_idx]
        kvmask_vmem_ref = bkvmask_ref.at[bkvmask_sem_idx]

        kv_len = kv_lens_ref[seq_idx]
        mask_len = kv_len
        mask_start = bkvmask_idx * bkv_sz
        mask_left = mask_len - mask_start
        load_kvmask_sz = jnp.minimum(bkv_sz, mask_left)

        q_len_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        load_q_sz = jnp.minimum(bq_sz, q_end - q_len_start)

        cur_seq_mask_start = cu_seq_mask_lens[seq_idx]
        cur_bq_mask_start = cur_seq_mask_start + bq_idx * bq_sz * kv_len

        def loop_body(i, _):
            start = cur_bq_mask_start + i * kv_len + mask_start
            _async_copy(
                custom_mask_ref.at[pl.ds(start, load_kvmask_sz)],
                kvmask_vmem_ref.at[i, pl.ds(0, load_kvmask_sz)],
                sem,
                wait,
            )
            _async_copy(
                zero_mask_ref.at[pl.ds(0, bkv_sz - load_kvmask_sz)],
                kvmask_vmem_ref.at[i, pl.ds(load_kvmask_sz, bkv_sz - load_kvmask_sz)],
                sem,
                wait,
            )

        lax.fori_loop(0, load_q_sz, loop_body, None, unroll=False)

    def _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False):
        sem = sems.at[0, bkv_sem_idx]
        vmem_ref = bkv_x2_ref.at[bkv_sem_idx, :, :num_kv_heads_x2_per_kv_packing]

        cache_hbm_shape = kv_cache_hbm_ref.shape
        cache_hbm_ref = kv_cache_hbm_ref.reshape(
            cache_hbm_shape[0] * cache_hbm_shape[1], *cache_hbm_shape[2:]
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

        bkv_sz_frm_cache = jnp.minimum(kv_left_frm_cache, bkv_sz)
        bkv_sz_frm_new = jnp.minimum(bkv_sz - bkv_sz_frm_cache, kv_left_frm_new)
        # sglang-jax: use cu_kv_lens for page_indices offset.
        start_kv_page_idx = cdiv(cu_kv_lens_ref[seq_idx], page_size)
        page_indices_offset = start_kv_page_idx + kv_p_start

        if not wait:
            # Make sure the current bkv buffer is safe to overwrite.
            wait_update_kv_cache(bkv_sem_idx)

            for i in range(bkv_p):
                sz = jnp.clip(kv_left_frm_cache - i * page_size, 0, page_size)
                page_idx = jnp.minimum(page_indices_offset + i, num_page_indices - 1)
                _async_copy(
                    cache_hbm_ref.at[pl.ds(page_indices_ref[page_idx] * page_size, sz)],
                    vmem_ref.at[pl.ds(i * page_size, sz)],
                    sem,
                    wait=False,
                )

            new_kv_len_start = q_end - kv_left_frm_new
            _async_copy(
                kv_hbm_ref.at[pl.ds(new_kv_len_start, bkv_sz_frm_new)],
                vmem_ref.at[pl.ds(bkv_sz_frm_cache, bkv_sz_frm_new)],
                sem,
                wait,
            )
        else:
            dst = vmem_ref.at[pl.ds(0, bkv_sz_frm_cache + bkv_sz_frm_new)]
            _async_copy(
                src=dst,
                dst=dst,
                sem=sem,
                wait=True,
            )
        return kv_len_start + bkv_sz_frm_cache, bkv_sz_frm_new

    def _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, *, wait=False):
        sem = sems.at[3, bkv_sem_idx]
        vmem_ref = bkv_x2_ref.at[bkv_sem_idx, :, :num_kv_heads_x2_per_kv_packing]
        bkv_id = offset // bkv_sz
        kv_p_start = offset // page_size
        kv_p_end = cdiv(offset + update_sz, page_size)
        ignore = offset % page_size
        p_ignore = kv_p_start - bkv_id * bkv_p
        # sglang-jax: use cu_kv_lens for page_indices offset.
        start_kv_page_idx = cdiv(cu_kv_lens_ref[seq_idx], page_size)
        page_indices_offset = start_kv_page_idx + kv_p_start

        cache_hbm_shape = updated_kv_cache_hbm_ref.shape
        cache_hbm_ref = updated_kv_cache_hbm_ref.reshape(
            cache_hbm_shape[0] * cache_hbm_shape[1], *cache_hbm_shape[2:]
        )

        def loop_body(i, states):
            update_sz, ignore = states
            sz = jnp.minimum(page_size - ignore, update_sz)

            _async_copy(
                vmem_ref.at[pl.ds((p_ignore + i) * page_size + ignore, sz)],
                cache_hbm_ref.at[
                    pl.ds(
                        page_indices_ref[page_indices_offset + i] * page_size + ignore,
                        sz,
                    )
                ],
                sem,
                wait,
            )
            return update_sz - sz, 0

        if not wait:
            lax.fori_loop(
                0,
                kv_p_end - kv_p_start,
                loop_body,
                (update_sz, ignore),
                unroll=False,
            )
        else:
            dst = cache_hbm_ref.at[pl.ds(0, update_sz)]
            _async_copy(
                src=dst,
                dst=dst,
                sem=sem,
                wait=True,
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

    def start_fetch_mask(seq_idx, bq_idx, bkvmask_idx, bkvmask_sem_idx):
        return _fetch_mask(seq_idx, bq_idx, bkvmask_idx, bkvmask_sem_idx)

    def wait_fetch_mask(seq_idx, bq_idx, bkvmask_idx, bkvmask_sem_idx):
        return _fetch_mask(seq_idx, bq_idx, bkvmask_idx, bkvmask_sem_idx, wait=True)

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

        @pl.when(jnp.logical_and(start_seq_idx <= old_seq_idx, old_seq_idx <= seq_idx))
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

    def strided_load(ref, start, sz, step, *, dtype=None):
        assert get_dtype_packing(ref.dtype) == 1
        assert len(ref.shape) == 2
        r, l = ref.shape  # noqa
        assert l % 128 == 0
        folds = l // 128
        ref = ref.reshape(r * folds, 128)
        start *= folds
        sz *= folds
        step *= folds
        assert sz % step == 0
        vec = jnp.concat([ref[pl.ds(start + i, sz // step, step)] for i in range(folds)], axis=1)
        if dtype is not None:
            vec = pltpu.bitcast(vec, dtype)
        return vec

    def strided_store(ref, start, sz, step, val):
        assert get_dtype_packing(ref.dtype) == 1
        assert ref.dtype == val.dtype
        assert ref.shape == val.shape
        assert len(ref.shape) == 2
        r, l = ref.shape  # noqa
        assert l % 128 == 0
        folds = l // 128
        ref = ref.reshape(r * folds, 128)
        start *= folds
        sz *= folds
        step *= folds
        assert sz % step == 0
        for i in range(folds):
            ref[pl.ds(start + i, sz // step, step)] = val[:, i * 128 : (i + 1) * 128]

    def load_bq(bq_sem_idx, kv_head_idx, start, sz):
        q_ref = (
            bq_x2_ref.bitcast(jnp.uint32)
            .at[bq_sem_idx, kv_head_idx]
            .reshape(bq_sz * num_q_heads_per_kv_head_per_packing, head_dim)
        )
        start *= num_q_heads_per_kv_head_per_packing
        sz *= num_q_heads_per_kv_head_per_packing
        return strided_load(q_ref, start, sz, 1, dtype=q_dtype)

    def load_bkv(bkv_sem_idx, kv_head_idx, start, sz):
        start *= bkv_stride
        sz *= bkv_stride
        step = bkv_stride
        kv_ref = bkv_x2_ref.bitcast(jnp.uint32).at[bkv_sem_idx].reshape(bkv_sz * step, head_dim)

        if kv_packing == 1:
            start += kv_head_idx * 2
            k = strided_load(kv_ref, start, sz, step, dtype=kv_dtype)
            v = strided_load(kv_ref, start + 1, sz, step, dtype=kv_dtype)
            k = pltpu.bitcast(k, kv_dtype)
            v = pltpu.bitcast(v, kv_dtype)
            return k, v

        num_kv_per_load = kv_packing // 2
        offset = kv_head_idx // num_kv_per_load
        kv_idx_in_load = kv_head_idx % num_kv_per_load
        kv = strided_load(kv_ref, start + offset, sz, step)
        bitwidth = 32 // kv_packing
        repack_ty = jnp.dtype(f"uint{bitwidth}")
        k = kv >> (kv_idx_in_load * 2 * bitwidth)
        v = k >> bitwidth
        k = pltpu.bitcast(k.astype(repack_ty), kv_dtype)
        v = pltpu.bitcast(v.astype(repack_ty), kv_dtype)
        return k, v

    def broadcast_minor(src, shape):
        if src.shape == shape:
            return src
        assert src.shape[:-1] == shape[:-1]
        assert src.shape[-1] % 128 == 0
        target_minor = align_to(shape[-1], src.shape[-1])
        return jnp.concatenate([src for _ in range(target_minor // src.shape[-1])], axis=-1)[
            ..., : shape[-1]
        ]

    def mask_and(mask, new_mask):
        if mask is None:
            return new_mask
        return jnp.logical_and(mask, new_mask)

    def process(static_q_len=None):
        if static_q_len is None:
            actual_bq_sz = bq_sz
            num_bq = cdiv(q_len, actual_bq_sz)
        else:
            actual_bq_sz = min(bq_sz, static_q_len)
            num_bq = cdiv(static_q_len, actual_bq_sz)

        actual_bq_csz = min(bq_csz, actual_bq_sz)

        def get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx):
            next_bq_idx = bq_idx + 1
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
            next_bq_sem_idx = lax.select(bq_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bq_idx, next_bq_sem_idx

        def get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx, *, num_bkv):
            next_bkv_idx = bkv_idx + 1
            is_last_bkv = next_bkv_idx == num_bkv
            next_bq_idx = lax.select(is_last_bkv, bq_idx + 1, bq_idx)
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
            next_bkv_sem_idx = lax.select(bkv_sem_idx == 0, 1, 0)

            next_bq_start_bkv_idx = 0
            if sliding_window is not None:
                next_bq_start_bkv_idx = (
                    jnp.maximum(kv_q_gap + (bq_idx + 1) * actual_bq_sz - sliding_window, 0)
                    // bkv_sz
                )
            next_bkv_idx = lax.select(is_last_bkv, next_bq_start_bkv_idx, next_bkv_idx)
            next_bkv_idx = lax.select(is_last_bq, next_seq_start_bkv_idx, next_bkv_idx)
            return next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx

        @pl.loop(0, num_bq, unroll=False)
        def compute_with_bq(bq_idx):
            acc_ref[...] = jnp.full_like(acc_ref, 0.0)

            # Initialize l, m before bkv loop.
            if attention_sink_ref is not None:
                # Attention sink: m = sink logits, l = 1.0
                # (pretend we've already seen a virtual token with logit = sink_value).
                m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
                l_ref[...] = jnp.full_like(l_ref, 1.0)
                for kv_head_idx in range(actual_num_kv_heads):
                    sinks = attention_sink_ref[kv_head_idx]  # [num_q_heads_per_kv_head, 128]
                    lm_start = 0
                    lm_size = actual_bq_sz * num_q_heads_per_kv_head
                    sink_tiled = jnp.tile(sinks, (actual_bq_sz, 1))
                    m_ref.at[kv_head_idx, pl.ds(lm_start, lm_size)][...] = sink_tiled.astype(
                        out_dtype
                    )
            else:
                l_ref[...] = jnp.full_like(l_ref, 0.0)
                m_ref[...] = jnp.full_like(m_ref, -jnp.inf)

            bq_sem_idx = sem_ids_ref[0]
            next_seq_idx, next_bq_idx, next_bq_sem_idx = get_next_bq_ids(
                seq_idx, bq_idx, bq_sem_idx
            )

            processed_q_len = kv_q_gap + bq_idx * actual_bq_sz
            start_bkv_idx = 0
            if sliding_window is not None:
                start_bkv_idx = jnp.maximum(processed_q_len - sliding_window, 0) // bkv_sz
            if use_causal_mask:
                effective_kv_len = jnp.minimum(kv_len, processed_q_len + actual_bq_sz)
            else:
                effective_kv_len = kv_len
            end_bkv_idx = cdiv(effective_kv_len, bkv_sz)

            # xai temperature computation
            xai_temperature_reg = None
            if xai_temperature_len is not None:
                prefix_len = kv_len - q_len
                local_q_offset = (
                    bq_idx * bq_sz
                    + lax.iota(jnp.int32, actual_bq_sz * num_q_heads_per_kv_head)
                    // num_q_heads_per_kv_head
                )
                absolute_q_position = prefix_len + local_q_offset
                xai_temperature_scale = 1.0 / jnp.log2(float(xai_temperature_len))
                _qtemp = jnp.log2(absolute_q_position.astype(jnp.float32)) * xai_temperature_scale
                xai_temperature_reg = jnp.where(
                    absolute_q_position > xai_temperature_len, _qtemp, 1.0
                )

            # Prefetch next bq
            @pl.when(next_seq_idx < end_seq_idx)
            def prefetch_next_bq():
                sem_ids_ref[0] = next_bq_sem_idx
                start_fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

            @pl.loop(start_bkv_idx, end_bkv_idx, unroll=False)
            def compute_with_bkv(bkv_idx):
                assert bkv_sz % kv_packing == 0

                # Get next bkv ids.
                bkv_sem_idx = sem_ids_ref[1]
                next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx = get_next_bkv_ids(
                    seq_idx, bq_idx, bkv_idx, bkv_sem_idx, num_bkv=end_bkv_idx
                )
                processed_kv_len = bkv_idx * bkv_sz

                # Prefetch next bkv
                @pl.when(next_seq_idx < end_seq_idx)
                def prefetch_next_bkv():
                    sem_ids_ref[1] = next_bkv_sem_idx
                    start_fetch_bkv(next_seq_idx, next_bkv_idx, next_bkv_sem_idx)
                    if custom_mask_ref is not None:
                        start_fetch_mask(next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx)

                # Wait for cur bq if not ready yet
                @pl.when(bkv_idx == start_bkv_idx)
                def wait_cur_bq():
                    wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)

                # Wait for cur bkv
                offset, update_sz = wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

                # Wait for custom mask if applicable
                if custom_mask_ref is not None:
                    wait_fetch_mask(seq_idx, bq_idx, bkv_idx, bkv_sem_idx)

                # Start updating bkv to kv cache if applicable.
                # Only needed in last bq loop.
                @pl.when(jnp.logical_and(update_sz > 0, bq_idx == num_bq - 1))
                def update_cur_bkv_to_cache():
                    start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)

                if debug_mode:
                    return

                # Load custom mask data for this block
                custom_mask_data = None
                if bkvmask_ref is not None:
                    custom_mask_data = bkvmask_ref[bkv_sem_idx, :actual_bq_sz, :, 0]

                # Flash attention with cur bkv and bq
                effective_bkv_sz = jnp.minimum(effective_kv_len - bkv_idx * bkv_sz, bkv_sz)
                effective_bkv_sz = jnp.maximum(effective_bkv_sz, 0)

                # Use static loop bound to avoid potential Pallas issues with
                # dynamic loop bounds. The @pl.when guard skips invalid iterations.
                max_num_loops = bkv_sz // bkv_csz

                @pl.loop(0, max_num_loops, unroll=False)
                def attention_loop(idx):
                    bkv_start = idx * bkv_csz

                    @pl.when(bkv_start < effective_bkv_sz)
                    def _():
                        for bq_start in range(0, actual_bq_sz, actual_bq_csz):
                            # Slice custom mask for this compute sub-block
                            cur_mask_data = None
                            if custom_mask_data is not None:
                                cur_mask_data = bkvmask_ref[
                                    bkv_sem_idx,
                                    pl.ds(bq_start, actual_bq_csz),
                                    pl.ds(bkv_start, bkv_csz),
                                    0,
                                ]

                            # Slice xai temperature for this compute sub-block
                            cur_xai_temp = None
                            if xai_temperature_reg is not None:
                                q_head_start = bq_start * num_q_heads_per_kv_head
                                q_head_sz = actual_bq_csz * num_q_heads_per_kv_head
                                cur_xai_temp = xai_temperature_reg[
                                    q_head_start : q_head_start + q_head_sz
                                ]

                            for kv_head_idx in range(actual_num_kv_heads):
                                bk_c, bv_c = load_bkv(
                                    bkv_sem_idx,
                                    kv_head_idx,
                                    bkv_start,
                                    bkv_csz,
                                )
                                bq_c = load_bq(bq_sem_idx, kv_head_idx, bq_start, actual_bq_csz)

                                lm_slice_start = bq_start * num_q_heads_per_kv_head
                                lm_slice_size = actual_bq_csz * num_q_heads_per_kv_head
                                lm_slice = (kv_head_idx, pl.ds(lm_slice_start, lm_slice_size))

                                cur_p, cur_v, cur_exp_m_diff = flash_attention_step1_qk_softmax(
                                    bq_c,
                                    bk_c,
                                    bv_c,
                                    l_ref.at[*lm_slice],
                                    m_ref.at[*lm_slice],
                                    processed_q_len=processed_q_len + bq_start,
                                    processed_kv_len=processed_kv_len + bkv_start,
                                    effective_kv_len=effective_kv_len,
                                    xai_temperature_reg=cur_xai_temp,
                                    custom_mask_data=cur_mask_data,
                                )
                                flash_attention_step2_pv(
                                    cur_p,
                                    cur_v,
                                    cur_exp_m_diff,
                                    acc_ref.at[*lm_slice],
                                )

            # Load acc and calculate final output.
            acc = acc_ref[...]
            l = broadcast_minor(l_ref[...], acc.shape)  # noqa
            out = (
                acc * pl.reciprocal(l, approx=True)
                if (l.dtype == jnp.float32 and out_dtype != jnp.float32)
                else lax.div(acc, l)
            ).astype(out_dtype)

            # Wait for previous bo to be fully sent before storing new bo.
            bo_sem_idx = sem_ids_ref[2]
            sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)
            wait_send_bo(bo_sem_idx)

            # Store output from acc to bo.
            out_ref = (
                bo_x2_ref.at[bo_sem_idx]
                .bitcast(jnp.int32)
                .reshape(
                    actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head_per_packing,
                    head_dim,
                )
            )
            out = pltpu.bitcast(out, out_ref.dtype).reshape(out_ref.shape)
            strided_store(out_ref, 0, out_ref.shape[0], 1, out)

            # Send cur bo
            start_send_bo(seq_idx, bq_idx, bo_sem_idx)

    ### ------- Kernel start ------- ###

    @pl.when(seq_idx == start_seq_idx)
    def prologue():
        start_fetch_bq(seq_idx=start_seq_idx, bq_idx=0, bq_sem_idx=0)
        # Initialize bkv_x2_ref to zeros to avoid NaN issues.
        # Use bitcast to int32 and preserve actual shape (which may include bank conflict padding)
        bkv_x2_int32_ref = bkv_x2_ref.bitcast(jnp.int32)
        zeros = jnp.zeros(bkv_x2_int32_ref.shape[1:], jnp.int32)
        bkv_x2_int32_ref[0] = zeros
        start_fetch_bkv(seq_idx=start_seq_idx, bkv_idx=cur_seq_start_bkv_idx, bkv_sem_idx=0)
        bkv_x2_int32_ref[1] = zeros
        if custom_mask_ref is not None:
            start_fetch_mask(start_seq_idx, 0, 0, 0)

    @pl.when(jnp.logical_and(start_seq_idx <= seq_idx, seq_idx < end_seq_idx))
    def pipeline():
        process(static_q_len=static_q_len)

    @pl.when(seq_idx == end_seq_idx - 1)
    def epilogue():
        for i in range(2):
            wait_send_bo(bo_sem_idx=i)
            wait_update_kv_cache(bkv_sem_idx=i)

    ### ------- Kernel end ------- ###


def has_bank_conflicts(stride, distance=24, num_banks=32):
    banks = set()
    for i in range(distance):
        bank = (i * stride) % num_banks
        if bank in banks:
            return True
        banks.add(bank)
    return False


def merge_kv(
    k: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    v: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
):
    assert k.shape == v.shape
    assert k.dtype == v.dtype
    max_num_tokens, actual_num_kv_heads, actual_head_dim = k.shape
    kv_packing = get_dtype_packing(k.dtype)
    actual_num_kv_heads_x2 = actual_num_kv_heads * 2
    num_kv_heads_x2 = align_to(actual_num_kv_heads_x2, kv_packing)

    head_dim = align_to(actual_head_dim, 128)
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


def prepare_inputs(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    k: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    v: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    attention_sink: jax.Array | float | None = None,  # f32[actual_num_q_heads]
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

    if attention_sink is not None:
        sink = jnp.asarray(attention_sink, dtype=jnp.float32)
        if sink.ndim == 0:
            sink = jnp.full((actual_num_q_heads,), sink, dtype=jnp.float32)
        sink = sink[: actual_num_kv_heads * actual_num_q_heads_per_kv_head]
        sink = sink.reshape(actual_num_kv_heads, actual_num_q_heads_per_kv_head)
        if num_q_heads_per_kv_head > actual_num_q_heads_per_kv_head:
            sink = jnp.pad(
                sink, ((0, 0), (0, num_q_heads_per_kv_head - actual_num_q_heads_per_kv_head))
            )
        attention_sink = sink.reshape(actual_num_kv_heads, num_q_heads_per_kv_head, 1)
        attention_sink = jnp.repeat(attention_sink, 128, axis=-1)

    return q, kv, attention_sink


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


def prepare_kv_cache_fused(
    kv_cache_fused: jax.Array,  # [total_num_pages, page_size, actual_num_kv_heads * 2, actual_head_dim]
):
    (
        total_num_pages,
        page_size,
        actual_num_kv_heads_interleaved_per_packing,
        packing,
        actual_head_dim,
    ) = kv_cache_fused.shape
    # assert actual_num_kv_heads_interleaved_per_packing % 2 == 0
    head_dim = align_to(actual_head_dim, 128)

    kv_cache_fused_processed = jnp.pad(
        kv_cache_fused,
        (
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
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

    actual_num_kv_heads_interleaved = actual_num_kv_heads * 2
    return kv_cache_fused[:, :, :actual_num_kv_heads_interleaved]


def static_validate_inputs(
    queries,
    keys,
    values,
    kv_cache_fused,
    kv_lens,
    page_indices,
    cu_q_lens,
    cu_kv_lens,
    distribution,
    *,
    causal: int = 1,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    xai_temperature_len: float | None = None,
    chunk_prefill_size: int | None = None,
    d_block_sizes: tuple[int, int, int, int] | None = None,
    p_block_sizes: tuple[int, int, int, int] | None = None,
    m_block_sizes: tuple[int, int, int, int] | None = None,
    vmem_limit_bytes: int | None = None,
    skip_kv_mask: bool = False,
    attention_sink=None,
):
    """Validate inputs to the RPA kernel statically."""
    q, k, v = queries, keys, values
    if not (len(q.shape) == len(k.shape) == len(v.shape) == 3):
        raise ValueError(f"Expected 3D array for {q.shape=}, {k.shape=}, {v.shape=}")
    if k.shape != v.shape:
        raise ValueError(f"Expected {k.shape=} to be equal to {v.shape=}")
    if not (q.shape[0] == k.shape[0] == v.shape[0]):
        raise ValueError(f"Expected {q.shape[0]=} to be equal to {k.shape[0]=} and {v.shape[0]=}")
    if not (q.shape[2] == k.shape[2] == v.shape[2]):
        raise ValueError(f"Expected {q.shape[2]=} to be equal to {k.shape[2]=} and {v.shape[2]=}")

    actual_num_q_heads = q.shape[1]
    actual_num_kv_heads = k.shape[1]

    if actual_num_q_heads % actual_num_kv_heads != 0:
        raise ValueError(
            f"Expected {actual_num_q_heads=} to be divisible by" f" {actual_num_kv_heads=}."
        )

    if kv_cache_fused is not None:
        kv_cache_processed = prepare_kv_cache_fused(kv_cache_fused)
        _, page_size, _, kv_packing, head_dim = kv_cache_processed.shape

        if not jnp.issubdtype(kv_cache_fused.dtype, jnp.floating):
            raise ValueError(f"Expected {kv_cache_fused.dtype=} to be a floating point.")

    if not (
        jnp.int32
        == kv_lens.dtype
        == page_indices.dtype
        == cu_q_lens.dtype
        == cu_kv_lens.dtype
        == distribution.dtype
    ):
        raise ValueError(
            f"Expected int32 dtype for {kv_lens.dtype=}, {page_indices.dtype=},"
            f" {cu_q_lens.dtype=}, {cu_kv_lens.dtype=}, {distribution.dtype=}"
        )

    if not (len(kv_lens.shape) == len(page_indices.shape) == len(cu_q_lens.shape) == 1):
        raise ValueError(
            f"Expected 1D array for {kv_lens.shape=}, {page_indices.shape=}," f" {cu_q_lens.shape=}"
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

    use_causal_mask = causal == 1
    if skip_kv_mask and use_causal_mask:
        raise ValueError("Cannot skip kv mask when using causal mask.")

    def _validate_block_sizes(block_sizes, prefix):
        if block_sizes is None:
            return
        bq_sz, bkv_sz, bq_csz, bkv_csz = block_sizes
        if not (bq_csz > 0 and bq_sz % bq_csz == 0):
            raise ValueError(
                f"{prefix} {bq_csz=} and {bq_sz=} must satisfy (0 < bq_csz and bq_sz"
                " % bq_csz == 0)."
            )
        if not (bkv_csz > 0 and bkv_sz % bkv_csz == 0):
            raise ValueError(
                f"{prefix} {bkv_csz=} and {bkv_sz=} must satisfy (0 < bkv_csz and"
                " bkv_sz % bkv_csz == 0)."
            )
        if kv_cache_fused is not None:
            if bkv_sz % page_size != 0:
                raise ValueError(f"{prefix} {bkv_sz=} must be divisible by {page_size=}.")
            if bkv_csz % page_size != 0:
                raise ValueError(f"{prefix} {bkv_csz=} must be divisible by {page_size=}.")

    _validate_block_sizes(d_block_sizes, "decode")
    _validate_block_sizes(p_block_sizes, "prefill")
    _validate_block_sizes(m_block_sizes, "mixed")

    if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
        raise ValueError(f"{vmem_limit_bytes=} must be positive.")

    del sm_scale
    del mask_value
    del q_scale
    del k_scale
    del v_scale
    del xai_temperature_len
    del skip_kv_mask
    del attention_sink


def dynamic_validate_inputs(
    queries,
    keys,
    values,
    kv_cache_fused,
    kv_lens,
    page_indices,
    cu_q_lens,
    cu_kv_lens,
    distribution,
    **kwargs,
):
    """Runtime validation of dynamic tensor values."""
    i, j, k = distribution[0], distribution[1], distribution[2]
    if not (i <= j <= k):
        raise ValueError(f"distribution must satisfy i <= j <= k, got {i}, {j}, {k}")
    max_num_seqs = kv_lens.shape[0]
    if k > max_num_seqs:
        raise ValueError(f"distribution[2]={k} exceeds {max_num_seqs=}")
    max_num_tokens = queries.shape[0]
    if cu_q_lens[k] > max_num_tokens:
        raise ValueError(f"cu_q_lens[{k}]={cu_q_lens[k]} exceeds {max_num_tokens=}")

    kv_cache_processed = prepare_kv_cache_fused(kv_cache_fused)
    total_num_pages = kv_cache_processed.shape[0]
    page_size = kv_cache_processed.shape[1]
    pages_per_seq = page_indices.shape[0] // max_num_seqs

    for seq_idx in range(int(k)):
        q_len = int(cu_q_lens[seq_idx + 1] - cu_q_lens[seq_idx])
        kv_len = int(kv_lens[seq_idx])
        if q_len <= 0:
            raise ValueError(f"Sequence {seq_idx}: q_len={q_len} must be > 0")
        if q_len > kv_len:
            raise ValueError(f"Sequence {seq_idx}: q_len={q_len} > kv_len={kv_len}")
        page_cnt = cdiv(kv_len, page_size)
        if page_cnt > pages_per_seq:
            raise ValueError(
                f"Sequence {seq_idx}: page_cnt={page_cnt} > pages_per_seq={pages_per_seq}"
            )
        for p in range(page_cnt):
            page_idx = int(page_indices[seq_idx * pages_per_seq + p])
            if page_idx < 0 or page_idx >= total_num_pages:
                raise ValueError(
                    f"Sequence {seq_idx}, page {p}: index={page_idx} "
                    f"out of [0, {total_num_pages})"
                )


def get_default_block_sizes(
    q_dtype,
    kv_dtype,
    actual_num_q_heads,
    actual_num_kv_heads,
    head_dim,
    page_size,
    max_num_tokens,
    max_num_seqs,
    pages_per_seq,
    *,
    case: RpaCase = RpaCase.MIXED,
    vmem_limit_bytes: int | None = None,
    use_custom_mask: bool = False,
    sliding_window: int | None = None,
):
    """Get (bq_sz, bkv_sz, bq_csz, bkv_csz) by some heuristic formulas."""
    tpu_version = get_tpu_version()

    kv_packing = get_dtype_packing(kv_dtype)
    num_kv_heads_x2 = next_power_of_2(align_to(actual_num_kv_heads * 2, kv_packing))
    head_dim = align_to(head_dim, 128)
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    num_q_heads_per_kv_head = next_power_of_2(actual_num_q_heads_per_kv_head)

    max_q = next_power_of_2(max_num_tokens)
    max_kv = pages_per_seq * page_size

    min_bkv_sz_to_peak = 16 * 1024 * 1024 * kv_packing // 4 // head_dim // num_kv_heads_x2

    # Cap bq_sz for non-decode to 32 to match v2 precision characteristics.
    # Larger bq causes bf16 rounding divergence in flash attention accumulation.
    MAX_BQ_SZ = 32

    match tpu_version:
        case 5 | 6:
            if case == RpaCase.DECODE:
                bq_sz = 1
                bkv_sz = (
                    min(min_bkv_sz_to_peak, max_kv) if sliding_window is None else sliding_window
                )
                bq_csz = 1
                bkv_csz = bkv_sz
            else:
                bq_sz = min(MAX_BQ_SZ, max_q // 2)
                bkv_sz = min(1024, max_kv)
                bq_csz = min(MAX_BQ_SZ, min(512 // num_q_heads_per_kv_head, max_q))
                bkv_csz = min(512, align_to(max_kv // 2, page_size)) if max_kv > 1 else page_size
        case 7:
            if case == RpaCase.DECODE:
                bq_sz = 1
                bkv_sz = (
                    min(min_bkv_sz_to_peak, max_kv) if sliding_window is None else sliding_window
                )
                bq_csz = 1
                bkv_csz = bkv_sz
            else:
                bq_sz = min(MAX_BQ_SZ, max_q // 2)
                bkv_sz = min(2048, max_kv // 2)
                bq_csz = min(MAX_BQ_SZ, min(1024 // num_q_heads_per_kv_head, max_q))
                bkv_csz = min(1024, align_to(max_kv // 2, page_size)) if max_kv > 1 else page_size
        case _:
            raise NotImplementedError(f"Unsupported {tpu_version=}.")

    bkv_alignment = max(page_size, kv_packing)
    bq_sz = max(1, bq_sz)
    bkv_sz = align_to(bkv_sz, bkv_alignment)
    bq_csz = max(1, bq_csz)
    bkv_csz = align_to(bkv_csz, bkv_alignment)

    # Reduce block sizes if VMEM estimate exceeds limit.
    # Use 30% of vmem_limit to account for compiler overhead (intermediates,
    # stack, spills, f32 softmax pipeline) not captured by get_vmem_estimate_bytes.
    if vmem_limit_bytes is not None:
        vmem_budget = int(vmem_limit_bytes * 0.30)
        while bq_sz > 1 or bkv_sz > bkv_alignment:
            vmem_est = get_vmem_estimate_bytes(
                actual_num_kv_heads,
                actual_num_q_heads_per_kv_head,
                head_dim,
                bq_sz,
                bkv_sz,
                q_dtype,
                kv_dtype,
                use_custom_mask=use_custom_mask,
                bkv_csz=bkv_csz,
            )
            if vmem_est <= vmem_budget:
                break
            if bkv_sz > bq_sz * bkv_alignment:
                bkv_sz = max(bkv_sz // 2, bkv_alignment)
                bkv_csz = bkv_sz
            else:
                bq_sz = max(bq_sz // 2, 1)
                bq_csz = bq_sz
            bkv_sz = align_to(bkv_sz, bkv_alignment)
            bkv_csz = align_to(bkv_csz, bkv_alignment)

    # Ensure csz <= sz.
    bkv_csz = min(bkv_csz, bkv_sz)
    bq_csz = min(bq_csz, bq_sz)

    # Ensure bkv_sz is evenly divisible by bkv_csz. If not, fall back to
    # bkv_csz = bkv_sz (disabling the nested attention loop for this config).
    if bkv_csz > 0 and bkv_sz % bkv_csz != 0:
        bkv_csz = bkv_sz

    logger.info(
        "get_default_block_sizes: case=%s, tpu_v=%d, bq=%d, bkv=%d, "
        "max_q=%d, max_kv=%d, heads=%d, head_dim=%d, page=%d, "
        "custom_mask=%s, vmem_limit=%s",
        case.symbol,
        tpu_version,
        bq_sz,
        bkv_sz,
        max_q,
        max_kv,
        actual_num_kv_heads,
        head_dim,
        page_size,
        use_custom_mask,
        vmem_limit_bytes,
    )

    return {
        "bq_sz": bq_sz,
        "bkv_sz": bkv_sz,
        "bq_csz": bq_csz,
        "bkv_csz": bkv_csz,
    }


def get_vmem_limit():
    try:
        # Use half of VMEM capacity as default to approximate the scoped VMEM limit.
        # The compiler's scoped VMEM allocation is typically ~50% of total capacity.
        vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes // 2
    except Exception:
        vmem_limit_bytes = DEFAULT_VMEM_LIMIT_BYTES
    return vmem_limit_bytes


@functools.partial(
    jax.jit,
    static_argnames=(
        "causal",
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "mask_value",
        "q_scale",
        "k_scale",
        "v_scale",
        "xai_temperature_len",
        "chunk_prefill_size",
        "d_block_sizes",
        "p_block_sizes",
        "m_block_sizes",
        "vmem_limit_bytes",
        "out_dtype",
        "skip_kv_mask",
        "disable_semaphore_checks",
        "debug_mode",
    ),
    donate_argnames=("queries", "keys", "values", "kv_cache_fused"),
)
def ragged_paged_attention(
    queries: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    kv_cache_fused: jax.Array,  # [total_num_pages, page_size, actual_num_kv_heads * 2, actual_head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[flat]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    cu_kv_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    custom_mask: jax.Array | None,
    attention_sink: jax.Array | None = None,
    *,
    causal: int = 1,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    xai_temperature_len: float | None = None,
    chunk_prefill_size: int | None = None,
    d_block_sizes: tuple[int, int, int, int] | None = None,
    p_block_sizes: tuple[int, int, int, int] | None = None,
    m_block_sizes: tuple[int, int, int, int] | None = None,
    vmem_limit_bytes: int | None = None,
    out_dtype=None,
    skip_kv_mask: bool = False,
    disable_semaphore_checks: bool = True,
    debug_mode: bool = False,
):
    """Ragged paged attention with fused KV cache.

    Args:
      queries: concatenated all sequences' queries.
      keys: concatenated all sequences' keys.
      values: concatenated all sequences' values.
      kv_cache_fused: paged KV cache with head interleaving [K1,V1,K2,V2,...].
      kv_lens: padded kv lengths.
      page_indices: flattened page indices look-up table.
      cu_q_lens: cumulative sum of effective query lengths.
      cu_kv_lens: cumulative sum of effective key/value lengths.
      distribution: (i, j, k) decode/prefill/mixed sequence ranges.
      custom_mask: custom attention mask for speculative decoding.
      attention_sink: per-head sink logits for streaming inference.
      causal: 1 for causal mask, 0 for custom mask.
      sm_scale: softmax scale applied to Q@K^T.
      sliding_window: sliding window size.
      soft_cap: logit soft cap.
      mask_value: mask value for masked positions.
      q_scale: query scale.
      k_scale: key scale.
      v_scale: value scale.
      xai_temperature_len: length-based temperature for Grok-style models.
      chunk_prefill_size: chunk prefill size.
      d_block_sizes: block sizes for decode (bq_sz, bkv_sz, bq_csz, bkv_csz).
      p_block_sizes: block sizes for prefill.
      m_block_sizes: block sizes for mixed.
      vmem_limit_bytes: vmem limit for the pallas kernel.
      debug_mode: if true, skip DMAs and flash attention.

    Returns:
      (output, updated_kv_cache_fused)
    """
    q, k, v = queries, keys, values

    if vmem_limit_bytes is None:
        vmem_limit_bytes = get_vmem_limit()

    static_validate_inputs(
        q,
        k,
        v,
        kv_cache_fused,
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
        causal=causal,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        xai_temperature_len=xai_temperature_len,
        chunk_prefill_size=chunk_prefill_size,
        d_block_sizes=d_block_sizes,
        p_block_sizes=p_block_sizes,
        m_block_sizes=m_block_sizes,
        vmem_limit_bytes=vmem_limit_bytes,
        skip_kv_mask=skip_kv_mask,
        attention_sink=attention_sink,
    )

    actual_num_q_heads = q.shape[1]
    actual_head_dim = q.shape[2]
    actual_num_kv_heads = k.shape[1]
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads

    q, kv, attention_sink = prepare_inputs(q, k, v, attention_sink)
    kv_cache_fused_processed = prepare_kv_cache_fused(kv_cache_fused)

    (
        _,
        max_num_tokens,
        num_q_heads_per_kv_head_per_q_packing,
        q_packing,
        head_dim,
    ) = q.shape
    page_size = kv_cache_fused_processed.shape[1]
    num_kv_heads_x2_per_kv_packing = kv_cache_fused_processed.shape[2]
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    pages_per_seq = num_page_indices // max_num_seqs
    num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_q_packing * q_packing
    if out_dtype is None:
        out_dtype = jnp.float32 if q.dtype == jnp.float32 else jnp.bfloat16

    # Prepare custom mask.
    if custom_mask is not None:
        if custom_mask.dtype == jnp.bool_:
            custom_mask = custom_mask.astype(jnp.int32)
        custom_mask = jnp.repeat(jnp.expand_dims(custom_mask, axis=1), repeats=head_dim, axis=1)

        # Prepare cu_seq_mask_lens for custom mask.
        q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
        seq_mask_lens = kv_lens * q_lens
        cu_seq_mask_lens = jnp.concatenate(
            [jnp.array([0], dtype=jnp.int32), jnp.cumsum(seq_mask_lens)]
        )
    else:
        cu_seq_mask_lens = jnp.array([0])

    # Scalar prefetch init values.
    init_sem_ids = jnp.zeros((3,), jnp.int32)
    init_bo_ids = jnp.full((4,), -1, jnp.int32)
    init_bkv_update_ids = jnp.full((6,), -1, jnp.int32)

    use_causal_mask = causal == 1
    tpu_version = get_tpu_version()

    def run_rpa_kernel(
        q,
        kv_cache,
        *,
        bq_sz,
        bkv_sz,
        bq_csz,
        bkv_csz,
        static_q_len=None,
        case: RpaCase = RpaCase.MIXED,
    ):
        in_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),  # q
            pl.BlockSpec(memory_space=pltpu.HBM),  # kv
            pl.BlockSpec(memory_space=pltpu.HBM),  # kv_cache
            (
                pl.BlockSpec(memory_space=pltpu.HBM) if custom_mask is not None else None
            ),  # custom_mask
            pl.BlockSpec(memory_space=pltpu.HBM),  # zero_mask
            (
                pl.BlockSpec(memory_space=pltpu.VMEM) if attention_sink is not None else None
            ),  # attention_sink
        ]

        out_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ]

        bkv_stride = num_kv_heads_x2_per_kv_packing
        if has_bank_conflicts(bkv_stride):
            bkv_stride += 1

        bkv_double_buf = pltpu.VMEM(
            (2, bkv_sz, bkv_stride, *kv_cache.shape[3:]),
            kv_cache.dtype,
        )

        bq_double_buf = pltpu.VMEM(
            (2, actual_num_kv_heads, bq_sz, *q.shape[2:]),
            q.dtype,
        )

        bo_double_buf = bq_double_buf

        if use_causal_mask:
            bkvmask_double_buf = None
        else:
            bkvmask_double_buf = pltpu.VMEM(
                (2, bq_sz, bkv_sz, head_dim),
                jnp.int32,
            )

        l_scratch = pltpu.VMEM(
            (actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128),
            out_dtype,
        )
        m_scratch = l_scratch

        acc_scratch = pltpu.VMEM(
            (actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, head_dim),
            out_dtype,
        )

        scratch_shapes = [
            bkvmask_double_buf,
            bkv_double_buf,
            bq_double_buf,
            bo_double_buf,
            pltpu.SemaphoreType.DMA((5, 2)),
            l_scratch,
            m_scratch,
            acc_scratch,
        ]

        scalar_prefetches = (
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            cu_seq_mask_lens,
            distribution,
            init_sem_ids,
            init_bo_ids,
            init_bkv_update_ids,
        )

        scope_name = f"RPA{case.symbol}-p_{page_size}-bq_{bq_sz}_{bq_csz}-bkv_{bkv_sz}_{bkv_csz}"
        if sliding_window is not None:
            scope_name += f"-sw_{sliding_window}"

        kernel = pl.pallas_call(
            functools.partial(
                _ragged_paged_attention_kernel,
                causal=use_causal_mask,
                sm_scale=sm_scale,
                sliding_window=sliding_window,
                soft_cap=soft_cap,
                mask_value=mask_value,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                xai_temperature_len=xai_temperature_len,
                static_q_len=static_q_len,
                bq_sz=bq_sz,
                bkv_sz=bkv_sz,
                bq_csz=bq_csz,
                bkv_csz=bkv_csz,
                case=case,
                skip_kv_mask=skip_kv_mask,
                tpu_version=tpu_version,
                debug_mode=debug_mode,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(scalar_prefetches),
                in_specs=in_specs,
                out_specs=out_specs,
                grid=(1,),
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary",),
                vmem_limit_bytes=vmem_limit_bytes,
                disable_bounds_checks=True,
                **(_semaphore_kwargs(disable_semaphore_checks)),
            ),
            out_shape=(
                [
                    pltpu.HBM(shape=q.shape, dtype=q.dtype),
                    pltpu.HBM(shape=kv_cache.shape, dtype=kv_cache.dtype),
                ]
                if tpu_version >= 7
                else [
                    jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
                    jax.ShapeDtypeStruct(shape=kv_cache.shape, dtype=kv_cache.dtype),
                ]
            ),
            input_output_aliases={
                9: 0,  # q -> output
                11: 1,  # kv_cache -> updated_kv_cache
            },
            name=scope_name,
        )

        zero_mask = jnp.zeros((bkv_sz, head_dim), dtype=jnp.int32)

        if tpu_version >= 7:

            @jax.jit
            def run(scalar_prefetches, q, kv, kv_cache):
                return kernel(
                    *scalar_prefetches,
                    pltpu.with_memory_space_constraint(q, pltpu.HBM),
                    pltpu.with_memory_space_constraint(kv, pltpu.HBM),
                    pltpu.with_memory_space_constraint(kv_cache, pltpu.HBM),
                    (
                        pltpu.with_memory_space_constraint(custom_mask, pltpu.HBM)
                        if custom_mask is not None
                        else custom_mask
                    ),
                    zero_mask,
                    attention_sink,
                )

        else:

            def run(scalar_prefetches, q, kv, kv_cache):
                return kernel(
                    *scalar_prefetches,
                    q,
                    kv,
                    kv_cache,
                    custom_mask,
                    zero_mask,
                    attention_sink,
                )

        return run(scalar_prefetches, q, kv, kv_cache)

    def _prepare_block_sizes(block_sizes, case):
        if block_sizes is None:
            return get_default_block_sizes(
                q.dtype,
                kv_cache_fused_processed.dtype,
                actual_num_q_heads,
                actual_num_kv_heads,
                head_dim,
                page_size,
                max_num_tokens,
                max_num_seqs,
                pages_per_seq,
                case=case,
                vmem_limit_bytes=vmem_limit_bytes,
                use_custom_mask=not use_causal_mask,
                sliding_window=sliding_window,
            )

        return {
            "bq_sz": block_sizes[0],
            "bkv_sz": block_sizes[1],
            "bq_csz": block_sizes[2],
            "bkv_csz": block_sizes[3],
        }

    # When chunk_prefill_size is None, PREFILL pallas_call is skipped.
    # Remap prefill sequences to MIXED so they are still processed.
    if chunk_prefill_size is None:
        distribution = distribution.at[1].set(distribution[0])

    # Decode-only
    q, kv_cache_fused_processed = run_rpa_kernel(
        q,
        kv_cache_fused_processed,
        **_prepare_block_sizes(d_block_sizes, RpaCase.DECODE),
        static_q_len=1,
        case=RpaCase.DECODE,
    )

    if chunk_prefill_size is not None:
        # Prefill-only
        q, kv_cache_fused_processed = run_rpa_kernel(
            q,
            kv_cache_fused_processed,
            **_prepare_block_sizes(p_block_sizes, RpaCase.PREFILL),
            static_q_len=chunk_prefill_size,
            case=RpaCase.PREFILL,
        )
    # Mixed
    q, kv_cache_fused_processed = run_rpa_kernel(
        q,
        kv_cache_fused_processed,
        **_prepare_block_sizes(m_block_sizes, RpaCase.MIXED),
        static_q_len=None,
        case=RpaCase.MIXED,
    )

    return (
        prepare_outputs(q, actual_num_q_heads_per_kv_head, actual_head_dim),
        prepare_updated_kv_cache_fused(
            kv_cache_fused_processed, actual_num_kv_heads, actual_head_dim
        ),
    )
