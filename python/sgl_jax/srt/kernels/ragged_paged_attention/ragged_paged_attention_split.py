# Adapted from sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py
# Split K/V variant

import functools
import logging

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
)

logger = logging.getLogger(__name__)

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024  # 100MB

def _ragged_paged_attention_kernel_split(
    # Prefetch
    kv_lens_ref,  # [padded_batch_size]
    page_indices_ref,  # [(padded_batch_size * model_context_len + page_size - 1) // page_size]
    cu_q_lens_ref,  # [padded_batch_size + 1]
    cu_kv_lens_ref,  # [padded_batch_size + 1]
    cu_seq_mask_lens,
    distribution_ref,  # [3] (decode_end, prefill_end, mixed_end)
    sem_ids_ref,  # [3] (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
    bo_ids_ref,  # [4] (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx)
    bkv_update_ids_ref,  # [6] (bkv_sem_0_seq_idx, bkv_sem_1_seq_idx, bkv_sem_0_offset, bkv_sem_1_offset, bkv_sem_0_sz, bkv_sem_1_sz)
    # Input
    q_hbm_ref,  # [actual_num_kv_heads, padded_num_tokens, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    k_hbm_ref,  # [padded_num_tokens, num_kv_heads // kv_packing, kv_packing, k_head_dim]
    v_hbm_ref,  # [padded_num_tokens, num_kv_heads // kv_packing, kv_packing, v_head_dim]
    k_cache_hbm_ref,  # [total_num_pages, page_size, num_kv_heads // kv_packing, kv_packing, k_head_dim]
    v_cache_hbm_ref,  # [total_num_pages, page_size, num_kv_heads // kv_packing, kv_packing, v_head_dim]
    custom_mask_ref,  # (flatten_total_kv_len, head_dim), int32
    attention_sink_ref,  # [actual_num_q_heads_padded] or None
    zero_mask_ref,  # (bkv_sz, head_dim), int32
    # Output
    o_hbm_ref,  # [actual_num_kv_heads, max_num_tokens, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    updated_k_cache_hbm_ref,  # [total_num_pages, page_size, num_kv_heads // kv_packing, kv_packing, k_head_dim]
    updated_v_cache_hbm_ref,  # [total_num_pages, page_size, num_kv_heads // kv_packing, kv_packing, v_head_dim]
    # Scratch
    bkvmask_ref,  # [2, bq_sz, bkv_sz, head_dim]
    bk_x2_ref,  # [2, bkv_sz, num_kv_heads // kv_packing, kv_packing, k_head_dim]
    bv_x2_ref,  # [2, bkv_sz, num_kv_heads // kv_packing, kv_packing, v_head_dim]
    bq_x2_ref,  # [2, actual_num_kv_heads, bq_sz, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    bo_x2_ref,  # [2, actual_num_kv_heads, bq_sz, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    sems,  # [5, 2]
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
    xai_temperature_len: float | None = None,
    chunk_prefill_size: int | None = None,
    bkv_p,
    bq_sz,
):
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
        k_head_dim,
    ) = k_cache_hbm_ref.shape
    v_head_dim = v_cache_hbm_ref.shape[-1]
    
    num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_packing * q_packing
    q_dtype = q_hbm_ref.dtype
    kv_dtype = k_cache_hbm_ref.dtype
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

    if sliding_window is None:
        bkv_idx_start = next_seq_bkv_idx_start = 0
    else:
        bkv_idx_start = jnp.maximum(kv_len - q_len - sliding_window, 0) // bkv_sz
        next_seq_idx = jnp.minimum(seq_idx + 1, num_seqs - 1)
        next_kv_len = kv_lens_ref[next_seq_idx]
        next_q_len = cu_q_lens_ref[next_seq_idx + 1] - q_end
        next_seq_bkv_idx_start = jnp.maximum(next_kv_len - next_q_len - sliding_window, 0) // bkv_sz

    def flash_attention_step1_qk_softmax(
        q,  # [actual_bq_sz * num_q_heads_per_kv_head, head_dim]
        k,  # [bkv_sz, k_head_dim]
        v,  # [bkv_sz, v_head_dim]
        *,
        bkv_idx,
        kv_head_idx,
        q_span,
        k_span,
        mask,
        xai_temperature_reg=None,
    ):
        head_l_ref = l_ref.at[kv_head_idx, : q.shape[0]]
        head_m_ref = m_ref.at[kv_head_idx, : q.shape[0]]

        def load_with_init(ref, init_val):
            return jnp.where(bkv_idx == bkv_idx_start, jnp.full_like(ref, init_val), ref[...])

        if q_scale is not None:
            q = q / q_scale
            if jnp.issubdtype(k.dtype, jnp.floating):
                dtype_info = jnp.finfo(k.dtype)
                minval = float(dtype_info.min)
                maxval = float(dtype_info.max)
                q = jnp.clip(q, min=minval, max=maxval)
            q = q.astype(k.dtype)

        common_dim = min(head_dim, k_head_dim)
        
        q_slice = q[:, :common_dim]
        k_slice = k[:, :common_dim]

        s = jnp.einsum("nd,md->nm", q_slice, k_slice, preferred_element_type=jnp.float32)
        s *= sm_scale
        if k_scale is not None:
            s *= k_scale
        if q_scale is not None:
            s *= q_scale

        if xai_temperature_reg is not None:
            s = s * xai_temperature_reg[:, None]

        if sliding_window is not None:
            mask = jnp.logical_or(mask, q_span - sliding_window >= k_span)

        if soft_cap is not None:
            s = soft_cap * jnp.tanh(s / soft_cap)
        s += jnp.where(mask, mask_value, 0.0)
        s_rowmax = jnp.max(s, axis=1, keepdims=True)
        m_prev = load_with_init(head_m_ref, -jnp.inf)
        m_curr = jnp.maximum(m_prev, s_rowmax)
        head_m_ref[...] = m_curr
        p = jnp.exp(s - broadcast_minor(m_curr, s.shape))

        p_rowsum = jnp.sum(p, axis=1, keepdims=True)
        exp_m_diff = jnp.exp(m_prev - m_curr)
        l_prev = load_with_init(head_l_ref, 0.0)
        l_curr = exp_m_diff * l_prev + p_rowsum
        head_l_ref[...] = l_curr

        return p, exp_m_diff

    def flash_attention_step2_pv(
        q_shape_0,
        v,  # [bkv_sz, v_head_dim]
        p,  # from step1
        exp_m_diff,  # from step1
        *,
        bkv_idx,
        kv_head_idx,
    ):
        head_acc_ref = acc_ref.at[kv_head_idx, :q_shape_0]

        def load_with_init(ref, init_val):
            return jnp.where(bkv_idx == 0, jnp.full_like(ref, init_val), ref[...])

        pv = jnp.einsum("nm,md->nd", p, v, preferred_element_type=jnp.float32)
        if v_scale is not None:
            pv *= v_scale
            
        pv_dim = pv.shape[-1]
        target_dim = head_dim
        if pv_dim < target_dim:
            pv = jnp.pad(pv, ((0, 0), (0, target_dim - pv_dim)))
        elif pv_dim > target_dim:
            pv = pv[:, :target_dim]
            
        o_prev = load_with_init(head_acc_ref, 0.0)
        o_curr = broadcast_minor(exp_m_diff, o_prev.shape) * o_prev + pv
        head_acc_ref[...] = o_curr

    def _async_copy(src, dst, sem, wait):
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
        k_vmem_ref = bk_x2_ref.at[bkv_sem_idx]
        v_vmem_ref = bv_x2_ref.at[bkv_sem_idx]

        k_cache_hbm_shape = k_cache_hbm_ref.shape
        k_cache_hbm_ref_flat = k_cache_hbm_ref.reshape(
            k_cache_hbm_shape[0] * k_cache_hbm_shape[1],
            *k_cache_hbm_shape[2:],
        )
        v_cache_hbm_shape = v_cache_hbm_ref.shape
        v_cache_hbm_ref_flat = v_cache_hbm_ref.reshape(
            v_cache_hbm_shape[0] * v_cache_hbm_shape[1],
            *v_cache_hbm_shape[2:],
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
        bkv_sz_frm_new = jnp.minimum(jnp.maximum(bkv_sz - kv_left_frm_cache, 0), kv_left_frm_new)

        start_kv_page_idx = cdiv(cu_kv_lens_ref[seq_idx], page_size)
        page_indices_offset = start_kv_page_idx + kv_p_start

        wait_update_kv_cache(bkv_sem_idx)

        if not wait:
            def loop_body(i, offset):
                sz = jnp.minimum(page_size, kv_left_frm_cache - i * page_size)
                p_idx = page_indices_ref[page_indices_offset + i]
                _async_copy(
                    k_cache_hbm_ref_flat.at[pl.ds(p_idx * page_size, sz)],
                    k_vmem_ref.at[pl.ds(i * page_size, sz)],
                    sem,
                    wait,
                )
                _async_copy(
                    v_cache_hbm_ref_flat.at[pl.ds(p_idx * page_size, sz)],
                    v_vmem_ref.at[pl.ds(i * page_size, sz)],
                    sem,
                    wait,
                )
                return offset + sz

            offset = lax.fori_loop(0, bkv_p_frm_cache, loop_body, 0, unroll=False)

            size = lax.select(bkv_sz_frm_new > 0, bkv_sz_frm_new, 0)
            new_kv_len_start = q_end - kv_left_frm_new
            
            _async_copy(
                k_hbm_ref.at[pl.ds(new_kv_len_start, size)],
                k_vmem_ref.at[pl.ds(offset, size)],
                sem,
                wait,
            )
            _async_copy(
                v_hbm_ref.at[pl.ds(new_kv_len_start, size)],
                v_vmem_ref.at[pl.ds(offset, size)],
                sem,
                wait,
            )

            return kv_len_start + offset, bkv_sz_frm_new
        else:
            offset = jnp.minimum(kv_left_frm_cache, page_size * bkv_p)
            dst_k = k_vmem_ref.at[pl.ds(0, offset + bkv_sz_frm_new)]
            dst_v = v_vmem_ref.at[pl.ds(0, offset + bkv_sz_frm_new)]
            _async_copy(src=dst_k, dst=dst_k, sem=sem, wait=True)
            _async_copy(src=dst_v, dst=dst_v, sem=sem, wait=True)
            return kv_len_start + offset, bkv_sz_frm_new

    def _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, *, wait=False):
        sem = sems.at[3, bkv_sem_idx]
        k_vmem_ref = bk_x2_ref.at[bkv_sem_idx]
        v_vmem_ref = bv_x2_ref.at[bkv_sem_idx]
        
        bkv_id = offset // bkv_sz
        kv_p_start = offset // page_size
        kv_p_end = cdiv(offset + update_sz, page_size)
        ignore = offset % page_size
        p_ignore = kv_p_start - bkv_id * bkv_p
        start_kv_page_idx = cdiv(cu_kv_lens_ref[seq_idx], page_size)
        page_indices_offset = start_kv_page_idx + kv_p_start

        k_cache_hbm_shape = updated_k_cache_hbm_ref.shape
        k_cache_hbm_ref_flat = updated_k_cache_hbm_ref.reshape(
            k_cache_hbm_shape[0] * k_cache_hbm_shape[1],
            *k_cache_hbm_shape[2:],
        )
        v_cache_hbm_shape = updated_v_cache_hbm_ref.shape
        v_cache_hbm_ref_flat = updated_v_cache_hbm_ref.reshape(
            v_cache_hbm_shape[0] * v_cache_hbm_shape[1],
            *v_cache_hbm_shape[2:],
        )

        if not wait:
            def loop_body(i, states):
                update_sz, ignore = states
                sz = jnp.minimum(page_size - ignore, update_sz)
                p_idx = page_indices_ref[page_indices_offset + i]
                _async_copy(
                    k_vmem_ref.at[pl.ds((p_ignore + i) * page_size + ignore, sz)],
                    k_cache_hbm_ref_flat.at[pl.ds(p_idx * page_size + ignore, sz)],
                    sem,
                    wait,
                )
                _async_copy(
                    v_vmem_ref.at[pl.ds((p_ignore + i) * page_size + ignore, sz)],
                    v_cache_hbm_ref_flat.at[pl.ds(p_idx * page_size + ignore, sz)],
                    sem,
                    wait,
                )
                return update_sz - sz, 0

            lax.fori_loop(
                0,
                kv_p_end - kv_p_start,
                loop_body,
                (update_sz, ignore),
                unroll=False,
            )
        else:
            dst_k = k_cache_hbm_ref_flat.at[pl.ds(0, update_sz)]
            dst_v = v_cache_hbm_ref_flat.at[pl.ds(0, update_sz)]
            _async_copy(src=dst_k, dst=dst_k, sem=sem, wait=True)
            _async_copy(src=dst_v, dst=dst_v, sem=sem, wait=True)

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
        @pl.when(jnp.logical_and(old_seq_idx >= 0, old_seq_idx <= seq_idx))
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
        res = pltpu.bitcast(q_ref[: actual_bq_sz * num_q_heads_per_kv_head_per_packing], q_dtype)
        return res

    def strided_load_kv_separate(bkv_sem_idx, start, step):
        head_idx_start = start // 2
        
        k_ref = bk_x2_ref.at[bkv_sem_idx]
        v_ref = bv_x2_ref.at[bkv_sem_idx]
        
        def unpack_heads(ref, head_start_idx, num_heads_to_load, head_dim_val):
            ref_flat = ref.reshape(bkv_sz, -1, head_dim_val)
            heads = ref_flat[:, head_start_idx : head_start_idx + num_heads_to_load, :]
            res = []
            for i in range(num_heads_to_load):
                res.append(heads[:, i, :])
            return res

        heads_per_load = max(1, kv_packing // 2)
        
        ks = unpack_heads(k_ref, head_idx_start, heads_per_load, k_head_dim)
        vs = unpack_heads(v_ref, head_idx_start, heads_per_load, v_head_dim)
        
        return list(zip(ks, vs))

    def broadcast_minor(src, shape):
        if src.shape == shape:
            return src
        assert src.shape[:-1] == shape[:-1]
        assert src.shape[-1] % 128 == 0
        target_minor = align_to(shape[-1], src.shape[-1])
        return jnp.concatenate([src for _ in range(target_minor // src.shape[-1])], axis=-1)[
            ..., : shape[-1]
        ]

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

            if sliding_window is None:
                next_bkv_idx_start = 0
            else:
                next_bkv_idx_start = lax.select(
                    is_last_bq,
                    next_seq_bkv_idx_start,
                    bkv_idx_start,
                )
            next_bkv_idx = lax.select(is_last_bkv, next_bkv_idx_start, next_bkv_idx)

            return next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx

        def compute_with_bq(bq_idx, _):
            bq_sem_idx = sem_ids_ref[0]
            next_seq_idx, next_bq_idx, next_bq_sem_idx = get_next_bq_ids(
                seq_idx, bq_idx, bq_sem_idx
            )
            
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

            @pl.when(next_seq_idx < num_seqs)
            def prefetch_next_bq():
                sem_ids_ref[0] = next_bq_sem_idx
                start_fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

            def compute_with_bkv(bkv_idx, _):
                bkv_sem_idx = sem_ids_ref[1]
                next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx = get_next_bkv_ids(
                    seq_idx, bq_idx, bkv_idx, bkv_sem_idx
                )

                @pl.when(next_seq_idx < num_seqs)
                def prefetch_next_bkv():
                    sem_ids_ref[1] = next_bkv_sem_idx
                    start_fetch_bkv(next_seq_idx, next_bkv_idx, next_bkv_sem_idx)
                    if custom_mask_ref is not None:
                        start_fetch_mask(next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx)

                @pl.when(bkv_idx == 0)
                def wait_cur_bq():
                    wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)

                offset, update_sz = wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

                if custom_mask_ref is not None:
                    wait_fetch_mask(seq_idx, bq_idx, bkv_idx, bkv_sem_idx)

                @pl.when(jnp.logical_and(update_sz > 0, bq_idx == 0))
                def update_cur_bkv_to_cache():
                    start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)

                def load_mask(q_span, k_span):
                    if bkvmask_ref is None:
                        return q_span < k_span
                    else:
                        mask = bkvmask_ref[bkv_sem_idx, :actual_bq_sz, :, 0]
                        num_q_heads_per_kv_head_mask = jnp.repeat(
                            mask, num_q_heads_per_kv_head, axis=0
                        )
                        return num_q_heads_per_kv_head_mask != 1

                prev_bq_shape_0 = None
                prev_kv_head_bv = None
                prev_kv_head_idx = None
                prev_kv_head_p = None
                prev_kv_head_exp_m_diff = None
                
                heads_per_load = max(1, kv_packing // 2)
                
                q_span = (
                    kv_len
                    - q_len
                    + bq_idx * bq_sz
                    + lax.broadcasted_iota(
                        jnp.int32, (actual_bq_sz * num_q_heads_per_kv_head, bkv_sz), 0
                    )
                    // num_q_heads_per_kv_head
                )
                k_span = bkv_idx * bkv_sz + lax.broadcasted_iota(
                    jnp.int32, (actual_bq_sz * num_q_heads_per_kv_head, bkv_sz), 1
                )
                attn_mask = load_mask(q_span, k_span)

                for kv_head_start in range(0, actual_num_kv_heads, heads_per_load):
                    bkv_lst = strided_load_kv_separate(
                        bkv_sem_idx,
                        kv_head_start * 2,
                        0,
                    )
                    
                    for i in range(len(bkv_lst)):
                        cur_kv_head_idx = kv_head_start + i
                        if cur_kv_head_idx >= actual_num_kv_heads:
                            break
                        cur_kv_head_bq = load_bq(
                            bq_sem_idx, cur_kv_head_idx, actual_bq_sz=actual_bq_sz
                        )

                        bk, bv = bkv_lst[i]
                        
                        cur_kv_head_p, cur_kv_head_exp_m_diff = flash_attention_step1_qk_softmax(
                            cur_kv_head_bq,
                            bk,
                            bv,
                            bkv_idx=bkv_idx,
                            kv_head_idx=cur_kv_head_idx,
                            q_span=q_span,
                            k_span=k_span,
                            mask=attn_mask,
                            xai_temperature_reg=xai_temperature_reg,
                        )
                        if prev_bq_shape_0 is not None:
                            flash_attention_step2_pv(
                                prev_bq_shape_0,
                                prev_kv_head_bv,
                                prev_kv_head_p,
                                prev_kv_head_exp_m_diff,
                                bkv_idx=bkv_idx,
                                kv_head_idx=prev_kv_head_idx,
                            )
                        prev_bq_shape_0 = cur_kv_head_bq.shape[0]
                        prev_kv_head_bv = bv
                        prev_kv_head_p = cur_kv_head_p
                        prev_kv_head_exp_m_diff = cur_kv_head_exp_m_diff
                        prev_kv_head_idx = cur_kv_head_idx

                assert prev_bq_shape_0 is not None
                flash_attention_step2_pv(
                    prev_bq_shape_0,
                    prev_kv_head_bv,
                    prev_kv_head_p,
                    prev_kv_head_exp_m_diff,
                    bkv_idx=bkv_idx,
                    kv_head_idx=prev_kv_head_idx,
                )

            lax.fori_loop(0, num_bkv, compute_with_bkv, None, unroll=False)

            acc = acc_ref[...]
            l = broadcast_minor(l_ref[...], acc.shape)
            out = (
                lax.div(acc, l)
                if q_dtype == jnp.float32
                else (acc * pl.reciprocal(l, approx=True)).astype(q_dtype)
            )
            if attention_sink_ref is not None:
                l_scalar = l_ref[..., 0]
                m_scalar = m_ref[..., 0]
                logsumexp = jnp.log(l_scalar) + m_scalar

                q_rows = l_scalar.shape[1]
                row_ids = lax.broadcasted_iota(jnp.int32, (q_rows,), 0)
                head_in_kv = row_ids % num_q_heads_per_kv_head
                kv_head_ids = lax.broadcasted_iota(jnp.int32, (actual_num_kv_heads, 1), 0)
                global_head = kv_head_ids * num_q_heads_per_kv_head + head_in_kv[None, :]

                sink_logits = attention_sink_ref[global_head].astype(jnp.float32)
                alpha = jnp.reciprocal(1.0 + jnp.exp(sink_logits - logsumexp))
                out = (out.astype(jnp.float32) * alpha[..., None]).astype(q_dtype)

            bo_sem_idx = sem_ids_ref[2]
            sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)
            wait_send_bo(bo_sem_idx)

            bo_x2_ref.at[bo_sem_idx].bitcast(jnp.int32).reshape(
                actual_num_kv_heads,
                bq_sz * num_q_heads_per_kv_head_per_packing,
                head_dim,
            )[...] = pltpu.bitcast(out, jnp.int32)

            start_send_bo(seq_idx, bq_idx, bo_sem_idx)

        lax.fori_loop(0, num_bq, compute_with_bq, None, unroll=False)

    @pl.when(seq_idx == 0)
    def prologue():
        start_fetch_bq(0, 0, 0)
        bk_x2_int32_ref = bk_x2_ref.bitcast(jnp.int32).reshape((2, -1, 8, 128))
        bv_x2_int32_ref = bv_x2_ref.bitcast(jnp.int32).reshape((2, -1, 8, 128))
        zeros_k = jnp.zeros(bk_x2_int32_ref.shape[1:], jnp.int32)
        zeros_v = jnp.zeros(bv_x2_int32_ref.shape[1:], jnp.int32)

        bk_x2_int32_ref[0] = zeros_k
        bv_x2_int32_ref[0] = zeros_v
        
        start_fetch_bkv(0, bkv_idx_start, 0)
        
        bk_x2_int32_ref[1] = zeros_k
        bv_x2_int32_ref[1] = zeros_v

        if custom_mask_ref is not None:
            start_fetch_mask(0, 0, 0, 0)

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
