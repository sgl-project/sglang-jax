# Adapted from https://github.com/vllm-project/tpu-inference
# Copyright 2026 The tpu-inference Authors. All rights reserved.
#
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reference (non-Pallas) implementation of MLA Ragged Paged Attention.

This file vendors only `ref_mla_ragged_paged_attention` and its direct
dependencies (`update_kv_cache`, `dynamic_validate_inputs`,
`static_validate_inputs`, `get_kv_cache_shape`) from tpu-inference's MLA v1
kernel. The full v1 Pallas kernel is intentionally not ported — sgl-jax uses
the v2 kernel for the actual compute path; this v1 reference exists purely
for numerical correctness checks against v2.

Note: this reference still uses the upstream padded `page_indices` layout
(`page_indices_start = i * pages_per_seq`). It is not adapted to sgl-jax's
ragged layout. To compare against the v2 kernel on sgl-jax metadata, callers
must construct padded inputs.
"""

import jax
import jax.numpy as jnp
from jax import lax

from sgl_jax.srt.kernels.mla.v2.kernel import align_to, cdiv, get_dtype_packing

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


def get_kv_cache_shape(
    total_num_pages,
    page_size,
    kv_dim,
    kv_dtype,
):
    kv_packing = get_dtype_packing(kv_dtype)
    return (
        total_num_pages,
        align_to(page_size, kv_packing) // kv_packing,
        kv_packing,
        align_to(kv_dim, 128),
    )


@jax.jit(donate_argnames=("cache_kv"))
def update_kv_cache(
    new_kv_c: jax.Array,  # [num_tokens, actual_lkv_dim]
    new_k_pe: jax.Array,  # [num_tokens, actual_r_dim]
    cache_kv: jax.Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim+r_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
) -> tuple[jax.Array, jax.Array]:
    """Update KV cache with new tokens."""
    actual_r_dim = new_k_pe.shape[-1]
    r_dim = align_to(actual_r_dim, 128)
    if actual_r_dim != r_dim:
        new_k_pe = jnp.pad(new_k_pe, ((0, 0), (0, r_dim - actual_r_dim)), constant_values=0)
    actual_lkv_dim = new_kv_c.shape[-1]
    lkv_dim = align_to(actual_lkv_dim, 128)
    if actual_lkv_dim != lkv_dim:
        new_kv_c = jnp.pad(new_kv_c, ((0, 0), (0, lkv_dim - actual_lkv_dim)), constant_values=0)
    kv_dim = r_dim + lkv_dim
    _, page_size_per_kv_packing, kv_packing, cache_kv_dim = cache_kv.shape
    assert kv_dim == cache_kv_dim
    page_size = page_size_per_kv_packing * kv_packing

    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    pages_per_seq = num_page_indices // max_num_seqs

    def seq_loop_body(i, cache_kv):
        q_start, q_end = cu_q_lens[i], cu_q_lens[i + 1]
        q_len = q_end - q_start
        kv_len = kv_lens[i]

        def token_loop_body(j, cache_kv_):
            token_idx_in_seq = kv_len - q_len + j
            page_num_in_seq = token_idx_in_seq // page_size
            page_indices_start = i * pages_per_seq
            page_idx = page_indices[page_indices_start + page_num_in_seq]
            row = (token_idx_in_seq % page_size) // kv_packing
            col = (token_idx_in_seq % page_size) % kv_packing

            cache_kv_ = cache_kv_.at[page_idx, row, col, ..., :lkv_dim].set(new_kv_c[q_start + j])
            cache_kv_ = cache_kv_.at[page_idx, row, col, ..., lkv_dim:].set(new_k_pe[q_start + j])
            return cache_kv_

        return lax.fori_loop(0, q_len, token_loop_body, cache_kv)

    cache_kv = lax.fori_loop(0, distribution[-1], seq_loop_body, cache_kv)

    return cache_kv


def ref_mla_ragged_paged_attention(
    ql_nope: jax.Array,  # [num_tokens, actual_num_q_heads, actual_lkv_dim]
    q_pe: jax.Array,  # [num_tokens, actual_num_q_heads, actual_r_dim]
    new_kv_c: jax.Array,  # [num_tokens, actual_lkv_dim]
    new_k_pe: jax.Array,  # [num_tokens, actual_r_dim]
    cache_kv: jax.Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
):

    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE

    dynamic_validate_inputs(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    updated_cache_kv = update_kv_cache(
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
    )
    # Pad ql_nope and q_pe to make the last dimension 128-byte aligned.
    actual_lkv_dim = ql_nope.shape[-1]
    lkv_dim = align_to(actual_lkv_dim, 128)
    if lkv_dim != actual_lkv_dim:
        ql_nope = jnp.pad(
            ql_nope,
            ((0, 0), (0, 0), (0, lkv_dim - actual_lkv_dim)),
            constant_values=0,
        )
    actual_r_dim = q_pe.shape[-1]
    r_dim = align_to(actual_r_dim, 128)
    if actual_r_dim != r_dim:
        q_pe = jnp.pad(q_pe, ((0, 0), (0, 0), (0, r_dim - actual_r_dim)), constant_values=0)

    q = jnp.concatenate([ql_nope, q_pe], axis=-1)
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs

    total_num_pages, page_size_per_kv_packing, kv_packing, _ = updated_cache_kv.shape
    page_size = page_size_per_kv_packing * kv_packing
    assert lkv_dim == ql_nope.shape[-1]
    assert r_dim == q_pe.shape[-1]
    assert lkv_dim + r_dim == updated_cache_kv.shape[-1]

    kv_c_cache = updated_cache_kv[..., :lkv_dim].reshape(total_num_pages, page_size, lkv_dim)
    k_pe_cache = updated_cache_kv[..., lkv_dim:].reshape(total_num_pages, page_size, r_dim)

    outputs = []

    for i in range(distribution[-1]):
        q_start, q_end = cu_q_lens[i], cu_q_lens[i + 1]
        q_len = q_end - q_start
        kv_len = kv_lens[i]

        q_i = q[q_start:q_end]  # [q_len, actual_num_q_heads, lkv_dim+r_dim]

        indices_start = i * pages_per_seq
        num_pages_i = cdiv(kv_len, page_size)
        indices_end = indices_start + num_pages_i
        indices = page_indices[indices_start:indices_end]

        # Gather paged kv_c and k_pe
        gathered_kv_c = kv_c_cache[indices]  # [num_pages_i, page_size, lkv_dim]
        gathered_k_pe = k_pe_cache[indices]  # [num_pages_i, page_size, r_dim]

        # Flatten pages to sequence
        flat_kv_c = gathered_kv_c.reshape(-1, lkv_dim)  # [num_pages_i * page_size, lkv_dim]
        flat_k_pe = gathered_k_pe.reshape(-1, r_dim)  # [num_pages_i * page_size, r_dim]

        # Prepare k and v for attention
        k_i = jnp.concatenate(
            [flat_kv_c[:kv_len], flat_k_pe[:kv_len]], axis=-1
        )  # [kv_len, lkv_dim+r_dim]
        v_i = flat_kv_c[:kv_len]  # [kv_len, lkv_dim]

        # MQA attention:
        # q:[q_len, actual_num_q_heads, lkv_dim+r_dim]
        # k:[kv_len, lkv_dim+r_dim]
        # v:[kv_len, lkv_dim]
        # attn: [actual_num_q_heads, q_len, kv_len]
        attn = jnp.einsum("qnh,kh->nqk", q_i, k_i, preferred_element_type=jnp.float32)
        attn *= sm_scale
        if k_scale is not None:
            attn *= k_scale
        if q_scale is not None:
            attn *= q_scale

        # Causal mask
        q_span = kv_len - q_len + jax.lax.broadcasted_iota(jnp.int32, attn.shape, 1)
        kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
        mask = q_span < kv_span
        if sliding_window is not None:
            mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
        if soft_cap is not None:
            attn = soft_cap * jnp.tanh(attn / soft_cap)
        attn = jnp.where(mask, mask_value, attn)
        attn = jax.nn.softmax(attn, axis=-1).astype(v_i.dtype)

        # out_i: [q_len, actual_num_q_heads, lkv_dim]
        out_i = jnp.einsum("nqk,kl->qnl", attn, v_i).astype(q_i.dtype)
        if v_scale is not None:
            out_i *= v_scale
        outputs.append(out_i)

    return (
        jnp.concatenate(outputs, axis=0),
        updated_cache_kv,
    )


# Expect to run this validation during runtime.
def dynamic_validate_inputs(
    ql_nope: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_lkv_dim]
    q_pe: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_r_dim]
    new_kv_c: jax.Array,  # [max_num_tokens, actual_lkv_dim]
    new_k_pe: jax.Array,  # [max_num_tokens, actual_r_dim]
    cache_kv: jax.Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
    # Debug params.
    debug_mode: bool = False,
):
    """Validate inputs to the MLA RPA kernel dynamically."""
    static_validate_inputs(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
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
        debug_mode=debug_mode,
    )


def static_validate_inputs(
    ql_nope: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_lkv_dim]
    q_pe: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_r_dim]
    new_kv_c: jax.Array,  # [max_num_tokens, actual_lkv_dim]
    new_k_pe: jax.Array,  # [max_num_tokens, actual_r_dim]
    cache_kv: jax.Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
    # Debug params.
    debug_mode: bool = False,
):
    """Validate inputs to the MLA RPA kernel statically."""
    if len(ql_nope.shape) != 3:
        raise ValueError(f"Expected 3D array for {ql_nope.shape=}")
    if len(q_pe.shape) != 3:
        raise ValueError(f"Expected 3D array for {q_pe.shape=}")
    if len(new_kv_c.shape) != 2:
        raise ValueError(f"Expected 2D array for {new_kv_c.shape=}")
    if len(new_k_pe.shape) != 2:
        raise ValueError(f"Expected 2D array for {new_k_pe.shape=}")

    if ql_nope.shape[:2] != q_pe.shape[:2]:
        raise ValueError(f"Expected {ql_nope.shape[:2]=} to be equal to {q_pe.shape[:2]=}")
    if ql_nope.shape[0] != new_kv_c.shape[0]:
        raise ValueError(f"Expected {ql_nope.shape[0]=} to be equal to {new_kv_c.shape[0]=}")
    if new_kv_c.shape[0] != new_k_pe.shape[0]:
        raise ValueError(f"Expected {new_kv_c.shape[0]=} to be equal to {new_k_pe.shape[0]=}")
    if ql_nope.shape[2] != new_kv_c.shape[1]:
        raise ValueError(f"Expected {ql_nope.shape[2]=} to be equal to {new_kv_c.shape[1]=}")
    if q_pe.shape[2] != new_k_pe.shape[1]:
        raise ValueError(f"Expected {q_pe.shape[2]=} to be equal to {new_k_pe.shape[1]=}")

    actual_lkv_dim = ql_nope.shape[2]
    actual_r_dim = q_pe.shape[2]
    lkv_dim = align_to(actual_lkv_dim, 128)
    r_dim = align_to(actual_r_dim, 128)

    (
        _,
        page_size_per_kv_packing,
        kv_packing,
        kv_dim,
    ) = cache_kv.shape

    if lkv_dim + r_dim != kv_dim:
        raise ValueError(f"Expected {lkv_dim=} + {r_dim=} to be equal to {kv_dim=}")

    if not (cache_kv.dtype == new_kv_c.dtype):
        raise ValueError(f"Expected {cache_kv.dtype=} to be equal to {new_kv_c.dtype=}.")
    if not (cache_kv.dtype == new_k_pe.dtype):
        raise ValueError(f"Expected {cache_kv.dtype=} to be equal to {new_k_pe.dtype=}.")

    # Integer kv quantization is currently not supported.
    if not jnp.issubdtype(cache_kv.dtype, jnp.floating):
        raise ValueError(f"Expected {cache_kv.dtype=} to be a floating point.")

    if kv_packing != get_dtype_packing(cache_kv.dtype):
        raise ValueError(f"{kv_packing=} does not match with {cache_kv.dtype=}")

    if not (
        jnp.int32 == kv_lens.dtype == page_indices.dtype == cu_q_lens.dtype == distribution.dtype
    ):
        raise ValueError(
            f"Expected int32 dtype for {kv_lens.dtype=}, {page_indices.dtype=},"
            f" {cu_q_lens.dtype=}, {distribution.dtype=}"
        )

    if not (len(kv_lens.shape) == len(page_indices.shape) == len(cu_q_lens.shape) == 1):
        raise ValueError(
            f"Expected 1D array for {kv_lens.shape=}, {page_indices.shape=}, {cu_q_lens.shape=}"
        )

    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    if num_page_indices % max_num_seqs != 0:
        raise ValueError(f"Expected {num_page_indices=} to be divisible by {max_num_seqs=}.")
    if cu_q_lens.shape != (max_num_seqs + 1,):
        raise ValueError(f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},).")
    if distribution.shape != (3,):
        raise ValueError(f"Expected {distribution.shape=} to be (3,).")

    page_size = page_size_per_kv_packing * kv_packing
    if page_size % kv_packing != 0:
        raise ValueError(f"{page_size=} must be divisible by {kv_packing=}.")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError(f"{sliding_window=} must be positive.")
    if soft_cap is not None and soft_cap == 0.0:
        raise ValueError(f"{soft_cap=} must not be 0.0.")
    if chunk_prefill_size is not None and chunk_prefill_size <= 0:
        raise ValueError(f"{chunk_prefill_size=} must be positive.")
    if num_kv_pages_per_block is not None and num_kv_pages_per_block <= 0:
        raise ValueError(f"{num_kv_pages_per_block=} must be positive.")
    if num_queries_per_block is not None and num_queries_per_block <= 0:
        raise ValueError(f"{num_queries_per_block=} must be positive.")
    if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
        raise ValueError(f"{vmem_limit_bytes=} must be positive.")

    # No constraints for the following inputs.
    del sm_scale
    del mask_value
    del q_scale
    del k_scale
    del v_scale
    del debug_mode
