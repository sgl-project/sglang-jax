# Adapted from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3.
# Copyright 2025 The tpu-inference Authors. All rights reserved.
# Modifications Copyright 2026 The SGLang-JAX Authors.
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

"""Variable-length ("varlen") attention for TPU.

Sequences are packed contiguously into a single ``[tokens, heads, head_dim]``
buffer and delimited by ``cu_seqlens`` (the FlashAttention ``flash_attn_varlen``
convention). There is no KV cache: for every sequence ``i`` the Q, K and V
share the interval ``[cu_seqlens[i], cu_seqlens[i + 1])``.

Two kernels back the single ``varlen_attention`` entry point, both a single
``grid=(1,)`` Pallas program that walks every sequence/Q-tile/KV-tile/head with
double-buffered DMAs:

* **MHA fast path** — ``num_q_heads == num_kv_heads`` and BF16 Q/K/V.
* **Packed fallback** — GQA (``num_q_heads != num_kv_heads``) or F32.

Only output rows below ``cu_seqlens[num_seqs]`` are defined.

Why the fast path is faster: the fallback keeps the RPA3 layout that word-packs
``q_per_kv`` for paged-cache compatibility. Without a cache to preserve, the MHA
path drops that and keeps Q/O token-major (``[tokens, heads, head_dim]``), so
``load_bq`` is a plain VMEM slice with no dummy lane, no bitcast/strided-gather,
and write-back is a transpose rather than a repack. Only K/V keep the
word-interleaved (``K0,V0,K1,V1,...``) packing that lets one DMA carry both.
GQA (where packing amortizes) and F32 stay on the fallback.
"""

from __future__ import annotations

import functools
from typing import Final

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.multimodal.kernels.varlen_tuned_block_sizes import (
    DEFAULT_KV_BLOCK,
    DEFAULT_Q_BLOCK,
    get_tuned_block_sizes,
)

DEFAULT_MASK_VALUE: Final[float] = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
DEFAULT_VMEM_LIMIT_BYTES: Final[int] = 120 * 1024 * 1024


# --------------------------------------------------------------------------- #
# Small host-side helpers
# --------------------------------------------------------------------------- #
def _align_to(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((value + alignment - 1) // alignment) * alignment


def _dtype_packing(dtype: jnp.dtype) -> int:
    """Number of values packed into one 32-bit TPU word."""
    bits = jnp.dtype(dtype).itemsize * 8
    if 32 % bits != 0:
        raise ValueError(f"Unsupported dtype bit width: {dtype} ({bits} bits)")
    return 32 // bits


def _has_bank_conflicts(stride: int, distance: int = 24, num_banks: int = 32) -> bool:
    banks: set[int] = set()
    for i in range(distance):
        bank = (i * stride) % num_banks
        if bank in banks:
            return True
        banks.add(bank)
    return False


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def static_validate_inputs(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    cu_seqlens: jax.Array,
    num_seqs: jax.Array,
    *,
    window_size: tuple[int, int] = (-1, -1),
    sm_scale: float | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    attention_sink: jax.Array | float | None = None,
    max_seq_len: int | None = None,
    num_queries_per_block: int | None = None,
    num_kv_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
) -> None:
    """Trace-time shape/dtype checks; safe to call under ``jax.jit``."""
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError(f"q, k and v must be rank-3: got {q.shape=}, {k.shape=}, {v.shape=}")
    if k.shape != v.shape:
        raise ValueError(f"k and v shapes differ: {k.shape} vs {v.shape}")
    if k.dtype != v.dtype:
        raise ValueError(f"k and v dtypes differ: {k.dtype} vs {v.dtype}")
    if q.dtype not in (jnp.bfloat16, jnp.float32):
        raise ValueError(f"Only BF16/F32 Q is supported, got {q.dtype}")
    if k.dtype not in (jnp.bfloat16, jnp.float32):
        raise ValueError(f"Only BF16/F32 K/V is supported, got {k.dtype}")

    _, num_q_heads, q_head_dim = q.shape
    _, num_kv_heads, kv_head_dim = k.shape
    if q_head_dim != kv_head_dim:
        raise ValueError(f"Q and K/V head dimensions differ: {q_head_dim} vs {kv_head_dim}")
    if q_head_dim <= 0:
        raise ValueError("Q and K/V head dimensions must be positive")
    if num_q_heads <= 0:
        raise ValueError("num_q_heads must be positive")
    if num_kv_heads <= 0 or num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads={num_q_heads} must be divisible by num_kv_heads={num_kv_heads}"
        )

    if num_seqs.shape != (1,) or num_seqs.dtype != jnp.int32:
        raise ValueError(f"num_seqs must be int32[1], got {num_seqs.shape=} {num_seqs.dtype=}")
    if cu_seqlens.ndim != 1:
        raise ValueError("cu_seqlens must be rank-1")
    if cu_seqlens.dtype != jnp.int32:
        raise ValueError("cu_seqlens must be int32")
    if cu_seqlens.shape[0] < 2:
        raise ValueError("cu_seqlens must contain at least two entries")

    if not isinstance(window_size, tuple) or len(window_size) != 2:
        raise ValueError(f"window_size must be a (left, right) tuple, got {window_size!r}")
    for name, value in zip(("left", "right"), window_size, strict=True):
        if isinstance(value, bool) or not isinstance(value, int) or value < -1:
            raise ValueError(f"window_size {name} must be -1 or a non-negative int, got {value!r}")
    if attention_sink is not None:
        sink = jnp.asarray(attention_sink)
        if sink.ndim not in (0, 1):
            raise ValueError(
                f"attention_sink must be a scalar or rank-1 [num_q_heads], got shape {sink.shape}"
            )
        if sink.ndim == 1 and sink.shape != (num_q_heads,):
            raise ValueError(f"attention_sink must have shape ({num_q_heads},), got {sink.shape}")
    if soft_cap is not None and soft_cap == 0.0:
        raise ValueError("soft_cap must not be zero")
    if max_seq_len is not None:
        if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int):
            raise ValueError(f"max_seq_len must be a Python int, got {max_seq_len!r}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        capacity = min(q.shape[0], k.shape[0])
        if max_seq_len > capacity:
            raise ValueError(
                f"max_seq_len={max_seq_len} exceeds the packed Q/KV capacity {capacity}"
            )
    if num_queries_per_block is not None and num_queries_per_block <= 0:
        raise ValueError("num_queries_per_block must be positive")
    if num_kv_per_block is not None and num_kv_per_block <= 0:
        raise ValueError("num_kv_per_block must be positive")
    if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
        raise ValueError("vmem_limit_bytes must be positive")

    # Keep names in the signature to make validation calls self-documenting.
    del sm_scale, mask_value, k_scale, v_scale


def dynamic_validate_metadata(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    cu_seqlens: jax.Array,
    num_seqs: jax.Array,
) -> None:
    """Eager host-side validation of packed metadata; not for use under jit."""
    static_validate_inputs(q, k, v, cu_seqlens, num_seqs)
    nseq = int(num_seqs[0])
    max_num_seqs = cu_seqlens.shape[0] - 1
    if not 1 <= nseq <= max_num_seqs:
        raise ValueError(f"num_seqs={nseq} is outside [1, {max_num_seqs}]")
    if int(cu_seqlens[0]) != 0:
        raise ValueError("cu_seqlens[0] must be zero")

    valid_tokens = int(cu_seqlens[nseq])
    if valid_tokens > q.shape[0]:
        raise ValueError("valid tokens exceed q capacity")
    if valid_tokens > k.shape[0]:
        raise ValueError("valid tokens exceed k/v capacity")

    for seq_idx in range(nseq):
        seq_start = int(cu_seqlens[seq_idx])
        seq_end = int(cu_seqlens[seq_idx + 1])
        seq_len = seq_end - seq_start
        if seq_len <= 0:
            raise ValueError(f"sequence {seq_idx}: seq_len={seq_len} must be positive")


# --------------------------------------------------------------------------- #
# Correctness reference
# --------------------------------------------------------------------------- #
def ref_varlen_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    cu_seqlens: jax.Array,
    num_seqs: jax.Array,
    *,
    window_size: tuple[int, int] = (-1, -1),
    sm_scale: float = 1.0,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
    attention_sink: jax.Array | float | None = None,
) -> jax.Array:
    """Pure-JAX reference for packed variable-length prefill attention."""
    static_validate_inputs(
        q,
        k,
        v,
        cu_seqlens,
        num_seqs,
        window_size=window_size,
        sm_scale=sm_scale,
        soft_cap=soft_cap,
        mask_value=mask_value,
        k_scale=k_scale,
        v_scale=v_scale,
        attention_sink=attention_sink,
    )
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE

    nseq = int(num_seqs[0])
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    q_per_kv = num_q_heads // num_kv_heads
    outputs: list[jax.Array] = []

    for seq_idx in range(nseq):
        seq_start = int(cu_seqlens[seq_idx])
        seq_end = int(cu_seqlens[seq_idx + 1])
        seq_len = seq_end - seq_start

        q_seq = q[seq_start:seq_end]
        k_seq = k[seq_start:seq_end]
        v_seq = v[seq_start:seq_end]
        if k_scale is not None:
            k_seq = (k_seq.astype(jnp.float32) * k_scale).astype(q.dtype)
        if v_scale is not None:
            v_seq = (v_seq.astype(jnp.float32) * v_scale).astype(q.dtype)

        k_seq = jnp.repeat(k_seq, q_per_kv, axis=1)
        v_seq = jnp.repeat(v_seq, q_per_kv, axis=1)
        logits = jnp.einsum(
            "qhd,khd->hqk",
            q_seq,
            k_seq,
            preferred_element_type=jnp.float32,
        )
        logits *= sm_scale

        q_pos = jnp.arange(seq_len, dtype=jnp.int32)
        kv_pos = jnp.arange(seq_len, dtype=jnp.int32)
        left_window, right_window = window_size
        keep = jnp.ones((seq_len, seq_len), dtype=jnp.bool_)
        if left_window >= 0:
            keep = jnp.logical_and(keep, kv_pos[None, :] >= q_pos[:, None] - left_window)
        if right_window >= 0:
            keep = jnp.logical_and(keep, kv_pos[None, :] <= q_pos[:, None] + right_window)
        if soft_cap is not None:
            logits = soft_cap * jnp.tanh(logits / soft_cap)
        logits = jnp.where(keep[None, :, :], logits, mask_value)
        if attention_sink is not None:
            sink = jnp.asarray(attention_sink, dtype=jnp.float32)
            if sink.ndim == 0:
                sink = jnp.full((num_q_heads,), sink, dtype=jnp.float32)
            sink_logits = jnp.broadcast_to(
                sink.reshape(num_q_heads, 1, 1),
                (num_q_heads, seq_len, 1),
            )
            probs = jax.nn.softmax(
                jnp.concatenate((sink_logits, logits), axis=-1),
                axis=-1,
            )[..., 1:]
        else:
            probs = jax.nn.softmax(logits, axis=-1)
        out = jnp.einsum(
            "hqk,khd->qhd",
            probs,
            v_seq,
            preferred_element_type=jnp.float32,
        ).astype(q.dtype)
        outputs.append(out)

    if not outputs:
        return jnp.zeros_like(q[:0])
    result = jnp.concatenate(outputs, axis=0)
    # Preserve the public padded-capacity shape.
    return jnp.pad(result, ((0, q.shape[0] - result.shape[0]), (0, 0), (0, 0)))


# --------------------------------------------------------------------------- #
# Shared kernel primitives
# --------------------------------------------------------------------------- #
def _make_async_copy(src, dst, sem, wait: bool):
    cp = pltpu.make_async_copy(src, dst, sem)
    if wait:
        cp.wait()
    else:
        cp.start()


def _strided_load(ref, start, size, step, *, dtype=None):
    """Gather ``size // step`` rows at stride ``step`` from a uint32 ref."""
    assert _dtype_packing(ref.dtype) == 1
    _, minor = ref.shape
    assert minor % 128 == 0
    folds = minor // 128
    ref2 = ref.reshape(ref.shape[0] * folds, 128)
    start *= folds
    size *= folds
    step *= folds
    pieces = [ref2[pl.ds(start + i, size // step, step)] for i in range(folds)]
    vec = jnp.concatenate(pieces, axis=1)
    if dtype is not None:
        vec = pltpu.bitcast(vec, dtype)
    return vec


def _broadcast_minor(src, shape):
    if src.shape == shape:
        return src
    target_minor = _align_to(shape[-1], src.shape[-1])
    return jnp.concatenate(
        [src for _ in range(target_minor // src.shape[-1])],
        axis=-1,
    )[..., : shape[-1]]


# =========================================================================== #
# MHA fast path (BF16, num_q_heads == num_kv_heads)
# =========================================================================== #
def _prepare_mha_layout(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    attention_sink: jax.Array | float | None,
    *,
    bq_sz: int,
) -> tuple[jax.Array, jax.Array, jax.Array | None, int, int]:
    """Build token-major exact-head Q/O and interleaved word-packed K/V."""
    original_q_capacity, num_heads, actual_head_dim = q.shape
    padded_head_dim = _align_to(actual_head_dim, 128)
    q_internal = jnp.pad(
        q,
        ((0, bq_sz - 1), (0, 0), (0, padded_head_dim - actual_head_dim)),
        constant_values=0,
    )

    sink_internal = None
    if attention_sink is not None:
        sink = jnp.asarray(attention_sink, dtype=jnp.float32)
        if sink.ndim == 0:
            sink = jnp.full((num_heads,), sink, dtype=jnp.float32)
        sink_internal = jnp.repeat(sink[:, None], 128, axis=-1)

    kv_packing = _dtype_packing(k.dtype)
    actual_combined_kv_heads = 2 * num_heads
    padded_combined_kv_heads = _align_to(actual_combined_kv_heads, kv_packing)
    # K0,V0,K1,V1,... makes K and V for one head share one BF16 word.
    kv_interleaved = jnp.stack((k, v), axis=2).reshape(
        k.shape[0], actual_combined_kv_heads, actual_head_dim
    )
    # KV carries no token-axis tail slack: the kernel clamps the final block's
    # DMA to the packed extent (RPA3-style) instead of reading a padded tail.
    kv_padded = jnp.pad(
        kv_interleaved,
        (
            (0, 0),
            (0, padded_combined_kv_heads - actual_combined_kv_heads),
            (0, padded_head_dim - actual_head_dim),
        ),
        constant_values=0,
    )
    kv_internal = kv_padded.reshape(
        kv_padded.shape[0],
        padded_combined_kv_heads // kv_packing,
        kv_packing,
        padded_head_dim,
    )
    return q_internal, kv_internal, sink_internal, actual_head_dim, original_q_capacity


def _restore_mha_output(
    out: jax.Array,
    actual_head_dim: int,
    original_q_capacity: int,
) -> jax.Array:
    """Restore public ``[T, H, D]`` layout and remove head-dim padding."""
    return out[:original_q_capacity, :, :actual_head_dim]


def _mha_kernel(
    # Scalar prefetch.
    cu_lens_ref,
    num_seqs_ref,
    sem_ids_ref,
    bo_ids_ref,
    # HBM inputs.
    q_hbm_ref,
    kv_hbm_ref,
    attention_sink_ref,
    # HBM output.
    o_hbm_ref,
    # VMEM scratch.
    bkv_x2_ref,
    bq_x2_ref,
    bo_x2_ref,
    sems,
    l_ref,
    m_ref,
    acc_ref,
    *,
    window_size: tuple[int, int],
    sm_scale: float,
    soft_cap: float | None,
    mask_value: float,
    k_scale: float | None,
    v_scale: float | None,
    bq_sz: int,
    bkv_sz: int,
):
    """One Pallas program containing all sequence, tile, and head loops."""
    num_seqs = num_seqs_ref[0]

    @pl.loop(0, num_seqs, unroll=False)
    def loop_sequence(seq_idx):
        _mha_sequence(
            seq_idx,
            num_seqs,
            cu_lens_ref,
            sem_ids_ref,
            bo_ids_ref,
            q_hbm_ref,
            kv_hbm_ref,
            attention_sink_ref,
            o_hbm_ref,
            bkv_x2_ref,
            bq_x2_ref,
            bo_x2_ref,
            sems,
            l_ref,
            m_ref,
            acc_ref,
            window_size=window_size,
            sm_scale=sm_scale,
            soft_cap=soft_cap,
            mask_value=mask_value,
            k_scale=k_scale,
            v_scale=v_scale,
            bq_sz=bq_sz,
            bkv_sz=bkv_sz,
        )


def _mha_sequence(
    seq_idx,
    num_seqs,
    cu_lens_ref,
    sem_ids_ref,
    bo_ids_ref,
    q_hbm_ref,
    kv_hbm_ref,
    attention_sink_ref,
    o_hbm_ref,
    bkv_x2_ref,
    bq_x2_ref,
    bo_x2_ref,
    sems,
    l_ref,
    m_ref,
    acc_ref,
    *,
    window_size: tuple[int, int],
    sm_scale: float,
    soft_cap: float | None,
    mask_value: float,
    k_scale: float | None,
    v_scale: float | None,
    bq_sz: int,
    bkv_sz: int,
):
    _, num_heads, head_dim = q_hbm_ref.shape
    _, combined_kv_heads_per_packing, kv_packing, _ = kv_hbm_ref.shape
    bkv_stride = bkv_x2_ref.shape[2]
    q_dtype = q_hbm_ref.dtype
    kv_dtype = kv_hbm_ref.dtype
    left_window, right_window = window_size

    q_start = cu_lens_ref[seq_idx]
    q_end = cu_lens_ref[seq_idx + 1]
    seq_len = q_end - q_start
    num_bq = pl.cdiv(seq_len, bq_sz)

    def _fetch_bq(target_seq_idx, bq_idx, sem_idx, *, wait=False):
        start = cu_lens_ref[target_seq_idx] + bq_idx * bq_sz
        _make_async_copy(
            q_hbm_ref.at[pl.ds(start, bq_sz)],
            bq_x2_ref.at[sem_idx],
            sems.at[1, sem_idx],
            wait,
        )

    def _fetch_bkv(target_seq_idx, bkv_idx, sem_idx, *, wait=False):
        start = cu_lens_ref[target_seq_idx] + bkv_idx * bkv_sz
        # KV is not tail-padded, so clamp the final block's DMA to the packed
        # extent.  Every issued fetch satisfies ``start < capacity`` (blocks stay
        # within their sequence), so ``load_sz`` is in ``[1, bkv_sz]``.  Rows past
        # ``load_sz`` keep stale VMEM but are masked out via ``physical_kv_len``.
        load_sz = jnp.minimum(bkv_sz, kv_hbm_ref.shape[0] - start)
        _make_async_copy(
            kv_hbm_ref.at[pl.ds(start, load_sz)],
            bkv_x2_ref.at[sem_idx, pl.ds(0, load_sz), :combined_kv_heads_per_packing],
            sems.at[0, sem_idx],
            wait,
        )

    def _send_bo(target_seq_idx, bq_idx, sem_idx, *, wait=False):
        start = cu_lens_ref[target_seq_idx] + bq_idx * bq_sz
        _make_async_copy(
            bo_x2_ref.at[sem_idx],
            o_hbm_ref.at[pl.ds(start, bq_sz)],
            sems.at[2, sem_idx],
            wait,
        )

    def start_fetch_bq(target_seq_idx, bq_idx, sem_idx):
        _fetch_bq(target_seq_idx, bq_idx, sem_idx)

    def wait_fetch_bq(target_seq_idx, bq_idx, sem_idx):
        _fetch_bq(target_seq_idx, bq_idx, sem_idx, wait=True)

    def start_fetch_bkv(target_seq_idx, bkv_idx, sem_idx):
        _fetch_bkv(target_seq_idx, bkv_idx, sem_idx)

    def wait_fetch_bkv(target_seq_idx, bkv_idx, sem_idx):
        _fetch_bkv(target_seq_idx, bkv_idx, sem_idx, wait=True)

    def start_send_bo(target_seq_idx, bq_idx, sem_idx):
        bo_ids_ref[sem_idx] = target_seq_idx
        bo_ids_ref[sem_idx + 2] = bq_idx
        _send_bo(target_seq_idx, bq_idx, sem_idx)

    def wait_send_bo(sem_idx):
        old_seq_idx = bo_ids_ref[sem_idx]
        old_bq_idx = bo_ids_ref[sem_idx + 2]

        @pl.when(jnp.logical_and(old_seq_idx >= 0, old_seq_idx <= seq_idx))
        def wait_old_output():
            _send_bo(old_seq_idx, old_bq_idx, sem_idx, wait=True)
            # A waited DMA semaphore cannot be waited a second time.  Mark the
            # slot empty so a sequence-boundary drain is safe before the usual
            # per-tile buffer reuse below.
            bo_ids_ref[sem_idx] = -1

    def load_bq(sem_idx, head_idx):
        return bq_x2_ref[sem_idx, :, head_idx, :]

    def load_bkv(sem_idx, head_idx):
        packed_ref = (
            bkv_x2_ref.bitcast(jnp.uint32).at[sem_idx].reshape(bkv_sz * bkv_stride, head_dim)
        )
        if kv_packing == 1:
            start = head_idx * 2
            k_block = _strided_load(
                packed_ref, start, bkv_sz * bkv_stride, bkv_stride, dtype=kv_dtype
            )
            v_block = _strided_load(
                packed_ref, start + 1, bkv_sz * bkv_stride, bkv_stride, dtype=kv_dtype
            )
            return k_block, v_block

        kv_heads_per_word = kv_packing // 2
        word_offset = head_idx // kv_heads_per_word
        kv_idx_in_word = head_idx % kv_heads_per_word
        packed = _strided_load(packed_ref, word_offset, bkv_sz * bkv_stride, bkv_stride)
        bitwidth = 32 // kv_packing
        repack_ty = jnp.dtype(f"uint{bitwidth}")
        k_bits = packed >> (kv_idx_in_word * 2 * bitwidth)
        v_bits = k_bits >> bitwidth
        return (
            pltpu.bitcast(k_bits.astype(repack_ty), kv_dtype),
            pltpu.bitcast(v_bits.astype(repack_ty), kv_dtype),
        )

    def flash_attention_step(
        q_block,
        k_block,
        v_block,
        l_head_ref,
        m_head_ref,
        acc_head_ref,
        *,
        processed_q,
        processed_kv,
        physical_kv_len,
    ):
        logits = jnp.matmul(q_block, k_block.T, preferred_element_type=jnp.float32)
        logits *= sm_scale
        if k_scale is not None:
            logits *= k_scale
        if soft_cap is not None:
            logits = soft_cap * jnp.tanh(logits / soft_cap)

        q_span = processed_q + lax.broadcasted_iota(jnp.int32, logits.shape, 0)
        k_span = processed_kv + lax.broadcasted_iota(jnp.int32, logits.shape, 1)
        v_span = processed_kv + lax.broadcasted_iota(jnp.int32, v_block.shape, 0)
        valid_kv = k_span < physical_kv_len
        keep = jnp.ones_like(valid_kv)
        if left_window >= 0:
            keep = jnp.logical_and(keep, k_span >= q_span - left_window)
        if right_window >= 0:
            keep = jnp.logical_and(keep, k_span <= q_span + right_window)
        logits = jnp.where(keep, logits, mask_value)
        # ``mask_value`` controls semantic window masking, but fixed-tile rows
        # beyond the real KV extent do not exist and must have zero softmax
        # probability even when the caller chooses a finite mask value.
        logits = jnp.where(valid_kv, logits, -jnp.inf)
        v_block = jnp.where(v_span < physical_kv_len, v_block, 0.0)

        rowmax = jnp.max(logits, axis=1, keepdims=True)
        m_prev = m_head_ref[...]
        m_curr = jnp.maximum(m_prev, rowmax)
        m_head_ref[...] = m_curr
        p = jnp.exp(logits - _broadcast_minor(m_curr, logits.shape))
        rowsum = jnp.sum(p, axis=1, keepdims=True)
        exp_diff = jnp.exp(m_prev - m_curr)
        l_prev = l_head_ref[...]
        l_head_ref[...] = exp_diff * l_prev + rowsum

        pv = jnp.matmul(p, v_block, preferred_element_type=jnp.float32)
        if v_scale is not None:
            pv *= v_scale
        acc_prev = acc_head_ref[...]
        acc_head_ref[...] = _broadcast_minor(exp_diff, acc_prev.shape) * acc_prev + pv

    def next_q_ids(bq_idx, sem_idx):
        next_bq_idx = bq_idx + 1
        is_last = next_bq_idx == num_bq
        next_bq_idx = lax.select(is_last, 0, next_bq_idx)
        next_seq_idx = lax.select(is_last, seq_idx + 1, seq_idx)
        return next_seq_idx, next_bq_idx, 1 - sem_idx

    @pl.when(seq_idx == 0)
    def prologue():
        start_fetch_bq(0, 0, 0)
        start_fetch_bkv(0, 0, 0)

    previous_seq_idx = jnp.maximum(seq_idx - 1, 0)
    previous_seq_len = q_start - cu_lens_ref[previous_seq_idx]

    @pl.when(jnp.logical_and(seq_idx > 0, jnp.mod(previous_seq_len, bq_sz) != 0))
    def drain_previous_sequence_outputs():
        # A final partial Q tile still issues a full-block DMA.  Drain both
        # output slots before the next packed sequence writes the overlapping
        # public token range, making the write order deterministic.
        wait_send_bo(0)
        wait_send_bo(1)

    @pl.loop(0, num_bq, unroll=False)
    def loop_q_block(bq_idx):
        acc_ref[...] = jnp.zeros_like(acc_ref)
        if attention_sink_ref is not None:
            l_ref[...] = jnp.ones_like(l_ref)
            m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
            for head_idx in range(num_heads):
                sink = attention_sink_ref[head_idx]
                m_ref.at[head_idx][...] = jnp.tile(sink[None, :], (bq_sz, 1))
        else:
            l_ref[...] = jnp.zeros_like(l_ref)
            m_ref[...] = jnp.full_like(m_ref, -jnp.inf)

        q_sem_idx = sem_ids_ref[0]
        next_seq_idx, next_bq_idx, next_q_sem_idx = next_q_ids(bq_idx, q_sem_idx)

        @pl.when(next_seq_idx < num_seqs)
        def prefetch_next_q():
            sem_ids_ref[0] = next_q_sem_idx
            start_fetch_bq(next_seq_idx, next_bq_idx, next_q_sem_idx)

        processed_q = bq_idx * bq_sz
        prune_window_blocks = mask_value == DEFAULT_MASK_VALUE or mask_value == float("-inf")
        start_bkv_idx = 0
        if left_window >= 0 and prune_window_blocks:
            start_bkv_idx = jnp.maximum(processed_q - left_window, 0) // bkv_sz
        iteration_kv_end = seq_len
        if right_window >= 0 and prune_window_blocks:
            iteration_kv_end = jnp.minimum(seq_len, processed_q + bq_sz + right_window)
        end_bkv_idx = pl.cdiv(iteration_kv_end, bkv_sz)

        @pl.loop(start_bkv_idx, end_bkv_idx, unroll=False)
        def loop_kv_block(bkv_idx):
            kv_sem_idx = sem_ids_ref[1]
            next_bkv_idx = bkv_idx + 1
            is_last_bkv = next_bkv_idx == end_bkv_idx
            next_q_for_kv = lax.select(is_last_bkv, bq_idx + 1, bq_idx)
            is_last_q = next_q_for_kv == num_bq
            next_q_for_kv = lax.select(is_last_q, 0, next_q_for_kv)
            next_seq_for_kv = lax.select(is_last_q, seq_idx + 1, seq_idx)
            next_bkv_idx = lax.select(is_last_bkv, 0, next_bkv_idx)
            if left_window >= 0 and prune_window_blocks:
                next_q_start_bkv = jnp.maximum(next_q_for_kv * bq_sz - left_window, 0) // bkv_sz
                next_bkv_idx = lax.select(is_last_bkv, next_q_start_bkv, next_bkv_idx)
            next_kv_sem_idx = 1 - kv_sem_idx

            @pl.when(next_seq_for_kv < num_seqs)
            def prefetch_next_kv():
                sem_ids_ref[1] = next_kv_sem_idx
                start_fetch_bkv(next_seq_for_kv, next_bkv_idx, next_kv_sem_idx)

            @pl.when(bkv_idx == start_bkv_idx)
            def wait_q():
                wait_fetch_bq(seq_idx, bq_idx, q_sem_idx)

            wait_fetch_bkv(seq_idx, bkv_idx, kv_sem_idx)
            processed_kv = bkv_idx * bkv_sz

            for head_idx in range(num_heads):
                q_block = load_bq(q_sem_idx, head_idx)
                k_block, v_block = load_bkv(kv_sem_idx, head_idx)
                lm_slice = (head_idx, pl.ds(0, bq_sz))
                flash_attention_step(
                    q_block,
                    k_block,
                    v_block,
                    l_ref.at[*lm_slice],
                    m_ref.at[*lm_slice],
                    acc_ref.at[*lm_slice],
                    processed_q=processed_q,
                    processed_kv=processed_kv,
                    physical_kv_len=seq_len,
                )

        acc = acc_ref[...]
        denominator = _broadcast_minor(l_ref[...], acc.shape)
        out = lax.div(acc, jnp.where(denominator == 0.0, 1.0, denominator)).astype(q_dtype)

        bo_sem_idx = sem_ids_ref[2]
        sem_ids_ref[2] = 1 - bo_sem_idx
        wait_send_bo(bo_sem_idx)
        bo_x2_ref.at[bo_sem_idx][...] = jnp.transpose(out, (1, 0, 2))
        start_send_bo(seq_idx, bq_idx, bo_sem_idx)

    @pl.when(seq_idx == num_seqs - 1)
    def epilogue():
        wait_send_bo(0)
        wait_send_bo(1)


def _mha_attention(
    q_internal: jax.Array,
    kv_internal: jax.Array,
    cu_seqlens: jax.Array,
    num_seqs: jax.Array,
    attention_sink_internal: jax.Array | None,
    *,
    window_size: tuple[int, int],
    sm_scale: float,
    soft_cap: float | None,
    mask_value: float,
    k_scale: float | None,
    v_scale: float | None,
    num_queries_per_block: int,
    num_kv_per_block: int,
    vmem_limit_bytes: int | None,
) -> jax.Array:
    """Launch the token-major MHA Pallas kernel on a prepared layout."""
    _, num_heads, head_dim = q_internal.shape
    combined_kv_heads_per_packing = kv_internal.shape[1]
    kv_packing = kv_internal.shape[2]
    bkv_stride = combined_kv_heads_per_packing
    if _has_bank_conflicts(bkv_stride):
        bkv_stride += 1

    bkv_double_buffer = pltpu.VMEM(
        (2, num_kv_per_block, bkv_stride, kv_packing, head_dim),
        kv_internal.dtype,
    )
    bq_double_buffer = pltpu.VMEM(
        (2, num_queries_per_block, num_heads, head_dim),
        q_internal.dtype,
    )
    bo_double_buffer = pltpu.VMEM(
        (2, num_queries_per_block, num_heads, head_dim),
        q_internal.dtype,
    )
    lm_scratch = pltpu.VMEM((num_heads, num_queries_per_block, 128), jnp.float32)
    acc_scratch = pltpu.VMEM((num_heads, num_queries_per_block, head_dim), jnp.float32)

    scalar_prefetches = (
        cu_seqlens,
        num_seqs,
        jnp.zeros((3,), dtype=jnp.int32),
        jnp.full((4,), -1, dtype=jnp.int32),
    )
    in_specs = (
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec(memory_space=pltpu.HBM),
        (pl.BlockSpec(memory_space=pltpu.VMEM) if attention_sink_internal is not None else None),
    )
    kernel = pl.pallas_call(
        functools.partial(
            _mha_kernel,
            window_size=window_size,
            sm_scale=sm_scale,
            soft_cap=soft_cap,
            mask_value=mask_value,
            k_scale=k_scale,
            v_scale=v_scale,
            bq_sz=num_queries_per_block,
            bkv_sz=num_kv_per_block,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_prefetches),
            in_specs=in_specs,
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            grid=(1,),
            scratch_shapes=(
                bkv_double_buffer,
                bq_double_buffer,
                bo_double_buffer,
                pltpu.SemaphoreType.DMA((3, 2)),
                lm_scratch,
                lm_scratch,
                acc_scratch,
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        out_shape=jax.ShapeDtypeStruct(q_internal.shape, q_internal.dtype),
        name="varlen_attention_mha",
    )
    return kernel(*scalar_prefetches, q_internal, kv_internal, attention_sink_internal)


# =========================================================================== #
# Packed fallback (GQA or F32)
# =========================================================================== #
def _prepare_packed_layout(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    attention_sink: jax.Array | float | None,
    *,
    bq_sz: int,
    bkv_sz: int,
) -> tuple[jax.Array, jax.Array, jax.Array | None, int, int, int]:
    """Pack Q/KV into complete 32-bit words and reserve static-DMA tail slack."""
    original_q_capacity, actual_num_q_heads, actual_head_dim = q.shape
    q_capacity = _align_to(original_q_capacity + bq_sz, bq_sz)
    q = jnp.pad(q, ((0, q_capacity - original_q_capacity), (0, 0), (0, 0)))
    actual_num_kv_heads = k.shape[1]
    actual_q_per_kv = actual_num_q_heads // actual_num_kv_heads

    q_packing = _dtype_packing(q.dtype)
    kv_packing = _dtype_packing(k.dtype)
    padded_q_per_kv = _align_to(actual_q_per_kv, q_packing)
    padded_head_dim = _align_to(actual_head_dim, 128)

    q_internal = (
        jnp.pad(
            q.reshape(q_capacity, actual_num_kv_heads, actual_q_per_kv, actual_head_dim),
            (
                (0, 0),
                (0, 0),
                (0, padded_q_per_kv - actual_q_per_kv),
                (0, padded_head_dim - actual_head_dim),
            ),
            constant_values=0,
        )
        .reshape(
            q_capacity,
            actual_num_kv_heads,
            padded_q_per_kv // q_packing,
            q_packing,
            padded_head_dim,
        )
        .swapaxes(0, 1)
    )

    original_kv_capacity = k.shape[0]
    kv_capacity = _align_to(original_kv_capacity + bkv_sz, bkv_sz)
    k = jnp.pad(k, ((0, kv_capacity - original_kv_capacity), (0, 0), (0, 0)))
    v = jnp.pad(v, ((0, kv_capacity - original_kv_capacity), (0, 0), (0, 0)))
    actual_combined_kv_heads = actual_num_kv_heads * 2
    padded_combined_kv_heads = _align_to(actual_combined_kv_heads, kv_packing)
    kv_internal = jnp.pad(
        jnp.concatenate((k, v), axis=-1).reshape(
            kv_capacity, actual_combined_kv_heads, actual_head_dim
        ),
        (
            (0, 0),
            (0, padded_combined_kv_heads - actual_combined_kv_heads),
            (0, padded_head_dim - actual_head_dim),
        ),
        constant_values=0,
    ).reshape(
        kv_capacity,
        padded_combined_kv_heads // kv_packing,
        kv_packing,
        padded_head_dim,
    )

    sink_internal = None
    if attention_sink is not None:
        sink = jnp.asarray(attention_sink, dtype=jnp.float32)
        if sink.ndim == 0:
            sink = jnp.full((actual_num_q_heads,), sink, dtype=jnp.float32)
        sink = sink.reshape(actual_num_kv_heads, actual_q_per_kv)
        if padded_q_per_kv > actual_q_per_kv:
            sink = jnp.pad(
                sink,
                ((0, 0), (0, padded_q_per_kv - actual_q_per_kv)),
                constant_values=0.0,
            )
        sink_internal = jnp.repeat(sink[..., None], 128, axis=-1)

    return (
        q_internal,
        kv_internal,
        sink_internal,
        actual_q_per_kv,
        actual_head_dim,
        original_q_capacity,
    )


def _restore_packed_output(
    out: jax.Array,
    actual_q_per_kv: int,
    actual_head_dim: int,
) -> jax.Array:
    num_kv_heads, token_capacity, q_per_kv_per_packing, q_packing, _ = out.shape
    return (
        out.swapaxes(0, 1)
        .reshape(
            token_capacity,
            num_kv_heads,
            q_per_kv_per_packing * q_packing,
            out.shape[-1],
        )[:, :, :actual_q_per_kv, :actual_head_dim]
        .reshape(token_capacity, num_kv_heads * actual_q_per_kv, actual_head_dim)
    )


def _packed_kernel(
    # Scalar prefetch.
    cu_lens_ref,
    num_seqs_ref,
    sem_ids_ref,
    bo_ids_ref,
    # HBM inputs.
    q_hbm_ref,
    kv_hbm_ref,
    attention_sink_ref,
    # HBM output.
    o_hbm_ref,
    # VMEM scratch.
    bkv_x2_ref,
    bq_x2_ref,
    bo_x2_ref,
    sems,
    l_ref,
    m_ref,
    acc_ref,
    *,
    window_size: tuple[int, int],
    sm_scale: float,
    soft_cap: float | None,
    mask_value: float,
    k_scale: float | None,
    v_scale: float | None,
    bq_sz: int,
    bkv_sz: int,
):
    """Single-program wrapper; all sequence/Q/KV/head loops live in the kernel."""
    num_seqs = num_seqs_ref[0]

    @pl.loop(0, num_seqs)
    def loop_sequence(seq_idx):
        _packed_sequence(
            seq_idx,
            num_seqs,
            cu_lens_ref,
            sem_ids_ref,
            bo_ids_ref,
            q_hbm_ref,
            kv_hbm_ref,
            attention_sink_ref,
            o_hbm_ref,
            bkv_x2_ref,
            bq_x2_ref,
            bo_x2_ref,
            sems,
            l_ref,
            m_ref,
            acc_ref,
            window_size=window_size,
            sm_scale=sm_scale,
            soft_cap=soft_cap,
            mask_value=mask_value,
            k_scale=k_scale,
            v_scale=v_scale,
            bq_sz=bq_sz,
            bkv_sz=bkv_sz,
        )


def _packed_sequence(
    seq_idx,
    num_seqs,
    cu_lens_ref,
    sem_ids_ref,
    bo_ids_ref,
    q_hbm_ref,
    kv_hbm_ref,
    attention_sink_ref,
    o_hbm_ref,
    bkv_x2_ref,
    bq_x2_ref,
    bo_x2_ref,
    sems,
    l_ref,
    m_ref,
    acc_ref,
    *,
    window_size: tuple[int, int],
    sm_scale: float,
    soft_cap: float | None,
    mask_value: float,
    k_scale: float | None,
    v_scale: float | None,
    bq_sz: int,
    bkv_sz: int,
):
    actual_num_kv_heads, _, q_per_kv_per_packing, q_packing, head_dim = q_hbm_ref.shape
    _, _, kv_packing, _ = kv_hbm_ref.shape
    q_per_kv = q_per_kv_per_packing * q_packing
    bkv_stride = bkv_x2_ref.shape[2]
    q_dtype = q_hbm_ref.dtype
    kv_dtype = kv_hbm_ref.dtype
    left_window, right_window = window_size
    combined_kv_heads_per_packing = kv_hbm_ref.shape[1]

    q_start = cu_lens_ref[seq_idx]
    q_end = cu_lens_ref[seq_idx + 1]
    seq_len = q_end - q_start
    num_bq = pl.cdiv(seq_len, bq_sz)

    def _fetch_bq(target_seq_idx, bq_idx, sem_idx, *, wait=False):
        start = cu_lens_ref[target_seq_idx] + bq_idx * bq_sz
        _make_async_copy(
            q_hbm_ref.at[:, pl.ds(start, bq_sz)],
            bq_x2_ref.at[sem_idx],
            sems.at[1, sem_idx],
            wait,
        )

    def _fetch_bkv(target_seq_idx, bkv_idx, sem_idx, *, wait=False):
        start = cu_lens_ref[target_seq_idx] + bkv_idx * bkv_sz
        _make_async_copy(
            kv_hbm_ref.at[pl.ds(start, bkv_sz)],
            bkv_x2_ref.at[sem_idx, :, :combined_kv_heads_per_packing],
            sems.at[0, sem_idx],
            wait,
        )

    def _send_bo(target_seq_idx, bq_idx, sem_idx, *, wait=False):
        start = cu_lens_ref[target_seq_idx] + bq_idx * bq_sz
        _make_async_copy(
            bo_x2_ref.at[sem_idx],
            o_hbm_ref.at[:, pl.ds(start, bq_sz)],
            sems.at[2, sem_idx],
            wait,
        )

    def start_fetch_bq(target_seq_idx, bq_idx, sem_idx):
        _fetch_bq(target_seq_idx, bq_idx, sem_idx)

    def wait_fetch_bq(target_seq_idx, bq_idx, sem_idx):
        _fetch_bq(target_seq_idx, bq_idx, sem_idx, wait=True)

    def start_fetch_bkv(target_seq_idx, bkv_idx, sem_idx):
        _fetch_bkv(target_seq_idx, bkv_idx, sem_idx)

    def wait_fetch_bkv(target_seq_idx, bkv_idx, sem_idx):
        _fetch_bkv(target_seq_idx, bkv_idx, sem_idx, wait=True)

    def start_send_bo(target_seq_idx, bq_idx, sem_idx):
        bo_ids_ref[sem_idx] = target_seq_idx
        bo_ids_ref[sem_idx + 2] = bq_idx
        _send_bo(target_seq_idx, bq_idx, sem_idx)

    def wait_send_bo(sem_idx):
        old_seq_idx = bo_ids_ref[sem_idx]
        old_bq_idx = bo_ids_ref[sem_idx + 2]

        @pl.when(jnp.logical_and(old_seq_idx >= 0, old_seq_idx <= seq_idx))
        def wait_old_output():
            _send_bo(old_seq_idx, old_bq_idx, sem_idx, wait=True)
            bo_ids_ref[sem_idx] = -1

    def load_bq(sem_idx, kv_head_idx):
        packed_ref = (
            bq_x2_ref.bitcast(jnp.uint32)
            .at[sem_idx, kv_head_idx]
            .reshape(bq_sz * q_per_kv_per_packing, head_dim)
        )
        return _strided_load(packed_ref, 0, bq_sz * q_per_kv_per_packing, 1, dtype=q_dtype)

    def load_bkv(sem_idx, kv_head_idx):
        packed_ref = (
            bkv_x2_ref.bitcast(jnp.uint32).at[sem_idx].reshape(bkv_sz * bkv_stride, head_dim)
        )
        if kv_packing == 1:
            start = kv_head_idx * 2
            k_block = _strided_load(
                packed_ref, start, bkv_sz * bkv_stride, bkv_stride, dtype=kv_dtype
            )
            v_block = _strided_load(
                packed_ref, start + 1, bkv_sz * bkv_stride, bkv_stride, dtype=kv_dtype
            )
            return k_block, v_block

        kv_heads_per_word = kv_packing // 2
        word_offset = kv_head_idx // kv_heads_per_word
        kv_idx_in_word = kv_head_idx % kv_heads_per_word
        packed = _strided_load(packed_ref, word_offset, bkv_sz * bkv_stride, bkv_stride)
        bitwidth = 32 // kv_packing
        repack_ty = jnp.dtype(f"uint{bitwidth}")
        k_bits = packed >> (kv_idx_in_word * 2 * bitwidth)
        v_bits = k_bits >> bitwidth
        return (
            pltpu.bitcast(k_bits.astype(repack_ty), kv_dtype),
            pltpu.bitcast(v_bits.astype(repack_ty), kv_dtype),
        )

    def flash_attention_step(
        q_block,
        k_block,
        v_block,
        l_head_ref,
        m_head_ref,
        acc_head_ref,
        *,
        processed_q,
        processed_kv,
        physical_kv_len,
    ):
        logits = jnp.matmul(q_block, k_block.T, preferred_element_type=jnp.float32)
        logits *= sm_scale
        if k_scale is not None:
            logits *= k_scale
        if soft_cap is not None:
            logits = soft_cap * jnp.tanh(logits / soft_cap)

        q_span = processed_q + (lax.broadcasted_iota(jnp.int32, logits.shape, 0) // q_per_kv)
        k_span = processed_kv + lax.broadcasted_iota(jnp.int32, logits.shape, 1)
        v_span = processed_kv + lax.broadcasted_iota(jnp.int32, v_block.shape, 0)
        valid_kv = k_span < physical_kv_len
        keep = jnp.ones_like(valid_kv)
        if left_window >= 0:
            keep = jnp.logical_and(keep, k_span >= q_span - left_window)
        if right_window >= 0:
            keep = jnp.logical_and(keep, k_span <= q_span + right_window)
        logits = jnp.where(keep, logits, mask_value)
        logits = jnp.where(valid_kv, logits, -jnp.inf)
        v_block = jnp.where(v_span < physical_kv_len, v_block, 0.0)

        rowmax = jnp.max(logits, axis=1, keepdims=True)
        m_prev = m_head_ref[...]
        m_curr = jnp.maximum(m_prev, rowmax)
        m_head_ref[...] = m_curr
        p = jnp.exp(logits - _broadcast_minor(m_curr, logits.shape))
        rowsum = jnp.sum(p, axis=1, keepdims=True)
        exp_diff = jnp.exp(m_prev - m_curr)
        l_prev = l_head_ref[...]
        l_head_ref[...] = exp_diff * l_prev + rowsum

        pv = jnp.matmul(p, v_block, preferred_element_type=jnp.float32)
        if v_scale is not None:
            pv *= v_scale
        acc_prev = acc_head_ref[...]
        acc_head_ref[...] = _broadcast_minor(exp_diff, acc_prev.shape) * acc_prev + pv

    def next_q_ids(bq_idx, sem_idx):
        next_bq_idx = bq_idx + 1
        is_last = next_bq_idx == num_bq
        next_bq_idx = lax.select(is_last, 0, next_bq_idx)
        next_seq_idx = lax.select(is_last, seq_idx + 1, seq_idx)
        return next_seq_idx, next_bq_idx, 1 - sem_idx

    @pl.when(seq_idx == 0)
    def prologue():
        start_fetch_bq(0, 0, 0)
        start_fetch_bkv(0, 0, 0)

    previous_seq_idx = jnp.maximum(seq_idx - 1, 0)
    previous_seq_len = q_start - cu_lens_ref[previous_seq_idx]

    @pl.when(jnp.logical_and(seq_idx > 0, jnp.mod(previous_seq_len, bq_sz) != 0))
    def drain_previous_sequence_outputs():
        wait_send_bo(0)
        wait_send_bo(1)

    @pl.loop(0, num_bq, unroll=False)
    def loop_q_block(bq_idx):
        acc_ref[...] = jnp.zeros_like(acc_ref)
        if attention_sink_ref is not None:
            # Virtual zero-value token: initialize online softmax with one
            # existing logit per Q head, but no contribution to the numerator.
            l_ref[...] = jnp.ones_like(l_ref)
            m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
            for kv_head_idx in range(actual_num_kv_heads):
                sinks = attention_sink_ref[kv_head_idx]
                m_ref.at[kv_head_idx, pl.ds(0, bq_sz * q_per_kv)][...] = jnp.tile(sinks, (bq_sz, 1))
        else:
            l_ref[...] = jnp.zeros_like(l_ref)
            m_ref[...] = jnp.full_like(m_ref, -jnp.inf)

        q_sem_idx = sem_ids_ref[0]
        next_seq_idx, next_bq_idx, next_q_sem_idx = next_q_ids(bq_idx, q_sem_idx)

        @pl.when(next_seq_idx < num_seqs)
        def prefetch_next_q():
            sem_ids_ref[0] = next_q_sem_idx
            start_fetch_bq(next_seq_idx, next_bq_idx, next_q_sem_idx)

        processed_q = bq_idx * bq_sz
        prune_window_blocks = mask_value == DEFAULT_MASK_VALUE or mask_value == float("-inf")
        start_bkv_idx = 0
        if left_window >= 0 and prune_window_blocks:
            start_bkv_idx = jnp.maximum(processed_q - left_window, 0) // bkv_sz
        iteration_kv_end = seq_len
        if right_window >= 0 and prune_window_blocks:
            iteration_kv_end = jnp.minimum(seq_len, processed_q + bq_sz + right_window)
        end_bkv_idx = pl.cdiv(iteration_kv_end, bkv_sz)

        @pl.loop(start_bkv_idx, end_bkv_idx, unroll=False)
        def loop_kv_block(bkv_idx):
            kv_sem_idx = sem_ids_ref[1]
            next_bkv_idx = bkv_idx + 1
            is_last_bkv = next_bkv_idx == end_bkv_idx
            next_q_for_kv = lax.select(is_last_bkv, bq_idx + 1, bq_idx)
            is_last_q = next_q_for_kv == num_bq
            next_q_for_kv = lax.select(is_last_q, 0, next_q_for_kv)
            next_seq_for_kv = lax.select(is_last_q, seq_idx + 1, seq_idx)
            next_bkv_idx = lax.select(is_last_bkv, 0, next_bkv_idx)
            if left_window >= 0 and prune_window_blocks:
                next_q_start_bkv = jnp.maximum(next_q_for_kv * bq_sz - left_window, 0) // bkv_sz
                next_bkv_idx = lax.select(is_last_bkv, next_q_start_bkv, next_bkv_idx)
            next_kv_sem_idx = 1 - kv_sem_idx

            @pl.when(next_seq_for_kv < num_seqs)
            def prefetch_next_kv():
                sem_ids_ref[1] = next_kv_sem_idx
                start_fetch_bkv(next_seq_for_kv, next_bkv_idx, next_kv_sem_idx)

            @pl.when(bkv_idx == start_bkv_idx)
            def wait_q():
                wait_fetch_bq(seq_idx, bq_idx, q_sem_idx)

            wait_fetch_bkv(seq_idx, bkv_idx, kv_sem_idx)
            processed_kv = bkv_idx * bkv_sz

            for kv_head_idx in range(actual_num_kv_heads):
                q_block = load_bq(q_sem_idx, kv_head_idx)
                k_block, v_block = load_bkv(kv_sem_idx, kv_head_idx)
                lm_slice = (kv_head_idx, pl.ds(0, bq_sz * q_per_kv))
                flash_attention_step(
                    q_block,
                    k_block,
                    v_block,
                    l_ref.at[*lm_slice],
                    m_ref.at[*lm_slice],
                    acc_ref.at[*lm_slice],
                    processed_q=processed_q,
                    processed_kv=processed_kv,
                    physical_kv_len=seq_len,
                )

        acc = acc_ref[...]
        denominator = _broadcast_minor(l_ref[...], acc.shape)
        out = lax.div(acc, jnp.where(denominator == 0.0, 1.0, denominator)).astype(q_dtype)

        bo_sem_idx = sem_ids_ref[2]
        sem_ids_ref[2] = 1 - bo_sem_idx
        wait_send_bo(bo_sem_idx)
        bo_x2_ref.at[bo_sem_idx][...] = out.reshape(
            actual_num_kv_heads,
            bq_sz,
            q_per_kv_per_packing,
            q_packing,
            head_dim,
        )
        start_send_bo(seq_idx, bq_idx, bo_sem_idx)

    @pl.when(seq_idx == num_seqs - 1)
    def epilogue():
        wait_send_bo(0)
        wait_send_bo(1)


def _packed_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    cu_seqlens: jax.Array,
    num_seqs: jax.Array,
    *,
    window_size: tuple[int, int],
    sm_scale: float,
    soft_cap: float | None,
    mask_value: float,
    k_scale: float | None,
    v_scale: float | None,
    attention_sink: jax.Array | float | None,
    num_queries_per_block: int,
    num_kv_per_block: int,
    vmem_limit_bytes: int | None,
) -> jax.Array:
    """GQA/F32 fallback: word-packed Q/KV kernel with an RPA3-style ``grid=(1,)``."""
    (
        q_internal,
        kv_internal,
        attention_sink_internal,
        actual_q_per_kv,
        actual_head_dim,
        original_q_capacity,
    ) = _prepare_packed_layout(
        q,
        k,
        v,
        attention_sink,
        bq_sz=num_queries_per_block,
        bkv_sz=num_kv_per_block,
    )
    actual_num_kv_heads, _, q_per_kv_per_packing, q_packing, head_dim = q_internal.shape
    combined_kv_heads_per_packing = kv_internal.shape[1]
    kv_packing = kv_internal.shape[2]
    q_per_kv = q_per_kv_per_packing * q_packing

    bkv_stride = combined_kv_heads_per_packing
    if _has_bank_conflicts(bkv_stride):
        bkv_stride += 1

    bkv_double_buffer = pltpu.VMEM(
        (2, num_kv_per_block, bkv_stride, kv_packing, head_dim),
        kv_internal.dtype,
    )
    bq_double_buffer = pltpu.VMEM(
        (2, actual_num_kv_heads, num_queries_per_block, q_per_kv_per_packing, q_packing, head_dim),
        q_internal.dtype,
    )
    bo_double_buffer = pltpu.VMEM(
        (2, actual_num_kv_heads, num_queries_per_block, q_per_kv_per_packing, q_packing, head_dim),
        q_internal.dtype,
    )
    lm_scratch = pltpu.VMEM(
        (actual_num_kv_heads, num_queries_per_block * q_per_kv, 128),
        jnp.float32,
    )
    acc_scratch = pltpu.VMEM(
        (actual_num_kv_heads, num_queries_per_block * q_per_kv, head_dim),
        jnp.float32,
    )

    scalar_prefetches = (
        cu_seqlens,
        num_seqs,
        jnp.zeros((3,), dtype=jnp.int32),
        jnp.full((4,), -1, dtype=jnp.int32),
    )
    in_specs = (
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec(memory_space=pltpu.HBM),
        (pl.BlockSpec(memory_space=pltpu.VMEM) if attention_sink_internal is not None else None),
    )
    kernel = pl.pallas_call(
        functools.partial(
            _packed_kernel,
            window_size=window_size,
            sm_scale=sm_scale,
            soft_cap=soft_cap,
            mask_value=mask_value,
            k_scale=k_scale,
            v_scale=v_scale,
            bq_sz=num_queries_per_block,
            bkv_sz=num_kv_per_block,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_prefetches),
            in_specs=in_specs,
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            grid=(1,),
            scratch_shapes=(
                bkv_double_buffer,
                bq_double_buffer,
                bo_double_buffer,
                pltpu.SemaphoreType.DMA((3, 2)),
                lm_scratch,
                lm_scratch,
                acc_scratch,
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        out_shape=jax.ShapeDtypeStruct(q_internal.shape, q_internal.dtype),
        name="varlen_attention_packed",
    )
    out_internal = kernel(*scalar_prefetches, q_internal, kv_internal, attention_sink_internal)
    restored = _restore_packed_output(out_internal, actual_q_per_kv, actual_head_dim)
    return restored[:original_q_capacity]


# =========================================================================== #
# Public entry point
# =========================================================================== #
@functools.partial(
    jax.jit,
    static_argnames=(
        "window_size",
        "sm_scale",
        "soft_cap",
        "mask_value",
        "k_scale",
        "v_scale",
        "max_seq_len",
        "num_queries_per_block",
        "num_kv_per_block",
        "vmem_limit_bytes",
    ),
)
def varlen_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    cu_seqlens: jax.Array,
    num_seqs: jax.Array,
    *,
    window_size: tuple[int, int] = (-1, -1),
    sm_scale: float = 1.0,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
    attention_sink: jax.Array | float | None = None,
    max_seq_len: int | None = None,
    num_queries_per_block: int | None = None,
    num_kv_per_block: int | None = None,
    vmem_limit_bytes: int | None = DEFAULT_VMEM_LIMIT_BYTES,
) -> jax.Array:
    """Variable-length packed full-prefill attention on TPU.

    Public layout::

      q:            [q_capacity,  num_q_heads,  head_dim]
      k, v:         [kv_capacity, num_kv_heads, head_dim]
      cu_seqlens:   int32[max_num_seqs + 1]
      num_seqs:     int32[1]

    For every valid sequence ``i``, Q, K and V share the packed interval
    ``[cu_seqlens[i], cu_seqlens[i + 1])``; there is no historical prefix.

    ``window_size=(left, right)`` uses FlashAttention semantics: ``-1`` means
    unbounded, so ``(-1, -1)`` is full attention and ``(-1, 0)`` is causal.
    ``attention_sink`` may be a scalar or ``float32[num_q_heads]``; it acts as
    one virtual zero-value token per Q head.

    ``max_seq_len`` is a static upper bound on
    ``max(diff(cu_seqlens[:num_seqs + 1]))`` used only to select tuned block
    sizes. Callers should compute it on the host. If omitted, the packed Q
    capacity is used as a conservative fallback.

    BF16 MHA (``num_q_heads == num_kv_heads``) takes a token-major fast path;
    GQA or F32 inputs fall back to the word-packed kernel. The wrapper reserves
    its fixed-DMA tail slack internally, so exact-capacity inputs are safe. If Q
    has extra capacity beyond the last valid token, that output suffix is
    intentionally unspecified.
    """
    static_validate_inputs(
        q,
        k,
        v,
        cu_seqlens,
        num_seqs,
        window_size=window_size,
        sm_scale=sm_scale,
        soft_cap=soft_cap,
        mask_value=mask_value,
        k_scale=k_scale,
        v_scale=v_scale,
        attention_sink=attention_sink,
        max_seq_len=max_seq_len,
        num_queries_per_block=num_queries_per_block,
        num_kv_per_block=num_kv_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE

    is_bf16_mha = q.shape[1] == k.shape[1] and q.dtype == jnp.bfloat16 and k.dtype == jnp.bfloat16
    if num_queries_per_block is None or num_kv_per_block is None:
        if is_bf16_mha:
            tuned_bq, tuned_bkv = get_tuned_block_sizes(
                q.shape[1],
                q.shape[2],
                q.shape[0] if max_seq_len is None else max_seq_len,
            )
        else:
            # The GQA/F32 packed kernel has a different VMEM layout and needs
            # an independently tuned table.
            tuned_bq, tuned_bkv = DEFAULT_Q_BLOCK, DEFAULT_KV_BLOCK
        if num_queries_per_block is None:
            num_queries_per_block = tuned_bq
        if num_kv_per_block is None:
            num_kv_per_block = tuned_bkv

    if not is_bf16_mha:
        return _packed_attention(
            q,
            k,
            v,
            cu_seqlens,
            num_seqs,
            window_size=window_size,
            sm_scale=sm_scale,
            soft_cap=soft_cap,
            mask_value=mask_value,
            k_scale=k_scale,
            v_scale=v_scale,
            attention_sink=attention_sink,
            num_queries_per_block=num_queries_per_block,
            num_kv_per_block=num_kv_per_block,
            vmem_limit_bytes=vmem_limit_bytes,
        )

    (
        q_internal,
        kv_internal,
        attention_sink_internal,
        actual_head_dim,
        original_q_capacity,
    ) = _prepare_mha_layout(
        q,
        k,
        v,
        attention_sink,
        bq_sz=num_queries_per_block,
    )
    out_internal = _mha_attention(
        q_internal,
        kv_internal,
        cu_seqlens,
        num_seqs,
        attention_sink_internal,
        window_size=window_size,
        sm_scale=sm_scale,
        soft_cap=soft_cap,
        mask_value=mask_value,
        k_scale=k_scale,
        v_scale=v_scale,
        num_queries_per_block=num_queries_per_block,
        num_kv_per_block=num_kv_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    return _restore_mha_output(out_internal, actual_head_dim, original_q_capacity)


__all__ = [
    "DEFAULT_KV_BLOCK",
    "DEFAULT_MASK_VALUE",
    "DEFAULT_Q_BLOCK",
    "DEFAULT_VMEM_LIMIT_BYTES",
    "dynamic_validate_metadata",
    "ref_varlen_attention",
    "static_validate_inputs",
    "varlen_attention",
]
