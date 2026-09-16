"""Rank-local HCA execution for SGLang-JAX."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.kernels.hca.attention import (
    ragged_attention,
    uniform_prefill_attention,
)
from sgl_jax.srt.kernels.hca.compressor import (
    hca_project_fused_pallas,
    hca_state_pool_emit_pallas,
    hca_state_pool_update_fused_pallas,
    hca_state_pool_update_ragged_fused_pallas,
    token_compress_prefill_pallas,
)
from sgl_jax.srt.kernels.hca.tuned_block_sizes import HCAKernelSchedule


def fused_projection_weight(wkv, wgate, fused_weight=None):
    """Default the fused ``[hidden, 2*head_dim]`` projection to ``[Wkv|Wgate]^T``."""
    if fused_weight is not None:
        return fused_weight
    return jnp.concatenate((wkv.astype(jnp.bfloat16), wgate.astype(jnp.bfloat16)), axis=0).T


@register_pytree_node_class
@dataclass(frozen=True)
class HCAMetadata:
    """One DP rank's unified decode/EXTEND HCA metadata.

    Every array length and the single static field are padded or bucketed by the
    backend to capacities derived only from the padded batch/token shape, so
    per-step metadata drift (boundary counts, page-table growth, ragged query
    mixes) never changes the compiled program shape.  Padded entries are inert:
    boundary indices use the out-of-range sentinel ``tokens``, query blocks use
    ``attention.INERT_QUERY_OFFSET``, decode ids use ``-1``, and page tables use
    the dummy physical page zero.
    """

    state_slots: jax.Array
    query_seq_ids: jax.Array
    cu_q_lens: jax.Array
    valid_token_mask: jax.Array
    boundary_token_indices: jax.Array
    window_page_indices: jax.Array
    window_cu_kv_lens: jax.Array
    seq_lens: jax.Array
    compressed_page_indices: jax.Array
    compressed_cu_kv_lens: jax.Array
    compressed_kv_lens: jax.Array
    query_block_request_ids: jax.Array
    query_block_offsets: jax.Array
    decode_request_ids: jax.Array
    max_queries_per_request: int

    def tree_flatten(self):
        return (
            (
                self.state_slots,
                self.query_seq_ids,
                self.cu_q_lens,
                self.valid_token_mask,
                self.boundary_token_indices,
                self.window_page_indices,
                self.window_cu_kv_lens,
                self.seq_lens,
                self.compressed_page_indices,
                self.compressed_cu_kv_lens,
                self.compressed_kv_lens,
                self.query_block_request_ids,
                self.query_block_offsets,
                self.decode_request_ids,
            ),
            (self.max_queries_per_request,),
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        (max_queries_per_request,) = aux
        return cls(*children, max_queries_per_request=max_queries_per_request)


def hca_step(
    x,
    q,
    new_kv,
    state_pool,
    window_cache,
    compressed_cache,
    wkv,
    wgate,
    ape,
    norm_weight,
    cos,
    sin,
    positions,
    attention_sink,
    metadata: HCAMetadata,
    *,
    mode: str,
    schedule: HCAKernelSchedule,
    softmax_scale: float,
    compress_ratio: int = 128,
    head_dim: int = 512,
    window_size: int = 128,
    norm_eps: float = 1e-6,
    fused_weight=None,
):
    """Run one complete stateful HCA step for ``mode``.

    Only compression is mode-specific: decode advances one token per request,
    uniform prefill compresses dense zero-prefix groups without reading state,
    and ragged rebuilds each completed group across a chunk boundary. Decode
    shares the ragged attention path, which degenerates correctly at
    ``q_len=1``; uniform prefill keeps its own faster entry point.
    """
    shared = {
        "compress_ratio": compress_ratio,
        "head_dim": head_dim,
        "norm_eps": norm_eps,
        "output_dtype": new_kv.dtype,
        "fused_weight": fused_weight,
        "schedule": schedule,
    }
    if mode == "decode":
        emitted, emit_mask, state_pool = _hca_compress_decode(
            x,
            state_pool,
            wkv,
            wgate,
            ape,
            norm_weight,
            positions,
            metadata.state_slots,
            cos,
            sin,
            valid_mask=metadata.valid_token_mask,
            has_boundary=bool(metadata.boundary_token_indices.shape[0]),
            **shared,
        )
    elif mode == "uniform":
        emitted, emit_mask, state_pool = _hca_compress_uniform_prefill(
            x, state_pool, wkv, wgate, ape, norm_weight, cos, sin, metadata, **shared
        )
    elif mode == "ragged":
        emitted, emit_mask, state_pool = _hca_compress_ragged(
            x,
            state_pool,
            wkv,
            wgate,
            ape,
            norm_weight,
            cos,
            sin,
            positions,
            metadata,
            **shared,
        )
    else:
        raise ValueError(f"unknown HCA step mode: {mode}")

    attention_options = {
        "schedule": schedule,
        "softmax_scale": softmax_scale,
        "window_size": window_size,
        "compress_ratio": compress_ratio,
    }
    if mode == "uniform":
        # Zero prefix and equal q_len let this specialization skip the history
        # gather, the padded-KV scatter, and the O(T) compressed-record write.
        # Measured 11.9% faster than the general path on the prefill grid, so
        # it stays separate on purpose.
        output, window_cache, compressed_cache = uniform_prefill_attention(
            q,
            new_kv,
            window_cache,
            compressed_cache,
            emitted,
            attention_sink,
            metadata,
            **attention_options,
        )
    else:
        output, window_cache, compressed_cache = ragged_attention(
            q,
            new_kv,
            window_cache,
            compressed_cache,
            positions,
            emitted,
            emit_mask,
            attention_sink,
            metadata,
            **attention_options,
        )
    return output, state_pool, window_cache, compressed_cache


def _hca_compress_uniform_prefill(
    x,
    state_pool,
    wkv,
    wgate,
    ape,
    norm_weight,
    cos,
    sin,
    metadata: HCAMetadata,
    *,
    schedule: HCAKernelSchedule,
    compress_ratio: int = 128,
    head_dim: int = 512,
    norm_eps: float = 1e-6,
    output_dtype=jnp.bfloat16,
    fused_weight=None,
):
    """Compress dense 128-token groups and preserve each request's tail state."""
    batch = metadata.seq_lens.shape[0]
    tokens = x.shape[0]
    if tokens % batch:
        raise ValueError("uniform-prefill EXTEND requires T divisible by B")
    sequence = tokens // batch
    x_by_request = x.reshape(batch, sequence, x.shape[-1])
    cutoff = sequence // compress_ratio * compress_ratio
    complete_entries = cutoff // compress_ratio
    emitted = jnp.zeros((tokens, head_dim), output_dtype)
    emit_mask = jnp.zeros((tokens,), jnp.bool_)
    if complete_entries:
        entries = token_compress_prefill_pallas(
            x_by_request[:, :cutoff],
            wkv,
            wgate,
            ape,
            norm_weight,
            cos[:cutoff:compress_ratio],
            sin[:cutoff:compress_ratio],
            schedule=schedule,
            compress_ratio=compress_ratio,
            head_dim=head_dim,
            norm_eps=norm_eps,
            out_dtype=output_dtype,
        )
        boundary_tokens = (
            jnp.arange(batch, dtype=jnp.int32)[:, None] * sequence
            + (jnp.arange(complete_entries, dtype=jnp.int32)[None, :] + 1) * compress_ratio
            - 1
        ).reshape(-1)
        emitted = emitted.at[boundary_tokens].set(
            entries.reshape(-1, head_dim).astype(output_dtype)
        )
        emit_mask = emit_mask.at[boundary_tokens].set(True)

    # A remainder-free prompt leaves the rows untouched: contiguous positions
    # overwrite the next group before emission reads it, so no stale state leaks.
    remainder = sequence - cutoff
    if remainder:
        fused_weight = fused_projection_weight(wkv, wgate, fused_weight)
        tail_positions = jnp.broadcast_to(
            jnp.arange(cutoff, sequence, dtype=jnp.int32)[None, :],
            (batch, remainder),
        )
        projected = hca_project_fused_pallas(
            x_by_request[:, cutoff:].reshape(batch * remainder, x_by_request.shape[-1]),
            fused_weight,
            ape,
            tail_positions.reshape(batch * remainder),
            schedule=schedule,
            compress_ratio=compress_ratio,
            head_dim=head_dim,
        ).reshape(batch, remainder, 2, head_dim)
        pad_rows = ((0, 0), (0, compress_ratio - remainder), (0, 0))
        tail = jnp.stack(
            (
                jnp.pad(projected[:, :, 0], pad_rows),
                jnp.pad(projected[:, :, 1], pad_rows, constant_values=-jnp.inf),
            ),
            axis=2,
        )
        per_request_slots = metadata.state_slots.reshape(batch, sequence)[:, 0]
        update = state_pool.at[per_request_slots]
        update_kwargs = {
            "mode": "promise_in_bounds",
            "unique_indices": True,
        }
        abstract_mesh = jax.sharding.get_abstract_mesh()
        if "data" in abstract_mesh.axis_names:
            data_axis = abstract_mesh.axis_names.index("data")
            if abstract_mesh.axis_types[data_axis] is jax.sharding.AxisType.Explicit:
                update_kwargs["out_sharding"] = P("data", None, None, None)
        state_pool = update.set(tail, **update_kwargs)

    return emitted, emit_mask, state_pool


def _hca_compress_decode(
    x,
    state_pool,
    wkv,
    wgate,
    ape,
    norm_weight,
    positions,
    state_slots,
    cos,
    sin,
    *,
    schedule: HCAKernelSchedule,
    valid_mask=None,
    compress_ratio: int = 128,
    head_dim: int = 512,
    norm_eps: float = 1e-6,
    output_dtype=jnp.bfloat16,
    fused_weight=None,
    has_boundary: bool = True,
):
    """Project decode tokens, update physical state rows, and emit boundaries."""
    fused_weight = fused_projection_weight(wkv, wgate, fused_weight)
    updated_state_pool = hca_state_pool_update_fused_pallas(
        x,
        state_pool,
        fused_weight,
        ape,
        positions,
        state_slots,
        schedule=schedule,
        compress_ratio=compress_ratio,
        head_dim=head_dim,
        valid_mask=valid_mask,
    )
    emit_mask = jnp.mod(positions + 1, compress_ratio) == 0
    if valid_mask is not None:
        # A padded token whose garbage position lands on a boundary would emit
        # from the dummy state row (all ``-inf`` scores -> NaN); mask it out.
        emit_mask = emit_mask & valid_mask
    if has_boundary:
        group_starts = jnp.maximum(positions + 1 - compress_ratio, 0)
        selected_cos = cos.at[group_starts].get(mode="fill", fill_value=0.0)
        selected_sin = sin.at[group_starts].get(mode="fill", fill_value=0.0)
        emitted = hca_state_pool_emit_pallas(
            updated_state_pool,
            state_slots,
            emit_mask,
            norm_weight,
            selected_cos,
            selected_sin,
            schedule=schedule,
            head_dim=head_dim,
            norm_eps=norm_eps,
            output_dtype=output_dtype,
        )
    else:
        emitted = jnp.zeros((x.shape[0], head_dim), output_dtype)
    return emitted, emit_mask, updated_state_pool


def _hca_compress_ragged(
    x,
    state_pool,
    wkv,
    wgate,
    ape,
    norm_weight,
    cos,
    sin,
    positions,
    metadata: HCAMetadata,
    *,
    schedule: HCAKernelSchedule,
    compress_ratio: int = 128,
    head_dim: int = 512,
    norm_eps: float = 1e-6,
    output_dtype=jnp.bfloat16,
    fused_weight=None,
):
    """Update ragged recurrent state and emit completed 128-token groups."""
    fused_weight = fused_projection_weight(wkv, wgate, fused_weight)
    # Padded tokens are projected but never selected: per-request gathers stay
    # inside [query_start, query_start + q_len), so shapes stay stable.
    return hca_state_pool_update_ragged_fused_pallas(
        x,
        state_pool,
        fused_weight,
        ape,
        norm_weight,
        cos,
        sin,
        positions,
        metadata.state_slots,
        metadata.cu_q_lens[:-1],
        metadata.seq_lens - jnp.diff(metadata.cu_q_lens),
        metadata.seq_lens,
        metadata.boundary_token_indices,
        schedule=schedule,
        compress_ratio=compress_ratio,
        head_dim=head_dim,
        norm_eps=norm_eps,
        output_dtype=output_dtype,
    )


__all__ = ["HCAMetadata", "hca_step"]
