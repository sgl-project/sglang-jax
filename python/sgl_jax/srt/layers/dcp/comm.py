"""DCP collectives for partial attention merge.

Runs inside a ``shard_map`` that names the ``tensor`` axis (group size =
TP = DCP).

When DCP == TP, Q is head-sharded. LSE-merge is only valid for the **same**
heads over different KV shards, so Q is all-gathered before attend and every
rank computes all heads.

Two merge paths exist:

``merge_scatter_dcp_attention`` (default) is the ``ag_rs`` pattern: all-reduce
the LSE, then ``psum_scatter`` the weighted outputs over the head axis so each
rank receives only its own heads, already summed.

``gather_merge_dcp_attention`` is the original all-gather then slice. It
materializes ``dcp_size`` copies of the all-head output and discards all but
``1/dcp_size`` of the merged result; at 8k/DCP=16 that one collective measured
71.9% of device self time. Kept as a reference for the equivalence test.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from sgl_jax.srt.layers.dcp.merge import merge_dcp_attention_jax


def allgather_heads(ql, qpe, axis_name: str = "tensor"):
    """Replicate head-TP shards: ``[T, H_local, D]`` → ``[T, H_all, D]``.

    Gathers on the head axis directly. Gathering on axis 0 instead yields
    ``[dcp_size, T, H_local, D]`` and then needs a transpose of the whole
    gathered array — half a gigabyte at 8k/DCP=16 — just to interleave the
    shards. ``tiled=True`` puts rank ``r``'s heads at ``[r * H_local,
    (r + 1) * H_local)``, the same order :func:`slice_local_heads` and the
    ``psum_scatter`` in :func:`merge_scatter_dcp_attention` assume.
    """
    if ql.dtype != qpe.dtype:
        ql_f = jax.lax.all_gather(ql, axis_name, axis=1, tiled=True)
        qpe_f = jax.lax.all_gather(qpe, axis_name, axis=1, tiled=True)
        return ql_f, qpe_f
    d = ql.shape[-1]
    packed = jax.lax.all_gather(jnp.concatenate([ql, qpe], axis=-1), axis_name, axis=1, tiled=True)
    return packed[..., :d], packed[..., d:]


def slice_local_heads(o, h_local: int, axis_name: str = "tensor"):
    """Undo :func:`allgather_heads`: take this rank's head slab."""
    rank = jax.lax.axis_index(axis_name)
    return jax.lax.dynamic_slice_in_dim(o, rank * h_local, h_local, axis=1)


def gather_merge_dcp_attention(
    local_o,
    local_lse,
    axis_name: str = "tensor",
):
    """All-gather ``(o, lse)`` on the DCP axis and LSE-merge.

    ``local_o`` is ``[*batch, dim]`` (this rank's locally-normalized output).
    ``local_lse`` is ``[*batch]`` natural-log LSE from the same softmax
    (``m + log(l)``). Empty shards pass ``lse = -inf`` and ``o = 0``.

    Heads on ``local_o`` must be the same set on every rank (see
    :func:`allgather_heads`).
    """
    partial_o = jax.lax.all_gather(local_o, axis_name, axis=0)
    partial_lse = jax.lax.all_gather(local_lse, axis_name, axis=0)
    merged_o, _merged_lse = merge_dcp_attention_jax(partial_o, partial_lse)
    return merged_o


def merge_scatter_dcp_attention(
    local_o,
    local_lse,
    h_local: int,
    head_axis: int = 1,
    axis_name: str = "tensor",
    scatter_dtype=None,
):
    """LSE-merge and return only this rank's head slab, without a full gather.

    Equivalent to ``slice_local_heads(gather_merge_dcp_attention(o, lse),
    h_local)`` but the ``dcp_size``-copy buffer is never built: only the LSE
    (one scalar per query-head) is replicated, and the outputs are combined by
    ``psum_scatter`` over ``head_axis``.

    ``local_o`` is ``[T, H_all, dim]``, ``local_lse`` is ``[T, H_all]``, and
    ``h_local`` is this rank's pre-:func:`allgather_heads` head count. Empty
    shards pass ``lse = -inf`` and drop out with weight 0.

    ``scatter_dtype`` narrows only the wire dtype of the ``psum_scatter``; the
    weighting and the final divide stay f32. bf16 halves the bytes of the single
    largest remaining collective and is the default (see
    ``DSA_DCP_MERGE_BF16``): measured at 9% of prefill wall time with tokens
    bit-identical to ``dcp=1``. The tail it risks — the reduction accumulates
    ``dcp_size`` partials weighted by ``exp(lse_r - max_lse) <= 1``, so a shard
    contributing little is the one whose contribution rounds away — is real but
    smaller than the bf16 the rest of the model already carries. Pass ``None``
    for f32.
    """
    h_all = local_o.shape[head_axis]
    if h_local <= 0 or h_all % h_local:
        raise ValueError(f"h_local {h_local} must divide all-head count {h_all}")

    lse = local_lse.astype(jnp.float32)
    finite = jnp.isfinite(lse)
    # Shift by the group max so the weights are in (0, 1]; same stabilization
    # merge_dcp_attention_jax does, but the max comes from a collective rather
    # than a reduction over a gathered axis.
    max_lse = jax.lax.pmax(jnp.where(finite, lse, jnp.float32(-jnp.inf)), axis_name)
    all_empty = ~jnp.isfinite(max_lse)
    max_safe = jnp.where(all_empty, jnp.float32(0.0), max_lse)
    weights = jnp.where(finite, jnp.exp(lse - max_safe), jnp.float32(0.0))
    weight_sum = jax.lax.psum(weights, axis_name)

    num = local_o.astype(jnp.float32) * weights[..., None]
    if scatter_dtype is not None:
        num = num.astype(scatter_dtype)
    num = jax.lax.psum_scatter(num, axis_name, scatter_dimension=head_axis, tiled=True)
    num = num.astype(jnp.float32)

    # psum_scatter hands shard ``i`` to rank ``i``, matching slice_local_heads.
    rank = jax.lax.axis_index(axis_name)
    own = jax.lax.dynamic_slice_in_dim(weight_sum, rank * h_local, h_local, axis=head_axis)
    own_empty = jax.lax.dynamic_slice_in_dim(all_empty, rank * h_local, h_local, axis=head_axis)
    o = num / jnp.maximum(own, 1e-30)[..., None]
    o = jnp.where(own_empty[..., None], jnp.float32(0.0), o)
    return o.astype(local_o.dtype)


def a2a_merge_dcp_attention(
    local_o,
    local_lse,
    h_local: int,
    axis_name: str = "tensor",
):
    """Like :func:`merge_scatter_dcp_attention`, with one all-to-all of ``[o | lse]``."""
    t, h_all, dv = local_o.shape
    if h_local <= 0 or h_all % h_local:
        raise ValueError(f"h_local {h_local} must divide all-head count {h_all}")
    n = h_all // h_local
    lse_bits = jax.lax.bitcast_convert_type(local_lse.astype(jnp.float32), local_o.dtype)
    if lse_bits.ndim == local_o.ndim - 1:
        lse_bits = lse_bits[..., None]
    c = lse_bits.shape[-1]
    packed = jnp.concatenate([local_o, lse_bits], axis=-1)
    recv = jax.lax.all_to_all(packed, axis_name, split_axis=1, concat_axis=0, tiled=True)
    recv = recv.reshape(n, t, h_local, dv + c)
    part_lse = recv[..., dv:]
    part_lse = jax.lax.bitcast_convert_type(part_lse if c > 1 else part_lse[..., 0], jnp.float32)
    o, _ = merge_dcp_attention_jax(recv[..., :dv], part_lse)
    return o
