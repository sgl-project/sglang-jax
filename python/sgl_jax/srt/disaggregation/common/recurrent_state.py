"""DP-aware recurrent state slot gather / scatter helpers for PD disaggregation.

``RecurrentStatePool`` buffers are global ``jax.Array``s sharded along the DP
mesh axis (``total_slots = size + dp_size``; each DP rank owns
``slots_per_rank + 1`` consecutive slots, local slot ``0`` being a dummy).
Request slot IDs (``req.recurrent_pool_idx``) are rank-local, so host code that
indexes the global buffers outside ``jax.shard_map`` must first map
``(dp_rank, slot_local_idx)`` to the global slot with
:func:`recurrent_slot_to_global_idx`.

Payload layout (matches ``TransferPayload``):

* ``"recurrent_state"``: ``list[jax.Array]`` per linear layer, each
  ``(num_heads, head_dim, head_dim)``.
* ``"conv_state"``: ``list[list[jax.Array]]`` per linear layer (inner list
  mirrors ``RecurrentStatePool.conv_buffers``), each ``(proj_size, kernel - 1)``.

On multi-host deployments every leaf is exposed as this host's local shard on
a 1-D local mesh (mirrors the paged-KV path in ``prefill._extract_req_kv``).
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.disaggregation.base.transfer import recurrent_slot_to_global_idx

RECURRENT_STATE_KEY = "recurrent_state"
CONV_STATE_KEY = "conv_state"

__all__ = [
    "CONV_STATE_KEY",
    "RECURRENT_STATE_KEY",
    "build_recurrent_state_specs",
    "gather_recurrent_slot",
    "global_recurrent_slot",
    "pd_transfer_prompt_len",
    "recurrent_slot_shardings",
    "scatter_recurrent_slot",
]


def pd_transfer_prompt_len(prompt_len: int, is_hybrid: bool) -> int:
    """Number of prompt tokens P computes and transfers (KV pages + state).

    D always re-runs the last prompt token as an extend on top of the received
    prefix. That is harmless for KV (D just rewrites that token's slot), but
    recurrent (linear-attention) and conv state are cumulative: if P had
    already consumed the last token, D would fold it in a second time. So for
    hybrid models P stops one token short and ships the state *before* it.

    A 1-token prompt is left intact: D's extend then has prefix_len == 0, so
    ``has_initial_state`` is False and the received state is never read.
    Both P and D must use this helper so their page counts agree.
    """

    prompt_len = int(prompt_len)
    return prompt_len - 1 if is_hybrid and prompt_len >= 2 else prompt_len


def global_recurrent_slot(
    recurrent_state_pool: Any, slot_local_idx: int, dp_rank: int | None
) -> int:
    """Map a request's rank-local recurrent slot to the global buffer index."""

    return recurrent_slot_to_global_idx(
        slot_local_idx,
        dp_rank=int(dp_rank or 0),
        dp_size=int(getattr(recurrent_state_pool, "dp_size", 1) or 1),
        slots_per_rank=int(recurrent_state_pool.slots_per_rank),
    )


def _drop_leading_axis(sharding: NamedSharding) -> NamedSharding:
    """Per-slot sharding: remove the (DP) slot axis, keep the remaining axes."""

    return NamedSharding(sharding.mesh, P(*tuple(sharding.spec)[1:]))


def recurrent_slot_shardings(recurrent_state_pool: Any) -> tuple[NamedSharding, NamedSharding]:
    """Return ``(temporal, conv)`` shardings for one gathered slot (replicated over DP)."""

    return (
        _drop_leading_axis(recurrent_state_pool.recurrent_sharding),
        _drop_leading_axis(recurrent_state_pool.conv_sharding),
    )


def _is_explicit(mesh) -> bool:
    return any(t == jax.sharding.AxisType.Explicit for t in getattr(mesh, "axis_types", ()))


def _with_leading_none(sharding: NamedSharding) -> NamedSharding:
    return NamedSharding(sharding.mesh, P(None, *tuple(sharding.spec)))


# The slot is indexed with an int32[1] array, not a scalar: a scalar index lowers
# to dynamic_slice, which Explicit sharding rejects on the DP-sharded slot axis
# (out dim 1 is not divisible by the data axis). A length-1 index array lowers to
# gather/scatter instead, like the paged-KV gather in prefill.py. Explicit meshes
# need the result sharding spelled out; Auto meshes reject it, so it is optional.
_GET_SLOT_FNS: dict[Any, Any] = {}
_SET_SLOT_FNS: dict[Any, Any] = {}


def _get_slot_fn(out_sharding: NamedSharding):
    fn = _GET_SLOT_FNS.get(out_sharding)
    if fn is None:
        rows_sharding = (
            _with_leading_none(out_sharding) if _is_explicit(out_sharding.mesh) else None
        )

        def _get(buf, slot_idx):
            rows = buf.at[slot_idx].get(out_sharding=rows_sharding)  # (1, *slot_shape)
            return jax.lax.squeeze(rows, (0,))

        fn = jax.jit(_get, out_shardings=out_sharding)
        _GET_SLOT_FNS[out_sharding] = fn
    return fn


def _set_slot_fn(out_sharding: NamedSharding):
    fn = _SET_SLOT_FNS.get(out_sharding)
    if fn is None:
        scatter_sharding = out_sharding if _is_explicit(out_sharding.mesh) else None

        def _set(buf, slot_idx, val):
            rows = jax.lax.expand_dims(val.astype(buf.dtype), (0,))  # (1, *slot_shape)
            return buf.at[slot_idx].set(rows, out_sharding=scatter_sharding)

        # Donate the pool buffer so the scatter is in place (O(slot) HBM).
        fn = jax.jit(_set, out_shardings=out_sharding, donate_argnums=(0,))
        _SET_SLOT_FNS[out_sharding] = fn
    return fn


def _replicated_slot_index(mesh, global_slot: int) -> jax.Array:
    return jax.device_put(np.asarray([global_slot], dtype=np.int32), NamedSharding(mesh, P()))


def _to_local_shard(arr: jax.Array) -> jax.Array:
    from sgl_jax.srt.disaggregation.prefill import _global_to_local_shard

    arr.block_until_ready()
    return _global_to_local_shard(arr)


def _local_spec(shape: tuple[int, ...], dtype: Any, spec: tuple) -> jax.ShapeDtypeStruct:
    """This host's 1/nproc slice of a single-sharded-dim global array on a local mesh."""

    from jax.sharding import Mesh

    sharded_dims = [i for i, s in enumerate(spec) if s is not None]
    if len(sharded_dims) != 1:
        raise ValueError(f"expected one sharded dim in recurrent state spec, got {spec}")
    sd = sharded_dims[0]
    nproc = jax.process_count()
    lshape = tuple(shape[i] // nproc if i == sd else shape[i] for i in range(len(shape)))
    lmesh = Mesh(np.asarray(jax.local_devices()), ("_local",))
    lspec = P(*("_local" if i == sd else None for i in range(len(shape))))
    return jax.ShapeDtypeStruct(lshape, dtype, sharding=NamedSharding(lmesh, lspec))


def gather_recurrent_slot(recurrent_state_pool: Any, global_slot: int) -> dict[str, Any]:
    """Gather one request's temporal + conv state from every linear layer."""

    rec_sharding, conv_sharding = recurrent_slot_shardings(recurrent_state_pool)
    slot = _replicated_slot_index(recurrent_state_pool.mesh, int(global_slot))
    get_rec = _get_slot_fn(rec_sharding)
    get_conv = _get_slot_fn(conv_sharding)
    recurrent = [get_rec(buf, slot) for buf in recurrent_state_pool.recurrent_buffers]
    conv = [[get_conv(cbuf, slot) for cbuf in inner] for inner in recurrent_state_pool.conv_buffers]
    if jax.process_count() > 1:
        recurrent = [_to_local_shard(x) for x in recurrent]
        conv = [[_to_local_shard(x) for x in inner] for inner in conv]
    return {RECURRENT_STATE_KEY: recurrent, CONV_STATE_KEY: conv}


def build_recurrent_state_specs(recurrent_state_pool: Any) -> dict[str, Any]:
    """Decode-side pull specs matching :func:`gather_recurrent_slot` output."""

    rec_sharding, conv_sharding = recurrent_slot_shardings(recurrent_state_pool)
    multi_host = jax.process_count() > 1

    def _spec(buf, sharding):
        shape = tuple(int(d) for d in buf.shape[1:])
        if multi_host:
            return _local_spec(shape, buf.dtype, tuple(sharding.spec))
        return jax.ShapeDtypeStruct(shape, buf.dtype, sharding=sharding)

    return {
        RECURRENT_STATE_KEY: [
            _spec(buf, rec_sharding) for buf in recurrent_state_pool.recurrent_buffers
        ],
        CONV_STATE_KEY: [
            [_spec(cbuf, conv_sharding) for cbuf in inner]
            for inner in recurrent_state_pool.conv_buffers
        ],
    }


def _to_global_slot_array(arr: jax.Array, global_shape: tuple[int, ...], sharding) -> jax.Array:
    if jax.process_count() > 1 and arr.is_fully_addressable:
        # Received this host's local shard; assemble into the pool mesh (zero-copy).
        return jax.make_array_from_single_device_arrays(
            global_shape, sharding, [s.data for s in arr.addressable_shards]
        )
    if getattr(arr, "sharding", None) != sharding:
        return jax.device_put(arr, sharding)
    return arr


def scatter_recurrent_slot(
    recurrent_state_pool: Any,
    global_slot: int,
    recurrent_state: list[Any],
    conv_state: list[Any],
) -> None:
    """Write received per-layer state into ``global_slot`` of every linear layer (in place)."""

    num_layers = len(recurrent_state_pool.recurrent_buffers)
    if len(recurrent_state) != num_layers or len(conv_state) != num_layers:
        raise ValueError(
            "recurrent state layer count mismatch: "
            f"pool={num_layers}, recurrent_state={len(recurrent_state)}, "
            f"conv_state={len(conv_state)}"
        )
    rec_sharding, conv_sharding = recurrent_slot_shardings(recurrent_state_pool)
    slot = _replicated_slot_index(recurrent_state_pool.mesh, int(global_slot))

    set_rec = _set_slot_fn(recurrent_state_pool.recurrent_sharding)
    for layer, value in enumerate(recurrent_state):
        buf = recurrent_state_pool.recurrent_buffers[layer]
        value = _to_global_slot_array(value, tuple(buf.shape[1:]), rec_sharding)
        recurrent_state_pool.recurrent_buffers[layer] = set_rec(buf, slot, value)

    set_conv = _set_slot_fn(recurrent_state_pool.conv_sharding)
    for layer, entry in enumerate(conv_state):
        inner_bufs = recurrent_state_pool.conv_buffers[layer]
        values = list(entry) if isinstance(entry, (list, tuple)) else [entry]
        if len(values) != len(inner_bufs):
            raise ValueError(
                f"conv_state layer {layer} expects {len(inner_bufs)} buffers, got {len(values)}"
            )
        for i, value in enumerate(values):
            cbuf = inner_bufs[i]
            value = _to_global_slot_array(value, tuple(cbuf.shape[1:]), conv_sharding)
            inner_bufs[i] = set_conv(cbuf, slot, value)
