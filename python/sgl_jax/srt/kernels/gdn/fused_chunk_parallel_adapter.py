"""Startup validation and state-pool adaptation for fused chunk-parallel GDN."""

from __future__ import annotations

import os
from collections.abc import Iterable

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

_TPU_LANE_WIDTH = 128
_TRACK_VALIDATION_ENV = "SGLANG_JAX_GDN_DEBUG_VALIDATE_TRACK_INDICES"


def _debug_track_validation_enabled() -> bool:
    return os.environ.get(_TRACK_VALIDATION_ENV, "0").lower() in {"1", "true"}


class GDNPrefillCapabilityError(RuntimeError):
    """An explicitly requested GDN prefill implementation is unsupported."""


def _mesh_devices(mesh) -> tuple[object, ...]:
    devices = getattr(mesh, "devices", None)
    if devices is None:
        return ()
    flat = getattr(devices, "flat", None)
    if flat is not None:
        return tuple(flat)
    if isinstance(devices, Iterable):
        return tuple(devices)
    return (devices,)


def validate_fused_chunk_parallel_capability(
    *,
    mesh,
    dtype,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
) -> None:
    """Validate only construction inputs; never probe the global JAX backend."""
    devices = _mesh_devices(mesh)
    mesh_is_tpu = bool(devices) and all(
        getattr(device, "platform", None) == "tpu" for device in devices
    )
    if not mesh_is_tpu:
        raise GDNPrefillCapabilityError(
            "fused_chunk_parallel requires a TPU mesh; CPU Pallas interpret is not "
            "a supported execution path for the fused DMA/state pipeline."
        )
    if dtype is None:
        raise GDNPrefillCapabilityError("fused_chunk_parallel requires a BF16 activation dtype.")
    if jnp.dtype(dtype) != jnp.dtype(jnp.bfloat16):
        raise GDNPrefillCapabilityError("fused_chunk_parallel requires BF16 activation dtype.")
    if min(num_k_heads, num_v_heads, head_k_dim, head_v_dim) <= 0:
        raise GDNPrefillCapabilityError(
            "fused_chunk_parallel requires positive head counts and dimensions."
        )
    for name, head_dim in (("head_k_dim", head_k_dim), ("head_v_dim", head_v_dim)):
        if head_dim % _TPU_LANE_WIDTH:
            raise GDNPrefillCapabilityError(
                f"fused_chunk_parallel requires {name} to be divisible by "
                f"the TPU lane width {_TPU_LANE_WIDTH}; got {head_dim}."
            )
    if num_v_heads % num_k_heads != 0:
        raise GDNPrefillCapabilityError(
            "fused_chunk_parallel requires num_v_heads to be divisible by num_k_heads."
        )
    if conv_kernel_size < 2:
        raise GDNPrefillCapabilityError(
            "fused_chunk_parallel requires conv_kernel_size >= 2 for its state shape."
        )


def _fused_chunk_parallel_kernel(*args, **kwargs):
    """Import the fused kernel only when the selected path executes."""
    from sgl_jax.srt.kernels.gdn.fused_chunk_parallel import fused_conv1d_gdn

    return fused_conv1d_gdn(*args, **kwargs)


def _validate_track_indices(
    track_indices: jax.Array | None,
    state_indices: jax.Array,
    pool_size: int,
    track_mask: jax.Array | None = None,
) -> jax.Array | None:
    """Reject track metadata that could alias or corrupt a state pool."""
    if track_indices is None:
        return
    if track_indices.shape != state_indices.shape:
        raise ValueError(
            "track_indices must have the same shape as state_indices; "
            f"got {track_indices.shape} and {state_indices.shape}."
        )
    if track_mask is not None and track_mask.shape != track_indices.shape:
        raise ValueError(
            "track_mask must have the same shape as track_indices; "
            f"got {track_mask.shape} and {track_indices.shape}."
        )

    try:
        track = np.asarray(track_indices)
        running = np.asarray(state_indices)
        mask = None if track_mask is None else np.asarray(track_mask, dtype=np.bool_)
    except jax.errors.TracerArrayConversionError:
        # Under JIT, keep the same checks as a runtime assertion. The eager
        # path above raises ValueError before the fused kernel is entered.
        active = track_indices != 0 if track_mask is None else track_mask.astype(jnp.bool_)
        duplicate = jnp.any(
            active[:, None]
            & active[None, :]
            & (track_indices[:, None] == track_indices[None, :])
            & ~jnp.eye(track_indices.size, dtype=jnp.bool_)
        )
        active_dummy = jnp.any(active & (track_indices == 0))
        out_of_range = jnp.any(active & ((track_indices < 0) | (track_indices >= pool_size)))
        aliases_running = jnp.any(
            active[:, None] & (track_indices[:, None] == state_indices[None, :])
        )
        invalid = active_dummy | duplicate | out_of_range | aliases_running

        def _raise_invalid(values):
            tracked, running_indices, active_mask = values

            def _raise_on_host(tracked_host, running_host, mask_host):
                _validate_track_indices(
                    np.asarray(tracked_host),
                    np.asarray(running_host),
                    pool_size,
                    track_mask=np.asarray(mask_host),
                )

            jax.debug.callback(
                _raise_on_host,
                tracked,
                running_indices,
                active_mask,
            )

        jax.lax.cond(
            invalid,
            _raise_invalid,
            lambda _: None,
            (track_indices, state_indices, active),
        )
        return invalid

    active = track != 0 if mask is None else mask
    active_track = track[active]
    if mask is not None and np.any(active_track == 0):
        raise ValueError("active track index cannot target dummy slot 0.")
    if np.any((active_track < 0) | (active_track >= pool_size)):
        raise ValueError("track_indices contains an out of range state-pool slot.")
    if np.unique(active_track).size != active_track.size:
        raise ValueError("track_indices contains a duplicate checkpoint slot.")
    if np.intersect1d(active_track, running).size:
        raise ValueError("track_indices aliases a running state slot.")
    return None


def _validate_track_indices_per_dp(
    track_indices: jax.Array | None,
    state_indices: jax.Array,
    pool_size: int,
    dp: int,
    track_mask: jax.Array | None = None,
) -> jax.Array | None:
    """Validate local state-pool indices independently for each DP rank."""
    if track_indices is None:
        return None
    if track_indices.shape != state_indices.shape:
        raise ValueError(
            "track_indices must have the same shape as state_indices; "
            f"got {track_indices.shape} and {state_indices.shape}."
        )
    if track_mask is not None and track_mask.shape != track_indices.shape:
        raise ValueError(
            "track_mask must have the same shape as track_indices; "
            f"got {track_mask.shape} and {track_indices.shape}."
        )
    if track_indices.size % dp:
        raise ValueError(f"track_indices size {track_indices.size} must be divisible by DP={dp}.")

    track_by_dp = track_indices.reshape(dp, -1)
    state_by_dp = state_indices.reshape(dp, -1)
    mask_by_dp = None if track_mask is None else track_mask.reshape(dp, -1)
    invalid_per_dp = []
    for dp_rank in range(dp):
        invalid = _validate_track_indices(
            track_by_dp[dp_rank],
            state_by_dp[dp_rank],
            pool_size,
            track_mask=None if mask_by_dp is None else mask_by_dp[dp_rank],
        )
        if invalid is not None:
            invalid_per_dp.append(invalid)
    if not invalid_per_dp:
        return None
    return jnp.any(jnp.stack(invalid_per_dp))


def _scatter_active(
    pool: jax.Array,
    indices: jax.Array,
    values: jax.Array,
) -> jax.Array:
    """Scatter request values while keeping the dummy slot 0 unchanged."""
    keep_dummy = (indices == 0).reshape((-1,) + (1,) * (pool.ndim - 1))
    safe_values = jnp.where(keep_dummy, pool[indices], values.astype(pool.dtype))
    return pool.at[indices].set(safe_values)


def _fused_chunk_parallel_prefill_local(
    mixed_qkv,
    b,
    a,
    conv_state,
    recurrent_state,
    conv_weight,
    a_log,
    dt_bias,
    cu_seqlens,
    state_indices,
    track_indices,
    has_initial_state,
    seq_lens,
    *,
    n_kq,
    n_v,
    d_k,
    d_v,
    kernel_size,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Adapt one local shard and its state pool to the fused-kernel ABI."""

    initial_conv = conv_state[state_indices]
    initial_recurrent = recurrent_state[state_indices]
    initial_conv = jnp.where(
        has_initial_state[:, None, None],
        initial_conv,
        jnp.zeros_like(initial_conv),
    )
    initial_recurrent = jnp.where(
        has_initial_state[:, None, None, None],
        initial_recurrent,
        jnp.zeros_like(initial_recurrent),
    )

    kernel_conv_pool = _scatter_active(
        conv_state.swapaxes(-1, -2),
        state_indices,
        initial_conv.swapaxes(-1, -2),
    )
    kernel_recurrent_pool = _scatter_active(
        recurrent_state,
        state_indices,
        initial_recurrent,
    )
    distribution = jnp.asarray(
        [0, 0, state_indices.shape[0]],
        dtype=jnp.int32,
    )

    (kernel_conv_result, kernel_recurrent_result), output = _fused_chunk_parallel_kernel(
        mixed_qkv,
        b,
        a,
        kernel_conv_pool,
        kernel_recurrent_pool,
        conv_weight[:, None, :],
        None,
        a_log,
        dt_bias,
        cu_seqlens,
        state_indices,
        distribution,
        seq_lens,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
    )

    query_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    has_tokens = query_lens > 0
    running_conv = kernel_conv_result[state_indices].swapaxes(-1, -2)
    running_recurrent = kernel_recurrent_result[state_indices]
    running_conv = jnp.where(
        has_tokens[:, None, None],
        running_conv,
        initial_conv,
    )
    running_recurrent = jnp.where(
        has_tokens[:, None, None, None],
        running_recurrent,
        initial_recurrent,
    )

    new_conv_state = _scatter_active(conv_state, state_indices, running_conv)
    new_recurrent_state = _scatter_active(
        recurrent_state,
        state_indices,
        running_recurrent,
    )
    if track_indices is not None:
        new_conv_state = _scatter_active(
            new_conv_state,
            track_indices,
            running_conv,
        )
        new_recurrent_state = _scatter_active(
            new_recurrent_state,
            track_indices,
            running_recurrent,
        )

    output = output.reshape(output.shape[0], n_v, d_v).astype(mixed_qkv.dtype)
    return (
        output,
        new_conv_state.astype(conv_state.dtype),
        new_recurrent_state.astype(recurrent_state.dtype),
    )


def fused_chunk_parallel_prefill(
    backend,
    mixed_qkv,
    conv_state,
    recurrent_state,
    b,
    a,
    conv_weight,
    a_log,
    dt_bias,
    seq_lens,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run the thin adapter once per DP/TP shard."""
    metadata = backend.forward_metadata
    cu_seqlens = metadata.cu_q_lens
    state_indices = metadata.recurrent_indices
    has_initial_state = metadata.has_initial_state
    track_indices = metadata.recurrent_track_indices
    track_mask = metadata.recurrent_track_mask
    if (track_indices is None) != (track_mask is None):
        raise ValueError("track_indices and track_mask must either both be set or both be None.")
    dp = int(backend.mesh.shape.get("data", 1))
    if conv_state.shape[0] % dp:
        raise ValueError(f"state pool size {conv_state.shape[0]} must be divisible by DP={dp}.")
    invalid_track = None
    if _debug_track_validation_enabled():
        invalid_track = _validate_track_indices_per_dp(
            track_indices,
            state_indices,
            conv_state.shape[0] // dp,
            dp,
            track_mask=track_mask,
        )

    tp = int(backend.mesh.shape.get("tensor", 1))
    data_axis = "data" if "data" in backend.mesh.shape else None
    n_kq = backend.num_k_heads // tp
    n_v = backend.num_v_heads // tp

    def _prefill_local(
        mixed_qkv_l,
        b_l,
        a_l,
        conv_state_l,
        recurrent_state_l,
        conv_weight_l,
        a_log_l,
        dt_bias_l,
        cu_seqlens_l,
        state_indices_l,
        has_initial_state_l,
        seq_lens_l,
        track_indices_l=None,
        track_mask_l=None,
    ):
        if track_indices_l is not None:
            track_indices_l = jnp.where(track_mask_l, track_indices_l, 0)
        return _fused_chunk_parallel_prefill_local(
            mixed_qkv_l,
            b_l,
            a_l,
            conv_state_l,
            recurrent_state_l,
            conv_weight_l,
            a_log_l,
            dt_bias_l,
            cu_seqlens_l,
            state_indices_l,
            track_indices_l,
            has_initial_state_l,
            seq_lens_l,
            n_kq=n_kq,
            n_v=n_v,
            d_k=backend.head_k_dim,
            d_v=backend.head_v_dim,
            kernel_size=backend.conv_kernel_size,
        )

    in_specs = [
        P(data_axis, "tensor"),
        P(data_axis, "tensor"),
        P(data_axis, "tensor"),
        P(data_axis, "tensor", None),
        P(data_axis, "tensor", None, None),
        P("tensor", None),
        P("tensor"),
        P("tensor"),
        P(data_axis),
        P(data_axis),
        P(data_axis),
        P(data_axis),
    ]
    args = [
        mixed_qkv,
        b,
        a,
        conv_state,
        recurrent_state,
        conv_weight,
        a_log,
        dt_bias,
        cu_seqlens,
        state_indices,
        has_initial_state,
        seq_lens,
    ]
    if track_indices is not None:
        in_specs += [P(data_axis), P(data_axis)]
        args += [track_indices, track_mask]

    def _run_fused(_):
        return jax.shard_map(
            _prefill_local,
            mesh=backend.mesh,
            in_specs=tuple(in_specs),
            out_specs=(
                P(data_axis, "tensor", None),
                P(data_axis, "tensor", None),
                P(data_axis, "tensor", None, None),
            ),
            check_vma=False,
        )(*args)

    if invalid_track is None:
        return _run_fused(None)

    def _reject_invalid(_):
        return (
            jnp.zeros(
                (mixed_qkv.shape[0], n_v, backend.head_v_dim),
                dtype=mixed_qkv.dtype,
            ),
            conv_state,
            recurrent_state,
        )

    # Runtime-invalid metadata takes this branch under the enclosing model JIT.
    # The fused shard_map is confined to the other branch, so no state update
    # can be computed or published after the fail-loud callback above.
    return jax.lax.cond(
        invalid_track,
        _reject_invalid,
        _run_fused,
        operand=None,
    )
