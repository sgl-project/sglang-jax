"""In-place slot cloning for recurrent-state copy-on-write.

The recurrent pool keeps temporal state and convolution state in separate HBM
buffers.  Each invocation aliases its buffer input to its output and copies the
selected source rows directly to their destination rows; it therefore avoids
the full-buffer materialization produced by ``buf.at[dst].set(buf[src])``.
"""

from __future__ import annotations

import functools
import math
import os

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp


def _interpret_enabled() -> bool:
    return os.environ.get("PALLAS_INTERPRET", "").strip().lower() in ("1", "true")


def _block_size(num_elements: int) -> int:
    """Choose a lane-friendly block that tiles a slot without a tail mask."""
    for candidate in range(min(num_elements, 128), 0, -1):
        if num_elements % candidate == 0:
            return candidate
    raise AssertionError("every positive integer has a divisor in [1, num_elements]")


def _clone_slots_kernel(src_ref, dst_ref, buffer_ref, out_ref, *, block_size: int):
    """Copy one contiguous payload tile for one (src, dst) pair."""
    pair = pl.program_id(0)
    tile = pl.program_id(1)
    src = src_ref[0]
    dst = dst_ref[0]
    offset = tile * block_size
    payload_slice = pl.ds(offset, block_size)

    # ``src == 0`` is the fixed-shape no-op sentinel. Reading the output in
    # that case makes the aliasing contract explicit and preserves slot zero.
    src_payload = buffer_ref[src, payload_slice]
    dst_payload = out_ref[dst, payload_slice]
    out_ref[dst, payload_slice] = jnp.where(src == 0, dst_payload, src_payload)


def _slow_clone(buffer: jax.Array, src_indices: jax.Array, dst_indices: jax.Array) -> jax.Array:
    """Portable reference used outside TPU Pallas lowering."""
    payload_dims = (1,) * (buffer.ndim - 1)
    values = jnp.where(
        (src_indices == 0).reshape((-1,) + payload_dims),
        buffer[dst_indices],
        buffer[src_indices],
    )
    return buffer.at[dst_indices].set(values)


def clone_slots_inplace(
    buffer: jax.Array,
    src_indices: jax.Array,
    dst_indices: jax.Array,
    *,
    interpret: bool | None = None,
) -> jax.Array:
    """Clone local pool slots into an aliased output buffer.

    ``buffer`` must have slot as its leading dimension. ``src_indices`` and
    ``dst_indices`` are rank-one, per-DP-rank-local arrays with identical
    shapes. A source index of zero leaves the corresponding destination slot
    unchanged. Callers must provide pairwise distinct destination slots that
    do not overlap any non-zero source slot in the same launch.
    """
    if buffer.ndim < 2:
        raise ValueError(f"buffer must have a slot and payload dimension, got {buffer.shape}")
    if src_indices.ndim != 1 or dst_indices.ndim != 1:
        raise ValueError(
            "src_indices and dst_indices must be rank-one, got "
            f"{src_indices.shape} and {dst_indices.shape}"
        )
    if src_indices.shape != dst_indices.shape:
        raise ValueError(
            f"src_indices shape {src_indices.shape} != dst_indices shape {dst_indices.shape}"
        )
    if src_indices.dtype != jnp.int32 or dst_indices.dtype != jnp.int32:
        raise ValueError("src_indices and dst_indices must use int32")
    if src_indices.shape[0] == 0:
        return buffer

    slot_payload = math.prod(buffer.shape[1:])
    block_size = _block_size(slot_payload)
    buffer_2d = buffer.reshape((buffer.shape[0], slot_payload))
    if interpret is None:
        # CPU is test-only for this TPU-oriented kernel; Pallas interpret keeps
        # the pool's unit tests runnable without a TPU backend.
        interpret = _interpret_enabled() or jax.default_backend() == "cpu"
    if not interpret and jax.default_backend() != "tpu":
        return _slow_clone(buffer, src_indices, dst_indices)

    kernel = functools.partial(_clone_slots_kernel, block_size=block_size)
    cloned_2d = pl.pallas_call(
        kernel,
        grid=(src_indices.shape[0], slot_payload // block_size),
        in_specs=[
            pl.BlockSpec((1,), lambda pair, _tile: (pair,)),
            pl.BlockSpec((1,), lambda pair, _tile: (pair,)),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        out_shape=jax.ShapeDtypeStruct(buffer_2d.shape, buffer_2d.dtype),
        input_output_aliases={2: 0},
        interpret=interpret,
        name="recurrent-h0-clone",
    )(src_indices, dst_indices, buffer_2d)
    return cloned_2d.reshape(buffer.shape)
