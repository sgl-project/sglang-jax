"""Reset compressor-state slots of new requests by DMA (Pallas TPU).

Every attention layer of a step must clear the recurrent compressor state of the
requests that start in that step (``state_init_mask``: first extend chunk only). The
XLA form ``state.at[slots].set(empty, mode="drop")`` reads a ``[T, *slot]`` template
and runs the scatter for every row of the batch even when no slot is valid, which is
the case for every decode step: at bs=64 the HCA C128 state (``[128, 2, D]`` f32 per
slot, 512 KiB) cost 1.6 ms per decode step across the 20 HCA layers on v7x, and the
CSA/indexer states another ~0.4 ms. This kernel copies one template slot into each
valid destination and does nothing else, so a step without new requests costs a
scalar loop over the batch.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _kernel(slots_ref, template_ref, _, state_ref, sem, *, capacity):
    count = slots_ref.shape[0]

    def valid(slot):
        return (slot >= 0) & (slot < capacity)

    def issue(i, carry):
        slot = slots_ref[i]

        @pl.when(valid(slot))
        def _start():
            pltpu.make_async_copy(template_ref, state_ref.at[slot], sem).start()

        return carry

    def wait(i, carry):
        slot = slots_ref[i]

        @pl.when(valid(slot))
        def _wait():
            pltpu.make_async_copy(template_ref, state_ref.at[slot], sem).wait()

        return carry

    lax.fori_loop(0, count, issue, 0)
    lax.fori_loop(0, count, wait, 0)


STATE_INIT_KERNEL_ENV = "DSV4_STATE_INIT_KERNEL"


def state_init_kernel_enabled() -> bool:
    """Select the DMA reset, or the numerical XLA fallback for diagnostics."""
    return os.environ.get(STATE_INIT_KERNEL_ENV, "1") == "1"


def init_state_slots(state, slots, template, *, capacity=None, interpret=None):
    """``state.at[slots].set(template)`` for every ``0 <= slot < capacity``, in place.

    ``state`` [S, *slot_shape], ``slots`` [T] int32 (entries outside ``[0, capacity)``
    are skipped; ``capacity`` defaults to ``S`` and lets callers keep a padding slot
    at the end untouched), ``template`` [*slot_shape] with ``state.dtype``. Duplicate
    slots write the same template twice, which is harmless. The result aliases ``state``.
    """
    state = jnp.asarray(state)
    template = jnp.asarray(template, state.dtype)
    slots = jnp.asarray(slots, jnp.int32)
    if template.shape != state.shape[1:]:
        raise ValueError("template must have the shape of one state slot")
    if slots.ndim != 1:
        raise ValueError("slots must be a 1-D int32 array")
    capacity = state.shape[0] if capacity is None else int(capacity)
    if not 0 <= capacity <= state.shape[0]:
        raise ValueError("capacity must not exceed the number of state slots")
    if interpret is None:
        interpret = jax.default_backend() != "tpu"
    return pl.pallas_call(
        lambda *refs: _kernel(*refs, capacity=capacity),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=(
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ),
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=(pltpu.SemaphoreType.DMA,),
        ),
        out_shape=jax.ShapeDtypeStruct(state.shape, state.dtype),
        input_output_aliases={2: 0},
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        interpret=interpret,
        name=f"dsv4-state-init-{'x'.join(str(d) for d in state.shape[1:])}",
    )(slots, template, state)
