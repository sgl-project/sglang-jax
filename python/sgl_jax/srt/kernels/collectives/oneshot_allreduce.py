"""Split-barrier one-shot all-reduce over one mesh axis, written with Pallas remote DMAs.

Replaces ``jax.lax.psum(x, axis)`` for small decode-shaped messages (<= 64 KiB) on a single v7x
host where the XLA emitter is latency-bound. The kernel form is :func:`split_flat_all_reduce`
(the HBM-latency form): every device pushes its operand into its own slot of an HBM receive slab
on every peer, and the N-slot sum is left to XLA (f32).

Constraints: axis size >= 2; ``x`` 2-D; f32 accumulation then cast back to ``x.dtype``.  Not
bit-identical to ``psum`` (different summation order) -- gate with the accuracy suite, not with
a bitwise compare.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.environ import envs
from sgl_jax.srt.utils.jax_utils import is_tpu_runtime

_MESH = pl.DeviceIdType.MESH


def _mesh_device_id(mesh_axis_names, reduce_axis, j):
    """Mesh coordinate of rank ``j`` on ``reduce_axis``; other axes keep the caller's index."""
    return tuple(j if a == reduce_axis else lax.axis_index(a) for a in mesh_axis_names)


# ---------------------------------------------------------------------------
# Split-barrier protocol (
# design: internal "latency_psum"). Every all-reduce is TWO pallas_calls that share a collective_id:
#   (1) ``barrier_alloc``: zero inputs; outputs = the receive slab (HBM) + the receive-side DMA
#       semaphore, XLA-allocated; the body only signals the runtime barrier semaphore to every peer.
#       Tagged async_type='collective-start' so XLA schedules it early (behind the upstream matmul).
#   (2) ``done``: the slab/semaphore of (1) as aliased (input_output_aliases), ``x`` read-only; waits
#       the barrier, pushes ``x`` into slot ``me`` of every peer, and exits WITHOUT an ack release.
#       Tagged async_type='collective-done'.
# No ack is safe because (a) start/done of one collective_id are a capacity-1 resource for the XLA
# scheduler (start(k+1) never precedes done(k) on any device) and (b) the receive slab of instance k
# is a live output until its consumer ran, so instance k+1 never reuses the address.
# Under ``interpret=True`` (CPU, 8 host devices) the barrier is signalled inside (2) and the DMA
# semaphores are scratch (the interpreter cannot carry semaphores across pallas_calls); the DMA
# pattern and the arithmetic are identical, so the CPU test checks the exchange, not the scheduling.
# ---------------------------------------------------------------------------
_SPLIT_SITE_CID = {"qmm": 4, "fused": 5, "moe": 5}  # two site classes, two barrier semaphores
_SPLIT_HIER_MAX_BYTES = 64 << 10  # the flat form only pays off for small operands
_SPLIT_LATENCY_META = "300000"  # latency hint on the start/done xla_metadata
_SEM_SPEC = pl.BlockSpec(memory_space=pltpu.MemorySpace.SEMAPHORE)
_VM_SPEC = pl.BlockSpec(memory_space=pltpu.VMEM)


def _is_sem(leaf):
    dt = getattr(leaf, "dtype", None)
    return isinstance(dt, pltpu.SemaphoreType) or "sem" in str(dt)


def _alloc_spec(leaf):
    """Call (1) out_specs: the memory space each MemoryRef was declared with."""
    if _is_sem(leaf):
        return _SEM_SPEC
    return pl.BlockSpec(memory_space=leaf.memory_space)


def _io_spec_hbm(leaf):
    return _SEM_SPEC if _is_sem(leaf) else pl.BlockSpec(memory_space=pl.ANY)


def _gen7_flags(interpret):
    if interpret:
        return {}
    try:
        if pltpu.get_tpu_info().generation >= 7:
            return {"XLA_TPU_STORE_TO_LOAD_FORWARDING_WINDOW": 99999}
    except Exception:  # noqa: BLE001 - AOT / no TPU: flag only matters on hardware
        pass
    return {}


def _split_cparams(cid, interpret, has_custom_barrier):
    return pltpu.CompilerParams(
        collective_id=cid,
        has_side_effects=pltpu.SideEffectType.DATAFLOW_SIDE_EFFECTING,
        flags=_gen7_flags(interpret),
        allow_collective_id_without_custom_barrier=not has_custom_barrier,
    )


class _async_meta:
    def __init__(self, kind, interpret):
        from jax.experimental import xla_metadata

        self.cm = (
            None
            if interpret
            else xla_metadata.set_xla_metadata(
                async_type=kind, latency_metadata=_SPLIT_LATENCY_META
            )
        )

    def __enter__(self):
        return self.cm.__enter__() if self.cm else None

    def __exit__(self, *a):
        return self.cm.__exit__(*a) if self.cm else None


def _barrier_alloc(to_alloc, cid, n, dev, interpret, anchor=None):
    """Call (1): allocate the receive slab / semaphore as outputs and signal the barrier.

    ``anchor`` (optional, never read) is a data input that pins the call's earliest position:
    a zero-input custom call floats to the top of a straight-line program (observed: all 157
    per-step instances hoisted together -> 157 live VMEM receive sets -> MSA failure). The
    caller passes the upstream matmul's input so the start lands after the previous layer and
    before this layer's matmul."""

    def start(*refs):
        if not interpret:
            bar = pltpu.get_barrier_semaphore()
            for j in range(n):
                pl.semaphore_signal(bar, 1, device_id=dev(j), device_id_type=_MESH)

    kw = {"interpret": pltpu.InterpretParams()} if interpret else {}
    args = () if anchor is None else (anchor,)
    call = pl.pallas_call(
        start,
        out_shape=to_alloc,
        in_specs=[pl.BlockSpec(memory_space=pl.ANY)] * len(args),
        out_specs=jax.tree.map(_alloc_spec, to_alloc),
        compiler_params=_split_cparams(cid, interpret, has_custom_barrier=not interpret),
        **kw,
    )
    with _async_meta("collective-start", interpret):
        outs = list(call(*args))
    if not interpret:
        # Constrain the allocated slab to HBM, the space it was declared in
        outs = [
            o if _is_sem(a) else pltpu.with_memory_space_constraint(o, pltpu.MemorySpace.HBM)
            for o, a in zip(outs, to_alloc)
        ]
    return tuple(outs)


def _done_call(kernel, aliased, x, scratch, cid, interpret):
    """Call (2): ``aliased`` leaves aliased in->out, ``x`` read-only VMEM, local send semaphores scratch."""
    n_io = len(aliased)
    kw = {"interpret": pltpu.InterpretParams()} if interpret else {}
    call = pl.pallas_call(
        kernel,
        out_shape=aliased,
        out_specs=tuple(_io_spec_hbm(a) for a in aliased),
        in_specs=tuple(_io_spec_hbm(a) for a in aliased) + (_VM_SPEC,),
        scratch_shapes=scratch,
        input_output_aliases={i: i for i in range(n_io)},
        compiler_params=_split_cparams(cid, interpret, has_custom_barrier=True),
        **kw,
    )
    with _async_meta("collective-done", interpret):
        return call(*aliased, x)


def _wait_barrier(n, dev, interpret):
    bar = pltpu.get_barrier_semaphore()
    if interpret:
        for j in range(n):
            pl.semaphore_signal(bar, 1, device_id=dev(j), device_id_type=_MESH)
    pl.semaphore_wait(bar, n)


def split_flat_all_reduce(
    x: jax.Array,
    reduce_axis: str,
    *,
    mesh_axis_names: tuple[str, ...],
    num_devices: int,
    interpret: bool = False,
    collective_id: int = 5,
    anchor: jax.Array | None = None,
) -> jax.Array:
    """Flat one-shot all-reduce on the split protocol with the receive slab in HBM (the measured form
    ``lat_hbm``): call (1) allocates ``recv[N, rows, cols]`` in HBM (no VMEM memory-space output,
    so no MSA colouring -- the VMEM forms fail to compile beyond a handful of straight-line
    instances, 10-04); call (2) pushes ``x`` into slot ``me`` of every peer (N-1 remote DMAs)
    plus a local copy, waits, no ack. The N-slot sum is left to XLA (f32). Costs ~+1 us per
    call vs a VMEM receive form (measured) and (N-1)x the bytes of a two-level form,
    so it is routed only for small operands (bs1 48 KB)."""
    if x.ndim != 2:
        raise ValueError(f"split_flat_all_reduce expects a 2-D operand, got {x.shape}")
    n = int(num_devices)
    if n < 2:
        raise ValueError(f"split_flat_all_reduce needs an axis size >= 2, got {n}")
    rows, cols = x.shape
    dt = x.dtype

    def dev(j):
        return _mesh_device_id(mesh_axis_names, reduce_axis, j)

    alloc = (pltpu.HBM((n, rows, cols), dt),)
    if not interpret:
        alloc += (pltpu.SemaphoreType.DMA((1,)),)
    scratch = [pltpu.SemaphoreType.DMA]
    if interpret:
        scratch += [pltpu.SemaphoreType.DMA]
    io = _barrier_alloc(alloc, collective_id, n, dev, interpret, anchor=anchor)
    n_io = len(io)

    def kernel(*refs):  # inputs (aliased..., x), outputs (aliased...), scratch
        x_ref = refs[n_io]
        outs = refs[n_io + 1 : 2 * n_io + 1]
        scr = refs[2 * n_io + 1 :]
        recv_ref = outs[0]
        ssem = scr[0]
        dsem = scr[1] if interpret else outs[1].at[0]
        me = lax.axis_index(reduce_axis)
        _wait_barrier(n, dev, interpret)
        dmas = []
        for off in range(1, n):
            t = (me + off) % n
            d = pltpu.make_async_remote_copy(
                x_ref, recv_ref.at[me], ssem, dsem, device_id=dev(t), device_id_type=_MESH
            )
            d.start()
            dmas.append(d)
        lc = pltpu.make_async_copy(x_ref, recv_ref.at[me], ssem)
        lc.start()
        lc.wait()
        for d in dmas:
            d.wait_recv()
        for d in dmas:
            d.wait_send()

    recv = _done_call(kernel, io, x, scratch, collective_id, interpret)[0]
    return recv.astype(jnp.float32).sum(0).astype(dt)


ALL_REDUCE_KERNELS = {
    "split_flat": split_flat_all_reduce,
}


# ---------------------------------------------------------------------------
# Dispatch helper for the serving path (environ.py):
# SGLANG_JAX_ONESHOT_AR_MIN_BYTES=<n>  route psum operands with n <= bytes <= MAX to the
#                                      Pallas one-shot kernel (default 4096; 0 = off).
# SGLANG_JAX_ONESHOT_AR_MAX_BYTES=<n>  upper bound (default 512 KiB: probe21 §20.20, TC psum
#                                      scales better from 768 KB up).
# SGLANG_JAX_ONESHOT_AR_SITES=fused,qmm  call sites that may use the kernel.
# The kernel form is split_flat (HBM receive slab), routed for operands <= SPLIT_HIER_MAX_BYTES
# (64 KiB), rest psum; collective_id per site class qmm=4 / fused,moe=5. The model-level fused
# MoE reduce path is only taken when the layer output is inside [MIN, MAX] bytes, i.e. where the
# kernel applies; outside (prefill chunks, cc64 3 MB) the graph is exactly the psum graph.
# ---------------------------------------------------------------------------
_logger = logging.getLogger(__name__)
_MIN_BYTES = envs.SGLANG_JAX_ONESHOT_AR_MIN_BYTES.get()
_MAX_BYTES = envs.SGLANG_JAX_ONESHOT_AR_MAX_BYTES.get()
# Which call sites may use the kernel: "fused" = the per-layer MoE(routed+shared) all-reduce
# assembled in the model (XLA merges those two psums into one all-reduce today), "moe" = the
# EPMoE-internal tensor psum, "qmm" = the quantized-matmul wrapper reduce (o_proj, dense MLP).
_SITES = set(filter(None, envs.SGLANG_JAX_ONESHOT_AR_SITES.get().split(",")))
_defer_depth = 0
_logged = set()


def _log_once(key, msg, *args):
    if key not in _logged:
        _logged.add(key)
        _logger.info(msg, *args)


class defer_reduce:
    """Inside this context the MoE reduce sites return their un-reduced per-device partial sums;
    the caller must finish with :func:`fused_all_reduce` on the summed partials."""

    def __enter__(self):
        global _defer_depth
        _defer_depth += 1

    def __exit__(self, *exc):
        global _defer_depth
        _defer_depth -= 1


def reduce_deferred() -> bool:
    return _defer_depth > 0


_force_fused_depth = 0


class force_fused_reduce:
    """Inside this context the model's fused MoE reduce path (``defer_reduce`` + one
    ``fused_all_reduce`` of routed + shared partials) is taken even when the one-shot
    kernel is not configured (``fused_moe_enabled`` is True here). Used by prefill
    micro-batching, which needs the MoE all-reduce issued at the model level so the
    per-device partial sum is a jit-level value (a pre-all-reduce anchor)."""

    def __enter__(self):
        global _force_fused_depth
        _force_fused_depth += 1

    def __exit__(self, *exc):
        global _force_fused_depth
        _force_fused_depth -= 1


def fused_moe_enabled(x: jax.Array | None = None) -> bool:
    """True when the model should take the fused MoE reduce path (routed + shared partials, one
    all-reduce). With ``x`` (the layer's hidden states) and the byte gate on, only when that
    all-reduce operand is inside the kernel's byte window -- the rest of the graph stays psum."""
    if _force_fused_depth > 0:
        return True
    if not (bool(_MIN_BYTES) and "fused" in _SITES):
        return False
    if x is not None:
        nbytes = int(x.shape[0]) * int(x.shape[-1]) * jnp.dtype(x.dtype).itemsize
        return _MIN_BYTES <= nbytes <= _MAX_BYTES
    return True


def fused_all_reduce(x: jax.Array, mesh, reduce_axis: str = "tensor") -> jax.Array:
    """One all-reduce over ``reduce_axis`` of per-device partials that the caller obtained under
    :class:`defer_reduce` (the partials travel as replicated-claimed global arrays, exactly as the
    shard_map wrappers already return them; ``check_vma=False`` keeps that contract)."""
    from jax import shard_map
    from jax.sharding import PartitionSpec as P

    # Explicit-sharding mode requires in_specs to match the operand's own spec (prefill hidden
    # states arrive as P('data', None)); fall back to fully replicated when no spec is visible.
    try:
        spec = jax.typeof(x).sharding.spec
    except Exception:  # noqa: BLE001
        spec = None
    if spec is None or len(spec) != x.ndim:
        spec = P(*([None] * x.ndim))
    return shard_map(
        lambda v: maybe_oneshot_psum(v, reduce_axis, mesh.axis_names, site="fused"),
        mesh=mesh,
        in_specs=spec,
        out_specs=spec,
        check_vma=False,
    )(x)


def maybe_oneshot_psum(
    x: jax.Array,
    reduce_axis: str,
    mesh_axis_names=None,
    *,
    site: str = "qmm",
    anchor: jax.Array | None = None,
) -> jax.Array:
    """``lax.psum`` unless the opt-in one-shot kernel applies to this operand and call site."""
    if site in ("moe", "qmm") and reduce_deferred() and fused_moe_enabled():
        return x  # partial sum; the model reduces routed+shared with one fused_all_reduce
    if not _MIN_BYTES or site not in _SITES or not is_tpu_runtime():
        return lax.psum(x, reduce_axis)
    if mesh_axis_names is None:
        try:
            mesh_axis_names = tuple(jax.sharding.get_abstract_mesh().axis_names)
        except Exception:  # noqa: BLE001 - no mesh context: keep XLA psum
            mesh_axis_names = ()
    n = int(lax.axis_size(reduce_axis)) if reduce_axis in mesh_axis_names else 0
    nbytes = x.size * x.dtype.itemsize
    eligible = (
        n >= 2
        and n % 2 == 0
        and x.ndim == 2
        and x.shape[1] % 128 == 0
        and x.dtype in (jnp.bfloat16, jnp.float32)
        and _MIN_BYTES <= nbytes <= _MAX_BYTES
        and nbytes <= _SPLIT_HIER_MAX_BYTES  # the flat form only pays off for small operands
    )
    key = (site, reduce_axis, tuple(x.shape), str(x.dtype), eligible)
    if not eligible:
        _log_once(
            key,
            "oneshot all-reduce[%s]: keep XLA psum for %s %s over %r (n=%d, bytes=%d)",
            site,
            x.shape,
            x.dtype,
            reduce_axis,
            n,
            nbytes,
        )
        return lax.psum(x, reduce_axis)
    _log_once(
        key,
        "oneshot all-reduce[%s]: Pallas split_flat kernel for %s %s over %r (n=%d, bytes=%d)",
        site,
        x.shape,
        x.dtype,
        reduce_axis,
        n,
        nbytes,
    )
    return split_flat_all_reduce(
        x,
        reduce_axis,
        mesh_axis_names=tuple(mesh_axis_names),
        num_devices=n,
        collective_id=_SPLIT_SITE_CID.get(site, 5),
        # bound the start's hoisting: the upstream matmul input when the caller has it,
        # else the operand itself (start then sits right before its done)
        anchor=x if anchor is None else anchor,
    )
