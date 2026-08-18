"""Backport jax 0.9.x `_slice_memref` to jax 0.8.1 for Pathways IFRT compat.

Root cause (verified via source diff, not guessed):
- jaxlib 0.8.1 `_slice_memref` emits `ir.StridedLayoutAttr` on the result type
  of `tpu.memref_slice`/`tpu.memref_squeeze` (lowering.py:1443,1460).
- Pathways worker image `jax-0.8.1` is actually g3@20260422 (same digest as
  `jax-0.9.2` tag), whose `MemRefReshapeOp::verify()` (tpu_ops.cc:565+) rejects
  non-identity/non-tiled source layouts.
- Direct libtpu skips verify() (in-memory IR); Pathways parseSourceString runs
  it -> "Only identity or tiled layouts supported".
- jaxlib 0.9.x `_slice_memref` dropped the explicit StridedLayoutAttr -> result
  type defaults to identity layout -> verify() passes.

This module monkeypatches 0.8.1's `_slice_memref` to the 0.9.x behavior. Call
`install()` once before any Pallas kernel is jit-compiled (e.g. in scheduler
init when `JAX_PLATFORMS=proxy`). No-op on jax>=0.9.0.
"""

from __future__ import annotations

import logging

import jax

logger = logging.getLogger(__name__)


def install() -> None:
    if tuple(int(x) for x in jax.__version__.split(".")[:2]) >= (0, 9):
        logger.info("[pathways_compat] jax %s >= 0.9, skip", jax.__version__)
        return

    from jax._src.pallas.mosaic import lowering as _lo

    if getattr(_lo._slice_memref, "_pathways_patched", False):
        return

    ir = _lo.ir
    tpu = _lo.tpu
    _fold = _lo._fold_and_get_constant_value
    _i2sss = _lo._indexer_to_start_size_stride

    def _slice_memref_patched(ref, indexer, ref_dtype, ref_block_shape):
        assert ref_block_shape is not None
        starts, sizes, strides, squeeze_dims, ref_block_shape = _i2sss(
            indexer, ref_block_shape, cast_to_index=False
        )
        if not all((s is None or s == 1) for s in strides):
            raise NotImplementedError("Strided slices of references are unsupported.")

        ir_dyn = ir.ShapedType.get_dynamic_size()
        static_sizes = []
        dynamic_sizes = []
        for s in sizes:
            if not isinstance(s, ir.Value):
                static_sizes.append(s)
            elif (v := _fold(s)) is not None:
                static_sizes.append(v)
            else:
                static_sizes.append(ir_dyn)
                dynamic_sizes.append(s)

        ref_ty = ir.MemRefType(ref.type)
        out_ty = ir.MemRefType.get(
            static_sizes, ref_ty.element_type, memory_space=ref_ty.memory_space
        )
        out = tpu.memref_slice(out_ty, ref, starts, dynamic_sizes)
        if any(squeeze_dims):
            ref_ty = out_ty
            out_ty = ir.MemRefType.get(
                [d for i, d in enumerate(ref_ty.shape) if not squeeze_dims[i]],
                ref_ty.element_type,
                memory_space=ref_ty.memory_space,
            )
            out = tpu.memref_squeeze(out_ty, out)
        return out, ref_block_shape

    _slice_memref_patched._pathways_patched = True
    _lo._slice_memref = _slice_memref_patched
    logger.info(
        "[pathways_compat] patched jax %s _slice_memref (drop StridedLayoutAttr)",
        jax.__version__,
    )
