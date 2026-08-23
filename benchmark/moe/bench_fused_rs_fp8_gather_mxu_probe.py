"""Target-TPU probe for MXU consumption of an irregularly gathered FP8 tile.

The preceding DMA-only probe proves that the HBM-to-VMEM gather is exact.  This
probe adds exactly one operation: an unscaled matmul from that same gathered
scratch tile.  It deliberately excludes packed weights, quantization scales,
routing, and the rest of fused MoE so a failure points at scratch reshape/MXU
input layout rather than those later stages.
"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.kernels.fused_moe.fused_rs.gmm_v2_gather_scatter import (
    GatherMetadata,
    dma_gather_gm_start,
    dma_gather_gm_wait,
)


_SIZE_M = 256
_TILE_M = 128
_NUM_ROWS = 64
_K = 512
_N = 128
_NUM_LANES = 128
_K_TILES = _K // _NUM_LANES
_CASES = (
    ("bf16_contiguous_offset0", jnp.bfloat16, "contiguous", 0),
    ("bf16_irregular_offset37", jnp.bfloat16, "irregular", 37),
    ("fp8_contiguous_offset0", jnp.float8_e4m3fn, "contiguous", 0),
    ("fp8_irregular_offset0", jnp.float8_e4m3fn, "irregular", 0),
    ("fp8_irregular_offset37", jnp.float8_e4m3fn, "irregular", 37),
)


def _gather_mxu_kernel(
    indices_ref,
    payload_ref,
    rhs_ref,
    output_ref,
    gathered_ref,
    rhs_scratch_ref,
    acc_scratch_ref,
    gather_sem_ref,
    rhs_sem_ref,
    *,
    m_start: int,
):
    is_valid = (
        jnp.arange(_TILE_M, dtype=jnp.int32) < _NUM_ROWS
    ).astype(jnp.int32)
    metadata = GatherMetadata(
        m_start=jnp.int32(m_start),
        m_end=jnp.int32(m_start + _NUM_ROWS),
        num_rows=jnp.int32(_NUM_ROWS),
        is_valid=is_valid,
        src_rows=[indices_ref[i] for i in range(_TILE_M)],
    )

    dma_gather_gm_start(payload_ref, gathered_ref, gather_sem_ref, metadata)
    rhs_copy = pltpu.make_async_copy(rhs_ref, rhs_scratch_ref, rhs_sem_ref)
    rhs_copy.start()
    dma_gather_gm_wait(gathered_ref, gather_sem_ref, metadata)
    rhs_copy.wait()

    # Match the fused-RS consumer exactly: load the VMEM ref before
    # reshaping the register array and retain the size_lhs_sublane axis.
    # Reshaping the ref itself emits an unsupported tpu.memref_reshape.
    lhs = gathered_ref[...].reshape(-1, _TILE_M, _K)
    rhs = rhs_scratch_ref[...]
    acc_scratch_ref[...] = jnp.matmul(
        lhs,
        rhs,
        preferred_element_type=jnp.float32,
    )

    copy_out = pltpu.make_async_copy(
        acc_scratch_ref, output_ref, gather_sem_ref
    )
    copy_out.start()
    copy_out.wait()


def _run_probe(
    payload: jax.Array,
    rhs: jax.Array,
    indices: jax.Array,
    *,
    m_start: int,
):
    kernel = pl.pallas_call(
        functools.partial(_gather_mxu_kernel, m_start=m_start),
        out_shape=jax.ShapeDtypeStruct((1, _TILE_M, _N), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=(
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ),
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=(
                pltpu.VMEM((_TILE_M, _K_TILES, _NUM_LANES), payload.dtype),
                pltpu.VMEM((_K, _N), rhs.dtype),
                pltpu.VMEM((1, _TILE_M, _N), jnp.float32),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
        ),
        name=f"fused-rs-gather-mxu-{payload.dtype.name}-offset{m_start}",
    )
    return kernel(indices, payload, rhs)


def _payload(dtype) -> jax.Array:
    rows = jnp.arange(_SIZE_M, dtype=jnp.float32)[:, None]
    cols = jnp.arange(_K, dtype=jnp.float32)[None, :]
    values = (((rows * 13 + cols * 7) % 201) - 100) / 16
    return values.astype(dtype).reshape(_SIZE_M, _K_TILES, _NUM_LANES)


def _rhs(dtype) -> jax.Array:
    rows = jnp.arange(_K, dtype=jnp.float32)[:, None]
    cols = jnp.arange(_N, dtype=jnp.float32)[None, :]
    values = (((rows * 5 + cols * 11) % 127) - 63) / 32
    return values.astype(dtype)


def _indices(mode: str) -> jax.Array:
    rows = jnp.arange(_TILE_M, dtype=jnp.int32)
    if mode == "contiguous":
        return rows
    if mode == "irregular":
        return (rows * 17 + 3) % _SIZE_M
    raise ValueError(f"unknown index mode: {mode}")


def _comparison(expected: np.ndarray, actual: np.ndarray) -> dict:
    expected_f64 = expected.astype(np.float64)
    actual_f64 = actual.astype(np.float64)
    delta = actual_f64 - expected_f64
    denominator = max(float(np.linalg.norm(expected_f64)), 1e-12)
    return {
        "exact": bool(np.array_equal(expected_f64, actual_f64)),
        "all_finite": bool(np.isfinite(actual_f64).all()),
        "rel_l2": float(np.linalg.norm(delta) / denominator),
        "max_abs": float(np.max(np.abs(delta))),
    }


def _compare_at_best_offset(expected: np.ndarray, output: np.ndarray) -> dict:
    candidates = []
    for offset in range(_TILE_M - _NUM_ROWS + 1):
        comparison = _comparison(expected, output[offset : offset + _NUM_ROWS])
        candidates.append(
            (comparison["rel_l2"], comparison["max_abs"], offset, comparison)
        )
    _, _, best_offset, best_comparison = min(
        candidates, key=lambda item: item[:3]
    )
    return {"best_offset": best_offset, **best_comparison}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--rel-l2-threshold", type=float, default=1e-4)
    args = parser.parse_args()
    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.jsonl.write_text("", encoding="utf-8")

    for case, dtype, index_mode, m_start in _CASES:
        payload = _payload(dtype)
        rhs = _rhs(dtype)
        indices = _indices(index_mode)
        output = _run_probe(payload, rhs, indices, m_start=m_start)
        expected = jnp.matmul(
            payload.reshape(_SIZE_M, _K)[indices[:_NUM_ROWS]],
            rhs,
            preferred_element_type=jnp.float32,
        )
        jax.block_until_ready((payload, rhs, indices, output, expected))

        comparison = _compare_at_best_offset(
            np.asarray(jax.device_get(expected), dtype=np.float32),
            np.asarray(jax.device_get(output), dtype=np.float32).reshape(
                _TILE_M, _N
            ),
        )
        contract_ok = (
            comparison["all_finite"]
            and comparison["rel_l2"] <= args.rel_l2_threshold
        )
        row = {
            "record_type": "fused_rs_fp8_gather_mxu_probe",
            "case": case,
            "status": "ok" if contract_ok else "correctness_failed",
            "contract_ok": bool(contract_ok),
            "dtype": jnp.dtype(dtype).name,
            "index_mode": index_mode,
            "size_m": _SIZE_M,
            "tile_m": _TILE_M,
            "num_rows": _NUM_ROWS,
            "k": _K,
            "n": _N,
            "m_start": m_start,
            "rel_l2_threshold": args.rel_l2_threshold,
            "comparison": comparison,
        }
        encoded = json.dumps(row, sort_keys=True)
        print(encoded, flush=True)
        with args.jsonl.open("a", encoding="utf-8") as output_file:
            output_file.write(encoded + "\n")


if __name__ == "__main__":
    main()
