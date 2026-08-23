"""Target-TPU exactness probe for fused-RS irregular hidden-state DMA gathers.

This intentionally excludes routing, weights, and MXU work.  It exercises the
same row-at-a-time HBM-to-VMEM helper used by fused-RS, copies the VMEM tile back
to HBM, and compares the valid destination rows with a JAX indexed reference.
BF16 is the control; FP8 is the suspected failing input layout.
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
_NUM_LANES = 128
_K_TILES = _K // _NUM_LANES
_CASES = (
    ("bf16_contiguous_offset0", jnp.bfloat16, "contiguous", 0),
    ("bf16_irregular_offset37", jnp.bfloat16, "irregular", 37),
    ("fp8_contiguous_offset0", jnp.float8_e4m3fn, "contiguous", 0),
    ("fp8_irregular_offset0", jnp.float8_e4m3fn, "irregular", 0),
    ("fp8_irregular_offset37", jnp.float8_e4m3fn, "irregular", 37),
)


def _dma_gather_probe_kernel(
    indices_ref,
    payload_ref,
    output_ref,
    offset_ref,
    scratch_ref,
    dma_sem_ref,
    *,
    m_start: int,
):
    is_valid = jnp.arange(_TILE_M, dtype=jnp.int32) < _NUM_ROWS
    metadata = GatherMetadata(
        m_start=jnp.int32(m_start),
        m_end=jnp.int32(m_start + _NUM_ROWS),
        num_rows=jnp.int32(_NUM_ROWS),
        is_valid=is_valid,
        src_rows=[indices_ref[i] for i in range(_TILE_M)],
    )
    dma_gather_gm_start(payload_ref, scratch_ref, dma_sem_ref, metadata)
    dma_gather_gm_wait(scratch_ref, dma_sem_ref, metadata)

    # Keep the observation path register-free: one contiguous DMA copies the
    # whole VMEM tile back to HBM.  The host compares only initialized rows.
    copy_out = pltpu.make_async_copy(scratch_ref, output_ref, dma_sem_ref)
    copy_out.start()
    copy_out.wait()

    sls = pltpu.get_tpu_info().get_sublane_tiling(payload_ref.dtype)
    offset_ref[0] = jnp.int32(m_start) % sls


def _run_probe(payload: jax.Array, indices: jax.Array, *, m_start: int):
    kernel = pl.pallas_call(
        functools.partial(_dma_gather_probe_kernel, m_start=m_start),
        out_shape=(
            jax.ShapeDtypeStruct((_TILE_M, _K_TILES, _NUM_LANES), payload.dtype),
            jax.ShapeDtypeStruct((1,), jnp.int32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=(
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ),
            scratch_shapes=(
                pltpu.VMEM((_TILE_M, _K_TILES, _NUM_LANES), payload.dtype),
                pltpu.SemaphoreType.DMA,
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
        ),
        name=f"fused-rs-hidden-dma-gather-{payload.dtype.name}-offset{m_start}",
    )
    return kernel(indices, payload)


def _payload(dtype) -> jax.Array:
    rows = jnp.arange(_SIZE_M, dtype=jnp.float32)[:, None]
    cols = jnp.arange(_K, dtype=jnp.float32)[None, :]
    values = (((rows * 13 + cols * 7) % 201) - 100) / 16
    return values.astype(dtype).reshape(_SIZE_M, _K_TILES, _NUM_LANES)


def _indices(mode: str) -> jax.Array:
    rows = jnp.arange(_TILE_M, dtype=jnp.int32)
    if mode == "contiguous":
        return rows
    if mode == "irregular":
        return (rows * 17 + 3) % _SIZE_M
    raise ValueError(f"unknown index mode: {mode}")


def _compare(expected: np.ndarray, actual: np.ndarray) -> dict:
    expected_f32 = expected.astype(np.float32)
    actual_f32 = actual.astype(np.float32)
    mismatch = expected_f32 != actual_f32
    mismatch_locations = np.argwhere(mismatch)
    first_mismatch = (
        mismatch_locations[0].astype(int).tolist()
        if mismatch_locations.size
        else None
    )
    return {
        "exact": bool(np.array_equal(expected_f32, actual_f32)),
        "mismatch_count": int(np.count_nonzero(mismatch)),
        "max_abs": float(np.max(np.abs(expected_f32 - actual_f32))),
        "first_mismatch": first_mismatch,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True)
    args = parser.parse_args()
    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.jsonl.write_text("", encoding="utf-8")

    for case, dtype, index_mode, m_start in _CASES:
        payload = _payload(dtype)
        indices = _indices(index_mode)
        gathered, offset = _run_probe(payload, indices, m_start=m_start)
        jax.block_until_ready((payload, indices, gathered, offset))

        payload_host = np.asarray(jax.device_get(payload))
        indices_host = np.asarray(jax.device_get(indices), dtype=np.int32)
        gathered_host = np.asarray(jax.device_get(gathered))
        offset_host = int(np.asarray(jax.device_get(offset), dtype=np.int32)[0])
        expected = payload_host[indices_host[:_NUM_ROWS]]
        actual = gathered_host[offset_host : offset_host + _NUM_ROWS]
        comparison = _compare(expected, actual)
        row = {
            "record_type": "fused_rs_fp8_dma_gather_probe",
            "case": case,
            "status": "ok" if comparison["exact"] else "correctness_failed",
            "dtype": jnp.dtype(dtype).name,
            "index_mode": index_mode,
            "size_m": _SIZE_M,
            "tile_m": _TILE_M,
            "num_rows": _NUM_ROWS,
            "k": _K,
            "m_start": m_start,
            "m_start_local": offset_host,
            "comparison": comparison,
        }
        encoded = json.dumps(row, sort_keys=True)
        print(encoded, flush=True)
        with args.jsonl.open("a", encoding="utf-8") as output:
            output.write(encoded + "\n")


if __name__ == "__main__":
    main()
