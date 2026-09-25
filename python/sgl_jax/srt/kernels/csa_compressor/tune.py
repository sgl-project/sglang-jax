"""Compressor format constants and measured TPU v6e scheduling constraints."""

from __future__ import annotations

from dataclasses import dataclass

CSA_COMPRESSION_RATIO = 4
CSA_STATE_SLOTS = 2 * CSA_COMPRESSION_RATIO

CSA_INDEX_DIM = 128
CSA_ATTENTION_DIM = 512
CSA_ROPE_DIM = 64
CSA_ROPE_FREQUENCY_DIM = CSA_ROPE_DIM // 2
CSA_MAIN_NOPE_DIM = CSA_ATTENTION_DIM - CSA_ROPE_DIM
CSA_MAIN_PROJECTED_DIM = 2 * CSA_ATTENTION_DIM
CSA_INDEX_PROJECTED_DIM = 2 * CSA_INDEX_DIM
CSA_DUAL_PROJECTION_DIM = 2 * (CSA_MAIN_PROJECTED_DIM + CSA_INDEX_PROJECTED_DIM)

CSA_CACHE_PACKING = 4

CSA_FP8_BLOCK_SIZE = 64
CSA_FP8_AMAX_FLOOR = 1e-4
CSA_NORM_EPS = 1e-6
CSA_MAIN_NOPE_SCALE_COUNT = CSA_MAIN_NOPE_DIM // CSA_FP8_BLOCK_SIZE
CSA_MAIN_NOPE_RECORD_BYTES = CSA_ATTENTION_DIM
CSA_MAIN_NOPE_PADDING_BYTES = (
    CSA_MAIN_NOPE_RECORD_BYTES - CSA_MAIN_NOPE_DIM - CSA_MAIN_NOPE_SCALE_COUNT
)
CSA_ROPE_RECORD_BYTES = 2 * CSA_ROPE_DIM
CSA_MAIN_RECORD_BYTES = CSA_MAIN_NOPE_RECORD_BYTES + CSA_ROPE_RECORD_BYTES
CSA_INDEX_SCALE_COUNT = 1
CSA_INDEX_RECORD_BYTES = 2 * CSA_INDEX_DIM
CSA_INDEX_PADDING_BYTES = CSA_INDEX_RECORD_BYTES - CSA_INDEX_DIM - CSA_INDEX_SCALE_COUNT


@dataclass(frozen=True)
class TPULayout:
    vector_lanes: int
    sublanes: int
    uint8_row_tile: int


TPU_V6E = TPULayout(
    vector_lanes=128,
    sublanes=8,
    uint8_row_tile=32,
)

# Measured projection K tile on TPU v6e; adjusted to divide the input width.
V6E_PROJECTION_K_TILE = 2048


@dataclass(frozen=True)
class CompressorSchedule:
    projection_k_tile: int
    query_tile: int


def get_compressor_schedule(hidden: int, *, device_kind: str) -> CompressorSchedule:
    """Select the measured v6e geometry; do not silently tune unknown devices as v6e."""
    if not any(marker in device_kind.lower() for marker in ("v6e", "v6 lite", "tpu v6")):
        raise ValueError(f"CSA compressor has no calibrated schedule for {device_kind!r}")
    return CompressorSchedule(
        projection_k_tile=get_csa_compressor_projection_k_tile(hidden),
        query_tile=TPU_V6E.vector_lanes,
    )


def get_csa_compressor_projection_k_tile(hidden: int) -> int:
    if hidden <= 0 or hidden % TPU_V6E.vector_lanes:
        raise ValueError("hidden must be a positive multiple of the TPU lane count")
    # Use the same K reduction partitions in decode and prefill before FP8 rounding.
    tile = min(hidden, V6E_PROJECTION_K_TILE)
    while hidden % tile:
        tile -= TPU_V6E.vector_lanes
    return tile


def get_csa_compressor_query_tile(sequence: int) -> int:
    # A lane-width query tile bounds pooling/projection scratch independently of context.
    return min(sequence, TPU_V6E.vector_lanes)
