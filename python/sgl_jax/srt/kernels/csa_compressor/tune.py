"""v6e schedule: measured query/K defaults; DMA alignment derives from compact tiling."""

from dataclasses import dataclass

from jax.experimental.pallas import tpu as pltpu


@dataclass(frozen=True)
class CompressorSchedule:
    projection_k_tile: int
    query_tile: int = 128  # Measured default, not a VMEM capacity limit.

    def decode_request_tile(self, requests: int) -> int:
        return min(requests, pltpu.Tiling.COMPACT.shape[0])

    def token_tile(self, tokens: int) -> int:
        sublanes = pltpu.Tiling.COMPACT.shape[0]
        return min(self.query_tile, max(sublanes, ((tokens + sublanes - 1) // sublanes) * sublanes))

    def input_geometry(self, tokens: int, sequence: int, element_bytes: int):
        rows = self.token_tile(sequence)
        # DMA alignment includes the sub-32-bit packing in each vector register.
        alignment = pltpu.Tiling.COMPACT.shape[0] * (4 // element_bytes)
        loaded = ((rows + 2 * alignment - 2) // alignment) * alignment
        padded = max(loaded, ((tokens + alignment - 1) // alignment) * alignment)
        return rows, alignment, loaded, padded

    def cache_write_run(self, records: int, element_bytes: int) -> int:
        # One metadata DMA lane row; the shared writer verifies physical contiguity.
        sublanes, lanes = pltpu.Tiling.COMPACT.shape
        alignment = sublanes * (4 // element_bytes)
        return min(lanes, ((records + alignment - 1) // alignment) * alignment)


def get_compressor_schedule(hidden: int, *, device_kind: str) -> CompressorSchedule:
    if not any(s in device_kind.lower() for s in ("v6e", "v6 lite")):
        raise ValueError(f"No calibrated compressor schedule for {device_kind!r}")
    if hidden <= 0 or hidden % 128:
        raise ValueError("hidden must be a positive multiple of 128")
    # Measured K ceiling; choose a lane-aligned divisor, not an estimated capacity limit.
    tile = min(hidden, 2048)
    while hidden % tile:
        tile -= 128
    return CompressorSchedule(tile)
