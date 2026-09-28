"""Platform-owned launch schedules for HCA."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace


def _align(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


@dataclass(frozen=True)
class HCAKernelSchedule:
    """Static launch geometry selected outside the mathematical kernels."""

    platform: str
    mxu_lanes: int
    sublanes: int
    projection_k_tile: int
    projection_batch_tile_max: int
    prefill_entries_per_step: int
    cache_write_tile: int
    boundary_small_tile: int
    boundary_large_tile: int
    query_block_size: int
    query_compute_block_size: int
    swa_dma_tile: int
    swa_compute_tile: int
    compressed_tile: int
    stream_rows: int
    vmem_budget_bytes: int

    def projection_tile_k(self, hidden: int) -> int:
        """Largest MXU-aligned K tile dividing the full projection width."""
        if hidden <= 0 or hidden % self.mxu_lanes:
            raise ValueError(f"hidden={hidden} must be a positive multiple of {self.mxu_lanes}")
        tile = min(self.projection_k_tile, hidden)
        while hidden % tile:
            tile -= self.mxu_lanes
        return tile

    @property
    def row_write_run(self) -> int:
        # A full BF16 tile spans two 32-bit sublane groups.
        return self.sublanes * (4 // 2)

    def boundary_rows(self, ratio: int, head_dim: int) -> int:
        # Two FP32 state slabs, pooled KV/scores and softmax work over both slabs.
        slabs = 2 * ratio * 2 * head_dim * 4
        pooling = 2 * ratio * head_dim * 4 * 3  # KV, scores, probabilities
        row_bytes = slabs + pooling + head_dim * (4 + 2) + self.mxu_lanes * 4
        return self._fit_rows(self.boundary_large_tile, self.vmem_budget_bytes // row_bytes)

    @staticmethod
    def _fit_rows(limit: int, capacity: int) -> int:
        """Use power-of-two launch buckets within the capacity and measured ceiling."""
        if capacity < 1:
            raise ValueError("one HCA work row exceeds the explicit VMEM budget")
        return 1 << (min(limit, capacity).bit_length() - 1)

    def stream_rows_per_step(
        self,
        tokens: int,
        head_dim: int,
        itemsize: int,
        heads: int,
        window_rows: int,
        page_rows: int,
    ) -> int:
        """Fit explicit decode buffers plus score/probability workspace."""
        heads = _align(heads, self.sublanes)
        kv_buffers = 2 * self.compressed_tile * head_dim * itemsize
        query_output = 2 * heads * head_dim * itemsize
        window = window_rows * head_dim * itemsize
        accumulator = heads * head_dim * 4
        softmax_state = 2 * heads * self.mxu_lanes * 4
        window_mask = self.sublanes * self.mxu_lanes * 4
        scores_probs = 2 * heads * self.compressed_tile * 4
        per_row = kv_buffers + query_output + window + accumulator + softmax_state
        per_row += window_mask + scores_probs
        fixed = page_rows * head_dim * itemsize + heads * 4
        return self._fit_rows(
            min(self.stream_rows, tokens), (self.vmem_budget_bytes - fixed) // per_row
        )

    def __post_init__(self) -> None:
        values = tuple(value for name, value in self.__dict__.items() if name != "platform")
        if not self.platform or any(value <= 0 for value in values):
            raise ValueError("HCA schedule fields must be positive")
        if self.projection_k_tile % self.mxu_lanes:
            raise ValueError("projection_k_tile must be MXU aligned")
        if self.projection_batch_tile_max % self.sublanes:
            raise ValueError("projection_batch_tile_max must be sublane aligned")
        if self.query_block_size % self.query_compute_block_size:
            raise ValueError("query_block_size must contain whole compute tiles")
        if self.swa_dma_tile != 2 * self.swa_compute_tile:
            raise ValueError("one SWA DMA tile must contain two compute tiles")


@dataclass(frozen=True)
class _HCAPlatformParameters:
    name: str
    device_markers: tuple[str, ...]
    mxu_lanes: int
    sublanes: int
    vmem_bytes: int
    projection_k_tile: int
    projection_batch_tile_max: int
    prefill_entries_per_step: int
    cache_write_tile: int
    boundary_tiles: tuple[int, int]
    query_block_size: int
    query_compute_block_size: int
    swa_dma_tile: int
    swa_compute_tile: int
    compressed_tiles: tuple[int, ...]
    stream_rows: int


# Geometry and scoped compiler budgets are platform constraints. Launch ceilings
# are measured schedules; capacity formulas below clamp them, not predict latency.
_PLATFORMS = (
    _HCAPlatformParameters(
        name="TPU v6e",
        device_markers=("v6e", "v6 lite", "tpu v6"),
        mxu_lanes=128,
        sublanes=8,
        vmem_bytes=32 * 1024 * 1024,  # Scoped compiler allocation budget, not physical capacity.
        projection_k_tile=2048,
        projection_batch_tile_max=128,
        prefill_entries_per_step=8,
        cache_write_tile=8,
        boundary_tiles=(4, 8),
        query_block_size=32,
        query_compute_block_size=16,
        swa_dma_tile=512,
        swa_compute_tile=256,
        compressed_tiles=(128, 256, 512, 1024, 2048),
        stream_rows=8,
    ),
)

# v7x uses the same 32 MiB scoped allocation as v6e and keeps the v6e tiles except
# for the query block: 128 queries per grid step measured 8K single-request prefill
# TTFT 262 -> 257 ms on v7x against 32 (the per-step q/SWA/output DMAs and the
# accumulator init amortise over four times the queries). v6e keeps 32 until
# measured there.
_PLATFORMS += (
    replace(
        _PLATFORMS[0],
        name="TPU v7x",
        device_markers=("tpu7x", "v7x", "tpu v7"),
        query_block_size=128,
    ),
)


def _platform_parameters(device_kind: str) -> _HCAPlatformParameters:
    normalized = device_kind.strip().lower()
    for platform in _PLATFORMS:
        if any(marker in normalized for marker in platform.device_markers):
            return platform
    supported = ", ".join(platform.name for platform in _PLATFORMS)
    raise ValueError(f"HCA has no schedule for {device_kind!r}; supported: {supported}")


def get_hca_kernel_schedule(
    device_kind: str,
    *,
    page_size: int,
    max_compressed_entries: int,
    local_heads: int,
    head_dim: int,
) -> HCAKernelSchedule:
    """Select one static schedule from platform and compiled-shape metadata."""
    if min(page_size, max_compressed_entries, local_heads, head_dim) <= 0:
        raise ValueError("HCA schedule shape fields must be positive")
    platform = _platform_parameters(device_kind)
    if head_dim % platform.mxu_lanes:
        raise ValueError(f"head_dim={head_dim} must be aligned to {platform.mxu_lanes}")
    # Queries per grid step: the platform table's value unless DSV4_HCA_QUERY_BLOCK
    # overrides it.
    query_block_size = int(os.environ.get("DSV4_HCA_QUERY_BLOCK", platform.query_block_size))
    if query_block_size <= 0 or query_block_size % platform.query_compute_block_size:
        raise ValueError(
            f"DSV4_HCA_QUERY_BLOCK={query_block_size} must be a positive multiple of "
            f"{platform.query_compute_block_size}"
        )

    def vmem_bytes(compressed_tile: int, query_compute: int) -> int:
        """Peak VMEM of one chunk-attention program, in bytes.

        Account for explicit buffers and a score tile; compiler scratch is not
        included. Heads are padded to match the kernel's sublane layout.
        """
        heads = _align(local_heads, platform.sublanes)
        rows = query_block_size * heads
        q_buffers = 2 * rows * head_dim * 2  # double-buffered across grid steps
        output_staging = rows * head_dim * 2
        accumulators = rows * head_dim * 4
        online_softmax = 2 * rows * platform.mxu_lanes * 4
        swa_buffers = 2 * platform.swa_dma_tile * 2 * head_dim  # BF16, double buffered
        compressed_buffers = compressed_tile * head_dim * 2  # single buffer
        # Scores exist one segment at a time, so the wider tile sets the peak.
        score_tile = query_compute * heads * max(compressed_tile, platform.swa_compute_tile) * 4
        return (
            q_buffers
            + output_staging
            + accumulators
            + online_softmax
            + swa_buffers
            + compressed_buffers
            + score_tile
            + min(page_size, 2) * head_dim * 2  # small-page DMA scratch
        )

    compatible = tuple(
        tile
        for tile in platform.compressed_tiles
        if tile % page_size == 0 and vmem_bytes(tile, 1) <= platform.vmem_bytes
    )
    if not compatible:
        raise ValueError(
            f"HCA page_size={page_size} is incompatible with {platform.name} compressed tiles"
        )
    compressed_tile = next(
        (tile for tile in compatible if tile >= max_compressed_entries),
        compatible[-1],
    )
    query_compute = platform.query_compute_block_size
    while query_compute > 1 and vmem_bytes(compressed_tile, query_compute) > platform.vmem_bytes:
        query_compute //= 2
    stream_rows = int(os.environ.get("DSV4_HCA_STREAM_ROWS", platform.stream_rows))
    if stream_rows < 1:
        raise ValueError("DSV4_HCA_STREAM_ROWS must be positive")
    small_boundary, large_boundary = platform.boundary_tiles
    return HCAKernelSchedule(
        platform=platform.name,
        mxu_lanes=platform.mxu_lanes,
        sublanes=platform.sublanes,
        projection_k_tile=platform.projection_k_tile,
        projection_batch_tile_max=platform.projection_batch_tile_max,
        prefill_entries_per_step=platform.prefill_entries_per_step,
        cache_write_tile=platform.cache_write_tile,
        boundary_small_tile=small_boundary,
        boundary_large_tile=large_boundary,
        query_block_size=query_block_size,
        query_compute_block_size=query_compute,
        swa_dma_tile=platform.swa_dma_tile,
        swa_compute_tile=platform.swa_compute_tile,
        compressed_tile=compressed_tile,
        stream_rows=stream_rows,
        vmem_budget_bytes=platform.vmem_bytes,
    )


__all__ = ["HCAKernelSchedule", "get_hca_kernel_schedule"]
