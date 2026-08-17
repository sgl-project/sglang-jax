"""Platform-owned launch schedules for HCA."""

from __future__ import annotations

from dataclasses import dataclass


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


# Values in this table are hardware schedules, not HCA model semantics. The v6e
# row is measured by benchmark/kernels/hca/bench_hca.py.
_PLATFORMS = (
    _HCAPlatformParameters(
        name="TPU v6e",
        device_markers=("v6e", "v6 lite", "tpu v6"),
        mxu_lanes=128,
        sublanes=8,
        vmem_bytes=32 * 1024 * 1024,
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

    def vmem_bytes(compressed_tile: int, query_compute: int) -> int:
        """Peak VMEM of one chunk-attention program, in bytes.

        Every buffer the kernel allocates is counted, so the budget below is a
        bound rather than an estimate.  Heads are padded to the sublane
        multiple, matching the kernel's own layout.
        """
        heads = _align(local_heads, platform.sublanes)
        rows = platform.query_block_size * heads
        q_buffers = 2 * rows * head_dim * 2  # double-buffered across grid steps
        output_staging = rows * head_dim * 2
        accumulators = rows * head_dim * 4
        online_softmax = 2 * rows * platform.mxu_lanes * 4
        swa_buffers = 2 * platform.swa_dma_tile * 2 * head_dim  # uint8, x2 buffers
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
        query_block_size=platform.query_block_size,
        query_compute_block_size=query_compute,
        swa_dma_tile=platform.swa_dma_tile,
        swa_compute_tile=platform.swa_compute_tile,
        compressed_tile=compressed_tile,
    )


__all__ = ["HCAKernelSchedule", "get_hca_kernel_schedule"]
