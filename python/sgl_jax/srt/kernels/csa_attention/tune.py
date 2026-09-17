"""Launch geometry for single-device CSA attention."""

from dataclasses import dataclass

LANES = 128
SUBLANES = 8
MXU_TILE = 256
# Measured v6e choice for 64 heads; 8/16-query tiles were slower.
MAX_QUERY_TILE = 32
# v6e measurements: two-page lookahead; 4/8 buffers added no useful benefit.
DMA_DEPTH = 3
APPEND_TILE = MXU_TILE
# Batched routing amortizes launch cost; 32-query tiles outperformed 8.
ROUTE_QUERY_TILE = 32


@dataclass(frozen=True)
class CSAAttentionSchedule:
    query_tile: int = MAX_QUERY_TILE
    # Shared-query tile; compact decode uses APPEND_TILE.
    selected_tile: int = MXU_TILE


def get_csa_attention_schedule(device_kind: str, *, decode: bool = False) -> CSAAttentionSchedule:
    if not any(marker in device_kind.lower() for marker in ("v6e", "v6 lite")):
        raise ValueError(f"CSA attention is not validated on {device_kind!r}")
    # Decode has no same-request queries to share a KV tile with.
    return CSAAttentionSchedule(query_tile=1 if decode else MAX_QUERY_TILE)
