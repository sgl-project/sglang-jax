from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sgl_jax.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool


@dataclasses.dataclass
class CacheInitParams:
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    page_size: int

    is_eagle: bool = False

    sliding_window_size: int | None = None

    # Recurrent radix: page-aligned ping-pong track buffer for page>=128 (PR#2).
    enable_recurrent_extra_buffer: bool = False

    # Track-boundary interval (multiple of page_size) for extra-buffer snapshots;
    # resolved to page_size by ServerArgs when extra-buffer is on.
    recurrent_track_interval: int | None = None
