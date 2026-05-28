from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from sgl_jax.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sgl_jax.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import Req


class ChunkCache(BasePrefixCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = page_size

    def reset(self):
        pass

    def match_prefix(self, **unused_kwargs) -> MatchResult:
        return MatchResult(
            device_indices=np.empty((0,), dtype=np.int32),
            last_device_node=None,
            last_host_node=None,
        )

    def cache_finished_req(self, req: Req, is_insert: bool = True):
        # is_insert is unused (no prefix tree); kept for signature parity.
        committed_kv_len = req.pop_committed_kv_cache()
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx,
            :committed_kv_len,
        ]
        self.token_to_kv_pool_allocator.free(
            kv_indices, req.dp_rank if req.dp_rank is not None else 0
        )

    def cache_unfinished_req(self, req: Req):
        req.prefix_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(req.fill_ids)
        ].copy()

    def evict(
        self,
        num_tokens: int,
        swa_num_tokens: int = 0,
        dp_rank: int | None = None,
    ):
        pass

    def inc_lock_ref(self, node: Any):
        return 0

    def dec_lock_ref(self, node: Any, swa_uuid_for_lock: str | None = None):
        return 0

    def pretty_print(self):
        return ""


class SWAChunkCache(ChunkCache):
    """ChunkCache with support for sliding window attention.

    Used when disable_radix_cache=True and the model is a hybrid SWA model.
    """

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: SWATokenToKVPoolAllocator,
        page_size: int,
        sliding_window_size: int,
    ):
        super().__init__(req_to_token_pool, token_to_kv_pool_allocator, page_size)
        self.sliding_window_size = sliding_window_size

    def supports_swa(self) -> bool:
        return True

    def full_evictable_size(self, dp_rank: int = 0) -> int:
        return 0

    def swa_evictable_size(self, dp_rank: int = 0) -> int:
        return 0

    def full_protected_size(self, dp_rank: int = 0) -> int:
        return 0

    def swa_protected_size(self, dp_rank: int = 0) -> int:
        return 0
