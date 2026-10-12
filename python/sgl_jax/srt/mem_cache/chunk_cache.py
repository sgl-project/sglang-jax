from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from sgl_jax.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sgl_jax.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    MatchPrefixParams,
    MatchResult,
)
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

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        return MatchResult(
            device_indices=np.empty((0,), dtype=np.int32),
            last_device_node=None,
            last_host_node=None,
            best_match_node=None,
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

    def evict(self, params: EvictParams) -> EvictResult:
        return EvictResult()

    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        return IncLockRefResult(delta=0)

    def dec_lock_ref(self, node: Any, params: DecLockRefParams | None = None):
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


class DeepseekV4ChunkCache(SWAChunkCache):
    """Own V4 history and SWA pages for one live request, without prefix reuse.

    R calls the SWA reclamation helper only after the submitted forward has
    finished. Releasing a request drops its ownership; the next zero-prefix
    forward must initialize any recycled compressor-state slot before use.
    """

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        result = super().match_prefix(params)
        req = params.req
        if req is not None and req.req_pool_idx is not None:
            result = result._replace(
                device_indices=self.req_to_token_pool.read(req.req_pool_idx, req.kv_committed_len)
            )
        return result

    def cache_unfinished_req(self, req: Req):
        req.prefix_indices = self.req_to_token_pool.read(req.req_pool_idx, req.kv_committed_len)

    def release_req(self, req: Req) -> None:
        """Release the committed prefix and allocated tail as one page extent."""
        if req.req_pool_idx is None:
            return
        slot = req.req_pool_idx
        indices = self.req_to_token_pool.read(slot, req.kv_allocated_len)
        self.token_to_kv_pool_allocator.free(
            indices[indices != 0], req.dp_rank if req.dp_rank is not None else 0
        )
        self.req_to_token_pool.req_to_token[slot].fill(0)
        req.prefix_indices = np.empty(0, dtype=np.int32)
        req.last_node = req.last_host_node = None
        req.cache_protected_len = 0
        req.kv_committed_freed = req.kv_overallocated_freed = True
        req.kv_committed_len = req.kv_allocated_len = 0
        req.swa_evicted_seqlen = 0
        self.req_to_token_pool.free(req)
