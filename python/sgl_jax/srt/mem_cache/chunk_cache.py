from __future__ import annotations

from collections import defaultdict
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
    build_swa_cache_ledger_snapshot,
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
        self.dec_lock_ref(getattr(req, "last_node", None), getattr(req, "cache_lock_params", None))
        req.cache_lock_params = None
        req.swa_uuid_for_lock = None

    def cache_unfinished_req(self, req: Req):
        req.prefix_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(req.fill_ids)
        ].copy()
        self.dec_lock_ref(getattr(req, "last_node", None), getattr(req, "cache_lock_params", None))
        lock_result = self.inc_lock_ref(getattr(req, "last_node", None))
        req.cache_lock_params = lock_result.to_dec_params()
        req.swa_uuid_for_lock = req.cache_lock_params.swa_uuid_for_lock

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
        self._ledger_event_totals = defaultdict(lambda: defaultdict(int))

    def supports_swa(self) -> bool:
        return True

    def evict_req_swa(self, req: Req, pre_len: int, dp_rank: int = 0) -> None:
        new_evicted = max(
            req.swa_evicted_seqlen,
            pre_len - self.sliding_window_size - self.page_size,
        )
        if self.page_size > 1:
            new_evicted = new_evicted // self.page_size * self.page_size
        if new_evicted <= req.swa_evicted_seqlen:
            return
        slots = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, req.swa_evicted_seqlen : new_evicted
        ]
        freed = self.token_to_kv_pool_allocator.count_swa_mapped(slots, dp_rank=dp_rank)
        self.token_to_kv_pool_allocator.free_swa(slots, dp_rank=dp_rank)
        self._ledger_event_totals[dp_rank]["swa_evicted_total"] += freed
        req.swa_evicted_seqlen = new_evicted

    def cache_ledger_snapshot(self, dp_rank: int, live_reqs):
        full_occurrences: list[int] = []
        swa_occurrences: list[int] = []
        mapping_pair_full_sources: list[int] = []
        mapping_pair_swa_destinations: list[int] = []
        for req in live_reqs or ():
            if (req.dp_rank or 0) != dp_rank or req.req_pool_idx is None:
                continue
            start = max(0, getattr(req, "cache_protected_len", 0))
            end = max(start, getattr(req, "kv_allocated_len", 0))
            row = self.req_to_token_pool.req_to_token[req.req_pool_idx, start:end]
            full_occurrences.extend(int(index) for index in row if int(index) != 0)
            swa_start = max(start, getattr(req, "swa_evicted_seqlen", 0))
            if swa_start < end:
                swa_row = self.req_to_token_pool.req_to_token[req.req_pool_idx, swa_start:end]
                mapped = self.token_to_kv_pool_allocator.translate_full_to_swa(
                    swa_row, dp_rank=dp_rank, require_mapped=False
                )
                for source, destination in zip(swa_row, mapped):
                    if int(destination) == 0:
                        continue
                    swa_occurrences.append(int(destination))
                    mapping_pair_full_sources.append(int(source))
                    mapping_pair_swa_destinations.append(int(destination))
        return build_swa_cache_ledger_snapshot(
            dp_rank=dp_rank,
            allocator=self.token_to_kv_pool_allocator,
            full_tree_evictable=[],
            full_tree_protected=[],
            swa_tree_evictable=[],
            swa_tree_protected=[],
            full_request_occurrences=full_occurrences,
            swa_request_occurrences=swa_occurrences,
            mapping_pair_full_sources=mapping_pair_full_sources,
            mapping_pair_swa_destinations=mapping_pair_swa_destinations,
            event_totals=self._ledger_event_totals.get(dp_rank, {}),
        )

    def full_evictable_size(self, dp_rank: int = 0) -> int:
        return 0

    def swa_evictable_size(self, dp_rank: int = 0) -> int:
        return 0

    def full_protected_size(self, dp_rank: int = 0) -> int:
        return 0

    def swa_protected_size(self, dp_rank: int = 0) -> int:
        return 0
