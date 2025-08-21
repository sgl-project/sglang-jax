from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import jax
from jax import numpy as jnp

from sgl_jax.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import Req


class ChunkCache(BasePrefixCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.cpu_device = jax.local_devices(backend="cpu")[0]

    def reset(self):
        pass

    def match_prefix(self, **unused_kwargs) -> MatchResult:
        return MatchResult(
            device_indices=np.empty((0,), dtype=np.int32),
            last_device_node=None,
            last_host_node=None,
        )

    def cache_finished_req(self, req: Req):
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx,
            # For decode server: if req.output_ids is empty, we want to free all req.origin_input_ids
            : len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0),
        ]
        self.req_to_token_pool.free(req.req_pool_idx)
        self.token_to_kv_pool_allocator.free(kv_indices)

    def cache_unfinished_req(self, req: Req):
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(req.fill_ids)
        ]

        req.prefix_indices = jax.device_put(kv_indices, self.cpu_device)

    def evict(self, num_tokens: int):
        pass

    def inc_lock_ref(self, node: Any):
        return 0

    def dec_lock_ref(self, node: Any, swa_uuid_for_lock: Optional[str] = None):
        return 0

    def pretty_print(self):
        return ""
