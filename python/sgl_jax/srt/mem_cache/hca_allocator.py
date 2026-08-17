"""Host-side request and page allocation for HCA's two KV tiers."""

from __future__ import annotations

import math

import numpy as np
from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool


def _align(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


class HCAKVPoolAllocator:
    """Coordinate HCA request slots, fixed SWA rings, and growing compressed pages."""

    def __init__(
        self,
        token_pool,
        request_pool: HybridReqToTokenPool,
    ):
        if not isinstance(request_pool, HybridReqToTokenPool):
            raise TypeError("HCA requires SGLang's HybridReqToTokenPool")
        if request_pool.size != token_pool.max_num_requests:
            raise ValueError("request and HCA KV pools must have the same capacity")
        if request_pool.max_context_len != token_pool.max_context_len:
            raise ValueError("request and HCA KV pools must have the same context limit")
        if request_pool.dp_size != token_pool.dp_size:
            raise ValueError("request and HCA KV pools must have the same dp_size")
        if request_pool.enable_recurrent_extra_buffer:
            raise ValueError("HCA does not use the linear-attention ping-pong buffer")

        self.token_pool = token_pool
        self.request_pool = request_pool
        self.page_size = token_pool.page_size
        self.dp_size = token_pool.dp_size
        self.window_allocator = token_pool.create_window_allocator()
        self.compressed_allocator = token_pool.create_compressed_allocator()
        self.window_slots = np.zeros((request_pool.size, token_pool.window_size), dtype=np.int32)
        max_entries = math.ceil(token_pool.max_context_len / token_pool.compress_ratio)
        self.max_entries = _align(max_entries, self.page_size)
        self.compressed_slots = np.zeros((request_pool.size, self.max_entries), dtype=np.int32)
        self.compressed_capacity = np.zeros((request_pool.size,), dtype=np.int32)
        self.request_dp_ranks = np.zeros((request_pool.size,), dtype=np.int32)

    def _global_state_slot(self, local_slot: int, dp_rank: int) -> int:
        return dp_rank * (self.request_pool.slots_per_rank + 1) + int(local_slot)

    def alloc(self, requests):
        """Allocate request, recurrent, and fixed-window resources atomically."""
        if not requests:
            return []
        dp_rank = requests[0].dp_rank if requests[0].dp_rank is not None else 0
        if any((req.dp_rank or 0) != dp_rank for req in requests):
            raise ValueError("one HCA allocation call must contain one DP rank")
        new_requests = [req for req in requests if req.req_pool_idx is None]
        window_tokens = len(new_requests) * self.token_pool.window_size
        if self.window_allocator.available_size(dp_rank) < window_tokens:
            return None

        result = self.request_pool.alloc(requests)
        if result is None:
            return None
        allocated = []
        try:
            for request in new_requests:
                slots = self.window_allocator.alloc(self.token_pool.window_size, dp_rank=dp_rank)
                if slots is None:
                    raise RuntimeError("HCA window allocation failed after preflight")
                req_index = int(request.req_pool_idx)
                self.window_slots[req_index] = slots
                self.request_dp_ranks[req_index] = dp_rank
                allocated.append(request)

            global_slots = [
                self._global_state_slot(request.recurrent_pool_idx, dp_rank)
                for request in new_requests
            ]
            if global_slots:
                self.request_pool.recurrent_state_pool.reset_slots(
                    np.asarray(global_slots, np.int32)
                )
            return result
        except Exception:
            for request in allocated:
                req_index = int(request.req_pool_idx)
                self.window_allocator.free(self.window_slots[req_index], dp_rank=dp_rank)
                self.window_slots[req_index].fill(0)
            for request in new_requests:
                if request.req_pool_idx is not None:
                    self.request_pool.free(request)
            raise

    def ensure_compressed_capacity(self, req_pool_indices, seq_lens) -> None:
        """Grow each request's compressed tier to its completed-record count."""
        req_pool_indices = np.asarray(req_pool_indices, np.int32)
        seq_lens = np.asarray(seq_lens, np.int32)
        desired = np.floor_divide(seq_lens, self.token_pool.compress_ratio)
        if np.any(desired > self.max_entries):
            raise ValueError("sequence exceeds HCA max_context_len")
        aligned = np.asarray(
            [_align(int(value), self.page_size) if value else 0 for value in desired],
            np.int32,
        )

        needed_by_rank = np.zeros((self.dp_size,), np.int64)
        for req_index, new_capacity in zip(req_pool_indices, aligned):
            old_capacity = self.compressed_capacity[req_index]
            if new_capacity > old_capacity:
                needed_by_rank[self.request_dp_ranks[req_index]] += new_capacity - old_capacity
        for rank, needed in enumerate(needed_by_rank):
            if needed > self.compressed_allocator.available_size(rank):
                raise RuntimeError(f"HCA compressed cache exhausted on DP rank {rank}")

        for req_index, new_capacity in zip(req_pool_indices, aligned):
            old_capacity = int(self.compressed_capacity[req_index])
            if new_capacity <= old_capacity:
                continue
            rank = int(self.request_dp_ranks[req_index])
            slots = self.compressed_allocator.alloc(int(new_capacity) - old_capacity, dp_rank=rank)
            if slots is None:
                raise RuntimeError(f"HCA compressed cache exhausted on DP rank {rank}")
            self.compressed_slots[req_index, old_capacity:new_capacity] = slots
            self.compressed_capacity[req_index] = new_capacity

    def free(self, request) -> None:
        """Release both KV tiers before returning the generic request slot."""
        if request.req_pool_idx is None:
            raise ValueError("request is not allocated")
        req_index = int(request.req_pool_idx)
        dp_rank = request.dp_rank if request.dp_rank is not None else 0
        window = self.window_slots[req_index]
        if np.any(window):
            self.window_allocator.free(window, dp_rank=dp_rank)
            window.fill(0)
        capacity = int(self.compressed_capacity[req_index])
        if capacity:
            self.compressed_allocator.free(
                self.compressed_slots[req_index, :capacity], dp_rank=dp_rank
            )
            self.compressed_slots[req_index].fill(0)
            self.compressed_capacity[req_index] = 0
        self.request_dp_ranks[req_index] = 0
        self.request_pool.free(request)

    def clear(self) -> None:
        self.request_pool.clear()
        self.window_allocator.clear()
        self.compressed_allocator.clear()
        self.window_slots.fill(0)
        self.compressed_slots.fill(0)
        self.compressed_capacity.fill(0)
        self.request_dp_ranks.fill(0)

    def page_tables(self, req_pool_indices, seq_lens):
        """Flattened logical-to-physical tables for both KV tiers of one batch.

        ``ensure_compressed_capacity`` must already cover ``seq_lens``: reading
        an unallocated compressed page would silently resolve to dummy page 0.
        """
        req_pool_indices = np.asarray(req_pool_indices, np.int32)
        seq_lens = np.asarray(seq_lens, np.int32)
        desired = np.floor_divide(seq_lens, self.token_pool.compress_ratio)
        if np.any(desired > self.compressed_capacity[req_pool_indices]):
            raise RuntimeError("ensure_compressed_capacity must run before page_tables")
        page_size = self.page_size
        window_pages = []
        window_cu = [0]
        physical_window_pages = self.window_slots[req_pool_indices, ::page_size] // page_size
        for seq_len, physical_pages in zip(seq_lens, physical_window_pages):
            logical_pages = max(1, math.ceil(int(seq_len) / page_size))
            window_pages.extend(
                physical_pages[np.arange(logical_pages, dtype=np.int32) % physical_pages.size]
            )
            window_cu.append(window_cu[-1] + logical_pages * page_size)

        compressed_lens = np.floor_divide(seq_lens, self.token_pool.compress_ratio).astype(np.int32)
        compressed_pages = []
        compressed_cu = [0]
        for req_index, logical_len in zip(req_pool_indices, compressed_lens):
            logical_pages = max(1, math.ceil(int(logical_len) / page_size))
            if logical_len:
                pages = (
                    self.compressed_slots[req_index, : logical_pages * page_size : page_size]
                    // page_size
                )
            else:
                pages = np.zeros((1,), np.int32)
            compressed_pages.extend(pages)
            compressed_cu.append(compressed_cu[-1] + logical_pages * page_size)
        return (
            np.asarray(window_pages, np.int32),
            np.asarray(window_cu, np.int32),
            np.asarray(compressed_pages, np.int32),
            np.asarray(compressed_cu, np.int32),
            compressed_lens,
        )


__all__ = ["HCAKVPoolAllocator"]
