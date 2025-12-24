import abc
import logging

import numpy as np

from sgl_jax.srt.mem_cache.memory_pool import KVCache, SWAKVPool

logger = logging.getLogger(__name__)


class BaseTokenToKVPoolAllocator(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        kvcache: KVCache,
        dp_size: int = 1,
    ):
        self.size = size
        self.page_size = page_size
        self.dp_size = dp_size
        self.size_per_rank = size // dp_size
        # self.dtype = dtype
        self._kvcache = kvcache

        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []

    def debug_print(self) -> str:
        return ""

    def available_size(self, dp_rank: int = 0) -> int:
        return (len(self.free_pages) + len(self.release_pages)) * self.page_size

    def get_kvcache(self) -> KVCache:
        return self._kvcache

    def restore_state(self, state):
        self.free_pages, self.release_pages = state

    def backup_state(self):
        return (self.free_pages, self.release_pages)

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            all_free_indices = np.concatenate(self.free_group)
            self.free(all_free_indices)

    def merge_and_sort_free(self, dp_rank: int = 0):
        if self.dp_size == 1:
            if len(self.release_pages) > 0:
                combined = np.concatenate((self.free_pages, self.release_pages))
                self.free_pages = np.sort(combined)  # No duplicates, just sort
                self.release_pages = np.empty((0,), dtype=np.int32)
        else:
            release_pages = self.release_pages_per_rank[dp_rank]
            if len(release_pages) > 0:
                combined = np.concatenate((self.free_pages_per_rank[dp_rank], release_pages))
                self.free_pages_per_rank[dp_rank] = np.sort(combined)
                self.release_pages_per_rank[dp_rank] = np.empty((0,), dtype=np.int32)

    def get_cpu_copy(self, *args, **kwargs):
        # JAX equivalent would be device_get
        raise NotImplementedError("get_cpu_copy not implemented for JAX")

    def load_cpu_copy(self, *args, **kwargs):
        # JAX equivalent would be device_put
        raise NotImplementedError("load_cpu_copy not implemented for JAX")

    def alloc_extend(self, *args, **kwargs):
        raise NotImplementedError("alloc_extend is only for paged allocator")

    def alloc_decode(self, *args, **kwargs):
        raise NotImplementedError("alloc_decode is only for paged allocator")

    @abc.abstractmethod
    def clear(self, dp_rank: int | None = None):
        raise NotImplementedError()

    @abc.abstractmethod
    def alloc(self, need_size: int, dp_rank: int = 0) -> np.ndarray | None:
        raise NotImplementedError()

    @abc.abstractmethod
    def free(self, free_index: np.ndarray, dp_rank: int = 0):
        raise NotImplementedError()


class TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    def __init__(
        self,
        size: int,
        kvcache: KVCache,
        dp_size: int = 1,
    ):
        # super().__init__(size, 1, dtype, kvcache)  # page_size=1 for token-level
        super().__init__(size, 1, kvcache, dp_size)  # page_size=1 for token-level

        if dp_size == 1:
            # Single rank: initialize state and bind single-rank methods
            self.free_slots = np.arange(1, self.size + 1, dtype=np.int32)
            self.origin_size = len(self.free_slots)
            self.is_not_in_free_group = True
            self.free_group = []

            # Dynamic method binding (avoid runtime if-else)
            self.alloc = self._alloc_single
            self.free = self._free_single
            self.available_size = self._available_size_single
            self.clear = self._clear_single
        else:
            # Multi-rank: initialize per-rank state with LOCAL index spaces
            # Each rank has independent [1, size_per_rank] indices (shard_map local view)
            self.free_slots_per_rank = {}

            for rank in range(dp_size):
                # Each rank: [1, 2, 3, ..., size_per_rank]
                self.free_slots_per_rank[rank] = np.arange(
                    1, self.size_per_rank + 1, dtype=np.int32
                )

            self.is_not_in_free_group = True
            self.free_group = {}

            # Dynamic method binding for multi-rank
            self.alloc = self._alloc_multi
            self.free = self._free_multi
            self.available_size = self._available_size_multi
            self.clear = self._clear_multi

    # Stub implementations to satisfy ABC requirements
    # These will be overridden by dynamic binding in __init__
    def alloc(self, need_size: int, dp_rank: int = 0) -> np.ndarray | None:
        raise NotImplementedError("alloc should be bound dynamically in __init__")

    def free(self, free_index: np.ndarray, dp_rank: int = 0):
        raise NotImplementedError("free should be bound dynamically in __init__")

    def clear(self, dp_rank: int | None = None):
        raise NotImplementedError("clear should be bound dynamically in __init__")

    # Single-rank implementation (bound when dp_size == 1)
    def _alloc_single(self, need_size: int, dp_rank: int = 0) -> np.ndarray | None:
        """Single-rank allocation (no DP overhead)."""
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size].copy()
        self.free_slots = self.free_slots[need_size:]
        return select_index

    # Multi-rank implementation (bound when dp_size > 1)
    def _alloc_multi(self, need_size: int, dp_rank: int = 0) -> np.ndarray | None:
        """Multi-rank allocation with local index spaces."""
        free_slots = self.free_slots_per_rank[dp_rank]
        if need_size > len(free_slots):
            return None
        select_index = free_slots[:need_size].copy()
        self.free_slots_per_rank[dp_rank] = free_slots[need_size:]
        return select_index

    # Single-rank implementation
    def _free_single(self, free_index: np.ndarray, dp_rank: int = 0):
        """Single-rank free (no DP overhead)."""
        if free_index.size == 0:
            return
        if self.is_not_in_free_group:
            self.free_slots = np.concatenate([self.free_slots, np.array(free_index)])
        else:
            self.free_group.append(np.array(free_index))

    # Multi-rank implementation
    def _free_multi(self, free_index: np.ndarray, dp_rank: int = 0):
        """Multi-rank free with per-rank free lists."""
        if free_index.size == 0:
            return
        if self.is_not_in_free_group:
            self.free_slots_per_rank[dp_rank] = np.concatenate(
                [self.free_slots_per_rank[dp_rank], np.array(free_index)]
            )
        else:
            if dp_rank not in self.free_group:
                self.free_group[dp_rank] = []
            self.free_group[dp_rank].append(np.array(free_index))

    # Single-rank implementation
    def _available_size_single(self, dp_rank: int = 0) -> int:
        """Single-rank available size."""
        return len(self.free_slots)

    # Multi-rank implementation
    def _available_size_multi(self, dp_rank: int = 0) -> int:
        """Multi-rank available size per rank."""
        return len(self.free_slots_per_rank[dp_rank])

    # Single-rank implementation
    def _clear_single(self, dp_rank: int | None = None):
        """Single-rank clear."""
        self.free_slots = np.arange(1, self.size + 1, dtype=np.int32)
        self.origin_size = len(self.free_slots)
        self.is_not_in_free_group = True
        self.free_group = []

    # Multi-rank implementation
    def _clear_multi(self, dp_rank: int | None = None):
        """Multi-rank clear (specific rank or all ranks)."""
        if dp_rank is None:
            # Clear all ranks (local index spaces)
            for rank in range(self.dp_size):
                self.free_slots_per_rank[rank] = np.arange(
                    1, self.size_per_rank + 1, dtype=np.int32
                )
        else:
            # Clear specific rank
            self.free_slots_per_rank[dp_rank] = np.arange(1, self.size_per_rank + 1, dtype=np.int32)
        self.is_not_in_free_group = True
        self.free_group = {} if self.dp_size > 1 else []

    def free_group_begin(self):
        """Begin batching free operations."""
        self.is_not_in_free_group = False
        if self.dp_size == 1:
            self.free_group = []
        else:
            self.free_group = {}

    def free_group_end(self):
        """Execute all batched free operations."""
        self.is_not_in_free_group = True

        if self.dp_size == 1:
            if self.free_group:
                self.free_slots = np.concatenate([self.free_slots] + self.free_group)
            self.free_group = []
        else:
            for dp_rank, free_list in self.free_group.items():
                if free_list:
                    self.free_slots_per_rank[dp_rank] = np.concatenate(
                        [self.free_slots_per_rank[dp_rank]] + free_list
                    )
            self.free_group = {}

    def get_cpu_copy(self, indices):
        return self._kvcache.get_cpu_copy(indices)

    def load_cpu_copy(self, kv_cache_cpu, indices):
        return self._kvcache.load_cpu_copy(kv_cache_cpu, indices)

    def backup_state(self):
        if self.dp_size == 1:
            return (self.free_slots, self.release_pages)
        else:
            return (self.free_slots_per_rank.copy(), self.release_pages)

    def restore_state(self, state):
        assert len(state) == 2
        if self.dp_size == 1:
            self.free_slots, self.release_pages = state
        else:
            self.free_slots_per_rank, self.release_pages = state


class PagedTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    def __init__(
        self,
        size: int,
        page_size: int,
        kvcache: KVCache,
        debug_mode: bool = False,
        dp_size: int = 1,
    ):
        # super().__init__(size, page_size, dtype, kvcache)
        super().__init__(size, page_size, kvcache, dp_size)
        self.num_pages = size // page_size
        self.pages_per_rank = self.num_pages // dp_size
        self.debug_mode = debug_mode

        if dp_size == 1:
            # Single rank: simple array
            self.free_pages = np.arange(1, self.num_pages + 1, dtype=np.int32)
            self.release_pages = np.empty(0, dtype=np.int32)
            self.is_not_in_free_group = True
            self.free_group = []

            # Dynamic method binding for single-rank
            self.alloc = self._alloc_single
            self.free = self._free_single
            self.clear = self._clear_single
            self.alloc_extend = self._alloc_extend_single
            self.alloc_decode = self._alloc_decode_single
            self.free_group_begin = self._free_group_begin_single
            self.free_group_end = self._free_group_end_single
        else:
            # Multi-rank: dict-based storage with LOCAL page indices
            # Each rank has independent [1, pages_per_rank] page indices
            self.free_pages_per_rank = {}
            self.release_pages_per_rank = {}
            for rank in range(dp_size):
                self.free_pages_per_rank[rank] = np.arange(
                    1, self.pages_per_rank + 1, dtype=np.int32
                )
                self.release_pages_per_rank[rank] = np.empty(0, dtype=np.int32)
            self.is_not_in_free_group = True
            self.free_group = {}

            # Dynamic method binding for multi-rank
            self.alloc = self._alloc_multi
            self.free = self._free_multi
            self.clear = self._clear_multi
            self.alloc_extend = self._alloc_extend_multi
            self.alloc_decode = self._alloc_decode_multi
            self.free_group_begin = self._free_group_begin_multi
            self.free_group_end = self._free_group_end_multi

    # Stub implementations to satisfy ABC requirements
    # These will be overridden by dynamic binding in __init__
    def alloc(self, need_size: int, dp_rank: int = 0) -> np.ndarray | None:
        raise NotImplementedError("alloc should be bound dynamically in __init__")

    def free(self, free_index: np.ndarray, dp_rank: int = 0):
        raise NotImplementedError("free should be bound dynamically in __init__")

    def clear(self, dp_rank: int | None = None):
        raise NotImplementedError("clear should be bound dynamically in __init__")

    def alloc_extend(
        self,
        prefix_lens: list[int],
        seq_lens: list[int],
        last_loc: list[int],
        extend_num_tokens: int,
        dp_rank: int = 0,
    ) -> np.ndarray | None:
        raise NotImplementedError("alloc_extend should be bound dynamically in __init__")

    def alloc_decode(
        self,
        seq_lens: list[int],
        last_loc: list[int],
        dp_rank: int = 0,
    ) -> np.ndarray | None:
        raise NotImplementedError("alloc_decode should be bound dynamically in __init__")

    # Single-rank implementation
    def _alloc_single(self, need_size: int, dp_rank: int = 0) -> np.ndarray | None:
        """Single-rank page-aligned allocation."""
        assert need_size % self.page_size == 0, "The allocation size should be page-aligned"
        num_pages = need_size // self.page_size

        if num_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_pages > len(self.free_pages):
            return None

        out_pages = self.free_pages[:num_pages].copy()
        self.free_pages = self.free_pages[num_pages:]

        # Generate contiguous indices using numpy internally
        page_indices = out_pages[:, None] * self.page_size + np.arange(self.page_size)
        out_indices = page_indices.reshape(-1)
        return out_indices

    # Multi-rank implementation
    def _alloc_multi(self, need_size: int, dp_rank: int = 0) -> np.ndarray | None:
        """Multi-rank page-aligned allocation."""
        assert need_size % self.page_size == 0, "The allocation size should be page-aligned"
        num_pages = need_size // self.page_size

        free_pages = self.free_pages_per_rank[dp_rank]
        if num_pages > len(free_pages):
            self.merge_and_sort_free(dp_rank)
            free_pages = self.free_pages_per_rank[dp_rank]
        if num_pages > len(free_pages):
            return None

        out_pages = free_pages[:num_pages].copy()
        self.free_pages_per_rank[dp_rank] = free_pages[num_pages:]

        # Generate contiguous indices using numpy internally
        page_indices = out_pages[:, None] * self.page_size + np.arange(self.page_size)
        out_indices = page_indices.reshape(-1)
        return out_indices

    def _alloc_extend_impl(
        self,
        allocated_pages: np.ndarray,
        prefix_lens_np: np.ndarray,
        extend_lens: np.ndarray,
        last_loc_np: np.ndarray,
        extend_num_tokens: int,
    ) -> tuple[np.ndarray, int]:
        """Common implementation for alloc_extend."""
        batch_size = len(prefix_lens_np)
        out_indices = np.zeros(extend_num_tokens, dtype=np.int32)
        current_output_idx = 0
        page_idx = 0

        for seq_idx in range(batch_size):
            pre_len = prefix_lens_np[seq_idx]
            last_loc = last_loc_np[seq_idx]
            extend_len = extend_lens[seq_idx]

            if extend_len == 0:
                continue

            # Part 1: Fill remaining space in current page
            current_page_capacity = (
                (pre_len + self.page_size - 1) // self.page_size
            ) * self.page_size
            part1_size = min(extend_len, current_page_capacity - pre_len)

            if part1_size > 0:
                part1_indices = np.arange(last_loc + 1, last_loc + 1 + part1_size, dtype=np.int32)
                out_indices[current_output_idx : current_output_idx + part1_size] = part1_indices
                current_output_idx += part1_size

            remaining_tokens = extend_len - part1_size
            if remaining_tokens == 0:
                continue

            # Part 2: Allocate complete new pages
            complete_pages = remaining_tokens // self.page_size
            part2_size = complete_pages * self.page_size

            if part2_size > 0:
                for _ in range(complete_pages):
                    page_start = allocated_pages[page_idx] * self.page_size
                    part2_indices = np.arange(
                        page_start, page_start + self.page_size, dtype=np.int32
                    )
                    out_indices[current_output_idx : current_output_idx + self.page_size] = (
                        part2_indices
                    )
                    current_output_idx += self.page_size
                    page_idx += 1

            # Part 3: Allocate partial page for remaining tokens
            remaining_tokens -= part2_size
            if remaining_tokens > 0:
                page_start = allocated_pages[page_idx] * self.page_size
                part3_indices = np.arange(page_start, page_start + remaining_tokens, dtype=np.int32)
                out_indices[current_output_idx : current_output_idx + remaining_tokens] = (
                    part3_indices
                )
                current_output_idx += remaining_tokens
                page_idx += 1

        return out_indices, page_idx

    # Single-rank implementation
    def _alloc_extend_single(
        self,
        prefix_lens: list[int],
        seq_lens: list[int],
        last_loc: list[int],
        extend_num_tokens: int,
        dp_rank: int = 0,
    ) -> np.ndarray | None:
        """Single-rank alloc_extend."""
        seq_lens_np = np.array(seq_lens)
        prefix_lens_np = np.array(prefix_lens)
        last_loc_np = np.array(last_loc)

        if self.debug_mode:
            assert np.all(
                (last_loc_np + 1) % self.page_size == prefix_lens_np % self.page_size
            ), f"last_loc_np: {last_loc_np}, prefix_lens_np: {prefix_lens_np}"

        extend_lens = seq_lens_np - prefix_lens_np
        num_pages_after = (seq_lens_np + self.page_size - 1) // self.page_size
        num_pages_before = (prefix_lens_np + self.page_size - 1) // self.page_size
        total_new_pages = np.sum(num_pages_after - num_pages_before)

        if total_new_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if total_new_pages > len(self.free_pages):
            return None

        allocated_pages = self.free_pages[:total_new_pages].copy()
        out_indices, pages_used = self._alloc_extend_impl(
            allocated_pages, prefix_lens_np, extend_lens, last_loc_np, extend_num_tokens
        )
        self.free_pages = self.free_pages[pages_used:]
        return out_indices

    # Multi-rank implementation
    def _alloc_extend_multi(
        self,
        prefix_lens: list[int],
        seq_lens: list[int],
        last_loc: list[int],
        extend_num_tokens: int,
        dp_rank: int = 0,
    ) -> np.ndarray | None:
        """Multi-rank alloc_extend."""
        seq_lens_np = np.array(seq_lens)
        prefix_lens_np = np.array(prefix_lens)
        last_loc_np = np.array(last_loc)

        if self.debug_mode:
            assert np.all(
                (last_loc_np + 1) % self.page_size == prefix_lens_np % self.page_size
            ), f"last_loc_np: {last_loc_np}, prefix_lens_np: {prefix_lens_np}"

        extend_lens = seq_lens_np - prefix_lens_np
        num_pages_after = (seq_lens_np + self.page_size - 1) // self.page_size
        num_pages_before = (prefix_lens_np + self.page_size - 1) // self.page_size
        total_new_pages = np.sum(num_pages_after - num_pages_before)

        free_pages = self.free_pages_per_rank[dp_rank]
        if total_new_pages > len(free_pages):
            self.merge_and_sort_free(dp_rank)
            free_pages = self.free_pages_per_rank[dp_rank]
        if total_new_pages > len(free_pages):
            return None

        allocated_pages = free_pages[:total_new_pages].copy()
        out_indices, pages_used = self._alloc_extend_impl(
            allocated_pages, prefix_lens_np, extend_lens, last_loc_np, extend_num_tokens
        )
        self.free_pages_per_rank[dp_rank] = self.free_pages_per_rank[dp_rank][pages_used:]
        return out_indices

    def _alloc_decode_impl(
        self,
        allocated_pages: np.ndarray,
        needs_new_page: np.ndarray,
        last_loc_np: np.ndarray,
        batch_size: int,
    ) -> tuple[np.ndarray, int]:
        """Common implementation for alloc_decode."""
        out_indices = np.zeros(batch_size, dtype=np.int32)
        page_idx = 0

        for seq_idx in range(batch_size):
            if needs_new_page[seq_idx]:
                # Sequence needs a new page - allocate first position of new page
                page_start = allocated_pages[page_idx] * self.page_size
                out_indices[seq_idx] = page_start
                page_idx += 1
            else:
                # Sequence continues in current page - allocate next position
                out_indices[seq_idx] = last_loc_np[seq_idx] + 1

        return out_indices, page_idx

    # Single-rank implementation
    def _alloc_decode_single(
        self,
        seq_lens: list[int],
        last_loc: list[int],
        dp_rank: int = 0,
    ) -> np.ndarray | None:
        """Single-rank alloc_decode."""
        seq_lens_np = np.array(seq_lens)
        last_loc_np = np.array(last_loc)

        if self.debug_mode:
            assert np.all(
                (last_loc_np + 2) % self.page_size == seq_lens_np % self.page_size
            ), f"last_loc_np: {last_loc_np}, seq_lens_np: {seq_lens_np}"

        batch_size = len(seq_lens_np)
        pre_lens = seq_lens_np - 1
        num_pages_after = (seq_lens_np + self.page_size - 1) // self.page_size
        num_pages_before = (pre_lens + self.page_size - 1) // self.page_size
        needs_new_page = num_pages_after > num_pages_before
        total_new_pages = np.sum(needs_new_page)

        if total_new_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if total_new_pages > len(self.free_pages):
            return None

        allocated_pages = self.free_pages[:total_new_pages].copy()
        out_indices, pages_used = self._alloc_decode_impl(
            allocated_pages, needs_new_page, last_loc_np, batch_size
        )
        self.free_pages = self.free_pages[pages_used:]
        return out_indices

    # Multi-rank implementation
    def _alloc_decode_multi(
        self,
        seq_lens: list[int],
        last_loc: list[int],
        dp_rank: int = 0,
    ) -> np.ndarray | None:
        """Multi-rank alloc_decode."""
        seq_lens_np = np.array(seq_lens)
        last_loc_np = np.array(last_loc)

        if self.debug_mode:
            assert np.all(
                (last_loc_np + 2) % self.page_size == seq_lens_np % self.page_size
            ), f"last_loc_np: {last_loc_np}, seq_lens_np: {seq_lens_np}"

        batch_size = len(seq_lens_np)
        pre_lens = seq_lens_np - 1
        num_pages_after = (seq_lens_np + self.page_size - 1) // self.page_size
        num_pages_before = (pre_lens + self.page_size - 1) // self.page_size
        needs_new_page = num_pages_after > num_pages_before
        total_new_pages = np.sum(needs_new_page)

        free_pages = self.free_pages_per_rank[dp_rank]
        if total_new_pages > len(free_pages):
            self.merge_and_sort_free(dp_rank)
            free_pages = self.free_pages_per_rank[dp_rank]
        if total_new_pages > len(free_pages):
            return None

        allocated_pages = free_pages[:total_new_pages].copy()
        out_indices, pages_used = self._alloc_decode_impl(
            allocated_pages, needs_new_page, last_loc_np, batch_size
        )
        self.free_pages_per_rank[dp_rank] = self.free_pages_per_rank[dp_rank][pages_used:]
        return out_indices

    # Single-rank implementation
    def _free_single(self, free_index: np.ndarray, dp_rank: int = 0):
        """Single-rank free."""
        if free_index.size == 0:
            return

        if self.is_not_in_free_group:
            free_index_np = np.array(free_index)
            free_pages = np.unique(free_index_np // self.page_size)
            free_pages = np.setdiff1d(free_pages, self.release_pages)
            free_pages = np.setdiff1d(free_pages, self.free_pages)
            if len(free_pages) > 0:
                self.release_pages = np.concatenate([free_pages, self.release_pages])
        else:
            self.free_group.append(np.array(free_index))

        if self.debug_mode:
            assert len(np.unique(self.free_pages)) == len(self.free_pages)

    # Multi-rank implementation
    def _free_multi(self, free_index: np.ndarray, dp_rank: int = 0):
        """Multi-rank free."""
        if free_index.size == 0:
            return

        if self.is_not_in_free_group:
            free_index_np = np.array(free_index)
            free_pages = np.unique(free_index_np // self.page_size)
            release_pages = self.release_pages_per_rank[dp_rank]
            free_pages = np.setdiff1d(free_pages, release_pages)
            free_pages = np.setdiff1d(free_pages, self.free_pages_per_rank[dp_rank])
            if len(free_pages) > 0:
                self.release_pages_per_rank[dp_rank] = np.concatenate([free_pages, release_pages])
        else:
            if dp_rank not in self.free_group:
                self.free_group[dp_rank] = []
            self.free_group[dp_rank].append(np.array(free_index))

        if self.debug_mode:
            assert len(np.unique(self.free_pages_per_rank[dp_rank])) == len(
                self.free_pages_per_rank[dp_rank]
            )

    # Single-rank free_group methods
    def _free_group_begin_single(self):
        """Begin batching free operations (single-rank)."""
        self.is_not_in_free_group = False
        self.free_group = []

    def _free_group_end_single(self):
        """Execute all batched free operations (single-rank)."""
        self.is_not_in_free_group = True
        if self.free_group:
            all_free_indices = np.concatenate(self.free_group)
            self.free(all_free_indices, dp_rank=0)
        self.free_group = []

    # Multi-rank free_group methods
    def _free_group_begin_multi(self):
        """Begin batching free operations (multi-rank)."""
        self.is_not_in_free_group = False
        self.free_group = {}

    def _free_group_end_multi(self):
        """Execute all batched free operations (multi-rank)."""
        self.is_not_in_free_group = True
        for dp_rank, free_list in self.free_group.items():
            if free_list:
                all_free_indices = np.concatenate(free_list)
                self.free(all_free_indices, dp_rank=dp_rank)
        self.free_group = {}

    # Single-rank implementation
    def _clear_single(self, dp_rank: int | None = None):
        """Single-rank clear."""
        self.free_pages = np.arange(1, self.num_pages + 1, dtype=np.int32)
        self.release_pages = np.empty(0, dtype=np.int32)
        self.free_group = []
        self.is_not_in_free_group = True

    # Multi-rank implementation
    def _clear_multi(self, dp_rank: int | None = None):
        """Multi-rank clear."""
        if dp_rank is None:
            # Clear all ranks
            for rank in range(self.dp_size):
                self.free_pages_per_rank[rank] = np.arange(
                    1, self.pages_per_rank + 1, dtype=np.int32
                )
                self.release_pages_per_rank[rank] = np.empty(0, dtype=np.int32)
        else:
            # Clear specific rank
            self.free_pages_per_rank[dp_rank] = np.arange(
                1, self.pages_per_rank + 1, dtype=np.int32
            )
            self.release_pages_per_rank[dp_rank] = np.empty(0, dtype=np.int32)
        self.free_group = {}
        self.is_not_in_free_group = True

    def get_cpu_copy(self, indices):
        return self._kvcache.get_cpu_copy(indices)

    def load_cpu_copy(self, kv_cache_cpu, indices):
        return self._kvcache.load_cpu_copy(kv_cache_cpu, indices)


class SWATokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """Allocator for SWA hybrid KV cache."""

    def __init__(
        self,
        size: int,
        size_swa: int,
        kvcache: SWAKVPool,
        dp_size: int = 1,
    ):
        super().__init__(size, 1, kvcache, dp_size)
        assert isinstance(kvcache, SWAKVPool)
        self._size_full = size
        self._size_swa = size_swa

        # Create DP-aware sub-allocators
        self.full_attn_allocator = TokenToKVPoolAllocator(
            size,
            kvcache.full_kv_pool,
            dp_size=dp_size,
        )
        self.swa_attn_allocator = TokenToKVPoolAllocator(
            size_swa,
            kvcache.swa_kv_pool,
            dp_size=dp_size,
        )
        self.full_to_swa_index_mapping = np.empty(
            size + size_swa + 1,
            dtype=np.int64,
        )
        self.clear()

        # Dynamic method binding for free_group operations
        if dp_size == 1:
            self.free_group_begin = self._free_group_begin_single
            self.free_group_end = self._free_group_end_single
        else:
            self.free_group_begin = self._free_group_begin_multi
            self.free_group_end = self._free_group_end_multi

        self._kvcache.full_to_swa_index_mapping = self.full_to_swa_index_mapping

    def available_size(self, dp_rank: int = 0):
        raise NotImplementedError()

    def full_available_size(self, dp_rank: int = 0):
        return self.full_attn_allocator.available_size(dp_rank=dp_rank)

    def swa_available_size(self, dp_rank: int = 0):
        return self.swa_attn_allocator.available_size(dp_rank=dp_rank)

    @property
    def size_full(self):
        return self._size_full

    @property
    def size_swa(self):
        return self._size_swa

    def debug_print(self) -> str:
        msg = ""
        msg += f"#swa-available-size: {self.swa_attn_allocator.available_size()}, "
        msg += f"#full-attn-available-size: {self.full_attn_allocator.available_size()}, "
        return msg

    def get_kvcache(self):
        return self._kvcache

    def alloc(self, need_size: int, dp_rank: int = 0):
        if need_size > self.full_attn_allocator.available_size(dp_rank=dp_rank):
            return None
        if need_size > self.swa_attn_allocator.available_size(dp_rank=dp_rank):
            return None

        alloc_full_indices = self.full_attn_allocator.alloc(need_size, dp_rank=dp_rank)
        alloc_swa_indices = self.swa_attn_allocator.alloc(need_size, dp_rank=dp_rank)
        if alloc_swa_indices is None:
            # Rollback full allocation if swa allocation fails
            self.full_attn_allocator.free(alloc_full_indices, dp_rank=dp_rank)
            return None
        self.full_to_swa_index_mapping[alloc_full_indices] = alloc_swa_indices
        return alloc_full_indices

    def free(self, free_index: np.array, dp_rank: int = 0):
        if len(free_index) == 0:
            return
        if self.is_not_in_free_group:
            self.full_attn_allocator.free(free_index, dp_rank=dp_rank)
            self.free_swa(free_index, dp_rank=dp_rank)
        else:
            if self.dp_size == 1:
                self.free_group.append(free_index)
            else:
                if dp_rank not in self.free_group:
                    self.free_group[dp_rank] = []
                self.free_group[dp_rank].append(free_index)
        assert (
            self.full_attn_allocator.available_size(dp_rank=dp_rank)
            <= self.full_attn_allocator.size_per_rank
            if self.dp_size > 1
            else self.full_attn_allocator.size
        )
        assert (
            self.swa_attn_allocator.available_size(dp_rank=dp_rank)
            <= self.swa_attn_allocator.size_per_rank
            if self.dp_size > 1
            else self.swa_attn_allocator.size
        )

    def free_swa(self, free_index: np.array, dp_rank: int = 0):
        map_vals = self.full_to_swa_index_mapping[free_index]
        swa_indices = map_vals[map_vals > 0]
        self.swa_attn_allocator.free(swa_indices, dp_rank=dp_rank)
        self.full_to_swa_index_mapping[free_index] = 0

    def backup_state(self):
        raise NotImplementedError

    def restore_state(self, state):
        raise NotImplementedError

    def clear(self, dp_rank: int | None = None):
        if dp_rank is None:
            # Clear all ranks
            self.swa_attn_allocator.clear()
            self.full_attn_allocator.clear()
        else:
            # Clear specific rank
            self.swa_attn_allocator.clear(dp_rank=dp_rank)
            self.full_attn_allocator.clear(dp_rank=dp_rank)
        self.full_to_swa_index_mapping.fill(0)
        self.is_not_in_free_group = True
        self.free_group = {} if self.dp_size > 1 else []

    # Single-rank free_group methods
    def _free_group_begin_single(self):
        """Begin batching free operations (single-rank)."""
        self.is_not_in_free_group = False
        self.free_group = []

    def _free_group_end_single(self):
        """Execute all batched free operations (single-rank)."""
        self.is_not_in_free_group = True
        if self.free_group:
            all_free_indices = np.concatenate(self.free_group)
            self.free(all_free_indices, dp_rank=0)
        self.free_group = []

    # Multi-rank free_group methods
    def _free_group_begin_multi(self):
        """Begin batching free operations (multi-rank)."""
        self.is_not_in_free_group = False
        self.free_group = {}

    def _free_group_end_multi(self):
        """Execute all batched free operations (multi-rank)."""
        self.is_not_in_free_group = True
        for dp_rank, free_list in self.free_group.items():
            if free_list:
                all_free_indices = np.concatenate(free_list)
                self.free(all_free_indices, dp_rank=dp_rank)
        self.free_group = {}
