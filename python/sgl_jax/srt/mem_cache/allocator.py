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
        return (len(self.free_pages[dp_rank]) + len(self.release_pages[dp_rank])) * self.page_size

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
        release_pages = self.release_pages[dp_rank]
        if len(release_pages) > 0:
            combined = np.concatenate((self.free_pages[dp_rank], release_pages))
            self.free_pages[dp_rank] = np.sort(combined)
            self.release_pages[dp_rank] = np.empty((0,), dtype=np.int32)

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

        # Each rank has independent [1, size_per_rank] indices (shard_map local view)
        self.free_slots = [
            np.arange(1, self.size_per_rank + 1, dtype=np.int32) for _ in range(dp_size)
        ]
        self.origin_size = len(self.free_slots[0])
        self.free_group = [[] for _ in range(dp_size)]
        self.is_not_in_free_group = True

    def alloc(self, need_size: int, dp_rank: int = 0) -> np.ndarray | None:
        slots = self.free_slots[dp_rank]
        if need_size > len(slots):
            return None
        select_index = slots[:need_size].copy()
        self.free_slots[dp_rank] = slots[need_size:]
        return select_index

    def free(self, free_index: np.ndarray, dp_rank: int = 0):
        if free_index.size == 0:
            return
        if self.is_not_in_free_group:
            self.free_slots[dp_rank] = np.concatenate([self.free_slots[dp_rank], free_index])
        else:
            self.free_group[dp_rank].append(free_index)

    def available_size(self, dp_rank: int = 0) -> int:
        return len(self.free_slots[dp_rank])

    def clear(self, dp_rank: int | None = None):
        ranks = range(self.dp_size) if dp_rank is None else [dp_rank]
        for rank in ranks:
            self.free_slots[rank] = np.arange(1, self.size_per_rank + 1, dtype=np.int32)
            self.free_group[rank] = []
        self.is_not_in_free_group = True

    def free_group_begin(self):
        self.is_not_in_free_group = False

    def free_group_end(self):
        self.is_not_in_free_group = True
        for rank in range(self.dp_size):
            if self.free_group[rank]:
                self.free_slots[rank] = np.concatenate(
                    [self.free_slots[rank]] + self.free_group[rank]
                )
                self.free_group[rank] = []

    def get_cpu_copy(self, indices):
        return self._kvcache.get_cpu_copy(indices)

    def load_cpu_copy(self, kv_cache_cpu, indices):
        return self._kvcache.load_cpu_copy(kv_cache_cpu, indices)

    def backup_state(self):
        return ([slots.copy() for slots in self.free_slots], self.release_pages)

    def restore_state(self, state):
        assert len(state) == 2
        self.free_slots, self.release_pages = state


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

        # Each rank has independent [1, pages_per_rank] page indices
        self.free_pages = [
            np.arange(1, self.pages_per_rank + 1, dtype=np.int32) for _ in range(dp_size)
        ]
        self.release_pages = [np.empty(0, dtype=np.int32) for _ in range(dp_size)]
        self.free_group = [[] for _ in range(dp_size)]
        self.is_not_in_free_group = True

    def alloc(self, need_size: int, dp_rank: int = 0) -> np.ndarray | None:
        """Page-aligned allocation."""
        assert need_size % self.page_size == 0, "The allocation size should be page-aligned"
        num_pages = need_size // self.page_size

        if num_pages > len(self.free_pages[dp_rank]):
            self.merge_and_sort_free(dp_rank)
        if num_pages > len(self.free_pages[dp_rank]):
            return None

        out_pages = self.free_pages[dp_rank][:num_pages].copy()
        self.free_pages[dp_rank] = self.free_pages[dp_rank][num_pages:]

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

    def alloc_extend(
        self,
        prefix_lens: list[int],
        seq_lens: list[int],
        last_loc: list[int],
        extend_num_tokens: int,
        dp_rank: int = 0,
    ) -> np.ndarray | None:
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

        if total_new_pages > len(self.free_pages[dp_rank]):
            self.merge_and_sort_free(dp_rank)
        if total_new_pages > len(self.free_pages[dp_rank]):
            return None

        allocated_pages = self.free_pages[dp_rank][:total_new_pages].copy()
        out_indices, pages_used = self._alloc_extend_impl(
            allocated_pages, prefix_lens_np, extend_lens, last_loc_np, extend_num_tokens
        )
        self.free_pages[dp_rank] = self.free_pages[dp_rank][pages_used:]
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

    def alloc_decode(
        self,
        seq_lens: list[int],
        last_loc: list[int],
        dp_rank: int = 0,
    ) -> np.ndarray | None:
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

        if total_new_pages > len(self.free_pages[dp_rank]):
            self.merge_and_sort_free(dp_rank)
        if total_new_pages > len(self.free_pages[dp_rank]):
            return None

        allocated_pages = self.free_pages[dp_rank][:total_new_pages].copy()
        out_indices, pages_used = self._alloc_decode_impl(
            allocated_pages, needs_new_page, last_loc_np, batch_size
        )
        self.free_pages[dp_rank] = self.free_pages[dp_rank][pages_used:]
        return out_indices

    def free(self, free_index: np.ndarray, dp_rank: int = 0):
        if free_index.size == 0:
            return

        if self.is_not_in_free_group:
            free_pages = np.unique(free_index // self.page_size)
            rel_pages = self.release_pages[dp_rank]
            f_pages = self.free_pages[dp_rank]
            free_pages = np.setdiff1d(free_pages, rel_pages)
            free_pages = np.setdiff1d(free_pages, f_pages)
            if len(free_pages) > 0:
                self.release_pages[dp_rank] = np.concatenate([free_pages, rel_pages])
        else:
            self.free_group[dp_rank].append(free_index)

        if self.debug_mode:
            assert len(np.unique(self.free_pages[dp_rank])) == len(self.free_pages[dp_rank])

    def free_group_begin(self):
        self.is_not_in_free_group = False

    def free_group_end(self):
        self.is_not_in_free_group = True
        for rank in range(self.dp_size):
            if self.free_group[rank]:
                all_free_indices = np.concatenate(self.free_group[rank])
                self.free(all_free_indices, dp_rank=rank)
                self.free_group[rank] = []

    def clear(self, dp_rank: int | None = None):
        ranks = range(self.dp_size) if dp_rank is None else [dp_rank]
        for rank in ranks:
            self.free_pages[rank] = np.arange(1, self.pages_per_rank + 1, dtype=np.int32)
            self.release_pages[rank] = np.empty(0, dtype=np.int32)
            self.free_group[rank] = []
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
        # Each rank needs its own mapping since they use local indices [1, size_per_rank]
        self.full_to_swa_index_mapping = [
            np.zeros(self.full_attn_allocator.size_per_rank + 1, dtype=np.int64)
            for _ in range(dp_size)
        ]
        self.clear()

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
        self.full_to_swa_index_mapping[dp_rank][alloc_full_indices] = alloc_swa_indices
        return alloc_full_indices

    def free(self, free_index: np.array, dp_rank: int = 0):
        if len(free_index) == 0:
            return
        if self.is_not_in_free_group:
            self.full_attn_allocator.free(free_index, dp_rank=dp_rank)
            self.free_swa(free_index, dp_rank=dp_rank)
        else:
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
        map_vals = self.full_to_swa_index_mapping[dp_rank][free_index]
        swa_indices = map_vals[map_vals > 0]
        self.swa_attn_allocator.free(swa_indices, dp_rank=dp_rank)
        self.full_to_swa_index_mapping[dp_rank][free_index] = 0

    def backup_state(self):
        raise NotImplementedError

    def restore_state(self, state):
        raise NotImplementedError

    def clear(self, dp_rank: int | None = None):
        if dp_rank is None:
            # Clear all ranks
            self.swa_attn_allocator.clear()
            self.full_attn_allocator.clear()
            for rank in range(self.dp_size):
                self.full_to_swa_index_mapping[rank].fill(0)
        else:
            # Clear specific rank
            self.swa_attn_allocator.clear(dp_rank=dp_rank)
            self.full_attn_allocator.clear(dp_rank=dp_rank)
            self.full_to_swa_index_mapping[dp_rank].fill(0)
        self.is_not_in_free_group = True
        self.free_group = [[] for _ in range(self.dp_size)]

    def free_group_begin(self):
        self.is_not_in_free_group = False

    def free_group_end(self):
        self.is_not_in_free_group = True
        for rank in range(self.dp_size):
            if self.free_group[rank]:
                all_free_indices = np.concatenate(self.free_group[rank])
                self.free(all_free_indices, dp_rank=rank)
                self.free_group[rank] = []
