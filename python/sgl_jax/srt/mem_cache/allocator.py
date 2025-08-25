import abc
import logging
from typing import List, Optional

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)


class BaseTokenToKVPoolAllocator(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: jnp.dtype,
        kvcache: KVCache,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self._kvcache = kvcache

        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []

    def debug_print(self) -> str:
        return ""

    def available_size(self) -> int:
        return len(self.free_pages) * self.page_size

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

    def merge_and_sort_free(self):
        if len(self.release_pages) > 0:
            self.free_pages = np.concatenate((self.free_pages, self.release_pages))
            self.free_pages = np.sort(self.free_pages)
            self.release_pages = np.empty((0,), dtype=np.int32)

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
    def clear(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def alloc(self, need_size: int) -> Optional[jnp.ndarray]:
        raise NotImplementedError()

    @abc.abstractmethod
    def free(self, free_index: jnp.ndarray):
        raise NotImplementedError()


class TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    def __init__(
        self,
        size: int,
        dtype: jnp.dtype,
        kvcache: KVCache,
    ):
        super().__init__(size, 1, dtype, kvcache)  # page_size=1 for token-level
        self.clear()

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_slots = np.arange(1, self.size + 1, dtype=np.int32)
        self.is_not_in_free_group = True
        self.free_group = []

    def available_size(self) -> int:
        # To avoid minor "len(free_slots) * 1" overhead
        return len(self.free_slots)

    def alloc(self, need_size: int) -> Optional[jnp.ndarray]:
        if need_size > self.available_size():
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return jnp.array(select_index)

    def free(self, free_index: jnp.ndarray):
        if free_index.size == 0:
            return

        if self.is_not_in_free_group:
            self.free_slots = np.concatenate([self.free_slots, np.array(free_index)])
        else:
            self.free_group.append(np.array(free_index))

    def get_cpu_copy(self, indices):
        return self._kvcache.get_cpu_copy(indices)

    def load_cpu_copy(self, kv_cache_cpu, indices):
        return self._kvcache.load_cpu_copy(kv_cache_cpu, indices)


class PagedTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: jnp.dtype,
        kvcache: KVCache,
        debug_mode: bool = False,
    ):
        super().__init__(size, page_size, dtype, kvcache)
        self.num_pages = size // page_size
        self.debug_mode = debug_mode
        self.clear()

    def alloc(self, need_size: int) -> Optional[jnp.ndarray]:
        # page-aligned allocation, returning contiguous indices of pages
        assert (
            need_size % self.page_size == 0
        ), "The allocation size should be page-aligned"

        num_pages = need_size // self.page_size
        if num_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_pages > len(self.free_pages):
            return None

        out_pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]

        # Generate contiguous indices using numpy internally
        page_indices = out_pages[:, None] * self.page_size + np.arange(self.page_size)
        out_indices = page_indices.reshape(-1)

        return jnp.array(out_indices)

    def alloc_extend(
        self,
        prefix_lens: List[int],
        seq_lens: List[int],
        last_loc: List[int],
        extend_num_tokens: int,
    ) -> Optional[jnp.ndarray]:
        # Convert to numpy for internal operations
        seq_lens_np = np.array(seq_lens)
        prefix_lens_np = np.array(prefix_lens)
        last_loc_np = np.array(last_loc)

        batch_size = len(seq_lens_np)
        extend_lens = seq_lens_np - prefix_lens_np

        # Calculate total pages needed for all sequences
        num_pages_after = (seq_lens_np + self.page_size - 1) // self.page_size
        num_pages_before = (prefix_lens_np + self.page_size - 1) // self.page_size
        num_new_pages_per_seq = num_pages_after - num_pages_before
        total_new_pages = np.sum(num_new_pages_per_seq)

        # Check if we have enough pages
        if total_new_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if total_new_pages > len(self.free_pages):
            return None

        # Get pages for allocation
        allocated_pages = self.free_pages[:total_new_pages].copy()
        self.free_pages = self.free_pages[total_new_pages:]

        # Allocate indices using three-part strategy
        out_indices = np.zeros(extend_num_tokens, dtype=np.int32)
        current_output_idx = 0
        page_idx = 0

        for seq_idx in range(batch_size):
            seq_len = seq_lens_np[seq_idx]
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
                part1_indices = np.arange(
                    last_loc + 1, last_loc + 1 + part1_size, dtype=np.int32
                )
                out_indices[current_output_idx : current_output_idx + part1_size] = (
                    part1_indices
                )
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
                    out_indices[
                        current_output_idx : current_output_idx + self.page_size
                    ] = part2_indices
                    current_output_idx += self.page_size
                    page_idx += 1

            # Part 3: Allocate partial page for remaining tokens
            remaining_tokens -= part2_size
            if remaining_tokens > 0:
                page_start = allocated_pages[page_idx] * self.page_size
                part3_indices = np.arange(
                    page_start, page_start + remaining_tokens, dtype=np.int32
                )
                out_indices[
                    current_output_idx : current_output_idx + remaining_tokens
                ] = part3_indices
                current_output_idx += remaining_tokens
                page_idx += 1

        # Debug validation
        if self.debug_mode:
            # Verify that last_loc+1 aligns with prefix_lens in page position
            for seq_idx in range(batch_size):
                if extend_lens[seq_idx] > 0:
                    expected_pos = (last_loc_np[seq_idx] + 1) % self.page_size
                    actual_pos = prefix_lens_np[seq_idx] % self.page_size
                    assert (
                        expected_pos == actual_pos
                    ), f"Seq {seq_idx}: last_loc+1 pos {expected_pos} != prefix_len pos {actual_pos}"

        return jnp.array(out_indices)

    def alloc_decode(
        self,
        seq_lens: List[int],
        last_loc: List[int],
    ) -> Optional[jnp.ndarray]:
        # Convert inputs to numpy for calculations
        seq_lens_np = np.array(seq_lens)
        last_loc_np = np.array(last_loc)

        batch_size = len(seq_lens_np)
        pre_lens = seq_lens_np - 1

        # Calculate which sequences need new pages
        num_pages_after = (seq_lens_np + self.page_size - 1) // self.page_size
        num_pages_before = (pre_lens + self.page_size - 1) // self.page_size
        needs_new_page = num_pages_after > num_pages_before
        total_new_pages = np.sum(needs_new_page)

        # Check if we have enough pages
        if total_new_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if total_new_pages > len(self.free_pages):
            return None

        # Allocate new pages
        allocated_pages = self.free_pages[:total_new_pages].copy()
        self.free_pages = self.free_pages[total_new_pages:]

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

        # Debug validation
        if self.debug_mode:
            for seq_idx in range(batch_size):
                # Verify that last_loc+2 aligns with seq_lens in page position
                expected_pos = (last_loc_np[seq_idx] + 2) % self.page_size
                actual_pos = seq_lens_np[seq_idx] % self.page_size
                assert (
                    expected_pos == actual_pos
                ), f"Seq {seq_idx}: last_loc+2 pos {expected_pos} != seq_len pos {actual_pos}"

        return jnp.array(out_indices)

    def free(self, free_index: jnp.ndarray):
        if free_index.size == 0:
            return

        if self.is_not_in_free_group:
            # Convert to numpy for internal operations
            free_index_np = np.array(free_index)
            free_page_indices = np.unique(free_index_np // self.page_size)
            self.release_pages = np.concatenate([free_page_indices, self.release_pages])
        else:
            self.free_group.append(np.array(free_index))

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = np.arange(1, self.num_pages + 1, dtype=np.int32)
        self.is_not_in_free_group = True
        self.free_group = []
        self.release_pages = np.empty(0, dtype=np.int32)

    def get_cpu_copy(self, indices):
        return self._kvcache.get_cpu_copy(indices)

    def load_cpu_copy(self, kv_cache_cpu, indices):
        return self._kvcache.load_cpu_copy(kv_cache_cpu, indices)
