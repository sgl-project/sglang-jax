import abc
import logging
from typing import Optional

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
        self.is_not_in_free_group = True
        self.free_group = []

    def debug_print(self) -> str:
        return ""

    def available_size(self) -> int:
        return len(self.free_pages) * self.page_size

    def get_kvcache(self) -> KVCache:
        return self._kvcache

    def restore_state(self, free_pages: jnp.ndarray):
        self.free_pages = free_pages

    def backup_state(self) -> jnp.ndarray:
        return self.free_pages

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            all_free_indices = jnp.concatenate(self.free_group)
            self.free(all_free_indices)

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
            self.free_group.append(free_index)

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
    ):
        super().__init__(size, page_size, dtype, kvcache)
        self.num_pages = size // page_size
        self.clear()

    def alloc(self, need_size: int) -> Optional[jnp.ndarray]:
        # page-aligned allocation, returning contiguous indices of pages
        assert (
            need_size % self.page_size == 0
        ), "The allocation size should be page-aligned"

        num_pages = need_size // self.page_size
        if num_pages > len(self.free_pages):
            return None

        out_pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]

        # Generate contiguous indices
        page_indices = out_pages[:, None] * self.page_size + jnp.arange(self.page_size)
        out_indices = page_indices.reshape(-1)

        return out_indices

    def alloc_extend(
        self,
        prefix_lens: jnp.ndarray,
        seq_lens: jnp.ndarray,
        last_loc: jnp.ndarray,
        extend_num_tokens: int,
    ) -> Optional[jnp.ndarray]:
        extend_lens = seq_lens - prefix_lens
        num_pages_after = (seq_lens + self.page_size - 1) // self.page_size
        num_pages_before = (prefix_lens + self.page_size - 1) // self.page_size
        num_new_pages = num_pages_after - num_pages_before
        total_new_pages = jnp.sum(num_new_pages)

        if total_new_pages > len(self.free_pages):
            return None

        # Simplified allocation - for production would need more sophisticated logic
        out_indices = jnp.zeros(extend_num_tokens, dtype=jnp.int32)

        # Update free pages
        self.free_pages = self.free_pages[total_new_pages:]

        return out_indices

    def alloc_decode(
        self,
        seq_lens: jnp.ndarray,
        last_loc: jnp.ndarray,
    ) -> Optional[jnp.ndarray]:
        pre_lens = seq_lens - 1
        num_pages_after = (seq_lens + self.page_size - 1) // self.page_size
        num_pages_before = (pre_lens + self.page_size - 1) // self.page_size
        num_new_pages = num_pages_after - num_pages_before
        total_new_pages = jnp.sum(num_new_pages)

        if total_new_pages > len(self.free_pages):
            return None

        bs = len(seq_lens)
        out_indices = jnp.zeros(bs, dtype=jnp.int32)

        # Update free pages
        self.free_pages = self.free_pages[total_new_pages:]

        return out_indices

    def free(self, free_index: jnp.ndarray):
        if free_index.size == 0:
            return

        if self.is_not_in_free_group:
            free_page_indices = jnp.unique(free_index // self.page_size)
            self.free_pages = jnp.concatenate([free_page_indices, self.free_pages])
        else:
            self.free_group.append(free_index)

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = jnp.arange(1, self.num_pages + 1, dtype=jnp.int32)
        self.is_not_in_free_group = True
        self.free_group = []

    def get_cpu_copy(self, indices):
        return self._kvcache.get_cpu_copy(indices)

    def load_cpu_copy(self, kv_cache_cpu, indices):
        return self._kvcache.load_cpu_copy(kv_cache_cpu, indices)
