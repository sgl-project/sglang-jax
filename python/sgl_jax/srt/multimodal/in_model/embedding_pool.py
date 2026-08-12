"""Device-resident, paged cache of encoder embeddings, keyed by item hash.

The pool mirrors the KV cache's paging model (see
:class:`sgl_jax.srt.mem_cache.allocator.PagedTokenToKVPoolAllocator`): a fixed
device buffer split into ``page_size``-row pages, a free-list of page ids, and
LRU eviction over whole cache entries.  Unlike the KV cache it is *content
addressed* -- an entry is keyed by ``MultimodalDataItem.hash`` (the whole
image / audio clip), not by token ids -- because multimodal embeddings share no
token-level prefix.

``pages`` stores the primary embedding and any deepstack planes contiguously as
``[num_pages, page_size, (1 + deepstack_dim) * H]``.

Writes are performed by a ``jit``+``donate`` scatter so the large device buffer
is updated in place (eager ``.at[].set`` would copy the whole pool per write).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.typing import ArrayLike

from sgl_jax.srt.multimodal.in_model.lane_packing import replicate_across_mesh


@dataclass(frozen=True)
class EmbeddingPoolEntry:
    """Locates one cached item inside the pool.

    ``page_ids`` are the (possibly non-contiguous) pages holding the item's
    tokens back-to-back; ``length`` is the item's true token count (the tail of
    the last page is padding).
    """

    page_ids: np.ndarray  # [num_pages_for_item], int32
    length: int


@partial(jax.jit, donate_argnames=("buffer",))
def _scatter_rows(buffer: jax.Array, slots: jax.Array, rows: jax.Array) -> jax.Array:
    """In-place masked scatter of bucket-shaped ``rows`` into ``buffer``.

    ``buffer`` is flattened over its leading ``(page, offset)`` axes. ``slots``
    and the leading axes of ``rows`` are flattened together; a negative slot
    marks padding and is converted to a positive out-of-bounds sentinel so JAX
    drops it instead of applying its usual negative-index wrapping.
    """
    flat = buffer.reshape(-1, *buffer.shape[2:])
    flat_slots = slots.reshape(-1)
    flat_rows = rows.reshape(-1, *buffer.shape[2:])
    if flat_slots.shape[0] != flat_rows.shape[0]:
        raise ValueError(
            f"slots/rows length mismatch: {flat_slots.shape[0]} != {flat_rows.shape[0]}"
        )
    safe_slots = jnp.where(flat_slots >= 0, flat_slots, flat.shape[0])
    flat = flat.at[safe_slots].set(flat_rows.astype(buffer.dtype), mode="drop")
    return flat.reshape(buffer.shape)


class EmbeddingPool:
    """Byte-bounded (page-count-bounded) LRU of encoder embeddings on device."""

    def __init__(
        self,
        num_pages: int,
        page_size: int,
        hidden: int,
        dtype: jnp.dtype,
        *,
        deepstack_dim: int = 0,
        mesh: Mesh | None = None,
    ) -> None:
        if num_pages <= 0 or page_size <= 0:
            raise ValueError("embedding pool needs positive num_pages and page_size")
        self.num_pages = num_pages
        self.page_size = page_size
        self.hidden = hidden
        self.deepstack_dim = deepstack_dim
        self.feature_width = hidden * (1 + deepstack_dim)
        self.mesh = mesh

        self._free_pages = np.arange(num_pages, dtype=np.int32)
        self._entries: OrderedDict[int, EmbeddingPoolEntry] = OrderedDict()

        self._pages = self._zeros((num_pages, page_size, self.feature_width), dtype)

    # -- buffers -----------------------------------------------------------
    @property
    def pages(self) -> jax.Array:
        return self._pages

    def _zeros(self, shape: tuple[int, ...], dtype: jnp.dtype) -> jax.Array:
        array = jnp.zeros(shape, dtype=dtype)
        if self.mesh is None:
            return array
        return jax.device_put(
            array,
            NamedSharding(self.mesh, PartitionSpec(*([None] * len(shape)))),
        )

    def _replicate(self, value: ArrayLike) -> jax.Array:
        if self.mesh is None:
            return jnp.asarray(value)
        return replicate_across_mesh(value, self.mesh)

    # -- allocation --------------------------------------------------------
    def _pages_for(self, length: int) -> int:
        return (length + self.page_size - 1) // self.page_size

    def _alloc(self, n_pages: int) -> np.ndarray | None:
        """Reserve ``n_pages`` pages, evicting LRU entries under pressure."""
        while len(self._free_pages) < n_pages and self._entries:
            _, evicted = self._entries.popitem(last=False)
            self._free_pages = np.concatenate([self._free_pages, evicted.page_ids])
        if len(self._free_pages) < n_pages:
            return None
        out = self._free_pages[:n_pages].copy()
        self._free_pages = self._free_pages[n_pages:]
        return out

    def _reserve(self, item_hash: int, n_pages: int) -> np.ndarray | None:
        """Drop any prior entry for ``item_hash`` and reserve ``n_pages`` fresh pages.

        The ``n_pages > num_pages`` guard fails fast *before* eviction, so an item
        too large for the whole pool never flushes the resident entries.
        """
        if n_pages > self.num_pages:
            return None
        previous = self._entries.pop(item_hash, None)
        if previous is not None:
            self._free_pages = np.concatenate([self._free_pages, previous.page_ids])
        return self._alloc(n_pages)

    def _slots(self, page_ids: np.ndarray) -> np.ndarray:
        """Flat row indices covered by ``page_ids`` (page-aligned)."""
        return (page_ids[:, None] * self.page_size + np.arange(self.page_size)).reshape(-1)

    # -- public API --------------------------------------------------------
    def lookup(self, item_hash: int) -> EmbeddingPoolEntry | None:
        """Return the entry for ``item_hash`` (moved to MRU) or ``None``."""
        entry = self._entries.pop(item_hash, None)
        if entry is not None:
            self._entries[item_hash] = entry
        return entry

    def write_packed(
        self,
        item_hashes: Sequence[int],
        packed_embeddings: ArrayLike,
        lengths: Sequence[int],
        *,
        write_mask: Sequence[bool] | None = None,
    ) -> tuple[EmbeddingPoolEntry | None, ...]:
        """Cache one padded encoder output whose items are packed in input order."""
        item_hashes = tuple(map(int, item_hashes))
        lengths = tuple(map(int, lengths))
        write_mask = (True,) * len(lengths) if write_mask is None else tuple(map(bool, write_mask))
        if len(item_hashes) != len(lengths):
            raise ValueError(f"item/length count mismatch: {len(item_hashes)} != {len(lengths)}")
        if len(write_mask) != len(lengths):
            raise ValueError(f"mask/length count mismatch: {len(write_mask)} != {len(lengths)}")

        packed_embeddings = self._replicate(packed_embeddings)
        if packed_embeddings.ndim != 2 or packed_embeddings.shape[1] != self.feature_width:
            raise ValueError(
                "packed embeddings must have shape "
                f"[capacity, {self.feature_width}], got {packed_embeddings.shape}"
            )
        capacity = int(packed_embeddings.shape[0])
        if any(length < 0 for length in lengths) or sum(lengths) > capacity:
            raise ValueError(f"invalid item lengths {lengths} for capacity {capacity}")

        planned: list[tuple[int, EmbeddingPoolEntry, int, int] | None] = []
        offset = 0
        for item_hash, length, should_write in zip(item_hashes, lengths, write_mask, strict=True):
            page_ids = self._reserve(item_hash, self._pages_for(length)) if should_write else None
            if page_ids is None:
                planned.append(None)
            else:
                entry = EmbeddingPoolEntry(page_ids, length)
                self._entries[item_hash] = entry
                planned.append((item_hash, entry, offset, length))
            offset += length

        slots = np.full(capacity, -1, dtype=np.int32)
        results: list[EmbeddingPoolEntry | None] = []
        for plan in planned:
            if plan is None:
                results.append(None)
                continue
            item_hash, entry, offset, length = plan
            if self._entries.get(item_hash) is not entry:
                results.append(None)
                continue
            slots[offset : offset + length] = self._slots(entry.page_ids)[:length]
            results.append(entry)

        if any(entry is not None and entry.length for entry in results):
            slots = self._replicate(slots)
            self._pages = _scatter_rows(self._pages, slots, packed_embeddings)
        return tuple(results)

    def precompile_packed_write(self, capacity: int) -> None:
        """Compile the packed writer for one encoder bucket without changing LRU state."""
        if capacity <= 0:
            raise ValueError("packed writer capacity must be positive")
        with jax.set_mesh(self.mesh) if self.mesh is not None else nullcontext():
            slots = self._replicate(np.full(capacity, -1, dtype=np.int32))
            rows = self._zeros((capacity, self.feature_width), self._pages.dtype)
            self._pages = _scatter_rows(self._pages, slots, rows)
            jax.block_until_ready(self._pages)

    def clear(self) -> None:
        """Free all pages (buffers are kept; only the free-list/table reset)."""
        self._entries.clear()
        self._free_pages = np.arange(self.num_pages, dtype=np.int32)
