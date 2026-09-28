"""Device-resident, paged cache of encoder embeddings, keyed by item hash.

The pool mirrors the KV cache's paging model (see
:class:`sgl_jax.srt.mem_cache.allocator.PagedTokenToKVPoolAllocator`): a fixed
device buffer split into ``page_size``-row pages, a free-list of page ids, and
LRU eviction over whole cache entries.  Unlike the KV cache it is *content
addressed* -- an entry is keyed by ``MultimodalDataItem.hash`` (the whole
image / audio clip), not by token ids -- because multimodal embeddings share no
token-level prefix.

``pages`` stores opaque encoder output rows as
``[num_pages, page_size, hidden]``. Any model-specific feature packing is
already reflected in ``hidden``.

Writes are performed by a ``jit``+``donate`` scatter so the large device buffer
is updated in place (eager ``.at[].set`` would copy the whole pool per write).
"""

from __future__ import annotations

from collections import OrderedDict
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
        mesh: Mesh | None = None,
    ) -> None:
        if num_pages <= 0 or page_size <= 0:
            raise ValueError("embedding pool needs positive num_pages and page_size")
        self.num_pages = num_pages
        self.page_size = page_size
        self.hidden = hidden
        self.mesh = mesh

        self._free_pages = np.arange(num_pages, dtype=np.int32)
        self._entries: OrderedDict[int, EmbeddingPoolEntry] = OrderedDict()

        self._pages = self._zeros((num_pages, page_size, hidden), dtype)

    # -- buffers -----------------------------------------------------------
    @property
    def pages(self) -> jax.Array:
        return self._pages

    def _zeros(self, shape: tuple[int, ...], dtype: jnp.dtype) -> jax.Array:
        if self.mesh is None:
            return jnp.zeros(shape, dtype=dtype)
        sharding = NamedSharding(
            self.mesh,
            PartitionSpec(*([None] * len(shape))),
        )
        with jax.set_mesh(self.mesh):
            return jnp.zeros(shape, dtype=dtype, out_sharding=sharding)

    def _replicate(self, value: ArrayLike) -> jax.Array:
        if self.mesh is None:
            return jnp.asarray(value)
        return replicate_across_mesh(value, self.mesh)

    # -- allocation --------------------------------------------------------
    def _reserve(self, item_hash: int, length: int) -> EmbeddingPoolEntry | None:
        """Replace an item and allocate its pages, evicting LRU entries."""
        n_pages = (length + self.page_size - 1) // self.page_size
        if n_pages > self.num_pages:
            return None
        previous = self._entries.pop(item_hash, None)
        if previous is not None:
            self._free_pages = np.concatenate([self._free_pages, previous.page_ids])

        while len(self._free_pages) < n_pages:
            _, entry = self._entries.popitem(last=False)
            self._free_pages = np.concatenate([self._free_pages, entry.page_ids])

        entry = EmbeddingPoolEntry(self._free_pages[:n_pages].copy(), length)
        self._free_pages = self._free_pages[n_pages:]
        self._entries[item_hash] = entry
        return entry

    # -- public API --------------------------------------------------------
    def contains(self, item_hash: int) -> bool:
        """Read-only scheduling hint; forward must recheck before using the cache."""
        return item_hash in self._entries

    def lookup(self, item_hash: int) -> EmbeddingPoolEntry | None:
        """Return the entry for ``item_hash`` (moved to MRU) or ``None``."""
        entry = self._entries.pop(item_hash, None)
        if entry is not None:
            self._entries[item_hash] = entry
        return entry

    def write_packed(
        self,
        item_hashes: list[int],
        packed_embeddings: ArrayLike,
        lengths: list[int],
        *,
        write_mask: list[bool] | None = None,
    ) -> list[EmbeddingPoolEntry | None]:
        """Cache one padded encoder output whose items are packed in input order."""
        if write_mask is None:
            write_mask = [True] * len(lengths)
        if len(item_hashes) != len(lengths):
            raise ValueError(f"item/length count mismatch: {len(item_hashes)} != {len(lengths)}")
        if len(write_mask) != len(lengths):
            raise ValueError(f"mask/length count mismatch: {len(write_mask)} != {len(lengths)}")

        packed_embeddings = self._replicate(packed_embeddings)
        if packed_embeddings.ndim != 2 or packed_embeddings.shape[1] != self.hidden:
            raise ValueError(
                "packed embeddings must have shape "
                f"[capacity, {self.hidden}], got {packed_embeddings.shape}"
            )
        capacity = int(packed_embeddings.shape[0])
        if any(length < 0 for length in lengths) or sum(lengths) > capacity:
            raise ValueError(f"invalid item lengths {lengths} for capacity {capacity}")

        results = [
            self._reserve(int(item_hash), int(length)) if should_write else None
            for item_hash, length, should_write in zip(
                item_hashes, lengths, write_mask, strict=True
            )
        ]

        # Later allocations can evict or replace earlier items in this same batch.
        # Build slots only after all placements are final.
        slots = np.full(capacity, -1, dtype=np.int32)
        offset = 0
        for i, (item_hash, length, entry) in enumerate(
            zip(item_hashes, lengths, results, strict=True)
        ):
            if entry is not None and self._entries.get(int(item_hash)) is entry:
                rows = np.arange(entry.length, dtype=np.int32)
                slots[offset : offset + length] = (
                    entry.page_ids[rows // self.page_size] * self.page_size + rows % self.page_size
                )
            else:
                results[i] = None
            offset += length

        if any(entry is not None and entry.length for entry in results):
            slots = self._replicate(slots)
            self._pages = _scatter_rows(self._pages, slots, packed_embeddings)
        return results

    def precompile_packed_write(self, capacity: int) -> None:
        """Compile the packed writer for one encoder bucket without changing LRU state."""
        if capacity <= 0:
            raise ValueError("packed writer capacity must be positive")
        with jax.set_mesh(self.mesh) if self.mesh is not None else nullcontext():
            slots = self._replicate(np.full(capacity, -1, dtype=np.int32))
            rows = self._zeros((capacity, self.hidden), self._pages.dtype)
            self._pages = _scatter_rows(self._pages, slots, rows)
            jax.block_until_ready(self._pages)

    def clear(self) -> None:
        """Free all pages (buffers are kept; only the free-list/table reset)."""
        self._entries.clear()
        self._free_pages = np.arange(self.num_pages, dtype=np.int32)
