"""Device-resident, paged cache of encoder embeddings, keyed by item hash.

The pool mirrors the KV cache's paging model (see
:class:`sgl_jax.srt.mem_cache.allocator.PagedTokenToKVPoolAllocator`): a fixed
device buffer split into ``page_size``-row pages, a free-list of page ids, and
LRU eviction over whole cache entries.  Unlike the KV cache it is *content
addressed* -- an entry is keyed by ``MultimodalDataItem.hash`` (the whole
image / audio clip), not by token ids -- because multimodal embeddings share no
token-level prefix.

Storage layout is chosen so the merge can gather from the pool with the *same*
kernel it uses for the encoder's packed output: ``pages`` is
``[num_pages, page_size, (1 + deepstack_dim) * H]`` -- the primary token
embedding plus ``deepstack_dim`` deepstack planes concatenated on the trailing
axis, exactly as :class:`PackedMultimodalEmbedding.output`. A token that is the
``k``-th token of an entry lives at page ``page_ids[k // page_size]``, offset
``k % page_size`` -- i.e. ``(page, offset)`` plays exactly the role of
``(row, pos)`` in the packed contract.

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

from sgl_jax.srt.multimodal.in_model.interface import Placement
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
        # One buffer holds the primary embedding + deepstack planes contiguously,
        # matching PackedMultimodalEmbedding.output's trailing feature width.
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
        placements: Sequence[Placement],
    ) -> tuple[EmbeddingPoolEntry | None, ...]:
        """Cache a batch directly from one bucket-shaped encoder output.

        ``packed_embeddings`` is ``[num_lanes, cap, (1 + deepstack_dim) * H]`` (the
        primary embedding with deepstack planes concatenated on the trailing
        axis) and each placement is a :class:`Placement` ``(row, offset,
        true_length)``. All items are planned on the host, then the device
        consumes the complete bucket exactly once using a fixed ``[num_lanes,
        cap]`` slot map. True lengths affect only map values and page allocation,
        never the JIT signature.

        If later items evict earlier items from the same batch, only entries
        still resident after planning are written. This keeps valid scatter
        destinations unique and avoids writing data that cannot be reused.
        """
        item_hashes = tuple(map(int, item_hashes))
        placements = tuple(
            Placement(*(int(value) for value in placement)) for placement in placements
        )
        if len(item_hashes) != len(placements):
            raise ValueError(
                f"item/placement count mismatch: {len(item_hashes)} != {len(placements)}"
            )

        packed_embeddings = self._replicate(packed_embeddings)
        if packed_embeddings.ndim != 3 or packed_embeddings.shape[2] != self.feature_width:
            raise ValueError(
                "packed embeddings must have shape "
                f"[num_lanes, cap, {self.feature_width}], got {packed_embeddings.shape}"
            )
        num_lanes, cap = map(int, packed_embeddings.shape[:2])
        for placement in placements:
            if (
                placement.row < 0
                or placement.row >= num_lanes
                or placement.offset < 0
                or placement.length < 0
                or placement.offset + placement.length > cap
            ):
                raise ValueError(
                    f"invalid packed placement {placement} for shape {packed_embeddings.shape}"
                )

        planned: list[tuple[int, EmbeddingPoolEntry, Placement] | None] = []
        for item_hash, placement in zip(item_hashes, placements, strict=True):
            length = placement.length
            page_ids = self._reserve(item_hash, self._pages_for(length))
            if page_ids is None:
                planned.append(None)
                continue

            entry = EmbeddingPoolEntry(page_ids, length)
            self._entries[item_hash] = entry
            planned.append((item_hash, entry, placement))

        slots = np.full((num_lanes, cap), -1, dtype=np.int32)
        results: list[EmbeddingPoolEntry | None] = []
        for plan in planned:
            if plan is None:
                results.append(None)
                continue
            item_hash, entry, (lane, offset, length) = plan
            if self._entries.get(item_hash) is not entry:
                results.append(None)
                continue
            slots[lane, offset : offset + length] = self._slots(entry.page_ids)[:length]
            results.append(entry)

        if any(entry is not None and entry.length for entry in results):
            slots = self._replicate(slots)
            self._pages = _scatter_rows(self._pages, slots, packed_embeddings)
        return tuple(results)

    def precompile_packed_write(self, num_lanes: int, cap: int) -> None:
        """Compile the packed writer for one encoder bucket without changing LRU state."""
        if num_lanes <= 0 or cap <= 0:
            raise ValueError("packed writer dimensions must be positive")
        with jax.set_mesh(self.mesh) if self.mesh is not None else nullcontext():
            slots = self._replicate(np.full((num_lanes, cap), -1, dtype=np.int32))
            rows = self._zeros((num_lanes, cap, self.feature_width), self._pages.dtype)
            self._pages = _scatter_rows(self._pages, slots, rows)
            jax.block_until_ready(self._pages)

    def clear(self) -> None:
        """Free all pages (buffers are kept; only the free-list/table reset)."""
        self._entries.clear()
        self._free_pages = np.arange(self.num_pages, dtype=np.int32)
