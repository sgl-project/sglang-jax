"""Grouped history + independent SWA pages with plan-before-commit allocation."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from sgl_jax.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator


@dataclass(frozen=True)
class DeepseekV4AllocationDemand:
    history_pages: int
    swa_pages: int
    page_size: int

    @property
    def history_tokens(self):
        return self.history_pages * self.page_size

    @property
    def swa_tokens(self):
        return self.swa_pages * self.page_size


class DeepseekV4TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """Pages have one request owner; no prefix sharing, no sub-page release.

    All returned addresses are rank-local ORIGINAL token slots, starting at P.
    Thus loc//4 and loc//128 address the corresponding flattened compressed
    buffers. SWA has the same page size but its own free list and physical IDs.
    free_swa accepts whole consumed pages (all written slots in each page),
    clears mappings, and leaves history intact. It is the lifecycle caller's
    responsibility to retain pages still needed by any query in a long chunk.

    Failed allocation/estimation has no effects. backup_state/restore_state is
    available for the runtime's wider request-slot/length/mapping transaction; this object
    does not mutate ReqToTokenPool or host request lengths itself.
    """

    def __init__(self, kvcache):
        super().__init__(kvcache.size, kvcache.page_size, kvcache, kvcache.dp_size)
        self.size_swa = kvcache.size_swa
        self.pages_per_rank = self.size_per_rank // self.page_size
        self.swa_pages_per_rank = self.size_swa // self.dp_size // self.page_size
        self._mapping = [
            np.zeros(self.size_per_rank + self.page_size, np.int32) for _ in range(self.dp_size)
        ]
        self.full_to_swa_index_mapping = self._mapping[0] if self.dp_size == 1 else self._mapping
        self._ends = [np.zeros(self.pages_per_rank + 1, np.int32) for _ in range(self.dp_size)]
        self._swa_page = [np.zeros(self.pages_per_rank + 1, np.int32) for _ in range(self.dp_size)]
        self.free_pages = [None] * self.dp_size
        self._swa_free = [None] * self.dp_size
        self.free_group = [[] for _ in range(self.dp_size)]
        self.clear()

    @property
    def size_full(self):
        return self.size

    def _rank(self, dp_rank):
        if not 0 <= dp_rank < self.dp_size:
            raise ValueError("invalid DP rank")

    def full_available_size(self, dp_rank=0):
        self._rank(dp_rank)
        return len(self.free_pages[dp_rank]) * self.page_size

    def swa_available_size(self, dp_rank=0):
        self._rank(dp_rank)
        return len(self._swa_free[dp_rank]) * self.page_size

    def available_size(self, dp_rank=0):
        # For exact partial-page admission, use estimate_extend/decode instead.
        return min(self.full_available_size(dp_rank), self.swa_available_size(dp_rank))

    def _plan(self, prefix_lens, seq_lens, last_loc, extend_num_tokens, dp_rank):
        self._rank(dp_rank)
        vectors = [np.asarray(v) for v in (prefix_lens, seq_lens, last_loc)]
        if any(v.ndim != 1 or (v.size and v.dtype.kind not in "iu") for v in vectors):
            raise ValueError("lengths/locations must be one-dimensional integer arrays")
        prefix, seq, last = [v.astype(np.int64) for v in vectors]
        if (
            not (prefix.shape == seq.shape == last.shape)
            or np.any(prefix < 0)
            or np.any(seq < prefix)
        ):
            raise ValueError("invalid extend lengths")
        if int(np.sum(seq - prefix)) != extend_num_tokens:
            raise ValueError("extend_num_tokens does not match lengths")
        p = self.page_size
        # Symbolic pages let estimation and allocation use the exact same plan
        # even when demand exceeds current capacity. Negative IDs denote new pages.
        segments, new_count, seen_tail = [], 0, set()
        for pre, end, tail in zip(prefix.tolist(), seq.tolist(), last.tolist()):
            count = end - pre
            if not count:
                continue
            if pre:
                page, offset = divmod(tail, p)
                if (
                    page <= 0
                    or page > self.pages_per_rank
                    or offset != (pre - 1) % p
                    or self._ends[dp_rank][page] != offset + 1
                    or page in seen_tail
                ):
                    raise ValueError("last_loc must identify a distinct live request tail")
                seen_tail.add(page)
                if pre % p:
                    take = min(count, p - pre % p)
                    segments.append((page, pre % p, take))
                    count -= take
            elif tail not in (-1, 0):
                raise ValueError("empty prefix must use padding last_loc -1 or 0")
            while count:
                new_count += 1
                take = min(count, p)
                segments.append((-new_count, 0, take))
                count -= take
        swa_needed = sum(page < 0 or self._swa_page[dp_rank][page] == 0 for page, _, _ in segments)
        demand = DeepseekV4AllocationDemand(new_count, swa_needed, p)
        return demand, segments

    def estimate_extend(self, prefix_lens, seq_lens, last_loc, extend_num_tokens, dp_rank=0):
        return self._plan(prefix_lens, seq_lens, last_loc, extend_num_tokens, dp_rank)[0]

    def estimate_decode(self, seq_lens, last_loc, dp_rank=0):
        return self.estimate_extend(
            np.asarray(seq_lens) - 1, seq_lens, last_loc, len(seq_lens), dp_rank
        )

    def can_allocate(self, demand, dp_rank=0):
        return demand.history_tokens <= self.full_available_size(
            dp_rank
        ) and demand.swa_tokens <= self.swa_available_size(dp_rank)

    def alloc_extend(self, prefix_lens, seq_lens, last_loc, extend_num_tokens, dp_rank=0):
        demand, segments = self._plan(prefix_lens, seq_lens, last_loc, extend_num_tokens, dp_rank)
        if not self.can_allocate(demand, dp_rank):
            return None
        pages = self.free_pages[dp_rank][: demand.history_pages]
        swa_pages = iter(self._swa_free[dp_rank][: demand.swa_pages])
        out, writes = [], []
        for page, offset, count in segments:
            if page < 0:
                page = int(pages[-page - 1])
            swa = int(self._swa_page[dp_rank][page]) or int(next(swa_pages))
            loc = page * self.page_size + np.arange(offset, offset + count, dtype=np.int32)
            swa_loc = swa * self.page_size + np.arange(offset, offset + count, dtype=np.int32)
            out.append(loc)
            writes.append((page, swa, offset + count, loc, swa_loc))
        result = np.concatenate(out) if out else np.empty(0, np.int32)
        # Every operation that can reject inputs or capacity occurs above. Commit
        # the new ledger and mappings together; never roll back by freeing tails.
        self.free_pages[dp_rank] = self.free_pages[dp_rank][demand.history_pages :]
        self._swa_free[dp_rank] = self._swa_free[dp_rank][demand.swa_pages :]
        for page, swa, end, loc, swa_loc in writes:
            self._ends[dp_rank][page] = end
            self._swa_page[dp_rank][page] = swa
            self._mapping[dp_rank][loc] = swa_loc
        return result

    def alloc_decode(self, seq_lens, last_loc, dp_rank=0):
        return self.alloc_extend(
            np.asarray(seq_lens) - 1, seq_lens, last_loc, len(seq_lens), dp_rank
        )

    def alloc(self, need_size, dp_rank=0):
        if need_size < 0 or need_size % self.page_size:
            raise ValueError("alloc size must be nonnegative and page aligned")
        return self.alloc_extend([0], [need_size], [-1], need_size, dp_rank)

    def _release_pages(self, indices, dp_rank, swa_only):
        self._rank(dp_rank)
        indices = np.asarray(indices)
        if indices.ndim != 1 or (indices.size and indices.dtype.kind not in "iu"):
            raise ValueError("free indices must be a one-dimensional integer array")
        if np.any(indices < self.page_size) or np.any(indices >= len(self._mapping[dp_rank])):
            raise ValueError("cannot free padding or out-of-range locations")
        pages = np.unique(indices // self.page_size).astype(np.int32)
        # Validate the entire release before changing either ledger. A repeated
        # release before reuse is harmless. Stale addresses after reuse are not
        # handles: the lifecycle caller must clear the request owner.
        for page in pages:
            end = self._ends[dp_rank][page]
            if swa_only:
                occupied = np.flatnonzero(
                    self._mapping[dp_rank][page * self.page_size : (page + 1) * self.page_size]
                )
            else:
                occupied = np.arange(end)
            requested = np.unique(indices[indices // self.page_size == page] % self.page_size)
            if not np.all(np.isin(occupied, requested)):
                raise ValueError("partial page release would free live tokens")
        return pages

    def free_swa(self, free_index, dp_rank=0):
        pages = self._release_pages(free_index, dp_rank, True)
        self._free_swa_pages(pages, dp_rank)

    def _free_swa_pages(self, pages, rank):
        physical = self._swa_page[rank][pages]
        physical = physical[physical != 0]
        self._swa_free[rank] = np.sort(np.concatenate((self._swa_free[rank], physical)))
        self._swa_page[rank][pages] = 0
        for page in pages:
            self._mapping[rank][page * self.page_size : (page + 1) * self.page_size] = 0

    def free(self, free_index, dp_rank=0):
        if not self.is_not_in_free_group:
            self._rank(dp_rank)
            self.free_group[dp_rank].append(np.asarray(free_index).copy())
            return
        pages = self._release_pages(free_index, dp_rank, False)
        live = pages[self._ends[dp_rank][pages] != 0]
        self._free_swa_pages(live, dp_rank)
        self._ends[dp_rank][live] = 0
        self.free_pages[dp_rank] = np.sort(np.concatenate((self.free_pages[dp_rank], live)))

    def count_swa_mapped(self, indices, dp_rank=0):
        self._rank(dp_rank)
        return int(np.count_nonzero(self._mapping[dp_rank][indices]))

    def clear(self, dp_rank=None):
        if dp_rank is not None:
            self._rank(dp_rank)
        for rank in range(self.dp_size) if dp_rank is None else [dp_rank]:
            self.free_pages[rank] = np.arange(1, self.pages_per_rank + 1, dtype=np.int32)
            self._swa_free[rank] = np.arange(1, self.swa_pages_per_rank + 1, dtype=np.int32)
            self._mapping[rank].fill(0)
            self._ends[rank].fill(0)
            self._swa_page[rank].fill(0)
            self.free_group[rank] = []
        self.is_not_in_free_group = True

    def backup_state(self):
        return deepcopy(
            (
                self.free_pages,
                self._swa_free,
                self._mapping,
                self._ends,
                self._swa_page,
                self.free_group,
                self.is_not_in_free_group,
            )
        )

    def restore_state(self, state):
        free, swa_free, mapping, ends, swa_page, group, outside_group = deepcopy(state)
        self.free_pages, self._swa_free = free, swa_free
        # Keep mapping references held by runtime consumers valid across a rollback.
        for old, saved in zip(self._mapping, mapping):
            old[:] = saved
        self._ends, self._swa_page = ends, swa_page
        self.free_group, self.is_not_in_free_group = group, outside_group

    def free_group_begin(self):
        if not self.is_not_in_free_group:
            raise ValueError("nested free groups are unsupported")
        self.is_not_in_free_group = False

    def free_group_end(self):
        groups = [np.concatenate(g) if g else np.empty(0, np.int32) for g in self.free_group]
        for rank, indices in enumerate(groups):
            self._release_pages(indices, rank, False)
        self.is_not_in_free_group = True
        for rank, indices in enumerate(groups):
            self.free(indices, rank)
        self.free_group = [[] for _ in range(self.dp_size)]

    def debug_print(self):
        return (
            f"V4 history={self.full_available_size()} SWA={self.swa_available_size()} token slots"
        )
