"""Host-only layout planning shared by serving and compilation inputs."""

from dataclasses import dataclass

import numpy as np

from sgl_jax.srt.utils.common_utils import pad_to_bucket


def _readonly(array):
    array.flags.writeable = False
    return array


def serving_shape_buckets(mode, token_buckets, request_buckets, cache_buckets):
    """Keep runtime and warmup on the same supported shape combinations.

    Extend still uses the maximum request/cache capacities. Changing that
    policy is separate from the layout refactor (issue #1724).
    """
    if mode.is_decode_or_idle():
        return request_buckets, request_buckets, cache_buckets
    return token_buckets, request_buckets[-1:], cache_buckets[-1:]


@dataclass(frozen=True, eq=False)
class SequenceLayout:
    """Compact request-major lengths and offsets; no dense B-by-S storage.

    Request order is rank-then-request. Token/page starts are rank-local, as
    required by the attention and logits kernels. Arrays own their snapshot.
    """

    request_offsets: np.ndarray
    kv_lengths: np.ndarray
    prefix_lengths: np.ndarray
    query_lengths: np.ndarray
    query_starts: np.ndarray
    page_counts: np.ndarray
    page_starts: np.ndarray

    @classmethod
    def create(cls, seq_lens, prefix_lens, query_lens, page_size):
        if page_size < 1:
            raise ValueError("page_size must be positive")
        counts = tuple(len(x) for x in seq_lens)
        offsets = np.cumsum((0, *counts), dtype=np.int32)
        kv = np.concatenate(seq_lens).astype(np.int32, copy=False)
        # Decode has one query token per request. Derive these arrays once
        # across all ranks instead of allocating temporary arrays per rank.
        prefix = (
            kv - 1
            if prefix_lens is None
            else np.concatenate(prefix_lens).astype(np.int32, copy=False)
        )
        query = (
            kv - prefix
            if query_lens is None
            else np.concatenate(query_lens).astype(np.int32, copy=False)
        )
        if kv.shape != prefix.shape or kv.shape != query.shape:
            raise ValueError("Sequence lengths must have matching request counts")
        pages = (kv + page_size - 1) // page_size
        query_starts = np.empty_like(query)
        page_starts = np.empty(len(pages), dtype=np.int64)
        for start, end in zip(offsets[:-1], offsets[1:]):
            if start == end:
                continue
            query_starts[start] = page_starts[start] = 0
            query[start : end - 1].cumsum(out=query_starts[start + 1 : end])
            pages[start : end - 1].cumsum(out=page_starts[start + 1 : end])
        return cls(*map(_readonly, (offsets, kv, prefix, query, query_starts, pages, page_starts)))

    def requests(self, rank):
        return slice(int(self.request_offsets[rank]), int(self.request_offsets[rank + 1]))


@dataclass(frozen=True, eq=False)
class BatchLayoutPlan:
    """One authority for DP capacities, rank slices and output selection."""

    request_counts: tuple[int, ...]
    token_counts: tuple[int, ...]
    request_capacity: int
    token_capacity: int
    sequences: SequenceLayout | None = None

    def __post_init__(self):
        if not self.request_counts or len(self.request_counts) != len(self.token_counts):
            raise ValueError("Request/token counts must describe the same nonempty DP mesh")
        for capacity, counts in (
            (self.request_capacity, self.request_counts),
            (self.token_capacity, self.token_counts),
        ):
            if capacity % self.dp_size or min(counts) < 0 or max(counts) > capacity // self.dp_size:
                raise ValueError(f"Capacity {capacity} cannot contain DP counts {counts}")

    @classmethod
    def from_buckets(cls, request_counts, token_counts, request_buckets, token_buckets, **kwargs):
        dp = len(request_counts)
        bs, _ = pad_to_bucket(max(request_counts) * dp, request_buckets)
        tokens, _ = pad_to_bucket(max(token_counts) * dp, token_buckets)
        return cls(tuple(request_counts), tuple(token_counts), bs, tokens, **kwargs)

    @property
    def dp_size(self):
        return len(self.request_counts)

    @property
    def requests_per_rank(self):
        return self.request_capacity // self.dp_size

    @property
    def tokens_per_rank(self):
        return self.token_capacity // self.dp_size

    @property
    def real_requests(self):
        return sum(self.request_counts)

    @property
    def real_tokens(self):
        return sum(self.token_counts)

    def request_slice(self, rank, *, padded=False):
        start = rank * self.requests_per_rank
        return slice(
            start, start + (self.requests_per_rank if padded else self.request_counts[rank])
        )

    def token_slice(self, rank, *, padded=False):
        start = rank * self.tokens_per_rank
        return slice(start, start + (self.tokens_per_rank if padded else self.token_counts[rank]))

    def request_selector(self):
        indices = np.empty(self.real_requests, dtype=np.int32)
        offset = 0
        for rank, count in enumerate(self.request_counts):
            slots = self.request_slice(rank)
            indices[offset : offset + count] = np.arange(slots.start, slots.stop, dtype=np.int32)
            offset += count
        return _readonly(indices)
