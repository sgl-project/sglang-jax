"""Correctness tests for NativeAttention masking under a paged KV layout.

Both mask paths are covered: ``_apply_decode_mask`` and ``_apply_extend_mask``. They share
the key axis -- one ``_merge_cache_loc``, same page-aligned blocks -- and had the same
defect, so the extend class mirrors the decode one case for case.

``cache_loc`` is built page-aligned per request (``ScheduleBatch._merge_cache_loc``):
each request occupies ``ceil(seq_len / page_size) * page_size`` slots and the per-request
offsets are ``cumsum(aligned_lens)``. ``native_backend._apply_decode_mask`` however
reconstructs the per-sequence key ranges from a dense ``cumsum(seq_lens)``.

For ``page_size > 1`` with more than one running sequence the mask boundaries therefore
drift off the real key positions. From the first sequence that leaves page padding behind,
every later query attends to a neighbour's page-tail slots -- which hold valid slot indices
(page ids start at 1, so ``loc > 0`` does not remove them) -- and misses some of its own
keys, including the most recent one.

The drift for sequence ``i`` is the total page padding of the sequences before it, so the
failure condition is exactly: some sequence other than the last has a length that is not a
multiple of ``page_size``. Tests whose padding totals are all zero (``page_size == 1``,
batch of one, exact page multiples, padding only on the last sequence) must keep passing --
they are the regression baseline for the fix.

Each test compares ``forward_attention`` against an independent per-sequence full-attention
reference across page sizes, sequence-length combinations, MHA / GQA / MQA head counts, the
sliding-window bound (which is derived from the same block offsets) and the batch-size /
cache_loc bucket padding that serving always applies.

Runtime note: cost here is XLA compilation, and it is driven by the number of distinct
``(num_queries, num_keys)`` shapes -- a new key count costs ~2s while a new head count costs
~0.15s and a repeated shape is free. The KV pool is therefore a fixed size (as it is in
serving, where it is allocated once at startup) and the sweeps vary one factor at a time:
sequence lengths at a single head config, head configs at a single sequence config.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from sgl_jax.srt.layers.attention.native_backend import forward_attention
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

HEAD_DIM = 8
# (num_heads, num_kv_heads): MHA, GQA 2:1, MQA 4:1
HEAD_CONFIGS = ((4, 4), (4, 2), (4, 1))
# Fixed like the serving KV pool, so k_cache/v_cache keep one shape across configs.
POOL_SLOTS = 2048
# Head config used when sweeping sequence lengths (the bug is head-agnostic).
DEFAULT_HEADS = (4, 2)


def _cpu_mesh() -> Mesh:
    return Mesh(np.array(jax.devices()).reshape(1, 1), ("data", "tensor"))


def _build_paged_fixture(page_size, seq_lens, num_heads, num_kv_heads, seed=0, extend_lens=None):
    """Hand-build the inputs ``forward_attention`` receives in serving.

    ``extend_lens=None`` builds a decode batch (one query row per request); passing it
    builds an extend batch (``extend_lens[i]`` query rows for request ``i``). The KV side
    is identical either way -- both paths go through the same ``_merge_cache_loc``.

    The req_to_token ledger mirrors PagedTokenToKVPoolAllocator: slot indices inside a
    page are consecutive (``req_to_token[i, p * ps + j] == req_to_token[i, p * ps] + j``)
    and page ids start at 1, because page 0 is never allocated -- that is what makes
    ``loc > 0`` a meaningful "real slot" test in forward_attention.

    The whole KV pool is filled with non-zero values on purpose: in production the
    allocated-but-unwritten page-tail slots hold stale KV from a previous request, so
    zero-filling them here would mask the very reads this test is meant to catch.
    """
    rng = np.random.default_rng(seed)
    num_reqs = len(seq_lens)
    # Decode contributes one query row per request; extend contributes extend_lens[i].
    num_queries = num_reqs if extend_lens is None else int(sum(extend_lens))
    pages_per_req = max(-(-int(length) // page_size) for length in seq_lens)
    max_ctx = pages_per_req * page_size
    assert num_reqs * max_ctx <= POOL_SLOTS, "fixture does not fit the fixed KV pool"

    page_ids = 1 + rng.permutation(POOL_SLOTS // page_size - 1)[: num_reqs * pages_per_req]
    req_to_token = (
        page_ids.reshape(num_reqs, pages_per_req)[:, :, None] * page_size + np.arange(page_size)
    ).reshape(num_reqs, max_ctx)
    req_to_token = req_to_token.astype(np.int32)

    aligned_lens = [-(-int(length) // page_size) * page_size for length in seq_lens]
    cache_loc = np.concatenate(
        [req_to_token[i, :aligned] for i, aligned in enumerate(aligned_lens)]
    ).astype(np.int32)

    return {
        "q": rng.normal(size=(num_queries, num_heads, HEAD_DIM)).astype(np.float32),
        "k_cache": rng.normal(size=(POOL_SLOTS, num_kv_heads, HEAD_DIM)).astype(np.float32),
        "v_cache": rng.normal(size=(POOL_SLOTS, num_kv_heads, HEAD_DIM)).astype(np.float32),
        "seq_lens": np.asarray(seq_lens, dtype=np.int32),
        "cache_loc": cache_loc,
        "own_slots": [req_to_token[i, : int(length)] for i, length in enumerate(seq_lens)],
        "extend_lens": None if extend_lens is None else np.asarray(extend_lens, dtype=np.int32),
    }


def _reference_decode_attention(fixture, num_heads, num_kv_heads, scale, sliding_window_size=None):
    """Per-sequence full attention over only that sequence's own KV slots.

    Deliberately independent of the backend: no flat concatenation, no mask, no notion of
    pages. Query head ``h`` reads kv head ``h // (num_heads // num_kv_heads)``, spelled out
    as an explicit loop rather than a repeat() so the head mapping is independent too.
    A sliding window simply keeps the last ``sliding_window_size`` keys of the sequence.
    """
    copies = num_heads // num_kv_heads
    num_reqs = len(fixture["own_slots"])
    out = np.zeros((num_reqs, num_heads, HEAD_DIM), dtype=np.float32)
    for i, slots in enumerate(fixture["own_slots"]):
        if sliding_window_size is not None:
            slots = slots[-sliding_window_size:]
        keys = fixture["k_cache"][slots]
        values = fixture["v_cache"][slots]
        for h in range(num_heads):
            kv_head = h // copies
            logits = (fixture["q"][i][h] @ keys[:, kv_head, :].T) * scale
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            out[i, h] = probs @ values[:, kv_head, :]
    return out.reshape(num_reqs, num_heads * HEAD_DIM)


def _reference_extend_attention(fixture, num_heads, num_kv_heads, scale, sliding_window_size=None):
    """Per-sequence causal attention over only that sequence's own KV slots.

    Same independence rules as the decode reference: no flat concatenation, no mask, no
    cumsum, no notion of pages -- a shared prefix-sum here would reproduce the very bug
    under test on both sides and the comparison would pass regardless.

    Row ``r`` of request ``i`` is the token at absolute position ``prefix_len_i + r``, so it
    sees its own keys ``[0, pos]``; a sliding window further trims that to the last
    ``sliding_window_size`` of them.
    """
    copies = num_heads // num_kv_heads
    extend_lens = fixture["extend_lens"]
    out = np.zeros((int(extend_lens.sum()), num_heads, HEAD_DIM), dtype=np.float32)
    row = 0
    for i, slots in enumerate(fixture["own_slots"]):
        keys = fixture["k_cache"][slots]
        values = fixture["v_cache"][slots]
        prefix_len = len(slots) - int(extend_lens[i])
        for r in range(int(extend_lens[i])):
            pos = prefix_len + r
            lo = 0 if sliding_window_size is None else max(0, pos + 1 - sliding_window_size)
            for h in range(num_heads):
                kv_head = h // copies
                logits = (fixture["q"][row][h] @ keys[lo : pos + 1, kv_head, :].T) * scale
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                out[row, h] = probs @ values[lo : pos + 1, kv_head, :]
            row += 1
    return out.reshape(-1, num_heads * HEAD_DIM)


class TestNativeAttentionPagedDecode(unittest.TestCase):
    def _assert_matches_reference(
        self,
        page_size,
        seq_lens,
        num_heads,
        num_kv_heads,
        seed=0,
        sliding_window_size=None,
        bs_bucket=None,
        loc_bucket=None,
    ):
        fixture = _build_paged_fixture(page_size, seq_lens, num_heads, num_kv_heads, seed)
        scale = 1.0 / np.sqrt(HEAD_DIM)
        num_reqs = len(seq_lens)
        q, cache_loc = fixture["q"], fixture["cache_loc"]
        if bs_bucket is not None:
            pad_rows = bs_bucket - num_reqs
            q = np.concatenate([q, np.zeros((pad_rows, num_heads, HEAD_DIM), np.float32)])
        if loc_bucket is not None:
            # The host cache_loc buffer is deliberately not re-zeroed between steps, so its
            # tail carries in-bounds slot indices left by earlier batches -- all > 0, which
            # means the `loc > 0` validity mask cannot remove them.
            stale = np.arange(101, 101 + loc_bucket - len(cache_loc), dtype=np.int32)
            cache_loc = np.concatenate([cache_loc, stale])

        got = forward_attention(
            jnp.asarray(q),
            jnp.asarray(fixture["k_cache"]),
            jnp.asarray(fixture["v_cache"]),
            jnp.asarray(fixture["seq_lens"]),
            jnp.asarray(cache_loc),
            None,  # extend_prefix_lens
            None,  # extend_seq_lens
            num_heads,
            num_kv_heads,
            scale,
            False,  # is_causal: a decode query attends over its whole prefix
            ForwardMode.DECODE,
            None,  # kv_sharding
            page_size=page_size,
            mesh=_cpu_mesh(),
            sliding_window_size=sliding_window_size,
        )
        expected = _reference_decode_attention(
            fixture, num_heads, num_kv_heads, scale, sliding_window_size
        )
        got = np.asarray(got)
        context = (
            f"page_size={page_size} seq_lens={list(seq_lens)} "
            f"heads={num_heads}/{num_kv_heads} window={sliding_window_size} "
            f"bs_bucket={bs_bucket} loc_bucket={loc_bucket}"
        )
        np.testing.assert_allclose(got[:num_reqs], expected, rtol=1e-5, atol=1e-5, err_msg=context)
        # Padding query rows are fully masked, so softmax degenerates to a uniform average
        # over every gathered value. That is discarded downstream, but it must stay finite:
        # a NaN here would poison the sampler rather than just produce a junk row.
        self.assertTrue(np.isfinite(got[num_reqs:]).all(), f"non-finite padding rows: {context}")

    def test_multi_sequence_unequal_lengths(self):
        """Several running sequences whose lengths are not page multiples."""
        for page_size, seq_lens in (
            (4, (3, 2)),  # minimal repro: seq1's keys sit at [4, 6), mask picks [3, 5)
            (4, (3, 4)),  # mirror of test_padding_only_on_last_sequence's (4, 3)
            (4, (5, 4)),  # leading sequence spans two pages
            (16, (7, 1, 9)),  # three sequences: drift accumulates
            (16, (1, 1, 1)),  # padding dominates every sequence
            (256, (100, 100)),  # the page size the bug report used
        ):
            with self.subTest(page_size=page_size, seq_lens=seq_lens):
                self._assert_matches_reference(page_size, seq_lens, *DEFAULT_HEADS)

    def test_head_configurations(self):
        """The drift is on the key axis, so it must reproduce for MHA, GQA and MQA alike."""
        for num_heads, num_kv_heads in HEAD_CONFIGS:
            with self.subTest(num_heads=num_heads, num_kv_heads=num_kv_heads):
                self._assert_matches_reference(4, (3, 2), num_heads, num_kv_heads)

    def test_page_size_1_is_dense(self):
        """page_size == 1 leaves no page padding, so the dense assumption holds."""
        for num_heads, num_kv_heads in HEAD_CONFIGS:
            with self.subTest(num_heads=num_heads, num_kv_heads=num_kv_heads):
                self._assert_matches_reference(1, (3, 2, 7), num_heads, num_kv_heads)

    def test_single_sequence(self):
        """A batch of one starts at offset 0, so no drift is possible."""
        for page_size in (4, 256):
            with self.subTest(page_size=page_size):
                self._assert_matches_reference(page_size, (3,), *DEFAULT_HEADS)

    def test_exact_page_multiples(self):
        """Every length an exact page multiple: aligned_lens == seq_lens."""
        self._assert_matches_reference(4, (4, 8, 4), *DEFAULT_HEADS)

    def test_sliding_window(self):
        """The SWA bound is derived from seq_ends, so it drifts with the block offsets too."""
        for window in (2, 8):  # 8 exceeds every sequence here: the window must be a no-op
            with self.subTest(window=window):
                self._assert_matches_reference(
                    4, (3, 2), *DEFAULT_HEADS, sliding_window_size=window
                )

    def test_bucket_padding(self):
        """Serving pads q to a batch-size bucket and cache_loc to a length bucket.

        Neither may disturb the real sequences. The cache_loc tail is the interesting half:
        it holds stale but valid slot indices, so only the mask keeps it out -- `loc > 0`
        lets it through.
        """
        for bs_bucket, loc_bucket in ((4, None), (4, 16), (2, 16), (8, 32)):
            with self.subTest(bs_bucket=bs_bucket, loc_bucket=loc_bucket):
                self._assert_matches_reference(
                    4, (3, 2), *DEFAULT_HEADS, bs_bucket=bs_bucket, loc_bucket=loc_bucket
                )

    def test_padding_only_on_last_sequence(self):
        """Padding behind the last sequence shifts nobody: this must pass either way."""
        self._assert_matches_reference(4, (4, 3), *DEFAULT_HEADS)
        self._assert_matches_reference(4, (8, 4, 1), *DEFAULT_HEADS)


class TestNativeAttentionPagedExtend(unittest.TestCase):
    """The extend half of the same defect.

    Extend shares the key axis with decode -- one ``_merge_cache_loc``, same page-aligned
    blocks -- and differs only on the query axis, where each request contributes
    ``extend_seq_lens[i]`` rows instead of one. ``_apply_extend_mask`` reconstructed the key
    blocks from a dense ``cumsum(seq_lens)`` and bounded validity with a single
    ``arange < sum(seq_lens)``, so it drifted exactly like the decode mask did, with the
    same necessary-and-sufficient condition: a request errs iff the page padding ahead of
    it is non-zero.

    The scalar bound is the part decode does not have. Block starts alone are not enough:
    ``k_batch_ids`` comes from a non-decreasing cumsum and so has no "unowned" value --
    every column, page tails included, is assigned to some request. Only a per-column end
    excludes them.
    """

    def _assert_matches_reference(
        self,
        page_size,
        seq_lens,
        extend_lens,
        num_heads,
        num_kv_heads,
        seed=0,
        sliding_window_size=None,
        token_bucket=None,
        loc_bucket=None,
    ):
        assert all(e <= s for e, s in zip(extend_lens, seq_lens))
        fixture = _build_paged_fixture(
            page_size, seq_lens, num_heads, num_kv_heads, seed, extend_lens=extend_lens
        )
        scale = 1.0 / np.sqrt(HEAD_DIM)
        num_queries = int(sum(extend_lens))
        prefix_lens = np.asarray(seq_lens, np.int32) - np.asarray(extend_lens, np.int32)
        q, cache_loc = fixture["q"], fixture["cache_loc"]
        if token_bucket is not None:
            pad_rows = token_bucket - num_queries
            q = np.concatenate([q, np.zeros((pad_rows, num_heads, HEAD_DIM), np.float32)])
        if loc_bucket is not None:
            # Same stale-tail construction as the decode test: the host cache_loc buffer is
            # never re-zeroed, so its tail holds in-bounds slot indices from earlier batches.
            stale = np.arange(101, 101 + loc_bucket - len(cache_loc), dtype=np.int32)
            cache_loc = np.concatenate([cache_loc, stale])

        got = forward_attention(
            jnp.asarray(q),
            jnp.asarray(fixture["k_cache"]),
            jnp.asarray(fixture["v_cache"]),
            jnp.asarray(fixture["seq_lens"]),
            jnp.asarray(cache_loc),
            jnp.asarray(prefix_lens),
            jnp.asarray(np.asarray(extend_lens, np.int32)),
            num_heads,
            num_kv_heads,
            scale,
            True,  # is_causal: extend queries are ordered within their own sequence
            ForwardMode.EXTEND,
            None,  # kv_sharding
            page_size=page_size,
            mesh=_cpu_mesh(),
            sliding_window_size=sliding_window_size,
        )
        expected = _reference_extend_attention(
            fixture, num_heads, num_kv_heads, scale, sliding_window_size
        )
        got = np.asarray(got)
        context = (
            f"page_size={page_size} seq_lens={list(seq_lens)} extend_lens={list(extend_lens)} "
            f"heads={num_heads}/{num_kv_heads} window={sliding_window_size} "
            f"token_bucket={token_bucket} loc_bucket={loc_bucket}"
        )
        np.testing.assert_allclose(
            got[:num_queries], expected, rtol=1e-5, atol=1e-5, err_msg=context
        )
        self.assertTrue(np.isfinite(got[num_queries:]).all(), f"non-finite padding rows: {context}")

    def test_multi_sequence_unequal_lengths(self):
        """Several requests whose lengths are not page multiples."""
        for page_size, seq_lens, extend_lens in (
            (4, (3, 2), (3, 2)),  # minimal repro: cold prefill of two prompts
            (16, (7, 1, 9), (7, 1, 9)),  # three requests: drift accumulates
            (256, (100, 100), (100, 100)),  # the page size the bug report used
        ):
            with self.subTest(page_size=page_size, seq_lens=seq_lens):
                self._assert_matches_reference(page_size, seq_lens, extend_lens, *DEFAULT_HEADS)

    def test_prefix_lens(self):
        """extend_prefix_lens is the axis decode does not have.

        A page-aligned prefix is what a radix-cache hit produces (match_prefix truncates to
        a page boundary); an unaligned one is what a continued chunk produces, because
        cache_unfinished_req hands back the tree-owned page-aligned part plus the
        request-owned tail. Both must place the causal diagonal at the absolute position.
        """
        for seq_lens, extend_lens, note in (
            ((9, 5), (3, 5), "unaligned prefix 6 + no prefix"),
            ((8, 5), (4, 5), "aligned prefix 4 + no prefix"),
            ((6, 5, 7), (2, 5, 3), "prefix > 0 mixed with prefix == 0"),
        ):
            with self.subTest(note=note):
                self._assert_matches_reference(4, seq_lens, extend_lens, *DEFAULT_HEADS)

    def test_head_configurations(self):
        """The drift is on the key axis, so it must reproduce for MHA, GQA and MQA alike."""
        for num_heads, num_kv_heads in HEAD_CONFIGS:
            with self.subTest(num_heads=num_heads, num_kv_heads=num_kv_heads):
                self._assert_matches_reference(4, (3, 2), (3, 2), num_heads, num_kv_heads)

    def test_page_size_1_is_dense(self):
        """page_size == 1 leaves no page padding, so the dense assumption holds."""
        for num_heads, num_kv_heads in HEAD_CONFIGS:
            with self.subTest(num_heads=num_heads, num_kv_heads=num_kv_heads):
                self._assert_matches_reference(1, (3, 2, 7), (3, 2, 4), num_heads, num_kv_heads)

    def test_single_sequence(self):
        """A batch of one starts at offset 0, so no drift is possible."""
        for page_size, seq_lens, extend_lens in ((4, (3,), (3,)), (4, (9,), (3,))):
            with self.subTest(page_size=page_size, seq_lens=seq_lens):
                self._assert_matches_reference(page_size, seq_lens, extend_lens, *DEFAULT_HEADS)

    def test_exact_page_multiples(self):
        """Every length an exact page multiple: aligned_lens == seq_lens."""
        self._assert_matches_reference(4, (4, 8, 4), (4, 4, 4), *DEFAULT_HEADS)

    def test_padding_only_on_last_sequence(self):
        """Padding behind the last request shifts nobody: this must pass either way."""
        self._assert_matches_reference(4, (4, 3), (2, 3), *DEFAULT_HEADS)
        self._assert_matches_reference(4, (8, 4, 1), (4, 4, 1), *DEFAULT_HEADS)

    def test_sliding_window(self):
        """The SWA bound rides on the same block offsets, so it drifts with them."""
        for window in (2, 8):  # 8 exceeds every sequence here: the window must be a no-op
            with self.subTest(window=window):
                self._assert_matches_reference(
                    4, (3, 2), (3, 2), *DEFAULT_HEADS, sliding_window_size=window
                )

    def test_bucket_padding(self):
        """Extend pins bs and cache_loc to the largest bucket and buckets the token count.

        Trailing zero-length requests are therefore the norm, not an edge case, and they
        make q_starts / k_starts repeat -- ``.at[].set(1)`` merges the duplicate markers and
        drops any index that lands past the end of the array.
        """
        for seq_lens, extend_lens, token_bucket, loc_bucket in (
            ((3, 2), (3, 2), 8, None),
            ((3, 2), (3, 2), 8, 16),
            ((3, 2, 0, 0), (3, 2, 0, 0), None, None),
            ((4, 3, 0, 0), (2, 3, 0, 0), 8, 16),
        ):
            with self.subTest(seq_lens=seq_lens, token_bucket=token_bucket):
                self._assert_matches_reference(
                    4,
                    seq_lens,
                    extend_lens,
                    *DEFAULT_HEADS,
                    token_bucket=token_bucket,
                    loc_bucket=loc_bucket,
                )


if __name__ == "__main__":
    unittest.main()
