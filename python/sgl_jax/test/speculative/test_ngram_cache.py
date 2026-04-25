"""Tests for the N-gram trie cache used in speculative decoding."""

import unittest

import numpy as np

from sgl_jax.srt.speculative.ngram_cache import NgramCache

# JAX is only required by ngram_worker.py. Make its import optional so that
# the rest of the tests in this module still run on CPU-only machines.
try:
    import jax  # noqa: F401

    HAS_JAX = True
except ImportError:  # pragma: no cover - exercised on CPU-only machines
    HAS_JAX = False


class TestTrieInsertAndQuery(unittest.TestCase):
    def test_insert_and_exact_match(self):
        cache = NgramCache(max_trie_depth=4)
        tokens = [10, 20, 30, 40, 50]
        cache.insert(tokens)

        draft = cache.query_bfs([30, 40], draft_token_num=4, max_bfs_breadth=3)
        self.assertEqual(draft.tokens[0], 50)

    def test_bfs_breadth_limit(self):
        cache = NgramCache(max_trie_depth=4)
        for suffix_tok in [100, 200, 300, 400, 500]:
            cache.insert([10, 20, suffix_tok])

        draft = cache.query_bfs([10, 20], draft_token_num=4, max_bfs_breadth=3)
        unique_root_children = set(draft.tokens[:3].tolist())
        self.assertLessEqual(len(unique_root_children), 3)

    def test_frequency_ranking(self):
        cache = NgramCache(max_trie_depth=4)
        for _ in range(5):
            cache.insert([10, 20, 100])
        for _ in range(2):
            cache.insert([10, 20, 200])
        cache.insert([10, 20, 300])

        draft = cache.query_bfs([10, 20], draft_token_num=4, max_bfs_breadth=3)
        self.assertEqual(draft.tokens[0], 100)

    def test_empty_cache_returns_valid_structure(self):
        cache = NgramCache(max_trie_depth=4)
        draft = cache.query_bfs([10, 20], draft_token_num=4, max_bfs_breadth=3)
        self.assertEqual(draft.tokens.shape, (4,))
        self.assertEqual(draft.retrive_index.shape, (4,))
        self.assertEqual(draft.retrive_next_token.shape, (4,))
        self.assertEqual(draft.retrive_next_sibling.shape, (4,))
        self.assertEqual(draft.mask.shape, (4, 4))
        self.assertEqual(draft.parent_idx.shape, (4,))
        # Empty draft uses -1 sentinel tokens
        self.assertTrue(np.all(draft.tokens == -1))

    def test_deep_chain(self):
        cache = NgramCache(max_trie_depth=8)
        tokens = list(range(1, 9))
        cache.insert(tokens)

        draft = cache.query_bfs([1, 2, 3], draft_token_num=4, max_bfs_breadth=1)
        self.assertEqual(draft.tokens[0], 4)
        self.assertEqual(draft.tokens[1], 5)
        self.assertEqual(draft.tokens[2], 6)
        self.assertEqual(draft.tokens[3], 7)

    def test_tree_structure_chain(self):
        cache = NgramCache(max_trie_depth=8)
        cache.insert([1, 2, 3, 4, 5])

        draft = cache.query_bfs([1, 2], draft_token_num=3, max_bfs_breadth=1)
        self.assertEqual(draft.tokens[0], 3)
        self.assertEqual(draft.tokens[1], 4)
        self.assertEqual(draft.tokens[2], 5)

        self.assertEqual(draft.retrive_next_token[0], 1)
        self.assertEqual(draft.retrive_next_token[1], 2)
        self.assertEqual(draft.retrive_next_token[2], -1)

        self.assertEqual(draft.retrive_next_sibling[0], -1)
        self.assertEqual(draft.retrive_next_sibling[1], -1)
        self.assertEqual(draft.retrive_next_sibling[2], -1)

        # Chain: parent_idx should be -1 -> 0 -> 1
        self.assertEqual(draft.parent_idx[0], -1)
        self.assertEqual(draft.parent_idx[1], 0)
        self.assertEqual(draft.parent_idx[2], 1)

    def test_tree_structure_branching(self):
        cache = NgramCache(max_trie_depth=4)
        cache.insert([1, 2, 10, 11])
        cache.insert([1, 2, 20, 21])

        draft = cache.query_bfs([1, 2], draft_token_num=4, max_bfs_breadth=3)
        tokens = draft.tokens.tolist()
        root_children = set(tokens[:2])
        self.assertTrue(10 in root_children or 20 in root_children)
        self.assertTrue(draft.retrive_next_sibling[0] == 1 or draft.retrive_next_sibling[1] == 0)

    def test_mask_ancestry(self):
        cache = NgramCache(max_trie_depth=8)
        cache.insert([1, 2, 3, 4, 5])

        draft = cache.query_bfs([1, 2], draft_token_num=3, max_bfs_breadth=1)
        self.assertTrue(draft.mask[0, 0])
        self.assertTrue(draft.mask[1, 0])
        self.assertTrue(draft.mask[1, 1])
        self.assertTrue(draft.mask[2, 0])
        self.assertTrue(draft.mask[2, 1])
        self.assertTrue(draft.mask[2, 2])

    def test_reset(self):
        cache = NgramCache(max_trie_depth=4)
        cache.insert([1, 2, 3, 4])
        cache.reset()
        draft = cache.query_bfs([1, 2], draft_token_num=4, max_bfs_breadth=3)
        self.assertTrue(np.all(draft.retrive_next_token == -1))

    def test_padding_uses_sentinel_tokens(self):
        """Padded positions must use -1 (unmatchable) not copies of real tokens."""
        cache = NgramCache(max_trie_depth=4)
        cache.insert([10, 20, 30])

        draft = cache.query_bfs([10, 20], draft_token_num=4, max_bfs_breadth=3)
        self.assertEqual(draft.tokens[0], 30)
        self.assertTrue(np.all(draft.tokens[1:] == -1))

    def test_single_suffix_match(self):
        """Even a single-token suffix should produce candidates."""
        cache = NgramCache(max_trie_depth=4)
        cache.insert([5, 10, 15])
        cache.insert([5, 10, 20])

        draft = cache.query_bfs([10], draft_token_num=4, max_bfs_breadth=3)
        real_tokens = set(draft.tokens[:2].tolist())
        self.assertIn(15, real_tokens)
        self.assertIn(20, real_tokens)

    def test_no_match_returns_root_children(self):
        """When suffix doesn't match, fall back to root's children."""
        cache = NgramCache(max_trie_depth=4)
        cache.insert([100, 200])
        cache.insert([100, 300])

        draft = cache.query_bfs([999], draft_token_num=4, max_bfs_breadth=3)
        self.assertEqual(draft.tokens[0], 100)

    def test_max_nodes_cap(self):
        """Trie should stop growing when max_nodes is reached."""
        cache = NgramCache(max_trie_depth=4, max_nodes=5)
        for i in range(100):
            cache.insert([i, i + 1, i + 2, i + 3])
        self.assertLessEqual(cache._node_count, 5)

    def test_parent_idx_branching(self):
        """Parent indices should correctly reflect the tree structure."""
        cache = NgramCache(max_trie_depth=4)
        cache.insert([1, 2, 10])
        cache.insert([1, 2, 20])

        draft = cache.query_bfs([1, 2], draft_token_num=4, max_bfs_breadth=3)
        # Both root children should have parent_idx == -1
        self.assertEqual(draft.parent_idx[0], -1)
        self.assertEqual(draft.parent_idx[1], -1)

    def test_empty_suffix(self):
        """Empty suffix should return root's children."""
        cache = NgramCache(max_trie_depth=4)
        cache.insert([10, 20, 30])
        draft = cache.query_bfs([], draft_token_num=4, max_bfs_breadth=3)
        self.assertEqual(draft.tokens[0], 10)

    def test_numpy_input(self):
        """insert/query should handle numpy arrays correctly."""
        cache = NgramCache(max_trie_depth=4)
        cache.insert(np.array([10, 20, 30, 40], dtype=np.int32))
        draft = cache.query_bfs(
            np.array([20, 30], dtype=np.int64), draft_token_num=2, max_bfs_breadth=3
        )
        self.assertEqual(draft.tokens[0], 40)

    def test_draft_token_num_one(self):
        """Edge case: requesting exactly one draft token."""
        cache = NgramCache(max_trie_depth=4)
        cache.insert([1, 2, 3])
        draft = cache.query_bfs([1, 2], draft_token_num=1, max_bfs_breadth=3)
        self.assertEqual(draft.tokens.shape, (1,))
        self.assertEqual(draft.tokens[0], 3)
        self.assertEqual(draft.mask.shape, (1, 1))


class TestSpecInfoIntegration(unittest.TestCase):
    @unittest.skipUnless(HAS_JAX, "jax is not installed")
    def test_ngram_algorithm_enum(self):
        from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

        algo = SpeculativeAlgorithm.from_string("NGRAM")
        self.assertEqual(algo, SpeculativeAlgorithm.NGRAM)
        self.assertTrue(algo.is_ngram())
        self.assertFalse(algo.is_eagle())
        self.assertFalse(algo.is_none())

    @unittest.skipUnless(HAS_JAX, "jax is not installed")
    def test_ngram_from_string_case_insensitive(self):
        from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

        self.assertEqual(SpeculativeAlgorithm.from_string("ngram"), SpeculativeAlgorithm.NGRAM)
        self.assertEqual(SpeculativeAlgorithm.from_string("Ngram"), SpeculativeAlgorithm.NGRAM)

    @unittest.skipUnless(HAS_JAX, "jax is not installed")
    def test_ngram_verify_input_keeps_allocate_lens_out_of_aux_data(self):
        import jax.numpy as jnp

        from sgl_jax.srt.speculative.ngram_worker import NgramVerifyInput

        spec = NgramVerifyInput(
            custom_mask=jnp.array([1, 0, 1], dtype=jnp.int32),
            draft_token_num=4,
            allocate_lens=np.array([3, 4], dtype=np.int32),
        )
        children, aux_data = spec.tree_flatten()

        self.assertEqual(set(aux_data.keys()), {"draft_token_num"})
        self.assertEqual(aux_data["draft_token_num"], 4)
        self.assertEqual(len(children), 2)
        np.testing.assert_array_equal(np.asarray(children[1]), np.array([3, 4], dtype=np.int32))

    @unittest.skipUnless(HAS_JAX, "jax is not installed")
    def test_ngram_verify_input_bs2_metadata_can_reuse_jit(self):
        import jax
        import jax.numpy as jnp

        from sgl_jax.srt.speculative.ngram_worker import NgramVerifyInput

        @jax.jit
        def f(spec):
            return spec.custom_mask + 1

        spec_a = NgramVerifyInput(
            custom_mask=jnp.array([1, 2], dtype=jnp.int32),
            draft_token_num=4,
            allocate_lens=np.array([3, 4], dtype=np.int32),
        )
        spec_b = NgramVerifyInput(
            custom_mask=jnp.array([1, 2], dtype=jnp.int32),
            draft_token_num=4,
            allocate_lens=np.array([5, 6], dtype=np.int32),
        )

        np.testing.assert_array_equal(np.asarray(f(spec_a)), np.array([2, 3], dtype=np.int32))
        np.testing.assert_array_equal(np.asarray(f(spec_b)), np.array([2, 3], dtype=np.int32))


def _count_node(cache: NgramCache, path: list[int]) -> int:
    """Walk the trie following ``path`` and return the terminal node's count.

    Returns ``0`` when the path is not present.
    """
    node = cache.root
    for tok in path:
        if tok not in node.children:
            return 0
        node = node.children[tok]
    return node.count


class TestInsertNewSuffixes(unittest.TestCase):
    """Tests for BUG E (trie count inflation) fix."""

    def test_insert_new_suffixes_no_double_count(self):
        """Incremental inserts must not re-increment counts for old n-grams."""
        cache = NgramCache(max_trie_depth=4)

        tokens_first = [1, 2, 3, 4, 5]
        cache.insert_new_suffixes(tokens_first, prev_len=0)

        # Record counts for every n-gram ending at positions 0..4.
        baseline: dict[tuple[int, ...], int] = {}
        for p in range(len(tokens_first)):
            for start in range(max(0, p - cache.max_trie_depth + 1), p + 1):
                ngram = tuple(tokens_first[start : p + 1])
                baseline[ngram] = _count_node(cache, list(ngram))

        # Now append a new token and insert only the new tail.
        tokens_extended = tokens_first + [6]
        cache.insert_new_suffixes(tokens_extended, prev_len=len(tokens_first))

        # All previously-seen n-grams must have their counts unchanged.
        for ngram, expected in baseline.items():
            self.assertEqual(
                _count_node(cache, list(ngram)),
                expected,
                f"count for {ngram} changed: expected {expected} " f"after incremental insert",
            )

        # n-grams ending at the new position 5 must have been inserted
        # (count >= 1).
        for start in range(max(0, 5 - cache.max_trie_depth + 1), 6):
            ngram = tuple(tokens_extended[start:6])
            self.assertGreaterEqual(_count_node(cache, list(ngram)), 1)

    def test_insert_is_equivalent_to_insert_new_suffixes_prev0(self):
        """``insert(tokens)`` must equal ``insert_new_suffixes(tokens, 0)``."""
        cache_a = NgramCache(max_trie_depth=4)
        cache_b = NgramCache(max_trie_depth=4)
        tokens = [10, 20, 30, 20, 30, 40, 30, 40, 50]
        cache_a.insert(tokens)
        cache_b.insert_new_suffixes(tokens, prev_len=0)

        self.assertEqual(cache_a._node_count, cache_b._node_count)
        # Walk both tries and compare counts at matching positions.
        for start in range(len(tokens)):
            for end in range(start + 1, min(len(tokens), start + 4) + 1):
                ngram = list(tokens[start:end])
                self.assertEqual(_count_node(cache_a, ngram), _count_node(cache_b, ngram))

    def test_repeated_incremental_insert_no_inflation(self):
        """Incremental inserts across many steps should not grow counts
        for old n-grams."""
        cache = NgramCache(max_trie_depth=3)
        tokens = [7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9]

        # First insertion of just the first 3 tokens.
        cache.insert_new_suffixes(tokens[:3], prev_len=0)
        first_count_78 = _count_node(cache, [7, 8])
        first_count_789 = _count_node(cache, [7, 8, 9])

        # Now grow the sequence one token at a time; the n-gram [7, 8] ending
        # at position 1 must not get re-incremented once we move past it.
        for new_len in range(4, len(tokens) + 1):
            cache.insert_new_suffixes(tokens[:new_len], prev_len=new_len - 1)

        # [7, 8] was inserted 4 times across position-endings (1, 4, 7, 10)
        # over the 12-token sequence.
        self.assertEqual(_count_node(cache, [7, 8]), 4)
        # Count for the 3-gram [7, 8, 9] should match 4 as well (ending at
        # positions 2, 5, 8, 11).
        self.assertEqual(_count_node(cache, [7, 8, 9]), 4)
        # Sanity: the initial counts after the 3-token insert were only 1.
        self.assertEqual(first_count_78, 1)
        self.assertEqual(first_count_789, 1)


class TestBreadthValidation(unittest.TestCase):
    """Tests for BUG C (``max_bfs_breadth > 1`` silently accepted) fix."""

    @unittest.skipUnless(HAS_JAX, "jax is not installed")
    def test_breadth_equal_one_accepted(self):
        """``max_bfs_breadth == 1`` is the only supported chain-mode value."""
        from sgl_jax.srt.speculative.ngram_worker import _ensure_chain_mode

        # Must not raise.
        _ensure_chain_mode(1)

    @unittest.skipUnless(HAS_JAX, "jax is not installed")
    def test_breadth_greater_than_one_raises(self):
        """Constructing an NgramWorker with branching breadth must fail."""
        from sgl_jax.srt.speculative.ngram_worker import _ensure_chain_mode

        for bad_value in (0, 2, 3, 8):
            with self.assertRaises(ValueError) as cm:
                _ensure_chain_mode(bad_value)
            self.assertIn("chain mode", str(cm.exception))


class TestBuildEmptyDraftNoCycle(unittest.TestCase):
    """Tests for BUG G (infinite position walk safety) related fixes."""

    def test_build_empty_draft_parent_idx_all_minus_one(self):
        """``_build_empty_draft`` must never produce a ``parent_idx`` cycle."""
        cache = NgramCache(max_trie_depth=4)
        draft = cache._build_empty_draft(4)
        self.assertEqual(draft.parent_idx.tolist(), [-1, -1, -1, -1])

    def test_query_bfs_empty_trie_produces_no_cycle(self):
        """Querying an empty trie must return an empty draft with no cycles."""
        cache = NgramCache(max_trie_depth=4)
        draft = cache.query_bfs([10, 20], draft_token_num=4, max_bfs_breadth=3)
        self.assertEqual(draft.parent_idx.tolist(), [-1, -1, -1, -1])
        # Walking the ancestry of every node must terminate in <= D hops.
        for i in range(4):
            p = int(draft.parent_idx[i])
            hops = 0
            while p >= 0 and hops < 4:
                p = int(draft.parent_idx[p])
                hops += 1
            self.assertLess(hops, 4, "ancestry walk exceeded draft_token_num hops")

    def test_query_bfs_single_real_node_no_self_loop(self):
        """If only one real node is produced, padded parents must not self-loop."""
        cache = NgramCache(max_trie_depth=4)
        # Insert exactly one 1-gram.
        cache.insert([42])
        draft = cache.query_bfs([], draft_token_num=4, max_bfs_breadth=3)
        # First slot is the real node (root-child), rest are padding.
        self.assertEqual(int(draft.tokens[0]), 42)
        self.assertEqual(int(draft.parent_idx[0]), -1)
        for i in range(1, 4):
            # Padding must attach to a real node index (0), never to itself.
            self.assertEqual(int(draft.tokens[i]), -1)
            self.assertEqual(int(draft.parent_idx[i]), 0)
            self.assertNotEqual(int(draft.parent_idx[i]), i)


class TestPaddingSafeInputIds(unittest.TestCase):
    """Tests for BUG D (padding ``-1`` fed to embedding layer) fix."""

    @unittest.skipUnless(HAS_JAX, "jax is not installed")
    def test_safe_input_ids_replaces_negatives(self):
        """``-1`` sentinels in the draft tokens must be replaced with ``0``."""
        from sgl_jax.srt.speculative.ngram_worker import _make_safe_input_ids

        draft = np.array([5, -1, 7, -1, 9], dtype=np.int32)
        safe = _make_safe_input_ids(draft)

        self.assertEqual(safe.dtype, np.int32)
        self.assertFalse(np.any(safe == -1))
        # Values that were not -1 must be unchanged.
        self.assertEqual(int(safe[0]), 5)
        self.assertEqual(int(safe[2]), 7)
        self.assertEqual(int(safe[4]), 9)
        # The replacement value must be 0 (harmless index into the
        # embedding table).
        self.assertEqual(int(safe[1]), 0)
        self.assertEqual(int(safe[3]), 0)

    @unittest.skipUnless(HAS_JAX, "jax is not installed")
    def test_safe_input_ids_no_negative_tokens_unchanged(self):
        """When no ``-1`` is present, the array must be unchanged."""
        from sgl_jax.srt.speculative.ngram_worker import _make_safe_input_ids

        draft = np.array([1, 2, 3, 4], dtype=np.int32)
        safe = _make_safe_input_ids(draft)
        self.assertTrue(np.array_equal(draft, safe))


class TestCustomMaskLayout(unittest.TestCase):
    """Tests for BUG B (``custom_mask`` wrong shape and layout) fix."""

    @unittest.skipUnless(HAS_JAX, "jax is not installed")
    def test_custom_mask_layout_single_request_chain(self):
        """Chain-mode single request: verify the flat mask block layout."""
        from sgl_jax.srt.speculative.ngram_worker import _build_custom_mask_flat

        # Chain tree of depth 3: D0 -> D1 -> D2.
        D = 3
        parent_idx = np.array([-1, 0, 1], dtype=np.int32)
        seq_lens_before_draft = np.array([5], dtype=np.int32)

        mask = _build_custom_mask_flat(seq_lens_before_draft, [parent_idx], D)

        # Block shape: D x (sl + D) = 3 x 8 = 24 entries.
        expected_rows = [
            # D0 attends to 5 prefix positions + itself at column 5 (+ no
            # ancestors since its parent is -1).
            [1, 1, 1, 1, 1, 1, 0, 0],
            # D1 attends to all 5 prefix + itself at column 6 + ancestor D0
            # at column 5.
            [1, 1, 1, 1, 1, 1, 1, 0],
            # D2 attends to all 5 prefix + itself at column 7 + ancestors
            # D0, D1 at columns 5, 6.
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]
        expected = np.array(expected_rows, dtype=np.int32).reshape(-1)

        self.assertEqual(mask.dtype, np.int32)
        self.assertEqual(mask.shape, (D * (5 + D),))
        self.assertTrue(
            np.array_equal(mask, expected),
            f"expected {expected.tolist()}, got {mask.tolist()}",
        )

    @unittest.skipUnless(HAS_JAX, "jax is not installed")
    def test_custom_mask_layout_uneven_batch(self):
        """Per-request blocks must respect each request's own seq_len."""
        from sgl_jax.srt.speculative.ngram_worker import _build_custom_mask_flat

        D = 2
        # Two requests with different prefix lengths.
        parent_idx_0 = np.array([-1, 0], dtype=np.int32)
        parent_idx_1 = np.array([-1, 0], dtype=np.int32)
        seq_lens_before_draft = np.array([3, 5], dtype=np.int32)

        mask = _build_custom_mask_flat(seq_lens_before_draft, [parent_idx_0, parent_idx_1], D)

        # Request 0: block size D * (3 + D) = 2 * 5 = 10
        # Request 1: block size D * (5 + D) = 2 * 7 = 14
        self.assertEqual(mask.shape, (10 + 14,))

        block0 = mask[:10].reshape(D, 3 + D)
        block1 = mask[10:].reshape(D, 5 + D)

        # Request 0: D0 attends to [prefix 0..2, self at col 3].
        self.assertTrue(np.array_equal(block0[0], np.array([1, 1, 1, 1, 0])))
        # D1 attends to [prefix 0..2, ancestor D0 at col 3, self at col 4].
        self.assertTrue(np.array_equal(block0[1], np.array([1, 1, 1, 1, 1])))
        # Request 1 analogous with 5 prefix columns.
        self.assertTrue(np.array_equal(block1[0], np.array([1, 1, 1, 1, 1, 1, 0])))
        self.assertTrue(np.array_equal(block1[1], np.array([1, 1, 1, 1, 1, 1, 1])))

    @unittest.skipUnless(HAS_JAX, "jax is not installed")
    def test_custom_mask_layout_empty_batch(self):
        """Empty batch must return an empty int32 array."""
        from sgl_jax.srt.speculative.ngram_worker import _build_custom_mask_flat

        mask = _build_custom_mask_flat(np.zeros(0, dtype=np.int32), [], 3)
        self.assertEqual(mask.dtype, np.int32)
        self.assertEqual(mask.shape, (0,))

    @unittest.skipUnless(HAS_JAX, "jax is not installed")
    def test_custom_mask_walk_cycle_is_bounded(self):
        """A pathological ``parent_idx`` self-loop must not hang."""
        from sgl_jax.srt.speculative.ngram_worker import _build_custom_mask_flat

        D = 3
        # Pathological: node 0 claims itself as parent (would infinite-loop
        # without the iteration bound in _build_custom_mask_flat).
        parent_idx = np.array([0, 0, 0], dtype=np.int32)
        seq_lens_before_draft = np.array([2], dtype=np.int32)

        mask = _build_custom_mask_flat(seq_lens_before_draft, [parent_idx], D)
        # The function must return without hanging; we don't care about the
        # exact mask values, just that the call terminates.
        self.assertEqual(mask.shape, (D * (2 + D),))


if __name__ == "__main__":
    unittest.main()
