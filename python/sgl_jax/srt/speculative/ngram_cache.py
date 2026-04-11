"""Trie-based N-gram cache for speculative decoding draft token generation.

Stores previously decoded token sequences in a trie structure. Given a suffix
of the most recently generated tokens, the trie is queried via BFS to produce
a tree of candidate draft tokens that the target model can verify in a single
forward pass.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


class TrieNode:
    __slots__ = ("children", "count")

    def __init__(self):
        self.children: dict[int, TrieNode] = {}
        self.count: int = 0


@dataclass
class DraftTree:
    """Result of a BFS query on the trie.

    Attributes:
        tokens: Flat array of draft token ids, length ``draft_token_num``.
        mask: Boolean mask of shape ``(draft_token_num, draft_token_num)``
              where ``mask[i, j] = True`` means token *i* can attend to token *j*.
        retrive_index: Maps each tree position to its index in the output logits.
        retrive_next_token: First-child index for tree traversal (-1 = leaf).
        retrive_next_sibling: Next-sibling index for tree traversal (-1 = none).
        parent_idx: Parent index for each node (-1 = root child).
    """

    tokens: np.ndarray
    mask: np.ndarray
    retrive_index: np.ndarray
    retrive_next_token: np.ndarray
    retrive_next_sibling: np.ndarray
    parent_idx: np.ndarray


class NgramCache:
    """Trie-based n-gram cache for speculative decoding.

    Not thread-safe -- relies on the scheduler being single-threaded
    (speculative decoding requires ``--disable-overlap-schedule``).

    Parameters:
        max_trie_depth: Maximum depth of the trie (longest stored n-gram).
        max_nodes: Soft cap on trie nodes; insertions stop when exceeded.
    """

    def __init__(self, max_trie_depth: int = 8, max_nodes: int = 1_000_000):
        self.root = TrieNode()
        self.max_trie_depth = max_trie_depth
        self.max_nodes = max_nodes
        self._node_count = 0

    def insert(self, token_ids: list[int] | np.ndarray) -> None:
        """Insert all valid n-gram suffixes from ``token_ids`` into the trie.

        This is a convenience wrapper around :meth:`insert_new_suffixes` that
        inserts every n-gram (of length ``<= max_trie_depth``) appearing in
        ``token_ids`` exactly once.
        """
        self.insert_new_suffixes(token_ids, prev_len=0)

    def insert_new_suffixes(self, token_ids: list[int] | np.ndarray, prev_len: int) -> None:
        """Insert only n-grams whose terminal position is >= ``prev_len``.

        This enables incremental updates when new tokens are appended to a
        sequence without re-incrementing counts for previously seen
        n-grams. For each new terminal position ``P`` in
        ``[prev_len, n - 1]`` and each length ``L`` in
        ``[1, min(max_trie_depth, P + 1)]``, the trie node representing
        the n-gram ``token_ids[P - L + 1 : P + 1]`` has its ``count``
        incremented by exactly one (and intermediate nodes along the
        walk are created if missing, but not incremented -- they count
        distinct n-grams of their own which will be visited for their
        own terminal positions).

        Parameters:
            token_ids: The full token sequence.
            prev_len: The sequence length that was previously inserted.
                Pass ``0`` to insert every n-gram from scratch.
        """
        n = len(token_ids)
        first_new_pos = max(int(prev_len), 0)
        for p in range(first_new_pos, n):
            max_len = min(self.max_trie_depth, p + 1)
            for length in range(1, max_len + 1):
                start = p - length + 1
                node = self.root
                abort = False
                for idx in range(start, p + 1):
                    tok = int(token_ids[idx])
                    if tok not in node.children:
                        if self._node_count >= self.max_nodes:
                            abort = True
                            break
                        node.children[tok] = TrieNode()
                        self._node_count += 1
                    node = node.children[tok]
                if abort:
                    # Node budget exhausted; stop processing the rest of
                    # this sequence to avoid inconsistent partial walks.
                    return
                # Increment only the terminal node of this n-gram walk.
                node.count += 1

    def query_bfs(
        self,
        suffix: list[int] | np.ndarray,
        draft_token_num: int,
        max_bfs_breadth: int,
    ) -> DraftTree:
        """Query the trie with ``suffix`` and return up to ``draft_token_num`` draft tokens.

        Uses BFS starting from the trie node matching the longest suffix prefix.
        Children at each level are ranked by frequency (count) and capped by
        ``max_bfs_breadth``.

        Returns a ``DraftTree`` with correctly constructed tree traversal indices.
        """
        start_node = self._find_best_match(suffix)

        tokens = np.full(draft_token_num, -1, dtype=np.int32)
        parent_idx = np.full(draft_token_num, -1, dtype=np.int32)
        children_map: dict[int, list[int]] = defaultdict(list)
        count = 0
        real_count = 0

        if start_node is None or not start_node.children:
            return self._build_empty_draft(draft_token_num)

        queue: deque[tuple[TrieNode, int]] = deque()

        sorted_children = sorted(
            start_node.children.items(), key=lambda x: x[1].count, reverse=True
        )[:max_bfs_breadth]

        for tok, child_node in sorted_children:
            if count >= draft_token_num:
                break
            tokens[count] = tok
            parent_idx[count] = -1
            children_map[-1].append(count)
            queue.append((child_node, count))
            count += 1

        while queue and count < draft_token_num:
            trie_node, pidx = queue.popleft()
            if not trie_node.children:
                continue

            sorted_ch = sorted(trie_node.children.items(), key=lambda x: x[1].count, reverse=True)[
                :max_bfs_breadth
            ]

            for tok, child_node in sorted_ch:
                if count >= draft_token_num:
                    break
                tokens[count] = tok
                parent_idx[count] = pidx
                children_map[pidx].append(count)
                queue.append((child_node, count))
                count += 1

        real_count = count

        # If we produced no real candidates (e.g. ``max_bfs_breadth == 0``
        # or an empty children dict), fall through to a safe empty tree
        # with ``parent_idx = [-1, -1, ...]``. Without this guard we
        # would compute ``pad_parent = max(-1, 0) = 0`` and then set
        # ``parent_idx[0] = 0``, producing a self-loop that can hang the
        # ancestry walks in ``_build_draft_tree`` and in the target
        # verify positions computation.
        if real_count == 0:
            return self._build_empty_draft(draft_token_num)

        # Padded positions use -1 (unmatchable sentinel) and are parented
        # to the last real node so the verify kernel rejects them
        # immediately. ``pad_parent`` is always a valid real-node index
        # (>= 0) because ``real_count >= 1`` at this point.
        if count < draft_token_num:
            pad_parent = real_count - 1
            for i in range(count, draft_token_num):
                tokens[i] = -1
                parent_idx[i] = pad_parent

        return self._build_draft_tree(tokens, parent_idx, children_map, draft_token_num)

    def _find_best_match(self, suffix: list[int] | np.ndarray) -> TrieNode | None:
        """Walk the trie to find the deepest node matching the full suffix.

        Tries the full suffix first, then drops the oldest token and retries
        with shorter suffixes until a complete match is found.

        Returns the deepest matching node with children, or ``self.root``
        as a last resort.
        """
        if len(suffix) == 0:
            return self.root

        for start in range(len(suffix)):
            node = self.root
            matched = True
            for i in range(start, len(suffix)):
                tok = int(suffix[i])
                if tok in node.children:
                    node = node.children[tok]
                else:
                    matched = False
                    break
            if matched and node.children:
                return node

        return self.root

    def _build_draft_tree(
        self,
        tokens: np.ndarray,
        parent_idx: np.ndarray,
        children_map: dict[int, list[int]],
        draft_token_num: int,
    ) -> DraftTree:
        """Build tree traversal indices from parent relationships."""
        retrive_index = np.arange(draft_token_num, dtype=np.int32)
        retrive_next_token = np.full(draft_token_num, -1, dtype=np.int32)
        retrive_next_sibling = np.full(draft_token_num, -1, dtype=np.int32)

        for pidx, child_indices in children_map.items():
            if not child_indices:
                continue
            if pidx >= 0:
                retrive_next_token[pidx] = child_indices[0]
            for j in range(len(child_indices) - 1):
                retrive_next_sibling[child_indices[j]] = child_indices[j + 1]

        mask = np.zeros((draft_token_num, draft_token_num), dtype=np.bool_)
        for i in range(draft_token_num):
            mask[i, i] = True
            p = parent_idx[i]
            while p >= 0:
                mask[i, p] = True
                p = parent_idx[p]

        return DraftTree(
            tokens=tokens,
            mask=mask,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            parent_idx=parent_idx,
        )

    def _build_empty_draft(self, draft_token_num: int) -> DraftTree:
        """Return a degenerate draft tree (no useful candidates)."""
        return DraftTree(
            tokens=np.full(draft_token_num, -1, dtype=np.int32),
            mask=np.eye(draft_token_num, dtype=np.bool_),
            retrive_index=np.arange(draft_token_num, dtype=np.int32),
            retrive_next_token=np.full(draft_token_num, -1, dtype=np.int32),
            retrive_next_sibling=np.full(draft_token_num, -1, dtype=np.int32),
            parent_idx=np.full(draft_token_num, -1, dtype=np.int32),
        )

    def reset(self) -> None:
        """Clear the entire trie."""
        self.root = TrieNode()
        self._node_count = 0
