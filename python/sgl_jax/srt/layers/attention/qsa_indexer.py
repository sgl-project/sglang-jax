"""Qwen Sparse Attention (QSA) indexer for Qwen3.8-Flash-Next.

The 12 full-attention layers of Flash-Next do not attend over the whole
context. This lightweight indexer aggregates the sequence into micro-blocks of
``compress_ratio`` tokens, scores at block granularity, and selects the
``indexer_budget / compress_ratio`` highest-scoring blocks. The sparse
attention kernel then attends over the tokens those blocks expand to.

Scoring itself is not implemented here. ``kernels/dsa/streamindex_topk`` already
computes ``sum_h relu(q_h . k) * w_h`` with a ``compression_ratio`` parameter
whose visibility rule (``e < (q_pos + 1) // compression_ratio``) is exactly
QSA's; passing all-ones weights reduces it to QSA's ``sum_h relu(q_h . k)``.

What is new here is the *write* side -- turning raw keys into compressed block keys
-- and the expansion of selected block ids back into token ids.

Four things are silently wrong if changed, all matching upstream SGLang's
``python/sglang/srt/layers/attention/qsa/qsa_indexer.py`` and vLLM's
``vllm/models/qwen4_exp``:

* compression is pool -> norm -> RoPE, never norm -> pool;
* the pool averages the whole group, in fp32; a max, or any subset of the
  group, changes the answer. A sum does not: the GemmaRMSNorm that follows is
  scale invariant, so the factor of ``compress_ratio`` survives only through
  its epsilon;
* the compressed key is rotated at its group's first position, not its own;
* the norm is GemmaRMSNorm, i.e. ``x * (1 + w)``; a plain RMSNorm is off by the
  unit offset and the error is silent.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.kernels.dsa.ref import streamindex_topk_ref
from sgl_jax.srt.kernels.dsa.streamindex_topk import streamindex_topk
from sgl_jax.srt.layers.layernorm import GemmaRMSNorm
from sgl_jax.srt.layers.linear import LinearBase


class QSAIndexer(nnx.Module):
    """Projections, group compression and block expansion for one QSA layer.

    Replicated across tensor-parallel ranks: with ``indexer_n_heads=4`` there is
    nothing worth sharding, and a replicated projection avoids a collective at
    every full-attention layer. The compressed key cache is replicated with it.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        indexer_n_heads: int,
        indexer_kv_heads: int,
        indexer_head_dim: int,
        indexer_budget: int,
        indexer_compress_ratio: int,
        rotary_dim: int,
        mesh: jax.sharding.Mesh,
        rms_norm_eps: float = 1e-6,
        params_dtype: jnp.dtype = jnp.bfloat16,
        scope_name: str = "indexer",
    ):
        if indexer_kv_heads != 1:
            raise ValueError(
                f"the QSA MQA operators require indexer_kv_heads=1, got {indexer_kv_heads}"
            )
        # Upstream's bound. A ratio of 1 never reaches a compression boundary,
        # so no group would ever close. The tighter constraint that the ratio
        # divides the KV page size belongs to the pool, which enforces it.
        if indexer_compress_ratio < 2:
            raise ValueError(
                f"indexer_compress_ratio must be at least 2, got {indexer_compress_ratio}"
            )
        if indexer_budget % indexer_compress_ratio:
            raise ValueError("indexer_budget must be divisible by indexer_compress_ratio")
        if rotary_dim > indexer_head_dim:
            raise ValueError(f"rotary_dim {rotary_dim} exceeds indexer_head_dim {indexer_head_dim}")

        self.n_heads = indexer_n_heads
        self.head_dim = indexer_head_dim
        self.budget = indexer_budget
        self.compress_ratio = indexer_compress_ratio
        self.block_topk = indexer_budget // indexer_compress_ratio
        self.rotary_dim = rotary_dim
        self.name = scope_name

        # One fused projection for both sides, split on the head axis below.
        # HF ships the weight as [out, in] and LinearBase wants (in, out), so
        # the weight mapping needs transpose=True for this entry.
        self.index_qk_proj = LinearBase(
            input_size=hidden_size,
            output_size=(indexer_n_heads + indexer_kv_heads) * indexer_head_dim,
            mesh=mesh,
            use_bias=False,
            params_dtype=params_dtype,
            kernel_axes=(None, None),
            scope_name="index_qk_proj",
        )
        # GemmaRMSNorm, i.e. x * (1 + w): a plain RMSNorm is off by the unit
        # offset and the error is silent.
        self.q_layernorm = GemmaRMSNorm(indexer_head_dim, epsilon=rms_norm_eps)
        self.k_layernorm = GemmaRMSNorm(indexer_head_dim, epsilon=rms_norm_eps)

    # ---------------------------------------------------------------- project

    def project(
        self, hidden_states: jax.Array, positions: jax.Array, rotary_emb: Any
    ) -> tuple[jax.Array, jax.Array]:
        """Split the fused projection into scoring queries and raw keys.

        Returns ``(query[T, n_heads, head_dim], raw_key[T, head_dim])``. The
        query is normed and rotated at its own position; the key is returned
        raw -- neither normed nor rotated -- because both happen later, on
        the pooled group, at the group's first position.
        """
        qk, _ = self.index_qk_proj(hidden_states)
        q_width = self.n_heads * self.head_dim
        query = qk[..., :q_width].reshape(-1, self.n_heads, self.head_dim)
        raw_key = qk[..., q_width:]

        query = self.q_layernorm(query)
        query = self._apply_rope(query, positions, rotary_emb)
        return query, raw_key

    def _apply_rope(self, x: jax.Array, positions: jax.Array, rotary_emb: Any) -> jax.Array:
        """Rotate the leading ``rotary_dim`` dims and pass the rest through.

        Written as a concatenation rather than ``x.at[..., :rotary_dim].set()``.
        The two are numerically identical; this is the form GLM's indexer uses,
        and it keeps the untouched tail visibly a pass-through.
        """
        head_size = getattr(rotary_emb, "head_size", self.rotary_dim)
        if head_size != self.rotary_dim:
            raise ValueError(
                f"rotary_emb.head_size must equal rotary_dim ({self.rotary_dim}) because "
                f"only the leading rotary dims are handed to it, got {head_size}"
            )
        squeeze = x.ndim == 2
        heads = x[:, None, :] if squeeze else x
        rot = heads[..., : self.rotary_dim]
        rot, _ = rotary_emb(positions, rot, rot)
        out = jnp.concatenate((rot, heads[..., self.rotary_dim :]), axis=-1)
        return out[:, 0, :] if squeeze else out

    # --------------------------------------------------------------- compress

    def compress_batch(
        self,
        raw_keys: jax.Array,
        positions: jax.Array,
        cu_q_lens: jax.Array,
        req_slots: jax.Array,
        rings: jax.Array,
        rotary_emb: Any,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Turn raw keys into one compressed key per completed group.

        A forward pass hands this a packed ``[T, D]`` stream whose requests
        have different lengths, so it walks the stream token by token rather
        than reshaping it into groups. For every token it materialises the
        ``compress_ratio`` keys ending at that token, reading this step's own
        keys while the index is still inside the request and the request's ring
        once it is not.

        That one window serves both outputs: a group's compressed key is the
        window of the token that closes it, and a request's new ring is the
        window of its last token masked to the tokens still open. The cost is
        ``compress_ratio`` static gathers, with no segment reduce, no dynamic
        loop, and no dependence on where a request's slice starts relative to a
        group boundary.

        Args:
          raw_keys:  [T, D]              un-normed, un-rotated keys, packed
          positions: [T] i32             each token's position in its request
          cu_q_lens: [S + 1] i32         request boundaries within the stream
          req_slots: [S] i32             batch index -> ReqToTokenPool slot
          rings:     [max_reqs, compress_ratio, D]
          rotary_emb: applied at each group's FIRST position, derived as
            ``group * compress_ratio`` rather than read back from the tokens.
            That derivation is why multimodal RoPE is out of reach here: an
            mrope position is three numbers and cannot be recovered from an
            index, so the ring would have to carry positions alongside keys.

        Returns ``(compressed[T, D], groups[T], seq_ids[T], rings_out)``.
        ``groups[i]`` is the group index token ``i`` closes, or -1 when it
        closes none -- so row ``i`` of ``compressed`` is written only where
        ``groups[i] >= 0``.

        Operands must be local, unsharded arrays. This runs inside the
        backend's ``jax.shard_map``, as ``DSASparseAttentionBackend._maybe_index``
        does, where each device sees a plain local view. Called from global
        scope on arrays that still carry axis annotations, the gathers below
        cannot resolve an output sharding and tracing fails.
        """
        compress_ratio = self.compress_ratio
        t_count = raw_keys.shape[0]
        n_seqs = cu_q_lens.shape[0] - 1

        # Which request each packed token belongs to. cu_q_lens is sorted, so
        # a searchsorted is the inverse of "request r owns tokens [cu[r], cu[r+1])".
        token_idx = jnp.arange(t_count, dtype=jnp.int32)
        seq_ids = jnp.clip(jnp.searchsorted(cu_q_lens[1:], token_idx, side="right"), 0, n_seqs - 1)
        q_start = cu_q_lens[seq_ids]
        in_seq = (token_idx >= q_start) & (token_idx < cu_q_lens[seq_ids + 1])
        offset_in_req = token_idx - q_start
        slots = req_slots[seq_ids]

        window = self._window_ending_at(token_idx, offset_in_req, slots, raw_keys, rings)

        # A token closes a group iff it is the last of one, i.e. its position
        # is the ratio-th in its group. -1 marks the rows nothing was written to.
        closes = in_seq & (positions % compress_ratio == compress_ratio - 1)
        groups = jnp.where(closes, positions // compress_ratio, -1)
        compressed = self._pool_and_rotate(
            window, jnp.maximum(groups, 0) * compress_ratio, rotary_emb
        )

        # Each request's new ring is the window of its last token this step,
        # keeping only the tokens whose group is still open, right aligned.
        # A request with no tokens this step has cu[r+1] == cu[r], so `last`
        # would point into its neighbour: compute anyway, then keep its old ring.
        last = jnp.clip(cu_q_lens[1:] - 1, 0, t_count - 1)
        has_tokens = cu_q_lens[1:] > cu_q_lens[:-1]
        carried = (positions[last] + 1) % compress_ratio
        keep = (
            jnp.arange(compress_ratio, dtype=jnp.int32)[None, :]
            >= (compress_ratio - carried)[:, None]
        )
        ring_new = jnp.where(keep[:, :, None], window[last], 0).astype(rings.dtype)
        ring_new = jnp.where(has_tokens[:, None, None], ring_new, rings[req_slots])
        rings_out = rings.at[req_slots].set(ring_new)

        return compressed, groups, seq_ids, rings_out

    def _window_ending_at(
        self,
        token_idx: jax.Array,
        offset_in_req: jax.Array,
        slots: jax.Array,
        raw_keys: jax.Array,
        rings: jax.Array,
    ) -> jax.Array:
        """[T, compress_ratio, D]: the keys ending at each token, oldest first.

        Stepping ``j`` tokens back from token ``i``: while ``j <= offset_in_req``
        the key is still in this step's stream at ``i - j``; past that it belongs
        to an earlier step and lives in the ring. The ring is right aligned, so
        the key ``j - offset_in_req`` positions before the stream's start sits at
        row ``compress_ratio - (j - offset_in_req)``.
        """
        compress_ratio = self.compress_ratio
        t_count = raw_keys.shape[0]
        columns = []
        for j in range(compress_ratio):
            from_stream = offset_in_req >= j
            # Both indices are clipped rather than guarded: the branch not taken
            # still reads somewhere in range, and `from_stream` discards it.
            src = jnp.clip(token_idx - j, 0, t_count - 1)
            ring_row = jnp.clip(compress_ratio - (j - offset_in_req), 0, compress_ratio - 1)
            columns.append(jnp.where(from_stream[:, None], raw_keys[src], rings[slots, ring_row]))
        # columns[0] is the token itself and columns[-1] the oldest key, so
        # reverse to put the group in reading order before pooling.
        return jnp.stack(columns[::-1], axis=1)

    def _pool_and_rotate(
        self, window: jax.Array, first_positions: jax.Array, rotary_emb: Any
    ) -> jax.Array:
        """fp32 mean -> GemmaRMSNorm -> RoPE at the group's first position."""
        pooled = jnp.mean(window.astype(jnp.float32), axis=1).astype(window.dtype)
        return self._apply_rope(self.k_layernorm(pooled), first_positions, rotary_emb)

    def select_blocks(
        self,
        query: jax.Array,
        compressed_cache: jax.Array,
        seq_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        cu_kv_lens: jax.Array,
        distribution: jax.Array,
        *,
        pages_per_seq: int,
        use_kernel: bool = True,
        one_token_per_seq: bool = False,
    ) -> jax.Array:
        """Top-k over compressed blocks; returns ``i32[T, block_topk]``.

        No QSA-specific scorer is needed. ``streamindex_topk`` computes
        ``sum_h relu(q_h . k) * w_h`` with the relu *before* the head sum, and
        its ``compression_ratio`` makes entry ``e`` visible only when
        ``e < (q_pos + 1) // compression_ratio``. QSA's score is
        ``sum_h relu(q_h . k)``
        under that same visibility rule, so all-ones weights reduce one to the
        other exactly -- QSA simply has no ``weights_proj`` to supply.

        ``seq_lens`` and ``cu_kv_lens`` both stay in uncompressed tokens; this
        method converts what needs converting. As in ``DSASparseAttentionBackend``
        the two implementations want different page tables: the Pallas kernel
        takes a fixed-stride one and walks it as ``seq * pages_per_seq``, while
        the reference takes the packed table and finds a sequence's first page
        at ``cu_kv_lens[seq] // cache.shape[1]``. That divisor is the *cache's*
        page size, which for the compressed cache is
        ``page_size // compress_ratio``, so the reference needs ``cu_kv_lens``
        in compressed entries or it lands that many times too far into the
        table. Every term of ``cu_kv_lens`` is
        a page-aligned token count, so the division is exact.
        """
        weights = jnp.ones(query.shape[:2], dtype=query.dtype)
        if use_kernel:
            return streamindex_topk(
                query,
                weights,
                compressed_cache,
                seq_lens,
                page_indices,
                cu_q_lens,
                distribution,
                k=self.block_topk,
                compression_ratio=self.compress_ratio,
            )
        return streamindex_topk_ref(
            query,
            weights,
            compressed_cache,
            seq_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens // self.compress_ratio,
            distribution,
            k=self.block_topk,
            pages_per_seq=pages_per_seq,
            compression_ratio=self.compress_ratio,
            one_token_per_seq=one_token_per_seq,
        )

    # ----------------------------------------------------------------- expand

    def expand(self, block_ids: jax.Array, logical_positions: jax.Array) -> jax.Array:
        """See :func:`expand_block_ids`; bound to this layer's shape constants."""
        return expand_block_ids(
            block_ids,
            logical_positions,
            compress_ratio=self.compress_ratio,
            budget=self.budget,
        )

    @property
    def indexer_weights_shape(self) -> tuple[int, ...]:
        """QSA has no ``weights_proj``; feed ``streamindex_topk`` ones of this shape."""
        return (self.n_heads,)


def expand_block_ids(
    block_ids: jax.Array,
    logical_positions: jax.Array,
    *,
    compress_ratio: int,
    budget: int,
) -> jax.Array:
    """Expand selected block ids to token ids, plus the open group's causal tail.

    Returns ``i32[T, budget + compress_ratio]``: the leading
    ``budget + compress_ratio - 1`` columns are request-local token ids
    (``-1`` padded) and the final column is the row's valid-entry count -- the
    sparse attention kernel's tile-loop bound, never a token id.

    The ``compress_ratio - 1`` extra slots are no safety margin. They
    hold the tail of the *open* group: the tokens that have not yet completed a
    group, so they are absent from the compressed cache and never got to
    compete in the top-k, yet are causally visible and must still be attended.
    A query at position ``p`` has ``(p + 1) % compress_ratio`` of them.
    """
    width = budget + compress_ratio - 1

    n_blocks = block_ids.shape[-1]
    # Block b covers tokens [b*compress_ratio, (b+1)*compress_ratio);
    # -1 padding stays -1.
    offsets = jnp.arange(compress_ratio, dtype=block_ids.dtype)
    expanded = (block_ids[..., None] * compress_ratio + offsets).reshape(
        *block_ids.shape[:-1], n_blocks * compress_ratio
    )
    expanded = jnp.where(
        block_ids[..., None].repeat(compress_ratio, axis=-1).reshape(expanded.shape) >= 0,
        expanded,
        -1,
    )
    expanded_count = jnp.sum(block_ids >= 0, axis=-1) * compress_ratio

    seen = logical_positions + 1
    tail_start = (seen // compress_ratio) * compress_ratio
    tail_count = seen - tail_start  # in [0, compress_ratio-1]

    # The tail starts right after the expanded blocks, at a column that varies
    # per row: a short sequence has fewer than budget/compress_ratio visible
    # blocks, so its expanded_count falls short of budget.
    cols = jnp.arange(width, dtype=jnp.int32)
    tail_offset = cols[None, :] - expanded_count[:, None]
    is_tail = (tail_offset >= 0) & (tail_offset < tail_count[:, None])
    tail_token = tail_start[:, None] + tail_offset

    padded = jnp.pad(
        expanded, ((0, 0), (0, max(0, width - expanded.shape[-1]))), constant_values=-1
    )
    padded = padded[:, :width]
    in_expanded = cols[None, :] < expanded_count[:, None]

    token_ids = jnp.where(in_expanded, padded, jnp.where(is_tail, tail_token, -1))
    count = (expanded_count + tail_count).astype(jnp.int32)
    return jnp.concatenate([token_ids.astype(jnp.int32), count[:, None]], axis=-1)
