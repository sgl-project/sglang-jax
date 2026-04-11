"""N-gram speculative decoding worker for sglang-jax.

Generates draft tokens by querying a trie-based n-gram cache built from
previously decoded tokens.  No draft model is required -- the target model
verifies the entire draft tree in a single forward pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.kernels.speculative.verify_tree_greedy_kernel import verify_tree_greedy
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.managers.scheduler import GenerationBatchResult
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata
from sgl_jax.srt.speculative.ngram_cache import NgramCache

if TYPE_CHECKING:
    from sgl_jax.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@register_pytree_node_class
@dataclass
class NgramVerifyInput:
    """Lightweight spec_info for ngram verify, compatible with get_eagle_forward_metadata.

    Registered as a JAX pytree so the wrapping ``ModelWorkerBatch`` can flow
    through ``jax.jit`` (the target ``forward`` is jitted and inspects every
    leaf of ``model_worker_batch``). ``custom_mask`` is the only jax.Array
    leaf; ``draft_token_num`` is static (compile-time) metadata; and
    ``allocate_lens`` is a host-side numpy array we keep out of the pytree
    to avoid forcing host->device transfers on every call.
    """

    custom_mask: np.ndarray | jax.Array
    draft_token_num: int
    allocate_lens: np.ndarray | None = None

    def tree_flatten(self):
        children = (self.custom_mask,)
        aux_data = {
            "draft_token_num": self.draft_token_num,
            "allocate_lens": self.allocate_lens,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            custom_mask=children[0],
            draft_token_num=aux_data["draft_token_num"],
            allocate_lens=aux_data["allocate_lens"],
        )


def _ensure_chain_mode(max_bfs_breadth: int) -> None:
    """Assert that ngram speculative decoding is running in chain mode.

    Branching-tree drafts require KV compaction support that the current
    implementation does not provide, so only chain mode (``breadth == 1``)
    is supported today.
    """
    if max_bfs_breadth != 1:
        raise ValueError(
            "ngram speculative decoding currently supports only chain mode "
            "(--speculative-ngram-max-bfs-breadth=1). Branching-tree KV "
            "compaction is not yet implemented; see sglang-jax issue #192."
        )


def _make_safe_input_ids(draft_tokens: np.ndarray) -> np.ndarray:
    """Replace padding sentinels (``-1``) with ``0`` before embedding lookup.

    JAX's ``.at[-1].get()`` wraps negative indices to the end of the
    embedding table, which would silently give padding positions random
    (last-row) embeddings. Substituting ``0`` ensures padding positions
    produce harmless embeddings; the verify kernel still rejects them
    because ``draft_tokens`` itself retains ``-1`` sentinels.
    """
    return np.where(draft_tokens == -1, 0, draft_tokens).astype(np.int32)


def _build_custom_mask_flat(
    seq_lens_before_draft: np.ndarray,
    parent_idx_per_req: list[np.ndarray],
    draft_token_num: int,
) -> np.ndarray:
    """Build the 1D flat ``custom_mask`` expected by ``ragged_paged_attention``.

    The kernel consumes per-request blocks of size
    ``q_len * kv_len = D * (seq_lens[i] + D)`` concatenated in request
    order (see ``ragged_paged_attention.py`` where
    ``cu_seq_mask_lens = cumsum(kv_lens * q_lens)``). Each block is
    row-major with query rows = draft token index and columns = KV
    positions.
    """
    bs = len(parent_idx_per_req)
    if bs == 0:
        return np.zeros(0, dtype=np.int32)

    per_req_mask_blocks: list[np.ndarray] = []
    for i in range(bs):
        sl = int(seq_lens_before_draft[i])
        parent_idx = parent_idx_per_req[i]
        block = np.zeros((draft_token_num, sl + draft_token_num), dtype=np.int32)
        for j in range(draft_token_num):
            # Attend to every prefix KV position for this request.
            if sl > 0:
                block[j, :sl] = 1
            # Self-attention on this draft token.
            block[j, sl + j] = 1
            # Walk up the tree ancestors with an iteration bound so that
            # any pathological ``parent_idx`` (e.g. a self-loop injected
            # by a degenerate draft) cannot hang the worker.
            p = int(parent_idx[j])
            iterations = 0
            while p >= 0 and iterations < draft_token_num:
                block[j, sl + p] = 1
                p = int(parent_idx[p])
                iterations += 1
        per_req_mask_blocks.append(block.reshape(-1))

    return np.concatenate(per_req_mask_blocks)


class NgramWorker:
    """Speculative decoding worker that uses n-gram lookup instead of a draft model."""

    def __init__(self, server_args: ServerArgs, target_worker: ModelWorker):
        self.server_args = server_args
        self.target_worker = target_worker
        self.mesh = target_worker.mesh
        self.model_config = target_worker.model_config
        self.page_size = server_args.page_size

        self.draft_token_num: int = server_args.speculative_num_draft_tokens
        self.speculative_num_draft_tokens: int = server_args.speculative_num_draft_tokens
        self.max_trie_depth: int = server_args.speculative_ngram_max_trie_depth
        self.max_bfs_breadth: int = server_args.speculative_ngram_max_bfs_breadth
        self.speculative_num_steps = server_args.speculative_num_steps

        _ensure_chain_mode(self.max_bfs_breadth)

        self.req_to_token_pool, self.token_to_kv_pool_allocator = target_worker.get_memory_pool()

        (
            self.precompile_token_paddings,
            self.precompile_bs_paddings,
            self.precompile_cache_loc_paddings,
        ) = target_worker.get_precompile_paddings()

        # Per-request ngram caches keyed by request id
        self._caches: dict[str, NgramCache] = {}
        # Tracks how many tokens of each request have already been inserted
        # into the trie, so that incremental updates skip already-inserted
        # n-grams (prevents BUG E count inflation).
        self._last_inserted_len: dict[str, int] = {}

    def _get_or_create_cache(self, req_id: str) -> NgramCache:
        if req_id not in self._caches:
            self._caches[req_id] = NgramCache(
                max_trie_depth=self.max_trie_depth,
            )
        return self._caches[req_id]

    def _evict_finished_requests(self, active_req_ids: set[str]) -> None:
        stale = [rid for rid in self._caches if rid not in active_req_ids]
        for rid in stale:
            del self._caches[rid]
            self._last_inserted_len.pop(rid, None)
        # Defensive cleanup: drop any tracker entries that lost their cache.
        stale_trackers = [rid for rid in self._last_inserted_len if rid not in self._caches]
        for rid in stale_trackers:
            self._last_inserted_len.pop(rid, None)

    def forward_batch_speculative_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
    ) -> GenerationBatchResult:
        """Run one round of ngram speculative decoding.

        For extend (prefill): run the target model and pre-populate ngram caches.
        For decode: draft via ngram, verify with target model.
        """
        reqs = getattr(model_worker_batch, "reqs", None)
        if reqs:
            active_ids = {req.rid for req in reqs}
            self._evict_finished_requests(active_ids)

        if model_worker_batch.forward_mode.is_extend():
            return self._forward_extend(model_worker_batch)
        else:
            return self._forward_decode(model_worker_batch)

    def _forward_extend(self, model_worker_batch: ModelWorkerBatch) -> GenerationBatchResult:
        """Handle prefill -- run the target model and pre-populate caches."""
        if model_worker_batch.sampling_info.temperatures.ndim == 1:
            model_worker_batch.sampling_info.temperatures = (
                model_worker_batch.sampling_info.temperatures[:, None]
            )
        sampling_metadata = SamplingMetadata.from_model_worker_batch(
            model_worker_batch,
            len(model_worker_batch.seq_lens) - model_worker_batch.real_bs,
            self.mesh,
            vocab_size=self.model_config.vocab_size,
        )
        logits_output, next_token_ids, cache_miss_count = (
            self.target_worker.forward_batch_generation(
                model_worker_batch, sampling_metadata=sampling_metadata
            )
        )

        # Pre-populate ngram caches from prefill tokens so the first
        # decode step has useful draft candidates.
        reqs = getattr(model_worker_batch, "reqs", None)
        if reqs:
            for req in reqs[: model_worker_batch.real_bs]:
                cache = self._get_or_create_cache(req.rid)
                prev_len = self._last_inserted_len.get(req.rid, 0)
                if len(req.origin_input_ids) > prev_len:
                    cache.insert_new_suffixes(req.origin_input_ids, prev_len)
                    self._last_inserted_len[req.rid] = len(req.origin_input_ids)

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            next_draft_input=None,
            allocate_lens=model_worker_batch.seq_lens[: model_worker_batch.real_bs],
            bid=model_worker_batch.bid,
            cache_miss_count=cache_miss_count,
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
        )

    def _forward_decode(self, model_worker_batch: ModelWorkerBatch) -> GenerationBatchResult:
        """Draft via ngram, verify with target model, return accepted tokens."""
        bs = model_worker_batch.real_bs
        D = self.draft_token_num

        # --- Draft phase (CPU, no model forward) ---
        (
            draft_tokens_np,
            parent_idx_per_req,
            retrive_index_np,
            retrive_next_token_np,
            retrive_next_sibling_np,
        ) = self._prepare_draft_tokens_from_batch(model_worker_batch)

        # BUG A fix: do NOT mutate ``batch.seq_lens`` in place.
        # ``model_worker_batch.seq_lens`` shares memory with
        # ``batch.seq_lens``; a persistent in-place decrement would drift
        # the tracked sequence length behind ``req.seqlen`` by one every
        # decode iteration because the scheduler will then do
        # ``batch.seq_lens += accept_lens`` on the already-decremented
        # value. Instead we use a local ``seq_lens_before_draft`` for all
        # downstream math, temporarily decrement
        # ``model_worker_batch.seq_lens`` only around the target forward
        # (mirroring EAGLE's ``prepare_for_verify`` semantics), and
        # restore the original value before returning so the scheduler's
        # accept_lens addition lands on the un-rewound value.
        seq_lens_original = model_worker_batch.seq_lens[:bs].copy()
        seq_lens_before_draft = seq_lens_original - 1

        # --- Allocate KV cache slots for draft tokens ---
        allocate_lens = seq_lens_before_draft.copy()
        new_allocate_lens = allocate_lens + D

        if self.page_size == 1:
            num_needed = int(np.sum(new_allocate_lens - allocate_lens))
            out_cache_loc = alloc_token_slots(model_worker_batch.tree_cache, num_needed)
        else:
            from sgl_jax.srt.managers.schedule_batch import get_last_loc

            last_loc = get_last_loc(
                self.req_to_token_pool.req_to_token,
                model_worker_batch.req_pool_indices[:bs],
                allocate_lens,
            )
            num_needed = int(np.sum(new_allocate_lens - allocate_lens))
            out_cache_loc = alloc_paged_token_slots_extend(
                model_worker_batch.tree_cache,
                allocate_lens,
                new_allocate_lens,
                last_loc,
                num_needed,
            )

        from sgl_jax.srt.speculative.eagle_util import assign_req_to_token_pool

        assign_req_to_token_pool(
            model_worker_batch.req_pool_indices[:bs],
            self.req_to_token_pool,
            allocate_lens,
            new_allocate_lens,
            out_cache_loc,
        )
        model_worker_batch.out_cache_loc = out_cache_loc

        # Build positions using tree depth (not linear index).
        # For branching trees, sibling nodes at the same depth must get
        # the same position so rotary embeddings are correct. We cap the
        # walk at ``D`` iterations (BUG G safety) so that any pathological
        # ``parent_idx`` cycle fails loud instead of hanging.
        positions = np.zeros(bs * D, dtype=np.int32)
        for i in range(bs):
            parent_idx = parent_idx_per_req[i]  # shape (D,)
            for j in range(D):
                depth = 0
                p = j
                iterations = 0
                while p >= 0 and parent_idx[p] >= 0 and iterations < D:
                    depth += 1
                    p = int(parent_idx[p])
                    iterations += 1
                assert iterations < D, (
                    f"parent_idx cycle detected at request {i}, position {j}: "
                    f"{parent_idx.tolist()}"
                )
                depth = max(depth, 0) if draft_tokens_np[i * D + j] != -1 else 0
                positions[i * D + j] = int(seq_lens_before_draft[i]) + depth

        # BUG B fix: build the attention ``custom_mask`` as a 1D flat int32
        # array with per-request blocks of size ``D * (seq_lens[i] + D)``
        # concatenated in request order. The kernel expects this layout
        # (see ``ragged_paged_attention.py``:
        # ``cu_seq_mask_lens = cumsum(kv_lens * q_lens)``) and then
        # promotes it to (flatten_total_kv_len, head_dim) internally. A
        # 2D ``(bs*D, total_kv)`` input would get expanded to 3D and
        # mismatch the kernel's expected DMA shape.
        custom_mask_flat = _build_custom_mask_flat(seq_lens_before_draft, parent_idx_per_req, D)

        # BUG D fix: ``draft_tokens_np`` may contain ``-1`` padding
        # sentinels which JAX's embedding ``.at[-1]`` interprets as
        # negative (wrap-around) indices. Replace padding with ``0``
        # before the model forward; ``draft_tokens_jax`` below still
        # carries the ``-1`` for the verify kernel so that padding
        # positions can never match.
        safe_input_ids = _make_safe_input_ids(draft_tokens_np)

        # Set up model_worker_batch for target verify. We keep
        # ``model_worker_batch.seq_lens`` at the rewound value only for
        # the duration of ``get_eagle_forward_metadata`` + target
        # forward, then restore it before returning.
        model_worker_batch.input_ids = safe_input_ids
        model_worker_batch.positions = positions
        model_worker_batch.forward_mode = ForwardMode.TARGET_VERIFY
        model_worker_batch.seq_lens[:bs] = seq_lens_before_draft

        # Create NgramVerifyInput as spec_info so get_eagle_forward_metadata
        # can access custom_mask and draft_token_num.
        model_worker_batch.spec_info = NgramVerifyInput(
            custom_mask=jnp.asarray(custom_mask_flat, dtype=jnp.int32),
            draft_token_num=D,
            allocate_lens=allocate_lens,
        )

        # Build cache_loc for the verify forward
        token_indices = self.req_to_token_pool.req_to_token[
            model_worker_batch.req_pool_indices[:bs]
        ]
        cache_loc_parts = []
        for i in range(bs):
            seq_len_with_draft = int(seq_lens_before_draft[i]) + D
            cache_loc_parts.append(token_indices[i, :seq_len_with_draft])
        model_worker_batch.cache_loc = (
            np.concatenate(cache_loc_parts) if cache_loc_parts else np.array([], dtype=np.int32)
        )

        try:
            # --- Forward target model ---
            forward_metadata = (
                self.target_worker.model_runner.attn_backend.get_eagle_forward_metadata(
                    model_worker_batch
                )
            )

            logits_output, _, cache_miss_count = self.target_worker.forward_batch_generation(
                model_worker_batch, skip_sample=True, forward_metadata=forward_metadata
            )
        finally:
            # Restore seq_lens so the scheduler's
            # ``batch.seq_lens += accept_lens`` lands on the un-rewound
            # value. Must happen even if forward_batch_generation raises.
            model_worker_batch.seq_lens[:bs] = seq_lens_original

        # --- Verify phase ---
        # Note: the verify kernel receives the ORIGINAL draft_tokens_np
        # (with ``-1`` padding intact) so that padding positions can
        # never match target predictions; the model forward used the
        # safe version for embedding lookup.
        draft_tokens_jax = jnp.asarray(draft_tokens_np, dtype=jnp.int32)
        retrive_index_jax = jnp.asarray(retrive_index_np, dtype=jnp.int32)
        retrive_next_token_jax = jnp.asarray(retrive_next_token_np, dtype=jnp.int32)
        retrive_next_sibling_jax = jnp.asarray(retrive_next_sibling_np, dtype=jnp.int32)

        try:
            ctx = jax.sharding.use_mesh(self.mesh)
        except AttributeError:
            try:
                ctx = jax.set_mesh(self.mesh)
            except AttributeError:
                ctx = self.mesh

        with ctx:
            accept_index, accept_length, predict = verify_tree_greedy(
                speculative_num_steps=D,
                num_draft_tokens=D,
                draft_tokens=draft_tokens_jax,
                retrive_index=retrive_index_jax,
                retrive_next_token=retrive_next_token_jax,
                retrive_next_sibling=retrive_next_sibling_jax,
                next_token_logits=logits_output.next_token_logits,
            )

        accept_length_np = np.asarray(jax.device_get(accept_length))
        predict_np = np.asarray(jax.device_get(predict))
        accept_index_np = np.asarray(jax.device_get(accept_index))

        # accept_length from kernel is 0-indexed (0 means 1 accepted token)
        accept_length_final = accept_length_np[:bs] + 1

        # Build next_token_ids in the stride format expected by
        # _resolve_spec_decode_token_ids: flat array of length bs*D.
        next_token_ids_flat = np.zeros(bs * D, dtype=np.int32)
        for i in range(bs):
            acc_len = int(accept_length_final[i])
            row = accept_index_np[i]
            for j in range(min(acc_len, len(row))):
                idx = int(row[j])
                if 0 <= idx < len(predict_np):
                    next_token_ids_flat[i * D + j] = int(predict_np[idx])

        # --- Free unaccepted KV slots for ALL requests (not just finished) ---
        # BUG F fix: in addition to freeing the KV cache slots, zero out
        # the corresponding ``req_to_token`` entries so that a subsequent
        # lookup can never see the stale slot id.
        for i in range(bs):
            used = int(accept_length_final[i])
            if used < D:
                start = int(seq_lens_before_draft[i]) + used
                end = int(seq_lens_before_draft[i]) + D
                req_idx = model_worker_batch.req_pool_indices[i]
                kv_indices = self.req_to_token_pool.req_to_token[req_idx, start:end]
                kv_indices = kv_indices[kv_indices != 0]
                if len(kv_indices) > 0:
                    self.token_to_kv_pool_allocator.free(kv_indices)
                # Clear stale entries so subsequent lookups can't re-use them.
                self.req_to_token_pool.req_to_token[req_idx, start:end] = 0

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids_flat,
            next_draft_input=None,
            accept_lens=accept_length_final,
            allocate_lens=allocate_lens,
            bid=model_worker_batch.bid,
            cache_miss_count=cache_miss_count,
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
        )

    def _prepare_draft_tokens_from_batch(
        self, model_worker_batch: ModelWorkerBatch
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """Generate draft tokens using the ngram cache.

        Position 0 (D0) is always the **verified_id** -- the last token the
        model actually produced.  This matches EAGLE's convention and ensures
        D0's KV is always correct.  Positions 1..D-1 are filled with ngram
        predictions from the trie cache.

        Returns:
            draft_tokens_np: Flat (bs*D,) array of draft token ids.
            parent_idx_per_req: List of (D,) arrays with parent indices per request.
            retrive_index_np: (bs, D) array of retrival indices.
            retrive_next_token_np: (bs, D) array of first-child indices.
            retrive_next_sibling_np: (bs, D) array of next-sibling indices.
        """
        bs = model_worker_batch.real_bs
        D = self.draft_token_num

        all_tokens = np.full(bs * D, -1, dtype=np.int32)
        all_retrive_index = np.zeros((bs, D), dtype=np.int32)
        all_retrive_next_token = np.full((bs, D), -1, dtype=np.int32)
        all_retrive_next_sibling = np.full((bs, D), -1, dtype=np.int32)
        parent_idx_per_req: list[np.ndarray] = []

        reqs = getattr(model_worker_batch, "reqs", None)
        if reqs is None:
            logger.warning("NgramWorker: no request info available, generating empty drafts")
            parent_idx_per_req = [np.full(D, -1, dtype=np.int32) for _ in range(bs)]
            return (
                all_tokens,
                parent_idx_per_req,
                all_retrive_index,
                all_retrive_next_token,
                all_retrive_next_sibling,
            )

        ngram_slots = D - 1  # slots available for ngram predictions (D1..D-1)

        for i, req in enumerate(reqs[:bs]):
            cache = self._get_or_create_cache(req.rid)

            all_toks = req.origin_input_ids + req.output_ids
            prev_len = self._last_inserted_len.get(req.rid, 0)
            if len(all_toks) > prev_len:
                cache.insert_new_suffixes(all_toks, prev_len)
                self._last_inserted_len[req.rid] = len(all_toks)

            # D0 = verified_id (last token the model actually produced)
            verified_id = int(all_toks[-1]) if len(all_toks) > 0 else 0
            base = i * D
            all_tokens[base] = verified_id

            # D1..D-1 = ngram predictions from the trie
            parent_idx_local = np.full(D, -1, dtype=np.int32)
            if ngram_slots > 0:
                suffix = all_toks[-self.max_trie_depth :]
                draft = cache.query_bfs(suffix, ngram_slots, self.max_bfs_breadth)

                all_tokens[base + 1 : base + D] = draft.tokens

                # Shift draft tree indices by +1 to account for D0
                for j in range(ngram_slots):
                    if draft.retrive_next_token[j] >= 0:
                        all_retrive_next_token[i, j + 1] = draft.retrive_next_token[j] + 1
                    if draft.retrive_next_sibling[j] >= 0:
                        all_retrive_next_sibling[i, j + 1] = draft.retrive_next_sibling[j] + 1

                # D0 -> D1 link (verified_id's first child is the first ngram token)
                all_retrive_next_token[i, 0] = 1

                # Build parent_idx: D0 is root (-1), D1..D-1 have parents shifted by +1
                parent_idx_local[0] = -1  # D0 is root
                for j in range(ngram_slots):
                    if draft.parent_idx[j] >= 0:
                        parent_idx_local[j + 1] = draft.parent_idx[j] + 1
                    else:
                        parent_idx_local[j + 1] = 0  # root children -> child of D0

            all_retrive_index[i] = np.arange(D, dtype=np.int32) + base
            parent_idx_per_req.append(parent_idx_local)

        return (
            all_tokens,
            parent_idx_per_req,
            all_retrive_index,
            all_retrive_next_token,
            all_retrive_next_sibling,
        )

    # ------------------------------------------------------------------
    # Precompile (placeholder for TPU JIT warmup)
    # ------------------------------------------------------------------

    def run_spec_decode_precompile(self):
        logger.info("[NGRAM] Ngram speculative decoding does not require precompilation.")
