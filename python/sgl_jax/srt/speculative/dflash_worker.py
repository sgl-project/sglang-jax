"""DFlash speculative decoding worker (stage C, greedy runtime path).

Scope: tp_size=1, dp_size=1, disable_overlap_schedule, greedy-only, page_size=1.
Ports the algorithmic structure of SGLang PyTorch PR 22077 to the sglang-jax
scheduler/worker contract. See ``docs/design/dflash_stage_c.md``.

The worker exposes the surface the scheduler needs
(``speculative_num_draft_tokens``, ``forward_batch_speculative_generation``,
``run_spec_decode_precompile``) and delegates everything else to the target
worker via ``__getattr__``.
"""

from __future__ import annotations

import copy
import logging

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sgl_jax.srt.speculative.dflash_info import (
    DFlashDraftInput,
    DFlashVerifyInput,
    build_dflash_draft_block,
    dflash_committed_slices,
)
from sgl_jax.srt.speculative.dflash_util import parse_dflash_draft_config, resolve_mask_token_id
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


class DFlashWorker:
    """DFlash draft/verify runtime worker (greedy, single-chip)."""

    def __init__(self, server_args, target_worker: ModelWorker):
        self.server_args = server_args
        self.target_worker = target_worker
        self.mesh = target_worker.mesh
        self.page_size = server_args.page_size
        self.device = server_args.device

        self.speculative_num_draft_tokens = int(server_args.speculative_num_draft_tokens)
        self.block_size = self.speculative_num_draft_tokens
        self.speculative_num_steps = int(server_args.speculative_num_steps)

        # ---- Build the draft ModelWorker (shares the target req->token pool) ----
        # sglang-jax's ModelWorker does not accept a KV allocator; EAGLE only
        # shares req_to_token_pool. We additionally alias the target's KV
        # allocator after construction so committed tokens land at the same
        # cache_loc in both target and draft KV pools (design decision:
        # "share allocator + independent KV buffer").
        req_to_token_pool, target_allocator = target_worker.get_memory_pool()

        draft_server_args = copy.deepcopy(server_args)
        draft_server_args.skip_tokenizer_init = True

        from sgl_jax.srt.models.dflash import DFlashDraftModel

        self._worker = ModelWorker(
            server_args=draft_server_args,
            mesh=self.mesh,
            req_to_token_pool=req_to_token_pool,
            is_draft_worker=True,
            model_class=DFlashDraftModel,
        )
        self.draft_model_runner = self._worker.model_runner
        self.draft_model = self.draft_model_runner.model

        # Alias the KV allocator so draft block allocation draws from the same
        # free list the target uses (no collision with committed slots).
        self.draft_model_runner.token_to_kv_pool_allocator = target_allocator
        self.token_to_kv_pool_allocator = target_allocator

        # ---- Borrow the target embedding + LM head (draft owns neither) ----
        target_model = target_worker.model_runner.model
        embed_weight, head_weight = target_model.get_embed_and_head()
        self.draft_model.set_embed_and_head(embed_weight, head_weight)
        self._target_lm_head = head_weight  # [vocab, hidden], for greedy head sampling
        self._target_embed = embed_weight  # [vocab, hidden]

        # ---- Resolve the draft mask token ----
        dflash_config = parse_dflash_draft_config(
            server_args.speculative_draft_model_path,
            revision=server_args.speculative_draft_model_revision,
        )
        self._mask_token_id = resolve_mask_token_id(
            dflash_config,
            getattr(target_worker, "tokenizer", None),
            vocab_size=int(target_worker.model_runner.model_config.vocab_size),
        )
        logger.info(
            "Initialized DFLASH worker: block_size=%d, mask_token_id=%d, draft_layers=%d",
            self.block_size,
            self._mask_token_id,
            len(self.draft_model.model.layers),
        )

    # Delegate anything not implemented here to the target worker.
    def __getattr__(self, name):
        # __getattr__ only fires for missing attributes; forward to target.
        return getattr(self.target_worker, name)

    # ------------------------------------------------------------------ #
    # Scheduler entry point
    # ------------------------------------------------------------------ #
    def forward_batch_speculative_generation(
        self, model_worker_batch: ModelWorkerBatch, launch_done=None
    ):
        if model_worker_batch.forward_mode.is_extend():
            return self._forward_prefill(model_worker_batch)
        return self._forward_decode(model_worker_batch)

    # ------------------------------------------------------------------ #
    # Prefill: target forward (capture FULL) -> materialize draft KV
    # ------------------------------------------------------------------ #
    def _forward_prefill(self, model_worker_batch: ModelWorkerBatch):
        from sgl_jax.srt.managers.scheduler import GenerationBatchResult
        from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata

        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL

        sampling_metadata = SamplingMetadata.from_model_worker_batch(
            model_worker_batch,
            len(model_worker_batch.seq_lens) - model_worker_batch.real_bs,
            self.mesh,
            vocab_size=self.target_worker.model_runner.model_config.vocab_size,
        )
        logits_output, _, cache_miss_count = self.target_worker.forward_batch_generation(
            model_worker_batch,
            sampling_metadata=sampling_metadata,
            skip_sample=True,
        )
        next_token_ids = jnp.argmax(logits_output.next_token_logits, axis=-1).astype(jnp.int32)

        # Aux hidden features for every prompt token: [sum(extend_seq_lens), K*H].
        target_hidden = logits_output.hidden_states
        extend_seq_lens = np.asarray(model_worker_batch.extend_seq_lens, dtype=np.int32)

        sel = np.asarray(model_worker_batch.logits_indices_selector)
        verified_id = np.asarray(jax.device_get(next_token_ids))[sel].astype(np.int32)

        draft_input = DFlashDraftInput(
            verified_id=verified_id,
            target_hidden=target_hidden,
            ctx_lens=extend_seq_lens,
            draft_seq_lens=np.asarray(model_worker_batch.extend_prefix_lens, dtype=np.int32),
        )

        # Materialize prompt hidden into the draft KV pool at the committed slots.
        self._append_target_hidden_to_draft_kv(model_worker_batch, draft_input, is_prefill=True)

        model_worker_batch.spec_info_padded = draft_input
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            next_draft_input=draft_input,
            bid=model_worker_batch.bid,
            cache_miss_count=cache_miss_count,
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
        )

    # ------------------------------------------------------------------ #
    # Decode: draft block -> target verify -> accept/commit
    # ------------------------------------------------------------------ #
    def _forward_decode(self, model_worker_batch: ModelWorkerBatch):
        from sgl_jax.srt.managers.scheduler import GenerationBatchResult

        draft_input: DFlashDraftInput = model_worker_batch.spec_info_padded
        assert isinstance(draft_input, DFlashDraftInput), (
            "DFLASH decode requires DFlashDraftInput carried over from prefill."
        )

        bs = int(model_worker_batch.seq_lens.shape[0])
        seq_lens = np.asarray(model_worker_batch.seq_lens, dtype=np.int32)

        # 1) Draft the fixed block and greedily sample draft proposals.
        verify_input = self._draft_block(model_worker_batch, draft_input, seq_lens, bs)

        # 2) Target verify over the block.
        verify_input.prepare_for_verify(model_worker_batch, self.page_size)
        forward_metadata = self.target_worker.model_runner.attn_backend.get_eagle_forward_metadata(
            model_worker_batch
        )
        logits_output, _, cache_miss_count = self.target_worker.forward_batch_generation(
            model_worker_batch, skip_sample=True, forward_metadata=forward_metadata
        )

        # 3) Greedy accept + bonus.
        accept_lens_out, next_token_ids_flat, new_verified_id, _accept_draft = verify_input.verify(
            logits_output.next_token_logits
        )
        accept_lens_np = np.asarray(jax.device_get(accept_lens_out), dtype=np.int32)
        new_seq_lens = seq_lens + accept_lens_np

        # 4) Update draft state for the next step: new seed + committed target
        #    hidden appended into the draft KV pool.
        next_target_hidden = self._slice_committed_target_hidden(
            logits_output.hidden_states, accept_lens_np, bs
        )
        next_draft_input = DFlashDraftInput(
            verified_id=np.asarray(jax.device_get(new_verified_id), dtype=np.int32),
            target_hidden=next_target_hidden,
            ctx_lens=accept_lens_np,
            draft_seq_lens=new_seq_lens.astype(np.int32),
        )
        next_draft_input.new_seq_lens = new_seq_lens.astype(np.int32)
        self._append_target_hidden_to_draft_kv(
            model_worker_batch, next_draft_input, is_prefill=False
        )
        model_worker_batch.spec_info_padded = next_draft_input

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids_flat,
            next_draft_input=next_draft_input,
            accept_lens=accept_lens_out,
            bid=model_worker_batch.bid,
            cache_miss_count=cache_miss_count,
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
        )

    # ------------------------------------------------------------------ #
    # Draft block forward
    # ------------------------------------------------------------------ #
    def _draft_block(
        self,
        model_worker_batch: ModelWorkerBatch,
        draft_input: DFlashDraftInput,
        seq_lens: np.ndarray,
        bs: int,
    ) -> DFlashVerifyInput:
        block_ids, positions = build_dflash_draft_block(
            verified_id=draft_input.verified_id,
            mask_token_id=self._mask_token_id,
            target_prefix_lens=seq_lens,
            block_size=self.block_size,
        )
        block_ids_flat = block_ids.reshape(-1)
        positions_flat = positions.reshape(-1)

        # Draft input embedding comes from the borrowed target embedding table.
        input_embedding = jnp.take(self._target_embed, jnp.asarray(block_ids_flat), axis=0)

        # Temporary draft-block KV slots (dropped after the draft forward).
        allocator = self.draft_model_runner.token_to_kv_pool_allocator
        state = allocator.backup_state()
        try:
            block_cache_loc = allocator.alloc(bs * self.block_size)
            if block_cache_loc is None:
                raise RuntimeError(
                    f"DFLASH draft OOM allocating {bs * self.block_size} block tokens."
                )
            draft_mwb = self._make_draft_block_mwb(
                model_worker_batch,
                block_ids_flat,
                positions_flat,
                block_cache_loc,
                seq_lens,
                bs,
            )
            forward_batch = ForwardBatch.init_new(draft_mwb, self.draft_model_runner)
            forward_batch.forward_mode = ForwardMode.TARGET_VERIFY
            forward_batch.input_embedding = input_embedding
            metadata = self.draft_model_runner.attn_backend.get_eagle_forward_metadata(draft_mwb)
            self.draft_model_runner.attn_backend.forward_metadata = metadata
            draft_logits_output, _, _ = self.draft_model_runner.forward(forward_batch)
        finally:
            allocator.restore_state(state)

        draft_hidden = draft_logits_output.hidden_states.reshape(bs, self.block_size, -1)
        # Greedy sample draft proposals d1..d15 from the target LM head.
        draft_next = self._greedy_sample_from_head(
            draft_hidden[:, 1:, :].reshape(-1, draft_hidden.shape[-1])
        ).reshape(bs, self.block_size - 1)

        draft_token = np.asarray(block_ids, dtype=np.int32).copy()
        draft_token[:, 1:] = np.asarray(jax.device_get(draft_next), dtype=np.int32)

        return DFlashVerifyInput(
            draft_token=jnp.asarray(draft_token.reshape(-1)),
            positions=jnp.asarray(positions_flat),
            draft_token_num=self.block_size,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )

    def _make_draft_block_mwb(
        self,
        base_mwb: ModelWorkerBatch,
        block_ids_flat: np.ndarray,
        positions_flat: np.ndarray,
        block_cache_loc,
        seq_lens: np.ndarray,
        bs: int,
    ) -> ModelWorkerBatch:
        mwb = copy.copy(base_mwb)
        mwb.forward_mode = ForwardMode.TARGET_VERIFY
        mwb.input_ids = np.asarray(block_ids_flat, dtype=np.int32)
        mwb.positions = np.asarray(positions_flat, dtype=np.int32)
        mwb.out_cache_loc = np.asarray(block_cache_loc, dtype=np.int32)
        mwb.seq_lens = np.asarray(seq_lens, dtype=np.int32)
        mwb.capture_hidden_mode = CaptureHiddenMode.NULL
        mwb.spec_algorithm = SpeculativeAlgorithm.DFLASH
        mwb.spec_info_padded = DFlashVerifyInput(
            draft_token=jnp.asarray(block_ids_flat),
            positions=jnp.asarray(positions_flat),
            draft_token_num=self.block_size,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        return mwb

    # ------------------------------------------------------------------ #
    # KV materialization + greedy head sampling
    # ------------------------------------------------------------------ #
    def _append_target_hidden_to_draft_kv(
        self, model_worker_batch: ModelWorkerBatch, draft_input: DFlashDraftInput, is_prefill: bool
    ) -> None:
        """Project committed target hidden and write draft K/V at their cache_loc.

        For prefill this is every prompt token; for decode it is the committed
        verify tokens. The draft cache_loc mirrors the target's committed slots
        (shared req_to_token_pool), so draft attention over the prefix is valid.
        """
        target_hidden = draft_input.target_hidden
        if target_hidden is None or int(np.asarray(draft_input.ctx_lens).sum()) == 0:
            return

        cache_loc = self._committed_cache_loc(model_worker_batch, draft_input, is_prefill)
        positions = self._committed_positions(draft_input, is_prefill)

        kv_list = self.draft_model.materialize_kv(target_hidden, jnp.asarray(positions))
        token_to_kv_pool = self.draft_model_runner.token_to_kv_pool
        cache_loc_dev = jnp.asarray(cache_loc)
        for layer_id, (k, v) in enumerate(kv_list):
            token_to_kv_pool.set_kv_buffer(layer_id, cache_loc_dev, k, v)

    def _committed_cache_loc(self, model_worker_batch, draft_input, is_prefill):
        """Flat cache_loc for the committed tokens, read from req_to_token_pool."""
        req_to_token = self.draft_model_runner.req_to_token_pool.req_to_token
        req_pool_indices = np.asarray(model_worker_batch.req_pool_indices, dtype=np.int64)
        starts, lengths = dflash_committed_slices(
            draft_input.ctx_lens, draft_input.draft_seq_lens, is_prefill
        )
        locs = []
        for rp, start, n in zip(req_pool_indices, starts, lengths):
            if n <= 0:
                continue
            locs.append(np.asarray(req_to_token[rp, start : start + n]))
        if not locs:
            return np.zeros((0,), dtype=np.int32)
        return np.concatenate(locs).astype(np.int32)

    def _committed_positions(self, draft_input, is_prefill):
        starts, lengths = dflash_committed_slices(
            draft_input.ctx_lens, draft_input.draft_seq_lens, is_prefill
        )
        pos = []
        for start, n in zip(starts, lengths):
            if n <= 0:
                continue
            pos.append(np.arange(start, start + n, dtype=np.int32))
        if not pos:
            return np.zeros((0,), dtype=np.int32)
        return np.concatenate(pos).astype(np.int32)

    def _greedy_sample_from_head(self, hidden_states: jax.Array) -> jax.Array:
        """argmax over the target LM head. tp_size=1 bring-up path (full vocab)."""
        logits = hidden_states @ self._target_lm_head.T
        return jnp.argmax(logits, axis=-1).astype(jnp.int32)

    def _slice_committed_target_hidden(self, hidden_states, accept_lens_np, bs):
        """Gather aux hidden for the committed verify tokens (front of each block)."""
        hs = hidden_states.reshape(bs, self.block_size, -1)
        rows = []
        for i in range(bs):
            a = int(accept_lens_np[i])
            if a <= 0:
                continue
            rows.append(hs[i, :a, :])
        if not rows:
            return hs[:0, 0, :]
        return jnp.concatenate(rows, axis=0)

    # ------------------------------------------------------------------ #
    # Precompile hook (scheduler calls this at startup)
    # ------------------------------------------------------------------ #
    def run_spec_decode_precompile(self):
        # Minimal bring-up: rely on the first real batch to trigger compilation.
        # A dedicated precompile sweep over bs paddings can be added once the
        # single-batch path is validated on TPU.
        logger.info("[DFLASH] run_spec_decode_precompile: deferred to first batch.")
