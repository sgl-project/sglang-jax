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
import os

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
        self._target_vocab_size = int(target_worker.model_runner.model_config.vocab_size)

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

        # Initialize JIT for the draft model runner (skipped during __init__
        # because is_draft_worker=True).
        self.draft_model_runner.initialize_jit()

        self.draft_layers = len(self.draft_model.model.layers)
        self._init_jit_kv_materialize()

        logger.info(
            "Initialized DFLASH worker: block_size=%d, mask_token_id=%d, draft_layers=%d",
            self.block_size,
            self._mask_token_id,
            self.draft_layers,
        )
        self._debug_tokens_remaining = (
            4 if os.getenv("SGL_JAX_DFLASH_DEBUG_TOKENS") == "1" else 0
        )
        self._sample_from_seed_hidden = (
            os.getenv("SGL_JAX_DFLASH_SAMPLE_FROM_SEED_HIDDEN") == "1"
        )
        self._debug_prefix_ab = os.getenv("SGL_JAX_DFLASH_DEBUG_PREFIX_AB") == "1"
        if self._debug_tokens_remaining > 0:
            logger.warning(
                "DFLASH debug enabled: tokens=%d sample_from_seed_hidden=%s prefix_ab=%s",
                self._debug_tokens_remaining,
                self._sample_from_seed_hidden,
                self._debug_prefix_ab,
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

        sel = np.asarray(model_worker_batch.logits_indices_selector)
        verified_id = np.asarray(jax.device_get(next_token_ids))[sel].astype(np.int32)
        extend_seq_lens = np.asarray(model_worker_batch.extend_seq_lens, dtype=np.int32)[
            sel
        ]
        extend_prefix_lens = np.asarray(
            model_worker_batch.extend_prefix_lens, dtype=np.int32
        )[sel]

        draft_input = DFlashDraftInput(
            verified_id=verified_id,
            target_hidden=target_hidden,
            ctx_lens=extend_seq_lens,
            draft_seq_lens=extend_prefix_lens,
            block_size=self.block_size,
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
        target_prefix_lens = seq_lens - 1
        draft_prefix_lens = np.asarray(draft_input.draft_seq_lens, dtype=np.int32)
        if draft_prefix_lens.shape[0] != bs:
            draft_prefix_lens = draft_prefix_lens[:bs]
            draft_input.draft_seq_lens = draft_prefix_lens
            draft_input.verified_id = np.asarray(draft_input.verified_id, dtype=np.int32)[:bs]
            draft_input.ctx_lens = np.asarray(draft_input.ctx_lens, dtype=np.int32)[:bs]

        # 1) Draft the fixed block and greedily sample draft proposals.
        verify_input = self._draft_block(
            model_worker_batch,
            draft_input,
            target_prefix_lens,
            draft_prefix_lens,
            bs,
        )

        # 2) Target verify over the block.
        verify_input.prepare_for_verify(model_worker_batch, self.page_size)
        model_worker_batch.forward_mode = ForwardMode.TARGET_VERIFY
        model_worker_batch.seq_lens = target_prefix_lens
        self._rebuild_cache_loc(model_worker_batch, target_prefix_lens, bs)
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
        if self._debug_tokens_remaining > 0:
            self._debug_tokens_remaining -= 1
            candidates_np = np.asarray(jax.device_get(verify_input.draft_token)).reshape(
                bs, self.block_size
            )
            target_predict_np = np.asarray(
                jax.device_get(
                    jnp.argmax(logits_output.next_token_logits, axis=-1).astype(jnp.int32)
                )
            ).reshape(bs, self.block_size)
            logger.warning(
                "DFLASH token debug: seq_lens=%s target_prefix=%s draft_prefix=%s "
                "accept=%s candidates0=%s target0=%s",
                seq_lens.tolist(),
                target_prefix_lens.tolist(),
                draft_prefix_lens.tolist(),
                np.asarray(jax.device_get(accept_lens_out), dtype=np.int32).tolist(),
                candidates_np[0].tolist() if len(candidates_np) else [],
                target_predict_np[0].tolist() if len(target_predict_np) else [],
            )
        accept_lens_np = np.asarray(jax.device_get(accept_lens_out), dtype=np.int32)
        new_seq_lens = seq_lens + accept_lens_np
        new_draft_kv_lens = target_prefix_lens + accept_lens_np

        # 4) Update draft state for the next step.
        next_target_hidden = self._slice_committed_target_hidden(
            logits_output.hidden_states, accept_lens_np, bs
        )
        next_draft_input = DFlashDraftInput(
            verified_id=np.asarray(jax.device_get(new_verified_id), dtype=np.int32),
            target_hidden=next_target_hidden,
            ctx_lens=accept_lens_np,
            draft_seq_lens=new_draft_kv_lens.astype(np.int32),
            block_size=self.block_size,
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
        target_prefix_lens: np.ndarray,
        draft_prefix_lens: np.ndarray,
        bs: int,
    ) -> DFlashVerifyInput:
        block_ids, positions = build_dflash_draft_block(
            verified_id=draft_input.verified_id,
            mask_token_id=self._mask_token_id,
            target_prefix_lens=target_prefix_lens,
            block_size=self.block_size,
        )
        block_ids_flat = block_ids.reshape(-1)
        positions_flat = positions.reshape(-1)

        # Draft input embedding comes from the borrowed target embedding table.
        from jax.sharding import NamedSharding, PartitionSpec as P

        embed = self._target_embed
        ids = jnp.asarray(block_ids_flat)
        out_sharding = NamedSharding(
            self.draft_model_runner.mesh, P(None, None)
        )
        input_embedding = embed.at[ids].get(out_sharding=out_sharding)

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
                draft_prefix_lens,
                bs,
            )
            forward_batch = ForwardBatch.init_new(draft_mwb, self.draft_model_runner)
            forward_batch.forward_mode = ForwardMode.TARGET_VERIFY
            forward_batch.input_embedding = input_embedding
            metadata = self.draft_model_runner.attn_backend.get_eagle_forward_metadata(draft_mwb)
            self.draft_model_runner.attn_backend.forward_metadata = metadata
            draft_logits_output, _, _ = self.draft_model_runner.forward(forward_batch, None)

            if self._debug_tokens_remaining > 0 and self._debug_prefix_ab:
                zero_prefix_lens = np.zeros_like(draft_prefix_lens)
                no_prefix_mwb = self._make_draft_block_mwb(
                    model_worker_batch,
                    block_ids_flat,
                    positions_flat,
                    block_cache_loc,
                    zero_prefix_lens,
                    bs,
                )
                no_prefix_batch = ForwardBatch.init_new(
                    no_prefix_mwb, self.draft_model_runner
                )
                no_prefix_batch.forward_mode = ForwardMode.TARGET_VERIFY
                no_prefix_batch.input_embedding = input_embedding
                no_prefix_metadata = (
                    self.draft_model_runner.attn_backend.get_eagle_forward_metadata(
                        no_prefix_mwb
                    )
                )
                self.draft_model_runner.attn_backend.forward_metadata = no_prefix_metadata
                no_prefix_output, _, _ = self.draft_model_runner.forward(
                    no_prefix_batch, None
                )
        finally:
            allocator.restore_state(state)

        draft_hidden = draft_logits_output.hidden_states.reshape(bs, self.block_size, -1)
        # Greedy sample draft proposals d1..d15 from the target LM head.
        proposal_hidden = (
            draft_hidden[:, :-1, :]
            if self._sample_from_seed_hidden
            else draft_hidden[:, 1:, :]
        )
        draft_next = self._greedy_sample_from_head(
            proposal_hidden.reshape(-1, draft_hidden.shape[-1])
        ).reshape(bs, self.block_size - 1)
        if self._debug_tokens_remaining > 0:
            proposal_flat = proposal_hidden.reshape(-1, draft_hidden.shape[-1])
            vocab_size = self._lm_head_vocab_size()
            proposal_logits = (proposal_flat @ self._target_lm_head.T)[:, :vocab_size]
            top_vals, top_ids = jax.lax.top_k(proposal_logits, k=2)
            hidden_norms = jnp.linalg.norm(draft_hidden[0].astype(jnp.float32), axis=-1)
            margin = (top_vals[:, 0] - top_vals[:, 1]).reshape(bs, self.block_size - 1)
            block_delta = jnp.max(
                jnp.abs(
                    draft_hidden[0, 1:].astype(jnp.float32)
                    - draft_hidden[0, 1:2].astype(jnp.float32)
                ),
                axis=-1,
            )
            logger.warning(
                "DFLASH draft stats: hidden_norm0=%s top1_0=%s margin0=%s "
                "block_delta0=%s",
                np.asarray(jax.device_get(hidden_norms), dtype=np.float32).round(3).tolist(),
                np.asarray(jax.device_get(top_ids[:, 0]), dtype=np.int32)
                .reshape(bs, self.block_size - 1)[0]
                .tolist(),
                np.asarray(jax.device_get(margin), dtype=np.float32).round(3)[0].tolist(),
                np.asarray(jax.device_get(block_delta), dtype=np.float32).round(5).tolist(),
            )
            if self._debug_prefix_ab:
                no_prefix_hidden = no_prefix_output.hidden_states.reshape(
                    bs, self.block_size, -1
                )
                no_prefix_proposal = (
                    no_prefix_hidden[:, :-1, :]
                    if self._sample_from_seed_hidden
                    else no_prefix_hidden[:, 1:, :]
                )
                no_prefix_logits = (
                    no_prefix_proposal.reshape(-1, no_prefix_hidden.shape[-1])
                    @ self._target_lm_head.T
                )[:, :vocab_size]
                no_prefix_top = jnp.argmax(no_prefix_logits, axis=-1).astype(jnp.int32)
                prefix_effect = jnp.max(
                    jnp.abs(
                        proposal_hidden.astype(jnp.float32)
                        - no_prefix_proposal.astype(jnp.float32)
                    )
                )
                logger.warning(
                    "DFLASH prefix A/B: prefix_lens=%s no_prefix_top1_0=%s "
                    "max_hidden_diff=%.6f",
                    draft_prefix_lens.tolist(),
                    np.asarray(jax.device_get(no_prefix_top), dtype=np.int32)
                    .reshape(bs, self.block_size - 1)[0]
                    .tolist(),
                    float(jax.device_get(prefix_effect)),
                )

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
        prefix_lens: np.ndarray,
        bs: int,
    ) -> ModelWorkerBatch:
        mwb = copy.copy(base_mwb)
        mwb.forward_mode = ForwardMode.TARGET_VERIFY
        mwb.input_ids = np.asarray(block_ids_flat, dtype=np.int32)
        mwb.positions = np.asarray(positions_flat, dtype=np.int32)
        mwb.out_cache_loc = np.asarray(block_cache_loc, dtype=np.int32)
        mwb.seq_lens = np.asarray(prefix_lens, dtype=np.int32)
        mwb.capture_hidden_mode = CaptureHiddenMode.NULL
        mwb.spec_algorithm = SpeculativeAlgorithm.DFLASH
        mwb.spec_info_padded = DFlashVerifyInput(
            draft_token=jnp.asarray(block_ids_flat),
            positions=jnp.asarray(positions_flat),
            draft_token_num=self.block_size,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        # The base MWB from speculative decode has an empty cache_loc placeholder.
        # Rebuild it from req_to_token_pool so the attention kernel knows which
        # pages hold the existing prefix KV (shared between target and draft).
        # In TARGET_VERIFY mode the kernel adds extend_seq_lens to seq_lens,
        # so cache_loc must cover prefix + block slots.
        req_to_token = self.draft_model_runner.req_to_token_pool.req_to_token
        req_pool_indices = np.asarray(base_mwb.req_pool_indices, dtype=np.int64)
        block_loc = np.asarray(block_cache_loc, dtype=np.int32)
        locs = []
        prefix_rows = []
        prefix_kv_lens = []
        kv_lens = []
        blk_offset = 0
        for i in range(bs):
            sl = int(prefix_lens[i])
            if sl > 0:
                rp = int(req_pool_indices[i])
                prefix_locs = np.asarray(req_to_token[rp, :sl], dtype=np.int32)
            else:
                prefix_locs = np.empty(0, dtype=np.int32)
            blk_locs = block_loc[blk_offset : blk_offset + self.block_size]
            prefix_rows.append(prefix_locs)
            prefix_kv_lens.append(sl)
            locs.append(np.concatenate([prefix_locs, blk_locs]))
            kv_lens.append(sl + self.block_size)
            blk_offset += self.block_size
        prefix_cache_loc = self._pack_cache_loc_rows(prefix_rows, prefix_kv_lens).reshape(bs, -1)
        mwb.spec_info_padded = DFlashVerifyInput(
            draft_token=jnp.asarray(block_ids_flat),
            positions=jnp.asarray(positions_flat),
            draft_token_num=self.block_size,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            prefix_cache_loc=jnp.asarray(prefix_cache_loc, dtype=jnp.int32),
            prefix_lens=jnp.asarray(prefix_lens, dtype=jnp.int32),
            dense_draft=True,
        )
        mwb.cache_loc = self._pack_cache_loc_rows(locs, kv_lens)
        return mwb

    # ------------------------------------------------------------------ #
    # KV materialization + greedy head sampling
    # ------------------------------------------------------------------ #
    def _init_jit_kv_materialize(self):
        """Build a single JIT function for materialize_kv + merge + KV write.

        Without this, each decode step dispatches ~60 individual JAX ops
        (FC, 5×k/v proj, 5×merge, 5×scatter) with ~25 ms tracing overhead
        each, totalling ~1.5 s.  One JIT call replaces them all.
        """
        from functools import partial as _partial

        from flax import nnx

        from sgl_jax.srt.mem_cache.memory_pool import _set_fused_kv_buffer, merge_kv

        runner = self.draft_model_runner
        pool = runner.token_to_kv_pool
        page_size = pool.page_size
        kv_part = pool.kv_partition_axis
        data_part = pool.attention_data_partition_axis
        mesh = pool.mesh
        n_layers = self.draft_layers

        model_def = runner._model_def
        model_state_def = runner._model_state_def

        @_partial(
            jax.jit,
            static_argnames=["model_state_def"],
            donate_argnames=["kv_buffers"],
        )
        def _jit_fn(
            model_def,
            model_state_def,
            model_state_leaves,
            target_hidden,
            positions,
            cache_loc,
            kv_buffers,
        ):
            state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
            model = nnx.merge(model_def, state)
            kv_list = model.materialize_kv(target_hidden, positions)
            new_bufs = []
            for i in range(n_layers):
                k, v = kv_list[i]
                fused = merge_kv(k, v)
                new_bufs.append(
                    _set_fused_kv_buffer(
                        fused,
                        cache_loc,
                        kv_buffers[i],
                        page_size,
                        kv_part,
                        data_part,
                        mesh,
                    )
                )
            return new_bufs

        self._jit_materialize_write = _partial(
            _jit_fn, model_def, model_state_def
        )

    def _append_target_hidden_to_draft_kv(
        self, model_worker_batch: ModelWorkerBatch, draft_input: DFlashDraftInput, is_prefill: bool
    ) -> None:
        """Project committed target hidden and write draft K/V at their cache_loc."""
        target_hidden = draft_input.target_hidden
        if target_hidden is None or int(np.asarray(draft_input.ctx_lens).sum()) == 0:
            return

        cache_loc = self._committed_cache_loc(model_worker_batch, draft_input, is_prefill)
        positions = self._committed_positions(draft_input, is_prefill)

        n = cache_loc.shape[0]
        if not is_prefill and n > 0:
            bs = int(np.asarray(draft_input.ctx_lens).shape[0])
            max_tokens = bs * self.block_size
            if n < max_tokens:
                pad_n = max_tokens - n
                cache_loc = np.concatenate(
                    [cache_loc, np.full(pad_n, cache_loc[0], dtype=cache_loc.dtype)]
                )
                positions = np.concatenate(
                    [positions, np.full(pad_n, positions[0], dtype=positions.dtype)]
                )
                target_hidden = np.asarray(target_hidden)
                target_hidden = np.concatenate(
                    [target_hidden, np.tile(target_hidden[0:1], (pad_n, 1))]
                )

        pool = self.draft_model_runner.token_to_kv_pool
        kv_buffers = list(pool.kv_buffer[: self.draft_layers])

        new_buffers = self._jit_materialize_write(
            self.draft_model_runner.model_state_leaves,
            jnp.asarray(target_hidden),
            jnp.asarray(positions),
            jnp.asarray(cache_loc),
            kv_buffers,
        )
        for i, buf in enumerate(new_buffers):
            pool.kv_buffer[i] = buf

        starts, lengths = dflash_committed_slices(
            draft_input.ctx_lens, draft_input.draft_seq_lens, is_prefill
        )
        draft_input.draft_seq_lens = (starts + lengths).astype(np.int32)
        draft_input.ctx_lens = np.zeros_like(np.asarray(draft_input.ctx_lens, dtype=np.int32))
        draft_input.target_hidden = target_hidden[:0]

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
        vocab_size = self._lm_head_vocab_size()
        logits = (hidden_states @ self._target_lm_head.T)[:, :vocab_size]
        return jnp.argmax(logits, axis=-1).astype(jnp.int32)

    def _lm_head_vocab_size(self) -> int:
        # Unit tests instantiate this class via __new__; avoid __getattr__ fallback there.
        return int(self.__dict__.get("_target_vocab_size", self._target_lm_head.shape[0]))

    def _rebuild_cache_loc(self, mwb: ModelWorkerBatch, prefix_lens: np.ndarray, bs: int):
        """Rebuild cache_loc from req_to_token_pool + out_cache_loc for verify."""
        req_to_token = self.target_worker.model_runner.req_to_token_pool.req_to_token
        req_pool_indices = np.asarray(mwb.req_pool_indices, dtype=np.int64)
        out_cache_loc = np.asarray(mwb.out_cache_loc, dtype=np.int32)
        locs = []
        kv_lens = []
        ocl_offset = 0
        for i in range(bs):
            sl = int(prefix_lens[i])
            rp = int(req_pool_indices[i])
            if sl > 0:
                prefix_locs = np.asarray(req_to_token[rp, :sl], dtype=np.int32)
            else:
                prefix_locs = np.empty(0, dtype=np.int32)
            n_verify = self.block_size
            verify_locs = out_cache_loc[ocl_offset : ocl_offset + n_verify]
            locs.append(np.concatenate([prefix_locs, verify_locs]))
            kv_lens.append(sl + n_verify)
            ocl_offset += n_verify
        mwb.cache_loc = self._pack_cache_loc_rows(locs, kv_lens)

    def _pack_cache_loc_rows(
        self, rows: list[np.ndarray], kv_lens: list[int]
    ) -> np.ndarray:
        """Pack per-request cache_loc rows into a bucket-stable flat layout.

        ``ForwardBatch.cache_loc`` is a JIT-visible child. If its length grows
        with sequence length, TPU decode retraces repeatedly. The FA metadata
        already expects each request row to have a uniform page stride, so pad
        the row width to the same min-16/power-of-two page bucket used by
        ``get_eagle_forward_metadata``.
        """
        if not rows:
            return np.empty(0, dtype=np.int32)

        max_kv = max(int(x) for x in kv_lens)
        pages_per_seq = (max_kv + self.page_size - 1) // self.page_size
        bucketed_pages = max(16, 1 << max(0, (pages_per_seq - 1)).bit_length())
        row_width = bucketed_pages * self.page_size

        packed = np.zeros((len(rows), row_width), dtype=np.int32)
        for i, row in enumerate(rows):
            n = min(len(row), row_width)
            packed[i, :n] = np.asarray(row[:n], dtype=np.int32)
        return packed.reshape(-1)

    def _slice_committed_target_hidden(self, hidden_states, accept_lens_np, bs):
        """Gather aux hidden for the committed verify tokens (front of each block)."""
        hs = np.asarray(jax.device_get(hidden_states)).reshape(bs, self.block_size, -1)
        rows = []
        for i in range(bs):
            a = int(accept_lens_np[i])
            if a <= 0:
                continue
            rows.append(hs[i, :a, :])
        if not rows:
            return np.zeros((0, hs.shape[-1]), dtype=hs.dtype)
        return np.concatenate(rows, axis=0)

    # ------------------------------------------------------------------ #
    # Precompile hook (scheduler calls this at startup)
    # ------------------------------------------------------------------ #
    def run_spec_decode_precompile(self):
        # Minimal bring-up: rely on the first real batch to trigger compilation.
        # A dedicated precompile sweep over bs paddings can be added once the
        # single-batch path is validated on TPU.
        logger.info("[DFLASH] run_spec_decode_precompile: deferred to first batch.")
