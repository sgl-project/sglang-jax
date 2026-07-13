"""DFlash speculative decoding worker (greedy runtime path).

Scope: dp_size=1, disable_overlap_schedule, greedy-only.
Ports the algorithmic structure of SGLang PyTorch PR 22077 to the sglang-jax
scheduler/worker contract. See ``docs/design/dflash_stage_c.md``.

The worker exposes the surface the scheduler needs
(``speculative_num_draft_tokens``, ``forward_batch_speculative_generation``,
``run_spec_decode_precompile``) and delegates everything else to the target
worker via ``__getattr__``.
"""

from __future__ import annotations

import contextlib
import copy
import logging
import os

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.logits_processor import LogitsMetadata
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
    compute_new_kv_slices,
    dflash_greedy_verify_impl,
)
from sgl_jax.srt.speculative.dflash_util import (
    parse_dflash_draft_config,
    resolve_mask_token_id,
)
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


class DFlashWorker:
    """DFlash draft/verify runtime worker (greedy, dp_size=1)."""

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

        self._draft_prefix_window = int(os.getenv("SGL_JAX_DFLASH_DRAFT_PREFIX_WINDOW", "0"))

        self.draft_layers = len(self.draft_model.model.layers)
        self._init_jit_target_verify()
        self._init_jit_kv_materialize()
        self._init_jit_draft_block()

        logger.info(
            "Initialized DFLASH worker: block_size=%d, mask_token_id=%d, draft_layers=%d",
            self.block_size,
            self._mask_token_id,
            self.draft_layers,
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
        extend_seq_lens = np.asarray(model_worker_batch.extend_seq_lens, dtype=np.int32)[sel]
        extend_prefix_lens = np.asarray(model_worker_batch.extend_prefix_lens, dtype=np.int32)[sel]

        draft_input = DFlashDraftInput(
            verified_id=verified_id,
            target_hidden=target_hidden,
            ctx_lens=extend_seq_lens,
            draft_seq_lens=extend_prefix_lens,
            block_size=self.block_size,
        )
        # @haifeng Need modify
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
        self._trim_dflash_draft_input_to_decode_batch(draft_input, bs)
        draft_prefix_lens = np.asarray(draft_input.draft_seq_lens, dtype=np.int32)

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
        (
            logits_output,
            cache_miss_count,
            accept_lens_out,
            next_token_ids_flat,
            new_verified_id,
            _accept_draft,
        ) = self._run_jit_target_verify(
            model_worker_batch,
            verify_input,
            forward_metadata,
        )
        accept_lens_np = np.asarray(jax.device_get(accept_lens_out), dtype=np.int32)
        new_seq_lens = seq_lens + accept_lens_np
        new_draft_kv_lens = target_prefix_lens + accept_lens_np

        next_draft_input = DFlashDraftInput(
            verified_id=np.asarray(jax.device_get(new_verified_id), dtype=np.int32),
            target_hidden=logits_output.hidden_states,
            ctx_lens=accept_lens_np,
            draft_seq_lens=new_draft_kv_lens.astype(np.int32),
            block_size=self.block_size,
        )
        next_draft_input.new_seq_lens = new_seq_lens.astype(np.int32)
        self._append_target_hidden_to_draft_kv(
            model_worker_batch,
            next_draft_input,
            is_prefill=False,
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

    def _trim_dflash_draft_input_to_decode_batch(
        self,
        draft_input: DFlashDraftInput,
        bs: int,
    ) -> None:
        """Trim stale host-side DFlash state to the current decode batch size."""
        draft_seq_lens = np.asarray(draft_input.draft_seq_lens, dtype=np.int32)
        state_bs = int(draft_seq_lens.shape[0])
        if state_bs == bs:
            return

        verified_id = np.asarray(draft_input.verified_id, dtype=np.int32)
        ctx_lens = np.asarray(draft_input.ctx_lens, dtype=np.int32)
        if state_bs > bs:
            draft_input.draft_seq_lens = draft_seq_lens[:bs]
            draft_input.verified_id = verified_id[:bs]
            draft_input.ctx_lens = ctx_lens[:bs]
            return

        raise ValueError(
            "DFLASH draft state is shorter than decode batch after prepare_for_decode: "
            f"state_bs={state_bs}, bs={bs}. Merged decode requests must be aligned "
            "from ScheduleBatch req state before entering DFlashWorker._forward_decode."
        )

    def _init_jit_target_verify(self):
        """Build target model forward + DFlash greedy verification as one JIT."""
        from functools import partial as _partial

        from flax import nnx
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        from sgl_jax.srt.lora.context_manager import LoraBatchContext
        from sgl_jax.srt.model_executor.model_runner import _maybe_apply_recurrent_cow

        runner = self.target_worker.model_runner
        model_def = runner._model_def
        model_state_def = runner._model_state_def
        draft_token_num = self.block_size
        token_sharding = NamedSharding(runner.mesh, P(None))

        @_partial(
            jax.jit,
            donate_argnames=["memory_pools"],
            static_argnames=["model_state_def", "draft_token_num"],
        )
        def target_verify(
            model_def,
            model_state_def,
            model_state_leaves,
            forward_batch,
            memory_pools,
            logits_metadata,
            draft_token,
            *,
            draft_token_num: int,
        ):
            model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
            model = nnx.merge(model_def, model_state)
            memory_pools = _maybe_apply_recurrent_cow(forward_batch, memory_pools)
            with LoraBatchContext.set_batch(forward_batch):
                output, pool_updates, aux, layers_topk_ids = model(
                    forward_batch, memory_pools, logits_metadata
                )
            accept_lens_out, next_token_ids_flat, new_verified_id, accept_draft = (
                dflash_greedy_verify_impl(
                    draft_token,
                    output.next_token_logits,
                    draft_token_num=draft_token_num,
                )
            )
            accept_lens_out = jax.sharding.reshard(accept_lens_out, token_sharding)
            next_token_ids_flat = jax.sharding.reshard(next_token_ids_flat, token_sharding)
            new_verified_id = jax.sharding.reshard(new_verified_id, token_sharding)
            accept_draft = jax.sharding.reshard(accept_draft, token_sharding)
            return (
                output,
                pool_updates,
                aux,
                layers_topk_ids,
                accept_lens_out,
                next_token_ids_flat,
                new_verified_id,
                accept_draft,
            )

        self._jit_target_verify = _partial(
            target_verify,
            model_def,
            model_state_def,
            draft_token_num=draft_token_num,
        )

    def _run_jit_target_verify(
        self,
        model_worker_batch: ModelWorkerBatch,
        verify_input: DFlashVerifyInput,
        forward_metadata,
    ):
        import jax._src.test_util as jtu

        target_worker = self.target_worker
        runner = target_worker.model_runner

        if target_worker.worker.server_args.enable_lora and target_worker.need_prepare_lora_batch:
            target_worker.prepare_lora_batch(model_worker_batch)

        if model_worker_batch.forward_batch is not None:
            forward_batch = model_worker_batch.forward_batch
        else:
            forward_batch = ForwardBatch.init_new(model_worker_batch, runner)
        forward_batch.input_ids = verify_input.draft_token

        runner.attn_backend.forward_metadata = forward_metadata
        logits_metadata = LogitsMetadata.from_model_worker_batch(model_worker_batch, self.mesh)

        def _call_and_replace():
            with jtu.count_pjit_cpp_cache_miss() as count:
                (
                    output,
                    pool_updates,
                    _,
                    layers_topk_ids,
                    accept_lens_out,
                    next_token_ids_flat,
                    new_verified_id,
                    accept_draft,
                ) = self._jit_target_verify(
                    runner.model_state_leaves,
                    forward_batch,
                    runner.memory_pools,
                    logits_metadata,
                    verify_input.draft_token,
                )
                cache_miss_count = count()

            if runner.tp_size == 1 and isinstance(pool_updates, list):
                target_sharding = runner.token_to_kv_pool.kv_sharding
                pool_updates = [jax.device_put(kv, target_sharding) for kv in pool_updates]
            runner.memory_pools.replace_all(pool_updates)
            return (
                output,
                cache_miss_count,
                layers_topk_ids,
                accept_lens_out,
                next_token_ids_flat,
                new_verified_id,
                accept_draft,
            )

        _kv_lock = getattr(runner.token_to_kv_pool, "_donate_lock", None)
        try:
            mesh_ctx = jax.sharding.use_mesh(self.mesh)
        except AttributeError:
            try:
                mesh_ctx = jax.set_mesh(self.mesh)
            except AttributeError:
                mesh_ctx = self.mesh

        lock_ctx = _kv_lock if _kv_lock is not None else contextlib.nullcontext()
        with mesh_ctx, lock_ctx:
            (
                output,
                cache_miss_count,
                layers_topk_ids,
                accept_lens_out,
                next_token_ids_flat,
                new_verified_id,
                accept_draft,
            ) = _call_and_replace()

        target_worker.dump_topk_ids(layers_topk_ids, model_worker_batch)
        target_worker.sync_queue.put((layers_topk_ids, model_worker_batch))

        return (
            output,
            cache_miss_count,
            accept_lens_out,
            next_token_ids_flat,
            new_verified_id,
            accept_draft,
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

        # Temporary draft-block KV slots (dropped after the draft forward).
        allocator = self.draft_model_runner.token_to_kv_pool_allocator
        state = allocator.backup_state()
        try:
            num_block_tokens = bs * self.block_size
            alloc_tokens = (
                (num_block_tokens + self.page_size - 1) // self.page_size
            ) * self.page_size
            block_cache_loc = allocator.alloc(alloc_tokens)
            if block_cache_loc is None:
                raise RuntimeError(
                    f"DFLASH draft OOM allocating {alloc_tokens} block KV slots "
                    f"for {num_block_tokens} block tokens."
                )
            block_cache_loc = np.asarray(block_cache_loc, dtype=np.int32)[:num_block_tokens]
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
            metadata = self.draft_model_runner.attn_backend.get_eagle_forward_metadata(draft_mwb)
            self.draft_model_runner.attn_backend.forward_metadata = metadata
            draft_token_flat = self._run_jit_draft_block(forward_batch, None)
        finally:
            # release the temp kv
            allocator.restore_state(state)

        return DFlashVerifyInput(
            draft_token=draft_token_flat,
            positions=jnp.asarray(positions_flat),
            draft_token_num=self.block_size,
            input_ids_host=block_ids_flat,
            positions_host=positions_flat,
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
        block_offset = 0
        for i in range(bs):
            prefix_len = int(prefix_lens[i])
            if prefix_len > 0:
                req_idx = int(req_pool_indices[i])
                if self._draft_prefix_window > 0:
                    prefix_start = max(0, prefix_len - self._draft_prefix_window)
                else:
                    prefix_start = 0
                prefix_locs = np.asarray(
                    req_to_token[req_idx, prefix_start:prefix_len], dtype=np.int32
                )
            else:
                prefix_locs = np.empty(0, dtype=np.int32)
            draft_prefix_len = len(prefix_locs)
            block_locs = block_loc[block_offset : block_offset + self.block_size]
            prefix_rows.append(prefix_locs)
            prefix_kv_lens.append(draft_prefix_len)
            locs.append(np.concatenate([prefix_locs, block_locs]))
            kv_lens.append(draft_prefix_len + self.block_size)
            block_offset += self.block_size
        prefix_cache_loc = self._pack_kv_cache(prefix_rows, prefix_kv_lens).reshape(bs, -1)
        draft_prefix_lens = np.asarray(prefix_kv_lens, dtype=np.int32)
        mwb.seq_lens = draft_prefix_lens
        mwb.spec_info_padded = DFlashVerifyInput(
            draft_token=jnp.asarray(block_ids_flat),
            positions=jnp.asarray(positions_flat),
            draft_token_num=self.block_size,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            prefix_cache_loc=jnp.asarray(prefix_cache_loc, dtype=jnp.int32),
            prefix_lens=jnp.asarray(draft_prefix_lens, dtype=jnp.int32),
            dense_draft=True,
        )
        mwb.cache_loc = self._pack_kv_cache(locs, kv_lens)
        return mwb

    # ------------------------------------------------------------------ #
    # KV materialization + greedy head sampling
    # ------------------------------------------------------------------ #
    def _init_jit_draft_block(self):
        """Build a draft model forward JIT that also samples draft proposals."""
        from functools import partial as _partial

        from flax import nnx
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        from sgl_jax.srt.lora.context_manager import LoraBatchContext
        from sgl_jax.srt.model_executor.model_runner import _maybe_apply_recurrent_cow

        runner = self.draft_model_runner
        model_def = runner._model_def
        model_state_def = runner._model_state_def
        block_size = self.block_size
        vocab_size = self._lm_head_vocab_size()
        embedding_sharding = NamedSharding(runner.mesh, P("data", "tensor"))
        logits_sharding = NamedSharding(runner.mesh, P("data", "tensor"))
        token_sharding = NamedSharding(runner.mesh, P(None))

        @_partial(
            jax.jit,
            donate_argnames=["memory_pools"],
            static_argnames=[
                "model_state_def",
                "block_size",
                "vocab_size",
            ],
        )
        def draft(
            model_def,
            model_state_def,
            model_state_leaves,
            forward_batch,
            memory_pools,
            logits_metadata,
            embed,
            lm_head,
            *,
            block_size: int,
            vocab_size: int,
        ):
            input_embedding = embed.at[forward_batch.input_ids].get(out_sharding=embedding_sharding)
            forward_batch.input_embedding = input_embedding
            model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
            model = nnx.merge(model_def, model_state)
            memory_pools = _maybe_apply_recurrent_cow(forward_batch, memory_pools)
            with LoraBatchContext.set_batch(forward_batch):
                output, pool_updates, _, _ = model(forward_batch, memory_pools, logits_metadata)

            draft_hidden = output.hidden_states.reshape(
                (-1, block_size, output.hidden_states.shape[-1])
            )
            proposal_hidden = draft_hidden[:, 1:, :]
            proposal_flat = proposal_hidden.reshape((-1, proposal_hidden.shape[-1]))
            logits = jnp.dot(
                proposal_flat,
                lm_head.T,
                out_sharding=logits_sharding,
            )[:, :vocab_size]
            draft_next = jnp.argmax(logits, axis=-1).astype(jnp.int32)
            draft_next = draft_next.reshape(proposal_hidden.shape[:-1])
            seed = forward_batch.input_ids.reshape((-1, block_size))[:, :1]
            draft_token = jnp.concatenate([seed, draft_next], axis=1).reshape(-1)
            draft_token = jax.sharding.reshard(draft_token, token_sharding)
            return pool_updates, draft_token

        self._jit_draft_block = _partial(
            draft,
            model_def,
            model_state_def,
            block_size=block_size,
            vocab_size=vocab_size,
        )

    def _run_jit_draft_block(self, forward_batch, logits_metadata):
        runner = self.draft_model_runner

        def _call_and_replace():
            pool_updates, draft_token = self._jit_draft_block(
                runner.model_state_leaves,
                forward_batch,
                runner.memory_pools,
                logits_metadata,
                self._target_embed,
                self._target_lm_head,
            )
            if runner.tp_size == 1 and isinstance(pool_updates, list):
                target_sharding = runner.token_to_kv_pool.kv_sharding
                pool_updates = [jax.device_put(kv, target_sharding) for kv in pool_updates]
            runner.memory_pools.replace_all(pool_updates)
            return draft_token

        _kv_lock = getattr(runner.token_to_kv_pool, "_donate_lock", None)
        if _kv_lock is None:
            return _call_and_replace()
        with _kv_lock:
            return _call_and_replace()

    def _init_jit_kv_materialize(self):
        """Build a single JIT function for materialize_kv + merge + KV write.

        Without this, each decode step dispatches ~60 individual JAX ops
        (FC, 5×k/v proj, 5×merge, 5×scatter) with ~25 ms tracing overhead
        each, totalling ~1.5 s.  One JIT call replaces them all.
        """
        from functools import partial as _partial

        from flax import nnx
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        from sgl_jax.srt.mem_cache.memory_pool import _set_fused_kv_buffer, merge_kv

        runner = self.draft_model_runner
        pool = runner.token_to_kv_pool
        page_size = pool.page_size
        kv_part = pool.kv_partition_axis
        data_part = pool.attention_data_partition_axis
        mesh = pool.mesh
        n_layers = self.draft_layers
        vector_sharding = NamedSharding(mesh, P("data"))
        hidden_sharding = NamedSharding(mesh, P("data", None))

        model_def = runner._model_def
        model_state_def = runner._model_state_def

        @_partial(
            jax.jit,
            static_argnames=["model_state_def", "is_decode"],
            donate_argnames=["kv_buffers"],
        )
        def draft_extend(
            model_def,
            model_state_def,
            model_state_leaves,
            target_hidden,
            positions,
            cache_loc,
            kv_buffers,
            *,
            is_decode: bool,
        ):
            positions = jax.sharding.reshard(positions.astype(jnp.int32), vector_sharding)
            cache_loc = jax.sharding.reshard(cache_loc.astype(jnp.int32), vector_sharding)
            target_hidden = jax.sharding.reshard(target_hidden, hidden_sharding)

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

        self._jit_materialize_write = _partial(draft_extend, model_def, model_state_def)

    def _append_target_hidden_to_draft_kv(
        self,
        model_worker_batch: ModelWorkerBatch,
        draft_input: DFlashDraftInput,
        is_prefill: bool,
    ) -> None:
        """Project committed target hidden and write draft K/V at their cache_loc."""
        target_hidden = draft_input.target_hidden
        if target_hidden is None or int(np.asarray(draft_input.ctx_lens).sum()) == 0:
            return

        if is_prefill:
            cache_loc = self._committed_cache_loc(model_worker_batch, draft_input, is_prefill)
            positions = self._committed_positions(draft_input, is_prefill)
        else:
            cache_loc, positions = self._committed_decode_row_padded_cache_loc_positions(
                model_worker_batch, draft_input
            )

        pool = self.draft_model_runner.token_to_kv_pool
        kv_buffers = list(pool.kv_buffer[: self.draft_layers])

        new_buffers = self._jit_materialize_write(
            self.draft_model_runner.model_state_leaves,
            jnp.asarray(target_hidden),
            jnp.asarray(positions),
            jnp.asarray(cache_loc),
            kv_buffers,
            is_decode=not is_prefill,
        )
        for i, buf in enumerate(new_buffers):
            pool.kv_buffer[i] = buf

        starts, lengths = compute_new_kv_slices(
            draft_input.ctx_lens, draft_input.draft_seq_lens, is_prefill
        )
        draft_input.draft_seq_lens = (starts + lengths).astype(np.int32)
        draft_input.ctx_lens = np.zeros_like(np.asarray(draft_input.ctx_lens, dtype=np.int32))
        draft_input.target_hidden = None if not is_prefill else target_hidden[:0]

    def _committed_cache_loc(self, model_worker_batch, draft_input, is_prefill):
        """Flat cache_loc for the committed tokens, read from req_to_token_pool."""
        req_to_token = self.draft_model_runner.req_to_token_pool.req_to_token
        req_pool_indices = np.asarray(model_worker_batch.req_pool_indices, dtype=np.int64)
        starts, lengths = compute_new_kv_slices(
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

    def _committed_decode_row_padded_cache_loc_positions(self, model_worker_batch, draft_input):
        """Decode KV write layout aligned with target verify hidden rows.

        Decode target hidden is row-major ``[bs, block_size, hidden]``. Keep the
        same fixed row layout for KV materialization and mark unaccepted columns
        with ``cache_loc=-1`` so the KV update kernel drops them. This avoids a
        device-side accepted-hidden compaction based on ``accept_lens``.
        """
        req_to_token = self.draft_model_runner.req_to_token_pool.req_to_token
        req_pool_indices = np.asarray(model_worker_batch.req_pool_indices, dtype=np.int64)
        starts, lengths = compute_new_kv_slices(
            draft_input.ctx_lens, draft_input.draft_seq_lens, is_prefill=False
        )
        bs = int(len(req_pool_indices))
        cache_loc = np.full((bs, self.block_size), -1, dtype=np.int32)
        positions = np.zeros((bs, self.block_size), dtype=np.int32)
        active_mask = self._active_decode_slot_mask(model_worker_batch, bs)
        for i, (rp, start, n) in enumerate(zip(req_pool_indices, starts, lengths)):
            valid_n = min(int(n), self.block_size)
            start = int(start)
            positions[i, :] = max(start, 0)
            if not active_mask[i] or valid_n <= 0 or start < 0:
                continue
            locs = np.asarray(req_to_token[int(rp), start : start + valid_n], dtype=np.int32)
            valid_n = min(valid_n, int(locs.shape[0]))
            if valid_n <= 0:
                continue
            cache_loc[i, :valid_n] = locs[:valid_n]
            positions[i, :valid_n] = np.arange(start, start + valid_n, dtype=np.int32)
            positions[i, valid_n:] = start + valid_n - 1
        return cache_loc.reshape(-1), positions.reshape(-1)

    @staticmethod
    def _active_decode_slot_mask(model_worker_batch, total_bs: int) -> np.ndarray:
        mask = np.zeros(total_bs, dtype=bool)
        real_bs_per_dp = getattr(model_worker_batch, "real_bs_per_dp", None)
        if real_bs_per_dp is None:
            mask[: int(getattr(model_worker_batch, "real_bs", total_bs))] = True
            return mask

        per_dp_bs = int(getattr(model_worker_batch, "per_dp_bs_size", total_bs))
        for dp_rank, real_bs in enumerate(real_bs_per_dp):
            start = dp_rank * per_dp_bs
            end = min(start + int(real_bs), total_bs)
            mask[start:end] = True
        return mask

    def _committed_positions(self, draft_input, is_prefill):
        starts, lengths = compute_new_kv_slices(
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
        mwb.cache_loc = self._pack_kv_cache(locs, kv_lens)

    def _pack_kv_cache(self, rows: list[np.ndarray], kv_lens: list[int]) -> np.ndarray:
        """Pack per-request cache_loc rows into a bucket-stable flat layout.

        ``ForwardBatch.cache_loc`` is a JIT-visible child. Keep each request row
        in a min-16/power-of-two page bucket so decode only recompiles when the
        KV length crosses a bucket. FA metadata removes row padding from
        ``page_indices`` before the kernel sees it, keeping request boundaries
        aligned with ``cu_kv_lens``.
        """
        if not rows:
            return np.empty(0, dtype=np.int32)

        max_kv = max(int(x) for x in kv_lens)
        pages_per_seq = (max_kv + self.page_size - 1) // self.page_size
        bucketed_pages = max(16, 1 << max(0, (pages_per_seq - 1)).bit_length())
        row_width = bucketed_pages * self.page_size

        packed = np.empty((len(rows), row_width), dtype=np.int32)
        for i, row in enumerate(rows):
            n = min(len(row), row_width)
            row_arr = np.asarray(row[:n], dtype=np.int32)
            pad_value = row_arr[-1] if n > 0 else np.int32(0)
            packed[i, :] = pad_value
            packed[i, :n] = row_arr
        return packed.reshape(-1)

    # ------------------------------------------------------------------ #
    # Precompile hook (scheduler calls this at startup)
    # ------------------------------------------------------------------ #
    def run_spec_decode_precompile(self):
        import time

        self.target_worker.run_precompile()

        pps_buckets = self._compute_pps_buckets()
        bs_buckets = self._compute_verify_bs_buckets()

        start = time.perf_counter()
        logger.info(
            "[DFLASH] Precompiling TARGET_VERIFY: bs_buckets=%s, pps_buckets=%s",
            bs_buckets,
            pps_buckets,
        )

        for bs in bs_buckets:
            for pps in pps_buckets:
                t0 = time.perf_counter()
                self._precompile_verify(bs, pps, target=True)
                logger.info(
                    "[DFLASH] Target verify bs=%d pps=%d compiled in %.1f secs",
                    bs,
                    pps,
                    time.perf_counter() - t0,
                )
                t0 = time.perf_counter()
                self._precompile_verify(bs, pps, target=False)
                logger.info(
                    "[DFLASH] Draft verify bs=%d pps=%d compiled in %.1f secs",
                    bs,
                    pps,
                    time.perf_counter() - t0,
                )

        logger.info(
            "[DFLASH] TARGET_VERIFY precompile finished in %.0f secs",
            time.perf_counter() - start,
        )

    def _compute_verify_bs_buckets(self) -> list[int]:
        buckets = []
        for bs in self.target_worker.compilation_manager.bs_buckets:
            bs = int(bs)
            # Skip the non-power-of-two max_padded_batch_size bucket that the
            # generic compilation manager appends as a catch-all (e.g. 204).
            # DFlash verify precompile is expensive, and normal serving sweeps
            # use power-of-two batch buckets.
            if bs > 0 and (bs & (bs - 1)) == 0:
                buckets.append(bs)
        if not buckets:
            buckets.append(int(self.target_worker.compilation_manager.max_padded_batch_size))
        return buckets

    def _compute_pps_buckets(self) -> list[int]:
        max_pps = (self.target_worker.max_req_len + self.page_size - 1) // self.page_size
        buckets = []
        pps = 16
        while pps <= max_pps:
            buckets.append(pps)
            pps *= 2
        if not buckets:
            buckets.append(16)
        final_bucket = max(16, 1 << max(0, (max_pps - 1)).bit_length())
        if buckets[-1] < final_bucket:
            buckets.append(final_bucket)
        # _make_verify_dummy_batch uses these values as the row-padded token
        # width of cache_loc. For paged attention the row width must be page
        # aligned because get_eagle_forward_metadata repacks rows into page
        # indices. Small buckets such as 16/32 are valid for page_size=1, but
        # invalid for page_size=64/128.
        min_width = max(self.block_size, self.page_size)
        aligned_buckets = []
        for bucket in buckets:
            bucket = max(int(bucket), int(min_width))
            if self.page_size > 1:
                bucket = ((bucket + self.page_size - 1) // self.page_size) * self.page_size
            if not aligned_buckets or aligned_buckets[-1] != bucket:
                aligned_buckets.append(bucket)
        return aligned_buckets

    def _precompile_verify(self, bs: int, pps: int, target: bool):
        from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

        mwb = self._make_verify_dummy_batch(bs, pps, is_draft=not target)

        if target:
            forward_metadata = (
                self.target_worker.model_runner.attn_backend.get_eagle_forward_metadata(mwb)
            )
            self.target_worker.forward_batch_generation(
                mwb,
                skip_sample=True,
                forward_metadata=forward_metadata,
            )
        else:
            forward_batch = ForwardBatch.init_new(mwb, self.draft_model_runner)
            forward_batch.forward_mode = ForwardMode.TARGET_VERIFY
            num_tokens = bs * self.block_size
            hidden_size = self.draft_model_runner.model_config.hidden_size
            forward_batch.input_embedding = jnp.zeros(
                (num_tokens, hidden_size),
                dtype=jnp.bfloat16 if self.server_args.dtype == "bfloat16" else jnp.float32,
            )
            metadata = self.draft_model_runner.attn_backend.get_eagle_forward_metadata(mwb)
            self.draft_model_runner.attn_backend.forward_metadata = metadata
            self.draft_model_runner.forward(forward_batch, None)

    def _make_verify_dummy_batch(
        self, bs: int, pps: int, is_draft: bool = False
    ) -> ModelWorkerBatch:
        from sgl_jax.srt.managers.schedule_batch import ModelWorkerSamplingInfo

        block_size = self.block_size
        num_tokens = bs * block_size
        prefix_len = max(0, pps - block_size)

        input_ids = np.ones(num_tokens, dtype=np.int32)
        positions = np.tile(np.arange(prefix_len, prefix_len + block_size, dtype=np.int32), bs)
        out_cache_loc = np.arange(1, num_tokens + 1, dtype=np.int32)
        cache_loc = np.zeros(bs * pps, dtype=np.int32)
        seq_lens = np.full(bs, prefix_len, dtype=np.int32)

        sampling_info = ModelWorkerSamplingInfo.generate_for_precompile_all_greedy(
            bs,
            self.target_worker.model_runner.model_config.vocab_size,
        )
        sampling_info.vocab_mask = None

        if is_draft:
            prefix_cache_loc = jnp.zeros((bs, pps), dtype=jnp.int32)
            prefix_lens_arr = jnp.full((bs,), prefix_len, dtype=jnp.int32)
            verify_input = DFlashVerifyInput(
                draft_token=jnp.ones(num_tokens, dtype=jnp.int32),
                positions=jnp.asarray(positions),
                draft_token_num=block_size,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                prefix_cache_loc=prefix_cache_loc,
                prefix_lens=prefix_lens_arr,
                dense_draft=True,
            )
            capture_hidden = CaptureHiddenMode.NULL
        else:
            verify_input = DFlashVerifyInput(
                draft_token=jnp.ones(num_tokens, dtype=jnp.int32),
                positions=jnp.asarray(positions),
                draft_token_num=block_size,
                capture_hidden_mode=CaptureHiddenMode.FULL,
            )
            capture_hidden = CaptureHiddenMode.FULL

        return ModelWorkerBatch(
            bid=1,
            forward_mode=ForwardMode.TARGET_VERIFY,
            input_ids=input_ids,
            real_input_ids_len=num_tokens,
            real_bs=bs,
            req_pool_indices=np.arange(bs, dtype=np.int32),
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            return_logprob=False,
            return_output_logprob_only=False,
            sampling_info=sampling_info,
            extend_input_logprob_token_ids=None,
            positions=positions,
            cache_loc=cache_loc,
            extend_prefix_lens=None,
            extend_seq_lens=None,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            extend_logprob_start_lens=None,
            logits_indices=None,
            input_logprob_indices=None,
            capture_hidden_mode=capture_hidden,
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            lora_ids=["0"] * bs,
            dp_size=1,
            per_dp_bs_size=bs,
            real_bs_per_dp=[bs],
            logits_indices_selector=np.arange(bs, dtype=np.int32),
            spec_info_padded=verify_input,
        )
