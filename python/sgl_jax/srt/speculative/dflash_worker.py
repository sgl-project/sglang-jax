from __future__ import annotations

import contextlib
import copy
import logging
import os
from dataclasses import dataclass

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
from sgl_jax.srt.speculative.base_worker import BaseDraftWorker, BaseSpecWorker
from sgl_jax.srt.speculative.dflash_info import (
    DFlashDraftInput,
    DFlashVerifyInput,
    build_dflash_draft_block,
    dflash_greedy_verify,
)
from sgl_jax.srt.speculative.dflash_util import (
    parse_dflash_draft_config,
    resolve_mask_token_id,
)
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


def _mask_dflash_draft_extend_cache_loc(
    cache_loc: jax.Array,
    accept_lens: jax.Array,
    active_mask: jax.Array,
) -> jax.Array:
    """Mask unaccepted and padded draft-KV writes inside jit_draft_extend."""
    tokens_per_row = cache_loc.shape[0] // accept_lens.shape[0]
    cache_rows = cache_loc.reshape((-1, tokens_per_row))
    accept_rows = accept_lens[:, None]
    active_rows = active_mask[:, None]
    token_offsets = jnp.arange(tokens_per_row, dtype=jnp.int32)[None, :]
    mesh = getattr(jax.typeof(cache_loc).sharding, "mesh", None)
    if mesh is not None and not getattr(mesh, "empty", False):
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        row_sharding = NamedSharding(mesh, P("data", None))
        replicated_2d = NamedSharding(mesh, P(None, None))
        cache_rows = jax.sharding.reshard(cache_rows, row_sharding)
        accept_rows = jax.sharding.reshard(accept_rows, row_sharding)
        active_rows = jax.sharding.reshard(active_rows, row_sharding)
        token_offsets = jax.sharding.reshard(token_offsets, replicated_2d)
    write_mask = active_rows & (token_offsets < accept_rows)
    return jnp.where(
        write_mask,
        cache_rows,
        jnp.int32(-1),
    ).reshape(-1)


@dataclass(frozen=True)
class DFlashVerifyBucketTemplate:
    extend_seq_lens: np.ndarray
    cu_q_lens: jax.Array
    active_mask: jax.Array
    distribution: jax.Array


@dataclass(frozen=True)
class DraftForwardPlan:
    forward_batch: ForwardBatch
    forward_metadata: object
    seq_lens: np.ndarray
    target_prefix_lens: np.ndarray
    positions_host: np.ndarray
    bs: int


@dataclass(frozen=True)
class TargetVerifyPlan:
    model_worker_batch: ModelWorkerBatch
    forward_batch: ForwardBatch
    forward_metadata: object
    logits_metadata: LogitsMetadata
    seq_lens: np.ndarray
    target_prefix_lens: np.ndarray
    draft_extend_positions: jax.Array
    draft_extend_cache_loc: jax.Array
    active_mask: jax.Array


class DFlashWorker(BaseSpecWorker, BaseDraftWorker):
    """DFlash draft/verify runtime worker (greedy, DP/TP aware)."""

    def __init__(self, server_args, target_worker: ModelWorker):
        super().__init__(
            server_args,
            target_worker,
            self,
        )
        self.block_size = self.speculative_num_draft_tokens

        req_to_token_pool = self.req_to_token_pool
        target_allocator = self.token_to_kv_pool_allocator

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
        self._draft_model_runner = self._worker.model_runner
        draft_model = self.draft_model_runner.model

        # Alias the KV allocator so draft block allocation draws from the same
        # free list the target uses (no collision with committed slots).
        self.draft_model_runner.token_to_kv_pool_allocator = target_allocator

        target_model = target_worker.model_runner.model
        embed_weight, head_weight = target_model.get_embed_and_head()
        self._target_lm_head = head_weight  # [vocab, hidden], for greedy head sampling
        self._target_embed = embed_weight  # [vocab, hidden]
        self._target_vocab_size = int(target_worker.model_runner.model_config.vocab_size)

        pool_pages = (
            int(target_worker.max_total_num_tokens) + self.page_size - 1
        ) // self.page_size
        max_req_pages = (
            int(target_worker.compilation_manager.max_req_len) + self.page_size - 1
        ) // self.page_size
        self._page_indices_pool_capacity = 1 << max(0, pool_pages - 1).bit_length()
        self._page_indices_per_seq_capacity = max(
            16,
            1 << max(0, max_req_pages - 1).bit_length(),
        )
        self._verify_bucket_templates: dict[tuple, DFlashVerifyBucketTemplate] = {}

        dflash_config = parse_dflash_draft_config(
            server_args.speculative_draft_model_path,
            revision=server_args.speculative_draft_model_revision,
        )
        self._mask_token_id = resolve_mask_token_id(
            dflash_config,
            getattr(target_worker, "tokenizer", None),
            vocab_size=int(target_worker.model_runner.model_config.vocab_size),
        )

        self._draft_prefix_window = int(os.getenv("SGL_JAX_DFLASH_DRAFT_PREFIX_WINDOW", "0"))
        if self._draft_prefix_window > 0:
            for layer in draft_model.model.layers:
                layer.self_attn.attn.sliding_window_size = self._draft_prefix_window

        # Initialize JIT for the draft model runner (skipped during __init__
        # because is_draft_worker=True). The optional prefix window must be set
        # before nnx.split captures the model graph.
        self.draft_model_runner.initialize_jit()

        self.draft_layers = len(draft_model.model.layers)
        self._init_jit_target_verify()
        self._init_jit_kv_materialize()
        self._init_jit_draft_block()

        logger.info(
            "Initialized DFLASH worker: block_size=%d, mask_token_id=%d, "
            "draft_layers=%d, page_indices_pool_capacity=%d, "
            "page_indices_per_seq_capacity=%d",
            self.block_size,
            self._mask_token_id,
            self.draft_layers,
            self._page_indices_pool_capacity,
            self._page_indices_per_seq_capacity,
        )

    def _page_indices_capacity(self, bs: int) -> int:
        return min(
            self._page_indices_pool_capacity,
            max(int(bs), 1) * self._page_indices_per_seq_capacity,
        )

    def _build_dflash_page_indices(
        self,
        model_worker_batch: ModelWorkerBatch,
        prefix_lens: np.ndarray,
        bs: int,
    ) -> np.ndarray:
        """Build fixed-capacity, DP-segmented page indices from req_to_token."""
        dp_size = int(getattr(model_worker_batch, "dp_size", 1))
        per_dp_bs = int(getattr(model_worker_batch, "per_dp_bs_size", bs))
        if dp_size * per_dp_bs != bs:
            raise ValueError(
                "DFLASH page layout has inconsistent DP metadata: "
                f"dp_size={dp_size}, per_dp_bs={per_dp_bs}, bs={bs}."
            )

        capacity = self._page_indices_capacity(bs)
        if capacity % dp_size != 0:
            raise ValueError(
                "DFLASH page_indices capacity must be divisible by dp_size: "
                f"capacity={capacity}, dp_size={dp_size}."
            )
        per_rank_capacity = capacity // dp_size

        prefix_lens = np.asarray(prefix_lens, dtype=np.int32)
        if prefix_lens.shape != (bs,):
            raise ValueError(
                f"DFLASH prefix_lens must have shape ({bs},), got {prefix_lens.shape}."
            )
        req_pool_indices = np.asarray(model_worker_batch.req_pool_indices, dtype=np.int64)
        if req_pool_indices.shape != (bs,):
            raise ValueError(
                "DFLASH req_pool_indices must match the padded batch: "
                f"shape={req_pool_indices.shape}, bs={bs}."
            )

        selector = getattr(model_worker_batch, "logits_indices_selector", None)
        if selector is None:
            real_bs = int(getattr(model_worker_batch, "real_bs", bs))
            selector = np.arange(real_bs, dtype=np.int32)
        else:
            selector = np.asarray(selector, dtype=np.int32)
        if selector.size and (int(selector.min()) < 0 or int(selector.max()) >= bs):
            raise ValueError(f"DFLASH active-slot selector is out of bounds: {selector}.")
        active = np.zeros(bs, dtype=bool)
        active[selector] = True

        req_to_token = self.req_to_token_pool.req_to_token
        kv_lens = np.where(active, prefix_lens + self.block_size, 0).astype(np.int32)
        invalid_prefix = active & (prefix_lens < 0)
        if np.any(invalid_prefix):
            slot = int(np.flatnonzero(invalid_prefix)[0])
            raise ValueError(
                f"DFLASH active slot {slot} has invalid prefix_len={int(prefix_lens[slot])}."
            )
        overflow = kv_lens > req_to_token.shape[1]
        if np.any(overflow):
            slot = int(np.flatnonzero(overflow)[0])
            raise ValueError(
                "DFLASH KV length exceeds req_to_token capacity: "
                f"slot={slot}, kv_len={int(kv_lens[slot])}, "
                f"capacity={req_to_token.shape[1]}."
            )

        page_counts = (kv_lens + self.page_size - 1) // self.page_size
        max_pages = int(page_counts.max(initial=0))
        if max_pages:
            page_offsets = np.arange(max_pages, dtype=np.int64) * self.page_size
            valid_pages = page_offsets[None, :] < kv_lens[:, None]
            safe_req_indices = np.where(active, req_pool_indices, 0)
            page_locs = np.asarray(
                req_to_token[safe_req_indices[:, None], page_offsets[None, :]],
                dtype=np.int32,
            )
            incomplete = valid_pages & (page_locs < 0)
            if np.any(incomplete):
                slot = int(np.argwhere(incomplete)[0, 0])
                raise RuntimeError(
                    "DFLASH paged KV slots are incomplete: "
                    f"slot={slot}, req_pool_index={int(req_pool_indices[slot])}, "
                    f"kv_len={int(kv_lens[slot])}."
                )
            page_ids = page_locs // self.page_size
        else:
            valid_pages = np.zeros((bs, 0), dtype=bool)
            page_ids = np.zeros((bs, 0), dtype=np.int32)

        rank_chunks = []
        for dp_rank in range(dp_size):
            start = dp_rank * per_dp_bs
            end = start + per_dp_bs
            chunk = page_ids[start:end][valid_pages[start:end]].astype(np.int32, copy=False)
            if len(chunk) > per_rank_capacity:
                raise ValueError(
                    "DFLASH page_indices exceed the per-rank capacity: "
                    f"rank={dp_rank}, required={len(chunk)}, capacity={per_rank_capacity}."
                )
            rank_chunks.append(
                np.pad(chunk, (0, per_rank_capacity - len(chunk)), constant_values=0)
            )
        return np.concatenate(rank_chunks).astype(np.int32)

    def _get_verify_bucket_template(
        self,
        model_worker_batch: ModelWorkerBatch,
        bs: int,
    ) -> DFlashVerifyBucketTemplate:
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        dp_size = int(getattr(model_worker_batch, "dp_size", 1))
        per_dp_bs = int(getattr(model_worker_batch, "per_dp_bs_size", bs))
        selector = getattr(model_worker_batch, "logits_indices_selector", None)
        if selector is None:
            selector = np.arange(int(getattr(model_worker_batch, "real_bs", bs)), dtype=np.int32)
        else:
            selector = np.asarray(selector, dtype=np.int32)
        key = (dp_size, per_dp_bs, tuple(selector.tolist()), self.block_size)
        cached = self._verify_bucket_templates.get(key)
        if cached is not None:
            return cached

        active_host = np.zeros(bs, dtype=np.bool_)
        active_host[selector] = True
        extend_seq_lens = active_host.astype(np.int32) * self.block_size
        cu_q_lens = np.zeros((dp_size, per_dp_bs + 1), dtype=np.int32)
        cu_q_lens[:, 1:] = np.cumsum(
            extend_seq_lens.reshape(dp_size, per_dp_bs),
            axis=1,
            dtype=np.int32,
        )
        local_n = active_host.reshape(dp_size, per_dp_bs).sum(axis=1, dtype=np.int32)
        distribution = np.column_stack([np.zeros_like(local_n), local_n, local_n]).reshape(-1)
        data_sharding = NamedSharding(self.mesh, P("data"))
        cached = DFlashVerifyBucketTemplate(
            extend_seq_lens=extend_seq_lens,
            cu_q_lens=jax.device_put(cu_q_lens.reshape(-1), data_sharding),
            active_mask=jax.device_put(active_host, data_sharding),
            distribution=jax.device_put(distribution, data_sharding),
        )
        self._verify_bucket_templates[key] = cached
        return cached

    @property
    def draft_model_runner(self):
        return self._draft_model_runner

    def __getattr__(self, name):
        target_worker = self.__dict__.get("_target_worker")
        if target_worker is None:
            raise AttributeError(name)
        return getattr(target_worker, name)

    def _prepare_overlap_sampling_info(self, model_worker_batch: ModelWorkerBatch):
        return

    def draft_extend_for_prefill(
        self,
        model_worker_batch: ModelWorkerBatch,
        target_hidden,
        next_token_ids,
    ) -> None:
        sel = np.asarray(model_worker_batch.logits_indices_selector)
        extend_seq_lens = np.asarray(model_worker_batch.extend_seq_lens, dtype=np.int32)[sel]
        extend_prefix_lens = np.asarray(model_worker_batch.extend_prefix_lens, dtype=np.int32)[sel]

        draft_input = DFlashDraftInput(
            verified_id=None,
            target_hidden=target_hidden,
            ctx_lens=extend_seq_lens,
            draft_seq_lens=extend_prefix_lens,
            block_size=self.block_size,
        )

        # Materialization only depends on target hidden states. Dispatch it before
        # waiting for the sampled token so PJRT can run target prefill ->
        # jit_draft_extend without a host synchronization gap.
        self._append_target_hidden_to_draft_kv(model_worker_batch, draft_input)
        draft_input.verified_id = np.asarray(jax.device_get(next_token_ids))[sel].astype(np.int32)
        model_worker_batch.spec_info_padded = draft_input

    def draft_extend_for_decode(
        self,
        model_worker_batch: ModelWorkerBatch,
        batch_output,
    ) -> None:
        next_draft_input = batch_output.next_draft_input
        assert isinstance(next_draft_input, DFlashDraftInput)
        plan = getattr(next_draft_input, "_target_verify_plan", None)
        if not isinstance(plan, TargetVerifyPlan):
            raise RuntimeError("DFLASH draft extend is missing its target verify plan.")

        accept_lens, verified_id = jax.device_get(
            (next_draft_input.ctx_lens, next_draft_input.verified_id)
        )
        accept_lens = np.asarray(accept_lens, dtype=np.int32)
        verified_id = np.asarray(verified_id, dtype=np.int32)
        next_draft_input.verified_id = verified_id
        next_draft_input.ctx_lens = np.zeros_like(accept_lens)
        next_draft_input.draft_seq_lens = plan.target_prefix_lens + accept_lens
        next_draft_input.new_seq_lens = plan.seq_lens + accept_lens
        next_draft_input.target_hidden = None
        batch_output.accept_lens = accept_lens
        self._compact_dflash_state_to_real_slots(
            next_draft_input,
            model_worker_batch.logits_indices_selector,
        )
        model_worker_batch.spec_info_padded = next_draft_input
        del next_draft_input._target_verify_plan
        del model_worker_batch._dflash_target_verify_plan

    @staticmethod
    def _compact_dflash_state_to_real_slots(
        draft_input: DFlashDraftInput,
        selector: np.ndarray,
    ) -> None:
        """Remove per-DP padding before publishing cross-round host state.

        ``ScheduleBatch._split_spec_info_per_rank`` consumes a compact
        rank-major state, while target verify runs on DP-padded slots. Keep
        ``new_seq_lens`` padded because the scheduler uses it immediately to
        advance each rank's request lengths.
        """
        selector = np.asarray(selector, dtype=np.int32)
        for field in ("verified_id", "ctx_lens", "draft_seq_lens"):
            value = getattr(draft_input, field, None)
            if value is None:
                continue
            value = np.asarray(value, dtype=np.int32)
            if selector.size and int(selector.max()) >= value.shape[0]:
                raise ValueError(
                    "DFLASH state selector is out of bounds: "
                    f"field={field}, shape={value.shape}, selector={selector}."
                )
            setattr(draft_input, field, value[selector])

    def draft(self, model_worker_batch: ModelWorkerBatch) -> None:
        draft_input: DFlashDraftInput = model_worker_batch.spec_info_padded
        assert isinstance(
            draft_input, DFlashDraftInput
        ), "DFLASH decode requires DFlashDraftInput carried over from prefill."

        bs = int(model_worker_batch.seq_lens.shape[0])
        seq_lens = np.asarray(model_worker_batch.seq_lens, dtype=np.int32)
        target_prefix_lens = seq_lens - 1
        self._trim_dflash_draft_input_to_decode_batch(draft_input, bs)
        draft_prefix_lens = np.asarray(draft_input.draft_seq_lens, dtype=np.int32)

        draft_plan = self._build_draft_forward_plan(
            model_worker_batch,
            draft_input,
            target_prefix_lens,
            draft_prefix_lens,
            bs,
        )
        self.draft_model_runner.attn_backend.forward_metadata = draft_plan.forward_metadata
        draft_token = self._run_jit_draft_block(draft_plan.forward_batch)

        # JAX dispatch is asynchronous. Bind the target model to the draft
        # proposal and shared device layout while jit_draft is executing.
        target_plan = self._build_target_verify_plan(
            model_worker_batch,
            draft_plan,
            draft_token,
        )
        self.target_worker.model_runner.attn_backend.forward_metadata = target_plan.forward_metadata
        model_worker_batch._dflash_target_verify_plan = target_plan

    def verify(self, model_worker_batch: ModelWorkerBatch, cur_allocate_lens=None):
        from sgl_jax.srt.managers.scheduler import GenerationBatchResult

        plan = getattr(model_worker_batch, "_dflash_target_verify_plan", None)
        if not isinstance(plan, TargetVerifyPlan):
            raise RuntimeError("DFLASH target verify plan was not prepared by the draft phase.")
        (
            logits_output,
            cache_miss_count,
            accept_lens_out,
            next_token_ids_flat,
            new_verified_id,
            _,
            layers_topk_ids,
        ) = self._run_jit_target_verify(
            plan,
        )

        next_draft_input = DFlashDraftInput(
            verified_id=new_verified_id,
            target_hidden=logits_output.hidden_states,
            ctx_lens=accept_lens_out,
            draft_seq_lens=None,
            block_size=self.block_size,
        )
        next_draft_input.new_seq_lens = None
        next_draft_input._target_verify_plan = plan

        # Start the small round-state copies as soon as target futures exist.
        # They can overlap draft KV materialization and are consumed only after
        # this method returns to draft_extend_for_decode.
        jax.copy_to_host_async((accept_lens_out, new_verified_id))
        self._run_jit_draft_extend(
            logits_output.hidden_states,
            plan.draft_extend_positions,
            plan.draft_extend_cache_loc,
            accept_lens=accept_lens_out,
            active_mask=plan.active_mask,
        )
        self.target_worker.dump_topk_ids(layers_topk_ids, plan.model_worker_batch)
        self.target_worker.sync_queue.put((layers_topk_ids, plan.model_worker_batch))

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
            "from ScheduleBatch req state before entering the DFlash draft phase."
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
        token_sharding = NamedSharding(runner.mesh, P("data"))

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
                dflash_greedy_verify(
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
        plan: TargetVerifyPlan,
    ):
        import jax._src.test_util as jtu

        target_worker = self.target_worker
        runner = target_worker.model_runner
        model_worker_batch = plan.model_worker_batch

        if target_worker.worker.server_args.enable_lora and target_worker.need_prepare_lora_batch:
            target_worker.prepare_lora_batch(model_worker_batch)

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
                    plan.forward_batch,
                    runner.memory_pools,
                    plan.logits_metadata,
                    plan.forward_batch.input_ids,
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

        return (
            output,
            cache_miss_count,
            accept_lens_out,
            next_token_ids_flat,
            new_verified_id,
            accept_draft,
            layers_topk_ids,
        )

    def _build_draft_forward_plan(
        self,
        model_worker_batch: ModelWorkerBatch,
        draft_input: DFlashDraftInput,
        target_prefix_lens: np.ndarray,
        draft_prefix_lens: np.ndarray,
        bs: int,
    ) -> DraftForwardPlan:
        block_ids, positions = build_dflash_draft_block(
            verified_id=draft_input.verified_id,
            mask_token_id=self._mask_token_id,
            target_prefix_lens=target_prefix_lens,
            block_size=self.block_size,
        )
        block_ids_flat = block_ids.reshape(-1)
        positions_flat = positions.reshape(-1)

        draft_mwb = self._make_draft_block_mwb(
            model_worker_batch,
            block_ids_flat,
            positions_flat,
            draft_prefix_lens,
        )
        forward_batch = ForwardBatch.init_new(draft_mwb, self.draft_model_runner)
        forward_batch.forward_mode = ForwardMode.TARGET_VERIFY
        # Reuse ForwardBatch's device token buffer.
        draft_mwb.spec_info_padded.draft_token = forward_batch.input_ids
        forward_batch.spec_info = draft_mwb.spec_info_padded
        active_mask = self._active_decode_slot_mask(model_worker_batch, bs)
        mismatched_prefix = active_mask & (draft_prefix_lens != target_prefix_lens)
        if np.any(mismatched_prefix):
            slots = np.flatnonzero(mismatched_prefix)
            raise RuntimeError(
                "DFLASH target/draft prefix layouts diverged for active slots: "
                f"slots={slots.tolist()}, "
                f"target={target_prefix_lens[slots].tolist()}, "
                f"draft={draft_prefix_lens[slots].tolist()}."
            )
        page_indices = self._build_dflash_page_indices(
            draft_mwb,
            draft_prefix_lens,
            bs,
        )
        template = self._get_verify_bucket_template(draft_mwb, bs)
        metadata = self.draft_model_runner.attn_backend.get_eagle_forward_metadata(
            draft_mwb,
            page_indices=page_indices,
            page_indices_capacity=self._page_indices_capacity(bs),
            extend_seq_lens=template.extend_seq_lens,
            cu_q_lens=template.cu_q_lens,
            distribution=template.distribution,
        )
        return DraftForwardPlan(
            forward_batch=forward_batch,
            forward_metadata=metadata,
            seq_lens=np.asarray(model_worker_batch.seq_lens, dtype=np.int32),
            target_prefix_lens=np.asarray(target_prefix_lens, dtype=np.int32),
            positions_host=positions_flat,
            bs=bs,
        )

    def _build_target_verify_plan(
        self,
        model_worker_batch: ModelWorkerBatch,
        draft_plan: DraftForwardPlan,
        draft_token: jax.Array,
    ) -> TargetVerifyPlan:
        bs = draft_plan.bs
        target_mwb = copy.copy(model_worker_batch)
        target_mwb.forward_mode = ForwardMode.TARGET_VERIFY
        target_mwb.input_ids = np.empty((0,), dtype=np.int32)
        target_mwb.positions = draft_plan.positions_host
        target_mwb.seq_lens = draft_plan.target_prefix_lens
        target_mwb.cache_loc = np.zeros(
            int(getattr(model_worker_batch, "dp_size", 1)), dtype=np.int32
        )
        target_mwb.capture_hidden_mode = CaptureHiddenMode.FULL
        target_mwb.forward_batch = None

        verify_input = DFlashVerifyInput(
            draft_token=draft_token,
            draft_token_num=self.block_size,
        )
        target_mwb.spec_info_padded = verify_input

        template = self._get_verify_bucket_template(target_mwb, bs)
        target_metadata = draft_plan.forward_metadata
        draft_extend_cache_loc = draft_plan.forward_batch.out_cache_loc

        # Reuse proposal positions, request indices, and other device buffers
        # from the draft plan. In particular, do not upload MASK ids for target
        # verify and overwrite them with draft_token afterwards.
        target_forward_batch = copy.copy(draft_plan.forward_batch)
        target_forward_batch.forward_mode = ForwardMode.TARGET_VERIFY
        target_forward_batch.input_ids = draft_token
        target_forward_batch.seq_lens = draft_plan.forward_batch.seq_lens
        target_forward_batch.out_cache_loc = draft_extend_cache_loc
        target_forward_batch.positions = draft_plan.forward_batch.positions
        target_forward_batch.cache_loc = None
        target_forward_batch.attn_backend = self.target_worker.model_runner.attn_backend
        target_forward_batch.spec_info = verify_input
        target_forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        target_forward_batch.input_embedding = None

        return TargetVerifyPlan(
            model_worker_batch=target_mwb,
            forward_batch=target_forward_batch,
            forward_metadata=target_metadata,
            logits_metadata=LogitsMetadata.from_model_worker_batch(target_mwb, self.mesh),
            seq_lens=draft_plan.seq_lens,
            target_prefix_lens=draft_plan.target_prefix_lens,
            draft_extend_positions=draft_plan.forward_batch.positions,
            draft_extend_cache_loc=draft_extend_cache_loc,
            active_mask=template.active_mask,
        )

    def _make_draft_block_mwb(
        self,
        base_mwb: ModelWorkerBatch,
        block_ids_flat: np.ndarray,
        positions_flat: np.ndarray,
        prefix_lens: np.ndarray,
    ) -> ModelWorkerBatch:
        mwb = copy.copy(base_mwb)
        mwb.forward_mode = ForwardMode.TARGET_VERIFY
        mwb.input_ids = np.asarray(block_ids_flat, dtype=np.int32)
        mwb.positions = np.asarray(positions_flat, dtype=np.int32)
        mwb.seq_lens = np.asarray(prefix_lens, dtype=np.int32)
        mwb.capture_hidden_mode = CaptureHiddenMode.NULL
        mwb.spec_algorithm = SpeculativeAlgorithm.DFLASH
        mwb.spec_info_padded = DFlashVerifyInput(
            draft_token=block_ids_flat,
            draft_token_num=self.block_size,
        )
        # DFlashDraftInput.prepare_for_decode already reserved and packed one
        # block per active request into the first half of each DP rank section.
        # Reuse that scheduler output rather than gathering the same slots from
        # req_to_token_pool again on every decode round.
        mwb.out_cache_loc = self._verify_write_cache_loc(base_mwb)
        mwb.cache_loc = None
        return mwb

    def _init_jit_draft_block(self):
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
        vocab_size = self._target_vocab_size
        embedding_sharding = NamedSharding(runner.mesh, P("data", "tensor"))
        logits_sharding = NamedSharding(runner.mesh, P("data", "tensor"))
        token_sharding = NamedSharding(runner.mesh, P("data"))

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
                output, pool_updates, _, _ = model(forward_batch, memory_pools, None)

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

    def _run_jit_draft_block(self, forward_batch):
        runner = self.draft_model_runner
        forward_batch.cache_loc = None

        def _call_and_replace():
            pool_updates, draft_token = self._jit_draft_block(
                runner.model_state_leaves,
                forward_batch,
                runner.memory_pools,
                self._target_embed,
                self._target_lm_head,
            )
            if runner.tp_size == 1 and isinstance(pool_updates, list):
                target_sharding = runner.token_to_kv_pool.kv_sharding
                pool_updates = [jax.device_put(kv, target_sharding) for kv in pool_updates]
            runner.memory_pools.replace_all(pool_updates)
            return draft_token

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
            return _call_and_replace()

    def _init_jit_kv_materialize(self):
        """Fuse draft KV projection, merge, and cache writes into one JIT."""
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
            static_argnames=["model_state_def"],
            donate_argnames=["kv_buffers"],
        )
        def draft_extend(
            model_def,
            model_state_def,
            model_state_leaves,
            target_hidden,
            positions,
            cache_loc,
            accept_lens,
            active_mask,
            kv_buffers,
        ):
            positions = jax.sharding.reshard(positions.astype(jnp.int32), vector_sharding)
            cache_loc = jax.sharding.reshard(cache_loc.astype(jnp.int32), vector_sharding)
            accept_lens = jax.sharding.reshard(accept_lens.astype(jnp.int32), vector_sharding)
            active_mask = jax.sharding.reshard(active_mask.astype(jnp.bool_), vector_sharding)
            target_hidden = jax.sharding.reshard(target_hidden, hidden_sharding)

            cache_loc = _mask_dflash_draft_extend_cache_loc(
                cache_loc,
                accept_lens,
                active_mask,
            )

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
    ) -> None:
        target_hidden = draft_input.target_hidden
        if target_hidden is None or int(np.asarray(draft_input.ctx_lens).sum()) == 0:
            return

        positions, cache_loc = self._prefill_draft_extend_metadata(
            model_worker_batch,
            target_hidden,
        )

        self._run_jit_draft_extend(target_hidden, positions, cache_loc)

        draft_input.draft_seq_lens = np.asarray(
            draft_input.draft_seq_lens, dtype=np.int32
        ) + np.asarray(draft_input.ctx_lens, dtype=np.int32)
        draft_input.ctx_lens = np.zeros_like(np.asarray(draft_input.ctx_lens, dtype=np.int32))
        draft_input.target_hidden = None

    def _run_jit_draft_extend(
        self,
        target_hidden,
        positions,
        cache_loc,
        *,
        accept_lens=None,
        active_mask=None,
    ):
        pool = self.draft_model_runner.token_to_kv_pool
        if accept_lens is None:
            # Prefill metadata is already on the host. Build its fixed masks with
            # NumPy so they become inputs to jit_draft_extend instead of separate
            # broadcast/compare JAX launches.
            cache_loc = np.asarray(cache_loc, dtype=np.int32)
            accept_lens = np.ones((target_hidden.shape[0],), dtype=np.int32)
            if active_mask is None:
                active_mask = cache_loc >= 0
        cache_loc = jnp.asarray(cache_loc)
        accept_lens = jnp.asarray(accept_lens)
        active_mask = jnp.asarray(active_mask)
        if cache_loc.shape[0] % accept_lens.shape[0] != 0:
            raise ValueError(
                "DFLASH draft extend cache rows do not match accept_lens: "
                f"cache_loc={cache_loc.shape}, accept_lens={accept_lens.shape}."
            )
        new_buffers = self._jit_materialize_write(
            self.draft_model_runner.model_state_leaves,
            jnp.asarray(target_hidden),
            jnp.asarray(positions),
            cache_loc,
            accept_lens,
            active_mask,
            list(pool.kv_buffer[: self.draft_layers]),
        )
        for i, buf in enumerate(new_buffers):
            pool.kv_buffer[i] = buf

    @staticmethod
    def _prefill_draft_extend_metadata(model_worker_batch, target_hidden):
        """Reuse target-prefill's DP-segmented token metadata.

        Target hidden rows use ``[rank0 tokens + pad | rank1 tokens + pad | ...]``.
        Reusing the matching positions/out-cache buffers preserves those rank
        boundaries for the ``P("data")`` KV materialization JIT.
        """
        positions = np.asarray(model_worker_batch.positions, dtype=np.int32).reshape(-1)
        cache_loc = np.asarray(model_worker_batch.out_cache_loc, dtype=np.int32).reshape(-1)
        if positions.shape != cache_loc.shape:
            raise ValueError(
                "DFLASH prefill positions/cache_loc shape mismatch: "
                f"{positions.shape} vs {cache_loc.shape}."
            )

        bucket_tokens = int(target_hidden.shape[0])
        metadata_tokens = int(positions.shape[0])
        if metadata_tokens != bucket_tokens:
            raise ValueError(
                "DFLASH prefill metadata must match the target hidden bucket: "
                f"metadata_tokens={metadata_tokens}, bucket_tokens={bucket_tokens}."
            )
        return positions, cache_loc

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

    def run_spec_decode_precompile(self):
        self._precompile_dflash_prefill()
        manager = self.target_worker.compilation_manager
        dp_size = int(manager.dp_size)
        bs_buckets = [
            int(bs)
            for bs in manager.bs_buckets
            if int(bs) >= dp_size and int(bs) % dp_size == 0 and int(bs) & (int(bs) - 1) == 0
        ]
        if not bs_buckets:
            max_bs = int(manager.max_padded_batch_size)
            if max_bs % dp_size != 0:
                raise ValueError(
                    "DFLASH precompile batch size must be divisible by dp_size: "
                    f"max_padded_batch_size={max_bs}, dp_size={dp_size}."
                )
            bs_buckets = [max_bs]

        logger.info(
            "[DFLASH] Precompiling one fixed-page variant per bs: bs=%s, page_indices_capacity=%s",
            bs_buckets,
            [self._page_indices_capacity(bs) for bs in bs_buckets],
        )
        for bs in bs_buckets:
            self._precompile_dflash_variant(bs)

    @staticmethod
    def _prefill_precompile_variants(manager) -> list[tuple[int, int]]:
        bs = int(manager.max_padded_batch_size)
        return [(bs, int(tokens)) for tokens in manager.token_buckets if int(tokens) >= bs]

    def _precompile_dflash_prefill(self) -> None:
        import time

        from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata

        manager = self.target_worker.compilation_manager
        variants = self._prefill_precompile_variants(manager)
        logger.info("[DFLASH] Precompiling prefill variants: %s", variants)
        start = time.perf_counter()

        for bs, num_tokens in variants:
            t0 = time.perf_counter()
            batch = manager._make_dummy_batch(
                bs,
                num_tokens,
                ForwardMode.EXTEND,
                manager.cache_loc_buckets[-1],
                speculative_algorithm=SpeculativeAlgorithm.DFLASH,
                dp_size=manager.dp_size,
                per_dp_bs_size=bs // manager.dp_size,
            )
            batch.capture_hidden_mode = CaptureHiddenMode.FULL
            sampling_metadata = SamplingMetadata.from_model_worker_batch(
                batch,
                0,
                self.mesh,
                self._target_vocab_size,
            )
            batch.forward_batch = ForwardBatch.init_new(
                batch,
                self.target_worker.model_runner,
            )
            logits_output, *_ = self.forward_target_extend(
                batch,
                sampling_metadata,
                skip_sample=True,
            )

            # Consume the real target output so hidden-state sharding matches
            # the serving dependency exactly.
            self._run_jit_draft_extend(
                logits_output.hidden_states,
                batch.positions,
                batch.out_cache_loc,
            )
            logger.info(
                "[DFLASH] Prefill bs=%d tokens=%d compiled in %.1f secs",
                bs,
                num_tokens,
                time.perf_counter() - t0,
            )

        logger.info(
            "[DFLASH] Prefill precompile finished in %.0f secs",
            time.perf_counter() - start,
        )

    def _precompile_dflash_variant(self, bs: int) -> None:
        row_width = max(self.block_size, 16 * self.page_size)
        page_indices = np.zeros(self._page_indices_capacity(bs), dtype=np.int32)
        draft_batch = self._make_verify_dummy_batch(bs, row_width, is_draft=True)
        draft_batch.out_cache_loc = self._verify_write_cache_loc(draft_batch)
        forward_batch = ForwardBatch.init_new(draft_batch, self.draft_model_runner)
        forward_batch.forward_mode = ForwardMode.TARGET_VERIFY
        template = self._get_verify_bucket_template(draft_batch, bs)
        draft_metadata = self.draft_model_runner.attn_backend.get_eagle_forward_metadata(
            draft_batch,
            page_indices=page_indices,
            page_indices_capacity=self._page_indices_capacity(bs),
            extend_seq_lens=template.extend_seq_lens,
            cu_q_lens=template.cu_q_lens,
            distribution=template.distribution,
        )
        self.draft_model_runner.attn_backend.forward_metadata = draft_metadata
        draft_token = self._run_jit_draft_block(forward_batch)

        # Match the serving dependency and sharding exactly: target verify
        # consumes the P("data") proposal produced by jit_draft.
        target_batch = self._make_verify_dummy_batch(bs, row_width)
        verify_input = DFlashVerifyInput(
            draft_token=draft_token,
            draft_token_num=self.block_size,
        )
        target_batch.spec_info_padded = verify_input
        target_metadata = draft_metadata
        draft_extend_cache_loc = forward_batch.out_cache_loc
        target_forward_batch = copy.copy(forward_batch)
        target_forward_batch.input_ids = draft_token
        target_forward_batch.out_cache_loc = draft_extend_cache_loc
        target_forward_batch.seq_lens = forward_batch.seq_lens
        target_forward_batch.attn_backend = self.target_worker.model_runner.attn_backend
        target_forward_batch.spec_info = verify_input
        target_forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        target_forward_batch.input_embedding = None
        target_plan = TargetVerifyPlan(
            model_worker_batch=target_batch,
            forward_batch=target_forward_batch,
            forward_metadata=target_metadata,
            logits_metadata=LogitsMetadata.from_model_worker_batch(target_batch, self.mesh),
            seq_lens=np.asarray(target_batch.seq_lens, dtype=np.int32) + 1,
            target_prefix_lens=np.asarray(target_batch.seq_lens, dtype=np.int32),
            draft_extend_positions=forward_batch.positions,
            draft_extend_cache_loc=draft_extend_cache_loc,
            active_mask=template.active_mask,
        )
        self.target_worker.model_runner.attn_backend.forward_metadata = target_metadata
        logits_output, _, accept_lens, *_ = self._run_jit_target_verify(target_plan)
        self._run_jit_draft_extend(
            logits_output.hidden_states,
            target_plan.draft_extend_positions,
            target_plan.draft_extend_cache_loc,
            accept_lens=accept_lens,
            active_mask=target_plan.active_mask,
        )

    def _verify_write_cache_loc(self, batch: ModelWorkerBatch) -> np.ndarray:
        dp_size = int(batch.dp_size)
        per_dp_tokens = int(batch.per_dp_bs_size) * self.block_size
        out_cache_loc = np.asarray(batch.out_cache_loc, dtype=np.int32)
        if out_cache_loc.shape[0] % dp_size != 0:
            raise ValueError(
                "DFLASH verify out_cache_loc is not divisible by dp_size: "
                f"shape={out_cache_loc.shape}, dp_size={dp_size}."
            )
        per_dp_ocl = out_cache_loc.shape[0] // dp_size
        if per_dp_ocl < per_dp_tokens:
            raise ValueError(
                "DFLASH verify out_cache_loc rank section is too short: "
                f"per_dp_ocl={per_dp_ocl}, verify_tokens={per_dp_tokens}."
            )
        return out_cache_loc.reshape(dp_size, per_dp_ocl)[:, :per_dp_tokens].reshape(-1)

    def _make_verify_dummy_batch(
        self, bs: int, row_width: int, is_draft: bool = False
    ) -> ModelWorkerBatch:
        block_size = self.block_size
        num_tokens = bs * block_size
        dp_size = int(self.target_worker.compilation_manager.dp_size)
        if bs % dp_size != 0:
            raise ValueError(
                "DFLASH verify dummy batch must be divisible by dp_size: "
                f"bs={bs}, dp_size={dp_size}."
            )
        per_dp_bs = bs // dp_size
        per_dp_tokens = per_dp_bs * block_size
        kv_len = min(row_width, self.target_worker.compilation_manager.max_req_len)
        prefix_len = max(0, kv_len - block_size)
        positions = np.tile(np.arange(prefix_len, prefix_len + block_size, dtype=np.int32), bs)
        batch = self.target_worker.compilation_manager._make_dummy_batch(
            bs,
            num_tokens,
            ForwardMode.TARGET_VERIFY,
            bs * row_width,
            speculative_algorithm=SpeculativeAlgorithm.DFLASH,
            dp_size=dp_size,
            per_dp_bs_size=per_dp_bs,
        )
        capture_hidden = CaptureHiddenMode.NULL if is_draft else CaptureHiddenMode.FULL
        batch.input_ids = np.ones(num_tokens, dtype=np.int32)
        batch.real_input_ids_len = num_tokens
        batch.seq_lens = np.full(bs, prefix_len, dtype=np.int32)
        batch.out_cache_loc = np.concatenate(
            [
                np.concatenate(
                    [
                        np.arange(
                            dp_rank * per_dp_tokens + 1,
                            (dp_rank + 1) * per_dp_tokens + 1,
                            dtype=np.int32,
                        ),
                        np.full(per_dp_tokens, -1, dtype=np.int32),
                    ]
                )
                for dp_rank in range(dp_size)
            ]
        )
        batch.positions = positions
        batch.cache_loc = np.zeros(dp_size, dtype=np.int32)
        batch.capture_hidden_mode = capture_hidden
        batch.spec_info_padded = DFlashVerifyInput(
            draft_token=jnp.ones(num_tokens, dtype=jnp.int32),
            draft_token_num=block_size,
        )
        return batch
