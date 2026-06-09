from __future__ import annotations

import dataclasses
import os
import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import jax
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
    from sgl_jax.srt.managers.tp_worker import ModelWorker


def replicate_to_mesh(
    mesh: jax.sharding.Mesh, *arrs: jax.Array
) -> tuple[jax.Array, ...] | jax.Array:
    """Replicate arrays across a mesh under explicit sharding.

    JIT outputs are typically vocab/data-sharded; spec-decode host orchestration
    (top_k, gather, build_tree) needs replicated arrays.
    """
    out = jax.device_put(arrs, NamedSharding(mesh, P()))
    return out[0] if len(out) == 1 else out


class BaseDraftWorker(ABC):
    """Draft model worker interface for speculative decoding.

    Concrete implementations hold the draft model runner and own all
    draft-specific logic (multi-step decode, tree building, extend).
    Standard EAGLE uses ``EagleDraftWorker``; MTP uses
    ``MultiLayerDraftWorker``.
    """

    @property
    @abstractmethod
    def draft_model_runner(self):
        """Primary model runner (multi-runner workers return a designated one)."""

    @abstractmethod
    def draft(self, model_worker_batch):
        pass

    @abstractmethod
    def draft_extend_for_prefill(self, model_worker_batch, hidden_states, next_token_ids):
        pass

    @abstractmethod
    def draft_extend_for_decode(self, model_worker_batch, batch_output):
        pass


class BaseSpecWorker:
    """Speculative decode orchestrator.

    Owns a ``target_worker`` (the full model) and a ``draft_worker``
    (the draft/MTP model). The orchestration loop (prefill → draft →
    verify → draft_extend) and ``verify()`` itself are spec-algorithm-
    agnostic, so they live here; subclasses only differ in which
    ``BaseDraftWorker`` they construct.
    """

    def __init__(self, server_args, target_worker: ModelWorker, draft_worker: BaseDraftWorker):
        self.server_args = server_args
        self._target_worker = target_worker
        self._draft_worker = draft_worker

        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.page_size = server_args.page_size
        self.mesh = target_worker.mesh

        from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self._can_use_fused_spec_decode = (
            self.speculative_algorithm.is_nextn()
            and self.topk == 1
            and self.speculative_num_steps > 1
            and self.speculative_num_draft_tokens == self.speculative_num_steps + 1
        )

        self.req_to_token_pool, self.token_to_kv_pool_allocator = target_worker.get_memory_pool()

        (
            self.precompile_token_paddings,
            self.precompile_bs_paddings,
            self.precompile_cache_loc_paddings,
        ) = target_worker.get_precompile_paddings()

    @property
    def target_worker(self) -> ModelWorker:
        return self._target_worker

    @property
    def draft_worker(self) -> BaseDraftWorker:
        return self._draft_worker

    def _can_use_fused_spec_prefill(self, model_worker_batch: ModelWorkerBatch) -> bool:
        if os.getenv("SGL_JAX_DISABLE_FUSED_SPEC_PREFILL") == "1":
            return False
        sampling_info = model_worker_batch.sampling_info
        penalizer = getattr(sampling_info, "penalizer_orchestrator", None)
        has_penalty = getattr(sampling_info, "linear_penalty", None) is not None or bool(
            getattr(penalizer, "is_required", False)
        )
        return (
            self._can_use_fused_spec_decode
            and sampling_info.is_all_greedy
            and not has_penalty
            and getattr(sampling_info, "vocab_mask", None) is None
            and not getattr(model_worker_batch, "return_logprob", False)
            and not getattr(model_worker_batch, "return_output_logprob_only", False)
        )

    # -- Main entry point --

    def _prepare_overlap_sampling_info(self, model_worker_batch: ModelWorkerBatch):
        sampling_info = model_worker_batch.sampling_info
        sampling_info.update_penalties()
        model_worker_batch.sampling_info = self.cur_sampling_info = dataclasses.replace(
            sampling_info,
            sampling_info_done=threading.Event(),
            penalizer_orchestrator=None,
        )

    def forward_batch_speculative_decode_overlap(self, model_worker_batch: ModelWorkerBatch):
        prepared = getattr(model_worker_batch, "spec_decode_verify_prepare", None)
        if prepared is None and not model_worker_batch.forward_mode.is_decode():
            raise NotImplementedError(
                "Spec decode-overlap entry only supports decode batches; "
                "prefill overlap uses forward_batch_speculative_generation()."
            )
        if not (self._can_use_fused_spec_decode and model_worker_batch.sampling_info.is_all_greedy):
            raise NotImplementedError("Spec overlap entry only supports fused greedy decode.")

        if prepared is None:
            self._prepare_overlap_sampling_info(model_worker_batch)
            sel = model_worker_batch.logits_indices_selector
            cur_allocate_lens = np.asarray(model_worker_batch.spec_info_padded.allocate_lens)[sel]
        else:
            cur_allocate_lens = prepared.cur_allocate_lens

        from sgl_jax.srt.speculative.draft_extend_fused import spec_decode_overlap

        result = spec_decode_overlap(self, model_worker_batch, cur_allocate_lens)
        launch_done = getattr(model_worker_batch, "launch_done", None)
        if launch_done is not None:
            launch_done.set()
        return result

    def prepare_speculative_decode_overlap(self, model_worker_batch: ModelWorkerBatch):
        """Prepare decode-overlap verify metadata before entering the forward path."""
        if not (self._can_use_fused_spec_decode and model_worker_batch.sampling_info.is_all_greedy):
            return
        self._prepare_overlap_sampling_info(model_worker_batch)
        sel = model_worker_batch.logits_indices_selector
        cur_allocate_lens = np.asarray(model_worker_batch.spec_info_padded.allocate_lens)[sel]

        from sgl_jax.srt.speculative.draft_extend_fused import (
            prepare_spec_decode_verify_phase,
        )

        model_worker_batch.spec_decode_verify_prepare = prepare_spec_decode_verify_phase(
            self, model_worker_batch, cur_allocate_lens
        )

    def forward_batch_speculative_generation(
        self, model_worker_batch: ModelWorkerBatch, launch_done=None
    ):
        from sgl_jax.srt.managers.scheduler import GenerationBatchResult
        from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata

        if launch_done is None:
            self._prepare_overlap_sampling_info(model_worker_batch)

        if model_worker_batch.forward_mode.is_extend():
            if self._can_use_fused_spec_prefill(model_worker_batch):
                from sgl_jax.srt.speculative.draft_extend_fused import (
                    prepare_spec_prefill_forward_batch,
                    spec_prefill,
                )

                if getattr(model_worker_batch, "forward_batch", None) is None:
                    prepare_spec_prefill_forward_batch(self, model_worker_batch)
                return spec_prefill(self, model_worker_batch, launch_done=launch_done)

            if model_worker_batch.sampling_info.temperatures.ndim == 1:
                model_worker_batch.sampling_info.temperatures = (
                    model_worker_batch.sampling_info.temperatures[:, None]
                )
            sampling_metadata = SamplingMetadata.from_model_worker_batch(
                model_worker_batch,
                len(model_worker_batch.seq_lens) - model_worker_batch.real_bs,
                self.mesh,
                vocab_size=self.target_worker.model_config.vocab_size,
            )
            logits_output, next_token_ids, cache_miss_count, bid, _seq_lens = (
                self.forward_target_extend(model_worker_batch, sampling_metadata)
            )
            if model_worker_batch.dp_size > 1:
                from jax.experimental.multihost_utils import process_allgather

                next_token_ids = process_allgather(next_token_ids, tiled=True)
            self.draft_worker.draft_extend_for_prefill(
                model_worker_batch, logits_output.hidden_states, next_token_ids
            )
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                next_draft_input=model_worker_batch.spec_info_padded,
                bid=bid,
                cache_miss_count=cache_miss_count,
                extend_input_len_per_req=None,
                extend_logprob_start_len_per_req=None,
            )

        # spec_info.allocate_lens is DP-padded (total_bs,) at dp>1; gather back
        # to global-flat (real_bs,) so reqs_info[0].spec_info stays flat-ordered.
        sel = model_worker_batch.logits_indices_selector
        cur_allocate_lens = np.asarray(model_worker_batch.spec_info_padded.allocate_lens)[sel]
        if self._can_use_fused_spec_decode and model_worker_batch.sampling_info.is_all_greedy:
            # Current fused route covers greedy NEXTN decode; more speculative
            # decode paths can be folded into this entry point over time.
            from sgl_jax.srt.speculative.draft_extend_fused import spec_decode

            batch_output = spec_decode(self, model_worker_batch, cur_allocate_lens)
            launch_done = getattr(model_worker_batch, "launch_done", None)
            if launch_done is not None:
                launch_done.set()
            return batch_output
        self.draft_worker.draft(model_worker_batch)
        batch_output = self.verify(model_worker_batch, cur_allocate_lens)
        self.draft_worker.draft_extend_for_decode(model_worker_batch, batch_output)
        launch_done = getattr(model_worker_batch, "launch_done", None)
        if launch_done is not None:
            launch_done.set()
        return batch_output

    def forward_target_extend(self, model_worker_batch: ModelWorkerBatch, sampling_metadata):
        from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode

        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        logits_output, next_token_ids, cache_miss_count = (
            self.target_worker.forward_batch_generation(
                model_worker_batch, sampling_metadata=sampling_metadata
            )
        )
        return (
            logits_output,
            next_token_ids,
            cache_miss_count,
            model_worker_batch.bid,
            model_worker_batch.seq_lens,
        )

    def verify(self, model_worker_batch: ModelWorkerBatch, cur_allocate_lens: jax.Array):
        from sgl_jax.srt.managers.scheduler import GenerationBatchResult
        from sgl_jax.srt.speculative.eagle_util import EagleDraftInput, EagleVerifyInput

        spec_info: EagleVerifyInput = model_worker_batch.spec_info_padded
        spec_info.allocate_lens = cur_allocate_lens
        spec_info.prepare_for_verify(model_worker_batch, self.page_size, self.target_worker)
        forward_metadata = self.target_worker.model_runner.attn_backend.get_eagle_forward_metadata(
            model_worker_batch
        )

        logits_output, _, cache_miss_count = self.target_worker.forward_batch_generation(
            model_worker_batch, skip_sample=True, forward_metadata=forward_metadata
        )
        logits_output.next_token_logits, logits_output.hidden_states = replicate_to_mesh(
            self.mesh, logits_output.next_token_logits, logits_output.hidden_states
        )
        spec_info.hidden_states = logits_output.hidden_states

        (
            predict,
            verified_id,
            accept_length,
            accept_index,
        ) = spec_info.sample(
            model_worker_batch,
            logits_output,
            self.draft_worker.draft_model_runner.rngs,
            self.mesh,
        )
        # accept_index uses -1 for rejected slots; gathering with -1 picks the
        # global last element, so dext later writes rejected tokens' draft-KV at
        # a foreign position inside each req's page (corrupts prefix KV for all
        # but the last req at bs>1). Redirect -1 to each req's own last slot.
        # accept_index has length bs*(spec_steps+1); the gathered tensors have
        # length bs*draft_token_num — equal at topk=1, distinct at topk>1.
        draft_n = self.speculative_num_draft_tokens
        accept_width = self.speculative_num_steps + 1
        req_ids = np.arange(len(accept_index)) // accept_width
        per_req_last = req_ids * draft_n + draft_n - 1
        safe_index = np.where(accept_index >= 0, accept_index, per_req_last)
        logits_output.next_token_logits = logits_output.next_token_logits[safe_index, :]
        logits_output.hidden_states = logits_output.hidden_states[safe_index, :]
        model_worker_batch.positions = model_worker_batch.positions[safe_index]
        new_seq_lens = model_worker_batch.seq_lens + accept_length + 1
        next_draft_input = EagleDraftInput(
            verified_id=verified_id,
            new_seq_lens=new_seq_lens,
            allocate_lens=cur_allocate_lens,
            hidden_states=logits_output.hidden_states,
        )

        model_worker_batch.spec_info_padded = next_draft_input
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            next_draft_input=next_draft_input,
            accept_lens=accept_length,
            bid=model_worker_batch.bid,
            cache_miss_count=cache_miss_count,
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
        )
