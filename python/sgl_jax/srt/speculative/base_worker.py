from __future__ import annotations

import dataclasses
import os
import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.speculative.overlap_utils import uses_host_eagle_state

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
    from sgl_jax.srt.managers.tp_worker import ModelWorker


def replicate_to_mesh(
    mesh: jax.sharding.Mesh, *arrs: jax.Array
) -> tuple[jax.Array, ...] | jax.Array:
    """Replicate arrays needed by host-side prefill state capture."""
    out = jax.device_put(arrs, NamedSharding(mesh, P()))
    return out[0] if len(out) == 1 else out


class BaseDraftWorker(ABC):
    """Draft model worker interface for speculative decoding.

    Concrete implementations hold the draft model runner and own the
    algorithm-specific fused draft/verify preparation.
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
    (the draft model). EAGLE/EAGLE3 and NEXTN use fused linear-chain paths;
    DFlash overrides the algorithm-specific draft and verify stages.
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
        can_use_linear_fused_verify = (
            self.topk == 1
            and self.speculative_num_steps > 1
            and self.speculative_num_draft_tokens == self.speculative_num_steps + 1
        )
        is_multi_layer_mtp = bool(getattr(draft_worker, "is_multi_layer_mtp", False))
        self._can_use_fused_eagle_verify = (
            (
                self.speculative_algorithm.is_eagle_family()
                or (self.speculative_algorithm.is_nextn() and not is_multi_layer_mtp)
            )
            and can_use_linear_fused_verify
            and server_args.attention_backend == "fa"
            and os.getenv("SGL_JAX_DISABLE_FUSED_EAGLE3_RECURRENT_DRAFT") != "1"
        )
        self._can_use_fused_mtp_verify = (
            self.speculative_algorithm.is_nextn()
            and is_multi_layer_mtp
            and can_use_linear_fused_verify
            and server_args.attention_backend == "fa"
        )
        if self.speculative_algorithm.is_eagle_family() and not self._can_use_fused_eagle_verify:
            raise ValueError(
                "EAGLE/EAGLE3 only support the fused topk=1 FA path; check num_steps, "
                "num_draft_tokens, attention_backend, and "
                "SGL_JAX_DISABLE_FUSED_EAGLE3_RECURRENT_DRAFT."
            )
        if self.speculative_algorithm.is_nextn() and not (
            self._can_use_fused_eagle_verify or self._can_use_fused_mtp_verify
        ):
            raise ValueError(
                "NEXTN only supports the fused topk=1 FA path; check num_steps, "
                "num_draft_tokens, and attention_backend."
            )

        self.req_to_token_pool, self.token_to_kv_pool_allocator = target_worker.get_memory_pool()

        (
            self.precompile_token_paddings,
            self.precompile_bs_paddings,
            self.precompile_cache_loc_paddings,
        ) = target_worker.get_precompile_paddings()
        self.spec_relay_buffers = None

    @property
    def target_worker(self) -> ModelWorker:
        return self._target_worker

    @property
    def draft_worker(self) -> BaseDraftWorker:
        return self._draft_worker

    def init_spec_relay_buffers(self):
        if self.spec_relay_buffers is not None:
            return
        from sgl_jax.srt.speculative.relay_buffer import create_spec_relay_buffers

        hidden_dtype = jnp.bfloat16 if self.server_args.dtype == "bfloat16" else jnp.float32
        self.spec_relay_buffers = create_spec_relay_buffers(
            self.mesh,
            self.req_to_token_pool,
            dp_size=self.server_args.dp_size,
            num_steps=self.speculative_num_steps,
            hidden_size=self.target_worker.model_config.hidden_size,
            hidden_dtype=hidden_dtype,
        )

    def _can_use_fused_spec_prefill(self, model_worker_batch: ModelWorkerBatch) -> bool:
        # EAGLE/EAGLE3 keep the ordinary target prefill, then run their draft prefix
        # forward through eagle_prefill_draft_extend. DFLASH overrides this.
        return False

    def _get_cur_allocate_lens(self, model_worker_batch: ModelWorkerBatch):
        allocate_lens = getattr(model_worker_batch.spec_info_padded, "allocate_lens", None)
        if allocate_lens is None:
            return None
        return np.asarray(allocate_lens)[model_worker_batch.logits_indices_selector]

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
        if not model_worker_batch.forward_mode.is_decode():
            raise NotImplementedError(
                "Spec decode-overlap entry only supports decode batches; "
                "prefill overlap uses forward_batch_speculative_generation()."
            )
        if not (
            self._can_use_fused_eagle_verify or getattr(self, "_can_use_fused_mtp_verify", False)
        ):
            raise NotImplementedError(
                "Spec overlap entry only supports fused topk=1 EAGLE/EAGLE3/NEXTN decode."
            )

        self.init_spec_relay_buffers()
        self._prepare_overlap_sampling_info(model_worker_batch)
        cur_allocate_lens = self._get_cur_allocate_lens(model_worker_batch)

        if getattr(self, "_can_use_fused_mtp_verify", False):
            from sgl_jax.srt.speculative.draft_extend_fused import (
                spec_decode_mtp_overlap,
            )

            result = spec_decode_mtp_overlap(self, model_worker_batch, cur_allocate_lens)
        else:
            from sgl_jax.srt.speculative.draft_extend_fused import (
                spec_decode_eagle_overlap,
            )

            result = spec_decode_eagle_overlap(self, model_worker_batch, cur_allocate_lens)
        launch_done = getattr(model_worker_batch, "launch_done", None)
        if launch_done is not None:
            launch_done.set()
        return result

    def forward_batch_speculative_prefill_overlap(self, model_worker_batch: ModelWorkerBatch):
        raise NotImplementedError(
            "EAGLE/EAGLE3 use ordinary target prefill plus fused draft-prefix extend; "
            "DFLASH implements its own fused prefill-overlap entry."
        )

    def forward_batch_speculative_generation(
        self, model_worker_batch: ModelWorkerBatch, launch_done=None
    ):
        from sgl_jax.srt.managers.scheduler import GenerationBatchResult
        from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata

        uses_host_state = uses_host_eagle_state(
            not self.server_args.disable_overlap_schedule,
            getattr(model_worker_batch, "spec_algorithm", None),
        )
        if launch_done is None and not uses_host_state:
            self._prepare_overlap_sampling_info(model_worker_batch)

        if model_worker_batch.forward_mode.is_extend():
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
            if model_worker_batch.sampling_info.is_all_greedy and not uses_host_state:
                logits_output, _, cache_miss_count, bid, _seq_lens = self.forward_target_extend(
                    model_worker_batch,
                    sampling_metadata,
                    skip_sample=True,
                )
                next_token_ids = jnp.argmax(logits_output.next_token_logits, axis=-1).astype(
                    jnp.int32
                )
            else:
                logits_output, next_token_ids, cache_miss_count, bid, _seq_lens = (
                    self.forward_target_extend(model_worker_batch, sampling_metadata)
                )
            if model_worker_batch.dp_size > 1:
                from jax.experimental.multihost_utils import process_allgather

                next_token_ids = process_allgather(next_token_ids, tiled=True)
            self.draft_worker.draft_extend_for_prefill(
                model_worker_batch, logits_output.hidden_states, next_token_ids
            )
            # The generic overlap loop waits for the current batch to finish
            # enqueueing before it resolves the previous batch.
            batch_launch_done = getattr(model_worker_batch, "launch_done", None)
            if batch_launch_done is not None:
                batch_launch_done.set()
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                next_draft_input=model_worker_batch.spec_info_padded,
                bid=bid,
                cache_miss_count=cache_miss_count,
                extend_input_len_per_req=None,
                extend_logprob_start_len_per_req=None,
            )

        # EAGLE carries DP-padded allocation lengths. Other algorithms can own
        # committed KV lengths directly and return None from the hook.
        cur_allocate_lens = self._get_cur_allocate_lens(model_worker_batch)
        if self._can_use_fused_eagle_verify:
            from sgl_jax.srt.speculative.draft_extend_fused import (
                eagle_recurrent_draft_extend_for_decode,
                spec_decode_verify,
            )

            # The first decode after prefill expands its seed in one fused
            # recurrent bootstrap. Steady-state rounds consume the chain from
            # the previous fused recurrent draft-extend.
            draft_to_target_token_ids = self.draft_worker.prepare_for_fused_verify(
                model_worker_batch
            )
            batch_output = spec_decode_verify(
                self,
                model_worker_batch,
                cur_allocate_lens,
                draft_to_target_token_ids=draft_to_target_token_ids,
                draft_padding_prepared=True,
            )
            eagle_recurrent_draft_extend_for_decode(
                self.draft_worker,
                model_worker_batch,
                batch_output,
            )
            launch_done = getattr(model_worker_batch, "launch_done", None)
            if launch_done is not None:
                launch_done.set()
            return batch_output
        if getattr(self, "_can_use_fused_mtp_verify", False):
            from sgl_jax.srt.speculative.draft_extend_fused import (
                mtp_draft_extend_for_decode,
                spec_decode_verify,
            )

            draft_to_target_token_ids = self.draft_worker.prepare_for_fused_verify(
                model_worker_batch
            )
            batch_output = spec_decode_verify(
                self,
                model_worker_batch,
                cur_allocate_lens,
                draft_to_target_token_ids=draft_to_target_token_ids,
                draft_padding_prepared=True,
            )
            mtp_draft_extend_for_decode(
                self.draft_worker,
                model_worker_batch,
                batch_output,
            )
            launch_done = getattr(model_worker_batch, "launch_done", None)
            if launch_done is not None:
                launch_done.set()
            return batch_output

        if self.speculative_algorithm.is_dflash():
            # DFlash owns a fused one-shot draft JIT and a fused target-verify
            # JIT. Keep their orchestration explicit instead of treating this
            # as a generic unfused fallback.
            self.draft_worker.draft(model_worker_batch)
            batch_output = self.draft_worker.verify(
                model_worker_batch,
                cur_allocate_lens,
            )
            self.draft_worker.draft_extend_for_decode(
                model_worker_batch,
                batch_output,
            )
            launch_done = getattr(model_worker_batch, "launch_done", None)
            if launch_done is not None:
                launch_done.set()
            return batch_output

        raise RuntimeError("No unfused speculative decode fallback is supported.")

    def forward_target_extend(
        self,
        model_worker_batch: ModelWorkerBatch,
        sampling_metadata,
        *,
        skip_sample: bool = False,
    ):
        from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode

        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        target_worker = getattr(self.target_worker, "worker", self.target_worker)
        logits_output, next_token_ids, cache_miss_count = target_worker.forward_batch_generation(
            model_worker_batch,
            sampling_metadata=sampling_metadata,
            skip_sample=skip_sample,
        )
        return (
            logits_output,
            next_token_ids,
            cache_miss_count,
            model_worker_batch.bid,
            model_worker_batch.seq_lens,
        )
