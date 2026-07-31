"""Fused topk=1 multi-layer NEXTN/MTP draft worker."""

from __future__ import annotations

import copy
import json

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.managers.scheduler import GenerationBatchResult
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sgl_jax.srt.speculative.eagle_draft_worker import EagleDraftWorker
from sgl_jax.srt.speculative.eagle_info import EagleDraftInput
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm


def _server_args_with_mtp_layer(server_args, layer_idx: int):
    layer_args = copy.copy(server_args)
    override = json.loads(layer_args.json_model_override_args or "{}")
    override["mtp_layer_idx"] = layer_idx
    layer_args.json_model_override_args = json.dumps(override)
    return layer_args


class MultiLayerDraftWorker(EagleDraftWorker):
    """One draft runner per MTP layer, fused into a linear topk=1 chain."""

    is_multi_layer_mtp = True

    def __init__(self, server_args, target_worker: ModelWorker):
        self.server_args = server_args
        self.target_worker_ref = target_worker
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        if not self.speculative_algorithm.is_eagle() or self.topk != 1:
            raise ValueError("MultiLayerDraftWorker only supports topk=1 EAGLE-style decoding.")
        self.hot_token_ids = None

        num_mtp_layers = getattr(
            target_worker.model_config.hf_config,
            "num_nextn_predict_layers",
            None,
        )
        if num_mtp_layers is not None and num_mtp_layers != self.speculative_num_steps:
            raise ValueError(
                f"--speculative-num-steps={self.speculative_num_steps} must equal "
                f"num_nextn_predict_layers={num_mtp_layers}."
            )

        req_to_token_pool, _ = target_worker.get_memory_pool()
        self._workers = [
            ModelWorker(
                _server_args_with_mtp_layer(server_args, layer_idx),
                target_worker.mesh,
                req_to_token_pool=req_to_token_pool,
                is_draft_worker=True,
            )
            for layer_idx in range(self.speculative_num_steps)
        ]
        self._worker = self._workers[0]

        EagleDraftInput.ALLOC_LEN_PER_DECODE = max(
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
        )

        for worker in self._workers:
            self._share_embed_head_one(target_worker, worker)
            worker.model_runner.initialize_jit()

        target_allocator = target_worker.model_runner.token_to_kv_pool_allocator
        target_swa_mapping = getattr(
            target_allocator,
            "full_to_swa_index_mapping",
            None,
        )
        if target_swa_mapping is not None:
            for worker in self._workers:
                object.__setattr__(
                    worker.model_runner.attn_backend,
                    "swa_index_mapping",
                    target_swa_mapping,
                )

        (
            self.precompile_token_paddings,
            self.precompile_bs_paddings,
            self.precompile_cache_loc_paddings,
        ) = target_worker.get_precompile_paddings()

    @staticmethod
    def _share_embed_head_one(target_worker: ModelWorker, draft_worker: ModelWorker):
        embed, head = target_worker.model_runner.model.get_embed_and_head()
        model = draft_worker.model_runner.model
        if getattr(model, "load_lm_head_from_target", False):
            model.set_embed_and_head(embed, head)
        else:
            model.set_embed(embed)

    @property
    def draft_model_runner(self):
        return self._workers[0].model_runner

    def runner(self, step: int):
        return self._workers[step].model_runner

    def draft(self, model_worker_batch: ModelWorkerBatch) -> None:
        raise RuntimeError("NEXTN draft() is unavailable; use the fused linear-chain path.")

    def prepare_for_fused_verify(self, model_worker_batch: ModelWorkerBatch):
        token_chain = model_worker_batch.spec_info_padded.topk_index
        if (
            token_chain is None
            or token_chain.ndim != 2
            or token_chain.shape[1] != self.speculative_num_steps
        ):
            raise ValueError(
                "NEXTN fused verify requires a precomputed token chain with shape "
                f"(batch, {self.speculative_num_steps}); got "
                f"{None if token_chain is None else token_chain.shape}."
            )
        self.padding_for_decode(model_worker_batch)
        return None

    def draft_extend_for_prefill(
        self,
        model_worker_batch: ModelWorkerBatch,
        hidden_states: jax.Array,
        next_token_ids: jax.Array,
    ) -> None:
        selector = np.asarray(model_worker_batch.logits_indices_selector)
        verified_id = np.asarray(jax.device_get(next_token_ids))[selector]
        model_worker_batch.spec_info_padded = EagleDraftInput(
            hidden_states=hidden_states,
            verified_id=verified_id,
            num_tokens_per_batch=np.asarray(1, dtype=jnp.int32),
            num_tokens_for_logprob_per_batch=np.asarray(1, dtype=jnp.int32),
            allocate_lens=model_worker_batch.seq_lens,
        )
        model_worker_batch.return_hidden_states = False
        model_worker_batch.spec_info_padded.prepare_for_extend_after_target_prefill(
            model_worker_batch
        )
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        model_worker_batch.spec_info_padded.capture_hidden_mode = CaptureHiddenMode.FULL

        from sgl_jax.srt.speculative.draft_extend_fused import (
            mtp_prefill_draft_extend,
        )

        selected_hidden, token_chain = mtp_prefill_draft_extend(
            self,
            model_worker_batch,
            hidden_states,
        )
        model_worker_batch.spec_info_padded.hidden_states = selected_hidden
        model_worker_batch.spec_info_padded.topk_index = token_chain
        model_worker_batch.spec_info_padded.verified_id = verified_id
        model_worker_batch.spec_info_padded.allocate_lens = np.asarray(
            model_worker_batch.seq_lens
        )[selector]

    def draft_extend_for_decode(
        self,
        model_worker_batch: ModelWorkerBatch,
        batch_output: GenerationBatchResult,
    ) -> None:
        from sgl_jax.srt.speculative.draft_extend_fused import (
            mtp_draft_extend_for_decode,
        )

        mtp_draft_extend_for_decode(self, model_worker_batch, batch_output)
