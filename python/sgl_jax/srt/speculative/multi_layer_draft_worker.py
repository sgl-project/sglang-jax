"""MiMo-V2.5-Pro multi-layer MTP draft worker (#1053 P1-4).

Differs from ``EagleDraftWorker`` in that the N draft steps each use a
*different* draft model runner (one per ``mtp_layer_idx``), with hidden
states flowing layer→layer (layer i's output hidden becomes layer i+1's
``spec_info.hidden_states``). Everything else — padding, tree build,
verify-input construction, replicate helpers — is reused from
``EagleDraftWorker``.
"""

from __future__ import annotations

import copy
import json
import logging

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.logits_processor import LogitsMetadata
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.managers.scheduler import GenerationBatchResult
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sgl_jax.srt.speculative.base_worker import replicate_to_mesh
from sgl_jax.srt.speculative.eagle_draft_worker import (
    EagleDraftWorker,
    select_top_k_tokens,
    topk_probs_from_logits,
    update_eagle_lists,
    update_forward_batch_info,
)
from sgl_jax.srt.speculative.eagle_util import EagleDraftInput
from sgl_jax.srt.utils.jax_utils import device_array

logger = logging.getLogger(__name__)


def _server_args_with_mtp_layer(server_args, layer_idx: int):
    sa = copy.copy(server_args)
    override = json.loads(sa.json_model_override_args or "{}")
    override["mtp_layer_idx"] = layer_idx
    sa.json_model_override_args = json.dumps(override)
    return sa


class MultiLayerDraftWorker(EagleDraftWorker):
    """N independent draft model runners, one per MTP layer.

    ``speculative_num_steps`` must equal the number of MTP layers (one
    draft step per layer). ``draft_model_runner`` resolves to layer 0's
    runner for code paths that need a single runner handle (KV-pool
    sizing, attention-backend type, mesh — all identical across layers).
    """

    def __init__(self, server_args, target_worker: ModelWorker):
        self.server_args = server_args
        self.target_worker_ref = target_worker
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.page_size = server_args.page_size
        from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.hot_token_ids = None

        self.num_mtp_layers = self.speculative_num_steps
        assert self.num_mtp_layers > 1
        cfg_mtp = getattr(target_worker.model_config.hf_config, "num_nextn_predict_layers", None)
        # MiMo-style configs omit num_nextn_predict_layers; only enforce equality
        # when the field exists (scheduler routes here iff n_mtp>1 either way).
        assert cfg_mtp is None or cfg_mtp == self.num_mtp_layers, (
            f"--speculative-num-steps={self.speculative_num_steps} must equal the "
            f"model's num_nextn_predict_layers={cfg_mtp} (one runner per MTP layer)"
        )
        req_to_token_pool, _ = target_worker.get_memory_pool()

        self._workers: list[ModelWorker] = []
        for i in range(self.num_mtp_layers):
            sa = _server_args_with_mtp_layer(server_args, i)
            self._workers.append(
                ModelWorker(
                    sa,
                    target_worker.mesh,
                    req_to_token_pool=req_to_token_pool,
                    is_draft_worker=True,
                )
            )
        self._worker = self._workers[0]

        EagleDraftInput.ALLOC_LEN_PER_DECODE = max(
            self.speculative_num_steps * self.topk, self.speculative_num_draft_tokens
        )

        for w in self._workers:
            self._share_embed_head_one(target_worker, w)

        target_slot_range = target_worker.model_runner.max_total_num_tokens
        for i, w in enumerate(self._workers):
            draft_pool = w.model_runner.max_total_num_tokens
            assert draft_pool >= target_slot_range, (
                f"MTP layer {i} draft KV pool ({draft_pool}) < target slot range "
                f"({target_slot_range})"
            )
            w.model_runner.initialize_jit()

        (
            self.precompile_token_paddings,
            self.precompile_bs_paddings,
            self.precompile_cache_loc_paddings,
        ) = target_worker.get_precompile_paddings()

    def _share_embed_head_one(self, target_worker: ModelWorker, draft_worker: ModelWorker):
        embed, head = target_worker.model_runner.model.get_embed_and_head()
        m = draft_worker.model_runner.model
        if getattr(m, "load_lm_head_from_target", False):
            m.set_embed_and_head(embed, head)
        else:
            m.set_embed(embed)

    @property
    def draft_model_runner(self):
        return self._workers[0].model_runner

    def runner(self, step: int):
        return self._workers[step].model_runner

    # ---- per-layer overrides ------------------------------------------------

    def draft_forward(self, model_worker_batch: ModelWorkerBatch):
        topk_p = model_worker_batch.spec_info.topk_p
        topk_index = model_worker_batch.spec_info.topk_index
        hidden_states = model_worker_batch.spec_info.hidden_states
        bs = model_worker_batch.seq_lens.shape[0]
        step_min_1 = self.speculative_num_steps - 1
        score_list = jnp.empty((bs, 1 + step_min_1 * self.topk, self.topk))
        token_list = jnp.empty(
            (bs, self.topk + step_min_1 * self.topk * self.topk), dtype=jnp.int32
        )
        parents_list = jnp.empty((bs, self.topk + 1 + step_min_1 * self.topk))
        scores = None
        positions_base = device_array(
            np.repeat(model_worker_batch.seq_lens, self.topk),
            sharding=NamedSharding(self.mesh, P()),
        )
        logits_metadata = LogitsMetadata.from_model_worker_batch(model_worker_batch, self.mesh)

        # Per-layer attention metadata: each layer has its own KV pool, so
        # page_indices differ; positions/seq_lens are shared so we only need
        # the i-th step's metadata from layer i.
        metadata_per_layer = [
            w.model_runner.attn_backend.get_eagle_multi_step_metadata(model_worker_batch)
            for w in self._workers
        ]

        forward_batch = ForwardBatch.init_new(model_worker_batch, self.draft_model_runner)
        forward_batch.out_cache_loc = np.empty((1,))
        forward_batch.cache_loc = np.empty((1,))
        forward_batch.spec_info = EagleDraftInput()
        forward_batch.spec_info.hidden_states = jnp.empty((bs * self.topk, hidden_states.shape[1]))

        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list, token_list, parents_list = update_eagle_lists(
                i, score_list, token_list, parents_list, tree_info, self.topk
            )
            if i == self.speculative_num_steps - 1:
                break

            forward_batch = update_forward_batch_info(
                forward_batch, i, input_ids, hidden_states, positions_base
            )
            mr = self.runner(i)
            mr.attn_backend.forward_metadata = metadata_per_layer[i][i]
            forward_batch.bid = model_worker_batch.bid
            logits_output, _, _ = mr.forward(forward_batch, logits_metadata=logits_metadata)

            topk_p, topk_index = topk_probs_from_logits(logits_output.next_token_logits, self.topk)
            if self.hot_token_ids is not None:
                topk_index = self.hot_token_ids[topk_index]
            hidden_states = replicate_to_mesh(self.mesh, logits_output.hidden_states)

        return score_list, token_list, parents_list

    def draft_extend_for_prefill(
        self,
        model_worker_batch: ModelWorkerBatch,
        hidden_states: jax.Array,
        next_token_ids: jax.Array,
    ) -> None:
        """Prefill-extend across all MTP layers.

        Layer 0 consumes the target's full-prefix hidden_states; layer i>0
        consumes layer i-1's output hidden. Each layer writes its own
        prefix KV. Only layer 0's topk/hidden are kept as the next-round
        draft state (draft step 0 starts from layer 0).
        """
        sel = np.asarray(model_worker_batch.logits_indices_selector)
        verified_id_np = np.asarray(jax.device_get(next_token_ids))[sel]
        model_worker_batch.spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            verified_id=verified_id_np,
            num_tokens_per_batch=np.asarray(1, dtype=jnp.int32),
            num_tokens_for_logprob_per_batch=np.asarray(1, dtype=jnp.int32),
            allocate_lens=model_worker_batch.seq_lens,
        )
        model_worker_batch.return_hidden_states = False
        model_worker_batch.spec_info.prepare_for_extend_after_target_prefill(
            model_worker_batch=model_worker_batch
        )
        # FULL: layer i+1 needs layer i's per-token hidden over the whole prefix.
        model_worker_batch.spec_info.capture_hidden_mode = CaptureHiddenMode.FULL
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        last_idx = model_worker_batch.logits_indices

        layer0_out = None
        cur_hidden = hidden_states
        for i, w in enumerate(self._workers):
            mr = w.model_runner
            model_worker_batch.spec_info.hidden_states = cur_hidden
            forward_batch = ForwardBatch.init_new(model_worker_batch, mr)
            forward_batch.return_logprob = False
            mr.attn_backend.forward_metadata = mr.attn_backend.get_eagle_forward_metadata(
                model_worker_batch
            )
            forward_batch.forward_mode = ForwardMode.EXTEND
            logits_output, _, _ = mr.forward(
                forward_batch,
                logits_metadata=LogitsMetadata.from_model_worker_batch(
                    model_worker_batch, self.mesh
                ),
            )
            cur_hidden = logits_output.hidden_states
            if i == 0:
                layer0_out = logits_output

        # Next-round draft state = layer 0's last-token hidden + topk (draft step 0
        # starts from layer 0 with target's last hidden, so we cache layer 0's
        # output here so step 0 can be skipped).
        rep_logits, rep_hidden = replicate_to_mesh(
            self.mesh, layer0_out.next_token_logits, layer0_out.hidden_states
        )
        layer0_out.next_token_logits = rep_logits[sel, :]
        layer0_out.hidden_states = rep_hidden[last_idx][sel]
        model_worker_batch.spec_info.hidden_states = hidden_states
        model_worker_batch.spec_info.allocate_lens = np.asarray(model_worker_batch.seq_lens)[sel]
        self.capture_for_decode(layer0_out, model_worker_batch.spec_info)

    def draft_extend_for_decode(
        self, model_worker_batch: ModelWorkerBatch, batch_output: GenerationBatchResult
    ) -> None:
        """Decode-extend across all MTP layers (each updates its own KV).

        Same chaining as prefill: layer 0 consumes target verify hidden,
        layer i>0 consumes layer i-1's output. Only layer 0's topk/hidden
        become the next-round draft state.
        """
        if batch_output.next_draft_input.verified_id.shape[0] <= 0:
            return
        target_hidden = batch_output.logits_output.hidden_states

        # prepare_for_extend_after_verify mutates mwb.seq_lens / input_ids /
        # spec_info in place — call it once (semantics are layer-independent),
        # then per layer only swap hidden_states + recompute forward_metadata
        # against that layer's KV pool.
        draft_input = EagleDraftInput(
            hidden_states=target_hidden, allocate_lens=batch_output.allocate_lens
        )
        mwb, logits_metadata = draft_input.prepare_for_extend_after_verify(
            model_worker_batch,
            self.draft_model_runner,
            batch_output,
            self.speculative_num_draft_tokens,
        )
        if mwb.input_ids.shape[0] <= 0:
            return

        layer0_logits = None
        cur_hidden = target_hidden
        for i, w in enumerate(self._workers):
            mr = w.model_runner
            mwb.spec_info.hidden_states = cur_hidden
            mr.attn_backend.forward_metadata = mr.attn_backend.get_eagle_forward_metadata(mwb)
            forward_batch = ForwardBatch.init_new(mwb, mr)
            logits_output, _, _ = mr.forward(forward_batch, logits_metadata=logits_metadata)
            if i == 0:
                layer0_logits = logits_output
            cur_hidden = logits_output.hidden_states

        # logits_indices_selector maps global-flat req k → DP-padded slot s_k.
        # rep_logits/rep_hidden are DP-padded (total_bs*(steps+1), …); gather
        # each req's accept_len-th entry by slot, producing global-flat
        # (real_bs, …) for the cross-round spec_info.
        sel = np.asarray(model_worker_batch.logits_indices_selector)
        accept_host = np.asarray(jax.device_get(batch_output.accept_lens))
        select_index = sel * (self.speculative_num_steps + 1) + accept_host[sel] - 1
        rep_logits, rep_hidden = replicate_to_mesh(
            self.mesh, layer0_logits.next_token_logits, layer0_logits.hidden_states
        )
        topk_p, topk_index = topk_probs_from_logits(rep_logits[select_index], self.topk)
        batch_output.next_draft_input.hidden_states = rep_hidden[select_index]
        batch_output.next_draft_input.topk_p = topk_p
        batch_output.next_draft_input.topk_index = topk_index
        batch_output.next_draft_input.verified_id = batch_output.next_draft_input.verified_id[
            select_index
        ]
        batch_output.allocate_lens = batch_output.allocate_lens[: model_worker_batch.real_bs]
        # accept_lens stays DP-padded (total_bs,) — scheduler per-rank seq_lens
        # update and _resolve_spec_decode_token_ids both index by DP slot.
        batch_output.accept_lens = accept_host
