"""Multi-layer NEXTN/MTP draft worker (#1053 P1-4).

The N draft steps each use a *different* draft model runner (one per
``mtp_layer_idx``). Per the NEXTN training recipe (DeepSeek/MiMo), every
layer i takes ``(embed(tok_{t+i}), target_hidden_t)`` — i.e. each layer
sees the **target** model's hidden, not the previous MTP layer's output.
So ``draft_extend`` forwards each layer once with the same target hidden
and a per-layer rotated ``input_ids`` (shift+append previous topk), and
``draft_forward`` does no model forward — it just reads the per-layer
topk that ``draft_extend`` already stored. Chain-style hidden passing is
only correct for architectures that explicitly chain MTP hidden states.
"""

from __future__ import annotations

import copy
import json
import logging

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.logits_processor import LogitsMetadata
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.managers.scheduler import GenerationBatchResult
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sgl_jax.srt.speculative.base_worker import (
    filter_spec_precompile_token_paddings,
    replicate_to_mesh,
)
from sgl_jax.srt.speculative.eagle_draft_worker import (
    EagleDraftWorker,
    select_top_k_tokens,
    topk_probs_from_logits,
    update_eagle_lists,
)
from sgl_jax.srt.speculative.eagle_util import EagleDraftInput

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
        assert (
            self.topk == 1
        ), f"MultiLayerDraftWorker fused decode requires topk=1, got {self.topk}"
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

        for i, w in enumerate(self._workers):
            w.model_runner.initialize_jit()

        # Share target's SWA index mapping with draft workers. The scheduler
        # allocates KV pages via the target's SWATokenToKVPoolAllocator which
        # maintains full_to_swa_index_mapping. Draft workers have independent
        # allocators whose mappings stay at zero. Without this, draft SWA
        # attention remaps all page indices to 0 → reads wrong KV data.
        target_allocator = target_worker.model_runner.token_to_kv_pool_allocator
        target_swa_mapping = getattr(target_allocator, "full_to_swa_index_mapping", None)
        if target_swa_mapping is not None:
            for w in self._workers:
                object.__setattr__(
                    w.model_runner.attn_backend, "swa_index_mapping", target_swa_mapping
                )

        (
            self.precompile_token_paddings,
            self.precompile_bs_paddings,
            self.precompile_cache_loc_paddings,
        ) = target_worker.get_precompile_paddings()
        self.precompile_token_paddings = filter_spec_precompile_token_paddings(
            server_args,
            self.precompile_token_paddings,
        )

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
        # dext already forwarded each MTP layer once and stored per-layer topk
        # in spec_info.topk_{p,index} with shape (bs, num_steps, topk). We only
        # feed those through select_top_k_tokens to assemble the tree lists —
        # no model forward here (mirrors GPU multi_layer_eagle_worker_v2).
        topk_p = model_worker_batch.spec_info_padded.topk_p
        topk_index = model_worker_batch.spec_info_padded.topk_index
        hidden_states = model_worker_batch.spec_info_padded.hidden_states
        if self.topk == 1:
            if isinstance(topk_index, np.ndarray):
                return None, np.asarray(topk_index[:, :, 0], dtype=np.int32), None
            return None, topk_index[:, :, 0].astype(jnp.int32), None

        bs = model_worker_batch.seq_lens.shape[0]
        step_min_1 = self.speculative_num_steps - 1
        score_list = jnp.zeros((bs, 1 + step_min_1 * self.topk, self.topk))
        token_list = jnp.zeros(
            (bs, self.topk + step_min_1 * self.topk * self.topk), dtype=jnp.int32
        )
        parents_list = jnp.zeros((bs, self.topk + 1 + step_min_1 * self.topk))
        scores = None
        for i in range(self.speculative_num_steps):
            _, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p[:, i], topk_index[:, i], hidden_states, scores, self.topk
            )
            score_list, token_list, parents_list = update_eagle_lists(
                i, score_list, token_list, parents_list, tree_info, self.topk
            )
        return score_list, token_list, parents_list

    @staticmethod
    def _rotate_ids(mwb: ModelWorkerBatch, last_tok: np.ndarray, sel_pos: np.ndarray) -> None:
        """In-place left-shift each req's input_ids by 1, then write last_tok[slot]
        at position sel_pos[slot]. Mirrors GPU rotate_input_ids_triton."""
        dp = mwb.dp_size
        per_dp_bs = mwb.per_dp_bs_size
        per_dp_tok = len(mwb.input_ids) // dp
        ext = mwb.extend_seq_lens
        for r in range(dp):
            pt = r * per_dp_tok
            for j in range(per_dp_bs):
                s = r * per_dp_bs + j
                el = int(ext[s])
                if el == 0:
                    continue
                seg = mwb.input_ids[pt : pt + el]
                seg[:-1] = seg[1:]
                seg[int(sel_pos[s])] = last_tok[s]
                pt += el

    def draft_extend_for_prefill(
        self,
        model_worker_batch: ModelWorkerBatch,
        hidden_states: jax.Array,
        next_token_ids: jax.Array,
    ) -> None:
        """Prefill-extend across all MTP layers.

        NEXTN (DeepSeek/MiMo) trains each MTP layer i with input
        ``(embed(tok_{t+i}), target_hidden_t)`` — every layer sees the *target*
        hidden, not the previous MTP layer's output. So per layer we keep
        ``hidden = target_hidden_states`` and only rotate ``input_ids`` (shift
        by 1, append previous layer's topk) so layer i's last-logit position is
        ``(topk_{i-1}, target_hidden_{last})``. Each layer's topk goes into
        ``spec_info.topk_index[:, i]`` and ``draft_forward`` does no model
        forward. Mirrors GPU sglang ``multi_layer_eagle_worker_v2`` for
        non-chain architectures.
        """
        sel = np.asarray(model_worker_batch.logits_indices_selector)
        verified_id_np = np.asarray(jax.device_get(next_token_ids))[sel]
        model_worker_batch.spec_info_padded = EagleDraftInput(
            hidden_states=hidden_states,
            verified_id=verified_id_np,
            num_tokens_per_batch=np.asarray(1, dtype=jnp.int32),
            num_tokens_for_logprob_per_batch=np.asarray(1, dtype=jnp.int32),
            allocate_lens=model_worker_batch.seq_lens,
        )
        model_worker_batch.return_hidden_states = False
        model_worker_batch.spec_info_padded.prepare_for_extend_after_target_prefill(
            model_worker_batch=model_worker_batch
        )
        # FULL: layer i+1 needs layer i's per-token hidden over the whole prefix.
        model_worker_batch.spec_info_padded.capture_hidden_mode = CaptureHiddenMode.FULL
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        last_idx = model_worker_batch.logits_indices

        # Pad verified_id from (real_bs,) to (padded_bs,) so jit'd MTP forward
        # sees the bucket shape every time (forward_batch.spec_info.verified_id
        # is a pytree leaf; without this each real_bs triggers a fresh trace).
        # Restored to real_bs after the loop so cross-round flat state stays
        # flat-ordered. Mirrors single-layer EagleDraftWorker.
        padded_bs = int(model_worker_batch.seq_lens.shape[0])
        if verified_id_np.shape[0] < padded_bs:
            model_worker_batch.spec_info_padded.verified_id = np.pad(
                verified_id_np, ((0, padded_bs - verified_id_np.shape[0]),)
            )

        layer0_out = None
        all_topk_p, all_topk_index = [], []
        ext = model_worker_batch.extend_seq_lens
        sel_pos = np.clip(ext - 1, 0, None).astype(np.int64)
        for i, w in enumerate(self._workers):
            mr = w.model_runner
            model_worker_batch.spec_info_padded.hidden_states = hidden_states
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
            if i == 0:
                layer0_out = logits_output
            tp, ti = topk_probs_from_logits(
                replicate_to_mesh(self.mesh, logits_output.next_token_logits), self.topk
            )
            all_topk_p.append(np.asarray(jax.device_get(tp)))
            all_topk_index.append(np.asarray(jax.device_get(ti)))
            if i < len(self._workers) - 1:
                self._rotate_ids(model_worker_batch, all_topk_index[-1][:, 0], sel_pos)

        # spec_info.hidden_states is only consumed by select_top_k_tokens (shape
        # bookkeeping at topk=1); keep layer0's last-token hidden for that.
        rep_hidden = replicate_to_mesh(self.mesh, layer0_out.hidden_states)
        dp_size = model_worker_batch.dp_size
        if dp_size > 1:
            per_dp_tokens = rep_hidden.shape[0] // dp_size
            per_dp_bs = last_idx.shape[0] // dp_size
            last_idx = last_idx.copy()
            for k in range(1, dp_size):
                last_idx[k * per_dp_bs : (k + 1) * per_dp_bs] += k * per_dp_tokens
        si = model_worker_batch.spec_info_padded
        si.hidden_states = np.asarray(jax.device_get(rep_hidden))[last_idx][sel]
        si.topk_p = np.stack(all_topk_p, axis=1)[sel]
        si.topk_index = np.stack(all_topk_index, axis=1)[sel]
        si.allocate_lens = np.asarray(model_worker_batch.seq_lens)[sel]
        si.verified_id = verified_id_np

    def draft_extend_for_decode(
        self, model_worker_batch: ModelWorkerBatch, batch_output: GenerationBatchResult
    ) -> None:
        """Decode-extend across all MTP layers."""
        from sgl_jax.srt.speculative.draft_extend_fused import (
            draft_extend_for_decode_fused,
        )

        return draft_extend_for_decode_fused(self, model_worker_batch, batch_output)
