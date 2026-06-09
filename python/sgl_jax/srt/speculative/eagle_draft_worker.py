import functools
import logging

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.managers.scheduler import GenerationBatchResult
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sgl_jax.srt.speculative.base_worker import BaseDraftWorker, replicate_to_mesh
from sgl_jax.srt.speculative.eagle_util import (
    EagleDraftInput,
    EagleVerifyInput,
    build_chain_verify_inputs,
    build_chain_verify_inputs_device,
    build_tree_kernel_efficient,
    build_tree_mask_for_draft_decode,
)
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm
from sgl_jax.srt.utils.common_utils import get_bool_env_var
from sgl_jax.srt.utils.jax_utils import device_array

logger = logging.getLogger(__name__)
RETURN_ORIGINAL_LOGPROB = get_bool_env_var("RETURN_ORIGINAL_LOGPROB")


class EagleDraftWorker(BaseDraftWorker):
    """EAGLE draft model worker.

    Holds a ``ModelWorker`` (the draft model runner) via composition and
    implements draft-specific logic: multi-step decode, tree building,
    prefill extend, and decode extend.
    """

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
        self.hot_token_ids = None

        req_to_token_pool, _ = target_worker.get_memory_pool()

        # Compose a ModelWorker for the draft model (instead of inheriting)
        # Must be created last to ensure model state is correct.
        self._worker = ModelWorker(
            server_args,
            target_worker.mesh,
            req_to_token_pool=req_to_token_pool,
            is_draft_worker=True,
        )

        EagleDraftInput.ALLOC_LEN_PER_DECODE = max(
            self.speculative_num_steps * self.topk, self.speculative_num_draft_tokens
        )

        self._share_embed_head(target_worker)

        self._worker.model_runner.initialize_jit()

        (
            precompile_token_paddings,
            precompile_bs_paddings,
            precompile_cache_loc_paddings,
        ) = target_worker.get_precompile_paddings()
        self.precompile_bs_paddings = precompile_bs_paddings
        self.precompile_cache_loc_paddings = precompile_cache_loc_paddings
        self.precompile_token_paddings = precompile_token_paddings

    def _share_embed_head(self, target_worker: ModelWorker):
        embed, head = target_worker.model_runner.model.get_embed_and_head()

        if self.speculative_algorithm.is_eagle3():
            if (
                hasattr(self.draft_model_runner.model, "load_lm_head_from_target")
                and self.draft_model_runner.model.load_lm_head_from_target
            ):
                self.draft_model_runner.model.set_embed_and_head(embed, head)
            else:
                self.draft_model_runner.model.set_embed(embed)

            if self.draft_model_runner.model.hot_token_ids is not None:
                self.hot_token_ids = device_array(
                    self.draft_model_runner.model.hot_token_ids,
                    sharding=(NamedSharding(self._worker.mesh, P())),
                )
        else:
            if self.hot_token_ids is not None:
                head = head.clone()
                self.hot_token_ids = device_array(
                    self.draft_model_runner.model.hot_token_ids,
                    sharding=(NamedSharding(self._worker.mesh, P())),
                )
                head.data = head.data[self.hot_token_ids]

            self.draft_model_runner.model.set_embed_and_head(embed, head)

    @property
    def draft_model_runner(self):
        return self._worker.get_model_runner()

    @property
    def mesh(self):
        return self._worker.mesh

    @property
    def model_config(self):
        return self._worker.model_config

    @property
    def compilation_manager(self):
        return self._worker.compilation_manager

    @property
    def max_req_len(self):
        return self._worker.max_req_len

    def get_max_padded_size(self):
        return self._worker.get_max_padded_size()

    # -- BaseDraftWorker interface --

    def draft(self, model_worker_batch: ModelWorkerBatch) -> None:
        self.padding_for_decode(model_worker_batch)
        score_list, token_list, parents_list = self.draft_forward(model_worker_batch)
        verified_seq_lens = model_worker_batch.seq_lens - 1
        bs = model_worker_batch.seq_lens.shape[0]

        if self.topk == 1:
            n = self.speculative_num_draft_tokens
            verified_id = model_worker_batch.spec_info_padded.verified_id
            if any(isinstance(x, jax.Array) for x in (verified_id, token_list, verified_seq_lens)):
                rep = NamedSharding(self.mesh, P())
                verified_id, token_list, verified_seq_lens = jax.device_put(
                    (verified_id, token_list, verified_seq_lens),
                    rep,
                )
                packed = build_chain_verify_inputs_device(
                    verified_id, token_list, verified_seq_lens, n, bs
                )
                packed = jax.device_put(packed, rep)
            else:
                packed_np = build_chain_verify_inputs(
                    np.asarray(verified_id, dtype=np.int32),
                    np.asarray(token_list, dtype=np.int32),
                    np.asarray(verified_seq_lens, dtype=np.int32),
                    n,
                    bs,
                )
                # One allgather instead of five: pack into a single (5, bs*n)
                # buffer, device_put once, then slice on device.
                packed = (
                    jax.device_put(packed_np, NamedSharding(self.mesh, P()))
                    if self.mesh is not None
                    else packed_np
                )
            draft_tokens = packed[0]
            position = packed[1]
            retrive_index = packed[2].reshape(bs, n)
            retrive_next_token = packed[3].reshape(bs, n)
            retrive_next_sibling = packed[4].reshape(bs, n)
            tree_mask = None
        else:
            max_seq_len = int(np.max(verified_seq_lens)) if verified_seq_lens.size > 0 else 1
            max_context_len = self._pick_context_len(max_seq_len)
            (
                tree_mask,
                position,
                retrive_index,
                retrive_next_token,
                retrive_next_sibling,
                draft_tokens,
            ) = build_tree_kernel_efficient(
                model_worker_batch.spec_info_padded.verified_id,
                score_list,
                token_list,
                parents_list,
                verified_seq_lens,
                np.sum(verified_seq_lens),
                self.topk,
                self.speculative_num_draft_tokens,
                max_context_len,
                bs,
                model_worker_batch.speculative_num_steps,
                self.mesh,
            )

        model_worker_batch.spec_info_padded = EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            seq_lens_sum=model_worker_batch.seq_lens_sum,
            seq_lens_cpu=model_worker_batch.seq_lens,
        )

    def draft_extend_for_prefill(
        self,
        model_worker_batch: ModelWorkerBatch,
        hidden_states: jax.Array,
        next_token_ids: jax.Array,
    ) -> None:
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
        model_worker_batch.spec_info_padded.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST

        padded_bs = int(model_worker_batch.seq_lens.shape[0])
        if verified_id_np.shape[0] < padded_bs:
            model_worker_batch.spec_info_padded.verified_id = np.pad(
                verified_id_np, ((0, padded_bs - verified_id_np.shape[0]),)
            )

        forward_batch = ForwardBatch.init_new(model_worker_batch, self.draft_model_runner)
        forward_batch.return_logprob = False

        forward_metadata = self.draft_model_runner.attn_backend.get_eagle_forward_metadata(
            model_worker_batch
        )

        self.draft_model_runner.attn_backend.forward_metadata = forward_metadata
        forward_batch.forward_mode = ForwardMode.EXTEND

        logits_output, _, _ = self.draft_model_runner.forward(
            forward_batch,
            logits_metadata=LogitsMetadata.from_model_worker_batch(model_worker_batch, self.mesh),
        )
        # Restore real_bs so split_spec_info_per_rank cuts on real_bs_per_dp.
        model_worker_batch.spec_info_padded.verified_id = verified_id_np
        rep_logits, rep_hidden = replicate_to_mesh(
            self.mesh, logits_output.next_token_logits, logits_output.hidden_states
        )
        if len(rep_hidden.shape) == 1:
            rep_hidden = jnp.expand_dims(rep_hidden, axis=0)
        logits_output.next_token_logits = rep_logits
        logits_output.hidden_states = rep_hidden
        assert isinstance(forward_batch.spec_info, EagleDraftInput)
        forward_batch.spec_info.allocate_lens = np.asarray(model_worker_batch.seq_lens)[sel]

        self.capture_for_decode(logits_output, forward_batch.spec_info, sel=sel)

    def draft_extend_for_decode(
        self, model_worker_batch: ModelWorkerBatch, batch_output: GenerationBatchResult
    ) -> None:
        if batch_output.next_draft_input.verified_id.shape[0] <= 0:
            return
        draft_input = EagleDraftInput(
            hidden_states=batch_output.logits_output.hidden_states,
            allocate_lens=batch_output.next_draft_input.allocate_lens,
        )
        model_worker_batch, logits_metadata = draft_input.prepare_for_extend_after_verify(
            model_worker_batch,
            self.draft_model_runner,
            batch_output,
            self.speculative_num_draft_tokens,
        )

        forward_batch = ForwardBatch.init_new(model_worker_batch, self.draft_model_runner)
        if forward_batch.input_ids.shape[0] <= 0:
            return
        draft_logits_output, _, _ = self.draft_model_runner.forward(
            forward_batch,
            logits_metadata=logits_metadata,
        )
        sel = np.asarray(model_worker_batch.logits_indices_selector)
        if hasattr(batch_output.accept_lens, "copy_to_host_async"):
            batch_output.accept_lens.copy_to_host_async()
        accept_host = np.asarray(batch_output.accept_lens)
        select_index = sel * (self.speculative_num_steps + 1) + accept_host[sel] - 1
        rep_logits, rep_hidden = replicate_to_mesh(
            self.mesh, draft_logits_output.next_token_logits, draft_logits_output.hidden_states
        )
        # topk_probs_from_logits runs as @jax.jit on bucket-shaped rep_logits;
        # keep this on device with stable shape.
        topk_p, topk_index = topk_probs_from_logits(rep_logits, self.topk)
        # Gather to real_bs on host to avoid variable-shape device gathers.
        jax.copy_to_host_async(topk_p)
        jax.copy_to_host_async(topk_index)
        jax.copy_to_host_async(rep_hidden)
        verified_id_arr = batch_output.next_draft_input.verified_id
        if hasattr(verified_id_arr, "copy_to_host_async"):
            jax.copy_to_host_async(verified_id_arr)
        topk_p = np.asarray(topk_p)[sel]
        topk_index = np.asarray(topk_index)[sel]
        hidden = np.asarray(rep_hidden)[select_index]
        verified_id = np.asarray(verified_id_arr)[select_index]

        batch_output.next_draft_input.hidden_states = hidden
        batch_output.next_draft_input.topk_p = topk_p
        batch_output.next_draft_input.topk_index = topk_index
        batch_output.next_draft_input.verified_id = verified_id
        batch_output.next_draft_input.allocate_lens = batch_output.next_draft_input.allocate_lens[
            : model_worker_batch.real_bs
        ]
        batch_output.accept_lens = accept_host

    # -- Internal draft helpers --

    def capture_for_decode(
        self, logits_output: LogitsProcessorOutput, draft_input: EagleDraftInput, sel=None
    ):
        topk_p, topk_index = topk_probs_from_logits(logits_output.next_token_logits, self.topk)
        hidden = replicate_to_mesh(self.mesh, logits_output.hidden_states)
        if sel is not None:
            jax.copy_to_host_async(topk_p)
            jax.copy_to_host_async(topk_index)
            jax.copy_to_host_async(hidden)
            topk_p = np.asarray(topk_p)[sel]
            topk_index = np.asarray(topk_index)[sel]
            hidden = np.asarray(hidden)[sel]
        draft_input.topk_p = topk_p
        draft_input.topk_index = topk_index
        draft_input.hidden_states = hidden

    def padding_for_decode(self, model_worker_batch: ModelWorkerBatch):
        # At dp>1 the incoming mwb is already DP-padded to total_bs (== a bucket
        # value, see _get_spec_decode_mwb_dp); use the larger of real_bs and the
        # incoming seq_lens length so we don't shrink below the DP layout.
        _, padding_bs_index = self.get_padding_bs(
            max(model_worker_batch.real_bs, len(model_worker_batch.seq_lens))
        )
        self.copy_model_worker_batch_to_cpu(model_worker_batch)
        model_worker_batch.spec_info_padded.prepare_for_draft_decode(
            model_worker_batch, self.topk, self.speculative_num_steps
        )
        model_worker_batch.seq_lens = model_worker_batch.seq_lens
        seq_lens_cpu = model_worker_batch.seq_lens
        page_size = self.page_size
        req_to_token_pool, _ = self.target_worker_ref.get_memory_pool()
        token_indices_with_all_reqs = req_to_token_pool.req_to_token[
            model_worker_batch.req_pool_indices
        ]
        spec_info = model_worker_batch.spec_info_padded
        assert isinstance(spec_info, EagleDraftInput)
        # At dp>1 spec_info arrays arrive at (real_bs,) but seq_lens_cpu is
        # (total_bs,); pad allocate_lens up front so valid_mask indexing works
        # (the per-field bs-padding loop below would do this anyway, just later).
        if len(spec_info.allocate_lens) < len(seq_lens_cpu):
            spec_info.allocate_lens = np.pad(
                spec_info.allocate_lens, (0, len(seq_lens_cpu) - len(spec_info.allocate_lens))
            )
        # DP-segmented cache_loc: rank r's slots occupy
        # [r*per_dp_cache_len : (r+1)*per_dp_cache_len) so the P("data") shard
        # gives each rank its own page_indices (not the contiguous-then-padding
        # layout where rank>0's shard lands in the padding region).
        total_cache_loc_size = self.precompile_cache_loc_paddings[padding_bs_index]
        dp_size = model_worker_batch.dp_size
        per_dp_bs = model_worker_batch.per_dp_bs_size if dp_size > 1 else len(seq_lens_cpu)
        assert total_cache_loc_size % dp_size == 0
        per_dp_cache_len = total_cache_loc_size // dp_size
        cache_loc_cpu = np.zeros(total_cache_loc_size, dtype=np.int32)
        valid_mask = seq_lens_cpu > 0
        if np.any(valid_mask):
            valid_indices = np.where(valid_mask)[0]
            valid_allocate_lens = spec_info.allocate_lens[valid_mask]
            aligned_lengths = ((valid_allocate_lens + page_size - 1) // page_size) * page_size
            intra_rank_off = np.zeros(dp_size, dtype=np.int64)
            for seq_idx, allocate_len, aligned_len in zip(
                valid_indices, valid_allocate_lens, aligned_lengths
            ):
                r = int(seq_idx) // per_dp_bs
                base = r * per_dp_cache_len + intra_rank_off[r]
                assert (
                    base + aligned_len <= (r + 1) * per_dp_cache_len
                ), f"rank {r} cache_loc overflow: {intra_rank_off[r]+aligned_len} > {per_dp_cache_len}"
                cache_loc_cpu[base : base + allocate_len] = token_indices_with_all_reqs[
                    seq_idx, :allocate_len
                ]
                intra_rank_off[r] += aligned_len

        model_worker_batch.cache_loc = cache_loc_cpu
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST

        topk_index = spec_info.topk_index
        if self.hot_token_ids is not None:
            model_worker_batch.spec_info_padded.topk_index = self.hot_token_ids[topk_index]
        if self.topk > 1:
            self.draft_model_runner.attn_backend.forward_metadata.custom_mask = (
                build_tree_mask_for_draft_decode(
                    model_worker_batch.seq_lens,
                    topk=topk_index.shape[1],
                    speculative_step_id=0,
                    parents_list=None,
                )
            )
        bs = self.precompile_bs_paddings[padding_bs_index]
        dp_size = model_worker_batch.dp_size
        per_dp_padded = bs // dp_size

        def _dp_segment_pad(arr, target_bs):
            """DP-segmented pad: pad each rank's section separately to per_dp_padded.

            Input arr shape (curr_bs, ...) with curr_bs = per_dp_curr * dp_size.
            Returns (target_bs, ...) with each rank's slice padded at the end.
            End-padding the whole array would let shard_map(P("data")) hand a
            following rank's data to a prior rank.
            """
            if arr is None or arr.shape[0] >= target_bs:
                return arr
            per_dp_curr = max(arr.shape[0] // dp_size, 1)
            if dp_size <= 1 or arr.shape[0] % dp_size != 0:
                # Fallback to end-pad if layout isn't DP-divisible (dp=1 path).
                pad_widths = [(0, target_bs - arr.shape[0])] + [(0, 0)] * (arr.ndim - 1)
                return np.pad(arr, pad_widths)
            reshaped = arr.reshape((dp_size, per_dp_curr) + arr.shape[1:])
            pad_widths = [(0, 0), (0, per_dp_padded - per_dp_curr)] + [(0, 0)] * (arr.ndim - 1)
            padded = np.pad(reshaped, pad_widths)
            return padded.reshape((target_bs,) + arr.shape[1:])

        spec_info_padded = model_worker_batch.spec_info_padded
        spec_info_padded.verified_id = _dp_segment_pad(spec_info_padded.verified_id, bs)
        spec_info_padded.topk_p = _dp_segment_pad(spec_info_padded.topk_p, bs)
        if bs - model_worker_batch.seq_lens.shape[0] > 0:
            model_worker_batch.seq_lens = _dp_segment_pad(model_worker_batch.seq_lens, bs)
            if spec_info_padded.allocate_lens is not None:
                spec_info_padded.allocate_lens = _dp_segment_pad(spec_info_padded.allocate_lens, bs)
        spec_info_padded.topk_index = _dp_segment_pad(spec_info_padded.topk_index, bs)
        spec_info_padded.hidden_states = _dp_segment_pad(spec_info_padded.hidden_states, bs)
        model_worker_batch.speculative_eagle_topk = self.topk
        model_worker_batch.speculative_num_steps = self.speculative_num_steps
        model_worker_batch.speculative_num_draft_tokens = self.speculative_num_draft_tokens
        model_worker_batch.input_ids = np.empty(bs * self.topk, np.int32)
        model_worker_batch.positions = np.empty(bs * self.topk, np.int32)

    def draft_forward(self, model_worker_batch: ModelWorkerBatch):
        topk_p, topk_index, hidden_states = (
            model_worker_batch.spec_info_padded.topk_p,
            model_worker_batch.spec_info_padded.topk_index,
            model_worker_batch.spec_info_padded.hidden_states,
        )
        bs = model_worker_batch.seq_lens.shape[0]
        step_min_1 = self.speculative_num_steps - 1
        score_list: jax.Array = jnp.empty((bs, 1 + step_min_1 * self.topk, self.topk))
        token_list: jax.Array = jnp.empty(
            (bs, self.topk + step_min_1 * self.topk * self.topk), dtype=jnp.int32
        )
        parents_list: jax.Array = jnp.empty((bs, self.topk + 1 + step_min_1 * self.topk))
        scores = None
        positions_base = device_array(
            np.repeat(model_worker_batch.seq_lens, self.topk),
            sharding=(NamedSharding(self.mesh, P())),
        )
        logits_metadata = None
        metadata_per_step = self.draft_model_runner.attn_backend.get_eagle_multi_step_metadata(
            model_worker_batch,
        )
        assert isinstance(metadata_per_step, list)
        logits_metadata = LogitsMetadata.from_model_worker_batch(
            model_worker_batch, self.draft_model_runner.mesh
        )
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
            self.draft_model_runner.attn_backend.forward_metadata = metadata_per_step[i]

            forward_batch.bid = model_worker_batch.bid
            logits_output, _, _ = self.draft_model_runner.forward(
                forward_batch,
                logits_metadata=logits_metadata,
            )

            topk_p, topk_index = topk_probs_from_logits(logits_output.next_token_logits, self.topk)

            if self.hot_token_ids is not None:
                topk_index = self.hot_token_ids[topk_index]
            hidden_states = replicate_to_mesh(self.mesh, logits_output.hidden_states)

        return score_list, token_list, parents_list

    def _pick_context_len(self, max_seq_len: int) -> int:
        if self.precompile_token_paddings:
            return self.precompile_token_paddings[-1]
        max_seq_len = max(int(max_seq_len), 1)
        return 1 << (max_seq_len - 1).bit_length()

    def copy_model_worker_batch_to_cpu(self, model_worker_batch: ModelWorkerBatch):
        mwb = model_worker_batch
        fields = [
            "input_ids",
            "seq_lens",
            "out_cache_loc",
            "positions",
            "req_pool_indices",
            "cache_loc",
        ]
        optional = ["extend_prefix_lens", "extend_seq_lens"]

        for name in fields:
            arr = getattr(mwb, name)
            if hasattr(arr, "copy_to_host_async"):
                arr.copy_to_host_async()
        for name in optional:
            arr = getattr(mwb, name)
            if arr is not None and hasattr(arr, "copy_to_host_async"):
                arr.copy_to_host_async()

        for name in fields:
            arr = getattr(mwb, name)
            setattr(mwb, name, np.asarray(arr))
        for name in optional:
            arr = getattr(mwb, name)
            if arr is not None:
                setattr(mwb, name, np.asarray(arr))

    def get_padding_bs(self, real_bs: int) -> int:
        self.precompile_bs_paddings.sort()
        select_bs_index = -1
        bs_padding_size = 0
        for i, size in enumerate(self.precompile_bs_paddings):
            if size >= real_bs:
                bs_padding_size = size - real_bs
                select_bs_index = i
                break
        if select_bs_index < 0:
            raise RuntimeError("did not get comperate padding bs, it should not happened")
        return bs_padding_size, select_bs_index


# ---------------------------------------------------------------------------
# Module-level JIT helpers (used exclusively by EagleDraftWorker)
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnames=["topk"])
def topk_probs_from_logits(
    logits: jax.Array, topk: int, axis: int = -1
) -> tuple[jax.Array, jax.Array]:
    """Return top-k probabilities without materializing the full softmax tensor."""
    working_logits = jnp.moveaxis(logits, axis, -1) if axis != -1 else logits
    # TODO(#1053 Phase 2): replace this all-gather with a sharded top-k +
    # logsumexp over the vocab axis for DP/TP scalability.
    sh = jax.typeof(working_logits).sharding
    if isinstance(sh, NamedSharding):
        working_logits = jax.sharding.reshard(working_logits, NamedSharding(sh.mesh, P()))
    topk_logits, topk_index = jax.lax.top_k(working_logits, topk)
    logsumexp = jax.nn.logsumexp(working_logits, axis=-1, keepdims=True)
    topk_probs = jnp.exp(topk_logits - logsumexp)

    if axis != -1:
        topk_probs = jnp.moveaxis(topk_probs, -1, axis)
        topk_index = jnp.moveaxis(topk_index, -1, axis)

    return topk_probs, topk_index


def fast_topk(values, topk, axis=-1):
    working_values = jnp.moveaxis(values, axis, -1) if axis != -1 else values
    result_vals, result_indices = jax.lax.top_k(working_values, topk)

    if axis != -1:
        result_vals = jnp.moveaxis(result_vals, -1, axis)
        result_indices = jnp.moveaxis(result_indices, -1, axis)

    return result_vals, result_indices


@functools.partial(jax.jit, static_argnames=["i", "topk"])
def update_eagle_lists(
    i: int,
    score_list: jax.Array,
    token_list: jax.Array,
    parents_list: jax.Array,
    tree_info: tuple[jax.Array, jax.Array, jax.Array],
    topk: int,
):
    bs = score_list.shape[0]
    scores_update, tokens_update, parents_update = tree_info
    if i == 0:
        score_list = score_list.at[:bs, :1, :].set(scores_update[:bs])
        token_list = token_list.at[:bs, :topk].set(tokens_update[:bs])
        parents_list = parents_list.at[:bs, : topk + 1].set(parents_update[:bs])
    else:
        score_start = 1 + (i - 1) * topk
        token_start = topk + (i - 1) * topk * topk
        parent_start = topk + 1 + (i - 1) * topk

        score_list = score_list.at[:bs, score_start : score_start + topk, :].set(scores_update[:bs])
        token_list = token_list.at[:bs, token_start : token_start + topk * topk].set(
            tokens_update[:bs]
        )
        parents_list = parents_list.at[:bs, parent_start : parent_start + topk].set(
            parents_update[:bs]
        )
    return score_list, token_list, parents_list


# FIXME(pc) this should be jitted or convert as np.ndarray
# @functools.partial(jax.jit, static_argnames=["i"])
def update_forward_batch_info(
    forward_batch: ForwardBatch,
    i: int,
    input_ids: jax.Array,
    hidden_states: jax.Array,
    positions_base: jax.Array,
) -> ForwardBatch:
    forward_batch.input_ids = input_ids
    # FIXME(pc) hiddenstate will become NAN when forward path is very long, we still have no reason for this
    forward_batch.spec_info.hidden_states = hidden_states
    forward_batch.positions = positions_base + i
    return forward_batch


def select_top_k_tokens(
    i: int,
    topk_p: jax.Array,
    topk_index: jax.Array,
    hidden_states: jax.Array,
    scores: jax.Array,
    topk: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    if i == 0:
        return select_top_k_tokens_step_0(topk_p, topk_index, hidden_states, scores, topk)
    else:
        return select_top_k_tokens_step_greater_0(
            jnp.asarray(i), topk_p, topk_index, hidden_states, scores, topk
        )


@functools.partial(jax.jit, static_argnames=["topk"])
def select_top_k_tokens_step_0(
    topk_p: jax.Array,
    topk_index: jax.Array,
    hidden_states: jax.Array,
    scores: jax.Array,
    topk: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    input_ids = topk_index.flatten()
    hidden_states = jnp.repeat(hidden_states, topk, axis=0)
    scores = topk_p
    tree_info = (
        jnp.expand_dims(topk_p, axis=1),
        topk_index,
        jnp.tile(
            jnp.expand_dims(jnp.arange(-1, topk, dtype=jnp.float32), axis=0),
            (topk_p.shape[0], 1),
        ),
    )
    return input_ids, hidden_states, scores, tree_info


@functools.partial(jax.jit, static_argnames=["topk"])
def select_top_k_tokens_step_greater_0(
    i: jax.Array,
    topk_p: jax.Array,
    topk_index: jax.Array,
    hidden_states: jax.Array,
    scores: jax.Array,
    topk: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    # scores carries the prior step's top-k probs; topk_p comes from the next
    # draft forward. Target and draft logits can be different dtypes (e.g. MiMo
    # V2 Flash), so promote before lax.mul which requires matching dtypes.
    scores = scores.astype(topk_p.dtype)
    expand_scores = jax.lax.mul(jnp.expand_dims(scores, axis=2), topk_p.reshape(-1, topk, topk))
    topk_cs_p, topk_cs_index = fast_topk(
        expand_scores.reshape(expand_scores.shape[0], -1), topk, axis=-1
    )
    scores = topk_cs_p
    topk_index = topk_index.reshape(-1, topk**2)
    input_ids = jnp.take_along_axis(topk_index, topk_cs_index, axis=1).flatten()
    if hidden_states.shape[0] > 0:
        selected_input_index = topk_cs_index.flatten() // topk + jnp.repeat(
            jnp.arange(0, hidden_states.shape[0], topk), topk
        )
        hidden_states = hidden_states[selected_input_index, :]
    tree_info = (
        expand_scores,
        topk_index,
        topk_cs_index + (topk**2 * (i - 1) + topk),
    )
    return input_ids, hidden_states, scores, tree_info
