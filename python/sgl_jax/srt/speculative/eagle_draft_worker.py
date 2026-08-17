import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.managers.scheduler import GenerationBatchResult
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sgl_jax.srt.speculative.base_worker import BaseDraftWorker, replicate_to_mesh
from sgl_jax.srt.speculative.eagle_info import EagleDraftInput
from sgl_jax.srt.speculative.overlap_utils import uses_host_eagle_state
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm
from sgl_jax.srt.utils.jax_utils import device_array


class EagleDraftWorker(BaseDraftWorker):
    """Topk=1 fused recurrent EAGLE/EAGLE3/single-layer NEXTN worker.

    Holds a ``ModelWorker`` (the draft model runner) via composition and
    prepares the prefix seed and fixed-width recurrent draft chain.
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
        if not self.speculative_algorithm.is_eagle() or self.topk != 1:
            raise ValueError("EagleDraftWorker only supports EAGLE/EAGLE3/NEXTN with topk=1.")
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
            # Classic EAGLE uses the target vocabulary directly and shares the
            # complete LM head; it has no draft-to-target vocabulary remap.
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
        raise RuntimeError(
            "EAGLE/EAGLE3 draft() is unavailable; use the fused verify/bootstrap path."
        )

    def prepare_for_fused_verify(self, model_worker_batch: ModelWorkerBatch):
        """Prepare a raw topk=1 token chain for fused verify."""
        topk_index = model_worker_batch.spec_info_padded.topk_index
        has_raw_recurrent_chain = self._has_precomputed_recurrent_chain(topk_index)
        self.padding_for_decode(
            model_worker_batch,
        )
        if not has_raw_recurrent_chain:
            from sgl_jax.srt.speculative.draft_extend_fused import bootstrap_eagle_chain

            model_worker_batch.spec_info_padded.topk_index = bootstrap_eagle_chain(
                self,
                model_worker_batch,
            )
        return self.hot_token_ids

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

        from sgl_jax.srt.speculative.draft_extend_fused import (
            eagle_prefill_draft_extend,
        )

        logits_output, forward_batch = eagle_prefill_draft_extend(
            self,
            model_worker_batch,
        )
        # Restore real_bs so split_spec_info_per_rank cuts on real_bs_per_dp.
        model_worker_batch.spec_info_padded.verified_id = verified_id_np
        assert isinstance(forward_batch.spec_info, EagleDraftInput)
        forward_batch.spec_info.allocate_lens = np.asarray(model_worker_batch.seq_lens)[sel]

        self.capture_for_decode(logits_output, forward_batch.spec_info, sel=sel)

    def draft_extend_for_decode(
        self, model_worker_batch: ModelWorkerBatch, batch_output: GenerationBatchResult
    ) -> None:
        raise RuntimeError(
            "EAGLE/EAGLE3 draft_extend_for_decode() is unavailable; "
            "use the fused recurrent path."
        )

    # -- Internal draft helpers --

    def capture_for_decode(
        self, logits_output: LogitsProcessorOutput, draft_input: EagleDraftInput, sel=None
    ):
        topk_index = jnp.argmax(logits_output.next_token_logits, axis=-1).astype(jnp.int32)[:, None]
        topk_index = replicate_to_mesh(self.mesh, topk_index)
        hidden = replicate_to_mesh(self.mesh, logits_output.hidden_states)
        if len(hidden.shape) == 1:
            hidden = jnp.expand_dims(hidden, axis=0)
        if sel is not None:
            jax.copy_to_host_async(topk_index)
            jax.copy_to_host_async(hidden)
            topk_index = np.asarray(topk_index)[sel]
            hidden = np.asarray(hidden)[sel]
        draft_input.topk_index = topk_index
        draft_input.hidden_states = hidden

    def padding_for_decode(
        self,
        model_worker_batch: ModelWorkerBatch,
    ):
        # At dp>1 the incoming mwb is already DP-padded to total_bs (== a bucket
        # value, see _get_spec_decode_mwb_dp); use the larger of real_bs and the
        # incoming seq_lens length so we don't shrink below the DP layout.
        padding_bs_index = self._get_padding_bs_index(
            max(model_worker_batch.real_bs, len(model_worker_batch.seq_lens))
        )
        self.copy_model_worker_batch_to_cpu(model_worker_batch)
        model_worker_batch.spec_info_padded.prepare_for_draft_decode(model_worker_batch)
        seq_lens_cpu = model_worker_batch.seq_lens
        page_size = self.page_size
        req_to_token_pool, _ = self.target_worker_ref.get_memory_pool()
        uses_host_state = uses_host_eagle_state(
            not self.server_args.disable_overlap_schedule, self.speculative_algorithm
        )
        if uses_host_state:
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
        cache_loc_cpu = self._get_decode_cache_loc_buffer(total_cache_loc_size)
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
                ), f"rank {r} cache_loc overflow: {intra_rank_off[r] + aligned_len} > {per_dp_cache_len}"
                if uses_host_state:
                    cache_loc_cpu[base : base + allocate_len] = token_indices_with_all_reqs[
                        seq_idx, :allocate_len
                    ]
                else:
                    page_offsets = np.arange(0, aligned_len, page_size)
                    cache_loc_cpu[base + page_offsets] = req_to_token_pool.req_to_token[
                        model_worker_batch.req_pool_indices[seq_idx], page_offsets
                    ]
                intra_rank_off[r] += aligned_len

        model_worker_batch.cache_loc = cache_loc_cpu
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST

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

    def _has_precomputed_recurrent_chain(self, topk_index) -> bool:
        return (
            self.speculative_algorithm.is_eagle()
            and self.topk == 1
            and topk_index is not None
            and len(topk_index.shape) == 2
            and topk_index.shape[1] == self.speculative_num_steps
        )

    def _get_decode_cache_loc_buffer(self, total_cache_loc_size: int):
        if not self.server_args.disable_overlap_schedule:
            return np.zeros(total_cache_loc_size, dtype=np.int32)
        cache_loc_buffers = getattr(self, "_decode_cache_loc_buffers", None)
        if cache_loc_buffers is None:
            cache_loc_buffers = self._decode_cache_loc_buffers = {}
        cache_loc_cpu = cache_loc_buffers.get(total_cache_loc_size)
        if cache_loc_cpu is None:
            cache_loc_cpu = np.zeros(total_cache_loc_size, dtype=np.int32)
            cache_loc_buffers[total_cache_loc_size] = cache_loc_cpu
        return cache_loc_cpu

    def copy_model_worker_batch_to_cpu(self, model_worker_batch: ModelWorkerBatch):
        mwb = model_worker_batch
        if self.server_args.disable_overlap_schedule:
            # padding_for_decode only consumes these two host arrays. The other
            # fields are replaced before the next forward, so copying them here
            # duplicates device-to-host traffic in the no-overlap path.
            fields = ["seq_lens", "req_pool_indices"]
            optional = []
        else:
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

    def _get_padding_bs_index(self, real_bs: int) -> int:
        self.precompile_bs_paddings.sort()
        select_bs_index = -1
        for i, size in enumerate(self.precompile_bs_paddings):
            if size >= real_bs:
                select_bs_index = i
                break
        if select_bs_index < 0:
            raise RuntimeError("did not get comperate padding bs, it should not happened")
        return select_bs_index
