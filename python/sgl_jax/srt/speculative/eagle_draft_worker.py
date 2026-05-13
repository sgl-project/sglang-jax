import logging
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from tqdm import tqdm

from sgl_jax.srt.layers.logits_processor import LogitsMetadata
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata
from sgl_jax.srt.speculative.base_spec_worker import BaseDraftWorker
from sgl_jax.srt.speculative.eagle_util import (
    EagleDraftInput,
    EagleVerifyInput,
    build_tree_kernel_efficient,
    build_tree_mask_for_draft_decode,
)
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm
from sgl_jax.srt.speculative.spec_utils import (
    select_top_k_tokens,
    topk_probs_from_logits,
    update_eagle_lists,
    update_forward_batch_info,
)
from sgl_jax.srt.utils.jax_utils import device_array

if TYPE_CHECKING:
    from sgl_jax.srt.managers.scheduler import GenerationBatchResult

logger = logging.getLogger(__name__)


def _take_with_optional_out_sharding(array: jax.Array, index: jax.Array, trailing_slice=False):
    out_sharding = getattr(array, "sharding", None)
    if not isinstance(out_sharding, (NamedSharding, P)):
        return array[index, :] if trailing_slice else array[index]
    if trailing_slice:
        return array.at[index, :].get(out_sharding=out_sharding)
    return array.at[index].get(out_sharding=out_sharding)


def _pad_1d_array(value, target_size: int, pad_value: int = -1) -> np.ndarray:
    value = np.asarray(value)
    pad_size = target_size - value.shape[0]
    if pad_size <= 0:
        return value
    return np.pad(value, (0, pad_size), constant_values=pad_value)


class EagleDraftWorker(ModelWorker, BaseDraftWorker):
    def __init__(self, server_args, target_worker: ModelWorker):
        self.server_args = server_args
        self._target_worker = target_worker
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.req_to_token_pool, self.token_to_kv_pool_allocator = target_worker.get_memory_pool()
        self.hot_token_ids = None

        ModelWorker.__init__(
            self,
            server_args,
            target_worker.mesh,
            req_to_token_pool=self.req_to_token_pool,
            is_draft_worker=True,
        )
        EagleDraftInput.ALLOC_LEN_PER_DECODE = max(
            self.speculative_num_steps * self.topk, self.speculative_num_draft_tokens
        )
        self._init_embed_and_head()
        self.model_runner.initialize_jit()
        (
            precompile_token_paddings,
            precompile_bs_paddings,
            precompile_cache_loc_paddings,
        ) = self._target_worker.get_precompile_paddings()
        self.precompile_bs_paddings = precompile_bs_paddings
        self.precompile_cache_loc_paddings = precompile_cache_loc_paddings
        self.precompile_token_paddings = precompile_token_paddings

    @property
    def target_worker(self):
        return self._target_worker

    @property
    def draft_model_runner(self):
        return self.model_runner

    def _init_embed_and_head(self):
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
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
                    self.draft_model_runner.model.hot_token_ids.value,
                    sharding=(
                        NamedSharding(self.model_runner.mesh, P())
                        if jax.process_count() == 1
                        else None
                    ),
                )
        else:
            if self.draft_model_runner.model.hot_token_ids is not None:
                head = head.clone()
                self.hot_token_ids = device_array(
                    self.draft_model_runner.model.hot_token_ids.value,
                    sharding=(
                        NamedSharding(self.model_runner.mesh, P())
                        if jax.process_count() == 1
                        else None
                    ),
                )
                head.data = head.data[self.hot_token_ids]
            self.draft_model_runner.model.set_embed_and_head(embed, head)

    def generate_model_worker_batch(self, *args, **kwargs):
        return self.compilation_manager.generate_model_worker_batch(*args, **kwargs)

    def _remap_hot_token_ids(self, token_ids: jax.Array) -> jax.Array:
        out_sharding = NamedSharding(
            self.model_runner.mesh,
            P("data", None) if token_ids.ndim == 2 else P("data"),
        )
        return self.hot_token_ids.at[token_ids].get(out_sharding=out_sharding)

    def copy_model_worker_batch_to_cpu(self, model_worker_batch: ModelWorkerBatch):
        model_worker_batch.input_ids = np.array(
            jax.device_get(model_worker_batch.input_ids), dtype=model_worker_batch.input_ids.dtype
        )
        model_worker_batch.seq_lens = np.array(
            jax.device_get(model_worker_batch.seq_lens), dtype=model_worker_batch.seq_lens.dtype
        )
        model_worker_batch.out_cache_loc = np.array(
            jax.device_get(model_worker_batch.out_cache_loc),
            dtype=model_worker_batch.out_cache_loc.dtype,
        )
        model_worker_batch.positions = np.array(
            jax.device_get(model_worker_batch.positions), dtype=model_worker_batch.positions.dtype
        )
        model_worker_batch.req_pool_indices = np.array(
            jax.device_get(model_worker_batch.req_pool_indices),
            dtype=model_worker_batch.req_pool_indices.dtype,
        )
        model_worker_batch.cache_loc = np.array(
            jax.device_get(model_worker_batch.cache_loc), dtype=model_worker_batch.cache_loc.dtype
        )
        model_worker_batch.extend_prefix_lens = (
            np.array(
                jax.device_get(model_worker_batch.extend_prefix_lens),
                dtype=model_worker_batch.extend_prefix_lens.dtype,
            )
            if model_worker_batch.extend_prefix_lens is not None
            else None
        )
        model_worker_batch.extend_seq_lens = (
            np.array(
                jax.device_get(model_worker_batch.extend_seq_lens),
                dtype=model_worker_batch.extend_seq_lens.dtype,
            )
            if model_worker_batch.extend_seq_lens is not None
            else None
        )

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

    def padding_for_decode(self, model_worker_batch: ModelWorkerBatch):
        _, padding_bs_index = self.get_padding_bs(model_worker_batch.real_bs)
        self.copy_model_worker_batch_to_cpu(model_worker_batch)
        model_worker_batch.spec_info.prepare_for_draft_decode(
            model_worker_batch, self.topk, self.speculative_num_steps
        )
        # get unpadded seq_lens
        model_worker_batch.seq_lens = model_worker_batch.seq_lens
        seq_lens_cpu = model_worker_batch.seq_lens
        page_size = self.page_size
        token_indices_with_all_reqs = self.req_to_token_pool.req_to_token[
            model_worker_batch.req_pool_indices
        ]
        spec_info = model_worker_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        cache_loc_flat = np.array([], dtype=np.int32)
        if len(seq_lens_cpu) > 0:
            # Filter out empty sequences
            valid_mask = seq_lens_cpu > 0
            if np.any(valid_mask):
                valid_indices = np.where(valid_mask)[0]
                valid_allocate_lens = spec_info.allocate_lens[valid_mask]
                # Calculate aligned lengths for all valid sequences at once
                aligned_lengths = ((valid_allocate_lens + page_size - 1) // page_size) * page_size
                total_aligned_length = np.sum(aligned_lengths)
                # Pre-allocate the result array
                cache_loc_flat = np.zeros(total_aligned_length, dtype=np.int32)
                # Fill the array efficiently
                offset = 0
                for i, (seq_idx, allocate_len, aligned_len) in enumerate(
                    zip(valid_indices, valid_allocate_lens, aligned_lengths)
                ):
                    # Copy the actual data
                    cache_loc_flat[offset : offset + allocate_len] = token_indices_with_all_reqs[
                        seq_idx, :allocate_len
                    ]
                    # Padding is already zero from initialization
                    offset += aligned_len
        total_cache_loc_size = self.precompile_cache_loc_paddings[padding_bs_index]
        assert total_cache_loc_size >= len(cache_loc_flat)
        cache_loc_cpu = np.empty(total_cache_loc_size, dtype=np.int32)
        if len(cache_loc_flat) > 0:
            cache_loc_cpu[: len(cache_loc_flat)] = cache_loc_flat
        # Initialize padding area to ensure multiprocess consistency
        if len(cache_loc_flat) < total_cache_loc_size:
            cache_loc_cpu[len(cache_loc_flat) :] = 0

        model_worker_batch.cache_loc = cache_loc_cpu
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST

        # out_cache_loc = model_worker_batch.out_cache_loc
        topk_index = model_worker_batch.spec_info.topk_index
        # if we need custom mask, we should create for all at once and update it within loop
        # we should optimize build_tree_mask_for_draft_decode to a kernel
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
        if bs - model_worker_batch.spec_info.verified_id.shape[0] > 0:
            model_worker_batch.spec_info.verified_id = np.pad(
                model_worker_batch.spec_info.verified_id,
                ((0, bs - model_worker_batch.spec_info.verified_id.shape[0]),),
            )
        if bs - model_worker_batch.spec_info.topk_p.shape[0] > 0:
            model_worker_batch.spec_info.topk_p = np.pad(
                model_worker_batch.spec_info.topk_p,
                (
                    (0, bs - model_worker_batch.spec_info.topk_p.shape[0]),
                    (0, 0),
                ),
            )
        if bs - model_worker_batch.seq_lens.shape[0] > 0:
            model_worker_batch.seq_lens = np.pad(
                model_worker_batch.seq_lens, ((0, bs - model_worker_batch.seq_lens.shape[0]),)
            )
            model_worker_batch.req_pool_indices = np.pad(
                model_worker_batch.req_pool_indices,
                ((0, bs - model_worker_batch.req_pool_indices.shape[0]),),
                constant_values=-1,
            )
            if model_worker_batch.spec_info.allocate_lens is not None:
                model_worker_batch.spec_info.allocate_lens = np.pad(
                    model_worker_batch.spec_info.allocate_lens,
                    ((0, bs - model_worker_batch.spec_info.allocate_lens.shape[0]),),
                )
        if bs - model_worker_batch.spec_info.topk_index.shape[0] > 0:
            model_worker_batch.spec_info.topk_index = np.pad(
                model_worker_batch.spec_info.topk_index,
                (
                    (0, bs - model_worker_batch.spec_info.topk_index.shape[0]),
                    (0, 0),
                ),
            )
        if bs - model_worker_batch.spec_info.hidden_states.shape[0] > 0:
            model_worker_batch.spec_info.hidden_states = np.pad(
                model_worker_batch.spec_info.hidden_states,
                (
                    (0, bs - model_worker_batch.spec_info.hidden_states.shape[0]),
                    (0, 0),
                ),
            )
        # Forward multiple steps
        model_worker_batch.speculative_eagle_topk = self.topk
        model_worker_batch.speculative_num_steps = self.speculative_num_steps
        model_worker_batch.speculative_num_draft_tokens = self.speculative_num_draft_tokens
        model_worker_batch.input_ids = np.empty(bs * self.topk, np.int32)
        model_worker_batch.positions = np.empty(bs * self.topk, np.int32)

    def draft(self, model_worker_batch: ModelWorkerBatch):
        self.padding_for_decode(model_worker_batch)
        score_list, token_list, parents_list = self.draft_forward(model_worker_batch)
        verified_seq_lens = model_worker_batch.seq_lens - 1
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
            model_worker_batch.spec_info.verified_id,
            score_list,
            token_list,
            parents_list,
            verified_seq_lens,
            np.sum(verified_seq_lens),
            self.topk,
            self.speculative_num_draft_tokens,
            max_context_len,
            model_worker_batch.seq_lens.shape[0],
            model_worker_batch.speculative_num_steps,
            self.mesh,
        )
        model_worker_batch.spec_info = EagleVerifyInput(
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
        return model_worker_batch.spec_info

    def pad_out_cache_loc_for_verify(self, model_worker_batch: ModelWorkerBatch) -> None:
        target_size = model_worker_batch.seq_lens.shape[0] * self.speculative_num_draft_tokens
        model_worker_batch.out_cache_loc = _pad_1d_array(
            model_worker_batch.out_cache_loc,
            target_size,
            -1,
        )

    def _pick_context_len(self, max_seq_len: int) -> int:
        max_seq_len = max(int(max_seq_len), 1)
        if self.precompile_token_paddings:
            for padding in self.precompile_token_paddings:
                if padding >= max_seq_len:
                    return padding
        return 1 << (max_seq_len - 1).bit_length()

    def draft_extend_for_prefill(
        self,
        model_worker_batch: ModelWorkerBatch,
        target_hidden_states: jax.Array,
        next_token_ids: jax.Array,
    ) -> EagleDraftInput:
        # FIXME(pc) move this all prepare to prepare_for_extend_after_target_prefill
        index_sharding = NamedSharding(self.model_runner.mesh, P("data"))
        real_indices = device_array(
            self._get_phase1_runtime_indices(model_worker_batch.real_bs),
            sharding=index_sharding,
        )
        padded_indices = device_array(
            np.arange(model_worker_batch.seq_lens.shape[0], dtype=np.int32),
            sharding=index_sharding,
        )
        model_worker_batch.spec_info = EagleDraftInput(
            hidden_states=target_hidden_states,
            verified_id=_take_with_optional_out_sharding(next_token_ids, padded_indices),
            num_tokens_per_batch=np.asarray(1, dtype=jnp.int32),
            num_tokens_for_logprob_per_batch=np.asarray(1, dtype=jnp.int32),
            allocate_lens=model_worker_batch.seq_lens,
        )
        model_worker_batch.return_hidden_states = False
        model_worker_batch.spec_info.prepare_for_extend_after_target_prefill(
            model_worker_batch=model_worker_batch
        )
        model_worker_batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.draft_model_runner)
        forward_batch.return_logprob = False

        # Set forward_metadata for draft_model_runner's attention backend
        forward_metadata = self.draft_model_runner.attn_backend.get_eagle_forward_metadata(
            model_worker_batch
        )

        self.draft_model_runner.attn_backend.forward_metadata = forward_metadata
        forward_batch.forward_mode = ForwardMode.EXTEND
        # last_idx = np.cumsum(model_worker_batch.extend_seq_lens, axis=0) - 1

        logits_output, _, _ = self.draft_model_runner.forward(
            forward_batch,
            logits_metadata=LogitsMetadata.from_model_worker_batch(model_worker_batch, self.mesh),
        )
        logits_output.next_token_logits = _take_with_optional_out_sharding(
            logits_output.next_token_logits, real_indices, trailing_slice=True
        )
        if len(logits_output.hidden_states.shape) == 1:
            logits_output.hidden_states = jnp.expand_dims(logits_output.hidden_states, axis=0)
        logits_output.hidden_states = _take_with_optional_out_sharding(
            logits_output.hidden_states, real_indices, trailing_slice=True
        )
        assert isinstance(forward_batch.spec_info, EagleDraftInput)
        forward_batch.spec_info.verified_id = _take_with_optional_out_sharding(
            forward_batch.spec_info.verified_id, real_indices
        )
        runtime_bs = real_indices.shape[0]
        forward_batch.spec_info.allocate_lens = model_worker_batch.seq_lens[:runtime_bs]

        self.capture_for_decode(logits_output, forward_batch.spec_info)
        self._trim_prefill_spec_info_to_real_bs(forward_batch.spec_info, model_worker_batch.real_bs)
        return forward_batch.spec_info

    def capture_for_decode(self, logits_output, draft_input: EagleDraftInput):
        topk_p, topk_index = topk_probs_from_logits(logits_output.next_token_logits, self.topk)
        if self.hot_token_ids is not None:
            topk_index = self._remap_hot_token_ids(topk_index)
        topk_index = np.asarray(jax.device_get(topk_index))
        draft_input.topk_p = topk_p
        draft_input.topk_index = topk_index
        draft_input.hidden_states = logits_output.hidden_states

    def _trim_prefill_spec_info_to_real_bs(
        self, draft_input: EagleDraftInput, real_bs: int
    ) -> None:
        keep_indices_host = np.arange(real_bs, dtype=np.int32)
        keep_indices = device_array(
            keep_indices_host,
            sharding=NamedSharding(self.model_runner.mesh, P("data")),
        )

        def take_rows(value):
            if value is None:
                return None
            if isinstance(value, jax.Array):
                return _take_with_optional_out_sharding(
                    value, keep_indices, trailing_slice=value.ndim > 1
                )
            return value[:real_bs]

        draft_input.hidden_states = take_rows(draft_input.hidden_states)
        draft_input.verified_id = take_rows(draft_input.verified_id)
        draft_input.topk_p = take_rows(draft_input.topk_p)
        draft_input.topk_index = take_rows(draft_input.topk_index)
        draft_input.allocate_lens = take_rows(draft_input.allocate_lens)

    def draft_extend_for_decode(
        self, model_worker_batch: ModelWorkerBatch, batch_output: "GenerationBatchResult"
    ) -> None:
        if batch_output.next_draft_input.verified_id.shape[0] <= 0:
            return
        draft_input = EagleDraftInput(
            hidden_states=batch_output.logits_output.hidden_states,
            allocate_lens=batch_output.allocate_lens,
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
        select_index = (
            np.arange(len(model_worker_batch.seq_lens[: model_worker_batch.real_bs]))
            * (self.speculative_num_steps + 1)
            + batch_output.accept_lens[: model_worker_batch.real_bs]
            - 1
        )
        draft_logits_output.next_token_logits = _take_with_optional_out_sharding(
            draft_logits_output.next_token_logits, select_index, trailing_slice=True
        )
        draft_logits_output.hidden_states = _take_with_optional_out_sharding(
            draft_logits_output.hidden_states, select_index, trailing_slice=True
        )
        topk_p, topk_index = topk_probs_from_logits(
            draft_logits_output.next_token_logits, self.topk
        )
        if self.hot_token_ids is not None:
            topk_index = self._remap_hot_token_ids(topk_index)
        topk_index = np.asarray(jax.device_get(topk_index))

        # prepare for next draft decode
        batch_output.next_draft_input.hidden_states = draft_logits_output.hidden_states
        batch_output.next_draft_input.topk_p = topk_p
        batch_output.next_draft_input.topk_index = topk_index
        batch_output.next_draft_input.verified_id = _take_with_optional_out_sharding(
            batch_output.next_draft_input.verified_id, select_index
        )
        batch_output.allocate_lens = batch_output.allocate_lens[: model_worker_batch.real_bs]
        batch_output.accept_lens = batch_output.accept_lens[: model_worker_batch.real_bs]

    def draft_forward(self, model_worker_batch: ModelWorkerBatch):
        topk_p, topk_index, hidden_states = (
            model_worker_batch.spec_info.topk_p,
            model_worker_batch.spec_info.topk_index,
            model_worker_batch.spec_info.hidden_states,
        )
        bs = model_worker_batch.seq_lens.shape[0]
        step_min_1 = self.speculative_num_steps - 1
        score_list: jax.Array = device_array(
            np.empty((bs, 1 + step_min_1 * self.topk, self.topk), dtype=np.float32),
            sharding=NamedSharding(self.model_runner.mesh, P("data", None, None)),
        )
        token_list: jax.Array = device_array(
            np.empty((bs, self.topk + step_min_1 * self.topk * self.topk), dtype=np.int32),
            sharding=NamedSharding(self.model_runner.mesh, P("data", None)),
        )
        parents_list: jax.Array = device_array(
            np.empty((bs, self.topk + 1 + step_min_1 * self.topk), dtype=np.int32),
            sharding=NamedSharding(self.model_runner.mesh, P("data", None)),
        )
        scores = None
        positions_base = device_array(
            np.repeat(model_worker_batch.seq_lens, self.topk),
            sharding=(NamedSharding(self.model_runner.mesh, P())),
        )
        logits_metadata = None
        metadata_per_step = self.draft_model_runner.attn_backend.get_eagle_multi_step_metadata(
            model_worker_batch,
        )
        assert isinstance(metadata_per_step, list)
        # we just use logits_metadata's forward mode and capture mode, it will not be modified within the loop
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
            # update_eagle_lists and update_forward_batch_info this two function will make accept rate very low if be jitted
            # FIXME we should find it out why lead this ?
            score_list, token_list, parents_list = update_eagle_lists(
                i, score_list, token_list, parents_list, tree_info, self.topk
            )
            if i == self.speculative_num_steps - 1:
                break

            forward_batch = update_forward_batch_info(
                forward_batch, i, input_ids, hidden_states, positions_base
            )
            self.draft_model_runner.attn_backend.forward_metadata = metadata_per_step[i]

            # Run forward
            forward_batch.bid = model_worker_batch.bid
            logits_output, _, _ = self.draft_model_runner.forward(
                forward_batch,
                logits_metadata=logits_metadata,
            )

            topk_p, topk_index = topk_probs_from_logits(logits_output.next_token_logits, self.topk)

            if self.hot_token_ids is not None:
                topk_index = self._remap_hot_token_ids(topk_index)
            hidden_states = logits_output.hidden_states

        return score_list, token_list, parents_list

    def run_spec_decode_precompile(self):
        self.precompile_spec_extend()
        self.precompile_spec_decode()
        self.precompile_runtime_jax_helpers()

    def _get_phase1_runtime_bs_candidates(self) -> list[int]:
        max_bs = max(self.precompile_bs_paddings) if self.precompile_bs_paddings else 0
        if max_bs <= 0:
            return []
        return [bs for bs in (1, 2, 4, 8, 16) if bs <= max_bs]

    def _get_phase1_runtime_bs_padding(self, real_bs: int) -> int:
        for bs in self._get_phase1_runtime_bs_candidates():
            if bs >= real_bs:
                return bs
        return real_bs

    def _get_phase1_runtime_indices(self, real_bs: int) -> np.ndarray:
        padded_bs = self._get_phase1_runtime_bs_padding(real_bs)
        indices = np.arange(padded_bs, dtype=np.int32)
        if padded_bs > real_bs:
            indices[real_bs:] = max(real_bs - 1, 0)
        return indices

    def _get_padding_bs_for_real_bs(self, real_bs: int) -> int:
        for bs in sorted(self.precompile_bs_paddings):
            if bs >= real_bs:
                return bs
        raise RuntimeError("did not get comperate padding bs, it should not happened")

    def precompile_runtime_jax_helpers(self):
        """Warm EAGLE runtime helper ops whose shapes follow real batch size.

        The model forward itself is padded to the configured precompile buckets, but
        several EAGLE post-processing helpers intentionally operate on the real
        number of active requests. Without warming these shapes, the first request
        drain through batch sizes such as 15, 13, or 6 can trigger persistent-cache
        misses for small JAX gather/top-k/reshard kernels.
        """
        max_bs = max(self.precompile_bs_paddings) if self.precompile_bs_paddings else 0
        if max_bs <= 0:
            return

        start_time = time.perf_counter()
        # The 4K/1K Phase-1 cache gate exercises bsz up to 16.  Larger buckets
        # such as max-running-requests=256 are still covered by the normal
        # padded model-forward precompile, but warming every real drain size up
        # to 256 would add excessive startup work for tiny helper kernels.
        max_runtime_bs = min(max_bs, 16)
        bs_candidates = list(range(1, max_runtime_bs + 1))
        logger.info("[SPEC_RUNTIME] Begin to precompile real_bs=%s", bs_candidates)

        data_sharding = NamedSharding(self.mesh, P("data"))
        data_2d_sharding = NamedSharding(self.mesh, P("data", None))
        replicated_sharding = NamedSharding(self.mesh, P(None))
        replicated_2d_sharding = NamedSharding(self.mesh, P(None, None))
        logits_sharding = NamedSharding(self.mesh, P("data", "tensor"))

        dtype = jnp.bfloat16 if self.server_args.dtype == "bfloat16" else jnp.float32
        hidden_size = self.model_config.hidden_size
        vocab_size = self.model_config.vocab_size

        with tqdm(bs_candidates, desc="[SPEC_RUNTIME] PRECOMPILE", leave=False) as pbar:
            for bs in pbar:
                pbar.set_postfix(real_bs=bs)
                indices = device_array(
                    np.arange(bs, dtype=np.int32),
                    sharding=data_sharding,
                )

                logits = device_array(
                    np.zeros((bs, vocab_size), dtype=np.float32),
                    sharding=logits_sharding,
                ).astype(dtype)
                topk_p, topk_index = topk_probs_from_logits(logits, self.topk)
                topk_p.block_until_ready()
                topk_index.block_until_ready()
                if self.hot_token_ids is not None:
                    token_ids_2d_host = np.zeros((bs, self.topk), dtype=np.int32)
                    token_ids_1d_host = np.zeros((bs * self.topk,), dtype=np.int32)
                    token_ids_2d_data = device_array(token_ids_2d_host, sharding=data_2d_sharding)
                    token_ids_1d_data = device_array(token_ids_1d_host, sharding=data_sharding)
                    token_ids_2d_replicated = device_array(
                        token_ids_2d_host, sharding=replicated_2d_sharding
                    )
                    token_ids_1d_replicated = device_array(
                        token_ids_1d_host, sharding=replicated_sharding
                    )
                    for token_ids in (
                        topk_index,
                        topk_index.flatten(),
                        token_ids_2d_data,
                        token_ids_1d_data,
                        token_ids_2d_replicated,
                        token_ids_1d_replicated,
                        token_ids_2d_host,
                        token_ids_1d_host,
                    ):
                        remapped_token_ids = self._remap_hot_token_ids(token_ids)
                        remapped_token_ids.block_until_ready()
                        np.asarray(jax.device_get(remapped_token_ids))

                hidden = device_array(
                    np.zeros((bs, hidden_size), dtype=np.float32),
                    sharding=data_2d_sharding,
                ).astype(dtype)
                verified_id = device_array(
                    np.zeros((bs,), dtype=np.int32),
                    sharding=data_sharding,
                )
                replicated_hidden = device_array(
                    np.zeros((bs, hidden_size), dtype=np.float32),
                    sharding=replicated_2d_sharding,
                ).astype(dtype)
                replicated_verified_id = device_array(
                    np.zeros((bs,), dtype=np.int32),
                    sharding=replicated_sharding,
                )
                _take_with_optional_out_sharding(
                    logits, indices, trailing_slice=True
                ).block_until_ready()
                _take_with_optional_out_sharding(
                    hidden, indices, trailing_slice=True
                ).block_until_ready()
                _take_with_optional_out_sharding(verified_id, indices).block_until_ready()
                _take_with_optional_out_sharding(
                    replicated_hidden, indices, trailing_slice=True
                ).block_until_ready()
                _take_with_optional_out_sharding(
                    replicated_verified_id, indices
                ).block_until_ready()

                for keep_bs in range(1, bs + 1):
                    keep_indices = device_array(
                        np.arange(keep_bs, dtype=np.int32),
                        sharding=data_sharding,
                    )
                    keep_indices_replicated = device_array(
                        np.arange(keep_bs, dtype=np.int32),
                        sharding=replicated_sharding,
                    )
                    keep_indices_host = np.arange(keep_bs, dtype=np.int32)
                    for keep_index in (
                        keep_indices,
                        keep_indices_replicated,
                        keep_indices_host,
                    ):
                        _take_with_optional_out_sharding(
                            logits, keep_index, trailing_slice=True
                        ).block_until_ready()
                        _take_with_optional_out_sharding(
                            topk_p, keep_index, trailing_slice=True
                        ).block_until_ready()
                        _take_with_optional_out_sharding(
                            topk_index, keep_index, trailing_slice=True
                        ).block_until_ready()
                        _take_with_optional_out_sharding(
                            hidden, keep_index, trailing_slice=True
                        ).block_until_ready()
                        _take_with_optional_out_sharding(
                            verified_id, keep_index
                        ).block_until_ready()
                        _take_with_optional_out_sharding(
                            replicated_hidden, keep_index, trailing_slice=True
                        ).block_until_ready()
                        _take_with_optional_out_sharding(
                            replicated_verified_id, keep_index
                        ).block_until_ready()

                if bs == max_runtime_bs and max_bs > max_runtime_bs:
                    padded_bs = max_bs
                    padded_logits = device_array(
                        np.zeros((padded_bs, vocab_size), dtype=np.float32),
                        sharding=logits_sharding,
                    ).astype(dtype)
                    padded_hidden = device_array(
                        np.zeros((padded_bs, hidden_size), dtype=np.float32),
                        sharding=data_2d_sharding,
                    ).astype(dtype)
                    padded_ids = device_array(
                        np.zeros((padded_bs,), dtype=np.int32),
                        sharding=data_sharding,
                    )
                    padded_indices = device_array(
                        np.arange(padded_bs, dtype=np.int32),
                        sharding=data_sharding,
                    )
                    _take_with_optional_out_sharding(padded_ids, padded_indices).block_until_ready()
                    for keep_bs in bs_candidates:
                        keep_indices = device_array(
                            np.arange(keep_bs, dtype=np.int32),
                            sharding=data_sharding,
                        )
                        _take_with_optional_out_sharding(
                            padded_logits, keep_indices, trailing_slice=True
                        ).block_until_ready()
                        _take_with_optional_out_sharding(
                            padded_hidden, keep_indices, trailing_slice=True
                        ).block_until_ready()
                        _take_with_optional_out_sharding(
                            padded_ids, keep_indices
                        ).block_until_ready()

                draft_extend_slots = bs * (self.speculative_num_steps + 1)
                draft_extend_logits = device_array(
                    np.zeros((draft_extend_slots, vocab_size), dtype=np.float32),
                    sharding=logits_sharding,
                ).astype(dtype)
                draft_extend_hidden = device_array(
                    np.zeros((draft_extend_slots, hidden_size), dtype=np.float32),
                    sharding=data_2d_sharding,
                ).astype(dtype)
                draft_extend_ids = device_array(
                    np.zeros((draft_extend_slots,), dtype=np.int32),
                    sharding=data_sharding,
                )
                for keep_bs in range(1, bs + 1):
                    select_index_host = np.arange(keep_bs, dtype=np.int32) * (
                        self.speculative_num_steps + 1
                    )
                    select_index_data = device_array(
                        select_index_host,
                        sharding=data_sharding,
                    )
                    select_index_replicated = device_array(
                        select_index_host,
                        sharding=replicated_sharding,
                    )
                    for select_index in (
                        select_index_host,
                        select_index_data,
                        select_index_replicated,
                    ):
                        selected_logits = _take_with_optional_out_sharding(
                            draft_extend_logits, select_index, trailing_slice=True
                        )
                        selected_logits.block_until_ready()
                        _take_with_optional_out_sharding(
                            draft_extend_hidden, select_index, trailing_slice=True
                        ).block_until_ready()
                        _take_with_optional_out_sharding(
                            draft_extend_ids, select_index
                        ).block_until_ready()
                        selected_topk_p, selected_topk_index = topk_probs_from_logits(
                            selected_logits, self.topk
                        )
                        selected_topk_p.block_until_ready()
                        selected_topk_index.block_until_ready()
                        if self.hot_token_ids is not None:
                            selected_topk_index = self._remap_hot_token_ids(selected_topk_index)
                            selected_topk_index.block_until_ready()

                if self.topk == 1:
                    continue

                step_min_1 = self.speculative_num_steps - 1
                score_list = device_array(
                    np.zeros((bs, 1 + step_min_1 * self.topk, self.topk), dtype=np.float32),
                    sharding=NamedSharding(self.mesh, P("data", None, None)),
                )
                token_list = device_array(
                    np.zeros(
                        (bs, self.topk + step_min_1 * self.topk * self.topk),
                        dtype=np.int32,
                    ),
                    sharding=data_2d_sharding,
                )
                parents_list = device_array(
                    np.zeros((bs, self.topk + 1 + step_min_1 * self.topk), dtype=np.int32),
                    sharding=data_2d_sharding,
                )
                hidden_states = device_array(
                    np.zeros((bs, hidden_size), dtype=dtype),
                    sharding=data_2d_sharding,
                )
                scores = None
                for i in range(self.speculative_num_steps):
                    _, hidden_states, scores, tree_info = select_top_k_tokens(
                        i, topk_p, topk_index, hidden_states, scores, self.topk
                    )
                    score_list, token_list, parents_list = update_eagle_lists(
                        i, score_list, token_list, parents_list, tree_info, self.topk
                    )
                    score_list.block_until_ready()
                    token_list.block_until_ready()
                    parents_list.block_until_ready()

        end_time = time.perf_counter()
        logger.info("[SPEC_RUNTIME] Precompile finished in %.0f secs", end_time - start_time)

    def precompile_spec_extend(self):
        start_time = time.perf_counter()
        real_bs_candidates = self._get_phase1_runtime_bs_candidates()
        precompile_pairs = []
        for num_tokens in self.precompile_token_paddings:
            for real_bs in real_bs_candidates:
                if num_tokens % real_bs != 0:
                    continue
                precompile_pairs.append(
                    (self._get_padding_bs_for_real_bs(real_bs), real_bs, num_tokens)
                )

        logger.info(
            "[SPEC_EXTEND] Begin to precompile bs_paddings=%s real_bs=%s token_paddings=%s",
            sorted(set(padded_bs for padded_bs, _, _ in precompile_pairs)),
            real_bs_candidates,
            self.precompile_token_paddings,
        )

        cache_loc_by_bs = dict(zip(self.precompile_bs_paddings, self.precompile_cache_loc_paddings))

        with tqdm(precompile_pairs, desc="[SPEC_EXTEND] PRECOMPILE", leave=False) as pbar:
            for padded_bs, real_bs, num_tokens in pbar:
                pbar.set_postfix(bs=padded_bs, real_bs=real_bs, tokens=num_tokens)
                tokens_per_req = num_tokens // real_bs
                if tokens_per_req <= 0:
                    continue
                if padded_bs > num_tokens:
                    logger.warning(
                        "bs=%s > num_tokens=%s, skip this pair",
                        padded_bs,
                        num_tokens,
                    )
                    continue
                model_worker_batch = self.generate_model_worker_batch(
                    padded_bs,
                    num_tokens,
                    ForwardMode.EXTEND,
                    cache_loc_by_bs[padded_bs],
                    do_penalties=False,
                    speculative_algotithm=self.speculative_algorithm,
                )
                model_worker_batch.return_output_logprob_only = False
                model_worker_batch.real_bs = real_bs
                model_worker_batch.real_input_ids_len = num_tokens
                model_worker_batch.seq_lens = np.zeros(padded_bs, dtype=np.int32)
                model_worker_batch.seq_lens[:real_bs] = tokens_per_req
                model_worker_batch.req_pool_indices = np.full(padded_bs, -1, dtype=np.int32)
                model_worker_batch.req_pool_indices[:real_bs] = np.arange(real_bs, dtype=np.int32)
                model_worker_batch.extend_seq_lens = np.zeros(padded_bs, dtype=np.int32)
                model_worker_batch.extend_seq_lens[:real_bs] = tokens_per_req
                model_worker_batch.extend_prefix_lens = np.zeros(padded_bs, dtype=np.int32)
                model_worker_batch.logits_indices = (
                    np.cumsum(model_worker_batch.extend_seq_lens, dtype=np.int32) - 1
                )
                model_worker_batch.logits_indices[real_bs:] = 0
                model_worker_batch.positions = np.arange(num_tokens, dtype=np.int32)
                model_worker_batch.out_cache_loc = np.arange(1, num_tokens + 1, dtype=np.int32)
                if model_worker_batch.sampling_info.temperatures.ndim == 1:
                    model_worker_batch.sampling_info.temperatures = (
                        model_worker_batch.sampling_info.temperatures[:, None]
                    )
                sampling_metadata = SamplingMetadata.from_model_worker_batch(
                    model_worker_batch,
                    len(model_worker_batch.seq_lens) - model_worker_batch.real_bs,
                    self.mesh,
                    vocab_size=self.model_config.vocab_size,
                )
                model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
                logits_output, next_token_ids, _ = self.target_worker.forward_batch_generation(
                    model_worker_batch, sampling_metadata=sampling_metadata
                )
                self.draft_extend_for_prefill(
                    model_worker_batch, logits_output.hidden_states, next_token_ids
                )
                np.asarray(jax.device_get(next_token_ids))
        end_time = time.perf_counter()
        logger.info("[SPEC_EXTEND] Precompile finished in %.0f secs", end_time - start_time)

    def precompile_spec_decode(self):
        start_time = time.perf_counter()
        max_bs = max(self.precompile_bs_paddings) if self.precompile_bs_paddings else 0
        runtime_bs_candidates = list(range(1, min(max_bs, 16) + 1))
        decode_bs_candidates = sorted(
            set(runtime_bs_candidates + list(self.precompile_bs_paddings))
        )
        logger.info(
            "[SPEC_DECODE] Begin to precompile bs_paddings=%s",
            decode_bs_candidates,
        )

        with tqdm(decode_bs_candidates, desc="[SPEC_DECODE] PRECOMPILE", leave=False) as pbar:
            for bs in pbar:
                pbar.set_postfix(bs=bs)
                # use same page aligned with precompile cache_loc_paddings
                aligned_cache_loc_size = (
                    (bs * self.max_req_len + self.page_size - 1) // self.page_size * self.page_size
                )
                model_worker_batch = self.generate_model_worker_batch(
                    bs,
                    bs,
                    ForwardMode.DECODE,
                    aligned_cache_loc_size,
                    do_penalties=False,
                    speculative_algotithm=self.speculative_algorithm,
                )
                spec_info = EagleDraftInput(
                    # FIXME(pc) dtype should according to serverargs
                    topk_p=jnp.ones(
                        (bs, self.topk),
                        dtype=jnp.bfloat16 if self.server_args.dtype == "bfloat16" else jnp.float32,
                    ),
                    topk_index=jnp.ones((bs, self.topk), dtype=jnp.int32),
                    hidden_states=jnp.ones(
                        (bs, self.model_config.hidden_size),
                        dtype=jnp.bfloat16 if self.server_args.dtype == "bfloat16" else jnp.float32,
                    ),
                    verified_id=jnp.ones((bs,), dtype=jnp.int32),
                    accept_length=jnp.ones((bs,), dtype=jnp.int32),
                    capture_hidden_mode=CaptureHiddenMode.LAST,
                    allocate_lens=model_worker_batch.seq_lens
                    + EagleDraftInput.ALLOC_LEN_PER_DECODE,
                )
                model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
                model_worker_batch.spec_info = spec_info
                model_worker_batch.speculative_eagle_topk = self.topk
                model_worker_batch.speculative_num_draft_tokens = self.speculative_num_draft_tokens
                model_worker_batch.speculative_num_steps = self.speculative_num_steps
                cur_allocate_lens = model_worker_batch.spec_info.allocate_lens
                verify_input = self.draft(model_worker_batch)
                model_worker_batch.spec_info = verify_input
                self.pad_out_cache_loc_for_verify(model_worker_batch)

                verify_input.allocate_lens = cur_allocate_lens
                verify_input.prepare_for_verify(
                    model_worker_batch, self.page_size, self.target_worker
                )
                forward_metadata = (
                    self.target_worker.model_runner.attn_backend.get_eagle_forward_metadata(
                        model_worker_batch
                    )
                )
                logits_output, _, _ = self.target_worker.forward_batch_generation(
                    model_worker_batch,
                    skip_sample=True,
                    forward_metadata=forward_metadata,
                )
                verify_input.hidden_states = logits_output.hidden_states
                (
                    _predict,
                    verified_id,
                    accept_length,
                    accept_index,
                ) = verify_input.sample(
                    model_worker_batch,
                    logits_output,
                    self.model_runner.rngs,
                    self.mesh,
                )
                logits_output.next_token_logits = _take_with_optional_out_sharding(
                    logits_output.next_token_logits, accept_index, trailing_slice=True
                )
                logits_output.hidden_states = _take_with_optional_out_sharding(
                    logits_output.hidden_states, accept_index, trailing_slice=True
                )
                model_worker_batch.positions = _take_with_optional_out_sharding(
                    model_worker_batch.positions, accept_index
                )
                batch_output = SimpleNamespace(
                    logits_output=logits_output,
                    next_draft_input=EagleDraftInput(
                        verified_id=verified_id,
                        new_seq_lens=model_worker_batch.seq_lens + accept_length,
                        allocate_lens=cur_allocate_lens,
                        hidden_states=logits_output.hidden_states,
                    ),
                    allocate_lens=cur_allocate_lens,
                    accept_lens=accept_length,
                )
                self.draft_extend_for_decode(model_worker_batch, batch_output)

        end_time = time.perf_counter()
        logger.info("[SPEC_DECODE] Precompile finished in %.0f secs", end_time - start_time)
