import itertools
import logging
import time
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
        topk_index = spec_info.topk_index
        if self.hot_token_ids is not None:
            model_worker_batch.spec_info.topk_index = self._remap_hot_token_ids(topk_index)
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
        model_worker_batch.spec_info = EagleDraftInput(
            hidden_states=target_hidden_states,
            verified_id=next_token_ids[: model_worker_batch.real_bs],
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
        logits_output.next_token_logits = logits_output.next_token_logits[
            : model_worker_batch.real_bs, :
        ]
        if len(logits_output.hidden_states.shape) == 1:
            logits_output.hidden_states = jnp.expand_dims(logits_output.hidden_states, axis=0)
        assert isinstance(forward_batch.spec_info, EagleDraftInput)
        forward_batch.spec_info.allocate_lens = model_worker_batch.seq_lens[
            : model_worker_batch.real_bs
        ]

        self.capture_for_decode(logits_output, forward_batch.spec_info)
        return forward_batch.spec_info

    def capture_for_decode(self, logits_output, draft_input: EagleDraftInput):
        topk_p, topk_index = topk_probs_from_logits(logits_output.next_token_logits, self.topk)
        draft_input.topk_p = topk_p
        draft_input.topk_index = topk_index
        draft_input.hidden_states = logits_output.hidden_states

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
        # FIXME precompile some kernel

    def precompile_spec_extend(self):
        start_time = time.perf_counter()
        logger.info(
            "[SPEC_EXTEND] Begin to precompile bs_paddings=%s token_paddings=%s",
            self.precompile_bs_paddings[-1:],
            self.precompile_token_paddings,
        )

        bs, _ = self.get_max_padded_size()
        pairs = list(itertools.product([bs], self.precompile_token_paddings))

        with tqdm(pairs, desc="[SPEC_EXTEND] PRECOMPILE", leave=False) as pbar:
            for pair in pbar:
                pair = list(pair)
                bs, num_tokens = pair[0], pair[1]
                pbar.set_postfix(bs=bs, tokens=num_tokens)
                if bs > num_tokens:
                    logger.warning("bs=%s > num_tokens=%s, skip this pair", bs, num_tokens)
                    continue
                model_worker_batch = self.generate_model_worker_batch(
                    bs,
                    num_tokens,
                    ForwardMode.EXTEND,
                    self.precompile_cache_loc_paddings[-1],
                    do_penalties=False,
                    speculative_algotithm=self.speculative_algorithm,
                )
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
        end_time = time.perf_counter()
        logger.info("[SPEC_EXTEND] Precompile finished in %.0f secs", end_time - start_time)

    def precompile_spec_decode(self):
        start_time = time.perf_counter()
        logger.info(
            "[SPEC_DECODE] Begin to precompile bs_paddings=%s",
            self.precompile_bs_paddings,
        )

        with tqdm(
            self.precompile_bs_paddings, desc="[SPEC_DECODE] PRECOMPILE", leave=False
        ) as pbar:
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
                self.draft(model_worker_batch)

        end_time = time.perf_counter()
        logger.info("[SPEC_DECODE] Precompile finished in %.0f secs", end_time - start_time)
