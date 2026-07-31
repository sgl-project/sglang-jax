import itertools
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from tqdm import tqdm

from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sgl_jax.srt.speculative.base_worker import BaseSpecWorker
from sgl_jax.srt.speculative.eagle_draft_worker import EagleDraftWorker
from sgl_jax.srt.speculative.eagle_info import EagleDraftInput

logger = logging.getLogger(__name__)


class EAGLEWorker(BaseSpecWorker):
    """Fused topk=1 EAGLE/EAGLE3 speculative decode orchestrator.

    Composes a ``target_worker`` (full model) with an ``EagleDraftWorker``
    (draft model).  Implements the ``BaseSpecWorker`` contract so the
    scheduler interface is unchanged.
    """

    def __init__(self, server_args, target_worker: ModelWorker, draft_worker=None):
        super().__init__(
            server_args,
            target_worker,
            draft_worker or EagleDraftWorker(server_args, target_worker),
        )

    # -- BaseSpecWorker provides target_worker/draft_worker/verify/
    #    forward_target_extend/forward_batch_speculative_generation --

    # -- Precompilation --

    def run_spec_decode_precompile(self):
        if not self.server_args.disable_overlap_schedule:
            self.init_spec_relay_buffers()
        self.precompile_spec_extend()
        self.precompile_spec_decode()
        # FIXME precompile some kernel

    def precompile_spec_extend(self):
        start_time = time.perf_counter()
        dp_size = self.server_args.dp_size
        logger.info(
            "[SPEC_EXTEND] Begin to precompile bs_paddings=%s token_paddings=%s dp_size=%d",
            self.precompile_bs_paddings[-1:],
            self.precompile_token_paddings,
            dp_size,
        )

        bs, _ = self.draft_worker.get_max_padded_size()
        pairs = list(itertools.product([bs], self.precompile_token_paddings))

        with tqdm(pairs, desc="[SPEC_EXTEND] PRECOMPILE", leave=False) as pbar:
            for pair in pbar:
                pair = list(pair)
                bs, num_tokens = pair[0], pair[1]
                pbar.set_postfix(bs=bs, tokens=num_tokens, dp_size=dp_size)
                if bs > num_tokens:
                    logger.warning("bs=%s > num_tokens=%s, skip this pair", bs, num_tokens)
                    continue
                if bs % dp_size != 0:
                    logger.warning(
                        "[SPEC_EXTEND] skip bs=%d (not divisible by dp_size=%d)", bs, dp_size
                    )
                    continue
                per_dp_bs = bs // dp_size
                model_worker_batch = self.draft_worker.compilation_manager._make_dummy_batch(
                    bs,
                    num_tokens,
                    ForwardMode.EXTEND,
                    self.precompile_cache_loc_paddings[-1],
                    speculative_algorithm=self.speculative_algorithm,
                    dp_size=dp_size,
                    per_dp_bs_size=per_dp_bs,
                )
                if not self._can_use_fused_spec_prefill(model_worker_batch):
                    logger.warning(
                        "[SPEC_EXTEND] skip fused precompile because fused spec prefill is disabled"
                    )
                    continue
                if self.spec_relay_buffers is not None:
                    self.forward_batch_speculative_prefill_overlap(model_worker_batch)
                    jax.block_until_ready(self.spec_relay_buffers)
                else:
                    self.forward_batch_speculative_generation(model_worker_batch)
        end_time = time.perf_counter()
        logger.info("[SPEC_EXTEND] Precompile finished in %.0f secs", end_time - start_time)

    def precompile_spec_decode(self):
        start_time = time.perf_counter()
        dp_size = self.server_args.dp_size
        logger.info(
            "[SPEC_DECODE] Begin to precompile bs_paddings=%s dp_size=%d",
            self.precompile_bs_paddings,
            dp_size,
        )

        with tqdm(
            self.precompile_bs_paddings, desc="[SPEC_DECODE] PRECOMPILE", leave=False
        ) as pbar:
            for bs in pbar:
                pbar.set_postfix(bs=bs, dp_size=dp_size)
                if bs % dp_size != 0:
                    logger.warning(
                        "[SPEC_DECODE] skip bs=%d (not divisible by dp_size=%d)", bs, dp_size
                    )
                    continue
                per_dp_bs = bs // dp_size
                aligned_cache_loc_size = (
                    (bs * self.draft_worker.max_req_len + self.page_size - 1)
                    // self.page_size
                    * self.page_size
                )

                def _make_decode_batch(
                    *,
                    bs=bs,
                    per_dp_bs=per_dp_bs,
                    aligned_cache_loc_size=aligned_cache_loc_size,
                ):
                    batch = self.draft_worker.compilation_manager._make_dummy_batch(
                        bs,
                        bs,
                        ForwardMode.DECODE,
                        aligned_cache_loc_size,
                        speculative_algorithm=self.speculative_algorithm,
                        dp_size=dp_size,
                        per_dp_bs_size=per_dp_bs,
                    )
                    # Pad out_cache_loc to the conservative decode allocation
                    # bucket used by runtime _get_spec_decode_mwb_dp.
                    ocl_target = bs * self.speculative_num_draft_tokens * 2
                    if batch.out_cache_loc.shape[0] < ocl_target:
                        pad_len = ocl_target - batch.out_cache_loc.shape[0]
                        batch.out_cache_loc = np.concatenate(
                            [
                                np.asarray(batch.out_cache_loc, dtype=np.int32),
                                np.full(pad_len, -1, dtype=np.int32),
                            ]
                        )
                    return batch

                model_worker_batch = _make_decode_batch()
                assert not model_worker_batch.return_logprob
                assert not model_worker_batch.return_output_logprob_only
                assert model_worker_batch.sampling_info.is_all_greedy
                num_steps = self.speculative_num_steps
                chain_shape = (bs, num_steps)
                data_sharding = NamedSharding(self.mesh, P("data"))
                spec_info = EagleDraftInput(
                    topk_index=jax.device_put(np.ones(chain_shape, dtype=np.int32), data_sharding),
                    hidden_states=np.ones(
                        (bs, self.draft_worker.model_config.hidden_size),
                        dtype=(
                            jnp.bfloat16 if self.server_args.dtype == "bfloat16" else np.float32
                        ),
                    ),
                    verified_id=jax.device_put(np.ones((bs,), dtype=np.int32), data_sharding),
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                    num_tokens_per_batch=np.asarray(1, dtype=np.int32),
                    num_tokens_for_logprob_per_batch=np.asarray(1, dtype=np.int32),
                    allocate_lens=model_worker_batch.seq_lens
                    + EagleDraftInput.ALLOC_LEN_PER_DECODE,
                )
                if self.spec_relay_buffers is not None:
                    model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
                    model_worker_batch.spec_info_padded = spec_info
                    model_worker_batch.speculative_eagle_topk = self.topk
                    model_worker_batch.speculative_num_draft_tokens = (
                        self.speculative_num_draft_tokens
                    )
                    model_worker_batch.speculative_num_steps = self.speculative_num_steps
                    self.forward_batch_speculative_decode_overlap(model_worker_batch)
                    jax.block_until_ready(self.spec_relay_buffers)

                    model_worker_batch = _make_decode_batch()
                    spec_info = EagleDraftInput(
                        future_indices=np.asarray(
                            model_worker_batch.req_pool_indices, dtype=np.int32
                        ),
                        capture_hidden_mode=CaptureHiddenMode.FULL,
                        num_tokens_per_batch=np.asarray(1, dtype=np.int32),
                        num_tokens_for_logprob_per_batch=np.asarray(1, dtype=np.int32),
                        allocate_lens=model_worker_batch.seq_lens
                        + EagleDraftInput.ALLOC_LEN_PER_DECODE,
                    )
                model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
                model_worker_batch.spec_info_padded = spec_info
                model_worker_batch.speculative_eagle_topk = self.topk
                model_worker_batch.speculative_num_draft_tokens = self.speculative_num_draft_tokens
                model_worker_batch.speculative_num_steps = self.speculative_num_steps
                if self.spec_relay_buffers is not None:
                    self.forward_batch_speculative_decode_overlap(model_worker_batch)
                    jax.block_until_ready(self.spec_relay_buffers)
                else:
                    self.forward_batch_speculative_generation(model_worker_batch)

        end_time = time.perf_counter()
        logger.info("[SPEC_DECODE] Precompile finished in %.0f secs", end_time - start_time)
