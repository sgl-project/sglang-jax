import itertools
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec
from tqdm import tqdm

from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
from sgl_jax.srt.layers.sampler import get_token_ids_logprobs, get_top_logprobs
from sgl_jax.srt.managers.schedule_batch import ScheduleBatch
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sgl_jax.srt.speculative.base_worker import BaseSpecWorker
from sgl_jax.srt.speculative.eagle_draft_worker import EagleDraftWorker
from sgl_jax.srt.speculative.eagle_util import EagleDraftInput, EagleVerifyOutput
from sgl_jax.srt.utils.common_utils import get_bool_env_var
from sgl_jax.srt.utils.jax_utils import device_array

logger = logging.getLogger(__name__)
RETURN_ORIGINAL_LOGPROB = get_bool_env_var("RETURN_ORIGINAL_LOGPROB")


class EAGLEWorker(BaseSpecWorker):
    """Standard EAGLE speculative decode orchestrator.

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

    # -- Logprob post-processing --

    def add_logprob_values(
        self,
        batch: ScheduleBatch,
        res: EagleVerifyOutput,
        logits_output: LogitsProcessorOutput,
    ):
        logits_output = res.logits_output
        top_logprobs_nums = batch.top_logprobs_nums
        token_ids_logprobs = batch.token_ids_logprobs
        accepted_indices = res.accepted_indices
        assert len(accepted_indices) == len(logits_output.next_token_logits)

        temperatures = batch.sampling_info.temperatures
        num_draft_tokens = batch.reqs_info[0].spec_info.draft_token_num
        temperatures = temperatures[accepted_indices // num_draft_tokens]
        if RETURN_ORIGINAL_LOGPROB:
            logprobs = jax.nn.log_softmax(logits_output.next_token_logits, axis=-1)
        else:
            logprobs = jax.nn.log_softmax(logits_output.next_token_logits / temperatures, axis=-1)
        batch_next_token_ids = res.verified_id
        num_tokens_per_req = [accept + 1 for accept in res.accept_length_per_req_cpu]

        top_logprobs_nums_repeat_interleaved = []
        token_ids_logprobs_repeat_interleaved = []
        for num, num_tokens in zip(top_logprobs_nums, num_tokens_per_req):
            top_logprobs_nums_repeat_interleaved.extend([num] * num_tokens)
        for token_ids, num_tokens in zip(token_ids_logprobs, num_tokens_per_req):
            token_ids_logprobs_repeat_interleaved.extend([token_ids] * num_tokens)

        if any(x > 0 for x in top_logprobs_nums):
            (
                logits_output.next_token_top_logprobs_val,
                logits_output.next_token_top_logprobs_idx,
            ) = get_top_logprobs(
                logprobs,
                top_logprobs_nums_repeat_interleaved,
            )

        if any(x is not None for x in token_ids_logprobs):
            logits_output.next_token_token_ids_logprobs_val = get_token_ids_logprobs(
                logprobs,
                token_ids_logprobs_repeat_interleaved,
                None,
            )
            logits_output.next_token_token_ids_logprobs_idx = None

        logits_output.next_token_logprobs = logprobs[
            jnp.arange(len(batch_next_token_ids), device=batch.sampling_info.device),
            batch_next_token_ids,
        ]

        pt = 0
        next_token_logprobs = logits_output.next_token_logprobs.tolist()
        verified_ids = batch_next_token_ids.tolist()
        for req, num_tokens in zip(batch.reqs, num_tokens_per_req, strict=True):
            for _ in range(num_tokens):
                if req.return_logprob:
                    req.output_token_logprobs_val.append(next_token_logprobs[pt])
                    req.output_token_logprobs_idx.append(verified_ids[pt])
                    if req.top_logprobs_num > 0:
                        req.output_top_logprobs_val.append(
                            res.logits_output.next_token_top_logprobs_val[pt]
                        )
                        req.output_top_logprobs_idx.append(
                            res.logits_output.next_token_top_logprobs_idx[pt]
                        )
                pt += 1

    # -- Precompilation --

    def run_spec_decode_precompile(self):
        self.target_worker.model_runner._compilation_phase = "spec_precompile"
        if hasattr(self.draft_worker, "draft_model_runner"):
            self.draft_worker.draft_model_runner._compilation_phase = "spec_precompile"
        self.precompile_spec_extend()
        self.precompile_spec_decode()
        self.precompile_real_bs_postprocess_primitives()
        self.target_worker.model_runner._compilation_phase = "runtime"
        if hasattr(self.draft_worker, "draft_model_runner"):
            self.draft_worker.draft_model_runner._compilation_phase = "runtime"
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
                model_worker_batch = self.draft_worker.compilation_manager._make_dummy_batch(
                    bs,
                    bs,
                    ForwardMode.DECODE,
                    aligned_cache_loc_size,
                    speculative_algorithm=self.speculative_algorithm,
                    dp_size=dp_size,
                    per_dp_bs_size=per_dp_bs,
                )
                spec_info = EagleDraftInput(
                    # FIXME(pc) dtype should according to serverargs
                    topk_p=jnp.ones(
                        (bs, self.topk),
                        dtype=jnp.bfloat16 if self.server_args.dtype == "bfloat16" else jnp.float32,
                    ),
                    topk_index=jnp.ones((bs, self.topk), dtype=jnp.int32),
                    hidden_states=jnp.ones(
                        (bs, self.draft_worker.model_config.hidden_size),
                        dtype=jnp.bfloat16 if self.server_args.dtype == "bfloat16" else jnp.float32,
                    ),
                    verified_id=jnp.ones((bs,), dtype=jnp.int32),
                    accept_length=jnp.ones((bs,), dtype=jnp.int32),
                    capture_hidden_mode=CaptureHiddenMode.LAST,
                    allocate_lens=model_worker_batch.seq_lens
                    + EagleDraftInput.ALLOC_LEN_PER_DECODE,
                )
                model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
                model_worker_batch.spec_info_padded = spec_info
                model_worker_batch.speculative_eagle_topk = self.topk
                model_worker_batch.speculative_num_draft_tokens = self.speculative_num_draft_tokens
                model_worker_batch.speculative_num_steps = self.speculative_num_steps
                # Pad out_cache_loc to bs * draft_token_num so verify/draft_extend
                # forward see the same shape runtime _get_spec_decode_mwb_dp emits.
                ocl_target = bs * self.speculative_num_draft_tokens
                if model_worker_batch.out_cache_loc.shape[0] < ocl_target:
                    pad_len = ocl_target - model_worker_batch.out_cache_loc.shape[0]
                    model_worker_batch.out_cache_loc = np.concatenate(
                        [
                            np.asarray(model_worker_batch.out_cache_loc, dtype=np.int32),
                            np.full(pad_len, -1, dtype=np.int32),
                        ]
                    )
                self.forward_batch_speculative_generation(model_worker_batch)

        end_time = time.perf_counter()
        logger.info("[SPEC_DECODE] Precompile finished in %.0f secs", end_time - start_time)

    def precompile_real_bs_postprocess_primitives(self):
        """Warm real-bs-indexed postprocess primitives before the server is ready.

        Model forward is padded to the largest bucket, but postprocess code still
        uses real-bs selectors for per-request state. Without this warmup, the
        first runtime request for real batch sizes such as 2/3/5/6 creates small
        persistent-cache misses for gather/scatter-update primitives even though
        the model HLO is already cached.
        """
        start_time = time.perf_counter()
        total_bs = self.precompile_bs_paddings[-1]
        if total_bs <= 0:
            return

        dtype = jnp.bfloat16 if self.server_args.dtype == "bfloat16" else jnp.float32
        mesh = self.target_worker.model_runner.mesh
        sharding = NamedSharding(mesh, PartitionSpec())
        vocab_size = self.target_worker.model_config.vocab_size
        hidden_size = self.draft_worker.model_config.hidden_size
        accept_width = self.speculative_num_steps + 1
        draft_rows = total_bs * accept_width

        (
            topk_p,
            topk_index,
            logits,
            hidden,
            verified_id,
        ) = device_array(
            (
                np.zeros((total_bs, self.topk), dtype=np.float32),
                np.zeros((total_bs, self.topk), dtype=np.int32),
                np.zeros((total_bs, vocab_size), dtype=np.float32),
                np.zeros((draft_rows, hidden_size), dtype=np.float32),
                np.zeros((draft_rows,), dtype=np.int32),
            ),
            sharding=sharding,
        )
        topk_p = topk_p.astype(dtype)
        logits = logits.astype(dtype)
        hidden = hidden.astype(dtype)

        logger.info(
            "[SPEC_POSTPROCESS] Begin to precompile real_bs selectors 1..%d",
            total_bs,
        )
        with tqdm(
            range(1, total_bs + 1),
            desc="[SPEC_POSTPROCESS] PRECOMPILE",
            leave=False,
        ) as pbar:
            for real_bs in pbar:
                pbar.set_postfix(real_bs=real_bs)
                selector = np.arange(real_bs, dtype=np.int32)
                select_index = selector * accept_width
                sel_jax = jnp.asarray(selector)
                select_index_jax = jnp.asarray(select_index)

                accept_length = device_array(
                    np.zeros((real_bs,), dtype=np.int32),
                    sharding=sharding,
                )
                accept_filter = jnp.zeros_like(accept_length)
                accept_filter = accept_filter.at[sel_jax].set(accept_length[sel_jax] + 1)

                warmed = (
                    topk_p[sel_jax],
                    topk_index[sel_jax],
                    logits[sel_jax],
                    hidden[select_index_jax],
                    verified_id[select_index_jax],
                    topk_p[selector],
                    topk_index[selector],
                    hidden[selector],
                    verified_id[selector],
                    accept_filter,
                )
                jax.block_until_ready(warmed)

        for left in range(1, total_bs):
            for right in range(1, total_bs - left + 1):
                left_hidden = hidden[:left]
                right_hidden = hidden[:right]
                left_verified_id = verified_id[:left]
                right_verified_id = verified_id[:right]
                left_topk_p = topk_p[:left]
                right_topk_p = topk_p[:right]
                left_topk_index = topk_index[:left]
                right_topk_index = topk_index[:right]
                warmed = (
                    jnp.concatenate([left_hidden, right_hidden], axis=0),
                    jnp.concatenate([left_verified_id, right_verified_id], axis=0),
                    jnp.concatenate([left_topk_p, right_topk_p], axis=0),
                    jnp.concatenate([left_topk_index, right_topk_index], axis=0),
                )
                jax.block_until_ready(warmed)

        end_time = time.perf_counter()
        logger.info(
            "[SPEC_POSTPROCESS] Precompile finished in %.0f secs",
            end_time - start_time,
        )
