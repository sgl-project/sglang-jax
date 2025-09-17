"""A tensor parallel worker."""

import itertools
import logging
import os

# Add import for GMM auto tune
import sys
import threading
import time
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.experimental.multihost_utils import broadcast_one_to_all
from tqdm import tqdm

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.gmm.auto_tune_tiling import TilingAutoTuner
from sgl_jax.srt.layers.gmm.tiling_manager import get_default_cache_dir
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
from sgl_jax.srt.managers.schedule_batch import (
    ModelWorkerBatch,
    global_server_args_dict,
)
from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool
from sgl_jax.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sgl_jax.srt.model_executor.model_runner import MockModelRunner, ModelRunner
from sgl_jax.srt.sampling.sampling_batch_info import SamplingBatchInfo, SamplingMetadata
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.utils.common_utils import (
    PRECOMPILE_DEFAULT_BS_PADDINGS,
    PRECOMPILE_DEFAULT_TOKEN_PADDINGS,
)

logger = logging.getLogger(__name__)


class ModelWorker:
    """A tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        mesh: jax.sharding.Mesh,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
    ):
        # Parse args
        self.tp_size = server_args.tp_size

        # Init model and tokenizer
        self.model_config = ModelConfig.from_server_args(
            server_args,
            model_path=server_args.model_path,
        )

        self.mesh = mesh
        self.page_size = server_args.page_size

        # Sync random seed across TP workers
        # Each process may have different random_seed. After broadcast, all processes will have the same random_seed.
        # self.random_seed = broadcast_one_to_all(server_args.random_seed).item()
        if server_args.random_seed is None:
            with jax.default_device(jax.local_devices()[0]):
                if jax.process_index() == 0:
                    seed_to_broadcast = server_args.random_seed
                else:
                    seed_to_broadcast = 0

                self.random_seed = broadcast_one_to_all(seed_to_broadcast).item()
        else:
            self.random_seed = server_args.random_seed

        # init model runner
        self.model_runner = ModelRunner(
            model_config=self.model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            tp_size=server_args.tp_size,
            server_args=server_args,
            mesh=self.mesh,
            req_to_token_pool=req_to_token_pool,
            rngs=nnx.Rngs(self.random_seed),
        )

        # set infer devices
        self.device = server_args.device

        # Profile number of tokens
        self.max_total_num_tokens = self.model_runner.max_total_num_tokens
        self.max_prefill_tokens = server_args.max_prefill_tokens
        self.chunked_prefill_size = server_args.chunked_prefill_size

        # Calculate max_running_requests from different constraints
        attn_backend_limit = self.model_runner.attn_backend.get_max_running_reqests(
            self.model_config.context_len, self.page_size
        )
        server_limit = (
            self.max_total_num_tokens // 2
            if server_args.max_running_requests is None
            else server_args.max_running_requests
        )
        pool_limit = self.model_runner.req_to_token_pool.size
        constraints = [server_limit, pool_limit, attn_backend_limit]
        self.max_running_requests = min(constraints)
        # Log each constraint for debugging
        logger.info(f"Max running requests constraints:")
        logger.info(
            f"  - Server limit: {server_limit} {'(max_total_tokens//2)' if server_args.max_running_requests is None else '(configured)'}"
        )
        logger.info(f"  - Token pool size: {pool_limit}")
        logger.info(
            f"  - Attention backend: {attn_backend_limit} (context_len={self.model_config.context_len}, page_size={self.page_size})"
        )
        logger.info(f"  → Final max_running_requests: {self.max_running_requests}")
        assert self.max_running_requests > 0, "max_running_request is zero"

        self.max_req_len = min(
            self.model_config.context_len - 1,
            self.max_total_num_tokens - 1,
        )
        self.max_req_input_len = self.max_req_len - 5
        assert (
            self.max_req_len > 0 and self.max_req_input_len > 0
        ), "Memory pool size is too small"

        # Sync random seed across TP workers
        # Each process may have different random_seed. After broadcast, all processes will have the same random_seed.
        # self.random_seed = broadcast_one_to_all(server_args.random_seed).item()

        # A reference make this class has the same member as TpModelWorkerClient
        self.worker = self

        self.max_padded_batch_size, self.max_padded_num_tokens = (
            self.get_max_padded_size()
        )

        # precompile
        self.precompile_token_paddings = server_args.precompile_token_paddings

        # normalize server_args.precompile_token_paddings
        # ensure every token padding value is not less than max_runnig_requests
        self.normalize_token_paddings()

        bs_padding_list = (
            server_args.precompile_bs_paddings
            if server_args.precompile_bs_paddings is not None
            else PRECOMPILE_DEFAULT_BS_PADDINGS
        )
        self.precompile_bs_paddings = []
        for bs in bs_padding_list:
            if bs <= self.max_padded_batch_size:
                self.precompile_bs_paddings.append(bs)
        self.precompile_bs_paddings.sort()
        if (
            len(self.precompile_bs_paddings) == 0
            or self.precompile_bs_paddings[-1] < self.max_padded_batch_size
        ):
            self.precompile_bs_paddings.append(self.max_padded_batch_size)

        # padding cache_loc_paddings
        # note: the length of following two cache_loc_paddings must keep the same to length of separate bs_paddings.
        self.precompile_cache_loc_paddings = [
            item * self.max_req_len for item in self.precompile_bs_paddings
        ]

    def normalize_token_paddings(self):
        normalized_token_paddings = []

        if self.precompile_token_paddings is None:
            self.precompile_token_paddings = PRECOMPILE_DEFAULT_TOKEN_PADDINGS
        for item in self.precompile_token_paddings:
            if (
                item >= self.max_padded_batch_size
                and item <= self.max_padded_num_tokens
            ):
                normalized_token_paddings.append(item)

        normalized_token_paddings.sort()
        if (
            len(normalized_token_paddings) == 0
            or normalized_token_paddings[-1] < self.max_padded_num_tokens
        ):
            normalized_token_paddings.append(self.max_padded_num_tokens)

        self.precompile_token_paddings = normalized_token_paddings

    def run_gmm_auto_tune(self):
        start_time = time.perf_counter()
        logger.info("[GMM AUTO-TUNE] Starting GMM tiling parameter auto-tuning")

        try:
            hf_config = self.model_config.hf_config

            # check if this is a MoE model
            num_experts = getattr(hf_config, "num_experts", None)
            if num_experts is None:
                return

            logger.info(
                f"[GMM AUTO-TUNE] MoE model detected with {num_experts} experts"
            )

            hidden_size = getattr(hf_config, "hidden_size", 4096)
            intermediate_size = getattr(hf_config, "intermediate_size", None)
            moe_intermediate_size = getattr(hf_config, "moe_intermediate_size", None)

            target_intermediate_size = (
                moe_intermediate_size or intermediate_size or hidden_size * 4
            )

            logger.info(
                f"[GMM AUTO-TUNE] Model config: hidden_size={hidden_size}, "
                f"intermediate_size={target_intermediate_size}, num_experts={num_experts}"
            )

            num_experts_per_tok = getattr(hf_config, "num_experts_per_tok", 8)
            # Generate all combinations: m = batch_size * seq_length * num_experts_per_tok
            logger.info(
                f"[GMM AUTO-TUNE] Using precompile parameters for shape generation"
            )
            logger.info(
                f"[GMM AUTO-TUNE] Batch size paddings: {self.precompile_bs_paddings}"
            )
            logger.info(
                f"[GMM AUTO-TUNE] Sequence length paddings: {self.precompile_token_paddings}"
            )
            logger.info(f"[GMM AUTO-TUNE] Experts per token: {num_experts_per_tok}")
            logger.info(
                f"[GMM AUTO-TUNE] Max padded num tokens: {self.max_padded_num_tokens}"
            )

            shapes = []
            skipped_count = 0

            logger.info(f"[GMM AUTO-TUNE] Adding decode-specific shapes (seq_length=1)")
            for batch_size in self.precompile_bs_paddings:
                seq_length = 1
                actual_tokens = batch_size * seq_length  # = batch_size

                if actual_tokens > self.max_padded_num_tokens:
                    continue

                # Decode: m = batch_size * num_experts_per_tok
                m = actual_tokens * num_experts_per_tok

                if m < 8 or hidden_size < 64 or target_intermediate_size < 64:
                    continue

                # Add shapes for decode gate/up projections (hidden -> intermediate)
                shapes.append((m, hidden_size, target_intermediate_size, num_experts))
                # Add shapes for decode down projections (intermediate -> hidden)
                shapes.append((m, target_intermediate_size, hidden_size, num_experts))

            logger.info(f"[GMM AUTO-TUNE] Adding prefill shapes (variable seq_length)")
            total_combinations = len(self.precompile_bs_paddings) + (
                len(self.precompile_bs_paddings) * len(self.precompile_token_paddings)
            )
            for batch_size in self.precompile_bs_paddings:
                for seq_length in self.precompile_token_paddings:
                    actual_tokens = batch_size * seq_length

                    if actual_tokens > self.max_padded_num_tokens:
                        skipped_count += 1
                        logger.debug(
                            f"[GMM AUTO-TUNE] Skipping shape bs={batch_size}, seq={seq_length} -> actual_tokens={actual_tokens} (exceeds max_padded_num_tokens={self.max_padded_num_tokens})"
                        )
                        continue

                    m = actual_tokens * num_experts_per_tok

                    if m < 64 or hidden_size < 64 or target_intermediate_size < 64:
                        skipped_count += 1
                        logger.debug(
                            f"[GMM AUTO-TUNE] Skipping tiny shape bs={batch_size}, seq={seq_length} -> m={m}, k={hidden_size}, n={target_intermediate_size}"
                        )
                        continue

                    shapes.append(
                        (m, hidden_size, target_intermediate_size, num_experts)
                    )

                    shapes.append(
                        (m, target_intermediate_size, hidden_size, num_experts)
                    )

            logger.info(
                f"[GMM AUTO-TUNE] Generated {len(shapes)} shape configurations from {total_combinations} combinations "
                f"(skipped {skipped_count} due to size constraints)"
            )
            if not shapes:
                logger.warning("[GMM AUTO-TUNE] No valid shapes to tune, skipping")
                return

            cache_dir = get_default_cache_dir()
            tuner = TilingAutoTuner(cache_dir=cache_dir)
            logger.info(f"[GMM AUTO-TUNE] Using cache directory: {cache_dir}")

            results = {}
            total_shapes = len(shapes)
            logger.info(f"[GMM AUTO-TUNE] Tuning {total_shapes} shape configurations")

            with tqdm(shapes, desc="[GMM AUTO-TUNE] Progress", leave=False) as pbar:
                for i, (m, k, n, num_groups) in enumerate(pbar):
                    pbar.set_postfix(shape=f"m={m},k={k},n={n},g={num_groups}")

                    optimal_tiling = tuner.tune_for_target_size(
                        m, k, n, num_groups, use_cache=True
                    )
                    cache_key = tuner._get_cache_key(m, k, n, num_groups)
                    results[cache_key] = optimal_tiling

            end_time = time.perf_counter()
            logger.info(
                f"[GMM AUTO-TUNE] Completed in {end_time - start_time:.1f} seconds"
            )
            logger.info(
                f"[GMM AUTO-TUNE] Successfully tuned {len(results)}/{total_shapes} configurations"
            )

            # Load all GMM tiling configurations into memory for fast access
            logger.info("[GMM AUTO-TUNE] Loading tiling configurations into memory")
            self.model_runner.load_all_gmm_tiling_configs()

        except Exception as e:
            logger.error(f"[GMM AUTO-TUNE] Auto-tuning failed: {e}")
            logger.info(
                "[GMM AUTO-TUNE] Continuing without auto-tuning, will use default tiling parameters"
            )

    def run_precompile(self):
        self.precompile_extend()
        self.precompile_decode()

    def precompile_extend(self):
        start_time = time.perf_counter()
        logger.info(
            f"[EXTEND] Begin to precompile bs_paddings={self.precompile_bs_paddings[-1:]} token_paddings={self.precompile_token_paddings}"
        )

        bs, _ = self.get_max_padded_size()
        pairs = list(itertools.product([bs], self.precompile_token_paddings))

        with tqdm(pairs, desc="[EXTEND] PRECOMPILE", leave=False) as pbar:
            for pair in pbar:
                pair = list(pair)
                bs, num_tokens = pair[0], pair[1]
                pbar.set_postfix(bs=bs, tokens=num_tokens)
                if bs > num_tokens:
                    logger.warning(f"{bs=} > {num_tokens=}, skip this pair")
                    continue
                model_worker_batch = self.generate_model_worker_batch(
                    bs,
                    num_tokens,
                    ForwardMode.EXTEND,
                    self.precompile_cache_loc_paddings[-1],
                )
                self.forward_batch_generation(model_worker_batch, None, True)

        end_time = time.perf_counter()
        logger.info("[EXTEND] Precompile finished in %.0f secs", end_time - start_time)

    def precompile_decode(self):
        start_time = time.perf_counter()
        logger.info(
            f"[DECODE] Begin to precompile bs_paddings={self.precompile_bs_paddings}"
        )

        with tqdm(
            self.precompile_bs_paddings, desc="[DECODE] PRECOMPILE", leave=False
        ) as pbar:
            for bs in pbar:
                pbar.set_postfix(bs=bs)
                model_worker_batch = self.generate_model_worker_batch(
                    bs,
                    bs,
                    ForwardMode.DECODE,
                    bs * self.max_req_len,
                )
                sampling_metadata = SamplingMetadata.from_model_worker_batch(
                    model_worker_batch, 0, self.mesh
                )
                self.forward_batch_generation(
                    model_worker_batch, None, False, sampling_metadata
                )

        end_time = time.perf_counter()
        logger.info("[DECODE] Precompile finished in %.0f secs", end_time - start_time)

    def get_max_padded_size(self):
        """Calculate the max padded batch size and token nums.

        Returns:
            tuple: (max_padded_batch_size, max_padded_num_tokens)
                - max_padded_batch_size: Maximum batch size, constrained by max_running_requests
                - max_padded_num_tokens: Maximum tokens, using chunked_prefill_size if enabled
        """
        # Use chunked prefill size if enabled (> 0), otherwise use max prefill tokens
        # Take minimum with max_prefill_tokens as upper bound
        max_padded_num_tokens = self.max_prefill_tokens
        if (
            self.chunked_prefill_size > 0
            and max_padded_num_tokens > self.chunked_prefill_size
        ):
            max_padded_num_tokens = self.chunked_prefill_size

        # Batch size is constrained by both max_running_requests and available tokens divide by page_size
        max_padded_batch_size = min(self.max_running_requests, max_padded_num_tokens)

        return max_padded_batch_size, max_padded_num_tokens

    def get_precompile_paddings(self):
        return (
            self.precompile_token_paddings,
            self.precompile_bs_paddings,
            self.precompile_cache_loc_paddings,
        )

    def generate_model_worker_batch(
        self,
        bs: int,
        num_tokens: int,
        mode: ForwardMode,
        max_cache_loc_size: int,
    ) -> ModelWorkerBatch:
        valid_input_ids = np.array([1] * bs, dtype=jnp.int32)
        invalid_input_ids = np.array([0] * (num_tokens - bs), dtype=jnp.int32)
        valid_out_cache_loc = np.arange(bs, dtype=jnp.int32)
        invalid_out_cache_loc = np.array([-1] * (num_tokens - bs), dtype=jnp.int32)
        valid_positions = np.array([0] * bs, dtype=jnp.int32)
        invalid_positions = np.array([0] * (num_tokens - bs), dtype=jnp.int32)
        invalid_cache_loc_size = max_cache_loc_size - bs
        if invalid_cache_loc_size < 0:
            raise ValueError(f"padding cache_loc_size {invalid_cache_loc_size} < 0!")

        valid_cache_loc = np.arange(bs)
        invalid_cache_loc = np.array([0] * (invalid_cache_loc_size), dtype=jnp.int32)

        return ModelWorkerBatch(
            bid=1,
            forward_mode=mode,
            input_ids=np.concat([valid_input_ids, invalid_input_ids], axis=0),
            real_input_ids_len=len(valid_input_ids),
            real_bs=bs,
            req_pool_indices=np.arange(bs, dtype=np.int32),
            seq_lens=np.array([1] * bs, dtype=np.int32),
            out_cache_loc=np.concat(
                [valid_out_cache_loc, invalid_out_cache_loc], axis=0
            ),
            return_logprob=False,
            sampling_info=SamplingBatchInfo.generate_for_precompile(
                bs,
            ),
            extend_input_logprob_token_ids=None,
            positions=np.concat([valid_positions, invalid_positions], axis=0),
            extend_start_loc=np.arange(bs, dtype=np.int64),
            cache_loc=np.concat([valid_cache_loc, invalid_cache_loc], axis=0),
            extend_prefix_lens=(
                np.array([0] * bs) if mode == ForwardMode.EXTEND else None
            ),
            extend_seq_lens=np.array([1] * bs) if mode == ForwardMode.EXTEND else None,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            extend_logprob_start_lens=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

    def get_model_runner(self):
        return self.model_runner

    def get_worker_info(self):
        return (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            global_server_args_dict,
            self.model_runner.req_to_token_pool.size,
            self.model_runner.req_to_token_pool.max_context_len,
            self.model_runner.token_to_kv_pool.size,
        )

    def get_pad_input_ids_func(self):
        return getattr(self.model_runner.model, "pad_input_ids", None)

    def get_memory_pool(self):
        return (
            self.model_runner.req_to_token_pool,
            self.model_runner.token_to_kv_pool_allocator,
        )

    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        launch_done: Optional[threading.Event] = None,
        skip_sample: bool = False,
        sampling_metadata: SamplingMetadata = None,
        forward_metadata=None,
    ) -> Tuple[Union[LogitsProcessorOutput, jax.Array, int], Optional[jax.Array]]:
        # Use pre-initialized ForwardBatch if available (for overlap scheduling optimization)
        if model_worker_batch.forward_batch is not None:
            forward_batch = model_worker_batch.forward_batch
        else:
            forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)

        if forward_metadata is None:
            forward_metadata = (
                self.worker.model_runner.attn_backend.get_forward_metadata(
                    model_worker_batch, self.mesh
                )
            )

        self.model_runner.attn_backend.forward_metadata = forward_metadata
        logits_output, cache_miss_count = self.model_runner.forward(
            forward_batch,
            logits_metadata=LogitsMetadata.from_model_worker_batch(
                model_worker_batch, self.mesh
            ),
        )

        if launch_done is not None:
            launch_done.set()

        sample_cache_miss_count = 0
        if skip_sample:
            next_token_ids_device = None
        else:
            import jax._src.test_util as jtu

            with jtu.count_pjit_cpp_cache_miss() as count:
                next_token_ids_device = self.model_runner.sample(
                    logits_output, sampling_metadata
                )
                sample_cache_miss_count = count()

        return (
            logits_output,
            next_token_ids_device,
            cache_miss_count + sample_cache_miss_count,
        )


class MockModelWorker:
    """A mock tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
    ):
        # Parse args
        self.tp_size = server_args.tp_size

        # Init model and tokenizer
        self.model_config = ModelConfig.from_server_args(
            server_args,
            model_path=server_args.model_path,
        )

        # Sync random seed across TP workers
        # Each process may have different random_seed. After broadcast, all processes will have the same random_seed.
        self.random_seed = broadcast_one_to_all(server_args.random_seed).item()

        # init model runner
        self.model_runner = MockModelRunner(
            model_config=self.model_config,
            rngs=jax.random.PRNGKey(self.random_seed),
            server_args=server_args,
        )

        # Profile number of tokens
        self.max_total_num_tokens = self.model_runner.max_total_num_tokens
        self.max_prefill_tokens = server_args.max_prefill_tokens
        self.max_running_requests = min(
            (
                self.max_total_num_tokens // 2
                if server_args.max_running_requests is None
                else server_args.max_running_requests
            ),
            self.model_runner.req_to_token_pool.size,
        )
        assert self.max_running_requests > 0, "max_running_request is zero"
        self.max_req_len = min(
            self.model_config.context_len - 1,
            self.max_total_num_tokens - 1,
        )
        self.max_req_input_len = self.max_req_len - 5
        assert (
            self.max_req_len > 0 and self.max_req_input_len > 0
        ), "Memory pool size is too small"

        # A reference make this class has the same member as TpModelWorkerClient
        self.worker = self

    def get_worker_info(self):
        return (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            global_server_args_dict,
            self.model_runner.req_to_token_pool.size,
            self.model_runner.req_to_token_pool.max_context_len,
            self.model_runner.token_to_kv_pool.size,
        )

    def get_memory_pool(self):
        return (self.model_runner.req_to_token_pool, self.model_runner.token_to_kv_pool)

    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        launch_done: Optional[threading.Event] = None,
        skip_sample: bool = False,
    ) -> Tuple[Union[LogitsProcessorOutput, jax.Array], Optional[jax.Array]]:
        return (
            LogitsProcessorOutput(
                next_token_logits=jnp.array([0, 1]),
            ),
            None,
        )
