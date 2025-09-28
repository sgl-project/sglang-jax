"""A tensor parallel worker."""

import itertools
import logging
import threading
import time
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.experimental.multihost_utils import broadcast_one_to_all
from jax.sharding import NamedSharding, PartitionSpec
from tqdm import tqdm

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
from sgl_jax.srt.managers.schedule_batch import (
    ModelWorkerBatch,
    global_server_args_dict,
)
from sgl_jax.srt.managers.utils import resolve_future_token_ids, set_future_token_ids
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
from sgl_jax.srt.utils.jax_utils import device_array

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
            (item * self.max_req_len + self.page_size - 1)
            // self.page_size
            * self.page_size
            for item in self.precompile_bs_paddings
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

    def run_precompile(self, future_token_ids_map=None):
        self.precompile_extend(future_token_ids_map)
        self.precompile_decode(future_token_ids_map)

    def precompile_extend(self, future_token_ids_map=None):
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
                sampling_metadata = SamplingMetadata.from_model_worker_batch(
                    model_worker_batch, 0, self.mesh
                )
                model_worker_batch.forward_batch = ForwardBatch.init_new(
                    model_worker_batch, self.model_runner
                )
                if future_token_ids_map is not None:
                    model_worker_batch.forward_batch.input_ids = (
                        resolve_future_token_ids(
                            model_worker_batch.forward_batch.input_ids,
                            future_token_ids_map,
                        )
                    )

                self.forward_batch_generation(
                    model_worker_batch, None, False, sampling_metadata
                )
        end_time = time.perf_counter()
        logger.info("[EXTEND] Precompile finished in %.0f secs", end_time - start_time)

    def precompile_decode(self, future_token_ids_map=None):
        start_time = time.perf_counter()
        logger.info(
            f"[DECODE] Begin to precompile bs_paddings={self.precompile_bs_paddings}"
        )

        with tqdm(
            self.precompile_bs_paddings, desc="[DECODE] PRECOMPILE", leave=False
        ) as pbar:
            for bs in pbar:
                pbar.set_postfix(bs=bs)
                # use same page aligned with precompile cache_loc_paddings
                aligned_cache_loc_size = (
                    (bs * self.max_req_len + self.page_size - 1)
                    // self.page_size
                    * self.page_size
                )
                model_worker_batch = self.generate_model_worker_batch(
                    bs,
                    bs,
                    ForwardMode.DECODE,
                    aligned_cache_loc_size,
                )
                sampling_metadata = SamplingMetadata.from_model_worker_batch(
                    model_worker_batch, 0, self.mesh
                )
                model_worker_batch.forward_batch = ForwardBatch.init_new(
                    model_worker_batch, self.model_runner
                )
                if future_token_ids_map is not None:
                    model_worker_batch.forward_batch.input_ids = (
                        resolve_future_token_ids(
                            model_worker_batch.forward_batch.input_ids,
                            future_token_ids_map,
                        )
                    )
                _, next_token_ids, _ = self.forward_batch_generation(
                    model_worker_batch, None, False, sampling_metadata
                )
                if future_token_ids_map is not None:
                    set_future_token_ids(future_token_ids_map, 0, next_token_ids)

        end_time = time.perf_counter()
        logger.info("[DECODE] Precompile finished in %.0f secs", end_time - start_time)

    def set_forward_metadata(self, model_worker_batch: ModelWorkerBatch):
        self.model_runner.attn_backend.forward_metadata = (
            self.worker.model_runner.attn_backend.get_forward_metadata(
                model_worker_batch
            )
        )

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
                    model_worker_batch
                )
            )

        self.model_runner.attn_backend.forward_metadata = forward_metadata
        # note: put positions on devices again because the forward_batch has been donated
        if not skip_sample:
            positions = (
                model_worker_batch.positions
                if model_worker_batch.forward_mode.is_decode()
                else model_worker_batch.seq_lens - 1
            )
            positions_device = device_array(
                positions,
                sharding=(
                    NamedSharding(self.model_runner.mesh, PartitionSpec())
                    if jax.process_count() == 1
                    else None
                ),
            )
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
                    logits_output,
                    sampling_metadata,
                    positions_device,
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
