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

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
from sgl_jax.srt.managers.schedule_batch import (
    ModelWorkerBatch,
    global_server_args_dict,
)
from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.model_executor.model_runner import MockModelRunner, ModelRunner
from sgl_jax.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.utils.common_utils import (
    JAX_PRECOMPILE_DEFAULT_DECODE_BS_PADDINGS,
    JAX_PRECOMPILE_DEFAULT_PREFILL_TOKEN_PADDINGS,
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

        # Sync random seed across TP workers
        # Each process may have different random_seed. After broadcast, all processes will have the same random_seed.
        # self.random_seed = broadcast_one_to_all(server_args.random_seed).item()

        # A reference make this class has the same member as TpModelWorkerClient
        self.worker = self

        # normalize server_args.jax_precompile_prefill_token_paddings
        # ensure every token padding value is not less than max_runnig_requests
        self.normalize_token_paddings(server_args, self.max_running_requests)

        # precompile
        self.precompile_prefill_token_paddings = (
            server_args.jax_precompile_prefill_token_paddings
        )
        self.precompile_decode_bs_paddings = (
            server_args.jax_precompile_decode_bs_paddings
            if server_args.jax_precompile_decode_bs_paddings is not None
            else JAX_PRECOMPILE_DEFAULT_DECODE_BS_PADDINGS
        )

    def normalize_token_paddings(
        self, server_args: ServerArgs, max_running_requests: int
    ):
        normalized_token_paddings = []
        if server_args.jax_precompile_prefill_token_paddings is None:
            server_args.jax_precompile_prefill_token_paddings = (
                JAX_PRECOMPILE_DEFAULT_PREFILL_TOKEN_PADDINGS
            )
        for item in server_args.jax_precompile_prefill_token_paddings:
            if item >= self.max_running_requests:
                normalized_token_paddings.append(item)
        if len(normalized_token_paddings) == 0:
            normalized_token_paddings.append(self.max_running_requests)
            logger.warning(
                f"Every padding of {server_args.jax_precompile_prefill_token_paddings=} is less than {max_running_requests=}, so set token_paddings as {normalized_token_paddings}"
            )

        server_args.jax_precompile_prefill_token_paddings = normalized_token_paddings

    def run_precompile(self):
        self.precompile_extend()
        self.precompile_decode()

    def precompile_extend(self):
        start_time = time.perf_counter()
        logger.info(f"[EXTEND] begin to precompile")
        for pair in itertools.product(
            [self.max_running_requests], self.precompile_prefill_token_paddings
        ):
            pair = list(pair)
            bs, num_tokens = pair[0], pair[1]
            logger.info(f"[EXTEND] precompile ({bs=}, {num_tokens=})")
            if bs > num_tokens:
                logger.warning(f"{bs=} > {num_tokens=}, skip this pair")
                continue
            model_worker_batch = self.generate_model_worker_batch(
                bs, num_tokens, ForwardMode.EXTEND
            )
            self.forward_batch_generation(model_worker_batch=model_worker_batch)

        end_time = time.perf_counter()
        logger.info("[EXTEND] precompile finished in %.0f secs", end_time - start_time)

    def generate_model_worker_batch(
        self, bs: int, num_tokens: int, mode: ForwardMode
    ) -> ModelWorkerBatch:
        valid_input_ids = np.array([1] * bs, dtype=jnp.int32)
        invalid_input_ids = np.array([0] * (num_tokens - bs), dtype=jnp.int32)
        valid_out_cache_loc = np.arange(bs, dtype=jnp.int32)
        invalid_out_cache_loc = np.array([-1] * (num_tokens - bs), dtype=jnp.int32)
        valid_positions = np.array([0] * bs, dtype=jnp.int32)
        invalid_positions = np.array([0] * (num_tokens - bs), dtype=jnp.int32)
        if mode == ForwardMode.EXTEND:
            valid_cache_loc = np.arange(bs)
            invalid_cache_loc = np.array([0] * (self.max_total_num_tokens - bs))
        else:
            valid_cache_loc = np.arange(bs * 2)
            padding_size = self.max_total_num_tokens - bs * 2
            if padding_size < 0:
                padding_size = bs * 2 - bs * self.max_total_num_tokens
            invalid_cache_loc = np.array([0] * (padding_size))
        return ModelWorkerBatch(
            bid=1,
            forward_mode=mode,
            input_ids=device_array(
                self.mesh, np.concat([valid_input_ids, invalid_input_ids], axis=0)
            ),
            real_input_ids_len=len(valid_input_ids),
            real_bs=bs,
            req_pool_indices=device_array(self.mesh, np.arange(bs, dtype=jnp.int32)),
            seq_lens=device_array(self.mesh, np.array([1] * bs, dtype=jnp.int32)),
            out_cache_loc=device_array(
                self.mesh,
                np.concat([valid_out_cache_loc, invalid_out_cache_loc], axis=0),
            ),
            return_logprob=False,
            sampling_info=SamplingBatchInfo.generate_for_precompile(
                bs, self.model_config.vocab_size, self.mesh
            ),
            extend_input_logprob_token_ids=None,
            positions=device_array(
                self.mesh, np.concat([valid_positions, invalid_positions], axis=0)
            ),
            extend_start_loc=device_array(self.mesh, np.arange(bs, dtype=jnp.int64)),
            cache_loc=device_array(
                self.mesh, np.concat([valid_cache_loc, invalid_cache_loc], axis=0)
            ),
            extend_prefix_lens=(
                device_array(self.mesh, np.array([0] * bs))
                if mode == ForwardMode.EXTEND
                else None
            ),
            extend_seq_lens=(
                device_array(self.mesh, np.array([1] * bs))
                if mode == ForwardMode.EXTEND
                else None
            ),
        )

    def precompile_decode(self):
        start_time = time.perf_counter()
        logger.info(f"[DECODE] begin to precompile")
        for bs in self.precompile_decode_bs_paddings:
            logger.info(f"[DECODE] precompile ({bs=})")
            model_worker_batch = self.generate_model_worker_batch(
                bs, bs, ForwardMode.DECODE
            )
            self.forward_batch_generation(model_worker_batch=model_worker_batch)

        end_time = time.perf_counter()
        logger.info("[DECODE] precompile finished in %.0f secs", end_time - start_time)

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
            # TODO: model_runner.token_to_kv_pool define in ModelRunner.init_memory_pool
            self.model_runner.token_to_kv_pool.size,
        )

    def get_tp_group(self):
        return self.model_runner.tp_group

    def get_pad_input_ids_func(self):
        # TODO: model_runner.model define in ModelRunner.load_model
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
    ) -> Tuple[Union[LogitsProcessorOutput, jax.Array, int], Optional[jax.Array]]:
        print(f"[TP_WORKER] forward_batch_generation called")
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        print(f"[TP_WORKER] ForwardBatch initialized")

        cache_miss_count = 0
        import jax._src.test_util as jtu

        print(f"[TP_WORKER] About to call model_runner.forward")
        with jtu.count_pjit_cpp_cache_miss() as count:
            logits_output = self.model_runner.forward(forward_batch)
            cache_miss_count = count()
        print(f"[TP_WORKER] model_runner.forward completed, cache_miss_count={cache_miss_count}")
        
        # 检查 logits_output 的健康状态  
        if hasattr(logits_output, 'next_token_logits') and logits_output.next_token_logits is not None:
            logits_sum = jnp.sum(logits_output.next_token_logits)
            logits_has_nan = jnp.any(jnp.isnan(logits_output.next_token_logits))
            logits_has_inf = jnp.any(jnp.isinf(logits_output.next_token_logits))
            
            jax.debug.print(
                "[TP_WORKER] Logits health check - sum={sum}, has_nan={has_nan}, has_inf={has_inf}, shape={shape}",
                sum=logits_sum,
                has_nan=logits_has_nan,
                has_inf=logits_has_inf,
                shape=logits_output.next_token_logits.shape
            )

        if launch_done is not None:
            print(f"[TP_WORKER] Setting launch_done")
            launch_done.set()
            print(f"[TP_WORKER] launch_done set")

        print(f"[TP_WORKER] Checking skip_sample={skip_sample}")
        if skip_sample:
            print(f"[TP_WORKER] Skipping sampling")
            next_token_ids = None
        else:
            print(f"[TP_WORKER] Calling model_runner.sample")
            next_token_ids = self.model_runner.sample(logits_output, model_worker_batch)
            print(f"[TP_WORKER] Sampling completed")

        print(f"[TP_WORKER] Processing indices")
        print(f"[TP_WORKER] model_worker_batch.real_bs: {model_worker_batch.real_bs}")
        print(f"[TP_WORKER] model_worker_batch.forward_mode: {model_worker_batch.forward_mode}")
        
        print(f"[TP_WORKER] Getting extend_start_loc")
        idx = model_worker_batch.extend_start_loc[: model_worker_batch.real_bs]
        print(f"[TP_WORKER] extend_start_loc obtained")
        
        if model_worker_batch.forward_mode == ForwardMode.EXTEND:
            print(f"[TP_WORKER] Computing cumsum for EXTEND mode")
            print(f"[TP_WORKER] extend_seq_lens shape: {model_worker_batch.extend_seq_lens.shape}")
            extend_seq_lens_slice = model_worker_batch.extend_seq_lens[: model_worker_batch.real_bs]
            print(f"[TP_WORKER] extend_seq_lens sliced")
            extend_seq_lens_minus_1 = extend_seq_lens_slice - 1
            print(f"[TP_WORKER] extend_seq_lens minus 1 computed")
            idx = np.cumsum(extend_seq_lens_minus_1)
            print(f"[TP_WORKER] cumsum computed, idx.shape: {idx.shape}, idx: {idx}")
        
        print(f"[TP_WORKER] idx computation completed")

        print(f"[TP_WORKER] Preparing return values")
        print(f"[TP_WORKER] next_token_ids type: {type(next_token_ids)}")
        if next_token_ids is not None:
            print(f"[TP_WORKER] next_token_ids.shape: {next_token_ids.shape}")
            print(f"[TP_WORKER] next_token_ids.dtype: {next_token_ids.dtype}")
            print(f"[TP_WORKER] idx: {idx}")
            
        truncated_logits = logits_output.truncate_logits_processor_output(idx)
        print(f"[TP_WORKER] Logits truncated")
        
        if next_token_ids is not None:
            print(f"[TP_WORKER] About to index next_token_ids")
            indexed_next_token_ids = next_token_ids[idx]
            print(f"[TP_WORKER] next_token_ids indexed, shape: {indexed_next_token_ids.shape}")
        else:
            indexed_next_token_ids = None
            
        print(f"[TP_WORKER] Returning final result")
        return (
            truncated_logits,
            indexed_next_token_ids,
            cache_miss_count,
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
        # TODO: model_runner.max_total_num_tokens define in ModelRunner.init_memory_pool
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
            # TODO: model_runner.token_to_kv_pool define in ModelRunner.init_memory_pool
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
