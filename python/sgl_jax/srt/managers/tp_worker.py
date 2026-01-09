"""A tensor parallel worker."""

import itertools
import logging
import os
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.experimental.multihost_utils import broadcast_one_to_all
from tqdm import tqdm

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.constrained.bitmask_ops import allocate_token_bitmask
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
from sgl_jax.srt.managers.schedule_batch import (
    ModelWorkerBatch,
    ModelWorkerSamplingInfo,
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

logger = logging.getLogger(__name__)


class ModelWorker:
    """A tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        mesh: jax.sharding.Mesh,
        req_to_token_pool: ReqToTokenPool | None = None,
        is_draft_worker: bool = False,
    ):
        # Parse args
        self.tp_size = server_args.tp_size
        self.dp_size = server_args.dp_size
        from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.server_args = server_args

        # LoRA configurations
        self.lora_paths = server_args.lora_paths
        self.max_loras_per_batch = server_args.max_loras_per_batch

        # Init model and tokenizer
        self.model_config = ModelConfig.from_server_args(
            server_args,
            model_path=(
                server_args.model_path
                if not is_draft_worker
                else server_args.speculative_draft_model_path
            ),
            model_revision=(
                server_args.revision
                if not is_draft_worker
                else server_args.speculative_draft_model_revision
            ),
            is_draft_model=is_draft_worker,
        )

        self.mesh = mesh
        self.page_size = server_args.page_size

        # need_prepare_lora_batch is False in overlap mode, default is True
        self.need_prepare_lora_batch = True

        # Sync random seed across TP workers
        # Each process may have different random_seed. After broadcast, all processes will have the same random_seed.
        if server_args.random_seed is None:
            with jax.default_device(jax.local_devices()[0]):
                seed_to_broadcast = server_args.random_seed if jax.process_index() == 0 else 0
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
            is_draft_worker=is_draft_worker,
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
        logger.info("Max running requests constraints:")
        logger.info(
            "  - Server limit: %s %s",
            server_limit,
            (
                "(max_total_tokens//2)"
                if server_args.max_running_requests is None
                else "(configured)"
            ),
        )
        logger.info("  - Token pool size: %s", pool_limit)
        logger.info(
            "  - Attention backend: %s (context_len=%s, page_size=%s)",
            attn_backend_limit,
            self.model_config.context_len,
            self.page_size,
        )
        logger.info("  → Final max_running_requests: %s", self.max_running_requests)

        # Validate and adjust max_running_requests for Data Parallelism
        dp_size = server_args.dp_size
        if self.max_running_requests < dp_size:
            raise ValueError(
                f"max_running_requests ({self.max_running_requests}) is less than dp_size ({dp_size}). "
                f"Please increase memory allocation or reduce dp_size."
            )
        if self.max_running_requests % dp_size != 0:
            original_value = self.max_running_requests
            self.max_running_requests = (self.max_running_requests // dp_size) * dp_size
            logger.warning(
                "Adjusted max_running_requests from %s to %s to be divisible by dp_size (%s)",
                original_value,
                self.max_running_requests,
                dp_size,
            )

        assert self.max_running_requests > 0, "max_running_request is zero"

        self.max_req_len = min(
            self.model_config.context_len - 1,
            self.max_total_num_tokens - 1,
        )
        self.max_req_input_len = self.max_req_len - 5
        assert self.max_req_len > 0 and self.max_req_input_len > 0, "Memory pool size is too small"

        # Sync random seed across TP workers
        # Each process may have different random_seed. After broadcast, all processes will have the same random_seed.
        # self.random_seed = broadcast_one_to_all(server_args.random_seed).item()

        # A reference make this class has the same member as TpModelWorkerClient
        self.worker = self

        self.max_padded_batch_size, self.max_padded_num_tokens = self.get_max_padded_size()

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
            # Ensure bs >= dp_size to avoid runtime errors in DP mode
            if bs <= self.max_padded_batch_size and bs >= self.dp_size:
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
            (item * self.max_req_len + self.page_size - 1) // self.page_size * self.page_size
            for item in self.precompile_bs_paddings
        ]

    def normalize_token_paddings(self):
        normalized_token_paddings = []
        dp_size = self.dp_size

        if self.precompile_token_paddings is None:
            # Multiply default token paddings by dp_size for DP mode
            self.precompile_token_paddings = [
                item * dp_size for item in PRECOMPILE_DEFAULT_TOKEN_PADDINGS
            ]

        for item in self.precompile_token_paddings:
            # Ensure item is divisible by dp_size
            if item % dp_size != 0:
                item = (item // dp_size) * dp_size
            if (
                item >= self.max_padded_batch_size
                and item <= self.max_padded_num_tokens
                and item >= dp_size
            ):
                normalized_token_paddings.append(item)

        normalized_token_paddings.sort()
        if (
            len(normalized_token_paddings) == 0
            or normalized_token_paddings[-1] < self.max_padded_num_tokens
        ):
            # max_padded_num_tokens is already multiplied by dp_size in get_max_padded_size
            normalized_token_paddings.append(self.max_padded_num_tokens)

        self.precompile_token_paddings = normalized_token_paddings

    def run_precompile(self, future_token_ids_map=None):
        self.precompile_extend(future_token_ids_map)
        self.precompile_decode(future_token_ids_map)

    def precompile_extend(self, future_token_ids_map=None):
        start_time = time.perf_counter()
        logger.info(
            "[EXTEND] Begin to precompile bs_paddings=%s token_paddings=%s",
            self.precompile_bs_paddings[-1:],
            self.precompile_token_paddings,
        )

        bs, _ = self.get_max_padded_size()
        pairs = list(itertools.product([bs], self.precompile_token_paddings))

        with tqdm(pairs, desc="[EXTEND] PRECOMPILE", leave=False) as pbar:
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
                    enable_static_lora=self.server_args.enable_static_lora,
                    dp_size=self.dp_size,
                    per_dp_bs_size=bs // self.dp_size,
                )
                # Prepare LoRA batch if LoRA is enabled
                if self.server_args.enable_lora:
                    self.prepare_lora_batch(model_worker_batch)
                sampling_metadata = SamplingMetadata.from_model_worker_batch(
                    model_worker_batch,
                    0,
                    self.mesh,
                    self.model_config.vocab_size,
                )
                model_worker_batch.forward_batch = ForwardBatch.init_new(
                    model_worker_batch, self.model_runner
                )
                if future_token_ids_map is not None:
                    model_worker_batch.forward_batch.input_ids = resolve_future_token_ids(
                        model_worker_batch.forward_batch.input_ids,
                        future_token_ids_map,
                    )

                self.forward_batch_generation(model_worker_batch, None, False, sampling_metadata)
        end_time = time.perf_counter()
        logger.info("[EXTEND] Precompile finished in %.0f secs", end_time - start_time)

    def precompile_decode(self, future_token_ids_map=None):
        start_time = time.perf_counter()
        logger.info(
            "[DECODE] Begin to precompile bs_paddings=%s",
            self.precompile_bs_paddings,
        )

        with tqdm(self.precompile_bs_paddings, desc="[DECODE] PRECOMPILE", leave=False) as pbar:
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
                    enable_static_lora=self.server_args.enable_static_lora,
                    dp_size=self.dp_size,
                    per_dp_bs_size=bs // self.dp_size,
                )
                # Prepare LoRA batch if LoRA is enabled
                if self.server_args.enable_lora:
                    self.prepare_lora_batch(model_worker_batch)
                sampling_metadata = SamplingMetadata.from_model_worker_batch(
                    model_worker_batch, 0, self.mesh, self.model_config.vocab_size
                )
                model_worker_batch.forward_batch = ForwardBatch.init_new(
                    model_worker_batch, self.model_runner
                )
                if future_token_ids_map is not None:
                    model_worker_batch.forward_batch.input_ids = resolve_future_token_ids(
                        model_worker_batch.forward_batch.input_ids,
                        future_token_ids_map,
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
            self.worker.model_runner.attn_backend.get_forward_metadata(model_worker_batch)
        )

    def get_max_padded_size(self):
        """Calculate the max padded batch size and token nums.

        Returns:
            tuple: (max_padded_batch_size, max_padded_num_tokens)
                - max_padded_batch_size: Maximum batch size, constrained by max_running_requests
                - max_padded_num_tokens: Maximum tokens for all DP ranks (multiplied by dp_size for prefill)
        """
        # Use chunked prefill size if enabled (> 0), otherwise use max prefill tokens
        # Take minimum with max_prefill_tokens as upper bound
        per_dp_num_tokens = self.max_prefill_tokens
        if self.chunked_prefill_size > 0 and per_dp_num_tokens > self.chunked_prefill_size:
            per_dp_num_tokens = self.chunked_prefill_size

        # For prefill, total tokens = per_dp_tokens * dp_size
        max_padded_num_tokens = per_dp_num_tokens * self.dp_size

        # Batch size is constrained by both max_running_requests and available tokens divide by page_size
        max_padded_batch_size = min(self.max_running_requests, max_padded_num_tokens)

        assert max_padded_batch_size % self.dp_size == 0, (
            "max_padded_batch_size must be divisible by dp_size, "
            f"but got max_padded_batch_size={max_padded_batch_size}, dp_size={self.dp_size}"
        )

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
        do_penalties: bool = False,
        speculative_algotithm=None,
        enable_static_lora: bool = None,
        dp_size: int = 1,
        per_dp_bs_size: int = 0,
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
        lora_ids = ["0"] * bs
        extend_seq_lens = np.array([1] * bs) if mode == ForwardMode.EXTEND else None
        logits_indices = np.array([0] * bs) if mode == ForwardMode.EXTEND else None

        return ModelWorkerBatch(
            bid=1,
            forward_mode=mode,
            input_ids=np.concat([valid_input_ids, invalid_input_ids], axis=0),
            real_input_ids_len=len(valid_input_ids),
            real_bs=bs,
            req_pool_indices=np.arange(bs, dtype=np.int32),
            seq_lens=np.array([1] * bs, dtype=np.int32),
            out_cache_loc=np.concat([valid_out_cache_loc, invalid_out_cache_loc], axis=0),
            return_logprob=False,
            return_output_logprob_only=True,
            sampling_info=(
                ModelWorkerSamplingInfo.generate_for_precompile(bs, self.model_config.vocab_size)
                if speculative_algotithm is None
                else SamplingBatchInfo.generate_for_precompile_all_greedy(
                    bs, self.model_config.vocab_size
                )
            ),
            extend_input_logprob_token_ids=None,
            positions=np.concat([valid_positions, invalid_positions], axis=0),
            cache_loc=np.concat([valid_cache_loc, invalid_cache_loc], axis=0),
            extend_prefix_lens=(np.array([0] * bs) if mode == ForwardMode.EXTEND else None),
            extend_seq_lens=extend_seq_lens,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            extend_logprob_start_lens=None,
            logits_indices=logits_indices,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            spec_algorithm=speculative_algotithm,
            lora_ids=lora_ids,  # Already set to [None] * bs above
            dp_size=dp_size,
            per_dp_bs_size=per_dp_bs_size,
        )

    def get_model_runner(self):
        return self.model_runner

    def prepare_lora_batch(self, model_worker_batch: ModelWorkerBatch):
        self.model_runner.lora_manager.prepare_lora_batch(model_worker_batch)
        if self.model_runner.lora_manager.has_new_weights:
            _, model_state = nnx.split(self.model_runner.model)
            self.model_runner.model_state_leaves, _ = jax.tree_util.tree_flatten(model_state)

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

    @property
    def sliding_window_size(self) -> int | None:
        return self.model_runner.sliding_window_size

    @property
    def is_hybrid(self) -> bool:
        return self.model_runner.is_hybrid

    def get_tokens_per_layer_info(self):
        return (
            self.model_runner.full_max_total_num_tokens,
            self.model_runner.swa_max_total_num_tokens,
        )

    def get_pad_input_ids_func(self):
        return getattr(self.model_runner.model, "pad_input_ids", None)

    def get_memory_pool(self):
        return (
            self.model_runner.req_to_token_pool,
            self.model_runner.token_to_kv_pool_allocator,
        )

    def _update_grammar_vocab_mask(
        self, batch: ModelWorkerBatch, sampling_metadata: SamplingMetadata
    ):
        if batch.sampling_info.grammars:
            # Overlap mode: wait for the mask prepared in set_next_batch_sampling_info_done
            if batch.sampling_info.sampling_info_done:
                batch.sampling_info.sampling_info_done.wait()
            else:
                batch.sampling_info.update_grammar_vocab_mask()
        if batch.sampling_info.vocab_mask is None:
            sampling_metadata.apply_vocab_mask = False
            sampling_metadata.vocab_mask = allocate_token_bitmask(
                len(batch.sampling_info.temperatures), batch.sampling_info.vocab_size
            )
        else:
            sampling_metadata.apply_vocab_mask = True
            sampling_metadata.vocab_mask = batch.sampling_info.vocab_mask

    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        launch_done: threading.Event | None = None,
        skip_sample: bool = False,
        sampling_metadata: SamplingMetadata = None,
        forward_metadata=None,
    ) -> tuple[LogitsProcessorOutput | jax.Array | int, jax.Array | None]:
        # Prepare LoRA batch if LoRA is enabled
        if self.worker.server_args.enable_lora and self.need_prepare_lora_batch:
            self.prepare_lora_batch(model_worker_batch)

        # Use pre-initialized ForwardBatch if available (for overlap scheduling optimization)
        if model_worker_batch.forward_batch is not None:
            forward_batch = model_worker_batch.forward_batch
        else:
            forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)

        if forward_metadata is None:
            forward_metadata = self.worker.model_runner.attn_backend.get_forward_metadata(
                model_worker_batch
            )

        if sampling_metadata is None:
            sampling_metadata = SamplingMetadata.from_model_worker_batch(
                model_worker_batch,
                0,
                self.mesh,
                self.model_config.vocab_size,
            )

        self.model_runner.attn_backend.forward_metadata = forward_metadata
        logits_output, cache_miss_count = self.model_runner.forward(
            forward_batch,
            logits_metadata=LogitsMetadata.from_model_worker_batch(model_worker_batch, self.mesh),
        )
        if launch_done is not None:
            launch_done.set()

        # SAVE last layer logits
        save_logits_file_info = os.getenv("DUMP_LAST_LAYER_LOGITS_FILENAMES", None)
        if save_logits_file_info:
            save_logits_with_txt(
                logits_output.next_token_logits[: model_worker_batch.real_bs, :],
                save_logits_file_info,
                forward_batch.forward_mode,
            )

        if skip_sample:
            next_token_ids_device = None
            new_logits_output = None
        else:
            import jax._src.test_util as jtu

            # Preprocess logits: update grammar vocab masks if needed
            if model_worker_batch.sampling_info:
                self._update_grammar_vocab_mask(model_worker_batch, sampling_metadata)

            with jtu.count_pjit_cpp_cache_miss() as count:
                next_token_ids_device, token_logprobs, new_logits_output = (
                    self.model_runner.sample(  # TODO @Brian: Data-Parallel sharding
                        logits_output,
                        sampling_metadata,
                    )
                )
                cache_miss_count += count()
            if model_worker_batch.return_output_logprob_only:
                logprobs = self.model_runner.compute_logprobs(token_logprobs, next_token_ids_device)
                logits_output.next_token_logprobs = logprobs[: model_worker_batch.real_bs]
        if new_logits_output is not None:
            logits_output = new_logits_output
            if logits_output.next_token_top_logprobs_val is not None:
                logits_output.next_token_top_logprobs_val = (
                    logits_output.next_token_top_logprobs_val.astype(jnp.float32).tolist()
                )
                logits_output.next_token_top_logprobs_idx = (
                    logits_output.next_token_top_logprobs_idx.tolist()
                )
            if logits_output.next_token_token_ids_logprobs_val is not None:
                logits_output.next_token_token_ids_logprobs_val = (
                    logits_output.next_token_token_ids_logprobs_val.astype(jnp.float32).tolist()
                )
                logits_output.next_token_token_ids_logprobs_idx = (
                    logits_output.next_token_token_ids_logprobs_idx.tolist()
                )
            if logits_output.input_token_ids_logprobs_val is not None:
                logits_output.input_token_ids_logprobs_val = (
                    logits_output.input_token_ids_logprobs_val.astype(jnp.float32).tolist()
                )
                logits_output.input_token_ids_logprobs_idx = (
                    logits_output.input_token_ids_logprobs_idx.tolist()
                )
            if logits_output.input_top_logprobs_val is not None:
                logits_output.input_top_logprobs_val = logits_output.input_top_logprobs_val.astype(
                    jnp.float32
                ).tolist()
                logits_output.input_top_logprobs_idx = logits_output.input_top_logprobs_idx.tolist()

        return (
            logits_output,
            next_token_ids_device,
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
        assert self.max_req_len > 0 and self.max_req_input_len > 0, "Memory pool size is too small"

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
        _model_worker_batch: ModelWorkerBatch,
        _launch_done: threading.Event | None = None,
        _skip_sample: bool = False,
        _sampling_metadata: SamplingMetadata | None = None,
    ) -> tuple[LogitsProcessorOutput | jax.Array, jax.Array | None]:
        return (
            LogitsProcessorOutput(
                next_token_logits=jnp.array([0, 1]),
            ),
            None,
        )


def save_logits_with_txt(
    arr: jax.Array,
    file_info: str,
    forward_mode: ForwardMode,
):
    # format: {prefill_file_name},{decode_file_name}
    file_slice = file_info.split(",")
    if forward_mode.is_extend():
        file_name = file_slice[0]
    elif forward_mode.is_decode():
        file_name = file_slice[1]
    else:
        raise ValueError(f"Unsupported {forward_mode} to save logits with txt")

    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    np.savetxt(file_name, np.asarray(jax.device_get(arr)).flatten(), fmt="%.15f")
