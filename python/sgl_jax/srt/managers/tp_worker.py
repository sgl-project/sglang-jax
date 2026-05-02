"""A tensor parallel worker."""

import logging
import os
import signal
import threading
from queue import Queue

import jax
import jax.numpy as jnp
import numpy as np
import psutil
from flax import nnx
from jax.experimental.multihost_utils import broadcast_one_to_all

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.constrained.bitmask_ops import allocate_token_bitmask
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
from sgl_jax.srt.layers.routed_experts_capturer import get_global_experts_capturer
from sgl_jax.srt.managers.schedule_batch import (
    ModelWorkerBatch,
    global_server_args_dict,
)
from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.model_executor.model_runner import MockModelRunner, ModelRunner
from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.utils import get_exception_traceback

logger = logging.getLogger(__name__)


class ModelWorker:
    """A tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        mesh: jax.sharding.Mesh,
        req_to_token_pool: ReqToTokenPool | None = None,
        is_draft_worker: bool = False,
        model_class=None,
        precompile_params: dict | None = None,
    ):
        # Parse args
        self.tp_size = server_args.tp_size
        self.dp_size = server_args.dp_size
        from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.server_args = server_args

        # pre_precompile
        self.precompile_params = precompile_params

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

        self.max_prefill_tokens = server_args.max_prefill_tokens
        self.chunked_prefill_size = server_args.chunked_prefill_size

        # init model runner
        self.model_runner = ModelRunner(
            model_config=self.model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            tp_size=server_args.tp_size,
            dp_size=server_args.dp_size,
            server_args=server_args,
            mesh=self.mesh,
            is_draft_worker=is_draft_worker,
            req_to_token_pool=req_to_token_pool,
            rngs=nnx.Rngs(self.random_seed),
            max_padding=max(self.max_prefill_tokens, self.chunked_prefill_size),
            model_class=model_class,
        )

        # set infer devices
        self.device = server_args.device

        # Profile number of tokens
        self.max_total_num_tokens = self.model_runner.max_total_num_tokens

        # Calculate max_running_requests from different constraints
        attn_backend_limit = (
            self.model_runner.attn_backend.get_max_running_reqests(
                self.model_config.context_len,
                self.page_size,
            )
            * self.dp_size
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

        # A single request lives on one DP rank, so max_req_len is bounded
        # by per-rank pool size, not the global (dp-scaled) pool size.
        per_rank_tokens = (
            self.max_total_num_tokens // self.dp_size
            if self.dp_size > 1
            else self.max_total_num_tokens
        )
        self.max_req_len = min(
            self.model_config.context_len - 1,
            per_rank_tokens - 1,
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
        from sgl_jax.srt.model_executor.compilation_manager import CompilationManager

        self.compilation_manager = CompilationManager(
            server_args=server_args,
            max_padded_batch_size=self.max_padded_batch_size,
            max_padded_num_tokens=self.max_padded_num_tokens,
            dp_size=self.dp_size,
            tp_size=self.tp_size,
            page_size=self.page_size,
            max_req_len=self.max_req_len,
            vocab_size=self.model_config.vocab_size,
            multimodal=server_args.multimodal,
        )
        self.precompile_token_paddings = self.compilation_manager.token_buckets
        self.precompile_bs_paddings = self.compilation_manager.bs_buckets
        self.precompile_cache_loc_paddings = self.compilation_manager.cache_loc_buckets

        self.parent_process = psutil.Process().parent()
        self.sync_queue = Queue()
        self.sync_expert_ids_d2h_thread = threading.Thread(
            target=self._sync_expert_ids_d2h_thread_func,
            daemon=True,
        )
        self.sync_expert_ids_d2h_thread.start()

    def _sync_expert_ids_d2h_thread_func(self):
        try:
            self._sync_experts_ids_d2h()
        except Exception:
            traceback = get_exception_traceback()
            logger.error("ModelWorker sync thread hit an exception: %s", traceback)
            self.parent_process.send_signal(signal.SIGQUIT)

    def _sync_experts_ids_d2h(self):
        while True:
            layers_topk_ids, model_worker_batch = self.sync_queue.get()
            get_global_experts_capturer().on_forward_end(layers_topk_ids, model_worker_batch)

    def run_precompile(self, future_token_ids_map=None):
        prepare_lora = self.prepare_lora_batch if self.server_args.enable_lora else None
        self.compilation_manager.precompile_all(
            forward_fn=self.forward_batch_generation,
            model_runner=self.model_runner,
            mesh=self.mesh,
            prepare_lora_fn=prepare_lora,
            future_token_ids_map=future_token_ids_map,
        )

    def set_forward_metadata(self, model_worker_batch: ModelWorkerBatch):
        self.model_runner.attn_backend.forward_metadata = (
            self.model_runner.attn_backend.get_forward_metadata(model_worker_batch)
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
            getattr(
                self.model_runner,
                "full_max_total_num_tokens",
                self.model_runner.max_total_num_tokens,
            ),
            getattr(
                self.model_runner,
                "swa_max_total_num_tokens",
                self.model_runner.max_total_num_tokens,
            ),
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
            forward_metadata = self.model_runner.attn_backend.get_forward_metadata(
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
        logits_metadata = LogitsMetadata.from_model_worker_batch(model_worker_batch, self.mesh)
        logits_output, cache_miss_count, layers_topk_ids = self.model_runner.forward(
            forward_batch,
            logits_metadata=logits_metadata,
        )

        self.dump_topk_ids(layers_topk_ids, model_worker_batch)

        if launch_done is not None:
            launch_done.set()

        self.sync_queue.put(
            (
                layers_topk_ids,
                model_worker_batch,
            )
        )

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
                next_token_ids_device, token_logprobs, new_logits_output = self.model_runner.sample(
                    logits_output,
                    sampling_metadata,
                )
                cache_miss_count += count()
            # `selector` reorders DP-interleaved per-req tensors back to
            # original request order. For DP=1 it's just np.arange(real_bs).
            selector = model_worker_batch.logits_indices_selector
            if model_worker_batch.return_output_logprob_only:
                logprobs = self.model_runner.compute_logprobs(token_logprobs, next_token_ids_device)
                logits_output.next_token_logprobs = jax.device_get(logprobs)[selector]
        if new_logits_output is not None:
            logits_output = new_logits_output
            self._materialize_logprobs_to_host(
                logits_output, model_worker_batch, logits_metadata, selector
            )

        return (
            logits_output,
            next_token_ids_device,
            cache_miss_count,
        )

    def _materialize_logprobs_to_host(
        self,
        logits_output: LogitsProcessorOutput,
        model_worker_batch: ModelWorkerBatch,
        logits_metadata: LogitsMetadata,
        selector: np.ndarray,
    ):
        """Reorder + per-req split logprob tensors from device to host lists.

        `next_token_*` tensors are batch-major and routed through `selector`
        to recover original-req order. `input_*` tensors are per prompt-token
        in DP-rank-then-req order which already matches the original-req
        order, so they are split directly using `extend_logprob_pruned_lens_cpu`.
        Per-req `top_logprobs_nums` / `token_ids_logprobs` give the trim k_i /
        gather columns. Output shape is `list[list[float]]` per req, matching
        the consumer contract in `scheduler_output_processor_mixin`.
        """

        def gather(arr, *, as_float=False):
            if as_float:
                arr = arr.astype(jnp.float32)
            return jax.device_get(arr)[selector]

        if logits_output.next_token_logprobs is not None:
            logits_output.next_token_logprobs = jax.device_get(logits_output.next_token_logprobs)[
                selector
            ]

        top_nums = model_worker_batch.top_logprobs_nums
        tok_ids = model_worker_batch.token_ids_logprobs

        if logits_output.next_token_top_logprobs_val is not None:
            vals = gather(logits_output.next_token_top_logprobs_val, as_float=True)
            idxs = gather(logits_output.next_token_top_logprobs_idx)
            logits_output.next_token_top_logprobs_val = [
                vals[i, : top_nums[orig]].tolist() for i, orig in enumerate(selector)
            ]
            logits_output.next_token_top_logprobs_idx = [
                idxs[i, : top_nums[orig]].tolist() for i, orig in enumerate(selector)
            ]

        if logits_output.next_token_token_ids_logprobs_val is not None:
            full = gather(logits_output.next_token_token_ids_logprobs_val, as_float=True)
            per_req_vals, per_req_idxs = [], []
            for i, orig in enumerate(selector):
                ids = tok_ids[orig] if tok_ids else None
                if ids:
                    per_req_vals.append(full[i, ids].tolist())
                    per_req_idxs.append(list(ids))
                else:
                    per_req_vals.append([])
                    per_req_idxs.append([])
            logits_output.next_token_token_ids_logprobs_val = per_req_vals
            logits_output.next_token_token_ids_logprobs_idx = per_req_idxs

        pruned_lens = logits_metadata.extend_logprob_pruned_lens_cpu

        if logits_output.input_top_logprobs_val is not None:
            vals = jax.device_get(logits_output.input_top_logprobs_val.astype(jnp.float32))
            idxs = jax.device_get(logits_output.input_top_logprobs_idx)
            per_req_vals, per_req_idxs = [], []
            pt = 0
            for k, plen in zip(logits_metadata.top_logprobs_nums, pruned_lens):
                if plen <= 0:
                    per_req_vals.append([])
                    per_req_idxs.append([])
                    continue
                per_req_vals.append(vals[pt : pt + plen, :k].tolist())
                per_req_idxs.append(idxs[pt : pt + plen, :k].tolist())
                pt += plen
            logits_output.input_top_logprobs_val = per_req_vals
            logits_output.input_top_logprobs_idx = per_req_idxs

        if logits_output.input_token_ids_logprobs_val is not None:
            full = jax.device_get(logits_output.input_token_ids_logprobs_val.astype(jnp.float32))
            per_req_vals, per_req_idxs = [], []
            pt = 0
            for ids, plen in zip(logits_metadata.token_ids_logprobs, pruned_lens):
                if plen <= 0:
                    per_req_vals.append([])
                    per_req_idxs.append([])
                    continue
                if ids:
                    per_req_vals.append(full[pt : pt + plen][:, ids].tolist())
                    per_req_idxs.append([list(ids) for _ in range(plen)])
                else:
                    per_req_vals.append([])
                    per_req_idxs.append([])
                pt += plen
            logits_output.input_token_ids_logprobs_val = per_req_vals
            logits_output.input_token_ids_logprobs_idx = per_req_idxs

    def dump_topk_ids(self, layers_topk_ids: list[jax.Array], model_worker_batch: ModelWorkerBatch):
        enable = self.server_args.enable_return_routed_experts
        dump_topk_ids_file_info = os.getenv("DUMP_TOPK_IDS_FILEINFO", None)
        if not enable or dump_topk_ids_file_info is None:
            return

        # format: {prefill_file_name},{decode_file_name}
        file_slice = dump_topk_ids_file_info.split(",")
        if model_worker_batch.forward_mode.is_extend():
            file_name = file_slice[0]
        elif model_worker_batch.forward_mode.is_decode():
            file_name = file_slice[1]
        else:
            raise ValueError(
                f"Unsupported {model_worker_batch.forward_mode} to save topk_ids with txt"
            )
        import datetime

        unpadded_input_len = model_worker_batch.get_original_input_len()
        layers_topk_ids_cpu = jax.device_get(layers_topk_ids)

        file_name = (
            f"{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}_{unpadded_input_len}"
            + file_name
        )

        valid_topk_ids = []
        for ids_cpu in layers_topk_ids_cpu:
            valid_ids = ids_cpu[
                :unpadded_input_len, : self.model_config.hf_text_config.num_experts_per_tok
            ]
            valid_topk_ids.append(valid_ids)

        # Stack to create (num_layers, seq_len, num_experts_per_tok)
        valid_topk_ids_stacked = np.stack(valid_topk_ids, axis=0)

        # Transpose to (seq_len, num_layers, num_experts_per_tok)
        seq_layer_topk_cpu = np.transpose(valid_topk_ids_stacked, (1, 0, 2))

        # os.makedirs(os.path.dirname(file_name), exist_ok=True)
        np.savetxt(file_name, np.asarray(seq_layer_topk_cpu).flatten(), fmt="%d")


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
        dp_size = server_args.dp_size
        per_rank_tokens = (
            self.max_total_num_tokens // dp_size if dp_size > 1 else self.max_total_num_tokens
        )
        self.max_req_len = min(
            self.model_config.context_len - 1,
            per_rank_tokens - 1,
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
