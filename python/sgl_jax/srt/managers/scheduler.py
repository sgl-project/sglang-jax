"""A scheduler that manages a tensor parallel TPU worker."""

import concurrent.futures as futures
import dataclasses
import faulthandler
import gc
import logging
import math
import os
import pickle
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace

import jax
import numpy as np
import pathwaysutils
import psutil
import setproctitle
import zmq
from jax import numpy as jnp
from jax.scipy import special as jsp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.global_config import global_config
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.constrained.base_grammar_backend import (
    INVALID_GRAMMAR_OBJ,
    create_grammar_backend,
)
from sgl_jax.srt.hf_transformers_utils import get_tokenizer
from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
from sgl_jax.srt.managers.communication import CommunicationBackend
from sgl_jax.srt.managers.io_struct import (
    AbortReq,
    ContinueGenerationReqInput,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    PauseGenerationReqInput,
    ProfileReq,
    ReleaseScoringCacheReqInput,
    ReleaseScoringCacheReqOutput,
    ScoreFromCacheReqInput,
    ScoreFromCacheReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    TokenizedGenerateReqInput,
)
from sgl_jax.srt.managers.schedule_batch import (
    FINISH_ABORT,
    Req,
    ScheduleBatch,
    acc_global_bid,
    global_server_args_dict,
)
from sgl_jax.srt.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
    SchedulePolicy,
)
from sgl_jax.srt.managers.scheduler_metrics_mixin import SchedulerMetricsMixin
from sgl_jax.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sgl_jax.srt.managers.scheduler_profiler_mixing import SchedulerProfilerMixin
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.managers.tp_worker_overlap_thread import ModelWorkerClient
from sgl_jax.srt.managers.utils import validate_input_length
from sgl_jax.srt.mem_cache.chunk_cache import ChunkCache
from sgl_jax.srt.mem_cache.radix_cache import RadixCache
from sgl_jax.srt.mem_cache.swa_radix_cache import SWARadixCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.multimodal.tokenizer_utils import resolve_tokenizer_subdir
from sgl_jax.srt.precision_tracer import precision_tracer
from sgl_jax.srt.sampling.sampling_params import SamplingParams
from sgl_jax.srt.server_args import PortArgs, ServerArgs
from sgl_jax.srt.speculative.eagle_util import EagleDraftInput
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm
from sgl_jax.srt.utils.common_utils import (
    configure_logger,
    get_bool_env_var,
    get_zmq_socket,
    kill_itself_when_parent_died,
    pyspy_dump_schedulers,
    set_random_seed,
)
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.utils import TypeBasedDispatcher, get_exception_traceback

logger = logging.getLogger(__name__)

# Test retract decode for debugging purposes
TEST_RETRACT = get_bool_env_var("SGLANG_TEST_RETRACT")
RECORD_STEP_TIME = get_bool_env_var("SGLANG_RECORD_STEP_TIME")
GRAMMAR_TIMEOUT = float(os.environ.get("SGLANG_GRAMMAR_TIMEOUT", 300))
SCORE_V2_ALLOW_REQPOOL_OVERSUBSCRIBE = get_bool_env_var(
    "SGLANG_SCORE_FROM_CACHE_V2_ALLOW_REQPOOL_OVERSUBSCRIBE"
)


class SyncError(Exception):
    pass


class SendDataError(Exception):
    pass


class ReceiveDataError(Exception):
    pass


@jax.jit(static_argnums=(2,))
def _compute_label_only_logprobs(next_token_logits, label_token_ids_arr, out_sharding):
    """Compute target-only logprobs for [batch, vocab] logits."""
    logits_f32 = next_token_logits.astype(jnp.float32)
    label_logits = logits_f32.at[:, label_token_ids_arr].get(out_sharding=out_sharding)
    normalizer = jsp.logsumexp(logits_f32, axis=-1, keepdims=True)
    return label_logits - normalizer


@dataclass
class GenerationBatchResult:
    logits_output: LogitsProcessorOutput | None
    next_token_ids: list[int] | None  # on device
    extend_input_len_per_req: list[int]
    extend_logprob_start_len_per_req: list[int]
    bid: int
    cache_miss_count: int
    # relay path: forward stream -> next step forward
    next_draft_input: EagleDraftInput | None = None

    allocate_lens: np.ndarray | None = None
    num_accepted_tokens: int | None = None
    accept_lens: np.ndarray | None = None


class Scheduler(
    SchedulerOutputProcessorMixin,
    SchedulerProfilerMixin,
    SchedulerMetricsMixin,
):
    """
    A scheduler that manages a tensor parallel TPU worker, which managaes fixed multi TPU devices.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs = None,
        communication_backend: CommunicationBackend = None,
        mesh: jax.sharding.Mesh = None,
        model_class: None = None,
        stage_sub_dir: str | None = None,
    ):
        if stage_sub_dir is not None:
            server_args = dataclasses.replace(server_args)
            server_args.model_sub_dir = stage_sub_dir
        # set jit cache
        jit_cache_dir = os.getenv("JAX_COMPILATION_CACHE_DIR", None)
        if jit_cache_dir is not None:
            jax.config.update("jax_compilation_cache_dir", jit_cache_dir)
            jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
            jax.config.update("jax_persistent_cache_enable_xla_caches", "all")
            from jax.experimental.compilation_cache import compilation_cache as cc

            cc.set_cache_dir(jit_cache_dir)

        # Parse args
        self.server_args = server_args
        self.node_rank = server_args.node_rank
        self.nnodes = server_args.nnodes
        if port_args is not None:
            self.pub_sub_addr = port_args.pub_sub_addr
            self.pub_sub_sync_addr = port_args.pub_sub_sync_addr
        self.tp_size = server_args.tp_size
        self.schedule_policy = server_args.schedule_policy
        self.skip_tokenizer_init = server_args.skip_tokenizer_init
        self.stream_interval = server_args.stream_interval
        self.max_seq_len = server_args.max_seq_len
        self.page_size = server_args.page_size
        self.enable_overlap = not server_args.disable_overlap_schedule
        if server_args.multimodal:
            logger.info("Multimodal mode enabled, disabling overlap schedule")
            self.enable_overlap = False
        self.spec_algorithm = SpeculativeAlgorithm.from_string(server_args.speculative_algorithm)

        # LoRA configurations
        self.lora_paths = server_args.lora_paths
        self.max_loras_per_batch = server_args.max_loras_per_batch

        # Init inter-process communication
        context = zmq.Context(2)
        self._comm_backend = None

        if self.node_rank == 0:
            # todo: support multi host
            if communication_backend is not None:
                self._comm_backend = communication_backend
            else:
                self.recv_from_tokenizer = get_zmq_socket(
                    context, zmq.PULL, port_args.scheduler_input_ipc_name, False
                )
                self.send_to_tokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.tokenizer_ipc_name, False
                )

                if server_args.skip_tokenizer_init:
                    # Directly send to the TokenizerManager
                    self.send_to_detokenizer = get_zmq_socket(
                        context, zmq.PUSH, port_args.tokenizer_ipc_name, False
                    )
                else:
                    # Send to the DetokenizerManager
                    self.send_to_detokenizer = get_zmq_socket(
                        context, zmq.PUSH, port_args.detokenizer_ipc_name, False
                    )

                self.recv_from_rpc = get_zmq_socket(
                    context, zmq.DEALER, port_args.rpc_ipc_name, False
                )
                if self.nnodes > 1:
                    self.publisher = get_zmq_socket(context, zmq.PUB, self.pub_sub_addr, bind=True)
                    self.publisher_sync = get_zmq_socket(
                        context, zmq.REP, self.pub_sub_sync_addr, bind=True
                    )
                    self.num_subscribers = self.nnodes - 1
        else:
            self.recv_from_tokenizer = None
            self.recv_from_rpc = None
            self.send_to_tokenizer = SimpleNamespace(send_pyobj=lambda x: None)
            self.send_to_detokenizer = SimpleNamespace(send_pyobj=lambda x: None)
            if self.nnodes > 1:
                self.subscriber = get_zmq_socket(context, zmq.SUB, self.pub_sub_addr, bind=False)
                self.subscriber.setsockopt(zmq.SUBSCRIBE, b"")
                self.subscriber.setsockopt(zmq.RCVTIMEO, 5000)
                self.subscriber_sync = get_zmq_socket(
                    context, zmq.REQ, self.pub_sub_sync_addr, bind=False
                )

        if self.nnodes > 1:
            self.sync_pub_sub()

        # Init tokenizer
        self.init_tokenizer()

        # Init grammar backend for structured output
        self.grammar_backend = None
        self.grammar_queue: list[Req] = []  # Requests waiting for grammar compilation
        if not server_args.skip_tokenizer_init and not server_args.multimodal:
            self.grammar_backend = create_grammar_backend(
                server_args,
                self.tokenizer,
                self.model_config.vocab_size,
                self.model_config.hf_eos_token_id,
            )
        else:
            self.grammar_backend = None

        if not self.is_generation:
            self.enable_overlap = False
            logger.info("Overlap scheduler is disabled for embedding models.")

        # init distribution
        if self.nnodes > 1:
            jax.distributed.initialize(server_args.dist_init_addr, self.nnodes, self.node_rank)

        platform = os.getenv("JAX_PLATFORMS", None)
        if platform == "proxy":
            pathwaysutils.initialize()
        if mesh is not None:
            self.mesh = mesh
        else:
            self.mesh = create_device_mesh(
                ici_parallelism=[-1, self.tp_size],
                dcn_parallelism=[1, 1],
                device_indexes=server_args.device_indexes,
            )

        if server_args.moe_backend == "fused":
            mesh_ep_size = self.mesh.shape.get("data", 1) * self.mesh.shape.get("tensor", 1)
            if server_args.ep_size != mesh_ep_size:
                logger.warning(
                    "moe_backend='fused' uses EP size = mesh(data*tensor)=%d, but --ep-size=%d. "
                    "If you expected separate EP and TP (e.g. ep_size=%d, tp_size=%d), note that the "
                    "fused MoE kernel currently treats the full 2D mesh as its EP group.",
                    mesh_ep_size,
                    server_args.ep_size,
                    server_args.ep_size,
                    server_args.tp_size,
                )

        TpWorkerClass = ModelWorkerClient if self.enable_overlap else ModelWorker

        self.tp_worker = TpWorkerClass(
            server_args=server_args,
            mesh=self.mesh,
            model_class=model_class,
        )

        # launch draft worker
        if self.spec_algorithm is not None and self.spec_algorithm.is_eagle():
            from sgl_jax.srt.speculative.eagle_worker import EAGLEWorker

            self.draft_worker = EAGLEWorker(
                server_args=server_args,
                target_worker=self.tp_worker,
            )

        # Get token and memory info from the model worker
        (
            self.max_total_num_tokens,  # total requests
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            _,
            worker_global_server_args_dict,
            _,
            _,
            _,
        ) = self.tp_worker.get_worker_info()

        global_server_args_dict.update(worker_global_server_args_dict)
        set_random_seed(self.random_seed)

        self.is_hybrid = self.tp_worker.is_hybrid
        if self.is_hybrid:
            self.sliding_window_size = self.tp_worker.sliding_window_size
            self.full_tokens_per_layer, self.swa_tokens_per_layer = (
                self.tp_worker.get_tokens_per_layer_info()
            )

        # Init memory pool and cache
        self.init_memory_pool_and_cache()

        # Init running status
        self.waiting_queue: list[Req] = []
        # The aborted requests
        self.aborted_reqs: dict[str, Req] = {}
        # The running decoding batch for continuous batching
        self.running_batch: ScheduleBatch = ScheduleBatch(reqs=[], batch_is_full=False)
        # The current forward batch
        self.cur_batch: ScheduleBatch | None = None
        # The last forward batch
        self.last_batch: ScheduleBatch | None = None
        self.forward_ct = 0
        self.forward_ct_decode = 0
        self.num_generated_tokens = 0
        self.last_prefill_tokens = 0
        self.last_decode_stats_tic = time.perf_counter()
        self.last_prefill_stats_tic = time.perf_counter()
        self.num_retracted_reqs: int = 0
        self.num_paused_reqs: int = 0
        self.accept_token = 0
        self.spec_num_forward_ct = 0
        self.draft_token = 0
        # Init chunked prefill
        self.chunked_prefill_size = server_args.chunked_prefill_size
        if self.chunked_prefill_size <= 0:  # -1 means disable
            self.chunked_prefill_size = None
        self.chunked_req = None
        self.is_mixed_chunk = (
            self.chunked_prefill_size is not None and server_args.enable_mixed_chunk
        )

        # Init pause/continue state
        self._engine_paused = False

        # Workstream B: Store cached nodes for prefill+extend
        # Map:
        # rid -> (
        #   last_node,
        #   swa_uuid_for_lock,
        #   input_ids,
        #   prefix_indices,
        #   extra_key,
        #   last_access_ts,
        # )
        self.scoring_cache_nodes = {}
        self.scoring_cache_timeout = float(server_args.multi_item_prefill_extend_cache_timeout)
        self._last_scoring_cache_gc = 0.0
        # Scoring-cache counters (vLLM-style query/hit/miss accounting).
        self.scoring_cache_lookup_queries = 0
        self.scoring_cache_lookup_hits = 0
        self.scoring_cache_lookup_misses = 0
        self.scoring_cache_lookup_by_path: dict[str, dict[str, int]] = {
            "extend": {"queries": 0, "hits": 0, "misses": 0},
            "score_from_cache_v2": {"queries": 0, "hits": 0, "misses": 0},
        }
        self.scoring_cache_handles_created = 0
        self.scoring_cache_handles_released = 0
        self.scoring_cache_handles_released_manual = 0
        self.scoring_cache_handles_released_expired = 0
        self.scoring_cache_handles_released_other = 0
        self.scoring_cache_handles_missing_node = 0
        # Ingress message metrics for tokenizer->scheduler and rpc->scheduler paths.
        self.ingress_recv_calls = 0
        self.ingress_nonempty_calls = 0
        self.ingress_max_batch_size = 0
        self.ingress_tokenizer_frames = 0
        self.ingress_rpc_frames = 0
        # "messages" here counts logical requests seen by scheduler after unpack.
        self.ingress_tokenizer_messages = 0
        self.ingress_rpc_messages = 0
        self.ingress_batch_size_histogram = {
            "eq_0": 0,
            "eq_1": 0,
            "2_to_4": 0,
            "5_to_16": 0,
            "gt_16": 0,
        }
        self.ingress_score_paths = {
            "tokenizer_multi_item_packed": 0,
            "tokenizer_cache_for_scoring": 0,
            "tokenizer_extend_from_cache": 0,
            "rpc_score_from_cache_v2": 0,
            "rpc_release_scoring_cache": 0,
        }
        # Fastpath v2 score-from-cache counters.
        self.score_from_cache_v2_attempted = 0
        self.score_from_cache_v2_succeeded = 0
        self.score_from_cache_v2_fallback = 0
        self.score_from_cache_v2_fallback_reasons: dict[str, int] = {}

        # Init schedule policy and new token estimation
        self.policy = SchedulePolicy(
            self.schedule_policy,
            self.tree_cache,
        )
        assert server_args.schedule_conservativeness >= 0, "Invalid schedule_conservativeness"
        self.init_new_token_ratio = min(
            global_config.default_init_new_token_ratio * server_args.schedule_conservativeness,
            1.0,
        )
        self.min_new_token_ratio = min(
            self.init_new_token_ratio * global_config.default_min_new_token_ratio_factor,
            1.0,
        )
        self.new_token_ratio_decay = (
            self.init_new_token_ratio - self.min_new_token_ratio
        ) / global_config.default_new_token_ratio_decay_steps
        self.new_token_ratio = self.init_new_token_ratio

        # Init watchdog thread
        self.watchdog_timeout = server_args.watchdog_timeout
        t = threading.Thread(target=self.watchdog_thread, daemon=True)
        t.start()
        self.parent_process = psutil.Process().parent()

        self.init_profier()

        self.init_metrics()

        # Init request dispatcher
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.handle_generate_request),
                (AbortReq, self.abort_request),
                (ProfileReq, self.profile),
                (FlushCacheReqInput, self.flush_cache_wrapped),
                (ReleaseScoringCacheReqInput, self.release_scoring_cache),
                (ScoreFromCacheReqInput, self.score_from_cache_v2),
                (GetInternalStateReq, self.get_internal_state),
                (SetInternalStateReq, self.set_internal_state),
                (PauseGenerationReqInput, self.pause_generation),
                (ContinueGenerationReqInput, self.continue_generation),
            ]
        )

        if not server_args.disable_precompile:
            if self.spec_algorithm is None or self.spec_algorithm.is_none():
                logger.info("[Scheduler] Begins to run worker precompile.")
                self.tp_worker.run_precompile()
                logger.info("[Scheduler] Completes worker precompile.")
            else:
                logger.info("[Scheduler] Begins to run spec_decode worker precompile.")
                self.draft_worker.run_spec_decode_precompile()
                logger.info("[Scheduler] Completes spec_decode worker precompile.")

    def sync_pub(self):
        logger.info(
            "[Publisher %s] Begins to synchronize, wait %s Subscribers",
            self.node_rank,
            self.nnodes - 1,
        )
        ready_count = 0
        try:
            while ready_count < self.num_subscribers:
                message = self.publisher_sync.recv_string()
                if message == "READY":
                    ready_count += 1
                    logger.info(
                        "[Publisher %s] receives %s READY signal",
                        self.node_rank,
                        ready_count,
                    )
                    self.publisher_sync.send_string("ACK")
                else:
                    self.publisher_sync.send_string("NACK")
        except zmq.Again:
            logger.error("[Publisher %s] Fails to synchronize due to timeout", self.node_rank)
            return False
        except Exception as e:
            logger.error("[Publisher %s] Encounters error: %s", self.node_rank, e)
            return False
        logger.info("[Publisher %s] Succeeds to synchronize!", self.node_rank)
        return True

    def sync_sub(self):
        logger.info("[Subscriber %s] Begins to synchronize", self.node_rank)
        try:
            self.subscriber_sync.send_string("READY")
            ack = self.subscriber_sync.recv_string()
            if ack == "ACK":
                logger.info("[Subscriber %s] Succeeds to synchronizes!", self.node_rank)
                return True
            else:
                logger.error(
                    "[Subscriber %s] Fails to synchroinze with ack: %s",
                    self.node_rank,
                    ack,
                )
                return False
        except Exception as e:
            logger.error("[Subscriber %s] Fails to synchronize with error: %s", self.node_rank, e)
            return False

    def sync_pub_sub(self):
        success = self.sync_pub() if self.node_rank == 0 else self.sync_sub()
        if not success:
            raise SyncError("Fail to synchronize between publisher and subscribers")

    def init_tokenizer(self):
        server_args = self.server_args
        self.model_config = ModelConfig.from_server_args(server_args)
        self.is_generation = self.model_config.is_generation
        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            tokenizer_subdir = ""
            if server_args.multimodal:
                tokenizer_subdir = resolve_tokenizer_subdir(
                    server_args.model_path, server_args.tokenizer_path
                )
            self.tokenizer = get_tokenizer(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                sub_dir=tokenizer_subdir,
            )

    def init_memory_pool_and_cache(self):
        server_args = self.server_args
        self.req_to_token_pool, self.token_to_kv_pool_allocator = self.tp_worker.get_memory_pool()

        if server_args.chunked_prefill_size is not None and server_args.disable_radix_cache:
            self.tree_cache = ChunkCache(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                page_size=self.page_size,
            )
        elif self.is_hybrid:
            self.tree_cache = SWARadixCache(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                sliding_window_size=self.sliding_window_size,
                page_size=self.page_size,
                disable=server_args.disable_radix_cache,
            )
        else:
            self.tree_cache = RadixCache(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                page_size=self.page_size,
                disable=server_args.disable_radix_cache,
                kv_head_num=self.model_config.get_num_kv_heads(self.tp_size),
                head_dim=self.model_config.head_dim,
                layer_num=self.model_config.num_hidden_layers,
                max_seq_len=server_args.max_seq_len,
                is_eagle=self.spec_algorithm is not None and self.spec_algorithm.is_eagle(),
            )

        self.decode_mem_cache_buf_multiplier = 1

    def event_loop_normal(self):
        """A normal scheduler loop."""
        while True:
            recv_reqs = (
                self._comm_backend.recv_requests()
                if self._comm_backend is not None
                else self.recv_requests()
            )
            self.process_input_requests(recv_reqs)

            # Skip batch processing when engine is paused
            if self._engine_paused:
                continue

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # When the server is idle, do self-check and re-init some states
                self.check_memory()
                self.check_tree_cache()
                self.new_token_ratio = self.init_new_token_ratio

                # Elegant wait if idle
                if self._comm_backend is not None:
                    self._comm_backend.wait_for_new_requests(0.001)

            self.last_batch = batch

    def event_loop_overlap(self):
        """A scheduler loop that overlaps the CPU processing and Accelerator computation."""
        self.result_queue = deque()

        while True:
            recv_reqs = (
                self._comm_backend.recv_requests()
                if self._comm_backend is not None
                else self.recv_requests()
            )
            self.process_input_requests(recv_reqs)

            # Skip batch processing when engine is paused
            if self._engine_paused:
                continue

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                batch.launch_done = threading.Event()
                with jax.profiler.TraceAnnotation("run_batch"):
                    result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), result))

                if self.last_batch is None:
                    # Create a dummy first batch to start the pipeline for overlap schedule.
                    # It is now used for triggering the sampling_info_done event.
                    tmp_batch = ScheduleBatch(
                        reqs=None,
                        forward_mode=ForwardMode.DUMMY_FIRST,
                        next_batch_sampling_info=self.tp_worker.cur_sampling_info,
                    )
                    with jax.profiler.TraceAnnotation("process_batch_result"):
                        self.process_batch_result(tmp_batch, None, batch.launch_done)

            if self.last_batch:
                # Process the results of the last batch
                tmp_batch, tmp_result = self.result_queue.popleft()
                tmp_batch.next_batch_sampling_info = (
                    self.tp_worker.cur_sampling_info if batch else None
                )
                # NOTE: we should use current launched batch's launch_done event Instead of the last batch's
                self.process_batch_result(
                    tmp_batch, tmp_result, batch.launch_done if batch else None
                )
            elif batch is None:
                # When the server is idle, do self-check and re-init some states
                self.check_memory()
                self.check_tree_cache()
                self.new_token_ratio = self.init_new_token_ratio

            self.last_batch = batch

    def run_publisher(self, recv_reqs):
        retry_count = 0
        while retry_count < 3:
            try:
                serialized_data = pickle.dumps(recv_reqs)
                self.publisher.send(serialized_data)
                return True
            except Exception as e:
                logger.error(
                    "[Publisher %s] Fails to send data with error: %s",
                    self.node_rank,
                    e,
                )
        return False

    def run_subscriber(self):
        retry_count = 0
        while retry_count < 3:
            try:
                serialized_data = self.subscriber.recv()
                return pickle.loads(serialized_data)
            except zmq.Again:
                logger.error(
                    "[Subscriber %s] Fails to receive data with timeout, and try again",
                    self.node_rank,
                )
            except Exception as e:
                logger.error(
                    "[Subscriber %s] Fails to receive or deserialize with error: %s, and try again",
                    self.node_rank,
                    e,
                )
        return None

    def broadcast_pyobj(self, recv_reqs):
        if self.node_rank == 0:
            if not self.run_publisher(recv_reqs):
                raise SendDataError(f"[Publisher {self.node_rank}] Fails to send data")
        else:
            recv_reqs = self.run_subscriber()
            if recv_reqs is None:
                raise ReceiveDataError(f"[Subscriber {self.node_rank}] Fails to receive data")
        return recv_reqs

    def recv_requests(self) -> list[Req]:
        """Receive results at node_rank = 0 and broadcast it to all other Node ranks."""
        self.ingress_recv_calls += 1
        if self.node_rank == 0:
            recv_reqs = []
            tokenizer_frame_count = 0
            rpc_frame_count = 0
            tokenizer_req_count = 0
            rpc_req_count = 0

            while True:
                try:
                    recv_obj = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                tokenizer_frame_count += 1
                unpacked_reqs = (
                    list(recv_obj) if isinstance(recv_obj, (list, tuple)) else [recv_obj]
                )
                recv_reqs.extend(unpacked_reqs)
                tokenizer_req_count += len(unpacked_reqs)
                for recv_req in unpacked_reqs:
                    if isinstance(recv_req, TokenizedGenerateReqInput):
                        if bool(getattr(recv_req, "is_multi_item_scoring", False)):
                            self.ingress_score_paths["tokenizer_multi_item_packed"] += 1
                        if bool(getattr(recv_req, "cache_for_scoring", False)):
                            self.ingress_score_paths["tokenizer_cache_for_scoring"] += 1
                        if bool(getattr(recv_req, "extend_from_cache", None)):
                            self.ingress_score_paths["tokenizer_extend_from_cache"] += 1

            while True:
                try:
                    recv_obj = self.recv_from_rpc.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                rpc_frame_count += 1
                unpacked_reqs = (
                    list(recv_obj) if isinstance(recv_obj, (list, tuple)) else [recv_obj]
                )
                recv_reqs.extend(unpacked_reqs)
                rpc_req_count += len(unpacked_reqs)
                for recv_rpc in unpacked_reqs:
                    if isinstance(recv_rpc, ScoreFromCacheReqInput):
                        self.ingress_score_paths["rpc_score_from_cache_v2"] += 1
                    elif isinstance(recv_rpc, ReleaseScoringCacheReqInput):
                        self.ingress_score_paths["rpc_release_scoring_cache"] += 1

            self.ingress_tokenizer_frames += tokenizer_frame_count
            self.ingress_rpc_frames += rpc_frame_count
            self.ingress_tokenizer_messages += tokenizer_req_count
            self.ingress_rpc_messages += rpc_req_count
            batch_size = tokenizer_req_count + rpc_req_count
            if batch_size > 0:
                self.ingress_nonempty_calls += 1
                if batch_size > self.ingress_max_batch_size:
                    self.ingress_max_batch_size = batch_size
            if batch_size == 0:
                self.ingress_batch_size_histogram["eq_0"] += 1
            elif batch_size == 1:
                self.ingress_batch_size_histogram["eq_1"] += 1
            elif batch_size <= 4:
                self.ingress_batch_size_histogram["2_to_4"] += 1
            elif batch_size <= 16:
                self.ingress_batch_size_histogram["5_to_16"] += 1
            else:
                self.ingress_batch_size_histogram["gt_16"] += 1
        else:
            recv_reqs = None

        if self.nnodes > 1:
            recv_reqs = self.broadcast_pyobj(recv_reqs)
        return recv_reqs

    def process_input_requests(self, recv_reqs: list):
        self._evict_expired_scoring_cache_nodes()
        for recv_req in recv_reqs:
            output = self._request_dispatcher(recv_req)
            if output is not None:
                self.send_to_tokenizer.send_pyobj(output)

    def _unpack_scoring_cache_entry(self, entry):
        # Backward-compatible unpack for entries created before `last_access_ts`
        # was added.
        if len(entry) == 6:
            return entry
        if len(entry) == 5:
            node, swa_uuid, input_ids, prefix_indices, extra_key = entry
            return node, swa_uuid, input_ids, prefix_indices, extra_key, 0.0
        raise RuntimeError(f"Invalid scoring cache entry format (len={len(entry)}).")

    def _record_scoring_cache_lookup(self, path: str, hit: bool) -> None:
        self.scoring_cache_lookup_queries += 1
        if hit:
            self.scoring_cache_lookup_hits += 1
        else:
            self.scoring_cache_lookup_misses += 1

        bucket = self.scoring_cache_lookup_by_path.setdefault(
            path,
            {"queries": 0, "hits": 0, "misses": 0},
        )
        bucket["queries"] += 1
        if hit:
            bucket["hits"] += 1
        else:
            bucket["misses"] += 1

    def _record_scoring_cache_handle_created(self) -> None:
        self.scoring_cache_handles_created += 1

    def _record_scoring_cache_handle_released(self, reason: str) -> None:
        self.scoring_cache_handles_released += 1
        if reason == "manual":
            self.scoring_cache_handles_released_manual += 1
        elif reason == "expired":
            self.scoring_cache_handles_released_expired += 1
        else:
            self.scoring_cache_handles_released_other += 1

    def _scoring_cache_metrics_snapshot(self) -> dict:
        query_total = self.scoring_cache_lookup_queries
        hit_total = self.scoring_cache_lookup_hits
        miss_total = self.scoring_cache_lookup_misses
        hit_rate = float(hit_total / query_total) if query_total > 0 else 0.0
        return {
            "active_handles": len(self.scoring_cache_nodes),
            "handles_created": self.scoring_cache_handles_created,
            "handles_released_total": self.scoring_cache_handles_released,
            "handles_released_manual": self.scoring_cache_handles_released_manual,
            "handles_released_expired": self.scoring_cache_handles_released_expired,
            "handles_released_other": self.scoring_cache_handles_released_other,
            "handles_missing_node": self.scoring_cache_handles_missing_node,
            "lookup_queries": query_total,
            "lookup_hits": hit_total,
            "lookup_misses": miss_total,
            "lookup_hit_rate": hit_rate,
            "lookup_by_path": {
                path: dict(stats) for path, stats in self.scoring_cache_lookup_by_path.items()
            },
        }

    def _release_scoring_cache_entry(self, rid: str, entry, reason: str) -> None:
        node, swa_uuid, *_ = self._unpack_scoring_cache_entry(entry)
        self._record_scoring_cache_handle_released(reason)
        if node is None:
            self.scoring_cache_handles_missing_node += 1
            logger.warning("Scoring cache entry rid=%s has no radix node (%s).", rid, reason)
            return
        try:
            if isinstance(self.tree_cache, SWARadixCache):
                self.tree_cache.dec_lock_ref(node, swa_uuid)
            else:
                self.tree_cache.dec_lock_ref(node)
        except Exception:
            logger.exception(
                "Failed to decrement scoring-cache lock ref for rid=%s (%s).",
                rid,
                reason,
            )

    def _touch_scoring_cache_entry(self, rid: str, now: float | None = None):
        entry = self.scoring_cache_nodes.get(rid)
        if entry is None:
            return
        node, swa_uuid, input_ids, prefix_indices, extra_key, _ = self._unpack_scoring_cache_entry(
            entry
        )
        self.scoring_cache_nodes[rid] = (
            node,
            swa_uuid,
            input_ids,
            prefix_indices,
            extra_key,
            time.monotonic() if now is None else now,
        )

    def _evict_expired_scoring_cache_nodes(self, now: float | None = None) -> int:
        timeout = self.scoring_cache_timeout
        if timeout <= 0:
            return 0

        now_ts = time.monotonic() if now is None else now
        # Throttle GC to avoid walking the dict too often.
        if now is None and now_ts - self._last_scoring_cache_gc < 0.5:
            return 0
        self._last_scoring_cache_gc = now_ts

        expired_rids: list[str] = []
        for rid, entry in self.scoring_cache_nodes.items():
            *_, last_access_ts = self._unpack_scoring_cache_entry(entry)
            if now_ts - last_access_ts > timeout:
                expired_rids.append(rid)

        for rid in expired_rids:
            entry = self.scoring_cache_nodes.pop(rid, None)
            if entry is None:
                continue
            self._release_scoring_cache_entry(rid, entry, reason="expired")

        if expired_rids:
            logger.info("Evicted %d expired scoring cache handles.", len(expired_rids))
        return len(expired_rids)

    def _resolve_extend_from_cache(
        self, recv_req: TokenizedGenerateReqInput
    ) -> tuple[tuple | None, str | None]:
        if not recv_req.extend_from_cache:
            return None, None

        self._evict_expired_scoring_cache_nodes()
        entry = self.scoring_cache_nodes.get(recv_req.extend_from_cache)
        if entry is None:
            self._record_scoring_cache_lookup(path="extend", hit=False)
            err = (
                f"Missing scoring cache handle '{recv_req.extend_from_cache}'. "
                "The cached prefix may have expired or been released."
            )
            logger.warning("Prefill+extend scheduler: %s", err)
            return None, err
        self._record_scoring_cache_lookup(path="extend", hit=True)

        cached_last_node, _, prefix_ids, prefix_indices, cached_extra_key, _ = (
            self._unpack_scoring_cache_entry(entry)
        )
        item_ids = recv_req.input_ids or []
        recv_req.input_ids = prefix_ids + item_ids
        cached_prefix_len = len(prefix_indices)
        suffix_len = max(0, len(item_ids))
        if recv_req.extra_key is None:
            recv_req.extra_key = cached_extra_key
        self._touch_scoring_cache_entry(recv_req.extend_from_cache)
        logger.debug(
            "Prefill+extend scheduler: extend request rid=%s handle=%s prefix_tokens=%d cached_prefix=%d item_tokens=%d merged_input_tokens=%d max_new_tokens=%s",
            recv_req.rid,
            recv_req.extend_from_cache,
            len(prefix_ids),
            cached_prefix_len,
            suffix_len,
            len(recv_req.input_ids),
            recv_req.sampling_params.max_new_tokens,
        )
        return (cached_last_node, prefix_indices), None

    def _record_score_from_cache_v2_fallback(self, reason: str):
        self.score_from_cache_v2_fallback += 1
        self.score_from_cache_v2_fallback_reasons[reason] = (
            self.score_from_cache_v2_fallback_reasons.get(reason, 0) + 1
        )

    def _score_from_cache_v2_fallback_output(
        self,
        recv_req: ScoreFromCacheReqInput,
        reason: str,
        error_msg: str = "",
        dispatch_count: int = 0,
        device_compute_s: float = 0.0,
        host_orchestration_s: float = 0.0,
    ) -> ScoreFromCacheReqOutput:
        self._record_score_from_cache_v2_fallback(reason)
        return ScoreFromCacheReqOutput(
            rid=recv_req.rid,
            success=False,
            scores=[],
            fallback_reason=reason,
            error_msg=error_msg,
            dispatch_count=dispatch_count,
            lifecycle_requests_sent=0,
            lifecycle_results_received=0,
            queue_wait_s=0.0,
            device_compute_s=device_compute_s,
            host_orchestration_s=host_orchestration_s,
        )

    def _score_from_cache_v2_validate_items(
        self, recv_req: ScoreFromCacheReqInput
    ) -> tuple[bool, str, str]:
        if not recv_req.cache_handle:
            return False, "missing_cache_handle", "cache_handle must be non-empty."
        if not isinstance(recv_req.items_2d, list):
            return False, "unsupported_shape", "items_2d must be a list of token lists."
        if not isinstance(recv_req.label_token_ids, list) or len(recv_req.label_token_ids) == 0:
            return False, "unsupported_shape", "label_token_ids must be a non-empty list."
        if any((not isinstance(token_id, int)) for token_id in recv_req.label_token_ids):
            return False, "unsupported_shape", "label_token_ids must contain ints."
        for token_id in recv_req.label_token_ids:
            if token_id < 0 or token_id >= self.model_config.vocab_size:
                return (
                    False,
                    "unsupported_shape",
                    f"label_token_ids must be in [0, {self.model_config.vocab_size - 1}].",
                )
        for idx, item in enumerate(recv_req.items_2d):
            if not isinstance(item, list):
                return (
                    False,
                    "unsupported_shape",
                    f"items_2d[{idx}] must be a list of token ids.",
                )
            if len(item) == 0:
                return (
                    False,
                    "unsupported_shape",
                    f"items_2d[{idx}] must contain at least one token.",
                )
            if any((not isinstance(token_id, int)) for token_id in item):
                return (
                    False,
                    "unsupported_shape",
                    f"items_2d[{idx}] must contain ints.",
                )
        return True, "", ""

    @staticmethod
    def _score_from_cache_v2_probs_from_logprobs(
        row_logprobs: list[float], apply_softmax: bool
    ) -> list[float]:
        if apply_softmax:
            finite_vals = [x for x in row_logprobs if x != float("-inf")]
            if not finite_vals:
                return [0.0 for _ in row_logprobs]
            max_logprob = max(finite_vals)
            exps = [math.exp(x - max_logprob) if x != float("-inf") else 0.0 for x in row_logprobs]
            denom = sum(exps)
            if denom <= 0:
                return [0.0 for _ in row_logprobs]
            return [x / denom for x in exps]
        return [math.exp(x) if x != float("-inf") else 0.0 for x in row_logprobs]

    @staticmethod
    def _estimate_score_from_cache_v2_words(prefix_len: int, items: list[list[int]]) -> int:
        # Conservative host-side int32-sized tensor estimate for this chunk.
        total_item_tokens = sum(len(item) for item in items)
        total_fill_tokens = sum(prefix_len + len(item) for item in items)
        max_item_len = max((len(item) for item in items), default=0)
        bs = len(items)
        # Terms loosely track main arrays: flat input ids, seq/prefix/extend lengths,
        # req_to_token writes, and token-id-logprob tensors.
        return (
            total_item_tokens
            + total_fill_tokens
            + (3 * bs)
            + (bs * max_item_len)
            + (bs * prefix_len)
        )

    def _release_score_from_cache_v2_chunk_reqs(
        self,
        reqs: list[Req],
        batch: ScheduleBatch | None = None,
    ) -> None:
        if batch is not None:
            try:
                out_cache_loc = getattr(batch, "out_cache_loc", None)
                if out_cache_loc is not None:
                    out_cache_loc_arr = np.asarray(out_cache_loc, dtype=np.int32)
                    if out_cache_loc_arr.size > 0:
                        self.token_to_kv_pool_allocator.free(out_cache_loc_arr)
            except Exception:
                logger.exception("Fastpath v2 cleanup failed while freeing chunk KV slots.")

            try:
                req_pool_indices = getattr(batch, "req_pool_indices", None)
                if req_pool_indices is not None:
                    req_pool_indices_list = np.asarray(req_pool_indices, dtype=np.int32).tolist()
                    if req_pool_indices_list:
                        self.req_to_token_pool.free(req_pool_indices_list)
            except Exception:
                logger.exception("Fastpath v2 cleanup failed while freeing chunk req slots.")

            for req in reqs:
                req.req_pool_idx = None
            return

        for req in reqs:
            if req.req_pool_idx is None:
                continue
            try:
                pre_len = len(req.prefix_indices)
                seq_len = pre_len + max(0, req.extend_input_len)
                if seq_len > 0:
                    token_locs = self.req_to_token_pool.read(req.req_pool_idx, seq_len)
                    token_locs = token_locs[pre_len:seq_len]
                    token_locs = token_locs[token_locs != 0]
                    if len(token_locs) > 0:
                        self.token_to_kv_pool_allocator.free(token_locs)
            except Exception:
                logger.exception(
                    "Fastpath v2 cleanup failed while freeing KV tokens for rid=%s.",
                    req.rid,
                )
            try:
                self.req_to_token_pool.free(req.req_pool_idx)
            except Exception:
                logger.exception(
                    "Fastpath v2 cleanup failed while freeing req slot for rid=%s.",
                    req.rid,
                )
            req.req_pool_idx = None

    def _run_score_from_cache_v2_chunk(
        self,
        cache_handle: str,
        chunk_items: list[list[int]],
        label_token_ids: list[int],
        apply_softmax: bool,
        cached_last_node,
        cached_prefix_indices,
        prefix_ids: list[int],
        cached_extra_key: str | None,
    ) -> tuple[list[list[float]], float, float]:
        batch: ScheduleBatch | None = None
        reqs = self._build_score_from_cache_v2_chunk_reqs(
            cache_handle=cache_handle,
            chunk_items=chunk_items,
            label_token_ids=label_token_ids,
            cached_last_node=cached_last_node,
            cached_prefix_indices=cached_prefix_indices,
            prefix_ids=prefix_ids,
            cached_extra_key=cached_extra_key,
            return_label_logprobs=True,
        )

        try:
            batch = ScheduleBatch.init_new(
                reqs=reqs,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                tree_cache=self.tree_cache,
                model_config=self.model_config,
                enable_overlap=self.enable_overlap,
                spec_algorithm=self.spec_algorithm,
                enable_custom_logit_processor=False,
                chunked_req=None,
                mesh=self.mesh,
            )
            batch.prepare_for_extend()
            batch.bid = acc_global_bid()
            result = self.run_batch(batch)

            if result.logits_output is None:
                raise RuntimeError("Missing logits output from score-from-cache v2 chunk.")

            logprob_vals = result.logits_output.next_token_token_ids_logprobs_val
            logprob_idxs = result.logits_output.next_token_token_ids_logprobs_idx
            if logprob_vals is None or logprob_idxs is None:
                raise RuntimeError(
                    "Missing token_ids_logprobs tensors from score-from-cache v2 chunk."
                )
            logprob_vals = np.asarray(jax.device_get(logprob_vals), dtype=np.float64)
            logprob_idxs = np.asarray(jax.device_get(logprob_idxs), dtype=np.int32)
            if logprob_vals.ndim != 2 or logprob_idxs.shape != logprob_vals.shape:
                raise RuntimeError(
                    f"Unexpected token_ids_logprobs shape: vals={logprob_vals.shape}, idxs={logprob_idxs.shape}."
                )
            if logprob_vals.shape[0] != len(reqs):
                raise RuntimeError(
                    f"Chunk output rows ({logprob_vals.shape[0]}) != request count ({len(reqs)})."
                )

            scores: list[list[float]] = []
            for row_vals, row_idxs in zip(logprob_vals, logprob_idxs):
                row_logprobs: list[float] = []
                for token_id in label_token_ids:
                    match = np.where(row_idxs == token_id)[0]
                    if len(match) == 0:
                        row_logprobs.append(float("-inf"))
                    else:
                        row_logprobs.append(float(row_vals[int(match[0])]))
                scores.append(
                    self._score_from_cache_v2_probs_from_logprobs(
                        row_logprobs=row_logprobs,
                        apply_softmax=apply_softmax,
                    )
                )

            chunk_device_compute_s = reqs[0].device_compute_time_s if reqs else 0.0
            chunk_host_overhead_s = reqs[0].host_overhead_time_s if reqs else 0.0
            return scores, chunk_device_compute_s, chunk_host_overhead_s
        finally:
            self._release_score_from_cache_v2_chunk_reqs(reqs, batch=batch)

    def _build_score_from_cache_v2_chunk_reqs(
        self,
        cache_handle: str,
        chunk_items: list[list[int]],
        label_token_ids: list[int],
        cached_last_node,
        cached_prefix_indices,
        prefix_ids: list[int],
        cached_extra_key: str | None,
        return_label_logprobs: bool,
    ) -> list[Req]:
        reqs: list[Req] = []
        chunk_uid = time.time_ns()
        for local_idx, item_ids in enumerate(chunk_items):
            sampling_params = SamplingParams(max_new_tokens=0)
            sampling_params.stop_strs = []
            sampling_params.stop_str_max_len = 0

            rid = f"{cache_handle}-scorev2-{chunk_uid}-{local_idx}"
            req = Req(
                rid=rid,
                origin_input_text=None,
                origin_input_ids=prefix_ids + item_ids,
                sampling_params=sampling_params,
                return_logprob=return_label_logprobs,
                return_output_logprob_only=False,
                top_logprobs_num=0,
                token_ids_logprob=label_token_ids if return_label_logprobs else None,
                stream=False,
                extra_key=cached_extra_key,
                eos_token_ids=self.model_config.hf_eos_token_id,
                vocab_size=self.model_config.vocab_size,
                is_multi_item_scoring=False,
                cache_for_scoring=False,
                extend_from_cache=cache_handle,
            )
            req.tokenizer = self.tokenizer
            req.logprob_start_len = len(req.origin_input_ids) - 1
            req.cached_last_node = cached_last_node
            req.cached_last_host_node = cached_last_node
            req.cached_prefix_indices = cached_prefix_indices
            req.cached_host_hit_length = 0

            error_msg = validate_input_length(
                req,
                self.max_req_input_len,
                self.server_args.allow_auto_truncate,
            )
            if error_msg:
                raise ValueError(error_msg)
            req.init_next_round_input(self.tree_cache)
            reqs.append(req)
        return reqs

    def _run_score_from_cache_v2_chunk_label_only(
        self,
        cache_handle: str,
        chunk_items: list[list[int]],
        label_token_ids: list[int],
        label_token_ids_arr: jax.Array,
        apply_softmax: bool,
        cached_last_node,
        cached_prefix_indices,
        prefix_ids: list[int],
        cached_extra_key: str | None,
    ) -> tuple[list[list[float]], float, float]:
        batch: ScheduleBatch | None = None
        reqs = self._build_score_from_cache_v2_chunk_reqs(
            cache_handle=cache_handle,
            chunk_items=chunk_items,
            label_token_ids=label_token_ids,
            cached_last_node=cached_last_node,
            cached_prefix_indices=cached_prefix_indices,
            prefix_ids=prefix_ids,
            cached_extra_key=cached_extra_key,
            return_label_logprobs=False,
        )
        try:
            chunk_wall_start = time.perf_counter()
            batch = ScheduleBatch.init_new(
                reqs=reqs,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                tree_cache=self.tree_cache,
                model_config=self.model_config,
                enable_overlap=self.enable_overlap,
                spec_algorithm=self.spec_algorithm,
                enable_custom_logit_processor=False,
                chunked_req=None,
                mesh=self.mesh,
            )
            batch.prepare_for_extend()
            batch.bid = acc_global_bid()
            (
                precompile_token_paddings,
                precompile_bs_paddings,
                precompile_cache_loc_paddings,
            ) = self.tp_worker.get_precompile_paddings()
            model_worker_batch = batch.get_model_worker_batch(
                precompile_token_paddings,
                precompile_bs_paddings,
                precompile_cache_loc_paddings,
                self.page_size,
                self.server_args.enable_static_lora,
            )

            forward_start = time.perf_counter()
            logits_output, _, _ = self.tp_worker.forward_batch_generation(
                model_worker_batch=model_worker_batch,
                launch_done=None,
                skip_sample=True,
                sampling_metadata=None,
            )

            if logits_output is None or logits_output.next_token_logits is None:
                raise RuntimeError(
                    "Missing next_token_logits from score-from-cache v2 label-only chunk."
                )

            next_token_logits = logits_output.next_token_logits[: model_worker_batch.real_bs, :]
            out_sharding = NamedSharding(self.mesh, P(None, None))
            row_logprobs_dev = _compute_label_only_logprobs(
                next_token_logits,
                label_token_ids_arr,
                out_sharding,
            )
            row_logprobs_dev.block_until_ready()
            forward_end = time.perf_counter()
            row_logprobs = np.asarray(jax.device_get(row_logprobs_dev), dtype=np.float64)

            if row_logprobs.ndim != 2:
                raise RuntimeError(f"Unexpected label-only logprob shape: {row_logprobs.shape}.")
            if row_logprobs.shape[0] != len(reqs):
                raise RuntimeError(
                    f"Chunk output rows ({row_logprobs.shape[0]}) != request count ({len(reqs)})."
                )
            if row_logprobs.shape[1] != len(label_token_ids):
                raise RuntimeError(
                    f"Chunk output labels ({row_logprobs.shape[1]}) != requested label count ({len(label_token_ids)})."
                )

            # Align with baseline v2 semantics: raw values are token probabilities
            # (not normalized across label ids). Apply optional softmax on those
            # raw probability values only when requested.
            token_prob_vals = np.exp(row_logprobs)
            if apply_softmax:
                row_max = np.max(token_prob_vals, axis=1, keepdims=True)
                stable = token_prob_vals - row_max
                exp_vals = np.exp(stable)
                denom = np.sum(exp_vals, axis=1, keepdims=True)
                scores_np = exp_vals / denom
            else:
                scores_np = token_prob_vals
            scores = scores_np.tolist()

            chunk_device_compute_s = max(0.0, forward_end - forward_start)
            chunk_total_s = max(0.0, time.perf_counter() - chunk_wall_start)
            chunk_host_overhead_s = max(0.0, chunk_total_s - chunk_device_compute_s)
            return scores, chunk_device_compute_s, chunk_host_overhead_s
        finally:
            self._release_score_from_cache_v2_chunk_reqs(reqs, batch=batch)

    def score_from_cache_v2(self, recv_req: ScoreFromCacheReqInput) -> ScoreFromCacheReqOutput:
        self.score_from_cache_v2_attempted += 1
        dispatch_count = 0
        queue_wait_s = 0.0
        device_compute_s = 0.0
        host_orchestration_s = 0.0
        score_start = time.perf_counter()

        try:
            if self.enable_overlap:
                return self._score_from_cache_v2_fallback_output(
                    recv_req,
                    reason="unsupported_scheduler_mode",
                    error_msg="score-from-cache v2 does not support overlap schedule.",
                )

            is_valid, fallback_reason, error_msg = self._score_from_cache_v2_validate_items(
                recv_req
            )
            if not is_valid:
                return self._score_from_cache_v2_fallback_output(
                    recv_req,
                    reason=fallback_reason,
                    error_msg=error_msg,
                )

            self._evict_expired_scoring_cache_nodes()
            entry = self.scoring_cache_nodes.get(recv_req.cache_handle)
            if entry is None:
                self._record_scoring_cache_lookup(path="score_from_cache_v2", hit=False)
                return self._score_from_cache_v2_fallback_output(
                    recv_req,
                    reason="missing_cache_handle",
                    error_msg=(
                        f"Missing scoring cache handle '{recv_req.cache_handle}'. "
                        "The cached prefix may have expired or been released."
                    ),
                )
            self._record_scoring_cache_lookup(path="score_from_cache_v2", hit=True)

            cached_last_node, _, prefix_ids, prefix_indices, cached_extra_key, _ = (
                self._unpack_scoring_cache_entry(entry)
            )
            if cached_last_node is None:
                return self._score_from_cache_v2_fallback_output(
                    recv_req,
                    reason="missing_cache_handle",
                    error_msg=f"Scoring cache handle '{recv_req.cache_handle}' has no radix node.",
                )

            label_only_logprob = bool(
                getattr(self.server_args, "multi_item_score_label_only_logprob", False)
            )
            if label_only_logprob:
                backend = str(getattr(self.server_args, "device", "")).lower()
                if backend not in {"tpu", "gpu", "cuda", "cpu"}:
                    return self._score_from_cache_v2_fallback_output(
                        recv_req,
                        reason="unsupported_backend",
                        error_msg=(
                            "Label-only logprob fastpath requires TPU/GPU/CPU backend, "
                            f"got device={backend!r}."
                        ),
                    )

            items_per_step = int(recv_req.items_per_step or 0)
            default_items_per_step = int(
                getattr(self.server_args, "multi_item_score_from_cache_v2_items_per_step", 64)
            )
            if default_items_per_step <= 0:
                default_items_per_step = 1
            if items_per_step <= 0:
                items_per_step = default_items_per_step
            requested_items_per_step = max(1, items_per_step)

            # Keep chunk size within request-slot capacity so large configured values
            # (e.g., 64 with max_running_requests=24) do not trigger alloc_req_slots failures.
            capacity_caps: list[int] = []
            max_running_requests = int(getattr(self.server_args, "max_running_requests", 0) or 0)
            if max_running_requests > 0 and not SCORE_V2_ALLOW_REQPOOL_OVERSUBSCRIBE:
                capacity_caps.append(max_running_requests)
            req_to_token_pool = getattr(self, "req_to_token_pool", None)
            if req_to_token_pool is not None and hasattr(req_to_token_pool, "available_size"):
                try:
                    req_pool_available = int(req_to_token_pool.available_size())
                except Exception:
                    req_pool_available = 0
                if req_pool_available > 0:
                    capacity_caps.append(req_pool_available)
            effective_capacity = min(capacity_caps) if capacity_caps else requested_items_per_step
            if effective_capacity <= 0:
                return self._score_from_cache_v2_fallback_output(
                    recv_req,
                    reason="req_slot_exhausted",
                    error_msg=(
                        "Fastpath v2 requires at least one free request slot "
                        f"(requested_items_per_step={requested_items_per_step})."
                    ),
                    dispatch_count=dispatch_count,
                    device_compute_s=device_compute_s,
                    host_orchestration_s=host_orchestration_s,
                )
            items_per_step = max(
                1,
                min(requested_items_per_step, default_items_per_step, effective_capacity),
            )

            total_items = len(recv_req.items_2d)
            if total_items == 0:
                self.score_from_cache_v2_succeeded += 1
                return ScoreFromCacheReqOutput(
                    rid=recv_req.rid,
                    success=True,
                    scores=[],
                    fallback_reason=None,
                    error_msg="",
                    dispatch_count=0,
                    lifecycle_requests_sent=0,
                    lifecycle_results_received=0,
                    queue_wait_s=0.0,
                    device_compute_s=0.0,
                    host_orchestration_s=0.0,
                )

            label_token_ids_arr = None
            if label_only_logprob:
                label_token_ids_arr = jnp.asarray(recv_req.label_token_ids, dtype=jnp.int32)

            for start in range(0, total_items, items_per_step):
                chunk_items = recv_req.items_2d[start : start + items_per_step]
                if not chunk_items:
                    continue

                int32_max = np.iinfo(np.int32).max
                max_seq_len = max((len(prefix_ids) + len(item) for item in chunk_items), default=0)
                estimated_words = self._estimate_score_from_cache_v2_words(
                    prefix_len=len(prefix_ids),
                    items=chunk_items,
                )
                if max_seq_len >= int32_max or estimated_words >= int(int32_max * 0.9):
                    return self._score_from_cache_v2_fallback_output(
                        recv_req,
                        reason="size_guard",
                        error_msg=(
                            "Fastpath v2 size guard triggered. "
                            f"max_seq_len={max_seq_len}, estimated_words={estimated_words}"
                        ),
                        dispatch_count=dispatch_count,
                        device_compute_s=device_compute_s,
                        host_orchestration_s=host_orchestration_s,
                    )

            self._touch_scoring_cache_entry(recv_req.cache_handle)

            all_scores: list[list[float]] = []
            first_dispatch_started = False
            for start in range(0, total_items, items_per_step):
                chunk_items = recv_req.items_2d[start : start + items_per_step]
                if not chunk_items:
                    continue
                if not first_dispatch_started:
                    queue_wait_s = max(0.0, time.perf_counter() - score_start)
                    first_dispatch_started = True
                chunk_host_start = time.perf_counter()
                if label_only_logprob:
                    chunk_scores, chunk_device_compute_s, chunk_host_overhead_s = (
                        self._run_score_from_cache_v2_chunk_label_only(
                            cache_handle=recv_req.cache_handle,
                            chunk_items=chunk_items,
                            label_token_ids=recv_req.label_token_ids,
                            label_token_ids_arr=label_token_ids_arr,
                            apply_softmax=recv_req.apply_softmax,
                            cached_last_node=cached_last_node,
                            cached_prefix_indices=prefix_indices,
                            prefix_ids=prefix_ids,
                            cached_extra_key=cached_extra_key,
                        )
                    )
                else:
                    chunk_scores, chunk_device_compute_s, chunk_host_overhead_s = (
                        self._run_score_from_cache_v2_chunk(
                            cache_handle=recv_req.cache_handle,
                            chunk_items=chunk_items,
                            label_token_ids=recv_req.label_token_ids,
                            apply_softmax=recv_req.apply_softmax,
                            cached_last_node=cached_last_node,
                            cached_prefix_indices=prefix_indices,
                            prefix_ids=prefix_ids,
                            cached_extra_key=cached_extra_key,
                        )
                    )
                all_scores.extend(chunk_scores)
                dispatch_count += 1
                device_compute_s += max(0.0, chunk_device_compute_s)
                # host_orchestration_s excludes device time by design.
                chunk_total = max(0.0, time.perf_counter() - chunk_host_start)
                host_orchestration_s += max(
                    0.0,
                    max(chunk_host_overhead_s, chunk_total - chunk_device_compute_s),
                )

            if len(all_scores) != total_items:
                return self._score_from_cache_v2_fallback_output(
                    recv_req,
                    reason="runtime_exception",
                    error_msg=(
                        f"score-from-cache v2 returned {len(all_scores)} scores for {total_items} items."
                    ),
                    dispatch_count=dispatch_count,
                    device_compute_s=device_compute_s,
                    host_orchestration_s=host_orchestration_s,
                )

            self.score_from_cache_v2_succeeded += 1
            return ScoreFromCacheReqOutput(
                rid=recv_req.rid,
                success=True,
                scores=all_scores,
                fallback_reason=None,
                error_msg="",
                dispatch_count=dispatch_count,
                lifecycle_requests_sent=0,
                lifecycle_results_received=0,
                queue_wait_s=queue_wait_s,
                device_compute_s=device_compute_s,
                host_orchestration_s=host_orchestration_s,
            )
        except Exception as e:
            logger.exception("score-from-cache v2 failed; falling back to baseline path.")
            return self._score_from_cache_v2_fallback_output(
                recv_req,
                reason="runtime_exception",
                error_msg=str(e),
                dispatch_count=dispatch_count,
                device_compute_s=device_compute_s,
                host_orchestration_s=host_orchestration_s,
            )

    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
    ):
        if self.server_args.log_requests:
            logger.debug(
                "Handle request: rid=%s, max_new_tokens=%s, token_ids_logprob=%s",
                recv_req.rid,
                recv_req.sampling_params.max_new_tokens,
                recv_req.token_ids_logprob,
            )

        cached_prefix_ctx, cache_lookup_error = self._resolve_extend_from_cache(recv_req)

        # Create a new request
        req = Req(
            recv_req.rid,
            recv_req.text,
            recv_req.input_ids,
            recv_req.sampling_params,
            return_logprob=recv_req.return_logprob,
            return_output_logprob_only=recv_req.return_output_logprob_only,
            top_logprobs_num=recv_req.top_logprobs_num,
            token_ids_logprob=recv_req.token_ids_logprob,
            stream=recv_req.stream,
            lora_id=recv_req.lora_id,
            extra_key=recv_req.extra_key,
            eos_token_ids=self.model_config.hf_eos_token_id,
            vocab_size=self.model_config.vocab_size,
            return_routed_experts=recv_req.return_routed_experts,
            return_hidden_states=recv_req.return_hidden_states,
            is_multi_item_scoring=recv_req.is_multi_item_scoring,
            multi_item_scoring_delimiter=recv_req.multi_item_scoring_delimiter,
            multi_item_algorithm=getattr(recv_req, "multi_item_algorithm", None),
            multi_item_mask_mode=getattr(recv_req, "multi_item_mask_mode", None),
            cache_for_scoring=recv_req.cache_for_scoring,
            extend_from_cache=recv_req.extend_from_cache,
        )
        req.tokenizer = self.tokenizer
        if cache_lookup_error is not None:
            req.set_finish_with_abort(cache_lookup_error)
            self._add_request_to_queue(req)
            return

        if cached_prefix_ctx is not None:
            cached_last_node, cached_prefix_indices = cached_prefix_ctx
            req.cached_last_node = cached_last_node
            req.cached_last_host_node = cached_last_node
            req.cached_prefix_indices = cached_prefix_indices
            req.cached_host_hit_length = 0

        if hasattr(recv_req, "mm_inputs") and recv_req.mm_inputs:
            req.mm_inputs = recv_req.mm_inputs
        # Validate prompt length
        error_msg = validate_input_length(
            req,
            self.max_req_input_len,
            self.server_args.allow_auto_truncate,
        )
        if error_msg:
            req.set_finish_with_abort(error_msg)
            self._add_request_to_queue(req)
            return

        # Copy more attributes
        if recv_req.logprob_start_len == -1 or not recv_req.return_logprob:
            # By default, only return the logprobs for output tokens
            req.logprob_start_len = len(req.origin_input_ids) - 1
        else:
            req.logprob_start_len = recv_req.logprob_start_len

        if req.logprob_start_len >= len(req.origin_input_ids):
            error_msg = f"{req.logprob_start_len=} is higher than the number of input tokens {len(req.origin_input_ids)=}. Please use a smaller logprob_start_len."
            req.logprob_start_len = len(req.origin_input_ids) - 1
            req.set_finish_with_abort(error_msg)
            self._add_request_to_queue(req)
            return

        req.sampling_params.max_new_tokens = min(
            (
                req.sampling_params.max_new_tokens
                if req.sampling_params.max_new_tokens is not None
                else 1 << 30
            ),
            self.max_req_len - len(req.origin_input_ids) - 1,
        )

        # Init grammar cache for this request
        add_to_grammar_queue = False
        if (
            req.sampling_params.json_schema is not None
            or req.sampling_params.regex is not None
            or req.sampling_params.ebnf is not None
            or req.sampling_params.structural_tag is not None
        ):
            if self.grammar_backend is None:
                error_msg = "Grammar-based generation (json_schema, regex, ebnf, structural_tag) is not supported when the server is launched with --grammar-backend none or the current grammar backend isn’t compatible with the model’s tokenizer"
                req.set_finish_with_abort(error_msg)
            else:
                if req.sampling_params.json_schema is not None:
                    key = ("json", req.sampling_params.json_schema)
                elif req.sampling_params.regex is not None:
                    key = ("regex", req.sampling_params.regex)
                elif req.sampling_params.ebnf is not None:
                    key = ("ebnf", req.sampling_params.ebnf)
                elif req.sampling_params.structural_tag:
                    key = ("structural_tag", req.sampling_params.structural_tag)

                value, cache_hit = self.grammar_backend.get_cached_or_future_value(key)
                req.grammar = value

                if not cache_hit:
                    req.grammar_key = key
                    add_to_grammar_queue = True
                else:
                    if value is INVALID_GRAMMAR_OBJ:  # We hit a cached invalid grammar.
                        error_msg = f"Invalid grammar request with cache hit: {key=}"
                        req.set_finish_with_abort(error_msg)

        if add_to_grammar_queue:
            req.queue_time_start = time.perf_counter()
            self.grammar_queue.append(req)
        else:
            self._add_request_to_queue(req)

    def move_ready_grammar_requests(self):
        """Poll grammar futures and move ready requests to waiting queue."""
        if not self.grammar_queue:
            return

        num_ready_reqs = 0
        num_timeout_reqs = 0

        for req in self.grammar_queue:
            try:
                if req.finished():  # Aborted by AbortReq
                    num_ready_reqs += 1
                    continue

                # Poll with short timeout
                req.grammar = req.grammar.result(timeout=0.03)
                # Cache the compiled grammar
                if self.grammar_backend and req.grammar_key:
                    self.grammar_backend.set_cache(req.grammar_key, req.grammar.copy())

                # Check if compilation resulted in invalid grammar
                if req.grammar is INVALID_GRAMMAR_OBJ:
                    req.set_finish_with_abort(f"Invalid grammar request: key={req.grammar_key}")

                num_ready_reqs += 1
            except futures._base.TimeoutError:
                req.grammar_wait_ct += 1
                # Check if we've exceeded the timeout
                if req.grammar_wait_ct > GRAMMAR_TIMEOUT / 0.03:
                    num_timeout_reqs = 1
                break

        # Handle timeout requests: cancel and mark as failed
        for i in range(num_ready_reqs, num_ready_reqs + num_timeout_reqs):
            req = self.grammar_queue[i]
            req.grammar.cancel()
            error_msg = f"Grammar preprocessing timed out for {req.grammar_key=}"
            req.set_finish_with_abort(error_msg)
            # Cache as invalid to avoid retrying
            if self.grammar_backend and req.grammar_key:
                self.grammar_backend.set_cache(req.grammar_key, INVALID_GRAMMAR_OBJ)
        num_ready_reqs += num_timeout_reqs

        # Move ready requests to waiting queue
        self._extend_requests_to_queue(self.grammar_queue[:num_ready_reqs])
        self.grammar_queue = self.grammar_queue[num_ready_reqs:]

    def get_internal_state(self, recv_req: GetInternalStateReq):
        ret = dict(global_server_args_dict)
        ret["last_gen_throughput"] = self.last_gen_throughput
        ret["memory_usage"] = {
            "kvcache": round(self.token_to_kv_pool_allocator.get_kvcache().mem_usage, 2),
            "token_capacity": int(self.max_total_num_tokens),
        }

        # state for pause/continue generation
        ret["engine_paused"] = self._engine_paused
        ret["waiting_queue_size"] = len(self.waiting_queue)
        ret["running_batch_size"] = (
            0 if self.running_batch.is_empty() else len(self.running_batch.reqs)
        )
        ret["prefill_decode_size"] = ret["waiting_queue_size"] + ret["running_batch_size"]
        ret["waiting_queue_rids"] = [req.rid for req in self.waiting_queue]
        ret["running_batch_rids"] = (
            [req.rid for req in self.running_batch.reqs]
            if not self.running_batch.is_empty()
            else []
        )

        # scheduling state
        ret["cur_batch_is_none"] = self.cur_batch is None
        ret["last_batch_is_none"] = self.last_batch is None
        ret["chunked_req_is_none"] = self.chunked_req is None

        # request cache stat
        if isinstance(self.tree_cache, ChunkCache):
            ret["tree_cache_size"] = 0
        else:
            ret["tree_cache_size"] = (
                self.tree_cache.total_size() if self.tree_cache is not None else 0
            )
        if self.req_to_token_pool is not None:
            ret["req_to_token_pool_total"] = self.req_to_token_pool.size
            ret["req_to_token_pool_available"] = self.req_to_token_pool.available_size()
            ret["req_to_token_pool_used"] = (
                self.req_to_token_pool.size - self.req_to_token_pool.available_size()
            )
        else:
            ret["req_to_token_pool_total"] = 0
            ret["req_to_token_pool_available"] = 0
            ret["req_to_token_pool_used"] = 0

        # physical kv cache stat
        ret["available_kv_tokens"] = self.token_to_kv_pool_allocator.available_size()

        # counters
        ret["num_generated_tokens"] = self.num_generated_tokens
        ret["forward_ct_decode"] = self.forward_ct_decode
        ret["new_token_ratio"] = self.new_token_ratio
        ret["init_new_token_ratio"] = self.init_new_token_ratio
        ret["score_from_cache_v2_metrics"] = {
            "attempted": self.score_from_cache_v2_attempted,
            "succeeded": self.score_from_cache_v2_succeeded,
            "fallback": self.score_from_cache_v2_fallback,
            "fallback_reasons": dict(self.score_from_cache_v2_fallback_reasons),
        }
        ret["scoring_cache_metrics"] = self._scoring_cache_metrics_snapshot()
        ret["ingress_metrics"] = {
            "recv_calls": self.ingress_recv_calls,
            "nonempty_calls": self.ingress_nonempty_calls,
            "max_batch_size": self.ingress_max_batch_size,
            "tokenizer_frames": self.ingress_tokenizer_frames,
            "rpc_frames": self.ingress_rpc_frames,
            "tokenizer_messages": self.ingress_tokenizer_messages,
            "rpc_messages": self.ingress_rpc_messages,
            "batch_size_histogram": dict(self.ingress_batch_size_histogram),
            "score_path_messages": dict(self.ingress_score_paths),
        }

        return GetInternalStateReqOutput(internal_state=ret)

    def set_internal_state(self, recv_req: SetInternalStateReq):
        """Handle internal state updates, including precision tracer configuration"""
        success = True
        error_msg = ""

        try:
            if "precision_tracer" in recv_req.state_data:
                tracer_config = recv_req.state_data["precision_tracer"]

                # Update precision_tracer state in this process
                if "trace_active" in tracer_config:
                    logger.info(
                        "[SCHEDULER] check trace_active: %s",
                        precision_tracer.get_trace_active(),
                    )
                    precision_tracer._trace_active = tracer_config["trace_active"]
                    logger.info(
                        "[SCHEDULER] Updated trace_active to: %s",
                        precision_tracer._trace_active,
                    )

                    # Reset counters when starting trace
                    if tracer_config["trace_active"]:
                        precision_tracer._request_counter = 0
                        precision_tracer._completed_requests_count = 0
                        precision_tracer._request_traces = {}
                        logger.info("[SCHEDULER] Reset request_counter, completed_count and traces")

                if "max_requests" in tracer_config:
                    precision_tracer._max_requests = tracer_config["max_requests"]
                    logger.info(
                        "[SCHEDULER] Updated max_requests to: %s",
                        precision_tracer._max_requests,
                    )

                if "output_file" in tracer_config:
                    precision_tracer._trace_output_file = tracer_config["output_file"]
                    logger.info(
                        "[SCHEDULER] Updated output_file to: %s",
                        precision_tracer._trace_output_file,
                    )

                if "save_tensor" in tracer_config:
                    precision_tracer._save_tensor = tracer_config["save_tensor"]
                    logger.info(
                        "[SCHEDULER] Updated save_tensor to: %s",
                        precision_tracer._save_tensor,
                    )

                logger.info("[SCHEDULER] Precision tracer state updated: %s", tracer_config)

        except Exception as e:
            success = False
            error_msg = str(e)
            logger.info("[SCHEDULER] Error updating internal state: %s", error_msg)

        return SetInternalStateReqOutput(
            request_id=recv_req.request_id, success=success, error_msg=error_msg
        )

    def flush_cache_wrapped(self, recv_req: FlushCacheReqInput):
        success, error_msg, flushed_items = self.flush_cache()
        return FlushCacheReqOutput(
            rid=recv_req.rid,
            error_msg=error_msg,
            success=success,
            flushed_items=flushed_items,
        )

    def _can_flush_cache(self) -> tuple[bool, str]:
        """Return whether cache flush can proceed and an optional error message."""

        def _batch_size(batch: ScheduleBatch | None) -> int:
            if batch is None:
                return 0
            return 0 if batch.is_empty() else batch.batch_size()

        waiting_reqs = len(self.waiting_queue)
        running_reqs = _batch_size(self.running_batch)
        current_batch_reqs = _batch_size(self.cur_batch)
        last_batch_reqs = _batch_size(self.last_batch)
        chunked_pending = self.chunked_req is not None
        pending_results = len(getattr(self, "result_queue", ())) if self.enable_overlap else 0

        has_pending = (
            waiting_reqs > 0
            or running_reqs > 0
            or current_batch_reqs > 0
            or last_batch_reqs > 0
            or chunked_pending
            or pending_results > 0
        )

        if has_pending:
            msg = (
                "Cache not flushed because there are pending requests. "
                f"waiting={waiting_reqs}, running={running_reqs}, "
                f"cur_batch={current_batch_reqs}, last_batch={last_batch_reqs}, "
                f"chunked={chunked_pending}, pending_results={pending_results}"
            )
            return False, msg

        return True, ""

    def flush_cache(self) -> tuple[bool, str, int]:
        can_flush, message = self._can_flush_cache()
        if not can_flush:
            logger.warning(message)
            return False, message, 0

        # Reset scheduling state
        self.cur_batch = None
        self.last_batch = None
        self.running_batch = ScheduleBatch(reqs=[], batch_is_full=False)
        self.chunked_req = None
        if self.enable_overlap:
            self.result_queue = deque()

        # Clear cache-related state
        if self.tree_cache is not None:
            self.tree_cache.reset()
        if self.req_to_token_pool is not None:
            self.req_to_token_pool.clear()
        if self.token_to_kv_pool_allocator is not None:
            self.token_to_kv_pool_allocator.clear()
        if self.grammar_backend is not None:
            self.grammar_backend.reset()

        self.num_generated_tokens = 0
        self.forward_ct_decode = 0
        self.new_token_ratio = self.init_new_token_ratio

        flushed_items = (
            self.token_to_kv_pool_allocator.available_size()
            if self.token_to_kv_pool_allocator is not None
            else 0
        )

        logger.info("Cache flushed successfully!")
        return True, "", flushed_items

    def _add_request_to_queue(self, req: Req):
        req.queue_time_start = time.perf_counter()
        req.queue_time_end = None
        self.waiting_queue.append(req)

    def _extend_requests_to_queue(self, reqs: list[Req], is_retracted: bool = False):
        if is_retracted:
            now = time.perf_counter()
            for req in reqs:
                req.queue_time_start = now
                req.queue_time_end = None
        self.waiting_queue.extend(reqs)

    def check_memory(self):
        if self.is_hybrid:
            (
                full_num_used,
                swa_num_used,
                _,
                _,
                full_available_size,
                full_evictable_size,
                swa_available_size,
                swa_evictable_size,
            ) = self._get_swa_token_info()
            # Strict mode: require perfect accounting with no tolerance
            full_protected = self.tree_cache.full_protected_size()
            swa_protected = self.tree_cache.swa_protected_size()
            memory_leak = (
                full_available_size + full_evictable_size + full_protected
            ) != self.full_tokens_per_layer or (
                swa_available_size + swa_evictable_size + swa_protected
            ) != self.swa_tokens_per_layer
            token_msg = (
                f"{self.full_tokens_per_layer=}, {full_available_size=}, {full_evictable_size=}, full_protected={full_protected} (used={full_num_used})\n"
                f"{self.swa_tokens_per_layer=}, {swa_available_size=}, {swa_evictable_size=}, swa_protected={swa_protected} (used={swa_num_used})\n"
            )
        else:
            _, _, available_size, evictable_size = self._get_token_info()
            protected_size = self.tree_cache.protected_size()
            memory_leak = (
                available_size + evictable_size + protected_size
            ) != self.max_total_num_tokens
            token_msg = f"{self.max_total_num_tokens=}, {available_size=}, {evictable_size=}, {protected_size=}\n"

        if memory_leak:
            msg = f"token_to_kv_pool_allocator memory leak detected! {token_msg}"
            raise ValueError(msg)

        req_total_size = self.req_to_token_pool.size

        if len(self.req_to_token_pool.free_slots) != req_total_size:
            msg = (
                "req_to_token_pool memory leak detected!"
                f"available_size={len(self.req_to_token_pool.free_slots)}, "
                f"total_size={self.req_to_token_pool.size}\n"
            )
            raise ValueError(msg)

    def check_tree_cache(self):
        if self.is_hybrid and isinstance(self.tree_cache, SWARadixCache):
            self.tree_cache.sanity_check()

    def _get_token_info(self):
        available_size = self.token_to_kv_pool_allocator.available_size()
        evictable_size = self.tree_cache.evictable_size()
        num_used = self.max_total_num_tokens - (available_size + evictable_size)
        token_usage = num_used / self.max_total_num_tokens
        return num_used, token_usage, available_size, evictable_size

    def _get_swa_token_info(self):
        full_available_size = self.token_to_kv_pool_allocator.full_available_size()
        full_evictable_size = self.tree_cache.full_evictable_size()
        swa_available_size = self.token_to_kv_pool_allocator.swa_available_size()
        swa_evictable_size = self.tree_cache.swa_evictable_size()
        full_num_used = self.full_tokens_per_layer - (full_available_size + full_evictable_size)
        swa_num_used = self.swa_tokens_per_layer - (swa_available_size + swa_evictable_size)
        full_token_usage = full_num_used / self.full_tokens_per_layer
        swa_token_usage = swa_num_used / self.swa_tokens_per_layer
        return (
            full_num_used,
            swa_num_used,
            full_token_usage,
            swa_token_usage,
            full_available_size,
            full_evictable_size,
            swa_available_size,
            swa_evictable_size,
        )

    def get_next_batch_to_run(self) -> ScheduleBatch | None:
        chunked_req_to_exclude = set()
        if self.chunked_req:
            # Move the chunked request out of the batch so that we can merge
            # only finished requests to running_batch.
            chunked_req_to_exclude.add(self.chunked_req)
            self.tree_cache.cache_unfinished_req(self.chunked_req)
            # chunked request keeps its rid but will get a new req_pool_idx
            self.req_to_token_pool.free(self.chunked_req.req_pool_idx)

        # Merge the prefill batch into the running batch
        if self.last_batch and self.last_batch.forward_mode.is_extend():
            if self.last_batch.chunked_req is not None:
                chunked_req_to_exclude.add(self.last_batch.chunked_req)

            # Filter batch
            last_bs = self.last_batch.batch_size()
            self.last_batch.filter_batch(chunked_req_to_exclude=list(chunked_req_to_exclude))
            if self.last_batch.batch_size() < last_bs:
                self.running_batch.batch_is_full = False

            # Merge the new batch into the running batch
            if not self.last_batch.is_empty() and not self.last_batch.is_prefill_only:
                if self.running_batch.is_empty():
                    self.running_batch = self.last_batch
                else:
                    # Merge running_batch with prefill batch
                    self.running_batch.merge_batch(self.last_batch)

        new_batch = self.get_new_batch_prefill()

        # if new_batch is not None:
        if new_batch:
            # Run prefill first if possible
            ret = new_batch
        else:
            # Run decode
            if not self.running_batch.is_empty():
                self.running_batch = self.update_running_batch(self.running_batch)
                ret = self.running_batch if not self.running_batch.is_empty() else None
            else:
                ret = None

        return ret

    def get_new_batch_prefill(self) -> ScheduleBatch | None:
        if self.grammar_queue:
            self.move_ready_grammar_requests()

        # `batch_is_full` is a soft throttle flag. If nothing is running, clear it so
        # prefill admission can resume and we don't get stuck in a full-but-idle state.
        if self.running_batch.is_empty() and self.running_batch.batch_is_full:
            self.running_batch.batch_is_full = False

        # Handle the cases where prefill is not allowed
        if (
            self.running_batch.batch_is_full or len(self.waiting_queue) == 0
        ) and self.chunked_req is None:
            return None

        running_bs = len(self.running_batch.reqs)
        if running_bs >= self.max_running_requests:
            self.running_batch.batch_is_full = True
            return None

        # ReqToTokenPool slots gate how many requests can enter EXTEND in this round.
        # Under prefill+extend scoring, a single user request can fan out into many
        # internal requests. If we ignore current slot pressure here, prepare_for_extend()
        # can raise and kill the scheduler process.
        req_slots_budget = self.req_to_token_pool.available_size()
        if req_slots_budget <= 0:
            self.running_batch.batch_is_full = True
            logger.debug("Deferring prefill: no req slots available in ReqToTokenPool.")
            return None

        # Get priority queue
        self.policy.calc_priority(self.waiting_queue)

        adder = PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            self.running_batch,
            self.new_token_ratio,
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
        )

        if self.chunked_req is not None:
            self.chunked_req.init_next_round_input()
            self.chunked_req = adder.add_chunked_req(self.chunked_req)

        # Collect existing LoRA IDs in the running batch if LoRA is enabled
        if self.lora_paths is not None:
            lora_set = (
                set([req.lora_id for req in self.running_batch.reqs])
                if self.running_batch is not None
                else set([])
            )

        # Get requests from the waiting queue to a new prefill batch
        for req in self.waiting_queue:
            if len(adder.can_run_list) >= req_slots_budget:
                self.running_batch.batch_is_full = True
                break

            if running_bs + len(adder.can_run_list) >= self.max_running_requests:
                self.running_batch.batch_is_full = True
                break

            # Check LoRA constraint: ensure we don't exceed max_loras_per_batch
            if (
                self.lora_paths is not None
                and len(
                    lora_set | set([req.lora_id for req in adder.can_run_list]) | set([req.lora_id])
                )
                > self.max_loras_per_batch
            ):
                break

            req.init_next_round_input(self.tree_cache)
            res = adder.add_one_req(req)

            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    self.running_batch.batch_is_full = True
                break

        # Update waiting queue
        can_run_list: list[Req] = adder.can_run_list
        if len(can_run_list) == 0:
            return None

        admit_ts = time.perf_counter()
        for req in can_run_list:
            if req.queue_time_start is None:
                continue
            req.queue_time_end = admit_ts
            req.queue_wait_time_s += max(0.0, req.queue_time_end - req.queue_time_start)
            req.queue_time_start = None

        self.log_prefill_stats(adder, can_run_list, running_bs)

        # Create a new batch
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            enable_custom_logit_processor=False,
            chunked_req=self.chunked_req,
            mesh=self.mesh,
            spec_algorithm=self.spec_algorithm,
        )

        new_batch.prepare_for_extend()

        # Update waiting queue and chunked request state only after we
        # successfully allocate req slots in prepare_for_extend().
        self.waiting_queue = [x for x in self.waiting_queue if x not in set(can_run_list)]

        if adder.new_chunked_req is not None and adder.new_chunked_req in set(can_run_list):
            assert self.chunked_req is None
            self.chunked_req = adder.new_chunked_req

        if self.chunked_req:
            self.chunked_req.is_chunked += 1

        # Mixed-style chunked prefill
        if (
            self.is_mixed_chunk
            and not self.running_batch.is_empty()
            and not (new_batch.return_logprob or self.running_batch.return_logprob)
        ):
            self.running_batch.filter_batch()
            if not self.running_batch.is_empty():
                self.running_batch.prepare_for_decode()
                new_batch.mix_with_running(self.running_batch)
                new_batch.decoding_reqs = self.running_batch.reqs

            self.running_batch = ScheduleBatch(
                reqs=[], batch_is_full=self.running_batch.batch_is_full, mesh=self.mesh
            )
        else:
            new_batch.decoding_reqs = None

        new_batch.bid = acc_global_bid()

        return new_batch

    def update_running_batch(self, batch: ScheduleBatch) -> ScheduleBatch | None:
        """Update the current running decoding batch."""
        initial_bs = batch.batch_size()

        batch.filter_batch()
        if batch.is_empty():
            batch.batch_is_full = False
            return batch

        # Check if decode out of memory
        if not batch.check_decode_mem(self.decode_mem_cache_buf_multiplier) or (
            TEST_RETRACT and batch.batch_size() > 10
        ):
            old_ratio = self.new_token_ratio

            retracted_reqs, new_token_ratio = batch.retract_decode(self.server_args)
            num_retracted_reqs = len(retracted_reqs)
            self.new_token_ratio = new_token_ratio

            logger.info(
                "KV cache pool is full. Retract requests. #retracted_reqs: %d, #new_token_ratio: %.4f -> %.4f",
                num_retracted_reqs,
                old_ratio,
                self.new_token_ratio,
            )

            self._extend_requests_to_queue(retracted_reqs, is_retracted=True)
        else:
            self.new_token_ratio = max(
                self.new_token_ratio - self.new_token_ratio_decay,
                self.min_new_token_ratio,
            )

        if batch.batch_size() < initial_bs:
            batch.batch_is_full = False

        # Update batch arrays
        batch.prepare_for_decode()
        return batch

    def run_batch(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Run a batch."""
        self.forward_ct += 1

        if self.server_args.log_requests:
            logger.debug(
                "Run batch: mode=%s, bs=%d, return_logprob=%s",
                batch.forward_mode,
                batch.batch_size(),
                batch.return_logprob,
            )

        # Whether to run the profiler
        self._profile_batch_predicate(batch)

        # Run forward
        assert self.is_generation
        batch_wall_start = time.perf_counter()
        forward_start = time.perf_counter()
        (
            precompile_token_paddings,
            precompile_bs_paddings,
            precompile_cache_loc_paddings,
        ) = self.tp_worker.get_precompile_paddings()
        if self.spec_algorithm is None or self.spec_algorithm.is_none():
            model_worker_batch = batch.get_model_worker_batch(
                precompile_token_paddings,
                precompile_bs_paddings,
                precompile_cache_loc_paddings,
                self.page_size,
                self.server_args.enable_static_lora,
            )

            if self.enable_overlap:
                with jax.profiler.TraceAnnotation(
                    f"forward_batch_generation_overlap {self.forward_ct}"
                ):

                    logits_output, next_token_ids, cache_miss_count = (
                        self.tp_worker.forward_batch_generation(
                            model_worker_batch, sampling_metadata=None
                        )
                    )
                next_token_ids = next_token_ids[: model_worker_batch.real_bs]
            else:
                logits_output, next_token_ids_device, cache_miss_count = (
                    self.tp_worker.forward_batch_generation(
                        model_worker_batch, sampling_metadata=None
                    )
                )
                next_token_ids = np.array(jax.device_get(next_token_ids_device))[
                    : model_worker_batch.real_bs
                ]
        else:
            model_worker_batch = batch.get_spec_model_worker_batch(
                precompile_token_paddings,
                precompile_bs_paddings,
                precompile_cache_loc_paddings,
                self.page_size,
                self.server_args.enable_static_lora,
            )
            batch_output = self.draft_worker.forward_batch_speculative_generation(
                model_worker_batch
            )
            if batch_output.accept_lens is not None:
                # Decode
                batch.seq_lens = batch.seq_lens + batch_output.accept_lens
            else:
                # Prefill
                batch.seq_lens = batch.seq_lens + 1
            batch.spec_info = batch_output.next_draft_input
            next_token_ids = batch_output.next_token_ids
            logits_output = batch_output.logits_output
            cache_miss_count = batch_output.cache_miss_count
        forward_end = time.perf_counter()
        batch_wall_end = time.perf_counter()
        bid = model_worker_batch.bid
        batch.output_ids = next_token_ids

        device_compute_s = max(0.0, forward_end - forward_start)
        host_overhead_s = max(0.0, (batch_wall_end - batch_wall_start) - device_compute_s)
        for req in batch.reqs:
            req.device_compute_time_s += device_compute_s
            req.host_overhead_time_s += host_overhead_s
            req.scheduler_dispatch_count += 1

        # These 2 values are needed for processing the output, but the values can be
        # modified by overlap schedule. So we have to copy them here so that
        # we can use the correct values in output processing.
        if batch.return_logprob:
            extend_input_len_per_req = [req.extend_input_len for req in batch.reqs]
        else:
            extend_input_len_per_req = None
        if batch.return_logprob:
            extend_logprob_start_len_per_req = [req.extend_logprob_start_len for req in batch.reqs]
        else:
            extend_logprob_start_len_per_req = None

        ret = GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids.tolist(),
            extend_input_len_per_req=extend_input_len_per_req,
            extend_logprob_start_len_per_req=extend_logprob_start_len_per_req,
            bid=bid,
            cache_miss_count=cache_miss_count,
        )
        if self.spec_algorithm is not None and self.spec_algorithm.is_eagle():
            assert isinstance(batch_output.next_draft_input, EagleDraftInput)
            ret.next_draft_input = batch_output.next_draft_input
            ret.accept_lens = batch_output.accept_lens
            ret.allocate_lens = batch_output.allocate_lens
        return ret

    def process_batch_result(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        launch_done: threading.Event | None = None,
    ):
        if batch.forward_mode.is_decode():
            self.process_batch_result_decode(batch, result, launch_done)
        elif batch.forward_mode.is_extend():
            self.process_batch_result_prefill(batch, result, launch_done)
        elif batch.forward_mode.is_idle():
            if self.enable_overlap:
                self.tp_worker.resolve_last_batch_result(launch_done)
                self.set_next_batch_sampling_info_done(batch)
        elif batch.forward_mode.is_dummy_first():
            self.set_next_batch_sampling_info_done(batch)

    def get_idle_batch(self):
        idle_batch = ScheduleBatch.init_new(
            [],
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.server_args.enable_custom_logit_processor,
            self.mesh,
            spec_algorithm=self.spec_algorithm,
        )
        idle_batch.prepare_for_idle()
        return idle_batch

    def set_next_batch_sampling_info_done(self, batch: ScheduleBatch):
        if batch.next_batch_sampling_info:
            # Update grammar vocab masks for next batch in overlap mode
            if batch.next_batch_sampling_info.grammars is not None:
                batch.next_batch_sampling_info.update_grammar_vocab_mask()
            batch.next_batch_sampling_info.sampling_info_done.set()

    def watchdog_thread(self):
        """A watch dog thread that will try to kill the server itself if one forward batch takes too long."""
        self.watchdog_last_forward_ct = 0
        self.watchdog_last_time = time.perf_counter()

        while True:
            current = time.perf_counter()
            if self.cur_batch is not None:
                if self.watchdog_last_forward_ct == self.forward_ct:
                    if current > self.watchdog_last_time + self.watchdog_timeout:
                        break
                else:
                    self.watchdog_last_forward_ct = self.forward_ct
                    self.watchdog_last_time = current
            time.sleep(self.watchdog_timeout // 2)

        pyspy_dump_schedulers()
        logger.error("Watchdog timeout (watchdog_timeout=%s)", self.watchdog_timeout)
        print(file=sys.stderr, flush=True)
        print(file=sys.stdout, flush=True)

        # Wait for some time so that the parent process can print the error.
        time.sleep(5)
        self.parent_process.send_signal(signal.SIGQUIT)

    def abort_request(self, recv_req: AbortReq):
        # Delete requests in the waiting queue
        to_del = []
        for i, req in enumerate(self.waiting_queue):
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                to_del.append(i)

        # Sort in reverse order to avoid index issues when deleting
        for i in reversed(to_del):
            # Abort method 1: directly pop from the queue
            # This only works for requests that have not started anything.
            # We still need to send something back to TokenizerManager to clean up the state.
            req = self.waiting_queue.pop(i)
            self.send_to_tokenizer.send_pyobj(AbortReq(rid=req.rid))
            logger.debug("Abort queued request. rid=%s", req.rid)

        # Delete the requests in the grammar queue
        for req in self.grammar_queue:
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                logger.debug("Abort grammar queue request. rid=%s", req.rid)
                if req.grammar:
                    req.grammar.cancel()
                req.set_finish_with_abort("Aborted by AbortReq.")

        # Delete requests in the running batch
        if self.cur_batch is self.running_batch or self.cur_batch is None:
            reqs = self.running_batch.reqs
        else:
            reqs = self.running_batch.reqs + self.cur_batch.reqs

        for req in reqs:
            if not req.finished() and (recv_req.abort_all or req.rid.startswith(recv_req.rid)):
                # Abort method 3: set `to_finish`
                # The request will still run one decode forward pass.
                # Then we reuse all existing code to clean up the KV cache allocation.
                logger.debug("Abort running request. rid=%s", req.rid)
                req.to_finish = FINISH_ABORT()

        # Abort method 4: Release cached nodes for prefill+extend
        self._release_scoring_cache_nodes(recv_req.rid, recv_req.abort_all)

    def _release_scoring_cache_nodes(self, rid_prefix: str | None, abort_all: bool) -> int:
        released = 0
        self._evict_expired_scoring_cache_nodes()
        if not abort_all and not rid_prefix:
            return released

        rids_to_remove = []
        for rid in self.scoring_cache_nodes:
            if abort_all or (rid_prefix and rid.startswith(rid_prefix)):
                rids_to_remove.append(rid)

        for rid in rids_to_remove:
            entry = self.scoring_cache_nodes.pop(rid, None)
            if entry is None:
                continue
            self._release_scoring_cache_entry(rid, entry, reason="manual")
            released += 1
            logger.debug("Released cached node for rid=%s", rid)
        return released

    def release_scoring_cache(
        self, recv_req: ReleaseScoringCacheReqInput
    ) -> ReleaseScoringCacheReqOutput:
        released = self._release_scoring_cache_nodes(recv_req.rid, abort_all=False)
        return ReleaseScoringCacheReqOutput(
            rid=recv_req.rid,
            success=True,
            released_items=released,
        )

    def pause_generation(self, recv_req: PauseGenerationReqInput):
        self._engine_paused = True

        # finish all in-flight request; in overlap mode, last_batch is running
        if self.enable_overlap and self.last_batch:
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)
            self.last_batch = None
            self.cur_batch = None

        if recv_req.mode == "retract":
            self.running_batch.filter_batch()
            if len(self.running_batch.reqs) != 0:
                # clear the kv cache
                retracted_reqs = self.running_batch.retract_all(self.server_args)
                for req in retracted_reqs:
                    self._add_request_to_queue(req)

            self.running_batch.batch_is_full = False
            self.chunked_req = None
            logger.info("Paused generation retracted")
        elif recv_req.mode == "in_place":
            logger.info("Paused generation in place")

    def continue_generation(self, recv_req: ContinueGenerationReqInput):
        self._engine_paused = False
        logger.info("Generation continued")


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    dp_rank: int | None,
    pipe_writer,
):
    def maybe_freeze_gc_after_warmup():
        if not getattr(server_args, "enable_gc_freeze", False):
            return
        if not hasattr(gc, "freeze"):
            logger.warning(
                "GC freeze requested but gc.freeze is unavailable on this Python runtime."
            )
            return
        try:
            freeze_before = gc.get_freeze_count() if hasattr(gc, "get_freeze_count") else -1
            collected = gc.collect()
            gc.freeze()
            freeze_after = gc.get_freeze_count() if hasattr(gc, "get_freeze_count") else -1
            logger.info(
                "Applied gc.freeze after warmup/precompile. collected=%d freeze_before=%d freeze_after=%d gc_count=%s",
                collected,
                freeze_before,
                freeze_after,
                gc.get_count(),
            )
            if getattr(server_args, "gc_freeze_rollback", False):
                if hasattr(gc, "unfreeze"):
                    gc.unfreeze()
                    rollback_count = (
                        gc.get_freeze_count() if hasattr(gc, "get_freeze_count") else -1
                    )
                    logger.warning(
                        "Rolled back gc.freeze due to --gc-freeze-rollback. freeze_count_after_rollback=%d gc_count=%s",
                        rollback_count,
                        gc.get_count(),
                    )
                else:
                    logger.warning(
                        "GC freeze rollback requested but gc.unfreeze is unavailable on this Python runtime."
                    )
        except Exception:
            logger.exception("Failed to apply gc.freeze after warmup/precompile.")

    # Generate the prefix
    prefix = ""
    if server_args.nnodes > 1:
        prefix += f" NP{server_args.node_rank}"

    # Config the process
    kill_itself_when_parent_died()
    setproctitle.setproctitle(f"sglang::scheduler{prefix.replace(' ', '_')}")
    faulthandler.enable()
    parent_process = psutil.Process().parent()

    # Configure the logger
    configure_logger(server_args, prefix=prefix)

    # Create a scheduler and run the event loop
    try:
        scheduler = Scheduler(server_args, port_args)
        maybe_freeze_gc_after_warmup()
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": scheduler.max_total_num_tokens,
                "max_req_input_len": scheduler.max_req_input_len,
            }
        )

        if scheduler.enable_overlap:
            scheduler.event_loop_overlap()
        else:
            scheduler.event_loop_normal()

    except Exception:
        traceback = get_exception_traceback()
        logger.error("Scheduler hit an exception: %s", traceback)
        parent_process.send_signal(signal.SIGQUIT)


def run_scheduler_loop_thread_after_create(
    server_args: ServerArgs,
    port_args: PortArgs,
):
    def maybe_freeze_gc_after_warmup():
        if not getattr(server_args, "enable_gc_freeze", False):
            return
        if not hasattr(gc, "freeze"):
            logger.warning(
                "GC freeze requested but gc.freeze is unavailable on this Python runtime."
            )
            return
        try:
            freeze_before = gc.get_freeze_count() if hasattr(gc, "get_freeze_count") else -1
            collected = gc.collect()
            gc.freeze()
            freeze_after = gc.get_freeze_count() if hasattr(gc, "get_freeze_count") else -1
            logger.info(
                "Applied gc.freeze after warmup/precompile. collected=%d freeze_before=%d freeze_after=%d gc_count=%s",
                collected,
                freeze_before,
                freeze_after,
                gc.get_count(),
            )
            if getattr(server_args, "gc_freeze_rollback", False):
                if hasattr(gc, "unfreeze"):
                    gc.unfreeze()
                    rollback_count = (
                        gc.get_freeze_count() if hasattr(gc, "get_freeze_count") else -1
                    )
                    logger.warning(
                        "Rolled back gc.freeze due to --gc-freeze-rollback. freeze_count_after_rollback=%d gc_count=%s",
                        rollback_count,
                        gc.get_count(),
                    )
                else:
                    logger.warning(
                        "GC freeze rollback requested but gc.unfreeze is unavailable on this Python runtime."
                    )
        except Exception:
            logger.exception("Failed to apply gc.freeze after warmup/precompile.")

    current_process = psutil.Process()
    # Create a scheduler and run the event loop
    try:
        scheduler = Scheduler(server_args, port_args)
        maybe_freeze_gc_after_warmup()
        scheduler_thread = threading.Thread(
            target=scheduler_loop_after_create,
            args=(server_args, scheduler),
            daemon=True,
        )
        scheduler_thread.start()
        return {
            "status": "ready",
            "max_total_num_tokens": scheduler.max_total_num_tokens,
            "max_req_input_len": scheduler.max_req_input_len,
            "scheduler": scheduler,
        }
    except Exception:
        traceback = get_exception_traceback()
        logger.error("Scheduler hit an exception: %s", traceback)
        current_process.send_signal(signal.SIGQUIT)


def scheduler_loop_after_create(server_args, scheduler):
    # Generate the prefix
    prefix = ""
    if server_args.nnodes > 1:
        prefix += f" NP{server_args.node_rank}"

    # Config the process
    current_thread = threading.current_thread()
    current_thread.name = f"sglang::scheduler{prefix.replace(' ', '_')}"
    faulthandler.enable()
    current_process = psutil.Process()

    # Configure the logger
    configure_logger(server_args, prefix=prefix)
    try:
        if scheduler.enable_overlap:
            scheduler.event_loop_overlap()
        else:
            scheduler.event_loop_normal()
    except Exception:
        traceback = get_exception_traceback()
        logger.error("Scheduler hit an exception: %s", traceback)
        current_process.send_signal(signal.SIGQUIT)
