"""A scheduler that manages a tensor parallel TPU worker."""

import faulthandler
import logging
import os
import pickle
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
# No need to import Optional if using `X | None` syntax, which is already prevalent.

import jax
import numpy as np
import psutil
import setproctitle
import zmq

from sgl_jax.global_config import global_config
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.hf_transformers_utils import get_tokenizer
from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
from sgl_jax.srt.managers.io_struct import (
    AbortReq,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    ProfileReq,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    TokenizedGenerateReqInput,
)
from sgl_jax.srt.managers.schedule_batch import (
    Req,
    ScheduleBatch,
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
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.precision_tracer import precision_tracer
from sgl_jax.srt.server_args import PortArgs, ServerArgs
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


class SyncError(Exception):
    pass


class SendDataError(Exception):
    pass


class ReceiveDataError(Exception):
    pass


@dataclass
class GenerationBatchResult:
    logits_output: LogitsProcessorOutput | None
    next_token_ids: list[int] | None  # on device
    extend_input_len_per_req: list[int]
    extend_logprob_start_len_per_req: list[int]
    bid: int
    cache_miss_count: int


class Scheduler(
    SchedulerOutputProcessorMixin,
    SchedulerProfilerMixin,
    SchedulerMetricsMixin,
):
    """
    A scheduler that manages a tensor parallel TPU worker, which manages fixed multi TPU devices.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
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
        self.pub_sub_addr = port_args.pub_sub_addr
        self.pub_sub_sync_addr = port_args.pub_sub_sync_addr
        self.tp_size = server_args.tp_size
        self.schedule_policy = server_args.schedule_policy
        self.skip_tokenizer_init = server_args.skip_tokenizer_init
        self.stream_interval = server_args.stream_interval
        self.max_seq_len = server_args.max_seq_len
        self.page_size = server_args.page_size
        self.enable_overlap = not server_args.disable_overlap_schedule

        # Init inter-process communication
        context = zmq.Context(2)

        if self.node_rank == 0:
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

            self.recv_from_rpc = get_zmq_socket(context, zmq.DEALER, port_args.rpc_ipc_name, False)
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
            # Perform synchronization between nodes if running in a distributed setup
            self.sync_pub_sub()

        # Init tokenizer
        self.init_tokenizer()

        if not self.is_generation:
            self.enable_overlap = False
            logger.info("Overlap scheduler is disabled for embedding models.")

        # init distribution
        if self.nnodes > 1:
            jax.distributed.initialize(server_args.dist_init_addr, self.nnodes, self.node_rank)
        self.mesh = create_device_mesh(
            ici_parallelism=[-1, self.tp_size, 1], dcn_parallelism=[1, 1, 1]
        )

        TpWorkerClass = ModelWorkerClient if self.enable_overlap else ModelWorker

        self.tp_worker = TpWorkerClass(
            server_args=server_args,
            mesh=self.mesh,
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

        # Init memory pool and cache
        self.init_memory_pool_and_cache()

        # Init running status
        self.waiting_queue: list[Req] = []
        # The aborted requests (dict[str, Req] for efficient lookup and context storage)
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

        # Init chunked prefill
        self.chunked_prefill_size = server_args.chunked_prefill_size
        if self.chunked_prefill_size <= 0:  # -1 means disable
            self.chunked_prefill_size = None
        self.chunked_req = None
        self.is_mixed_chunk = (
            self.chunked_prefill_size is not None and server_args.enable_mixed_chunk
        )

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
                (GetInternalStateReq, self.get_internal_state),
                (SetInternalStateReq, self.set_internal_state),
            ]
        )

        if not server_args.disable_jax_precompile:
            logger.info("[Scheduler] Begins to run worker precompile.")
            self.tp_worker.run_precompile()
            logger.info("[Scheduler] Completes worker precompile.")

    def sync_pub(self) -> bool:
        """
        Publisher (node 0) synchronization logic. Waits for all subscribers to be ready.
        """
        logger.info(
            "[Publisher %s] Begins to synchronize, waiting for %s Subscribers",
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
                        "[Publisher %s] Received %s READY signal",
                        self.node_rank,
                        ready_count,
                    )
                    self.publisher_sync.send_string("ACK")
                else:
                    logger.warning(
                        "[Publisher %s] Received unexpected message during sync: %s",
                        self.node_rank, message
                    )
                    self.publisher_sync.send_string("NACK")
        except zmq.Again:
            logger.error("[Publisher %s] Failed to synchronize due to timeout", self.node_rank)
            return False
        except Exception as e:
            logger.error("[Publisher %s] Encountered error during sync: %s", self.node_rank, e)
            return False
        logger.info("[Publisher %s] Successfully synchronized!", self.node_rank)
        return True

    def sync_sub(self) -> bool:
        """
        Subscriber synchronization logic. Sends a READY signal and waits for ACK from publisher.
        """
        logger.info("[Subscriber %s] Begins to synchronize with publisher", self.node_rank)
        try:
            self.subscriber_sync.send_string("READY")
            ack = self.subscriber_sync.recv_string()
            if ack == "ACK":
                logger.info("[Subscriber %s] Successfully synchronized!", self.node_rank)
                return True
            else:
                logger.error(
                    "[Subscriber %s] Failed to synchronize. Received unexpected ACK: %s",
                    self.node_rank, ack
                )
                return False
        except zmq.Again:
            logger.error(
                "[Subscriber %s] Failed to synchronize due to timeout when receiving ACK",
                self.node_rank
            )
            return False
        except Exception as e:
            logger.error("[Subscriber %s] Encountered error during sync: %s", self.node_rank, e)
            return False

    def sync_pub_sub(self) -> bool:
        """
        Orchestrates synchronization across nodes.
        Node 0 acts as the publisher, other nodes as subscribers.
        """
        if self.node_rank == 0:
            return self.sync_pub()
        else:
            return self.sync_sub()