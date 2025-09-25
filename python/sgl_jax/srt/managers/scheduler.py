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
from typing import Dict, List, Optional, Union

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
from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata
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
    logits_output: Optional[LogitsProcessorOutput]
    next_token_ids: Optional[List[int]]  # on device
    extend_input_len_per_req: List[int]
    extend_logprob_start_len_per_req: List[int]
    bid: int
    cache_miss_count: int


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

            self.recv_from_rpc = get_zmq_socket(
                context, zmq.DEALER, port_args.rpc_ipc_name, False
            )
            if self.nnodes > 1:
                self.publisher = get_zmq_socket(
                    context, zmq.PUB, self.pub_sub_addr, bind=True
                )
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
                self.subscriber = get_zmq_socket(
                    context, zmq.SUB, self.pub_sub_addr, bind=False
                )
                self.subscriber.setsockopt(zmq.SUBSCRIBE, b"")
                self.subscriber.setsockopt(zmq.RCVTIMEO, 5000)
                self.subscriber_sync = get_zmq_socket(
                    context, zmq.REQ, self.pub_sub_sync_addr, bind=False
                )

        if self.nnodes > 1:
            self.sync_pub_sub()

        # Init tokenizer
        self.init_tokenizer()

        if not self.is_generation:
            self.enable_overlap = False
            logger.info("Overlap scheduler is disabled for embedding models.")

        # init distribution
        if self.nnodes > 1:
            jax.distributed.initialize(
                server_args.dist_init_addr, self.nnodes, self.node_rank
            )
        self.mesh = create_device_mesh(
            ici_parallelism=[-1, self.tp_size, 1], dcn_parallelism=[1, 1, 1]
        )

        if self.enable_overlap:
            TpWorkerClass = ModelWorkerClient
        else:
            TpWorkerClass = ModelWorker

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
        self.waiting_queue: List[Req] = []
        # The aborted requests
        self.aborted_reqs: Dict[str, Req] = {}
        # The running decoding batch for continuous batching
        self.running_batch: ScheduleBatch = ScheduleBatch(reqs=[], batch_is_full=False)
        # The current forward batch
        self.cur_batch: Optional[ScheduleBatch] = None
        # The last forward batch
        self.last_batch: Optional[ScheduleBatch] = None
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
        assert (
            server_args.schedule_conservativeness >= 0
        ), "Invalid schedule_conservativeness"
        self.init_new_token_ratio = min(
            global_config.default_init_new_token_ratio
            * server_args.schedule_conservativeness,
            1.0,
        )
        self.min_new_token_ratio = min(
            self.init_new_token_ratio
            * global_config.default_min_new_token_ratio_factor,
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
            logger.info(f"[Scheduler] Begins to run worker precompile.")
            self.tp_worker.run_precompile()
            logger.info(f"[Scheduler] Completes worker precompile.")

    def sync_pub(self):
        logger.info(
            f"[Publisher {self.node_rank}] Begins to synchronize, wait {self.nnodes-1} Subscribers"
        )
        ready_count = 0
        try:
            while ready_count < self.num_subscribers:
                message = self.publisher_sync.recv_string()
                if message == "READY":
                    ready_count += 1
                    logger.info(
                        f"[Publisher {self.node_rank}] receives {ready_count} READY signal"
                    )
                    self.publisher_sync.send_string("ACK")
                else:
                    self.publisher_sync.send_string("NACK")
        except zmq.Again:
            logger.error(
                f"[Publisher {self.node_rank}] Fails to synchronize due to timeout"
            )
            return False
        except Exception as e:
            logger.error(f"[Publisher {self.node_rank}] Encounters error: {e}")
            return False
        logger.info(f"[Publisher {self.node_rank}] Succeeds to synchronize!")
        return True

    def sync_sub(self):
        logger.info(f"[Subscriber {self.node_rank}] Begins to synchronize")
        try:
            self.subscriber_sync.send_string("READY")
            ack = self.subscriber_sync.recv_string()
            if ack == "ACK":
                logger.info(f"[Subscriber {self.node_rank}] Succeeds to synchronizes!")
                return True
            else:
                logger.error(
                    f"[Subscriber {self.node_rank}] Fails to synchroinze with ack: {ack}"
                )
                return False
        except Exception as e:
            logger.error(
                f"[Subscriber {self.node_rank}] Fails to synchronize with error: {e}"
            )
            return False

    def sync_pub_sub(self):
        success = False
        if self.node_rank == 0:
            success = self.sync_pub()
        else:
            success = self.sync_sub()
        if not success:
            raise SyncError("Fail to synchronize between publisher and subscribers")

    def init_tokenizer(self):
        server_args = self.server_args
        self.model_config = ModelConfig.from_server_args(server_args)
        self.is_generation = self.model_config.is_generation
        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            self.tokenizer = get_tokenizer(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
            )

    def init_memory_pool_and_cache(self):
        server_args = self.server_args
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            self.tp_worker.get_memory_pool()
        )

        if (
            server_args.chunked_prefill_size is not None
            and server_args.disable_radix_cache
        ):
            self.tree_cache = ChunkCache(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
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
            )

        self.decode_mem_cache_buf_multiplier = 1

    def event_loop_normal(self):
        """A normal scheduler loop."""
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
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

            self.last_batch = batch

    def event_loop_overlap(self):
        """A scheduler loop that overlaps the CPU processing and Accelerator computation."""
        self.result_queue = deque()

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                batch.launch_done = threading.Event()
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
                    f"[Publisher {self.node_rank}] Fails to send data with error: {e}"
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
                    f"[Subscriber {self.node_rank}] Fails to receive data with timeout, and try again"
                )
            except Exception as e:
                logger.error(
                    f"[Subscriber {self.node_rank}] Fails to receive or deserialize with error: {e}, and try again"
                )
        return None

    def broadcast_pyobj(self, recv_reqs):
        if self.node_rank == 0:
            if not self.run_publisher(recv_reqs):
                raise SendDataError(f"[Publisher {self.node_rank}] Fails to send data")
        else:
            recv_reqs = self.run_subscriber()
            if recv_reqs is None:
                raise ReceiveDataError(
                    f"[Subscriber {self.node_rank}] Fails to receive data"
                )
        return recv_reqs

    def recv_requests(self) -> List[Req]:
        """Receive results at node_rank = 0 and broadcast it to all other Node ranks."""
        if self.node_rank == 0:
            recv_reqs = []

            while True:
                try:
                    recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                recv_reqs.append(recv_req)

            while True:
                try:
                    recv_rpc = self.recv_from_rpc.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                recv_reqs.append(recv_rpc)
        else:
            recv_reqs = None

        if self.nnodes > 1:
            recv_reqs = self.broadcast_pyobj(recv_reqs)
        return recv_reqs

    def process_input_requests(self, recv_reqs: List):
        for recv_req in recv_reqs:
            output = self._request_dispatcher(recv_req)
            if output is not None:
                self.send_to_tokenizer.send_pyobj(output)

    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
    ):
        # Create a new request
        req = Req(
            recv_req.rid,
            recv_req.text,
            recv_req.input_ids,
            recv_req.sampling_params,
            return_logprob=recv_req.return_logprob,
            top_logprobs_num=recv_req.top_logprobs_num,
            token_ids_logprob=recv_req.token_ids_logprob,
            stream=recv_req.stream,
            eos_token_ids=self.model_config.hf_eos_token_id,
            vocab_size=self.model_config.vocab_size,
        )
        req.tokenizer = self.tokenizer

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

        self._add_request_to_queue(req)

    def get_internal_state(self, recv_req: GetInternalStateReq):
        ret = dict(global_server_args_dict)
        ret["last_gen_throughput"] = self.last_gen_throughput
        ret["memory_usage"] = {
            "kvcache": round(
                self.token_to_kv_pool_allocator.get_kvcache().mem_usage, 2
            ),
            "token_capacity": int(self.max_total_num_tokens),
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
                        f"[SCHEDULER] check trace_active: {precision_tracer.get_trace_active()}"
                    )
                    precision_tracer._trace_active = tracer_config["trace_active"]
                    logger.info(
                        f"[SCHEDULER] Updated trace_active to: {precision_tracer._trace_active}"
                    )

                    # Reset counters when starting trace
                    if tracer_config["trace_active"]:
                        precision_tracer._request_counter = 0
                        precision_tracer._completed_requests_count = 0
                        precision_tracer._request_traces = {}
                        logger.info(
                            f"[SCHEDULER] Reset request_counter, completed_count and traces"
                        )

                if "max_requests" in tracer_config:
                    precision_tracer._max_requests = tracer_config["max_requests"]
                    logger.info(
                        f"[SCHEDULER] Updated max_requests to: {precision_tracer._max_requests}"
                    )

                if "output_file" in tracer_config:
                    precision_tracer._trace_output_file = tracer_config["output_file"]
                    logger.info(
                        f"[SCHEDULER] Updated output_file to: {precision_tracer._trace_output_file}"
                    )

                logger.info(
                    f"[SCHEDULER] Precision tracer state updated: {tracer_config}"
                )

        except Exception as e:
            success = False
            error_msg = str(e)
            logger.info(f"[SCHEDULER] Error updating internal state: {error_msg}")

        return SetInternalStateReqOutput(
            request_id=recv_req.request_id, success=success, error_msg=error_msg
        )

    def _add_request_to_queue(self, req: Req):
        req.queue_time_start = time.perf_counter()
        self.waiting_queue.append(req)

    def _extend_requests_to_queue(self, reqs: List[Req], is_retracted: bool = False):
        self.waiting_queue.extend(reqs)

    def check_memory(self):
        _, _, available_size, evictable_size = self._get_token_info()
        protected_size = self.tree_cache.protected_size()
        memory_leak = (available_size + evictable_size) != self.max_total_num_tokens
        token_msg = f"{self.max_total_num_tokens=}, {available_size=}, {evictable_size=}, {protected_size=}\n"

        if memory_leak:
            msg = "token_to_kv_pool_allocator memory leak detected! " f"{token_msg}"
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
        pass

    def _get_token_info(self):
        available_size = self.token_to_kv_pool_allocator.available_size()
        evictable_size = self.tree_cache.evictable_size()
        num_used = self.max_total_num_tokens - (available_size + evictable_size)
        token_usage = num_used / self.max_total_num_tokens
        return num_used, token_usage, available_size, evictable_size

    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
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
            self.last_batch.filter_batch(
                chunked_req_to_exclude=list(chunked_req_to_exclude)
            )
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

    def get_new_batch_prefill(self) -> Optional[ScheduleBatch]:
        # Handle the cases where prefill is not allowed
        if (
            self.running_batch.batch_is_full or len(self.waiting_queue) == 0
        ) and self.chunked_req is None:
            return None

        running_bs = len(self.running_batch.reqs)
        if running_bs >= self.max_running_requests:
            self.running_batch.batch_is_full = True
            return None

        # Get priority queue
        self.policy.calc_priority(self.waiting_queue)

        # Prefill policy
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

        # Get requests from the waiting queue to a new prefill batch
        for req in self.waiting_queue:
            if req.finished():
                self.aborted_reqs[req.rid] = req
                continue

            if running_bs + len(adder.can_run_list) >= self.max_running_requests:
                self.running_batch.batch_is_full = True
                break

            req.init_next_round_input(self.tree_cache)
            res = adder.add_one_req(req)

            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    self.running_batch.batch_is_full = True
                break

        # Update waiting queue
        can_run_list: List[Req] = adder.can_run_list
        if len(can_run_list) == 0:
            return None

        self.waiting_queue = [
            x for x in self.waiting_queue if x not in set(can_run_list)
        ]

        if adder.new_chunked_req is not None:
            assert self.chunked_req is None
            self.chunked_req = adder.new_chunked_req

        if self.chunked_req:
            self.chunked_req.is_chunked += 1

        self.log_prefill_stats(adder, can_run_list, running_bs)

        # Create a new batch
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            False,
            self.chunked_req,
            self.mesh,
        )

        new_batch.prepare_for_extend()

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

        return new_batch

    def update_running_batch(self, batch: ScheduleBatch) -> Optional[ScheduleBatch]:
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
                "KV cache pool is full. Retract requests. "
                f"#retracted_reqs: {num_retracted_reqs}, "
                f"#new_token_ratio: {old_ratio:.4f} -> {self.new_token_ratio:.4f}"
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

    def run_batch(self, batch: ScheduleBatch) -> Union[GenerationBatchResult]:
        """Run a batch."""
        self.forward_ct += 1

        # Whether to run the profiler
        self._profile_batch_predicate(batch)

        # Run forward
        assert self.is_generation

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
        )

        sampling_metadata = SamplingMetadata.from_model_worker_batch(
            model_worker_batch,
            len(model_worker_batch.seq_lens) - model_worker_batch.real_bs,
            self.mesh,
        )

        if self.enable_overlap:
            # Pre-initialize ForwardBatch for overlap scheduling optimization
            from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

            model_worker_batch.forward_batch = ForwardBatch.init_new(
                model_worker_batch, self.tp_worker.get_model_runner()
            )
            logits_output, next_token_ids, cache_miss_count = (
                self.tp_worker.forward_batch_generation(
                    model_worker_batch, sampling_metadata=sampling_metadata
                )
            )
            next_token_ids = next_token_ids[: model_worker_batch.real_bs]
        else:
            logits_output, next_token_ids_device, cache_miss_count = (
                self.tp_worker.forward_batch_generation(
                    model_worker_batch, sampling_metadata=sampling_metadata
                )
            )
            next_token_ids = np.array(jax.device_get(next_token_ids_device))[
                : model_worker_batch.real_bs
            ]

        bid = model_worker_batch.bid
        batch.output_ids = next_token_ids

        # These 2 values are needed for processing the output, but the values can be
        # modified by overlap schedule. So we have to copy them here so that
        # we can use the correct values in output processing.
        if batch.return_logprob:
            extend_input_len_per_req = [req.extend_input_len for req in batch.reqs]
        else:
            extend_input_len_per_req = None
        if batch.return_logprob:
            extend_logprob_start_len_per_req = [
                req.extend_logprob_start_len for req in batch.reqs
            ]
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

        return ret

    def process_batch_result(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult],
        launch_done: Optional[threading.Event] = None,
    ):

        if batch.forward_mode.is_decode():
            self.process_batch_result_decode(batch, result)
        elif batch.forward_mode.is_extend():
            self.process_batch_result_prefill(batch, result)
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
        )
        idle_batch.prepare_for_idle()
        return idle_batch

    def set_next_batch_sampling_info_done(self, batch: ScheduleBatch):
        if batch.next_batch_sampling_info:
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
        logger.error(f"Watchdog timeout ({self.watchdog_timeout=})")
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
            self.send_to_tokenizer.send_pyobj(AbortReq(req.rid))
            logger.debug(f"Abort queued request. {req.rid=}")

        # Delete requests in the running batch
        if self.cur_batch is self.running_batch or self.cur_batch is None:
            reqs = self.running_batch.reqs
        else:
            reqs = self.running_batch.reqs + self.cur_batch.reqs

        for req in reqs:
            if not req.finished() and (
                recv_req.abort_all or req.rid.startswith(recv_req.rid)
            ):
                # Abort method 3: set `to_abort=True`
                # The request will still run one decode forward pass.
                # Then we reuse all existing code to clean up the KV cache allocation.
                logger.debug(f"Abort running request. {req.rid=}")
                req.to_abort = True


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    dp_rank: Optional[int],
    pipe_writer,
):
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
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)


def run_scheduler_thread(
    server_args: ServerArgs,
    port_args: PortArgs,
    dp_rank: Optional[int],
    pipe_writer,
):
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

    # Create a scheduler and run the event loop
    try:
        scheduler = Scheduler(server_args, port_args)
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
        logger.error(f"Scheduler hit an exception: {traceback}")
        current_process.send_signal(signal.SIGQUIT)
