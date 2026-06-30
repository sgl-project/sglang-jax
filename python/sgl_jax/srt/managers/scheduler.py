"""A scheduler that manages a tensor parallel TPU worker."""

import concurrent.futures as futures
import dataclasses
import faulthandler
import json
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

import jax
import numpy as np
import pathwaysutils
import psutil
import setproctitle
import zmq

from sgl_jax.global_config import global_config
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.constrained.base_grammar_backend import (
    INVALID_GRAMMAR_OBJ,
    create_grammar_backend,
)
from sgl_jax.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
from sgl_jax.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sgl_jax.srt.disaggregation.runtime import install_disaggregation_wiring
from sgl_jax.srt.hf_transformers_utils import get_tokenizer
from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
from sgl_jax.srt.managers.communication import CommunicationBackend
from sgl_jax.srt.managers.dp_schedule_policy import (
    pick_cache_aware_dp,
    req_prefix_match_key,
)
from sgl_jax.srt.managers.io_struct import (
    AbortReq,
    ContinueGenerationReqInput,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    PauseGenerationReqInput,
    ProfileReq,
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
    CLIP_MAX_NEW_TOKENS_ESTIMATION,
    IGNORE_EOS_RESERVE_TOKENS,
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
from sgl_jax.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sgl_jax.srt.mem_cache.chunk_cache import ChunkCache
from sgl_jax.srt.mem_cache.kv_cache_builder import build_kv_cache
from sgl_jax.srt.mem_cache.radix_cache import RadixKey
from sgl_jax.srt.mem_cache.swa_radix_cache import SWARadixCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.multimodal.tokenizer_utils import resolve_tokenizer_subdir
from sgl_jax.srt.precision_tracer import precision_tracer
from sgl_jax.srt.server_args import PortArgs, ServerArgs
from sgl_jax.srt.speculative.eagle_util import EagleDraftInput
from sgl_jax.srt.speculative.overlap_utils import (
    can_use_spec_decode_overlap,
    can_use_spec_prefill_overlap,
    publish_spec_decode_new_seq_lens,
    use_legacy_eagle3_non_overlap,
)
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
TEST_RETRACT_INTERVAL = int(os.environ.get("SGLANG_TEST_RETRACT_INTERVAL", "3"))
TEST_RETRACT_NO_PREFILL_BS = int(os.environ.get("SGLANG_TEST_RETRACT_NO_PREFILL_BS", str(2**31)))
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
    next_token_ids: object | None
    extend_input_len_per_req: list[int]
    extend_logprob_start_len_per_req: list[int]
    bid: int
    cache_miss_count: int
    # relay path: forward stream -> next step forward
    next_draft_input: EagleDraftInput | None = None
    spec_relay_buffers: object | None = None
    prefill_relay_future_indices: object | None = None

    num_accepted_tokens: int | None = None
    accept_lens: np.ndarray | None = None


class Scheduler(
    SchedulerOutputProcessorMixin,
    SchedulerProfilerMixin,
    SchedulerMetricsMixin,
    SchedulerDisaggregationPrefillMixin,
    SchedulerDisaggregationDecodeMixin,
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
        precompile_params: dict | None = None,
    ):
        if stage_sub_dir is not None:
            server_args = dataclasses.replace(server_args)
            server_args.model_sub_dir = stage_sub_dir
        self._setup_jit_cache(server_args)

        # Parse args
        self.server_args = server_args
        self.node_rank = server_args.node_rank
        self.nnodes = server_args.nnodes
        if port_args is not None:
            self.pub_sub_addr = port_args.pub_sub_addr
            self.pub_sub_sync_addr = port_args.pub_sub_sync_addr

        self.dp_size = server_args.dp_size
        self.tp_size = server_args.tp_size
        self.schedule_policy = server_args.schedule_policy
        self.dp_schedule_policy = server_args.dp_schedule_policy
        self.skip_tokenizer_init = server_args.skip_tokenizer_init
        self.stream_interval = server_args.stream_interval
        self.max_seq_len = server_args.max_seq_len
        self.page_size = server_args.page_size
        self.enable_overlap = not server_args.disable_overlap_schedule
        if server_args.multimodal:
            logger.info("Multimodal mode enabled, disabling overlap schedule")
            self.enable_overlap = False
        if server_args.disaggregation_mode != "null":
            logger.info("PD disaggregation mode enabled, disabling overlap schedule")
            self.enable_overlap = False
        self.spec_algorithm = SpeculativeAlgorithm.from_string(server_args.speculative_algorithm)

        # PD disaggregation runtime attributes. They are populated by
        # install_disaggregation_wiring() when disaggregation_mode != "null".
        self.disagg_kv_manager = None
        self.disagg_bootstrap_client = None
        self.disagg_bootstrap_server = None
        self.disagg_heartbeat = None
        self.disagg_bootstrap_key = None
        self.disagg_shutdown = None
        self.disagg_use_d2h_staging = False
        self.disagg_prefill_queue = None
        self.disagg_prealloc_queue = None
        self.disagg_transfer_queue = None
        self.disagg_decode_watchdog = None
        # Decode-side cache of prefill registry (sglang-style local per-room
        # resolution) + reqs deferred because no prefill was registered yet.
        self.disagg_prefill_info_cache = None
        self._pd_pending_bootstrap = []

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
            if not jax.distributed.is_initialized():
                jax.distributed.initialize(server_args.dist_init_addr, self.nnodes, self.node_rank)
            else:
                logger.info("JAX distributed already initialized, skipping re-initialization")

        platform = os.getenv("JAX_PLATFORMS", None)
        if platform == "proxy":
            pathwaysutils.initialize()
            from sgl_jax.srt.kernels._pathways_compat import install

            install()

        self.pd = server_args.pd_disaggregation
        if mesh is not None:
            self.mesh = mesh
        elif self.pd == "pathways":
            from sgl_jax.srt.disaggregation.pathways_pd import make_slice_meshes

            server_args.disable_radix_cache = True
            if not server_args.quantization_config_path:
                # Dynamic quant: cache holds BF16; after quant the fp8 model
                # plus BF16 cache exceeds device HBM (139G>103G on v7x).
                # setdefault so SGLANG_PD_WEIGHT_CACHE=0 can disable for >768G
                # checkpoints that overflow the c4-192 head-node host RAM.
                os.environ.setdefault("SGLANG_PD_WEIGHT_CACHE", "1")
            os.environ.setdefault("SGLANG_MOE_BULK_READ", "1")
            self._pd_n_prefill = max(1, server_args.pd_num_prefill)
            self.p_meshes, self.mesh = make_slice_meshes(
                self.dp_size, self.tp_size, self._pd_n_prefill
            )
            self.p_mesh = self.p_meshes[0]
        else:
            self.mesh = create_device_mesh(
                ici_parallelism=[self.dp_size, self.tp_size // self.dp_size],
                dcn_parallelism=[1, 1],
                device_indexes=server_args.device_indexes,
            )

        if server_args.moe_backend in ("fused", "fused_v2"):
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

        self.tp_worker_p = None
        if self.pd:
            import copy as _copy

            assert self.spec_algorithm is None or self.spec_algorithm.is_none()
            assert not getattr(server_args, "enable_mixed_chunk", False)
            d_mesh = self.mesh
            p_args = _copy.deepcopy(server_args)
            p_args.mem_fraction_static = server_args.pd_prefill_mem_fraction
            p_args.disable_radix_cache = True
            d_args = _copy.deepcopy(server_args)
            d_args.mem_fraction_static = server_args.pd_decode_mem_fraction
            d_args.disable_radix_cache = True
            from sgl_jax.srt.mem_cache.allocator import (
                SWATokenToKVPoolAllocator as _SWAAlloc,
            )
            from sgl_jax.srt.mem_cache.chunk_cache import ChunkCache as _CC
            from sgl_jax.srt.mem_cache.chunk_cache import SWAChunkCache as _SWACC

            n_p = getattr(self, "_pd_n_prefill", 1)
            p_meshes = getattr(self, "p_meshes", [self.p_mesh])
            os.environ["SGLANG_CI_SMALL_KV_SIZE"] = str(server_args.pd_prefill_max_tokens)
            self.tp_workers_p = []
            self.p_r2ts, self.p_allocs, self.p_trees = [], [], []
            for i, pm in enumerate(p_meshes):
                logger.info("[pathways_pd] loading P worker %d/%d on %s", i, n_p, pm.shape)
                # P worker stays sync (ModelWorker): prefill must finish writing
                # KV into the P pool before gather_to_dmesh reads it.
                w = ModelWorker(
                    server_args=p_args,
                    mesh=pm,
                    model_class=model_class,
                    precompile_params=precompile_params,
                )
                self.tp_workers_p.append(w)
                r2t, alloc = w.get_memory_pool()
                self.p_r2ts.append(r2t)
                self.p_allocs.append(alloc)
                if isinstance(alloc, _SWAAlloc):
                    self.p_trees.append(
                        _SWACC(
                            r2t,
                            alloc,
                            page_size=server_args.page_size,
                            sliding_window_size=w.sliding_window_size,
                        )
                    )
                else:
                    self.p_trees.append(_CC(r2t, alloc, page_size=server_args.page_size))
            self.tp_worker_p = self.tp_workers_p[0]
            self.p_r2t, self.p_alloc, self.p_tree = (
                self.p_r2ts[0],
                self.p_allocs[0],
                self.p_trees[0],
            )
            logger.info("[pathways_pd] loading D worker on %s", d_mesh.shape)
            os.environ["SGLANG_CI_SMALL_KV_SIZE"] = str(server_args.pd_decode_max_tokens)
            self.tp_worker = TpWorkerClass(
                server_args=d_args,
                mesh=d_mesh,
                model_class=model_class,
                precompile_params=precompile_params,
            )
            del os.environ["SGLANG_CI_SMALL_KV_SIZE"]
            d_kv = self.tp_worker.model_runner.token_to_kv_pool
            d_kv._donate_lock = threading.Lock()
            if self.pd == "pathways":
                from queue import Empty as _QEmpty
                from queue import Queue as _Queue

                from sgl_jax.srt.disaggregation.pathways_pd import PathwaysPDKVTransfer

                d_alloc = self.tp_worker.model_runner.token_to_kv_pool_allocator
                self.kv_transfers = [
                    PathwaysPDKVTransfer(
                        pm,
                        d_mesh,
                        w.model_runner.token_to_kv_pool,
                        d_kv,
                        p_alloc=self.p_allocs[i],
                        d_alloc=d_alloc,
                        page_size=server_args.page_size,
                    )
                    for i, (pm, w) in enumerate(zip(p_meshes, self.tp_workers_p))
                ]
                self.kv_transfer = self.kv_transfers[0]
                self._pd_prefill_qs = [_Queue(maxsize=2) for _ in range(n_p)]
                self._pd_prefill_q = self._pd_prefill_qs[0]
                self._pd_ready_q: _Queue = _Queue()
                self._pd_qempty = _QEmpty
                # reqs pushed to prefill_q but not yet drained into running_batch;
                # admission must reserve D r2t slots for them (single-item drain
                # lets ready_q backlog while running_batch undercounts).
                self._pd_inflight = 0
                self._pd_next_p = 0
                self._pd_empty_running_p = [
                    ScheduleBatch.init_new(
                        reqs=[[] for _ in range(self.dp_size)],
                        req_to_token_pool=self.p_r2ts[i],
                        token_to_kv_pool_allocator=self.p_allocs[i],
                        tree_cache=self.p_trees[i],
                        model_config=self.model_config,
                        enable_overlap=self.enable_overlap,
                        dp_size=self.dp_size,
                        spec_algorithm=self.spec_algorithm,
                        mesh=p_meshes[i],
                    )
                    for i in range(n_p)
                ]
                self._pd_prefill_threads = [
                    threading.Thread(
                        target=self._pd_prefill_loop,
                        args=(i,),
                        name=f"pd-prefill-{i}",
                        daemon=True,
                    )
                    for i in range(n_p)
                ]
                for t in self._pd_prefill_threads:
                    t.start()
            self._pd_pending_migrate: ScheduleBatch | None = None
            logger.info(
                "[pathways_pd] n_prefill=%d P pool=%d tok, D pool=%d tok",
                n_p,
                self.tp_worker_p.max_total_num_tokens,
                self.tp_worker.max_total_num_tokens,
            )
        else:
            self.tp_worker = TpWorkerClass(
                server_args=server_args,
                mesh=self.mesh,
                model_class=model_class,
                precompile_params=precompile_params,
            )

        # launch draft worker
        self._spec_multi_layer = False
        if self.spec_algorithm is not None and self.spec_algorithm.is_eagle():
            # Multi-layer vs single-layer is a model property (how many MTP heads
            # the target ships), not a CLI-algorithm property. NEXTN with a single
            # MTP head behaves exactly like EAGLE (same head run N times).
            # DeepSeek-style configs expose num_nextn_predict_layers; MiMo-style
            # configs don't, so fall back to --speculative-num-steps under NEXTN
            # (one MTP weight set per step).
            n_mtp = getattr(self.tp_worker.model_config.hf_config, "num_nextn_predict_layers", None)
            if n_mtp is None and self.spec_algorithm.is_nextn():
                n_mtp = server_args.speculative_num_steps
            self._spec_multi_layer = n_mtp is not None and n_mtp > 1
            if self._spec_multi_layer:
                from sgl_jax.srt.speculative.multi_layer_eagle_worker import (
                    MultiLayerEAGLEWorker as _SpecWorkerCls,
                )
            else:
                from sgl_jax.srt.speculative.eagle_worker import (
                    EAGLEWorker as _SpecWorkerCls,
                )

            self.draft_worker = _SpecWorkerCls(
                server_args=server_args,
                target_worker=self.tp_worker,
            )
            if self.enable_overlap and hasattr(self.draft_worker, "init_spec_relay_buffers"):
                self.draft_worker.init_spec_relay_buffers()

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

        # Adjust max_running_requests to be divisible by dp_size
        if self.max_running_requests % self.dp_size != 0:
            self.max_running_requests = (self.max_running_requests // self.dp_size) * self.dp_size
        self.per_dp_max_running_requests = self.max_running_requests // self.dp_size

        self.is_hybrid = self.tp_worker.is_hybrid
        self.sliding_window_size = None
        if self.is_hybrid:
            self.sliding_window_size = self.tp_worker.sliding_window_size
            self.full_tokens_per_layer, self.swa_tokens_per_layer = (
                self.tp_worker.get_tokens_per_layer_info()
            )

        # Init memory pool and cache
        self.init_memory_pool_and_cache()

        # Init running status
        self.waiting_queue: list[Req] = []
        # Pending incoming generate requests waiting for dp assignment
        self.pending_dp_reqs: list[TokenizedGenerateReqInput] = []
        # The aborted requests
        self.aborted_reqs: dict[str, Req] = {}
        # The running decoding batch for continuous batching
        self.running_batch: ScheduleBatch = ScheduleBatch.init_new(
            reqs=[[] for _ in range(self.dp_size)],  # Empty list for each DP rank
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=self.enable_overlap,
            dp_size=self.dp_size,
            spec_algorithm=self.spec_algorithm,
            mesh=self.mesh,
        )
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
        self.chunked_reqs = [None] * self.dp_size  # Per-DP chunked requests
        self.is_mixed_chunk = (
            self.chunked_prefill_size is not None and server_args.enable_mixed_chunk
        )

        # Init pause/continue state
        self._engine_paused = False

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

        # Initialize DP scheduling state
        self.dp_round_robin_counter = 0

        # Init request dispatcher
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.handle_generate_request),
                (AbortReq, self.abort_request),
                (ProfileReq, self.profile),
                (FlushCacheReqInput, self.flush_cache_wrapped),
                (GetInternalStateReq, self.get_internal_state),
                (SetInternalStateReq, self.set_internal_state),
                (PauseGenerationReqInput, self.pause_generation),
                (ContinueGenerationReqInput, self.continue_generation),
            ]
        )

        if not server_args.disable_precompile and not self.pd:
            if self.spec_algorithm is None or self.spec_algorithm.is_none():
                logger.info("[Scheduler] Begins to run worker precompile.")
                self.tp_worker.run_precompile()
                logger.info("[Scheduler] Completes worker precompile.")
            else:
                logger.info("[Scheduler] Begins to run spec_decode worker precompile.")
                self.draft_worker.run_spec_decode_precompile()
                logger.info("[Scheduler] Completes spec_decode worker precompile.")

    def _setup_jit_cache(self, server_args: ServerArgs) -> None:
        jit_cache_dir = os.getenv("JAX_COMPILATION_CACHE_DIR", None)
        device_indexes = server_args.device_indexes
        cache_status = None
        # libtpu (tpu-v6e + libtpu 0.0.30) crashes during JAX persistent
        # compilation-cache use when the device subset does not start at
        # device 0 (e.g. device_indexes=[2, 3]). Disable the cache for
        # such schedulers and override any cache config a sibling
        # scheduler may have set in the same process. See
        # sgl-project/sglang-jax#1216.
        if (
            jit_cache_dir is not None
            and device_indexes is not None
            and min(device_indexes, default=0) > 0
        ):
            jax.config.update("jax_compilation_cache_dir", "")
            jit_cache_dir = None
            # jax.config.update alone does not take effect once the
            # compilation_cache module's _cache_initialized flag is set
            # by an earlier sibling scheduler in the same process; that
            # flag is one-shot. cc.reset_cache() is the only public API
            # that clears it, forcing the next cache lookup to re-read
            # the (now-empty) cache_dir config and stay disabled.
            from jax.experimental.compilation_cache import compilation_cache as cc

            cc.reset_cache()
            cache_status = (
                f"disabled for non-zero-base device subset: device_indexes={device_indexes}"
            )
        if jit_cache_dir is not None:
            jax.config.update("jax_compilation_cache_dir", jit_cache_dir)
            # Default the compile-time write threshold to 0 (cache every compile) for
            # local/dev. When JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS is set (CI sets
            # it to 1 to skip tiny entries and cut small-file GCS writes), defer to JAX's
            # own parsing so the behavior — including validation of bad values — matches
            # upstream JAX.
            if "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS" not in os.environ:
                jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
            # Disable the size gate; the compile-time threshold still controls writes.
            jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
            # Include XLA sub-caches such as kernel/autotune data.
            jax.config.update("jax_persistent_cache_enable_xla_caches", "all")
            from jax.experimental.compilation_cache import compilation_cache as cc

            cc.set_cache_dir(jit_cache_dir)
            min_compile_time = jax.config.jax_persistent_cache_min_compile_time_secs
            cache_status = f"enabled, dir={jit_cache_dir}, min_compile_time={min_compile_time}s"

        if cache_status is None:
            cache_status = "not configured (JAX_COMPILATION_CACHE_DIR unset)"
        logger.info("XLA persistent compilation cache: %s", cache_status)

    def _is_spec_decode_enabled(self) -> bool:
        return self.spec_algorithm is not None and not self.spec_algorithm.is_none()

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
                tokenizer_backend=server_args.tokenizer_backend,
                sub_dir=tokenizer_subdir,
            )

    def init_memory_pool_and_cache(self):
        self.req_to_token_pool, self.token_to_kv_pool_allocator = self.tp_worker.get_memory_pool()
        self.tree_cache = build_kv_cache(
            server_args=self.server_args,
            model_config=self.model_config,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            page_size=self.page_size,
            is_hybrid=self.is_hybrid,
            sliding_window_size=self.sliding_window_size,
            tp_size=self.tp_size,
            spec_algorithm=self.spec_algorithm,
        )

    def _select_round_robin_dp(self) -> int:
        dp_rank = self.dp_round_robin_counter % self.dp_size
        self.dp_round_robin_counter += 1
        return dp_rank

    @staticmethod
    def _get_input_token_len(req: Req | TokenizedGenerateReqInput) -> int:
        if isinstance(req, Req):
            return len(req.origin_input_ids)

        if not isinstance(req, TokenizedGenerateReqInput):
            return 0

        input_ids = req.input_ids
        if input_ids is None:
            return 0
        if isinstance(input_ids, list):
            if len(input_ids) == 0:
                return 0
            if isinstance(input_ids[0], int):
                return len(input_ids)
            if isinstance(input_ids[0], list):
                return sum(len(ids) for ids in input_ids if isinstance(ids, list))
        return 0

    @staticmethod
    def _extract_max_new_tokens(sampling_params: object) -> int:
        """Extract max_new_tokens from sampling params with a conservative fallback."""
        default_max_new_tokens = 128
        value = None

        if sampling_params is None:
            return default_max_new_tokens

        if isinstance(sampling_params, dict):
            value = sampling_params.get("max_new_tokens", default_max_new_tokens)
        elif isinstance(sampling_params, list):
            if len(sampling_params) > 0:
                first = sampling_params[0]
                if isinstance(first, dict):
                    value = first.get("max_new_tokens", default_max_new_tokens)
                else:
                    value = getattr(first, "max_new_tokens", default_max_new_tokens)
            else:
                value = default_max_new_tokens
        else:
            value = getattr(sampling_params, "max_new_tokens", default_max_new_tokens)

        if value is None:
            return CLIP_MAX_NEW_TOKENS_ESTIMATION
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return default_max_new_tokens

    @staticmethod
    def _extract_ignore_eos(sampling_params: object) -> bool:
        if sampling_params is None:
            return False
        if isinstance(sampling_params, dict):
            return bool(sampling_params.get("ignore_eos", False))
        return bool(getattr(sampling_params, "ignore_eos", False))

    def _estimate_req_tokens(self, req: Req | TokenizedGenerateReqInput) -> int:
        """Estimate per-request token load as input + expected output."""
        input_token_len = self._get_input_token_len(req)
        sampling_params = getattr(req, "sampling_params", None)
        est_max_new_tokens = self._extract_max_new_tokens(sampling_params)
        est_max_new_tokens = min(est_max_new_tokens, CLIP_MAX_NEW_TOKENS_ESTIMATION)
        ignore_eos = self._extract_ignore_eos(sampling_params)

        # Align with handle_generate_request() clipping rule:
        # max_new_tokens <= max_req_len - input_len - 1
        max_by_req_len = max(0, self.max_req_len - input_token_len - 1)
        est_max_new_tokens = min(est_max_new_tokens, max_by_req_len)

        # Align with ignore_eos token estimation in PrefillAdder:
        # ignore_eos requests use ratio=1.0 and page-aligned token budgeting.
        new_token_ratio = 1.0 if ignore_eos else self.new_token_ratio
        est_output_tokens = int(est_max_new_tokens * new_token_ratio)
        if ignore_eos:
            est_output_tokens = (
                (est_output_tokens + self.page_size - 1) // self.page_size
            ) * self.page_size
            est_output_tokens += IGNORE_EOS_RESERVE_TOKENS

        if isinstance(req, Req):
            # For running requests, scheduler load should reflect total reserved footprint.
            return input_token_len + est_output_tokens

        return input_token_len + est_output_tokens

    def _get_dp_load_snapshot(self) -> tuple[list[int], list[int]]:
        """Return per-DP (request_count, token_count) for in-flight scheduled work."""
        req_counts = [0] * self.dp_size
        token_counts = [0] * self.dp_size

        for dp_rank, info in enumerate(self.running_batch.reqs_info):
            if not info.reqs:
                continue
            req_counts[dp_rank] += len(info.reqs)
            token_counts[dp_rank] += sum(self._estimate_req_tokens(req) for req in info.reqs)

        # In overlap mode, last_batch can still be in-flight (e.g., prefill/extend) but not
        # yet merged into running_batch. Include it to avoid underestimating DP load.
        if self.last_batch and self.last_batch.forward_mode.is_extend():
            for dp_rank, info in enumerate(self.last_batch.reqs_info):
                if not info.reqs and info.chunked_req is None:
                    continue

                running_ids = set()
                running_info = self.running_batch.reqs_info[dp_rank]
                if running_info.reqs:
                    running_ids = {req.rid for req in running_info.reqs}

                for req in info.reqs or []:
                    if req.rid in running_ids:
                        continue
                    req_counts[dp_rank] += 1
                    token_counts[dp_rank] += self._estimate_req_tokens(req)

                if info.chunked_req is not None and info.chunked_req.rid not in running_ids:
                    req_counts[dp_rank] += 1
                    token_counts[dp_rank] += self._estimate_req_tokens(info.chunked_req)

        return req_counts, token_counts

    def _dp_load_and_eligible(
        self, extra_counts: list[int], extra_token_counts: list[int]
    ) -> tuple[list[int], list[int], list[int]]:
        """Per-DP (running + pending) load and the ranks that can accept a request.

        A rank is eligible when its batch is not full and it is under the
        per-rank running cap. Returns ``(eligible_ranks, counts, token_counts)``.
        """
        running_counts, running_token_counts = self._get_dp_load_snapshot()
        counts = [running_counts[i] + extra_counts[i] for i in range(self.dp_size)]
        token_counts = [
            running_token_counts[i] + extra_token_counts[i] for i in range(self.dp_size)
        ]
        eligible = [
            dp_rank
            for dp_rank in range(self.dp_size)
            if not self.running_batch.reqs_info[dp_rank].batch_is_full
            and counts[dp_rank] < self.per_dp_max_running_requests
        ]
        return eligible, counts, token_counts

    def _select_min_running_dp(
        self,
        extra_counts: list[int] | None = None,
        extra_token_counts: list[int] | None = None,
    ) -> int | None:
        """Select a DP rank with the minimum (running requests, scheduled tokens) load.

        Returns None if all DP ranks are full.
        """
        if self.dp_size == 1:
            return 0

        if extra_counts is None:
            extra_counts = [0] * self.dp_size
        if extra_token_counts is None:
            extra_token_counts = [0] * self.dp_size

        eligible, counts, token_counts = self._dp_load_and_eligible(
            extra_counts, extra_token_counts
        )
        if not eligible:
            return None

        return min(eligible, key=lambda dp_rank: (counts[dp_rank], token_counts[dp_rank], dp_rank))

    def _cached_prefix_len(self, token_ids: list[int], extra_key: str | None, dp_rank: int) -> int:
        """Length of the longest cached prefix for ``token_ids`` on ``dp_rank``.

        Probes the dp-keyed tree (no alloc, no CoW), but incurs the normal
        ``match_prefix`` side effects (LRU refresh, possible node split). Returns
        0 for non-radix caches (ChunkCache returns an empty match).
        """
        if self.tree_cache is None:
            return 0
        result = self.tree_cache.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids, extra_key, dp_rank))
        )
        return len(result.device_indices)

    def _select_cache_aware_dp(
        self,
        req: TokenizedGenerateReqInput,
        extra_counts: list[int],
        extra_token_counts: list[int],
    ) -> int | None:
        """Route ``req`` by cache affinity with soft load balancing.

        Probes each eligible rank's cached prefix length, then defers to
        ``pick_cache_aware_dp``: balance on large load skew, else least-loaded
        among the ranks holding a substantial cached prefix. Returns None if all
        DP ranks are full.
        """
        if self.dp_size == 1:
            return 0

        eligible, counts, token_counts = self._dp_load_and_eligible(
            extra_counts, extra_token_counts
        )
        if not eligible:
            return None

        token_ids, extra_key = req_prefix_match_key(req)
        matches: dict[int, int] = {}
        prompt_len = len(token_ids) if token_ids else 0
        if token_ids:
            for dp_rank in eligible:
                matches[dp_rank] = self._cached_prefix_len(token_ids, extra_key, dp_rank)

        return pick_cache_aware_dp(eligible, counts, token_counts, matches, prompt_len)

    def select_dp_for_request(self, recv_reqs: list[Req]) -> list[Req]:
        """Assign dp_rank to incoming requests using the configured DP policy.

        Requests without a dp assignment (min-running + all full) are queued and
        retried in the next loop to keep ordering deterministic across nodes.
        """
        if recv_reqs is None:
            recv_reqs = []

        # Preserve FIFO order: older pending requests first, then new arrivals.
        combined_reqs = []
        if self.pending_dp_reqs:
            combined_reqs.extend(self.pending_dp_reqs)
            self.pending_dp_reqs = []
        if recv_reqs:
            combined_reqs.extend(recv_reqs)

        if self.dp_size == 1:
            for req in combined_reqs:
                # Only assign dp_rank to TokenizedGenerateReqInput
                if isinstance(req, TokenizedGenerateReqInput):
                    req.dp_rank = 0
            return combined_reqs

        pending_counts = [0] * self.dp_size
        pending_token_counts = [0] * self.dp_size
        ready_reqs: list[Req] = []

        for req in combined_reqs:
            # Only assign dp_rank to TokenizedGenerateReqInput
            if not isinstance(req, TokenizedGenerateReqInput):
                ready_reqs.append(req)
                continue

            # Skip if dp_rank already set (e.g., sticky sessions)
            if req.dp_rank is not None:
                if 0 <= req.dp_rank < self.dp_size:
                    pending_counts[req.dp_rank] += 1
                    pending_token_counts[req.dp_rank] += self._estimate_req_tokens(req)
                ready_reqs.append(req)
                continue

            if self.dp_schedule_policy == "round_robin":
                req.dp_rank = self._select_round_robin_dp()
                ready_reqs.append(req)
                continue

            if self.dp_schedule_policy == "cache_aware":
                dp_rank = self._select_cache_aware_dp(
                    req,
                    extra_counts=pending_counts,
                    extra_token_counts=pending_token_counts,
                )
            else:
                dp_rank = self._select_min_running_dp(
                    extra_counts=pending_counts,
                    extra_token_counts=pending_token_counts,
                )
            if dp_rank is None:
                # All DP ranks are full; keep the request pending.
                self.pending_dp_reqs.append(req)
                continue

            req.dp_rank = dp_rank
            pending_counts[dp_rank] += 1
            pending_token_counts[dp_rank] += self._estimate_req_tokens(req)
            ready_reqs.append(req)

        return ready_reqs

    def event_loop_normal(self):
        """A normal scheduler loop."""
        while True:
            recv_reqs = (
                self._comm_backend.recv_requests()
                if self._comm_backend is not None
                else self.recv_requests()
            )
            # Assign DP rank to incoming requests
            recv_reqs = self.select_dp_for_request(recv_reqs)
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
        _pd_iter_trace = self.pd == "pathways"

        import gc as _gc

        _gc.collect()
        _gc.freeze()
        _gc.set_threshold(700, 10, 10000)
        logger.info(
            "[pd-gc] gc.freeze() frozen=%d thresholds=%s",
            _gc.get_freeze_count(),
            _gc.get_threshold(),
        )

        while True:
            _it0 = time.perf_counter() if _pd_iter_trace else 0.0
            recv_reqs = (
                self._comm_backend.recv_requests()
                if self._comm_backend is not None
                else self.recv_requests()
            )
            # Assign DP rank to incoming requests
            recv_reqs = self.select_dp_for_request(recv_reqs)
            self.process_input_requests(recv_reqs)
            _it1 = time.perf_counter() if _pd_iter_trace else 0.0

            # Skip batch processing when engine is paused
            if self._engine_paused:
                continue

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch
            _it2 = time.perf_counter() if _pd_iter_trace else 0.0

            if batch:
                batch.launch_done = threading.Event()
                with jax.profiler.TraceAnnotation("run_batch"):
                    result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), result))

                if self.last_batch is None:
                    # Create a dummy first batch to start the pipeline for overlap schedule.
                    # It is now used for triggering the sampling_info_done event.
                    tmp_batch = ScheduleBatch.init_new(
                        reqs=[[] for _ in range(self.dp_size)],
                        req_to_token_pool=self.req_to_token_pool,
                        token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                        tree_cache=self.tree_cache,
                        model_config=self.model_config,
                        enable_overlap=self.enable_overlap,
                        dp_size=self.dp_size,
                        spec_algorithm=self.spec_algorithm,
                        mesh=self.mesh,
                    )
                    tmp_batch.forward_mode = ForwardMode.DUMMY_FIRST
                    tmp_batch.next_batch_sampling_info = (
                        self._current_sampling_info_owner().cur_sampling_info
                    )
                    with jax.profiler.TraceAnnotation("process_batch_result"):
                        self.process_batch_result(tmp_batch, None, batch.launch_done)

            if self.last_batch:
                # Process the results of the last batch
                tmp_batch, tmp_result = self.result_queue.popleft()
                tmp_batch.next_batch_sampling_info = (
                    self._current_sampling_info_owner().cur_sampling_info if batch else None
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
            if _pd_iter_trace:
                _it3 = time.perf_counter()
                if _it3 - _it0 > 0.5:
                    logger.info(
                        "[pd-iter] total=%.0fms recv=%.0f get_batch=%.0f run+proc=%.0f "
                        "running=%d inflight=%d",
                        (_it3 - _it0) * 1e3,
                        (_it1 - _it0) * 1e3,
                        (_it2 - _it1) * 1e3,
                        (_it3 - _it2) * 1e3,
                        sum(len(i.reqs) for i in self.running_batch.reqs_info),
                        getattr(self, "_pd_inflight", 0),
                    )

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

    def process_input_requests(self, recv_reqs: list):
        for recv_req in recv_reqs:
            output = self._request_dispatcher(recv_req)
            if output is not None:
                if self._comm_backend is not None:
                    self._comm_backend.send_pyobj(output)
                else:
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
            return_output_logprob_only=recv_req.return_output_logprob_only,
            top_logprobs_num=recv_req.top_logprobs_num,
            token_ids_logprob=recv_req.token_ids_logprob,
            stream=recv_req.stream,
            lora_id=recv_req.lora_id,
            extra_key=recv_req.extra_key,
            dp_rank=recv_req.dp_rank,
            eos_token_ids=self.model_config.hf_eos_token_id,
            vocab_size=self.model_config.vocab_size,
            return_routed_experts=recv_req.return_routed_experts,
            return_hidden_states=recv_req.return_hidden_states,
        )
        req.tokenizer = self.tokenizer
        # PD disaggregation routing keys.
        req.bootstrap_host = recv_req.bootstrap_host
        req.bootstrap_port = recv_req.bootstrap_port
        req.bootstrap_room = recv_req.bootstrap_room
        req.disagg_transfer_id = recv_req.disagg_transfer_id or req.rid
        if hasattr(recv_req, "mm_inputs") and recv_req.mm_inputs:
            req.mm_inputs = recv_req.mm_inputs
            multimodal_embedding = recv_req.mm_inputs.get("multimodal_embedding")
            req.multimodal_embedding = multimodal_embedding
            if (
                recv_req.mm_inputs.get("deepstack_visual_pos_mask") is not None
                and recv_req.mm_inputs.get("deepstack_visual_embedding") is not None
            ):
                req.apply_for_deepstack = True
                req.deepstack_visual_pos_mask = recv_req.mm_inputs.get("deepstack_visual_pos_mask")
                req.deepstack_visual_embedding = recv_req.mm_inputs.get(
                    "deepstack_visual_embedding"
                )
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
                    schema = req.sampling_params.json_schema
                    if isinstance(schema, dict):
                        schema = json.dumps(schema, sort_keys=True)
                    key = ("json", schema)
                elif req.sampling_params.regex is not None:
                    key = ("regex", req.sampling_params.regex)
                elif req.sampling_params.ebnf is not None:
                    key = ("ebnf", req.sampling_params.ebnf)
                elif req.sampling_params.structural_tag:
                    tag = req.sampling_params.structural_tag
                    if hasattr(tag, "model_dump_json"):
                        tag = tag.model_dump_json()
                    elif isinstance(tag, dict):
                        tag = json.dumps(tag, sort_keys=True)
                    key = ("structural_tag", tag)

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
            0 if self.running_batch.is_empty() else self.running_batch.batch_size()
        )
        ret["prefill_decode_size"] = ret["waiting_queue_size"] + ret["running_batch_size"]
        ret["waiting_queue_rids"] = [req.rid for req in self.waiting_queue]
        all_reqs = [req for info in self.running_batch.reqs_info for req in info.reqs if info.reqs]
        ret["running_batch_rids"] = [req.rid for req in all_reqs] if len(all_reqs) != 0 else []

        # scheduling state
        ret["cur_batch_is_none"] = self.cur_batch is None
        ret["last_batch_is_none"] = self.last_batch is None
        ret["chunked_req_is_none"] = all(r is None for r in self.chunked_reqs)

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

        # PD disaggregation queues
        ret["disagg_prefill_queue_size"] = len(self.disagg_prefill_queue or ())
        ret["disagg_prealloc_queue_size"] = len(self.disagg_prealloc_queue or ())
        ret["disagg_transfer_queue_size"] = len(self.disagg_transfer_queue or ())

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
        pending_dp_reqs = len(self.pending_dp_reqs)
        running_reqs = _batch_size(self.running_batch)
        current_batch_reqs = _batch_size(self.cur_batch)
        last_batch_reqs = _batch_size(self.last_batch)
        chunked_pending = any(req is not None for req in self.chunked_reqs)
        pending_results = len(getattr(self, "result_queue", ())) if self.enable_overlap else 0

        has_pending = (
            waiting_reqs > 0
            or pending_dp_reqs > 0
            or running_reqs > 0
            or current_batch_reqs > 0
            or last_batch_reqs > 0
            or chunked_pending
            or pending_results > 0
        )

        pd_prefill = len(self.disagg_prefill_queue or ())
        pd_prealloc = len(self.disagg_prealloc_queue or ())
        pd_transfer = len(self.disagg_transfer_queue or ())
        has_pending = has_pending or pd_prefill > 0 or pd_prealloc > 0 or pd_transfer > 0

        if has_pending:
            msg = (
                "Cache not flushed because there are pending requests. "
                f"waiting={waiting_reqs}, pending_dp={pending_dp_reqs}, running={running_reqs}, "
                f"cur_batch={current_batch_reqs}, last_batch={last_batch_reqs}, "
                f"chunked={chunked_pending}, pending_results={pending_results}, "
                f"pd_prefill={pd_prefill}, pd_prealloc={pd_prealloc}, pd_transfer={pd_transfer}"
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
        self.running_batch = ScheduleBatch.init_new(
            reqs=[[] for _ in range(self.dp_size)],
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=self.enable_overlap,
            dp_size=self.dp_size,
            spec_algorithm=self.spec_algorithm,
            mesh=self.mesh,
        )
        self.pending_dp_reqs = []
        self.chunked_reqs = [None] * self.dp_size
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
        self.waiting_queue.append(req)
        if req.bootstrap_room is not None:
            mark = getattr(self, "_pd_mark_time", None)
            if mark is not None:
                mark(req, "queue_entry")

    def _extend_requests_to_queue(self, reqs: list[Req], is_retracted: bool = False):
        self.waiting_queue.extend(reqs)

    def check_memory(self):
        if self.is_hybrid:
            # Per-rank invariant: available + evictable + protected == size_per_rank.
            # Checking per-rank avoids one rank's over-count masking another's leak.
            full_size_per_rank = self.token_to_kv_pool_allocator.full_attn_allocator.size_per_rank
            swa_size_per_rank = self.token_to_kv_pool_allocator.swa_attn_allocator.size_per_rank
            leak_msgs = []
            for dp in range(self.dp_size):
                full_avail = self.token_to_kv_pool_allocator.full_available_size(dp)
                full_evict = self.tree_cache.full_evictable_size(dp_rank=dp)
                full_protected = self.tree_cache.full_protected_size(dp_rank=dp)
                swa_avail = self.token_to_kv_pool_allocator.swa_available_size(dp)
                swa_evict = self.tree_cache.swa_evictable_size(dp_rank=dp)
                swa_protected = self.tree_cache.swa_protected_size(dp_rank=dp)
                if full_avail + full_evict + full_protected != full_size_per_rank:
                    leak_msgs.append(
                        f"[dp={dp}][full] expected={full_size_per_rank}, "
                        f"{full_avail=}, {full_evict=}, {full_protected=}"
                    )
                if swa_avail + swa_evict + swa_protected != swa_size_per_rank:
                    leak_msgs.append(
                        f"[dp={dp}][swa] expected={swa_size_per_rank}, "
                        f"{swa_avail=}, {swa_evict=}, {swa_protected=}"
                    )
            if leak_msgs:
                raise ValueError(
                    "token_to_kv_pool_allocator memory leak detected!\n" + "\n".join(leak_msgs)
                )
        else:
            size_per_rank = self.token_to_kv_pool_allocator.size_per_rank
            leak_msgs = []
            for dp in range(self.dp_size):
                avail = self.token_to_kv_pool_allocator.available_size(dp)
                evict = self.tree_cache.evictable_size(dp_rank=dp)
                protected = self.tree_cache.protected_size(dp_rank=dp)
                if avail + evict + protected != size_per_rank:
                    leak_msgs.append(
                        f"[dp={dp}] expected={size_per_rank}, " f"{avail=}, {evict=}, {protected=}"
                    )
            if leak_msgs:
                raise ValueError(
                    "token_to_kv_pool_allocator memory leak detected!\n" + "\n".join(leak_msgs)
                )

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
        available_size = sum(
            [self.token_to_kv_pool_allocator.available_size(dp) for dp in range(self.dp_size)]
        )
        # Sum evictable size across all DP ranks
        evictable_size = sum(
            [self.tree_cache.evictable_size(dp_rank=dp) for dp in range(self.dp_size)]
        )
        num_used = self.max_total_num_tokens - (available_size + evictable_size)
        token_usage = num_used / self.max_total_num_tokens
        return num_used, token_usage, available_size, evictable_size

    def _get_swa_token_info(self):
        full_available_size = sum(
            [self.token_to_kv_pool_allocator.full_available_size(dp) for dp in range(self.dp_size)]
        )
        full_evictable_size = sum(
            [self.tree_cache.full_evictable_size(dp_rank=dp) for dp in range(self.dp_size)]
        )
        swa_available_size = sum(
            [self.token_to_kv_pool_allocator.swa_available_size(dp) for dp in range(self.dp_size)]
        )
        swa_evictable_size = sum(
            [self.tree_cache.swa_evictable_size(dp_rank=dp) for dp in range(self.dp_size)]
        )
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
        if self.pd == "pathways":
            return self._pd_get_next_batch_async()
        # PD: 先尝试迁移之前 D 满时暂存的 batch（D running req 完成后 free 出空间）
        if self.pd and self._pd_pending_migrate is not None:  # noqa: SIM102
            if self._pd_migrate(self._pd_pending_migrate):
                if self.running_batch.is_empty():
                    self.running_batch = self._pd_pending_migrate
                else:
                    self.running_batch.merge_batch(self._pd_pending_migrate)
                self._pd_pending_migrate = None

        # Process chunked requests for each DP rank
        chunked_req_to_exclude = {}
        _chunk_tree = self.p_tree if self.pd else self.tree_cache
        for dp_rank in range(self.dp_size):
            if self.chunked_reqs[dp_rank] is not None:
                # Move the chunked request out of the batch so that we can merge
                # only finished requests to running_batch.
                chunked_req_to_exclude[dp_rank] = self.chunked_reqs[dp_rank]
                _chunk_tree.cache_unfinished_req(self.chunked_reqs[dp_rank])

        # Merge the prefill batch into the running batch
        if self.last_batch and self.last_batch.forward_mode.is_extend():
            # Consistency check: each last_batch.reqs_info[dp_rank].chunked_req should match
            # what's in chunked_req_to_exclude (since self.chunked_reqs should contain the same requests)
            for dp_rank in range(self.dp_size):
                info = self.last_batch.reqs_info[dp_rank]
                if info.chunked_req is not None:
                    # Verify consistency: info.chunked_req should match self.chunked_reqs[dp_rank]
                    if dp_rank in chunked_req_to_exclude:
                        assert (
                            chunked_req_to_exclude[dp_rank] is info.chunked_req
                        ), f"Chunked request mismatch for DP rank {dp_rank}"
                    else:
                        # This shouldn't happen, but handle it gracefully
                        chunked_req_to_exclude[dp_rank] = info.chunked_req

            # Filter batch
            # Track per-DP batch sizes before filtering
            last_bs_per_dp = [
                len(info.reqs) if info.reqs else 0 for info in self.last_batch.reqs_info
            ]

            self.last_batch.filter_batch(chunked_req_to_exclude=chunked_req_to_exclude)

            # Update batch_is_full per DP rank
            for dp_rank in range(self.dp_size):
                info = self.last_batch.reqs_info[dp_rank]
                current_bs = len(info.reqs) if info.reqs else 0
                if current_bs < last_bs_per_dp[dp_rank]:
                    # Batch size decreased for this DP rank, mark as not full
                    info.batch_is_full = False
                    self.running_batch.reqs_info[dp_rank].batch_is_full = False

            # Merge the new batch into the running batch
            if not self.last_batch.is_empty() and not self.last_batch.is_prefill_only:
                if self.pd and not self._pd_migrate(self.last_batch):
                    # D 满，暂存等 D running req 完成 free 后再 migrate
                    assert self._pd_pending_migrate is None
                    self._pd_pending_migrate = self.last_batch
                elif self.running_batch.is_empty():
                    self.running_batch = self.last_batch
                elif (
                    not self._is_spec_decode_enabled()
                    or self.enable_overlap
                    or use_legacy_eagle3_non_overlap(self.enable_overlap, self.spec_algorithm)
                ):
                    # Spec overlap keeps prefill and decode as separate forwards, but
                    # once prefill has produced req-granular relay state it can join
                    # the next decode batch through the normal batch merge.
                    self.running_batch.merge_batch(self.last_batch)

        # For prefill-only batch, filter out finished requests since they
        # won't go through the decode step.
        if self.running_batch.is_prefill_only:
            self.running_batch.filter_batch()
            if self.running_batch.is_empty():
                for info in self.running_batch.reqs_info:
                    info.batch_is_full = False

        # decode-first interleave: when enabled, force 1:N decode:prefill ratio
        # to bound Max ITL by ~prefill_chunk_time instead of letting a burst of
        # waiting prefills starve running decodes (observed 110s spike at c64).
        df = getattr(self, "_decode_first_n", None)
        if df is None:
            df = int(os.environ.get("SGL_DECODE_FIRST_INTERLEAVE", "0"))
            self._decode_first_n = df
            self._consec_decode = 0
        skip_prefill = (
            df > 0
            and not self.running_batch.is_empty()
            and not self.running_batch.is_prefill_only
            and self._consec_decode < df
        )
        if skip_prefill or (self.pd and self._pd_pending_migrate is not None):
            new_batch = None
        elif self.pd:
            with self._pd_swap_p_pool():
                new_batch = self.get_new_batch_prefill()
        else:
            new_batch = self.get_new_batch_prefill()

        if new_batch:
            # Run prefill first if possible
            self._consec_decode = 0
            ret = new_batch
        else:
            # Run decode (skip for prefill-only batches)
            if not self.running_batch.is_empty() and not self.running_batch.is_prefill_only:
                self.running_batch = self.update_running_batch(self.running_batch)
                ret = self.running_batch if not self.running_batch.is_empty() else None
                if ret is not None:
                    self._consec_decode += 1
            else:
                ret = None
                self._consec_decode = 0

        return ret

    def get_new_batch_prefill(self) -> ScheduleBatch | None:
        if self.grammar_queue:
            self.move_ready_grammar_requests()

        # Handle the cases where prefill is not allowed
        has_chunked_reqs = any(req is not None for req in self.chunked_reqs)
        if self.is_hybrid:
            for info in self.running_batch.reqs_info:
                info.batch_is_full = False

        if (
            self._is_spec_decode_enabled()
            and not self.enable_overlap
            and not use_legacy_eagle3_non_overlap(self.enable_overlap, self.spec_algorithm)
            and not self.running_batch.is_empty()
        ):
            return None
        if (
            self.running_batch.batch_is_full or len(self.waiting_queue) == 0
        ) and not has_chunked_reqs:
            return None

        running_bs = self.running_batch.batch_size()
        running_bs_per_dp = [
            len(info.reqs) if info.reqs else 0 for info in self.running_batch.reqs_info
        ]

        if TEST_RETRACT and running_bs > TEST_RETRACT_NO_PREFILL_BS:
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
            running_bs_per_dp if self.is_mixed_chunk else 0,
            dp_size=self.dp_size,
        )

        # Process existing chunked requests for each DP rank
        for dp_rank in range(self.dp_size):
            if self.chunked_reqs[dp_rank] is not None:
                self.chunked_reqs[dp_rank].init_next_round_input()
                self.chunked_reqs[dp_rank] = adder.add_chunked_req(self.chunked_reqs[dp_rank])

        # Collect existing LoRA IDs in the running batch if LoRA is enabled
        if self.lora_paths is not None:
            lora_set = set()
            if self.running_batch is not None:
                for info in self.running_batch.reqs_info:
                    if info.reqs:
                        lora_set.update([req.lora_id for req in info.reqs])

        # Get requests from the waiting queue to a new prefill batch
        for req in self.waiting_queue:
            # Get DP rank for this request
            dp_rank = req.dp_rank
            assert (
                dp_rank is not None
            ), "dp_rank is None in waiting_queue; dp should be assigned before enqueue."

            # Check whether dp is full load
            if self.running_batch.reqs_info[dp_rank].batch_is_full or (
                len(self.running_batch.reqs_info[dp_rank].reqs) + len(adder.can_run_list[dp_rank])
                >= self.per_dp_max_running_requests
            ):
                continue

            # Skip DP ranks with an ongoing chunked request to avoid
            # creating a second chunked req on the same rank.
            if self.chunked_reqs[dp_rank] is not None:
                continue

            # Check LoRA constraint: ensure we don't exceed max_loras_per_batch
            # This is GLOBAL - must be same across all DP ranks
            if (
                self.lora_paths is not None
                and len(
                    lora_set
                    | set([req.lora_id for reqs in adder.can_run_list.values() for req in reqs])
                    | set([req.lora_id])
                )
                > self.max_loras_per_batch
            ):
                break

            mgr = getattr(self, "disagg_kv_manager", None)
            _host_pool = mgr.host_pool if mgr is not None else None
            _admit_ok, _reserved_bid = _reserve_host_slot_for_pd(
                _host_pool, getattr(self, "disagg_use_d2h_staging", False), req
            )
            if not _admit_ok:
                continue  # host pool full: leave req in waiting_queue, retry next round

            req.init_next_round_input(self.tree_cache)
            res = adder.add_one_req(req)

            if res != AddReqResult.CONTINUE:
                if _reserved_bid is not None and _host_pool is not None:
                    _host_pool.release(_reserved_bid)
                if res == AddReqResult.NO_TOKEN:
                    # Mark this specific DP rank as exhausted
                    self.running_batch.reqs_info[dp_rank].batch_is_full = True

                    # Check if all DP ranks are exhausted
                    if self.running_batch.batch_is_full:
                        break

                    # Continue to try requests from other DP ranks
                    continue
                else:
                    # OTHER: Global budget exhausted, stop entirely
                    break
            if _reserved_bid is not None:
                req.disagg_host_buffer_id = _reserved_bid

        # Update waiting queue
        # Flatten can_run_list for operations that need all requests
        all_can_run_reqs = [req for reqs in adder.can_run_list.values() for req in reqs]
        if len(all_can_run_reqs) == 0:
            return None

        can_run_set = set(all_can_run_reqs)
        self.waiting_queue = [x for x in self.waiting_queue if x not in can_run_set]

        # Update chunked requests for each DP rank
        for dp_rank in range(self.dp_size):
            if adder.new_chunked_reqs[dp_rank] is not None:
                assert (
                    self.chunked_reqs[dp_rank] is None
                ), f"Chunked request already exists for DP rank {dp_rank} when adding new chunked req"
                self.chunked_reqs[dp_rank] = adder.new_chunked_reqs[dp_rank]
            # Increment for any chunked req (new OR continuing) to keep
            # process_batch_result_prefill from sampling on intermediate chunks.
            if self.chunked_reqs[dp_rank] is not None:
                self.chunked_reqs[dp_rank].is_chunked += 1

        self.log_prefill_stats(adder, all_can_run_reqs, running_bs)

        # Use adder.can_run_list directly as reqs_per_dp (already grouped by DP rank)
        reqs_per_dp = [adder.can_run_list.get(i, []) for i in range(self.dp_size)]

        # Use self.chunked_reqs directly as chunked_reqs_per_dp
        chunked_reqs_per_dp = self.chunked_reqs.copy()

        # Create a new batch
        new_batch = ScheduleBatch.init_new(
            reqs_per_dp,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.dp_size,
            enable_custom_logit_processor=False,
            chunked_reqs=chunked_reqs_per_dp,
            mesh=self.mesh,
            spec_algorithm=self.spec_algorithm,
        )

        new_batch.prepare_for_extend()

        # Mixed-style chunked prefill
        if (
            self.is_mixed_chunk
            and not self._is_spec_decode_enabled()
            and not self.running_batch.is_empty()
            and not (new_batch.return_logprob or self.running_batch.return_logprob)
        ):
            self.running_batch.filter_batch()
            if not self.running_batch.is_empty():
                self.running_batch.prepare_for_decode()
                new_batch.mix_with_running(self.running_batch)
                for dp_rank in range(self.dp_size):
                    running_info = self.running_batch.reqs_info[dp_rank]
                    new_info = new_batch.reqs_info[dp_rank]
                    if running_info.reqs:
                        new_info.decoding_reqs = running_info.reqs

            self.running_batch = ScheduleBatch.init_new(
                reqs=[[] for _ in range(self.dp_size)],
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                tree_cache=self.tree_cache,
                model_config=self.model_config,
                enable_overlap=self.enable_overlap,
                dp_size=self.dp_size,
                spec_algorithm=self.spec_algorithm,
                mesh=self.mesh,
            )

        new_batch.bid = acc_global_bid()

        return new_batch

    def update_running_batch(self, batch: ScheduleBatch) -> ScheduleBatch | None:
        """Update the current running decoding batch."""
        initial_bs = batch.batch_size()

        batch.filter_batch()
        if batch.is_empty():
            # Mark all DP ranks as not full when batch is empty
            for info in batch.reqs_info:
                info.batch_is_full = False
            return batch

        # Check if decode out of memory
        if (kv_full_retract_flag := not batch.check_decode_mem()) or (
            TEST_RETRACT and self.forward_ct % TEST_RETRACT_INTERVAL == 0
        ):
            old_ratio = self.new_token_ratio

            retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode(self.server_args)
            num_retracted_reqs = len(retracted_reqs)
            self.new_token_ratio = new_token_ratio

            # Send abort responses so clients get an error instead of a hung connection
            for req in reqs_to_abort:
                abort_out = AbortReq(rid=req.rid)
                if self._comm_backend is not None:
                    self._comm_backend.send_pyobj(abort_out)
                else:
                    self.send_to_tokenizer.send_pyobj(abort_out)

            if kv_full_retract_flag:
                logger.warning(
                    "KV cache pool is full. Retract requests."
                    " #retracted_reqs: %d, #aborted_reqs: %d,"
                    " #new_token_ratio: %.4f -> %.4f",
                    num_retracted_reqs,
                    len(reqs_to_abort),
                    old_ratio,
                    self.new_token_ratio,
                )
            else:
                logger.info(
                    "Testing retraction." " #retracted_reqs: %d, #aborted_reqs: %d",
                    num_retracted_reqs,
                    len(reqs_to_abort),
                )

            self._extend_requests_to_queue(retracted_reqs, is_retracted=True)
        else:
            self.new_token_ratio = max(
                self.new_token_ratio - self.new_token_ratio_decay,
                self.min_new_token_ratio,
            )

        if batch.batch_size() < initial_bs:
            # Re-check per-DP batch_is_full status after filtering
            for dp_rank in range(self.dp_size):
                info = batch.reqs_info[dp_rank]
                current_bs = len(info.reqs) if info.reqs else 0
                if current_bs < self.per_dp_max_running_requests:
                    info.batch_is_full = False

        if batch.is_empty():
            return batch

        # Update batch arrays
        batch.prepare_for_decode()
        return batch

    def _extract_dp_output_ids(
        self,
        next_token_ids_flat: np.ndarray,
        model_worker_batch,
        batch: ScheduleBatch,
    ):
        """Extract output IDs from DP-formatted array and assign to reqs_info.

        Args:
            next_token_ids_flat: np.ndarray with format [dp0_tokens..., dp1_tokens..., ...]
                                 where each DP section has per_dp_bs_size tokens (including padding)
            model_worker_batch: ModelWorkerBatch with per_dp_bs_size and dp_size
            batch: ScheduleBatch to update reqs_info[*].output_ids
        """
        per_dp_bs_size = model_worker_batch.per_dp_bs_size

        for dp_rank in range(batch.dp_size):
            info = batch.reqs_info[dp_rank]
            num_real_reqs = len(info.reqs) if info.reqs else 0

            if num_real_reqs == 0:
                info.output_ids = np.array([], dtype=np.int32)
            else:
                info.output_ids = next_token_ids_flat[
                    dp_rank * per_dp_bs_size : dp_rank * per_dp_bs_size + num_real_reqs
                ]

    def run_batch(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Run a batch."""
        self.forward_ct += 1

        # Whether to run the profiler
        self._profile_batch_predicate(batch)

        # Run forward
        assert self.is_generation
        _worker = self.tp_worker_p if self.pd and batch.forward_mode.is_extend() else self.tp_worker
        (
            precompile_token_paddings,
            precompile_bs_paddings,
            precompile_cache_loc_paddings,
        ) = _worker.get_precompile_paddings()
        if self.spec_algorithm is None or self.spec_algorithm.is_none():
            model_worker_batch = batch.get_model_worker_batch(
                precompile_token_paddings,
                precompile_bs_paddings,
                precompile_cache_loc_paddings,
                self.page_size,
                self.server_args.enable_static_lora,
            )

            if self.enable_overlap and not (self.pd and batch.forward_mode.is_extend()):
                with jax.profiler.TraceAnnotation(
                    f"forward_batch_generation_overlap {self.forward_ct}"
                ):

                    logits_output, next_token_ids, cache_miss_count = (
                        self.tp_worker.forward_batch_generation(
                            model_worker_batch, sampling_metadata=None
                        )
                    )
                self._extract_dp_output_ids(next_token_ids, model_worker_batch, batch)
            else:
                logits_output, next_token_ids_device, cache_miss_count = (
                    _worker.forward_batch_generation(model_worker_batch, sampling_metadata=None)
                )
                if self.pd:
                    next_token_ids = self._pd_gather_output(
                        next_token_ids_device, batch.forward_mode.is_extend()
                    )
                elif self.dp_size > 1:
                    # In multi-host DP, next_token_ids may span non-addressable
                    # devices.  Replicate first so device_get can proceed.
                    from jax.experimental.multihost_utils import process_allgather

                    next_token_ids_device = process_allgather(next_token_ids_device, tiled=True)
                    next_token_ids = np.array(jax.device_get(next_token_ids_device))
                else:
                    next_token_ids = np.array(jax.device_get(next_token_ids_device))
                self._extract_dp_output_ids(next_token_ids, model_worker_batch, batch)
        else:
            (
                model_worker_batch,
                batch_output,
                next_token_ids,
                logits_output,
                cache_miss_count,
                defer_spec_output,
                defer_spec_prefill_output,
            ) = self._run_speculative_batch(
                batch,
                precompile_token_paddings,
                precompile_bs_paddings,
                precompile_cache_loc_paddings,
            )
        bid = model_worker_batch.bid

        # These 2 values are needed for processing the output, but the values can be
        # modified by overlap schedule. So we have to copy them here so that
        # we can use the correct values in output processing.
        if batch.return_logprob:
            # Collect extend_input_len from all DP ranks
            extend_input_len_per_req = []
            for info in batch.reqs_info:
                if info.reqs:
                    extend_input_len_per_req.extend([req.extend_input_len for req in info.reqs])
        else:
            extend_input_len_per_req = None
        if batch.return_logprob:
            # Collect extend_logprob_start_len from all DP ranks
            extend_logprob_start_len_per_req = []
            for info in batch.reqs_info:
                if info.reqs:
                    extend_logprob_start_len_per_req.extend(
                        [req.extend_logprob_start_len for req in info.reqs]
                    )
        else:
            extend_logprob_start_len_per_req = None
        spec_relay_buffers = None
        prefill_relay_future_indices = None
        if self.spec_algorithm is not None and not self.spec_algorithm.is_none():
            spec_relay_buffers = getattr(batch_output, "spec_relay_buffers", None)
            prefill_relay_future_indices = getattr(
                batch_output, "prefill_relay_future_indices", None
            )

        ret = GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=(
                batch_output.next_token_ids
                if (
                    self.spec_algorithm is not None
                    and self.spec_algorithm.is_eagle()
                    and (batch.forward_mode.is_decode() or defer_spec_prefill_output)
                    and self.enable_overlap
                )
                else next_token_ids.tolist()
            ),
            extend_input_len_per_req=extend_input_len_per_req,
            extend_logprob_start_len_per_req=extend_logprob_start_len_per_req,
            bid=bid,
            cache_miss_count=cache_miss_count,
            spec_relay_buffers=spec_relay_buffers,
            prefill_relay_future_indices=prefill_relay_future_indices,
        )
        if (
            self.spec_algorithm is not None
            and self.spec_algorithm.is_eagle()
            and batch_output.next_draft_input is not None
        ):
            assert isinstance(batch_output.next_draft_input, EagleDraftInput)
            ret.next_draft_input = batch_output.next_draft_input
            ret.accept_lens = batch_output.accept_lens
        return ret

    # ---- pathways_pd helpers ----
    def _pd_prefill_loop(self, p_idx: int = 0):
        """Pathways-PD async prefill thread: P-slice forward + cross-slice KV
        gather/device_put run here so the main loop never blocks on P mesh.
        scatter into D pool stays on the main thread (see _pd_drain_ready).
        Multi-P: one thread per P slice, each bound to its own
        worker/pool/transfer; main thread round-robins batches across queues."""
        from sgl_jax.srt.disaggregation.pathways_pd import slots_to_ordered_pages

        worker = self.tp_workers_p[p_idx]
        p_r2t = self.p_r2ts[p_idx]
        kv_transfer = self.kv_transfers[p_idx]
        q = self._pd_prefill_qs[p_idx]
        paddings = worker.get_precompile_paddings()
        while True:
            batch = q.get()
            if batch is None:
                return
            try:
                t0 = time.perf_counter()
                mwb = batch.get_model_worker_batch(
                    *paddings, self.page_size, self.server_args.enable_static_lora
                )
                _, ntok_dev, _ = worker.forward_batch_generation(mwb, sampling_metadata=None)
                t_disp = time.perf_counter()
                ntok = np.asarray(jax.device_get(ntok_dev))
                t_fwd = time.perf_counter()
                self._extract_dp_output_ids(ntok, mwb, batch)
                result = GenerationBatchResult(
                    logits_output=None,
                    next_token_ids=ntok.tolist(),
                    extend_input_len_per_req=None,
                    extend_logprob_start_len_per_req=None,
                    bid=mwb.bid,
                    cache_miss_count=0,
                )
                all_reqs = [r for info in batch.reqs_info if info.reqs for r in info.reqs]
                p_pages_per_req = []
                for r in all_reqs:
                    seq_len = len(r.fill_ids)
                    p_slots = p_r2t.req_to_token[r.req_pool_idx, :seq_len].copy()
                    p_pages_per_req.append(
                        (r, p_slots, slots_to_ordered_pages(p_slots, self.page_size))
                    )
                p_pages_all = np.concatenate([p for _, _, p in p_pages_per_req])
                d_stacked, bucket = kv_transfer.gather_to_dmesh(p_pages_all)
                t_gather = time.perf_counter()
                logger.info(
                    "[pd-timing] P fwd disp=%.0fms wait=%.0fms gather=%.0fms reqs=%d",
                    (t_disp - t0) * 1e3,
                    (t_fwd - t_disp) * 1e3,
                    (t_gather - t_fwd) * 1e3,
                    len(all_reqs),
                )
                self._pd_ready_q.put(
                    (p_idx, batch, result, p_pages_per_req, p_pages_all, d_stacked, bucket, t0)
                )
            except Exception:
                logger.exception("[pd_prefill %d] thread error", p_idx)
                psutil.Process().parent().send_signal(signal.SIGQUIT)
                return

    def _pd_drain_ready(self) -> None:
        """Main-thread side of async PD: pop ONE ready item per call (avoid
        burst stalling decode), rewrite req state to D side, scatter, then
        process_prefill under D context so finished reqs release D pool."""
        from sgl_jax.srt.disaggregation.pathways_pd import slots_to_ordered_pages

        try:
            item = self._pd_ready_q.get_nowait()
        except self._pd_qempty:
            return
        p_idx, batch, result, p_pages_per_req, p_pages_all, d_stacked, bucket, t0 = item
        p_alloc, p_r2t = self.p_allocs[p_idx], self.p_r2ts[p_idx]
        d_pages_all = []
        _ann = jax.profiler.TraceAnnotation("pd_drain_ready")
        _ann.__enter__()
        for r, p_slots, p_pages in p_pages_per_req:
            seq_len = len(r.fill_ids)
            n_pages = len(p_pages)
            d_slots = self.token_to_kv_pool_allocator.alloc(
                n_pages * self.page_size, dp_rank=r.dp_rank or 0
            )
            if d_slots is None:
                raise RuntimeError("[pd_async] D pool OOM during insert")
            d_pages = slots_to_ordered_pages(d_slots, self.page_size)
            p_slots64 = np.asarray(p_slots, np.int64)
            offsets = p_slots64 % self.page_size
            page_pos = {int(pg): k for k, pg in enumerate(p_pages)}
            tok_page_k = np.array([page_pos[int(s)] for s in p_slots64 // self.page_size], np.int64)
            d_slot_per_tok = (
                d_pages[tok_page_k].astype(np.int64) * self.page_size + offsets
            ).astype(np.int32)
            p_alloc.free(p_slots, dp_rank=r.dp_rank or 0)
            p_r2t.free_slots.append(r.req_pool_idx)
            r.req_pool_idx = None
            self.req_to_token_pool.alloc([r])
            self.req_to_token_pool.req_to_token[r.req_pool_idx, :seq_len] = d_slot_per_tok
            r.prefix_indices = d_slot_per_tok
            r.last_node = None
            r.cache_protected_len = 0
            r.kv_committed_len = seq_len
            r.kv_allocated_len = seq_len
            d_pages_all.append(d_pages)
        _ts0 = time.perf_counter()
        with jax.profiler.TraceAnnotation("pd_scatter"):
            self.kv_transfers[p_idx].scatter_from_dmesh(
                np.concatenate(d_pages_all), d_stacked, bucket
            )
        _ts = (time.perf_counter() - _ts0) * 1e3
        _tq = (time.perf_counter() - t0) * 1e3
        if _ts > 100 or _tq > 3000:
            logger.info(
                "[pd-timing] scatter=%.0fms P->drain=%.0fms reqs=%d",
                _ts,
                _tq,
                len(p_pages_per_req),
            )
        batch.req_to_token_pool = self.req_to_token_pool
        batch.token_to_kv_pool_allocator = self.token_to_kv_pool_allocator
        batch.tree_cache = self.tree_cache
        for info in batch.reqs_info:
            if info.reqs:
                info.req_pool_indices = np.array(
                    [r.req_pool_idx for r in info.reqs], dtype=np.int64
                )
        # process_prefill AFTER reqs are on D side: finished reqs (EOS on first
        # token) release_kv_cache against D pool here, not leak.
        with jax.profiler.TraceAnnotation("pd_process_prefill"):
            self.process_batch_result_prefill(batch, result, None)
        self._pd_inflight -= len(p_pages_per_req)
        _ann.__exit__(None, None, None)
        batch.filter_batch()
        if not batch.is_empty():
            if self.running_batch.is_empty():
                self.running_batch = batch
            else:
                self.running_batch.merge_batch(batch)
        if self.forward_ct % 20 == 0:
            logger.info(
                "[pd_async] insert %d reqs e2e=%.1fms (D avail=%d)",
                len(p_pages_per_req),
                (time.perf_counter() - t0) * 1e3,
                self.token_to_kv_pool_allocator.available_size(),
            )

    def _pd_get_next_batch_async(self):
        """Pathways-PD async scheduling: main loop only ever returns decode
        batches; prefill batches are pushed to _pd_prefill_qs[i] (round-robin
        across P slices) and run on prefill threads concurrently with D decode."""
        self._pd_drain_ready()
        n_p = self._pd_n_prefill
        for _ in range(n_p):
            i = self._pd_next_p
            self._pd_next_p = (i + 1) % n_p
            if self._pd_prefill_qs[i].full():
                continue
            saved = self.per_dp_max_running_requests
            d_running = sum(len(x.reqs) for x in self.running_batch.reqs_info if x.reqs)
            # swap_p_pool replaces running_batch with an empty one so PrefillAdder
            # stops mixing D-side future-token reservation into the P pool budget;
            # the D r2t slot constraint moves here explicitly (was implicit via
            # get_new_batch_prefill's len(running_batch.reqs) check).
            self.per_dp_max_running_requests = max(0, saved - self._pd_inflight - d_running)
            try:
                with self._pd_swap_p_pool(i):
                    new_batch = self.get_new_batch_prefill()
            finally:
                self.per_dp_max_running_requests = saved
            if new_batch is None:
                break
            assert all(
                r is None for r in self.chunked_reqs
            ), "[pd_async] chunked prefill not yet supported (IL must be < chunked_prefill_size)"
            self._pd_inflight += sum(len(info.reqs) for info in new_batch.reqs_info if info.reqs)
            self._pd_prefill_qs[i].put_nowait(new_batch)
            break
        if not self.running_batch.is_empty() and not self.running_batch.is_prefill_only:
            self.running_batch = self.update_running_batch(self.running_batch)
            return self.running_batch if not self.running_batch.is_empty() else None
        return None

    def _pd_gather_output(self, arr, is_prefill: bool) -> np.ndarray:
        """sub-mesh forward output（仅半边 process addressable）→ 全 process numpy。"""
        if self.pd == "pathways":
            return np.asarray(jax.device_get(arr))
        from jax.experimental.multihost_utils import broadcast_one_to_all

        if getattr(arr, "addressable_shards", None):
            local = np.asarray(jax.device_get(arr.addressable_shards[0].data))
        else:
            local = np.zeros(arr.shape, arr.dtype)
        src_devs = (self.p_mesh if is_prefill else self.mesh).devices.flatten()
        src_pid = min(int(d.process_index) for d in src_devs)
        return np.asarray(broadcast_one_to_all(local, is_source=(jax.process_index() == src_pid)))

    def _pd_swap_p_pool(self, p_idx: int = 0):
        """get_new_batch_prefill / process_batch_result_prefill 期间临时把
        self.{tree_cache,req_to_token_pool,token_to_kv_pool_allocator,mesh} 指向 P[p_idx] 侧。
        """
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            saved = (
                self.tree_cache,
                self.req_to_token_pool,
                self.token_to_kv_pool_allocator,
                self.mesh,
                self.running_batch,
                self.max_total_num_tokens,
            )
            self.tree_cache = self.p_trees[p_idx]
            self.req_to_token_pool = self.p_r2ts[p_idx]
            self.token_to_kv_pool_allocator = self.p_allocs[p_idx]
            self.max_total_num_tokens = self.tp_workers_p[p_idx].max_total_num_tokens
            self.mesh = self.p_meshes[p_idx] if hasattr(self, "p_meshes") else self.p_mesh
            # PrefillAdder reads running_batch for rem_total_tokens AND writes
            # batch_is_full back into it on NO_TOKEN; without this swap the
            # D-side decode reqs' future-token reservation starves the P pool
            # admission and the sticky batch_is_full stalls push until a D req
            # finishes (multi-P 1.16x root cause).
            empties = getattr(self, "_pd_empty_running_p", None)
            if empties is not None:
                empty = empties[p_idx]
                for info in empty.reqs_info:
                    info.batch_is_full = False
                self.running_batch = empty
            try:
                yield
            finally:
                (
                    self.tree_cache,
                    self.req_to_token_pool,
                    self.token_to_kv_pool_allocator,
                    self.mesh,
                    self.running_batch,
                    self.max_total_num_tokens,
                ) = saved

        return _ctx()

    def _pd_migrate(self, batch: ScheduleBatch) -> bool:
        """把 batch（prefill 完成、已 filter 掉 chunked/finished）中的 reqs 从 P pool
        迁移到 D pool（KV ppermute + r2t/alloc/req 状态重写），使其可 merge 进 D
        running_batch。D pool 不够时返回 False（caller 应暂存等 D free）。"""
        from sgl_jax.srt.disaggregation.pathways_pd import migrate_reqs_p_to_d

        all_reqs = []
        for info in batch.reqs_info:
            if info.reqs:
                all_reqs.extend(info.reqs)
        if not all_reqs:
            return True
        need = sum(
            ((len(r.fill_ids) + self.page_size - 1) // self.page_size) * self.page_size
            for r in all_reqs
        )
        if self.token_to_kv_pool_allocator.available_size() < need:
            return False
        t0 = time.perf_counter()
        migrate_reqs_p_to_d(
            all_reqs,
            self.page_size,
            self.p_r2t,
            self.p_alloc,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.kv_transfer,
        )
        # batch 元信息改绑 D pool（merge_batch 不检查，但后续 prepare_for_decode 用）
        batch.req_to_token_pool = self.req_to_token_pool
        batch.token_to_kv_pool_allocator = self.token_to_kv_pool_allocator
        batch.tree_cache = self.tree_cache
        for info in batch.reqs_info:
            if info.reqs:
                info.req_pool_indices = np.array(
                    [r.req_pool_idx for r in info.reqs], dtype=np.int64
                )
        if self.forward_ct % 50 == 0:
            logger.info(
                "[pathways_pd] migrate %d reqs in %.1fms (D avail=%d)",
                len(all_reqs),
                (time.perf_counter() - t0) * 1e3,
                self.token_to_kv_pool_allocator.available_size(),
            )
        return True

    def process_batch_result(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        launch_done: threading.Event | None = None,
    ):
        if batch.forward_mode.is_decode():
            self.process_batch_result_decode(batch, result, launch_done)
        elif batch.forward_mode.is_extend():
            if self.pd:
                with self._pd_swap_p_pool():
                    self.process_batch_result_prefill(batch, result, launch_done)
                return
            self.process_batch_result_prefill(batch, result, launch_done)
        elif batch.forward_mode.is_idle():
            if self.enable_overlap:
                self.tp_worker.resolve_last_batch_result(launch_done)
                self.set_next_batch_sampling_info_done(batch)
        elif batch.forward_mode.is_dummy_first():
            self.set_next_batch_sampling_info_done(batch)

    def set_next_batch_sampling_info_done(self, batch: ScheduleBatch):
        if batch.next_batch_sampling_info:
            # Update grammar vocab masks for next batch in overlap mode
            if batch.next_batch_sampling_info.grammars is not None:
                batch.next_batch_sampling_info.update_grammar_vocab_mask()
            batch.next_batch_sampling_info.sampling_info_done.set()

    def _current_sampling_info_owner(self):
        if self.spec_algorithm is not None and not self.spec_algorithm.is_none():
            return self.draft_worker
        return self.tp_worker

    def _run_speculative_batch(
        self,
        batch: ScheduleBatch,
        precompile_token_paddings,
        precompile_bs_paddings,
        precompile_cache_loc_paddings,
    ):
        if batch.forward_mode.is_extend():
            # Spec extend always uses the padded mwb so target and draft
            # see identical shapes regardless of dp_size / multi-layer
            # (#1090 + #1053 P1-5b assert dp>1 spec extend must go here).
            model_worker_batch = batch.get_model_worker_batch(
                precompile_token_paddings,
                precompile_bs_paddings,
                precompile_cache_loc_paddings,
                self.page_size,
                self.server_args.enable_static_lora,
            )
        else:
            model_worker_batch = batch.get_spec_model_worker_batch(
                precompile_token_paddings,
                precompile_bs_paddings,
                precompile_cache_loc_paddings,
                self.page_size,
                self.server_args.enable_static_lora,
                draft_token_num=self.draft_worker.speculative_num_draft_tokens,
            )

        use_spec_decode_overlap = can_use_spec_decode_overlap(
            self.enable_overlap, self.spec_algorithm, batch
        )
        use_spec_prefill_overlap = can_use_spec_prefill_overlap(
            self.enable_overlap, self.spec_algorithm, batch
        ) and self.draft_worker._can_use_fused_spec_prefill(model_worker_batch)
        use_legacy_eagle3_decode = batch.forward_mode.is_decode() and use_legacy_eagle3_non_overlap(
            self.enable_overlap, self.spec_algorithm
        )
        if use_spec_decode_overlap:
            batch_output, published_new_seq_lens = (
                self.draft_worker.forward_batch_speculative_decode_overlap(model_worker_batch)
            )
        elif use_spec_prefill_overlap:
            batch_output = self.draft_worker.forward_batch_speculative_prefill_overlap(
                model_worker_batch
            )
            published_new_seq_lens = None
        else:
            batch_output = self.draft_worker.forward_batch_speculative_generation(
                model_worker_batch
            )
            if use_legacy_eagle3_decode:
                published_new_seq_lens = None
            else:
                published_new_seq_lens = (
                    publish_spec_decode_new_seq_lens(batch_output)
                    if batch.forward_mode.is_decode()
                    else None
                )

        if batch_output.next_draft_input is not None:
            per_rank_spec = ScheduleBatch._split_spec_info_per_rank(
                batch_output.next_draft_input, model_worker_batch.real_bs_per_dp
            )
            for r, s in enumerate(per_rank_spec):
                batch.reqs_info[r].spec_info = s

        if not use_spec_decode_overlap:
            if use_legacy_eagle3_decode and batch_output.accept_lens is not None:
                new_seq_lens = np.asarray(jax.device_get(batch_output.accept_lens))
                advance_from_accept_lens = True
            else:
                new_seq_lens = (
                    np.asarray(jax.device_get(published_new_seq_lens))
                    if published_new_seq_lens is not None
                    else None
                )
                advance_from_accept_lens = False
            per_dp_bs = model_worker_batch.per_dp_bs_size
            for dp_rank, info in enumerate(batch.reqs_info):
                if info.seq_lens is None or len(info.seq_lens) == 0:
                    continue
                if new_seq_lens is not None:
                    off = dp_rank * per_dp_bs
                    delta = new_seq_lens[off : off + len(info.seq_lens)]
                    if advance_from_accept_lens:
                        info.seq_lens = info.seq_lens + delta
                    else:
                        info.seq_lens = delta
                else:
                    info.seq_lens = info.seq_lens + 1

        defer_spec_output = use_spec_decode_overlap or use_spec_prefill_overlap
        next_token_ids = None
        if not defer_spec_output:
            next_token_ids = np.asarray(jax.device_get(batch_output.next_token_ids))
            self._extract_dp_output_ids(next_token_ids, model_worker_batch, batch)

        return (
            model_worker_batch,
            batch_output,
            next_token_ids,
            batch_output.logits_output,
            batch_output.cache_miss_count,
            defer_spec_output,
            use_spec_prefill_overlap,
        )

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
            abort_out = AbortReq(rid=req.rid)
            if self._comm_backend is not None:
                self._comm_backend.send_pyobj(abort_out)
            else:
                self.send_to_tokenizer.send_pyobj(abort_out)
            logger.debug("Abort queued request. rid=%s", req.rid)

        # Delete the requests in the grammar queue
        for req in self.grammar_queue:
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                logger.debug("Abort grammar queue request. rid=%s", req.rid)
                if req.grammar:
                    req.grammar.cancel()
                req.set_finish_with_abort("Aborted by AbortReq.")

        # Delete requests in the running batch
        reqs = []
        for info in self.running_batch.reqs_info:
            if info.reqs:
                reqs.extend(info.reqs)

        if self.cur_batch is not None and self.cur_batch is not self.running_batch:
            for info in self.cur_batch.reqs_info:
                if info.reqs:
                    reqs.extend(info.reqs)

        for req in reqs:
            if not req.finished() and (recv_req.abort_all or req.rid.startswith(recv_req.rid)):
                # Abort method 3: set `to_finish`
                # The request will still run one decode forward pass.
                # Then we reuse all existing code to clean up the KV cache allocation.
                logger.debug("Abort running request. rid=%s", req.rid)
                req.to_finish = FINISH_ABORT()

        # Abort PD disaggregation queues
        prefill_q = self.disagg_prefill_queue
        if prefill_q is not None:
            for entry in prefill_q.abort_matching(recv_req.rid, recv_req.abort_all):
                logger.debug("Abort prefill queue request. rid=%s", entry.req_id)
                entry.sender.abort()
                if entry.on_terminal is not None:
                    try:
                        entry.on_terminal()
                    except Exception:
                        logger.exception(
                            "on_terminal for aborted prefill req_id=%s raised",
                            entry.req_id,
                        )

        prealloc_q = self.disagg_prealloc_queue
        if prealloc_q is not None:
            for entry in prealloc_q.abort_matching(recv_req.rid, recv_req.abort_all):
                logger.debug("Abort prealloc queue request. rid=%s", entry.req_id)
                if entry.receiver is not None:
                    entry.receiver.abort()
                if entry.kv_indices is not None:
                    self._release_decode_kv_indices(entry.kv_indices)
                self._abort_decode_request(entry.req, "abort_request")

        transfer_q = self.disagg_transfer_queue
        if transfer_q is not None:
            for entry in transfer_q.abort_matching(recv_req.rid, recv_req.abort_all):
                logger.debug("Abort transfer queue request. rid=%s", entry.req_id)
                if entry.receiver is not None:
                    entry.receiver.abort()
                if entry.kv_indices is not None:
                    self._release_decode_kv_indices(entry.kv_indices)
                self._abort_decode_request(entry.req, "abort_request")

        # Decode reqs deferred because no prefill was registered yet hold no KV
        # or receiver, but abort_request must still drop them so a cancelled
        # request is not re-admitted on the next decode tick.
        pending_bootstrap = getattr(self, "_pd_pending_bootstrap", None)
        if pending_bootstrap:
            survivors = []
            for req in pending_bootstrap:
                if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                    logger.debug("Abort pending-bootstrap request. rid=%s", req.rid)
                    self._abort_decode_request(req, "abort_request")
                else:
                    survivors.append(req)
            self._pd_pending_bootstrap = survivors

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
            all_reqs = [
                req for info in self.running_batch.reqs_info for req in info.reqs if info.reqs
            ]
            if len(all_reqs) != 0:
                # clear the kv cache
                retracted_reqs = self.running_batch.retract_all(self.server_args)
                for req in retracted_reqs:
                    self._add_request_to_queue(req)

            self.chunked_reqs = [None] * self.dp_size
            logger.info("Paused generation retracted")
        elif recv_req.mode == "in_place":
            logger.info("Paused generation in place")

    def continue_generation(self, recv_req: ContinueGenerationReqInput):
        self._engine_paused = False
        logger.info("Generation continued")


def _reserve_host_slot_for_pd(host_pool, use_d2h_staging, req):
    """D1 admission. Returns (admit_ok, reserved_buffer_id).

    For a D2H-staged PD req, reserve a host-pool slot. If the pool is
    full, (False, None) tells the caller to skip the req this round so it
    stays in the waiting queue (backpressure). Non-PD / non-staged reqs
    are always admitted with no reservation.
    """
    if (
        host_pool is None
        or not use_d2h_staging
        or getattr(req, "bootstrap_room", None) is None
        or getattr(req, "disagg_host_buffer_id", None) is not None
    ):
        return True, None
    buffer_id = host_pool.reserve()
    if buffer_id is None:
        return False, None
    return True, buffer_id


def dispatch_scheduler_event_loop(scheduler: Scheduler, server_args: ServerArgs) -> None:
    """Choose and run the appropriate scheduler event loop."""

    mode = server_args.disaggregation_mode
    if mode == "prefill":
        scheduler.event_loop_normal_disagg_prefill()
    elif mode == "decode":
        scheduler.event_loop_normal_disagg_decode()
    elif scheduler.enable_overlap:
        scheduler.event_loop_overlap()
    else:
        scheduler.event_loop_normal()


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    dp_rank: int | None,
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
        install_disaggregation_wiring(scheduler, server_args)
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": scheduler.max_total_num_tokens,
                "max_req_input_len": scheduler.max_req_input_len,
            }
        )

        dispatch_scheduler_event_loop(scheduler, server_args)

    except Exception:
        traceback = get_exception_traceback()
        logger.error("Scheduler hit an exception: %s", traceback)
        parent_process.send_signal(signal.SIGQUIT)


def run_scheduler_loop_thread_after_create(
    server_args: ServerArgs,
    port_args: PortArgs,
):
    current_process = psutil.Process()
    # Create a scheduler and run the event loop
    try:
        scheduler = Scheduler(server_args, port_args)
        install_disaggregation_wiring(scheduler, server_args)
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
        dispatch_scheduler_event_loop(scheduler, server_args)
    except Exception:
        traceback = get_exception_traceback()
        logger.error("Scheduler hit an exception: %s", traceback)
        current_process.send_signal(signal.SIGQUIT)
