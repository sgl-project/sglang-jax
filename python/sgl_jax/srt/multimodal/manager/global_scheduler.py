import dataclasses
import logging
import os
import queue
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import psutil
import setproctitle
import zmq

from sgl_jax.srt.managers.io_struct import AbortReq
from sgl_jax.srt.multimodal.manager.device_manager import DeviceManager
from sgl_jax.srt.multimodal.manager.io_struct import TokenizedGenerateMMReqInput
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.manager.stage import Stage
from sgl_jax.srt.multimodal.manager.utils import load_stage_configs_from_yaml
from sgl_jax.srt.multimodal.models.static_configs import get_stage_config_path
from sgl_jax.srt.server_args import PortArgs, ServerArgs
from sgl_jax.srt.utils import configure_logger, kill_itself_when_parent_died
from sgl_jax.srt.utils.common_utils import get_zmq_socket
from sgl_jax.utils import TypeBasedDispatcher, get_exception_traceback

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ReqTrackingState:
    """Tracks a request's state in the pipeline.

    Attributes:
        req: The request object.
        current_stage: The stage index where the request is currently being
            processed (0-indexed). A value of -1 indicates the request has
            not yet entered the pipeline.
    """

    req: Req
    current_stage: int = 0


class GlobalScheduler:
    """Orchestrates multimodal request scheduling and stage dispatch.

    GlobalScheduler is responsible for receiving tokenized requests from the
    tokenizer via a ZMQ PULL socket, converting them into `Req` objects, and
    dispatching them into a pipeline of `Stage` instances. It also collects
    outputs from stages and forwards final results to the detokenizer via a
    ZMQ PUSH socket.

    Key responsibilities:
    - Initialize ZMQ sockets for inter-process communication.
    - Load stage configurations and build a list of `Stage` instances.
    - Manage `DeviceManager` and a request store for tracking in-flight
        requests.
    - Start each stage in its own thread and coordinate stage input/output
        queues (using `queue.Queue`).
    - Run the main event loop that pulls requests, dispatches them, and
        collects stage results.

    Notes:
    - `start_stage` spawns threads which execute `Stage.run_stage`.
    - `run_global_scheduler_process` wraps this class for running in a
        separate process and sets up process-level bookkeeping (title,
        parent-death handling, logging).
    """

    def __init__(self, server_args: ServerArgs, port_args: PortArgs) -> None:
        """Initialize the GlobalScheduler.

        Creates ZMQ sockets for receiving tokenized requests and sending
        detokenized outputs, loads stage configurations, initializes the
        `DeviceManager`, and prepares in/out queues and stage threads.
        """
        context = zmq.Context(2)
        self.server_args = server_args
        self.recv_from_tokenizer = get_zmq_socket(
            context, zmq.PULL, port_args.scheduler_input_ipc_name, False
        )
        self.send_to_detokenizer = get_zmq_socket(
            context, zmq.PUSH, port_args.detokenizer_ipc_name, False
        )

        stage_config_path = get_stage_config_path(server_args.model_path)
        logger.info("Loading stage config from: %s", stage_config_path)
        self.stage_configs = load_stage_configs_from_yaml(stage_config_path)
        self.device_manager = DeviceManager()
        self._init_stage()
        self.req_store = dict()

    def _init_stage(self):
        """Build Stage instances and wire stage queues.

        Builds stages in parallel using a ThreadPoolExecutor, sorts them by
        index, creates per-stage input/output queues, attaches the queues to
        each `Stage`, starts stage threads, and prepares the request
        dispatcher mapping.
        """

        def _build_stage(idx_cfg: tuple[int, Any]) -> tuple[int, Stage]:
            idx, cfg = idx_cfg
            return idx, Stage(cfg, device_manager=self.device_manager, server_args=self.server_args)

        with ThreadPoolExecutor(
            max_workers=min(len(self.stage_configs), max(1, os.cpu_count() or 1))
        ) as executor:
            futures = [
                executor.submit(_build_stage, (idx, cfg))
                for idx, cfg in enumerate(self.stage_configs)
            ]
            results: list[tuple[int, Stage]] = []
            for fut in as_completed(futures):
                results.append(fut.result())
        results.sort(key=lambda x: x[0])
        self.stage_list = [st for _, st in results]
        self.in_queues = [queue.Queue() for _ in range(len(self.stage_list))]
        self.out_queues = [queue.Queue() for _ in range(len(self.stage_list))]
        for i, stage in enumerate(self.stage_list):
            stage.set_in_queue(self.in_queues[i])
            stage.set_out_queue(self.out_queues[i])
        self.start_stage()
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateMMReqInput, self.convert_request),
                (AbortReq, self.handle_abort_request),
            ]
        )

    def handle_abort_request(self, abort_req: AbortReq):
        """Handle a client abort request.

        Aborts requests matching the given rid (or all requests if abort_all
        is True). The abort is performed in two phases:

        1. Remove matching requests from req_store and send abort notifications
           only to stages where the request is currently at or will pass through
           (current_stage and subsequent stages).
        2. Send AbortReq back to detokenizer (which forwards to tokenizer) to
           notify the client that the request has been aborted.

        For requests that are already being processed by a stage, the stage
        scheduler will check for abort status and skip remaining work.
        """
        logger.info(
            "Received abort request for rid=%s, abort_all=%s",
            abort_req.rid,
            abort_req.abort_all,
        )

        # Find all requests to abort
        rids_to_abort = []
        if abort_req.abort_all:
            rids_to_abort = list(self.req_store.keys())
        else:
            # Match requests whose rid starts with the given rid
            for rid in list(self.req_store.keys()):
                if rid.startswith(abort_req.rid):
                    rids_to_abort.append(rid)

        if not rids_to_abort:
            logger.info("No matching requests found for abort request rid=%s", abort_req.rid)
            return None

        # Abort each matching request
        for rid in rids_to_abort:
            tracking_state = self.req_store.pop(rid, None)
            if tracking_state is not None:
                current_stage = tracking_state.current_stage
                logger.info("Aborting request rid=%s at stage %d", rid, current_stage)

                # Send abort signal only to current stage (subsequent stages won't
                # receive the request because event_loop checks req_store)
                stage_abort_req = AbortReq(
                    rid=rid,
                    aborted_message="Aborted by client request",
                )
                try:
                    self.in_queues[current_stage].put_nowait(stage_abort_req)
                except Exception as e:
                    logger.warning("Failed to send abort to stage %d queue: %s", current_stage, e)

                # Send AbortReq to detokenizer -> tokenizer to notify client
                self.send_to_detokenizer.send_pyobj(stage_abort_req)

        return None

    def convert_request(self, input: TokenizedGenerateMMReqInput):
        """Convert a tokenized input into internal `Req`.

        Parses input size, constructs a `Req` object, ensures the request id
        is unique in `req_store`, and stores the request with tracking state.
        """

        size_str = input.size if input.size else "1024*1024"

        req = Req(
            rid=input.rid,
            input_ids=input.input_ids,
            negative_input_ids=input.negative_input_ids,
            num_outputs_per_prompt=input.n,
            height=int(size_str.split("*")[0]),
            width=int(size_str.split("*")[1]),
            num_frames=input.num_frames,
            num_inference_steps=(
                input.num_inference_steps if input.num_inference_steps is not None else 50
            ),
            data_type=input.data_type,
            save_output=input.save_output,
        )
        if req.rid in self.req_store:
            raise RuntimeError(f"{req.rid} is already in req_store")
        # Store with tracking state, starting at stage 0
        self.req_store[req.rid] = ReqTrackingState(req=req, current_stage=0)
        return req

    def start_stage(self):
        """Start each stage in its own thread and wait for readiness.

        Spawns a thread per `Stage` running `Stage.run_stage` and then blocks
        on each stage's output queue for a readiness message. Raises if a
        stage fails to initialize.
        """

        import threading

        for stage in self.stage_list:
            thread = threading.Thread(target=stage.run_stage)
            thread.start()
        for i, q in enumerate(self.out_queues):
            status = q.get()
            if status["status"] != "ready":
                raise Exception(f"stage {i} init failed")

    def recv_request(self):
        """Non-blockingly drain incoming requests from the tokenizer socket.

        Returns a list of Python objects received via ZMQ. The method uses
        `zmq.NOBLOCK` to avoid blocking when there are no messages.
        """

        recv_reqs = []
        while True:
            try:
                recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
            except zmq.ZMQError:
                break
            recv_reqs.append(recv_req)
        return recv_reqs

    def event_loop(self):
        """Main event loop: receive, dispatch, and collect stage results.

        Loop behavior:
        - Drain incoming tokenized requests and dispatch them via the
          `TypeBasedDispatcher` to create per-stage `Req` objects.
        - If no new requests are available, poll each stage for outputs via
          `stage.try_collect()` and either forward final outputs to the
          detokenizer socket or enqueue the next-stage requests.
        The loop runs forever; external supervision should manage process
        lifecycle and termination.
        """

        while True:
            reqs = self.recv_request()
            if len(reqs) > 0 and self.server_args.log_requests:
                logger.info("recv_reqs from tokenizer %s", reqs)
            if reqs:
                for req in reqs:
                    dispatched_req = self._request_dispatcher(req)
                    if dispatched_req is not None:
                        stage_reqs = dispatched_req.to_stage_reqs(self.stage_configs[0].scheduler)
                        for stage_req in stage_reqs:
                            self.in_queues[0].put_nowait(stage_req)
            else:
                for i, stage in enumerate(self.stage_list):
                    stage_result = Req.from_stage(stage.try_collect(), self.req_store)
                    if stage_result is not None and self.server_args.log_requests:
                        logger.info(
                            "stage-%d result_type=%s is_final=%s",
                            i,
                            type(stage_result),
                            self.stage_configs[i].final_output,
                        )
                    if stage_result is None:
                        continue

                    # Check if request was aborted (rid no longer in req_store)
                    if stage_result.rid not in self.req_store:
                        logger.info(
                            "Skipping aborted request rid=%s from stage-%d",
                            stage_result.rid,
                            i,
                        )
                        continue

                    if self.stage_configs[i].final_output:
                        self.send_to_detokenizer.send_pyobj(stage_result)
                        del self.req_store[stage_result.rid]
                    else:
                        # Update tracking state to next stage before dispatching
                        next_stage = i + 1
                        self.req_store[stage_result.rid].current_stage = next_stage

                        stage_reqs = stage_result.to_stage_reqs(
                            self.stage_configs[next_stage].scheduler
                        )
                        for stage_req in stage_reqs:
                            self.in_queues[next_stage].put_nowait(stage_req)


def run_global_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
):
    """Run the GlobalScheduler inside a separate process context.

    This helper performs process-level setup (parent-death handling,
    process title, logger configuration), constructs the `GlobalScheduler`,
    signals readiness to the parent via `pipe_writer`, and then starts the
    scheduler's event loop. Exceptions are logged and the parent is signaled
    to terminate on fatal errors.
    """

    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang-jax::global_scheduler")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        scheduler = GlobalScheduler(server_args, port_args)
        # TODO: Implement event loop
        pipe_writer.send(
            {
                "status": "ready",
            }
        )
        scheduler.event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error("GlobalScheduler hit an exception: %s", traceback)
        parent_process.send_signal(signal.SIGQUIT)
    return scheduler
