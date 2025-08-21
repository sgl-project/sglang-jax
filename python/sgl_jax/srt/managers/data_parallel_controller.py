"""Data parallel controller for distributing requests across DP ranks."""

import logging
import multiprocessing as mp
import os
import signal
import threading
import time

import psutil
import zmq

try:
    import setproctitle
except ImportError:
    setproctitle = None

from sgl_jax.srt.layers.dp_attention import compute_dp_attention_world_info
from sgl_jax.srt.managers.io_struct import (
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sgl_jax.srt.managers.scheduler import run_scheduler_process
from sgl_jax.srt.server_args import PortArgs, ServerArgs
from sgl_jax.srt.utils import get_zmq_socket
from sgl_jax.srt.utils.common_utils import configure_logger


def get_exception_traceback():
    """Simple exception traceback function."""
    import traceback

    return traceback.format_exc()


logger = logging.getLogger(__name__)


class DataParallelController:
    """A controller that launches schedulers and dispatches requests across DP ranks."""

    def __init__(self, server_args: ServerArgs, port_args: PortArgs):
        self.server_args = server_args
        self.port_args = port_args

        # Launch local scheduler first
        self.scheduler_proc = None
        attn_tp_size, attn_dp_rank, dp_rank = compute_dp_attention_world_info(
            self.server_args.enable_dp_attention,
            self.server_args.node_rank,
            self.server_args.tp_size,
            self.server_args.dp_size,
        )
        self.attn_tp_size = attn_tp_size
        self.attn_dp_rank = attn_dp_rank
        self.dp_rank = dp_rank
        logger.debug(
            f"DP rank: {dp_rank}, attn_dp_rank: {attn_dp_rank}, attn_tp_size: {attn_tp_size}"
        )
        ready_event = threading.Event()
        self._launch_local_scheduler(
            self.server_args, self.port_args, dp_rank, ready_event
        )
        logger.debug(f"Node {self.server_args.node_rank} Scheduler launched")
        # Only node 0 sets up communication to all DP ranks
        if server_args.node_rank == 0:
            self._setup_dp_communication()

        # Round-robin counter for request dispatching
        self.round_robin_counter = 0

    def _launch_local_scheduler(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        dp_rank: int,
        ready_event: threading.Event,
    ):
        """Launch scheduler on current node."""
        logger.info(
            f"Launching scheduler for Node rank {self.server_args.node_rank} DP rank {dp_rank}"
        )
        reader, writer = mp.Pipe(duplex=False)

        proc = mp.Process(
            target=run_scheduler_process,
            args=(
                self.server_args,
                PortArgs.init_new(self.server_args, dp_rank),
                self.dp_rank,
                writer,
            ),
        )
        proc.start()
        self.scheduler_proc = proc
        logger.debug(
            f"Node {self.server_args.node_rank} Scheduler started and waiting for info"
        )
        scheduler_info = reader.recv()
        logger.debug(
            f"Node {self.server_args.node_rank} Scheduler Info: {scheduler_info}"
        )
        ready_event.set()
        logger.debug(f"Node {self.server_args.node_rank} Scheduler ready and set event")
        return scheduler_info

    def _setup_dp_communication(self):
        """Set up ZMQ communication to all DP rank schedulers (only on node 0)."""
        logger.info("Setting up DP communication to all schedulers")

        # Initialize ZMQ context
        self.context = zmq.Context(1 + self.server_args.dp_size)

        # Receive from tokenizer
        self.recv_from_tokenizer = get_zmq_socket(
            self.context, zmq.PULL, self.port_args.scheduler_input_ipc_name, False
        )

        # Initialize connections to each DP rank's scheduler
        self.workers = [None] * self.server_args.dp_size

        # Create port args for each DP rank to get their scheduler addresses
        if self.server_args.enable_dp_attention:
            dp_port_args = self.get_dp_port_args(self.server_args)
        else:
            dp_port_args = []

        if self.server_args.node_rank == 0:
            for dp_rank in range(self.server_args.dp_size):
                self.workers[dp_rank] = get_zmq_socket(
                    self.context,
                    zmq.PUSH,
                    dp_port_args[dp_rank].scheduler_input_ipc_name,
                    False,
                )

    def get_dp_port_args(self, server_args):
        # self.launch_tensor_parallel_group(server_args, port_args, 0, None)
        logger.debug(f"Launching DP attention schedulers")
        dp_port_args = []
        for dp_rank in range(server_args.dp_size):
            dp_port_args.append(PortArgs.init_new(server_args, dp_rank))
        return dp_port_args

    def round_robin_scheduler(self, req):
        """Dispatch request using round-robin strategy."""
        if self.server_args.node_rank == 0:
            self.workers[self.round_robin_counter].send_pyobj(req)
            self.round_robin_counter = (self.round_robin_counter + 1) % len(
                self.workers
            )

    def event_loop(self):
        """Main event loop for processing requests (only on node 0)."""
        if self.server_args.node_rank != 0:
            return  # Only node 0 runs the event loop

        logger.info("DataParallelController starting event loop")

        while True:
            try:
                # Non-blocking receive
                recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)

                if isinstance(
                    recv_req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)
                ):
                    # Dispatch tokenized requests using round-robin
                    self.round_robin_scheduler(recv_req)
                else:
                    # Send other control messages to all workers
                    for worker in self.workers:
                        worker.send_pyobj(recv_req)

            except zmq.ZMQError:
                # No message available, sleep briefly to avoid busy waiting
                time.sleep(0.001)
                continue


def run_data_parallel_controller_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
):
    setproctitle.setproctitle("sgl-jax::data_parallel_controller")
    parent_process = psutil.Process().parent()
    configure_logger(server_args, prefix=f"DPController{server_args.node_rank}")
    try:
        logger.debug(f"Node {server_args.node_rank} DataParallelController started")
        controller = DataParallelController(server_args, port_args)
        logger.debug(f"Node {server_args.node_rank} DataParallelController launched")
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": 1000000,  # controller.max_total_num_tokens,
                "max_req_input_len": 1000000,  # controller.max_req_input_len,
            }
        )
        if server_args.node_rank == 0:
            logger.debug(f"Node {server_args.node_rank} running event loop")
            controller.event_loop()
        logger.debug(f"Node {server_args.node_rank} scheduler joined")
        controller.scheduler_proc.join()
        logger.error(
            f"Scheduler or DataParallelController {controller.scheduler_proc.pid} terminated with {controller.scheduler_proc.exitcode}"
        )
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DataParallelController hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
