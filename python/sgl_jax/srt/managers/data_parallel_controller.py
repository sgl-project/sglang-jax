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
from sgl_jax.srt.server_args import PortArgs, ServerArgs
from sgl_jax.srt.utils import get_zmq_socket


# Import scheduler only when needed to avoid circular dependencies
def _import_scheduler():
    from sgl_jax.srt.managers.scheduler import run_scheduler_process

    return run_scheduler_process


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
        attn_tp_size, dp_rank = compute_dp_attention_world_info(
            self.server_args.enable_dp_attention,
            self.server_args.node_rank,
            self.server_args.tp_size,
            self.server_args.dp_size,
        )
        self._launch_local_scheduler(self.server_args, self.port_args, dp_rank, None)

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

        # Import scheduler function when needed
        run_scheduler_process = _import_scheduler()

        proc = mp.Process(
            target=run_scheduler_process,
            args=(
                self.server_args,
                self.port_args,
                dp_rank,
                writer,
            ),
        )
        proc.start()
        self.scheduler_proc = proc
        scheduler_info = reader.recv()
        ready_event.set()
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
            dp_port_args = self.launch_dp_attention_schedulers(
                self.server_args, self.port_args
            )
        else:
            dp_port_args = self.launch_dp_schedulers(self.server_args, self.port_args)

        if self.server_args.node_rank == 0:
            self.workers[dp_rank] = get_zmq_socket(
                self.context,
                zmq.PUSH,
                dp_port_args[dp_rank].scheduler_input_ipc_name,
                True,
            )

    def launch_dp_schedulers(self, server_args, port_args):
        base_gpu_id = 0

        threads = []
        sockets = []
        dp_port_args = []
        ready_events = []
        for dp_rank in range(server_args.dp_size):
            tmp_port_args = PortArgs.init_new(server_args)
            tmp_port_args.tokenizer_ipc_name = port_args.tokenizer_ipc_name
            tmp_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name
            dp_port_args.append(tmp_port_args)

            # This port is checked free in PortArgs.init_new.
            # We hold it first so that the next dp worker gets a different port
            sockets.append(bind_port(tmp_port_args.nccl_port))

            ready_event = threading.Event()
            ready_events.append(ready_event)

            # Create a thread for each worker
            thread = threading.Thread(
                target=self.launch_tensor_parallel_group_thread,
                args=(server_args, tmp_port_args, base_gpu_id, dp_rank, ready_event),
            )
            threads.append(thread)
            base_gpu_id += server_args.tp_size * server_args.gpu_id_step

        # Free all sockets before starting the threads to launch TP workers
        for sock in sockets:
            sock.close()

        # Start all threads
        for thread in threads:
            thread.start()
        for event in ready_events:
            event.wait()

        return dp_port_args

    def launch_dp_attention_schedulers(self, server_args, port_args):
        self.launch_tensor_parallel_group(server_args, port_args, 0, None)
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

    try:
        controller = DataParallelController(server_args, port_args)
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": controller.max_total_num_tokens,
                "max_req_input_len": controller.max_req_input_len,
            }
        )
        if server_args.node_rank == 0:
            controller.event_loop()
        for proc in controller.scheduler_procs:
            proc.join()
            logger.error(
                f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}"
            )
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DataParallelController hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
