"""Data parallel controller for distributing requests across DP ranks."""

import logging
import multiprocessing as mp
import os
import signal
import threading
import time

import zmq

try:
    import setproctitle
except ImportError:
    setproctitle = None

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
        self.scheduler_pipe_reader = None
        self._launch_local_scheduler()

        # Only node 0 sets up communication to all DP ranks
        if server_args.node_rank == 0:
            self._setup_dp_communication()

        # Round-robin counter for request dispatching
        self.round_robin_counter = 0

    def _launch_local_scheduler(self):
        """Launch scheduler on current node."""
        logger.info(f"Launching scheduler for DP rank {self.server_args.node_rank}")
        reader, writer = mp.Pipe(duplex=False)

        # Import scheduler function when needed
        run_scheduler_process = _import_scheduler()

        proc = mp.Process(
            target=run_scheduler_process,
            args=(
                self.server_args,
                self.port_args,
                None,
                writer,
            ),
        )
        proc.start()
        self.scheduler_proc = proc
        self.scheduler_pipe_reader = reader

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
        for dp_rank in range(self.server_args.dp_size):
            if dp_rank == 0:
                # Node 0 uses the original port_args
                scheduler_port_args = self.port_args
            else:
                # Other nodes get new port args
                # Note: We need to know the address of other nodes' schedulers
                # For now, we'll create port args but won't connect until we know addresses
                scheduler_port_args = PortArgs.init_new(self.server_args, dp_rank)

            # Connect to this DP rank's scheduler
            self.workers[dp_rank] = get_zmq_socket(
                self.context,
                zmq.PUSH,
                scheduler_port_args.scheduler_input_ipc_name,
                True,
            )

    def round_robin_scheduler(self, req):
        """Dispatch request using round-robin strategy."""
        if self.server_args.node_rank == 0:
            self.workers[self.round_robin_counter].send_pyobj(req)
            self.round_robin_counter = (self.round_robin_counter + 1) % len(
                self.workers
            )

    def get_scheduler_info(self):
        """Get scheduler info from local scheduler."""
        data = self.scheduler_pipe_reader.recv()
        return data

    def event_loop(self):
        """Main event loop for processing requests (only on node 0)."""
        if self.server_args.node_rank != 0:
            return  # Only node 0 runs the event loop

        logger.info("DataParallelController starting event loop")

        while True:
            try:
                # Non-blocking receive
                jax.experimental.multihost_utils.broadcast_one_to_all
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
    """Run the data parallel controller process."""
    if setproctitle is not None:
        setproctitle.setproctitle("sgl-jax::data_parallel_controller")

    # Kill itself when the parent process is dead
    def kill_itself_when_parent_died():
        parent_process_id = os.getppid()

        def monitor_parent():
            while True:
                try:
                    if os.getppid() != parent_process_id:
                        logger.warning("Parent process died. Killing controller.")
                        os.kill(os.getpid(), signal.SIGTERM)
                        break
                except:
                    os.kill(os.getpid(), signal.SIGTERM)
                    break
                time.sleep(1)

        monitor_thread = threading.Thread(target=monitor_parent, daemon=True)
        monitor_thread.start()

    kill_itself_when_parent_died()

    try:
        controller = DataParallelController(server_args, port_args)

        # Wait for local scheduler to be ready
        scheduler_info = controller.get_scheduler_info()

        # Send ready signal with scheduler info
        pipe_writer.send(
            {
                "status": "ready",
                "controller_rank": server_args.node_rank,
                **scheduler_info,
            }
        )

        # Start event loop (only for node 0)
        if server_args.node_rank == 0:
            controller.event_loop()

        # If not node 0, wait for scheduler process
        for proc in controller.scheduler_procs:
            proc.join()
            logger.error(
                f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}"
            )

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DataParallelController hit an exception: {traceback}")
        pipe_writer.send({"status": "error", "error": traceback})
        # Send signal to parent to kill the whole process tree
        parent_process = os.getppid()
        os.kill(parent_process, signal.SIGQUIT)
