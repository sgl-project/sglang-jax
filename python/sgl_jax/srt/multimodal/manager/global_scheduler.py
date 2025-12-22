import logging
import signal

import psutil
import setproctitle
import zmq

from sgl_jax.srt.server_args import PortArgs, ServerArgs
from sgl_jax.srt.utils import configure_logger, kill_itself_when_parent_died
from sgl_jax.srt.utils.common_utils import get_zmq_socket
from sgl_jax.utils import get_exception_traceback

logger = logging.getLogger(__name__)


class GlobalScheduler:
    def __init__(self, server_args: ServerArgs, port_args: PortArgs) -> None:
        context = zmq.Context(2)
        self.recv_from_tokenizer = get_zmq_socket(
            context, zmq.PULL, port_args.scheduler_input_ipc_name, False
        )
        self.send_to_detokenizer = get_zmq_socket(
            context, zmq.PUSH, port_args.detokenizer_ipc_name, False
        )

    def recv_request(self):
        recv_reqs = []
        while True:
            try:
                recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
            except zmq.ZMQError:
                break
            recv_reqs.append(recv_req)
        return recv_reqs

    def event_loop(self):
        import time

        while True:
            reqs = self.recv_request()
            print("recv_reqs from tokenizer", reqs)
            time.sleep(3)
            if reqs:
                self.send_to_detokenizer.send_pyobj(reqs)


def run_global_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
):
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
