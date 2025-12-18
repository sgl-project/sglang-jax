import logging
import signal
import time

import psutil
import setproctitle

from sgl_jax.srt.server_args import PortArgs, ServerArgs
from sgl_jax.srt.utils import (
    configure_logger,
    get_exception_traceback,
    kill_itself_when_parent_died,
)

logger = logging.getLogger(__name__)


class GlobalScheduler:
    def __init__(self, server_args: ServerArgs, port_args: PortArgs) -> None:
        pass


def run_global_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
):
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang-jax::global_scheduler")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        manager = GlobalScheduler(server_args, port_args)
        # TODO: Implement event loop
        while True:
            time.sleep(1)
    except Exception:
        traceback = get_exception_traceback()
        logger.error("GlobalScheduler hit an exception: %s", traceback)
        parent_process.send_signal(signal.SIGQUIT)
    return manager
