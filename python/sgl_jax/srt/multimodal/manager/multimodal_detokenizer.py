import logging
import signal

import psutil
import setproctitle

from sgl_jax.srt.managers.detokenizer_manager import DetokenizerManager
from sgl_jax.srt.server_args import PortArgs, ServerArgs
from sgl_jax.srt.utils import (
    configure_logger,
    get_exception_traceback,
    kill_itself_when_parent_died,
)

logger = logging.getLogger(__name__)


class MultimodalDetokenizer(DetokenizerManager):
    pass


def run_multimodal_detokenizer_process(
    server_args: ServerArgs,
    port_args: PortArgs,
):
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang-jax::multimodal_detokenizer")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        manager = MultimodalDetokenizer(server_args, port_args)
        manager.event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error("MultimodalDetokenizerManager hit an exception: %s", traceback)
        parent_process.send_signal(signal.SIGQUIT)
