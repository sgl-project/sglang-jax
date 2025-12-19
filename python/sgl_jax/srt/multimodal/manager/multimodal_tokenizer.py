import logging
import signal

import psutil
import setproctitle

from sgl_jax.srt.managers.tokenizer_manager import TokenizerManager
from sgl_jax.srt.server_args import PortArgs, ServerArgs
from sgl_jax.srt.utils import configure_logger, kill_itself_when_parent_died
from sgl_jax.utils import get_exception_traceback

logger = logging.getLogger(__name__)


class MultimodalTokenizer(TokenizerManager):
    pass


def run_multimodal_tokenizer_process(
    server_args: ServerArgs,
    port_args: PortArgs,
):
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang-jax::multimodal_tokenizer")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        tokenizer = MultimodalTokenizer(server_args, port_args)
        tokenizer.event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error("MultimodalTokenizerManager hit an exception: %s", traceback)
        parent_process.send_signal(signal.SIGQUIT)

    return tokenizer
