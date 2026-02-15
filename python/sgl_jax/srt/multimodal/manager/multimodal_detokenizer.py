import logging
import signal

import imageio
import numpy as np
import psutil
import setproctitle

from sgl_jax.srt.managers.detokenizer_manager import DetokenizerManager
from sgl_jax.srt.managers.io_struct import AbortReq, BatchTokenIDOut, ProfileReqOutput
from sgl_jax.srt.multimodal.manager.io_struct import DataType
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.server_args import PortArgs, ServerArgs
from sgl_jax.srt.utils import configure_logger, kill_itself_when_parent_died
from sgl_jax.utils import TypeBasedDispatcher, get_exception_traceback

logger = logging.getLogger(__name__)


class MultimodalDetokenizer(DetokenizerManager):
    """Collects final multimodal outputs and persists or returns them.

    The `MultimodalDetokenizer` receives completed `Req` objects (typically
    produced by pipeline stages) and is responsible for converting raw
    output arrays into image/video files when requested. It wires a
    `TypeBasedDispatcher` to handle different result types; currently it
    supports saving `Req` outputs via `save_result`.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        """Initialize the detokenizer manager and request dispatcher.

        Args:
            server_args: Global server configuration used for logging and
                behavior control.
            port_args: Port arguments (kept for API compatibility / future
                use).
        """
        super().__init__(server_args, port_args)
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (BatchTokenIDOut, self.handle_batch_token_id_out),
                (Req, self.save_result),
                (AbortReq, self._handle_abort_req),
                (ProfileReqOutput, self._forward_profile_output),
            ]
        )

    def _forward_profile_output(self, output: ProfileReqOutput):
        """Forward ProfileReqOutput through to the tokenizer manager."""
        return output

    def _handle_abort_req(self, abort_req: AbortReq):
        """Forward an AbortReq to the tokenizer manager.

        When the GlobalScheduler aborts a request, it sends an AbortReq to
        the detokenizer. This method simply passes it through to the
        tokenizer manager so it can notify the waiting client coroutine.
        """
        logger.info("Forwarding abort request for rid=%s to tokenizer", abort_req.rid)
        return abort_req

    def save_result(self, req: Req):
        """Process and optionally save the `Req` output.

        Behavior:
        - Validates presence of `req.output` and logs a warning if empty.
        - Normalizes pixel range from model output (assumed in [-1, 1]) to
          uint8 images.
        - If `req.save_output` is true, writes a video (`.mp4`) for
          `DataType.VIDEO` or an image (`.jpg`) otherwise and sets
          `req.output_file_name`.

        Returns the original `Req` wrapped in a list to match dispatcher
        expectations.
        """

        if req.output is None or len(req.output) == 0:
            logger.warning("No output to save for request id: %s", req.rid)
            return [req]

        if req.data_type == DataType.AUDIO:
            return [req]

        sample = req.output[0]
        if sample.ndim == 3:
            # for images, dim t is missing
            sample = sample.unsqueeze(1)
        frames = []
        for x in sample:
            frames.append((np.clip(x / 2 + 0.5, 0, 1) * 255).astype(np.uint8))

        # Save outputs if requested
        if req.save_output:
            # if req.output_file_name:
            if req.data_type == DataType.VIDEO:
                req.output_file_name = req.rid + ".mp4"
                imageio.mimsave(
                    req.output_file_name,
                    frames,
                    fps=req.fps,
                    format=req.data_type.get_default_extension(),
                )
            else:
                req.output_file_name = req.rid + ".jpg"
                imageio.imwrite(req.output_file_name, frames[0])
            logger.info("Saved output to %s", req.output_file_name)
        else:
            logger.info("No output path provided, output not saved")

        return [req]


def run_multimodal_detokenizer_process(
    server_args: ServerArgs,
    port_args: PortArgs,
):
    """Process entrypoint for the multimodal detokenizer.

    Performs process-level setup, constructs a `MultimodalDetokenizer`, and
    runs its event loop. On unhandled exceptions the parent process is
    signaled to terminate.
    """

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
