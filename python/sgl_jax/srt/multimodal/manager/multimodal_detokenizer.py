import logging
import os
import signal
import subprocess

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
                # Default to 25fps (LTX-2 reference) when not specified
                video_fps = req.fps if req.fps is not None else 25
                imageio.mimsave(
                    req.output_file_name,
                    frames,
                    fps=video_fps,
                    format=req.data_type.get_default_extension(),
                    quality=8,
                )
            else:
                req.output_file_name = req.rid + ".jpg"
                imageio.imwrite(req.output_file_name, frames[0])
            logger.info("Saved output to %s", req.output_file_name)

            # Mux audio into video if a WAV file was produced by the VAE scheduler
            if req.data_type == DataType.VIDEO:
                wav_path = os.path.join("outputs", req.rid + ".wav")
                if os.path.exists(wav_path):
                    self._mux_audio(req.output_file_name, wav_path)
        else:
            logger.info("No output path provided, output not saved")

        return [req]

    def _mux_audio(self, video_path: str, wav_path: str):
        """Mux a WAV audio track into an existing MP4 video file."""
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            logger.warning("imageio_ffmpeg not available, skipping audio mux")
            return

        muxed_path = video_path + ".tmp.mp4"
        cmd = [
            ffmpeg_exe, "-y",
            "-i", video_path,
            "-i", wav_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            muxed_path,
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=60)
            os.replace(muxed_path, video_path)
            os.remove(wav_path)
            logger.info("Muxed audio into %s", video_path)
        except Exception as e:
            logger.warning("Failed to mux audio: %s", e)
            if os.path.exists(muxed_path):
                os.remove(muxed_path)


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
