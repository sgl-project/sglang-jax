import logging
import signal

import imageio
import numpy as np
import psutil
import setproctitle

from sgl_jax.srt.managers.detokenizer_manager import DetokenizerManager
from sgl_jax.srt.multimodal.manager.io_struct import DataType
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.server_args import PortArgs, ServerArgs
from sgl_jax.srt.utils import configure_logger, kill_itself_when_parent_died
from sgl_jax.utils import TypeBasedDispatcher, get_exception_traceback

logger = logging.getLogger(__name__)


class MultimodalDetokenizer(DetokenizerManager):
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        super().__init__(server_args, port_args)
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (Req, self.save_result),
            ]
        )

    def save_result(self, req: Req):
        print("save_result...")
        sample = req.output[0][0]
        if sample.ndim == 3:
            # for images, dim t is missing
            sample = sample.unsqueeze(1)
        # videos = rearrange(sample, "t h w c -> t c h w")
        frames = []
        for x in sample:
            # x = torchvision.utils.make_grid(x, nrow = 6)
            # x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            frames.append((x * 255).astype(np.uint8))

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
