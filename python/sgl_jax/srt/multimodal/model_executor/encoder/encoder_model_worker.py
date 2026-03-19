import logging

from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.model_executor.encoder.encoder_model_runner import EncoderModelRunner
from sgl_jax.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class EncoderModelWorker:
    def __init__(
        self,
        server_args: ServerArgs = None,
        mesh=None,
        model_class: str | list[str] = None,
        stage_sub_dir: str = None,
        tokenizer: str = None,
    ):
        self.mesh = mesh
        self.model_runner = EncoderModelRunner(
            server_args, mesh, model_class=model_class, stage_sub_dir=stage_sub_dir, tokenizer=tokenizer
        )

    def forward(self, batch: Req):
        # Implement the encoder model inference logic here
        # return batch
        return self.model_runner.forward(batch)