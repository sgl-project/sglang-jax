import jax

from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.model_executor.vit.vit_model_runner import VitModelRunner


class VitModelWorker:
    """Worker shell for the ViT stage."""

    def __init__(
        self, server_args: MultimodalServerArgs, mesh: jax.sharding.Mesh = None, model_class=None
    ):
        self.mesh = mesh
        self.model_runner = VitModelRunner(server_args, self.mesh, model_class=model_class)
        self.initialize()

    def initialize(self):
        pass

    def forward(self, batch: Req, mesh: jax.sharding.Mesh):
        return self.model_runner.forward(batch, mesh)
