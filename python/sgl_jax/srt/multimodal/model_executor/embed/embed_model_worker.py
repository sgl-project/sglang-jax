import jax

from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.schedule_batch import Req


class EmbedModelWorker:
    """Worker shell for the Embed stage."""

    def __init__(
        self, server_args: MultimodalServerArgs, mesh: jax.sharding.Mesh = None, model_class=None
    ):
        self.mesh = mesh
        self.model_runner = None  # VitModelRunner(server_args, self.mesh, model_class=model_class)
        self.initialize()

    def initialize(self):
        pass

    def forward(self, batch: Req, mesh: jax.sharding.Mesh):
        # mock result
        return batch
        # return self.model_runner.forward(batch, mesh)
