import jax

from sgl_jax.srt.model_executor.base_model_runner import BaseModelRunner
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.schedule_batch import Req


class VitModelRunner(BaseModelRunner):
    """Runner shell for ViT stage execution."""

    def __init__(
        self, server_args: MultimodalServerArgs, mesh: jax.sharding.Mesh = None, model_class=None
    ):
        self.server_args = server_args
        self.mesh = mesh
        self.model_class = model_class
        self.model = None

    def forward(self, batch: Req, mesh: jax.sharding.Mesh):
        return batch
