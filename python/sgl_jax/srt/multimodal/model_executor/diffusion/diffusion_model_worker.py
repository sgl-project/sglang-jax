from collections.abc import Callable

import jax

from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.model_executor.diffusion.diffusion_model_runner import (
    DiffusionModelRunner,
)


class DiffusionModelWorker:
    def __init__(
        self,
        server_args: MultimodalServerArgs,
        mesh: jax.sharding.Mesh = None,
        model_class=None,
        stage_sub_dir: str | None = None,
    ):
        self.mesh = mesh
        self.model_runner = DiffusionModelRunner(
            server_args, self.mesh, model_class=model_class, stage_sub_dir=stage_sub_dir
        )
        self.initialize()

    def initialize(self):
        pass
        # self.model_loader.load_model()
        # init cache here if needed
        # init different attention backend if needed

    def forward(
        self,
        batch: Req,
        mesh: jax.sharding.Mesh,
        abort_checker: Callable[[], bool] | None = None,
    ) -> bool:
        """Generate video from text embeddings using the diffusion model.

        Args:
            batch: Request batch containing text embeddings and parameters.
            mesh: JAX device mesh for sharding.
            abort_checker: Optional callback that returns True if the request
                should be aborted. Called between diffusion steps.

        Returns:
            True if the request was aborted, False otherwise.
        """
        return self.model_runner.forward(batch, mesh, abort_checker=abort_checker)
