from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.model_executor.vae.vae_model_runner import VaeModelRunner
from sgl_jax.srt.server_args import ServerArgs


class VaeModelWorker:
    def __init__(
        self,
        server_args: ServerArgs = None,
        mesh=None,
        model_class=None,
        stage_sub_dir: str | None = None,
    ):
        self.mesh = mesh
        self.model_runner = VaeModelRunner(
            server_args, mesh, model_class=model_class, stage_sub_dir=stage_sub_dir
        )
        # Initialize model here based on model_config

    def forward(self, batch: Req):
        # Implement the vae model inference logic here
        # return batch
        return self.model_runner.forward(batch.latents, "decode")
