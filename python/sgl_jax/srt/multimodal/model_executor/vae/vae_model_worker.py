from sgl_jax.srt.multimodal.model_executor.vae.vae_model_runner import VaeModelRunner
from sgl_jax.srt.server_args import ServerArgs


class VaeModelWorker:
    def __init__(self, server_args: ServerArgs, model_config, mesh):
        self.model_config = model_config
        self.mesh = mesh
        self.model_runner = VaeModelRunner(server_args, model_config, mesh)
        # Initialize model here based on model_config

    def forward(self, batch):
        # Implement the vae model inference logic here
        # return batch
        for i in range(len(batch)):
            self.model_runner.forward(batch[i].x)
