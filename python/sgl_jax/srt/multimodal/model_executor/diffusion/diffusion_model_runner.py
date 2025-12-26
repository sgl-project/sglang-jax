from python.sgl_jax.srt.model_loader.loader import get_model_loader
from sgl_jax.srt.model_executor.base_model_runner import BaseModelRunner


class DiffusionModelRunner(BaseModelRunner):
    def __init__(self, model_loader, model_config):
        super().__init__(model_loader, model_config)
        self.model_loader = get_model_loader(model_config, None)

        # Additional initialization for diffusion model if needed
        # e.g., setting up noise schedulers, diffusion steps, etc.

    def initialize(self):
        self.model = self.model_loader.load_model(model_config=self.model_config)

        # Any additional initialization specific to diffusion models

    def forward(self, batch, mesh):
        # Implement the diffusion model inference logic here
        # This might include steps like adding noise, denoising, etc.
        return self.model(**batch)
