import logging
import time

import jax.numpy as jnp

from python.sgl_jax.srt.model_loader.loader import get_model_loader
from sgl_jax.srt.model_executor.base_model_runner import BaseModelRunner

logger = logging.getLogger(__name__)


# DiffusionModelRunner is responsible for running denoising steps within diffusion model inference
class DiffusionModelRunner(BaseModelRunner):
    def __init__(self, model_config):
        super().__init__(model_config)
        self.model_loader = get_model_loader(model_config, None)
        self.transformer_model = None
        self.transformer_2_model = None
        self.solver = None
        self.guidance = None
        # cache-dit state (for delayed mounting and idempotent control)
        self._cache_dit_enabled = False
        self._cached_num_steps = None

        # Additional initialization for diffusion model if needed
        # e.g., setting up noise schedulers, diffusion steps, etc.

    def initialize(self):
        self.model = self.model_loader.load_model(model_config=self.model_config)

        # Any additional initialization specific to diffusion models

    def forward(self, batch, mesh):
        # Implement the diffusion model inference logic here
        # This might include steps like adding noise, denoising, etc.

        time_steps = batch.get("time_steps", None)
        latents = jnp.empty((1, 1))  # Placeholder for latents
        solver_state = None
        solver_state = self.solver.set_time_steps(
            solver_state, num_inference=time_steps, shape=latents.transpose(0, 4, 1, 2, 3).shape
        )
        for step in time_steps:
            logging.info("Starting diffusion step %d", step)
            start_time = time.time()
            # Perform denoising step

            logging.info(
                "Finished diffusion step %d in %.2f seconds", step, time.time() - start_time
            )

        return self.model(**batch)
