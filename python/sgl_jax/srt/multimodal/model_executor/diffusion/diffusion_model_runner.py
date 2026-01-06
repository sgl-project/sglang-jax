import logging
import time

import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.configs.load_config import LoadConfig
from sgl_jax.srt.model_executor.base_model_runner import BaseModelRunner
from sgl_jax.srt.model_loader.loader import JAXModelLoader
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.configs.dits.configs import WanModelConfig
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.models.diffusion_solvers.unipc_multistep_scheduler import (
    UniPCMultistepScheduler,
    UniPCMultistepSchedulerState,
)
from sgl_jax.srt.multimodal.models.wan2_1.diffusion.wan2_1_dit import (
    WanTransformer3DModel,
)

logger = logging.getLogger(__name__)


# DiffusionModelRunner is responsible for running denoising steps within diffusion model inference
class DiffusionModelRunner(BaseModelRunner):
    def __init__(self, server_args: MultimodalServerArgs, mesh: jax.sharding.Mesh = None):
        self.mesh = mesh
        load_config = LoadConfig(
            load_format=server_args.load_format,
            download_dir=server_args.download_dir,
        )
        self.model_loader = JAXModelLoader(load_config, mesh)

        self.transformer_model = None
        self.transformer_2_model = None
        self.solver = None
        self.guidance = None
        self._cache_dit_enabled = False
        self._cached_num_steps = None
        # TODO: load model_config from server_args based on model architecture
        self.model_config = WanModelConfig
        # Additional initialization for diffusion model if needed
        # e.g., setting up noise schedulers, diffusion steps, etc.
        self.initialize()

    def initialize(self):
        # self.model = self.model_loader.load_model(model_config=self.model_config)
        rngs = nnx.Rngs(0)
        with jax.set_mesh(self.mesh):
            self.model = WanTransformer3DModel(self.model_config, rngs=rngs)
        self.solver: UniPCMultistepScheduler = UniPCMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            solver_order=2,  # Order 2 for guided sampling
            prediction_type="flow_prediction",
            use_flow_sigmas=True,  # Enable flow-based sigma schedule
            flow_shift=3.0,  # 5.0 for 720P, 3.0 for 480P
            timestep_spacing="linspace",
            predict_x0=True,
            solver_type="bh2",
            lower_order_final=True,
            dtype=jnp.float32,
        )
        self.solver_state = self.solver.create_state()
        # Any additional initialization specific to diffusion models

    def forward(self, batch: Req, mesh: jax.sharding.Mesh):
        # Implement the diffusion model inference logic here
        # This might include steps like adding noise, denoising, etc.

        num_inference_steps = batch.num_inference_steps
        # time_steps = batch.timesteps
        text_embeds = batch.prompt_embeds
        print(f"{text_embeds=}")
        guidance_scale = batch.guidance_scale
        do_classifier_free_guidance = guidance_scale > 1.0
        batch.latents = jax.random.normal(
            jax.random.PRNGKey(46),
            (
                1,
                self.model_config.num_frames,
                self.model_config.latent_size[0],
                self.model_config.latent_size[1],
                self.model_config.latent_input_dim,
            ),
            dtype=jnp.float32,
        )  # Placeholder for latents
        latents = batch.latents
        solver_state: UniPCMultistepSchedulerState = self.solver.set_timesteps(
            self.solver_state,
            num_inference_steps=num_inference_steps,
            shape=latents.transpose(0, 4, 1, 2, 3).shape,
        )
        for step in range(num_inference_steps):
            start_time = time.time()
            logging.info("Starting diffusion step %d", step)
            t_scalar = jnp.array(solver_state.timesteps, dtype=jnp.int32)[step]
            t_batch = jnp.broadcast_to(t_scalar, latents.shape[0])
            if do_classifier_free_guidance:
                latents = jnp.concatenate([latents] * 2)
            # Transpose to channel-first (B, T, H, W, C) -> (B, C, T, H, W) for model
            latents_cf = latents.transpose(0, 4, 1, 2, 3)
            # Perform denoising step
            noise_pred: jax.Array = self.model(
                hidden_states=latents_cf,
                encoder_hidden_states=text_embeds,
                timesteps=t_batch,
                encoder_hidden_states_image=None,
                guidance_scale=None,
            )
            if do_classifier_free_guidance:
                bsz = latents.shape[0] // 2
                noise_uncond = noise_pred[bsz:]
                noise_pred = noise_pred[:bsz]
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
                latents = latents[:bsz]
            # noise_pred is already channel-first (B, C, T, H, W) from model
            # latents is channel-last (B, T, H, W, C), need to transpose for solver
            latents, solver_state = self.solver.step(
                solver_state,
                noise_pred,  # already (B, C, T, H, W)
                t_scalar,
                latents.transpose(0, 4, 1, 2, 3),  # (B, T, H, W, C) -> (B, C, T, H, W)
            )

            latents = latents.transpose(0, 2, 3, 4, 1)  # back to channel-last
            logging.info(
                "Finished diffusion step %d in %.2f seconds", step, time.time() - start_time
            )
        batch.latents = latents
        return batch
