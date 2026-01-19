import logging
import time
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from jax import NamedSharding
from jax.sharding import PartitionSpec
from tqdm import tqdm

from sgl_jax.srt.configs.load_config import LoadConfig
from sgl_jax.srt.model_executor.base_model_runner import BaseModelRunner
from sgl_jax.srt.model_loader.loader import JAXModelLoader, get_model_loader
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.models.diffusion_solvers.flow_unipc_multistep_scheduler import (
    FlowUniPCMultistepScheduler,
)
from sgl_jax.srt.multimodal.configs.config_registry import get_diffusion_config
from sgl_jax.srt.utils.jax_utils import device_array

logger = logging.getLogger(__name__)


# DiffusionModelRunner is responsible for running denoising steps within diffusion model inference
class DiffusionModelRunner(BaseModelRunner):
    def __init__(
        self, server_args: MultimodalServerArgs, mesh: jax.sharding.Mesh = None, model_class=None
    ):
        self.server_args = server_args
        self.mesh = mesh
        load_config = LoadConfig(
            load_format=server_args.load_format,
            download_dir=server_args.download_dir,
            sub_dir="transformer",
        )
        self.model_loader = JAXModelLoader(load_config, mesh)

        self.transformer_model = None
        self.transformer_2_model = None
        self.solver = None
        self.guidance = None
        self._cache_dit_enabled = False
        self._cached_num_steps = None
        self.model_class = model_class
        # TODO: load model_config from server_args based on model architecture
        self.model_config = get_diffusion_config(self.server_args.model_path)
        self.model_config.model_path = self.server_args.model_path
        self.model_config.model_class = self.model_class
        # Additional initialization for diffusion model if needed
        # e.g., setting up noise schedulers, diffusion steps, etc.
        self.initialize()

    def initialize(self):
        # self.model = self.model_loader.load_model(model_config=self.model_config)
        self.model_loader = get_model_loader(
            mesh=self.mesh, load_config=LoadConfig(sub_dir="transformer")
        )
        self.model = self.model_loader.load_model(model_config=self.model_config)
        self.solver: FlowUniPCMultistepScheduler = FlowUniPCMultistepScheduler(
            shift=self.model_config.flow_shift
        )
        # self.solver_state = self.solver.create_state()
        # Any additional initialization specific to diffusion models
        self.initialize_jit()

    def initialize_jit(self):
        model_def, model_state = nnx.split(self.model)
        model_state_leaves, model_state_def = jax.tree_util.tree_flatten(model_state)

        @partial(
            jax.jit,
            static_argnames=["model_state_def"],
        )
        def forward_model(
            model_def,
            model_state_def,
            model_state_leaves,
            hidden_states,
            encoder_hidden_states,
            timesteps,
            encoder_hidden_states_image,
            guidance_scale,
        ):
            model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
            model = nnx.merge(model_def, model_state)
            return model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timesteps=timesteps,
                encoder_hidden_states_image=encoder_hidden_states_image,
                guidance_scale=guidance_scale,
            )

        def forward_wrapper(
            hidden_states,
            encoder_hidden_states,
            timesteps,
            encoder_hidden_states_image,
            guidance_scale,
        ):
            return forward_model(
                model_def,
                model_state_def,
                model_state_leaves,
                hidden_states,
                encoder_hidden_states,
                timesteps,
                encoder_hidden_states_image,
                guidance_scale,
            )

        self.jitted_forward = forward_wrapper

    def forward(
        self,
        batch: Req,
        mesh: jax.sharding.Mesh,
        abort_checker: Callable[[], bool] | None = None,
    ) -> bool:
        """Run diffusion inference with optional abort checking.

        Args:
            batch: Request batch containing embeddings and parameters.
            mesh: JAX device mesh for sharding.
            abort_checker: Optional callback that returns True if the request
                should be aborted. Called between diffusion steps.

        Returns:
            True if the request was aborted, False otherwise.
        """
        num_inference_steps = batch.num_inference_steps
        guidance_scale = batch.guidance_scale
        do_classifier_free_guidance = guidance_scale > 1.0

        # Handle prompt embeddings

        prompt_embeds = batch.prompt_embeds
        # Add batch dimension if needed: (L, D) -> (1, L, D)
        if prompt_embeds.ndim == 2:
            prompt_embeds = jnp.expand_dims(prompt_embeds, axis=0)

        # Pad to 512 tokens (do this first for both positive and negative)
        def pad_to_512(embeds):
            if embeds.shape[1] < 512:
                pad_width = 512 - embeds.shape[1]
                return jnp.pad(
                    embeds, ((0, 0), (0, pad_width), (0, 0)), mode="constant", constant_values=0
                )
            return embeds

        # text_embeds shape: (B, 512, D)
        prompt_embeds = pad_to_512(prompt_embeds)
        if do_classifier_free_guidance:
            if batch.negative_prompt_embeds is not None:
                neg_embeds = batch.negative_prompt_embeds
                # Add batch dimension to negative embeds if needed
                if neg_embeds.ndim == 2:
                    neg_embeds = jnp.expand_dims(neg_embeds, axis=0)

                # Pad negative embeds to 512 as well
                neg_embeds = pad_to_512(neg_embeds)

                # Now both are (1, 512, D), concatenate to (2, 512, D)
                prompt_embeds = jnp.concatenate([prompt_embeds, neg_embeds], axis=0)
            else:
                pass

        text_embeds = device_array(
            prompt_embeds, sharding=NamedSharding(self.mesh, PartitionSpec())
        )

        self.prepare_latents(batch)
        latents = device_array(batch.latents, sharding=NamedSharding(self.mesh, PartitionSpec()))
        self.solver.set_timesteps(
            num_inference_steps=num_inference_steps,
            shape=latents.transpose(0, 4, 1, 2, 3).shape,
        )
        self.solver.set_begin_index(0)
        start_time = time.time()

        for step in tqdm(range(num_inference_steps), desc="Diffusion steps"):
            # Check for abort between steps
            if abort_checker is not None and abort_checker():
                logger.info(
                    "Diffusion aborted at step %d/%d for rid=%s",
                    step,
                    num_inference_steps,
                    batch.rid,
                )
                return True  # Aborted

            t_scalar = jnp.array(self.solver.timesteps, dtype=jnp.int32)[step]
            if do_classifier_free_guidance:
                latents = jnp.concatenate([latents] * 2, axis=0)
            # Create timestep batch AFTER latents concat to match batch size
            t_batch = jnp.broadcast_to(t_scalar, (latents.shape[0],))
            # Transpose to channel-first (B, T, H, W, C) -> (B, C, T, H, W) for model
            latents_cf = latents.transpose(0, 4, 1, 2, 3)
            # Perform denoising step
            noise_pred: jax.Array = self.jitted_forward(
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
            latents = self.solver.step(
                model_output=noise_pred,  # already (B, C, T, H, W)
                timestep=t_scalar,
                sample=latents.transpose(0, 4, 1, 2, 3),  # (B, T, H, W, C) -> (B, C, T, H, W)
                return_dict=False,
            )[0]

            latents = latents.transpose(0, 2, 3, 4, 1)  # back to channel-last

        logger.info("Finished diffusion step %d in %.2f seconds", step, time.time() - start_time)
        batch.latents = jax.device_get(latents)
        return False  # Not aborted

    def prepare_latents(self, batch: Req):
        if batch.latents is not None:
            return
        assert batch.width % self.model_config.scale_factor_spatial == 0
        assert batch.height % self.model_config.scale_factor_spatial == 0
        if batch.num_frames is not None:
            assert (batch.num_frames - 1) % self.model_config.scale_factor_temporal == 0
        latents = jax.random.normal(
            jax.random.PRNGKey(46),
            (
                1,
                (
                    (batch.num_frames - 1) // self.model_config.scale_factor_temporal + 1
                    if batch.num_frames is not None
                    else 1
                ),
                batch.width // self.model_config.scale_factor_spatial,
                batch.height // self.model_config.scale_factor_spatial,
                self.model_config.latent_input_dim,
            ),
            dtype=jnp.float32,
        )  # Placeholder for latents
        batch.latents = latents
