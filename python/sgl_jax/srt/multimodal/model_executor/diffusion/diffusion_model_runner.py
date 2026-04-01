import logging
import time
from collections.abc import Callable
from functools import partial

import jax
import numpy as np
import jax.numpy as jnp
from flax import nnx
from jax import NamedSharding
from jax.sharding import PartitionSpec
from tqdm import tqdm

from sgl_jax.srt.configs.load_config import LoadConfig
from sgl_jax.srt.model_executor.base_model_runner import BaseModelRunner
from sgl_jax.srt.model_loader.loader import JAXModelLoader, get_model_loader
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.configs.config_registry import (
    get_diffusion_config,
    get_vae_config,
)
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.models.diffusion_solvers.flow_match_euler_discrete_scheduler import (
    FlowMatchEulerDiscreteScheduler,
)
from sgl_jax.srt.multimodal.models.diffusion_solvers.flow_unipc_multistep_scheduler import (
    FlowUniPCMultistepScheduler,
)
from sgl_jax.srt.utils.jax_utils import device_array

logger = logging.getLogger(__name__)


# DiffusionModelRunner is responsible for running denoising steps within diffusion model inference
class DiffusionModelRunner(BaseModelRunner):
    def __init__(
        self,
        server_args: MultimodalServerArgs,
        mesh: jax.sharding.Mesh = None,
        model_class=None,
        stage_sub_dir: str | None = None,
        scheduler: str | None = None,
    ):
        self.server_args = server_args
        self.mesh = mesh
        self.scheduler = scheduler
        load_sub_dir = "transformer" if stage_sub_dir is None else stage_sub_dir
        if load_sub_dir == "":
            load_sub_dir = None
        self.load_sub_dir = load_sub_dir
        load_config = LoadConfig(
            load_format=server_args.load_format,
            download_dir=server_args.download_dir,
            sub_dir=load_sub_dir,
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

        # Load VAE config for FLUX (provides scale_factor_spatial, scaling_factor, etc.)
        if self.is_flux():
            self.vae_config = get_vae_config(self.server_args.model_path)

        self.initialize()

    def is_flux(self) -> bool:
        model_name = getattr(self.model_class, "__name__", str(self.model_class))
        return "Flux" in model_name

    def initialize(self):
        load_sub_dir = self.load_sub_dir
        self.model_loader = get_model_loader(
            mesh=self.mesh, load_config=LoadConfig(sub_dir=load_sub_dir)
        )
        self.model = self.model_loader.load_model(model_config=self.model_config)
        if self.scheduler == "FlowUniPCMultistepScheduler":
            self.solver = FlowUniPCMultistepScheduler(
                shift=self.model_config.flow_shift
            )
        elif self.scheduler == "FlowMatchEulerDiscreteScheduler":
            # FLUX uses dynamic shifting (no flow_shift needed)
            self.solver = FlowMatchEulerDiscreteScheduler(
                shift=1.0,
                use_dynamic_shifting=True,
                base_shift=0.5,
                max_shift=1.15,
            )
        else:
            raise ValueError(f"Unsupported diffusion scheduler: {self.scheduler}")
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
            inputs,
        ):
            model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
            model = nnx.merge(model_def, model_state)
            return model(**inputs)

        def forward_wrapper(**kwargs):
            return forward_model(
                model_def,
                model_state_def,
                model_state_leaves,
                kwargs,
            )

        self.jitted_forward = forward_wrapper

    # ── FLUX helper methods ──

    @staticmethod
    def _pack_latents_flux(latents, batch_size, num_channels, height, width):
        """[B, C, H, W] -> [B, H/2*W/2, C*4]"""
        latents = latents.reshape(batch_size, num_channels, height // 2, 2, width // 2, 2)
        latents = jnp.transpose(latents, (0, 2, 4, 1, 3, 5))
        return latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)

    @staticmethod
    def _unpack_latents_flux(latents, height, width, in_channels):
        """[B, seq, C*4] -> [B, C, H, W]"""
        batch_size = latents.shape[0]
        num_channels = in_channels // 4  # 16
        latents = latents.reshape(batch_size, height // 2, width // 2, num_channels, 2, 2)
        latents = jnp.transpose(latents, (0, 3, 1, 4, 2, 5))
        return latents.reshape(batch_size, num_channels, height, width)

    @staticmethod
    def _prepare_latent_image_ids(height, width, vae_scale_factor):
        """Prepare img_ids for FLUX RoPE. Returns [H/2*W/2, 3]."""
        h = height // (vae_scale_factor * 2)
        w = width // (vae_scale_factor * 2)
        img_ids = jnp.zeros((int(h), int(w), 3))
        img_ids = img_ids.at[..., 1].add(jnp.arange(int(h))[:, None])
        img_ids = img_ids.at[..., 2].add(jnp.arange(int(w))[None, :])
        return img_ids.reshape(int(h) * int(w), 3)

    @staticmethod
    def _calculate_mu(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15):
        """Port of GPU calculate_shift() from flux.py pipeline."""
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        return image_seq_len * m + b

    # ── Forward dispatch ──

    def forward(
        self,
        batch: Req,
        mesh: jax.sharding.Mesh,
        abort_checker: Callable[[], bool] | None = None,
        step_callback: Callable[[], None] | None = None,
    ) -> bool:
        if self.is_flux():
            return self._forward_flux(batch, mesh, abort_checker, step_callback)
        else:
            return self._forward_wan(batch, mesh, abort_checker, step_callback)

    # ── FLUX forward ──

    def _forward_flux(
        self,
        batch: Req,
        mesh: jax.sharding.Mesh,
        abort_checker: Callable[[], bool] | None = None,
        step_callback: Callable[[], None] | None = None,
    ) -> bool:
        num_inference_steps = batch.num_inference_steps
        vae_scale_factor = self.vae_config.scale_factor_spatial  # 8
        in_channels = self.model_config.in_channels  # 64

        # 1. Extract encoder outputs
        prompt_embeds = batch.prompt_embeds  # T5: [B, seq_len, 4096]
        if isinstance(prompt_embeds, list):
            prompt_embeds = prompt_embeds[0]
        if prompt_embeds.ndim == 2:
            prompt_embeds = jnp.expand_dims(prompt_embeds, 0)

        pooled_projections = batch.pooled_embeds[0]  # CLIP: [B, 768]
        if pooled_projections.ndim == 1:
            pooled_projections = jnp.expand_dims(pooled_projections, 0)

        # 2. Prepare latents [B, C//4, H', W']
        height_latent = 2 * (batch.height // (vae_scale_factor * 2))
        width_latent = 2 * (batch.width // (vae_scale_factor * 2))
        num_channels = in_channels // 4  # 16
        latents = jax.random.normal(
            jax.random.PRNGKey(batch.seed or 42),
            (1, num_channels, height_latent, width_latent),
            dtype=jnp.float32,
        )

        # 3. Pack latents: [B, C//4, H', W'] -> [B, H'/2*W'/2, C]
        latents = self._pack_latents_flux(latents, 1, num_channels, height_latent, width_latent)

        # 4. Prepare position IDs
        img_ids = self._prepare_latent_image_ids(batch.height, batch.width, vae_scale_factor)
        txt_ids = jnp.zeros((prompt_embeds.shape[1], 3))

        # 5. Compute mu for dynamic timestep shifting
        image_seq_len = (batch.height // (vae_scale_factor * 2)) * (batch.width // (vae_scale_factor * 2))
        mu = self._calculate_mu(image_seq_len)

        # 6. Set timesteps with dynamic shifting (match GPU: sigmas = linspace(1, 1/N, N))
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps).tolist()
        self.solver.set_timesteps(sigmas=sigmas, mu=mu)

        # 7. Build guidance tensor (embedded_cfg_scale=3.5 for FLUX.1-dev)
        embedded_cfg_scale = 3.5
        guidance = jnp.full((1,), embedded_cfg_scale)

        # 8. Move to device
        latents = device_array(latents, sharding=NamedSharding(self.mesh, PartitionSpec()))
        prompt_embeds = device_array(prompt_embeds, sharding=NamedSharding(self.mesh, PartitionSpec()))
        pooled_projections = device_array(pooled_projections, sharding=NamedSharding(self.mesh, PartitionSpec()))
        img_ids = device_array(img_ids, sharding=NamedSharding(self.mesh, PartitionSpec()))
        txt_ids = device_array(txt_ids, sharding=NamedSharding(self.mesh, PartitionSpec()))
        guidance = device_array(guidance, sharding=NamedSharding(self.mesh, PartitionSpec()))

        # 9. Denoising loop
        start_time = time.time()
        for step in tqdm(range(num_inference_steps), desc="FLUX diffusion"):
            if abort_checker is not None and abort_checker():
                logger.info(
                    "Diffusion aborted at step %d/%d for rid=%s",
                    step, num_inference_steps, batch.rid,
                )
                return True

            t = jnp.array(self.solver.timesteps[step])
            # Model internally multiplies by 1000, so pass sigma [0,1] not timestep [0,1000]
            t_for_model = jnp.broadcast_to(t / 1000.0, (latents.shape[0],))

            noise_pred = self.jitted_forward(
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_projections,
                timestep=t_for_model,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
            )

            latents = self.solver.step(
                model_output=noise_pred,
                timestep=t,
                sample=latents,
                return_dict=False,
            )[0]

            if step_callback is not None:
                step_callback()

        logger.info("Finished FLUX diffusion in %.2f seconds", time.time() - start_time)

        # 10. Unpack latents: [B, seq, C] -> [B, C//4, H', W']
        latents = self._unpack_latents_flux(latents, height_latent, width_latent, in_channels)

        # 11. Store 4D result [B, C, H', W']
        batch.latents = jax.device_get(latents)
        return False

    # ── Wan forward (original logic) ──

    def _forward_wan(
        self,
        batch: Req,
        mesh: jax.sharding.Mesh,
        abort_checker: Callable[[], bool] | None = None,
        step_callback: Callable[[], None] | None = None,
    ) -> bool:
        num_inference_steps = batch.num_inference_steps
        guidance_scale = batch.guidance_scale
        do_classifier_free_guidance = guidance_scale > 1.0

        # Handle prompt embeddings
        prompt_embeds = batch.prompt_embeds
        if prompt_embeds.ndim == 2:
            prompt_embeds = jnp.expand_dims(prompt_embeds, axis=0)

        def pad_to_512(embeds):
            if embeds.shape[1] < 512:
                pad_width = 512 - embeds.shape[1]
                return jnp.pad(
                    embeds, ((0, 0), (0, pad_width), (0, 0)), mode="constant", constant_values=0
                )
            return embeds

        prompt_embeds = pad_to_512(prompt_embeds)
        if do_classifier_free_guidance:
            if batch.negative_prompt_embeds is not None:
                neg_embeds = batch.negative_prompt_embeds
                if neg_embeds.ndim == 2:
                    neg_embeds = jnp.expand_dims(neg_embeds, axis=0)
                neg_embeds = pad_to_512(neg_embeds)
                prompt_embeds = jnp.concatenate([prompt_embeds, neg_embeds], axis=0)

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
        import jax._src.test_util as jtu

        for step in tqdm(range(num_inference_steps), desc="Diffusion steps"):
            if abort_checker is not None and abort_checker():
                logger.info(
                    "Diffusion aborted at step %d/%d for rid=%s",
                    step, num_inference_steps, batch.rid,
                )
                return True

            jax.profiler.StepTraceAnnotation("diffusion_step", step_num=step)
            t_scalar = jnp.array(self.solver.timesteps, dtype=jnp.int32)[step]
            if do_classifier_free_guidance:
                latents = jnp.concatenate([latents] * 2, axis=0)
            t_batch = jnp.broadcast_to(t_scalar, (latents.shape[0],))
            latents_cf = latents.transpose(0, 4, 1, 2, 3)
            with jtu.count_pjit_cpp_cache_miss() as count:
                noise_pred: jax.Array = self.jitted_forward(
                    hidden_states=latents_cf,
                    encoder_hidden_states=text_embeds,
                    timesteps=t_batch,
                    encoder_hidden_states_image=None,
                    guidance_scale=None,
                )
                if count() > 0:
                    logger.info("diffusion cache miss count: %d", count())
            if do_classifier_free_guidance:
                bsz = latents.shape[0] // 2
                noise_uncond = noise_pred[bsz:]
                noise_pred = noise_pred[:bsz]
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
                latents = latents[:bsz]
            latents = self.solver.step(
                model_output=noise_pred,
                timestep=t_scalar,
                sample=latents.transpose(0, 4, 1, 2, 3),
                return_dict=False,
            )[0]
            latents = latents.transpose(0, 2, 3, 4, 1)
            debug_data[f"step{step}_t"] = float(t)
            debug_data[f"step{step}_noise_pred"] = jax.device_get(noise_pred)
            debug_data[f"step{step}_latents"] = jax.device_get(latents)

            if step_callback is not None:
                step_callback()

        logger.info("Finished diffusion step %d in %.2f seconds", step, time.time() - start_time)
        batch.latents = jax.device_get(latents)
        return False

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
                batch.height // self.model_config.scale_factor_spatial,
                batch.width // self.model_config.scale_factor_spatial,
                self.model_config.latent_input_dim,
            ),
            dtype=jnp.float32,
        )
        batch.latents = latents

    def mock_data(self, batch: Req) -> Req:
        """Generate mock diffusion output for stage-by-stage debugging."""
        if self.is_flux():
            vae_scale_factor = self.vae_config.scale_factor_spatial  # 8
            in_channels = self.model_config.in_channels  # 64
            height_latent = 2 * (batch.height // (vae_scale_factor * 2))
            width_latent = 2 * (batch.width // (vae_scale_factor * 2))
            num_channels = in_channels // 4  # 16
            key = jax.random.PRNGKey(42)
            batch.latents = jax.random.normal(key, (1, num_channels, height_latent, width_latent))
        else:
            key = jax.random.PRNGKey(42)
            batch.latents = jax.random.normal(key, (
                1,
                (batch.num_frames - 1) // self.model_config.scale_factor_temporal + 1,
                batch.height // self.model_config.scale_factor_spatial,
                batch.width // self.model_config.scale_factor_spatial,
                self.model_config.latent_input_dim,
            ))
        return batch
