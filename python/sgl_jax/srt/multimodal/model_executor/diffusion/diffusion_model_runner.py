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
from sgl_jax.srt.multimodal.configs.config_registry import get_diffusion_config
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.models.diffusion_solvers.flow_unipc_multistep_scheduler import (
    FlowUniPCMultistepScheduler,
)
from sgl_jax.srt.multimodal.models.diffusion_solvers.euler_scheduler import EulerScheduler
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
    ):
        self.server_args = server_args
        self.mesh = mesh
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
        if self.model_class is None:
            if hasattr(self.model_config, "scheduler_type") and self.model_config.scheduler_type == "EulerScheduler":
                from sgl_jax.srt.multimodal.models.ltx2 import LTX2Transformer3DModel
                self.model_config.model_class = LTX2Transformer3DModel
            else:
                from sgl_jax.srt.multimodal.models.wan.diffusion.wan_dit import WanModel
                self.model_config.model_class = WanModel
        # Additional initialization for diffusion model if needed
        # e.g., setting up noise schedulers, diffusion steps, etc.
        self.initialize()

    def initialize(self):
        load_sub_dir = self.load_sub_dir
        self.model_loader = get_model_loader(
            mesh=self.mesh, load_config=LoadConfig(sub_dir=load_sub_dir)
        )
        self.model = self.model_loader.load_model(model_config=self.model_config)
        use_dynamic_shifting = getattr(self.model_config, "use_dynamic_shifting", False)
        scheduler_type = getattr(self.model_config, "scheduler_type", "FlowUniPCMultistepScheduler")
        
        if scheduler_type == "EulerScheduler":
            self.solver = EulerScheduler(
                base_shift=getattr(self.model_config, "base_shift", 0.95),
                max_shift=getattr(self.model_config, "max_shift", 2.05),
                stretch=getattr(self.model_config, "stretch", True),
                terminal=getattr(self.model_config, "terminal", 0.1),
            )
        else:
            self.solver = FlowUniPCMultistepScheduler(
                shift=self.model_config.flow_shift if not use_dynamic_shifting else None,
                use_dynamic_shifting=use_dynamic_shifting,
            )
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
            attention_mask,
            stg_mask,
            audio_latent,
            audio_context,
            modality_mask=None,
        ):
            model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
            model = nnx.merge(model_def, model_state)
            # Both WanModel and LTX2Transformer3DModel accept hidden_states,
            # encoder_hidden_states, timesteps as common args, plus **kwargs
            # for model-specific extras (attention_mask, stg_mask, etc.)
            return model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timesteps=timesteps,
                encoder_hidden_states_image=encoder_hidden_states_image,
                guidance_scale=guidance_scale,
                attention_mask=attention_mask,
                stg_mask=stg_mask,
                modality_mask=modality_mask,
                audio_latent=audio_latent,
                audio_context=audio_context,
            )

        def forward_wrapper(
            hidden_states,
            encoder_hidden_states,
            timesteps,
            encoder_hidden_states_image=None,
            guidance_scale=None,
            attention_mask=None,
            stg_mask=None,
            audio_latent=None,
            audio_context=None,
            modality_mask=None,
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
                attention_mask,
                stg_mask,
                audio_latent,
                audio_context,
                modality_mask,
            )

        self.jitted_forward = forward_wrapper

    def forward(
        self,
        batch: Req,
        mesh: jax.sharding.Mesh,
        abort_checker: Callable[[], bool] | None = None,
        step_callback: Callable[[], None] | None = None,
    ) -> bool:
        """Run diffusion inference with optional abort checking.

        Args:
            batch: Request batch containing embeddings and parameters.
            mesh: JAX device mesh for sharding.
            abort_checker: Optional callback that returns True if the request
                should be aborted. Called between diffusion steps.
            step_callback: Optional callback invoked after each denoising step
                completes, used for profiling step counting.

        Returns:
            True if the request was aborted, False otherwise.
        """
        num_inference_steps = batch.num_inference_steps
        guidance_scale = batch.guidance_scale
        stg_scale = batch.stg_scale
        do_spatio_temporal_guidance = stg_scale > 0.0 and getattr(self.model_config, "stg_mode", False)
        do_classifier_free_guidance = guidance_scale > 1.0 and not do_spatio_temporal_guidance
        # LTX-2 uses Euler scheduler and requires guidance in x0 space with variance rescaling.
        # WAN 2.1 uses FlowUniPC which handles x0 conversion internally, so velocity-space guidance is correct.
        use_x0_guidance = isinstance(self.solver, EulerScheduler)

        # Modality guidance: 4th forward pass that skips A2V/V2A cross-attention.
        # Reference default: modality_scale=3.0 for both video and audio.
        # Only enabled when STG is active (requires multi-pass batching).
        modality_scale = getattr(batch, "modality_scale", 3.0)
        do_modality_guidance = (
            do_spatio_temporal_guidance
            and modality_scale != 1.0
            and getattr(self.model_config, "is_audio_enabled", False)
        )
        audio_modality_scale = getattr(batch, "audio_modality_scale", 3.0)

        # Audio uses separate guidance parameters (PyTorch: DEFAULT_AUDIO_GUIDER_PARAMS)
        audio_guidance_scale = getattr(batch, "audio_guidance_scale", 7.0)
        audio_stg_scale = getattr(batch, "audio_stg_scale", 1.0)
        audio_rescale_scale = getattr(batch, "audio_rescale_scale", 0.7)

        # Handle prompt embeddings

        prompt_embeds = batch.prompt_embeds
        # Add batch dimension if needed: (L, D) -> (1, L, D)
        if prompt_embeds.ndim == 2:
            prompt_embeds = jnp.expand_dims(prompt_embeds, axis=0)

        # Pad text embeddings to the model's expected sequence length
        pad_dim = getattr(self.model_config, "max_sequence_length", 512)
        def pad_to_max_len(embeds):
            if embeds.shape[1] < pad_dim:
                pad_width = pad_dim - embeds.shape[1]
                return jnp.pad(
                    embeds, ((0, 0), (0, pad_width), (0, 0)), mode="constant", constant_values=0
                )
            return embeds

        # Handle audio text embeddings (same padding/concatenation as video)
        has_audio = (
            getattr(self.model_config, "is_audio_enabled", False)
            and batch.audio_prompt_embeds is not None
        )
        audio_embeds_raw = None
        if has_audio:
            audio_embeds_raw = batch.audio_prompt_embeds
            if audio_embeds_raw.ndim == 2:
                audio_embeds_raw = jnp.expand_dims(audio_embeds_raw, axis=0)
            audio_embeds_raw = pad_to_max_len(audio_embeds_raw)

        # text_embeds shape: (B, max_len, D)
        prompt_embeds = pad_to_max_len(prompt_embeds)
        if do_spatio_temporal_guidance:
            if batch.negative_prompt_embeds is not None:
                neg_embeds = batch.negative_prompt_embeds
                if neg_embeds.ndim == 2:
                    neg_embeds = jnp.expand_dims(neg_embeds, axis=0)
                neg_embeds = pad_to_max_len(neg_embeds)
                if do_modality_guidance:
                    # STG+modality: [pos, neg, pos(ptb), pos(mod)]
                    prompt_embeds = jnp.concatenate(
                        [prompt_embeds, neg_embeds, prompt_embeds, prompt_embeds], axis=0
                    )
                else:
                    # STG only: [pos, neg, pos(ptb)]
                    prompt_embeds = jnp.concatenate([prompt_embeds, neg_embeds, prompt_embeds], axis=0)
                if has_audio:
                    audio_neg = batch.audio_negative_prompt_embeds
                    if audio_neg.ndim == 2:
                        audio_neg = jnp.expand_dims(audio_neg, axis=0)
                    audio_neg = pad_to_max_len(audio_neg)
                    if do_modality_guidance:
                        audio_embeds_raw = jnp.concatenate(
                            [audio_embeds_raw, audio_neg, audio_embeds_raw, audio_embeds_raw], axis=0
                        )
                    else:
                        audio_embeds_raw = jnp.concatenate(
                            [audio_embeds_raw, audio_neg, audio_embeds_raw], axis=0
                        )
        elif do_classifier_free_guidance:
            if batch.negative_prompt_embeds is not None:
                neg_embeds = batch.negative_prompt_embeds
                # Add batch dimension to negative embeds if needed
                if neg_embeds.ndim == 2:
                    neg_embeds = jnp.expand_dims(neg_embeds, axis=0)

                # Pad negative embeds to max_sequence_length as well
                neg_embeds = pad_to_max_len(neg_embeds)

                # Now both are (1, max_len, D), concatenate to (2, max_len, D)
                prompt_embeds = jnp.concatenate([prompt_embeds, neg_embeds], axis=0)
                if has_audio:
                    audio_neg = batch.audio_negative_prompt_embeds
                    if audio_neg.ndim == 2:
                        audio_neg = jnp.expand_dims(audio_neg, axis=0)
                    audio_neg = pad_to_max_len(audio_neg)
                    audio_embeds_raw = jnp.concatenate([audio_embeds_raw, audio_neg], axis=0)

        text_embeds = device_array(
            prompt_embeds, sharding=NamedSharding(self.mesh, PartitionSpec())
        )
        audio_text_embeds = None
        if has_audio:
            audio_text_embeds = device_array(
                audio_embeds_raw, sharding=NamedSharding(self.mesh, PartitionSpec())
            )

        self.prepare_latents(batch)
        latents = device_array(batch.latents, sharding=NamedSharding(self.mesh, PartitionSpec()))

        # Prepare audio latents if audio is enabled
        audio_latents = None
        if has_audio:
            self._prepare_audio_latents(batch)
            audio_latents = device_array(
                batch.audio_latents, sharding=NamedSharding(self.mesh, PartitionSpec())
            )

        # For dynamic shifting (LTX-2), mu is computed from the token count of
        # the latent shape. The scheduler's set_timesteps computes it automatically
        # when mu=None.
        self.solver.set_timesteps(
            num_inference_steps=num_inference_steps,
            shape=latents.transpose(0, 4, 1, 2, 3).shape,
            mu=None,
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

            jax.profiler.StepTraceAnnotation("diffusion_step", step_num=step)
            # EulerScheduler: pass raw sigma (float 0-1) — the DiT internally
            # multiplies by timestep_scale_multiplier (1000).
            # FlowUniPC: pass integer timestep (0-999) as expected by WAN.
            if use_x0_guidance:
                t_scalar = self.solver._sigmas[step]
            else:
                t_scalar = jnp.array(self.solver.timesteps, dtype=jnp.int32)[step]
            if do_modality_guidance:
                latents_in = jnp.concatenate([latents] * 4, axis=0)
            elif do_spatio_temporal_guidance:
                latents_in = jnp.concatenate([latents] * 3, axis=0)
            elif do_classifier_free_guidance:
                latents_in = jnp.concatenate([latents] * 2, axis=0)
            else:
                latents_in = latents
            # Create timestep batch AFTER latents concat to match batch size
            t_batch = jnp.broadcast_to(t_scalar, (latents_in.shape[0],))
            # Transpose to channel-first (B, T, H, W, C) -> (B, C, T, H, W) for model
            latents_cf = latents_in.transpose(0, 4, 1, 2, 3)
            # Perform denoising step
            modality_mask_val = None
            if do_modality_guidance:
                bsz = latents.shape[0]
                # STG mask: [1, 1, 0, 1] — only ptb (3rd) gets self-attn zeroed
                stg_mask = jnp.concatenate([
                    jnp.ones((2 * bsz, 1, 1), dtype=jnp.float32),
                    jnp.zeros((bsz, 1, 1), dtype=jnp.float32),
                    jnp.ones((bsz, 1, 1), dtype=jnp.float32),
                ], axis=0)
                # Modality mask: [1, 1, 1, 0] — only mod (4th) gets cross-modal zeroed
                modality_mask_val = jnp.concatenate([
                    jnp.ones((3 * bsz, 1, 1), dtype=jnp.float32),
                    jnp.zeros((bsz, 1, 1), dtype=jnp.float32),
                ], axis=0)
            elif do_spatio_temporal_guidance:
                bsz = latents.shape[0]
                stg_mask = jnp.concatenate([
                    jnp.ones((2 * bsz, 1, 1), dtype=jnp.float32),
                    jnp.zeros((bsz, 1, 1), dtype=jnp.float32)
                ], axis=0)
            else:
                stg_mask = None
            # LTX-2 NEVER passes attention masks to the transformer.
            # The model expects all max_text_len tokens without masking.
            attention_mask = None

            # Prepare audio latent input (same CFG/STG/modality batching as video)
            audio_latent_in = None
            if audio_latents is not None:
                if do_modality_guidance:
                    audio_latent_in = jnp.concatenate([audio_latents] * 4, axis=0)
                elif do_spatio_temporal_guidance:
                    audio_latent_in = jnp.concatenate([audio_latents] * 3, axis=0)
                elif do_classifier_free_guidance:
                    audio_latent_in = jnp.concatenate([audio_latents] * 2, axis=0)
                else:
                    audio_latent_in = audio_latents

            model_output = self.jitted_forward(
                hidden_states=latents_cf,
                encoder_hidden_states=text_embeds,
                timesteps=t_batch,
                encoder_hidden_states_image=None,
                guidance_scale=None,
                attention_mask=attention_mask,
                stg_mask=stg_mask,
                audio_latent=audio_latent_in,
                audio_context=audio_text_embeds,
                modality_mask=modality_mask_val,
            )

            # Unpack video and audio predictions
            if isinstance(model_output, tuple):
                noise_pred, audio_noise_pred = model_output
            else:
                noise_pred = model_output
                audio_noise_pred = None

            if do_modality_guidance:
                # 4-way split: [cond, uncond, ptb, mod]
                bsz = latents_in.shape[0] // 4
                v_cond = noise_pred[:bsz]
                v_uncond = noise_pred[bsz:2*bsz]
                v_ptb = noise_pred[2*bsz:3*bsz]
                v_mod = noise_pred[3*bsz:]

                if use_x0_guidance:
                    sample_cf = latents.transpose(0, 4, 1, 2, 3).astype(jnp.float32)
                    sigma = t_scalar
                    x0_cond = sample_cf - v_cond.astype(jnp.float32) * sigma
                    x0_uncond = sample_cf - v_uncond.astype(jnp.float32) * sigma
                    x0_ptb = sample_cf - v_ptb.astype(jnp.float32) * sigma
                    x0_mod = sample_cf - v_mod.astype(jnp.float32) * sigma

                    # Reference MultiModalGuider.calculate:
                    # pred = cond + (cfg-1)*(cond-uncond) + stg*(cond-ptb) + (mod-1)*(cond-mod)
                    x0_guided = (
                        x0_cond
                        + (guidance_scale - 1) * (x0_cond - x0_uncond)
                        + stg_scale * (x0_cond - x0_ptb)
                        + (modality_scale - 1) * (x0_cond - x0_mod)
                    )

                    rescale_scale = batch.rescale_scale
                    factor = x0_cond.std() / (x0_guided.std() + 1e-8)
                    factor = rescale_scale * factor + (1 - rescale_scale)
                    x0_guided = x0_guided * factor

                    noise_pred = ((sample_cf - x0_guided) / (sigma + 1e-8)).astype(noise_pred.dtype)
                else:
                    noise_pred = (
                        v_uncond
                        + guidance_scale * (v_cond - v_uncond)
                        + stg_scale * (v_cond - v_ptb)
                        + (modality_scale - 1) * (v_cond - v_mod)
                    )

            elif do_spatio_temporal_guidance:
                bsz = latents_in.shape[0] // 3
                v_cond = noise_pred[:bsz]
                v_uncond = noise_pred[bsz:2*bsz]
                v_ptb = noise_pred[2*bsz:]

                if use_x0_guidance:
                    # LTX-2: guidance in x0 space (matching reference guiders.py)
                    sample_cf = latents.transpose(0, 4, 1, 2, 3).astype(jnp.float32)
                    sigma = t_scalar
                    x0_cond = sample_cf - v_cond.astype(jnp.float32) * sigma
                    x0_uncond = sample_cf - v_uncond.astype(jnp.float32) * sigma
                    x0_ptb = sample_cf - v_ptb.astype(jnp.float32) * sigma

                    x0_guided = (
                        x0_cond
                        + (guidance_scale - 1) * (x0_cond - x0_uncond)
                        + stg_scale * (x0_cond - x0_ptb)
                    )

                    # Variance rescaling
                    rescale_scale = batch.rescale_scale
                    factor = x0_cond.std() / (x0_guided.std() + 1e-8)
                    factor = rescale_scale * factor + (1 - rescale_scale)
                    x0_guided = x0_guided * factor

                    noise_pred = ((sample_cf - x0_guided) / (sigma + 1e-8)).astype(noise_pred.dtype)
                else:
                    # WAN 2.1: guidance in velocity space (FlowUniPC handles x0 internally)
                    noise_pred = (
                        v_uncond
                        + guidance_scale * (v_cond - v_uncond)
                        + stg_scale * (v_cond - v_ptb)
                    )

            elif do_classifier_free_guidance:
                bsz = latents_in.shape[0] // 2
                v_cond = noise_pred[:bsz]
                v_uncond = noise_pred[bsz:]

                if use_x0_guidance:
                    # LTX-2: guidance in x0 space with variance rescaling
                    sample_cf = latents.transpose(0, 4, 1, 2, 3).astype(jnp.float32)
                    sigma = t_scalar
                    x0_cond = sample_cf - v_cond.astype(jnp.float32) * sigma
                    x0_uncond = sample_cf - v_uncond.astype(jnp.float32) * sigma

                    x0_guided = x0_uncond + guidance_scale * (x0_cond - x0_uncond)

                    rescale_scale = batch.rescale_scale
                    factor = x0_cond.std() / (x0_guided.std() + 1e-8)
                    factor = rescale_scale * factor + (1 - rescale_scale)
                    x0_guided = x0_guided * factor

                    noise_pred = ((sample_cf - x0_guided) / (sigma + 1e-8)).astype(noise_pred.dtype)
                else:
                    # WAN 2.1: guidance in velocity space
                    noise_pred = v_uncond + guidance_scale * (v_cond - v_uncond)

            # Audio guidance: separate parameters (PyTorch DEFAULT_AUDIO_GUIDER_PARAMS: cfg=7.0)
            if audio_noise_pred is not None and audio_latents is not None:
                if do_modality_guidance:
                    # 4-way split: [cond, uncond, ptb, mod]
                    bsz_a = 1
                    a_cond = audio_noise_pred[:bsz_a]
                    a_uncond = audio_noise_pred[bsz_a:2*bsz_a]
                    a_ptb = audio_noise_pred[2*bsz_a:3*bsz_a]
                    a_mod = audio_noise_pred[3*bsz_a:]
                    if use_x0_guidance:
                        sigma = t_scalar
                        a_sample = audio_latents.astype(jnp.float32)
                        ax0_cond = a_sample - a_cond.astype(jnp.float32) * sigma
                        ax0_uncond = a_sample - a_uncond.astype(jnp.float32) * sigma
                        ax0_ptb = a_sample - a_ptb.astype(jnp.float32) * sigma
                        ax0_mod = a_sample - a_mod.astype(jnp.float32) * sigma
                        ax0_guided = (
                            ax0_cond
                            + (audio_guidance_scale - 1) * (ax0_cond - ax0_uncond)
                            + audio_stg_scale * (ax0_cond - ax0_ptb)
                            + (audio_modality_scale - 1) * (ax0_cond - ax0_mod)
                        )
                        factor = ax0_cond.std() / (ax0_guided.std() + 1e-8)
                        factor = audio_rescale_scale * factor + (1 - audio_rescale_scale)
                        ax0_guided = ax0_guided * factor
                        audio_noise_pred = ((a_sample - ax0_guided) / (sigma + 1e-8)).astype(audio_noise_pred.dtype)
                    else:
                        audio_noise_pred = (
                            a_uncond
                            + audio_guidance_scale * (a_cond - a_uncond)
                            + audio_stg_scale * (a_cond - a_ptb)
                            + (audio_modality_scale - 1) * (a_cond - a_mod)
                        )
                elif do_spatio_temporal_guidance:
                    bsz_a = 1
                    a_cond = audio_noise_pred[:bsz_a]
                    a_uncond = audio_noise_pred[bsz_a:2*bsz_a]
                    a_ptb = audio_noise_pred[2*bsz_a:]
                    if use_x0_guidance:
                        sigma = t_scalar
                        a_sample = audio_latents.astype(jnp.float32)
                        ax0_cond = a_sample - a_cond.astype(jnp.float32) * sigma
                        ax0_uncond = a_sample - a_uncond.astype(jnp.float32) * sigma
                        ax0_ptb = a_sample - a_ptb.astype(jnp.float32) * sigma
                        ax0_guided = (
                            ax0_cond
                            + (audio_guidance_scale - 1) * (ax0_cond - ax0_uncond)
                            + audio_stg_scale * (ax0_cond - ax0_ptb)
                        )
                        factor = ax0_cond.std() / (ax0_guided.std() + 1e-8)
                        factor = audio_rescale_scale * factor + (1 - audio_rescale_scale)
                        ax0_guided = ax0_guided * factor
                        audio_noise_pred = ((a_sample - ax0_guided) / (sigma + 1e-8)).astype(audio_noise_pred.dtype)
                    else:
                        audio_noise_pred = a_uncond + audio_guidance_scale * (a_cond - a_uncond) + audio_stg_scale * (a_cond - a_ptb)
                elif do_classifier_free_guidance:
                    bsz_a = 1
                    a_cond = audio_noise_pred[:bsz_a]
                    a_uncond = audio_noise_pred[bsz_a:]
                    if use_x0_guidance:
                        sigma = t_scalar
                        a_sample = audio_latents.astype(jnp.float32)
                        ax0_cond = a_sample - a_cond.astype(jnp.float32) * sigma
                        ax0_uncond = a_sample - a_uncond.astype(jnp.float32) * sigma
                        ax0_guided = ax0_uncond + audio_guidance_scale * (ax0_cond - ax0_uncond)
                        factor = ax0_cond.std() / (ax0_guided.std() + 1e-8)
                        factor = audio_rescale_scale * factor + (1 - audio_rescale_scale)
                        ax0_guided = ax0_guided * factor
                        audio_noise_pred = ((a_sample - ax0_guided) / (sigma + 1e-8)).astype(audio_noise_pred.dtype)
                    else:
                        audio_noise_pred = a_uncond + audio_guidance_scale * (a_cond - a_uncond)

            # noise_pred is now velocity in channel-first (B, C, T, H, W) from model
            # latents is channel-last (B, T, H, W, C), need to transpose for solver
            latents = self.solver.step(
                model_output=noise_pred,  # velocity (B, C, T, H, W)
                timestep=t_scalar,
                sample=latents.transpose(0, 4, 1, 2, 3),  # (B, T, H, W, C) -> (B, C, T, H, W)
                return_dict=False,
            )[0]

            latents = latents.transpose(0, 2, 3, 4, 1)  # back to channel-last

            # Audio Euler step: audio_latents += audio_noise_pred * dt
            if audio_noise_pred is not None and audio_latents is not None:
                # Euler step for audio: x_{t+1} = x_t + v * dt
                # dt = sigma_{t+1} - sigma_t
                next_sigma = self.solver._sigmas[step + 1] if step + 1 < num_inference_steps else 0.0
                dt = next_sigma - t_scalar
                audio_latents = audio_latents + audio_noise_pred * dt

            if step_callback is not None:
                step_callback()

        logger.info("Finished diffusion step %d in %.2f seconds", step, time.time() - start_time)
        batch.latents = jax.device_get(latents)
        if audio_latents is not None:
            batch.audio_latents = jax.device_get(audio_latents)
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
                batch.height // self.model_config.scale_factor_spatial,
                batch.width // self.model_config.scale_factor_spatial,
                self.model_config.latent_input_dim,
            ),
            dtype=jnp.float32,
        )  # Placeholder for latents
        batch.latents = latents

    def _prepare_audio_latents(self, batch: Req):
        """Initialize random audio latents matching video duration.

        Audio latent shape: (B, T_audio, C*F) where:
        - T_audio = round(duration * sample_rate / hop_length / downsample_factor)
        - C = 8 (z_channels), F = 16 (mel_bins / 4)
        - C*F = 128 = audio_in_channels
        """
        if getattr(batch, "audio_latents", None) is not None:
            return
        duration = batch.num_frames / getattr(self.model_config, "fps", 25.0)
        # Audio VAE: sample_rate=16000, hop_length=160, downsample=4
        latents_per_second = 16000.0 / 160.0 / 4.0  # = 25
        audio_time_steps = round(duration * latents_per_second)
        audio_in_channels = getattr(self.model_config, "audio_in_channels", 128)
        batch.audio_latents = jax.random.normal(
            jax.random.PRNGKey(43),
            (1, audio_time_steps, audio_in_channels),
            dtype=jnp.float32,
        )
