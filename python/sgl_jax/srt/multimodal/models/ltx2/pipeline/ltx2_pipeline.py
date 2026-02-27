"""
LTX-2 Inference Pipeline for Text-to-Video Generation

This module provides the end-to-end pipeline for generating videos from text prompts
using the LTX-2 diffusion transformer model.
"""

import logging
from typing import Optional, Union

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class LTX2Pipeline(nnx.Module):
    """
    End-to-end pipeline for LTX-2 text-to-video generation.

    This pipeline orchestrates:
    1. Text encoding (prompt → embeddings)
    2. Noise initialization
    3. Iterative denoising (diffusion process)
    4. VAE decoding (latents → video pixels)

    Args:
        text_encoder: LTX2GemmaTextEncoder for encoding prompts
        transformer: LTX2Transformer3DModel for denoising
        vae_decoder: LTX2VAEDecoder for decoding latents
        scheduler: LTX2Scheduler for sigma schedule
        tokenizer: Gemma tokenizer for text preprocessing
        mesh: JAX sharding mesh
        dtype: Data type for computations
        max_sequence_length: Maximum sequence length for tokenization (default: 512)
    """

    def __init__(
        self,
        text_encoder,
        transformer,
        vae_decoder,
        scheduler,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        max_sequence_length: int = 512,
    ):
        self.text_encoder = text_encoder
        self.transformer = transformer
        self.vae_decoder = vae_decoder
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.mesh = mesh
        self.dtype = dtype
        self.max_sequence_length = max_sequence_length

        logger.info("Initialized LTX2Pipeline")

    def __call__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_frames: int = 121,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        seed: Optional[int] = None,
        **kwargs,
    ) -> jax.Array:
        """
        Generate a video from a text prompt.

        Args:
            prompt: Text description of the video to generate
            negative_prompt: Negative prompt for CFG (if None, uses empty string)
            num_frames: Number of frames to generate (must be 8k+1, e.g. 121, 97, 81)
            height: Video height in pixels (must be divisible by 32)
            width: Video width in pixels (must be divisible by 32)
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale (1.0 = no guidance)
            seed: Random seed for reproducibility

        Returns:
            Generated video tensor [B, 3, num_frames, height, width]
        """
        # Validate inputs
        assert (num_frames - 1) % 8 == 0, f"num_frames must be 8k+1, got {num_frames}"
        assert height % 32 == 0, f"height must be divisible by 32, got {height}"
        assert width % 32 == 0, f"width must be divisible by 32, got {width}"

        batch_size = 1
        do_classifier_free_guidance = guidance_scale > 1.0

        # Step 1: Encode text prompt (and negative prompt if CFG is enabled)
        logger.info(f"Encoding prompt: {prompt}")
        text_embeddings, negative_embeddings = self._encode_prompt(
            prompt,
            negative_prompt=negative_prompt,
            batch_size=batch_size,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        # Step 2: Initialize latent noise
        logger.info(f"Initializing latents: {num_frames} frames, {height}x{width}")
        latents = self._initialize_latents(
            batch_size, num_frames, height, width, seed
        )

        # Step 3: Get sigma schedule
        sigmas = self.scheduler.execute(
            steps=num_inference_steps,
            latent=latents,
        )

        # Step 4: Denoising loop
        logger.info(f"Running denoising for {num_inference_steps} steps (guidance_scale={guidance_scale})")
        latents = self._denoise_latents(
            latents=latents,
            text_embeddings=text_embeddings,
            negative_embeddings=negative_embeddings,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        # Step 5: Decode latents to video
        logger.info("Decoding latents to video pixels")
        video = self._decode_latents(latents)

        logger.info(f"Generated video shape: {video.shape}")
        return video

    def _encode_prompt(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        batch_size: int = 1,
        do_classifier_free_guidance: bool = True,
    ) -> tuple[jax.Array, Optional[jax.Array]]:
        """
        Encode text prompt and optional negative prompt to embeddings.

        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt for CFG (if None, uses empty string)
            batch_size: Batch size
            do_classifier_free_guidance: Whether to generate negative embeddings

        Returns:
            Tuple of (positive_embeddings, negative_embeddings)
            negative_embeddings is None if do_classifier_free_guidance is False
        """
        # Tokenize the prompt
        tokenizer_output = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="np",
        )

        # Convert to JAX arrays
        input_ids = jnp.array(tokenizer_output["input_ids"], dtype=jnp.int32)

        # Repeat for batch if needed
        if batch_size > 1:
            input_ids = jnp.repeat(input_ids, batch_size, axis=0)

        # Create ForwardBatch and encode
        forward_batch = ForwardBatch(input_ids=input_ids)
        text_embeddings = self.text_encoder(forward_batch, token_to_kv_pool=None)

        logger.info(f"Encoded prompt to shape: {text_embeddings.shape}")

        # Encode negative prompt if CFG is enabled
        negative_embeddings = None
        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = ""

            # Tokenize negative prompt
            neg_tokenizer_output = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=self.max_sequence_length,
                truncation=True,
                return_tensors="np",
            )

            neg_input_ids = jnp.array(neg_tokenizer_output["input_ids"], dtype=jnp.int32)

            # Repeat for batch if needed
            if batch_size > 1:
                neg_input_ids = jnp.repeat(neg_input_ids, batch_size, axis=0)

            # Encode negative prompt
            neg_forward_batch = ForwardBatch(input_ids=neg_input_ids)
            negative_embeddings = self.text_encoder(neg_forward_batch, token_to_kv_pool=None)

            logger.info(f"Encoded negative prompt to shape: {negative_embeddings.shape}")

        return text_embeddings, negative_embeddings

    def _initialize_latents(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        seed: Optional[int] = None,
    ) -> jax.Array:
        """
        Initialize random noise latents.

        Args:
            batch_size: Batch size
            num_frames: Number of video frames
            height: Video height
            width: Video width
            seed: Random seed

        Returns:
            Noise tensor [B, 128, T, H, W] where:
                T = (num_frames - 1) // 8 + 1
                H = height // 32
                W = width // 32
        """
        # Calculate latent dimensions
        latent_frames = (num_frames - 1) // 8 + 1
        latent_height = height // 32
        latent_width = width // 32
        latent_channels = 128

        # Initialize random key
        if seed is None:
            seed = 0
        key = jax.random.PRNGKey(seed)

        # Sample random noise
        latents = jax.random.normal(
            key,
            shape=(batch_size, latent_channels, latent_frames, latent_height, latent_width),
            dtype=self.dtype,
        )

        logger.info(f"Initialized latents with shape: {latents.shape}")
        return latents

    def _denoise_latents(
        self,
        latents: jax.Array,
        text_embeddings: jax.Array,
        negative_embeddings: Optional[jax.Array],
        sigmas: jax.Array,
        guidance_scale: float,
        do_classifier_free_guidance: bool,
    ) -> jax.Array:
        """
        Iterative denoising using the diffusion transformer with classifier-free guidance.

        Args:
            latents: Initial noise [B, 128, T, H, W]
            text_embeddings: Positive text conditioning [B, L, 3840]
            negative_embeddings: Negative text conditioning [B, L, 3840] or None
            sigmas: Sigma schedule [num_steps + 1]
            guidance_scale: CFG scale
            do_classifier_free_guidance: Whether to apply CFG

        Returns:
            Denoised latents [B, 128, T, H, W]
        """
        num_steps = len(sigmas) - 1

        # Compute position grid dimensions
        # The transformer patch embeds with size (1, 2, 2), so grid size is:
        # T_grid = T // 1 = T
        # H_grid = H // 2
        # W_grid = W // 2
        _, _, T, H, W = latents.shape
        video_positions = (T, H // 2, W // 2)

        for step_idx in range(num_steps):
            sigma = sigmas[step_idx]
            sigma_next = sigmas[step_idx + 1]

            # Prepare timesteps
            timesteps = jnp.full((latents.shape[0],), sigma, dtype=self.dtype)

            if do_classifier_free_guidance:
                # Duplicate latents for conditional and unconditional predictions
                latent_model_input = jnp.concatenate([latents, latents], axis=0)
                timesteps_input = jnp.concatenate([timesteps, timesteps], axis=0)

                # Concatenate text embeddings: [positive, negative]
                text_embeddings_input = jnp.concatenate([text_embeddings, negative_embeddings], axis=0)

                # Forward pass
                noise_pred = self.transformer(
                    video_latent=latent_model_input,
                    audio_latent=None,
                    video_context=text_embeddings_input,
                    audio_context=None,
                    timesteps=timesteps_input,
                    video_positions=video_positions,
                    audio_positions=None,
                    video_context_mask=None,
                    audio_context_mask=None,
                )

                # Get video prediction
                noise_pred = noise_pred[0] if isinstance(noise_pred, tuple) else noise_pred

                # Split conditional and unconditional predictions
                noise_pred_cond, noise_pred_uncond = jnp.split(noise_pred, 2, axis=0)

                # Apply classifier-free guidance
                video_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # No CFG, just forward pass with positive embeddings
                noise_pred = self.transformer(
                    video_latent=latents,
                    audio_latent=None,
                    video_context=text_embeddings,
                    audio_context=None,
                    timesteps=timesteps,
                    video_positions=video_positions,
                    audio_positions=None,
                    video_context_mask=None,
                    audio_context_mask=None,
                )

                # Get video prediction
                video_pred = noise_pred[0] if isinstance(noise_pred, tuple) else noise_pred

            # Apply Euler step
            dt = sigma_next - sigma
            velocity = self._to_velocity(latents, sigma, video_pred)
            latents = latents + velocity * dt

            if (step_idx + 1) % 10 == 0:
                logger.info(f"Denoising step {step_idx + 1}/{num_steps}")

        return latents

    def _to_velocity(
        self,
        sample: jax.Array,
        sigma: float,
        denoised: jax.Array,
    ) -> jax.Array:
        """
        Compute velocity for flow matching.

        Args:
            sample: Current noisy sample
            sigma: Current noise level
            denoised: Denoised prediction

        Returns:
            Velocity vector
        """
        return (denoised - sample) / (1 - sigma + 1e-8)

    def _decode_latents(
        self,
        latents: jax.Array,
    ) -> jax.Array:
        """
        Decode latents to video pixels using VAE decoder.

        Args:
            latents: Latent tensor [B, 128, T, H, W]

        Returns:
            Video tensor [B, 3, T', H', W'] where:
                T' = (T - 1) * 8 + 1
                H' = H * 32
                W' = W * 32
        """
        video = self.vae_decoder(latents)
        return video

    @staticmethod
    def from_pretrained(
        model_path: str,
        text_encoder_path: str,
        vae_path: str,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        max_sequence_length: int = 512,
    ):
        """
        Load a pretrained LTX-2 pipeline.

        Args:
            model_path: Path to transformer weights
            text_encoder_path: Path to Gemma text encoder
            vae_path: Path to VAE weights
            mesh: JAX sharding mesh
            dtype: Data type
            max_sequence_length: Maximum sequence length for tokenization

        Returns:
            LTX2Pipeline instance
        """
        from sgl_jax.srt.hf_transformers_utils import get_tokenizer
        from sgl_jax.srt.multimodal.models.ltx2.text_encoder import LTX2GemmaTextEncoder
        from sgl_jax.srt.multimodal.models.ltx2.diffusion import LTX2Transformer3DModel
        from sgl_jax.srt.multimodal.models.ltx2.vae import LTX2VAEDecoder
        from sgl_jax.srt.multimodal.models.ltx2.scheduler import LTX2Scheduler

        # Load tokenizer
        logger.info(f"Loading tokenizer from {text_encoder_path}")
        tokenizer = get_tokenizer(
            tokenizer_name=text_encoder_path,
            trust_remote_code=True,
        )

        # TODO: Load model components with proper weight loading
        # text_encoder = LTX2GemmaTextEncoder.from_pretrained(text_encoder_path, mesh=mesh, dtype=dtype)
        # transformer = LTX2Transformer3DModel.from_pretrained(model_path, mesh=mesh, dtype=dtype)
        # vae_decoder = LTX2VAEDecoder.from_pretrained(vae_path, mesh=mesh, dtype=dtype)
        # scheduler = LTX2Scheduler()

        raise NotImplementedError(
            "from_pretrained not yet implemented - need to implement weight loading for components"
        )
