import logging
import time
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tqdm import tqdm

from sgl_jax.srt.model_executor.base_model_runner import BaseModelRunner
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.configs.config_registry import get_diffusion_config
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.model_loader.loader import get_model_loader
from sgl_jax.srt.configs.load_config import LoadConfig

logger = logging.getLogger(__name__)

# --- Helper functions copied and verified from the working reference ---

def compute_sigma_schedule(steps: int, T: int, H: int, W: int, max_shift: float = 2.05, base_shift: float = 0.95, stretch: bool = True, terminal: float = 0.1) -> np.ndarray:
    tokens = T * H * W
    sigmas = np.linspace(1.0, 0.0, steps + 1)
    mm = (max_shift - base_shift) / (4096 - 1024)
    b = base_shift - mm * 1024
    sigma_shift = tokens * mm + b
    
    out_sigmas = np.exp(sigma_shift) / (np.exp(sigma_shift) + (1 / sigmas - 1))
    out_sigmas[-1] = 0
            
    if stretch and len(non_zero_sigmas := out_sigmas[out_sigmas != 0]) > 0:
        scale_factor = (1.0 - non_zero_sigmas[-1]) / (1.0 - terminal)
        if scale_factor > 0:
            out_sigmas[out_sigmas != 0] = 1.0 - ((1.0 - non_zero_sigmas) / scale_factor)
            
    return out_sigmas.astype(np.float32)

def compute_video_pe_numpy(T: int, H: int, W: int, fps: float, theta: float, inner_dim: int, num_attention_heads: int):
    scale_factors = (8, 32, 32)
    coords = [
        np.arange(T, dtype=np.float32) * scale_factors[0] / fps,
        np.arange(H, dtype=np.float32) * scale_factors[1],
        np.arange(W, dtype=np.float32) * scale_factors[2],
    ]
    grid = np.stack(np.meshgrid(*coords, indexing='ij'), axis=0)
    
    dim_per_head = inner_dim // num_attention_heads
    dim_list = [dim_per_head - 4 * (dim_per_head // 6), 2 * (dim_per_head // 6), 2 * (dim_per_head // 6)]
    pos_flat = grid.reshape(3, -1)
    
    freqs_list = [np.outer(pos_flat[i], 1.0 / (theta ** (np.arange(0, d, 2, dtype=np.float32) / d))) for i, d in enumerate(dim_list)]
    freqs = np.concatenate(freqs_list, axis=-1)
    freqs = np.broadcast_to(freqs[None, None, :, :], (1, num_attention_heads, freqs.shape[-2], freqs.shape[-1]))
    return np.cos(freqs), np.sin(freqs)


class LTX2ModelRunner(BaseModelRunner):
    def __init__(self, server_args: MultimodalServerArgs, mesh: jax.sharding.Mesh, model_class, stage_sub_dir: str | None = None):
        self.server_args = server_args
        self.mesh = mesh
        self.model_config = get_diffusion_config(server_args.model_path)
        self.model_config.model_path = server_args.model_path
        self.model_config.model_class = model_class
        self.initialize()

    def initialize(self):
        loader_config = LoadConfig(load_format=self.server_args.load_format, sub_dir=None)
        self.model_loader = get_model_loader(mesh=self.mesh, load_config=loader_config)
        self.model = self.model_loader.load_model(model_config=self.model_config)
        logger.info("LTX2ModelRunner initialized via standard SGLang loader.")

    def forward(self, batch: Req, mesh: jax.sharding.Mesh, abort_checker: Callable = None, step_callback: Callable = None) -> bool:
        guidance_scale, stg_scale, rescale_scale = batch.guidance_scale, getattr(batch, "stg_scale", 1.0), 0.7
        
        self.prepare_latents(batch)
        latents = jnp.array(batch.latents)
        _, _, T, H, W = latents.shape
        
        sigmas = compute_sigma_schedule(batch.num_inference_steps, T, H, W)
        pos_ctx = jnp.array(batch.prompt_embeds, dtype=jnp.bfloat16)
        neg_ctx = jnp.array(batch.negative_prompt_embeds, dtype=jnp.bfloat16)
        video_pe = compute_video_pe_numpy(T, H, W, batch.fps, self.model_config.positional_embedding_theta, self.model_config.attention_head_dim * self.model_config.num_attention_heads, self.model_config.num_attention_heads)

        @partial(nnx.jit, static_argnames=["stg_blocks"])
        def dit_forward(model, latent, timesteps, context, stg_mask, stg_blocks):
            return model(video_latent=latent, video_context=context, timesteps=timesteps, video_pe=video_pe, stg_mask=stg_mask, stg_blocks=stg_blocks)[0]

        logger.info("Warming up JIT...")
        t0 = time.time()
        dit_forward(self.model, jnp.concatenate([latents]*3), jnp.array([sigmas[0]]*3), jnp.concatenate([pos_ctx, neg_ctx, pos_ctx]), jnp.array([[[1.]],[[1.]],[[0.]]]), (29,)).block_until_ready()
        logger.info(f"JIT warmup done in {time.time() - t0:.1f}s")
        
        loop_start_time = time.time()
        for step in tqdm(range(batch.num_inference_steps), desc="Denoising"):
            if abort_checker and abort_checker(): return True
            sigma = float(sigmas[step])
            
            v_pred = dit_forward(self.model, jnp.concatenate([latents]*3).astype(jnp.bfloat16), jnp.array([sigma]*3), jnp.concatenate([pos_ctx, neg_ctx, pos_ctx]), jnp.array([[[1.]],[[1.]],[[0.]]]), (29,))
            v_cond, v_uncond, v_ptb = jnp.split(v_pred.astype(jnp.float32), 3)
            
            x0_cond = latents - v_cond * sigma
            x0_uncond = latents - v_uncond * sigma
            x0_ptb = latents - v_ptb * sigma
            x0_pred = x0_cond + (guidance_scale - 1.0) * (x0_cond - x0_uncond) + stg_scale * (x0_cond - x0_ptb)
            
            # This is the CORRECT rescaling logic from the reference script
            factor = jnp.std(x0_cond, ddof=1) / jnp.maximum(jnp.std(x0_pred, ddof=1), 1e-6)
            factor = rescale_scale * factor + (1.0 - rescale_scale)
            x0_pred *= factor
            
            video_pred = (latents - x0_pred) / sigma
            latents += video_pred * (float(sigmas[step + 1]) - sigma)
            if step_callback: step_callback()

        logger.info(f"Finished {batch.num_inference_steps} steps in {time.time() - loop_start_time:.2f}s")
        batch.latents = jax.device_get(latents)
        return False

    def prepare_latents(self, batch: Req):
        if batch.latents is not None: return
        cfg = self.model_config
        T = ((batch.num_frames - 1) // cfg.scale_factor_temporal) + 1
        H, W = batch.height // cfg.scale_factor_spatial, batch.width // cfg.scale_factor_spatial
        batch.latents = jax.random.normal(jax.random.PRNGKey(42), (1, cfg.latent_input_dim, T, H, W))
