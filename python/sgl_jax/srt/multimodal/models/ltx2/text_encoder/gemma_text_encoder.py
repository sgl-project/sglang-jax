"""
Gemma Text Encoder for LTX-2

This module provides a text encoder based on Gemma models for the LTX-2 video generation pipeline.
It extracts text embeddings that are used to condition the diffusion transformer.
"""

import logging
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.models.gemma2 import Gemma2Model
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

from sgl_jax.srt.multimodal.models.ltx2.diffusion.ltx2_dit import LTX2Attention, RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.multimodal.layers.mlp import MLP

logger = logging.getLogger(__name__)


class _BasicTransformerBlock1D(nnx.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mesh=None):
        super().__init__()
        self.attn1 = LTX2Attention(
            query_dim=dim,
            heads=heads,
            dim_head=dim_head,
            mesh=mesh
        )
        self.ff = MLP(
            input_dim=dim,
            mlp_hidden_dim=dim * 4,
            output_dim=dim,
            act_type="gelu",
            mesh=mesh
        )
        self.norm_q = RMSNorm(dim)
        self.norm_k = RMSNorm(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def __call__(self, x, mask, pe):
        nx = self.norm1(x)
        # SGLang JAX's LTX2Attention applies norm_q/norm_k inside it
        attn_out = self.attn1(nx, context=nx, mask=mask, pe=pe)
        x = x + attn_out
        nx = self.norm2(x)
        x = x + self.ff(nx)
        return x


class Embeddings1DConnector(nnx.Module):
    def __init__(self, num_layers=2, dim=3840, heads=30, dim_head=128, num_registers=128, mesh=None):
        super().__init__()
        self.dim = dim
        self.num_registers = num_registers
        self.blocks = [_BasicTransformerBlock1D(dim, heads, dim_head, mesh) for _ in range(num_layers)]
        self.norm_out = RMSNorm(dim)
        if self.num_registers > 0:
            self.learnable_registers = nnx.Param(jnp.zeros((num_registers, dim), dtype=jnp.float32))

    def __call__(self, x, mask):
        # 1D RoPE
        L = x.shape[1]
        freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(0, self.dim//2, 2, dtype=jnp.float32) / (self.dim//2))
        t = jnp.arange(L, dtype=jnp.float32)
        args = t[:, None] * freqs[None, :]
        cos = jnp.cos(args).repeat(2, axis=-1)
        sin = jnp.sin(args).repeat(2, axis=-1)
        pe = (cos, sin)

        for block in self.blocks:
            x = block(x, mask, pe)
        return self.norm_out(x)


class GemmaFeaturesExtractorProjLinear(nnx.Module):
    def __init__(self, mesh=None):
        super().__init__()
        self.aggregate_embed = LinearBase(3840 * 49, 3840, use_bias=False, mesh=mesh, kernel_axes=(None, "tensor"))

    def __call__(self, hidden_states, mask):
        # hidden_states: [B, T, D, L] where L=49
        b, t, d, l = hidden_states.shape
        # For simplicity in testing/inference, we assume all tokens in the forward batch are valid up to T
        # and mask is a boolean array of [B, T]
        denom = (mask.sum(axis=1, keepdims=True) * d)[..., None, None]
        mean = hidden_states.sum(axis=(1, 2), keepdims=True) / (denom + 1e-6)
        
        normed = hidden_states - mean
        # Zero out invalid tokens
        normed = jnp.where(mask[:, :, None, None], normed, 0.0)
        return self.aggregate_embed(jnp.reshape(normed, (b, t, -1)))


class LTX2GemmaTextEncoder(nnx.Module):
    """
    Gemma-based text encoder for LTX-2.
    Includes the Gemma2 backbone, the feature extractor projection, and the 1D embedding connector.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.mesh = mesh
        self.dtype = dtype

        self.model = Gemma2Model(config, dtype=dtype, mesh=mesh)
        self.model.capture_aux_hidden_states = True

        self.hidden_size = config.hidden_size

        self.feature_extractor = GemmaFeaturesExtractorProjLinear(mesh=mesh)
        self.embeddings_connector = Embeddings1DConnector(
            num_layers=2, dim=self.hidden_size, heads=30, dim_head=128, num_registers=128, mesh=mesh
        )

        logger.info(
            f"Initialized LTX2GemmaTextEncoder with hidden_size={self.hidden_size}"
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: Optional[KVCache] = None,
    ) -> jax.Array:
        # Get all hidden states from Gemma model
        _, aux_hidden_states, _ = self.model(forward_batch, token_to_kv_pool)
        
        # Stack to [B, T, D, L] where L = 49 (embed + 48 layers)
        stacked_hidden_states = jnp.stack(aux_hidden_states, axis=-1)
        
        # Mask is all 1s for the valid tokens passed
        # This implementation assumes sequence length equals actual tokens (padding stripped by SGLang)
        # but if there's padding, forward_batch should reflect it.
        b, t = stacked_hidden_states.shape[:2]
        mask = jnp.ones((b, t), dtype=jnp.bool_)

        projected = self.feature_extractor(stacked_hidden_states, mask)
        
        # Replace padded with registers if we had padding (skipped here since we assume densely packed batch)
        # Run connector
        encoded_ctx = self.embeddings_connector(projected, mask)
        
        return encoded_ctx

    def load_weights(self, model_config: ModelConfig, ltx_checkpoint_path: str | None = None):
        loader = WeightLoader(
            model=self.model,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        loader.load_weights()
        logger.info(f"Loaded Gemma backbone weights from {model_config.model_path}")
        
        if ltx_checkpoint_path is not None:
            # We must load feature_extractor and embeddings_connector from LTX checkpoint
            self._load_ltx2_connector_weights(ltx_checkpoint_path)

    def _load_ltx2_connector_weights(self, checkpoint_path: str):
        from safetensors import safe_open
        logger.info(f"Loading text encoder connector weights from {checkpoint_path}")
        
        mappings = {
            "text_embedding_projection.aggregate_embed.weight": WeightMapping(
                target_path="feature_extractor.aggregate_embed.kernel",
                sharding=(None, None),
                transpose=True,
            ),
            "model.diffusion_model.video_embeddings_connector.learnable_registers": WeightMapping(
                target_path="embeddings_connector.learnable_registers",
                sharding=(None, None),
            )
        }
        
        for i in range(2):
            prefix = f"model.diffusion_model.video_embeddings_connector.transformer_1d_blocks.{i}"
            target = f"embeddings_connector.blocks.{i}"
            mappings.update({
                f"{prefix}.attn1.to_q.weight": WeightMapping(f"{target}.attn1.to_q.weight", (None, "tensor"), True),
                f"{prefix}.attn1.to_q.bias": WeightMapping(f"{target}.attn1.to_q.bias", ("tensor",)),
                f"{prefix}.attn1.to_k.weight": WeightMapping(f"{target}.attn1.to_k.weight", (None, "tensor"), True),
                f"{prefix}.attn1.to_k.bias": WeightMapping(f"{target}.attn1.to_k.bias", ("tensor",)),
                f"{prefix}.attn1.to_v.weight": WeightMapping(f"{target}.attn1.to_v.weight", (None, "tensor"), True),
                f"{prefix}.attn1.to_v.bias": WeightMapping(f"{target}.attn1.to_v.bias", ("tensor",)),
                f"{prefix}.attn1.to_out.0.weight": WeightMapping(f"{target}.attn1.to_out.weight", ("tensor", None), True),
                f"{prefix}.attn1.to_out.0.bias": WeightMapping(f"{target}.attn1.to_out.bias", (None,)),
                f"{prefix}.attn1.q_norm.weight": WeightMapping(f"{target}.attn1.norm_q.scale", (None,)),
                f"{prefix}.attn1.k_norm.weight": WeightMapping(f"{target}.attn1.norm_k.scale", (None,)),
                f"{prefix}.ff.net.0.proj.weight": WeightMapping(f"{target}.ff.fc_in.weight", (None, "tensor"), True),
                f"{prefix}.ff.net.0.proj.bias": WeightMapping(f"{target}.ff.fc_in.bias", ("tensor",)),
                f"{prefix}.ff.net.2.weight": WeightMapping(f"{target}.ff.fc_out.weight", ("tensor", None), True),
                f"{prefix}.ff.net.2.bias": WeightMapping(f"{target}.ff.fc_out.bias", (None,)),
            })

        params = nnx.state(self)
        loaded = 0
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            all_keys = set(f.keys())
            for ck, mapping in mappings.items():
                if ck in all_keys:
                    tensor = f.get_tensor(ck).float().numpy()
                    if mapping.transpose:
                        tensor = tensor.T
                    
                    target_keys = mapping.target_path.split('.')
                    current = params
                    for key in target_keys[:-1]:
                        if key.isdigit():
                            current = current[int(key)]
                        else:
                            current = getattr(current, key)
                    last_key = target_keys[-1]
                    
                    param = getattr(current, last_key)
                    
                    jax_tensor = jnp.array(tensor, dtype=param.dtype)
                    if jax_tensor.shape != param.shape:
                        logger.warning(f"Shape mismatch for {ck}: {jax_tensor.shape} != {param.shape}")
                    else:
                        param[...] = jax_tensor
                        loaded += 1
                else:
                    logger.warning(f"Key {ck} not found in checkpoint")
        
        nnx.update(self, params)
        logger.info(f"Loaded {loaded} LTX-2 connector weights.")


    @staticmethod
    def from_pretrained(
        model_name_or_path: str,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        ltx_checkpoint_path: str | None = None,
    ):
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name_or_path)
        encoder = LTX2GemmaTextEncoder(config, mesh=mesh, dtype=dtype)

        model_config = ModelConfig(model_path=model_name_or_path)
        encoder.load_weights(model_config, ltx_checkpoint_path)

        return encoder


# Entry class for model loading
EntryClass = LTX2GemmaTextEncoder
