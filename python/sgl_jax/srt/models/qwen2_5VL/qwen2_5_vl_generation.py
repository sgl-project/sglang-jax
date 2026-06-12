import logging

import jax.numpy as jnp

from sgl_jax.srt.layers.embeddings import MRotaryEmbedding
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.qwen2 import Qwen2Model

logger = logging.getLogger(__name__)


class Qwen2_5_VL_Model(Qwen2Model):
    """Qwen2Model with MRoPE support for Qwen2.5-VL."""

    def __init__(
        self,
        config,
        mesh,
        dtype=jnp.bfloat16,
    ):
        super().__init__(config=config, mesh=mesh, dtype=dtype)
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        self._mrope_section = rope_scaling.get("mrope_section")
        self._mrope_interleaved = rope_scaling.get("mrope_interleaved", False)
        if self._mrope_section:
            rope_theta = getattr(config, "rope_theta", 1000000)
            max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
            for layer in self.layers:
                head_dim = layer.self_attn.head_dim
                layer.self_attn.rotary_emb = MRotaryEmbedding(
                    head_size=head_dim,
                    rotary_dim=head_dim,
                    max_position_embeddings=max_position_embeddings,
                    base=rope_theta,
                    is_neox_style=True,
                    dtype=dtype,
                    mrope_section=self._mrope_section,
                    mrope_interleaved=self._mrope_interleaved,
                )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ):
        residual = None
        input_embeds = (
            forward_batch.input_embedding
            if forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
            else None
        )
        hidden_states = (
            self.embed_tokens(forward_batch.input_ids) if input_embeds is None else input_embeds
        )
        rope_positions = (
            forward_batch.mrope_positions
            if self._mrope_section and forward_batch.mrope_positions is not None
            else forward_batch.positions
        )
        layers_kv_fused = []
        layers_callback_flag = []
        for layer in self.layers:
            hidden_states, residual, kv_fused, callback_flag = layer(
                rope_positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
            )
            layers_kv_fused.append(kv_fused)
            layers_callback_flag.extend(callback_flag)

        if residual is not None:
            hidden_states += residual
        hidden_states = self.norm(hidden_states)

        return hidden_states, layers_kv_fused, layers_callback_flag
