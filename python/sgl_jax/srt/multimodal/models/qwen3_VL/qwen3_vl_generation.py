"""Qwen3-VL Generation Model for SGLang-JAX.

This module implements the text decoder for Qwen3-VL with M-RoPE
(Multimodal Rotary Position Embeddings) for handling multimodal sequences.

Key features:
- Interleaved M-RoPE: T/H/W position embeddings with interleaved layout
- Q/K normalization: RMSNorm applied to query and key before RoPE
- Extends Qwen2Model from sglang-jax base models
"""

import logging

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.hf_transformers_utils import get_hf_text_config
from sgl_jax.srt.layers.embeddings import ParallelLMHead, apply_rotary_emb
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.qwen2 import Qwen2Model
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


def _apply_interleaved_rope(x: jax.Array, mrope_section: list[int]) -> jax.Array:
    """Apply interleaved MRoPE layout.

    Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
    interleaved [THTHWHTHW...TT] pattern.

    Args:
        x: Frequencies of shape (3, seq_len, head_dim // 2) for T, H, W
        mrope_section: Section sizes (e.g., [24, 20, 20])

    Returns:
        Interleaved frequencies of shape (seq_len, head_dim // 2)
    """
    x_t = x[0]
    x_t = x_t.at[..., 1 : mrope_section[1] * 3 : 3].set(x[1, ..., 1 : mrope_section[1] * 3 : 3])
    x_t = x_t.at[..., 2 : mrope_section[2] * 3 : 3].set(x[2, ..., 2 : mrope_section[2] * 3 : 3])
    return x_t


class MRotaryEmbedding:
    """Rotary Embedding with Multimodal Sections for Qwen3-VL.

    Implements the M-RoPE mechanism that partitions the head dimension
    into sections for temporal (T), height (H), and width (W) positions.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: jnp.dtype,
        mrope_section: list[int],
        mrope_interleaved: bool = True,
    ) -> None:
        del max_position_embeddings
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        self.mrope_section = list(mrope_section)
        self.mrope_interleaved = mrope_interleaved

        inv_freq_np = 1.0 / (base ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim))
        self._inv_freq_np = inv_freq_np

        # Validate and adjust section sizes
        expected_sum = rotary_dim // 2
        actual_sum = sum(self.mrope_section)
        if actual_sum != expected_sum:
            logger.warning(
                "MRoPE section sum mismatch: expected %s, got %s. Adjusting.",
                expected_sum,
                actual_sum,
            )
            if actual_sum > 0:
                scale_factor = expected_sum / actual_sum
                self.mrope_section = [
                    max(1, int(section * scale_factor)) for section in self.mrope_section
                ]
                current_sum = sum(self.mrope_section)
                if current_sum != expected_sum:
                    self.mrope_section[-1] += expected_sum - current_sum
            else:
                self.mrope_section = [expected_sum // len(self.mrope_section)] * len(
                    self.mrope_section
                )
                remainder = expected_sum % len(self.mrope_section)
                for i in range(remainder):
                    self.mrope_section[i] += 1

    def __call__(
        self,
        positions: jax.Array,
        query: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Apply M-RoPE to query and key.

        Args:
            positions: Position IDs. Shape (seq,) for text-only or (3, seq) for multimodal
            query: Query tensor of shape (seq, hidden_dim)
            key: Key tensor of shape (seq, hidden_dim)

        Returns:
            Rotated query and key tensors
        """
        inv_freq = jnp.asarray(self._inv_freq_np, dtype=self.dtype)

        if positions.ndim == 1:
            # Text-only: simple 1D positions
            freqs = jnp.einsum("n,d->nd", positions.astype(jnp.float32), inv_freq)
            cos = jnp.cos(freqs).astype(self.dtype)
            sin = jnp.sin(freqs).astype(self.dtype)
        else:
            # Multimodal: 3D positions (T, H, W)
            freqs = jnp.einsum("tn,d->tnd", positions.astype(jnp.float32), inv_freq)
            cos = jnp.cos(freqs).astype(self.dtype)
            sin = jnp.sin(freqs).astype(self.dtype)

            if self.mrope_interleaved:
                cos = _apply_interleaved_rope(cos, self.mrope_section)
                sin = _apply_interleaved_rope(sin, self.mrope_section)
            else:
                # Concatenate sections
                cos_slices = []
                sin_slices = []
                offset = 0
                for i, section in enumerate(self.mrope_section):
                    cos_slices.append(cos[i, :, offset : offset + section])
                    sin_slices.append(sin[i, :, offset : offset + section])
                    offset += section
                cos = jnp.concatenate(cos_slices, axis=-1)
                sin = jnp.concatenate(sin_slices, axis=-1)

        num_tokens = positions.shape[-1]
        query_shape = query.shape
        query = query.reshape(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = jnp.concatenate((query_rot, query_pass), axis=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.reshape(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = jnp.concatenate((key_rot, key_pass), axis=-1).reshape(key_shape)

        return query, key


class Qwen3_VL_Model(Qwen2Model):
    """Qwen3-VL text decoder with M-RoPE support.

    Extends Qwen2Model to handle multimodal position embeddings.
    """

    def __init__(
        self,
        config,
        mesh,
        dtype=jnp.bfloat16,
    ):
        super().__init__(config=config, mesh=mesh, dtype=dtype)

        # Override rotary embeddings with M-RoPE
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        self._mrope_section = rope_scaling.get("mrope_section")
        self._mrope_interleaved = rope_scaling.get("mrope_interleaved", True)

        if self._mrope_section:
            rope_theta = getattr(config, "rope_theta", 5000000)
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

        # Use input embeddings if provided (for multimodal prefill)
        input_embeds = (
            forward_batch.input_embedding
            if forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
            else None
        )
        hidden_states = (
            self.embed_tokens(forward_batch.input_ids) if input_embeds is None else input_embeds
        )

        # Use M-RoPE positions if available
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


class Qwen3_VL_Generation(nnx.Module):
    """Qwen3-VL model for conditional generation.

    Architecture:
    - Vision encoder (separate module): Processes images/videos to embeddings
    - Language model (self.model): Generates text with M-RoPE

    Usage Pattern:
    1. PREFILL (once per image):
       - Process vision with Qwen3_VL_VisionModel
       - Merge embeddings with get_input_embeddings()
       - Call __call__() with merged embeddings
    2. DECODE (many times for text generation):
       - Call __call__() without embeddings (uses text tokens only)
    """

    def __init__(self, config=None, dtype=None, mesh=None):
        super().__init__()
        self.mesh = mesh
        self.config = config
        self.text_config = get_hf_text_config(config) or config
        self.dtype = dtype or jnp.bfloat16

        self.model = Qwen3_VL_Model(self.text_config, mesh=mesh, dtype=self.dtype)

        if not getattr(self.text_config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                self.text_config.vocab_size,
                self.text_config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=("tensor", None),
            )

        self.logits_processor = LogitsProcessor(self.text_config.vocab_size, mesh=self.mesh)

        # Multimodal token IDs
        self.image_token_id = getattr(self.config, "image_token_id", 151655)
        self.video_token_id = getattr(self.config, "video_token_id", 151656)

    def load_weights(self, model_config: ModelConfig):
        """Load model weights from safetensors."""
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_qwen3_weight_mappings()
        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("Qwen3-VL (LLM) weights loaded successfully!")

    def _create_qwen3_weight_mappings(self) -> dict:
        """Create weight mappings for text decoder."""
        mappings = {
            "model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.norm.weight": WeightMapping(
                target_path="model.norm.scale",
                sharding=(None,),
                transpose=False,
            ),
        }

        if not getattr(self.text_config, "tie_word_embeddings", False):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding",
                sharding=("tensor", None),
                transpose=False,
            )

        num_layers = self.text_config.num_hidden_layers
        for layer_idx in range(num_layers):
            mappings.update(self._create_layer_mappings(layer_idx))

        return mappings

    def _create_layer_mappings(self, layer_idx: int) -> dict:
        """Create weight mappings for a single decoder layer."""
        prefix = f"model.layers.{layer_idx}"
        target_prefix = f"model.layers.{layer_idx}"

        mappings = {
            # Layer norms
            f"{prefix}.input_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.input_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.post_attention_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.post_attention_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            # Attention projections
            f"{prefix}.self_attn.q_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=False,
            ),
            f"{prefix}.self_attn.k_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.k_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=True,
            ),
            f"{prefix}.self_attn.v_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.v_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=True,
            ),
            f"{prefix}.self_attn.o_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.o_proj.weight",
                sharding=("tensor", None),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=False,
            ),
            # Q/K norms (Qwen3-VL specific)
            f"{prefix}.self_attn.q_norm.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_norm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.self_attn.k_norm.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.k_norm.scale",
                sharding=(None,),
                transpose=False,
            ),
            # MLP
            f"{prefix}.mlp.gate_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.gate_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.up_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.up_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.down_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.down_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
        }

        # Add bias mappings if attention uses bias
        if getattr(self.text_config, "attention_bias", False):
            mappings.update(
                {
                    f"{prefix}.self_attn.q_proj.bias": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.q_proj.bias",
                        sharding=(None,),
                        transpose=False,
                        head_dim_padding=True,
                        kv_head_padding=False,
                    ),
                    f"{prefix}.self_attn.k_proj.bias": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.k_proj.bias",
                        sharding=(None,),
                        transpose=False,
                        head_dim_padding=True,
                        kv_head_padding=True,
                    ),
                    f"{prefix}.self_attn.v_proj.bias": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.v_proj.bias",
                        sharding=(None,),
                        transpose=False,
                        head_dim_padding=True,
                        kv_head_padding=True,
                    ),
                }
            )

        return mappings

    def get_embed_and_head(self):
        """Get embedding and lm_head weights for tied embeddings handling."""
        if getattr(self.text_config, "tie_word_embeddings", False):
            weight = self.model.embed_tokens.embedding.value
            return (weight, weight)
        return (self.model.embed_tokens.embedding.value, self.lm_head.embedding.value)

    def set_embed_and_head(
        self,
        embed_weight: jax.Array | None = None,
        head_weight: jax.Array | None = None,
    ) -> None:
        """Set embedding and lm_head weights."""
        if embed_weight is not None:
            self.model.embed_tokens.embedding.value = embed_weight
        if head_weight is not None:
            self.lm_head.embedding.value = head_weight

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ):
        """Forward pass for text generation.

        Args:
            forward_batch: Batch information including input_ids and positions
            token_to_kv_pool: KV cache for inference
            logits_metadata: Metadata for logits processing

        Returns:
            Tuple of (logits, layers_kv_fused, layers_callback_flag, None)
        """
        hidden_states, layers_kv_fused, layers_callback_flag = self.model(
            forward_batch, token_to_kv_pool
        )

        if not getattr(self.text_config, "tie_word_embeddings", False):
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)

        return output, layers_kv_fused, layers_callback_flag, None
