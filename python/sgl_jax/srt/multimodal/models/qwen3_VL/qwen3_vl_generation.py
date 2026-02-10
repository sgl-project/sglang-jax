"""Qwen3-VL Generation Model for SGLang-JAX.

Self-contained text decoder — no dependency on qwen2.py. Uses sglang-jax
sharded layers (LinearBase, Embed, RMSNorm, RadixAttention) with Qwen3-VL
specific features:
  - Q/K normalization (RMSNorm on query/key before RoPE)
  - M-RoPE (Multimodal Rotary Position Embeddings)

Reference: qwen3-vl/q3vljax/qwen3_vl/modeling.py
"""

import logging

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, apply_rotary_emb
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


# =============================================================================
# M-RoPE utilities
# =============================================================================


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


# =============================================================================
# Config helper
# =============================================================================


def _get_mrope_section(config) -> list[int]:
    """Extract mrope_section from either our config or HuggingFace config.

    Our Qwen3VLTextConfig:  config.mrope_section = (24, 20, 20)
    HuggingFace config:     config.rope_scaling = {'mrope_section': [24, 20, 20], ...}
    """
    # Our config: direct attribute
    if hasattr(config, "mrope_section") and config.mrope_section:
        return list(config.mrope_section)
    # HF config: nested in rope_scaling dict
    rope_scaling = getattr(config, "rope_scaling", None) or {}
    return rope_scaling.get("mrope_section", [24, 20, 20])


# =============================================================================
# Model components (self-contained, using sglang-jax sharded layers)
# =============================================================================


class Qwen3VL_MLP(nnx.Module):
    """SiLU-gated MLP for Qwen3-VL text decoder."""

    def __init__(self, config, mesh, layer_id: int = 0, dtype=jnp.bfloat16):
        self.gate_proj = LinearBase(
            input_size=config.hidden_size,
            output_size=config.intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
        )
        self.up_proj = LinearBase(
            input_size=config.hidden_size,
            output_size=config.intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
        )
        self.down_proj = LinearBase(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            kernel_axes=("tensor", None),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        gate, _ = self.gate_proj(hidden_states)
        up, _ = self.up_proj(hidden_states)
        output, _ = self.down_proj(jax.nn.silu(gate) * up)
        return output


class Qwen3VL_Attention(nnx.Module):
    """Qwen3-VL text decoder attention with Q/K norms, GQA, and M-RoPE."""

    def __init__(self, config, mesh, layer_id: int = 0, dtype=jnp.bfloat16):
        self.layer_id = layer_id
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.q_head_num = config.num_attention_heads
        self.kv_head_num = config.num_key_value_heads
        self.q_size = self.q_head_num * self.head_dim
        self.kv_size = self.kv_head_num * self.head_dim
        self.scaling = self.head_dim**-0.5

        use_bias = getattr(config, "attention_bias", False)

        self.q_proj = LinearBase(
            input_size=config.hidden_size,
            output_size=self.q_size,
            use_bias=use_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.k_proj = LinearBase(
            input_size=config.hidden_size,
            output_size=self.kv_size,
            use_bias=use_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.v_proj = LinearBase(
            input_size=config.hidden_size,
            output_size=self.kv_size,
            use_bias=use_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.o_proj = LinearBase(
            input_size=self.q_size,
            output_size=config.hidden_size,
            use_bias=False,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
        )

        # Qwen3-VL specific: Q/K normalization before RoPE
        self.q_norm = RMSNorm(
            self.head_dim,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
        )
        self.k_norm = RMSNorm(
            self.head_dim,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
        )

        # M-RoPE
        mrope_section = _get_mrope_section(config)
        rope_theta = getattr(config, "rope_theta", 5_000_000)

        self.rotary_emb = MRotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position_embeddings=32768,
            base=rope_theta,
            is_neox_style=True,
            dtype=dtype,
            mrope_section=mrope_section,
        )

        self.attn = RadixAttention(
            num_heads=self.q_head_num,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.kv_head_num,
            layer_id=layer_id,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, jax.Array]:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        q = q.reshape(-1, self.q_head_num, self.head_dim)
        k = k.reshape(-1, self.kv_head_num, self.head_dim)
        v = v.reshape(-1, self.kv_head_num, self.head_dim)

        # Qwen3-VL specific: normalize Q/K before RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = self.rotary_emb(positions, q, k)
        attn_output, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool)

        output, _ = self.o_proj(attn_output)
        return output, kv_fused


class Qwen3VL_DecoderLayer(nnx.Module):
    """Single decoder layer for Qwen3-VL."""

    def __init__(self, config, mesh, layer_id: int = 0, dtype=jnp.bfloat16):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3VL_Attention(
            config=config,
            mesh=mesh,
            layer_id=layer_id,
            dtype=dtype,
        )
        self.mlp = Qwen3VL_MLP(
            config=config,
            mesh=mesh,
            layer_id=layer_id,
            dtype=dtype,
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: jax.Array | None = None,
    ):
        layer_callback_flag = []

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states, kv_fused = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )

        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual, kv_fused, layer_callback_flag


class Qwen3_VL_Model(nnx.Module):
    """Qwen3-VL text decoder with M-RoPE (self-contained, no Qwen2 dependency)."""

    def __init__(self, config, mesh, dtype=jnp.bfloat16):
        self.config = config

        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            kernel_axes=("tensor", None),
            param_dtype=dtype,
            mesh=mesh,
        )

        self.layers = nnx.data(
            [
                Qwen3VL_DecoderLayer(
                    config=config,
                    layer_id=i,
                    dtype=dtype,
                    mesh=mesh,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
        )

        # Store mrope_section for position routing
        self._mrope_section = _get_mrope_section(config)

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


# =============================================================================
# Top-level generation model
# =============================================================================


class Qwen3_VL_Generation(nnx.Module):
    """Qwen3-VL model for conditional generation.

    Self-contained implementation (no Qwen2 dependency).
    Accepts either our Qwen3VLConfig or a HuggingFace config with .text_config.

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
        self.dtype = dtype or jnp.bfloat16

        # Extract text config (works with both our Qwen3VLConfig and HF config)
        self.text_config = getattr(config, "text_config", None) or config

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
        """Create weight mappings for text decoder.

        HF safetensors keys use ``model.language_model.`` prefix;
        our JAX model paths use ``model.`` (no language_model).
        """
        mappings = {
            "model.language_model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.language_model.norm.weight": WeightMapping(
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
        prefix = f"model.language_model.layers.{layer_idx}"
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
