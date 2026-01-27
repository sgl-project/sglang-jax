"""MiMo Audio Backbone model implementation for sglang-jax."""
# Forced sync update

from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, RotaryEmbedding
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.qwen2 import Qwen2DecoderLayer  # Import Qwen2 components
from sgl_jax.srt.multimodal.configs.audio.mimo_audio_backbone_config import (
    MiMoAudioArguments,
    MiMoAudioBackboneConfig,
    MiMoSamplerConfig,
)
from sgl_jax.srt.multimodal.models.mimo_audio.mimo_audio_backbone_weights_mapping import (
    to_mappings,
)
from sgl_jax.srt.utils.weight_utils import WeightLoader


@dataclass
class Qwen2ConfigAdapter:
    """Adapter to make MiMoAudioBackboneConfig compatible with Qwen2DecoderLayer."""
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    head_dim: int = None
    rope_scaling: dict = None


class MiMoAudioMLP(nnx.Module):
    """SwiGLU MLP for MiMo Audio model."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id

        self.gate_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
        )

        self.up_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
        )

        self.down_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            kernel_axes=("tensor", None),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
        )

        self.act_fn = jax.nn.silu

    def __call__(self, hidden_states: jax.Array):
        a1, _ = self.gate_proj(hidden_states)
        a2, _ = self.up_proj(hidden_states)
        intermediate_parallel = a2 * self.act_fn(a1)
        output, _ = self.down_proj(intermediate_parallel)
        return output


class MiMoAudioAttention(nnx.Module):
    """Attention layer for MiMo Audio model.

    Supports two modes:
    - RadixAttention: Used when token_to_kv_pool is provided (main model).
    - Standard Attention: Used when token_to_kv_pool is None (Patch Encoder/Decoder).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        use_bias: bool = True,
        use_causal_mask: bool = True,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.head_dim = head_dim
        self.q_head_num = num_heads
        self.kv_head_num = num_kv_heads
        self.use_causal_mask = use_causal_mask
        self.scaling = head_dim**-0.5

        # Q/K/V/O projections
        self.q_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_heads * head_dim,
            use_bias=use_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.k_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_kv_heads * head_dim,
            use_bias=use_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.v_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_kv_heads * head_dim,
            use_bias=use_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.o_proj = LinearBase(
            input_size=num_heads * head_dim,
            output_size=hidden_size,
            use_bias=False,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
        )

        # Rotary Position Embedding
        self.rotary_emb = RotaryEmbedding(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            dtype=dtype,
        )

        # RadixAttention for KV cache mode
        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            scaling=self.scaling,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        positions: jax.Array,
        forward_batch: Optional[ForwardBatch] = None,
        token_to_kv_pool: Optional[KVCache] = None,
    ) -> Tuple[jax.Array, Optional[jax.Array]]:
        """Forward pass with two branches: RadixAttention or Standard Attention.

        Branch 1 (RadixAttention): Used when token_to_kv_pool is provided (main model).
        Branch 2 (Standard): Used when token_to_kv_pool is None (Patch Encoder/Decoder).
        """
        # Branch 1: RadixAttention (for main model with KV cache)
        if token_to_kv_pool is not None:
            # DEBUG: Check hidden_states before Q projection (Layer 0 only)
            if self.layer_id == 0:
                jax.debug.print(
                    "Attn0 input_hidden: min={h_min}, max={h_max}, has_nan={has_nan}",
                    h_min=jnp.min(hidden_states),
                    h_max=jnp.max(hidden_states),
                    has_nan=jnp.any(jnp.isnan(hidden_states))
                )
                # Check Q projection weights
                q_weight = self.q_proj.weight.value
                jax.debug.print(
                    "Attn0 q_proj_weight: shape={shape}, min={w_min}, max={w_max}, mean={w_mean}",
                    shape=q_weight.shape,
                    w_min=jnp.min(q_weight),
                    w_max=jnp.max(q_weight),
                    w_mean=jnp.mean(q_weight)
                )

            q, _ = self.q_proj(hidden_states)
            k, _ = self.k_proj(hidden_states)
            v, _ = self.v_proj(hidden_states)

            # DEBUG: Check Q/K/V value ranges after projection (Layer 0 only)
            if self.layer_id == 0:
                jax.debug.print(
                    "Attn0 after_proj: q_min={q_min}, q_max={q_max}, k_min={k_min}, k_max={k_max}",
                    q_min=jnp.min(q),
                    q_max=jnp.max(q),
                    k_min=jnp.min(k),
                    k_max=jnp.max(k)
                )

            q = q.reshape(-1, self.q_head_num, self.head_dim)
            k = k.reshape(-1, self.kv_head_num, self.head_dim)
            v = v.reshape(-1, self.kv_head_num, self.head_dim)

            # DEBUG: Check Q/K/V before RoPE (Layer 0 only)
            if self.layer_id == 0:
                jax.debug.print(
                    "Attn0 before_rope: q_nan={q_nan}, k_nan={k_nan}, v_nan={v_nan}",
                    q_nan=jnp.any(jnp.isnan(q)),
                    k_nan=jnp.any(jnp.isnan(k)),
                    v_nan=jnp.any(jnp.isnan(v))
                )

            q, k = self.rotary_emb(positions, q, k)

            # DEBUG: Check Q/K after RoPE (Layer 0 only)
            if self.layer_id == 0:
                jax.debug.print(
                    "Attn0 after_rope: q_nan={q_nan}, k_nan={k_nan}, positions={positions}",
                    q_nan=jnp.any(jnp.isnan(q)),
                    k_nan=jnp.any(jnp.isnan(k)),
                    positions=positions
                )

            attn_output, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool)

            # DEBUG: Check attention output (Layer 0 only)
            if self.layer_id == 0:
                jax.debug.print(
                    "Attn0 after_radix_attn: output_nan={output_nan}, output_inf={output_inf}",
                    output_nan=jnp.any(jnp.isnan(attn_output)),
                    output_inf=jnp.any(jnp.isinf(attn_output))
                )

            output, _ = self.o_proj(attn_output)
            return output, kv_fused

        # Branch 2: Standard Attention (Stateless / Patch Encoder/Decoder)
        batch_size, seq_len, _ = hidden_states.shape

        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        q = q.reshape(batch_size, seq_len, self.q_head_num, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.kv_head_num, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.kv_head_num, self.head_dim)

        # RoPE
        positions_flat = positions.reshape(-1) if positions.ndim > 1 else jnp.tile(positions, batch_size)
        q_flat = q.reshape(-1, self.q_head_num, self.head_dim)
        k_flat = k.reshape(-1, self.kv_head_num, self.head_dim)
        q_rot, k_rot = self.rotary_emb(positions_flat, q_flat, k_flat)
        q = q_rot.reshape(batch_size, seq_len, self.q_head_num, self.head_dim)
        k = k_rot.reshape(batch_size, seq_len, self.kv_head_num, self.head_dim)

        # GQA: Repeat KV heads if needed
        if self.kv_head_num < self.q_head_num:
            k = jnp.repeat(k, self.q_head_num // self.kv_head_num, axis=2)
            v = jnp.repeat(v, self.q_head_num // self.kv_head_num, axis=2)

        # Transpose to [B, H, T, D]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Compute attention scores
        attn_weights = jnp.einsum("bhqd,bhkd->bhqk", q, k) * self.scaling

        # Apply causal mask if needed
        if self.use_causal_mask:
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
            attn_weights = jnp.where(causal_mask[None, None, :, :], attn_weights, -jnp.inf)

        attn_weights = jax.nn.softmax(attn_weights, axis=-1).astype(v.dtype)
        attn_output = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        output, _ = self.o_proj(attn_output)
        return output, None


class MiMoAudioDecoderLayer(nnx.Module):
    """Transformer decoder layer for MiMo Audio model."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        max_position_embeddings: int,
        rope_theta: float,
        rms_norm_eps: float,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        use_bias: bool = True,
        use_causal_mask: bool = True,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.self_attn = MiMoAudioAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            mesh=mesh,
            layer_id=layer_id,
            use_bias=use_bias,
            use_causal_mask=use_causal_mask,
            dtype=dtype,
        )
        self.mlp = MiMoAudioMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            mesh=mesh,
            layer_id=layer_id,
            dtype=dtype,
        )
        self.input_layernorm = RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        positions: jax.Array,
        forward_batch: Optional[ForwardBatch] = None,
        token_to_kv_pool: Optional[KVCache] = None,
        residual: jax.Array | None = None,
    ) -> Tuple[jax.Array, jax.Array, Optional[jax.Array], list]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        # DEBUG: Check residual and LayerNorm output before attention (Layer 0 only)
        if self.self_attn.layer_id == 0:
            jax.debug.print(
                "Layer0 residual_before_attn: min={r_min}, max={r_max}",
                r_min=jnp.min(residual),
                r_max=jnp.max(residual)
            )
            jax.debug.print(
                "Layer0 after_input_layernorm: min={h_min}, max={h_max}, has_nan={has_nan}",
                h_min=jnp.min(hidden_states),
                h_max=jnp.max(hidden_states),
                has_nan=jnp.any(jnp.isnan(hidden_states))
            )

        attn_output, kv_fused = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )

        # DEBUG: Check attention output for layer 0 only
        if self.self_attn.layer_id == 0:
            jax.debug.print(
                "Layer0 attn_output: has_nan={has_nan}, has_inf={has_inf}",
                has_nan=jnp.any(jnp.isnan(attn_output)),
                has_inf=jnp.any(jnp.isinf(attn_output))
            )

        hidden_states = attn_output + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)

        # DEBUG: Check MLP output for layer 0 only
        if self.self_attn.layer_id == 0:
            jax.debug.print(
                "Layer0 mlp_output: has_nan={has_nan}, has_inf={has_inf}",
                has_nan=jnp.any(jnp.isnan(mlp_output)),
                has_inf=jnp.any(jnp.isinf(mlp_output))
            )

        return mlp_output, residual, kv_fused, []


class MiMoAudioTransformer(nnx.Module):
    """Reusable transformer model for MiMo Audio (main/local/input_local).

    For the main model (has_embedder=True, use_kv_cache=True), uses Qwen2DecoderLayer
    for better stability. For patch encoder/decoder, uses MiMoAudioDecoderLayer.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        max_position_embeddings: int,
        rope_theta: float,
        rms_norm_eps: float,
        vocab_size: int,
        mesh: jax.sharding.Mesh,
        use_bias: bool = True,
        use_causal_mask: bool = True,
        has_embedder: bool = True,
        use_qwen2_layers: bool = False,  # New: use Qwen2DecoderLayer
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.has_embedder = has_embedder
        self.hidden_size = hidden_size
        self.use_qwen2_layers = use_qwen2_layers

        if has_embedder:
            self.embed_tokens = Embed(
                num_embeddings=vocab_size,
                features=hidden_size,
                dtype=dtype,
                kernel_axes=("tensor", None),
                param_dtype=dtype,
                mesh=mesh,
            )

        if use_qwen2_layers:
            # Use Qwen2DecoderLayer for main model (more stable)
            config = Qwen2ConfigAdapter(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                intermediate_size=intermediate_size,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                rms_norm_eps=rms_norm_eps,
                head_dim=head_dim,
            )
            self.layers = nnx.List([
                Qwen2DecoderLayer(
                    config=config,
                    mesh=mesh,
                    layer_id=i,
                    dtype=dtype,
                )
                for i in range(num_layers)
            ])
        else:
            # Use MiMoAudioDecoderLayer for patch encoder/decoder
            self.layers = nnx.List([
                MiMoAudioDecoderLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                    max_position_embeddings=max_position_embeddings,
                    rope_theta=rope_theta,
                    rms_norm_eps=rms_norm_eps,
                    mesh=mesh,
                    layer_id=i,
                    use_bias=use_bias,
                    use_causal_mask=use_causal_mask,
                    dtype=dtype,
                )
                for i in range(num_layers)
            ])

        self.norm = RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
        )

    def __call__(
        self,
        input_ids_or_embeds: jax.Array,
        positions: jax.Array,
        forward_batch: Optional[ForwardBatch] = None,
        token_to_kv_pool: Optional[KVCache] = None,
    ) -> Tuple[jax.Array, list, list]:
        """
        Args:
            input_ids_or_embeds: Embeddings [Total_Tokens, hidden_size] or [B, T, H]
            positions: Position IDs
            forward_batch: ForwardBatch info (Optional)
            token_to_kv_pool: KVCache pool (Optional)

        Returns:
            hidden_states: Output embeddings
            layers_kv_fused: List of KV outputs (None for standard attention)
            layers_callback_flag: Callback flags
        """
        if self.has_embedder and jnp.issubdtype(input_ids_or_embeds.dtype, jnp.integer):
            # Rank 2 [B, T] or Rank 1 [Total]
            hidden_states = self.embed_tokens(input_ids_or_embeds)
        else:
            hidden_states = input_ids_or_embeds

        residual = None
        layers_kv_fused = []
        layers_callback_flag = []

        for layer_idx, layer in enumerate(self.layers):
            if self.use_qwen2_layers:
                # Qwen2DecoderLayer: (positions, hidden_states, forward_batch, kv_pool, residual)
                hidden_states, residual, kv_fused, callback_flag = layer(
                    positions,
                    hidden_states,
                    forward_batch,
                    token_to_kv_pool,
                    residual,
                )
            else:
                # MiMoAudioDecoderLayer: (hidden_states, positions, forward_batch, kv_pool, residual)
                hidden_states, residual, kv_fused, callback_flag = layer(
                    hidden_states,
                    positions,
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


class MiMoAudioForCausalLM(nnx.Module):
    """MiMo Audio model for causal language modeling with audio token generation."""

    def __init__(
        self,
        config: MiMoAudioBackboneConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        args: MiMoAudioArguments | None = None,
    ):
        self.config = config
        self.args = args or MiMoAudioArguments()
        self.mesh = mesh
        self.dtype = dtype

        # Main Qwen2 model (36 layers) - use Qwen2DecoderLayer for stability
        self.model = MiMoAudioTransformer(
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            vocab_size=config.vocab_size,
            mesh=mesh,
            use_bias=config.attention_bias,
            use_causal_mask=True,
            has_embedder=True,
            use_qwen2_layers=True,  # Use Qwen2DecoderLayer for main model
            dtype=dtype,
        )

        # Patch decoder (16 layers, no embedder) - generates audio tokens for each group
        self.patch_decoder = MiMoAudioTransformer(
            hidden_size=config.local_dim,
            num_layers=config.local_layers,
            num_heads=config.local_attn_heads,
            num_kv_heads=config.local_attn_heads,
            head_dim=config.local_head_dim,
            intermediate_size=config.local_ffn_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            vocab_size=config.vocab_size,
            mesh=mesh,
            use_bias=config.attention_bias,
            use_causal_mask=True,
            has_embedder=False,
            dtype=dtype,
        )

        # Patch encoder (6 layers, no embedder, bidirectional attention) - encodes speech embeddings
        self.patch_encoder = MiMoAudioTransformer(
            hidden_size=config.input_local_dim,
            num_layers=config.input_local_layers,
            num_heads=config.local_attn_heads,
            num_kv_heads=config.local_attn_heads,
            head_dim=config.local_head_dim,
            intermediate_size=config.local_ffn_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            vocab_size=config.vocab_size,
            mesh=mesh,
            use_bias=config.attention_bias,
            use_causal_mask=False,  # bidirectional attention
            has_embedder=False,
            dtype=dtype,
        )

        # LM head for text
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )

        # LM heads for audio channels (8 channels)
        # Note: speech_vocab_sizes (e.g. 1025, 129) are not divisible by TP, so no sharding
        self.patch_decoder_lm_heads = nnx.List([
            LinearBase(
                input_size=config.local_dim,
                output_size=config.speech_vocab_sizes[i],
                use_bias=False,
                kernel_axes=(None, None),
                params_dtype=dtype,
                mesh=mesh,
            )
            for i in range(config.audio_channels)
        ])

        # Speech embeddings (8 channels)
        # Note: speech_vocab_sizes not divisible by TP, so no sharding on vocab axis
        self.speech_embeddings = nnx.List([
            Embed(
                num_embeddings=config.speech_vocab_sizes[i],
                features=config.input_local_dim,
                dtype=dtype,
                kernel_axes=(None, None),
                param_dtype=dtype,
                mesh=mesh,
            )
            for i in range(config.audio_channels)
        ])

        # Projection layers
        self.speech_group_downcast = LinearBase(
            input_size=config.input_local_dim * config.group_size,
            output_size=config.hidden_size,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )

        self.hidden_states_downcast = LinearBase(
            input_size=config.hidden_size,
            output_size=config.local_dim,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )

        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=self.mesh)

    def load_weights(self, model_config: ModelConfig):
        """Load weights from safetensors file."""
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        loader.load_weights_from_safetensors(to_mappings(self.config))

    def apply_patch_encoder(
        self,
        speech_embeddings: jax.Array,
    ) -> jax.Array:
        """Apply patch encoder to speech embeddings.

        Each group is processed independently with bidirectional attention.

        Args:
            speech_embeddings: [B, T_groups, group_size, hidden_size]

        Returns:
            Processed embeddings: [B, T_groups, group_size, hidden_size]
        """
        B, T_groups, group_size, hidden_size = speech_embeddings.shape

        # Reshape for processing: [B * T_groups, group_size, hidden_size]
        input_embeddings = speech_embeddings.reshape(B * T_groups, group_size, hidden_size)

        # Create position IDs
        positions = jnp.arange(group_size)

        # Process through patch encoder
        output, _, _ = self.patch_encoder(input_embeddings, positions)

        # Reshape back: [B, T_groups, group_size, hidden_size]
        return output.reshape(B, T_groups, group_size, hidden_size)

    def _prepare_input_embeds(
        self,
        input_ids: jax.Array,
    ) -> jax.Array:
        """Prepare input embeddings from interleaved text and speech tokens.

        Args:
            input_ids: [B, 1 + audio_channels, seq_len] where first channel is text

        Returns:
            Combined embeddings: [B, T_groups, hidden_size]
        """
        B = input_ids.shape[0]

        # Extract text input IDs (downsampled by group_size)
        text_input_ids = input_ids[:, 0, :: self.config.group_size]  # [B, T_groups]

        # Extract speech input IDs and reshape
        speech_input_ids = input_ids[:, 1:, :]  # [B, audio_channels, seq_len]
        speech_input_ids = speech_input_ids.reshape(
            B, self.config.audio_channels, -1, self.config.group_size
        ).transpose(0, 2, 1, 3)  # [B, T_groups, audio_channels, group_size]

        # Identify speech positions
        is_speech = text_input_ids == self.args.empty_idx  # [B, T_groups]

        # Initialize speech embeddings
        T_groups = is_speech.shape[1]
        speech_embeds = jnp.zeros(
            (B, T_groups, self.config.group_size, self.config.input_local_dim),
            dtype=self.dtype,
        )

        # Accumulate embeddings from all audio channels
        for idx in range(self.config.audio_channels):
            cur_empty = self.config.speech_empty_ids[idx]
            cur_embed = self.speech_embeddings[idx]
            cur_speech_ids = speech_input_ids[:, :, idx, :]  # [B, T_groups, group_size]

            # DEBUG: Print token ID range for each channel
            jax.debug.print(
                "Channel {idx}: min_id={min_id}, max_id={max_id}, vocab_size={vocab_size}",
                idx=idx,
                min_id=jnp.min(cur_speech_ids),
                max_id=jnp.max(cur_speech_ids),
                vocab_size=self.config.speech_vocab_sizes[idx]
            )

            cur_speech_embeds = cur_embed(cur_speech_ids)  # [B, T_groups, group_size, dim]

            # Mask out empty tokens
            cur_mask = cur_speech_ids == cur_empty
            cur_speech_embeds = cur_speech_embeds * ~cur_mask[..., None]
            speech_embeds = speech_embeds + cur_speech_embeds

        # DEBUG: Check speech_embeds for NaN after accumulation
        jax.debug.print(
            "speech_embeds stats: has_nan={has_nan}, has_inf={has_inf}",
            has_nan=jnp.any(jnp.isnan(speech_embeds)),
            has_inf=jnp.any(jnp.isinf(speech_embeds))
        )

        # Mask out non-speech positions
        speech_embeds = speech_embeds * is_speech[:, :, None, None]

        # Apply patch encoder
        speech_embeds = self.apply_patch_encoder(speech_embeds)
        jax.debug.print(
            "after_patch_encoder: has_nan={has_nan}, has_inf={has_inf}",
            has_nan=jnp.any(jnp.isnan(speech_embeds)),
            has_inf=jnp.any(jnp.isinf(speech_embeds))
        )

        # Mask again after transformer
        speech_embeds = speech_embeds * is_speech[:, :, None, None]

        # Downsample speech embeddings: [B, T_groups, group_size * dim] -> [B, T_groups, hidden_size]
        speech_grouped_embeds, _ = self.speech_group_downcast(
            speech_embeds.reshape(B, T_groups, -1)
        )
        jax.debug.print(
            "speech_grouped_embeds: has_nan={has_nan}, has_inf={has_inf}",
            has_nan=jnp.any(jnp.isnan(speech_grouped_embeds)),
            has_inf=jnp.any(jnp.isinf(speech_grouped_embeds))
        )

        # Get text embeddings
        text_input_ids_safe = jnp.where(text_input_ids == -100, 0, text_input_ids)
        text_embeds = self.model.embed_tokens(text_input_ids_safe)
        jax.debug.print(
            "text_embeds: has_nan={has_nan}, has_inf={has_inf}",
            has_nan=jnp.any(jnp.isnan(text_embeds)),
            has_inf=jnp.any(jnp.isinf(text_embeds))
        )

        # Zero out embeddings for empty/masked tokens
        text_zero_mask = (text_input_ids == self.args.empty_idx) | (text_input_ids == -100)
        text_embeds = text_embeds * ~text_zero_mask[..., None]

        # Combine text and speech embeddings
        result = text_embeds + speech_grouped_embeds
        jax.debug.print(
            "combined_embeds: has_nan={has_nan}, has_inf={has_inf}",
            has_nan=jnp.any(jnp.isnan(result)),
            has_inf=jnp.any(jnp.isinf(result))
        )
        return result

    def forward_simple(
        self,
        input_ids: jax.Array,
        positions: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """Simplified forward pass using standard attention (no KV cache).

        This method is used when RadixAttention/KV cache pool is not available,
        such as in multi-stage pipelines where mesh context is managed externally.

        Args:
            input_ids: [B, 1 + audio_channels, seq_len]
            positions: Position IDs [T_groups]

        Returns:
            text_logits: [B, T_groups, vocab_size]
            local_hidden_states: [B, 1, local_dim]
        """
        # Prepare input embeddings (MiMo specific)
        inputs_embeds = self._prepare_input_embeds(input_ids)

        B, T_groups, H = inputs_embeds.shape

        # Forward through main transformer using Branch 2 (no KV cache)
        # Passing None for forward_batch and token_to_kv_pool triggers simple attention
        hidden_states, _, _ = self.model(inputs_embeds, positions, None, None)

        # Get text logits using ParallelLMHead's embedding matrix
        hidden_states_promoted, embedding = self.lm_head.promote_dtype(hidden_states)
        text_logits = jnp.matmul(hidden_states_promoted, embedding.T)  # [B, T_groups, vocab_size]

        # Downcast hidden states for local transformer
        local_hidden_states, _ = self.hidden_states_downcast(
            hidden_states[:, -1:, :]
        )  # [B, 1, local_dim]

        return text_logits, local_hidden_states

    def forward(
        self,
        input_ids: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ) -> Tuple[jax.Array, jax.Array, None, list, list]:
        """Forward pass through main transformer.

        Args:
            input_ids: [B, 1 + audio_channels, seq_len]
            forward_batch: Batch metadata
            token_to_kv_pool: KV Cache pool
            logits_metadata: Metadata for logits processing

        Returns:
            text_logits: LogitsProcessorOutput
            local_hidden_states: [B, 1, local_dim]
            cache: None (Managed by RadixAttention)
            layers_kv_fused: KV outputs
            layers_callback_flag: Callback flags
        """
        # Prepare input embeddings (MiMo specific)
        inputs_embeds = self._prepare_input_embeds(input_ids)
        jax.debug.print(
            "FORWARD inputs_embeds: has_nan={has_nan}, has_inf={has_inf}",
            has_nan=jnp.any(jnp.isnan(inputs_embeds)),
            has_inf=jnp.any(jnp.isinf(inputs_embeds))
        )

        # Flatten embeddings for RadixAttention: [Total_Tokens, H]
        B, T_groups, H = inputs_embeds.shape
        inputs_embeds_flat = inputs_embeds.reshape(-1, H)

        # Forward through main transformer
        hidden_states, layers_kv_fused, layers_callback_flag = self.model(
            inputs_embeds_flat, forward_batch.positions, forward_batch, token_to_kv_pool
        )
        jax.debug.print(
            "FORWARD hidden_states_after_model: has_nan={has_nan}, has_inf={has_inf}",
            has_nan=jnp.any(jnp.isnan(hidden_states)),
            has_inf=jnp.any(jnp.isinf(hidden_states))
        )

        # Get logits for the last positions using LogitsProcessor
        text_logits = self.logits_processor(hidden_states, self.lm_head, logits_metadata)

        # Reshape hidden_states back to [B, T_groups, H] to get local_hidden_states
        hidden_states = hidden_states.reshape(B, T_groups, H)

        # Downcast hidden states for local transformer
        local_hidden_states, _ = self.hidden_states_downcast(
            hidden_states[:, -1:, :]
        )  # [B, 1, local_dim]

        return text_logits, local_hidden_states, None, layers_kv_fused, layers_callback_flag

    def patch_decode(
        self,
        local_embeds: jax.Array,
        key: jax.Array,
        sampler_config: Optional[MiMoSamplerConfig] = None,
    ) -> jax.Array:
        """Generate audio tokens for one group using patch decoder.

        Each call is independent - cache is not shared between patches.

        Args:
            local_embeds: [B, 1, local_dim]
            key: Random key for sampling
            sampler_config: Sampling configuration

        Returns:
            local_tokens: [B, group_size, audio_channels]
        """
        if sampler_config is None:
            sampler_config = MiMoSamplerConfig()

        B = local_embeds.shape[0]
        delay_iters = self.config.group_size + max(self.config.delay_pattern)

        local_tokens = jnp.zeros(
            (B, self.config.group_size, self.config.audio_channels), dtype=jnp.int32
        )

        for t in range(delay_iters):
            positions = jnp.array([t])

            # Forward through patch decoder without cache (simpler, avoids sharding issues)
            hidden_state, _, _ = self.patch_decoder(local_embeds, positions)

            next_local_embeds = jnp.zeros_like(local_embeds)

            for idx in range(self.config.audio_channels):
                cur_start = self.config.delay_pattern[idx]
                cur_end = cur_start + self.config.group_size
                cur_empty = self.config.speech_empty_ids[idx]

                if cur_start <= t < cur_end:
                    # Get logits for this channel
                    cur_lm_head = self.patch_decoder_lm_heads[idx]
                    cur_logits, _ = cur_lm_head(hidden_state[:, -1, :])  # [B, vocab_size]

                    # Mask out empty token
                    cur_logits = cur_logits.at[:, cur_empty].set(-jnp.inf)

                    # Sample token
                    key, subkey = jax.random.split(key)
                    if sampler_config.do_sample:
                        # Apply temperature
                        cur_logits = cur_logits / sampler_config.temperature
                        cur_token = jax.random.categorical(subkey, cur_logits)
                    else:
                        cur_token = jnp.argmax(cur_logits, axis=-1)

                    # Store token
                    local_tokens = local_tokens.at[:, t - cur_start, idx].set(cur_token)

                    # Get embedding for next step
                    cur_input_embed = self.speech_embeddings[idx](cur_token[:, None])
                    next_local_embeds = next_local_embeds + cur_input_embed

            local_embeds = next_local_embeds

        return local_tokens


EntryClass = MiMoAudioForCausalLM
