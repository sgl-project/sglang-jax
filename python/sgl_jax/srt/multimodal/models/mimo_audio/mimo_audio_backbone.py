"""MiMo Audio Backbone model implementation for sglang-jax."""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, RotaryEmbedding
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.multimodal.configs.audio.mimo_audio_backbone_config import (
    MiMoAudioArguments,
    MiMoAudioBackboneConfig,
    MiMoSamplerConfig,
)
from sgl_jax.srt.multimodal.models.mimo_audio.mimo_audio_backbone_weights_mapping import (
    to_mappings,
)
from sgl_jax.srt.utils.weight_utils import WeightLoader


class MiMoAudioMLP(nnx.Module):
    """MLP layer for MiMo Audio model."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
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

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        gate, _ = self.gate_proj(hidden_states)
        up, _ = self.up_proj(hidden_states)
        intermediate = up * self.act_fn(gate)
        output, _ = self.down_proj(intermediate)
        return output


class MiMoAudioAttention(nnx.Module):
    """Standard attention layer for MiMo Audio model.

    Unlike RadixAttention used in LLM serving, this uses standard attention
    since audio token generation doesn't require KV cache sharing.

    Supports two cache modes:
    - Fixed-size cache with position index (for patch_decoder with fixed steps)
    - Dynamic concatenation cache (for main transformer)
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
        use_bias: bool = True,
        use_causal_mask: bool = True,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.use_causal_mask = use_causal_mask
        self.scaling = head_dim**-0.5

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

        self.rotary_emb = RotaryEmbedding(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            dtype=dtype,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        positions: jax.Array,
        attention_mask: Optional[jax.Array] = None,
        cache: Optional[Tuple[jax.Array, jax.Array, int]] = None,
    ) -> Tuple[jax.Array, Optional[Tuple[jax.Array, jax.Array, int]]]:
        """
        Args:
            hidden_states: [B, T, hidden_size]
            positions: [B, T] or [T]
            attention_mask: Optional attention mask
            cache: Optional (key_cache, value_cache, cache_pos) for fixed-size cache
                   where cache_pos is the current write position

        Returns:
            output: [B, T, hidden_size]
            new_cache: Updated cache tuple (k, v, new_pos)
        """
        batch_size, seq_len, _ = hidden_states.shape

        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        # Reshape to [B, T, num_heads, head_dim]
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply rotary embeddings
        q_flat = q.reshape(-1, self.num_heads, self.head_dim)
        k_flat = k.reshape(-1, self.num_kv_heads, self.head_dim)
        positions_flat = positions.reshape(-1) if positions.ndim > 1 else jnp.tile(positions, batch_size)

        q_flat, k_flat = self.rotary_emb(positions_flat, q_flat, k_flat)

        q = q_flat.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k_flat.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Handle KV cache
        if cache is not None:
            cached_k, cached_v, cache_pos = cache
            # Force k/v to be replicated to match cache sharding
            k = jnp.asarray(k)
            v = jnp.asarray(v)
            # Update cache at current position using fixed-size indexing
            cached_k = jax.lax.dynamic_update_slice(
                cached_k, k, (0, cache_pos, 0, 0)
            )
            cached_v = jax.lax.dynamic_update_slice(
                cached_v, v, (0, cache_pos, 0, 0)
            )
            new_cache_pos = cache_pos + seq_len
            # Use all cached KV up to current position for attention
            kv_len = new_cache_pos
            k_for_attn = jax.lax.dynamic_slice(
                cached_k, (0, 0, 0, 0), (batch_size, kv_len, self.num_kv_heads, self.head_dim)
            )
            v_for_attn = jax.lax.dynamic_slice(
                cached_v, (0, 0, 0, 0), (batch_size, kv_len, self.num_kv_heads, self.head_dim)
            )
            new_cache = (cached_k, cached_v, new_cache_pos)
        else:
            k_for_attn = k
            v_for_attn = v
            kv_len = seq_len
            new_cache = None

        # Repeat KV heads if needed (GQA)
        if self.num_kv_heads < self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k_for_attn = jnp.repeat(k_for_attn, repeat_factor, axis=2)
            v_for_attn = jnp.repeat(v_for_attn, repeat_factor, axis=2)

        # Compute attention: [B, num_heads, T_q, T_kv]
        q = q.transpose(0, 2, 1, 3)  # [B, num_heads, T, head_dim]
        k_for_attn = k_for_attn.transpose(0, 2, 1, 3)
        v_for_attn = v_for_attn.transpose(0, 2, 1, 3)

        attn_weights = jnp.einsum("bhqd,bhkd->bhqk", q, k_for_attn) * self.scaling

        # Apply causal mask if needed
        if self.use_causal_mask:
            # For cached attention, we need to mask based on absolute positions
            if cache is not None:
                # Query attends to all positions up to and including its own
                causal_mask = jnp.tril(jnp.ones((seq_len, kv_len), dtype=jnp.bool_), k=kv_len - seq_len)
            else:
                causal_mask = jnp.tril(jnp.ones((seq_len, kv_len), dtype=jnp.bool_))
            attn_weights = jnp.where(causal_mask[None, None, :, :], attn_weights, -jnp.inf)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = jax.nn.softmax(attn_weights, axis=-1).astype(v_for_attn.dtype)

        # Apply attention to values
        attn_output = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v_for_attn)

        # Reshape back: [B, T, num_heads * head_dim]
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        output, _ = self.o_proj(attn_output)
        return output, new_cache


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
            use_bias=use_bias,
            use_causal_mask=use_causal_mask,
            dtype=dtype,
        )
        self.mlp = MiMoAudioMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            mesh=mesh,
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
        attention_mask: Optional[jax.Array] = None,
        cache: Optional[Tuple[jax.Array, jax.Array, int]] = None,
    ) -> Tuple[jax.Array, Optional[Tuple[jax.Array, jax.Array, int]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, new_cache = self.self_attn(
            hidden_states, positions, attention_mask, cache
        )

        hidden_states = residual + hidden_states
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_cache


class MiMoAudioTransformer(nnx.Module):
    """Reusable transformer model for MiMo Audio (main/local/input_local)."""

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
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.has_embedder = has_embedder
        self.hidden_size = hidden_size

        if has_embedder:
            self.embed_tokens = Embed(
                num_embeddings=vocab_size,
                features=hidden_size,
                dtype=dtype,
                kernel_axes=("tensor", None),
                param_dtype=dtype,
                mesh=mesh,
            )

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
                use_bias=use_bias,
                use_causal_mask=use_causal_mask,
                dtype=dtype,
            )
            for _ in range(num_layers)
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
        attention_mask: Optional[jax.Array] = None,
        cache: Optional[list] = None,
    ) -> Tuple[jax.Array, Optional[list]]:
        """
        Args:
            input_ids_or_embeds: Token IDs [B, T] if has_embedder else embeddings [B, T, hidden_size]
            positions: Position IDs [B, T] or [T]
            attention_mask: Optional attention mask
            cache: Optional list of layer caches

        Returns:
            hidden_states: [B, T, hidden_size]
            new_cache: Updated cache list
        """
        if self.has_embedder and input_ids_or_embeds.ndim == 2 and jnp.issubdtype(input_ids_or_embeds.dtype, jnp.integer):
            hidden_states = self.embed_tokens(input_ids_or_embeds)
        else:
            hidden_states = input_ids_or_embeds

        new_cache = [] if cache is not None else None

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            hidden_states, layer_new_cache = layer(
                hidden_states, positions, attention_mask, layer_cache
            )
            if new_cache is not None:
                new_cache.append(layer_new_cache)

        hidden_states = self.norm(hidden_states)
        return hidden_states, new_cache

    def init_cache(
        self,
        batch_size: int,
        max_seq_len: int,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> list:
        """Initialize fixed-size KV cache for all layers.

        Args:
            batch_size: Batch size
            max_seq_len: Maximum sequence length (pre-allocated size)
            dtype: Data type for cache

        Returns:
            List of (k_cache, v_cache, cache_pos) tuples for each layer
        """
        cache = []
        for layer in self.layers:
            num_kv_heads = layer.self_attn.num_kv_heads
            head_dim = layer.self_attn.head_dim
            # Pre-allocate fixed-size cache
            k_cache = jnp.zeros((batch_size, max_seq_len, num_kv_heads, head_dim), dtype=dtype)
            v_cache = jnp.zeros((batch_size, max_seq_len, num_kv_heads, head_dim), dtype=dtype)
            cache_pos = 0  # Current write position
            cache.append((k_cache, v_cache, cache_pos))
        return cache


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

        # Main Qwen2 model (36 layers)
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
        self.lm_head = LinearBase(
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
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
        output, _ = self.patch_encoder(input_embeddings, positions)

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
            cur_speech_embeds = cur_embed(cur_speech_ids)  # [B, T_groups, group_size, dim]

            # Mask out empty tokens
            cur_mask = cur_speech_ids == cur_empty
            cur_speech_embeds = cur_speech_embeds * ~cur_mask[..., None]
            speech_embeds = speech_embeds + cur_speech_embeds

        # Mask out non-speech positions
        speech_embeds = speech_embeds * is_speech[:, :, None, None]

        # Apply patch encoder
        speech_embeds = self.apply_patch_encoder(speech_embeds)

        # Mask again after transformer
        speech_embeds = speech_embeds * is_speech[:, :, None, None]

        # Downsample speech embeddings: [B, T_groups, group_size * dim] -> [B, T_groups, hidden_size]
        speech_grouped_embeds, _ = self.speech_group_downcast(
            speech_embeds.reshape(B, T_groups, -1)
        )

        # Get text embeddings
        text_input_ids_safe = jnp.where(text_input_ids == -100, 0, text_input_ids)
        text_embeds = self.model.embed_tokens(text_input_ids_safe)

        # Zero out embeddings for empty/masked tokens
        text_zero_mask = (text_input_ids == self.args.empty_idx) | (text_input_ids == -100)
        text_embeds = text_embeds * ~text_zero_mask[..., None]

        # Combine text and speech embeddings
        return text_embeds + speech_grouped_embeds

    def forward(
        self,
        input_ids: jax.Array,
        cache: Optional[list] = None,
    ) -> Tuple[jax.Array, jax.Array, Optional[list]]:
        """Forward pass through main transformer.

        Args:
            input_ids: [B, 1 + audio_channels, seq_len]
            cache: Optional KV cache

        Returns:
            text_logits: [B, 1, vocab_size]
            local_hidden_states: [B, 1, local_dim]
            cache: Updated cache
        """
        text_input_ids = input_ids[:, 0, :: self.config.group_size]
        B, T_groups = text_input_ids.shape

        # Prepare input embeddings
        inputs_embeds = self._prepare_input_embeds(input_ids)

        # Create position IDs
        positions = jnp.arange(T_groups)

        # Forward through main transformer
        hidden_states, cache = self.model(
            inputs_embeds, positions, attention_mask=None, cache=cache
        )

        # Get text logits from last position
        text_logits, _ = self.lm_head(hidden_states[:, -1:, :])  # [B, 1, vocab_size]

        # Downcast hidden states for local transformer
        local_hidden_states, _ = self.hidden_states_downcast(
            hidden_states[:, -1:, :]
        )  # [B, 1, local_dim]

        return text_logits, local_hidden_states, cache

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

            # Forward through patch decoder without cache for now (simpler, avoids sharding issues)
            hidden_state, _ = self.patch_decoder(local_embeds, positions, cache=None)

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
