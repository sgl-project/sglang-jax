"""
Gemma Text Encoder for LTX-2

This module provides a text encoder based on Gemma models for the LTX-2 video generation pipeline.
It extracts text embeddings that are used to condition the diffusion transformer.
"""

import logging

import jax
import jax.numpy as jnp
from flax import nnx
from jax import ShapeDtypeStruct
from jax.sharding import NamedSharding, PartitionSpec
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import RotaryEmbedding
from sgl_jax.srt.layers.layernorm import GemmaRMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
from sgl_jax.srt.models.gemma2 import Gemma2Model
from sgl_jax.srt.multimodal.layers.attention.layer import simple_attention
from sgl_jax.srt.multimodal.layers.mlp import MLP
from sgl_jax.srt.multimodal.models.ltx2.diffusion.ltx2_dit import LTX2Attention, RMSNorm
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


class _BasicTransformerBlock1D(nnx.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mesh=None):
        super().__init__()
        self.attn1 = LTX2Attention(
            query_dim=dim,
            heads=heads,
            dim_head=dim_head,
            mesh=mesh,
            rope_type="interleaved",
        )
        self.ff = MLP(
            input_dim=dim,
            mlp_hidden_dim=dim * 4,
            output_dim=dim,
            act_type="gelu_pytorch_tanh",
            mesh=mesh
        )
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def __call__(self, x, mask, pe, attn_mask=None):
        nx = self.norm1(x)
        effective_mask = attn_mask if attn_mask is not None else mask
        attn_out = self.attn1(nx, context=nx, mask=effective_mask, pe=pe)
        x = x + attn_out
        nx = self.norm2(x)
        x = x + self.ff(nx)
        return x


class Embeddings1DConnector(nnx.Module):
    def __init__(self, num_layers=2, dim=3840, heads=30, dim_head=128, num_registers=128, mesh=None):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.num_registers = num_registers
        self.blocks = [_BasicTransformerBlock1D(dim, heads, dim_head, mesh) for _ in range(num_layers)]
        if hasattr(nnx, 'data'):
            self.blocks = nnx.data(self.blocks)
        self.norm_out = RMSNorm(dim)
        if self.num_registers > 0:
            self.learnable_registers = nnx.Param(jnp.zeros((num_registers, dim), dtype=jnp.float32))

    def _compute_pe(self, L):
        """Compute interleaved RoPE positional embeddings for the connector.

        Uses max_pos=[4096] from checkpoint config and f64 frequency grid.
        """
        theta = 10000.0
        n_freqs = self.dim // 2  # 1920
        max_pos = 4096  # From checkpoint config: connector_positional_embedding_max_pos=[4096]
        # Frequency grid: f64 for theta^linspace (matching PyTorch)
        pow_indices = jnp.power(theta, jnp.linspace(0.0, 1.0, n_freqs, dtype=jnp.float64))
        indices = (pow_indices * (jnp.pi / 2.0)).astype(jnp.float32)
        positions = jnp.arange(L, dtype=jnp.float32)
        frac = positions / max_pos
        pos_scaled = frac * 2.0 - 1.0
        freqs = indices[None, :] * pos_scaled[:, None]  # [L, 1920]
        cos_freq = jnp.repeat(jnp.cos(freqs), 2, axis=-1)  # [L, 3840]
        sin_freq = jnp.repeat(jnp.sin(freqs), 2, axis=-1)  # [L, 3840]
        return (cos_freq[None], sin_freq[None])  # [1, L, 3840]

    def _replace_padded_with_registers(self, x, num_valid):
        """Replace padded positions with tiled learnable register embeddings.

        Reference: embeddings_connector.py _replace_padded_with_learnable_registers.
        After this, valid tokens are at the start and registers fill the rest.
        Attention mask becomes all-valid (no masking needed).

        Args:
            x: [1, target_len, dim] - input with zero padding after num_valid positions
            num_valid: number of valid (non-padded) tokens
        Returns:
            x with padded positions replaced by tiled registers
        """
        target_len = x.shape[1]
        num_tiles = target_len // self.num_registers
        registers = jnp.tile(self.learnable_registers.value, (num_tiles, 1))  # [target_len, dim]
        # Create mask: 1 for valid tokens (positions < num_valid), 0 for registers
        valid_mask = (jnp.arange(target_len) < num_valid)[None, :, None]  # [1, target_len, 1]
        x = jnp.where(valid_mask, x, registers[None].astype(x.dtype))
        return x

    def __call__(self, x, num_valid_tokens=None):
        """Process embeddings through the connector with learnable register replacement.

        Reference (embeddings_connector.py): pads to 1024, replaces padding with
        tiled learnable registers, runs self-attention with all positions valid.
        The DiT was trained with 1024-length text embeddings including registers.

        Args:
            x: [1, seq_len, dim] - projected text embeddings for a SINGLE prompt
            num_valid_tokens: number of valid tokens (rest will be filled with registers).
                If None, uses x.shape[1] (no padding/registers).
        Returns:
            [1, 1024, dim] - connector output with registers
        """
        TARGET_LEN = 1024  # Reference always pads to 1024
        original_len = x.shape[1]

        # Pad to TARGET_LEN if needed
        if original_len < TARGET_LEN:
            x = jnp.pad(x, ((0, 0), (0, TARGET_LEN - original_len), (0, 0)))

        # Replace padded positions with learnable registers
        n_valid = num_valid_tokens if num_valid_tokens is not None else original_len
        if self.num_registers > 0:
            x = self._replace_padded_with_registers(x, n_valid)

        L = x.shape[1]  # Now TARGET_LEN
        pe = self._compute_pe(L)

        # No attention mask — all positions (tokens + registers) attend to each other
        for block in self.blocks:
            x = block(x, mask=None, pe=pe, attn_mask=None)
        return self.norm_out(x)


class GemmaFeaturesExtractorProjLinear(nnx.Module):
    def __init__(self, mesh=None):
        super().__init__()
        self.aggregate_embed = LinearBase(3840 * 49, 3840, use_bias=False, mesh=mesh, kernel_axes=(None, "tensor"))

    def __call__(self, hidden_states, mask, request_ids=None, batch_size=1):
        # hidden_states: [B, T, D, L] where L=49
        # mask: [B, T] boolean - True for valid tokens (all True when no padding)
        b, t, d, l = hidden_states.shape
        eps = 1e-6

        if request_ids is not None and batch_size > 1:
            # Per-request normalization: each request gets its own mean and range
            # so positive and negative prompts are normalized independently.
            hs = hidden_states[0]  # [T, D, L]
            mean_per_token = jnp.zeros((t, 1, l), dtype=hs.dtype)
            range_per_token = jnp.ones((t, 1, l), dtype=hs.dtype)
            for i in range(batch_size):
                req_mask = (request_ids == i) & mask[0]  # [T]
                req_sum = (hs * req_mask[:, None, None]).sum(axis=(0, 1), keepdims=True)  # [1, 1, L]
                req_count = req_mask.sum() * d
                req_mean = req_sum / (req_count + eps)  # [1, 1, L]
                # Masked min/max per layer for range normalization
                masked_hs = jnp.where(req_mask[:, None, None], hs, jnp.inf)
                req_min = masked_hs.min(axis=(0, 1), keepdims=True)  # [1, 1, L]
                masked_hs = jnp.where(req_mask[:, None, None], hs, -jnp.inf)
                req_max = masked_hs.max(axis=(0, 1), keepdims=True)  # [1, 1, L]
                req_range = req_max - req_min  # [1, 1, L]
                mean_per_token = jnp.where(req_mask[:, None, None], req_mean, mean_per_token)
                range_per_token = jnp.where(req_mask[:, None, None], req_range, range_per_token)
            normed = 8.0 * (hs - mean_per_token) / (range_per_token + eps)
            normed = normed[jnp.newaxis]  # [1, T, D, L]
        else:
            # Single request: global mean and range per layer
            denom = (mask.sum(axis=1, keepdims=True) * d)[..., None, None]
            mean = hidden_states.sum(axis=(1, 2), keepdims=True) / (denom + eps)
            # Masked min/max for range normalization
            x_min = jnp.where(mask[:, :, None, None], hidden_states, jnp.inf).min(axis=(1, 2), keepdims=True)
            x_max = jnp.where(mask[:, :, None, None], hidden_states, -jnp.inf).max(axis=(1, 2), keepdims=True)
            range_ = x_max - x_min
            normed = 8.0 * (hidden_states - mean) / (range_ + eps)

        # Zero out invalid tokens
        normed = jnp.where(mask[:, :, None, None], normed, 0.0)
        out, _ = self.aggregate_embed(jnp.reshape(normed, (b, t, -1)))
        return out


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
        # Augment Gemma2Model layers with Gemma3-specific features (QK norms,
        # per-layer RoPE theta/scaling) needed by forward_no_cache.
        self._augment_gemma3_layers(config, dtype)

        self.hidden_size = config.hidden_size

        self.feature_extractor = GemmaFeaturesExtractorProjLinear(mesh=mesh)
        self.embeddings_connector = Embeddings1DConnector(
            num_layers=2, dim=self.hidden_size, heads=30, dim_head=128, num_registers=128, mesh=mesh
        )
        # Audio embeddings connector — same architecture, independent weights.
        # Both connectors receive the same Gemma feature extractor output but
        # produce separate conditioning for video and audio diffusion.
        self.audio_embeddings_connector = Embeddings1DConnector(
            num_layers=2, dim=self.hidden_size, heads=30, dim_head=128, num_registers=128, mesh=mesh
        )

        logger.info(
            f"Initialized LTX2GemmaTextEncoder with hidden_size={self.hidden_size}"
        )

    def __call__(
        self,
        forward_batch,
        token_to_kv_pool=None,
        logits_metadata=None,
    ) -> jax.Array:
        batch_size = forward_batch.batch_size  # Python int (static in JIT)
        seq_lens = forward_batch.seq_lens      # [batch_size] traced JAX array
        # Raw (unpadded) tokens are sent to Gemma. Causal attention on raw tokens
        # gives the same hidden states as PyTorch's left-padded + attention_mask
        # approach, because RoPE is shift-invariant and real tokens only attend to
        # preceding real tokens in both cases. After feature extraction, we pad
        # each request's features to 1024 for the connector using gather.
        PROMPT_LEN = 1024

        # Get all hidden states from Gemma model (causal attention)
        _, aux_hidden_states, layers_kv_fused = self.model(forward_batch, token_to_kv_pool)

        # Stack to [N_total, D, L] where L = 49 (embed + 48 layers)
        stacked_hidden_states = jnp.stack(aux_hidden_states, axis=-1)
        n, d, l = stacked_hidden_states.shape
        stacked_hidden_states = stacked_hidden_states.reshape(1, n, d, l)
        # All tokens are valid (no padding in Gemma input)
        mask = jnp.ones((1, n), dtype=jnp.bool_)

        cum = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(seq_lens)])

        # Build per-request IDs for feature extractor normalization.
        positions = jnp.arange(n)
        request_ids = jnp.zeros(n, dtype=jnp.int32)
        for i in range(batch_size):
            in_request = (positions >= cum[i]) & (positions < cum[i + 1])
            request_ids = jnp.where(in_request, i, request_ids)

        projected = self.feature_extractor(
            stacked_hidden_states, mask, request_ids=request_ids, batch_size=batch_size
        )  # [1, N_total, 3840]

        # Build connector input for each request using gather.
        # Each request has seq_lens[i] features; we gather them into a
        # [1024, 3840] tensor (features at front, zeros at back).
        # JAX clips out-of-bounds indices, and invalid positions are zeroed out.
        connector_outputs = []
        audio_connector_outputs = []
        for i in range(batch_size):
            local_pos = jnp.arange(PROMPT_LEN)
            global_pos = cum[i] + local_pos  # Traced; may exceed N_total for invalid positions
            valid = local_pos < seq_lens[i]  # True for real features, False for padding
            # Clamp out-of-bounds to 0 (JAX also clips, but this is explicit)
            safe_pos = jnp.where(valid, global_pos, jnp.int32(0))
            chunk = projected[0][safe_pos]  # [1024, 3840] - gather from projected
            chunk = jnp.where(valid[:, None], chunk, 0.0)  # Zero out padding positions
            num_valid = seq_lens[i]

            encoded_chunk = self.embeddings_connector(
                chunk[None], num_valid_tokens=num_valid
            )
            connector_outputs.append(encoded_chunk[0])  # [1024, 3840]

            # Audio connector gets the same input but has independent weights
            audio_chunk = self.audio_embeddings_connector(
                chunk[None], num_valid_tokens=num_valid
            )
            audio_connector_outputs.append(audio_chunk[0])  # [1024, 3840]
        encoded_ctx = jnp.concatenate(connector_outputs, axis=0)
        audio_encoded_ctx = jnp.concatenate(audio_connector_outputs, axis=0)

        # Dummy logits for interface compatibility
        dummy = jnp.zeros((batch_size, self.config.vocab_size), dtype=self.dtype)
        dummy = jax.device_put(dummy, NamedSharding(self.mesh, PartitionSpec(None, "tensor")))
        return LogitsProcessorOutput(next_token_logits=dummy, hidden_states=encoded_ctx, audio_hidden_states=audio_encoded_ctx), layers_kv_fused, [], None

    def _augment_gemma3_layers(self, config, dtype):
        """Add Gemma3-specific features to Gemma2Model layers.

        The base Gemma2Model doesn't have QK normalization or per-layer RoPE
        theta/scaling. Gemma3 requires these. We add them here so that
        forward_no_cache can use them without modifying the shared gemma2.py.
        """
        is_gemma3 = getattr(config, "model_type", "").startswith("gemma3")
        if not is_gemma3:
            return

        for i, layer in enumerate(self.model.layers):
            attn = layer.self_attn

            # Add QK normalization (Gemma3 has this, Gemma2 does not)
            attn.q_norm = GemmaRMSNorm(config.head_dim, epsilon=1e-6)
            attn.k_norm = GemmaRMSNorm(config.head_dim, epsilon=1e-6)

            # Replace RoPE with correct per-layer theta and scaling.
            # Gemma3 sliding_attention layers use a local rope base freq
            # with no scaling. Global layers use the main rope_theta with
            # linear scaling.
            is_sliding = config.layer_types[i] == "sliding_attention"
            if is_sliding and hasattr(config, "rope_local_base_freq"):
                layer_rope_theta = config.rope_local_base_freq
                scaling_factor = 1.0
            else:
                layer_rope_theta = config.rope_theta
                scaling_factor = 1.0
                if getattr(config, "rope_scaling", None):
                    scaling_factor = config.rope_scaling.get("factor", 1.0)

            attn.rotary_emb = RotaryEmbedding(
                head_size=config.head_dim,
                rotary_dim=config.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=layer_rope_theta,
                is_neox_style=True,
                dtype=dtype,
                scaling_factor=scaling_factor,
            )

    def forward_no_cache(self, input_ids_pos, input_ids_neg):
        """Run the full text encoder pipeline without KV cache.

        Runs Gemma backbone using simple_attention (no RadixAttention/KV cache),
        then feature extractor and connector. This is appropriate since the text
        encoder runs once per request (prefill-only, no autoregressive decode).

        Args:
            input_ids_pos: list[int] — tokenized positive prompt
            input_ids_neg: list[int] — tokenized negative prompt

        Returns:
            (prompt_embeds, negative_prompt_embeds) — each [1024, hidden_size]
        """
        PROMPT_LEN = 1024

        all_ids_raw = [input_ids_pos, input_ids_neg]
        all_hidden_stacks = []  # will hold [1024, D, L] per prompt
        
        # We need a shared mask for the Gemma feature extractor projection
        all_masks = []
        original_lens = []

        for ids in all_ids_raw:
            original_len = len(ids)
            original_lens.append(original_len)
            
            # 1. Pad input_ids upfront (matching PyTorch)
            padded_ids = ids + [0] * (PROMPT_LEN - original_len)
            ids_array = jnp.array(padded_ids, dtype=jnp.int32)
            
            # 2. Create attention mask
            attention_mask = [True] * original_len + [False] * (PROMPT_LEN - original_len)
            mask_array = jnp.array(attention_mask, dtype=jnp.bool_)
            all_masks.append(mask_array)
            
            seq_len = ids_array.shape[0] # Now exactly 1024

            # Embedding
            hidden = self.model.embed_tokens(ids_array)
            hidden = hidden * jnp.array(
                [self.model.hidden_size**0.5], dtype=hidden.dtype
            )

            aux_hidden_states = [hidden]

            # Run through all Gemma layers with simple_attention
            for i, layer in enumerate(self.model.layers):
                hidden = self._layer_forward_no_cache(layer, hidden, seq_len, mask_array)
                if i < len(self.model.layers) - 1:
                    aux_hidden_states.append(hidden)

            hidden = self.model.norm(hidden)
            aux_hidden_states.append(hidden)

            # Stack: [seq_len, D, L] where L = num_layers + 1
            stacked = jnp.stack(aux_hidden_states, axis=-1)
            all_hidden_stacks.append(stacked)

        # Concatenate both prompts for feature extraction
        combined = jnp.concatenate(all_hidden_stacks, axis=0)  # [N_total=2048, D, L]
        n_total = combined.shape[0]
        combined = combined.reshape(1, n_total, combined.shape[1], combined.shape[2])
        
        # Mask needs to reflect real vs padded for the feature extractor
        mask = jnp.concatenate(all_masks, axis=0).reshape(1, n_total) # [1, 2048]

        # Per-request IDs for feature extractor normalization (0 for pos, 1 for neg)
        positions = jnp.arange(n_total)
        request_ids = jnp.where(positions < PROMPT_LEN, 0, 1)

        projected = self.feature_extractor(
            combined, mask, request_ids=request_ids, batch_size=2
        )  # [1, N_total, 3840]

        # Run connector per prompt (both video and audio connectors)
        results = []
        audio_results = []
        for i in range(2):
            # Because we padded the input tokens upfront, the projected features 
            # are ALREADY length 1024 per prompt. We just need to slice them out.
            start_idx = i * PROMPT_LEN
            end_idx = start_idx + PROMPT_LEN
            chunk = projected[0][start_idx:end_idx]  # [1024, 3840]
            
            num_valid = original_lens[i]
            
            # The connector will replace the 0-padded features with the learnable registers
            encoded_chunk = self.embeddings_connector(
                chunk[None], num_valid_tokens=num_valid
            )
            results.append(encoded_chunk[0])  # [1024, 3840]

            audio_chunk = self.audio_embeddings_connector(
                chunk[None], num_valid_tokens=num_valid
            )
            audio_results.append(audio_chunk[0])  # [1024, 3840]

        return results[0], results[1], audio_results[0], audio_results[1]

    def _layer_forward_no_cache(self, layer, hidden_states, seq_len, mask_array):
        """Run a single Gemma decoder layer using simple_attention (no KV cache)."""
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        attn = layer.self_attn
        q, _ = attn.q_proj(hidden_states)
        k, _ = attn.k_proj(hidden_states)
        v, _ = attn.v_proj(hidden_states)

        q = q.reshape(-1, attn.num_heads, attn.head_dim)
        k = k.reshape(-1, attn.num_kv_heads, attn.head_dim)
        v = v.reshape(-1, attn.num_kv_heads, attn.head_dim)

        q_norm = getattr(attn, "q_norm", None)
        k_norm = getattr(attn, "k_norm", None)
        if q_norm is not None:
            q = q_norm(q)
        if k_norm is not None:
            k = k_norm(k)

        # Apply RoPE
        positions = jnp.arange(seq_len, dtype=jnp.int32)
        q, k = attn.rotary_emb(positions, q, k)

        # Reshape for simple_attention: [1, S, H, D]
        q = q[None]
        k = k[None]
        v = v[None]

        # GQA: repeat KV heads to match query heads
        if attn.num_kv_heads != attn.num_heads:
            copies = attn.num_heads // attn.num_kv_heads
            k = jnp.repeat(k, copies, axis=2)
            v = jnp.repeat(v, copies, axis=2)

        # Create 2D boolean mask for attention: [1, S_k]
        attn_mask = mask_array[None]

        attn_output = simple_attention(q, k, v, scale=attn.scaling, causal=True, mask=attn_mask)
        attn_output = attn_output[0]  # [S, H, D]
        attn_output = attn_output.reshape(-1, attn.num_heads * attn.head_dim)
        output, _ = attn.o_proj(attn_output)

        hidden_states = layer.post_attention_layernorm(output)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = layer.pre_feedforward_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = layer.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def _create_gemma_weight_mappings(self) -> dict:
        """Build Gemma weight mappings for gemma-3-12b-it checkpoint.

        The gemma-3-12b-it safetensors use 'language_model.model.*' key prefix
        (multimodal Gemma 3 format). Target paths use 'model.*' to reach
        through self.model (Gemma2Model).
        """
        # Source prefix in gemma-3-12b-it safetensors
        src = "language_model.model"
        # Target prefix for self.model (Gemma2Model stored at self.model)
        tgt = "model"

        mappings = {
            f"{src}.embed_tokens.weight": WeightMapping(
                target_path=f"{tgt}.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            f"{src}.norm.weight": WeightMapping(
                target_path=f"{tgt}.norm.weight", sharding=(None,), transpose=False
            ),
        }

        num_layers = self.config.num_hidden_layers
        for layer_idx in range(num_layers):
            sp = f"{src}.layers.{layer_idx}"
            tp = f"{tgt}.layers.{layer_idx}"
            mappings.update({
                f"{sp}.input_layernorm.weight": WeightMapping(
                    f"{tp}.input_layernorm.weight", (None,)),
                f"{sp}.post_attention_layernorm.weight": WeightMapping(
                    f"{tp}.post_attention_layernorm.weight", (None,)),
                f"{sp}.post_feedforward_layernorm.weight": WeightMapping(
                    f"{tp}.post_feedforward_layernorm.weight", (None,)),
                f"{sp}.pre_feedforward_layernorm.weight": WeightMapping(
                    f"{tp}.pre_feedforward_layernorm.weight", (None,)),
                f"{sp}.self_attn.q_proj.weight": WeightMapping(
                    f"{tp}.self_attn.q_proj.weight", (None, "tensor"), True,
                    head_dim_padding=True, kv_head_padding=False),
                f"{sp}.self_attn.k_proj.weight": WeightMapping(
                    f"{tp}.self_attn.k_proj.weight", (None, "tensor"), True,
                    head_dim_padding=True, kv_head_padding=True),
                f"{sp}.self_attn.v_proj.weight": WeightMapping(
                    f"{tp}.self_attn.v_proj.weight", (None, "tensor"), True,
                    head_dim_padding=True, kv_head_padding=True),
                f"{sp}.self_attn.o_proj.weight": WeightMapping(
                    f"{tp}.self_attn.o_proj.weight", ("tensor", None), True,
                    head_dim_padding=True, kv_head_padding=False),
                f"{sp}.self_attn.q_norm.weight": WeightMapping(
                    f"{tp}.self_attn.q_norm.weight", (None,)),
                f"{sp}.self_attn.k_norm.weight": WeightMapping(
                    f"{tp}.self_attn.k_norm.weight", (None,)),
                f"{sp}.mlp.gate_proj.weight": WeightMapping(
                    f"{tp}.mlp.gate_proj.weight", (None, "tensor"), True),
                f"{sp}.mlp.up_proj.weight": WeightMapping(
                    f"{tp}.mlp.up_proj.weight", (None, "tensor"), True),
                f"{sp}.mlp.down_proj.weight": WeightMapping(
                    f"{tp}.mlp.down_proj.weight", ("tensor", None), True),
            })

        return mappings

    def load_weights(self, model_config: ModelConfig, ltx_checkpoint_path: str | None = None):  # noqa: ARG002 — ltx_checkpoint_path kept for backward compat
        import os

        # Auto-discover Gemma-3 checkpoint from HF cache
        from sgl_jax.srt.multimodal.models.ltx2.utils import get_hf_snapshot_dir
        gemma_path = get_hf_snapshot_dir("google/gemma-3-12b-it") or model_config.model_path

        # Use a temporary config to point the loader to the right directory
        temp_config = ModelConfig(model_path=gemma_path)

        loader = WeightLoader(
            model=self,
            model_config=temp_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        loader.load_weights_from_safetensors(self._create_gemma_weight_mappings())
        logger.info(f"Loaded Gemma backbone weights from {gemma_path}")

        # Materialize any remaining abstract params (connector weights not in Gemma checkpoint)
        def _replace_abstract(x):
            if isinstance(x, ShapeDtypeStruct):
                pspec = PartitionSpec()
                if hasattr(x, "sharding") and x.sharding is not None and hasattr(x.sharding, "spec"):
                    pspec = x.sharding.spec
                concrete_sharding = NamedSharding(self.mesh, pspec)
                return jax.device_put(jnp.zeros(x.shape, x.dtype), concrete_sharding)
            return x

        state = nnx.state(self)
        concrete_state = jax.tree_util.tree_map(_replace_abstract, state)
        nnx.update(self, concrete_state)
        logger.info("Materialized abstract state for text encoder connector params")

        # Set RMSNorm scales to ones (they default to zeros from materialize_abstract_state,
        # but PyTorch defaults are weight=1). These norms are NOT in the LTX checkpoint.
        for connector in [self.embeddings_connector, self.audio_embeddings_connector]:
            for block in connector.blocks:
                block.norm1.scale = nnx.Param(jnp.ones_like(block.norm1.scale.value))
                block.norm2.scale = nnx.Param(jnp.ones_like(block.norm2.scale.value))
            connector.norm_out.scale = nnx.Param(
                jnp.ones_like(connector.norm_out.scale.value)
            )
        logger.info("Set connector norm scales to ones (video + audio)")

        # Load feature extractor and connector weights from LTX-2 checkpoint
        self._load_ltx2_connector_weights(model_config)

    def _load_ltx2_connector_weights(self, model_config: ModelConfig):
        """Load connector weights from LTX-2 checkpoint using WeightLoader.

        Uses the same WeightLoader infrastructure as the DiT and VAE for
        consistent sharding and mesh-aware tensor placement. Temporarily
        swaps model_config.model_path to the LTX-2 checkpoint directory
        (same pattern as DiT load_weights).
        """
        from sgl_jax.srt.multimodal.models.ltx2.utils import (
            get_ltx2_checkpoint_dir, cleanup_ltx2_checkpoint_dir,
        )

        ckpt_dir = get_ltx2_checkpoint_dir()
        if ckpt_dir is None:
            logger.warning("LTX-2 checkpoint not found; connector weights not loaded")
            return

        logger.info(f"Loading text encoder connector weights via WeightLoader from {ckpt_dir}")

        mappings = {
            "text_embedding_projection.aggregate_embed.weight": WeightMapping(
                target_path="feature_extractor.aggregate_embed.weight",
                sharding=(None, None),
                transpose=True,
            ),
            "model.diffusion_model.video_embeddings_connector.learnable_registers": WeightMapping(
                target_path="embeddings_connector.learnable_registers",
                sharding=(None, None),
            ),
            "model.diffusion_model.audio_embeddings_connector.learnable_registers": WeightMapping(
                target_path="audio_embeddings_connector.learnable_registers",
                sharding=(None, None),
            ),
        }

        # Video and audio connectors share the same structure, different weights
        connector_configs = [
            ("model.diffusion_model.video_embeddings_connector", "embeddings_connector"),
            ("model.diffusion_model.audio_embeddings_connector", "audio_embeddings_connector"),
        ]
        for ckpt_prefix, model_prefix in connector_configs:
            for i in range(2):
                prefix = f"{ckpt_prefix}.transformer_1d_blocks.{i}"
                target = f"{model_prefix}.blocks.{i}"
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

        # Temporarily swap model_path to the LTX checkpoint directory
        # (same pattern as DiT load_weights at ltx2_dit.py:1215).
        original_path = model_config.model_path
        model_config.model_path = ckpt_dir

        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        loader.load_weights_from_safetensors(mappings)
        model_config.model_path = original_path
        cleanup_ltx2_checkpoint_dir(ckpt_dir)
        logger.info("Loaded LTX-2 connector weights via WeightLoader.")

    @staticmethod
    def from_pretrained(
        model_name_or_path: str,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name_or_path)
        encoder = LTX2GemmaTextEncoder(config, mesh=mesh, dtype=dtype)

        model_config = ModelConfig(model_path=model_name_or_path)
        encoder.load_weights(model_config)

        return encoder


# Entry class for model loading
EntryClass = LTX2GemmaTextEncoder
