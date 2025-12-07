import logging
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.lax
from flax import nnx
from jax import numpy as jnp
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.activation import GeluAndMul
from sgl_jax.srt.layers.embeddings import (
    Embed,
    ParallelLMHead,
    RotaryEmbedding,
    _yarn_find_correction_range,
    _yarn_get_mscale,
)
from sgl_jax.srt.layers.fused_moe import FusedEPMoE
from sgl_jax.srt.layers.layernorm import RMSNorm, dual_rmsnorm_forward
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    LogitsProcessorOutput,
)
from sgl_jax.srt.layers.moe import EPMoE
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)

init_fn = nnx.initializers.uniform()


def _yarn_linear_ramp_mask(low: float, high: float, dim: int, dtype: jnp.dtype) -> jax.Array:
    """Create a linear ramp mask for YaRN scaling."""
    if low == high:
        low -= 0.001  # Prevent singularity

    linear_func = (jnp.arange(dim, dtype=dtype) - low) / (high - low)
    ramp_func = jnp.clip(linear_func, 0, 1)
    return ramp_func


def get_rope_scaling(config: PretrainedConfig) -> dict[str, Any] | None:
    """Extract RoPE scaling configuration from model config."""
    rope_type = getattr(config, "rope_type", None)
    if rope_type:
        original_max_position_embeddings = getattr(config, "original_max_position_embeddings", None)
        scaling_factor = getattr(config, "scaling_factor", None)
        extrapolation_factor = getattr(config, "extrapolation_factor", 1.0)
        attn_factor = getattr(config, "attn_factor", 1.0)
        beta_fast = getattr(config, "beta_fast", 32)
        beta_slow = getattr(config, "beta_slow", 1)
        rope_scaling = {
            "extra_method": rope_type,
            "max_position_embeddings": original_max_position_embeddings,
            "scaling_factor": scaling_factor,
            "extrapolation_factor": extrapolation_factor,
            "attn_factor": attn_factor,
            "beta_fast": beta_fast,
            "beta_slow": beta_slow,
            "dtype": jnp.float32,
        }
        return rope_scaling
    else:
        return None


class ScalingRotaryEmbedding(RotaryEmbedding):
    """Scale the RotaryEmbedding in a way similar to YaRN method.
    https://arxiv.org/pdf/2309.00071.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: jnp.dtype,
        *,
        extra_method: str = "yarn_log",
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extra_method = extra_method
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation
        self.mscale = float(_yarn_get_mscale(self.scaling_factor) * attn_factor)
        super().__init__(head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype)

    def _compute_inv_freq(self, scaling_factor: float) -> jax.Array:
        pos_freqs = self.base ** (
            jnp.arange(0, self.rotary_dim, 2, dtype=jnp.float32) / self.rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
        )
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (
            1 - _yarn_linear_ramp_mask(low, high, self.rotary_dim // 2, dtype=jnp.float32)
        ) * self.extrapolation_factor

        if self.extra_method in ["original"]:
            inv_freq = inv_freq_extrapolation
        elif self.extra_method in ["yarn", "yarn_linear"]:
            inv_freq = (
                inv_freq_interpolation * (1 - inv_freq_mask)
                + inv_freq_extrapolation * inv_freq_mask
            )
        elif self.extra_method == "yarn_log":
            inv_freq = jnp.exp(
                jnp.log(inv_freq_extrapolation) * inv_freq_mask
                + jnp.log(inv_freq_interpolation) * (1.0 - inv_freq_mask)
            )
        elif self.extra_method == "theta_scale":
            import math

            exponents = jnp.arange(0, self.rotary_dim, 2, dtype=jnp.float32)
            theta_scale_exponent = self.base ** (
                math.log(self.max_position_embeddings * self.scaling_factor / (2 * math.pi))
                / math.log(self.max_position_embeddings / (2 * math.pi))
            )
            inv_freq = jnp.array(
                1.0 / (theta_scale_exponent ** (exponents / self.rotary_dim)),
                dtype=jnp.float32,
            )
        else:
            raise ValueError(f"Unknown extrapolation method: {self.extra_method}")
        return inv_freq


class Grok1MLP(nnx.Module):
    """Standard MLP layer for Grok-1 model."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()

        self.gate_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            mesh=mesh,
        )
        self.up_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            mesh=mesh,
        )
        self.down_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )
        self.act_fn = GeluAndMul(approximate="tanh")
        self.layer_id = layer_id
        self.reduce_results = reduce_results

    def __call__(self, x: jax.Array) -> jax.Array:
        gate, _ = self.gate_proj(x)
        up, _ = self.up_proj(x)
        x, _ = self.act_fn(gate, up)
        x, _ = self.down_proj(x)
        return x


class Grok1MoE(nnx.Module):
    """A tensor-parallel Mixture of Experts implementation for Grok-1.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, with outputs reduced across ranks.
    """

    experts: FusedEPMoE | EPMoE

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.top_k = top_k
        self.layer_id = layer_id
        self.mesh = mesh

        # Gate always runs at full precision for stability
        # (see https://arxiv.org/pdf/2101.03961)
        self.gate = LinearBase(
            input_size=hidden_size,
            output_size=num_experts,
            use_bias=False,
            params_dtype=jnp.float32,
            kernel_axes=(None, None),
            mesh=mesh,
        )

        self.router_logit_softcapping = getattr(config, "router_logit_softcapping", 30.0)

        # Select MoE backend based on config
        self.moe_backend = getattr(config, "moe_backend", "epmoe")
        self.use_fused = self.moe_backend == "fused"

        if self.use_fused:
            self.experts = FusedEPMoE(
                config=config,
                num_experts=num_experts,
                num_experts_per_tok=self.top_k,
                intermediate_dim=intermediate_size,
                mesh=mesh,
                activation="gelu",
                ep_size=config.ep_size,
                weight_dtype=dtype,
                dtype=dtype,
                layer_id=layer_id,
                renormalize_topk_logits=False,  # Match sglang behavior
            )
        else:
            self.experts = EPMoE(
                config=config,
                num_experts=num_experts,
                num_experts_per_tok=self.top_k,
                intermediate_dim=intermediate_size,
                mesh=mesh,
                activation="gelu",
                ep_size=config.ep_size,
                weight_dtype=dtype,
                dtype=dtype,
                layer_id=layer_id,
            )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        # Router computation with soft capping
        router_logits, _ = self.gate(hidden_states)

        # Apply soft capping for stability (matching sglang implementation)
        if self.router_logit_softcapping != 0:
            router_logits = router_logits / self.router_logit_softcapping
            router_logits = jax.nn.tanh(router_logits) * self.router_logit_softcapping

        if self.use_fused:
            # Fused kernel: pass router_logits directly
            # Top-K selection is handled internally by the kernel
            assert isinstance(self.experts, FusedEPMoE)
            return self.experts(hidden_states, router_logits)
        else:
            # EPMoE: compute top-k routing weights using sglang-style approach:
            # 1. Compute global softmax over ALL experts (not just top-k)
            # 2. Select top-k experts based on logits
            # 3. Extract corresponding weights (no renormalization)
            assert isinstance(self.experts, EPMoE)
            top_k_weights, top_k_indices = self._custom_topk(
                router_logits, self.top_k, renormalize=False
            )

            return self.experts(hidden_states, top_k_weights, top_k_indices)

    def _custom_topk(
        self, router_logits: jax.Array, top_k: int, renormalize: bool = False
    ) -> tuple[jax.Array, jax.Array]:
        """
        Compute top-k routing weights using sglang's approach.

        This matches sglang's fused_moe_router implementation:
        1. Compute softmax over ALL experts (global normalization)
        2. Select top-k experts based on original logits
        3. Extract weights for selected experts

        Key difference from naive approach:
        - Naive: top_k on logits, then softmax on selected → weights sum to 1
        - sglang: softmax on all, then select top_k → weights sum to <1

        This preserves the "confidence" of selected experts relative to all experts.

        Args:
            router_logits: [batch_size, num_experts] logits from gate
            top_k: number of experts to select
            renormalize: if True, renormalize selected weights to sum to 1

        Returns:
            top_k_weights: [batch_size, top_k] weights for selected experts
            top_k_indices: [batch_size, top_k] indices of selected experts
        """
        # Compute global softmax over ALL experts (numerically stable)
        # This matches sglang's implementation in router.py lines 71-76, 244-246
        max_logit = jnp.max(router_logits, axis=-1, keepdims=True)
        exp_logits = jnp.exp(router_logits - max_logit)
        sum_exp_logits = jnp.sum(exp_logits, axis=-1, keepdims=True)
        all_weights = exp_logits / sum_exp_logits  # Global softmax weights

        # Select top-k experts based on logits (not weights)
        # jax.lax.top_k returns values and indices in descending order
        _, top_k_indices = jax.lax.top_k(router_logits, top_k)

        # Extract weights for the selected top-k experts
        # Use take_along_axis to gather weights corresponding to top_k_indices
        top_k_weights = jnp.take_along_axis(all_weights, top_k_indices, axis=-1)

        # Optional: renormalize weights to sum to 1 (sglang uses renormalize=False)
        if renormalize:
            top_k_weights = top_k_weights / jnp.sum(top_k_weights, axis=-1, keepdims=True)

        return top_k_weights, top_k_indices


class Grok1Attention(nnx.Module):
    """Multi-head attention layer for Grok-1 with RoPE positional encoding."""

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = hidden_size

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        self.head_dim = getattr(config, "head_dim", 128)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.mesh = mesh

        rope_scaling = get_rope_scaling(config)
        self.rope_rotate_half_dims = getattr(config, "rope_rotate_half_dims", False)

        # Separate Q, K, V projections (to match checkpoint format)
        self.q_proj = LinearBase(
            input_size=hidden_size,
            output_size=self.q_size,
            use_bias=False,
            params_dtype=jnp.bfloat16,
            kernel_axes=(None, "tensor"),
            mesh=mesh,
        )

        self.k_proj = LinearBase(
            input_size=hidden_size,
            output_size=self.kv_size,
            use_bias=False,
            params_dtype=jnp.bfloat16,
            kernel_axes=(None, "tensor"),
            mesh=mesh,
        )

        self.v_proj = LinearBase(
            input_size=hidden_size,
            output_size=self.kv_size,
            use_bias=False,
            params_dtype=jnp.bfloat16,
            kernel_axes=(None, "tensor"),
            mesh=mesh,
        )

        self.o_proj = LinearBase(
            input_size=self.num_heads * self.head_dim,  # Use local num_heads
            output_size=hidden_size,
            use_bias=False,
            params_dtype=jnp.bfloat16,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )

        # Initialize rotary embeddings based on scaling configuration
        if rope_scaling is not None:
            self.rotary_emb: RotaryEmbedding = ScalingRotaryEmbedding(
                self.head_dim,
                rotary_dim=(
                    self.head_dim if not self.rope_rotate_half_dims else self.head_dim // 2
                ),
                base=int(self.rope_theta),
                is_neox_style=True,
                **rope_scaling,
            )
        else:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                rotary_dim=(
                    self.head_dim if not self.rope_rotate_half_dims else self.head_dim // 2
                ),
                max_position_embeddings=max_position,
                base=int(self.rope_theta),
                is_neox_style=True,
                dtype=jnp.bfloat16,
            )

        logit_cap = max(getattr(config, "attn_logit_softcapping", 30.0), 0.0)

        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            logit_cap=logit_cap,
        )
        self.attn.xai_temperature_len = getattr(config, "attn_temperature_len", -1)

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, jax.Array]:
        # Short circuit for empty sequences
        if hidden_states.shape[0] == 0:
            return hidden_states, hidden_states

        # Project Q, K, V separately
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        # Apply rotary position embeddings
        q, k = self.rotary_emb(positions, q, k)

        q = q.reshape(-1, self.num_heads, self.head_dim)
        k = k.reshape(-1, self.num_kv_heads, self.head_dim)
        v = v.reshape(-1, self.num_kv_heads, self.head_dim)

        # Apply attention (backend may return tuple)
        attn_ret, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool)
        attn_output = attn_ret[0] if isinstance(attn_ret, tuple) else attn_ret

        # Project output
        output, _ = self.o_proj(attn_output)
        return output, kv_fused


class Grok1DecoderLayer(nnx.Module):
    """A single decoder layer of the Grok-1 model."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        mesh: jax.sharding.Mesh,
    ) -> None:
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.residual_moe = getattr(config, "residual_moe", False)
        self.layer_id = layer_id
        self.mesh = mesh

        # Self-attention
        rope_theta = getattr(config, "rope_theta", 10000)
        self.self_attn = Grok1Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=(
                config.context_len
                if hasattr(config, "context_len")
                else config.max_position_embeddings
            ),
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            mesh=mesh,
        )

        # Feed-forward networks
        if self.num_experts > 0:
            intermediate_size = getattr(
                config,
                "moe_intermediate_size",
                getattr(config, "intermediate_size", None),
            )
            if intermediate_size is None:
                raise ValueError(
                    "MoE layers require 'moe_intermediate_size' or 'intermediate_size' in config."
                )
            self.block_sparse_moe = Grok1MoE(
                config=config,
                layer_id=layer_id,
                num_experts=config.num_local_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                mesh=mesh,
            )
            if self.residual_moe:
                self.mlp = Grok1MLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    layer_id=layer_id,
                    reduce_results=False,
                    mesh=mesh,
                )
        else:
            raise NotImplementedError()

        # Layer normalization (using eps instead of epsilon to match PyTorch)
        self.pre_attn_norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
        )
        self.post_attn_norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
        )
        self.pre_moe_norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
        )
        self.post_moe_norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
        )

        # Setup FFN function based on configuration (matching PyTorch logic)
        if self.num_experts > 0:
            self.ffn: Callable[[jax.Array], jax.Array] | Grok1MoE
            if self.residual_moe:
                self.ffn = self.moe_with_rmoe
            else:
                self.ffn = self.block_sparse_moe
        else:
            raise NotImplementedError()

    def moe_with_rmoe(self, x: jax.Array) -> jax.Array:
        """Combine MoE and residual MLP outputs (matches PyTorch implementation)."""
        mlp_result = self.mlp(x)
        moe_result = self.block_sparse_moe(x)
        # Scale factor from the paper: 1/sqrt(2)
        return (mlp_result + moe_result) / 1.4142135623730951

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: jax.Array | None = None,
        deferred_norm: RMSNorm | None = None,
    ) -> tuple[jax.Array, jax.Array, RMSNorm, jax.Array]:

        # Self Attention block (matching PyTorch logic exactly)
        if deferred_norm is not None:
            assert residual is not None
            assert deferred_norm.scale is not None
            assert self.pre_attn_norm.scale is not None
            # Apply deferred norm from previous layer and pre-attention norm
            hidden_states, residual = dual_rmsnorm_forward(
                hidden_states,
                residual,
                deferred_norm.scale,
                self.pre_attn_norm.scale,
                deferred_norm.epsilon,
            )
        else:
            # First layer or no deferred norm - use fused_rmsnorm equivalent
            hidden_states, residual = self.pre_attn_norm(hidden_states), hidden_states

        # Self-attention
        hidden_states, kv_fused = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )

        # # Apply post-attention norm and pre-MoE norm (matching PyTorch fused_dual_residual_rmsnorm)
        assert self.post_attn_norm.scale is not None
        assert self.pre_moe_norm.scale is not None
        hidden_states, residual = dual_rmsnorm_forward(
            hidden_states,
            residual,
            self.post_attn_norm.scale,
            self.pre_moe_norm.scale,
            self.post_attn_norm.epsilon,
        )

        # Feed-forward network
        if self.residual_moe:
            hidden_states = self.moe_with_rmoe(hidden_states)
        else:
            hidden_states = self.block_sparse_moe(hidden_states)

        # Return with deferred post-MoE norm (matching PyTorch)
        return hidden_states, residual, self.post_moe_norm, kv_fused


class Grok1Model(nnx.Module):
    """The main Grok-1 transformer model."""

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        mesh: jax.sharding.Mesh,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.mesh = mesh

        # Token embeddings (matching PyTorch VocabParallelEmbedding behavior)
        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=jnp.bfloat16,
            kernel_axes=("tensor", None),
            param_dtype=jnp.bfloat16,
            mesh=self.mesh,
        )

        # Transformer layers
        self.layers = nnx.List(
            [
                Grok1DecoderLayer(config=config, layer_id=i, mesh=mesh)
                for i in range(config.num_hidden_layers)
            ]
        )

        # Final layer norm (using eps to match PyTorch)
        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
        )

    def __call__(
        self,
        input_ids: jax.Array,
        positions: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        input_embeds: jax.Array | None = None,
    ) -> tuple[jax.Array, list[jax.Array]]:
        # Get embeddings, hidden_states: [B, N, hidden_size]
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
            # Apply embedding scaling if specified (matching PyTorch)
            if hasattr(self.config, "embedding_multiplier_scale"):
                hidden_states = hidden_states * self.config.embedding_multiplier_scale
        else:
            hidden_states = input_embeds

        # Process through transformer layers with deferred normalization
        residual, deferred_norm = None, None
        layers_kv_fused = []
        for layer in self.layers:
            hidden_states, residual, deferred_norm, kv_fused = layer(
                positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
                deferred_norm,
            )
            layers_kv_fused.append(kv_fused)
        # Apply final normalization (matching PyTorch fused_dual_residual_rmsnorm)
        assert deferred_norm is not None
        assert residual is not None
        assert deferred_norm.scale is not None
        assert self.norm.scale is not None
        hidden_states, _ = dual_rmsnorm_forward(
            hidden_states,
            residual,
            deferred_norm.scale,
            self.norm.scale,
            deferred_norm.epsilon,
        )

        return hidden_states, layers_kv_fused


class Grok1ForCausalLM(nnx.Module):
    """Grok-1 model with a causal language modeling head."""

    def __init__(
        self,
        config: PretrainedConfig,
        dtype: jnp.dtype,
        mesh: jax.sharding.Mesh,
    ) -> None:
        super().__init__()
        assert dtype == jnp.bfloat16
        self.config = config
        self.mesh = mesh

        # Configuration flags (matching PyTorch implementation)
        self.replicate_lm_head = getattr(config, "replicate_lm_head", False)

        # Main model
        self.model = Grok1Model(config, mesh=mesh)

        # Language modeling head (matching PyTorch logic for replicated vs parallel)
        self.lm_head: LinearBase | ParallelLMHead
        if self.replicate_lm_head:
            # ReplicatedLinear equivalent
            self.lm_head = LinearBase(
                input_size=config.hidden_size,
                output_size=config.vocab_size,
                use_bias=False,
                params_dtype=jnp.bfloat16,
                kernel_axes=("tensor", None),
                mesh=mesh,
            )
        else:
            self.lm_head = ParallelLMHead(
                num_embeddings=config.vocab_size,
                features=config.hidden_size,
                param_dtype=jnp.bfloat16,
                kernel_axes=("tensor", None),
            )
        soft_cap = getattr(config, "final_logit_softcapping", 0.0) if config else 0.0
        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=mesh, soft_cap=soft_cap)

        self.loaded_param_names: set[str] = set()

    def named_parameters(self) -> list[tuple[str, nnx.Param[Any]]]:
        """Recursively yield (full_path, nnx.Param) over all parameters in this module.
        Paths resemble PyTorch-style dotted names, starting from this module.
        """
        params = {}

        def _collect(obj: Any, prefix: str) -> None:
            from flax import nnx as _nnx

            # nnx.Param
            if isinstance(obj, _nnx.Param):
                params[prefix] = obj
                return
            # dict-like
            if isinstance(obj, dict):
                for k, v in obj.items():
                    name = f"{prefix}.{k}" if prefix else str(k)
                    _collect(v, name)
                return
            # list/tuple
            if isinstance(obj, (list, tuple)):
                for idx, v in enumerate(obj):
                    name = f"{prefix}.{idx}" if prefix else str(idx)
                    _collect(v, name)
                return
            # objects with attributes
            if hasattr(obj, "__dict__"):
                for k, v in obj.__dict__.items():
                    # Skip private/internal
                    if k.startswith("_"):
                        continue
                    name = f"{prefix}.{k}" if prefix else k
                    _collect(v, name)

        _collect(self, "")
        return list(params.items())

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ) -> tuple[LogitsProcessorOutput, list[jax.Array], bool]:
        """Forward pass through the model using unified forward_batch API."""
        input_ids = forward_batch.input_ids
        positions = forward_batch.positions
        hidden_states, layers_kv_fused = self.model(
            input_ids, positions, forward_batch, token_to_kv_pool, None
        )
        output = self.logits_processor(hidden_states, cast(Embed, self.lm_head), logits_metadata)

        return output, layers_kv_fused, True

    def load_weights(self, model_config: ModelConfig) -> None:
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=jnp.bfloat16,
        )

        weight_mappings = self._create_grok_weight_mappings()

        loader.load_weights_from_safetensors(
            weight_mappings,
            safetensors_partition=8,
        )
        logger.info("Grok weights loaded successfully!")

    def _create_grok_weight_mappings(self) -> dict[str, WeightMapping]:
        mappings = {
            "model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.norm.weight": WeightMapping(
                target_path="model.norm.scale", sharding=(None,), transpose=False
            ),
            "lm_head.weight": WeightMapping(
                target_path="lm_head.embedding", sharding=("tensor", None), transpose=False
            ),
        }

        num_layers = self.config.num_hidden_layers
        for layer_idx in range(num_layers):
            layer_mappings = self._create_layer_mappings(layer_idx)
            mappings.update(layer_mappings)

        return mappings

    def _create_layer_mappings(self, layer_idx: int) -> dict[str, WeightMapping]:
        prefix = f"model.layers.{layer_idx}"
        target_prefix = f"model.layers.{layer_idx}"

        mappings = {
            # self_attn - separate q, k, v projections
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
            f"{prefix}.pre_attn_norm.weight": WeightMapping(
                target_path=f"{target_prefix}.pre_attn_norm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.post_attn_norm.weight": WeightMapping(
                target_path=f"{target_prefix}.post_attn_norm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.pre_moe_norm.weight": WeightMapping(
                target_path=f"{target_prefix}.pre_moe_norm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.post_moe_norm.weight": WeightMapping(
                target_path=f"{target_prefix}.post_moe_norm.scale",
                sharding=(None,),
                transpose=False,
            ),
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
            f"{prefix}.block_sparse_moe.gate.weight": WeightMapping(
                target_path=f"{target_prefix}.block_sparse_moe.gate.weight",
                sharding=(None, None),
                transpose=True,
            ),
        }

        moe_backend = getattr(self.config, "moe_backend", "epmoe")
        use_fused = moe_backend == "fused"

        if use_fused:
            # Fused MoE Mapping
            # w1: fused gate(w1) + up(w3) -> (num_experts, 2, hidden, intermediate)
            # w2: down(w2) -> (num_experts, intermediate, hidden)

            # 1. Fused w1 (gate + up)
            target_path_w1 = [f"{target_prefix}.block_sparse_moe.w1"]
            # Add source keys for w1 (gate) and w3 (up)
            # Note: Grok experts are 0..N-1
            for name in ["w1", "w3"]:
                target_path_w1.extend(
                    [
                        f"{prefix}.block_sparse_moe.experts.{i}.{name}.weight"
                        for i in range(self.config.num_local_experts)
                    ]
                )

            mappings[f"__MOE_EXPERTS__{prefix}.block_sparse_moe.w1"] = WeightMapping(
                target_path=target_path_w1,
                sharding=("expert", None, None, "tensor"),  # (E, 2, H, I/TP)
                transpose=True,
                concat_axis=0,  # concat along E axis
                fuse_moe_weights=True,
                fuse_gate_up=("w1", "w3"),
            )

            # 2. w2 (down)
            target_path_w2 = [f"{target_prefix}.block_sparse_moe.w2"]
            target_path_w2.extend(
                [
                    f"{prefix}.block_sparse_moe.experts.{i}.w2.weight"
                    for i in range(self.config.num_local_experts)
                ]
            )

            mappings[f"__MOE_EXPERTS__{prefix}.block_sparse_moe.w2"] = WeightMapping(
                target_path=target_path_w2,
                sharding=("expert", "tensor", None),  # (E, I/TP, H)
                transpose=True,
                concat_axis=-1,
            )

        else:
            # Standard EPMoE Mapping
            for name, target_name in [("w1", "wi_0"), ("w3", "wi_1"), ("w2", "wo")]:
                target_path = [f"{target_prefix}.block_sparse_moe.experts.{target_name}"]
                target_path.extend(
                    [
                        f"{prefix}.block_sparse_moe.experts.{i}.{name}.weight"
                        for i in range(self.config.num_local_experts)
                    ]
                )

                sharding = (
                    ("expert", "tensor", None)
                    if target_name == "wo"
                    else ("expert", None, "tensor")
                )

                if name == "w2":
                    # w2 (down_proj) -> wo
                    mappings[f"__MOE_EXPERTS__{prefix}.block_sparse_moe.experts.{target_name}"] = (
                        WeightMapping(
                            target_path=target_path,
                            sharding=sharding,
                            transpose=True,
                            concat_axis=-1,
                        )
                    )
                else:
                    # w1/w3 (gate/up) -> wi_0/wi_1
                    mappings[f"__MOE_EXPERTS__{prefix}.block_sparse_moe.experts.{target_name}"] = (
                        WeightMapping(
                            target_path=target_path,
                            sharding=sharding,
                            transpose=True,
                            concat_axis=0,
                        )
                    )

        return mappings


EntryClass = Grok1ForCausalLM
