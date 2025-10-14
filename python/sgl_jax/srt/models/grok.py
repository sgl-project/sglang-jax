import logging
from typing import Iterable, Optional, Tuple

import jax
import jax.lax
from flax import nnx
from jax import numpy as jnp
from transformers import PretrainedConfig

from sgl_jax.srt.layers.activation import GeluAndMul
from sgl_jax.srt.layers.embeddings import (
    Embed,
    ParallelLMHead,
    RotaryEmbedding,
    _yarn_find_correction_range,
    _yarn_get_mscale,
)
from sgl_jax.srt.layers.layernorm import RMSNorm, dual_rmsnorm_forward
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import EPMoE
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

init_fn = nnx.initializers.uniform()


def _yarn_linear_ramp_mask(
    low: float, high: float, dim: int, dtype: jnp.dtype
) -> jax.Array:
    """Create a linear ramp mask for YaRN scaling."""
    if low == high:
        low -= 0.001  # Prevent singularity

    linear_func = (jnp.arange(dim, dtype=dtype) - low) / (high - low)
    ramp_func = jnp.clip(linear_func, 0, 1)
    return ramp_func


def get_rope_scaling(config):
    """Extract RoPE scaling configuration from model config."""
    rope_type = getattr(config, "rope_type", None)
    if rope_type:
        original_max_position_embeddings = getattr(
            config, "original_max_position_embeddings", None
        )
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
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

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
            1
            - _yarn_linear_ramp_mask(low, high, self.rotary_dim // 2, dtype=jnp.float32)
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
                math.log(
                    self.max_position_embeddings * self.scaling_factor / (2 * math.pi)
                )
                / math.log(self.max_position_embeddings / (2 * math.pi))
            )
            inv_freq = jnp.array(
                1.0 / (theta_scale_exponent ** (exponents / self.rotary_dim)),
                dtype=jnp.float32,
            )
        else:
            raise ValueError(f"Unknown extrapolation method: {self.extra_method}")
        return inv_freq

    def _compute_cos_sin_cache(self) -> jax.Array:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = jnp.arange(
            self.max_position_embeddings * self.scaling_factor, dtype=jnp.float32
        )
        freqs = jnp.einsum("i,j -> ij", t, inv_freq)
        cos = jnp.cos(freqs)
        sin = jnp.sin(freqs)
        cache = jnp.concatenate((cos, sin), axis=-1)
        return cache


class Grok1MLP(nnx.Module):
    """Standard MLP layer for Grok-1 model."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        rngs: nnx.Rngs = None,
        dtype: jnp.dtype = jnp.bfloat16,
        reduce_results: bool = True,
        split_gate_up: bool = False,
    ) -> None:
        super().__init__()

        self.gate_up_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size * 2,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            rngs=rngs,
        )
        self.down_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=("tensor", None),
            rngs=rngs,
        )
        self.act_fn = GeluAndMul(approximate="tanh")
        self.layer_id = layer_id
        self.reduce_results = reduce_results

    def __call__(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x, _ = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Grok1MoE(nnx.Module):
    """A tensor-parallel Mixture of Experts implementation for Grok-1.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, with outputs reduced across ranks.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        rngs: nnx.Rngs,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        reduce_results: bool = True,
        inplace: bool = True,
        no_combine: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Gate always runs at full precision for stability
        # (see https://arxiv.org/pdf/2101.03961)
        self.gate = LinearBase(
            input_size=hidden_size,
            output_size=num_experts,
            use_bias=False,
            params_dtype=jnp.float32,
            kernel_axes=(None, None),
            rngs=rngs,
        )

        self.router_logit_softcapping = getattr(
            config, "router_logit_softcapping", 30.0
        )

        # Determine MoE implementation based on parallelism
        expert_parallel_size = mesh.shape.get("data", 1) * mesh.shape.get("tensor", 1)
        if num_experts % expert_parallel_size != 0:
            logging.warning(
                "num_experts(%d) not divisible by expert_parallel_size(%d); falling back to expert_parallel_size=1",
                num_experts,
                expert_parallel_size,
            )
            expert_parallel_size = 1
        self.experts = EPMoE(
            config=config,
            num_experts=num_experts,
            num_experts_per_tok=top_k,
            expert_parallel_size=expert_parallel_size,
            mesh=mesh,
            intermediate_dim=intermediate_size,
            dtype=dtype,
            activation="gelu",
            layer_id=layer_id,
            rngs=rngs,
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        # Router computation with soft capping
        router_logits, _ = self.gate(hidden_states)

        # Apply soft capping for stability (matching PyTorch implementation)
        if self.router_logit_softcapping != 0:
            router_logits = router_logits / self.router_logit_softcapping
            router_logits = jax.nn.tanh(router_logits) * self.router_logit_softcapping

        # Note: The PyTorch version applies softmax in the routing function,
        # but here we pass the logits directly to the experts layer
        # which handles the routing internally
        return self.experts(hidden_states, router_logits)


class Grok1Attention(nnx.Module):
    """Multi-head attention layer for Grok-1 with RoPE positional encoding."""

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        reduce_results: bool = True,
        *,
        mesh: jax.sharding.Mesh,
        rngs: nnx.Rngs,
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

        rope_scaling = get_rope_scaling(config)
        self.rope_rotate_half_dims = getattr(config, "rope_rotate_half_dims", False)

        # QKV projection (matches PyTorch QKVParallelLinear structure)
        self.qkv_proj = LinearBase(
            input_size=hidden_size,
            output_size=self.q_size + 2 * self.kv_size,  # Q + K + V
            use_bias=False,
            params_dtype=jnp.bfloat16,
            kernel_axes=(None, "tensor") if ("tensor" in mesh.shape) else (None, None),
            rngs=rngs,
        )

        self.o_proj = LinearBase(
            input_size=self.num_heads * self.head_dim,  # Use local num_heads
            output_size=hidden_size,
            use_bias=False,
            params_dtype=jnp.bfloat16,
            kernel_axes=("tensor", None) if ("tensor" in mesh.shape) else (None, None),
            rngs=rngs,
        )

        # Initialize rotary embeddings based on scaling configuration
        if rope_scaling is not None:
            self.rotary_emb = ScalingRotaryEmbedding(
                self.head_dim,
                rotary_dim=(
                    self.head_dim
                    if not self.rope_rotate_half_dims
                    else self.head_dim // 2
                ),
                base=int(self.rope_theta),
                is_neox_style=True,
                **rope_scaling,
            )
            pos_encoding_mode = "NONE"
        else:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                rotary_dim=(
                    self.head_dim
                    if not self.rope_rotate_half_dims
                    else self.head_dim // 2
                ),
                max_position=max_position,
                base=int(self.rope_theta),
                is_neox_style=True,
            )
            pos_encoding_mode = "NONE"

        # Attention configuration
        logit_cap = max(getattr(config, "attn_logit_softcapping", 30.0), 0.0)
        logit_capping_method = getattr(config, "attn_logit_softcapping_method", "tanh")

        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            # logit_cap=logit_cap,
            # pos_encoding_mode=pos_encoding_mode,
            # logit_capping_method=logit_capping_method,
        )
        self.attn.xai_temperature_len = getattr(config, "attn_temperature_len", -1)

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> jax.Array:
        # Short circuit for empty sequences
        if hidden_states.shape[0] == 0:
            return hidden_states

        # Project to QKV
        qkv, _ = self.qkv_proj(hidden_states)

        # Split QKV (matching PyTorch split logic)
        q, k, v = jnp.split(qkv, [self.q_size, self.q_size + self.kv_size], axis=-1)
        jax.debug.print("before attn qkv: {qkv}", qkv=qkv)

        # Apply rotary position embeddings
        q, k = self.rotary_emb(positions, q, k)

        # Apply attention (backend may return tuple)
        jax.debug.print("before attn q: {q}", q=q)
        jax.debug.print("before attn k: {k}", k=k)
        jax.debug.print("before attn v: {v}", v=v)
        attn_ret = self.attn(q, k, v, forward_batch, token_to_kv_pool)
        attn_output = attn_ret[0] if isinstance(attn_ret, tuple) else attn_ret
        jax.debug.print(
            "after attn attn_output: {attn_output}, shape: {shape}",
            attn_output=attn_output,
            shape=attn_output.shape,
        )

        # Project output
        output, _ = self.o_proj(attn_output)
        return output


class Grok1DecoderLayer(nnx.Module):
    """A single decoder layer of the Grok-1 model."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        rngs: nnx.Rngs,
        mesh: jax.sharding.Mesh,
    ) -> None:
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.residual_moe = getattr(config, "residual_moe", False)
        self.layer_id = layer_id

        # Self-attention
        rope_theta = getattr(config, "rope_theta", 10000)
        # Cache tensor-parallel axis size to avoid psum when no mesh context
        self.tp_axis_size = mesh.shape.get("tensor", 1)
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
            reduce_results=False,
            rngs=rngs,
            mesh=mesh,
        )

        # Feed-forward networks
        split_gate_up = not getattr(config, "merge_gate_up", True)
        if self.num_experts > 0:
            self.block_sparse_moe = Grok1MoE(
                config=config,
                layer_id=layer_id,
                num_experts=config.num_local_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=getattr(
                    config,
                    "moe_intermediate_size",
                    getattr(config, "intermediate_size", None),
                ),
                rngs=rngs,
                mesh=mesh,
                reduce_results=not self.residual_moe,
                inplace=False,  # not self.residual_moe,
                no_combine=False,  # self.residual_moe,
            )
            if self.residual_moe:
                self.mlp = Grok1MLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    layer_id=layer_id,
                    rngs=rngs,
                    reduce_results=False,
                    split_gate_up=split_gate_up,
                )
        else:
            raise NotImplementedError()

        # Layer normalization (using eps instead of epsilon to match PyTorch)
        self.pre_attn_norm = RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, rngs=rngs
        )
        self.post_attn_norm = RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, rngs=rngs
        )
        self.pre_moe_norm = RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, rngs=rngs
        )
        self.post_moe_norm = RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, rngs=rngs
        )

        # Setup FFN function based on configuration (matching PyTorch logic)
        if self.num_experts > 0:
            if self.residual_moe:
                self.ffn = self.moe_with_rmoe
            else:
                self.ffn = self.block_sparse_moe
        else:
            raise NotImplementedError()

    def moe_with_rmoe(self, x):
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
        residual: Optional[jax.Array] = None,
        deferred_norm: Optional[RMSNorm] = None,
    ) -> Tuple[jax.Array, jax.Array, RMSNorm]:

        # Self Attention block (matching PyTorch logic exactly)
        if deferred_norm is not None:
            assert residual is not None
            # Apply deferred norm from previous layer and pre-attention norm
            hidden_states, residual = dual_rmsnorm_forward(
                hidden_states,
                residual,
                deferred_norm.weight,
                self.pre_attn_norm.weight,
                deferred_norm.variance_epsilon,
            )
        else:
            # First layer or no deferred norm - use fused_rmsnorm equivalent
            hidden_states, residual = self.pre_attn_norm(hidden_states), hidden_states

        # Self-attention
        jax.debug.print(
            "before self_attn hidden_states: {hidden_states}",
            hidden_states=hidden_states,
        )
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )
        jax.debug.print(
            "after self_attn hidden_states: {hidden_states}",
            hidden_states=hidden_states,
        )

        # Apply post-attention norm and pre-MoE norm (matching PyTorch fused_dual_residual_rmsnorm)
        hidden_states, residual = dual_rmsnorm_forward(
            hidden_states,
            residual,
            self.post_attn_norm.weight,
            self.pre_moe_norm.weight,
            self.post_attn_norm.variance_epsilon,
        )

        # Feed-forward network
        if self.residual_moe:
            hidden_states = self.moe_with_rmoe(hidden_states)
        else:
            hidden_states = self.block_sparse_moe(hidden_states)

        # Return with deferred post-MoE norm (matching PyTorch)
        return hidden_states, residual, self.post_moe_norm


class Grok1Model(nnx.Module):
    """The main Grok-1 transformer model."""

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        rngs: nnx.Rngs,
        mesh: jax.sharding.Mesh,
    ) -> None:
        super().__init__()
        # TODO (chhzh123): remove this
        config.num_hidden_layers = 1
        print(f"config: {config}")
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Token embeddings (matching PyTorch VocabParallelEmbedding behavior)
        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            rngs=rngs,
        )

        # Transformer layers
        self.layers = [
            Grok1DecoderLayer(config=config, layer_id=i, mesh=mesh, rngs=rngs)
            for i in range(config.num_hidden_layers)
        ]

        # Final layer norm (using eps to match PyTorch)
        self.norm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps, rngs=rngs)

    def __call__(
        self,
        input_ids: jax.Array,
        positions: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        input_embeds: jax.Array = None,
    ) -> jax.Array:
        # Get embeddings
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
            # Apply embedding scaling if specified (matching PyTorch)
            if hasattr(self.config, "embedding_multiplier_scale"):
                hidden_states = hidden_states * self.config.embedding_multiplier_scale
        else:
            hidden_states = input_embeds

        # Process through transformer layers with deferred normalization
        residual, deferred_norm = None, None
        for layer in self.layers:
            hidden_states, residual, deferred_norm = layer(
                positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
                deferred_norm,
            )

        # Apply final normalization (matching PyTorch fused_dual_residual_rmsnorm)
        hidden_states, _ = dual_rmsnorm_forward(
            hidden_states,
            residual,
            deferred_norm.weight,
            self.norm.weight,
            deferred_norm.variance_epsilon,
        )

        return hidden_states


class Grok1ForCausalLM(nnx.Module):
    """Grok-1 model with a causal language modeling head."""

    def __init__(
        self,
        config: PretrainedConfig,
        # TODO(jcyang): should make the following two args keyword-only
        rngs: Optional[nnx.Rngs] = None,
        mesh: Optional[jax.sharding.Mesh] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh = mesh

        # Configuration flags (matching PyTorch implementation)
        self.replicate_lm_head = getattr(config, "replicate_lm_head", False)

        # Main model
        self.model = Grok1Model(config, rngs=rngs, mesh=mesh)

        # Language modeling head (matching PyTorch logic for replicated vs parallel)
        if self.replicate_lm_head:
            # ReplicatedLinear equivalent
            self.lm_head = LinearBase(
                input_size=config.hidden_size,
                output_size=config.vocab_size,
                use_bias=False,
                params_dtype=jnp.bfloat16,
                kernel_axes=None,
                rngs=rngs,
            )
            # Skip all-gather for replicated head
            self.logits_processor = LogitsProcessor(
                config.vocab_size, lm_head=self.lm_head, mesh=mesh
            )
        else:
            self.lm_head = ParallelLMHead(
                num_embeddings=config.vocab_size,
                features=config.hidden_size,
                param_dtype=jnp.bfloat16,
                rngs=rngs,
            )
            self.logits_processor = LogitsProcessor(
                config.vocab_size, lm_head=self.lm_head, mesh=mesh
            )

        self.loaded_param_names = set()

    def named_parameters(self):
        """Recursively yield (full_path, nnx.Param) over all parameters in this module.
        Paths resemble PyTorch-style dotted names, starting from this module.
        """
        params = {}

        def _collect(obj, prefix: str):
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
    ) -> tuple[jax.Array, list, bool]:
        """Forward pass through the model using unified forward_batch API."""
        input_ids = forward_batch.input_ids
        positions = forward_batch.positions
        jax.debug.print("input_ids: {input_ids}", input_ids=input_ids)
        jax.debug.print("positions: {positions}", positions=positions)
        hidden_states = self.model(
            input_ids, positions, forward_batch, token_to_kv_pool, None
        )
        output = self.logits_processor(hidden_states, logits_metadata)
        # Return values consistent with other models: (output, layers_kv_fused, layers_callback_flag)
        # Grok model does not expose per-layer KV tensors here, so return an empty list and True flag.
        return output, [], True

    def load_weights(
        self,
        weights: Iterable[Tuple[str, jax.Array]],
        ignore_parent_name: bool = False,
        check_hit_names: bool = True,
        model_config: PretrainedConfig = None,
    ) -> dict[str, jax.Array]:
        """Load model weights from iterator (matching PyTorch interface)."""
        if model_config is None:
            model_config = self.config

        # Weight mapping configurations (matching PyTorch stacked_params_mapping)
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Expert params mapping (matching PyTorch expert_params_mapping)
        num_experts = model_config.num_local_experts
        expert_params_mapping = []
        for expert_id in range(num_experts):
            expert_params_mapping.extend(
                [
                    (
                        "block_sparse_moe.experts.wi_0",
                        f"block_sparse_moe.experts.{expert_id}.w1",
                        expert_id,
                        0,
                    ),
                    (
                        "block_sparse_moe.experts.wi_1",
                        f"block_sparse_moe.experts.{expert_id}.w3",
                        expert_id,
                        1,
                    ),
                    (
                        "block_sparse_moe.experts.wo",
                        f"block_sparse_moe.experts.{expert_id}.w2",
                        expert_id,
                        0,
                    ),
                ]
            )

        params_dict = dict(self.named_parameters())
        all_names = set(params_dict.keys())
        hit_names = set()

        def load_weight_wrapper(name: str, loaded_weight: jax.Array, *args, **kwargs):
            # Fuse constant multipliers into the weights (matching PyTorch)
            if "lm_head" in name:
                loaded_weight = (
                    loaded_weight.astype(jnp.float32)
                    * model_config.output_multiplier_scale
                )

            original_name = name
            if ignore_parent_name:
                name = name.split(".")[-1]

            if name not in params_dict:
                logger.info(f"Skipping {name=} in load_weights_wrapper")
                return

            param = params_dict[name]
            # Use default weight loader (JAX equivalent)
            if hasattr(param, "value"):
                param.value = loaded_weight
            else:
                # Handle different parameter types
                setattr(self, name.split(".")[-1], loaded_weight)

            hit_names.add(name)
            self.loaded_param_names.add(original_name)

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # Handle stacked parameters
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for quantized models
                if name.endswith(".bias") and name not in params_dict:
                    continue
                load_weight_wrapper(name, loaded_weight, shard_id)
                break
            else:
                # Handle expert parameters
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    load_weight_wrapper(
                        name,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Handle regular parameters
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name is None:
                        continue
                    load_weight_wrapper(name=name, loaded_weight=loaded_weight)

        # Validation (matching PyTorch logic)
        if check_hit_names:
            if len(hit_names) > 5:
                missing = all_names - hit_names
                missing_exclude_scales = {x for x in missing if "scale" not in x}
                logger.info(
                    f"#all_names: {len(all_names)}, #hit_names: {len(hit_names)}, #missing_exclude_scales: {len(missing_exclude_scales)}",
                )
                if len(missing_exclude_scales) > 0:
                    raise ValueError(
                        f"load_weights failed because some weights are missing: {missing_exclude_scales=}."
                    )
            elif len(hit_names) == 0:
                raise ValueError(
                    f"load_weights failed because it did not hit any names. {all_names=} {hit_names=}"
                )

        return hit_names

    def get_num_params_analytical(self):
        """Calculate number of parameters analytically (matching PyTorch)."""
        cfg = self.config
        moe_intermediate_size = getattr(
            cfg,
            "moe_intermediate_size",
            getattr(cfg, "intermediate_size", None),
        )
        residual_moe = getattr(cfg, "residual_moe", False)
        if cfg.num_local_experts > 0:
            num_experts = cfg.num_local_experts + (1 if residual_moe else 0)
        else:
            num_experts = 1

        wq = (
            cfg.num_hidden_layers
            * cfg.hidden_size
            * cfg.num_attention_heads
            * cfg.head_dim
        )
        wkv = (
            cfg.num_hidden_layers
            * cfg.hidden_size
            * cfg.num_key_value_heads
            * cfg.head_dim
            * 2
        )
        out = (
            cfg.num_hidden_layers
            * cfg.hidden_size
            * cfg.num_attention_heads
            * cfg.head_dim
        )
        ffn1 = (
            cfg.num_hidden_layers
            * num_experts
            * cfg.hidden_size
            * moe_intermediate_size
            * 2
        )
        ffn2 = (
            cfg.num_hidden_layers
            * num_experts
            * cfg.hidden_size
            * moe_intermediate_size
        )
        embed = cfg.hidden_size * cfg.vocab_size * 2
        return wq + wkv + out + ffn1 + ffn2 + embed

    def get_num_params_torch(self):
        """Get actual number of parameters (JAX equivalent)."""

        def count_params(pytree):
            return sum(x.size for x in jax.tree_leaves(pytree) if hasattr(x, "size"))

        # Get mesh size for proper scaling
        mesh_size = self.mesh.size if self.mesh else 1
        return count_params(self) * mesh_size


class Grok1ModelForCausalLM(Grok1ForCausalLM):
    """An alias for backward-compatibility (matching PyTorch)."""

    pass


# Entry classes for model registration (matching PyTorch)
EntryClass = [Grok1ForCausalLM, Grok1ModelForCausalLM]
