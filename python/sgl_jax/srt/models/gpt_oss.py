import logging
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, RotaryEmbedding
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.precision_tracer import precision_tracer
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)

init_fn = nnx.initializers.uniform()

# import dataclasses
import math
# from functools import partial
from typing import Tuple

import jax
from flax import nnx
# from jax import P
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike



# @dataclasses.dataclass(frozen=True)
# class ModelConfig:
#     num_hidden_layers: int
#     num_experts: int
#     experts_per_token: int
#     vocab_size: int
#     hidden_size: int
#     intermediate_size: int
#     swiglu_limit: float
#     head_dim: int
#     num_attention_heads: int
#     num_key_value_heads: int
#     sliding_window: int
#     initial_context_length: int
#     rope_theta: float
#     rope_scaling_factor: float
#     rope_ntk_alpha: float
#     rope_ntk_beta: float

#     @classmethod
#     def _from_param(cls, **kwargs):
#         return cls(**kwargs)

#     @classmethod
#     def default(cls):
#         """Default OSS model configuration."""
#         return cls._from_param(
#             num_hidden_layers=36,
#             num_experts=128,
#             experts_per_token=4,
#             vocab_size=201088,
#             hidden_size=2880,
#             intermediate_size=2880,
#             swiglu_limit=7.0,
#             head_dim=64,
#             num_attention_heads=64,
#             num_key_value_heads=8,
#             sliding_window=128,
#             initial_context_length=4096,
#             rope_theta=150000.0,
#             rope_scaling_factor=32.0,
#             rope_ntk_alpha=1.0,
#             rope_ntk_beta=32.0,
#         )



# class RMSNorm(nnx.Module):
#     def __init__(self, num_features: int, cfg: ModelConfig, *, rngs: nnx.Rngs, eps: float = 1e-05):
#         self.num_features = num_features
#         self.eps = eps
#         self.scale = nnx.Param(nnx.initializers.ones_init()(rngs.params(), (num_features,)))
        

#     @jax.named_scope("rms_norm")
#     def __call__(self, x: Array) -> Array:
#         assert x.shape[-1] == self.num_features
#         dtype = x.dtype
#         t = jnp.astype(x, jnp.float32)
#         rms = jnp.sqrt(jnp.mean(t**2, axis=-1, keepdims=True) + self.eps)
#         return jnp.astype((t / rms) * self.scale.value, dtype)


def _apply_rotary_emb(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    """Apply rotary embedding to x."""
    cos = jnp.expand_dims(cos, axis=-2).astype(x.dtype)
    sin = jnp.expand_dims(sin, axis=-2).astype(x.dtype)
    x1, x2 = jnp.split(x, 2, axis=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return jnp.concatenate((o1, o2), axis=-1)


class RotaryEmbedding(nnx.Module):
    def __init__(
        self,
        head_dim: int,
        base: float,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
    ):
        self.head_dim = head_dim
        self.base = base
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta

    def _compute_concentration_and_inv_freq(self, num_tokens: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """See YaRN paper: https://arxiv.org/abs/2309.00071"""
        freq = self.base ** (jnp.arange(0, self.head_dim, 2, dtype=jnp.float32) / self.head_dim)

        if self.scaling_factor > 1.0:
            concentration = 0.1 * jnp.log(self.scaling_factor) + 1.0  # YaRN concentration

            d_half = self.head_dim / 2
            # NTK by parts
            low = (
                d_half
                * jnp.log(self.initial_context_length / (self.ntk_beta * 2 * jnp.pi))
                / jnp.log(self.base)
            )
            high = (
                d_half
                * jnp.log(self.initial_context_length / (self.ntk_alpha * 2 * jnp.pi))
                / jnp.log(self.base)
            )

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (jnp.arange(d_half, dtype=jnp.float32) - low) / (high - low)
            mask = 1 - jnp.clip(ramp, 0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin(self, num_tokens: int):
        concentration, inv_freq = self._compute_concentration_and_inv_freq(num_tokens)
        t = jnp.arange(num_tokens, dtype=jnp.float32)
        freqs = jnp.einsum("i,j->ij", t, inv_freq)
        cos = jnp.cos(freqs) * concentration
        sin = jnp.sin(freqs) * concentration
        return cos, sin

    @jax.named_scope("rope")
    def __call__(self, query: jnp.ndarray, key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        num_tokens = query.shape[0]
        cos, sin = self._compute_cos_sin(num_tokens)

        query_shape = query.shape
        query = query.reshape(num_tokens, -1, self.head_dim)
        query = _apply_rotary_emb(query, cos, sin)
        query = query.reshape(query_shape)

        key_shape = key.shape
        key = key.reshape(num_tokens, -1, self.head_dim)
        key = _apply_rotary_emb(key, cos, sin)
        key = key.reshape(key_shape)
        return query, key


def sdpa(
    Q: jnp.ndarray,
    K: jnp.ndarray,
    V: jnp.ndarray,
    S: jnp.ndarray,
    sm_scale: float,
    sliding_window: int = 0,
) -> jnp.ndarray:
    """Scaled dot-product attention with sliding window and sink tokens."""
    # sliding_window == 0 means no sliding window
    n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)

    # Expand K and V to match Q's q_mult dimension
    K = jnp.expand_dims(K, axis=2)  # [n_tokens, n_heads, 1, d_head]
    K = jnp.broadcast_to(K, (n_tokens, n_heads, q_mult, d_head))
    V = jnp.expand_dims(V, axis=2)  # [n_tokens, n_heads, 1, d_head]
    V = jnp.broadcast_to(V, (n_tokens, n_heads, q_mult, d_head))

    # Expand S to match attention shape
    S = S.reshape(n_heads, q_mult, 1, 1)
    S = jnp.broadcast_to(S, (n_heads, q_mult, n_tokens, 1))

    # Create causal mask
    mask = jnp.triu(jnp.full((n_tokens, n_tokens), -jnp.inf), k=1)
    if sliding_window > 0:
        mask += jnp.tril(jnp.full((n_tokens, n_tokens), -jnp.inf), k=-sliding_window)

    # Compute attention scores
    QK = jnp.einsum("qhmd,khmd->hmqk", Q, K)
    QK = QK * sm_scale
    QK = QK + mask[None, None, :, :]

    # Concatenate sink tokens
    QK = jnp.concatenate([QK, S], axis=-1)

    # Softmax
    W = jax.nn.softmax(QK, axis=-1)
    W = W[..., :-1]  # Remove sink dimension

    # Apply attention weights
    attn = jnp.einsum("hmqk,khmd->qhmd", W, V)
    return attn.reshape(n_tokens, -1)


class AttentionBlock(nnx.Module):
    def __init__(self,
                 config: ModelConfig,
                 mesh: jax.sharding.Mesh,
                 layer_idx: int,
                 dtype: jnp.dtype):
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        # Only apply sliding window to every other layer
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0

        # Sink tokens parameter
        self.sinks = nnx.Param(jnp.zeros((config.num_attention_heads,), dtype))
        # self.norm = RMSNorm(config.hidden_size, config, rngs=rngs)
        self.norm = RMSNorm(config.hidden_size, epsilon=1e-5, param_dtype=dtype)

        # QKV projection
        qkv_dim = config.head_dim * (config.num_attention_heads + 2 * config.num_key_value_heads)
        # self.qkv = nnx.Linear(config.hidden_size, qkv_dim, use_bias=False, dtype=jnp.bfloat16, rngs=rngs)
        self.qkv = LinearBase(
            config.hidden_size,
            qkv_dim,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"))

        # Output projection
        # self.out = nnx.Linear(
        #         config.head_dim * config.num_attention_heads,
        #         config.hidden_size,
        #         use_bias=False,
        #         dtype=jnp.bfloat16,
        #         rngs=nnx.Rngs(0),
        #     )
        self.out = LinearBase(
            config.head_dim * config.num_attention_heads,
            config.hidden_size,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"))

        self.sm_scale = 1.0 / math.sqrt(config.head_dim)
        self.rope = RotaryEmbedding(
            config.head_dim,
            config.rope_theta,
            initial_context_length=config.initial_context_length,
            scaling_factor=config.rope_scaling["factor"],
            # ntk_alpha=config.rope_ntk_alpha,
            ntk_alpha=config.rope_scaling["beta_slow"],
            # ntk_beta=config.rope_ntk_beta,
            ntk_beta=config.rope_scaling["beta_fast"]
        )

    def __call__(self, x: Array) -> Array:
        t = self.norm(x)
        qkv, _ = self.qkv(t)

        # Split QKV
        # qkv shape: [batch, seq_len, qkv_dim]
        # qkv_dim = num_attention_heads * head_dim + 2 * num_key_value_heads * head_dim
        q_dim = self.num_attention_heads * self.head_dim
        kv_dim = self.num_key_value_heads * self.head_dim
        q = qkv[:, :, :q_dim]  # [batch, seq_len, num_attention_heads * head_dim]
        k = qkv[:, :, q_dim:q_dim + kv_dim]  # [batch, seq_len, num_key_value_heads * head_dim]
        v = qkv[:, :, q_dim + kv_dim:q_dim + 2 * kv_dim]  # [batch, seq_len, num_key_value_heads * head_dim]

        # Reshape for attention
        q = q.reshape(-1, self.num_key_value_heads, self.num_attention_heads // self.num_key_value_heads, self.head_dim)
        k = k.reshape(-1, self.num_key_value_heads, self.head_dim)
        v = v.reshape(-1, self.num_key_value_heads, self.head_dim)

        # Apply RoPE
        q, k = self.rope(q, k)

        # Apply attention
        t = sdpa(q, k, v, self.sinks.value, self.sm_scale, self.sliding_window)
        t = self.out(t)
        t = x + t
        return t


def swiglu(x: jnp.ndarray, alpha: float = 1.702, limit: float = 7.0) -> jnp.ndarray:
    """Swish-Gated Linear Unit activation."""
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = jnp.clip(x_glu, a_min=None, a_max=limit)
    x_linear = jnp.clip(x_linear, a_min=-limit, a_max=limit)
    out_glu = x_glu * jax.nn.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)


class MLPBlock(nnx.Module):
    def __init__(self, 
                 config: ModelConfig,
                 mesh: jax.sharding.Mesh,
                 dtype: jnp.dtype):
        # self.num_experts = config.num_experts
        self.num_experts = 128
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit

        # self.norm = RMSNorm(config.hidden_size, config, rngs=rngs)
        self.norm = RMSNorm(config.hidden_size, epsilon=1e-5, dtype=dtype)

        # Gate projection
        # self.gate = nnx.Linear(config.hidden_size, config.num_experts, use_bias=False, dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
        self.gate = LinearBase(
            config.hidden_size,
            self.num_experts,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"))

        # MLP weights (per expert)
        # mlp1: [num_experts, intermediate_size * 2, hidden_size]
        self.mlp1_weight = nnx.Param(
                nnx.initializers.normal()(
                    nnx.Rngs(0).params(),
                    (self.num_experts, config.intermediate_size * 2, config.hidden_size),
                )
            )


        # mlp1 bias: [num_experts, intermediate_size * 2]
        self.mlp1_bias = nnx.Param(
                nnx.initializers.normal()(nnx.Rngs(0).params(), (self.num_experts, config.intermediate_size * 2))
            )

        # mlp2: [num_experts, hidden_size, intermediate_size]
        self.mlp2_weight = nnx.Param(
                nnx.initializers.normal()(
                    nnx.Rngs(0).params(),
                    (self.num_experts, config.hidden_size, config.intermediate_size),
                )
            )

        # mlp2 bias: [num_experts, hidden_size]
        self.mlp2_bias = nnx.Param(nnx.initializers.normal()(nnx.Rngs(0).params(), (self.num_experts, config.hidden_size)))

    def __call__(self, x: Array) -> Array:
        t = self.norm(x)
        g = self.gate(t)  # [batch, seq_len, num_experts] or [batch, num_experts]

        # Top-k expert selection
        # torch.topk operates on dim=-1 (last dimension), so if g is [batch, seq_len, num_experts],
        # topk returns [batch, seq_len, experts_per_token]
        expert_weights, expert_indices = jax.lax.top_k(g, k=self.experts_per_token)
        # expert_weights: [batch, seq_len, experts_per_token] or [batch, experts_per_token]
        # expert_indices: [batch, seq_len, experts_per_token] or [batch, experts_per_token]
        # torch softmax on dim=1 corresponds to axis=1 in jax (over experts_per_token dimension)
        expert_weights = jax.nn.softmax(expert_weights, axis=-1)

        # MLP #1
        # Gather expert weights
        # If expert_indices is [batch, seq_len, experts_per_token], then:
        # mlp1_weight[expert_indices, ...] -> [batch, seq_len, experts_per_token, intermediate_size * 2, hidden_size]
        # Access Param value - use direct indexing which should work with nnx.Param
        # If _value is ShapeDtypeStruct, nnx will handle it during computation
        try:
            mlp1_weight = self.mlp1_weight[expert_indices, ...]
            mlp1_bias = self.mlp1_bias[expert_indices, ...]
        except (TypeError, AttributeError) as e:
            # Fallback: try to access _value directly
            from flax.nnx import variablelib
            from jax import ShapeDtypeStruct
            if isinstance(self.mlp1_weight, variablelib.Param):
                mlp1_weight_val = getattr(self.mlp1_weight, '_value', None)
                if mlp1_weight_val is not None and not isinstance(mlp1_weight_val, ShapeDtypeStruct):
                    mlp1_weight = mlp1_weight_val[expert_indices, ...]
                else:
                    # If still ShapeDtypeStruct, this means weights weren't loaded
                    # Try to get from the model's state directly
                    raise RuntimeError(f"mlp1_weight is ShapeDtypeStruct - weights not loaded. Error: {e}")
            else:
                raise
        
        # einsum: handle both 2D and 3D cases
        # mlp1_weight: [batch, seq_len, experts_per_token, intermediate_size * 2, hidden_size] or [batch, experts_per_token, intermediate_size * 2, hidden_size]
        # t: [batch, seq_len, hidden_size] or [batch, hidden_size]
        # Use 'k' for hidden_size dimension to avoid conflict with 'c' (intermediate_size * 2)
        if len(mlp1_weight.shape) == 5:
            # 3D case: [batch, seq_len, experts_per_token, intermediate_size * 2, hidden_size]
            # bseck: batch, seq_len, experts_per_token, intermediate_size*2, hidden_size
            # bsk: batch, seq_len, hidden_size
            # bsec: batch, seq_len, experts_per_token, intermediate_size*2
            t = jnp.einsum("bseck,bsk->bsec", mlp1_weight, t) + mlp1_bias
        else:
            # 2D case: [batch, experts_per_token, intermediate_size * 2, hidden_size]
            # beck: batch, experts_per_token, intermediate_size*2, hidden_size
            # bk: batch, hidden_size
            # bec: batch, experts_per_token, intermediate_size*2
            t = jnp.einsum("beck,bk->bec", mlp1_weight, t) + mlp1_bias
        t = swiglu(t, limit=self.swiglu_limit)

        # MLP #2
        try:
            mlp2_weight = self.mlp2_weight[expert_indices, ...]
            mlp2_bias = self.mlp2_bias[expert_indices, ...]
        except (TypeError, AttributeError) as e:
            from flax.nnx import variablelib
            from jax import ShapeDtypeStruct
            if isinstance(self.mlp2_weight, variablelib.Param):
                mlp2_weight_val = getattr(self.mlp2_weight, '_value', None)
                if mlp2_weight_val is not None and not isinstance(mlp2_weight_val, ShapeDtypeStruct):
                    mlp2_weight = mlp2_weight_val[expert_indices, ...]
                else:
                    raise RuntimeError(f"mlp2_weight is ShapeDtypeStruct - weights not loaded. Error: {e}")
            else:
                raise
        
        # einsum: handle both 2D and 3D cases
        # mlp2_weight: [batch, seq_len, experts_per_token, hidden_size, intermediate_size] or [batch, experts_per_token, hidden_size, intermediate_size]
        # t after swiglu: [batch, seq_len, experts_per_token, intermediate_size] or [batch, experts_per_token, intermediate_size]
        # From PyTorch: einsum("beck,bek->bec") where:
        # - mlp2_weight: [batch, experts_per_token, hidden_size, intermediate_size] (b, e, c, k)
        # - t: [batch, experts_per_token, intermediate_size] (b, e, k)
        # - output: [batch, experts_per_token, hidden_size] (b, e, c)
        if len(mlp2_weight.shape) == 5:
            # 3D case: [batch, seq_len, experts_per_token, hidden_size, intermediate_size]
            # t: [batch, seq_len, experts_per_token, intermediate_size]
            # Use 'bseck,bsek->bsec' where s=seq_len, e=experts_per_token, c=hidden_size, k=intermediate_size
            t = jnp.einsum("bseck,bsek->bsec", mlp2_weight, t) + mlp2_bias
        else:
            # 2D case: [batch, experts_per_token, hidden_size, intermediate_size]
            # t: [batch, experts_per_token, intermediate_size]
            t = jnp.einsum("beck,bek->bec", mlp2_weight, t) + mlp2_bias
        
        # Combine expert outputs with weights
        # expert_weights: [batch, seq_len, experts_per_token] or [batch, experts_per_token]
        # t: [batch, seq_len, experts_per_token, hidden_size] or [batch, experts_per_token, hidden_size]
        # From PyTorch: torch.einsum("bec,be->bc", t, expert_weights)
        # where t is [batch, experts_per_token, hidden_size] and expert_weights is [batch, experts_per_token]
        if len(expert_weights.shape) == 3:
            # 3D case: [batch, seq_len, experts_per_token]
            # t: [batch, seq_len, experts_per_token, hidden_size]
            # Output: [batch, seq_len, hidden_size]
            t = jnp.einsum("bsec,bse->bsc", t, expert_weights)
        else:
            # 2D case: [batch, experts_per_token]
            # t: [batch, experts_per_token, hidden_size]
            # Output: [batch, hidden_size]
            t = jnp.einsum("bec,be->bc", t, expert_weights)

        return x + t


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_idx: int,
        dtype: jnp.dtype):
        self.layer_idx = layer_idx
        self.attn = AttentionBlock(config, mesh, layer_idx, dtype)
        self.mlp = MLPBlock(config, mesh, dtype)

    def __call__(self, x: Array) -> Array:
        layer_callback_flag = []
        x = self.attn(x)
        attn_callback_flag = precision_tracer.jit_pure_callback_record(
            x, "self_attn_output", "SELF_ATTN", self.layer_idx
        )
        layer_callback_flag.append(attn_callback_flag)
        
        x = self.mlp(x)
        mlp_callback_flag = precision_tracer.jit_pure_callback_record(
            x, "mlp_output", "MLP", self.layer_idx
        )
        layer_callback_flag.append(mlp_callback_flag)
        kv_fused = None
        return x, kv_fused, layer_callback_flag


class GptOssModel(nnx.Module):
    def __init__(
        self,   
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype
    ):
        # self.embedding = nnx.Embed(
        #         num_embeddings=config.vocab_size,
        #         features=config.hidden_size,
        #         dtype=jnp.bfloat16,
        #         rngs=rngs,
        #     )
        self.embedding = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            kernel_axes=("tensor", None),
            param_dtype=dtype,
            mesh=mesh,
        )

        self.block = nnx.data(
            [TransformerBlock(config, mesh, layer_idx, dtype)
             for layer_idx in range(config.num_hidden_layers)]
        )

        # self.norm = RMSNorm(config.hidden_size, config, rngs=rngs)
        self.norm = RMSNorm(config.hidden_size,epsilon=1e-5, dtype=dtype)

        # self.unembedding = nnx.Linear(
        #         config.hidden_size,
        #         config.vocab_size,
        #         use_bias=False,
        #         dtype=jnp.bfloat16,
        #         rngs=rngs,
        #     )
        self.unembedding = LinearBase(
            config.hidden_size,
            config.vocab_size,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"))

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache
    ):
        layers_kv_fused = []
        layers_callback_flag = []

        x = self.embedding(forward_batch.input_ids)

        callback_flag = precision_tracer.jit_pure_callback_record(
            x, "embedding", "EMBEDDING"
        )
        layers_callback_flag.append(callback_flag)

        return x, None, layers_callback_flag
        for block in self.block:
            x, kv_fused, callback_flag = block(x)
            layers_kv_fused.append(kv_fused)
            layers_callback_flag.extend(callback_flag)

        x = self.norm(x)
        x = self.unembedding(x)

        callback_flag = precision_tracer.jit_pure_callback_record(
            x, "transformer_output", "TRANSFORMER"
        )
        layers_callback_flag.append(callback_flag)
        return x, layers_kv_fused, layers_callback_flag


class GptOssForCausalLM(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        logger.info("GptOssForCausalLM config dtype: %s", self.dtype)
        self.model = GptOssModel(config, mesh, dtype)
        pass

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_llama_weight_mappings()
        loader.load_weights_from_safetensors(weight_mappings)

    def _create_llama_weight_mappings(self) -> dict:
        mappings = {
            "model.embed_tokens.weight": WeightMapping(
                target_path="model.embedding.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            }
        return mappings

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ):
        x, layers_kv_fused, layers_callback_flag = self.model(forward_batch, token_to_kv_pool)
        return x, layers_kv_fused, layers_callback_flag


EntryClass = GptOssForCausalLM
