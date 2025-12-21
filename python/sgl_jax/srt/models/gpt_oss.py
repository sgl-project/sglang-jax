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

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

logger = logging.getLogger(__name__)

init_fn = nnx.initializers.uniform()

import math
from typing import Tuple

import jax
from flax import nnx
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike

def mxfp4_dequantization(
    scales,
    blocks,
    mesh: jax.sharding.Mesh,
    sharding_spec,
    dtype: jnp.dtype):
    nr_elem__per_blk = blocks.shape[-1] * 2

    high_bits = jnp.right_shift(blocks, 4) & 0x0F
    low_bits = blocks & 0x0F

    interleaved = jnp.stack([low_bits, high_bits], axis=-1)

    interleaved = interleaved.reshape(blocks.shape[:-1] + (nr_elem__per_blk,))

    e2m1__to__float = {
        0b1111:-6.0,
        0b1110:-4.0,
        0b1101:-3.0,
        0b1100:-2.0,
        0b1011:-1.5,
        0b1010:-1.0,
        0b1001:-0.5,
        0b1000:-0.0,
        0b0000:+0.0,
        0b0001:+0.5,
        0b0010:+1.0,
        0b0011:+1.5,
        0b0100:+2.0,
        0b0101:+3.0,
        0b0110:+4.0,
        0b0111:+6.0,
    }
    e2m1_lut = [e2m1__to__float[i] for i in range(16)]
    e2m1_lut = jnp.array(e2m1_lut, dtype=jnp.float16)
    e2m1_lut = jax.device_put(e2m1_lut, jax.sharding.NamedSharding(mesh, P()))

    interleaved = e2m1_lut.at[interleaved].get(out_sharding=interleaved.sharding)
    scales = scales.astype(dtype=jnp.int16)
    scales = scales - 127

    scales = scales[..., jnp.newaxis]

    dequantized = jax.numpy.ldexp(interleaved, scales).astype(dtype=dtype)

    dequantized = dequantized.reshape(dequantized.shape[:-2] + (-1,))
    dequantized = dequantized.transpose((0, 2, 1))
    dequantized = jax.device_put(dequantized, jax.sharding.NamedSharding(mesh, P(*sharding_spec)))

    return dequantized




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

    def _compute_concentration_and_inv_freq(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
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

    def _compute_cos_sin(self, positions):
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        positions = positions.astype(jnp.float32)
        freqs = jnp.einsum("i,j->ij", positions, inv_freq)
        cos = jnp.cos(freqs) * concentration
        sin = jnp.sin(freqs) * concentration
        return cos, sin

    def __call__(self, positions, query: jnp.ndarray, key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        num_tokens = query.shape[0]
        cos, sin = self._compute_cos_sin(positions)

        query_shape = query.shape
        query = query.reshape(num_tokens, -1, self.head_dim)
        query = _apply_rotary_emb(query, cos, sin)
        query = query.reshape(query_shape)

        key_shape = key.shape
        key = key.reshape(num_tokens, -1, self.head_dim)
        key = _apply_rotary_emb(key, cos, sin)
        key = key.reshape(key_shape)
        return query, key


class AttentionBlock(nnx.Module):
    def __init__(self,
                 config: ModelConfig,
                 mesh: jax.sharding.Mesh,
                 layer_idx: int,
                 dtype: jnp.dtype):
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0

        self.sinks = nnx.Param(jnp.zeros((config.num_attention_heads,), dtype))
        self.norm = RMSNorm(config.hidden_size, epsilon=1e-5, param_dtype=dtype)

        q_dim = self.num_attention_heads * self.head_dim
        self.q_proj = LinearBase(
            config.hidden_size,
            q_dim,
            mesh,
            use_bias=True,
            params_dtype=dtype,
            kernel_axes=("tensor", None))


        kv_dim = self.num_key_value_heads * self.head_dim
        self.k_proj = LinearBase(
            config.hidden_size,
            kv_dim,
            mesh,
            use_bias=True,
            params_dtype=dtype,
            kernel_axes=("tensor", None))
        self.v_proj = LinearBase(
            config.hidden_size,
            kv_dim,
            mesh,
            use_bias=True,
            params_dtype=dtype,
            kernel_axes=("tensor", None))

        self.sm_scale = 1.0 / math.sqrt(config.head_dim)
        self.rope = RotaryEmbedding(
            config.head_dim,
            config.rope_theta,
            initial_context_length=config.initial_context_length,
            scaling_factor=config.rope_scaling["factor"],
            # TODO
            # ntk_alpha=config.rope_ntk_alpha,
            ntk_alpha=config.rope_scaling["beta_slow"],
            # ntk_beta=config.rope_ntk_beta,
            ntk_beta=config.rope_scaling["beta_fast"]
        )

        self.attn = RadixAttention(
            num_heads=self.num_attention_heads,
            head_dim=self.head_dim,
            scaling=None,
            num_kv_heads=self.num_key_value_heads,
            layer_id=layer_idx,
        )

        self.o_proj = LinearBase(
            config.head_dim * config.num_attention_heads,
            config.hidden_size,
            mesh,
            use_bias=True,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"))



    def __call__(
        self,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache
    ):
        positions = forward_batch.positions
        t = self.norm(hidden_states)
        q, _ = self.q_proj(t)
        k, _ = self.k_proj(t)
        v, _ = self.v_proj(t)

        q = q.reshape(-1, self.num_attention_heads, self.head_dim)
        k = k.reshape(-1, self.num_key_value_heads, self.head_dim)
        v = v.reshape(-1, self.num_key_value_heads, self.head_dim)

        q, k = self.rope(positions, q, k)

        attn_output, kv_fused = self.attn(
            q, k, v,
            forward_batch, token_to_kv_pool,
            sinks=self.sinks.value,)

        t, _ = self.o_proj(attn_output)
        t = hidden_states + t
        return t, kv_fused



class MLPBlock(nnx.Module):
    def __init__(self, 
                 config: ModelConfig,
                 mesh: jax.sharding.Mesh,
                 dtype: jnp.dtype):
        self.num_experts = config.num_local_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit
        self.hidden_size = config.hidden_size
        self.expert_dim = config.intermediate_size
        self.alpha = 1.702
        self.limit = 7.0

        self.norm = RMSNorm(config.hidden_size, epsilon=1e-5, dtype=dtype)

        self.router = LinearBase(
            config.hidden_size,
            self.num_experts,
            mesh,
            use_bias=True,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"))

        self.gate_up_proj_scales = nnx.Param(
            jnp.zeros((self.num_experts, 2 * self.expert_dim, config.hidden_size // 32), dtype=jnp.uint8)
        )
        self.gate_up_proj_blocks = nnx.Param(
            jnp.zeros((self.num_experts, 2 * self.expert_dim, config.hidden_size // 32, 32 // 2), dtype=jnp.uint8)
        )
        self.gate_up_proj_bias = nnx.Param(
            jnp.zeros((self.num_experts, 2 * self.expert_dim), dtype=jnp.bfloat16)
        )

        self.down_proj_scales = nnx.Param(
            jnp.zeros((self.num_experts, self.hidden_size, self.expert_dim // 32), dtype=jnp.uint8)
        )
        self.down_proj_blocks = nnx.Param(
            jnp.zeros((self.num_experts, self.hidden_size, self.expert_dim // 32, 32 // 2), dtype=jnp.uint8)
        )
        self.down_proj_bias = nnx.Param(
            jnp.zeros((self.num_experts, self.hidden_size), dtype=jnp.bfloat16)
        )


    def init_weight(
        self,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype,
    ):
        self.gate_up_proj = nnx.Param(mxfp4_dequantization(
            self.gate_up_proj_scales.value,
            self.gate_up_proj_blocks.value,
            mesh, ("tensor", None, None), dtype)
        )
        delattr(self, "gate_up_proj_scales")
        delattr(self, "gate_up_proj_blocks")

        self.down_proj = nnx.Param(mxfp4_dequantization(
            self.down_proj_scales.value,
            self.down_proj_blocks.value,
            mesh, ("tensor", None, None), dtype)
        )
        delattr(self, "down_proj_scales")
        delattr(self, "down_proj_blocks")

    def __call__(self, x: Array) -> Array:
        t = self.norm(x)
        router_logits, _ = self.router(t)

        router_logits = jax.device_put(router_logits, P(None, None))

        router_top_value, router_indices = jax.lax.top_k(router_logits, k=self.experts_per_token)

        router_top_value = jax.device_put(router_top_value, P(None, None))
        router_indices = jax.device_put(router_indices, P(None, None))

        router_top_value = jax.nn.softmax(router_top_value, axis=-1)

        nr_tokens = router_logits.shape[0]
        token_indices = jnp.arange(nr_tokens)[:, None].repeat(self.experts_per_token, axis=1)
    
        router_scores = jnp.zeros_like(router_logits).at[token_indices, router_indices].set(router_top_value)
    

        t = t[None, ... ]
        t = jnp.repeat(t, self.num_experts, axis=0)
        
        gate_up = jnp.matmul(t, self.gate_up_proj) + self.gate_up_proj_bias[:, None, :]
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]

        gate = jnp.clip(gate, a_min=None, a_max=self.limit)
        up = jnp.clip(up, a_min=-self.limit, a_max=self.limit)
    
        glu = gate * jax.nn.sigmoid(gate * self.alpha)
        next_states = jnp.matmul((up + 1) * glu, self.down_proj) + self.down_proj_bias[..., None, :]
        
        next_states = next_states * router_scores.transpose((1, 0))[..., None]
        next_states = jnp.sum(next_states, axis=0)
        return next_states + x


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_idx: int,
        dtype: jnp.dtype
    ):
        self.layer_idx = layer_idx
        self.attn = AttentionBlock(config, mesh, layer_idx, dtype)
        self.mlp = MLPBlock(config, mesh, dtype)

    def __call__(
        self,
        hidden_states,
        forward_batch,
        token_to_kv_pool
    ) -> Array:
        layer_callback_flag = []
        t, kv_fused = self.attn(
            hidden_states,
            forward_batch,
            token_to_kv_pool)
        attn_callback_flag = precision_tracer.jit_pure_callback_record(
            t, "self_attn_output", "SELF_ATTN", self.layer_idx
        )
        layer_callback_flag.append(attn_callback_flag)
        
        x = self.mlp(t)
        mlp_callback_flag = precision_tracer.jit_pure_callback_record(
            x, "mlp_output", "MLP", self.layer_idx
        )
        layer_callback_flag.append(mlp_callback_flag)
        return x, kv_fused, layer_callback_flag


class GptOssModel(nnx.Module):
    def __init__(
        self,   
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype
    ):
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

        self.norm = RMSNorm(config.hidden_size, epsilon=1e-5, dtype=dtype)

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

        for block in self.block:
            x, kv_fused, callback_flag = block(x, forward_batch, token_to_kv_pool)
            layers_kv_fused.append(kv_fused)
            layers_callback_flag.extend(callback_flag)

        x = self.norm(x)

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

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.dtype,
            kernel_axes=("tensor", None),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=self.mesh)

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_llama_weight_mappings()
        loader.load_weights_from_safetensors(weight_mappings)

        for layer_id in range(self.config.num_hidden_layers):
            self.model.block[layer_id].mlp.init_weight(self.mesh, self.dtype)

        params = nnx.state(self.model)
        nnx.update(self.model, params)
        logger.info("GptOss weights loaded successfully!")

    def _create_llama_weight_mappings(self) -> dict:
        mappings = {
            "model.embed_tokens.weight": WeightMapping(
                target_path="model.embedding.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
        }

        for layer_idx in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            target_prefix = f"model.block.{layer_idx}"
            mappings.update({
                f"{prefix}.input_layernorm.weight": WeightMapping(
                    target_path=f"{target_prefix}.attn.norm.scale",
                    sharding=("tensor", ),
                    transpose=False,
                ),
            })
            mappings.update({
                f"{prefix}.self_attn.q_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.attn.q_proj.bias",
                    sharding=("tensor", ),
                    transpose=False,
                ),
            })
            mappings.update({
                f"{prefix}.self_attn.q_proj.weight": WeightMapping(
                    target_path=f"{target_prefix}.attn.q_proj.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                ),
            })
            mappings.update({
                f"{prefix}.self_attn.k_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.attn.k_proj.bias",
                    sharding=("tensor", ),
                    transpose=False,
                ),
            })
            mappings.update({
                f"{prefix}.self_attn.k_proj.weight": WeightMapping(
                    target_path=f"{target_prefix}.attn.k_proj.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                ),
            })
            mappings.update({
                f"{prefix}.self_attn.v_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.attn.v_proj.bias",
                    sharding=("tensor", ),
                    transpose=False,
                ),
            })
            mappings.update({
                f"{prefix}.self_attn.v_proj.weight": WeightMapping(
                    target_path=f"{target_prefix}.attn.v_proj.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                ),
            })
            mappings.update({
                f"{prefix}.self_attn.o_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.attn.o_proj.bias",
                    sharding=("tensor", ),
                    transpose=False,
                ),
            })
            mappings.update({
                f"{prefix}.self_attn.o_proj.weight": WeightMapping(
                    target_path=f"{target_prefix}.attn.o_proj.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                ),
            })
            mappings.update({
                f"{prefix}.self_attn.sinks": WeightMapping(
                    target_path=f"{target_prefix}.attn.sinks",
                    sharding=("tensor", ),
                    transpose=False,
                ),
            })
            mappings.update({
                f"{prefix}.post_attention_layernorm.weight": WeightMapping(
                    target_path=f"{target_prefix}.mlp.norm.scale",
                    sharding=("tensor", ),
                    transpose=False,
                ),
            })
            mappings.update({
                f"{prefix}.mlp.router.weight": WeightMapping(
                    target_path=f"{target_prefix}.mlp.router.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                ),
            })
            mappings.update({
                f"{prefix}.mlp.router.bias": WeightMapping(
                    target_path=f"{target_prefix}.mlp.router.bias",
                    sharding=("tensor", ),
                    transpose=False,
                ),
            })
            mappings.update({
                f"{prefix}.mlp.experts.gate_up_proj_scales": WeightMapping(
                    target_path=f"{target_prefix}.mlp.gate_up_proj_scales",
                    sharding=("tensor", None, None),
                    transpose=False,
                ),
            })
            mappings.update({
                f"{prefix}.mlp.experts.gate_up_proj_blocks": WeightMapping(
                    target_path=f"{target_prefix}.mlp.gate_up_proj_blocks",
                    sharding=("tensor", None, None, None),
                    transpose=False,
                ),
            })
            mappings.update({
                f"{prefix}.mlp.experts.gate_up_proj_bias": WeightMapping(
                    target_path=f"{target_prefix}.mlp.gate_up_proj_bias",
                    sharding=("tensor", None),
                    transpose=False,
                ),
            })
            mappings.update({
                f"{prefix}.mlp.experts.down_proj_scales": WeightMapping(
                    target_path=f"{target_prefix}.mlp.down_proj_scales",
                    sharding=("tensor", None, None),
                    transpose=False,
                ),
            })
            mappings.update({
                f"{prefix}.mlp.experts.down_proj_blocks": WeightMapping(
                    target_path=f"{target_prefix}.mlp.down_proj_blocks",
                    sharding=("tensor", None, None, None),
                    transpose=False,
                ),
            })
            mappings.update({
                f"{prefix}.mlp.experts.down_proj_bias": WeightMapping(
                    target_path=f"{target_prefix}.mlp.down_proj_bias",
                    sharding=("tensor", None),
                    transpose=False,
                ),
            })

        
        mappings.update({
            f"model.norm.weight": WeightMapping(
                target_path=f"model.norm.scale",
                sharding=("tensor", ),
                transpose=False,
            ),
        })
        mappings.update({
            f"lm_head.weight": WeightMapping(
                target_path=f"lm_head.embedding",
                sharding=(None, "tensor"),
                transpose=False,
            ),
        })

        return mappings

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ):
        x, layers_kv_fused, layers_callback_flag = self.model(forward_batch, token_to_kv_pool)
        x = jax.device_put(x, jax.sharding.NamedSharding(self.mesh, P()))
        output = self.logits_processor(x, self.lm_head, logits_metadata)
        output = jax.device_put(output, jax.sharding.NamedSharding(self.mesh, P(None, "tensor")))
        return output, layers_kv_fused, layers_callback_flag


EntryClass = GptOssForCausalLM
