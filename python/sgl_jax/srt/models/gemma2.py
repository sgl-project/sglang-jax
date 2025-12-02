import logging

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, RotaryEmbedding
from sgl_jax.srt.layers.layernorm import GemmaRMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


class Gemma2MLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
        mesh: jax.sharding.Mesh = None,
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
            use_bias=False,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
        )

        self.act_fn = jax.nn.gelu

    def __call__(self, hidden_states: jnp.ndarray):
        a1, _ = self.gate_proj(hidden_states)
        a2, _ = self.up_proj(hidden_states)
        intermediate_parallel = a2 * self.act_fn(a1)
        output, _ = self.down_proj(intermediate_parallel)
        return output


class Gemma2Attention(nnx.Module):
    def __init__(
        self,
        layer_id: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float = 10000,
        query_pre_attn_scalar: int = 256,
        sliding_window_size: int = 0,
        logit_cap: float = 0,
        attention_bias: bool = False,
        dtype: jnp.dtype = jnp.bfloat16,
        mesh: jax.sharding.Mesh = None,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = query_pre_attn_scalar**-0.5

        self.q_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.num_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.k_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.num_kv_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.v_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.num_kv_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.o_proj = LinearBase(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            use_bias=attention_bias,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
        )

        self.rotary_emb = RotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            dtype=dtype,
        )
        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=sliding_window_size,
            logit_cap=logit_cap,
        )
        self.layer_id = layer_id

    def __call__(
        self,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        q = q.reshape(-1, self.num_heads, self.head_dim)
        k = k.reshape(-1, self.num_kv_heads, self.head_dim)
        v = v.reshape(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(forward_batch.positions, q, k)
        attn_output, kv_fused = self.attn(
            q, k, v, forward_batch=forward_batch, token_to_kv_pool=token_to_kv_pool
        )
        output, _ = self.o_proj(attn_output)
        return output, kv_fused


class Gemma2DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
        mesh: jax.sharding.Mesh = None,
    ):
        self.layer_id = layer_id
        use_sliding_window = config.layer_types[layer_id] == "sliding_attention"
        self.self_attn = Gemma2Attention(
            layer_id,
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.max_position_embeddings,
            rope_theta=config.rope_theta,
            query_pre_attn_scalar=config.query_pre_attn_scalar,
            sliding_window_size=config.sliding_window if use_sliding_window else 0,
            logit_cap=config.attn_logit_softcapping,
            attention_bias=config.attention_bias,
            dtype=dtype,
            mesh=mesh,
        )
        self.mlp = Gemma2MLP(
            config.hidden_size,
            config.intermediate_size,
            layer_id=layer_id,
            dtype=dtype,
            mesh=mesh,
        )

        self.input_layernorm = GemmaRMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
        )
        self.pre_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
        )
        self.post_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: jax.Array | None = None,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states, kv_fused = self.self_attn(
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)

        return hidden_states, residual, kv_fused


class Gemma2Model(nnx.Module):
    """Gemma2 model"""

    def __init__(
        self,
        config: PretrainedConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        mesh: jax.sharding.Mesh = None,
    ):
        self.embed_tokens = Embed(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )

        self.layers = nnx.data(
            [
                Gemma2DecoderLayer(
                    config,
                    layer_id=i,
                    dtype=dtype,
                    mesh=mesh,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = GemmaRMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

        self.hidden_size = config.hidden_size

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ):
        hidden_states = self.embed_tokens(forward_batch.input_ids)
        hidden_states *= jnp.array([self.hidden_size**0.5], dtype=hidden_states.dtype)

        residual = None
        layers_kv_fused = []
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual, kv_fused = layer(
                hidden_states, forward_batch, token_to_kv_pool, residual
            )
            layers_kv_fused.append(kv_fused)

        if residual is not None:
            hidden_states += residual

        hidden_states = self.norm(hidden_states)
        return hidden_states, layers_kv_fused


class Gemma2ForCausalLM(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        mesh: jax.sharding.Mesh = None,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        self.model = Gemma2Model(config, dtype=self.dtype, mesh=mesh)
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size,
            soft_cap=self.config.final_logit_softcapping,
            mesh=self.mesh,
        )

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self, model_config=model_config, mesh=self.mesh, dtype=self.dtype
        )

        weight_mappings = self._create_weight_mappings()

        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("Gemma2 weights loaded successfully!")

    def _create_weight_mappings(self) -> dict:
        mappings = {
            "model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.norm.weight": WeightMapping(
                target_path="model.norm.weight", sharding=(None,), transpose=False
            ),
        }

        num_layers = self.config.num_hidden_layers
        for layer_idx in range(num_layers):
            layer_mappings = self._create_layer_mappings(layer_idx)
            mappings.update(layer_mappings)

        return mappings

    def _create_layer_mappings(self, layer_idx: int) -> dict:
        prefix = f"model.layers.{layer_idx}"

        return {
            f"{prefix}.input_layernorm.weight": WeightMapping(
                target_path=f"{prefix}.input_layernorm.weight",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.post_attention_layernorm.weight": WeightMapping(
                target_path=f"{prefix}.post_attention_layernorm.weight",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.post_feedforward_layernorm.weight": WeightMapping(
                target_path=f"{prefix}.post_feedforward_layernorm.weight",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.pre_feedforward_layernorm.weight": WeightMapping(
                target_path=f"{prefix}.pre_feedforward_layernorm.weight",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.self_attn.q_proj.weight": WeightMapping(
                target_path=f"{prefix}.self_attn.q_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=False,
            ),
            f"{prefix}.self_attn.k_proj.weight": WeightMapping(
                target_path=f"{prefix}.self_attn.k_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=True,
            ),
            f"{prefix}.self_attn.v_proj.weight": WeightMapping(
                target_path=f"{prefix}.self_attn.v_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=True,
            ),
            f"{prefix}.self_attn.o_proj.weight": WeightMapping(
                target_path=f"{prefix}.self_attn.o_proj.weight",
                sharding=("tensor", None),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=False,
            ),
            f"{prefix}.mlp.gate_proj.weight": WeightMapping(
                target_path=f"{prefix}.mlp.gate_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.up_proj.weight": WeightMapping(
                target_path=f"{prefix}.mlp.up_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.down_proj.weight": WeightMapping(
                target_path=f"{prefix}.mlp.down_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
        }

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, layers_kv_fused = self.model(forward_batch, token_to_kv_pool)
        output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)
        return output, layers_kv_fused, True


EntryClass = Gemma2ForCausalLM
