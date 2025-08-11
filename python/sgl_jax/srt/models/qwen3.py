from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from transformers import PretrainedConfig

from sgl_jax.srt.debug_tracer import global_tracer, trace_function
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, RotaryEmbedding
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsProcessor
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils import (
    flatten_pytree_with_paths,
    get_expected_param_paths,
    update_state_recursive,
)


class QWen3Attention(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = None,
        layer_id: int = 0,
        attention_bias: bool = False,
        rngs: nnx.Rngs = None,
    ):
        self.layer_id = layer_id
        assert num_heads % num_kv_heads == 0
        self.head_dim = head_dim or hidden_size // num_heads

        self.q_size = num_heads * self.head_dim
        self.kv_size = num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_norm = RMSNorm(self.head_dim, epsilon=rms_norm_eps, rngs=rngs)
        self.k_norm = RMSNorm(self.head_dim, epsilon=rms_norm_eps, rngs=rngs)

        self.q_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            rngs=rngs,
        )
        self.k_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_kv_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            rngs=rngs,
        )
        self.v_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_kv_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            rngs=rngs,
        )
        self.o_proj = LinearBase(
            input_size=num_heads * self.head_dim,
            output_size=hidden_size,
            use_bias=attention_bias,
            kernel_axes=("tensor", None),
            rngs=rngs,
        )
        self.rotary_emb = RotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            is_neox_style=False,
            dtype=jnp.bfloat16,
        )

        self.attn = Attention(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            scale=self.scaling,
        )

    @trace_function(stage="ATTENTION", include_args=False, include_output=True)
    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
    ) -> jax.Array:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        q_by_head = q.reshape(-1, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.reshape(q.shape)

        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.reshape(k.shape)

        q, k = self.rotary_emb(positions, q, k)
        global_tracer.print(q, f"rotary_emb_output_q", f"attention_layer_id_{self.layer_id}")
        global_tracer.print(k, f"rotary_emb_output_k", f"attention_layer_id_{self.layer_id}")
        attn_output = self.attn(q, k, v, forward_batch=forward_batch, is_causal=True)
        global_tracer.print(attn_output, f"attn_output", f"attention_layer_id_{self.layer_id}")

        output, _ = self.o_proj(attn_output)
        return output


class Qwen3MLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int = 0,
        rngs: nnx.Rngs = None,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.layer_id = layer_id

        self.gate_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            rngs=rngs,
        )

        self.up_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            rngs=rngs,
        )

        self.down_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            kernel_axes=("tensor", None),
            use_bias=False,
            params_dtype=dtype,
            rngs=rngs,
        )

        self.act_fn = jax.nn.silu

    @trace_function(stage="MLP", include_args=False, include_output=True)
    def __call__(self, hidden_states: jnp.ndarray):
        a1, _ = self.gate_proj(hidden_states)
        a2, _ = self.up_proj(hidden_states)
        global_tracer.print(a1, f"a1_output", f"mlp_layer_id_{self.layer_id}")
        global_tracer.print(a2, f"a2_output", f"mlp_layer_id_{self.layer_id}")
        intermediate_parallel = a2 * self.act_fn(a1)
        intermediate_parallel = jax.lax.with_sharding_constraint(
            intermediate_parallel, PartitionSpec(None, "tensor")
        )
        global_tracer.print(
            intermediate_parallel, f"act_fn_output", f"mlp_layer_id_{self.layer_id}"
        )
        output, _ = self.down_proj(intermediate_parallel)

        return output


class QWen3DecoderLayer(nnx.Module):
    def __init__(self, config: PretrainedConfig, layer_id: int = 0, rngs: nnx.Rngs = None):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        head_dim = getattr(config, "head_dim", None)
        self.self_attn = QWen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=head_dim,
            rms_norm_eps=config.rms_norm_eps,
            layer_id=layer_id,
            attention_bias=config.attention_bias,
            rngs=rngs,
        )

        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            layer_id=layer_id,
            rngs=rngs,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, rngs=rngs
        )

    @trace_function(stage="QWen3DecoderLayer", include_args=False, include_output=True)
    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        residual: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        global_tracer.print(
            hidden_states, f"post_attention_layernorm_output", f"decoder_layer_id_{self.layer_id}"
        )
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class QWen3Model(nnx.Module):
    def __init__(self, config: PretrainedConfig, rngs: nnx.Rngs = None):
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            rngs=rngs,
        )

        self.layers = [
            QWen3DecoderLayer(
                config=config,
                layer_id=i,
                rngs=rngs,
            )
            for i in range(config.num_hidden_layers)
        ]

        self.norm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps, rngs=rngs)

    @trace_function(stage="TRANSFORMER", include_args=False, include_output=True)
    def __call__(
        self,
        input_ids: jax.Array,
        positions: jax.Array,
        forward_batch: ForwardBatch,
    ):
        residual = None
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, forward_batch, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLMJaxModel(nnx.Module):
    def __init__(self, config: PretrainedConfig, rngs: nnx.Rngs = None):
        self.config = config
        self.model = QWen3Model(config, rngs)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, rngs=rngs)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self._setup_debug_tracer()

    def _setup_debug_tracer(self):
        try:
            global_tracer.set_model(self)
        except Exception as e:
            print(f"Warning: Could not setup debug tracer: {str(e)}")

    def load_pytree_weights(self, pytree):
        flat_weights = flatten_pytree_with_paths(pytree)
        model_state = nnx.state(self)
        expected_paths = get_expected_param_paths(model_state)
        missing_paths = expected_paths - set(flat_weights.keys())
        if missing_paths:
            raise ValueError(f"Missing weights for parameters: {sorted(missing_paths)}")

        update_state_recursive(model_state, flat_weights)

        pspecs = nnx.get_partition_spec(model_state)
        pstate = jax.lax.with_sharding_constraint(model_state, pspecs)
        nnx.update(self, pstate)

    def __call__(
        self,
        input_ids: jax.Array,
        positions: jax.Array,
        forward_batch: ForwardBatch,
    ):
        hidden_states = self.model(input_ids, positions, forward_batch)
        result = self.logits_processor(hidden_states, self.lm_head, forward_batch)
        return result


EntryClass = Qwen3ForCausalLMJaxModel
