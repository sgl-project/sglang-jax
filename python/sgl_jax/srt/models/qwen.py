import logging
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.debug_tracer import global_tracer, trace_function
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, RotaryEmbedding
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsProcessor
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import load_hf_weights

logger = logging.getLogger(__name__)


class QWenMLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int = 0,
        rngs: nnx.Rngs = None,
        dtype: jnp.dtype = jnp.float16,
    ):
        self.layer_id = layer_id

        self.w1 = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            rngs=rngs,
        )

        self.w2 = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            rngs=rngs,
        )

        self.c_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            use_bias=False,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            rngs=rngs,
        )

        self.act_func = jax.nn.silu

    @trace_function(stage="MLP", include_args=False, include_output=True)
    def __call__(self, hidden_states: jnp.ndarray):
        return _mlp_forward(
            hidden_states,
            self.w1.weight.value,
            self.w2.weight.value,
            self.c_proj.weight.value,
        )


# @jax.jit
def _mlp_forward(hidden_states: jax.Array, w1: jax.Array, w2: jax.Array, c_proj: jax.Array):
    a1 = jnp.dot(hidden_states, w1)
    a2 = jnp.dot(hidden_states, w2)
    intermediate_parallel = a1 * jax.nn.silu(a2)
    intermediate_parallel = jax.lax.with_sharding_constraint(
        intermediate_parallel, PartitionSpec(None, "tensor")
    )
    output = jnp.dot(intermediate_parallel, c_proj)
    return output


class QWenAttention(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_position_embeddings: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.float16,
        rngs: nnx.Rngs = None,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        head_size = hidden_size // num_heads
        self.scaling = head_size**-0.5

        self.q_proj = LinearBase(
            input_size=hidden_size,
            output_size=hidden_size,
            use_bias=True,
            kernel_axes=(None, "tensor"),
            rngs=rngs,
            params_dtype=dtype,
        )
        self.k_proj = LinearBase(
            input_size=hidden_size,
            output_size=hidden_size,
            use_bias=True,
            kernel_axes=(None, "tensor"),
            rngs=rngs,
            params_dtype=dtype,
        )
        self.v_proj = LinearBase(
            input_size=hidden_size,
            output_size=hidden_size,
            use_bias=True,
            kernel_axes=(None, "tensor"),
            rngs=rngs,
            params_dtype=dtype,
        )
        self.c_proj = LinearBase(
            input_size=num_heads * head_size,
            output_size=hidden_size,
            use_bias=False,
            kernel_axes=("tensor", None),
            rngs=rngs,
            params_dtype=dtype,
        )

        # Use torch version of RotaryEmbedding directly
        self.rotary_emb = RotaryEmbedding(
            head_size=head_size,
            rotary_dim=head_size,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            dtype=dtype,
        )
        self.scaling = head_size**-0.5
        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=head_size,
            scaling=self.scaling,
            num_kv_heads=num_heads,
            layer_id=layer_id,
        )

    @trace_function(stage="ATTENTION", include_args=False, include_output=True)
    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        layer_id: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        q, k = self.rotary_emb(positions, q, k)
        attn_output, k, v = self.attn(q, k, v, forward_batch=forward_batch)
        output, _ = self.c_proj(attn_output)
        return output, k, v


class QWenBlock(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.float16,
        rngs: nnx.Rngs = None,
    ):
        self.layer_id = layer_id

        self.ln_1 = RMSNorm(config.hidden_size, epsilon=config.layer_norm_epsilon, rngs=rngs)

        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        self.attn = QWenAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            layer_id=layer_id,
            dtype=dtype,
            rngs=rngs,
        )

        self.ln_2 = RMSNorm(config.hidden_size, epsilon=config.layer_norm_epsilon, rngs=rngs)

        self.mlp = QWenMLP(
            config.hidden_size,
            config.intermediate_size // 2,
            layer_id=layer_id,
            dtype=dtype,
            rngs=rngs,
        )

    @trace_function(stage="BLOCK", include_args=False, include_output=True)
    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        residual = hidden_states

        global_tracer.print(
            hidden_states,
            f"RMSNorm_pre_attn_input",
            f"rmsnorm_layer_id_{self.layer_id}",
        )
        hidden_states = self.ln_1(hidden_states)
        global_tracer.print(
            hidden_states,
            f"RMSNorm_pre_attn_output",
            f"rmsnorm_layer_id_{self.layer_id}",
        )

        attn_output, k, v = self.attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            layer_id=self.layer_id,
        )

        hidden_states = residual + attn_output

        residual = hidden_states

        global_tracer.print(
            hidden_states, f"RMSNorm_pre_mlp_input", f"rmsnorm_layer_id_{self.layer_id}"
        )
        hidden_states = self.ln_2(hidden_states)
        global_tracer.print(
            hidden_states,
            f"RMSNorm_pre_mlp_output",
            f"rmsnorm_layer_id_{self.layer_id}",
        )

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, k, v


class QWenModel(nnx.Module):
    """QWen model"""

    def __init__(
        self, config: PretrainedConfig, dtype: jnp.dtype = jnp.float16, rngs: nnx.Rngs = None
    ):
        vocab_size = ((config.vocab_size + 63) // 64) * 64

        self.embed_tokens = Embed(
            num_embeddings=vocab_size,
            features=config.hidden_size,
            rngs=rngs,
            dtype=dtype,
            param_dtype=dtype,
        )

        self.h = [
            QWenBlock(
                config,
                layer_id=i,
                dtype=dtype,
                rngs=rngs,
            )
            for i in range(config.num_hidden_layers)
        ]

        self.ln_f = RMSNorm(config.hidden_size, epsilon=config.layer_norm_epsilon, rngs=rngs)

    @trace_function(stage="TRANSFORMER", include_args=False, include_output=True)
    def __call__(
        self,
        input_ids: jax.Array,
        positions: jax.Array,
        forward_batch: ForwardBatch,
    ):
        global_tracer.print(input_ids, "embedding_input", "embedding_all")
        hidden_states = self.embed_tokens(input_ids)
        global_tracer.print(hidden_states, "embedding_output", "embedding_all")

        layers_k = []
        layers_v = []

        for layer in self.h:
            hidden_states, k, v = layer(positions, hidden_states, forward_batch)
            layers_k.append(k)
            layers_v.append(v)

        global_tracer.print(hidden_states, "RMSNorm_final_input", "rmsnorm_final")
        hidden_states = self.ln_f(hidden_states)
        global_tracer.print(hidden_states, "RMSNorm_final_output", "rmsnorm_final")

        return hidden_states, layers_k, layers_v


class QWenLMHeadModel(nnx.Module):
    """QWen language head model"""

    def __init__(self, config: ModelConfig, rngs: nnx.Rngs = None, mesh: jax.sharding.Mesh = None):
        self.mesh = mesh
        self.config = config
        self.dtype = config.dtype
        logger.info(f"QWenLMHeadModel config dtype: {self.dtype}")
        self.transformer = QWenModel(config.hf_config, dtype=self.dtype, rngs=rngs)
        vocab_size = ((config.hf_config.vocab_size + 63) // 64) * 64
        self.lm_head = ParallelLMHead(vocab_size, config.hidden_size, rngs=rngs)
        self.logits_processor = LogitsProcessor(vocab_size)
        self._setup_debug_tracer()

    def _setup_debug_tracer(self):
        try:
            global_tracer.set_model(self)
        except Exception as e:
            print(f"Warning: Could not setup debug tracer: {str(e)}")

    def load_weights(self, rng_key: jax.Array):
        self.rng = nnx.Rngs(rng_key)

        mappings = {
            "transformer.wte": ("transformer.embed_tokens.embedding", (None, None)),
            "transformer.ln_f": ("transformer.ln_f.weight", (None,)),
        }

        if not self.config.hf_config.tie_word_embeddings:
            mappings.update(
                {
                    "lm_head": ("lm_head.embedding", (None, None)),
                }
            )

        num_layers = self.config.hf_config.num_hidden_layers
        for layer_idx in range(num_layers):
            mappings.update(
                {
                    f"transformer.h.{layer_idx}.ln_1": (
                        f"transformer.h.{layer_idx}.ln_1.weight",
                        (None,),
                    ),
                    f"transformer.h.{layer_idx}.ln_2": (
                        f"transformer.h.{layer_idx}.ln_2.weight",
                        (None,),
                    ),
                }
            )

            mappings.update(
                {
                    f"transformer.h.{layer_idx}.attn.c_attn": (
                        [
                            f"transformer.h.{layer_idx}.attn.q_proj.weight",
                            f"transformer.h.{layer_idx}.attn.k_proj.weight",
                            f"transformer.h.{layer_idx}.attn.v_proj.weight",
                        ],
                        (None, "tensor"),
                    ),
                    f"transformer.h.{layer_idx}.attn.c_attn.bias": (
                        [
                            f"transformer.h.{layer_idx}.attn.q_proj.bias",
                            f"transformer.h.{layer_idx}.attn.k_proj.bias",
                            f"transformer.h.{layer_idx}.attn.v_proj.bias",
                        ],
                        (None,),
                    ),
                    f"transformer.h.{layer_idx}.attn.c_proj": (
                        f"transformer.h.{layer_idx}.attn.c_proj.weight",
                        ("tensor", None),
                    ),
                }
            )

            mappings.update(
                {
                    f"transformer.h.{layer_idx}.mlp.w1": (
                        f"transformer.h.{layer_idx}.mlp.w1.weight",
                        (None, "tensor"),
                    ),
                    f"transformer.h.{layer_idx}.mlp.w2": (
                        f"transformer.h.{layer_idx}.mlp.w2.weight",
                        (None, "tensor"),
                    ),
                    f"transformer.h.{layer_idx}.mlp.c_proj": (
                        f"transformer.h.{layer_idx}.mlp.c_proj.weight",
                        ("tensor", None),
                    ),
                }
            )
        load_hf_weights(
            model_config=self.config,
            model=self,
            mappings=mappings,
            mesh=self.mesh,
            dtype=self.dtype,
        )

    def __call__(
        self,
        input_ids: jax.Array,
        positions: jax.Array,
        forward_batch: ForwardBatch,
    ):
        hidden_states, layers_k, layers_v = self.transformer(input_ids, positions, forward_batch)
        result = self.logits_processor(hidden_states, self.lm_head, forward_batch)

        if global_tracer.is_session_active():
            input_data = {"input_ids": input_ids, "input_shape": list(input_ids.shape)}

            output_data = {"output_type": str(type(result).__name__)}

            if hasattr(result, "next_token_logits") and result.next_token_logits is not None:
                output_data.update(
                    {
                        "logits": result.next_token_logits,
                        "logits_shape": list(result.next_token_logits.shape),
                    }
                )

            global_tracer.accumulate_step(input_data, output_data)

            if global_tracer.should_auto_save():
                global_tracer.end_session()
        return result, layers_k, layers_v


EntryClass = QWenLMHeadModel
