import functools
import logging
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jax.experimental import multihost_utils, shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec
from jax.sharding import PartitionSpec as P
from transformers import PretrainedConfig

from python.sgl_jax.srt.utils.jax_utils import device_array
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.debug_tracer import global_tracer, trace_function
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, RotaryEmbedding
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsProcessor
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

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
def _mlp_forward(
    hidden_states: jax.Array, w1: jax.Array, w2: jax.Array, c_proj: jax.Array
):
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
        if forward_batch.forward_mode.is_idle():
            return None, None, None
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
        mesh: jax.sharding.Mesh = None,
    ):
        self.layer_id = layer_id

        self.ln_1 = RMSNorm(
            config.hidden_size, epsilon=config.layer_norm_epsilon, rngs=rngs
        )

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
        self.mesh = mesh
        self.ln_2 = RMSNorm(
            config.hidden_size, epsilon=config.layer_norm_epsilon, rngs=rngs
        )

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
        logger.info(
            f"before attention hidden_states shape{hidden_states.shape} mode {forward_batch.forward_mode}"
        )
        attn_output, k, v = self.attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            layer_id=self.layer_id,
        )
        if attn_output is not None:
            hidden_states = residual + attn_output

        logger.info(
            f"after attention hidden_states shape{hidden_states.shape} {forward_batch.enable_dp_attention}"
        )

        if forward_batch.enable_dp_attention:
            hidden_states = functools.partial(
                shard_map.shard_map,
                mesh=self.mesh,
                in_specs=P(None),
                out_specs=P(None),
                check_rep=False,
            )(lambda x: jax.lax.all_gather(x, "data"))(hidden_states)
            hidden_states = jnp.cumsum(hidden_states, axis=0)
            logger.info(f"after attention dp hidden_states shape{hidden_states.shape}")

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
        self,
        config: PretrainedConfig,
        dtype: jnp.dtype = jnp.float16,
        rngs: nnx.Rngs = None,
        mesh: jax.sharding.Mesh = None,
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
                mesh=mesh,
            )
            for i in range(config.num_hidden_layers)
        ]

        self.ln_f = RMSNorm(
            config.hidden_size, epsilon=config.layer_norm_epsilon, rngs=rngs
        )

    @trace_function(stage="TRANSFORMER", include_args=False, include_output=True)
    def __call__(
        self,
        input_ids: jax.Array,
        positions: jax.Array,
        forward_batch: ForwardBatch,
    ):
        global_tracer.print(input_ids, "embedding_input", "embedding_all")
        logger.info(f"input_ids: {input_ids} {input_ids.dtype}")
        hidden_states = self.embed_tokens(input_ids)
        global_tracer.print(hidden_states, "embedding_output", "embedding_all")

        layers_k = []
        layers_v = []

        for layer in self.h:
            logger.info(f"positions: {positions} {positions.dtype} {layer.layer_id}")
            hidden_states, k, v = layer(positions, hidden_states, forward_batch)
            if k is not None and v is not None:
                layers_k.append(k)
                layers_v.append(v)
        logger.info(
            f"hidden_states: {hidden_states} {hidden_states.dtype} {hidden_states.shape}"
        )
        global_tracer.print(hidden_states, "RMSNorm_final_input", "rmsnorm_final")
        hidden_states = self.ln_f(hidden_states)
        logger.info(
            f"hidden_states: {hidden_states} {hidden_states.dtype} {hidden_states.shape}"
        )
        global_tracer.print(hidden_states, "RMSNorm_final_output", "rmsnorm_final")

        return hidden_states, layers_k, layers_v


class QWenLMHeadModel(nnx.Module):
    """QWen language head model"""

    def __init__(
        self, config: ModelConfig, rngs: nnx.Rngs = None, mesh: jax.sharding.Mesh = None
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = config.dtype
        logger.info(f"QWenLMHeadModel config dtype: {self.dtype}")
        self.transformer = QWenModel(
            config.hf_config, dtype=self.dtype, rngs=rngs, mesh=mesh
        )
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

        loader = WeightLoader(
            model=self, model_config=self.config, mesh=self.mesh, dtype=self.dtype
        )

        weight_mappings = self._create_qwen_weight_mappings()

        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("Qwen weights loaded successfully!")

    def _create_qwen_weight_mappings(self) -> dict:
        mappings = {
            "transformer.wte.weight": WeightMapping(
                target_path="transformer.embed_tokens.embedding",
                sharding=(None, None),
                transpose=False,
            ),
            "transformer.ln_f.weight": WeightMapping(
                target_path="transformer.ln_f.weight", sharding=(None,), transpose=False
            ),
        }

        if not getattr(self.config.hf_config, "tie_word_embeddings", True):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding", sharding=(None, None), transpose=False
            )

        num_layers = self.config.hf_config.num_hidden_layers
        for layer_idx in range(num_layers):
            layer_mappings = self._create_layer_mappings(layer_idx)
            mappings.update(layer_mappings)

        return mappings

    def _create_layer_mappings(self, layer_idx: int) -> dict:
        prefix = f"transformer.h.{layer_idx}"

        return {
            f"{prefix}.ln_1.weight": WeightMapping(
                target_path=f"{prefix}.ln_1.weight", sharding=(None,), transpose=False
            ),
            f"{prefix}.ln_2.weight": WeightMapping(
                target_path=f"{prefix}.ln_2.weight", sharding=(None,), transpose=False
            ),
            f"{prefix}.attn.c_attn.weight": WeightMapping(
                target_path=[
                    f"{prefix}.attn.q_proj.weight",
                    f"{prefix}.attn.k_proj.weight",
                    f"{prefix}.attn.v_proj.weight",
                ],
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=True,
            ),
            f"{prefix}.attn.c_attn.bias": WeightMapping(
                target_path=[
                    f"{prefix}.attn.q_proj.bias",
                    f"{prefix}.attn.k_proj.bias",
                    f"{prefix}.attn.v_proj.bias",
                ],
                sharding=(None,),
                transpose=False,
                head_dim_padding=True,
                kv_head_padding=True,
            ),  # bias 不转置
            f"{prefix}.attn.c_proj.weight": WeightMapping(
                target_path=f"{prefix}.attn.c_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
            f"{prefix}.mlp.w1.weight": WeightMapping(
                target_path=f"{prefix}.mlp.w1.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.w2.weight": WeightMapping(
                target_path=f"{prefix}.mlp.w2.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.c_proj.weight": WeightMapping(
                target_path=f"{prefix}.mlp.c_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
        }

    def __call__(
        self,
        input_ids: jax.Array,
        positions: jax.Array,
        forward_batch: ForwardBatch,
    ):
        logger.info(f"input_ids: {input_ids} {input_ids.dtype}")
        positions = positions.astype(jnp.int32)
        logger.info(f"positions: {positions} {positions.dtype}")
        logger.info(f"forward_batch: {forward_batch}")
        hidden_states, layers_k, layers_v = self.transformer(
            input_ids, positions, forward_batch
        )
        result = self.logits_processor(hidden_states, self.lm_head, forward_batch)

        if global_tracer.is_session_active():
            input_data = {"input_ids": input_ids, "input_shape": list(input_ids.shape)}

            output_data = {"output_type": str(type(result).__name__)}

            if (
                hasattr(result, "next_token_logits")
                and result.next_token_logits is not None
            ):
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
