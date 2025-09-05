import functools
import logging
import time
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jax.experimental import shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
from jax.sharding import PartitionSpec as P
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.debug_tracer import global_tracer, trace_function
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, RotaryEmbedding
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)

mesh = None


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
        logger.info("qwen attention input 1 ")
        q, _ = self.q_proj(hidden_states)
        logger.info("qwen attention input 2 ")
        # k, _ = self.k_proj(hidden_states)
        # logger.info("qwen attention input 3 ")
        # v, _ = self.v_proj(hidden_states)
        # logger.info("qwen attention input 4 ")
        # q, k = self.rotary_emb(positions, q, k)
        # logger.info("qwen attention input 5 ")
        # attn_output, k, v = self.attn(q, k, v, forward_batch=forward_batch)
        # logger.info("qwen attention input 6 ")
        # output, _ = self.c_proj(attn_output)
        # logger.info("qwen attention input 7 ")
        return None, None, None


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
        self.mesh = mesh
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
        logger.info(f"before attn hiddenstate shape {hidden_states.shape}")
        global_tracer.print(
            hidden_states,
            f"RMSNorm_pre_attn_input",
            f"rmsnorm_layer_id_{self.layer_id}",
        )
        hidden_states = self.ln_1(hidden_states)
        logger.info(f"after ln1 hiddenstate shape {hidden_states.shape}")
        global_tracer.print(
            hidden_states,
            f"RMSNorm_pre_attn_output",
            f"rmsnorm_layer_id_{self.layer_id}",
        )
        hidden_states = jnp.arange(32 * 4096).reshape(32, 4096)
        hidden_states = jax.device_put(
            hidden_states, NamedSharding(self.mesh, P("data", None))
        )
        attn_output, k, v = self.attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            layer_id=self.layer_id,
        )
        logger.info(f"after attn hiddenstate shape {hidden_states.shape}")
        hidden_states = hidden_states

        logger.info(f"after residual hiddenstate shape {hidden_states.shape}")
        residual = hidden_states
        # Remove shard_map all_gather to avoid memory overflow (87.62G > 31.25G)
        # Each all_gather creates 1.28G * 32 layers = 40.96G temporary memory
        logger.info(f"after allgather hiddenstate shape {hidden_states.shape}")
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
        logger.info(f"after mlp hiddenstate shape {hidden_states.shape}")
        # hidden_states = hidden_states[0, :, :]
        logger.info(f"after mlp hiddenstate shape {hidden_states.shape}")
        hidden_states = residual + hidden_states
        # process scatter
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
        logger.info("before embedding")
        global_tracer.print(input_ids, "embedding_input", "embedding_all")
        hidden_states = self.embed_tokens(input_ids)
        logger.info(f"*********************hidden_states shape: {hidden_states.shape}")
        global_tracer.print(hidden_states, "embedding_output", "embedding_all")
        logger.info("after embedding")
        layers_k = []
        layers_v = []

        for layer in self.h:
            logger.info(f"start layer {layer.layer_id}")
            hidden_states, k, v = layer(positions, hidden_states, forward_batch)
            logger.info(f"after layer {layer.layer_id}")
            layers_k.append(k)
            layers_v.append(v)
        logger.info("before ln_f")
        global_tracer.print(hidden_states, "RMSNorm_final_input", "rmsnorm_final")
        hidden_states = self.ln_f(hidden_states)
        global_tracer.print(hidden_states, "RMSNorm_final_output", "rmsnorm_final")
        logger.info("after ln_f")
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
            config.hf_config, dtype=self.dtype, rngs=rngs, mesh=self.mesh
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
            ),
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

    def compute_logits(
        self,
        hidden_states: jax.Array,
        logits_metadata: LogitsMetadata,
    ):
        import time

        time.sleep(100)

        logger.info("start compute_logits")

        ret = self.logits_processor(
            hidden_states, self.lm_head, logits_metadata, self.mesh
        )
        logger.info(f"return compute_logits {ret}")
        return ret

    def __call__(
        self,
        input_ids: jax.Array,
        positions: jax.Array,
        forward_batch: ForwardBatch,
    ):
        hidden_states, layers_k, layers_v = self.transformer(
            input_ids, positions, forward_batch
        )
        logger.info("after transformer")
        return hidden_states, layers_k, layers_v


EntryClass = QWenLMHeadModel
