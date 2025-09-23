import logging
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, RotaryEmbedding
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)

init_fn = nnx.initializers.uniform()


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

    def __call__(self, hidden_states: jnp.ndarray):
        a1, _ = self.w1(hidden_states)
        a2, _ = self.w2(hidden_states)
        intermediate_parallel = a1 * jax.nn.silu(a2)
        output, _ = self.c_proj(intermediate_parallel)
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
        self.head_size = head_size
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

        q = q.reshape(-1, self.num_heads, self.head_size)
        k = k.reshape(-1, self.num_heads, self.head_size)
        v = v.reshape(-1, self.num_heads, self.head_size)

        q, k = self.rotary_emb(positions, q, k)
        attn_output, kv_fused = self.attn(q, k, v, forward_batch=forward_batch)
        output, _ = self.c_proj(attn_output)
        return output, kv_fused


class QWenBlock(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.float16,
        rngs: nnx.Rngs = None,
    ):
        self.layer_id = layer_id

        self.ln_1 = nnx.RMSNorm(
            config.hidden_size,
            epsilon=config.layer_norm_epsilon,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rngs,
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

        self.ln_2 = nnx.RMSNorm(
            config.hidden_size,
            epsilon=config.layer_norm_epsilon,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rngs,
        )

        self.mlp = QWenMLP(
            config.hidden_size,
            config.intermediate_size // 2,
            layer_id=layer_id,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        residual = hidden_states

        hidden_states = self.ln_1(hidden_states)
        attn_output, kv_fused = self.attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            layer_id=self.layer_id,
        )
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, kv_fused


class QWenModel(nnx.Module):
    """QWen model"""

    def __init__(
        self,
        config: PretrainedConfig,
        dtype: jnp.dtype = jnp.float16,
        rngs: nnx.Rngs = None,
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

        self.ln_f = nnx.RMSNorm(
            config.hidden_size,
            epsilon=config.layer_norm_epsilon,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rngs,
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
    ):
        hidden_states = self.embed_tokens(forward_batch.input_ids)

        layers_kv_fused = []

        for layer in self.h:
            hidden_states, kv_fused = layer(
                forward_batch.positions, hidden_states, forward_batch
            )
            layers_kv_fused.append(kv_fused)

        hidden_states = self.ln_f(hidden_states)

        return hidden_states, layers_kv_fused


class QWenLMHeadModel(nnx.Module):
    """QWen language head model"""

    def __init__(
        self, config: ModelConfig, rngs: nnx.Rngs = None, mesh: jax.sharding.Mesh = None
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = config.dtype
        logger.info(f"QWenLMHeadModel config dtype: {self.dtype}")
        self.transformer = QWenModel(config.hf_config, dtype=self.dtype, rngs=rngs)
        vocab_size = ((config.hf_config.vocab_size + 63) // 64) * 64
        self.lm_head = ParallelLMHead(vocab_size, config.hidden_size, rngs=rngs)
        self.logits_processor = LogitsProcessor(vocab_size, self.lm_head, self.mesh)

    def load_weights(self, rng_key: jax.Array):
        self.rng = nnx.Rngs(rng_key)

        loader = WeightLoader(
            model=self,
            model_config=self.config,
            mesh=self.mesh,
            dtype=self.dtype,
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
                target_path="transformer.ln_f.scale", sharding=(None,), transpose=False
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
                target_path=f"{prefix}.ln_1.scale", sharding=(None,), transpose=False
            ),
            f"{prefix}.ln_2.weight": WeightMapping(
                target_path=f"{prefix}.ln_2.scale", sharding=(None,), transpose=False
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

    def __call__(
        self,
        forward_batch: ForwardBatch,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, layers_kv_fused = self.transformer(forward_batch)
        output = self.logits_processor(hidden_states, logits_metadata)
        return output, layers_kv_fused, True


EntryClass = QWenLMHeadModel
