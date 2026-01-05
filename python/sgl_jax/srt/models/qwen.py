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
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)

init_fn = nnx.initializers.uniform()


class QWenMLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.float16,
    ):
        self.layer_id = layer_id

        self.w1 = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
        )

        self.w2 = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
        )

        self.c_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            use_bias=False,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
        )

        self.act_func = jax.nn.silu

    def __call__(self, hidden_states: jax.Array):
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
        mesh: jax.sharding.Mesh,
        rope_theta: float = 10000,
        rope_scaling: dict[str, Any] | None = None,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.float16,
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
            params_dtype=dtype,
        )
        self.k_proj = LinearBase(
            input_size=hidden_size,
            output_size=hidden_size,
            use_bias=True,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
        )
        self.v_proj = LinearBase(
            input_size=hidden_size,
            output_size=hidden_size,
            use_bias=True,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
        )
        self.c_proj = LinearBase(
            input_size=num_heads * head_size,
            output_size=hidden_size,
            use_bias=False,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
        )

        self.rotary_emb = RotaryEmbedding(
            head_size=head_size,
            rotary_dim=head_size,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            dtype=dtype,
        )
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
        token_to_kv_pool: KVCache,
        layer_id: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        q = q.reshape(-1, self.num_heads, self.head_size)
        k = k.reshape(-1, self.num_heads, self.head_size)
        v = v.reshape(-1, self.num_heads, self.head_size)

        q, k = self.rotary_emb(positions, q, k)
        attn_output, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool)
        output, _ = self.c_proj(attn_output)
        return output, kv_fused


class QWenBlock(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.float16,
    ):
        self.layer_id = layer_id

        self.ln_1 = RMSNorm(
            config.hidden_size,
            epsilon=config.layer_norm_epsilon,
            param_dtype=dtype,
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
            mesh=mesh,
        )

        self.ln_2 = RMSNorm(
            config.hidden_size,
            epsilon=config.layer_norm_epsilon,
            param_dtype=dtype,
        )

        self.mlp = QWenMLP(
            config.hidden_size,
            config.intermediate_size // 2,
            layer_id=layer_id,
            dtype=dtype,
            mesh=mesh,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ):
        residual = hidden_states

        hidden_states = self.ln_1(hidden_states)
        attn_output, kv_fused = self.attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
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
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.float16,
    ):
        vocab_size = ((config.vocab_size + 63) // 64) * 64

        self.embed_tokens = Embed(
            num_embeddings=vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            kernel_axes=("tensor", None),
            param_dtype=dtype,
            mesh=mesh,
        )

        self.layers = nnx.data(
            [
                QWenBlock(
                    config,
                    layer_id=i,
                    dtype=dtype,
                    mesh=mesh,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.ln_f = RMSNorm(
            config.hidden_size,
            epsilon=config.layer_norm_epsilon,
            param_dtype=dtype,
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ):
        hidden_states = self.embed_tokens(forward_batch.input_ids)

        layers_kv_fused = []

        for layer in self.layers:
            hidden_states, kv_fused = layer(
                forward_batch.positions, hidden_states, forward_batch, token_to_kv_pool
            )
            layers_kv_fused.append(kv_fused)

        hidden_states = self.ln_f(hidden_states)

        return hidden_states, layers_kv_fused


class QWenLMHeadModel(nnx.Module):
    """QWen language head model"""

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        logger.info("QWenLMHeadModel config dtype: %s", self.dtype)
        self.model = QWenModel(config, dtype=self.dtype, mesh=mesh)
        vocab_size = ((config.vocab_size + 63) // 64) * 64
        if not getattr(self.config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                vocab_size,
                config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=("tensor", None),
            )
        self.logits_processor = LogitsProcessor(vocab_size, mesh=self.mesh)

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )

        weight_mappings = self._create_qwen_weight_mappings()

        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("Qwen weights loaded successfully!")

    def _create_qwen_weight_mappings(self) -> dict:
        mappings = {
            "transformer.wte.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "transformer.ln_f.weight": WeightMapping(
                target_path="model.ln_f.scale", sharding=(None,), transpose=False
            ),
        }

        if not getattr(self.config, "tie_word_embeddings", False):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding", sharding=("tensor", None), transpose=False
            )

        num_layers = self.config.num_hidden_layers
        for layer_idx in range(num_layers):
            layer_mappings = self._create_layer_mappings(layer_idx)
            mappings.update(layer_mappings)

        return mappings

    def _create_layer_mappings(self, layer_idx: int) -> dict:
        prefix = f"transformer.h.{layer_idx}"
        target_prefix = f"model.layers.{layer_idx}"

        return {
            f"{prefix}.ln_1.weight": WeightMapping(
                target_path=f"{target_prefix}.ln_1.scale", sharding=(None,), transpose=False
            ),
            f"{prefix}.ln_2.weight": WeightMapping(
                target_path=f"{target_prefix}.ln_2.scale", sharding=(None,), transpose=False
            ),
            f"{prefix}.attn.c_attn.weight": WeightMapping(
                target_path=[
                    f"{target_prefix}.attn.q_proj.weight",
                    f"{target_prefix}.attn.k_proj.weight",
                    f"{target_prefix}.attn.v_proj.weight",
                ],
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=True,
            ),
            f"{prefix}.attn.c_attn.bias": WeightMapping(
                target_path=[
                    f"{target_prefix}.attn.q_proj.bias",
                    f"{target_prefix}.attn.k_proj.bias",
                    f"{target_prefix}.attn.v_proj.bias",
                ],
                sharding=(None,),
                transpose=False,
                head_dim_padding=True,
                kv_head_padding=True,
            ),
            f"{prefix}.attn.c_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.attn.c_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
            f"{prefix}.mlp.w1.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.w1.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.w2.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.w2.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.c_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.c_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
        }

    def get_embed_and_head(self):
        return (
            self.transformer.embed_tokens.embedding.value,
            self.lm_head.embedding.value,
        )

    def set_embed_and_head(
        self,
        embed_weight: jax.Array | None = None,
        head_weight: jax.Array | None = None,
    ) -> None:
        """Set word embedding and LM Head weights.

        Args:
            embed_weight: Embedding matrix with shape [vocab_size, hidden_size].
            head_weight:  LM Head matrix with shape [vocab_size, hidden_size].
        """

        # Set embedding weight
        if embed_weight is not None:
            self.transformer.embed_tokens.embedding.value = embed_weight

        # Set LM Head weight
        if head_weight is not None:
            self.lm_head.embedding.value = head_weight

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, layers_kv_fused = self.model(forward_batch, token_to_kv_pool)
        if not getattr(self.config, "tie_word_embeddings", False):
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)
        return output, layers_kv_fused, True


EntryClass = QWenLMHeadModel
