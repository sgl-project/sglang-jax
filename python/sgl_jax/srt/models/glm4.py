"""Inference-only GLM-4 dense model (Glm4ForCausalLM, e.g. GLM-4-9B-0414 / GLM-Z1).

Differs from Llama in four places:
- sandwich norms: extra RMSNorm after attention and after the MLP,
- fused ``gate_up_proj`` checkpoint weight (split at load time),
- partial (``partial_rotary_factor``) GPT-J-style (interleaved) RoPE,
- q/k/v projections carry a bias while ``o_proj`` does not.
"""

import logging

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, get_rope
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.model_loader.weights import WeightLoader, WeightSpec
from sgl_jax.srt.models.llama import LlamaMLP

logger = logging.getLogger(__name__)


def _rope_parameters(config: PretrainedConfig) -> dict:
    rope = dict(getattr(config, "rope_parameters", None) or {})
    rope.setdefault("rope_theta", getattr(config, "rope_theta", 10000.0))
    rope.setdefault("partial_rotary_factor", getattr(config, "partial_rotary_factor", 0.5))
    rope.setdefault("rope_type", "default")
    return rope


class Glm4Attention(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.mesh = mesh
        self.hidden_size = config.hidden_size
        self.q_head_num = config.num_attention_heads
        self.kv_head_num = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or (
            self.hidden_size // self.q_head_num
        )
        attention_bias = getattr(config, "attention_bias", True)

        rope = _rope_parameters(config)
        rotary_dim = int(self.head_dim * rope["partial_rotary_factor"])
        rope_scaling = {k: v for k, v in rope.items() if k != "partial_rotary_factor"}

        self.q_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.q_head_num * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.k_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.kv_head_num * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.v_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.kv_head_num * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.o_proj = LinearBase(
            input_size=self.q_head_num * self.head_dim,
            output_size=self.hidden_size,
            use_bias=False,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
        )
        # GLM-4 rotates interleaved pairs (GPT-J style) over the first
        # partial_rotary_factor * head_dim channels.
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=rotary_dim,
            max_position=getattr(config, "max_position_embeddings", 32768),
            base=rope["rope_theta"],
            is_neox_style=False,
            rope_scaling=rope_scaling,
            dtype=dtype,
        )
        self.attn = RadixAttention(
            num_heads=self.q_head_num,
            head_dim=self.head_dim,
            scaling=self.head_dim**-0.5,
            num_kv_heads=self.kv_head_num,
            layer_id=layer_id,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ):
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        heads = NamedSharding(self.mesh, P("data", "tensor", None))
        q = q.reshape(-1, self.q_head_num, self.head_dim, out_sharding=heads)
        k = k.reshape(-1, self.kv_head_num, self.head_dim, out_sharding=heads)
        v = v.reshape(-1, self.kv_head_num, self.head_dim, out_sharding=heads)

        q, k = self.rotary_emb(positions, q, k)
        attn_output, kv_fused = self.attn(
            q, k, v, forward_batch=forward_batch, token_to_kv_pool=token_to_kv_pool
        )
        output, _ = self.o_proj(attn_output)
        return output, kv_fused


class Glm4DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.layer_id = layer_id
        self.self_attn = Glm4Attention(config, mesh=mesh, layer_id=layer_id, dtype=dtype)
        self.mlp = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            mesh=mesh,
            layer_id=layer_id,
            dtype=dtype,
        )

        def norm():
            return RMSNorm(
                config.hidden_size, epsilon=config.rms_norm_eps, param_dtype=dtype, dtype=dtype
            )

        self.input_layernorm = norm()
        self.post_self_attn_layernorm = norm()
        self.post_attention_layernorm = norm()
        self.post_mlp_layernorm = norm()

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, kv_fused = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, kv_fused


class Glm4Model(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.embed_tokens = Embed(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            kernel_axes=("tensor", None),
            param_dtype=dtype,
            mesh=mesh,
        )
        self.layers = nnx.data(
            [
                Glm4DecoderLayer(config, mesh=mesh, layer_id=i, dtype=dtype)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps, param_dtype=dtype)

    def __call__(self, forward_batch: ForwardBatch, token_to_kv_pool: KVCache):
        hidden_states = self.embed_tokens(forward_batch.input_ids)
        layers_kv_fused = []
        for layer in self.layers:
            hidden_states, kv_fused = layer(
                forward_batch.positions, hidden_states, forward_batch, token_to_kv_pool
            )
            layers_kv_fused.append(kv_fused)
        hidden_states = self.norm(hidden_states)
        return hidden_states, layers_kv_fused


class Glm4ForCausalLM(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        self.model = Glm4Model(config, mesh=mesh, dtype=dtype)
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        if not self.tie_word_embeddings:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                dtype=dtype,
                param_dtype=dtype,
                mesh=mesh,
                enable_dp_lm_head=getattr(config, "enable_dp_lm_head", False),
            )
        self.logits_processor = LogitsProcessor(
            config.vocab_size,
            mesh=mesh,
            enable_dp_lm_head=getattr(config, "enable_dp_lm_head", False),
        )

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self, model_config=model_config, mesh=self.mesh, dtype=self.dtype
        )
        loader.load(self._create_weight_mappings())
        logger.info("GLM-4 weights loaded successfully!")

    def _create_weight_mappings(self) -> dict:
        mappings = {
            "model.embed_tokens.weight": WeightSpec(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.norm.weight": WeightSpec(
                target_path="model.norm.scale", sharding=(None,), transpose=False
            ),
        }
        if not self.tie_word_embeddings:
            mappings["lm_head.weight"] = self.lm_head.weight_mapping("lm_head.embedding")
        for layer_idx in range(self.config.num_hidden_layers):
            mappings.update(self._create_layer_mappings(layer_idx))
        return mappings

    def _create_layer_mappings(self, layer_idx: int) -> dict:
        prefix = f"model.layers.{layer_idx}"
        mappings = {}
        for norm in (
            "input_layernorm",
            "post_self_attn_layernorm",
            "post_attention_layernorm",
            "post_mlp_layernorm",
        ):
            mappings[f"{prefix}.{norm}.weight"] = WeightSpec(
                target_path=f"{prefix}.{norm}.scale", sharding=(None,), transpose=False
            )

        for proj, is_kv in (("q_proj", False), ("k_proj", True), ("v_proj", True)):
            mappings[f"{prefix}.self_attn.{proj}.weight"] = WeightSpec(
                target_path=f"{prefix}.self_attn.{proj}.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=is_kv,
            )
            if getattr(self.config, "attention_bias", True):
                mappings[f"{prefix}.self_attn.{proj}.bias"] = WeightSpec(
                    target_path=f"{prefix}.self_attn.{proj}.bias",
                    sharding=(None,),
                    transpose=False,
                    head_dim_padding=True,
                    kv_head_padding=is_kv,
                )
        mappings[f"{prefix}.self_attn.o_proj.weight"] = WeightSpec(
            target_path=f"{prefix}.self_attn.o_proj.weight",
            sharding=("tensor", None),
            transpose=True,
            head_dim_padding=True,
            kv_head_padding=False,
        )

        intermediate = self.config.intermediate_size
        mappings[f"{prefix}.mlp.gate_up_proj.weight"] = WeightSpec(
            target_path=[f"{prefix}.mlp.gate_proj.weight", f"{prefix}.mlp.up_proj.weight"],
            sharding=(None, "tensor"),
            transpose=True,
            split_sizes=(intermediate, intermediate),
            split_axis=1,
        )
        mappings[f"{prefix}.mlp.down_proj.weight"] = WeightSpec(
            target_path=f"{prefix}.mlp.down_proj.weight",
            sharding=("tensor", None),
            transpose=True,
        )
        return mappings

    def get_embed_and_head(self):
        head = self.model.embed_tokens if self.tie_word_embeddings else self.lm_head
        return self.model.embed_tokens.embedding.value, head.embedding.value

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        kv_pool = memory_pools.token_to_kv_pool
        hidden_states, layers_kv_fused = self.model(forward_batch, kv_pool)
        head = self.model.embed_tokens if self.tie_word_embeddings else self.lm_head
        output = self.logits_processor(hidden_states, head, logits_metadata)
        return output, {"token_to_kv_pool": layers_kv_fused}, True, None


EntryClass = Glm4ForCausalLM
