import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PretrainedConfig

from sgl_jax.srt.layers.embeddings import Embed, RotaryEmbedding
from sgl_jax.srt.layers.layernorm import GemmaRMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

if TYPE_CHECKING:
    from sgl_jax.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


def _get_text_config(config: PretrainedConfig) -> PretrainedConfig:
    return getattr(config, "text_config", config)


def _get_layer_type(config: PretrainedConfig, layer_id: int) -> str:
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None and layer_id < len(layer_types):
        return layer_types[layer_id]
    sliding_window_pattern = getattr(
        config,
        "sliding_window_pattern",
        getattr(config, "_sliding_window_pattern", None),
    )
    if sliding_window_pattern:
        if (layer_id + 1) % sliding_window_pattern:
            return "sliding_attention"
    return "full_attention"


def get_attention_sliding_window_size(config: PretrainedConfig) -> int:
    sliding_window = getattr(config, "sliding_window", 0) or 0
    return max(sliding_window, 0)


def _get_rope_theta(config: PretrainedConfig, layer_type: str) -> float:
    rope_parameters = getattr(config, "rope_parameters", None) or {}
    if isinstance(rope_parameters, dict) and layer_type in rope_parameters:
        return rope_parameters[layer_type].get("rope_theta", 10000.0)
    if layer_type == "sliding_attention":
        return getattr(config, "rope_local_base_freq", 10000.0)
    if isinstance(rope_parameters, dict):
        return rope_parameters.get("rope_theta", getattr(config, "rope_theta", 10000.0))
    return getattr(config, "rope_theta", 10000.0)


def _get_forward_batch_input_embeds(forward_batch: ForwardBatch) -> jax.Array | None:
    forward_mode = getattr(forward_batch, "forward_mode", None)
    is_extend = getattr(forward_mode, "is_extend_or_draft_extend_or_mixed", None)
    if is_extend is None or not is_extend():
        return None
    return getattr(forward_batch, "input_embedding", None)


class Gemma3MLP(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        hidden_activation = getattr(config, "hidden_activation", "gelu_pytorch_tanh")
        if hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(
                "Gemma3MLP expects `hidden_activation='gelu_pytorch_tanh'`, "
                f"got {hidden_activation!r}."
            )

        self.gate_proj = LinearBase(
            input_size=config.hidden_size,
            output_size=config.intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="gate_proj",
        )
        self.up_proj = LinearBase(
            input_size=config.hidden_size,
            output_size=config.intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="up_proj",
        )
        self.down_proj = LinearBase(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            use_bias=False,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="down_proj",
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        gate, _ = self.gate_proj(hidden_states)
        up, _ = self.up_proj(hidden_states)
        hidden_states = up * jax.nn.gelu(gate, approximate=True)
        output, _ = self.down_proj(hidden_states)
        return output


class Gemma3Attention(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.num_heads
        )
        self.scaling = getattr(config, "query_pre_attn_scalar", self.head_dim) ** -0.5
        self.layer_type = _get_layer_type(config, layer_id)
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window_size = (
            get_attention_sliding_window_size(config) if self.is_sliding else 0
        )

        self.q_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.num_heads * self.head_dim,
            use_bias=getattr(config, "attention_bias", False),
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="q_proj",
        )
        self.k_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.num_kv_heads * self.head_dim,
            use_bias=getattr(config, "attention_bias", False),
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="k_proj",
        )
        self.v_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.num_kv_heads * self.head_dim,
            use_bias=getattr(config, "attention_bias", False),
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="v_proj",
        )
        self.o_proj = LinearBase(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            use_bias=getattr(config, "attention_bias", False),
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="o_proj",
        )

        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        rotary_dim = int(partial_rotary_factor * self.head_dim)
        self.rotary_emb = RotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=rotary_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=_get_rope_theta(config, self.layer_type),
            is_neox_style=getattr(config, "rope_is_neox_style", True),
            dtype=dtype,
        )
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=self.sliding_window_size,
            logit_cap=0,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, epsilon=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, epsilon=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, jax.Array]:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        q = q.reshape(-1, self.num_heads, self.head_dim)
        k = k.reshape(-1, self.num_kv_heads, self.head_dim)
        v = v.reshape(-1, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(forward_batch.positions, q, k)

        attn_output, kv_fused = self.attn(
            q,
            k,
            v,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )
        output, _ = self.o_proj(attn_output)
        return output, kv_fused


class Gemma3DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.self_attn = Gemma3Attention(config, layer_id, mesh=mesh, dtype=dtype)
        self.mlp = Gemma3MLP(config, layer_id=layer_id, dtype=dtype, mesh=mesh)
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
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
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


class Gemma3Model(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.hidden_size = config.hidden_size
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
                Gemma3DecoderLayer(
                    config,
                    layer_id=i,
                    dtype=dtype,
                    mesh=mesh,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        input_embeds: jax.Array | None = None,
    ) -> tuple[jax.Array, list[jax.Array]]:
        if input_embeds is None:
            input_embeds = _get_forward_batch_input_embeds(forward_batch)
        if input_embeds is None:
            hidden_states = self.embed_tokens(forward_batch.input_ids)
            hidden_states *= jnp.asarray(self.hidden_size**0.5, dtype=hidden_states.dtype)
        else:
            # Match HF inputs_embeds semantics: callers provide final LM-space embeddings.
            hidden_states = input_embeds

        residual = None
        layers_kv_fused = []
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual, kv_fused = layer(
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
            )
            layers_kv_fused.append(kv_fused)

        if residual is not None:
            hidden_states += residual
        hidden_states = self.norm(hidden_states)
        return hidden_states, layers_kv_fused


class Gemma3ForCausalLM(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.hf_config = config
        self.config = _get_text_config(config)
        self.dtype = dtype
        self.model = Gemma3Model(self.config, dtype=self.dtype, mesh=mesh)
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size,
            soft_cap=getattr(self.config, "final_logit_softcapping", None),
            mesh=self.mesh,
        )

    def get_attention_sliding_window_size(self) -> int:
        return get_attention_sliding_window_size(self.config)

    def load_weights(self, model_config: "ModelConfig"):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        loader.load_weights_from_safetensors(self._create_weight_mappings())
        logger.info("Gemma3 weights loaded successfully!")

    def _create_weight_mappings(self) -> dict[str, WeightMapping]:
        hf_config = getattr(self, "hf_config", SimpleNamespace())
        architectures = getattr(hf_config, "architectures", []) or []
        if "Gemma3ForConditionalGeneration" in architectures:
            # Text weights in conditional checkpoints are nested under language_model.
            # The conditional architecture itself is registered when vision support lands.
            return self._create_text_weight_mappings(hf_prefix="language_model.")
        return self._create_text_weight_mappings()

    def _create_text_weight_mappings(
        self,
        hf_prefix: str = "",
    ) -> dict[str, WeightMapping]:
        def hf_key(name: str) -> str:
            return f"{hf_prefix}{name}"

        mappings = {
            hf_key("model.embed_tokens.weight"): WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            hf_key("model.norm.weight"): WeightMapping(
                target_path="model.norm.weight",
                sharding=(None,),
                transpose=False,
            ),
        }

        for layer_idx in range(self.config.num_hidden_layers):
            mappings.update(
                self._create_layer_mappings(layer_idx, hf_prefix=hf_prefix)
            )

        return mappings

    def _create_layer_mappings(
        self,
        layer_idx: int,
        hf_prefix: str = "",
    ) -> dict[str, WeightMapping]:
        prefix = f"model.layers.{layer_idx}"
        hf_layer_prefix = f"{hf_prefix}{prefix}"

        return {
            f"{hf_layer_prefix}.input_layernorm.weight": WeightMapping(
                target_path=f"{prefix}.input_layernorm.weight",
                sharding=(None,),
                transpose=False,
            ),
            f"{hf_layer_prefix}.post_attention_layernorm.weight": WeightMapping(
                target_path=f"{prefix}.post_attention_layernorm.weight",
                sharding=(None,),
                transpose=False,
            ),
            f"{hf_layer_prefix}.post_feedforward_layernorm.weight": WeightMapping(
                target_path=f"{prefix}.post_feedforward_layernorm.weight",
                sharding=(None,),
                transpose=False,
            ),
            f"{hf_layer_prefix}.pre_feedforward_layernorm.weight": WeightMapping(
                target_path=f"{prefix}.pre_feedforward_layernorm.weight",
                sharding=(None,),
                transpose=False,
            ),
            f"{hf_layer_prefix}.self_attn.q_proj.weight": WeightMapping(
                target_path=f"{prefix}.self_attn.q_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=False,
            ),
            f"{hf_layer_prefix}.self_attn.k_proj.weight": WeightMapping(
                target_path=f"{prefix}.self_attn.k_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=True,
            ),
            f"{hf_layer_prefix}.self_attn.v_proj.weight": WeightMapping(
                target_path=f"{prefix}.self_attn.v_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=True,
            ),
            f"{hf_layer_prefix}.self_attn.o_proj.weight": WeightMapping(
                target_path=f"{prefix}.self_attn.o_proj.weight",
                sharding=("tensor", None),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=False,
            ),
            f"{hf_layer_prefix}.self_attn.q_norm.weight": WeightMapping(
                target_path=f"{prefix}.self_attn.q_norm.weight",
                sharding=(None,),
                transpose=False,
            ),
            f"{hf_layer_prefix}.self_attn.k_norm.weight": WeightMapping(
                target_path=f"{prefix}.self_attn.k_norm.weight",
                sharding=(None,),
                transpose=False,
            ),
            f"{hf_layer_prefix}.mlp.gate_proj.weight": WeightMapping(
                target_path=f"{prefix}.mlp.gate_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{hf_layer_prefix}.mlp.up_proj.weight": WeightMapping(
                target_path=f"{prefix}.mlp.up_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{hf_layer_prefix}.mlp.down_proj.weight": WeightMapping(
                target_path=f"{prefix}.mlp.down_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
        }

    def get_embed_and_head(self):
        embedding = self.model.embed_tokens.embedding.value
        return embedding, embedding

    def set_embed_and_head(
        self,
        embed_weight: jax.Array | None = None,
        head_weight: jax.Array | None = None,
    ) -> None:
        tied_weight = embed_weight if embed_weight is not None else head_weight
        if tied_weight is not None:
            self.model.embed_tokens.embedding.value = tied_weight

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        kv_pool = memory_pools.token_to_kv_pool
        hidden_states, layers_kv_fused = self.model(forward_batch, kv_pool)
        output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)
        return output, {"token_to_kv_pool": layers_kv_fused}, True, None


EntryClass = Gemma3ForCausalLM
