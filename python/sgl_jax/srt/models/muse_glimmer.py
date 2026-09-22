"""Text-only Muse Glimmer implementation for SGLang-JAX."""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, get_rope
from sgl_jax.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.precision_tracer import precision_tracer
from sgl_jax.srt.utils.profiling_utils import named_scope
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


def _query_prescale(config: PretrainedConfig) -> float:
    explicit = getattr(config, "scale_query_by", None)
    if explicit is not None:
        return float(explicit)
    factor = float(getattr(config, "qk_scale_factor", 1.0))
    sqrt_head_dim = float(config.head_dim) ** 0.5
    return factor / sqrt_head_dim if factor >= sqrt_head_dim else factor


class MuseGlimmerMLP(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int,
        dtype: jnp.dtype,
    ) -> None:
        if getattr(config, "hidden_activation", "silu") != "silu":
            raise ValueError("Muse Glimmer requires SiLU activation")
        self.layer_id = layer_id
        self.gate_proj = LinearBase(
            config.hidden_size,
            config.intermediate_size,
            mesh=mesh,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            scope_name="gate_proj",
        )
        self.up_proj = LinearBase(
            config.hidden_size,
            config.intermediate_size,
            mesh=mesh,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            scope_name="up_proj",
        )
        self.down_proj = LinearBase(
            config.intermediate_size,
            config.hidden_size,
            mesh=mesh,
            use_bias=False,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            scope_name="down_proj",
        )

    @named_scope
    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        gate, _ = self.gate_proj(hidden_states)
        up, _ = self.up_proj(hidden_states)
        output, _ = self.down_proj(jax.nn.silu(gate) * up)
        return output


class MuseGlimmerAttention(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int,
        dtype: jnp.dtype,
    ) -> None:
        self.layer_id = layer_id
        self.hidden_size = int(config.hidden_size)
        self.q_head_num = int(config.num_attention_heads)
        self.kv_head_num = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim)
        self.mesh = mesh
        layer_types = getattr(config, "layer_types", [])
        layer_type = layer_types[layer_id] if layer_id < len(layer_types) else "sliding_attention"
        self.use_rope = layer_type == "sliding_attention"
        self.sliding_window = int(getattr(config, "sliding_window", 2048)) if self.use_rope else 0
        self.query_prescale = _query_prescale(config)

        self.q_proj = LinearBase(
            self.hidden_size,
            self.q_head_num * self.head_dim,
            mesh=mesh,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            scope_name="q_proj",
        )
        self.k_proj = LinearBase(
            self.hidden_size,
            self.kv_head_num * self.head_dim,
            mesh=mesh,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            scope_name="k_proj",
        )
        self.v_proj = LinearBase(
            self.hidden_size,
            self.kv_head_num * self.head_dim,
            mesh=mesh,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            scope_name="v_proj",
        )
        self.gate_proj = LinearBase(
            self.hidden_size,
            self.q_head_num * self.head_dim,
            mesh=mesh,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            scope_name="gate_proj",
        )
        self.o_proj = LinearBase(
            self.q_head_num * self.head_dim,
            self.hidden_size,
            mesh=mesh,
            use_bias=False,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            scope_name="o_proj",
        )
        self.qk_norm = RMSNorm(
            self.head_dim,
            epsilon=float(config.rms_norm_eps),
            use_scale=False,
            scope_name="qk_norm",
        )

        rope_parameters = dict(getattr(config, "rope_parameters", {}) or {})
        rope_theta = float(rope_parameters.get("rope_theta", 500000.0))
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=int(getattr(config, "max_position_embeddings", 131072)),
            base=rope_theta,
            is_neox_style=True,
            rope_scaling=rope_parameters,
            dtype=dtype,
        )
        self.attn = RadixAttention(
            num_heads=self.q_head_num,
            head_dim=self.head_dim,
            scaling=self.head_dim**-0.5,
            num_kv_heads=self.kv_head_num,
            layer_id=layer_id,
            sliding_window_size=self.sliding_window,
        )

    @named_scope
    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, jax.Array]:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        q = q.reshape(
            -1,
            self.q_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )
        k = k.reshape(
            -1,
            self.kv_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )
        v = v.reshape(
            -1,
            self.kv_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )
        q = self.qk_norm(q) * self.query_prescale
        k = self.qk_norm(k)
        if self.use_rope:
            q, k = self.rotary_emb(positions, q, k)
        attn_output, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool)
        gate, _ = self.gate_proj(hidden_states)
        attn_output = attn_output * jax.nn.sigmoid(gate.reshape(attn_output.shape))
        output, _ = self.o_proj(attn_output)
        return output, kv_fused


class MuseGlimmerDecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int,
        dtype: jnp.dtype,
    ) -> None:
        hidden_size = int(config.hidden_size)
        self.layer_id = layer_id
        self.input_layernorm = GemmaRMSNorm(
            hidden_size, epsilon=float(config.rms_norm_eps), add_unit_offset=True
        )
        self.self_attn = MuseGlimmerAttention(config, mesh, layer_id, dtype)
        self.post_attention_layernorm = GemmaRMSNorm(
            hidden_size,
            epsilon=float(getattr(config, "post_norm_eps", 1e-8)),
            add_unit_offset=True,
        )
        self.pre_feedforward_layernorm = GemmaRMSNorm(
            hidden_size, epsilon=float(config.rms_norm_eps), add_unit_offset=True
        )
        self.mlp = MuseGlimmerMLP(config, mesh, layer_id, dtype)
        self.post_feedforward_layernorm = GemmaRMSNorm(
            hidden_size,
            epsilon=float(getattr(config, "post_norm_eps", 1e-8)),
            add_unit_offset=True,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, jax.Array, list[jax.Array]]:
        callbacks = []
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        callbacks.append(
            precision_tracer.jit_pure_callback_record(
                hidden_states, "input_layernorm_output", "INPUT_LAYERNORM", self.layer_id
            )
        )
        hidden_states, kv_fused = self.self_attn(
            positions, hidden_states, forward_batch, token_to_kv_pool
        )
        callbacks.append(
            precision_tracer.jit_pure_callback_record(
                hidden_states, "self_attn_output", "SELF_ATTN", self.layer_id
            )
        )
        hidden_states = residual + self.post_attention_layernorm(hidden_states)
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.post_feedforward_layernorm(hidden_states)
        callbacks.append(
            precision_tracer.jit_pure_callback_record(
                hidden_states, "mlp_output", "MLP", self.layer_id
            )
        )
        return hidden_states, kv_fused, callbacks


class MuseGlimmerModel(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype,
    ) -> None:
        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            kernel_axes=("tensor", None),
            param_dtype=dtype,
            mesh=mesh,
        )
        self.embed_norm = RMSNorm(
            config.hidden_size,
            epsilon=float(config.rms_norm_eps),
            use_scale=False,
            scope_name="embed_norm",
        )
        self.layers = nnx.data(
            [
                MuseGlimmerDecoderLayer(config, mesh, layer_id, dtype)
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = GemmaRMSNorm(
            config.hidden_size,
            epsilon=float(config.rms_norm_eps),
            add_unit_offset=False,
        )
        self.layers_to_capture: list[int] = []

    def __call__(self, forward_batch: ForwardBatch, token_to_kv_pool: KVCache):
        if (
            forward_batch.input_embedding is not None
            and forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
        ):
            hidden_states = forward_batch.input_embedding
        else:
            hidden_states = self.embed_norm(self.embed_tokens(forward_batch.input_ids))

        kv_fused = []
        callbacks = []
        aux_hidden_states = []
        for layer_id, layer in enumerate(self.layers):
            hidden_states, layer_kv, layer_callbacks = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
            )
            if layer_id + 1 in self.layers_to_capture:
                aux_hidden_states.append(hidden_states)
            kv_fused.append(layer_kv)
            callbacks.extend(layer_callbacks)
        hidden_states = self.norm(hidden_states)
        callbacks.append(
            precision_tracer.jit_pure_callback_record(
                hidden_states, "transformer_output", "TRANSFORMER"
            )
        )
        return hidden_states, aux_hidden_states, kv_fused, callbacks


class MuseGlimmerForConditionalGeneration(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.mesh = mesh
        self.config = getattr(config, "text_config", config)
        self.dtype = dtype
        self.model = MuseGlimmerModel(self.config, mesh, dtype)
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
        )
        self.output_multiplier = float(getattr(self.config, "output_multiplier", 1.0))
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size,
            soft_cap=float(getattr(self.config, "final_logit_softcapping", 0.0) or 0.0),
            mesh=mesh,
        )
        self.capture_aux_hidden_states = False

    def load_weights(self, model_config: ModelConfig) -> None:
        loader = WeightLoader(self, model_config, self.mesh, self.dtype)
        mappings = self._create_weight_mappings()
        if not loader.dummy_mode:
            weight_info = loader._scan_weight_info()
            mappings = {key: value for key, value in mappings.items() if key in weight_info}
        loader.load_weights_from_safetensors(mappings)
        logger.info("Muse Glimmer weights loaded successfully")

    def _create_weight_mappings(self) -> dict[str, WeightMapping]:
        prefix = "model.language_model"
        mappings: dict[str, WeightMapping] = {
            f"{prefix}.embed_tokens.weight": WeightMapping(
                "model.embed_tokens.embedding", sharding=("tensor", None)
            ),
            f"{prefix}.norm.weight": WeightMapping("model.norm.weight", sharding=(None,)),
            "lm_head.weight": WeightMapping("lm_head.embedding", sharding=("tensor", None)),
        }
        for layer_id in range(self.config.num_hidden_layers):
            source = f"{prefix}.layers.{layer_id}"
            target = f"model.layers.{layer_id}"
            for name in (
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ):
                mappings[f"{source}.{name}.weight"] = WeightMapping(
                    f"{target}.{name}.weight", sharding=(None,)
                )
            for name in ("q_proj", "k_proj", "v_proj", "gate_proj"):
                mappings[f"{source}.self_attn.{name}.weight"] = WeightMapping(
                    f"{target}.self_attn.{name}.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                )
            mappings[f"{source}.self_attn.o_proj.weight"] = WeightMapping(
                f"{target}.self_attn.o_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            )
            for name in ("gate_proj", "up_proj"):
                mappings[f"{source}.mlp.{name}.weight"] = WeightMapping(
                    f"{target}.mlp.{name}.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                )
            mappings[f"{source}.mlp.down_proj.weight"] = WeightMapping(
                f"{target}.mlp.down_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            )
        return mappings

    def get_embed_and_head(self) -> tuple[jax.Array, jax.Array]:
        return self.model.embed_tokens.embedding.value, self.lm_head.embedding.value

    def set_embed_and_head(
        self,
        embed_weight: jax.Array | None = None,
        head_weight: jax.Array | None = None,
    ) -> None:
        if embed_weight is not None:
            self.model.embed_tokens.embedding.value = embed_weight
        if head_weight is not None:
            self.lm_head.embedding.value = head_weight

    def set_dflash_layers_to_capture(self, layer_ids: list[int] | None) -> None:
        if layer_ids is None:
            raise ValueError("DFLASH requires explicit target layer ids")
        self.capture_aux_hidden_states = True
        self.model.layers_to_capture = [int(layer_id) + 1 for layer_id in layer_ids]

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, aux_hidden_states, kv_fused, callbacks = self.model(
            forward_batch, memory_pools.token_to_kv_pool
        )
        if not self.capture_aux_hidden_states:
            aux_hidden_states = None
        output = self.logits_processor(
            hidden_states * self.output_multiplier,
            self.lm_head,
            logits_metadata,
            aux_hidden_states=aux_hidden_states,
        )
        return output, {"token_to_kv_pool": kv_fused}, callbacks, None


class MuseGlimmerForCausalLM(MuseGlimmerForConditionalGeneration):
    pass


EntryClass = [MuseGlimmerForConditionalGeneration, MuseGlimmerForCausalLM]
