import logging
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.scipy.special import ndtri
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, RotaryEmbedding
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.profiling_utils import named_scope
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


def _text_config(config: PretrainedConfig) -> PretrainedConfig:
    return getattr(config, "text_config", config)


def _layer_value(value: Any, layer_id: int, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, list | tuple):
        return value[layer_id]
    return value


def _layer_type(config: PretrainedConfig, layer_id: int) -> str:
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        return "full_attention"
    return layer_types[layer_id]


def _rope_theta(config: PretrainedConfig, layer_type: str) -> float:
    rope_parameters = getattr(config, "rope_parameters", None) or {}
    if isinstance(rope_parameters, dict):
        layer_rope = rope_parameters.get(layer_type, {})
        if isinstance(layer_rope, dict) and "rope_theta" in layer_rope:
            return layer_rope["rope_theta"]
    if layer_type == "sliding_attention":
        return getattr(config, "rope_local_base_freq", getattr(config, "rope_theta", 10000.0))
    return getattr(config, "rope_theta", 1000000.0)


def _shared_kv_owner(config: PretrainedConfig, layer_id: int) -> int | None:
    num_shared = getattr(config, "num_kv_shared_layers", 0) or 0
    if num_shared <= 0:
        return None

    first_shared = config.num_hidden_layers - num_shared
    if first_shared <= 0 or layer_id < first_shared:
        return None

    cur_type = _layer_type(config, layer_id)
    prev_layers = [_layer_type(config, i) for i in range(first_shared)]
    for owner in range(first_shared - 1, -1, -1):
        if prev_layers[owner] == cur_type:
            return owner
    return first_shared - 1


class Gemma3nRMSNorm(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        epsilon: float = 1e-6,
        with_scale: bool = True,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.epsilon = epsilon
        self.with_scale = with_scale
        if with_scale:
            self.weight = nnx.Param(jnp.ones((hidden_size,), dtype=dtype))
        else:
            self.weight = None

    @named_scope
    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        orig_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.epsilon)
        if self.weight is not None:
            hidden_states = hidden_states * self.weight.value.astype(jnp.float32)
        return hidden_states.astype(orig_dtype)


class Gemma3nScaledEmbed(Embed):
    def __init__(
        self,
        num_embeddings: int,
        features: int,
        embed_scale: float,
        dtype: jnp.dtype,
        mesh: jax.sharding.Mesh,
    ):
        super().__init__(
            num_embeddings=num_embeddings,
            features=features,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )
        self.embed_scale = embed_scale

    @named_scope
    def __call__(self, inputs: jax.Array) -> jax.Array:
        output = super().__call__(inputs)
        return output * jnp.asarray(self.embed_scale, dtype=output.dtype)


class Gemma3nMLP(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.intermediate_size = int(_layer_value(config.intermediate_size, layer_id))
        activation_pattern = getattr(config, "activation_sparsity_pattern", None)
        self.activation_sparsity = float(_layer_value(activation_pattern, layer_id, 0.0) or 0.0)

        self.gate_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="gate_proj",
        )
        self.up_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="up_proj",
        )
        self.down_proj = LinearBase(
            input_size=self.intermediate_size,
            output_size=self.hidden_size,
            kernel_axes=("tensor", None),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="down_proj",
        )

    def _gaussian_topk(self, inputs: jax.Array) -> jax.Array:
        multiplier = ndtri(jnp.asarray(self.activation_sparsity, dtype=jnp.float32)).astype(
            inputs.dtype
        )
        inputs_mean = jnp.mean(inputs, axis=-1, keepdims=True)
        inputs_std = jnp.std(inputs, axis=-1, keepdims=True)
        cutoff_x = inputs_mean + inputs_std * multiplier
        return jax.nn.relu(inputs - cutoff_x)

    @named_scope
    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        gate, _ = self.gate_proj(hidden_states)
        if self.activation_sparsity > 0.0:
            gate = self._gaussian_topk(gate)
        up, _ = self.up_proj(hidden_states)
        hidden_states = jax.nn.gelu(gate, approximate=True) * up
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class Gemma3nLaurelBlock(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        laurel_rank = getattr(config, "laurel_rank", config.hidden_size)
        self.linear_left = LinearBase(
            input_size=config.hidden_size,
            output_size=laurel_rank,
            kernel_axes=(None, None),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="linear_left",
        )
        self.linear_right = LinearBase(
            input_size=laurel_rank,
            output_size=config.hidden_size,
            kernel_axes=(None, None),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="linear_right",
        )
        self.post_laurel_norm = Gemma3nRMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, dtype=dtype
        )

    @named_scope
    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        laurel_states, _ = self.linear_left(hidden_states)
        laurel_states, _ = self.linear_right(laurel_states)
        return hidden_states + self.post_laurel_norm(laurel_states)


class Gemma3nAltUp(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.altup_num_inputs = getattr(config, "altup_num_inputs", 1)
        self.altup_active_idx = getattr(config, "altup_active_idx", 0)
        self.altup_correct_scale = getattr(config, "altup_correct_scale", True)
        self.router_input_scale = config.hidden_size**-1.0

        self.correct_output_scale = nnx.Param(jnp.zeros((config.hidden_size,), dtype=dtype))
        self.correction_coefs = LinearBase(
            input_size=self.altup_num_inputs,
            output_size=self.altup_num_inputs,
            kernel_axes=(None, None),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="correction_coefs",
        )
        self.prediction_coefs = LinearBase(
            input_size=self.altup_num_inputs,
            output_size=self.altup_num_inputs**2,
            kernel_axes=(None, None),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="prediction_coefs",
        )
        self.modality_router = LinearBase(
            input_size=config.hidden_size,
            output_size=self.altup_num_inputs,
            kernel_axes=(None, None),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="modality_router",
        )
        self.router_norm = Gemma3nRMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, dtype=dtype
        )

    def compute_router_modalities(self, hidden_states: jax.Array) -> jax.Array:
        router_inputs = self.router_norm(hidden_states) * jnp.asarray(
            self.router_input_scale, dtype=hidden_states.dtype
        )
        routed, _ = self.modality_router(router_inputs)
        return jnp.tanh(routed.astype(jnp.float32)).astype(routed.dtype)

    @named_scope
    def predict(self, hidden_states: jax.Array) -> jax.Array:
        modalities = self.compute_router_modalities(hidden_states[self.altup_active_idx])
        all_coefs, _ = self.prediction_coefs(modalities)
        all_coefs = all_coefs.reshape(
            modalities.shape[0], self.altup_num_inputs, self.altup_num_inputs
        )
        all_coefs = jnp.transpose(all_coefs, (0, 2, 1))
        predictions = jnp.einsum("ath,tba->bth", hidden_states, all_coefs)
        return (predictions + hidden_states).astype(hidden_states.dtype)

    @named_scope
    def correct(self, predictions: jax.Array, activated: jax.Array) -> jax.Array:
        modalities = self.compute_router_modalities(activated)
        innovation = activated - predictions[self.altup_active_idx]
        all_coefs, _ = self.correction_coefs(modalities)
        all_coefs = jnp.transpose(all_coefs + 1.0, (1, 0))[:, :, None]
        return (predictions + innovation[None, :, :] * all_coefs).astype(activated.dtype)

    def scale_corrected_output(self, corrected: jax.Array) -> jax.Array:
        return corrected * self.correct_output_scale.value.astype(corrected.dtype)


class Gemma3nAttention(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.kv_shared_layer_index = _shared_kv_owner(config, layer_id)
        self.is_kv_shared_layer = self.kv_shared_layer_index is not None
        self.layer_type = _layer_type(config, layer_id)
        self.is_sliding = self.layer_type == "sliding_attention"
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.mesh = mesh

        self.q_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.num_heads * self.head_dim,
            kernel_axes=(None, "tensor"),
            use_bias=getattr(config, "attention_bias", False),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="q_proj",
        )
        self.q_norm = Gemma3nRMSNorm(self.head_dim, epsilon=config.rms_norm_eps, dtype=dtype)

        if not self.is_kv_shared_layer:
            self.k_proj = LinearBase(
                input_size=self.hidden_size,
                output_size=self.num_kv_heads * self.head_dim,
                kernel_axes=(None, "tensor"),
                use_bias=getattr(config, "attention_bias", False),
                params_dtype=dtype,
                mesh=mesh,
                scope_name="k_proj",
            )
            self.v_proj = LinearBase(
                input_size=self.hidden_size,
                output_size=self.num_kv_heads * self.head_dim,
                kernel_axes=(None, "tensor"),
                use_bias=getattr(config, "attention_bias", False),
                params_dtype=dtype,
                mesh=mesh,
                scope_name="v_proj",
            )
            self.k_norm = Gemma3nRMSNorm(self.head_dim, epsilon=config.rms_norm_eps, dtype=dtype)
            self.v_norm = Gemma3nRMSNorm(
                self.head_dim, epsilon=config.rms_norm_eps, with_scale=False, dtype=dtype
            )
        else:
            self.k_proj = None
            self.v_proj = None
            self.k_norm = None
            self.v_norm = None

        self.o_proj = LinearBase(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            kernel_axes=("tensor", None),
            use_bias=getattr(config, "attention_bias", False),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="o_proj",
        )

        self.rotary_emb = RotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position_embeddings=getattr(config, "max_position_embeddings", 32768),
            base=_rope_theta(config, self.layer_type),
            is_neox_style=True,
            dtype=dtype,
        )

        sliding_window = 0
        if self.is_sliding:
            # Gemma 3n/HF treats the local window as inclusive. RadixAttention's
            # sliding-window boundary is exclusive, so pass one less.
            sliding_window = max(0, getattr(config, "sliding_window", 0) - 1)

        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=1.0,
            num_kv_heads=self.num_kv_heads,
            layer_id=self.kv_shared_layer_index if self.is_kv_shared_layer else layer_id,
            sliding_window_size=sliding_window,
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
        q = q.reshape(
            -1,
            self.num_heads,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )
        q = self.q_norm(q)

        if self.is_kv_shared_layer:
            k = jnp.zeros(
                (q.shape[0], self.num_kv_heads, self.head_dim),
                dtype=q.dtype,
            )
            v = jnp.zeros_like(k)
            q, _ = self.rotary_emb(positions, q, k)
        else:
            k, _ = self.k_proj(hidden_states)
            v, _ = self.v_proj(hidden_states)
            k = k.reshape(
                -1,
                self.num_kv_heads,
                self.head_dim,
                out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
            )
            v = v.reshape(
                -1,
                self.num_kv_heads,
                self.head_dim,
                out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
            )
            k = self.k_norm(k)
            v = self.v_norm(v)
            q, k = self.rotary_emb(positions, q, k)

        attn_output, kv_fused = self.attn(
            q,
            k,
            v,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
            save_kv_cache=not self.is_kv_shared_layer,
        )
        output, _ = self.o_proj(attn_output)

        if self.is_kv_shared_layer:
            kv_fused = token_to_kv_pool.get_fused_kv_buffer(self.layer_id)
        return output, kv_fused


class Gemma3nDecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = config.hidden_size
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        self.altup_active_idx = getattr(config, "altup_active_idx", 0)
        self.altup_correct_scale = getattr(config, "altup_correct_scale", True)

        self.altup = Gemma3nAltUp(config, mesh=mesh, dtype=dtype)
        self.input_layernorm = Gemma3nRMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, dtype=dtype
        )
        self.laurel = Gemma3nLaurelBlock(config, mesh=mesh, dtype=dtype)
        self.self_attn = Gemma3nAttention(config, mesh=mesh, layer_id=layer_id, dtype=dtype)
        self.post_attention_layernorm = Gemma3nRMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, dtype=dtype
        )
        self.pre_feedforward_layernorm = Gemma3nRMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, dtype=dtype
        )
        self.mlp = Gemma3nMLP(config, mesh=mesh, layer_id=layer_id, dtype=dtype)
        self.post_feedforward_layernorm = Gemma3nRMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, dtype=dtype
        )

        self.per_layer_input_gate = LinearBase(
            input_size=config.hidden_size,
            output_size=config.hidden_size_per_layer_input,
            kernel_axes=(None, None),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="per_layer_input_gate",
        )
        self.per_layer_projection = LinearBase(
            input_size=config.hidden_size_per_layer_input,
            output_size=config.hidden_size,
            kernel_axes=(None, None),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="per_layer_projection",
        )
        self.post_per_layer_input_norm = Gemma3nRMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, dtype=dtype
        )

    @named_scope
    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        per_layer_input: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, jax.Array]:
        predictions = self.altup.predict(hidden_states)
        active_prediction = predictions[self.altup_active_idx]

        active_prediction_normed = self.input_layernorm(active_prediction)
        laurel_output = self.laurel(active_prediction_normed)

        attn, kv_fused = self.self_attn(
            positions=positions,
            hidden_states=active_prediction_normed,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )
        attn = self.post_attention_layernorm(attn)

        attn_gated = active_prediction + attn
        attn_laurel = (attn_gated + laurel_output) * jnp.asarray(
            2**-0.5, dtype=active_prediction.dtype
        )

        attn_norm = self.pre_feedforward_layernorm(attn_laurel)
        attn_ffw = self.mlp(attn_norm)
        attn_ffw_norm = self.post_feedforward_layernorm(attn_ffw)
        corrected_predictions = self.altup.correct(predictions, attn_laurel + attn_ffw_norm)

        first_prediction = corrected_predictions[self.altup_active_idx]
        if self.altup_correct_scale:
            first_prediction = self.altup.scale_corrected_output(first_prediction)

        first_prediction, _ = self.per_layer_input_gate(first_prediction)
        first_prediction = jax.nn.gelu(first_prediction, approximate=True)
        first_prediction = first_prediction * per_layer_input
        first_prediction, _ = self.per_layer_projection(first_prediction)
        first_prediction = self.post_per_layer_input_norm(first_prediction)

        if getattr(self.config, "altup_num_inputs", 1) > 1:
            corrected_predictions = corrected_predictions.at[1:].add(first_prediction)
        return corrected_predictions, kv_fused


class Gemma3nModel(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.hidden_size = config.hidden_size
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        self.num_hidden_layers = config.num_hidden_layers
        self.altup_num_inputs = getattr(config, "altup_num_inputs", 1)

        self.embed_tokens = Gemma3nScaledEmbed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            embed_scale=config.hidden_size**0.5,
            dtype=dtype,
            mesh=mesh,
        )
        self.embed_tokens_per_layer = Gemma3nScaledEmbed(
            num_embeddings=config.vocab_size_per_layer_input,
            features=config.num_hidden_layers * config.hidden_size_per_layer_input,
            embed_scale=config.hidden_size_per_layer_input**0.5,
            dtype=dtype,
            mesh=mesh,
        )
        self.per_layer_model_projection = LinearBase(
            input_size=config.hidden_size,
            output_size=config.num_hidden_layers * config.hidden_size_per_layer_input,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="per_layer_model_projection",
        )
        self.per_layer_projection_norm = Gemma3nRMSNorm(
            config.hidden_size_per_layer_input, epsilon=config.rms_norm_eps, dtype=dtype
        )

        self.altup_projections = nnx.data(
            [
                LinearBase(
                    input_size=config.hidden_size,
                    output_size=config.hidden_size,
                    kernel_axes=(None, "tensor"),
                    use_bias=False,
                    params_dtype=dtype,
                    mesh=mesh,
                    scope_name=f"altup_projection_{i}",
                )
                for i in range(self.altup_num_inputs - 1)
            ]
        )
        self.altup_unembed_projections = nnx.data(
            [
                LinearBase(
                    input_size=config.hidden_size,
                    output_size=config.hidden_size,
                    kernel_axes=(None, "tensor"),
                    use_bias=False,
                    params_dtype=dtype,
                    mesh=mesh,
                    scope_name=f"altup_unembed_projection_{i}",
                )
                for i in range(self.altup_num_inputs - 1)
            ]
        )

        self.layers = nnx.data(
            [
                Gemma3nDecoderLayer(config, mesh=mesh, layer_id=i, dtype=dtype)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = Gemma3nRMSNorm(config.hidden_size, epsilon=config.rms_norm_eps, dtype=dtype)

    def get_per_layer_inputs(self, input_ids: jax.Array) -> jax.Array:
        input_ids = jnp.where(
            (input_ids >= 0) & (input_ids < self.config.vocab_size_per_layer_input),
            input_ids,
            jnp.zeros_like(input_ids),
        )
        embeddings = self.embed_tokens_per_layer(input_ids)
        return embeddings.reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(
        self,
        input_embeds: jax.Array,
        per_layer_inputs: jax.Array | None = None,
    ) -> jax.Array:
        per_layer_projection, _ = self.per_layer_model_projection(input_embeds)
        per_layer_projection = per_layer_projection * jnp.asarray(
            self.hidden_size**-0.5, dtype=input_embeds.dtype
        )
        per_layer_projection = per_layer_projection.reshape(
            *input_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection

        if per_layer_projection.shape != per_layer_inputs.shape:
            per_layer_inputs = per_layer_inputs[..., : self.config.num_hidden_layers, :]

        return (per_layer_projection + per_layer_inputs) * jnp.asarray(
            2**-0.5, dtype=input_embeds.dtype
        )

    def _rescale_to_target_magnitude(
        self,
        hidden_states: jax.Array,
        target_magnitude: jax.Array,
    ) -> jax.Array:
        magnitude_square = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
        eps = jnp.asarray(1e-5, dtype=hidden_states.dtype)
        new_magnitude = jnp.sqrt(jnp.maximum(magnitude_square, eps))
        return hidden_states * (target_magnitude / new_magnitude)

    @named_scope
    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, list[jax.Array]]:
        if (
            forward_batch.input_embedding is not None
            and forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
        ):
            input_embeds = forward_batch.input_embedding
            per_layer_inputs = self.get_per_layer_inputs(forward_batch.input_ids)
        else:
            input_embeds = self.embed_tokens(forward_batch.input_ids)
            per_layer_inputs = self.get_per_layer_inputs(forward_batch.input_ids)

        per_layer_inputs = self.project_per_layer_inputs(input_embeds, per_layer_inputs)
        target_magnitude = jnp.mean(jnp.square(input_embeds), axis=-1, keepdims=True) ** 0.5

        temp_hidden_states = [input_embeds]
        for projection in self.altup_projections:
            current_hidden_state, _ = projection(input_embeds)
            current_hidden_state = self._rescale_to_target_magnitude(
                current_hidden_state.astype(input_embeds.dtype), target_magnitude
            )
            temp_hidden_states.append(current_hidden_state)
        hidden_states = jnp.stack(temp_hidden_states, axis=0)

        layers_kv_fused = []
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, kv_fused = layer(
                positions=forward_batch.positions,
                hidden_states=hidden_states,
                per_layer_input=per_layer_inputs[:, layer_idx, :],
                forward_batch=forward_batch,
                token_to_kv_pool=token_to_kv_pool,
            )
            layers_kv_fused.append(kv_fused)

        target_magnitude = jnp.mean(jnp.square(hidden_states[0]), axis=-1, keepdims=True) ** 0.5
        temp_hidden_states = [hidden_states[0]]
        for i, projection in enumerate(self.altup_unembed_projections, start=1):
            current_hidden_state, _ = projection(hidden_states[i])
            current_hidden_state = self._rescale_to_target_magnitude(
                current_hidden_state.astype(hidden_states[0].dtype), target_magnitude
            )
            temp_hidden_states.append(current_hidden_state)
        hidden_states = jnp.mean(jnp.stack(temp_hidden_states, axis=0), axis=0)
        hidden_states = self.norm(hidden_states)
        return hidden_states, layers_kv_fused


class Gemma3nForCausalLM(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = _text_config(config)
        self.dtype = dtype
        self.model = Gemma3nModel(self.config, mesh=mesh, dtype=dtype)
        if not getattr(self.config, "tie_word_embeddings", True):
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                dtype=dtype,
                param_dtype=dtype,
                kernel_axes=("tensor", None),
                mesh=mesh,
            )
        self.logits_processor = LogitsProcessor(self.config.vocab_size, mesh=self.mesh)

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_weight_mappings()
        if not loader.dummy_mode:
            weight_info = loader._scan_weight_info()
            weight_mappings = {k: v for k, v in weight_mappings.items() if k in weight_info}
        loader.load_weights_from_safetensors(weight_mappings)

        if hasattr(self, "lm_head") and isinstance(
            self.lm_head.embedding.value, jax.ShapeDtypeStruct
        ):
            self.lm_head.embedding = self.model.embed_tokens.embedding

        logger.info("Gemma3n weights loaded successfully!")

    def _create_weight_mappings(self) -> dict[str, WeightMapping]:
        mappings: dict[str, WeightMapping] = {
            "model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.embed_tokens_per_layer.weight": WeightMapping(
                target_path="model.embed_tokens_per_layer.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.per_layer_model_projection.weight": WeightMapping(
                target_path="model.per_layer_model_projection.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            "model.per_layer_projection_norm.weight": WeightMapping(
                target_path="model.per_layer_projection_norm.weight",
                sharding=(None,),
                transpose=False,
            ),
            "model.norm.weight": WeightMapping(
                target_path="model.norm.weight",
                sharding=(None,),
                transpose=False,
            ),
        }

        if hasattr(self, "lm_head"):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding",
                sharding=("tensor", None),
                transpose=False,
            )

        for i in range(getattr(self.config, "altup_num_inputs", 1) - 1):
            mappings[f"model.altup_projections.{i}.weight"] = WeightMapping(
                target_path=f"model.altup_projections.{i}.weight",
                sharding=(None, "tensor"),
                transpose=True,
            )
            mappings[f"model.altup_unembed_projections.{i}.weight"] = WeightMapping(
                target_path=f"model.altup_unembed_projections.{i}.weight",
                sharding=(None, "tensor"),
                transpose=True,
            )

        for layer_idx in range(self.config.num_hidden_layers):
            mappings.update(self._create_layer_mappings(layer_idx))

        aliases = {}
        for key, mapping in mappings.items():
            if key.startswith("model."):
                aliases[f"model.language_model.{key[len('model.') :]}"] = mapping
                aliases[f"language_model.{key}"] = mapping
        mappings.update(aliases)
        return mappings

    def _create_layer_mappings(self, layer_idx: int) -> dict[str, WeightMapping]:
        prefix = f"model.layers.{layer_idx}"
        is_shared = _shared_kv_owner(self.config, layer_idx) is not None
        mappings: dict[str, WeightMapping] = {
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
            f"{prefix}.pre_feedforward_layernorm.weight": WeightMapping(
                target_path=f"{prefix}.pre_feedforward_layernorm.weight",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.post_feedforward_layernorm.weight": WeightMapping(
                target_path=f"{prefix}.post_feedforward_layernorm.weight",
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
            f"{prefix}.self_attn.q_norm.weight": WeightMapping(
                target_path=f"{prefix}.self_attn.q_norm.weight",
                sharding=(None,),
                transpose=False,
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
            f"{prefix}.altup.correct_output_scale": WeightMapping(
                target_path=f"{prefix}.altup.correct_output_scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.altup.correction_coefs.weight": WeightMapping(
                target_path=f"{prefix}.altup.correction_coefs.weight",
                sharding=(None, None),
                transpose=True,
            ),
            f"{prefix}.altup.prediction_coefs.weight": WeightMapping(
                target_path=f"{prefix}.altup.prediction_coefs.weight",
                sharding=(None, None),
                transpose=True,
            ),
            f"{prefix}.altup.modality_router.weight": WeightMapping(
                target_path=f"{prefix}.altup.modality_router.weight",
                sharding=(None, None),
                transpose=True,
            ),
            f"{prefix}.altup.router_norm.weight": WeightMapping(
                target_path=f"{prefix}.altup.router_norm.weight",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.laurel.linear_left.weight": WeightMapping(
                target_path=f"{prefix}.laurel.linear_left.weight",
                sharding=(None, None),
                transpose=True,
            ),
            f"{prefix}.laurel.linear_right.weight": WeightMapping(
                target_path=f"{prefix}.laurel.linear_right.weight",
                sharding=(None, None),
                transpose=True,
            ),
            f"{prefix}.laurel.post_laurel_norm.weight": WeightMapping(
                target_path=f"{prefix}.laurel.post_laurel_norm.weight",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.per_layer_input_gate.weight": WeightMapping(
                target_path=f"{prefix}.per_layer_input_gate.weight",
                sharding=(None, None),
                transpose=True,
            ),
            f"{prefix}.per_layer_projection.weight": WeightMapping(
                target_path=f"{prefix}.per_layer_projection.weight",
                sharding=(None, None),
                transpose=True,
            ),
            f"{prefix}.post_per_layer_input_norm.weight": WeightMapping(
                target_path=f"{prefix}.post_per_layer_input_norm.weight",
                sharding=(None,),
                transpose=False,
            ),
        }

        if not is_shared:
            mappings.update(
                {
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
                    f"{prefix}.self_attn.k_norm.weight": WeightMapping(
                        target_path=f"{prefix}.self_attn.k_norm.weight",
                        sharding=(None,),
                        transpose=False,
                    ),
                }
            )

        if getattr(self.config, "attention_bias", False):
            mappings[f"{prefix}.self_attn.q_proj.bias"] = WeightMapping(
                target_path=f"{prefix}.self_attn.q_proj.bias",
                sharding=(None,),
                transpose=False,
                head_dim_padding=True,
                kv_head_padding=False,
            )
            mappings[f"{prefix}.self_attn.o_proj.bias"] = WeightMapping(
                target_path=f"{prefix}.self_attn.o_proj.bias",
                sharding=(None,),
                transpose=False,
            )
            if not is_shared:
                mappings[f"{prefix}.self_attn.k_proj.bias"] = WeightMapping(
                    target_path=f"{prefix}.self_attn.k_proj.bias",
                    sharding=(None,),
                    transpose=False,
                    head_dim_padding=True,
                    kv_head_padding=True,
                )
                mappings[f"{prefix}.self_attn.v_proj.bias"] = WeightMapping(
                    target_path=f"{prefix}.self_attn.v_proj.bias",
                    sharding=(None,),
                    transpose=False,
                    head_dim_padding=True,
                    kv_head_padding=True,
                )

        return mappings

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, layers_kv_fused = self.model(forward_batch, memory_pools.token_to_kv_pool)
        if hasattr(self, "lm_head"):
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)
        return output, {"token_to_kv_pool": layers_kv_fused}, True, None


class Gemma3nForConditionalGeneration(Gemma3nForCausalLM):
    pass


EntryClass = [Gemma3nForCausalLM, Gemma3nForConditionalGeneration]
