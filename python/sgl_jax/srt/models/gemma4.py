import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import safetensors
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.eplb.expert_location import topk_ids_logical_to_physical
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, get_rope
from sgl_jax.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import EPMoE, GateLogit, TopK
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.precision_tracer import precision_tracer
from sgl_jax.srt.utils.profiling_utils import named_scope
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)

init_fn = nnx.initializers.uniform()


class Gemma4Router(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        mesh: jax.sharding.Mesh | None = None,
    ):
        self.hidden_size = config.hidden_size
        self.num_experts = getattr(config, "num_experts", 128)
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)

        self.norm = RMSNorm(
            self.hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            use_scale=False,
            scope_name="norm",
        )
        self.scale = nnx.Param(jnp.ones((self.hidden_size,), dtype=dtype))
        self.root_size = self.hidden_size**-0.5
        self.proj = GateLogit(
            input_size=self.hidden_size,
            num_experts=self.num_experts,
            weight_dtype=dtype,
        )
        self.per_expert_scale = nnx.Param(jnp.ones((self.num_experts,), dtype=dtype))

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        x = self.norm(hidden_states)
        x = x * self.root_size
        x = x * self.scale.value.astype(x.dtype)
        router_logits = self.proj(x)
        return router_logits


class Gemma4MLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.layer_id = layer_id

        self.gate_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="gate_proj",
        )

        self.up_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="up_proj",
        )

        self.down_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            kernel_axes=("tensor", None),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="down_proj",
        )

        self.act_fn = jax.nn.gelu

    @named_scope
    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        a1, _ = self.gate_proj(hidden_states)
        a2, _ = self.up_proj(hidden_states)
        intermediate_parallel = a2 * self.act_fn(a1)
        output, _ = self.down_proj(intermediate_parallel)
        return output


class Gemma4Attention(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        max_position_embeddings: int,
        attention_bias: bool,
        dtype: jnp.dtype,
        mesh: jax.sharding.Mesh,
    ):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)

        self.layer_type = "full_attention"
        if hasattr(config, "layer_types") and layer_id < len(config.layer_types):
            self.layer_type = config.layer_types[layer_id]

        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = getattr(config, "sliding_window", 0) if self.is_sliding else 0

        rope_parameters = getattr(config, "rope_parameters", {})
        rope_params = dict(rope_parameters.get(self.layer_type, {}))
        if not self.is_sliding:
            if "rope_type" not in rope_params:
                rope_params["rope_type"] = "proportional"
            if "partial_rotary_factor" not in rope_params:
                rope_params["partial_rotary_factor"] = 0.25
            rope_theta = rope_params.get("rope_theta", getattr(config, "rope_theta", 1000000.0))
        else:
            if "rope_type" not in rope_params:
                rope_params["rope_type"] = "default"
            if "partial_rotary_factor" not in rope_params:
                rope_params["partial_rotary_factor"] = 1.0
            rope_theta = rope_params.get(
                "rope_theta", getattr(config, "rope_local_base_freq", 10000.0)
            )

        if not self.is_sliding:
            self.head_dim = getattr(
                config,
                "head_dim",
                self.hidden_size // self.num_heads,
            )
        else:
            self.head_dim = getattr(
                config,
                "swa_head_dim",
                getattr(config, "head_dim", self.hidden_size // self.num_heads),
            )

        self.use_k_eq_v = (not self.is_sliding) and getattr(config, "attention_k_eq_v", False)
        if self.use_k_eq_v:
            self.num_kv_heads = getattr(
                config,
                "num_key_value_heads",
                self.num_heads,
            )
        else:
            self.num_kv_heads = getattr(
                config,
                "swa_num_key_value_heads",
                getattr(config, "num_key_value_heads", self.num_heads),
            )

        self.q_head_num = self.num_heads
        self.kv_head_num = self.num_kv_heads
        self.mesh = mesh

        self.q_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.num_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="q_proj",
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, epsilon=rms_norm_eps, add_unit_offset=False)

        self.k_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.num_kv_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="k_proj",
        )
        self.k_norm = GemmaRMSNorm(self.head_dim, epsilon=rms_norm_eps, add_unit_offset=False)

        if self.use_k_eq_v:
            self.v_proj = None
        else:
            self.v_proj = LinearBase(
                input_size=self.hidden_size,
                output_size=self.num_kv_heads * self.head_dim,
                use_bias=attention_bias,
                kernel_axes=(None, "tensor"),
                params_dtype=dtype,
                mesh=mesh,
                scope_name="v_proj",
            )
        self.v_norm = RMSNorm(
            self.head_dim,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            use_scale=False,
            scope_name="v_norm",
        )

        self.o_proj = LinearBase(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            use_bias=attention_bias,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="o_proj",
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            rope_scaling=rope_params,
            dtype=dtype,
        )

        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=1.0,
            num_kv_heads=self.num_kv_heads,
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
        if self.v_proj is None:
            v = k
        else:
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

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = self.rotary_emb(positions, q, k)
        v = self.v_norm(v)

        attn_output, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool)
        output, _ = self.o_proj(attn_output)
        return output, kv_fused


class Gemma4DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.mesh = mesh
        max_position_embeddings = getattr(config, "max_position_embeddings", 256000)
        attention_bias = getattr(config, "attention_bias", False)
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)

        self.layer_scalar = nnx.Param(jnp.ones((1,), dtype=dtype))

        self.input_layernorm = GemmaRMSNorm(
            config.hidden_size,
            epsilon=rms_norm_eps,
            add_unit_offset=False,
        )
        self.self_attn = Gemma4Attention(
            config=config,
            layer_id=layer_id,
            max_position_embeddings=max_position_embeddings,
            attention_bias=attention_bias,
            dtype=dtype,
            mesh=mesh,
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size,
            epsilon=rms_norm_eps,
            add_unit_offset=False,
        )
        self.pre_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size,
            epsilon=rms_norm_eps,
            add_unit_offset=False,
        )
        self.mlp = Gemma4MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            layer_id=layer_id,
            dtype=dtype,
            mesh=mesh,
        )
        self.post_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size,
            epsilon=rms_norm_eps,
            add_unit_offset=False,
        )

        self.enable_moe_block = getattr(config, "enable_moe_block", False)
        if self.enable_moe_block:
            num_experts = getattr(config, "num_experts", 128)
            num_experts_per_tok = getattr(
                config, "num_experts_per_tok", getattr(config, "top_k_experts", 8)
            )
            moe_intermediate_size = getattr(
                config, "moe_intermediate_size", config.intermediate_size
            )

            self.router = Gemma4Router(
                config=config,
                dtype=dtype,
                mesh=mesh,
            )
            self.topk = TopK(
                topk=num_experts_per_tok,
                renormalize=True,
                layer_id=layer_id,
            )
            self.experts = EPMoE(
                hidden_size=config.hidden_size,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                intermediate_dim=moe_intermediate_size,
                mesh=mesh,
                ep_size=getattr(config, "ep_size", 1),
                weight_dtype=dtype,
                dtype=dtype,
                activation="gelu",
                layer_id=layer_id,
                quantization_config=getattr(config, "quantization_config", None),
            )
            self.post_feedforward_layernorm_1 = GemmaRMSNorm(
                config.hidden_size,
                epsilon=rms_norm_eps,
                add_unit_offset=False,
            )
            self.post_feedforward_layernorm_2 = GemmaRMSNorm(
                config.hidden_size,
                epsilon=rms_norm_eps,
                add_unit_offset=False,
            )
            self.pre_feedforward_layernorm_2 = GemmaRMSNorm(
                config.hidden_size,
                epsilon=rms_norm_eps,
                add_unit_offset=False,
            )
        else:
            self.router = None
            self.topk = None
            self.experts = None
            self.post_feedforward_layernorm_1 = None
            self.post_feedforward_layernorm_2 = None
            self.pre_feedforward_layernorm_2 = None

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ):
        layer_callback_flag = []
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        layer_norm_callback_flag = precision_tracer.jit_pure_callback_record(
            hidden_states, "input_layernorm_output", "INPUT_LAYERNORM", self.layer_id
        )
        layer_callback_flag.append(layer_norm_callback_flag)

        hidden_states, kv_fused = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )

        attn_callback_flag = precision_tracer.jit_pure_callback_record(
            hidden_states, "self_attn_output", "SELF_ATTN", self.layer_id
        )
        layer_callback_flag.append(attn_callback_flag)

        attn_output = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + attn_output
        residual = hidden_states

        expert_ids = None
        if self.enable_moe_block:
            hidden_states_1 = self.pre_feedforward_layernorm(hidden_states)
            hidden_states_1 = self.mlp(hidden_states_1)
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states_1)

            router_logits = self.router(hidden_states)
            topk_weights, topk_ids = self.topk(router_logits, dispatch_info=None)
            expert_scales = self.router.per_expert_scale.value.at[topk_ids].get(
                out_sharding=NamedSharding(self.mesh, P("data"))
            )
            topk_weights = topk_weights * expert_scales

            dispatch_info = getattr(forward_batch, "expert_location_metadata", None)
            if dispatch_info is not None:
                topk_ids = topk_ids_logical_to_physical(topk_ids, dispatch_info, self.layer_id)

            hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states)
            hidden_states_2 = self.experts(hidden_states_2, topk_weights, topk_ids)
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

            hidden_states = hidden_states_1 + hidden_states_2
            expert_ids = jax.sharding.reshard(topk_ids, P(None))
        else:
            mlp_input = self.pre_feedforward_layernorm(hidden_states)
            hidden_states = self.mlp(mlp_input)

        mlp_output = self.post_feedforward_layernorm(hidden_states)
        outputs = residual + mlp_output
        outputs = outputs * self.layer_scalar.value

        mlp_callback_flag = precision_tracer.jit_pure_callback_record(
            outputs, "mlp_output", "MLP", self.layer_id
        )
        layer_callback_flag.append(mlp_callback_flag)

        return outputs, kv_fused, layer_callback_flag, expert_ids


class Gemma4Model(nnx.Module):
    """Gemma 4 core model structure architecture.

    Supports dynamic selection between full global attention and local sliding window attention layers.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            kernel_axes=("tensor", None),
            param_dtype=dtype,
            mesh=mesh,
        )

        self.layers = nnx.data(
            [
                Gemma4DecoderLayer(
                    config=config,
                    layer_id=i,
                    dtype=dtype,
                    mesh=mesh,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = GemmaRMSNorm(
            config.hidden_size,
            epsilon=getattr(config, "rms_norm_eps", 1e-6),
            add_unit_offset=False,
        )
        self.hidden_size = config.hidden_size
        self.layers_to_capture = []

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ):
        if (
            forward_batch.input_embedding is not None
            and forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
        ):
            hidden_states = forward_batch.input_embedding
        else:
            hidden_states = self.embed_tokens(forward_batch.input_ids)
            hidden_states *= jnp.array([self.hidden_size**0.5], dtype=hidden_states.dtype)

        layers_kv_fused = []
        layers_callback_flag = []
        layers_topk_ids = []
        aux_hidden_states = []
        for layer_id, layer in enumerate(self.layers):
            if layer_id in self.layers_to_capture:
                aux_hidden_states.append(hidden_states)
            hidden_states, kv_fused, callback_flag, topk_ids = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
            )
            layers_kv_fused.append(kv_fused)
            layers_callback_flag.extend(callback_flag)
            layers_topk_ids.append(topk_ids)

        hidden_states = self.norm(hidden_states)

        callback_flag = precision_tracer.jit_pure_callback_record(
            hidden_states, "transformer_output", "TRANSFORMER"
        )
        layers_callback_flag.append(callback_flag)
        return (
            hidden_states,
            aux_hidden_states,
            layers_kv_fused,
            layers_callback_flag,
            layers_topk_ids,
        )


class Gemma4ForCausalLM(nnx.Module):
    """Gemma 4 Causal LM wrapper supporting sharded weights loading and logits evaluation."""

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = getattr(config, "text_config", config)
        self.dtype = dtype
        logger.info("Gemma4ForCausalLM config dtype: %s", self.dtype)
        self.model = Gemma4Model(self.config, dtype=self.dtype, mesh=mesh)
        if not getattr(self.config, "tie_word_embeddings", True):
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=("tensor", None),
            )
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size,
            soft_cap=getattr(self.config, "final_logit_softcapping", 0.0),
            mesh=self.mesh,
        )
        self.capture_aux_hidden_states = False

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )

        weight_mappings = self._create_gemma4_weight_mappings()
        if not loader.dummy_mode:
            # Filter weight mappings to match exact safetensors index keys, preventing false-positive "weight not found" errors
            weight_info = loader._scan_weight_info()
            weight_mappings = {k: v for k, v in weight_mappings.items() if k in weight_info}

        loader.load_weights_from_safetensors(weight_mappings)

        if getattr(self.config, "enable_moe_block", False) and not loader.dummy_mode:
            weight_info = loader._scan_weight_info()
            for layer_idx, layer in enumerate(self.model.layers):
                if layer.enable_moe_block and layer.experts is not None:
                    key_gu = f"model.language_model.layers.{layer_idx}.experts.gate_up_proj"
                    if key_gu not in weight_info:
                        key_gu = (
                            f"model.language_model.layers.{layer_idx}.experts.gate_up_proj.weight"
                        )
                    if key_gu not in weight_info:
                        key_gu = f"model.layers.{layer_idx}.experts.gate_up_proj"
                    if key_gu in weight_info:
                        fn = weight_info[key_gu][0]["file"]
                        with safetensors.safe_open(fn, framework="np", device="cpu") as f:
                            tensor = f.get_tensor(key_gu)
                            F = tensor.shape[1] // 2
                            w0 = np.transpose(tensor[:, :F, :], (0, 2, 1))
                            w1 = np.transpose(tensor[:, F:, :], (0, 2, 1))
                            sharding_w0 = (
                                jax.sharding.NamedSharding(
                                    layer.experts.moe_mesh, P("expert", None, "tensor")
                                )
                                if hasattr(layer.experts, "moe_mesh")
                                else None
                            )
                            layer.experts.wi_0.value = (
                                jax.device_put(w0.astype(jnp.float32), sharding_w0).astype(
                                    self.dtype
                                )
                                if sharding_w0
                                else jnp.array(w0, dtype=self.dtype)
                            )
                            layer.experts.wi_1.value = (
                                jax.device_put(w1.astype(jnp.float32), sharding_w0).astype(
                                    self.dtype
                                )
                                if sharding_w0
                                else jnp.array(w1, dtype=self.dtype)
                            )

                    key_down = f"model.language_model.layers.{layer_idx}.experts.down_proj"
                    if key_down not in weight_info:
                        key_down = (
                            f"model.language_model.layers.{layer_idx}.experts.down_proj.weight"
                        )
                    if key_down not in weight_info:
                        key_down = f"model.layers.{layer_idx}.experts.down_proj"
                    if key_down in weight_info:
                        fn = weight_info[key_down][0]["file"]
                        with safetensors.safe_open(fn, framework="np", device="cpu") as f:
                            tensor = f.get_tensor(key_down)
                            wo = np.transpose(tensor, (0, 2, 1))
                            sharding_wo = (
                                jax.sharding.NamedSharding(
                                    layer.experts.moe_mesh, P("expert", "tensor", None)
                                )
                                if hasattr(layer.experts, "moe_mesh")
                                else None
                            )
                            layer.experts.wo.value = (
                                jax.device_put(wo.astype(jnp.float32), sharding_wo).astype(
                                    self.dtype
                                )
                                if sharding_wo
                                else jnp.array(wo, dtype=self.dtype)
                            )

        if getattr(self.config, "enable_moe_block", False) and loader.dummy_mode:
            ep_size = getattr(self.config, "ep_size", 1)
            world_size = self.mesh.shape.get("data", 1) * self.mesh.shape.get("tensor", 1)
            tp_size = world_size // ep_size
            devices = self.mesh.devices.flatten()
            moe_mesh = jax.sharding.Mesh(
                devices.reshape(ep_size, tp_size),
                axis_names=("expert", "tensor"),
                axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
            )
            for layer in self.model.layers:
                if layer.enable_moe_block and layer.experts is not None:
                    sharding_wi = jax.sharding.NamedSharding(moe_mesh, P("expert", None, "tensor"))
                    sharding_wo = jax.sharding.NamedSharding(moe_mesh, P("expert", "tensor", None))

                    shape_wi0 = layer.experts.wi_0.value.shape
                    shape_wi1 = layer.experts.wi_1.value.shape
                    shape_wo = layer.experts.wo.value.shape

                    layer.experts.wi_0.value = jax.device_put(
                        jnp.zeros(shape_wi0, dtype=self.dtype), sharding_wi
                    )
                    layer.experts.wi_1.value = jax.device_put(
                        jnp.zeros(shape_wi1, dtype=self.dtype), sharding_wi
                    )
                    layer.experts.wo.value = jax.device_put(
                        jnp.zeros(shape_wo, dtype=self.dtype), sharding_wo
                    )

        if hasattr(self, "lm_head") and isinstance(
            self.lm_head.embedding.value, jax.ShapeDtypeStruct
        ):
            logger.info("Tying lm_head weights to embed_tokens (lm_head not in safetensors)")
            self.lm_head.embedding = self.model.embed_tokens.embedding

        logger.info("Gemma4 weights loaded successfully!")

    def _create_gemma4_weight_mappings(self) -> dict:
        mappings = {
            "model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.norm.weight": WeightMapping(
                target_path="model.norm.weight", sharding=(None,), transpose=False
            ),
        }

        if hasattr(self, "lm_head"):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding", sharding=("tensor", None), transpose=False
            )

        num_layers = self.config.num_hidden_layers
        for layer_idx in range(num_layers):
            layer_mappings = self._create_layer_mappings(layer_idx)
            mappings.update(layer_mappings)

        multimodal_mappings = {f"language_model.{k}": v for k, v in mappings.items()}
        mappings.update(multimodal_mappings)

        model_lm_mappings = {
            k.replace("model.", "model.language_model."): v
            for k, v in mappings.items()
            if k.startswith("model.")
        }
        mappings.update(model_lm_mappings)

        return mappings

    def _create_layer_mappings(self, layer_idx: int) -> dict:
        prefix = f"model.layers.{layer_idx}"
        target_prefix = f"model.layers.{layer_idx}"
        layer_type = "full_attention"
        if hasattr(self.config, "layer_types") and layer_idx < len(self.config.layer_types):
            layer_type = self.config.layer_types[layer_idx]
        is_sliding = layer_type == "sliding_attention"
        use_k_eq_v = (not is_sliding) and getattr(self.config, "attention_k_eq_v", False)

        mappings = {
            f"{prefix}.layer_scalar": WeightMapping(
                target_path=f"{target_prefix}.layer_scalar",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.input_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.input_layernorm.weight",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.post_attention_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.post_attention_layernorm.weight",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.pre_feedforward_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.pre_feedforward_layernorm.weight",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.post_feedforward_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.post_feedforward_layernorm.weight",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.self_attn.q_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                kv_head_padding=False,
            ),
            f"{prefix}.self_attn.k_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.k_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                kv_head_padding=use_k_eq_v,
            ),
            f"{prefix}.self_attn.o_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.o_proj.weight",
                sharding=("tensor", None),
                transpose=True,
                kv_head_padding=False,
            ),
            f"{prefix}.self_attn.q_norm.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_norm.weight",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.self_attn.k_norm.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.k_norm.weight",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.mlp.gate_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.gate_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.up_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.up_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.down_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.down_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
        }

        if not use_k_eq_v:
            mappings[f"{prefix}.self_attn.v_proj.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.v_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                kv_head_padding=use_k_eq_v,
            )

        if getattr(self.config, "enable_moe_block", False):
            moe_norm_mappings = {
                f"{prefix}.router.scale": WeightMapping(
                    target_path=f"{target_prefix}.router.scale", sharding=(None,), transpose=False
                ),
                f"{prefix}.router.per_expert_scale": WeightMapping(
                    target_path=f"{target_prefix}.router.per_expert_scale",
                    sharding=(None,),
                    transpose=False,
                ),
                f"{prefix}.router.proj.weight": WeightMapping(
                    target_path=f"{target_prefix}.router.proj.kernel",
                    sharding=(None, None),
                    transpose=True,
                ),
                f"{prefix}.post_feedforward_layernorm_1.weight": WeightMapping(
                    target_path=f"{target_prefix}.post_feedforward_layernorm_1.weight",
                    sharding=(None,),
                    transpose=False,
                ),
                f"{prefix}.post_feedforward_layernorm_2.weight": WeightMapping(
                    target_path=f"{target_prefix}.post_feedforward_layernorm_2.weight",
                    sharding=(None,),
                    transpose=False,
                ),
                f"{prefix}.pre_feedforward_layernorm_2.weight": WeightMapping(
                    target_path=f"{target_prefix}.pre_feedforward_layernorm_2.weight",
                    sharding=(None,),
                    transpose=False,
                ),
            }
            mappings.update(moe_norm_mappings)

        if getattr(self.config, "attention_bias", False):
            bias_mappings = {
                f"{prefix}.self_attn.q_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.q_proj.bias",
                    sharding=(None,),
                    transpose=False,
                    kv_head_padding=False,
                ),
                f"{prefix}.self_attn.k_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.k_proj.bias",
                    sharding=(None,),
                    transpose=False,
                    kv_head_padding=use_k_eq_v,
                ),
                f"{prefix}.self_attn.o_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.o_proj.bias",
                    sharding=(None,),
                    transpose=False,
                ),
            }
            if not use_k_eq_v:
                bias_mappings[f"{prefix}.self_attn.v_proj.bias"] = WeightMapping(
                    target_path=f"{target_prefix}.self_attn.v_proj.bias",
                    sharding=(None,),
                    transpose=False,
                    kv_head_padding=use_k_eq_v,
                )
            mappings.update(bias_mappings)

        return mappings

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: Any,
        logits_metadata: LogitsMetadata,
    ):
        kv_pool = memory_pools.token_to_kv_pool
        hidden_states, aux_hidden_states, layers_kv_fused, layers_callback_flag, layers_topk_ids = (
            self.model(forward_batch, kv_pool)
        )
        if not getattr(self.config, "tie_word_embeddings", True):
            output = self.logits_processor(
                hidden_states, self.lm_head, logits_metadata, aux_hidden_states=aux_hidden_states
            )
        else:
            output = self.logits_processor(
                hidden_states,
                self.model.embed_tokens,
                logits_metadata,
                aux_hidden_states=aux_hidden_states,
            )

        return output, {"token_to_kv_pool": layers_kv_fused}, layers_callback_flag, layers_topk_ids


class Gemma4ForConditionalGeneration(Gemma4ForCausalLM):
    pass


EntryClass = [Gemma4ForCausalLM, Gemma4ForConditionalGeneration]
