import logging
from typing import Any

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig, MoEBackend
from sgl_jax.srt.eplb.expert_location import ExpertLocationMetadata
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, RotaryEmbedding
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import (
    EPMoE,
    FusedEPMoE,
    GateLogit,
    TopK,
    create_moe_weights_mapping,
)
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)

init_fn = nnx.initializers.uniform()


class BailingMoEAttention(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int,
        mesh: jax.sharding.Mesh,
        rope_theta: float = 10000,
        rope_scaling: dict[str, Any] | None = None,
        head_dim: int | None = None,
        rms_norm_eps: float = None,
        use_qk_norm: bool = True,
        rotary_dim: int = 0,
        layer_id: int = 0,
        attention_bias: bool = False,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        assert num_heads % num_kv_heads == 0

        self.head_dim = head_dim or hidden_size // num_heads
        self.q_head_num = num_heads
        self.kv_head_num = num_kv_heads

        self.q_size = num_heads * self.head_dim
        self.kv_size = num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.use_qk_norm = use_qk_norm

        if use_qk_norm:
            self.q_norm = RMSNorm(
                self.head_dim, epsilon=rms_norm_eps, param_dtype=dtype, scope_name="q_norm"
            )
            self.k_norm = RMSNorm(
                self.head_dim, epsilon=rms_norm_eps, param_dtype=dtype, scope_name="k_norm"
            )
        else:
            self.q_norm = None
            self.k_norm = None

        self.q_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="q_proj",
        )
        self.k_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_kv_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="k_proj",
        )
        self.v_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_kv_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="v_proj",
        )
        self.c_proj = LinearBase(
            input_size=num_heads * self.head_dim,
            output_size=hidden_size,
            use_bias=attention_bias,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="c_proj",
        )
        self.rotary_emb = RotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            dtype=dtype,
        )
        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> jax.Array:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        q = q.reshape(-1, self.q_head_num, self.head_dim)
        k = k.reshape(-1, self.kv_head_num, self.head_dim)
        v = v.reshape(-1, self.kv_head_num, self.head_dim)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k = self.rotary_emb(positions, q, k)
        attn_output, kv_fused = self.attn(
            q, k, v, forward_batch=forward_batch, token_to_kv_pool=token_to_kv_pool
        )

        output, _ = self.c_proj(attn_output)
        return output, kv_fused


class BailingMoEMLP(nnx.Module):
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

        self.act_fn = jax.nn.silu

    def __call__(self, hidden_states: jax.Array):
        a1, _ = self.gate_proj(hidden_states)
        a2, _ = self.up_proj(hidden_states)
        intermediate_parallel = a2 * self.act_fn(a1)
        output, _ = self.down_proj(intermediate_parallel)
        return output


class BailingMoEDecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 40960)
        self.head_dim = getattr(config, "head_dim", None)
        use_qk_norm = getattr(config, "use_qk_norm", False)
        if hasattr(config, "partial_rotary_factor"):
            rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        elif hasattr(config, "rotary_dim"):
            rotary_dim = config.rotary_dim
        else:
            rotary_dim = self.head_dim

        self.self_attn = BailingMoEAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=self.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            use_qk_norm=use_qk_norm,
            rotary_dim=rotary_dim,
            layer_id=layer_id,
            attention_bias=getattr(config, "attention_bias", False),
            dtype=dtype,
            mesh=mesh,
        )

        first_k_dense_replace = getattr(config, "first_k_dense_replace", 0)

        if layer_id < first_k_dense_replace:
            self.mlp = BailingMoEMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                layer_id=layer_id,
                dtype=dtype,
                mesh=mesh,
            )
            self.is_moe_layer = False
            self.moe_gate = None
        else:
            num_shared_experts = getattr(config, "num_shared_experts", 0)
            router_dtype = getattr(config, "router_dtype", None)
            if router_dtype is None:
                router_dtype = None
            elif router_dtype == "fp32":
                router_dtype = jnp.float32
            else:
                router_dtype = jnp.bfloat16
            self.moe_gate = GateLogit(
                input_size=config.hidden_size,
                num_experts=config.num_experts,
                enable_expert_bias=getattr(config, "moe_router_enable_expert_bias", False),
                weight_dtype=router_dtype,
                score_func=getattr(config, "score_function", "sigmoid"),
            )

            self.moe_backend = getattr(config, "moe_backend", MoEBackend.EPMOE)
            self.use_fused = self.moe_backend == MoEBackend.FUSED
            moe_shared_expert_intermediate_size = getattr(
                config,
                "moe_shared_expert_intermediate_size",
                config.moe_intermediate_size,
            )

            self.topk = TopK(
                topk=config.num_experts_per_tok,
                renormalize=config.norm_topk_prob,
                num_expert_group=config.n_group,
                topk_group=config.topk_group,
                routed_scaling_factor=config.routed_scaling_factor,
                layer_id=layer_id,
            )

            if self.use_fused:
                self.mlp = FusedEPMoE(
                    hidden_size=config.hidden_size,
                    num_experts=config.num_experts,
                    num_experts_per_tok=config.num_experts_per_tok,
                    intermediate_dim=config.moe_intermediate_size,
                    mesh=mesh,
                    ep_size=config.ep_size,
                    weight_dtype=dtype,
                    dtype=dtype,
                    layer_id=layer_id,
                    renormalize_topk_logits=config.norm_topk_prob,
                    routed_scaling_factor=config.routed_scaling_factor,
                    use_grouped_topk=config.n_group > 0,
                    num_groups=config.n_group,
                    top_k_groups=config.topk_group,
                    num_shared_experts=num_shared_experts,
                    moe_shared_expert_intermediate_size=moe_shared_expert_intermediate_size,
                    quantization_config=getattr(config, "quantization_config", None),
                )
            else:
                self.mlp = EPMoE(
                    hidden_size=config.hidden_size,
                    num_experts=config.num_experts,
                    num_experts_per_tok=config.num_experts_per_tok,
                    intermediate_dim=config.moe_intermediate_size,
                    mesh=mesh,
                    ep_size=config.ep_size,
                    weight_dtype=dtype,
                    dtype=dtype,
                    layer_id=layer_id,
                    quantization_config=getattr(config, "quantization_config", None),
                )

            if num_shared_experts > 0 and not self.use_fused:
                self.shared_experts = BailingMoEMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=moe_shared_expert_intermediate_size * num_shared_experts,
                    layer_id=layer_id,
                    dtype=dtype,
                    mesh=mesh,
                )
            else:
                self.shared_experts = None
            self.is_moe_layer = True

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="input_layernorm",
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="post_attention_layernorm",
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: jax.Array | None = None,
        dispatch_info: ExpertLocationMetadata | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states, kv_fused = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )
        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.is_moe_layer:
            if self.shared_experts is not None:
                shared_output = self.shared_experts(hidden_states)
            else:
                shared_output = None
            router_logits = self.moe_gate(hidden_states)

            correction_bias = self.moe_gate.bias.value if self.moe_gate.bias is not None else None
            topk_weights, topk_ids = self.topk(
                router_logits,
                correction_bias,
                dispatch_info=dispatch_info,
            )

            if self.use_fused:
                token_valid_mask = forward_batch.get_token_valid_mask(hidden_states.shape[0])
                topk_ids = jnp.where(token_valid_mask[:, None], topk_ids, -1)
                hidden_states = self.mlp(
                    hidden_states,
                    topk_weights,
                    topk_ids,
                )
            else:
                hidden_states = self.mlp(hidden_states, topk_weights, topk_ids)

            if shared_output is not None:
                hidden_states = hidden_states + shared_output
        else:
            hidden_states = self.mlp(hidden_states)
            topk_ids = None

        return hidden_states, residual, kv_fused, topk_ids


class BailingMoEModel(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )

        self.layers = nnx.data(
            [
                BailingMoEDecoderLayer(
                    config=config,
                    layer_id=i,
                    dtype=dtype,
                    mesh=mesh,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, param_dtype=dtype, scope_name="norm"
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> jax.Array:
        hidden_states = self.embed_tokens(forward_batch.input_ids)
        residual = None
        layers_kv_fused = []
        layers_topk_ids = []
        for layer in self.layers:
            hidden_states, residual, kv_fused, topk_ids = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
                dispatch_info=forward_batch.expert_location_metadata,
            )
            layers_kv_fused.append(kv_fused)
            layers_topk_ids.append(topk_ids)

        if residual is not None:
            hidden_states += residual

        hidden_states = self.norm(hidden_states)
        # jax.debug.print("hidden_states: {hidden_states}", hidden_states=hidden_states)
        return hidden_states, layers_kv_fused, layers_topk_ids


class BailingMoEForCausalLM(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        self.model = BailingMoEModel(config, dtype=self.dtype, mesh=mesh)
        if not getattr(self.config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=("tensor", None),
            )
        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=self.mesh)

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_bailing_moe_weight_mappings(model_config)
        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("Weights loaded successfully!")

    def _create_bailing_moe_weight_mappings(self, model_config: ModelConfig) -> dict:
        mappings = {
            "model.word_embeddings.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.norm.weight": WeightMapping(
                target_path="model.norm.scale", sharding=(None,), transpose=False
            ),
        }

        if not getattr(self.config, "tie_word_embeddings", False):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding", sharding=("tensor", None), transpose=False
            )

        num_layers = self.config.num_hidden_layers
        first_k_dense_replace = self.config.first_k_dense_replace

        quant_config = getattr(model_config, "quantization_config", None)
        is_static_quant = quant_config is not None and quant_config.is_static_checkpoint

        for layer_idx in range(num_layers):
            layer_mappings = self._create_moe_layer_mappings(
                layer_idx, layer_idx < first_k_dense_replace, is_static_quant=is_static_quant
            )
            mappings.update(layer_mappings)

        return mappings

    def _create_moe_layer_mappings(
        self, layer_idx: int, is_mlp_layer: bool, is_static_quant: bool = False
    ) -> dict:
        prefix = f"model.layers.{layer_idx}"
        target_prefix = f"model.layers.{layer_idx}"

        mappings = {
            f"{prefix}.input_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.input_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.post_attention_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.post_attention_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
        }

        if is_static_quant:
            # QKV
            mappings[f"{prefix}.attention.query_key_value.weight"] = WeightMapping(
                target_path=[
                    f"{target_prefix}.self_attn.q_proj.weight_q",
                    f"{target_prefix}.self_attn.k_proj.weight_q",
                    f"{target_prefix}.self_attn.v_proj.weight_q",
                ],
                sharding=("tensor", None),
                transpose=False,
                kv_head_padding=True,
            )
            mappings[f"{prefix}.attention.query_key_value.weight_scale"] = WeightMapping(
                target_path=[
                    f"{target_prefix}.self_attn.q_proj.weight_scale",
                    f"{target_prefix}.self_attn.k_proj.weight_scale",
                    f"{target_prefix}.self_attn.v_proj.weight_scale",
                ],
                sharding=("tensor", None),
                transpose=False,
                kv_head_padding=True,
            )

            # Dense
            mappings[f"{prefix}.attention.dense.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.c_proj.weight_q",
                sharding=(None, "tensor"),
                transpose=False,
            )
            mappings[f"{prefix}.attention.dense.weight_scale"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.c_proj.weight_scale",
                sharding=(None, None),
                transpose=False,
            )
        else:
            mappings[f"{prefix}.attention.query_key_value.weight"] = WeightMapping(
                target_path=[
                    f"{target_prefix}.self_attn.q_proj.weight",
                    f"{target_prefix}.self_attn.k_proj.weight",
                    f"{target_prefix}.self_attn.v_proj.weight",
                ],
                sharding=(None, "tensor"),
                transpose=True,
                kv_head_padding=True,
            )
            mappings[f"{prefix}.attention.dense.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.c_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            )

        # QK Norm
        if getattr(self.config, "use_qk_norm", True):
            mappings[f"{prefix}.attention.query_layernorm.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_norm.scale", sharding=(None,)
            )
            mappings[f"{prefix}.attention.key_layernorm.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.k_norm.scale", sharding=(None,)
            )

        if is_mlp_layer:

            def add_mlp_mapping(hf_name, target_name, sharding_std):
                full_hf_key = f"{prefix}.mlp.{hf_name}.weight"
                if is_static_quant:
                    sharding_quant = (
                        (sharding_std[1], sharding_std[0])
                        if len(sharding_std) == 2
                        else sharding_std
                    )

                    mappings[full_hf_key] = WeightMapping(
                        target_path=f"{target_prefix}.mlp.{target_name}.weight_q",
                        sharding=sharding_quant,
                        transpose=False,
                    )

                    scale_key = f"{prefix}.mlp.{hf_name}.weight_scale"
                    scale_sharding = (sharding_quant[0],)
                    if target_name == "down_proj":
                        scale_sharding = (None,)

                    mappings[scale_key] = WeightMapping(
                        target_path=f"{target_prefix}.mlp.{target_name}.weight_scale",
                        sharding=scale_sharding,
                        transpose=False,
                    )
                else:
                    mappings[full_hf_key] = WeightMapping(
                        target_path=f"{target_prefix}.mlp.{target_name}.weight",
                        sharding=sharding_std,
                        transpose=True,
                    )

            add_mlp_mapping("gate_proj", "gate_proj", (None, "tensor"))
            add_mlp_mapping("up_proj", "up_proj", (None, "tensor"))
            add_mlp_mapping("down_proj", "down_proj", ("tensor", None))

        else:
            mappings[f"{prefix}.mlp.gate.weight"] = WeightMapping(
                target_path=f"{target_prefix}.moe_gate.kernel",
                sharding=(None, None),
                transpose=True,
            )
            if getattr(self.config, "moe_router_enable_expert_bias", False):
                mappings[f"{prefix}.mlp.gate.expert_bias"] = WeightMapping(
                    target_path=f"{target_prefix}.moe_gate.bias", sharding=(None,)
                )

            num_logical_experts = getattr(self.config, "num_experts", 256)
            moe_backend = getattr(self.config, "moe_backend", "epmoe")
            use_fused = moe_backend == "fused"

            BLOCK_SIZE = 256
            hidden_size = self.config.hidden_size
            inter_size = getattr(self.config, "moe_intermediate_size", 2048)

            # Get physical to logical mapping for redundant experts
            from sgl_jax.srt.eplb.expert_location import (
                get_global_expert_location_metadata,
            )

            metadata = get_global_expert_location_metadata()
            phy_to_log = None
            num_physical_experts = num_logical_experts
            if metadata is not None:
                num_physical_experts = metadata.num_physical_experts
                physical_to_logical_map = np.array(jax.device_get(metadata.physical_to_logical_map))
                phy_to_log = physical_to_logical_map[layer_idx]
                sample = phy_to_log[: min(10, phy_to_log.shape[0])].tolist()
                logger.info(
                    "Layer %s: logical=%s, physical=%s, redundancy=%.2fx",
                    layer_idx,
                    num_logical_experts,
                    num_physical_experts,
                    num_physical_experts / num_logical_experts,
                )
                logger.info(
                    "Layer %s EPLB map: size=%s min=%s max=%s sample=%s",
                    layer_idx,
                    phy_to_log.shape[0],
                    int(phy_to_log.min()),
                    int(phy_to_log.max()),
                    sample,
                )

            moe_mappings = create_moe_weights_mapping(
                prefix=prefix,
                target_prefix=target_prefix,
                num_experts=num_logical_experts,
                expert_type_names=("gate_proj", "up_proj", "down_proj"),
                moe_backend=moe_backend,
                physical_to_logical_map=phy_to_log,
            )

            if is_static_quant:
                new_moe_mappings = {}
                for key, mapping in moe_mappings.items():
                    target_param = mapping.target_path[0]
                    src_paths = mapping.target_path[1:]

                    new_moe_mappings[key] = WeightMapping(
                        target_path=[target_param] + src_paths,
                        sharding=mapping.sharding,
                        transpose=mapping.transpose,
                        concat_axis=mapping.concat_axis,
                        physical_to_logical_map=mapping.physical_to_logical_map,
                    )

                    scale_key = key + "_scale"
                    target_scale_param = target_param + "_scale"
                    scale_src_paths = [p.replace(".weight", ".weight_scale") for p in src_paths]

                    is_w2 = target_param.endswith("w2") or target_param.endswith("wo")
                    out_dim = hidden_size if is_w2 else inter_size

                    if use_fused:
                        in_dim = inter_size if is_w2 else hidden_size
                        num_blocks = in_dim // BLOCK_SIZE
                        # Use physical experts count for reshape (after redundant expert cloning)
                        scale_reshape = (num_physical_experts, 1, 1, out_dim)
                        logger.info("scale_reshape: %s", scale_reshape)
                        scale_repeat = (1, num_blocks)

                        scale_sharding = None
                        if mapping.sharding:
                            scale_sharding = (
                                mapping.sharding[0],
                                mapping.sharding[1],
                                None,
                                mapping.sharding[2],
                            )

                        new_moe_mappings[scale_key] = WeightMapping(
                            target_path=[target_scale_param] + scale_src_paths,
                            sharding=scale_sharding,
                            transpose=False,
                            reshape=scale_reshape,
                            repeat=scale_repeat,
                            concat_axis=mapping.concat_axis,
                            physical_to_logical_map=mapping.physical_to_logical_map,
                        )

                    else:
                        # Use physical experts count for reshape (after redundant expert cloning)
                        scale_reshape = (num_physical_experts, 1, 1, out_dim)
                        scale_repeat = None
                        scale_sharding = None
                        if mapping.sharding:
                            target_dim_sharding = None
                            if is_w2 and len(mapping.sharding) > 2:
                                target_dim_sharding = mapping.sharding[2]
                            elif not is_w2 and len(mapping.sharding) > 1:
                                target_dim_sharding = mapping.sharding[1]
                            scale_sharding = (mapping.sharding[0], target_dim_sharding, None)

                        new_moe_mappings[scale_key] = WeightMapping(
                            target_path=[target_scale_param] + scale_src_paths,
                            sharding=scale_sharding,
                            transpose=False,
                            reshape=scale_reshape,
                            repeat=scale_repeat,
                            concat_axis=mapping.concat_axis,
                            physical_to_logical_map=mapping.physical_to_logical_map,
                        )

                mappings.update(new_moe_mappings)
            else:
                mappings.update(moe_mappings)

            num_shared = getattr(self.config, "num_shared_experts", 0)
            if num_shared > 0:
                if use_fused:
                    shared_map = [
                        ("gate_proj", "w1_shared"),
                        ("up_proj", "w3_shared"),
                        ("down_proj", "w2_shared"),
                    ]

                    for hf_name, target_name in shared_map:
                        full_hf_key = f"{prefix}.mlp.shared_experts.{hf_name}.weight"
                        target_path = f"{target_prefix}.mlp.{target_name}"

                        if is_static_quant:
                            mappings[full_hf_key] = WeightMapping(
                                target_path=target_path,
                                sharding=(None, None),
                                transpose=True,
                            )
                            scale_key = f"{prefix}.mlp.shared_experts.{hf_name}.weight_scale"

                            is_w2 = "down_proj" in hf_name
                            se_inter = (
                                getattr(self.config, "moe_shared_expert_intermediate_size", 2048)
                                * num_shared
                            )
                            out_dim = hidden_size if is_w2 else se_inter

                            scale_reshape = (1, 1, out_dim)

                            mappings[scale_key] = WeightMapping(
                                target_path=target_path + "_scale",
                                sharding=(None, None, None),
                                reshape=scale_reshape,
                                transpose=False,
                            )
                        else:
                            mappings[full_hf_key] = WeightMapping(
                                target_path=target_path,
                                sharding=(None, None),
                                transpose=True,
                            )
                else:

                    def add_shared_expert_mapping(hf_name, target_name, sharding_std):
                        full_hf_key = f"{prefix}.mlp.shared_experts.{hf_name}.weight"

                        target_base = f"{target_prefix}.shared_experts.{target_name}"

                        if is_static_quant:
                            sharding_quant = (
                                (sharding_std[1], sharding_std[0])
                                if len(sharding_std) == 2
                                else sharding_std
                            )

                            mappings[full_hf_key] = WeightMapping(
                                target_path=f"{target_base}.weight_q",
                                sharding=sharding_quant,
                                transpose=False,
                            )

                            scale_key = f"{prefix}.mlp.shared_experts.{hf_name}.weight_scale"

                            scale_sharding = (sharding_quant[0],)
                            if target_name == "down_proj":
                                scale_sharding = (None,)

                            mappings[scale_key] = WeightMapping(
                                target_path=f"{target_base}.weight_scale",
                                sharding=scale_sharding,
                                transpose=False,
                            )
                        else:
                            mappings[full_hf_key] = WeightMapping(
                                target_path=f"{target_base}.weight",
                                sharding=sharding_std,
                                transpose=True,
                            )

                    add_shared_expert_mapping("gate_proj", "gate_proj", (None, "tensor"))
                    add_shared_expert_mapping("up_proj", "up_proj", (None, "tensor"))
                    add_shared_expert_mapping("down_proj", "down_proj", ("tensor", None))

        return mappings

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, layers_kv_fused, layers_topk_ids = self.model(
            forward_batch,
            token_to_kv_pool,
        )
        if not getattr(self.config, "tie_word_embeddings", False):
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)
        return output, layers_kv_fused, True, layers_topk_ids


class BailingMoeForCausalLM(BailingMoEForCausalLM):
    pass


class BailingMoeV2ForCausalLM(BailingMoEForCausalLM):
    pass


EntryClass = [BailingMoEForCausalLM, BailingMoeForCausalLM, BailingMoeV2ForCausalLM]
