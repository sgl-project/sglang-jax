import logging

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import AttentionArch, ModelConfig
from sgl_jax.srt.eplb.expert_location import ExpertLocationMetadata
from sgl_jax.srt.layers.attention.mla import MLAAttention
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
from sgl_jax.srt.layers.gate import GateLogit, TopK
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import EPMoE, create_moe_weights_mapping
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


class KimiLinearConfig(PretrainedConfig):
    # https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base/blob/main/configuration_kimi.py
    model_type = "kimi_linear"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        model_type="kimi_linear",
        vocab_size=163840,
        hidden_size=4096,
        head_dim=None,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_theta=10000.0,
        rope_scaling=None,
        tie_word_embeddings=False,
        moe_intermediate_size: int | None = None,
        moe_renormalize: bool = True,
        moe_router_activation_func: str = "sigmoid",
        num_experts: int | None = None,
        num_experts_per_token: int | None = None,
        num_shared_experts: int = 0,
        routed_scaling_factor: float = 1.0,
        first_k_dense_replace: int = 0,
        moe_layer_freq: int = 1,
        use_grouped_topk: bool = True,
        num_expert_group: int = 1,
        topk_group: int = 1,
        q_lora_rank: int | None = None,
        kv_lora_rank: int | None = None,
        qk_nope_head_dim: int | None = None,
        qk_rope_head_dim: int | None = None,
        v_head_dim: int | None = None,
        mla_use_nope: bool | None = False,
        num_nextn_predict_layers: int = 0,
        linear_attn_config: dict | None = None,
        **kwargs,
    ):
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mla_use_nope = mla_use_nope

        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.moe_renormalize = moe_renormalize
        self.num_shared_experts = num_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.moe_router_activation_func = moe_router_activation_func
        assert self.moe_router_activation_func in ("softmax", "sigmoid")
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.num_nextn_predict_layers = num_nextn_predict_layers

        if linear_attn_config is not None:
            assert linear_attn_config["kda_layers"] is not None
            assert linear_attn_config["full_attn_layers"] is not None
        self.linear_attn_config = linear_attn_config

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def is_mla(self):
        return (
            self.q_lora_rank is not None
            or self.kv_lora_rank is not None
            or self.qk_nope_head_dim is not None
            or self.qk_rope_head_dim is not None
            or self.v_head_dim is not None
            or self.mla_use_nope is True
        )

    @property
    def is_moe(self):
        return self.num_experts is not None

    @property
    def is_linear_attn(self) -> bool:
        return not (
            self.linear_attn_config is None
            or (
                isinstance(self.linear_attn_config, dict)
                and self.linear_attn_config["kda_layers"] is not None
                and len(self.linear_attn_config["kda_layers"]) == 0
            )
        )

    def is_kda_layer(self, layer_idx: int):
        return (
            self.linear_attn_config is not None
            and (layer_idx + 1) in self.linear_attn_config["kda_layers"]
        )


class KimiMLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        super().__init__()
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

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        gate, _ = self.gate_proj(hidden_states)
        up, _ = self.up_proj(hidden_states)
        output, _ = self.down_proj(jax.nn.silu(gate) * up)
        return output


class KimiMoE(nnx.Module):
    shared_experts: KimiMLP | None
    moe_gate: GateLogit
    topk: TopK

    def __init__(
        self,
        layer_id: int,
        config: KimiLinearConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        super().__init__()
        self.moe_layer = config.is_moe
        # Gate
        # https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base/blob/main/modeling_kimi.py#L532
        self.moe_gate = GateLogit(
            input_size=config.hidden_size,
            num_experts=config.num_experts,
            enable_expert_bias=True,  # 这里怎么填
            weight_dtype=dtype,
            score_func=config.moe_router_activation_func,
        )

        # TopK routing
        self.topk = TopK(
            topk=config.num_experts_per_token,
            renormalize=config.moe_renormalize,
            num_expert_group=config.num_expert_group,
            topk_group=config.topk_group,
            routed_scaling_factor=config.routed_scaling_factor,
            layer_idx=layer_id,
        )

        self.experts = EPMoE(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_token,
            intermediate_dim=config.moe_intermediate_size,
            mesh=mesh,
            weight_dtype=dtype,
            dtype=dtype,
            layer_id=layer_id,
            ep_size=config.ep_size,
        )

        if config.num_shared_experts > 0:
            self.shared_experts = KimiMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size * config.num_shared_experts,
                mesh=mesh,
                dtype=dtype,
            )
        else:
            self.shared_experts = None

    def __call__(self, hidden_states: jax.Array, dispatch_info=None) -> tuple[jax.Array, jax.Array]:
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)

        router_logits = self.moe_gate(hidden_states)
        correction_bias = self.moe_gate.bias.value if self.moe_gate.bias is not None else None
        topk_weights, topk_ids = self.topk(
            router_logits, correction_bias, dispatch_info=dispatch_info
        )
        hidden_states = self.experts(hidden_states, topk_weights, topk_ids)

        if self.shared_experts is not None:
            hidden_states = hidden_states + shared_output

        return hidden_states, topk_ids


class KimiMLAAttention(nnx.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        layer_id: int = 0,
    ) -> None:
        super().__init__()

        self.mesh = mesh
        self.dtype = dtype
        self.layer_id = layer_id

        self.self_attn = MLAAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            mesh=mesh,
            layer_id=layer_id,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            rope_interleave=True,
            dtype=dtype,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, jax.Array]:
        return self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )


class KimiDeltaAttention(nnx.Module):
    """Placeholder for KDA (linear attention). Not yet implemented."""

    def __init__(self, layer_idx: int, hidden_size: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, None]:
        raise NotImplementedError("KimiDeltaAttention (KDA) is not yet implemented")


class KimiDecoderLayer(nnx.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        mesh: jax.sharding.Mesh,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.is_kda = config.is_kda_layer(layer_idx)

        # Attention
        if self.is_kda:
            self.self_attn = KimiDeltaAttention(
                layer_idx=layer_idx,
                hidden_size=config.hidden_size,
            )
        else:
            self.self_attn = KimiMLAAttention(
                config=config,
                mesh=mesh,
                dtype=dtype,
                layer_id=layer_idx,
            )

        # FFN
        if (
            config.is_moe
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            self.block_sparse_moe = KimiMoE(
                layer_id=layer_idx,
                config=config,
                mesh=mesh,
                dtype=dtype,
            )
            self.mlp = self.block_sparse_moe
        else:
            self.mlp = KimiMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                mesh=mesh,
                dtype=dtype,
            )
        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=jnp.float32,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=jnp.float32,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: jax.Array | None = None,
        dispatch_info: ExpertLocationMetadata | None = None,
    ):
        # Pre-norm residual pattern
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        # Attention
        hidden_states, kv_fused = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )
        # Post-attention residual + norm
        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Fully Connected
        if isinstance(self.mlp, KimiMoE):
            hidden_states, topk_ids = self.mlp(hidden_states, dispatch_info=dispatch_info)
        else:
            hidden_states = self.mlp(hidden_states)
            topk_ids = None

        return hidden_states, residual, kv_fused, topk_ids


class KimiLinearModel(nnx.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.config = config
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
                KimiDecoderLayer(
                    config=config,
                    mesh=mesh,
                    layer_idx=i,
                    dtype=dtype,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=jnp.float32,
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, list, list]:
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
        return hidden_states, layers_kv_fused, layers_topk_ids


class KimiLinearForCausalLM(nnx.Module):
    @classmethod
    def patch_model_config(cls, config: ModelConfig) -> None:
        config.attention_arch = AttentionArch.MLA
        qk_nope = getattr(config.hf_text_config, "qk_nope_head_dim", 0)
        qk_rope = getattr(config.hf_text_config, "qk_rope_head_dim", 0)
        if qk_nope and qk_rope:
            config.head_dim = qk_nope + qk_rope

    def __init__(
        self, config: KimiLinearConfig, mesh: jax.sharding.Mesh, dtype: jnp.dtype = jnp.bfloat16
    ) -> None:
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        self.model = KimiLinearModel(
            config=config,
            mesh=mesh,
            dtype=dtype,
        )

        if not getattr(config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                dtype=dtype,
                param_dtype=dtype,
                kernel_axes=("tensor", None),
            )

        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=mesh)

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

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_weight_mappings()
        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("Weights loaded successfully!")

    def _create_weight_mappings(self) -> dict:
        mappings = {
            "model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.norm.weight": WeightMapping(
                target_path="model.norm.scale",
                sharding=(None,),
                transpose=False,
            ),
        }

        if not getattr(self.config, "tie_word_embeddings", False):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding",
                sharding=("tensor", None),
                transpose=False,
            )

        num_layers = self.config.num_hidden_layers
        first_k_dense_replace = self.config.first_k_dense_replace

        for layer_idx in range(num_layers):
            is_dense = layer_idx < first_k_dense_replace
            is_kda = self.config.is_kda_layer(layer_idx)
            layer_mappings = self._create_layer_mappings(
                layer_idx, is_dense=is_dense, is_kda=is_kda
            )
            mappings.update(layer_mappings)

        return mappings

    def _create_layer_mappings(self, layer_idx: int, *, is_dense: bool, is_kda: bool) -> dict:
        prefix = f"model.layers.{layer_idx}"
        target_prefix = f"model.layers.{layer_idx}"

        mappings = {
            # Layer norms (all layers)
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

        # --- Attention mappings ---
        if is_kda:
            # KDA layer weights are mapped after KDA module is implemented.
            pass
        else:
            # MLA layer — target goes through KimiMLAAttention wrapper:
            # self.self_attn (KimiMLAAttention) -> self.self_attn (MLAAttention)
            attn_target = f"{target_prefix}.self_attn.self_attn"
            mappings[f"{prefix}.self_attn.q_proj.weight"] = WeightMapping(
                target_path=f"{attn_target}.q_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            )
            mappings[f"{prefix}.self_attn.o_proj.weight"] = WeightMapping(
                target_path=f"{attn_target}.o_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            )
            mappings[f"{prefix}.self_attn.kv_a_proj_with_mqa.weight"] = WeightMapping(
                target_path=f"{attn_target}.kv_a_proj.weight",
                sharding=(None, None),
                transpose=True,
            )
            mappings[f"{prefix}.self_attn.kv_a_layernorm.weight"] = WeightMapping(
                target_path=f"{attn_target}.kv_a_layernorm.scale",
                sharding=(None,),
                transpose=False,
            )
            mappings[f"{prefix}.self_attn.kv_b_proj.weight"] = WeightMapping(
                target_path=f"{attn_target}.kv_b_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            )

        # --- FFN mappings ---
        if is_dense:
            # Dense MLP
            for proj_name, sharding in [
                ("gate_proj", (None, "tensor")),
                ("up_proj", (None, "tensor")),
                ("down_proj", ("tensor", None)),
            ]:
                mappings[f"{prefix}.mlp.{proj_name}.weight"] = WeightMapping(
                    target_path=f"{target_prefix}.mlp.{proj_name}.weight",
                    sharding=sharding,
                    transpose=True,
                )
        else:
            # MoE — gate/topk/experts/shared_experts are inside KimiMoE (block_sparse_moe)
            moe_target = f"{target_prefix}.block_sparse_moe"

            # Gate
            mappings[f"{prefix}.block_sparse_moe.gate.weight"] = WeightMapping(
                target_path=f"{moe_target}.moe_gate.kernel",
                sharding=(None, None),
                transpose=True,
            )
            mappings[f"{prefix}.block_sparse_moe.gate.e_score_correction_bias"] = WeightMapping(
                target_path=f"{moe_target}.moe_gate.bias",
                sharding=(None,),
            )

            # Expert weights
            num_logical_experts = self.config.num_experts

            from sgl_jax.srt.eplb.expert_location import (
                get_global_expert_location_metadata,
            )

            metadata = get_global_expert_location_metadata()
            phy_to_log = None
            if metadata is not None:
                physical_to_logical_map = np.array(jax.device_get(metadata.physical_to_logical_map))
                phy_to_log = physical_to_logical_map[layer_idx]

            moe_mappings = create_moe_weights_mapping(
                prefix=prefix,
                target_prefix=moe_target,
                num_experts=num_logical_experts,
                expert_type_names=("w1", "w3", "w2"),
                moe_backend="epmoe",
                moe_path="experts",
                physical_to_logical_map=phy_to_log,
            )
            mappings.update(moe_mappings)

            # Shared experts
            for proj_name, sharding in [
                ("gate_proj", (None, "tensor")),
                ("up_proj", (None, "tensor")),
                ("down_proj", ("tensor", None)),
            ]:
                mappings[f"{prefix}.block_sparse_moe.shared_experts.{proj_name}.weight"] = (
                    WeightMapping(
                        target_path=f"{moe_target}.shared_experts.{proj_name}.weight",
                        sharding=sharding,
                        transpose=True,
                    )
                )

        return mappings


EntryClass = [KimiLinearForCausalLM]
