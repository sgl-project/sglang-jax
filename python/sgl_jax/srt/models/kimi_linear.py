"""Kimi-Linear-48B-A3B-Instruct model implementation.

Hybrid architecture: MLA (Multi-Latent Attention) + KDA (Key-Delta Attention)
with MoE (Mixture of Experts) sparse FFN.

Reference: https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct
"""

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
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import EPMoE, GateLogit, TopK, create_moe_weights_mapping
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


class KimiMLP(nnx.Module):
    """SwiGLU MLP used for dense layers and shared experts."""

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

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        gate, _ = self.gate_proj(hidden_states)
        up, _ = self.up_proj(hidden_states)
        output, _ = self.down_proj(up * self.act_fn(gate))
        return output


class KimiDecoderLayer(nnx.Module):
    """Single decoder layer with MLA/KDA attention dispatch and dense/MoE FFN dispatch."""

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.is_kda = _is_kda_layer(config, layer_id)

        # --- Attention ---
        if self.is_kda:
            self.self_attn = None  # KDA not yet implemented
        else:
            rope_scaling = getattr(config, "rope_scaling", None)
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
                rope_theta=getattr(config, "rope_theta", 10000.0),
                rope_scaling=rope_scaling,
                rope_interleave=getattr(config, "rope_interleave", True),
                max_position_embeddings=getattr(config, "max_position_embeddings", 163840),
                dtype=dtype,
            )

        # --- FFN ---
        first_k_dense_replace = getattr(config, "first_k_dense_replace", 0)
        if layer_id < first_k_dense_replace:
            # Dense MLP (layer 0)
            self.is_moe_layer = False
            self.mlp = KimiMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                mesh=mesh,
                layer_id=layer_id,
                dtype=dtype,
            )
            self.moe_gate = None
        else:
            # MoE (layers 1-26)
            self.is_moe_layer = True
            self.mlp = None

            self.moe_gate = GateLogit(
                input_size=config.hidden_size,
                num_experts=config.num_experts,
                enable_expert_bias=True,
                score_func="sigmoid",
            )

            self.topk = TopK(
                topk=config.num_experts_per_token,
                renormalize=getattr(config, "moe_renormalize", False),
                num_expert_group=getattr(config, "num_expert_group", 1),
                topk_group=getattr(config, "topk_group", 1),
                routed_scaling_factor=getattr(config, "routed_scaling_factor", 1.0),
                layer_id=layer_id,
            )

            self.block_sparse_moe = EPMoE(
                hidden_size=config.hidden_size,
                num_experts=config.num_experts,
                num_experts_per_tok=config.num_experts_per_token,
                intermediate_dim=config.moe_intermediate_size,
                mesh=mesh,
                ep_size=config.ep_size,
                weight_dtype=dtype,
                dtype=dtype,
                layer_id=layer_id,
                quantization_config=getattr(config, "quantization_config", None),
            )

            num_shared_experts = getattr(config, "num_shared_experts", 0)
            if num_shared_experts > 0:
                shared_intermediate = (
                    getattr(
                        config,
                        "moe_shared_expert_intermediate_size",
                        config.moe_intermediate_size,
                    )
                    * num_shared_experts
                )
                self.shared_experts = KimiMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=shared_intermediate,
                    mesh=mesh,
                    layer_id=layer_id,
                    dtype=dtype,
                )
            else:
                self.shared_experts = None

        # --- Layer Norms ---
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=getattr(config, "rms_norm_eps", 1e-6),
            param_dtype=jnp.float32,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=getattr(config, "rms_norm_eps", 1e-6),
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
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None]:
        # Pre-norm residual pattern
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        # Attention
        if self.is_kda:
            raise NotImplementedError(f"KDA attention not yet implemented (layer {self.layer_id})")
        else:
            hidden_states, kv_fused = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                token_to_kv_pool=token_to_kv_pool,
            )

        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # FFN
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

            hidden_states = self.block_sparse_moe(hidden_states, topk_weights, topk_ids)

            if shared_output is not None:
                hidden_states = hidden_states + shared_output
        else:
            hidden_states = self.mlp(hidden_states)
            topk_ids = None

        return hidden_states, residual, kv_fused, topk_ids


class KimiLinearModel(nnx.Module):
    """Kimi-Linear transformer model (without lm_head)."""

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
            kernel_axes=(None, "tensor"),
            mesh=mesh,
        )

        self.layers = nnx.data(
            [
                KimiDecoderLayer(
                    config=config,
                    mesh=mesh,
                    layer_id=i,
                    dtype=dtype,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=getattr(config, "rms_norm_eps", 1e-6),
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
    """Kimi-Linear causal language model."""

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        self.model = KimiLinearModel(config, dtype=dtype, mesh=mesh)

        if not getattr(config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                dtype=dtype,
                param_dtype=dtype,
                kernel_axes=("tensor", None),
            )

        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=mesh)

    @classmethod
    def patch_model_config(cls, mc: ModelConfig) -> None:
        mc.attention_arch = AttentionArch.MLA
        qk_nope = getattr(mc.hf_text_config, "qk_nope_head_dim", 0)
        qk_rope = getattr(mc.hf_text_config, "qk_rope_head_dim", 0)
        if qk_nope and qk_rope:
            mc.head_dim = qk_nope + qk_rope

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_weight_mappings(model_config)
        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("Weights loaded successfully!")

    def _create_weight_mappings(self, model_config: ModelConfig) -> dict:
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
        first_k_dense_replace = getattr(self.config, "first_k_dense_replace", 0)

        for layer_idx in range(num_layers):
            is_dense = layer_idx < first_k_dense_replace
            is_kda = _is_kda_layer(self.config, layer_idx)
            layer_mappings = self._create_layer_mappings(
                layer_idx, is_dense=is_dense, is_kda=is_kda
            )
            mappings.update(layer_mappings)

        return mappings

    def _create_layer_mappings(self, layer_idx: int, *, is_dense: bool, is_kda: bool) -> dict:
        prefix = f"model.layers.{layer_idx}"
        target_prefix = f"model.layers.{layer_idx}"

        mappings = {
            # Layer norms
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

        # --- Attention-specific mappings ---
        if is_kda:
            # KDA layer weights are mapped after KDA module is implemented.
            pass
        else:
            # MLA layer
            mappings[f"{prefix}.self_attn.q_proj.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            )
            mappings[f"{prefix}.self_attn.o_proj.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.o_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            )
            mappings[f"{prefix}.self_attn.kv_a_proj_with_mqa.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.kv_a_proj.weight",
                sharding=(None, None),
                transpose=True,
            )
            mappings[f"{prefix}.self_attn.kv_a_layernorm.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.kv_a_layernorm.scale",
                sharding=(None,),
                transpose=False,
            )
            mappings[f"{prefix}.self_attn.kv_b_proj.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.kv_b_proj.weight",
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
            # MoE gate
            mappings[f"{prefix}.block_sparse_moe.gate.weight"] = WeightMapping(
                target_path=f"{target_prefix}.moe_gate.kernel",
                sharding=(None, None),
                transpose=True,
            )
            mappings[f"{prefix}.block_sparse_moe.gate.e_score_correction_bias"] = WeightMapping(
                target_path=f"{target_prefix}.moe_gate.bias",
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
                target_prefix=target_prefix,
                num_experts=num_logical_experts,
                expert_type_names=("w1", "w3", "w2"),
                moe_backend="epmoe",
                moe_path="block_sparse_moe",
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
                        target_path=f"{target_prefix}.shared_experts.{proj_name}.weight",
                        sharding=sharding,
                        transpose=True,
                    )
                )

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


def _is_kda_layer(config: PretrainedConfig, layer_idx: int) -> bool:
    """Check if a layer uses KDA (Key-Delta Attention).

    The config's ``linear_attn_config["kda_layers"]`` list is 1-indexed.
    """
    kda_layers = config.linear_attn_config["kda_layers"]
    return (layer_idx + 1) in kda_layers


EntryClass = [KimiLinearForCausalLM]
