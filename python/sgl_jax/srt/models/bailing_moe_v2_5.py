"""BailingMoeV2.5: hybrid MLA + Linear Attention model for TPU inference.

Implements the Ling-V2.5 architecture which alternates between linear attention
layers and MLA (softmax) attention layers based on layer_group_size, with MoE
feed-forward for most layers.
"""

import logging
from typing import Any

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig, MoEBackend
from sgl_jax.srt.eplb.expert_location import ExpertLocationMetadata
from sgl_jax.srt.layers.attention.fla.linear_attention_backend import (
    LinearAttentionBackend,
)
from sgl_jax.srt.layers.attention.mla import MLAAttention
from sgl_jax.srt.models.bailing_moe_v2_5_linear_attention import (
    BailingMoeV2_5LinearAttention,
)
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
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


def is_linear_layer(layer_idx, layer_group_size):
    """Determine if a layer uses linear attention (True) or MLA/softmax (False)."""
    if layer_idx is None:
        return False
    if isinstance(layer_group_size, list):
        return layer_group_size[layer_idx] == 1
    if layer_group_size > 0:
        return (layer_idx + 1) % layer_group_size != 0
    else:
        return False


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


class _DummyAttn:
    """Stub so that ``layer.self_attn.attn.sliding_window_size`` works for
    linear-attention layers (which have no KV cache / sliding window)."""
    sliding_window_size = None


class BailingMoeV2_5DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        is_linear: bool = True,
        dtype: jnp.dtype = jnp.bfloat16,
        linear_attn_backend: LinearAttentionBackend | None = None,
    ):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.is_linear = is_linear

        # --- Attention ---
        if is_linear:
            self.self_attn = BailingMoeV2_5LinearAttention(
                config=config,
                layer_idx=layer_id,
                mesh=mesh,
                backend=linear_attn_backend,
                dtype=dtype,
            )
            # Add dummy attn for hybrid SWA compatibility
            # Linear attention layers are treated as "full attention" (no sliding window)
            self.self_attn.attn = _DummyAttn()
        else:
            # MLA attention
            self.self_attn = MLAAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                q_lora_rank=getattr(config, "q_lora_rank", None),
                kv_lora_rank=config.kv_lora_rank,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                mesh=mesh,
                layer_id=layer_id,
                rope_theta=getattr(config, "rope_theta", 10000.0),
                rope_scaling=getattr(config, "rope_scaling", None),
                rope_interleave=True,
                max_position_embeddings=getattr(config, "max_position_embeddings", 4096),
                dtype=dtype,
            )

        # --- MLP ---
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
        recurrent_state: jax.Array | None = None,
        dispatch_info: ExpertLocationMetadata | None = None,
    ) -> tuple:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        # Attention
        new_recurrent_state = None
        kv_fused = None
        if self.is_linear:
            hidden_states, new_recurrent_state = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                recurrent_state=recurrent_state,
            )
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

        # MLP
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

            hidden_states = self.mlp(hidden_states, topk_weights, topk_ids)

            if shared_output is not None:
                hidden_states = hidden_states + shared_output
        else:
            hidden_states = self.mlp(hidden_states)
            topk_ids = None

        return hidden_states, residual, kv_fused, topk_ids, new_recurrent_state


class BailingMoeV2_5Model(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.mesh = mesh

        # Ensure partial_rotary_factor is set for linear attention layers
        if not hasattr(config, "partial_rotary_factor") and hasattr(config, "rotary_dim"):
            head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            config.partial_rotary_factor = config.rotary_dim / head_dim

        layer_group_size = getattr(config, "layer_group_size", 1)
        self.layer_group_size = layer_group_size

        # Determine attention type per layer
        self.attention_types = [
            0 if is_linear_layer(i, layer_group_size) else 1
            for i in range(config.num_hidden_layers)
        ]
        logger.info(
            "Attention types: %s (0=linear, 1=MLA)",
            self.attention_types,
        )

        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )

        # Create shared LinearAttentionBackend for all linear attention layers
        has_linear_layers = any(at == 0 for at in self.attention_types)
        if has_linear_layers:
            self.linear_attn_backend = LinearAttentionBackend(mesh=mesh)
        else:
            self.linear_attn_backend = None

        self.layers = nnx.data(
            [
                BailingMoeV2_5DecoderLayer(
                    config=config,
                    layer_id=i,
                    is_linear=(self.attention_types[i] == 0),
                    dtype=dtype,
                    mesh=mesh,
                    linear_attn_backend=self.linear_attn_backend,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, param_dtype=dtype, scope_name="norm"
        )

        # Count linear attention layers for recurrent state management
        self.linear_layer_indices = [
            i for i in range(config.num_hidden_layers) if self.attention_types[i] == 0
        ]
        self.num_linear_layers = len(self.linear_layer_indices)

        # Set on config so model_runner can create RecurrentStatePool
        config.num_linear_attention_layers = self.num_linear_layers

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        recurrent_state_pool=None,
    ) -> tuple:
        hidden_states = self.embed_tokens(forward_batch.input_ids)
        residual = None
        layers_kv_fused = []
        layers_topk_ids = []

        recurrent_id = 0
        for i, layer in enumerate(self.layers):
            # Get recurrent state for linear attention layers
            recurrent_state = None
            if self.attention_types[i] == 0:
                if recurrent_state_pool is not None:
                    recurrent_state = recurrent_state_pool.get_state(
                        recurrent_id, forward_batch.req_pool_indices
                    )
                else:
                    # Fallback: zero state
                    batch_size = forward_batch.batch_size
                    num_heads = self.config.num_attention_heads
                    head_dim = self.config.hidden_size // num_heads
                    recurrent_state = jnp.zeros(
                        (batch_size, num_heads, head_dim, head_dim), dtype=jnp.float32
                    )

            hidden_states, residual, kv_fused, topk_ids, new_recurrent_state = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
                recurrent_state=recurrent_state,
                dispatch_info=forward_batch.expert_location_metadata,
            )

            # Append kv_fused for ALL layers (framework expects one per layer).
            # Linear layers return None; MLA layers return updated KV cache.
            layers_kv_fused.append(kv_fused)

            if self.attention_types[i] == 0:
                # Linear layer: scatter updated state back into pool
                if recurrent_state_pool is not None and new_recurrent_state is not None:
                    recurrent_state_pool.set_state(
                        recurrent_id, forward_batch.req_pool_indices, new_recurrent_state
                    )
                recurrent_id += 1

            layers_topk_ids.append(topk_ids)

        if residual is not None:
            hidden_states += residual

        hidden_states = self.norm(hidden_states)

        # Return pool states for functional JIT update
        updated_pool_states = recurrent_state_pool.states if recurrent_state_pool is not None else []
        return hidden_states, layers_kv_fused, layers_topk_ids, updated_pool_states


class BailingMoeV2_5ForCausalLM(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        self.model = BailingMoeV2_5Model(config, dtype=self.dtype, mesh=mesh)
        # Expose linear_attn_backend at the top level for model_runner to pick up
        # via getattr(model, "linear_attn_backend", None)
        self.linear_attn_backend = self.model.linear_attn_backend
        if not getattr(self.config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=("tensor", None),
            )
        effective_vocab_size = getattr(config, "effective_vocab_size", config.vocab_size)
        self.logits_processor = LogitsProcessor(effective_vocab_size, mesh=self.mesh)

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
        layer_group_size = getattr(self.config, "layer_group_size", 1)

        quant_config = getattr(model_config, "quantization_config", None)
        is_static_quant = quant_config is not None and quant_config.is_static_checkpoint

        for layer_idx in range(num_layers):
            is_linear = is_linear_layer(layer_idx, layer_group_size)
            is_dense = layer_idx < first_k_dense_replace
            layer_mappings = self._create_layer_mappings(
                layer_idx,
                is_linear=is_linear,
                is_dense=is_dense,
                is_static_quant=is_static_quant,
            )
            mappings.update(layer_mappings)

        return mappings

    def _create_layer_mappings(
        self,
        layer_idx: int,
        is_linear: bool,
        is_dense: bool,
        is_static_quant: bool = False,
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

        # Attention weight mappings
        if is_linear:
            mappings.update(self._create_linear_attn_mappings(
                prefix, target_prefix, is_static_quant
            ))
        else:
            mappings.update(self._create_mla_attn_mappings(
                prefix, target_prefix, is_static_quant
            ))

        # MLP weight mappings
        if is_dense:
            mappings.update(self._create_dense_mlp_mappings(
                prefix, target_prefix, is_static_quant
            ))
        else:
            mappings.update(self._create_moe_mappings(
                prefix, target_prefix, layer_idx, is_static_quant
            ))

        return mappings

    def _create_linear_attn_mappings(
        self, prefix: str, target_prefix: str, is_static_quant: bool
    ) -> dict:
        mappings = {}

        if is_static_quant:
            # QKV fused
            mappings[f"{prefix}.attention.query_key_value.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.qkv_proj.weight_q",
                sharding=("tensor", None),
                transpose=False,
            )
            mappings[f"{prefix}.attention.query_key_value.weight_scale"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.qkv_proj.weight_scale",
                sharding=("tensor", None),
                transpose=False,
            )
            # Dense
            mappings[f"{prefix}.attention.dense.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.dense.weight_q",
                sharding=(None, "tensor"),
                transpose=False,
            )
            mappings[f"{prefix}.attention.dense.weight_scale"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.dense.weight_scale",
                sharding=(None, None),
                transpose=False,
            )
            # G proj
            mappings[f"{prefix}.attention.g_proj.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.g_proj.weight_q",
                sharding=("tensor", None),
                transpose=False,
            )
            mappings[f"{prefix}.attention.g_proj.weight_scale"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.g_proj.weight_scale",
                sharding=("tensor", None),
                transpose=False,
            )
        else:
            # QKV fused
            mappings[f"{prefix}.attention.query_key_value.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.qkv_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            )
            # Dense
            mappings[f"{prefix}.attention.dense.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.dense.weight",
                sharding=("tensor", None),
                transpose=True,
            )
            # G proj
            mappings[f"{prefix}.attention.g_proj.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.g_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            )

        # G norm weight (always float32, not quantized)
        mappings[f"{prefix}.attention.g_norm.weight"] = WeightMapping(
            target_path=f"{target_prefix}.self_attn.g_norm.weight",
            sharding=(None,),
            transpose=False,
        )

        # QK norm
        if getattr(self.config, "use_qk_norm", True):
            mappings[f"{prefix}.attention.query_layernorm.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_norm.scale",
                sharding=(None,),
            )
            mappings[f"{prefix}.attention.key_layernorm.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.k_norm.scale",
                sharding=(None,),
            )

        return mappings

    def _create_mla_attn_mappings(
        self, prefix: str, target_prefix: str, is_static_quant: bool
    ) -> dict:
        mappings = {}

        if is_static_quant:
            # Q path
            mappings[f"{prefix}.attention.q_a_proj.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_a_proj.weight_q",
                sharding=(None, None),
                transpose=False,
            )
            mappings[f"{prefix}.attention.q_a_proj.weight_scale"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_a_proj.weight_scale",
                sharding=(None,),
                transpose=False,
            )
            mappings[f"{prefix}.attention.q_b_proj.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_b_proj.weight_q",
                sharding=("tensor", None),
                transpose=False,
            )
            mappings[f"{prefix}.attention.q_b_proj.weight_scale"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_b_proj.weight_scale",
                sharding=("tensor", None),
                transpose=False,
            )
            # KV path
            mappings[f"{prefix}.attention.kv_a_proj_with_mqa.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.kv_a_proj.weight_q",
                sharding=(None, None),
                transpose=False,
            )
            mappings[f"{prefix}.attention.kv_a_proj_with_mqa.weight_scale"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.kv_a_proj.weight_scale",
                sharding=(None,),
                transpose=False,
            )
            mappings[f"{prefix}.attention.kv_b_proj.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.kv_b_proj.weight_q",
                sharding=("tensor", None),
                transpose=False,
            )
            mappings[f"{prefix}.attention.kv_b_proj.weight_scale"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.kv_b_proj.weight_scale",
                sharding=("tensor", None),
                transpose=False,
            )
            # Dense
            mappings[f"{prefix}.attention.dense.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.o_proj.weight_q",
                sharding=(None, "tensor"),
                transpose=False,
            )
            mappings[f"{prefix}.attention.dense.weight_scale"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.o_proj.weight_scale",
                sharding=(None, None),
                transpose=False,
            )
        else:
            # Q path
            mappings[f"{prefix}.attention.q_a_proj.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_a_proj.weight",
                sharding=(None, None),
                transpose=True,
            )
            mappings[f"{prefix}.attention.q_b_proj.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_b_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            )
            # KV path
            mappings[f"{prefix}.attention.kv_a_proj_with_mqa.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.kv_a_proj.weight",
                sharding=(None, None),
                transpose=True,
            )
            mappings[f"{prefix}.attention.kv_b_proj.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.kv_b_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            )
            # Dense / o_proj
            mappings[f"{prefix}.attention.dense.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.o_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            )

        # Layer norms (always float32)
        mappings[f"{prefix}.attention.q_a_layernorm.weight"] = WeightMapping(
            target_path=f"{target_prefix}.self_attn.q_a_layernorm.scale",
            sharding=(None,),
        )
        mappings[f"{prefix}.attention.kv_a_layernorm.weight"] = WeightMapping(
            target_path=f"{target_prefix}.self_attn.kv_a_layernorm.scale",
            sharding=(None,),
        )

        return mappings

    def _create_dense_mlp_mappings(
        self, prefix: str, target_prefix: str, is_static_quant: bool
    ) -> dict:
        mappings = {}

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

        return mappings

    def _create_moe_mappings(
        self, prefix: str, target_prefix: str, layer_idx: int, is_static_quant: bool
    ) -> dict:
        mappings = {}

        # Gate
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
            logger.info(
                "Layer %s: logical=%s, physical=%s, redundancy=%.2fx",
                layer_idx,
                num_logical_experts,
                num_physical_experts,
                num_physical_experts / num_logical_experts,
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
                    scale_reshape = (num_physical_experts, 1, 1, out_dim)
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

        # Shared experts
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
        recurrent_state_pool=None,
    ):
        hidden_states, layers_kv_fused, layers_topk_ids, updated_recurrent_states = self.model(
            forward_batch,
            token_to_kv_pool,
            recurrent_state_pool=recurrent_state_pool,
        )
        if not getattr(self.config, "tie_word_embeddings", False):
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)
        return output, layers_kv_fused, True, layers_topk_ids, updated_recurrent_states


class BailingMoeLinearForCausalLM(BailingMoeV2_5ForCausalLM):
    pass


EntryClass = [BailingMoeV2_5ForCausalLM, BailingMoeLinearForCausalLM]
