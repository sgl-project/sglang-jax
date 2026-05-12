import copy
import functools
import logging

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import AttentionArch, ModelConfig, MoEBackend
from sgl_jax.srt.eplb.expert_location import ExpertLocationMetadata
from sgl_jax.srt.layers.attention.fla.group_rmsnorm import GroupRMSNorm
from sgl_jax.srt.layers.embeddings import (
    Embed,
    ParallelLMHead,
    _deepseek_yarn_get_mscale,
    get_rope,
)
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
from sgl_jax.srt.layers.radix_lightning_attention import RadixLightningAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.deepseek_v3 import DeepseekV3Attention, DeepseekV3MLP
from sgl_jax.srt.utils.debug_utils import maybe_dump_jax_array
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


def is_linear_layer(layer_idx: int | None, layer_group_size: int) -> bool:
    if layer_idx is None:
        return False
    if layer_group_size <= 0:
        return False
    return (layer_idx + 1) % layer_group_size != 0


class BailingMoELinearAttention(nnx.Module):
    """BailingMoeV2.5 GLA block wired through RadixLightningAttention."""

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
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.mesh = mesh
        self.linear_silu = getattr(config, "use_linear_silu", getattr(config, "linear_silu", False))
        self.linear_rope = getattr(config, "linear_rope", True)

        inner_size = self.num_heads * self.head_dim
        qkv_bias = getattr(config, "use_bias", False) or getattr(config, "use_qkv_bias", False)
        self.qkv_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=3 * inner_size,
            use_bias=qkv_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="query_key_value",
        )
        self.g_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=inner_size,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="g_proj",
        )
        self.dense = LinearBase(
            input_size=inner_size,
            output_size=self.hidden_size,
            use_bias=getattr(config, "use_bias", False),
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="dense",
        )

        if getattr(config, "use_qk_norm", False):
            self.q_norm = RMSNorm(
                self.head_dim,
                epsilon=config.rms_norm_eps,
                param_dtype=dtype,
                scope_name="query_layernorm",
            )
            self.k_norm = RMSNorm(
                self.head_dim,
                epsilon=config.rms_norm_eps,
                param_dtype=dtype,
                scope_name="key_layernorm",
            )
        else:
            self.q_norm = None
            self.k_norm = None

        if hasattr(config, "rotary_dim"):
            rotary_dim = config.rotary_dim
        elif hasattr(config, "partial_rotary_factor"):
            rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        else:
            rotary_dim = self.head_dim
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=rotary_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            is_neox_style=True,
            rope_scaling=getattr(config, "rope_scaling", None),
            dtype=dtype,
        )

        self.g_norm = GroupRMSNorm(
            hidden_size=inner_size,
            num_groups=getattr(config, "group_norm_size", 1),
            epsilon=config.rms_norm_eps,
            scope_name="g_norm",
            param_dtype=dtype,
            kernel_axes=("tensor",),
            mesh=mesh,
        )
        self.attn = RadixLightningAttention(
            layer_id=layer_id,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        recurrent_state_pool,
    ) -> tuple[jax.Array, tuple]:
        _dump = functools.partial(
            maybe_dump_jax_array, component="gla_attn_detail", layer_id=self.layer_id
        )

        qkv, _ = self.qkv_proj(hidden_states)
        qkv = qkv.astype(jnp.float32)
        if self.linear_silu:
            qkv = jax.nn.silu(qkv)

        qkv = jax.lax.reshape(
            qkv,
            (hidden_states.shape[0], 3, self.num_heads, self.head_dim),
            out_sharding=NamedSharding(self.mesh, P("data", None, "tensor", None)),
        )
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        _dump(q, name="q_raw")
        _dump(k, name="k_raw")
        _dump(v, name="v_raw")

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
            _dump(q, name="q_after_norm")
            _dump(k, name="k_after_norm")

        if self.linear_rope:
            q, k = self.rotary_emb(positions, q, k)
            _dump(q, name="q_after_rope")
            _dump(k, name="k_after_rope")

        attn_output, pool_updates = self.attn(forward_batch, q, k, v, recurrent_state_pool)
        _dump(attn_output, name="attn_output")
        attn_output = attn_output.astype(hidden_states.dtype)

        gate, _ = self.g_proj(hidden_states)
        _dump(gate, name="gate")
        g_normed = self.g_norm(attn_output)
        _dump(g_normed, name="g_norm_output")
        attn_output = g_normed * jax.nn.sigmoid(gate)
        _dump(attn_output, name="gated_output")
        output, _ = self.dense(attn_output)
        _dump(output, name="final_output")
        return output, pool_updates


class BailingMoEGQAAttention(nnx.Module):
    """Minimal full-attention fallback for non-MLA BailingV2.5 configs."""

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.layer_id = layer_id
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.q_head_num = config.num_attention_heads
        self.kv_head_num = getattr(config, "num_key_value_heads", self.q_head_num)
        self.q_size = self.q_head_num * self.head_dim
        self.kv_size = self.kv_head_num * self.head_dim
        self.use_qk_norm = getattr(config, "use_qk_norm", False)

        qkv_bias = getattr(config, "use_bias", False) or getattr(config, "use_qkv_bias", False)
        self.qkv_proj = LinearBase(
            input_size=config.hidden_size,
            output_size=self.q_size + 2 * self.kv_size,
            use_bias=qkv_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="query_key_value",
        )
        self.dense = LinearBase(
            input_size=self.q_size,
            output_size=config.hidden_size,
            use_bias=getattr(config, "use_bias", False),
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="dense",
        )
        if self.use_qk_norm:
            self.q_norm = RMSNorm(
                self.head_dim,
                epsilon=config.rms_norm_eps,
                param_dtype=dtype,
                scope_name="query_layernorm",
            )
            self.k_norm = RMSNorm(
                self.head_dim,
                epsilon=config.rms_norm_eps,
                param_dtype=dtype,
                scope_name="key_layernorm",
            )
        else:
            self.q_norm = None
            self.k_norm = None

        if hasattr(config, "rotary_dim"):
            rotary_dim = config.rotary_dim
        elif hasattr(config, "partial_rotary_factor"):
            rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        else:
            rotary_dim = self.head_dim
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=rotary_dim,
            max_position=config.max_position_embeddings,
            base=getattr(config, "rope_theta", 600000),
            is_neox_style=True,
            rope_scaling=getattr(config, "rope_scaling", None),
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
    ) -> tuple[jax.Array, jax.Array]:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = jnp.split(qkv, [self.q_size, self.q_size + self.kv_size], axis=-1)
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
        output, _ = self.dense(attn_output)
        return output, kv_fused


class BailingMoELinearDecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.attention_type = getattr(
            config,
            "attention_type",
            0 if is_linear_layer(layer_id, getattr(config, "layer_group_size", 1)) else 1,
        )
        self.use_mla = getattr(config, "full_attention_type", "mla") == "mla"

        if self.attention_type == 0:
            self.self_attn = BailingMoELinearAttention(
                config=config,
                layer_id=layer_id,
                mesh=mesh,
                dtype=dtype,
            )
            self.is_linear_attention = True
        elif self.attention_type == 1:
            if self.use_mla:
                self.self_attn = DeepseekV3Attention(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    q_lora_rank=getattr(config, "q_lora_rank", None),
                    kv_lora_rank=getattr(config, "kv_lora_rank", 512),
                    qk_nope_head_dim=getattr(config, "qk_nope_head_dim", 128),
                    qk_rope_head_dim=getattr(config, "qk_rope_head_dim", 64),
                    v_head_dim=getattr(config, "v_head_dim", 128),
                    mesh=mesh,
                    layer_id=layer_id,
                    rope_theta=getattr(config, "rope_theta", 600000),
                    rope_scaling=getattr(config, "rope_scaling", None),
                    rope_interleave=getattr(config, "rope_interleave", True),
                    max_position_embeddings=getattr(config, "max_position_embeddings", 262144),
                    dtype=dtype,
                    use_absorbed=getattr(config, "use_absorbed_mla", True),
                )
                rope_scaling = getattr(config, "rope_scaling", None)
                if rope_scaling is not None and rope_scaling.get("mscale_all_dim", 0):
                    mscale = _deepseek_yarn_get_mscale(
                        rope_scaling["factor"], rope_scaling["mscale_all_dim"]
                    )
                    self.self_attn.attn_mqa.scaling *= mscale * mscale
                    self.self_attn.attn_mha.scaling *= mscale * mscale
            else:
                self.self_attn = BailingMoEGQAAttention(
                    config=config,
                    mesh=mesh,
                    layer_id=layer_id,
                    dtype=dtype,
                )
            self.is_linear_attention = False
        else:
            raise ValueError(f"Unsupported BailingMoeV2.5 attention type: {self.attention_type}")

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

        self._init_mlp(config, mesh, layer_id, dtype)

    def _init_mlp(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int,
        dtype: jnp.dtype,
    ) -> None:
        first_k_dense_replace = getattr(config, "first_k_dense_replace", 0)
        num_experts = getattr(config, "num_experts", 1)
        if num_experts == 1 or layer_id < first_k_dense_replace:
            self.mlp = DeepseekV3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                mesh=mesh,
                dtype=dtype,
            )
            self.is_moe_layer = False
            self.moe_gate = None
            self.shared_experts = None
            return

        router_dtype = getattr(config, "router_dtype", None)
        if router_dtype == "fp32" or router_dtype is None:
            router_dtype = jnp.float32
        else:
            router_dtype = jnp.bfloat16

        self.moe_gate = GateLogit(
            input_size=config.hidden_size,
            num_experts=num_experts,
            enable_expert_bias=getattr(config, "moe_router_enable_expert_bias", False),
            weight_dtype=router_dtype,
            score_func=getattr(config, "score_function", "sigmoid"),
        )
        self.topk = TopK(
            topk=config.num_experts_per_tok,
            renormalize=getattr(config, "norm_topk_prob", False),
            num_expert_group=getattr(config, "n_group", 0),
            topk_group=getattr(config, "topk_group", 0),
            routed_scaling_factor=getattr(config, "routed_scaling_factor", 1.0),
            layer_id=layer_id,
        )

        moe_backend = getattr(config, "moe_backend", MoEBackend.EPMOE)
        self.use_fused = moe_backend == MoEBackend.FUSED or moe_backend == "fused"
        if self.use_fused:
            self.mlp = FusedEPMoE(
                hidden_size=config.hidden_size,
                num_experts=num_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                intermediate_dim=config.moe_intermediate_size,
                mesh=mesh,
                ep_size=getattr(config, "ep_size", 1),
                weight_dtype=dtype,
                dtype=dtype,
                layer_id=layer_id,
                renormalize_topk_logits=getattr(config, "norm_topk_prob", False),
                routed_scaling_factor=getattr(config, "routed_scaling_factor", 1.0),
                use_grouped_topk=getattr(config, "n_group", 0) > 0,
                num_groups=getattr(config, "n_group", 0),
                top_k_groups=getattr(config, "topk_group", 0),
                num_shared_experts=getattr(config, "num_shared_experts", 0),
                moe_shared_expert_intermediate_size=getattr(
                    config, "moe_shared_expert_intermediate_size", config.moe_intermediate_size
                ),
                quantization_config=getattr(config, "quantization_config", None),
            )
        else:
            self.mlp = EPMoE(
                hidden_size=config.hidden_size,
                num_experts=num_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                intermediate_dim=config.moe_intermediate_size,
                mesh=mesh,
                ep_size=getattr(config, "ep_size", 1),
                weight_dtype=dtype,
                dtype=dtype,
                layer_id=layer_id,
                quantization_config=getattr(config, "quantization_config", None),
            )

        num_shared_experts = getattr(config, "num_shared_experts", 0)
        if num_shared_experts > 0 and not self.use_fused:
            shared_intermediate = (
                getattr(
                    config,
                    "moe_shared_expert_intermediate_size",
                    config.moe_intermediate_size,
                )
                * num_shared_experts
            )
            self.shared_experts = DeepseekV3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=shared_intermediate,
                mesh=mesh,
                dtype=dtype,
            )
        else:
            self.shared_experts = None
        self.is_moe_layer = True

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        recurrent_state_pool,
        residual: jax.Array | None = None,
        dispatch_info: ExpertLocationMetadata | None = None,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        if self.is_linear_attention:
            hidden_states, pool_update = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                recurrent_state_pool=recurrent_state_pool,
            )
            maybe_dump_jax_array(
                hidden_states,
                component="gla_attention",
                name="output",
                layer_id=self.layer_id,
                forward_mode=forward_batch.forward_mode,
            )
            kv_fused = None
        else:
            hidden_states, kv_fused = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                token_to_kv_pool=token_to_kv_pool,
            )
            maybe_dump_jax_array(
                hidden_states,
                component="mla_attention" if self.use_mla else "gqa_attention",
                name="output",
                layer_id=self.layer_id,
                forward_mode=forward_batch.forward_mode,
            )
            pool_update = None

        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.is_moe_layer:
            shared_output = self.shared_experts(hidden_states) if self.shared_experts else None
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

        maybe_dump_jax_array(
            hidden_states,
            component="mlp",
            name="output",
            layer_id=self.layer_id,
            forward_mode=forward_batch.forward_mode,
        )

        return (
            hidden_states,
            residual,
            kv_fused,
            pool_update,
            jax.sharding.reshard(topk_ids, P(None)),
        )


class BailingMoELinearModel(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.vocab_size = config.vocab_size
        self.layer_group_size = getattr(config, "layer_group_size", 1)
        self.decoder_attention_types = [
            0 if is_linear_layer(i, self.layer_group_size) else 1
            for i in range(config.num_hidden_layers)
        ]
        if self.layer_group_size <= 0 or config.num_hidden_layers % self.layer_group_size != 0:
            raise ValueError(
                f"num_hidden_layers={config.num_hidden_layers} must be divisible by "
                f"positive layer_group_size={self.layer_group_size}"
            )

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
                BailingMoELinearDecoderLayer(
                    config=self._layer_config(config, i),
                    mesh=mesh,
                    layer_id=i,
                    dtype=dtype,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="norm",
        )

    def _layer_config(self, config: PretrainedConfig, layer_id: int) -> PretrainedConfig:
        layer_config = copy.copy(config)
        layer_config.attention_type = self.decoder_attention_types[layer_id]
        return layer_config

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
    ):
        maybe_dump_jax_array(
            forward_batch.input_ids,
            component="embed",
            name="input_ids",
            forward_mode=forward_batch.forward_mode,
        )
        hidden_states = self.embed_tokens(forward_batch.input_ids)
        maybe_dump_jax_array(
            hidden_states,
            component="embed",
            name="hidden_states",
            forward_mode=forward_batch.forward_mode,
        )
        residual = None
        layers_kv_fused = []
        layers_topk_ids = []

        recurrent_state_pool = getattr(memory_pools, "recurrent_state_pool", None)
        recurrent_updates = None
        conv_updates = None
        if recurrent_state_pool is not None:
            recurrent_updates = list(recurrent_state_pool.recurrent_buffers)
            conv_updates = list(recurrent_state_pool.conv_buffers)

        for layer in self.layers:
            hidden_states, residual, kv_fused, pool_update, topk_ids = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                memory_pools.token_to_kv_pool,
                recurrent_state_pool,
                residual,
                dispatch_info=forward_batch.expert_location_metadata,
            )
            if kv_fused is not None:
                layers_kv_fused.append(kv_fused)
            if pool_update is not None:
                if recurrent_state_pool is None:
                    raise ValueError("Linear Bailing layer requires recurrent_state_pool")
                layer_idx = recurrent_state_pool.layers_mapping[layer.layer_id]
                recurrent_updates[layer_idx] = pool_update[0]
            layers_topk_ids.append(topk_ids)

        if residual is not None:
            hidden_states += residual
        hidden_states = self.norm(hidden_states)

        if recurrent_state_pool is None:
            pool_updates = layers_kv_fused
        else:
            pool_updates = {
                "token_to_kv_pool": layers_kv_fused,
                "recurrent_state_pool": (recurrent_updates, conv_updates),
            }
        return hidden_states, pool_updates, layers_topk_ids


class BailingMoeV2_5ForCausalLM(nnx.Module):
    @classmethod
    def patch_model_config(cls, mc: ModelConfig) -> None:
        cfg = mc.hf_text_config
        if getattr(cfg, "full_attention_type", "mla") != "mla":
            return
        mc.attention_arch = AttentionArch.MLA
        mc.head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        mc.kv_lora_rank = cfg.kv_lora_rank
        mc.qk_nope_head_dim = cfg.qk_nope_head_dim
        mc.qk_rope_head_dim = cfg.qk_rope_head_dim
        mc.v_head_dim = cfg.v_head_dim

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        self.model = BailingMoELinearModel(config, mesh=mesh, dtype=dtype)
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
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, pool_updates, layers_topk_ids = self.model(forward_batch, memory_pools)
        if not getattr(self.config, "tie_word_embeddings", False):
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)
        return output, pool_updates, True, layers_topk_ids

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        mappings = self._create_bailing_moe_linear_weight_mappings(model_config)
        loader.load_weights_from_safetensors(mappings)
        for layer in self.model.layers:
            if isinstance(layer.self_attn, DeepseekV3Attention):
                layer.self_attn.post_load_weights()
        logger.info("BailingMoeV2.5 weights loaded successfully!")

    def _create_bailing_moe_linear_weight_mappings(self, model_config: ModelConfig) -> dict:
        mappings = {
            "model.word_embeddings.weight": WeightMapping(
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

        decoder_attention_types = getattr(self.model, "decoder_attention_types", None)
        if decoder_attention_types is None:
            decoder_attention_types = [
                0 if is_linear_layer(i, getattr(self.config, "layer_group_size", 1)) else 1
                for i in range(self.config.num_hidden_layers)
            ]

        for layer_idx, attention_type in enumerate(decoder_attention_types):
            is_mlp_layer = layer_idx < getattr(self.config, "first_k_dense_replace", 0) or (
                getattr(self.config, "num_experts", 1) == 1
            )
            mappings.update(
                self._create_layer_mappings(
                    layer_idx=layer_idx,
                    attention_type=attention_type,
                    is_mlp_layer=is_mlp_layer,
                    model_config=model_config,
                )
            )
        return mappings

    def _create_layer_mappings(
        self,
        layer_idx: int,
        attention_type: int,
        is_mlp_layer: bool,
        model_config: ModelConfig,
    ) -> dict:
        prefix = f"model.layers.{layer_idx}"
        target = f"model.layers.{layer_idx}"
        mappings = {
            f"{prefix}.input_layernorm.weight": WeightMapping(
                target_path=f"{target}.input_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.post_attention_layernorm.weight": WeightMapping(
                target_path=f"{target}.post_attention_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
        }

        if attention_type == 0:
            self._add_linear_attention_mappings(mappings, prefix, target)
        elif getattr(self.config, "full_attention_type", "mla") == "mla":
            self._add_mla_attention_mappings(mappings, prefix, target)
        else:
            self._add_gqa_attention_mappings(mappings, prefix, target)

        if is_mlp_layer:
            self._add_dense_mlp_mappings(mappings, prefix, target)
        else:
            self._add_moe_mappings(mappings, prefix, target, layer_idx, model_config)
        return mappings

    def _add_linear_attention_mappings(self, mappings: dict, prefix: str, target: str) -> None:
        ap = f"{prefix}.attention"
        tp = f"{target}.self_attn"
        mappings[f"{ap}.query_key_value.weight"] = WeightMapping(
            target_path=f"{tp}.qkv_proj.weight",
            sharding=(None, "tensor"),
            transpose=True,
        )
        mappings[f"{ap}.g_proj.weight"] = WeightMapping(
            target_path=f"{tp}.g_proj.weight",
            sharding=(None, "tensor"),
            transpose=True,
        )
        mappings[f"{ap}.dense.weight"] = WeightMapping(
            target_path=f"{tp}.dense.weight",
            sharding=("tensor", None),
            transpose=True,
        )
        mappings[f"{ap}.g_norm.weight"] = WeightMapping(
            target_path=f"{tp}.g_norm.weight",
            sharding=("tensor",),
            transpose=False,
        )
        if getattr(self.config, "use_qk_norm", True):
            mappings[f"{ap}.query_layernorm.weight"] = WeightMapping(
                target_path=f"{tp}.q_norm.scale",
                sharding=(None,),
                transpose=False,
            )
            mappings[f"{ap}.key_layernorm.weight"] = WeightMapping(
                target_path=f"{tp}.k_norm.scale",
                sharding=(None,),
                transpose=False,
            )

    def _add_mla_attention_mappings(self, mappings: dict, prefix: str, target: str) -> None:
        ap = f"{prefix}.attention"
        tp = f"{target}.self_attn"
        if getattr(self.config, "q_lora_rank", None) is None:
            mappings[f"{ap}.q_proj.weight"] = WeightMapping(
                target_path=f"{tp}.q_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            )
        else:
            mappings[f"{ap}.q_a_proj.weight"] = WeightMapping(
                target_path=f"{tp}.q_a_proj.weight",
                sharding=(None, None),
                transpose=True,
            )
            mappings[f"{ap}.q_a_layernorm.weight"] = WeightMapping(
                target_path=f"{tp}.q_a_layernorm.scale",
                sharding=(None,),
                transpose=False,
            )
            mappings[f"{ap}.q_b_proj.weight"] = WeightMapping(
                target_path=f"{tp}.q_b_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            )
        mappings[f"{ap}.kv_a_proj_with_mqa.weight"] = WeightMapping(
            target_path=f"{tp}.kv_a_proj.weight",
            sharding=(None, None),
            transpose=True,
        )
        mappings[f"{ap}.kv_a_layernorm.weight"] = WeightMapping(
            target_path=f"{tp}.kv_a_layernorm.scale",
            sharding=(None,),
            transpose=False,
        )
        mappings[f"{ap}.kv_b_proj.weight"] = WeightMapping(
            target_path=f"{tp}.kv_b_proj.weight",
            sharding=(None, "tensor"),
            transpose=True,
        )
        mappings[f"{ap}.dense.weight"] = WeightMapping(
            target_path=f"{tp}.o_proj.weight",
            sharding=("tensor", None),
            transpose=True,
        )
        mappings[f"{ap}.o_proj.weight"] = WeightMapping(
            target_path=f"{tp}.o_proj.weight",
            sharding=("tensor", None),
            transpose=True,
        )

    def _add_gqa_attention_mappings(self, mappings: dict, prefix: str, target: str) -> None:
        ap = f"{prefix}.attention"
        tp = f"{target}.self_attn"
        mappings[f"{ap}.query_key_value.weight"] = WeightMapping(
            target_path=f"{tp}.qkv_proj.weight",
            sharding=(None, "tensor"),
            transpose=True,
        )
        mappings[f"{ap}.dense.weight"] = WeightMapping(
            target_path=f"{tp}.dense.weight",
            sharding=("tensor", None),
            transpose=True,
        )
        if getattr(self.config, "use_qk_norm", True):
            mappings[f"{ap}.query_layernorm.weight"] = WeightMapping(
                target_path=f"{tp}.q_norm.scale",
                sharding=(None,),
                transpose=False,
            )
            mappings[f"{ap}.key_layernorm.weight"] = WeightMapping(
                target_path=f"{tp}.k_norm.scale",
                sharding=(None,),
                transpose=False,
            )

    def _add_dense_mlp_mappings(self, mappings: dict, prefix: str, target: str) -> None:
        for proj, sharding in [
            ("gate_proj", (None, "tensor")),
            ("up_proj", (None, "tensor")),
            ("down_proj", ("tensor", None)),
        ]:
            mappings[f"{prefix}.mlp.{proj}.weight"] = WeightMapping(
                target_path=f"{target}.mlp.{proj}.weight",
                sharding=sharding,
                transpose=True,
            )

    def _add_moe_mappings(
        self,
        mappings: dict,
        prefix: str,
        target: str,
        layer_idx: int,
        model_config: ModelConfig,
    ) -> None:
        mappings[f"{prefix}.mlp.gate.weight"] = WeightMapping(
            target_path=f"{target}.moe_gate.kernel",
            sharding=(None, None),
            transpose=True,
        )
        if getattr(self.config, "moe_router_enable_expert_bias", False):
            mappings[f"{prefix}.mlp.gate.expert_bias"] = WeightMapping(
                target_path=f"{target}.moe_gate.bias",
                sharding=(None,),
                transpose=False,
            )

        from sgl_jax.srt.eplb.expert_location import get_global_expert_location_metadata

        phy_to_log = None
        metadata = get_global_expert_location_metadata()
        if metadata is not None:
            phy_to_log = np.array(jax.device_get(metadata.physical_to_logical_map))[layer_idx]

        moe_backend = getattr(self.config, "moe_backend", "epmoe")
        moe_mappings = create_moe_weights_mapping(
            prefix=prefix,
            target_prefix=target,
            num_experts=getattr(self.config, "num_experts", 256),
            expert_type_names=("gate_proj", "up_proj", "down_proj"),
            moe_backend=moe_backend,
            physical_to_logical_map=phy_to_log,
        )
        mappings.update(moe_mappings)

        num_shared = getattr(self.config, "num_shared_experts", 0)
        use_fused = moe_backend == "fused"
        if num_shared > 0 and not use_fused:
            for proj, sharding in [
                ("gate_proj", (None, "tensor")),
                ("up_proj", (None, "tensor")),
                ("down_proj", ("tensor", None)),
            ]:
                mappings[f"{prefix}.mlp.shared_experts.{proj}.weight"] = WeightMapping(
                    target_path=f"{target}.shared_experts.{proj}.weight",
                    sharding=sharding,
                    transpose=True,
                )


# Backward-compatible name for any local imports created before the file rename.
BailingMoeV2_5LinearAttention = BailingMoELinearAttention

EntryClass = [BailingMoeV2_5ForCausalLM]
