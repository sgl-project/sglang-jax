import logging
from typing import Any, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PretrainedConfig
from sgl_jax.srt.configs.quantization_config import QuantizationConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, get_rope
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import EPMoE, GateLogit, TopK
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.mimo_weight_utils import (
    mimo_apply_kv_head_padding,
    mimo_expand_linear_block_scale,
)
from sgl_jax.srt.utils.profiling_utils import named_scope
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


def _get_mimo_num_experts(config: PretrainedConfig) -> int:
    """MiMo checkpoints use `n_routed_experts` instead of `num_experts`."""
    num_experts = getattr(config, "num_experts", None)
    if num_experts is None:
        num_experts = getattr(config, "n_routed_experts", None)
    return 8 if num_experts is None else num_experts


# Custom MoE mapping function to handle quantization and correct path patterns
def create_moe_weights_mapping_quantized(
    prefix: str,
    target_prefix: str,
    num_experts: int,
    expert_type_names: tuple[str, str, str] = (
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
    moe_backend: str = "epmoe",
    moe_path: str = "mlp",
    source_expert_pattern: str = "experts.{i}",
    is_quantized: bool = False,
    hidden_size: int = 4096,
    intermediate_size: int = 4096,
    weight_block_size: int = 256,
) -> dict:
    """Generate a unified mapping dictionary for MoE layer expert weights with quantization support."""
    if moe_backend == "epmoe":
        expert_type_map = {
            expert_type_names[0]: "wi_0",
            expert_type_names[1]: "wi_1",
            expert_type_names[2]: "wo",
        }
    elif moe_backend == "fused":
        expert_type_map = {
            expert_type_names[0]: "w1",
            expert_type_names[1]: "w3",
            expert_type_names[2]: "w2",
        }
    else:
        raise ValueError(f"Unsupported MoE backend: {moe_backend}")

    mappings = {}
    for source_name, target_name in expert_type_map.items():
        # Target path for JAX model parameters
        target_path_base = f"{target_prefix}.{moe_path}.{target_name}"

        # Source weight paths
        expert_keys = [
            f"{prefix}.{moe_path}.{source_expert_pattern.format(i=i)}.{source_name}.weight"
            for i in range(num_experts)
        ]

        if moe_backend == "epmoe":
            # Weights are transposed from HF [n, k] to [k, n], stacked to [g, k, n].
            # wi_0/wi_1: [E, hidden_size, intermediate_dim] -> P("expert", None, "tensor")
            # wo:        [E, intermediate_dim, hidden_size] -> P("expert", "tensor", None)
            sharding = (
                ("expert", "tensor", None) if target_name == "wo" else ("expert", None, "tensor")
            )
            transpose = True
        elif moe_backend == "fused":
            sharding = (("data", "tensor"), None, None)
            transpose = True
        else:
            raise ValueError(f"Unsupported MoE backend: {moe_backend}")

        mappings[f"__MOE_EXPERTS__{target_path_base}"] = WeightMapping(
            target_path=[target_path_base] + expert_keys,
            sharding=sharding,
            transpose=transpose,
        )
        
        # Map Scales (if quantized)
        if is_quantized:
            target_scale_name = f"{target_name}_scale"
            target_path_scale = f"{target_prefix}.{moe_path}.{target_scale_name}"
            
            expert_scale_keys = [
                f"{prefix}.{moe_path}.{source_expert_pattern.format(i=i)}.{source_name}.weight_scale_inv"
                for i in range(num_experts)
            ]
            
            # Use the kernel's 4D scale layout (E, K/wsz, 1, N).
            # For static fused MoE checkpoints with 128-block quant, the upper-layer
            # adaptation pass may requantize loaded weights/scales to subc=256 later.
            if moe_backend == "fused":
                scale_sharding = (("data", "tensor"), None, None, None)
                if target_name == "w2":
                    scale_reshape = (num_experts, intermediate_size // weight_block_size, 1, hidden_size)
                else:
                    scale_reshape = (num_experts, hidden_size // weight_block_size, 1, intermediate_size)
            else:
                # EPMoE scale layout: (E, k_blocks, 1, out_size)
                # wi_0/wi_1: k_dim=hidden_size, out_size=intermediate_size → tensor-parallel on out_size
                # wo:        k_dim=intermediate_size, out_size=hidden_size → no tensor-parallel
                if target_name == "wo":
                    scale_sharding = ("expert", None, None, None)
                    k_blocks = intermediate_size // weight_block_size
                    out_size = hidden_size
                else:
                    scale_sharding = ("expert", None, None, "tensor")
                    k_blocks = hidden_size // weight_block_size
                    out_size = intermediate_size
                scale_reshape = (num_experts, k_blocks, 1, out_size)

            mappings[f"__MOE_EXPERTS__{target_path_scale}"] = WeightMapping(
                target_path=[target_path_scale] + expert_scale_keys,
                sharding=scale_sharding,
                transpose=False,
                reshape=scale_reshape,
            )

    return mappings

class MiMoV2MLP(nnx.Module):
    """MiMo V2 MLP layer with gate, up, and down projections."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
        gate_up_down_bias: bool | None = False,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        self.layer_id = layer_id

        self.gate_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=gate_up_down_bias,
            params_dtype=dtype,
            mesh=mesh,
        )

        self.up_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=gate_up_down_bias,
            params_dtype=dtype,
            mesh=mesh,
        )

        self.down_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            kernel_axes=("tensor", None),
            use_bias=gate_up_down_bias,
            params_dtype=dtype,
            mesh=mesh,
        )

        self.act_fn = jax.nn.silu

    def __call__(self, hidden_states: jax.Array):
        a1, _ = self.gate_proj(hidden_states)
        a2, _ = self.up_proj(hidden_states)
        intermediate_parallel = a2 * self.act_fn(a1)
        output, _ = self.down_proj(intermediate_parallel)
        return output

class MiMoV2Moe(nnx.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        mesh: jax.sharding.Mesh = None,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        # Prefer the explicitly threaded quant_config (decoder layer passes it in).
        # Some model configs may not have `config.quantization_config` attached when
        # fused MoE modules are constructed, which would silently disable MoE weight
        # quantization in FusedEPMoE/EPMoE.
        config_quant_config = getattr(config, "quantization_config", None)
        self.quantization_config = quant_config if quant_config is not None else config_quant_config
        self.quantized_dtype = (
            self.quantization_config.get_moe_weight_dtype()
            if self.quantization_config is not None
            else None
        )
        self.activation_quantized_dtype = (
            self.quantization_config.get_moe_activation_dtype()
            if self.quantization_config is not None
            else None
        )
        if quant_config is not None and config_quant_config is None:
            logger.info(
                "MiMoV2Moe layer=%s using explicit quant_config because config.quantization_config is missing",
                layer_id,
            )

        num_experts = _get_mimo_num_experts(config)
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 2)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", config.intermediate_size)

        self.moe_gate = GateLogit(
            input_size=config.hidden_size,
            num_experts=num_experts,
            weight_dtype=dtype,
            score_func=getattr(config, "scoring_func", "softmax"),
        )

        moe_backend = getattr(config, "moe_backend", "epmoe")
        if moe_backend == "fused":
            raise NotImplementedError(
                "MiMo-V2-Flash does not support moe_backend='fused'. "
                "Use '--moe-backend epmoe' instead."
            )

        self.topk_method = getattr(config, "topk_method", "greedy")
        if self.topk_method == "noaux_tc":
            self.correction_bias = nnx.Param(
                jnp.zeros(num_experts, dtype=jnp.float32)
            )
        else:
            self.correction_bias = None

        self.topk = TopK(
            topk=num_experts_per_tok,
            renormalize=getattr(config, "norm_topk_prob", True),
        )

        self.experts = EPMoE(
            hidden_size=config.hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            intermediate_dim=moe_intermediate_size,
            mesh=mesh,
            ep_size=config.ep_size,
            weight_dtype=dtype,
            dtype=dtype,
            layer_id=layer_id,
            quantization_config=self.quantization_config,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
    ):
        router_logits = self.moe_gate(hidden_states)
        correction_bias = self.correction_bias.value if self.correction_bias is not None else None
        topk_weights, topk_ids = self.topk(router_logits, correction_bias=correction_bias)
        mlp_output = self.experts(hidden_states, topk_weights, topk_ids)

        return mlp_output, topk_ids


class MiMoMoeAttention(nnx.Module):
    """MiMo MoE attention layer with QKV projections and output projection."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int,
        mesh: jax.sharding.Mesh,
        rope_theta: float = 1000000,
        rope_scaling: dict[str, Any] | None = None,
        head_dim: int | None = None,
        v_head_dim: int | None = None,
        sliding_window_size: int | None = None,
        attention_bias: bool = False,
        attention_sink_bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        partial_rotary_factor: float = 1.0,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
        qkv_bias: bool | None = True,
        o_bias: bool | None = False,
        v_scale: float | None = None,
    ):
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        assert (
            num_heads % num_kv_heads == 0
        ), "Please use other tp partition strategy for this model."
        self.head_dim = head_dim or hidden_size // num_heads
        self.q_head_num = num_heads
        self.k_head_num = num_kv_heads
        self.v_head_dim = v_head_dim if v_head_dim is not None else self.head_dim
        self.v_scale = v_scale
    
        self.q_size = num_heads * self.head_dim
        self.k_size = num_kv_heads * self.head_dim
        self.v_size = num_kv_heads * self.v_head_dim
        self.scaling = self.head_dim**-0.5

        logger.info(
            "[MiMoV2Flash] layer=%d q_heads=%d k_heads=%d head_dim=%d v_head_dim=%d v_size=%d",
            layer_id,
            self.q_head_num,
            self.k_head_num,
            self.head_dim,
            self.v_head_dim,
            self.v_size,
        )

        # Respect model config flags: fall back to attention_bias when qkv_bias not set
        qkv_bias = qkv_bias if qkv_bias is not None else False
        o_bias = o_bias if o_bias is not None else False

        self.q_proj = LinearBase(
            input_size=hidden_size,
            output_size=self.q_size,
            use_bias=qkv_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="q_proj",
            force_dequant=True,
        )
        self.k_proj = LinearBase(
            input_size=hidden_size,
            output_size=self.k_size,
            use_bias=qkv_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="k_proj",
        )
        self.v_proj = LinearBase(
            input_size=hidden_size,
            output_size=self.v_size,
            use_bias=qkv_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="v_proj",
        )
        self.o_proj = LinearBase(
            input_size=num_heads * self.v_head_dim,
            output_size=hidden_size,
            use_bias=o_bias,
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
            rope_scaling=rope_scaling,
            dtype=dtype,
            partial_rotary_factor=partial_rotary_factor,
        )

        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
            v_head_dim=self.v_head_dim,
            sliding_window_size=sliding_window_size, # -1 for no sliding window
            
        )

        self.attention_sink_bias = (
            nnx.Param(jnp.zeros(self.q_head_num, dtype=dtype))
            if attention_sink_bias
            else None
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

        q = q.reshape(-1, self.q_head_num, self.head_dim)
        k = k.reshape(-1, k.shape[-1] // self.head_dim, self.head_dim)
        v = v.reshape(-1, v.shape[-1] // self.v_head_dim, self.v_head_dim)

        # NOTE: config.attention_value_scale (0.707) exists in config.json but the
        # official HuggingFace reference code (modeling_mimo_v2_flash.py) does NOT
        # apply it at runtime.  Applying it here would incorrectly scale V down,
        # degrading output quality across all 48 layers.  Left as a no-op.

        q, k = self.rotary_emb(positions, q, k)

        attn_output, kv_fused = self.attn(
            q,
            k,
            v,
            forward_batch,
            token_to_kv_pool,
            attention_sink=self.attention_sink_bias,
        )

        # Some backends still return q_head_num * head_dim here and need an
        # explicit slice, while split-KV attention already returns q_head_num *
        # v_head_dim. Handle both layouts defensively.
        if self.head_dim != self.v_head_dim:
            expected_head_dim = self.q_head_num * self.head_dim
            expected_v_head_dim = self.q_head_num * self.v_head_dim
            if attn_output.shape[-1] == expected_head_dim:
                attn_output = attn_output.reshape(-1, self.q_head_num, self.head_dim)
                attn_output = attn_output[..., : self.v_head_dim]
                attn_output = attn_output.reshape(-1, expected_v_head_dim)
            elif attn_output.shape[-1] != expected_v_head_dim:
                raise ValueError(
                    "Unexpected attention output width for MiMo attention: "
                    f"got {attn_output.shape[-1]}, expected {expected_head_dim} "
                    f"or {expected_v_head_dim}"
                )

        output, _ = self.o_proj(attn_output)
        return output, kv_fused


class MiMoMoeDecoderLayer(nnx.Module):
    """MiMo MoE decoder layer with attention and MoE/MLP."""

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        quant_config: Optional[QuantizationConfig] = None,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        # head_dim = getattr(config, "head_dim", None)

        v_scale = getattr(config, "attention_value_scale", None)

        if self.is_swa_layer(config):
            self.self_attn = MiMoMoeAttention(
                hidden_size=config.hidden_size,
                num_heads=config.swa_num_attention_heads,
                num_kv_heads=config.swa_num_key_value_heads,
                max_position_embeddings=max_position_embeddings,
                rope_theta=getattr(config, "swa_rope_theta", rope_theta),
                rope_scaling=rope_scaling,
                head_dim=config.swa_head_dim,
                v_head_dim=getattr(config, "swa_v_head_dim", None),
                # swa_sliding_window_size may not exist; fall back to sliding_window_size
                sliding_window_size=getattr(
                    config,
                    "swa_sliding_window_size",
                    getattr(config, "sliding_window_size", None),
                ),
                attention_bias=config.attention_bias,
                attention_sink_bias=getattr(
                    config, "add_swa_attention_sink_bias", False
                ),
                layer_id=layer_id,
                dtype=dtype,
                qkv_bias=getattr(config, "qkv_bias", getattr(config, "attention_bias", False)),
                o_bias=getattr(config, "o_bias", False),
                partial_rotary_factor=getattr(config, "partial_rotary_factor", 1.0),
                mesh=mesh,
                quant_config=quant_config,
                v_scale=v_scale,
            )
        else:
            self.self_attn = MiMoMoeAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                head_dim=config.head_dim,
                v_head_dim=getattr(config, "v_head_dim", None),
                # Full attention layers use no sliding window (0 → None in RadixAttention)
                sliding_window_size=0,
                attention_bias=config.attention_bias,
                attention_sink_bias=getattr(
                    config, "add_full_attention_sink_bias", False
                ),
                layer_id=layer_id,
                dtype=dtype,
                qkv_bias=getattr(config, "qkv_bias", getattr(config, "attention_bias", False)),
                o_bias=getattr(config, "o_bias", False),
                partial_rotary_factor=getattr(config, "partial_rotary_factor", 1.0),
                mesh=mesh,
                quant_config=quant_config,
                v_scale=v_scale,
            )

        self.is_layer_sparse = self.is_moe_layer(config)

        if self.is_layer_sparse:
            self.mlp = MiMoV2Moe(
                config=config,
                quant_config=getattr(config, "quantization_config", None),
                layer_id=layer_id,
                mesh=mesh,
                dtype=dtype,
            )
        else:  
            self.mlp = MiMoV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                layer_id=layer_id,
                dtype=dtype,
                gate_up_down_bias=False,
                mesh=mesh,
                quant_config=getattr(config, "quantization_config", None),
            )

        # Optional shared expert path
        shared_sz = getattr(config, "shared_expert_intermediate_size", 0)
        if shared_sz and shared_sz > 0:
            self.shared_experts = MiMoV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=shared_sz,
                layer_id=layer_id,
                dtype=dtype,
                gate_up_down_bias=False,
                mesh=mesh,
                quant_config=getattr(config, "quantization_config", None),
            )
            self.shared_expert_gate = LinearBase(
                input_size=config.hidden_size,
                output_size=1,
                use_bias=False,
                kernel_axes=(None, None),
                params_dtype=dtype,
                mesh=mesh,
            )
        else:
            self.shared_experts = None
            self.shared_expert_gate = None

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.layernorm_epsilon,
            param_dtype=dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.layernorm_epsilon,
            param_dtype=dtype,
        )

    def __call__(
        self,
        positions: jax.Array,
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
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )

        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # optional shared expert output
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
            if self.shared_expert_gate is not None:
                gate, _ = self.shared_expert_gate(hidden_states)
                shared_output = jax.nn.sigmoid(gate) * shared_output
        else:
            shared_output = None

        if self.is_layer_sparse:
            mlp_output, topk_ids = self.mlp(hidden_states, forward_batch)
        else:
            mlp_output = self.mlp(hidden_states)
            topk_ids = None

        hidden_states = mlp_output if shared_output is None else (mlp_output + shared_output)

        return hidden_states, residual, kv_fused, topk_ids

    def is_moe_layer(self, config) -> bool:
        moe_freq = getattr(config, "moe_layer_freq", None)
        return (
            moe_freq is not None
            and 0 <= self.layer_id < len(moe_freq)
            and not isinstance(moe_freq, int)
            and moe_freq[self.layer_id]
        )

    def is_swa_layer(self, config) -> bool:
        # MiMo config uses hybrid_layer_pattern (list of 0/1 ints, 1=SWA)
        hybrid_layer_pattern = getattr(config, "hybrid_layer_pattern", None)
        if hybrid_layer_pattern is not None and 0 <= self.layer_id < len(hybrid_layer_pattern):
            return hybrid_layer_pattern[self.layer_id] == 1
        # fallback for string-format hybrid_pattern (e.g. ["full", "swa", ...])
        hybrid_pattern = getattr(config, "hybrid_pattern", None)
        if not hybrid_pattern:
            return False
        return 0 <= self.layer_id < len(hybrid_pattern) and hybrid_pattern[self.layer_id] == "swa"


class MiMoV2Model(nnx.Module):
    """MiMo MoE model with embedding, layers, and normalization."""

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
            kernel_axes=("tensor", None),
            param_dtype=dtype,
            mesh=mesh,
        )

        self.layers = nnx.data(
            [
                MiMoMoeDecoderLayer(
                    config=config,
                    layer_id=i,
                    dtype=dtype,
                    mesh=mesh,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=config.layernorm_epsilon,
            param_dtype=dtype,
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ):
        residual = None
        hidden_states = self.embed_tokens(forward_batch.input_ids)
        layers_kv_fused = []
        layers_topk_ids = []

        for i, layer in enumerate(self.layers):
            hidden_states, residual, kv_fused, topk_ids = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
            )
            layers_kv_fused.append(kv_fused)
            layers_topk_ids.append(topk_ids)

        if residual is not None:
            hidden_states += residual

        hidden_states = self.norm(hidden_states)

        return hidden_states, layers_kv_fused, layers_topk_ids


class MiMoV2FlashForCausalLM(nnx.Module):
    """MiMo MoE model for causal language modeling."""

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        logger.info("MiMoV2FlashForCausalLM config dtype: %s", self.dtype)
        self.model = MiMoV2Model(config, dtype=self.dtype, mesh=mesh)
        if not getattr(self.config, "tie_word_embeddings", True):
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=("tensor", None),
            )
        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=self.mesh)

    def _apply_kv_head_padding(self, weight, hf_key, model_config, sharding_size, head_dim):
        """Hook for WeightLoader: MiMo-specific KV head padding with per-layer heads."""
        return mimo_apply_kv_head_padding(weight, hf_key, model_config, sharding_size, head_dim)

    def _expand_linear_block_scale(self, scale, model_param, jax_path, model_config):
        """Hook for WeightLoader: MiMo-specific block scale expansion."""
        return mimo_expand_linear_block_scale(scale, model_param, jax_path, model_config)

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )

        weight_mappings = self._create_MiMo_moe_weight_mappings(model_config)

        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("MiMoV2Flash weights loaded successfully!")

    def _create_MiMo_moe_weight_mappings(self, model_config: ModelConfig) -> dict:
        mappings = {
            "model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.norm.weight": WeightMapping(
                target_path="model.norm.scale", sharding=(None,), transpose=False
            ),
        }

        if not getattr(self.config, "tie_word_embeddings", True):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding", sharding=("tensor", None), transpose=False
            )

        num_layers = self.config.num_hidden_layers

        for layer_idx in range(num_layers):
            layer_mappings = self._create_moe_layer_mappings(layer_idx, model_config)
            mappings.update(layer_mappings)

        return mappings

    def _create_moe_layer_mappings(self, layer_idx: int, model_config: ModelConfig) -> dict:
        prefix = f"model.layers.{layer_idx}"
        target_prefix = f"model.layers.{layer_idx}"
        
        quant_config = getattr(self.config, "quantization_config", None)
        is_quantized = quant_config is not None
        ignored_layers = None
        if is_quantized:
            if isinstance(quant_config, dict):
                ignored_layers = quant_config.get("ignored_layers", None)
            else:
                ignored_layers = getattr(quant_config, "ignored_layers", None)
        if is_quantized and isinstance(quant_config, dict):
            weight_block_size = quant_config.get("weight_block_size", None)
        elif is_quantized:
            weight_block_size = getattr(quant_config, "weight_block_size", None)
        else:
            weight_block_size = None
        has_blockwise_linear_scale = bool(weight_block_size is not None)

        def _col_linear_scale_sharding():
            # Blockwise scale must be replicated (P(None, None)) so the kernel can index
            # the full global scale using lax.axis_index-based global offsets.
            return (None, None) if has_blockwise_linear_scale else ("tensor",)

        def _row_linear_scale_sharding():
            # Same reasoning as _col_linear_scale_sharding: replicated for blockwise.
            return (None, None) if has_blockwise_linear_scale else (None,)

        def _ignore(layer_suffix: str) -> bool:
            if not ignored_layers:
                return False
            target = f"{prefix}.{layer_suffix}"
            return target in ignored_layers

        def _is_swa_layer() -> bool:
            hybrid_layer_pattern = getattr(self.config, "hybrid_layer_pattern", None)
            if hybrid_layer_pattern is not None and 0 <= layer_idx < len(hybrid_layer_pattern):
                return hybrid_layer_pattern[layer_idx] == 1
            hybrid_pattern = getattr(self.config, "hybrid_pattern", None)
            if not hybrid_pattern:
                return False
            return 0 <= layer_idx < len(hybrid_pattern) and hybrid_pattern[layer_idx] == "swa"

        mappings = {}
        is_swa_layer = _is_swa_layer()
        
        # QKV Projections
        # q_proj
        if is_quantized:
             mappings[f"{prefix}.self_attn.q_proj.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_proj.weight_q",
                sharding=("tensor", None),
                transpose=False, # Transpose removed for QuantizedLinear [out, in]
                head_dim_padding=True,
                kv_head_padding=False,
            )
             mappings[f"{prefix}.self_attn.q_proj.weight_scale_inv"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_proj.weight_scale",
                sharding=_col_linear_scale_sharding(),
                transpose=False,
            )
        else:
            mappings[f"{prefix}.self_attn.q_proj.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=False,
            )

        # k_proj
        if is_quantized:
             mappings[f"{prefix}.self_attn.k_proj.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.k_proj.weight_q",
                sharding=("tensor", None), # Split axis 0 (output)
                transpose=False,
                head_dim_padding=True,
                kv_head_padding=True,
            )
             mappings[f"{prefix}.self_attn.k_proj.weight_scale_inv"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.k_proj.weight_scale",
                sharding=_col_linear_scale_sharding(),
                transpose=False,
                kv_head_padding=True,
            )
        else:
            mappings[f"{prefix}.self_attn.k_proj.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.k_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=True,
            )

        # v_proj
        if is_quantized:
             mappings[f"{prefix}.self_attn.v_proj.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.v_proj.weight_q",
                sharding=("tensor", None), # Split axis 0
                transpose=False,
                head_dim_padding=False,
                kv_head_padding=True,
                # repeat removed
            )
             mappings[f"{prefix}.self_attn.v_proj.weight_scale_inv"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.v_proj.weight_scale",
                sharding=_col_linear_scale_sharding(),
                transpose=False,
                kv_head_padding=True,
            )
        else:
            mappings[f"{prefix}.self_attn.v_proj.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.v_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                # v_head_dim 可能与 head_dim 不同；对 v_proj 做 head_dim padding 会让权重/偏置长度
                # 与实际输出维度不一致（例如输出 512 而 bias 被 pad 到 1024），因此关闭。
                head_dim_padding=False,
                kv_head_padding=True,
            )

        # o_proj (Row Parallel: Input is sharded)
        ignore_o_proj = _ignore("self_attn.o_proj")
        if is_quantized and not ignore_o_proj:
            mappings[f"{prefix}.self_attn.o_proj.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.o_proj.weight_q",
                sharding=(None, "tensor"), # Split axis 1 (Input)
                transpose=False,
                head_dim_padding=True,
                kv_head_padding=False,
            )
            mappings[f"{prefix}.self_attn.o_proj.weight_scale_inv"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.o_proj.weight_scale",
                sharding=_row_linear_scale_sharding(),
                transpose=False,
            )
        else:
            mappings[f"{prefix}.self_attn.o_proj.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.o_proj.weight",
                sharding=("tensor", None),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=False,
            )
            
        # Layernorms (standard)
        mappings[f"{prefix}.input_layernorm.weight"] = WeightMapping(
            target_path=f"{target_prefix}.input_layernorm.scale",
            sharding=(None,),
            transpose=False,
        )
        mappings[f"{prefix}.post_attention_layernorm.weight"] = WeightMapping(
            target_path=f"{target_prefix}.post_attention_layernorm.scale",
            sharding=(None,),
            transpose=False,
        )

        # Bias mappings follow the actual bias flags (qkv_bias / o_bias) rather than attention_bias.
        if getattr(self.config, "qkv_bias", getattr(self.config, "attention_bias", False)):
            mappings[f"{prefix}.self_attn.q_proj.bias"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_proj.bias",
                sharding=("tensor",),  # Shard by TP
                transpose=False,
                head_dim_padding=True,
                kv_head_padding=False,
            )
            mappings[f"{prefix}.self_attn.k_proj.bias"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.k_proj.bias",
                sharding=("tensor",),  # Shard by TP
                transpose=False,
                head_dim_padding=True,
                kv_head_padding=True,
            )
            mappings[f"{prefix}.self_attn.v_proj.bias"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.v_proj.bias",
                sharding=("tensor",),  # Shard by TP
                transpose=False,
                head_dim_padding=False,
                kv_head_padding=True,  # Allow automatic replication for TP>KV
            )

        if getattr(self.config, "o_bias", False):
            mappings[f"{prefix}.self_attn.o_proj.bias"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.o_proj.bias",
                sharding=(None,),  # Output (hidden_size) is not sharded
                transpose=False,
            )

        has_attention_sink_bias = (
            is_swa_layer and getattr(self.config, "add_swa_attention_sink_bias", False)
        ) or (
            (not is_swa_layer) and getattr(self.config, "add_full_attention_sink_bias", False)
        )
        if has_attention_sink_bias:
            mappings[f"{prefix}.self_attn.attention_sink_bias"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.attention_sink_bias",
                sharding=(None,),
                transpose=False,
            )

        is_sparse = (
            hasattr(self.config, "moe_layer_freq")
            and 0 <= layer_idx < len(self.config.moe_layer_freq)
            and not isinstance(self.config.moe_layer_freq, int)
            and self.config.moe_layer_freq[layer_idx]
        )

        if is_sparse:
            # MoE mappings for expert layers
            # MoE gate is usually not quantized in the same way or handled by specific MoE logic
            mappings[f"{prefix}.mlp.gate.weight"] = WeightMapping(
                target_path=f"{target_prefix}.mlp.moe_gate.kernel",
                sharding=(None, None),
                transpose=True,
            )
            if getattr(self.config, "topk_method", "greedy") == "noaux_tc":
                mappings[f"{prefix}.mlp.gate.e_score_correction_bias"] = WeightMapping(
                    target_path=f"{target_prefix}.mlp.correction_bias",
                    sharding=(None,),
                    transpose=False,
                )

            num_experts = _get_mimo_num_experts(self.config)
            moe_backend = getattr(self.config, "moe_backend", "epmoe")
            moe_intermediate_size = getattr(self.config, "moe_intermediate_size", self.config.intermediate_size)
            moe_quant_block_size = getattr(model_config, "moe_quant_block_size", 256)

            moe_mappings = create_moe_weights_mapping_quantized(
                prefix=prefix,
                target_prefix=target_prefix,
                num_experts=num_experts,
                moe_backend=moe_backend,
                moe_path="mlp.experts",
                source_expert_pattern="{i}", 
                is_quantized=is_quantized,
                hidden_size=self.config.hidden_size,
                intermediate_size=moe_intermediate_size,
                weight_block_size=moe_quant_block_size,
            )
            mappings.update(moe_mappings)
        else:
            # Standard MLP mappings
            if is_quantized:
                # gate_proj
                mappings[f"{prefix}.mlp.gate_proj.weight"] = WeightMapping(
                    target_path=f"{target_prefix}.mlp.gate_proj.weight_q",
                    sharding=("tensor", None), # Split axis 0
                    transpose=False,
                )
                mappings[f"{prefix}.mlp.gate_proj.weight_scale_inv"] = WeightMapping(
                    target_path=f"{target_prefix}.mlp.gate_proj.weight_scale",
                    sharding=_col_linear_scale_sharding(),
                    transpose=False,
                )
                # up_proj
                mappings[f"{prefix}.mlp.up_proj.weight"] = WeightMapping(
                    target_path=f"{target_prefix}.mlp.up_proj.weight_q",
                    sharding=("tensor", None), # Split axis 0
                    transpose=False,
                )
                mappings[f"{prefix}.mlp.up_proj.weight_scale_inv"] = WeightMapping(
                    target_path=f"{target_prefix}.mlp.up_proj.weight_scale",
                    sharding=_col_linear_scale_sharding(),
                    transpose=False,
                )
                # down_proj
                mappings[f"{prefix}.mlp.down_proj.weight"] = WeightMapping(
                    target_path=f"{target_prefix}.mlp.down_proj.weight_q",
                    sharding=(None, "tensor"), # Split axis 1
                    transpose=False,
                )
                mappings[f"{prefix}.mlp.down_proj.weight_scale_inv"] = WeightMapping(
                    target_path=f"{target_prefix}.mlp.down_proj.weight_scale",
                    sharding=_row_linear_scale_sharding(),
                    transpose=False,
                )
            else:
                mappings[f"{prefix}.mlp.gate_proj.weight"] = WeightMapping(
                    target_path=f"{target_prefix}.mlp.gate_proj.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                )
                mappings[f"{prefix}.mlp.up_proj.weight"] = WeightMapping(
                    target_path=f"{target_prefix}.mlp.up_proj.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                )
                mappings[f"{prefix}.mlp.down_proj.weight"] = WeightMapping(
                    target_path=f"{target_prefix}.mlp.down_proj.weight",
                    sharding=("tensor", None),
                    transpose=True,
                )

        # Optional shared expert weight mapping (singular in source naming)
        if (
            getattr(self.config, "shared_expert_intermediate_size", 0)
            and getattr(self.config, "shared_expert_intermediate_size", 0) > 0
        ):
            if is_quantized:
                 # gate_proj
                mappings[f"{prefix}.mlp.shared_expert.gate_proj.weight"] = WeightMapping(
                    target_path=f"{target_prefix}.shared_experts.gate_proj.weight_q",
                    sharding=("tensor", None),
                    transpose=False,
                )
                mappings[f"{prefix}.mlp.shared_expert.gate_proj.weight_scale_inv"] = WeightMapping(
                    target_path=f"{target_prefix}.shared_experts.gate_proj.weight_scale",
                    sharding=_col_linear_scale_sharding(),
                    transpose=False,
                )
                # up_proj
                mappings[f"{prefix}.mlp.shared_expert.up_proj.weight"] = WeightMapping(
                    target_path=f"{target_prefix}.shared_experts.up_proj.weight_q",
                    sharding=("tensor", None),
                    transpose=False,
                )
                mappings[f"{prefix}.mlp.shared_expert.up_proj.weight_scale_inv"] = WeightMapping(
                    target_path=f"{target_prefix}.shared_experts.up_proj.weight_scale",
                    sharding=_col_linear_scale_sharding(),
                    transpose=False,
                )
                # down_proj
                mappings[f"{prefix}.mlp.shared_expert.down_proj.weight"] = WeightMapping(
                    target_path=f"{target_prefix}.shared_experts.down_proj.weight_q",
                    sharding=(None, "tensor"),
                    transpose=False,
                )
                mappings[f"{prefix}.mlp.shared_expert.down_proj.weight_scale_inv"] = WeightMapping(
                    target_path=f"{target_prefix}.shared_experts.down_proj.weight_scale",
                    sharding=_row_linear_scale_sharding(),
                    transpose=False,
                )
            else:
                mappings[f"{prefix}.mlp.shared_expert.gate_proj.weight"] = WeightMapping(
                    target_path=f"{target_prefix}.shared_experts.gate_proj.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                )
                mappings[f"{prefix}.mlp.shared_expert.up_proj.weight"] = WeightMapping(
                    target_path=f"{target_prefix}.shared_experts.up_proj.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                )
                mappings[f"{prefix}.mlp.shared_expert.down_proj.weight"] = WeightMapping(
                    target_path=f"{target_prefix}.shared_experts.down_proj.weight",
                    sharding=("tensor", None),
                    transpose=True,
                )
            
            mappings[f"{prefix}.mlp.shared_expert_gate.weight"] = WeightMapping(
                target_path=f"{target_prefix}.shared_expert_gate.weight",
                sharding=(None, None),
                transpose=True,
                # Explicitly ignore padding for gate as it's typically 1D or small
            )

        return mappings

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, layers_kv_fused, layers_topk_ids = self.model(
            forward_batch, token_to_kv_pool
        )

        if not getattr(self.config, "tie_word_embeddings", True):
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)

        return output, layers_kv_fused, True, layers_topk_ids


EntryClass = [MiMoV2FlashForCausalLM]
# Optionally add an alias if the config uses a different name
# EntryClass.append(MiMoV2ForCausalLM)
