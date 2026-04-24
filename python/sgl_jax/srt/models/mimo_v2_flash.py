"""MiMo-V2-Flash model implementation for SGLang-JAX."""

import logging
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, get_rope
from sgl_jax.srt.layers.fused_moe import FusedEPMoE
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import EPMoE, GateLogit, TopK, create_moe_weights_mapping
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import (
    WeightLoader,
    WeightMapping,
    replicate_kv_heads,
)

logger = logging.getLogger(__name__)


class MiMoV2MLP(nnx.Module):
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
        )
        self.up_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
        )
        self.down_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            kernel_axes=("tensor", None),
            use_bias=False,
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
        mesh: jax.sharding.Mesh = None,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id

        num_experts = getattr(config, "n_routed_experts", getattr(config, "num_experts", 8))
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 2)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", config.intermediate_size)

        self.moe_gate = GateLogit(
            input_size=config.hidden_size,
            num_experts=num_experts,
            weight_dtype=dtype,
            score_func=getattr(config, "scoring_func", "softmax"),
        )

        self.topk_method = getattr(config, "topk_method", "greedy")
        if self.topk_method == "noaux_tc":
            self.correction_bias = nnx.Param(jnp.zeros(num_experts, dtype=jnp.float32))
        else:
            self.correction_bias = None

        self.moe_backend = getattr(config, "moe_backend", "epmoe")
        self.use_fused = self.moe_backend == "fused"

        self.topk = TopK(
            topk=num_experts_per_tok,
            renormalize=getattr(config, "norm_topk_prob", True),
        )

        if self.use_fused:
            self.experts = FusedEPMoE(
                hidden_size=config.hidden_size,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                intermediate_dim=moe_intermediate_size,
                mesh=mesh,
                activation="silu",
                ep_size=config.ep_size,
                weight_dtype=dtype,
                dtype=dtype,
                layer_id=layer_id,
                renormalize_topk_logits=getattr(config, "norm_topk_prob", True),
                quantization_config=getattr(config, "quantization_config", None),
            )
        else:
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
                quantization_config=getattr(config, "quantization_config", None),
            )

    def __call__(self, hidden_states: jax.Array, forward_batch: ForwardBatch):
        router_logits = self.moe_gate(hidden_states)
        correction_bias = self.correction_bias.value if self.correction_bias is not None else None
        topk_weights, topk_ids = self.topk(router_logits, correction_bias=correction_bias)
        if self.use_fused:
            token_valid_mask = forward_batch.get_token_valid_mask(hidden_states.shape[0])
            topk_ids = jnp.where(token_valid_mask[:, None], topk_ids, -1)
        mlp_output = self.experts(hidden_states, topk_weights, topk_ids)
        return mlp_output, topk_ids


class MiMoV2Attention(nnx.Module):
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
        attention_sink_bias: bool = False,
        partial_rotary_factor: float = 1.0,
        attention_value_scale: float | None = None,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.head_dim = head_dim or hidden_size // num_heads
        self.q_head_num = num_heads
        self.k_head_num = num_kv_heads
        self.v_head_dim = v_head_dim if v_head_dim is not None else self.head_dim
        self.attention_value_scale = attention_value_scale

        self.q_size = num_heads * self.head_dim
        self.k_size = num_kv_heads * self.head_dim
        self.v_size = num_kv_heads * self.v_head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = LinearBase(
            input_size=hidden_size,
            output_size=self.q_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
        )
        self.k_proj = LinearBase(
            input_size=hidden_size,
            output_size=self.k_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
        )
        self.v_proj = LinearBase(
            input_size=hidden_size,
            output_size=self.v_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
        )
        self.o_proj = LinearBase(
            input_size=num_heads * self.v_head_dim,
            output_size=hidden_size,
            kernel_axes=("tensor", None),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
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
            sliding_window_size=sliding_window_size,
        )

        self.attention_sink_bias = (
            nnx.Param(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (self.q_head_num,),
                    dtype=dtype,
                    out_sharding=P("tensor"),
                )
            )
            if attention_sink_bias
            else None
        )

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
        # Pad V to match Q/K head_dim for fused KV cache
        if self.v_head_dim != self.head_dim:
            pad_size = self.head_dim - self.v_head_dim
            v = jnp.pad(v, ((0, 0), (0, 0), (0, pad_size)))

        q, k = self.rotary_emb(positions, q, k)
        if self.attention_value_scale is not None:
            v = v * self.attention_value_scale

        attn_output, kv_fused = self.attn(
            q,
            k,
            v,
            forward_batch,
            token_to_kv_pool,
            attention_sink=self.attention_sink_bias.value if self.attention_sink_bias else None,
        )

        # V was padded to head_dim for fused KV cache; slice back to v_head_dim
        # so o_proj receives the correct input size.
        if self.head_dim != self.v_head_dim:
            expected_v_head_dim = self.q_head_num * self.v_head_dim
            if attn_output.shape[-1] != expected_v_head_dim:
                padded_head_dim = attn_output.shape[-1] // self.q_head_num
                attn_output = attn_output.reshape(-1, self.q_head_num, padded_head_dim)
                attn_output = attn_output[..., : self.v_head_dim]
                attn_output = attn_output.reshape(-1, expected_v_head_dim)

        output, _ = self.o_proj(attn_output)
        return output, kv_fused


class MiMoV2DecoderLayer(nnx.Module):
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
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        attention_value_scale = getattr(config, "attention_value_scale", None)

        if self._is_swa_layer(config):
            self.self_attn = MiMoV2Attention(
                hidden_size=config.hidden_size,
                num_heads=config.swa_num_attention_heads,
                num_kv_heads=config.swa_num_key_value_heads,
                max_position_embeddings=max_position_embeddings,
                rope_theta=getattr(config, "swa_rope_theta", rope_theta),
                rope_scaling=rope_scaling,
                head_dim=config.swa_head_dim,
                v_head_dim=getattr(config, "swa_v_head_dim", None),
                sliding_window_size=getattr(config, "sliding_window_size", None),
                attention_sink_bias=getattr(config, "add_swa_attention_sink_bias", False),
                partial_rotary_factor=getattr(config, "partial_rotary_factor", 1.0),
                attention_value_scale=attention_value_scale,
                layer_id=layer_id,
                dtype=dtype,
                mesh=mesh,
            )
        else:
            self.self_attn = MiMoV2Attention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                head_dim=config.head_dim,
                v_head_dim=getattr(config, "v_head_dim", None),
                sliding_window_size=0,  # full attention
                attention_sink_bias=getattr(config, "add_full_attention_sink_bias", False),
                partial_rotary_factor=getattr(config, "partial_rotary_factor", 1.0),
                attention_value_scale=attention_value_scale,
                layer_id=layer_id,
                dtype=dtype,
                mesh=mesh,
            )

        self.is_layer_sparse = self._is_moe_layer(config)

        if self.is_layer_sparse:
            self.mlp = MiMoV2Moe(
                config=config,
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
                mesh=mesh,
            )

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
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None]:
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

        if self.is_layer_sparse:
            mlp_output, topk_ids = self.mlp(hidden_states, forward_batch)
        else:
            mlp_output = self.mlp(hidden_states)
            topk_ids = None

        hidden_states = mlp_output
        return hidden_states, residual, kv_fused, topk_ids

    def _is_moe_layer(self, config) -> bool:
        moe_freq = getattr(config, "moe_layer_freq", None)
        return (
            moe_freq is not None
            and 0 <= self.layer_id < len(moe_freq)
            and not isinstance(moe_freq, int)
            and moe_freq[self.layer_id]
        )

    def _is_swa_layer(self, config) -> bool:
        hybrid = getattr(config, "hybrid_layer_pattern", None)
        if hybrid is not None and 0 <= self.layer_id < len(hybrid):
            return hybrid[self.layer_id] == 1
        return False


class MiMoV2Model(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
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
                MiMoV2DecoderLayer(config=config, layer_id=i, dtype=dtype, mesh=mesh)
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=config.layernorm_epsilon,
            param_dtype=dtype,
        )

    def __call__(self, forward_batch: ForwardBatch, token_to_kv_pool: KVCache):
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
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        self.model = MiMoV2Model(config, dtype=self.dtype, mesh=mesh)
        # Buffer to hold raw FP8 K/V weights+scales for per-head fused dequant.
        # Populated during weight loading, consumed by _dequant_fused_kv_heads.
        self._kv_buffers: dict[int, dict] = {}

        if not getattr(self.config, "tie_word_embeddings", True):
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
        self._quant_config = model_config.quantization_config
        weight_mappings = self._create_weight_mappings()
        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("MiMoV2Flash weights loaded successfully!")

        # Post-load: dequantize FP8 attention + layer-0 MLP to bf16.
        # Q/K head_dim padding (192→256) is handled by the kernel internally.
        if self._is_static_quant:
            self._dequantize_fp8_to_bf16()
            # K+V: fused per-head dequant (cross K/V boundary blocks)
            self._dequant_fused_kv_heads()
            # KV head replication for TP alignment (after K/V are bf16 LinearBase)
            self._ensure_kv_head_replication()

    def _is_quant_ignored(self, hf_path: str) -> bool:
        """Check if a HuggingFace weight path is in the quantization ignored_layers list."""
        quant_cfg = getattr(self, "_quant_config", None)
        if quant_cfg is None or not quant_cfg.is_static_checkpoint:
            return True  # not quantized at all
        ignored = quant_cfg.ignored_layers or []
        return any(hf_path == ig or hf_path.endswith(f".{ig}") for ig in ignored)

    @property
    def _is_static_quant(self) -> bool:
        quant_cfg = getattr(self, "_quant_config", None)
        return quant_cfg is not None and quant_cfg.is_static_checkpoint

    # ------------------------------------------------------------------
    # Post-load transforms: dequantize FP8 → BF16, pad head dims
    # ------------------------------------------------------------------

    def _dequantize_quantized_linear(self, ql, head_dim=None) -> LinearBase:
        """Dequantize a single QuantizedLinear to bf16 LinearBase.

        weight_q may be in HF layout [out, in] or model layout [in, out]
        depending on the transpose flag used during loading.
        weight_scale is in kernel-ready 3D layout [in_blocks, 1, out_dim].

        Handles kv_head_padding: when weight_q has been replicated along the
        output axis (e.g. 4 kv_heads → 16 for TP), the scale only covers the
        original (unreplicated) output dimension.  We extract one copy of the
        original heads, dequantize, then re-replicate in bf16.

        Args:
            head_dim: If set, enables per-head block quant handling. Required
                when head_dim % block_size != 0 (e.g., head_dim=192 with
                block_size=128), as the HF checkpoint uses per-head block
                boundaries instead of uniform blocks.
        """
        weight_q = ql.weight_q.value
        weight_scale = ql.weight_scale.value
        logger.info(
            "Dequant debug: weight_q.shape=%s weight_scale.shape=%s kernel_axes=%s head_dim=%s",
            weight_q.shape,
            weight_scale.shape,
            ql.kernel_axes,
            head_dim,
        )

        if weight_scale.ndim == 3:
            weight_bf16 = self._block_dequant(weight_q, weight_scale, head_dim=head_dim)
        elif weight_scale.ndim == 2:
            # 2D block-quant scale: (out_blocks, in_blocks) in HF layout.
            # Expand to 3D kernel-ready (in_blocks, 1, out_dim) then reuse _block_dequant.
            from sgl_jax.srt.kernels.quantized_matmul.blockwise_utils import (
                expand_block_scale,
            )

            # Determine out_dim: weight_q is (in, out) model layout or (out, in) HF layout.
            _, in_blocks = weight_scale.shape
            out_dim = weight_q.shape[1] if weight_q.shape[0] % in_blocks == 0 else weight_q.shape[0]
            block_size_out = 128
            scale_3d = expand_block_scale(weight_scale, out_dim, block_size_out)
            weight_bf16 = self._block_dequant(weight_q, scale_3d, head_dim=head_dim)
        elif weight_scale.ndim == 1:
            out_dim = weight_scale.shape[0]
            if weight_q.shape[1] == out_dim:
                weight_bf16 = (weight_q.astype(jnp.float32) * weight_scale[None, :]).astype(
                    jnp.bfloat16
                )
            else:
                weight_bf16 = (
                    jnp.transpose(weight_q).astype(jnp.float32) * weight_scale[None, :]
                ).astype(jnp.bfloat16)
        else:
            raise ValueError(f"Unexpected weight_scale ndim={weight_scale.ndim}")

        # weight_bf16 is now in model layout [in, out]
        in_features, out_features = weight_bf16.shape

        with jax.set_mesh(ql.mesh):
            new_linear = LinearBase(
                input_size=in_features,
                output_size=out_features,
                kernel_axes=ql.kernel_axes,
                use_bias=ql.bias is not None,
                params_dtype=jnp.bfloat16,
                mesh=ql.mesh,
            )
            new_linear.weight = nnx.Param(weight_bf16)
            if ql.bias is not None:
                new_linear.bias = nnx.Param(ql.bias.value.astype(jnp.bfloat16))
        return new_linear

    def _block_dequant(
        self, weight_q: jax.Array, weight_scale: jax.Array, head_dim: int | None = None
    ) -> jax.Array:
        """Block-dequantize weight_q using 3D scale [in_blocks, 1, out_dim].

        Returns bf16 weight in model layout [in_dim, out_dim].
        Handles kv_head_padding where weight_q is larger than scale coverage
        by tiling the scale to match.

        Args:
            head_dim: If set, enables per-head block quant handling when scale
                out_dim doesn't match weight dims. The scale is re-indexed using
                per-head block boundaries instead of uniform block mapping.
        """
        import math

        in_blocks = weight_scale.shape[0]
        out_dim = weight_scale.shape[2]

        # Detect layout and kv-padding
        dim0, dim1 = weight_q.shape

        if dim1 == out_dim:
            # Model layout [in, out], no kv-padding
            # Check if per-head re-indexing is needed: when head_dim doesn't
            # divide evenly by block_size, the pre-expanded uniform scale has
            # wrong block assignments at head boundaries.
            if head_dim is not None and head_dim % 128 != 0:
                block_size = 128
                blocks_per_head = math.ceil(head_dim / block_size)
                num_heads = out_dim // head_dim
                # Build per-head gather: map each output channel to a channel
                # position in the uniformly-expanded scale that holds the
                # correct per-head block's value.
                gather_idx = jnp.array(
                    [
                        ((j // head_dim) * blocks_per_head + (j % head_dim) // block_size)
                        * block_size
                        for j in range(out_dim)
                    ]
                )
                weight_scale = weight_scale[:, :, gather_idx]
                logger.info(
                    "Per-head block re-index (pre-expanded): %d heads × head_dim=%d, "
                    "%d blocks/head, scale shape=%s",
                    num_heads,
                    head_dim,
                    blocks_per_head,
                    weight_scale.shape,
                )
        elif dim0 == out_dim:
            # HF layout [out, in] — transpose to model layout
            weight_q = jnp.transpose(weight_q)
        elif head_dim is not None and dim1 != out_dim and dim0 != out_dim:
            # Per-head block quant: scale was expanded for wrong n_out.
            # Determine layout: in_dim must be divisible by in_blocks.
            if dim0 % in_blocks == 0:
                actual_out = dim1  # model layout [in, out]
            else:
                weight_q = jnp.transpose(weight_q)
                actual_out = dim0  # was HF layout [out, in]

            # Re-index scale using per-head block boundaries.
            # The wrongly-expanded scale has uniform block mapping (ch // 128),
            # but we need per-head mapping where each head's blocks are independent.
            block_size = out_dim // (out_dim // 128) if out_dim >= 128 else 128
            block_size = 128  # from weight_block_size config
            blocks_per_head = math.ceil(head_dim / block_size)
            num_heads = actual_out // head_dim

            # Build gather index: for each output channel, find the corresponding
            # channel in the uniformly-expanded scale that has the correct block's value.
            gather_idx = jnp.array(
                [
                    ((j // head_dim) * blocks_per_head + (j % head_dim) // block_size) * block_size
                    for j in range(actual_out)
                ]
            )
            weight_scale = weight_scale[:, :, gather_idx]  # (in_blocks, 1, actual_out)
            out_dim = actual_out

            logger.info(
                "Per-head block dequant: %d heads × head_dim=%d, %d blocks/head, "
                "remapped scale to (%s)",
                num_heads,
                head_dim,
                blocks_per_head,
                weight_scale.shape,
            )
        elif dim1 > out_dim and dim1 % out_dim == 0:
            # Model layout [in, kv_padded_out] — tile scale to match
            kv_replicas = dim1 // out_dim
            weight_scale = jnp.tile(weight_scale, (1, 1, kv_replicas))
            out_dim = dim1
            logger.info(
                "Detected kv_head_padding: tiling scale %dx to match %d out channels",
                kv_replicas,
                out_dim,
            )
        elif dim0 > out_dim and dim0 % out_dim == 0:
            # HF layout [kv_padded_out, in] — transpose and tile scale
            kv_replicas = dim0 // out_dim
            weight_q = jnp.transpose(weight_q)
            weight_scale = jnp.tile(weight_scale, (1, 1, kv_replicas))
            out_dim = weight_q.shape[1]
            logger.info(
                "Detected kv_head_padding (HF layout): tiling scale %dx to match %d out channels",
                kv_replicas,
                out_dim,
            )
        else:
            raise ValueError(
                f"Cannot match weight_q shape {weight_q.shape} with scale out_dim={out_dim}"
            )

        # weight_q is now [in_dim, out_dim] in model layout
        in_dim = weight_q.shape[0]
        block_k = in_dim // in_blocks
        weight_f = weight_q.astype(jnp.float32).reshape(in_blocks, block_k, out_dim)
        weight_bf16 = (weight_f * weight_scale).reshape(in_dim, out_dim).astype(jnp.bfloat16)

        return weight_bf16

    def _dequantize_fp8_to_bf16(self):
        """Dequantize FP8 QuantizedLinear → bf16 LinearBase.

        Targets:
        - All layers: self_attn.q_proj, k_proj, v_proj
        - Layer 0: mlp.gate_proj, up_proj, down_proj (dense MLP only)
        """
        from sgl_jax.srt.layers.linear import QuantizedLinear

        for layer_idx, layer in enumerate(self.model.layers):
            attn = layer.self_attn
            # Only Q and o_proj go through QuantizedLinear path.
            # K/V are handled by _dequant_fused_kv_heads (per-head fused dequant).
            for proj_name in ("q_proj",):
                proj = getattr(attn, proj_name)
                if isinstance(proj, QuantizedLinear):
                    hd = attn.head_dim
                    setattr(attn, proj_name, self._dequantize_quantized_linear(proj, head_dim=hd))
                    logger.info("Dequantized layer %d %s → bf16", layer_idx, proj_name)

            # Layer 0 dense MLP
            if layer_idx == 0 and not layer.is_layer_sparse:
                for proj_name in ("gate_proj", "up_proj", "down_proj"):
                    proj = getattr(layer.mlp, proj_name)
                    if isinstance(proj, QuantizedLinear):
                        setattr(layer.mlp, proj_name, self._dequantize_quantized_linear(proj))
                        logger.info("Dequantized layer 0 MLP %s → bf16", proj_name)

        logger.info("FP8 → BF16 dequantization complete.")

    def _ensure_kv_head_replication(self):
        """Replicate KV heads for TP alignment when the weight loader missed them."""
        attn = self.model.layers[0].self_attn
        replicate_kv_heads(
            layers=self.model.layers,
            mesh=self.mesh,
            head_dim=attn.head_dim,
            v_head_dim=attn.v_head_dim,
            target_kv_heads=attn.k_head_num,
        )

    def _uniform_block_dequant(self, weight, scale, block_size):
        """Simple uniform block dequant for weight[out_dim, in_dim] * scale[out_blocks, in_blocks].

        Used for layers where K/V are quantized uniformly across all heads
        (no cross-boundary scale sharing between K and V).
        """
        out_dim, in_dim = weight.shape
        out_blocks = scale.shape[0]
        padded_out = out_blocks * block_size
        in_blocks = scale.shape[1]
        if padded_out > out_dim:
            weight = jnp.pad(weight, ((0, padded_out - out_dim), (0, 0)))
        w_4d = weight.astype(jnp.float32).reshape(out_blocks, block_size, in_blocks, block_size)
        s_4d = scale[:, None, :, None]
        result = (w_4d * s_4d).reshape(padded_out, in_dim)[:out_dim, :].astype(jnp.bfloat16)
        return result

    def _dequant_fused_kv_heads(self):
        """Dequantize FP8 K+V weights with per-layer quantization scheme detection.

        Different layers may use different quantization schemes:
        - Per-head fused: K+V quantized as fused [K(head_dim), V(v_head_dim)] per KV head.
          Block boundaries cross K/V boundary, so they must be fused for correct dequant.
          Signature: k_scale_blocks == num_kv_heads * ceil(head_dim/block_size)
        - Uniform: K and V quantized independently across the whole tensor.
          No cross-boundary issue, can dequant K and V separately.
          Signature: k_scale_blocks == ceil(num_kv_heads * head_dim / block_size)
        """
        import math

        from jax.sharding import NamedSharding

        kv_buffers = self._kv_buffers
        if not kv_buffers:
            return

        head_dim = self.config.head_dim
        v_head_dim = getattr(self.config, "v_head_dim", head_dim)
        quant_cfg = getattr(self, "_quant_config", None)
        block_size = int(quant_cfg.weight_block_size[0]) if quant_cfg else 128

        fused_dim = head_dim + v_head_dim
        blocks_per_head = math.ceil(fused_dim / block_size)
        padded_dim = blocks_per_head * block_size
        k_blocks_per_head = math.ceil(head_dim / block_size)
        v_blocks_per_head = blocks_per_head - k_blocks_per_head

        tp_sharding = NamedSharding(self.mesh, P(None, "tensor"))

        for layer_idx in sorted(kv_buffers.keys()):
            buf = kv_buffers[layer_idx]
            k_weight = buf["k_weight"]
            k_scale = buf["k_scale"]
            v_weight = buf["v_weight"]
            v_scale = buf["v_scale"]

            in_dim = k_weight.shape[1]
            in_blocks = in_dim // block_size

            # Per-layer num_kv_heads from actual weight shape
            num_kv_heads = k_weight.shape[0] // head_dim
            k_scale_blocks = k_scale.shape[0]

            # Detect quantization scheme for this layer
            expected_per_head = num_kv_heads * k_blocks_per_head
            expected_uniform = math.ceil(num_kv_heads * head_dim / block_size)
            is_per_head = (
                k_scale_blocks == expected_per_head and expected_per_head != expected_uniform
            )

            if is_per_head:
                # Per-head fused: K+V must be fused because scale blocks cross K/V boundary
                k_w = k_weight.reshape(num_kv_heads, head_dim, in_dim)
                v_w = v_weight.reshape(num_kv_heads, v_head_dim, in_dim)
                k_s = k_scale.reshape(num_kv_heads, k_blocks_per_head, in_blocks)
                v_s = v_scale.reshape(num_kv_heads, v_blocks_per_head, in_blocks)
                fused_w = jnp.concatenate([k_w, v_w], axis=1)
                fused_s = jnp.concatenate([k_s, v_s], axis=1)
                if fused_dim < padded_dim:
                    fused_w = jnp.pad(fused_w, ((0, 0), (0, padded_dim - fused_dim), (0, 0)))
                fused_5d = fused_w.astype(jnp.float32).reshape(
                    num_kv_heads, blocks_per_head, block_size, in_blocks, block_size
                )
                scale_5d = fused_s[:, :, None, :, None]
                dequanted = (
                    (fused_5d * scale_5d)
                    .reshape(num_kv_heads, padded_dim, in_dim)[:, :fused_dim, :]
                    .astype(jnp.bfloat16)
                )
                k_bf16 = dequanted[:, :head_dim, :].reshape(num_kv_heads * head_dim, in_dim)
                v_bf16 = dequanted[:, head_dim:, :].reshape(num_kv_heads * v_head_dim, in_dim)
            else:
                # Uniform: K and V can be dequanted independently
                k_bf16 = self._uniform_block_dequant(k_weight, k_scale, block_size)
                v_bf16 = self._uniform_block_dequant(v_weight, v_scale, block_size)

            # Transpose [out, in] → [in, out], shard, replace
            k_bf16 = jax.device_put(jnp.transpose(k_bf16), tp_sharding)
            v_bf16 = jax.device_put(jnp.transpose(v_bf16), tp_sharding)

            attn = self.model.layers[layer_idx].self_attn
            in_k, out_k = k_bf16.shape
            in_v, out_v = v_bf16.shape

            with jax.set_mesh(self.mesh):
                k_linear = LinearBase(
                    input_size=in_k,
                    output_size=out_k,
                    kernel_axes=(None, "tensor"),
                    use_bias=False,
                    params_dtype=jnp.bfloat16,
                    mesh=self.mesh,
                )
                k_linear.weight = nnx.Param(k_bf16)
                attn.k_proj = k_linear

                v_linear = LinearBase(
                    input_size=in_v,
                    output_size=out_v,
                    kernel_axes=(None, "tensor"),
                    use_bias=False,
                    params_dtype=jnp.bfloat16,
                    mesh=self.mesh,
                )
                v_linear.weight = nnx.Param(v_bf16)
                attn.v_proj = v_linear

            if layer_idx % 10 == 0 or layer_idx == 0:
                logger.info(
                    "Layer %d KV dequant: %s, heads=%d, K=%s V=%s",
                    layer_idx,
                    "per-head" if is_per_head else "uniform",
                    num_kv_heads,
                    k_bf16.shape,
                    v_bf16.shape,
                )

        kv_buffers.clear()
        logger.info("FP8 KV dequantization complete for all layers.")

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

        if not getattr(self.config, "tie_word_embeddings", True):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding",
                sharding=("tensor", None),
                transpose=False,
            )

        for layer_idx in range(self.config.num_hidden_layers):
            mappings.update(self._create_layer_mappings(layer_idx))

        return mappings

    def _create_layer_mappings(self, layer_idx: int) -> dict:
        prefix = f"model.layers.{layer_idx}"
        target = prefix

        mappings = {}
        is_fp8 = self._is_static_quant

        # Attention projections
        for proj, sharding, kv_pad, hd_pad in [
            ("q_proj", (None, "tensor"), False, True),
            ("k_proj", (None, "tensor"), not is_fp8, True),
            ("v_proj", (None, "tensor"), not is_fp8, False),
            ("o_proj", ("tensor", None), False, True),
        ]:
            hf_key = f"{prefix}.self_attn.{proj}"
            ignored = self._is_quant_ignored(hf_key)

            # FP8 K/V: bypass QuantizedLinear, store raw FP8 data for fused
            # per-head dequant (cross K/V boundary blocks).
            if is_fp8 and not ignored and proj in ("k_proj", "v_proj"):
                kv_key = "K" if proj == "k_proj" else "V"
                mappings[f"{hf_key}.weight"] = WeightMapping(
                    target_path=f"__KV_{kv_key}_WEIGHT__{layer_idx}",
                    sharding=(None, None),
                    transpose=False,
                )
                mappings[f"{hf_key}.weight_scale_inv"] = WeightMapping(
                    target_path=f"__KV_{kv_key}_SCALE__{layer_idx}",
                    sharding=(None, None),
                    transpose=False,
                )
                continue

            weight_suffix = "weight" if (not is_fp8 or ignored) else "weight_q"

            mappings[f"{hf_key}.weight"] = WeightMapping(
                target_path=f"{target}.self_attn.{proj}.{weight_suffix}",
                sharding=sharding,
                transpose=True,
                head_dim_padding=hd_pad,
                kv_head_padding=kv_pad,
            )

            if is_fp8 and not ignored:
                mappings[f"{hf_key}.weight_scale_inv"] = WeightMapping(
                    target_path=f"{target}.self_attn.{proj}.weight_scale",
                    sharding=(None, None),
                    transpose=False,
                )

        # Attention sink bias
        is_swa = (
            hasattr(self.config, "hybrid_layer_pattern")
            and 0 <= layer_idx < len(self.config.hybrid_layer_pattern)
            and self.config.hybrid_layer_pattern[layer_idx] == 1
        )
        has_sink_bias = (is_swa and getattr(self.config, "add_swa_attention_sink_bias", False)) or (
            not is_swa and getattr(self.config, "add_full_attention_sink_bias", False)
        )
        if has_sink_bias:
            mappings[f"{prefix}.self_attn.attention_sink_bias"] = WeightMapping(
                target_path=f"{target}.self_attn.attention_sink_bias",
                sharding=("tensor",),
                transpose=False,
            )

        # Layernorms
        mappings[f"{prefix}.input_layernorm.weight"] = WeightMapping(
            target_path=f"{target}.input_layernorm.scale",
            sharding=(None,),
            transpose=False,
        )
        mappings[f"{prefix}.post_attention_layernorm.weight"] = WeightMapping(
            target_path=f"{target}.post_attention_layernorm.scale",
            sharding=(None,),
            transpose=False,
        )

        # MLP / MoE
        is_sparse = (
            hasattr(self.config, "moe_layer_freq")
            and 0 <= layer_idx < len(self.config.moe_layer_freq)
            and not isinstance(self.config.moe_layer_freq, int)
            and self.config.moe_layer_freq[layer_idx]
        )

        if is_sparse:
            # MoE gate
            mappings[f"{prefix}.mlp.gate.weight"] = WeightMapping(
                target_path=f"{target}.mlp.moe_gate.kernel",
                sharding=(None, None),
                transpose=True,
            )

            # Correction bias for noaux_tc
            if getattr(self.config, "topk_method", "greedy") == "noaux_tc":
                mappings[f"{prefix}.mlp.gate.e_score_correction_bias"] = WeightMapping(
                    target_path=f"{target}.mlp.correction_bias",
                    sharding=(None,),
                    transpose=False,
                )

            # Expert weights — use standard create_moe_weights_mapping
            num_experts = getattr(
                self.config,
                "n_routed_experts",
                getattr(self.config, "num_experts", 8),
            )
            moe_backend = getattr(self.config, "moe_backend", "epmoe")

            moe_mappings = create_moe_weights_mapping(
                prefix=prefix,
                target_prefix=target,
                num_experts=num_experts,
                moe_backend=moe_backend,
                moe_path="mlp.experts",
                source_expert_pattern="{i}",
            )

            if is_fp8:
                augmented = {}
                use_model_mesh_for_scale = moe_backend == "fused"
                for key, mapping in moe_mappings.items():
                    augmented[key] = mapping
                    # Add scale mapping for each MoE group
                    target_param = mapping.target_path[0]
                    src_paths = mapping.target_path[1:]
                    scale_key = key + "_scale"
                    scale_target = target_param + "_scale"
                    scale_srcs = [p.replace(".weight", ".weight_scale_inv") for p in src_paths]
                    scale_sharding = (
                        (("data", "tensor"), None, None)
                        if use_model_mesh_for_scale
                        else ("expert", None, None)
                    )
                    augmented[scale_key] = WeightMapping(
                        target_path=[scale_target] + scale_srcs,
                        sharding=scale_sharding,
                        transpose=use_model_mesh_for_scale,
                        concat_axis=mapping.concat_axis,
                        physical_to_logical_map=mapping.physical_to_logical_map,
                    )
                moe_mappings = augmented

            mappings.update(moe_mappings)
        else:
            # Dense MLP
            for proj, sharding in [
                ("gate_proj", (None, "tensor")),
                ("up_proj", (None, "tensor")),
                ("down_proj", ("tensor", None)),
            ]:
                hf_key = f"{prefix}.mlp.{proj}"
                weight_suffix = "weight_q" if is_fp8 else "weight"
                mappings[f"{hf_key}.weight"] = WeightMapping(
                    target_path=f"{target}.mlp.{proj}.{weight_suffix}",
                    sharding=sharding,
                    transpose=True,
                )
                if is_fp8:
                    mappings[f"{hf_key}.weight_scale_inv"] = WeightMapping(
                        target_path=f"{target}.mlp.{proj}.weight_scale",
                        sharding=(None, None),
                        transpose=False,
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
