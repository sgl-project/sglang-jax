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
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import EPMoE, GateLogit, TopK, create_moe_weights_mapping
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

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
            quantization_config=getattr(config, "quantization_config", None),
        )

    def __call__(self, hidden_states: jax.Array, forward_batch: ForwardBatch):
        router_logits = self.moe_gate(hidden_states)
        correction_bias = self.correction_bias.value if self.correction_bias is not None else None
        topk_weights, topk_ids = self.topk(router_logits, correction_bias=correction_bias)
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
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.head_dim = head_dim or hidden_size // num_heads
        self.q_head_num = num_heads
        self.k_head_num = num_kv_heads
        self.v_head_dim = v_head_dim if v_head_dim is not None else self.head_dim

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
            pass
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
            for proj_name in ("q_proj", "k_proj", "v_proj"):
                proj = getattr(attn, proj_name)
                if isinstance(proj, QuantizedLinear):
                    # Pass head_dim for per-head block quant handling.
                    # Q/K use head_dim, V uses v_head_dim.
                    hd = attn.v_head_dim if proj_name == "v_proj" else attn.head_dim
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

        # Fix kv_head_padding for v_proj: the weight loader's _apply_kv_head_padding
        # uses head_dim (Q/K) for shape matching, so it misses v_proj when
        # v_head_dim != head_dim. Replicate kv_heads here.
        self._ensure_kv_head_replication()

    def _ensure_kv_head_replication(self):
        """Replicate KV heads for TP alignment when the weight loader missed them.

        When v_head_dim != head_dim, the weight loader's _apply_kv_head_padding
        can't pattern-match v_proj shapes. We fix that here by checking actual
        weight dims against expected (tp-aligned) dims and replicating as needed.
        """
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        for layer_idx, layer in enumerate(self.model.layers):
            attn = layer.self_attn
            # attn.k_head_num is already the TP-aligned count (e.g., 16)
            target_kv_heads = attn.k_head_num

            for proj_name, head_dim in [
                ("k_proj", attn.head_dim),
                ("v_proj", attn.v_head_dim),
            ]:
                proj = getattr(attn, proj_name)
                w = proj.weight.value
                expected_size = target_kv_heads * head_dim
                actual_size = w.shape[1]

                if actual_size == expected_size:
                    continue

                # Weight is smaller than expected — needs replication
                if actual_size > 0 and expected_size % actual_size == 0:
                    num_replicas = expected_size // actual_size
                    orig_kv = actual_size // head_dim

                    logger.info(
                        "KV head replication: layer %d %s %d→%d heads (%d→%d)",
                        layer_idx,
                        proj_name,
                        orig_kv,
                        target_kv_heads,
                        actual_size,
                        expected_size,
                    )

                    w_full = jax.device_put(w, NamedSharding(self.mesh, P()))
                    w_3d = w_full.reshape(w.shape[0], orig_kv, head_dim)
                    w_rep = jnp.repeat(w_3d, num_replicas, axis=1)
                    w_new = w_rep.reshape(w.shape[0], expected_size)
                    w_new = jax.device_put(w_new, NamedSharding(self.mesh, P(None, "tensor")))

                    with jax.set_mesh(self.mesh):
                        new_linear = LinearBase(
                            input_size=w.shape[0],
                            output_size=expected_size,
                            kernel_axes=proj.kernel_axes,
                            use_bias=False,
                            params_dtype=jnp.bfloat16,
                            mesh=self.mesh,
                        )
                        new_linear.weight = nnx.Param(w_new)
                    setattr(attn, proj_name, new_linear)

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
                for key, mapping in moe_mappings.items():
                    augmented[key] = mapping
                    # Add scale mapping for each MoE group
                    target_param = mapping.target_path[0]
                    src_paths = mapping.target_path[1:]
                    scale_key = key + "_scale"
                    scale_target = target_param + "_scale"
                    scale_srcs = [p.replace(".weight", ".weight_scale_inv") for p in src_paths]
                    augmented[scale_key] = WeightMapping(
                        target_path=[scale_target] + scale_srcs,
                        sharding=("expert", None, None),
                        transpose=False,
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
