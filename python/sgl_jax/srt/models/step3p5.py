"""Inference-only Step 3.5 Flash model.

Hybrid sliding/full-attention GQA + sigmoid-gated MoE. ``model_type="step3p5"``.
Reference: HF modeling_step3p5.py / configuration_step3p5.py.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, get_rope
from sgl_jax.srt.layers.layernorm import GemmaRMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import EPMoE, GateLogit, TopK
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.precision_tracer import precision_tracer
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)

_HEAD_DIM = 128


class Step3p5Attention(nnx.Module):
    """Per-head QK-norm + per-layer partial RoPE + head-wise gate + optional SWA.

    full: 64 Q-heads, theta 5e6, partial 0.5, llama3 scaling. sliding: 96 Q-heads,
    theta 1e4, partial 1.0, no scaling. Both 8 KV-heads, head_dim 128.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        attn_impl: str = "flash",
    ) -> None:
        # "flash" → RadixAttention (TPU kernel, production default).
        # "naive" → pure-JAX fp32 oracle (CPU-runnable, Plan 4 reference).
        self.attn_impl = attn_impl
        self.layer_id = layer_id
        self.dtype = dtype
        self.head_dim = _HEAD_DIM
        self.mesh = mesh

        layer_types = config.layer_types or []
        layer_type = layer_types[layer_id] if layer_id < len(layer_types) else "full_attention"
        self._layer_type = layer_type

        is_sliding = layer_type == "sliding_attention"

        if is_sliding and config.attention_other_setting is not None:
            self.num_heads = config.attention_other_setting["num_attention_heads"]
            self.num_kv_heads = config.attention_other_setting.get(
                "num_attention_groups", config.num_attention_groups
            )
        else:
            self.num_heads = config.num_attention_heads
            self.num_kv_heads = config.num_attention_groups

        # TP kv-head replication: when tensor parallelism exceeds the kv-head count,
        # replicate kv heads so each tensor shard holds >=1 head (mirrors
        # MiniMaxM2Attention). On tp=1 (CPU / naive) replicas=1, so num_kv_heads is
        # unchanged and the naive GQA grouping is unaffected.
        tp = mesh.shape.get("tensor", 1)
        replicas = (
            (tp + self.num_kv_heads - 1) // self.num_kv_heads if tp > self.num_kv_heads else 1
        )
        self.num_kv_heads = self.num_kv_heads * replicas

        self.q_size = self.num_heads * _HEAD_DIM
        self.kv_size = self.num_kv_heads * _HEAD_DIM
        self.scaling = _HEAD_DIM**-0.5

        self.q_proj = LinearBase(
            input_size=config.hidden_size,
            output_size=self.q_size,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="q_proj",
        )
        self.k_proj = LinearBase(
            input_size=config.hidden_size,
            output_size=self.kv_size,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="k_proj",
        )
        self.v_proj = LinearBase(
            input_size=config.hidden_size,
            output_size=self.kv_size,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="v_proj",
        )
        self.o_proj = LinearBase(
            input_size=self.q_size,
            output_size=config.hidden_size,
            use_bias=False,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="o_proj",
        )

        # Per-head zero-centered QK-norm (weight [head_dim], shared across heads), applied before RoPE.
        self.q_norm = GemmaRMSNorm(_HEAD_DIM, epsilon=config.rms_norm_eps, add_unit_offset=True)
        self.k_norm = GemmaRMSNorm(_HEAD_DIM, epsilon=config.rms_norm_eps, add_unit_offset=True)

        if isinstance(config.rope_theta, list):
            rope_theta_i = float(config.rope_theta[layer_id])
        else:
            rope_theta_i = float(config.rope_theta)

        if config.partial_rotary_factors is not None:
            partial_rotary_factor_i = float(config.partial_rotary_factors[layer_id])
        else:
            partial_rotary_factor_i = 1.0

        # llama3 scaling only on yarn_only_types (full_attention) layers.
        yarn_only_types = config.yarn_only_types or []
        rope_scaling_i = config.rope_scaling if layer_type in yarn_only_types else None

        self.rotary_emb = get_rope(
            head_size=_HEAD_DIM,
            rotary_dim=_HEAD_DIM,
            max_position=config.max_position_embeddings,
            base=rope_theta_i,  # type: ignore[arg-type]
            rope_scaling=rope_scaling_i,
            partial_rotary_factor=partial_rotary_factor_i,
            is_neox_style=True,
            dtype=dtype,
        )

        # Head-wise gate: g_proj(hidden) -> per-head sigmoid scalar, applied before o_proj.
        self.use_head_wise_attn_gate: bool = bool(getattr(config, "use_head_wise_attn_gate", True))
        if self.use_head_wise_attn_gate:
            self.g_proj = LinearBase(
                input_size=config.hidden_size,
                output_size=self.num_heads,
                use_bias=False,
                kernel_axes=(None, "tensor"),
                params_dtype=dtype,
                mesh=mesh,
                scope_name="g_proj",
            )

        # sliding_window_size=0 means full attention (gemma2 convention).
        sliding_window_size = config.sliding_window if is_sliding else 0
        # Keep a plain int copy for the naive path (RadixAttention stores 0→None).
        self._naive_sliding_window: int | None = int(config.sliding_window) if is_sliding else None
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=_HEAD_DIM,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads,
            sliding_window_size=sliding_window_size,
            layer_id=layer_id,
        )

    def _naive_attention(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
    ) -> jax.Array:
        """Pure-JAX fp32 attention oracle (CPU-runnable, single-sequence prefill).

        Inputs post QK-norm + RoPE: q [T, num_heads, 128], k/v [T, num_kv_heads, 128].
        Returns attn_output [T, num_heads * 128].

        Window predicate (HF create_sliding_window_causal_mask / native_backend):
          valid[q, k] := (k_pos <= q_pos) AND (q_pos - k_pos < W)
        For full-attention layers W is absent (no window mask applied).

        GQA: kv-heads repeated via np.repeat to match num_heads, same grouping order as
        native_backend._apply_extend_mask (repeat_interleave along heads axis).

        Softmax computed in fp32. Caution: this provides the naive==HF oracle (CPU);
        flash==naive is completed in Plan 4 (TPU). Until then, the production flash
        path is NOT yet verified end-to-end via naive==HF ∧ flash==naive ⇒ flash==HF.
        """
        T = q.shape[0]
        q_fp = jnp.asarray(q, dtype=jnp.float32)  # [T, num_heads, 128]
        k_fp = jnp.asarray(k, dtype=jnp.float32)  # [T, num_kv_heads, 128]
        v_fp = jnp.asarray(v, dtype=jnp.float32)  # [T, num_kv_heads, 128]

        # GQA: expand kv-heads to num_heads via repeat_interleave (axis=1).
        # out_sharding required under explicit-sharding mesh mode.
        num_q_per_kv = self.num_heads // self.num_kv_heads
        _unsharded_3d = NamedSharding(self.mesh, P(None, None, None))
        k_exp = jnp.repeat(k_fp, num_q_per_kv, axis=1, out_sharding=_unsharded_3d)
        v_exp = jnp.repeat(v_fp, num_q_per_kv, axis=1, out_sharding=_unsharded_3d)

        # Scores: [T_q, num_heads, T_k] in fp32.
        scores = (
            jnp.einsum("qhd,khd->qhk", q_fp, k_exp, preferred_element_type=jnp.float32)
            * self.scaling
        )  # noqa: E501

        # Causal mask: k_pos <= q_pos.
        q_idx = jnp.arange(T, dtype=jnp.int32)
        k_idx = jnp.arange(T, dtype=jnp.int32)
        causal = q_idx[:, None] >= k_idx[None, :]  # [T, T]

        if self._naive_sliding_window is not None:
            window = q_idx[:, None] - k_idx[None, :] < self._naive_sliding_window
            valid = causal & window
        else:
            valid = causal

        neg_inf = jnp.finfo(jnp.float32).min / 2
        # Broadcast valid [T, T] to [T, 1, T] for heads.
        scores = jnp.where(valid[:, None, :], scores, neg_inf)

        # Softmax over key axis in fp32.
        scores = scores - jnp.max(scores, axis=-1, keepdims=True)
        exp_s = jnp.exp(scores)
        attn_w = exp_s / jnp.sum(exp_s, axis=-1, keepdims=True)  # [T, H, T]

        # Weighted sum: [T, num_heads, 128].
        out = jnp.einsum("qhk,khd->qhd", attn_w, v_exp, preferred_element_type=jnp.float32)
        return out.reshape(T, self.num_heads * _HEAD_DIM)

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, jax.Array]:
        # Gate reads the attention INPUT (HF ref line 489), not the attn output.
        if self.use_head_wise_attn_gate:
            gate_states, _ = self.g_proj(hidden_states)

        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        # Reshape to heads. out_sharding is required under JAX 0.8 sharding-in-types:
        # the proj output's last dim is sharded on "tensor", and JAX cannot infer the
        # post-reshape sharding (mirrors qwen3/minimax_m2). The head dim carries the
        # tensor sharding; kv heads are already replicated to >= tp (see __init__).
        _head_sharding = NamedSharding(self.mesh, P("data", "tensor", None))
        q = q.reshape(-1, self.num_heads, _HEAD_DIM, out_sharding=_head_sharding)
        k = k.reshape(-1, self.num_kv_heads, _HEAD_DIM, out_sharding=_head_sharding)

        # QK-norm before RoPE.
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)

        v = v.reshape(-1, self.num_kv_heads, _HEAD_DIM, out_sharding=_head_sharding)

        if self.attn_impl == "naive":
            attn_output = self._naive_attention(q, k, v)
            # kv_fused placeholder: callers (Step3p5 decoder, not yet built) store the
            # fused KV buffer for the TPU kernel; naive path has no KV pool.
            kv_fused = jnp.empty((0,), dtype=jnp.float32)
        else:
            attn_output, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool)

        # Per-head gate before o_proj (HF ref lines 528-531).
        if self.use_head_wise_attn_gate:
            gate = jax.nn.sigmoid(gate_states)
            out = attn_output.reshape(-1, self.num_heads, _HEAD_DIM) * gate[..., None]
            attn_output = out.reshape(-1, self.num_heads * _HEAD_DIM)

        output, _ = self.o_proj(attn_output)
        return output, kv_fused


class Step3p5MLP(nnx.Module):
    """SwiGLU dense/shared-expert MLP.

    down(silu(gate(x)) * up(x)), with optional asymmetric clamp matching HF Step3p5MLP.
    ``swiglu_limit``: gate clamped upper-only, up double-sided. None = no clamp.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        swiglu_limit: float | None = None,
    ) -> None:
        self.swiglu_limit = swiglu_limit
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
        gate = jax.nn.silu(gate)
        if self.swiglu_limit is not None:
            gate = jnp.clip(gate, max=self.swiglu_limit)
            up = jnp.clip(up, -self.swiglu_limit, self.swiglu_limit)
        output, _ = self.down_proj(gate * up)
        return output


def _swiglu_limit_for(limits: list[float] | None, layer_id: int) -> float | None:
    """Return per-layer swiglu limit or None if absent/zero (matches HF logic)."""
    if limits is None or layer_id >= len(limits):
        return None
    v = limits[layer_id]
    return float(v) if v else None


def _moe_layer_ids(config: PretrainedConfig) -> list[int]:
    """Return sorted list of MoE layer indices from config.moe_layers_enum."""
    moe_enum = getattr(config, "moe_layers_enum", None)
    if moe_enum is None:
        return list(range(1, config.num_hidden_layers))
    if isinstance(moe_enum, str):
        return [int(i) for i in moe_enum.strip().split(",")]
    return list(moe_enum)


class Step3p5MoE(nnx.Module):
    """Sigmoid-gated MoE with bias routing + always-on shared expert.

    Mirrors HF Step3p5MoEMLP + separate share_expert in Step3p5DecoderLayer.
    Forward: moe_out(hidden) + shared_experts(hidden).
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        mesh: jax.sharding.Mesh,
        ep_size: int = 1,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        hidden_size: int = config.hidden_size
        num_experts: int = config.moe_num_experts
        topk: int = config.moe_top_k
        moe_intermediate_size: int = config.moe_intermediate_size
        share_expert_dim: int = config.share_expert_dim
        routed_scaling_factor: float = config.moe_router_scaling_factor
        renormalize: bool = config.norm_expert_weight

        # Gate: fp32 kernel, HIGHEST precision dot (need_fp32_gate).
        # GateLogit already uses fp32 kernel + HIGHEST precision.
        self.moe_gate = GateLogit(
            input_size=hidden_size,
            num_experts=num_experts,
            score_func="sigmoid",
            enable_expert_bias=config.use_moe_router_bias,
            weight_dtype=dtype,
        )

        self.topk = TopK(
            topk=topk,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            layer_id=layer_id,
        )

        swiglu_limit_routed = _swiglu_limit_for(getattr(config, "swiglu_limits", None), layer_id)
        self.experts = EPMoE(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=topk,
            intermediate_dim=moe_intermediate_size,
            ep_size=ep_size,
            mesh=mesh,
            weight_dtype=dtype,
            dtype=dtype,
            layer_id=layer_id,
            swiglu_limit=swiglu_limit_routed,
        )

        swiglu_limit_shared = _swiglu_limit_for(
            getattr(config, "swiglu_limits_shared", None), layer_id
        )
        self.shared_experts = Step3p5MLP(
            hidden_size=hidden_size,
            intermediate_size=share_expert_dim,
            mesh=mesh,
            dtype=dtype,
            swiglu_limit=swiglu_limit_shared,
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        router_logits = self.moe_gate(hidden_states)
        correction_bias = self.moe_gate.bias.value if self.moe_gate.bias is not None else None
        topk_weights, topk_ids = self.topk(router_logits, correction_bias)
        moe_out = self.experts(hidden_states, topk_weights, topk_ids)
        shared_out = self.shared_experts(hidden_states)
        return moe_out + shared_out


class Step3p5DecoderLayer(nnx.Module):
    """Pre-norm fused-residual decoder layer (DeepSeek/M2 pattern).

    Supports both dense (layers 0-2) and MoE (layers 3+) FFN blocks,
    with per-layer attn_impl threading for CPU-runnable naive oracle.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        attn_impl: str = "flash",
    ) -> None:
        self.layer_id = layer_id

        self.self_attn = Step3p5Attention(
            config=config,
            layer_id=layer_id,
            mesh=mesh,
            dtype=dtype,
            attn_impl=attn_impl,
        )

        moe_ids = _moe_layer_ids(config)
        self.is_moe_layer = layer_id in moe_ids

        if self.is_moe_layer:
            self.mlp = Step3p5MoE(
                config=config,
                layer_id=layer_id,
                mesh=mesh,
                ep_size=getattr(config, "ep_size", 1),
                dtype=dtype,
            )
        else:
            # Dense MLP; shared swiglu limit applies (None for layers 0-2 per HF).
            swiglu_limit = _swiglu_limit_for(
                getattr(config, "swiglu_limits_shared", None), layer_id
            )
            self.mlp = Step3p5MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                mesh=mesh,
                dtype=dtype,
                swiglu_limit=swiglu_limit,
            )

        self.input_layernorm = GemmaRMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, add_unit_offset=True
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, add_unit_offset=True
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, list]:
        # Pre-norm fused residual (deepseek/M2 pattern).
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states = hidden_states + residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        ln_flag = precision_tracer.jit_pure_callback_record(
            hidden_states, "input_layernorm_output", "INPUT_LAYERNORM", self.layer_id
        )

        hidden_states, kv_fused = self.self_attn(
            positions, hidden_states, forward_batch, token_to_kv_pool
        )

        attn_flag = precision_tracer.jit_pure_callback_record(
            hidden_states, "self_attn_output", "SELF_ATTN", self.layer_id
        )

        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        mlp_flag = precision_tracer.jit_pure_callback_record(
            hidden_states, "mlp_output", "MLP", self.layer_id
        )

        return hidden_states, residual, kv_fused, [ln_flag, attn_flag, mlp_flag]


class Step3p5Model(nnx.Module):
    """Stack of Step3p5DecoderLayer + final GemmaRMSNorm."""

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        attn_impl: str = "flash",
    ) -> None:
        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )

        # Only the first num_hidden_layers decoder layers; MTP layers 45/46/47 excluded.
        self.layers = nnx.data(
            [
                Step3p5DecoderLayer(
                    config=config,
                    layer_id=i,
                    mesh=mesh,
                    dtype=dtype,
                    attn_impl=attn_impl,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = GemmaRMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, add_unit_offset=True
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, list, list]:
        hidden_states = self.embed_tokens(forward_batch.input_ids)
        residual = None
        layers_kv_fused: list = []
        layers_callback_flags: list = []

        for layer in self.layers:
            hidden_states, residual, kv_fused, cb_flags = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
            )
            layers_kv_fused.append(kv_fused)
            layers_callback_flags.extend(cb_flags)

        if residual is not None:
            hidden_states = hidden_states + residual

        hidden_states = self.norm(hidden_states)

        xfmr_flag = precision_tracer.jit_pure_callback_record(
            hidden_states, "transformer_output", "TRANSFORMER"
        )
        layers_callback_flags.append(xfmr_flag)

        return hidden_states, layers_kv_fused, layers_callback_flags


class Step3p5ForCausalLM(nnx.Module):
    """Step 3.5 Flash causal LM (untied lm_head, sigmoid MoE, hybrid SWA)."""

    @classmethod
    def patch_model_config(cls, mc: ModelConfig) -> None:
        # Pin head_dim=128 for KV pool / attention backend sizing.
        mc.head_dim = 128

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        attn_impl: str = "flash",
    ) -> None:
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        self.attn_impl = attn_impl
        logger.info("Step3p5ForCausalLM dtype=%s attn_impl=%s", dtype, attn_impl)

        self.model = Step3p5Model(config, mesh=mesh, dtype=dtype, attn_impl=attn_impl)

        # Untied lm_head (tie_word_embeddings=False).
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
        kv_pool = memory_pools.token_to_kv_pool
        hidden_states, layers_kv_fused, layers_callback_flags = self.model(forward_batch, kv_pool)
        output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        return output, {"token_to_kv_pool": layers_kv_fused}, True, None

    def load_weights(self, model_config: ModelConfig) -> None:
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_weight_mappings(model_config)
        # Step 3.5 has no legitimately-derived params (no MLA absorb, no quant
        # scales, untied lm_head — every nnx.Param must come from the checkpoint).
        loader.load_weights_from_safetensors(weight_mappings, assert_all_assigned=True)
        logger.info("Step3p5 weights loaded successfully!")

    def _create_weight_mappings(self, model_config: ModelConfig) -> dict:
        mappings: dict = {
            "model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.norm.weight": WeightMapping(
                target_path="model.norm.weight",
                sharding=(None,),
                transpose=False,
            ),
            "lm_head.weight": WeightMapping(
                target_path="lm_head.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
        }

        moe_ids = set(_moe_layer_ids(self.config))
        for layer_idx in range(self.config.num_hidden_layers):
            is_moe = layer_idx in moe_ids
            mappings.update(self._create_layer_mappings(layer_idx, is_moe))

        return mappings

    def _create_layer_mappings(self, layer_idx: int, is_moe: bool) -> dict:
        prefix = f"model.layers.{layer_idx}"
        target = f"model.layers.{layer_idx}"
        mappings: dict = {}

        # Layer norms: GemmaRMSNorm stores param as `.weight` (not `.scale`).
        mappings[f"{prefix}.input_layernorm.weight"] = WeightMapping(
            target_path=f"{target}.input_layernorm.weight",
            sharding=(None,),
            transpose=False,
        )
        mappings[f"{prefix}.post_attention_layernorm.weight"] = WeightMapping(
            target_path=f"{target}.post_attention_layernorm.weight",
            sharding=(None,),
            transpose=False,
        )

        # Attention projections (uniform — shapes auto-resolve per layer_type).
        ap = f"{prefix}.self_attn"
        tp = f"{target}.self_attn"
        mappings[f"{ap}.q_proj.weight"] = WeightMapping(
            target_path=f"{tp}.q_proj.weight",
            sharding=(None, "tensor"),
            transpose=True,
            kv_head_padding=False,
        )
        mappings[f"{ap}.k_proj.weight"] = WeightMapping(
            target_path=f"{tp}.k_proj.weight",
            sharding=(None, "tensor"),
            transpose=True,
            kv_head_padding=True,
        )
        mappings[f"{ap}.v_proj.weight"] = WeightMapping(
            target_path=f"{tp}.v_proj.weight",
            sharding=(None, "tensor"),
            transpose=True,
            kv_head_padding=True,
        )
        mappings[f"{ap}.o_proj.weight"] = WeightMapping(
            target_path=f"{tp}.o_proj.weight",
            sharding=("tensor", None),
            transpose=True,
            kv_head_padding=False,
        )
        mappings[f"{ap}.g_proj.weight"] = WeightMapping(
            target_path=f"{tp}.g_proj.weight",
            sharding=(None, "tensor"),
            transpose=True,
            kv_head_padding=False,
        )
        # QK-norm: GemmaRMSNorm → .weight; not head-sharded (head_dim only).
        mappings[f"{ap}.q_norm.weight"] = WeightMapping(
            target_path=f"{tp}.q_norm.weight",
            sharding=(None,),
            transpose=False,
        )
        mappings[f"{ap}.k_norm.weight"] = WeightMapping(
            target_path=f"{tp}.k_norm.weight",
            sharding=(None,),
            transpose=False,
        )

        if not is_moe:
            # Dense FFN (layers 0-2): mlp.{gate,up,down}_proj.
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
            return mappings

        # MoE router gate: [288, 4096] in HF → moe_gate.kernel [4096, 288] after transpose.
        mappings[f"{prefix}.moe.gate.weight"] = WeightMapping(
            target_path=f"{target}.mlp.moe_gate.kernel",
            sharding=(None, None),
            transpose=True,
        )
        # Router bias: float32 [288].
        mappings[f"{prefix}.moe.router_bias"] = WeightMapping(
            target_path=f"{target}.mlp.moe_gate.bias",
            sharding=(None,),
            transpose=False,
        )

        # Stacked expert weights via Plan 1 loader (__MOE_EXPERTS__ pattern).
        # Checkpoint stores pre-stacked [E, out, in]; the loader detects 3D + shape[0]==E
        # (single-source branch) and calls _create_prestacked_moe_lazy_tensor, which applies
        # transpose(0,2,1): [E, out, in] → [E, in, out] matching EPMoE wi_0/wi_1/wo layout.
        # Sharding ("expert", None, None) shards axis 0 on the moe mesh's expert axis so each
        # device loads ONLY its experts (num_experts/ep_size), never the full stacked tensor —
        # mirrors minimax_m2 / deepseek_v3 / qwen3_moe. The non-expert dims stay unsharded, so
        # transpose(0,2,1) keeps ("expert", None, None) (no mesh-context conflict); the
        # assignment to EPMoE params reshards to their ("expert", None, "tensor") layout.
        # REQUIRES --ep-size > 1 at runtime, else the expert axis has size 1 and the full
        # stacked tensor is replicated on every device (HBM blow-up on the real 288-expert model).
        for src_proj, tgt_name in [
            ("gate_proj", "wi_0"),
            ("up_proj", "wi_1"),
            ("down_proj", "wo"),
        ]:
            tgt_base = f"{target}.mlp.experts.{tgt_name}"
            src_key = f"{prefix}.moe.{src_proj}.weight"
            mappings[f"__MOE_EXPERTS__{tgt_base}"] = WeightMapping(
                target_path=[tgt_base, src_key],
                sharding=("expert", None, None),
                transpose=True,
            )

        # Shared expert (always-on): share_expert.{gate,up,down}_proj (HF path, no "moe." prefix).
        for proj, sharding in [
            ("gate_proj", (None, "tensor")),
            ("up_proj", (None, "tensor")),
            ("down_proj", ("tensor", None)),
        ]:
            mappings[f"{prefix}.share_expert.{proj}.weight"] = WeightMapping(
                target_path=f"{target}.mlp.shared_experts.{proj}.weight",
                sharding=sharding,
                transpose=True,
            )

        return mappings


EntryClass = [Step3p5ForCausalLM]
