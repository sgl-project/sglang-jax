"""Ling3 (BailingMoE V3) — hybrid KDA + MLA + MoE.

Layer pattern (Ling3-Tiny: 24 main layers, layer_group_size=4):
- layer 0: dense MLP, KDA attention
- layer i in {1..23}: MoE MLP
  - if (i+1) % 4 == 0 → MLA attention (0-based [3,7,11,15,19,23])
  - else              → KDA attention
- layer 24: MTP (skipped at inference for the spike).

Reuses:
- DeepseekV3Attention for MLA (subclassed as BailingMLA to insert head-wise gate
  before o_proj — relies on the P1.1 _attention_core / _apply_o_proj split).
- KimiDeltaAttention's KDA wiring as a starting point, but Ling3 has
  no_kda_lora=True so the f_a/f_b and g_a/g_b LoRA pairs collapse into single
  f_proj / g_proj direct projections.
- BailingMoE-style MoE blocks: mlp.gate (GateLogit), TopK, EPMoE, shared_experts.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.bailing_moe_v3 import BailingMoeV3Config
from sgl_jax.srt.configs.model_config import AttentionArch, ModelConfig, MoEBackend
from sgl_jax.srt.eplb.expert_location import ExpertLocationMetadata
from sgl_jax.srt.layers.attention.fla.gated_rmsnorm import GatedRMSNorm
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
from sgl_jax.srt.layers.fused_moe import FusedEPMoE
from sgl_jax.srt.layers.gate import GateLogit, TopK
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import EPMoE, create_moe_weights_mapping
from sgl_jax.srt.layers.radix_linear_attention import RadixLinearAttention
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.deepseek_v3 import DeepseekV3Attention
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class BailingMoeV3MLP(nnx.Module):
    """Standard SwiGLU MLP — used both as the dense layer-0 MLP and as the
    shared-expert path for MoE layers."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        swiglu_limit: float | None = None,
    ):
        super().__init__()
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
            # Matches maxtext/src/MaxText/layers/linears.py:500-543:
            # gate (post-silu) clamped single-sided, up clamped double-sided.
            gate = jnp.clip(gate, max=self.swiglu_limit)
            up = jnp.clip(up, -self.swiglu_limit, self.swiglu_limit)
        output, _ = self.down_proj(gate * up)
        return output


# ---------------------------------------------------------------------------
# MLA: head-wise gated o_proj
# ---------------------------------------------------------------------------


class BailingMLA(DeepseekV3Attention):
    """MLA with a head-wise sigmoid gate inserted before o_proj.

    Reuses parent _attention_core (Q/KV/RoPE/attn). Only _apply_o_proj is
    overridden — multiplies the pre-o_proj tensor by sigmoid(g_proj(hidden_states))
    broadcast over (num_heads, v_head_dim), then defers to parent o_proj.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        rope_interleave: bool = True,
        max_position_embeddings: int = 8192,
        dtype: jnp.dtype = jnp.bfloat16,
        use_absorbed: bool = True,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            mesh=mesh,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rope_interleave=rope_interleave,
            max_position_embeddings=max_position_embeddings,
            dtype=dtype,
            use_absorbed=use_absorbed,
            skip_rope=False,
        )
        # Head-wise gate: hidden_size -> num_heads (one scalar gate per head).
        # Replicated along the head dimension so each TP shard produces all gates.
        self.g_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_heads,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="g_proj",
        )

    def _apply_o_proj(
        self,
        pre_o_proj: jax.Array,
        hidden_states: jax.Array,
    ) -> jax.Array:
        gate, _ = self.g_proj(hidden_states)
        gate = jax.nn.sigmoid(gate.astype(jnp.float32)).astype(pre_o_proj.dtype)
        # pre_o_proj is flat [T, num_heads * v_head_dim]; gate is [T, num_heads].
        T = pre_o_proj.shape[0]
        gated = pre_o_proj.reshape(T, self.num_heads, self.v_head_dim) * gate[:, :, None]
        gated = gated.reshape(T, self.num_heads * self.v_head_dim)
        return super()._apply_o_proj(gated, hidden_states)


# ---------------------------------------------------------------------------
# KDA: Kimi-style delta attention without f/g LoRA
# ---------------------------------------------------------------------------


class BailingKDAAttention(nnx.Module):
    """KDA layer for Ling3 (no_kda_lora=True).

    Mirrors KimiDeltaAttention except:
    - f_a_proj/f_b_proj collapse into a single f_proj (hidden -> projection_size).
    - g_a_proj/g_b_proj collapse into a single g_proj (hidden -> projection_size).
    - RadixLinearAttention is constructed with kda_lower_bound=config.kda_lower_bound
      (=-5.0 for Ling3-Tiny) so the chunk_kda kernel and decode _fused_kda_gate
      switch to kda_lower_bound * sigmoid(exp(A) * (g + dt_bias)).
    """

    def __init__(
        self,
        config: BailingMoeV3Config,
        layer_idx: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        super().__init__()
        linear_config = config.linear_attn_config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.conv_size = linear_config["short_conv_kernel_size"]
        self.head_dim = linear_config["head_dim"]
        self.k_head_dim = self.head_dim
        self.v_head_dim = config.v_head_dim or self.head_dim
        self.num_heads = linear_config["num_heads"]
        self.num_k_heads = self.num_heads
        self.num_v_heads = self.num_heads
        self.projection_k_size = self.num_k_heads * self.k_head_dim
        self.projection_size = self.num_heads * self.head_dim
        self.rms_norm_eps = config.rms_norm_eps

        # Q/K/V projections — sharded along the head axis.
        self.q_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.projection_k_size,
            mesh=mesh, use_bias=False, params_dtype=dtype,
            kernel_axes=(None, "tensor"), scope_name="q_proj",
        )
        self.k_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.projection_k_size,
            mesh=mesh, use_bias=False, params_dtype=dtype,
            kernel_axes=(None, "tensor"), scope_name="k_proj",
        )
        self.v_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.projection_size,
            mesh=mesh, use_bias=False, params_dtype=dtype,
            kernel_axes=(None, "tensor"), scope_name="v_proj",
        )

        # Short conv weight containers — never called, only their .weight.value
        # is read by short_convolution. Layout [D, K]; D sharded on "tensor".
        self.q_conv1d = LinearBase(
            self.projection_k_size, self.conv_size,
            mesh, use_bias=False, params_dtype=dtype,
            kernel_axes=("tensor", None), scope_name="q_conv1d",
        )
        self.k_conv1d = LinearBase(
            self.projection_k_size, self.conv_size,
            mesh, use_bias=False, params_dtype=dtype,
            kernel_axes=("tensor", None), scope_name="k_conv1d",
        )
        self.v_conv1d = LinearBase(
            self.projection_size, self.conv_size,
            mesh, use_bias=False, params_dtype=dtype,
            kernel_axes=("tensor", None), scope_name="v_conv1d",
        )

        # KDA recurrent params.
        self.A_log = nnx.Param(
            jnp.zeros(
                (1, 1, self.num_heads, 1), dtype=jnp.float32,
                out_sharding=P(None, None, "tensor", None),
            )
        )
        self.dt_bias = nnx.Param(
            jnp.zeros(
                (self.projection_size,), dtype=jnp.float32,
                out_sharding=P("tensor"),
            )
        )

        # Decay-gate (f) and output-gate (g): direct hidden -> projection_size,
        # NO LoRA bottleneck (no_kda_lora=True for Ling3).
        self.f_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.projection_size,
            mesh=mesh, use_bias=False, params_dtype=dtype,
            kernel_axes=(None, "tensor"), scope_name="f_proj",
        )
        self.g_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.projection_size,
            mesh=mesh, use_bias=False, params_dtype=dtype,
            kernel_axes=(None, "tensor"), scope_name="g_proj",
        )
        self.b_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.num_heads,
            mesh=mesh, use_bias=False, params_dtype=dtype,
            kernel_axes=(None, "tensor"), scope_name="b_proj",
        )

        self.o_norm = GatedRMSNorm(self.head_dim, epsilon=self.rms_norm_eps)
        self.o_proj = LinearBase(
            input_size=self.projection_size,
            output_size=self.hidden_size,
            mesh=mesh, use_bias=False, params_dtype=dtype,
            kernel_axes=("tensor", None), scope_name="o_proj",
        )

        self.attn = RadixLinearAttention(
            layer_id=self.layer_idx,
            num_q_heads=self.num_k_heads,
            num_k_heads=self.num_k_heads,
            num_v_heads=self.num_v_heads,
            head_q_dim=self.k_head_dim,
            head_k_dim=self.k_head_dim,
            head_v_dim=self.v_head_dim,
            q_conv1d=self.q_conv1d,
            k_conv1d=self.k_conv1d,
            v_conv1d=self.v_conv1d,
            bias=None,
            activation=jax.nn.silu,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            kda_lower_bound=config.kda_lower_bound,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        recurrent_state_pool,
    ):
        del positions  # KDA is position-agnostic.

        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        raw_gate, _ = self.f_proj(hidden_states)
        raw_gate = raw_gate.reshape(hidden_states.shape[0], self.num_heads, self.head_dim)
        beta = jax.nn.sigmoid(self.b_proj(hidden_states)[0].astype(jnp.float32))

        o, recurrent_state_pool = self.attn(
            forward_batch, q, k, v, raw_gate, beta, recurrent_state_pool,
        )
        o = o.reshape(hidden_states.shape[0], self.num_heads, self.head_dim)

        output_gate, _ = self.g_proj(hidden_states)
        output_gate = output_gate.reshape(
            hidden_states.shape[0], self.num_heads, self.head_dim
        )
        # GatedRMSNorm: RMSNorm(o) * sigmoid(output_gate).
        o = self.o_norm(o, output_gate).reshape(
            hidden_states.shape[0], self.projection_size
        )
        o, _ = self.o_proj(o)
        return o, recurrent_state_pool


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class BailingMoeV3DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: BailingMoeV3Config,
        mesh: jax.sharding.Mesh,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.is_kda = config.is_kda_layer(layer_idx)

        # Attention
        if self.is_kda:
            self.self_attn = BailingKDAAttention(
                config=config,
                layer_idx=layer_idx,
                mesh=mesh,
                dtype=dtype,
            )
        else:
            self.self_attn = BailingMLA(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                mesh=mesh,
                layer_id=layer_idx,
                rope_theta=config.rope_theta,
                rope_scaling=config.rope_scaling,
                rope_interleave=config.rope_interleave,
                max_position_embeddings=config.max_position_embeddings,
                dtype=dtype,
            )

        # MLP — layer 0 dense, layers 1..23 MoE.
        self.is_moe_layer = layer_idx >= config.first_k_dense_replace
        self.moe_backend = getattr(config, "moe_backend", MoEBackend.EPMOE)
        self.use_fused = self.moe_backend == "fused"

        if not self.is_moe_layer:
            self.mlp = BailingMoeV3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                mesh=mesh,
                dtype=dtype,
            )
            self.moe_gate = None
        else:
            # Router. router_dtype="fp32" so GateLogit weight/bias are fp32.
            router_dtype = jnp.float32 if config.router_dtype == "fp32" else dtype
            self.moe_gate = GateLogit(
                input_size=config.hidden_size,
                num_experts=config.num_experts,
                enable_expert_bias=config.moe_router_enable_expert_bias,
                weight_dtype=router_dtype,
                score_func=config.score_function,
            )
            self.topk = TopK(
                topk=config.num_experts_per_tok,
                renormalize=config.norm_topk_prob,
                num_expert_group=config.n_group,
                topk_group=config.topk_group,
                routed_scaling_factor=config.routed_scaling_factor,
                layer_id=layer_idx,
            )

            if self.use_fused:
                if (
                    config.expert_swiglu_limit(layer_idx) is not None
                    or config.shared_expert_swiglu_limit(layer_idx) is not None
                ):
                    raise NotImplementedError(
                        f"layer {layer_idx}: FusedEPMoE path does not yet implement "
                        "SwiGLU clamp; fused kernel folds silu*up inside its fused "
                        "moe op. Use the EPMoE backend (default) for Flash, or "
                        "extend fused_moe to plumb the clamp before enabling fused."
                    )
                self.experts = FusedEPMoE(
                    hidden_size=config.hidden_size,
                    num_experts=config.num_experts,
                    num_experts_per_tok=config.num_experts_per_tok,
                    intermediate_dim=config.moe_intermediate_size,
                    mesh=mesh,
                    weight_dtype=dtype,
                    dtype=dtype,
                    layer_id=layer_idx,
                    ep_size=getattr(config, "ep_size", 1),
                    activation_fn=config.score_function,
                    renormalize_topk_logits=config.norm_topk_prob,
                    routed_scaling_factor=config.routed_scaling_factor,
                    use_grouped_topk=config.n_group > 0,
                    num_groups=config.n_group,
                    top_k_groups=config.topk_group,
                    num_shared_experts=config.num_shared_experts,
                    moe_shared_expert_intermediate_size=config.moe_shared_expert_intermediate_size,
                )
                self.shared_experts = None
            else:
                self.experts = EPMoE(
                    hidden_size=config.hidden_size,
                    num_experts=config.num_experts,
                    num_experts_per_tok=config.num_experts_per_tok,
                    intermediate_dim=config.moe_intermediate_size,
                    mesh=mesh,
                    weight_dtype=dtype,
                    dtype=dtype,
                    layer_id=layer_idx,
                    ep_size=getattr(config, "ep_size", 1),
                    swiglu_limit=config.expert_swiglu_limit(layer_idx),
                )
                if config.num_shared_experts > 0:
                    self.shared_experts = BailingMoeV3MLP(
                        hidden_size=config.hidden_size,
                        intermediate_size=config.moe_shared_expert_intermediate_size
                        * config.num_shared_experts,
                        mesh=mesh,
                        dtype=dtype,
                        swiglu_limit=config.shared_expert_swiglu_limit(layer_idx),
                    )
                else:
                    self.shared_experts = None

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
        memory_pools,
        residual: jax.Array | None = None,
        dispatch_info: ExpertLocationMetadata | None = None,
    ):
        # Pre-norm residual pattern.
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        # Attention.
        if self.is_kda:
            kv_pool = memory_pools.recurrent_state_pool
        else:
            kv_pool = memory_pools.token_to_kv_pool
        hidden_states, kv_fused = self.self_attn(
            positions, hidden_states, forward_batch, kv_pool,
        )

        # Post-attention residual + norm.
        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP.
        if self.is_moe_layer:
            shared_output = self.shared_experts(hidden_states) if self.shared_experts is not None else None

            router_logits = self.moe_gate(hidden_states)
            correction_bias = (
                self.moe_gate.bias.value if self.moe_gate.bias is not None else None
            )
            topk_weights, topk_ids = self.topk(
                router_logits, correction_bias, dispatch_info=dispatch_info
            )
            if self.use_fused:
                token_valid_mask = forward_batch.get_token_valid_mask(
                    hidden_states.shape[0]
                )
                topk_ids = jnp.where(token_valid_mask[:, None], topk_ids, -1)

            hidden_states = self.experts(hidden_states, topk_weights, topk_ids)
            if shared_output is not None:
                hidden_states = hidden_states + shared_output
        else:
            hidden_states = self.mlp(hidden_states)
            topk_ids = None

        return hidden_states, residual, kv_fused, topk_ids


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class BailingMoeV3Model(nnx.Module):
    def __init__(
        self,
        config: BailingMoeV3Config,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
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
                BailingMoeV3DecoderLayer(
                    config=config, mesh=mesh, layer_idx=i, dtype=dtype,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=dtype,
            scope_name="norm",
        )

    def __call__(self, forward_batch: ForwardBatch, memory_pools):
        hidden_states = self.embed_tokens(forward_batch.input_ids)

        residual = None
        layers_kv_fused = []
        layers_recurrent_buffers = []
        layers_conv_buffers = []
        layers_topk_ids = []

        for layer in self.layers:
            hidden_states, residual, attn_state, topk_ids = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                memory_pools,
                residual,
                dispatch_info=forward_batch.expert_location_metadata,
            )
            if layer.is_kda:
                rec_buf, conv_buf_list = attn_state
                layers_recurrent_buffers.append(rec_buf)
                layers_conv_buffers.append(conv_buf_list)
            else:
                layers_kv_fused.append(attn_state)
            layers_topk_ids.append(topk_ids)

        if residual is not None:
            hidden_states += residual
        hidden_states = self.norm(hidden_states)
        return (
            hidden_states,
            layers_kv_fused,
            (layers_recurrent_buffers, layers_conv_buffers),
            layers_topk_ids,
        )


# ---------------------------------------------------------------------------
# CausalLM wrapper + weight loader
# ---------------------------------------------------------------------------


class BailingMoeV3ForCausalLM(nnx.Module):
    @classmethod
    def patch_model_config(cls, mc: ModelConfig) -> None:
        mc.attention_arch = AttentionArch.MLA
        mc.head_dim = (
            mc.hf_text_config.qk_nope_head_dim + mc.hf_text_config.qk_rope_head_dim
        )

    def __init__(
        self,
        config: BailingMoeV3Config,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        self.model = BailingMoeV3Model(config=config, mesh=mesh, dtype=dtype)

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
        memory_pools,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, layers_kv_fused, layers_recurrent_state, layers_topk_ids = (
            self.model(forward_batch, memory_pools)
        )
        if not getattr(self.config, "tie_word_embeddings", False):
            output = self.logits_processor(
                hidden_states, self.lm_head, logits_metadata
            )
        else:
            output = self.logits_processor(
                hidden_states, self.model.embed_tokens, logits_metadata
            )
        return (
            output,
            {
                "token_to_kv_pool": layers_kv_fused,
                "recurrent_state_pool": layers_recurrent_state,
            },
            True,
            layers_topk_ids,
        )

    # -----------------------------------------------------------------
    # Weight loading
    # -----------------------------------------------------------------

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_weight_mappings()
        # Strictness check (per plan §2): every ckpt key must be either covered
        # by a mapping or pruned by WeightLoader._is_excluded_layer_weight (which
        # auto-drops layers >= num_hidden_layers — that's how MTP at layer 24
        # gets skipped). Raise on any other unmapped key BEFORE invoking the
        # loader so we don't silently drop weights.
        self._assert_full_ckpt_coverage(loader, weight_mappings)
        loader.load_weights_from_safetensors(weight_mappings)
        for layer in self.model.layers:
            if not layer.is_kda:
                layer.self_attn.post_load_weights()
        logger.info("Ling3 weights loaded successfully.")

    def _assert_full_ckpt_coverage(
        self,
        loader: WeightLoader,
        weight_mappings: dict,
    ) -> None:
        import re

        ckpt_keys = set(loader._scan_weight_info().keys())

        covered: set[str] = set()
        for mapping_key in weight_mappings:
            if mapping_key.startswith("__MOE_EXPERTS__"):
                # Aggregated MoE entries: mapping.target_path is [target, *expert_keys].
                m = weight_mappings[mapping_key]
                expert_keys = m.target_path[1:] if isinstance(m.target_path, list) else []
                covered.update(k for k in expert_keys if k in ckpt_keys)
                continue
            if "*" in mapping_key:
                pattern = re.escape(mapping_key).replace(r"\*", r"(.*?)")
                covered.update(k for k in ckpt_keys if re.fullmatch(pattern, k))
            else:
                if mapping_key in ckpt_keys:
                    covered.add(mapping_key)

        # Loader auto-skips ckpt keys for layers >= num_hidden_layers (MTP).
        auto_skipped = {k for k in ckpt_keys if loader._is_excluded_layer_weight(k)}

        unmapped = ckpt_keys - covered - auto_skipped
        if unmapped:
            sample = sorted(unmapped)[:10]
            raise RuntimeError(
                f"Ling3 loader: {len(unmapped)} ckpt key(s) have no mapping and "
                f"were not auto-skipped. First {len(sample)}: {sample}"
            )

    def _create_weight_mappings(self) -> dict:
        mappings: dict[str, WeightMapping] = {
            # Ling3 ckpt uses model.word_embeddings.weight (NOT embed_tokens).
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

        num_layers = self.config.num_hidden_layers
        first_k_dense_replace = self.config.first_k_dense_replace
        moe_backend = getattr(self.config, "moe_backend", "epmoe")

        for layer_idx in range(num_layers):
            is_dense = layer_idx < first_k_dense_replace
            is_kda = self.config.is_kda_layer(layer_idx)
            mappings.update(
                self._create_layer_mappings(
                    layer_idx,
                    is_dense=is_dense,
                    is_kda=is_kda,
                    moe_backend=moe_backend,
                )
            )

        return mappings

    def _create_layer_mappings(
        self, layer_idx: int, *, is_dense: bool, is_kda: bool, moe_backend: str
    ) -> dict:
        prefix = f"model.layers.{layer_idx}"
        target_prefix = f"model.layers.{layer_idx}"
        mappings: dict[str, WeightMapping] = {
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

        # ----- attention -----
        # Ling3 ckpt uses .attention.* (NOT .self_attn.*) as the source prefix.
        attn_src = f"{prefix}.attention"
        attn_target = f"{target_prefix}.self_attn"

        if is_kda:
            for proj in ("q_proj", "k_proj", "v_proj", "f_proj", "g_proj", "b_proj"):
                mappings[f"{attn_src}.{proj}.weight"] = WeightMapping(
                    target_path=f"{attn_target}.{proj}.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                )
            mappings[f"{attn_src}.o_proj.weight"] = WeightMapping(
                target_path=f"{attn_target}.o_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            )
            conv_size = self.config.linear_attn_config["short_conv_kernel_size"]
            num_heads = self.config.linear_attn_config["num_heads"]
            head_dim = self.config.linear_attn_config["head_dim"]
            projection_size = num_heads * head_dim
            for conv_name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                mappings[f"{attn_src}.{conv_name}.weight"] = WeightMapping(
                    # conv1d weights live under self_attn.attn (RadixLinearAttention).
                    target_path=f"{attn_target}.attn.{conv_name}.weight",
                    sharding=("tensor", None),
                    transpose=False,
                    reshape=(projection_size, conv_size),
                )
            mappings[f"{attn_src}.o_norm.weight"] = WeightMapping(
                target_path=f"{attn_target}.o_norm.weight",
                sharding=(None,),
                transpose=False,
            )
            mappings[f"{attn_src}.dt_bias"] = WeightMapping(
                target_path=f"{attn_target}.attn.dt_bias",
                sharding=("tensor",),
                transpose=False,
            )
            mappings[f"{attn_src}.A_log"] = WeightMapping(
                target_path=f"{attn_target}.A_log",
                sharding=(None, None, "tensor", None),
                transpose=False,
                # ckpt stores [H]; KDA layer expects [1, 1, H, 1] (Kimi shape).
                reshape=(1, 1, self.config.num_attention_heads, 1),
            )
        else:
            # MLA (Q[-LoRA] + KV-LoRA + head-wise gate).
            # Q projection: Tiny (q_lora_rank=256) uses Q-LoRA (q_a_proj +
            # q_a_layernorm + q_b_proj); Flash (q_lora_rank=null) uses a flat
            # q_proj. Semantics aligned with the q-LoRA / flat-q switch in
            # deepseek_v3.py::_create_weight_mappings.
            if self.config.q_lora_rank is None:
                mappings[f"{attn_src}.q_proj.weight"] = WeightMapping(
                    target_path=f"{attn_target}.q_proj.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                )
            else:
                mappings[f"{attn_src}.q_a_proj.weight"] = WeightMapping(
                    target_path=f"{attn_target}.q_a_proj.weight",
                    sharding=(None, None),
                    transpose=True,
                )
                mappings[f"{attn_src}.q_a_layernorm.weight"] = WeightMapping(
                    target_path=f"{attn_target}.q_a_layernorm.scale",
                    sharding=(None,),
                    transpose=False,
                )
                mappings[f"{attn_src}.q_b_proj.weight"] = WeightMapping(
                    target_path=f"{attn_target}.q_b_proj.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                )
            mappings[f"{attn_src}.kv_a_proj_with_mqa.weight"] = WeightMapping(
                target_path=f"{attn_target}.kv_a_proj.weight",
                sharding=(None, None),
                transpose=True,
            )
            mappings[f"{attn_src}.kv_a_layernorm.weight"] = WeightMapping(
                target_path=f"{attn_target}.kv_a_layernorm.scale",
                sharding=(None,),
                transpose=False,
            )
            mappings[f"{attn_src}.kv_b_proj.weight"] = WeightMapping(
                target_path=f"{attn_target}.kv_b_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            )
            # Ling3 head-wise MLA gate: ckpt [16,1536] -> JAX [1536,16].
            mappings[f"{attn_src}.g_proj.weight"] = WeightMapping(
                target_path=f"{attn_target}.g_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            )
            # MLA output projection: ckpt name is `dense`, our target is o_proj.
            mappings[f"{attn_src}.dense.weight"] = WeightMapping(
                target_path=f"{attn_target}.o_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            )

        # ----- FFN -----
        if is_dense:
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
            # MoE — Ling3 ckpt uses mlp.gate / mlp.gate.expert_bias /
            # mlp.experts.{i} / mlp.shared_experts.
            mappings[f"{prefix}.mlp.gate.weight"] = WeightMapping(
                target_path=f"{target_prefix}.moe_gate.kernel",
                sharding=(None, None),
                transpose=True,
            )
            mappings[f"{prefix}.mlp.gate.expert_bias"] = WeightMapping(
                target_path=f"{target_prefix}.moe_gate.bias",
                sharding=(None,),
            )

            # Routed experts. The Ling3 ckpt stores experts under `mlp.experts.{i}.{proj}.weight`,
            # but our nnx tree exposes them as `experts.{wi_0|wi_1|wo}` (EPMoE) or
            # `experts.{w1|w3|w2}` (FusedEPMoE). Source and target prefixes don't share
            # the moe_path segment, so we build the __MOE_EXPERTS__ mappings manually
            # rather than going through create_moe_weights_mapping (which assumes
            # source_path = `{prefix}.{moe_path}.experts.{i}...` and
            # target_path = `{target_prefix}.{moe_path}.{wi_0|...}` with the same moe_path).
            from sgl_jax.srt.eplb.expert_location import (
                get_global_expert_location_metadata,
            )

            metadata = get_global_expert_location_metadata()
            phy_to_log = None
            if metadata is not None:
                physical_to_logical_map = np.array(
                    jax.device_get(metadata.physical_to_logical_map)
                )
                phy_to_log = physical_to_logical_map[layer_idx]

            if moe_backend == "fused":
                expert_target_map = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
                fused_sharding = (("data", "tensor"), None, None)
            else:  # epmoe
                expert_target_map = {"gate_proj": "wi_0", "up_proj": "wi_1", "down_proj": "wo"}

            for source_name, target_name in expert_target_map.items():
                if moe_backend == "epmoe":
                    sharding = (
                        ("expert", "tensor", None) if target_name == "wo"
                        else ("expert", None, "tensor")
                    )
                else:
                    sharding = fused_sharding
                target_path_base = f"{target_prefix}.experts.{target_name}"
                expert_keys = [
                    f"{prefix}.mlp.experts.{i}.{source_name}.weight"
                    for i in range(self.config.num_experts)
                ]
                mappings[f"__MOE_EXPERTS__{target_path_base}"] = WeightMapping(
                    target_path=[target_path_base] + expert_keys,
                    sharding=sharding,
                    transpose=True,
                    physical_to_logical_map=phy_to_log,
                )

            # Shared experts.
            if self.config.num_shared_experts > 0:
                for proj_name, sharding in [
                    ("gate_proj", (None, "tensor")),
                    ("up_proj", (None, "tensor")),
                    ("down_proj", ("tensor", None)),
                ]:
                    mappings[f"{prefix}.mlp.shared_experts.{proj_name}.weight"] = (
                        WeightMapping(
                            target_path=f"{target_prefix}.shared_experts.{proj_name}.weight",
                            sharding=sharding,
                            transpose=True,
                        )
                    )

        return mappings


EntryClass = BailingMoeV3ForCausalLM
