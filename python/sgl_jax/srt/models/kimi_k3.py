"""Kimi-K3 for sglang-jax.

K3 = Kimi-Linear + two architectural additions, so this reuses ``kimi_linear``'s KDA, MLA and MoE
wiring and adds only what differs:

* **SITU** activation in the dense MLP (``hidden_act == "situ"``).
* **AttnRes** — instead of a plain additive residual, each layer mixes a *learned softmax-weighted
  sum* over depth-checkpointed residuals. This is why ``KimiK3DecoderLayer`` cannot reuse
  ``KimiDecoderLayer``: sglang-jax's fused-residual optimization (carry ``residual``, add it lazily
  at the next norm) is incompatible with a residual that is recomputed by a learned mixer.

Architecture (from the released config.json): 93 layers, ``kda_layers`` lists 69 => 24 gated MLA,
3:1 interleave; hidden 7168; 896 experts, top-16, +2 shared; ``attn_res_block_size`` 12 => at most
8 AttnRes candidates; MXFP4 weights at group_size 32.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.configs.kimi_k3 import KimiK3Config
from sgl_jax.srt.layers.gate import GateLogit, TopK
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.moe import EPMoE
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.kimi_k3_layers import AttentionResidual, situ_and_mul
from sgl_jax.srt.models.kimi_k3_residual import initial_block_residuals
from sgl_jax.srt.models.kimi_linear import KimiDeltaAttention
from sgl_jax.srt.models.deepseek_v3 import DeepseekV3Attention as KimiMLAAttention


class KimiK3MLP(nnx.Module):
    """Dense MLP with K3's SITU gate.

    Kimi-Linear's ``KimiMLP`` hardcodes ``jax.nn.silu``; K3 ships ``hidden_act: "situ"`` with
    ``beta=4.0`` and ``linear_beta=25.0``, so the activation is a parameter here. gate and up are
    concatenated before the activation because SITU consumes a single tensor and splits it, which
    is how the reference's ``SituAndMul`` is defined.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mesh: jax.sharding.Mesh,
        situ_beta: float | None = None,
        situ_linear_beta: float | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        super().__init__()
        self.situ_beta = situ_beta
        self.situ_linear_beta = situ_linear_beta
        mk = lambda i, o, ax, name: LinearBase(  # noqa: E731
            input_size=i, output_size=o, kernel_axes=ax, use_bias=False,
            params_dtype=dtype, mesh=mesh, scope_name=name)
        self.gate_proj = mk(hidden_size, intermediate_size, (None, "tensor"), "gate_proj")
        self.up_proj = mk(hidden_size, intermediate_size, (None, "tensor"), "up_proj")
        self.down_proj = mk(intermediate_size, hidden_size, ("tensor", None), "down_proj")

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        gate, _ = self.gate_proj(hidden_states)
        up, _ = self.up_proj(hidden_states)
        if self.situ_beta is not None:
            act = situ_and_mul(
                jnp.concatenate([gate, up], axis=-1), self.situ_beta, self.situ_linear_beta
            )
        else:
            act = jax.nn.silu(gate) * up
        out, _ = self.down_proj(act)
        return out


class KimiK3DecoderLayer(nnx.Module):
    """One K3 decoder layer: KDA-or-MLA attention, two AttnRes mixers, MoE-or-dense FFN."""

    def __init__(
        self,
        config: KimiK3Config,
        mesh: jax.sharding.Mesh,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.is_kda = config.is_kda_layer(layer_idx)
        self.attn_res_block_size = config.attn_res_block_size

        if self.is_kda:
            self.self_attn = KimiDeltaAttention(
                config=config, mesh=mesh, dtype=dtype, layer_idx=layer_idx
            )
        else:
            self.self_attn = KimiMLAAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                mesh=mesh,
                layer_id=layer_idx,
                dtype=dtype,
                use_absorbed=getattr(config, "use_absorbed_mla", True),
                skip_rope=config.mla_use_nope,
            )

        is_moe = (
            getattr(config, "num_experts", None)
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        )
        self.is_moe_layer = bool(is_moe)

        if not self.is_moe_layer:
            self.mlp = KimiK3MLP(
                config.hidden_size, config.intermediate_size, mesh,
                config.activation_situ_beta, config.activation_situ_linear_beta, dtype)
            self.moe_gate = None
            self.shared_experts = None
        else:
            self.mlp = None
            self.moe_gate = GateLogit(
                input_size=config.hidden_size,
                num_experts=config.num_experts,
                enable_expert_bias=True,
                weight_dtype=dtype,
                score_func=config.moe_router_activation_func,
            )
            self.topk = TopK(
                topk=config.num_experts_per_token,
                renormalize=config.moe_renormalize,
                num_expert_group=config.num_expert_group,
                topk_group=config.topk_group,
                routed_scaling_factor=config.routed_scaling_factor,
                layer_id=layer_idx,
                mesh=mesh,
            )
            self.block_sparse_moe = EPMoE(
                hidden_size=config.hidden_size,
                num_experts=config.num_experts,
                num_experts_per_tok=config.num_experts_per_token,
                intermediate_dim=config.moe_intermediate_size,
                mesh=mesh,
                weight_dtype=dtype,
                dtype=dtype,
                layer_id=layer_idx,
                ep_size=getattr(config, "ep_size", 1),
            )
            self.shared_experts = (
                KimiK3MLP(
                    config.hidden_size,
                    config.moe_intermediate_size * config.num_shared_experts,
                    mesh, config.activation_situ_beta, config.activation_situ_linear_beta, dtype)
                if config.num_shared_experts > 0 else None
            )

        self.input_layernorm = RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, param_dtype=dtype,
            scope_name="input_layernorm")
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, param_dtype=dtype,
            scope_name="post_attention_layernorm")

        if config.uses_attn_res:
            self.self_attention_res = AttentionResidual(
                config.hidden_size, config.rms_norm_eps, mesh, dtype, "self_attention_res")
            self.mlp_res = AttentionResidual(
                config.hidden_size, config.rms_norm_eps, mesh, dtype, "mlp_res")
        else:
            self.self_attention_res = None
            self.mlp_res = None

    def _ffn(self, hidden_states, forward_batch, dispatch_info):
        if not self.is_moe_layer:
            return self.mlp(hidden_states), None
        shared = self.shared_experts(hidden_states) if self.shared_experts is not None else None
        router_logits = self.moe_gate(hidden_states)
        bias = self.moe_gate.bias.value if self.moe_gate.bias is not None else None
        topk_w, topk_ids = self.topk(router_logits, bias, dispatch_info=dispatch_info)
        out = self.block_sparse_moe(hidden_states, topk_w, topk_ids)
        if shared is not None:
            out = out + shared
        return out, topk_ids

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        memory_pools,
        prefix_sum: jax.Array | None = None,
        block_residuals: jax.Array | None = None,
        dispatch_info=None,
    ):
        """Follows ``KimiDecoderLayer.forward`` in the PyTorch reference exactly.

        Returns ``(hidden_states, prefix_sum, block_residuals, kv_fused, topk_ids)``. The plain
        pre-norm path is kept for configs without AttnRes so this class covers both.
        """
        kv_pool = (memory_pools.recurrent_state_pool if self.is_kda
                   else memory_pools.token_to_kv_pool)

        if self.attn_res_block_size is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, kv_fused = self.self_attn(
                positions, hidden_states, forward_batch, kv_pool)
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            out, topk_ids = self._ffn(hidden_states, forward_batch, dispatch_info)
            return residual + out, None, None, kv_fused, topk_ids

        # --- AttnRes path -------------------------------------------------------------
        prefix_sum = hidden_states
        if block_residuals.shape[-2] > 0:
            hidden_states = self.self_attention_res(prefix_sum, block_residuals)
        if self.layer_idx % self.attn_res_block_size == 0:
            block_residuals = jnp.concatenate(
                (block_residuals, jnp.expand_dims(prefix_sum, axis=-2)), axis=-2)
            prefix_sum = None

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, kv_fused = self.self_attn(
            positions, hidden_states, forward_batch, kv_pool)
        # prefix_sum is None exactly at a checkpoint boundary -- the running sum restarts from
        # this layer's attention output rather than continuing across the boundary.
        prefix_sum = hidden_states if prefix_sum is None else prefix_sum + hidden_states

        hidden_states = self.mlp_res(prefix_sum, block_residuals)
        hidden_states = self.post_attention_layernorm(hidden_states)
        out, topk_ids = self._ffn(hidden_states, forward_batch, dispatch_info)
        return prefix_sum + out, prefix_sum, block_residuals, kv_fused, topk_ids


def make_initial_block_residuals(num_tokens: int, hidden_size: int, dtype=jnp.bfloat16):
    """Empty candidate set the first layer starts from."""
    return initial_block_residuals(num_tokens, hidden_size, dtype)
