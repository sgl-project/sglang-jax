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

import os

import logging

import jax
import numpy as np
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.configs.kimi_k3 import KimiK3Config
from sgl_jax.srt.layers.gate import GateLogit, TopK
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.moe import EPMoE
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as _P
from sgl_jax.srt.layers.quantization.mxfp4 import MXFP4_GROUP_SIZE
from sgl_jax.srt.models.kimi_k3_layers import (
    AttentionResidual,
    mla_output_gate,
    situ_and_mul,
)
from sgl_jax.srt.models.kimi_k3_residual import initial_block_residuals
from sgl_jax.srt.models.kimi_linear import KimiDeltaAttention
from sgl_jax.srt.models.deepseek_v3 import DeepseekV3Attention as KimiMLAAttention

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------------------
# Field-based config accessors.
#
# The model must NOT depend on the config CLASS. K3's checkpoint ships its own
# `configuration_kimi_k3.py` whose class is ALSO called KimiK3Config, and with
# --trust-remote-code that one is what HF constructs -- shadowing ours and lacking our helper
# methods (observed: "'KimiK3Config' object has no attribute 'is_kda_layer'"). Reading fields
# works for both, and for a plain PretrainedConfig.
# ---------------------------------------------------------------------------------------------

def cfg_is_kda_layer(config, layer_idx: int) -> bool:
    """kda_layers is 1-BASED in the checkpoint, hence the +1.

    Reads the FIELD unconditionally and never prefers a ``config.is_kda_layer`` method. Both
    config classes derive the answer from ``linear_attn_config``, so this is equivalent -- and
    preferring the method caused infinite recursion once a shim delegating back here was
    installed on the config (RecursionError at load).
    """
    la = getattr(config, "linear_attn_config", None) or {}
    kl = la.get("kda_layers")
    return bool(kl) and (layer_idx + 1) in kl


def cfg_uses_attn_res(config) -> bool:
    return getattr(config, "attn_res_block_size", None) is not None


class KimiK3DeltaAttention(KimiDeltaAttention):
    """KDA with K3's FULL-RANK output gate.

    Kimi-Linear factors the output gate through a rank-128 bottleneck
    (``g_b_proj(g_a_proj(h))``). K3 sets ``linear_attn_config.use_full_rank_gate: True`` and ships
    a single ``g_proj.weight`` of ``[12288, 7168]`` -- no ``g_a_proj``/``g_b_proj`` at all, so the
    inherited mapping asks for tensors that do not exist and the projections stay empty
    (``dot_general ... got (7168,) and (0,)``).

    Only the OUTPUT gate changes. The FORGET gate stays low-rank in K3 too
    (``f_a_proj [128, 7168]`` -> ``f_b_proj [12288, 128]``), so it is inherited unchanged.
    """

    def __init__(self, config, mesh, dtype=jnp.bfloat16, layer_idx: int = 0):
        super().__init__(config=config, mesh=mesh, dtype=dtype, layer_idx=layer_idx)
        la = getattr(config, "linear_attn_config", None) or {}
        self.use_full_rank_gate = bool(la.get("use_full_rank_gate", False))
        if self.use_full_rank_gate:
            self.g_proj = LinearBase(
                input_size=config.hidden_size,
                output_size=self.projection_size,
                kernel_axes=(None, "tensor"),
                use_bias=False,
                params_dtype=dtype,
                mesh=mesh,
                scope_name="g_proj",
            )
            # drop the unused low-rank pair so no empty params survive into the forward
            self.g_a_proj = None
            self.g_b_proj = None

    def _output_gate(self, hidden_states):
        if self.use_full_rank_gate:
            gate, _ = self.g_proj(hidden_states)
            return gate
        g_a, _ = self.g_a_proj(hidden_states)
        gate, _ = self.g_b_proj(g_a)
        return gate

    def __call__(self, positions, hidden_states, forward_batch, recurrent_state_pool):
        """Mirrors KimiDeltaAttention.__call__ with the output gate swapped."""
        del positions

        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        raw_gate, _ = self.f_b_proj(self.f_a_proj(hidden_states)[0])
        raw_gate = raw_gate.reshape(hidden_states.shape[0], self.num_heads, self.head_dim)
        beta = jax.nn.sigmoid(self.b_proj(hidden_states)[0].astype(jnp.float32))

        o, recurrent_state_pool = self.attn(
            forward_batch, q, k, v, raw_gate, beta, recurrent_state_pool
        )
        o = o.reshape(hidden_states.shape[0], self.num_heads, self.head_dim)

        output_gate = self._output_gate(hidden_states).reshape(
            hidden_states.shape[0], self.num_heads, self.head_dim
        )
        o = self.o_norm(o, output_gate).reshape(hidden_states.shape[0], self.projection_size)
        o, _ = self.o_proj(o)
        return o, recurrent_state_pool


class KimiK3MLAAttention(KimiMLAAttention):
    """MLA with K3's optional output gate (``mla_use_output_gate``).

    Kimi-Linear's MLA has no gate. K3 ships ``self_attn.g_proj.weight`` ``[12288, 7168]`` on every
    MLA layer -- ``num_attention_heads * v_head_dim`` -- and the reference applies it to the
    attention output BEFORE ``o_proj``, matching the torch reference::

        attn_out *= self.g_proj(hidden_states)[0].sigmoid()
        return self.o_proj(attn_out)[0]

    ``o_proj`` is linear but the gate is elementwise on its INPUT, so there is no algebraically
    equivalent place to apply it afterwards -- the forward has to be re-entered at that point,
    which is why the shared front-half below is duplicated rather than delegated.
    """

    def __init__(self, *args, use_output_gate: bool = False, **kwargs):
        hidden_size = kwargs.get("hidden_size", args[0] if args else None)
        dtype = kwargs.get("dtype", jnp.bfloat16)
        super().__init__(*args, **kwargs)
        self.use_output_gate = bool(use_output_gate)
        if self.use_output_gate:
            # [hidden -> num_heads * v_head_dim]; shapes are GLOBAL here (sglang-jax annotates
            # sharding rather than slicing per device), so this is 7168 -> 96*128 = 12288,
            # matching the checkpoint's g_proj.weight [12288, 7168] exactly.
            self.g_proj = LinearBase(
                input_size=hidden_size,
                output_size=self.num_heads * self.v_head_dim,
                kernel_axes=(None, "tensor"),
                use_bias=False,
                params_dtype=dtype,
                mesh=self.mesh,
                scope_name="g_proj",
            )

    def __call__(self, positions, hidden_states, forward_batch, token_to_kv_pool):
        """Mirrors DeepseekV3Attention.__call__, gating attn_output before o_proj."""
        if not self.use_output_gate:
            return super().__call__(positions, hidden_states, forward_batch, token_to_kv_pool)

        if self.q_lora_rank is None:
            q, _ = self.q_proj(hidden_states)
        else:
            q_compressed, _ = self.q_a_proj(hidden_states)
            q_compressed = self.q_a_layernorm(q_compressed)
            q, _ = self.q_b_proj(q_compressed)
        q = q.reshape(-1, self.num_heads, self.qk_head_dim)
        q_nope = q[:, :, : self.qk_nope_head_dim]
        q_rope = q[:, :, self.qk_nope_head_dim :]

        kv_a_out, _ = self.kv_a_proj(hidden_states)
        compressed = kv_a_out[:, : self.kv_lora_rank]
        k_rope_raw = kv_a_out[:, self.kv_lora_rank :]
        compressed = self.kv_a_layernorm(compressed)

        k_rope = k_rope_raw.reshape(-1, 1, self.qk_rope_head_dim)
        if self.rotary_emb is not None:
            q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)

        if self.use_absorbed:
            attn_output, kv_fused = self._forward_mqa(
                q_nope, q_rope, compressed, k_rope, forward_batch, token_to_kv_pool
            )
        else:
            attn_output, kv_fused = self._forward_mha(
                q_nope, q_rope, compressed, k_rope, forward_batch, token_to_kv_pool
            )

        gate, _ = self.g_proj(hidden_states)
        attn_output = mla_output_gate(attn_output, gate)

        output, _ = self.o_proj(attn_output)
        return output, kv_fused


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


def _local_expert_range(moe, num_experts: int) -> range:
    """Experts this PROCESS must read, from the MoE's own expert-axis sharding.

    Derived from the mesh rather than recomputed, so the fetch plan and the device placement
    cannot drift apart -- a mismatch loads cleanly and leaves the missing experts as zeros, which
    is a wrong model that runs.
    """
    mesh = getattr(moe, "moe_mesh", None)
    ep_size = int(getattr(moe, "ep_size", 1) or 1)
    if mesh is None or ep_size <= 1:
        return range(num_experts)

    try:
        axis = mesh.axis_names.index("expert")
    except (AttributeError, ValueError):
        return range(num_experts)

    # which positions along the expert axis have a device in THIS process
    local = {
        int(idx[axis])
        for idx, device in np.ndenumerate(mesh.devices)
        if device in mesh.devices.flat and device.process_index == jax.process_index()
    }
    if not local:
        return range(num_experts)
    per = num_experts // ep_size
    lo, hi = min(local) * per, (max(local) + 1) * per
    return range(lo, hi)


class KimiK3EPMoE(EPMoE):
    """K3's routed experts: SITU activation, and weights that stay fp4 in HBM.

    Two corrections over the stock EPMoE, both of which are silent if missed:

    **SITU.** The reference's ``KimiBlockSparseMLP`` uses ``SituAndMul`` when
    ``hidden_act == "situ"`` -- and SITU is not ``f(gate) * up`` for any elementwise ``f``, so it
    cannot be expressed through the stock silu/gelu branch. Leaving the default (``silu``) loads
    cleanly, runs, and computes a different model.

    **fp4.** The released K3 experts are MXFP4 (e2m1 values + per-32 e8m0 block scales). This
    reference path widens them to bf16 with their block scales and runs the standard grouped
    matmul; correct and portable. Keeping the experts sub-byte through a fused GMM kernel is a
    separate optimization (bf16 experts are 5,072 GiB at full depth versus 1,347 GiB kept as fp4,
    measured 0.535 B/value on v7x), tracked outside this change.
    """

    def __init__(self, *args, situ_beta=None, situ_linear_beta=None, fp4: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.situ_beta = situ_beta
        self.situ_linear_beta = situ_linear_beta
        self.fp4 = bool(fp4)
        if self.fp4:
            # gmm_v2 reads block scales as [E, k_blocks, 1, N] and checks them against
            # weight_block_size; MXFP4 groups 32 values along K. HF convention is
            # [block_n, block_k], and only block_k is consumed.
            self.weight_block_size = [1, MXFP4_GROUP_SIZE]

            # Declared HERE, not filled in by the loader: EPMoE sets `self.wi_0_scale = None` in
            # __init__, which makes it a STATIC attribute, and nnx then refuses to have a Param
            # assigned over it ("Cannot assign data value ... to static attribute"). Declaring
            # them up front also pins the shapes, so a checkpoint whose group count disagrees
            # fails at assignment instead of reaching the kernel.
            #
            # K = the contracting dim of each weight: hidden (3584) for wi_*, intermediate (3072)
            # for wo -- which is why the two block counts differ (112 vs 96).
            k_blocks_wi = self.hidden_size // MXFP4_GROUP_SIZE
            k_blocks_wo = self.intermediate_dim // MXFP4_GROUP_SIZE
            wi_spec = _P("expert", None, None, "tensor")
            wo_spec = _P("expert", None, None, None)
            # Under EPMoE's OWN abstract mesh: the "expert" axis exists only there. The model mesh
            # is ("data", "tensor"), so creating these outside the context raises
            #   ValueError: Resource axis: expert ... is not found in mesh: ('data', 'tensor')
            with jax.sharding.use_abstract_mesh(self.updated_mesh):
                for name, blocks, out_dim, spec in (
                    ("wi_0_scale", k_blocks_wi, self.intermediate_dim, wi_spec),
                    ("wi_1_scale", k_blocks_wi, self.intermediate_dim, wi_spec),
                    ("wo_scale", k_blocks_wo, self.hidden_size, wo_spec),
                ):
                    # nnx.data is required: EPMoE.__init__ has already set these to None, which
                    # marks them STATIC on the pytree, and nnx then refuses a Param over them.
                    setattr(self, name, nnx.data(nnx.Param(
                        jnp.zeros((self.num_experts, blocks, 1, out_dim),
                                  dtype=jnp.float32, out_sharding=spec),
                        out_sharding=spec,
                    )))

    def _apply_activation(self, layer_w0, layer_w1):
        if self.situ_beta is None:
            return super()._apply_activation(layer_w0, layer_w1)
        # SituAndMul consumes cat([gate, up]) and splits it internally, matching the reference's
        # `torch.cat([w1(x), w3(x)], dim=-1)` exactly.
        return situ_and_mul(
            jnp.concatenate([layer_w0, layer_w1], axis=-1),
            self.situ_beta,
            self.situ_linear_beta,
        )

    def _call_gmm(self, **kwargs):
        if not self.fp4:
            return super()._call_gmm(**kwargs)
        # Reference MXFP4 path: widen the e2m1 expert weights to bf16 with their per-32 e8m0 block
        # scales, then run the standard grouped matmul. `rhs` is native float4_e2m1fn [E, K, N] and
        # `rhs_scale` is fp32 [E, num_k_blocks, 1, N] (one scale per MXFP4_GROUP_SIZE along K).
        rhs = kwargs.pop("rhs")
        scale = kwargs.pop("rhs_scale", None)
        kwargs.pop("rhs_bias", None)
        kwargs.pop("activation_quantized_dtype", None)
        w = rhs.astype(jnp.bfloat16)
        if scale is not None:
            num_experts, size_k, size_n = w.shape
            num_k_blocks = scale.shape[1]
            block = size_k // num_k_blocks
            s = jnp.broadcast_to(
                scale.reshape(num_experts, num_k_blocks, 1, size_n),
                (num_experts, num_k_blocks, block, size_n),
            ).reshape(num_experts, size_k, size_n)
            w = w * s.astype(jnp.bfloat16)
        return super()._call_gmm(rhs=w, **kwargs)


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
        self.is_kda = cfg_is_kda_layer(config, layer_idx)
        self.attn_res_block_size = getattr(config, 'attn_res_block_size', None)

        if self.is_kda:
            self.self_attn = KimiK3DeltaAttention(
                config=config, mesh=mesh, dtype=dtype, layer_idx=layer_idx
            )
        else:
            self.self_attn = KimiK3MLAAttention(
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
                use_output_gate=getattr(config, "mla_use_output_gate", False),
            )

        is_moe = (
            getattr(config, "num_experts", None)
            and layer_idx >= getattr(config, 'first_k_dense_replace', 0)
            and layer_idx % getattr(config, 'moe_layer_freq', 1) == 0
        )
        self.is_moe_layer = bool(is_moe)

        if not self.is_moe_layer:
            self.mlp = KimiK3MLP(
                config.hidden_size, config.intermediate_size, mesh,
                getattr(config, 'activation_situ_beta', None), getattr(config, 'activation_situ_linear_beta', None), dtype)
            self.moe_gate = None
            self.shared_experts = None
        else:
            self.mlp = None
            self.moe_gate = GateLogit(
                input_size=config.hidden_size,
                num_experts=config.num_experts,
                enable_expert_bias=True,
                weight_dtype=dtype,
                score_func=getattr(config, 'moe_router_activation_func', "sigmoid"),
            )
            self.topk = TopK(
                topk=config.num_experts_per_token,
                renormalize=getattr(config, 'moe_renormalize', True),
                num_expert_group=getattr(config, 'num_expert_group', 1),
                topk_group=getattr(config, 'topk_group', 1),
                routed_scaling_factor=getattr(config, 'routed_scaling_factor', 1.0),
                layer_id=layer_idx,
                mesh=mesh,
            )
            # --- LatentMoE ------------------------------------------------------------
            # K3's routed experts do NOT operate on the residual stream. They live in a
            # `routed_expert_hidden_size` (3584) LATENT, half the 7168 hidden. The checkpoint is
            # unambiguous: expert `w1.weight_packed` is [3072, 1792] -> unpacked [3072, 3584],
            # and `routed_expert_{down,up}_proj.weight` are [3584, 7168] / [7168, 3584].
            #
            # Feeding the experts the full hidden is not a silent error -- gmm_v2's
            # `validate_inputs` asserts `lhs.shape == (size_m, size_k)` -- but it IS the reason
            # a plain Kimi-Linear MoE cannot load K3.
            #
            # Both projections are ReplicatedLinear in the reference (no TP), and their I/O is
            # replicated on both sides here too (residual stream in, EPMoE's P(None) x out).
            self.routed_expert_hidden_size = getattr(
                config, "routed_expert_hidden_size", None)
            expert_hidden_size = self.routed_expert_hidden_size or config.hidden_size
            if self.routed_expert_hidden_size is not None:
                self.routed_expert_down_proj = LinearBase(
                    input_size=config.hidden_size,
                    output_size=expert_hidden_size,
                    kernel_axes=(None, None),
                    use_bias=False,
                    params_dtype=dtype,
                    mesh=mesh,
                    scope_name="routed_expert_down_proj",
                )
                self.routed_expert_up_proj = LinearBase(
                    input_size=expert_hidden_size,
                    output_size=config.hidden_size,
                    kernel_axes=(None, None),
                    use_bias=False,
                    params_dtype=dtype,
                    mesh=mesh,
                    scope_name="routed_expert_up_proj",
                )
                self.routed_expert_norm = (
                    RMSNorm(expert_hidden_size, epsilon=config.rms_norm_eps,
                            param_dtype=dtype, scope_name="routed_expert_norm")
                    if getattr(config, "latent_moe_use_norm", False) else None
                )
            else:
                self.routed_expert_down_proj = None
                self.routed_expert_up_proj = None
                self.routed_expert_norm = None

            self.block_sparse_moe = KimiK3EPMoE(
                situ_beta=(getattr(config, "activation_situ_beta", None)
                           if getattr(config, "hidden_act", "silu") == "situ" else None),
                situ_linear_beta=getattr(config, "activation_situ_linear_beta", None),
                # fp4 is the default; KIMI_K3_MOE_FP4=0 falls back to bf16 experts, which is how
                # the two paths are A/B'd and an escape hatch if the fp4 kernel misbehaves.
                fp4=os.environ.get("KIMI_K3_MOE_FP4", "1") != "0",
                hidden_size=expert_hidden_size,
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
                    config.moe_intermediate_size * getattr(config, 'num_shared_experts', 0),
                    mesh, getattr(config, 'activation_situ_beta', None), getattr(config, 'activation_situ_linear_beta', None), dtype)
                if getattr(config, 'num_shared_experts', 0) > 0 else None
            )

        self.input_layernorm = RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, param_dtype=dtype,
            scope_name="input_layernorm")
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, param_dtype=dtype,
            scope_name="post_attention_layernorm")

        if cfg_uses_attn_res(config):
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
        # Shared experts and the router both read the FULL hidden state (gate.weight is
        # [896, 7168], shared_experts.gate_proj is [6144, 7168]); only the ROUTED path goes
        # through the latent. Getting that split wrong type-checks but computes a different model.
        shared = self.shared_experts(hidden_states) if self.shared_experts is not None else None
        router_logits = self.moe_gate(hidden_states)
        bias = self.moe_gate.bias.value if self.moe_gate.bias is not None else None
        topk_w, topk_ids = self.topk(router_logits, bias, dispatch_info=dispatch_info)

        routed_in = hidden_states
        if self.routed_expert_down_proj is not None:
            routed_in, _ = self.routed_expert_down_proj(hidden_states)

        out = self.block_sparse_moe(routed_in, topk_w, topk_ids)

        # KimiRoutedOutputTransform: norm THEN up_proj (norm is on the latent, 3584-wide).
        if self.routed_expert_norm is not None:
            out = self.routed_expert_norm(out)
        if self.routed_expert_up_proj is not None:
            out, _ = self.routed_expert_up_proj(out)

        if shared is not None:
            out = out + shared
        return out, topk_ids

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        memory_pools,
        block_residuals: jax.Array | None = None,
        dispatch_info=None,
    ):
        """Follows ``KimiDecoderLayer.forward`` in the PyTorch reference exactly.

        Returns ``(hidden_states, block_residuals, kv_fused, topk_ids)``.

        ``prefix_sum`` is deliberately NOT threaded between layers: the reference re-initializes
        it from ``hidden_states`` at the top of every layer, and only ``block_residuals`` carries
        across. Passing it in would silently change what every AttnRes mixes.
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
            return residual + out, None, kv_fused, topk_ids

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
        return prefix_sum + out, block_residuals, kv_fused, topk_ids


def make_initial_block_residuals(num_tokens: int, hidden_size: int, dtype=jnp.bfloat16):
    """Empty candidate set the first layer starts from."""
    return initial_block_residuals(num_tokens, hidden_size, dtype)


class KimiK3Model(nnx.Module):
    """Embedding -> N decoder layers -> output AttnRes -> final norm.

    The **output_attn_res** below is a THIRD AttentionResidual beyond the two per layer: after the
    last layer the model mixes the final hidden state against the accumulated block residuals one
    more time, before the final norm. Missing it silently drops K3's last depth-mixing step -- the
    model would still run and produce plausible-looking text.
    """

    def __init__(self, config: KimiK3Config, mesh: jax.sharding.Mesh, dtype=jnp.bfloat16):
        from sgl_jax.srt.layers.embeddings import Embed

        self.config = config
        self.vocab_size = config.vocab_size
        self.attn_res_block_size = getattr(config, 'attn_res_block_size', None)
        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size, features=config.hidden_size,
            dtype=dtype, param_dtype=dtype, kernel_axes=("tensor", None), mesh=mesh)
        self.layers = nnx.data([
            KimiK3DecoderLayer(config=config, mesh=mesh, layer_idx=i, dtype=dtype)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, dtype=dtype,
            param_dtype=dtype, scope_name="norm")
        self.output_attn_res = (
            AttentionResidual(config.hidden_size, config.rms_norm_eps, mesh, dtype,
                              "output_attn_res")
            if cfg_uses_attn_res(config) else None)

    def __call__(self, forward_batch: ForwardBatch, memory_pools):
        hidden_states = self.embed_tokens(forward_batch.input_ids)

        # Derive the empty candidate set FROM hidden_states rather than building a fresh
        # jnp.zeros: a fresh array is unsharded (P(None,None,None)) while hidden_states carries
        # P('data',None,None), and jnp.concatenate rejects mismatched shardings --
        #   ShardingTypeError: All operands should have the same sharding
        # Slicing to zero width inherits both the sharding and the dtype.
        block_residuals = (
            hidden_states[:, None, :][:, :0, :]
            if self.attn_res_block_size is not None else None)

        kv_fused_list, rec_bufs, conv_bufs, topk_list = [], [], [], []
        for layer in self.layers:
            hidden_states, block_residuals, attn_state, topk_ids = layer(
                forward_batch.positions, hidden_states, forward_batch, memory_pools,
                block_residuals, dispatch_info=forward_batch.expert_location_metadata)
            if layer.is_kda:
                rec, conv = attn_state
                rec_bufs.append(rec); conv_bufs.append(conv)
            else:
                kv_fused_list.append(attn_state)
            topk_list.append(topk_ids)

        if block_residuals is not None:
            hidden_states = self.output_attn_res(hidden_states, block_residuals)
        hidden_states = self.norm(hidden_states)
        return hidden_states, kv_fused_list, (rec_bufs, conv_bufs), topk_list


class KimiK3ForCausalLM(nnx.Module):
    """Top-level K3 causal LM, mirroring ``KimiLinearForCausalLM``."""

    @classmethod
    def patch_model_config(cls, config) -> None:
        from sgl_jax.srt.configs.model_config import AttentionArch

        config.attention_arch = AttentionArch.MLA
        qk_nope = getattr(config.hf_text_config, "qk_nope_head_dim", 0)
        qk_rope = getattr(config.hf_text_config, "qk_rope_head_dim", 0)
        if qk_nope and qk_rope:
            config.head_dim = qk_nope + qk_rope

    def __init__(self, config: KimiK3Config, mesh: jax.sharding.Mesh, dtype=jnp.bfloat16):
        from sgl_jax.srt.layers.embeddings import ParallelLMHead
        from sgl_jax.srt.layers.logits_processor import LogitsProcessor

        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        self.model = KimiK3Model(config=config, mesh=mesh, dtype=dtype)
        if not getattr(config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                config.vocab_size, config.hidden_size, dtype=dtype,
                param_dtype=dtype, kernel_axes=("tensor", None))
        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=mesh)

    def __call__(self, forward_batch: ForwardBatch, memory_pools, logits_metadata):
        hidden_states, kv_fused, recurrent_state, topk_ids = self.model(
            forward_batch, memory_pools)
        head = (self.model.embed_tokens
                if getattr(self.config, "tie_word_embeddings", False) else self.lm_head)
        output = self.logits_processor(hidden_states, head, logits_metadata)
        return (output,
                {"token_to_kv_pool": kv_fused, "recurrent_state_pool": recurrent_state},
                True, topk_ids)

    def load_weights(self, model_config):
        from sgl_jax.srt.utils.weight_utils import WeightLoader

        # must run BEFORE the mappings are built -- every key in them is prefixed
        self._detect_text_prefix(model_config)

        loader = WeightLoader(model=self, model_config=model_config,
                              mesh=self.mesh, dtype=self.dtype)
        loader.load_weights_from_safetensors(self._create_weight_mappings())
        self._fixup_kda_a_log(model_config)
        self._fixup_moe_mxfp4(model_config)
        for layer in self.model.layers:
            if not layer.is_kda:
                layer.self_attn.post_load_weights()

    def _fixup_moe_mxfp4(self, model_config):
        """Load the MXFP4 MoE experts, which the inherited mapping cannot see.

        Kimi-Linear's expert mapping asks for ``...experts.<e>.w{1,2,3}.weight``. K3 ships
        ``.weight_packed`` (uint8, fp4 pairs) plus ``.weight_scale`` (uint8, e8m0 per 32), so
        those keys simply do not exist and every expert param stays at its zero-size placeholder
        -- surfacing later as ``dot_general ... got (7168,) and (0,)``.

        DeepSeek-V3 solves the analogous FP8 case by registering a parallel scale group
        (``.weight`` -> ``.weight_scale_inv``) and converting after stacking. MXFP4 needs a
        different conversion (unpack fp4 pairs, decode e8m0, apply per-32 scale, transpose), so
        this loads and dequantizes directly.

        Sharding is taken from the EXISTING parameter rather than reconstructed, so it is correct
        by construction regardless of the EP/TP mesh layout.
        """
        from sgl_jax.srt.layers.quantization.mxfp4_moe import (
            EXPERT_PROJ_TO_EPMOE,
            dequant_expert_weight,
            e8m0_scale_to_kernel_layout,
            unpack_fp4_to_e2m1,
        )
        from sgl_jax.srt.layers.quantization.mxfp4_streaming import open_source

        cfg = self.config
        num_experts = int(getattr(cfg, "num_experts", 0) or 0)
        if not num_experts:
            return
        # KIMI_K3_WEIGHTS_URI points the EXPERT load at GCS while --model-path stays a local dir
        # holding the config and tokenizer. The two are separable because experts are ~93% of the
        # checkpoint's bytes (2.723 T of 2.78 T params) and are loaded here, not by the generic
        # WeightLoader -- so streaming them is what turns 1.42 TiB of local staging into ~106 GiB.
        path = (
            os.environ.get("KIMI_K3_WEIGHTS_URI")
            or getattr(model_config, "model_path", None)
            or getattr(cfg, "_name_or_path", None)
        )
        if not path:
            logger.warning("MoE MXFP4 fixup skipped: no model path")
            return

        # A gs:// path is read by RANGE and never staged: the full release is 1.42 TiB against a
        # node's ~919 GB of tmpfs, so "download it first" is not an option at full depth. A local
        # directory keeps the original safetensors path.
        def _wanted(key: str) -> bool:
            return ".block_sparse_moe.experts." in key and key.endswith(
                ("weight_packed", "weight_scale")
            )

        source = open_source(path, keep=_wanted)
        logger.info(
            "MoE MXFP4 fixup: reading experts from %s (%s)",
            path, "streamed byte ranges" if str(path).startswith("gs://") else "local files",
        )
        try:
            n_done = 0
            for li, layer in enumerate(self.model.layers):
                if not getattr(layer, "is_moe_layer", False):
                    continue
                moe = layer.block_sparse_moe
                # the MoE module decides; the loader does not second-guess it
                use_fp4 = bool(getattr(moe, "fp4", False))

                # Only the experts some LOCAL device owns. On a single-process pod that is all of
                # them -- every device is addressable, so nothing is saved there. Across hosts at
                # ep_size=32 a process owns 8 of 32 expert groups (224 of 896) and never requests
                # the other 672.
                wanted_experts = _local_expert_range(moe, num_experts)

                # Prefetch the WHOLE LAYER -- all three projections at once. A shard stores each
                # expert's w1/w2/w3 together, so asking for one projection leaves ~33 MB of the
                # other two between consecutive wanted ranges and the coalescer cannot bridge it.
                # Measured on this fleet: one GET per tensor = 46 s/group (~3.5 h for the model);
                # per-projection coalescing = 25 s; whole-layer = 15.4 s for 3.9 GB (~24 min).
                stem = f"{self.TEXT_PREFIX}model.layers.{li}.block_sparse_moe.experts"
                source.prefetch([
                    f"{stem}.{e}.{proj}.{suffix}"
                    for e in wanted_experts
                    for proj in EXPERT_PROJ_TO_EPMOE
                    for suffix in ("weight_packed", "weight_scale")
                ])

                for proj, target in EXPERT_PROJ_TO_EPMOE.items():
                    param = getattr(moe, target, None)
                    if param is None:
                        continue
                    scale_param = getattr(moe, f"{target}_scale", None) if use_fp4 else None

                    base_fmt = (
                        f"{self.TEXT_PREFIX}model.layers.{li}.block_sparse_moe.experts.{{e}}.{proj}"
                    )
                    if not source.has(f"{base_fmt.format(e=0)}.weight_packed"):
                        continue

                    # MEASURED: GcsSource.get() pops the cache (single-use, so the cache cannot
                    # grow into the local copy streaming exists to avoid). But _expert(e) is
                    # called twice per projection -- once building the weight, once the scale --
                    # and the second pass finds an empty cache and re-reads every tensor as an
                    # individual, UN-coalesced GET. Measured on one projection: 7.0 s coalesced
                    # vs 43.1 s individual, a 6.2x penalty paid 276 times.
                    _expert_cache: dict = {}

                    def _expert(e, _fmt=base_fmt, _fp4=use_fp4, _cache=_expert_cache):
                        pk, sk = f"{_fmt.format(e=e)}.weight_packed", f"{_fmt.format(e=e)}.weight_scale"
                        if not source.has(sk):
                            raise KeyError(f"{pk} present but {sk} missing")
                        if e in _cache:
                            w, sc = _cache[e]
                        else:
                            w, sc = source.get(pk), source.get(sk)
                            _cache[e] = (w, sc)
                        if _fp4:
                            return unpack_fp4_to_e2m1(jnp.asarray(w)), e8m0_scale_to_kernel_layout(
                                jnp.asarray(sc)
                            )
                        return dequant_expert_weight(jnp.asarray(w), jnp.asarray(sc), jnp.bfloat16), None

                    # EPMoE's OWN mesh: it builds `moe_mesh` with axis_names ("expert", "tensor")
                    # (moe.py:84-88) and creates its params under that. The model mesh is
                    # ("data", "tensor"), so binding the spec to it raises
                    #   ValueError: Resource axis: expert ... is not found in mesh
                    target_mesh = getattr(moe, "moe_mesh", None) or self.mesh

                    def _build(param_obj, pick):
                        """Assemble one sharded param, fetching ONLY this host's slices.

                        make_array_from_callback is what the shared loader uses for exactly this
                        (weight_utils.py:1253). Building the global array host-side and then
                        device_put'ing it -- what this did before -- works on one host and fails
                        on four with
                          RuntimeError: Fetching value for `jax.Array` that spans non-addressable
                          (non process local) devices
                        because the global array is not process-local. The callback form is also
                        where EP filtering becomes real: it is invoked only for slices this
                        process owns, so the other ranks' experts are never fetched.
                        """
                        spec = getattr(getattr(param_obj.value, "sharding", None), "spec", None)
                        global_shape = tuple(param_obj.value.shape)
                        if spec is None:
                            return None
                        sharding = NamedSharding(target_mesh, spec)

                        def _slice(index):
                            expert_idx = index[0]
                            ids = (
                                range(*expert_idx.indices(global_shape[0]))
                                if isinstance(expert_idx, slice)
                                else [int(expert_idx)]
                            )
                            local = np.stack([np.asarray(pick(_expert(e))) for e in ids], axis=0)
                            rest = tuple(index[1:])
                            return local[(slice(None), *rest)] if rest else local

                        return jax.make_array_from_callback(global_shape, sharding, _slice)

                    built = _build(param, lambda t: t[0])
                    if built is None:
                        continue
                    param.value = built

                    # The scale is NOT optional in fp4: the kernel multiplies by it, so a missing
                    # or zero scale produces finite, plausible, wrong numbers rather than an error.
                    if use_fp4 and scale_param is not None:
                        scale_built = _build(scale_param, lambda t: t[1])
                        if scale_built is not None:
                            scale_param.value = scale_built

                    _expert_cache.clear()   # bounded to one projection, never the model
                    n_done += 1
                    if n_done % 12 == 0 or li <= 1:
                        # 92 MoE layers x 3 projections = 276 groups and one line at the end:
                        # without progress a stall and a silent death look identical from outside.
                        logger.info(
                            "MoE MXFP4 fixup: %d groups done (layer %d/%d, %s)",
                            n_done, li, len(self.model.layers), target)
            logger.info(
                "MoE MXFP4 fixup: loaded %d expert groups as %s",
                n_done, "native fp4 + block scales" if use_fp4 else "bf16")
        finally:
            source.close()

    def _fixup_kda_a_log(self, model_config):
        """Load KDA ``A_log``, narrowing the padded checkpoint tensor to ``num_heads``.

        K3 ships ``A_log`` with **128** entries while the KDA has **96** heads. The other tensors
        pin the geometry unambiguously: ``o_norm.weight`` is ``[128]`` (the kernel validates it
        equals ``head_dim``), ``b_proj.weight`` is ``[96, 7168]`` (beta is per-head), and
        ``q/k/v_proj`` are ``[12288, ...]`` = 96*128. So ``A_log`` is stored padded to
        ``head_dim`` and only its first ``num_heads`` entries are meaningful.

        This mirrors the torch reference's A_log loader exactly::

            shard_size = parameter.shape[0]                       # num_heads // tp
            loaded_weight.narrow(0, rank * shard_size, shard_size)

        i.e. take the FIRST ``num_heads`` entries and shard them across TP. The kernel then
        indexes it per head -- ``A_log.reshape(H,1,1,1,1)``, ``-exp(A)[:,None,None] * softplus(g)``
        (chunk_kda.py:716,303) -- so a wrong length here would silently mis-gate every head
        rather than fail.

        The old ``[1, 1, H, 1]`` layout is also accepted, matching the reference's own comment
        ("Load either the old [1,1,H,1] or current [H] layout").
        """
        import glob
        import os

        import numpy as np

        from safetensors import safe_open

        cfg = self.config
        la = getattr(cfg, "linear_attn_config", None) or {}
        num_heads = int(la.get("num_heads", 0))
        if not num_heads:
            return

        path = getattr(model_config, "model_path", None) or getattr(cfg, "_name_or_path", None)
        files = sorted(glob.glob(os.path.join(path, "*.safetensors"))) if path else []
        if not files:
            logger.warning("A_log fixup skipped: no safetensors under %r", path)
            return

        wanted = {
            f"{self.TEXT_PREFIX}model.layers.{i}.self_attn.A_log": i
            for i in range(cfg.num_hidden_layers)
            if cfg_is_kda_layer(cfg, i)
        }
        found = 0
        for f in files:
            with safe_open(f, framework="np") as h:
                keys = set(h.keys())
                for key, layer_idx in list(wanted.items()):
                    if key not in keys:
                        continue
                    raw = h.get_tensor(key)
                    if raw.ndim == 4:            # old [1, 1, H, 1] layout
                        raw = raw.reshape(-1)
                    if raw.shape[0] < num_heads:
                        raise ValueError(
                            f"{key}: A_log has {raw.shape[0]} entries, fewer than "
                            f"num_heads={num_heads}"
                        )
                    # Place with the SAME sharding the param was declared with
                    # (kimi_linear.py: out_sharding=P(None, None, "tensor", None)). Assigning a
                    # plain unsharded array leaves the per-rank view empty, and the backend then
                    # fails on `layer.A_log.value.reshape(H)` with shape (0,) -- H there is the
                    # PER-RANK head count (q.shape[-2]), not the global 96.
                    from jax.sharding import NamedSharding
                    from jax.sharding import PartitionSpec as _P

                    a = jnp.asarray(raw[:num_heads], dtype=jnp.float32).reshape(
                        1, 1, num_heads, 1
                    )
                    # count on the HOST copy, before placement: reading it back afterwards is a
                    # cross-process fetch ("Fetching value for `jax.Array` that spans
                    # non-addressable devices") because the tensor axis is sharded. A debug log
                    # line is not worth an all-gather, and on 4 hosts it is fatal rather than slow.
                    nonzero = int((np.asarray(a) != 0).sum())
                    a = jax.device_put(
                        a, NamedSharding(self.mesh, _P(None, None, "tensor", None))
                    )
                    attn = self.model.layers[layer_idx].self_attn
                    before = getattr(getattr(attn, "A_log", None), "value", None)
                    # Mutate the EXISTING Param in place. RadixLinearAttention captures a
                    # reference to A_log at construction, so replacing the Param object leaves
                    # the backend holding the old one.
                    if getattr(attn, "A_log", None) is not None:
                        attn.A_log.value = a
                    else:
                        attn.A_log = nnx.Param(a)
                    logger.info(
                        "A_log L%d: raw%s -> %s (was %s), nonzero=%d",
                        layer_idx, tuple(raw.shape), tuple(a.shape),
                        tuple(before.shape) if before is not None else None,
                        nonzero,
                    )
                    del wanted[key]
                    found += 1
        if wanted:
            raise KeyError(f"A_log missing for {len(wanted)} KDA layers: {list(wanted)[:3]}")
        logger.info("A_log fixup: loaded %d KDA layers, narrowed to %d heads", found, num_heads)

    # K3's RELEASE prefixes every text parameter with `language_model.` because it is multimodal
    # (the same checkpoint carries `mm_projector.*` and a vision tower). That is a property of the
    # checkpoint, not of the architecture -- a text-only export, or Kimi-Linear weights run through
    # this class, emit bare `model.*` keys. Assuming the prefix costs a KeyError naming every layer
    # at once, which reads like a missing-weights bug rather than a naming one.
    #
    # So: default to the release's layout, but let _detect_text_prefix() correct it from the
    # checkpoint's own keys before any mapping is built.
    TEXT_PREFIX = "language_model."

    def _detect_text_prefix(self, model_config) -> None:
        """Set TEXT_PREFIX from what the checkpoint actually contains.

        Cheap: reads key NAMES from one shard's header, never a tensor. Silent no-op if the
        checkpoint cannot be located -- the declared default still applies.
        """
        import glob
        import os

        from safetensors import safe_open

        path = getattr(model_config, "model_path", None) or getattr(
            self.config, "_name_or_path", None
        )
        files = sorted(glob.glob(os.path.join(path, "*.safetensors"))) if path else []
        if not files:
            return
        with safe_open(files[0], "numpy") as handle:
            keys = list(handle.keys())
        prefix = "language_model." if any(k.startswith("language_model.") for k in keys) else ""
        if prefix != type(self).TEXT_PREFIX:
            logger.info(
                "text prefix: %r (checkpoint keys start with %r)",
                prefix,
                keys[0].split(".")[0] if keys else "",
            )
        self.TEXT_PREFIX = prefix

    # Borrow Kimi-Linear's per-layer mapping builder WITHOUT inheriting from it. The parent's
    # _create_weight_mappings calls self._create_layer_mappings(...), so an unbound delegation
    # needs the helper present on this class too. Inheriting instead would make K3 a
    # KimiLinearForCausalLM subclass, which is exactly the substitution the registry test forbids
    # (K3's text_config points at Kimi-Linear; nothing must be able to satisfy it by inheritance).
    from sgl_jax.srt.models.kimi_linear import (  # noqa: E402
        KimiLinearForCausalLM as _KL,
    )

    _create_layer_mappings = _KL._create_layer_mappings
    del _KL

    def _create_weight_mappings(self) -> dict:
        """Kimi-Linear's mappings, re-prefixed, plus K3's AttnRes parameters.

        Verified against the released checkpoint's index rather than inferred: the AttnRes norm and
        proj are SIBLINGS with `_norm` / `_proj` suffixes
        (`...layers.N.self_attention_res_norm.weight`), not a nested module, and there is a
        model-level pair (`model.output_attn_res_{norm,proj}.weight`) on top of the two per layer.
        """
        from sgl_jax.srt.models.kimi_linear import KimiLinearForCausalLM
        from sgl_jax.srt.utils.weight_utils import WeightMapping

        # Kimi-Linear's mapping builder calls self.config.is_kda_layer(i) directly
        # (kimi_linear.py:665). With --trust-remote-code the config is the CHECKPOINT's
        # KimiK3Config, which has no such method, so install a field-reading shim before
        # delegating. Shimming beats duplicating the whole mapping builder, which would then
        # drift from upstream.
        if not callable(getattr(self.config, "is_kda_layer", None)):
            cfg = self.config
            cfg.is_kda_layer = lambda i, _c=cfg: cfg_is_kda_layer(_c, i)

        base = KimiLinearForCausalLM._create_weight_mappings(self)
        mappings = {self.TEXT_PREFIX + k: v for k, v in base.items()}

        if not cfg_uses_attn_res(self.config):
            return mappings

        def _pair(ckpt_stem: str, target_stem: str):
            # norm: [hidden] RMSNorm scale, replicated. proj: [hidden, 1] scorer, replicated --
            # sharding it would need an all-reduce to produce a single scalar per candidate.
            mappings[f"{self.TEXT_PREFIX}{ckpt_stem}_norm.weight"] = WeightMapping(
                target_path=f"{target_stem}.norm.scale", sharding=(None,), transpose=False)
            mappings[f"{self.TEXT_PREFIX}{ckpt_stem}_proj.weight"] = WeightMapping(
                target_path=f"{target_stem}.proj.weight", sharding=(None, None), transpose=True)

        for i in range(self.config.num_hidden_layers):
            _pair(f"model.layers.{i}.self_attention_res",
                  f"model.layers.{i}.self_attention_res")
            _pair(f"model.layers.{i}.mlp_res", f"model.layers.{i}.mlp_res")
        _pair("model.output_attn_res", "model.output_attn_res")

        # --- KDA full-rank output gate ------------------------------------------------------
        # K3 has g_proj [12288, 7168]; Kimi-Linear has g_a_proj/g_b_proj. Drop the low-rank keys
        # (absent from the checkpoint) and map the full-rank one.
        la = getattr(self.config, "linear_attn_config", None) or {}
        if la.get("use_full_rank_gate"):
            for i in range(self.config.num_hidden_layers):
                if not cfg_is_kda_layer(self.config, i):
                    continue
                for gone in ("g_a_proj", "g_b_proj"):
                    mappings.pop(f"{self.TEXT_PREFIX}model.layers.{i}.self_attn.{gone}.weight", None)
                mappings[f"{self.TEXT_PREFIX}model.layers.{i}.self_attn.g_proj.weight"] = (
                    WeightMapping(
                        target_path=f"model.layers.{i}.self_attn.g_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                    )
                )

        # --- MLA: q-LoRA + output gate ------------------------------------------------------
        # Kimi-Linear's MLA projects Q in one shot (`q_proj`) because its q_lora_rank is None.
        # K3's is LoRA-factored -- q_a_proj [1536, 7168] -> q_a_layernorm [1536] -> q_b_proj
        # [18432, 1536] -- so the inherited `q_proj` key does not exist in the checkpoint and the
        # param stays empty (`dot_general ... got (7168,) and (0,)`).
        if getattr(self.config, "q_lora_rank", None) is not None:
            for i in range(self.config.num_hidden_layers):
                if cfg_is_kda_layer(self.config, i):
                    continue
                stem = f"{self.TEXT_PREFIX}model.layers.{i}.self_attn"
                tgt = f"model.layers.{i}.self_attn"
                mappings.pop(f"{stem}.q_proj.weight", None)
                mappings[f"{stem}.q_a_proj.weight"] = WeightMapping(
                    target_path=f"{tgt}.q_a_proj.weight",
                    sharding=(None, None),
                    transpose=True,
                )
                mappings[f"{stem}.q_a_layernorm.weight"] = WeightMapping(
                    target_path=f"{tgt}.q_a_layernorm.scale",
                    sharding=(None,),
                    transpose=False,
                )
                mappings[f"{stem}.q_b_proj.weight"] = WeightMapping(
                    target_path=f"{tgt}.q_b_proj.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                )

        if getattr(self.config, "mla_use_output_gate", False):
            for i in range(self.config.num_hidden_layers):
                if cfg_is_kda_layer(self.config, i):
                    continue
                mappings[f"{self.TEXT_PREFIX}model.layers.{i}.self_attn.g_proj.weight"] = (
                    WeightMapping(
                        target_path=f"model.layers.{i}.self_attn.g_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                    )
                )

        # --- LatentMoE down/norm/up ---------------------------------------------------------
        if getattr(self.config, "routed_expert_hidden_size", None) is not None:
            for i in range(self.config.num_hidden_layers):
                stem = f"{self.TEXT_PREFIX}model.layers.{i}.block_sparse_moe"
                # checkpoint nests these under `block_sparse_moe.`; the JAX modules hang off the
                # DECODER LAYER (next to block_sparse_moe), so the target stem is one level up.
                tgt = f"model.layers.{i}"
                for proj in ("routed_expert_down_proj", "routed_expert_up_proj"):
                    mappings[f"{stem}.{proj}.weight"] = WeightMapping(
                        target_path=f"{tgt}.{proj}.weight",
                        sharding=(None, None),
                        transpose=True,
                    )
                if getattr(self.config, "latent_moe_use_norm", False):
                    mappings[f"{stem}.routed_expert_norm.weight"] = WeightMapping(
                        target_path=f"{tgt}.routed_expert_norm.scale",
                        sharding=(None,),
                        transpose=False,
                    )

        # --- KDA A_log: EXCLUDED from the mapping, see _fixup_kda_a_log ---------------------
        # A_log needs a narrow (slice) that WeightMapping cannot express, so it is dropped here
        # and loaded by hand after the main pass.
        for i in range(self.config.num_hidden_layers):
            if cfg_is_kda_layer(self.config, i):
                mappings.pop(f"{self.TEXT_PREFIX}model.layers.{i}.self_attn.A_log", None)

        return mappings

    # ------------------------------------------------------------------------------------
    # (retained for reference; superseded by _fixup_kda_a_log)
    # ------------------------------------------------------------------------------------
    def _unused_a_log_reshape_mapping(self):
        """

        # sglang-jax's KimiDeltaAttention still declares A_log as (1, 1, H, 1) with
        # P(None, None, "tensor", None) -- the OLD layout. K3's released checkpoint ships the
        # CURRENT layout, a flat [H]. The PyTorch reference documents exactly this and accepts
        # both (`_load_a_log`: "Load either the old [1,1,H,1] or current [H] layout").
        #
        # Without a reshape the loader asserts on a rank-1 tensor against a rank-4 spec:
        #   AssertionError: (1, P(None, None, 'tensor', None))
        # which is at least loud. The dangerous variant would be a silent broadcast.
        for i in range(self.config.num_hidden_layers):
            if not cfg_is_kda_layer(self.config, i):
                continue
            mappings[f"{self.TEXT_PREFIX}model.layers.{i}.self_attn.A_log"] = WeightMapping(
                target_path=f"model.layers.{i}.self_attn.A_log",
                sharding=(None, None, "tensor", None),
                transpose=False,
                reshape=(1, 1, -1, 1),
            )
        """
        return {}

class KimiK3ForConditionalGeneration(KimiK3ForCausalLM):
    """Registry entry for K3's declared architecture.

    The registry keys by ``cls.__name__`` and the released config's top-level
    ``architectures`` is ``["KimiK3ForConditionalGeneration"]``, so the class must carry exactly
    that name to be selected.

    > [!warning] Why this class must exist, and must not be skipped
    > K3's ``text_config`` declares ``model_type: "kimi_linear"`` and
    > ``architectures: ["KimiLinearForCausalLM"]``. Anything that routes on the TEXT config
    > therefore lands on Kimi-Linear, which has no AttnRes and no SITU -- and would load K3's
    > weights, run, and emit fluent text with two architectural components silently missing.
    > Routing must come from the TOP-LEVEL architectures, which is what this name pins.

    Text-only for now: the released checkpoint also carries a vision tower and ``mm_projector.*``,
    which this class does not construct; the text stack is served on its own.
    """


EntryClass = [KimiK3ForCausalLM, KimiK3ForConditionalGeneration]
