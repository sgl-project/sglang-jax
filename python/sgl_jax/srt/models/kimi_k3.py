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
        self.is_kda = cfg_is_kda_layer(config, layer_idx)
        self.attn_res_block_size = getattr(config, 'attn_res_block_size', None)

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

        loader = WeightLoader(model=self, model_config=model_config,
                              mesh=self.mesh, dtype=self.dtype)
        loader.load_weights_from_safetensors(self._create_weight_mappings())
        for layer in self.model.layers:
            if not layer.is_kda:
                layer.self_attn.post_load_weights()

    # K3 checkpoints prefix every text parameter with `language_model.` because the release is
    # multimodal (the same checkpoint also carries `mm_projector.*` and a vision tower). Kimi-Linear
    # is text-only and emits bare `model.*` keys, so every inherited mapping is re-prefixed.
    TEXT_PREFIX = "language_model."

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

        # --- KDA A_log layout migration -----------------------------------------------------
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

        return mappings

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
    which this class does not construct (the vLLM lane serves K3 text-only via
    ``--language-model-only``).
    """


EntryClass = [KimiK3ForCausalLM, KimiK3ForConditionalGeneration]
