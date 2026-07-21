"""BailingMoeV3 (Ling-V3-Flash / ring_v3_lite) hybrid model for sglang-jax.

Architecture (model_type: bailing_hybrid):
  - Decoder layers grouped by ``layer_group_size`` (=4). The last layer in each
    group uses MLA (DeepSeek-style latent attention); all other layers use KDA
    (Kimi Delta Attention, linear attention with 1-tier gate projections).
  - MoE (sigmoid router, group-limited top-k, shared expert) on every layer
    with ``layer_idx >= first_k_dense_replace`` (=1); layer 0 is a dense MLP.

KDA layers carry recurrent state (SSM + conv buffers) via
``RecurrentStatePool``; MLA layers carry a paged latent KV cache via
``MHATokenToKVPool``. The hybrid memory pool dispatches transparently.

Differences from KimiLinear (Kimi K2):
  - 1-tier gate projections (f_a_proj/f_proj, g_a_proj/g_proj) instead of
    Kimi K2's 2-tier (f_a→f_b, g_a→g_b).
  - A_log is 1D [num_heads] (per-head scalar) instead of [1,1,H,1].
  - Output norm: RMS-norm + sigmoid gate × learned weight (instead of
    GatedRMSNorm module).
  - Conv weights stored as ``conv_q``/``conv_k``/``conv_v``.
  - Gate checkpoint name aliases: ``f_a_proj`` (ring_v3_lite) / ``f_proj``
    (ling_v3_flash) → both mapped to ``f_a_proj``; same for output gate.
"""

from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.model_config import AttentionArch, ModelConfig, MoEBackend
from sgl_jax.srt.eplb.expert_location import ExpertLocationMetadata, get_global_expert_location_metadata
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
    FusedEPMoEV2,
    FusedTPMoEV4,
    GateLogit,
    TopK,
    create_moe_weights_mapping,
)
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.layers.radix_linear_attention import RadixLinearAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.deepseek_v3 import DeepseekV3Attention, DeepseekV3MLP
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


def _is_kda_layer(layer_idx: int, layer_group_size: int) -> bool:
    """Last layer in each group is MLA; the rest are KDA. Mirrors modeling."""
    return (layer_idx + 1) % layer_group_size != 0


# ---------------------------------------------------------------------------
# KDA Attention (Ling V3 style — 1-tier gate projections)
# ---------------------------------------------------------------------------


class BailingMoeV3KDAAttention(nnx.Module):
    """KDA attention with 1-tier gate projections (Ling V3 / ring_v3_lite).

    Differences from ``KimiDeltaAttention`` (Kimi K2):
      - Forget gate: single ``f_a_proj`` (H → num_heads × head_dim), no
        ``f_b_proj`` second tier.
      - Output gate: single ``g_a_proj`` (H → num_heads × head_dim), no
        ``g_b_proj`` second tier.
      - ``A_log`` is 1D ``[num_heads]`` (not 4D).
      - Output norm: RMS-norm + sigmoid(gate) × learned ``o_norm_weight``
        (not ``GatedRMSNorm``).
      - Conv weight containers named ``conv_q`` / ``conv_k`` / ``conv_v``.
    """

    def __init__(
        self,
        config,
        layer_idx: int = 0,
        mesh: jax.sharding.Mesh | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        self.num_heads = config.num_attention_heads
        self.projection_size = self.num_heads * self.head_dim
        self.conv_kernel_size = getattr(config, "short_conv_kernel_size", 4)
        self.rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)

        # Q/K/V projections
        self.q_proj = LinearBase(
            self.hidden_size,
            self.projection_size,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="q_proj",
        )
        self.k_proj = LinearBase(
            self.hidden_size,
            self.projection_size,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="k_proj",
        )
        self.v_proj = LinearBase(
            self.hidden_size,
            self.projection_size,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="v_proj",
        )

        # Depthwise short-conv weights stored as LinearBase containers
        # [D, K] layout consumed by ``short_convolution`` / ``KDAAttnBackend``.
        self.conv_q = LinearBase(
            self.projection_size,
            self.conv_kernel_size,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=("tensor", None),
            scope_name="conv_q",
        )
        self.conv_k = LinearBase(
            self.projection_size,
            self.conv_kernel_size,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=("tensor", None),
            scope_name="conv_k",
        )
        self.conv_v = LinearBase(
            self.projection_size,
            self.conv_kernel_size,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=("tensor", None),
            scope_name="conv_v",
        )

        # 1-tier gate projections (direct H → projection_size, no bottleneck)
        self.f_a_proj = LinearBase(
            self.hidden_size,
            self.projection_size,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="f_a_proj",
        )
        self.g_a_proj = LinearBase(
            self.hidden_size,
            self.projection_size,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="g_a_proj",
        )

        # Beta projection: H → num_heads (per-head scalar)
        self.b_proj = LinearBase(
            self.hidden_size,
            self.num_heads,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="b_proj",
        )

        # A_log: [num_heads] per-head decay scalars (1D, unlike Kimi K2's 4D)
        self.A_log = nnx.Param(
            jnp.zeros(
                (self.num_heads,),
                dtype=jnp.float32,
                out_sharding=P("tensor"),
            )
        )
        # dt_bias: [projection_size] channel-wise bias
        self.dt_bias = nnx.Param(
            jnp.zeros(
                (self.projection_size,),
                dtype=jnp.float32,
                out_sharding=P("tensor"),
            )
        )
        # Output norm: learned per-head-dim weight
        self.o_norm_weight = nnx.Param(
            jnp.ones(
                (self.head_dim,),
                dtype=jnp.float32,
                out_sharding=P(None),
            )
        )

        # Output projection
        self.o_proj = LinearBase(
            self.projection_size,
            self.hidden_size,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=("tensor", None),
            scope_name="o_proj",
        )

        # RadixLinearAttention dispatcher — conv LinearBases are passed as
        # parameter containers (never called). The KDA backend accesses
        # ``layer.q_conv1d.weight.value`` etc.
        self.attn = RadixLinearAttention(
            layer_id=self.layer_idx,
            num_q_heads=self.num_heads,
            num_k_heads=self.num_heads,
            num_v_heads=self.num_heads,
            head_q_dim=self.head_dim,
            head_k_dim=self.head_dim,
            head_v_dim=self.head_dim,
            q_conv1d=self.conv_q,
            k_conv1d=self.conv_k,
            v_conv1d=self.conv_v,
            bias=None,
            activation=jax.nn.silu,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )

    def __call__(
        self,
        positions: jax.Array | None,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        recurrent_state_pool,
    ) -> tuple[jax.Array, object]:
        del positions  # KDA does not use RoPE

        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        # Forget gate: 1-tier projection [T, H] → [T, projection_size]
        raw_gate, _ = self.f_a_proj(hidden_states)
        raw_gate = raw_gate.reshape(hidden_states.shape[0], self.num_heads, self.head_dim)

        # Beta: per-head scalar gate
        beta = jax.nn.sigmoid(self.b_proj(hidden_states)[0].astype(jnp.float32))

        # KDA attention via RadixLinearAttention → KDAAttnBackend
        o, recurrent_state_pool = self.attn(
            forward_batch,
            q,
            k,
            v,
            raw_gate,
            beta,
            recurrent_state_pool,
        )
        o = o.reshape(hidden_states.shape[0], self.num_heads, self.head_dim)

        # Output gate: 1-tier projection
        output_gate, _ = self.g_a_proj(hidden_states)
        output_gate = output_gate.reshape(hidden_states.shape[0], self.num_heads, self.head_dim)

        # Output norm (gated RMSNorm, computed in fp32 to match reference):
        #   rmsnorm(o) × o_norm_weight × sigmoid(gate).
        # RMS uses mean-of-squares (variance), NOT sum.
        o = o.astype(jnp.float32)
        var = jnp.mean(jnp.square(o), axis=-1, keepdims=True)
        o = o * jax.lax.rsqrt(var + self.rms_norm_eps)
        o = o * self.o_norm_weight.value.astype(jnp.float32)
        o = o * jax.nn.sigmoid(output_gate.astype(jnp.float32))
        o = o.astype(hidden_states.dtype)
        o = o.reshape(hidden_states.shape[0], self.projection_size)
        o, _ = self.o_proj(o)

        return o, recurrent_state_pool


# ---------------------------------------------------------------------------
# Decoder Layer (hybrid: KDA or MLA + MoE/dense FFN)
# ---------------------------------------------------------------------------


class BailingMoeV3DecoderLayer(nnx.Module):
    """Hybrid decoder layer: KDA or MLA attention + MoE or dense MLP."""

    def __init__(
        self,
        config,
        mesh: jax.sharding.Mesh,
        layer_idx: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        lgs = getattr(config, "layer_group_size", 4)
        self.is_kda = _is_kda_layer(layer_idx, lgs)

        # Attention
        if self.is_kda:
            self.self_attn = BailingMoeV3KDAAttention(
                config=config,
                mesh=mesh,
                dtype=dtype,
                layer_idx=layer_idx,
            )
        else:
            self.self_attn = DeepseekV3Attention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                q_lora_rank=getattr(config, "q_lora_rank", None),
                kv_lora_rank=getattr(config, "kv_lora_rank", 512),
                qk_nope_head_dim=getattr(config, "qk_nope_head_dim", 128),
                qk_rope_head_dim=getattr(config, "qk_rope_head_dim", 64),
                v_head_dim=getattr(config, "v_head_dim", 128),
                mesh=mesh,
                layer_id=layer_idx,
                rope_theta=getattr(config, "rope_theta", 600000.0),
                rope_scaling=getattr(config, "rope_scaling", None),
                rope_interleave=getattr(config, "rope_interleave", True),
                max_position_embeddings=getattr(config, "max_position_embeddings", 262144),
                dtype=dtype,
                use_absorbed=getattr(config, "use_absorbed_mla", True),
                use_gate=getattr(config, "gated_attention_proj_granularity_type", None)
                is not None,
            )
            rope_scaling = getattr(config, "rope_scaling", None)
            if rope_scaling is not None and rope_scaling.get("mscale_all_dim", 0):
                mscale = _deepseek_yarn_get_mscale(
                    rope_scaling["factor"], rope_scaling["mscale_all_dim"]
                )
                self.self_attn.attn_mqa.scaling *= mscale * mscale
                self.self_attn.attn_mha.scaling *= mscale * mscale

        # FFN (MoE or dense)
        is_sparse = (
            getattr(config, "num_experts", 0) > 0
            and layer_idx >= getattr(config, "first_k_dense_replace", 0)
        )
        self.is_moe_layer = is_sparse
        self.moe_backend = getattr(config, "moe_backend", MoEBackend.EPMOE)
        self.use_fused = self.moe_backend in (
            MoEBackend.FUSED,
            MoEBackend.FUSED_V2,
            MoEBackend.FUSED_V4,
        )

        if not is_sparse:
            self.mlp = DeepseekV3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                mesh=mesh,
                dtype=dtype,
            )
            self.moe_gate = None
            self.shared_experts = None
        else:
            # Router
            self.moe_gate = GateLogit(
                input_size=config.hidden_size,
                num_experts=config.num_experts,
                enable_expert_bias=getattr(config, "moe_router_enable_expert_bias", False),
                weight_dtype=dtype,
                score_func=getattr(config, "score_function", "sigmoid"),
            )

            # TopK routing
            self.topk = TopK(
                topk=config.num_experts_per_tok,
                renormalize=getattr(config, "norm_topk_prob", True),
                num_expert_group=getattr(config, "n_group", 0),
                topk_group=getattr(config, "topk_group", 0),
                routed_scaling_factor=getattr(config, "routed_scaling_factor", 1.0),
                layer_id=layer_idx,
            )

            if self.use_fused:
                # Named ``mlp`` (mutually exclusive with the dense path) so the
                # MoE expert weight source path matches the Bailing checkpoint
                # layout ``model.layers.N.mlp.experts.*``.
                if self.moe_backend == MoEBackend.FUSED_V4:
                    fused_cls = FusedTPMoEV4
                elif self.moe_backend == MoEBackend.FUSED_V2:
                    fused_cls = FusedEPMoEV2
                else:
                    fused_cls = FusedEPMoE
                self.mlp = fused_cls(
                    hidden_size=config.hidden_size,
                    num_experts=config.num_experts,
                    num_experts_per_tok=config.num_experts_per_tok,
                    intermediate_dim=config.moe_intermediate_size,
                    mesh=mesh,
                    weight_dtype=dtype,
                    dtype=dtype,
                    layer_id=layer_idx,
                    ep_size=getattr(config, "ep_size", 1),
                    renormalize_topk_logits=getattr(config, "norm_topk_prob", True),
                    routed_scaling_factor=getattr(config, "routed_scaling_factor", 1.0),
                    use_grouped_topk=getattr(config, "n_group", 0) > 0,
                    num_groups=getattr(config, "n_group", 0),
                    top_k_groups=getattr(config, "topk_group", 0),
                    num_shared_experts=0,  # Shared experts handled externally
                    moe_shared_expert_intermediate_size=config.moe_intermediate_size,
                )
            else:
                self.mlp = EPMoE(
                    hidden_size=config.hidden_size,
                    num_experts=config.num_experts,
                    num_experts_per_tok=config.num_experts_per_tok,
                    intermediate_dim=config.moe_intermediate_size,
                    mesh=mesh,
                    weight_dtype=dtype,
                    dtype=dtype,
                    layer_id=layer_idx,
                    ep_size=getattr(config, "ep_size", 1),
                )

            # Shared experts (external DeepseekV3MLP)
            num_shared = getattr(config, "num_shared_experts", 0)
            if num_shared > 0:
                shared_intermediate = (
                    getattr(
                        config,
                        "moe_shared_expert_intermediate_size",
                        config.moe_intermediate_size,
                    )
                    * num_shared
                )
                self.shared_experts = DeepseekV3MLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=shared_intermediate,
                    mesh=mesh,
                    dtype=dtype,
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
        # Pre-norm residual pattern
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        # Attention
        if self.is_kda:
            kv_pool = memory_pools.recurrent_state_pool
        else:
            kv_pool = memory_pools.token_to_kv_pool

        hidden_states, kv_fused = self.self_attn(
            positions,
            hidden_states,
            forward_batch,
            kv_pool,
        )

        # Post-attention residual + norm
        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP (MoE or dense)
        if self.is_moe_layer:
            shared_output = self.shared_experts(hidden_states) if self.shared_experts else None

            router_logits = self.moe_gate(hidden_states)
            correction_bias = self.moe_gate.bias.value if self.moe_gate.bias is not None else None
            topk_weights, topk_ids = self.topk(
                router_logits, correction_bias, dispatch_info=dispatch_info
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

        return hidden_states, residual, kv_fused, topk_ids


# ---------------------------------------------------------------------------
# Transformer Model
# ---------------------------------------------------------------------------


class BailingMoeV3Model(nnx.Module):
    """BailingMoeV3 transformer body (no LM head)."""

    def __init__(
        self,
        config,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
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
                    config=config,
                    mesh=mesh,
                    layer_idx=i,
                    dtype=dtype,
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

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools,
    ) -> tuple[jax.Array, list, list, list]:
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
# Top-Level Causal LM
# ---------------------------------------------------------------------------


class BailingMoeV3ForCausalLM(nnx.Module):
    """BailingMoeV3 (Ling-V3-Flash / ring_v3_lite) for causal LM."""

    @classmethod
    def patch_model_config(cls, config: ModelConfig) -> None:
        """Set attention_arch and head_dim for MLA detection."""
        config.attention_arch = AttentionArch.MLA
        qk_nope = getattr(config.hf_text_config, "qk_nope_head_dim", 0)
        qk_rope = getattr(config.hf_text_config, "qk_rope_head_dim", 0)
        if qk_nope and qk_rope:
            config.head_dim = qk_nope + qk_rope

    def __init__(
        self, config, mesh: jax.sharding.Mesh, dtype: jnp.dtype = jnp.bfloat16
    ) -> None:
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        self.model = BailingMoeV3Model(
            config=config,
            mesh=mesh,
            dtype=dtype,
        )

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
        hidden_states, layers_kv_fused, layers_recurrent_state, layers_topk_ids = self.model(
            forward_batch,
            memory_pools,
        )
        if not getattr(self.config, "tie_word_embeddings", False):
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
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

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_weight_mappings()
        loader.load_weights_from_safetensors(weight_mappings)
        self._post_load_weights()
        logger.info("BailingMoeV3 weights loaded successfully!")

    def _post_load_weights(self):
        """Split kv_b_proj into absorbed w_uk/w_uv for MLA layers; guard KDA gates."""
        # FusedTPMoEV4 EP→TP reshard (must run before MLA/KDA post-load hooks
        # that may null out weight attributes).
        for layer in self.model.layers:
            mlp = getattr(layer, "mlp", None)
            if isinstance(mlp, FusedTPMoEV4):
                mlp.reshape_weights_for_tp()

        for layer in self.model.layers:
            attn = layer.self_attn
            if hasattr(attn, "post_load_weights"):
                attn.post_load_weights()

        # Guard against silently-unmapped KDA gate weights. The forget/output
        # gate projection is named ``f_a_proj`` (ring_v3_lite) or ``f_proj``
        # (ling_v3_flash) in the checkpoint; both names are registered as
        # aliases in ``_add_kda_mappings``. If a checkpoint uses neither name,
        # the loader skips it and the projection stays at zero init — fail
        # loudly instead of producing wrong output silently.
        lgs = getattr(self.config, "layer_group_size", 4)
        for i, layer in enumerate(self.model.layers):
            if not _is_kda_layer(i, lgs):
                continue
            attn = layer.self_attn
            for gate_name in ("f_a_proj", "g_a_proj"):
                kernel = getattr(attn, gate_name).weight.value
                if not bool(jax.device_get(jnp.any(kernel != 0))):
                    raise RuntimeError(
                        f"KDA layer {i} gate '{gate_name}' is all-zero after "
                        f"weight loading: the checkpoint exposes neither "
                        f"'{gate_name}' nor its alias "
                        f"('f_proj'/'g_proj'). Add the correct source name to "
                        f"_add_kda_mappings."
                    )

    def _create_weight_mappings(self) -> dict:
        config = self.config
        lgs = getattr(config, "layer_group_size", 4)
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        num_heads = config.num_attention_heads
        proj_k = num_heads * head_dim
        conv_kernel_size = getattr(config, "short_conv_kernel_size", 4)

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

        if not getattr(config, "tie_word_embeddings", False):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding",
                sharding=("tensor", None),
                transpose=False,
            )

        for layer_idx in range(config.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            target = f"model.layers.{layer_idx}"
            is_kda = _is_kda_layer(layer_idx, lgs)

            mappings.update(self._create_layer_mappings(
                layer_idx, prefix, target,
                is_kda=is_kda,
                proj_k=proj_k,
                conv_kernel_size=conv_kernel_size,
            ))

        return mappings

    def _create_layer_mappings(
        self,
        layer_idx: int,
        prefix: str,
        target: str,
        *,
        is_kda: bool,
        proj_k: int,
        conv_kernel_size: int,
    ) -> dict:
        config = self.config
        mappings = {}

        # Layer norms (all layers)
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

        # Attention mappings
        if is_kda:
            self._add_kda_mappings(mappings, prefix, target, proj_k, conv_kernel_size)
        else:
            self._add_mla_mappings(mappings, prefix, target)

        # FFN mappings
        is_sparse = (
            getattr(config, "num_experts", 0) > 0
            and layer_idx >= getattr(config, "first_k_dense_replace", 0)
        )
        if is_sparse:
            self._add_moe_mappings(mappings, prefix, target, layer_idx)
        else:
            self._add_dense_mappings(mappings, prefix, target)

        return mappings

    # ---- KDA attention mappings ----

    def _add_kda_mappings(
        self, mappings: dict, prefix: str, target: str,
        proj_k: int, conv_kernel_size: int,
    ) -> None:
        """Register weight mappings for a KDA attention layer.

        Gate projections are registered with BOTH checkpoint names per gate:
        ``f_a_proj`` (ring_v3_lite) and ``f_proj`` (ling_v3_flash) both map to
        ``f_a_proj.weight``. The same dual-registration applies to the output
        gate (``g_a_proj`` / ``g_proj``). The loader iterates over checkpoint
        keys, so only the name that actually exists is consumed; the other is
        silently skipped.
        """
        ckpt = f"{prefix}.attention"
        sa = f"{target}.self_attn"

        # Linear projections (unfused checkpoint layout)
        for name in ("q_proj", "k_proj", "v_proj"):
            mappings[f"{ckpt}.{name}.weight"] = WeightMapping(
                target_path=f"{sa}.{name}.weight",
                sharding=(None, "tensor"),
                transpose=True,
            )

        # Forget gate: register both source names → same target
        for src in ("f_a_proj", "f_proj"):
            mappings[f"{ckpt}.{src}.weight"] = WeightMapping(
                target_path=f"{sa}.f_a_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            )
        # Output gate: register both source names → same target
        for src in ("g_a_proj", "g_proj"):
            mappings[f"{ckpt}.{src}.weight"] = WeightMapping(
                target_path=f"{sa}.g_a_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            )

        mappings[f"{ckpt}.b_proj.weight"] = WeightMapping(
            target_path=f"{sa}.b_proj.weight",
            sharding=(None, "tensor"),
            transpose=True,
        )
        mappings[f"{ckpt}.o_proj.weight"] = WeightMapping(
            target_path=f"{sa}.o_proj.weight",
            sharding=("tensor", None),
            transpose=True,
        )

        # Conv1d: HF shape [dim, 1, kernel] → reshape to [dim, kernel], no transpose.
        # The conv LinearBases live under RadixLinearAttention (self_attn.attn),
        # registered as q_conv1d / k_conv1d / v_conv1d.
        for src_name, tgt_name in [
            ("q_conv1d", "q_conv1d"),
            ("k_conv1d", "k_conv1d"),
            ("v_conv1d", "v_conv1d"),
        ]:
            mappings[f"{ckpt}.{src_name}.weight"] = WeightMapping(
                target_path=f"{sa}.attn.{tgt_name}.weight",
                sharding=("tensor", None),
                transpose=False,
                reshape=(proj_k, conv_kernel_size),
            )

        # A_log [num_heads], dt_bias [proj_k], o_norm_weight [head_dim].
        # These target float32 nnx.Params; the loader casts to the target dtype.
        # A_log/dt_bias are shared with RadixLinearAttention (self_attn.attn);
        # nnx canonicalizes A_log under self_attn directly but dt_bias under
        # .attn (mirrors KimiDeltaAttention's resolved param paths).
        mappings[f"{ckpt}.A_log"] = WeightMapping(
            target_path=f"{sa}.A_log",
            sharding=("tensor",),
            transpose=False,
        )
        mappings[f"{ckpt}.dt_bias"] = WeightMapping(
            target_path=f"{sa}.attn.dt_bias",
            sharding=("tensor",),
            transpose=False,
        )
        mappings[f"{ckpt}.o_norm.weight"] = WeightMapping(
            target_path=f"{sa}.o_norm_weight",
            sharding=(None,),
            transpose=False,
        )

    # ---- MLA attention mappings ----

    def _add_mla_mappings(
        self, mappings: dict, prefix: str, target: str
    ) -> None:
        """Register weight mappings for an MLA attention layer.

        When ``q_lora_rank`` is None (ling_v3_flash), a single ``q_proj`` is
        used. When present (ring_v3_lite), the LoRA path ``q_a_proj`` →
        ``q_a_layernorm`` → ``q_b_proj`` is used instead.
        """
        config = self.config
        ckpt = f"{prefix}.attention"
        sa = f"{target}.self_attn"

        # Q path
        if getattr(config, "q_lora_rank", None) is not None:
            mappings[f"{ckpt}.q_a_proj.weight"] = WeightMapping(
                target_path=f"{sa}.q_a_proj.weight",
                sharding=(None, None),
                transpose=True,
            )
            mappings[f"{ckpt}.q_a_layernorm.weight"] = WeightMapping(
                target_path=f"{sa}.q_a_layernorm.scale",
                sharding=(None,),
                transpose=False,
            )
            mappings[f"{ckpt}.q_b_proj.weight"] = WeightMapping(
                target_path=f"{sa}.q_b_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            )
        else:
            mappings[f"{ckpt}.q_proj.weight"] = WeightMapping(
                target_path=f"{sa}.q_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            )

        # KV path
        mappings[f"{ckpt}.kv_a_proj_with_mqa.weight"] = WeightMapping(
            target_path=f"{sa}.kv_a_proj.weight",
            sharding=(None, None),
            transpose=True,
        )
        mappings[f"{ckpt}.kv_a_layernorm.weight"] = WeightMapping(
            target_path=f"{sa}.kv_a_layernorm.scale",
            sharding=(None,),
            transpose=False,
        )
        mappings[f"{ckpt}.kv_b_proj.weight"] = WeightMapping(
            target_path=f"{sa}.kv_b_proj.weight",
            sharding=(None, "tensor"),
            transpose=True,
        )

        # Output projection: HF emits ``attention.dense``; register both names
        # so the correct one is consumed regardless of checkpoint variant.
        mappings[f"{ckpt}.dense.weight"] = WeightMapping(
            target_path=f"{sa}.o_proj.weight",
            sharding=("tensor", None),
            transpose=True,
        )
        mappings[f"{ckpt}.o_proj.weight"] = WeightMapping(
            target_path=f"{sa}.o_proj.weight",
            sharding=("tensor", None),
            transpose=True,
        )

        # Optional head-wise output gate (gated_attention_proj_granularity_type).
        if getattr(config, "gated_attention_proj_granularity_type", None) is not None:
            mappings[f"{ckpt}.g_proj.weight"] = WeightMapping(
                target_path=f"{sa}.g_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            )

    # ---- Dense MLP mappings ----

    @staticmethod
    def _add_dense_mappings(mappings: dict, prefix: str, target: str) -> None:
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

    # ---- MoE mappings ----

    def _add_moe_mappings(
        self, mappings: dict, prefix: str, target: str, layer_idx: int,
    ) -> None:
        config = self.config

        # FusedTPMoEV4 replicates all experts on every chip; EPLB phy→log
        # remapping is meaningless and dangerous here.
        moe_backend = getattr(config, "moe_backend", None)
        if moe_backend is None:
            moe_backend = getattr(self, "moe_backend", None)
        if moe_backend == MoEBackend.FUSED_V4:
            metadata = get_global_expert_location_metadata()
            if metadata is not None:
                raise NotImplementedError(
                    "moe_backend=fused_v4 is incompatible with EPLB expert metadata. "
                    "FusedTPMoEV4 replicates all experts across chips; phy→log "
                    "remapping has no valid semantics."
                )

        # Router gate
        mappings[f"{prefix}.mlp.gate.weight"] = WeightMapping(
            target_path=f"{target}.moe_gate.kernel",
            sharding=(None, None),
            transpose=True,
        )
        if getattr(config, "moe_router_enable_expert_bias", False):
            mappings[f"{prefix}.mlp.gate.expert_bias"] = WeightMapping(
                target_path=f"{target}.moe_gate.bias",
                sharding=(None,),
                transpose=False,
            )

        # Expert weights
        num_logical_experts = config.num_experts

        metadata = get_global_expert_location_metadata()
        phy_to_log = None
        if metadata is not None:
            physical_to_logical_map = np.array(
                jax.device_get(metadata.physical_to_logical_map)
            )
            phy_to_log = physical_to_logical_map[layer_idx]

        moe_backend = getattr(config, "moe_backend", "epmoe")
        moe_mappings = create_moe_weights_mapping(
            prefix=prefix,
            target_prefix=target,
            num_experts=num_logical_experts,
            expert_type_names=("gate_proj", "up_proj", "down_proj"),
            moe_backend=moe_backend,
            moe_path="mlp",
            physical_to_logical_map=phy_to_log,
        )
        mappings.update(moe_mappings)

        # Shared experts
        num_shared = getattr(config, "num_shared_experts", 0)
        if num_shared > 0:
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


EntryClass = [BailingMoeV3ForCausalLM]
