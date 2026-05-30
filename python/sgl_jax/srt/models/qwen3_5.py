"""Qwen3.5-35B-A3B hybrid-attention MoE (text-only for M1).

Layer layout (40 total, ``full_attention_interval=4``): full-attention at
indices 3, 7, ..., 39 (10 layers); Gated DeltaNet (GDN linear attention) at
all other indices (30 layers). Per RFC §3.3.

Key conventions confirmed against the upstream torch reference
(``_external_repos/sglang/python/sglang/srt/models/qwen3_5.py``):

* All transformer-level norms are ``GemmaRMSNorm`` (``(1 + weight)`` scaling):
  input/post-attention/q_norm/k_norm + the final model norm. Only the
  GDN-internal output norm is a plain ``RMSNorm`` followed by an explicit
  ``silu(z)`` gate (norm-before-gate, silu — NOT the fla sigmoid GatedRMSNorm).
* Full attention has an output gate: ``q_proj`` emits ``2*num_heads*head_dim``;
  per head the layout is ``[q(head_dim) | gate(head_dim)]``; ``gate`` skips
  q_norm/RoPE and applies as ``sigmoid(gate)`` after attention.
* MoE mirrors qwen2_moe: ``GateLogit`` + ``TopK`` + ``FusedEPMoE`` routed path,
  plus a dense ``Qwen2MoeMLP`` shared expert gated by ``sigmoid(shared_gate)``.
* GDN fuses HF's 4 in-proj keys into 2 JAX projections
  (``in_proj_qkvz`` = [Q|K|V|Z], ``in_proj_ba`` = [B|A]); the model reshards
  the sliced q/k/v/z/a/b to ``P("data","tensor")`` so each TP rank sees its
  head-striped shard (the GDN backend's ``shard_map`` contract). The conv1d
  weight is stripe-rearranged at load time (see the weight loader in P3).
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
from sgl_jax.srt.eplb.expert_location import ExpertLocationMetadata
from sgl_jax.srt.layers.embeddings import Embed, MRotaryEmbedding, ParallelLMHead
from sgl_jax.srt.layers.fused_moe import FusedEPMoE
from sgl_jax.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import GateLogit, TopK
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.layers.radix_linear_attention import RadixLinearAttention
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.qwen2_moe import Qwen2MoeMLP
from sgl_jax.srt.utils.weight_utils import WeightMapping

logger = logging.getLogger(__name__)


# =============================================================================
# Full attention (10 layers) — output-gated, GemmaRMSNorm q/k-norm, partial M-RoPE
# =============================================================================
class Qwen3_5Attention(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        text_cfg = config.text_config
        self.mesh = mesh
        self.layer_id = layer_id
        self.hidden_size = text_cfg.hidden_size
        self.num_heads = text_cfg.num_attention_heads
        self.num_kv_heads = text_cfg.num_key_value_heads
        self.head_dim = text_cfg.head_dim
        self.attn_output_gate = bool(getattr(text_cfg, "attn_output_gate", True))
        self.scaling = self.head_dim**-0.5

        q_mult = 2 if self.attn_output_gate else 1
        self.q_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.num_heads * q_mult * self.head_dim,
            mesh=mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="q_proj",
        )
        self.k_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.num_kv_heads * self.head_dim,
            mesh=mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="k_proj",
        )
        self.v_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.num_kv_heads * self.head_dim,
            mesh=mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="v_proj",
        )
        self.o_proj = LinearBase(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            mesh=mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=("tensor", None),
            scope_name="o_proj",
        )

        # Per-head GemmaRMSNorm over head_dim (Gemma (1+w) convention).
        self.q_norm = GemmaRMSNorm(self.head_dim, epsilon=text_cfg.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, epsilon=text_cfg.rms_norm_eps)

        # Partial M-RoPE: rotary_dim = head_dim * partial_rotary_factor = 64.
        # MRotaryEmbedding does NOT apply the partial factor itself, so pass the
        # final rotary_dim. mrope_section sums to rotary_dim // 2 = 32.
        partial = float(text_cfg.partial_rotary_factor)
        rotary_dim = int(self.head_dim * partial)
        rope_scaling = text_cfg.rope_scaling or {}
        mrope_section = rope_scaling["mrope_section"]
        mrope_interleaved = bool(rope_scaling.get("mrope_interleaved", False))
        self.rotary_emb = MRotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=rotary_dim,
            max_position_embeddings=text_cfg.max_position_embeddings,
            base=int(text_cfg.rope_theta),
            is_neox_style=True,
            dtype=dtype,
            mrope_section=mrope_section,
            mrope_interleaved=mrope_interleaved,
        )

        self.attn = RadixAttention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            layer_id=layer_id,
        )

    def __call__(self, positions, hidden_states, forward_batch, token_to_kv_pool):
        T = hidden_states.shape[0]
        q_raw, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        if self.attn_output_gate:
            # [T, num_heads, 2*head_dim] then split last dim → [q | gate] per head.
            q_gate = q_raw.reshape(
                T,
                self.num_heads,
                2 * self.head_dim,
                out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
            )
            q = q_gate[..., : self.head_dim]
            gate = q_gate[..., self.head_dim :]
        else:
            q = q_raw.reshape(
                T,
                self.num_heads,
                self.head_dim,
                out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
            )
            gate = None

        k = k.reshape(
            T,
            self.num_kv_heads,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )
        v = v.reshape(
            T,
            self.num_kv_heads,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )

        # q_norm/k_norm per head (gate skips both norm and RoPE).
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)

        attn_out, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool)

        if gate is not None:
            attn_out = attn_out.reshape(
                T,
                self.num_heads,
                self.head_dim,
                out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
            )
            attn_out = attn_out * jax.nn.sigmoid(gate)
            attn_out = attn_out.reshape(T, self.num_heads * self.head_dim)

        out, _ = self.o_proj(attn_out)
        return out, kv_fused


# =============================================================================
# Gated DeltaNet (30 layers) — fused in_proj_qkvz / in_proj_ba + GDN backend
# =============================================================================
class Qwen3_5GatedDeltaNet(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        text_cfg = config.text_config
        self.mesh = mesh
        self.layer_id = layer_id
        self.hidden_size = text_cfg.hidden_size
        self.num_k_heads = text_cfg.linear_num_key_heads
        self.num_v_heads = text_cfg.linear_num_value_heads
        self.head_k_dim = text_cfg.linear_key_head_dim
        self.head_v_dim = text_cfg.linear_value_head_dim
        self.conv_kernel_size = text_cfg.linear_conv_kernel_dim

        self.key_dim = self.num_k_heads * self.head_k_dim  # 2048
        self.value_dim = self.num_v_heads * self.head_v_dim  # 4096
        qkvz_out = 2 * self.key_dim + 2 * self.value_dim  # 12288 = [Q|K|V|Z]
        ba_out = 2 * self.num_v_heads  # 64 = [B|A]
        conv_dim = 2 * self.key_dim + self.value_dim  # 8192 = [Q|K|V]

        self.in_proj_qkvz = LinearBase(
            input_size=self.hidden_size,
            output_size=qkvz_out,
            mesh=mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="in_proj_qkvz",
        )
        self.in_proj_ba = LinearBase(
            input_size=self.hidden_size,
            output_size=ba_out,
            mesh=mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="in_proj_ba",
        )
        # conv1d is a parameter container only (never called); weight laid out
        # as [conv_dim, K] for the GDN backend's depthwise conv. The loader
        # stripe-rearranges it into rank-major [q_d|k_d|v_d] under TP>1.
        self.conv1d = LinearBase(
            input_size=conv_dim,
            output_size=self.conv_kernel_size,
            mesh=mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=("tensor", None),
            scope_name="conv1d",
        )

        self.A_log = nnx.Param(
            jnp.zeros((self.num_v_heads,), dtype=jnp.float32, out_sharding=P("tensor"))
        )
        self.dt_bias = nnx.Param(
            jnp.ones((self.num_v_heads,), dtype=jnp.float32, out_sharding=P("tensor"))
        )

        # GDN output norm: plain RMSNorm over head_v_dim, then explicit silu(z)
        # gate (norm-before-gate, silu — matches torch RMSNormGated).
        self.norm = RMSNorm(self.head_v_dim, epsilon=text_cfg.rms_norm_eps, param_dtype=dtype)
        self.out_proj = LinearBase(
            input_size=self.value_dim,
            output_size=self.hidden_size,
            mesh=mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=("tensor", None),
            scope_name="out_proj",
        )

        # RadixLinearAttention is a weight container + dispatch entry; the
        # actual GDN kernel runs in the hybrid backend (forward_batch.attn_backend).
        self.self_attn = RadixLinearAttention(
            layer_id=layer_id,
            num_q_heads=self.num_k_heads,
            num_k_heads=self.num_k_heads,
            num_v_heads=self.num_v_heads,
            head_q_dim=self.head_k_dim,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv1d=self.conv1d,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )

    def _shard_dt(self, x):
        # Reshard a sliced [T, C] tensor so its channel axis is head-striped
        # across "tensor" (each TP rank gets its head shard). No-op at TP=1.
        return jax.sharding.reshard(x, P("data", "tensor"))

    def __call__(self, positions, hidden_states, forward_batch, recurrent_state_pool):
        del positions  # GDN is position-agnostic.
        T = hidden_states.shape[0]
        qkvz, _ = self.in_proj_qkvz(hidden_states)  # [T, 2*key_dim + 2*value_dim]
        ba, _ = self.in_proj_ba(hidden_states)  # [T, 2*num_v_heads]

        kd, vd = self.key_dim, self.value_dim
        q = self._shard_dt(qkvz[:, :kd])
        k = self._shard_dt(qkvz[:, kd : 2 * kd])
        v = self._shard_dt(qkvz[:, 2 * kd : 2 * kd + vd])
        z = self._shard_dt(qkvz[:, 2 * kd + vd :])
        b = self._shard_dt(ba[:, : self.num_v_heads])
        a = self._shard_dt(ba[:, self.num_v_heads :])

        core_out, attn_state = self.self_attn(forward_batch, q, k, v, a, b, recurrent_state_pool)
        # core_out: [T, value_dim]. Per-head RMSNorm then silu(z) gate.
        core_out = core_out.reshape(
            T,
            self.num_v_heads,
            self.head_v_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )
        core_out = self.norm(core_out)
        core_out = core_out.reshape(
            T, self.value_dim, out_sharding=NamedSharding(self.mesh, P("data", "tensor"))
        )
        core_out = core_out * jax.nn.silu(z)
        out, _ = self.out_proj(core_out)
        return out, attn_state


# =============================================================================
# MoE block — routed FusedEPMoE + sigmoid-gated dense shared expert
# =============================================================================
class Qwen3_5MoeBlock(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        text_cfg = config.text_config
        self.layer_id = layer_id
        hidden = text_cfg.hidden_size
        inter = text_cfg.moe_intermediate_size
        se_inter = text_cfg.shared_expert_intermediate_size
        self.num_experts = text_cfg.num_experts
        self.top_k = text_cfg.num_experts_per_tok
        renorm = bool(getattr(text_cfg, "norm_topk_prob", True))
        # ep_size is injected on the ROOT hf_config by the model runner.
        ep_size = getattr(config, "ep_size", 1)

        self.moe_gate = GateLogit(
            input_size=hidden,
            num_experts=self.num_experts,
            weight_dtype=dtype,
        )
        self.topk = TopK(topk=self.top_k, renormalize=renorm, layer_id=layer_id)
        self.experts = FusedEPMoE(
            hidden_size=hidden,
            num_experts=self.num_experts,
            num_experts_per_tok=self.top_k,
            ep_size=ep_size,
            mesh=mesh,
            intermediate_dim=inter,
            weight_dtype=dtype,
            dtype=dtype,
            activation="silu",
            layer_id=layer_id,
            renormalize_topk_logits=renorm,
        )
        # Dense shared expert (always present for 35B-A3B) gated per-token.
        self.shared_experts = Qwen2MoeMLP(
            hidden_size=hidden,
            intermediate_size=se_inter,
            mesh=mesh,
            layer_id=layer_id,
            dtype=dtype,
            gate_up_down_bias=False,
        )
        self.shared_expert_gate = LinearBase(
            input_size=hidden,
            output_size=1,
            mesh=mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, None),
            scope_name="shared_expert_gate",
        )

    def __call__(self, hidden_states, forward_batch, dispatch_info=None):
        shared_out = self.shared_experts(hidden_states)
        gate_logit, _ = self.shared_expert_gate(hidden_states)  # [T, 1]
        shared_out = jax.nn.sigmoid(gate_logit) * shared_out

        router_logits = self.moe_gate(hidden_states)
        topk_weights, topk_ids = self.topk(router_logits, dispatch_info=dispatch_info)
        token_valid_mask = forward_batch.get_token_valid_mask(hidden_states.shape[0])
        topk_ids = jnp.where(token_valid_mask[:, None], topk_ids, -1)

        routed_out = self.experts(hidden_states, topk_weights, topk_ids)
        return routed_out + shared_out, jax.sharding.reshard(topk_ids, P(None))


# =============================================================================
# Decoder layer — picks full vs GDN attention per index
# =============================================================================
class Qwen3_5DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        text_cfg = config.text_config
        interval = int(text_cfg.full_attention_interval)
        self.is_full_attn = ((layer_id + 1) % interval) == 0

        # One-shot cross-check that the interval-derived schedule matches HF's
        # explicit layer_types (only on layer 0 to avoid 40x duplicate logs).
        if layer_id == 0:
            derived = [
                "full_attention" if ((i + 1) % interval == 0) else "linear_attention"
                for i in range(text_cfg.num_hidden_layers)
            ]
            declared = list(text_cfg.layer_types)
            assert derived == declared, (
                "Qwen3.5 layer_types mismatch — interval-derived vs HF text_config:\n"
                f"  derived  = {derived}\n"
                f"  declared = {declared}"
            )

        if self.is_full_attn:
            self.self_attn = Qwen3_5Attention(config, mesh, layer_id, dtype=dtype)
        else:
            self.self_attn = Qwen3_5GatedDeltaNet(config, mesh, layer_id, dtype=dtype)

        self.mlp = Qwen3_5MoeBlock(config, mesh, layer_id, dtype=dtype)
        self.input_layernorm = GemmaRMSNorm(text_cfg.hidden_size, epsilon=text_cfg.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            text_cfg.hidden_size, epsilon=text_cfg.rms_norm_eps
        )

    def __call__(
        self,
        positions,
        hidden_states,
        forward_batch,
        memory_pools,
        residual=None,
        dispatch_info: ExpertLocationMetadata | None = None,
    ):
        # Deferred-residual pre-norm pattern (mirrors qwen2_moe / kimi_linear).
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states = hidden_states + residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        if self.is_full_attn:
            pool = memory_pools.token_to_kv_pool
        else:
            pool = memory_pools.recurrent_state_pool
        hidden_states, attn_state = self.self_attn(positions, hidden_states, forward_batch, pool)

        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states, topk_ids = self.mlp(hidden_states, forward_batch, dispatch_info)
        return hidden_states, residual, attn_state, topk_ids


# =============================================================================
# Model / CausalLM / multimodal-style wrapper
# =============================================================================
class Qwen3_5MoeModel(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        text_cfg = config.text_config
        self.config = config
        self.embed_tokens = Embed(
            num_embeddings=text_cfg.vocab_size,
            features=text_cfg.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )
        self.layers = nnx.data(
            [
                Qwen3_5DecoderLayer(config, mesh, layer_id=i, dtype=dtype)
                for i in range(text_cfg.num_hidden_layers)
            ]
        )
        self.norm = GemmaRMSNorm(text_cfg.hidden_size, epsilon=text_cfg.rms_norm_eps)

    def __call__(self, forward_batch: ForwardBatch, memory_pools):
        hidden_states = self.embed_tokens(forward_batch.input_ids)
        residual = None
        layers_kv_fused = []
        layers_rec_buffers = []
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
            if layer.is_full_attn:
                layers_kv_fused.append(attn_state)
            else:
                rec_buf, conv_buf_list = attn_state
                layers_rec_buffers.append(rec_buf)
                layers_conv_buffers.append(conv_buf_list)
            layers_topk_ids.append(topk_ids)

        if residual is not None:
            hidden_states = hidden_states + residual
        hidden_states = self.norm(hidden_states)
        return (
            hidden_states,
            layers_kv_fused,
            (layers_rec_buffers, layers_conv_buffers),
            layers_topk_ids,
        )


class Qwen3_5MoeForCausalLM(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.model = Qwen3_5MoeModel(config, mesh, dtype=dtype)

    def __call__(self, forward_batch: ForwardBatch, memory_pools):
        return self.model(forward_batch, memory_pools)


class Qwen3_5MoeForConditionalGeneration(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.mesh = mesh
        self.dtype = dtype

        # Hidden-state-only language model (RFC §3.3). M2 replaces self.visual.
        self.language_model = Qwen3_5MoeForCausalLM(config, mesh, dtype=dtype)
        self.visual = None

        text_cfg = config.text_config
        self.tie_word_embeddings = bool(getattr(config, "tie_word_embeddings", False))
        if not self.tie_word_embeddings:
            self.lm_head = ParallelLMHead(
                text_cfg.vocab_size,
                text_cfg.hidden_size,
                dtype=dtype,
                param_dtype=dtype,
                kernel_axes=("tensor", None),
                mesh=mesh,
            )
        self.logits_processor = LogitsProcessor(text_cfg.vocab_size, mesh=mesh)

        # Fused-projection CPU staging buffers drained by the P3 weight loader.
        # {layer_idx: {hf_source: {"weight": np.ndarray}}}.
        self._fused_qkvz_buffers: dict[int, dict] = {}
        self._fused_ba_buffers: dict[int, dict] = {}

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
        pixel_values=None,
    ):
        assert pixel_values is None, "Qwen3.5 M1 is text-only; multimodal lands in M2"
        hidden_states, layers_kv_fused, layers_rec_state, layers_topk_ids = self.language_model(
            forward_batch, memory_pools
        )
        head = self.language_model.model.embed_tokens if self.tie_word_embeddings else self.lm_head
        output = self.logits_processor(hidden_states, head, logits_metadata)
        return (
            output,
            {
                "token_to_kv_pool": layers_kv_fused,
                "recurrent_state_pool": layers_rec_state,
            },
            True,
            layers_topk_ids,
        )

    def load_weights(self, model_config: ModelConfig):
        # Real loader (GDN 4->2 concat + conv1d stripe + MoE pre-fused split)
        # lands in P3 (Task P3.4).
        raise NotImplementedError("Qwen3.5 weight loader lands in P3.")


# =============================================================================
# Weight mapping table (HF source key -> JAX target). The fused-projection
# sentinels (__FUSED_QKVZ_* / __FUSED_BA_*) and the conv1d stripe + MoE
# gate_up split are made functional by the P3 WeightLoader changes; here the
# table is pure data so every HF text key is accounted for (coverage test).
# =============================================================================
_VISUAL_SKIP_PATTERNS = [r"^model\.visual\..+"]
_MTP_SKIP_PATTERNS = [r"^mtp\..+"]


def _create_qwen3_5_weight_mappings(hf_config):
    """Return (mappings, visual_skip_patterns, mtp_skip_patterns).

    Source keys mirror the 35B-A3B safetensors layout: full-attn layers use
    ``...self_attn.*``, GDN layers use ``...linear_attn.*``; MoE experts are
    pre-fused (``mlp.experts.gate_up_proj`` / ``...down_proj``). Targets are
    rooted at the multimodal wrapper's ``language_model.model.*`` prefix.
    """
    tc = hf_config.text_config
    interval = int(tc.full_attention_interval)
    num_layers = int(tc.num_hidden_layers)
    key_dim = tc.linear_num_key_heads * tc.linear_key_head_dim
    value_dim = tc.linear_num_value_heads * tc.linear_value_head_dim
    conv_dim = 2 * key_dim + value_dim
    conv_k = int(tc.linear_conv_kernel_dim)

    mappings: dict[str, WeightMapping] = {}

    # Top-level
    mappings["model.language_model.embed_tokens.weight"] = WeightMapping(
        target_path="language_model.model.embed_tokens.embedding",
        sharding=("tensor", None),
        transpose=False,
    )
    mappings["model.language_model.norm.weight"] = WeightMapping(
        target_path="language_model.model.norm.weight",
        sharding=(None,),
        transpose=False,
    )
    mappings["lm_head.weight"] = WeightMapping(
        target_path="lm_head.embedding",
        sharding=("tensor", None),
        transpose=False,
    )

    for i in range(num_layers):
        is_full = ((i + 1) % interval) == 0
        src = f"model.language_model.layers.{i}"
        dst = f"language_model.model.layers.{i}"

        # GemmaRMSNorm -> .weight (both layer types)
        mappings[f"{src}.input_layernorm.weight"] = WeightMapping(
            target_path=f"{dst}.input_layernorm.weight", sharding=(None,), transpose=False
        )
        mappings[f"{src}.post_attention_layernorm.weight"] = WeightMapping(
            target_path=f"{dst}.post_attention_layernorm.weight", sharding=(None,), transpose=False
        )

        if is_full:
            mappings[f"{src}.self_attn.q_proj.weight"] = WeightMapping(
                target_path=f"{dst}.self_attn.q_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            )
            mappings[f"{src}.self_attn.k_proj.weight"] = WeightMapping(
                target_path=f"{dst}.self_attn.k_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                kv_head_padding=True,
            )
            mappings[f"{src}.self_attn.v_proj.weight"] = WeightMapping(
                target_path=f"{dst}.self_attn.v_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                kv_head_padding=True,
            )
            mappings[f"{src}.self_attn.o_proj.weight"] = WeightMapping(
                target_path=f"{dst}.self_attn.o_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            )
            mappings[f"{src}.self_attn.q_norm.weight"] = WeightMapping(
                target_path=f"{dst}.self_attn.q_norm.weight", sharding=(None,), transpose=False
            )
            mappings[f"{src}.self_attn.k_norm.weight"] = WeightMapping(
                target_path=f"{dst}.self_attn.k_norm.weight", sharding=(None,), transpose=False
            )
        else:
            # GDN: 4 HF in-proj keys -> 2 fused JAX params via __FUSED_* sentinels.
            mappings[f"{src}.linear_attn.in_proj_qkv.weight"] = WeightMapping(
                target_path=f"__FUSED_QKVZ_QKV_WEIGHT__{i}", sharding=(None, None), transpose=False
            )
            mappings[f"{src}.linear_attn.in_proj_z.weight"] = WeightMapping(
                target_path=f"__FUSED_QKVZ_Z_WEIGHT__{i}", sharding=(None, None), transpose=False
            )
            mappings[f"{src}.linear_attn.in_proj_b.weight"] = WeightMapping(
                target_path=f"__FUSED_BA_B_WEIGHT__{i}", sharding=(None, None), transpose=False
            )
            mappings[f"{src}.linear_attn.in_proj_a.weight"] = WeightMapping(
                target_path=f"__FUSED_BA_A_WEIGHT__{i}", sharding=(None, None), transpose=False
            )
            # conv1d HF shape [conv_dim, 1, K] -> [conv_dim, K]; loader stripes it.
            mappings[f"{src}.linear_attn.conv1d.weight"] = WeightMapping(
                target_path=f"{dst}.self_attn.conv1d.weight",
                sharding=("tensor", None),
                transpose=False,
                reshape=(conv_dim, conv_k),
            )
            mappings[f"{src}.linear_attn.A_log"] = WeightMapping(
                target_path=f"{dst}.self_attn.A_log", sharding=("tensor",), transpose=False
            )
            mappings[f"{src}.linear_attn.dt_bias"] = WeightMapping(
                target_path=f"{dst}.self_attn.dt_bias", sharding=("tensor",), transpose=False
            )
            mappings[f"{src}.linear_attn.norm.weight"] = WeightMapping(
                target_path=f"{dst}.self_attn.norm.scale", sharding=(None,), transpose=False
            )
            mappings[f"{src}.linear_attn.out_proj.weight"] = WeightMapping(
                target_path=f"{dst}.self_attn.out_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            )

        # MoE (all 40 layers). Experts are pre-fused on disk.
        mappings[f"{src}.mlp.gate.weight"] = WeightMapping(
            target_path=f"{dst}.mlp.moe_gate.kernel", sharding=(None, None), transpose=True
        )
        mappings[f"{src}.mlp.experts.gate_up_proj"] = WeightMapping(
            target_path=[f"{dst}.mlp.experts.w1", f"{dst}.mlp.experts.w3"],
            sharding=(("data", "tensor"), None, None),
            transpose=False,
        )
        mappings[f"{src}.mlp.experts.down_proj"] = WeightMapping(
            target_path=f"{dst}.mlp.experts.w2",
            sharding=(("data", "tensor"), None, None),
            transpose=False,
        )
        mappings[f"{src}.mlp.shared_expert.gate_proj.weight"] = WeightMapping(
            target_path=f"{dst}.mlp.shared_experts.gate_proj.weight",
            sharding=(None, "tensor"),
            transpose=True,
        )
        mappings[f"{src}.mlp.shared_expert.up_proj.weight"] = WeightMapping(
            target_path=f"{dst}.mlp.shared_experts.up_proj.weight",
            sharding=(None, "tensor"),
            transpose=True,
        )
        mappings[f"{src}.mlp.shared_expert.down_proj.weight"] = WeightMapping(
            target_path=f"{dst}.mlp.shared_experts.down_proj.weight",
            sharding=("tensor", None),
            transpose=True,
        )
        mappings[f"{src}.mlp.shared_expert_gate.weight"] = WeightMapping(
            target_path=f"{dst}.mlp.shared_expert_gate.weight",
            sharding=(None, None),
            transpose=True,
        )

    return mappings, _VISUAL_SKIP_PATTERNS, _MTP_SKIP_PATTERNS


EntryClass = Qwen3_5MoeForConditionalGeneration
