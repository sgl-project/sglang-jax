"""DeepSeek V3 model implementation.

Architecture: Multi-head Latent Attention (MLA) + Mixture-of-Experts (MoE)
with shared experts and dense layers for the first K layers.

Reference: deepseek_jax.py (pure JAX/NNX reference implementation).
This file adapts that reference to the sglang-jax serving infrastructure,
reusing existing modules: RadixAttention, GateLogit, TopK, EPMoE/FusedEPMoE.
"""

import logging

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import AttentionArch, ModelConfig, MoEBackend
from sgl_jax.srt.eplb.expert_location import ExpertLocationMetadata
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
    GateLogit,
    TopK,
    create_moe_weights_mapping,
)
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


class DeepseekV3MLP(nnx.Module):
    """SwiGLU MLP used for dense layers and shared experts."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
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
        output, _ = self.down_proj(jax.nn.silu(gate) * up)
        return output


class DeepseekV3Attention(nnx.Module):
    """Multi-head Latent Attention (non-absorbed mode) for DeepSeek V2/V3.

    Decompresses the latent state into Q/K/V during forward and reuses the
    standard RadixAttention + MHATokenToKVPool path — ~43x more KV cache than
    the absorbed form; will be replaced once the MLA Pallas kernel is
    production-ready. Matches the convention used by every other model file
    (Llama, Qwen*, Grok, ...) where the model owns its own attention wrapper
    around RadixAttention rather than pulling from a shared layers module.

    Data flow:
      Q path:  hidden -> q_a_proj -> norm -> q_b_proj -> split(q_nope, q_rope)
               (or hidden -> q_proj -> split, when q_lora_rank is None)
      KV path: hidden -> kv_a_proj -> split(compressed, k_rope)
               compressed -> norm -> kv_b_proj -> split(k_nope, v)
      RoPE applied only to q_rope and k_rope.
      Assembly: Q = concat(q_nope, q_rope'), K = concat(k_nope, k_rope').
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
        max_position_embeddings: int = 163840,
        dtype: jnp.dtype = jnp.bfloat16,
        use_absorbed: bool = True,
    ):
        super().__init__()
        self.mesh = mesh
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank

        if q_lora_rank is None:
            self.q_proj = LinearBase(
                hidden_size,
                num_heads * self.qk_head_dim,
                mesh,
                use_bias=False,
                params_dtype=dtype,
                kernel_axes=(None, "tensor"),
                scope_name="q_proj",
            )
        else:
            self.q_a_proj = LinearBase(
                hidden_size,
                q_lora_rank,
                mesh,
                use_bias=False,
                params_dtype=dtype,
                kernel_axes=(None, None),
                scope_name="q_a_proj",
            )
            self.q_a_layernorm = RMSNorm(q_lora_rank, dtype=dtype)
            self.q_b_proj = LinearBase(
                q_lora_rank,
                num_heads * self.qk_head_dim,
                mesh,
                use_bias=False,
                params_dtype=dtype,
                kernel_axes=(None, "tensor"),
                scope_name="q_b_proj",
            )

        self.kv_a_proj = LinearBase(
            hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, None),
            scope_name="kv_a_proj",
        )
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, dtype=dtype)
        self.kv_b_proj = LinearBase(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="kv_b_proj",
        )

        self.o_proj = LinearBase(
            num_heads * v_head_dim,
            hidden_size,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=("tensor", None),
            scope_name="o_proj",
        )

        self.rotary_emb = get_rope(
            head_size=qk_rope_head_dim,
            rotary_dim=qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=int(rope_theta),
            is_neox_style=not rope_interleave,
            rope_scaling=rope_scaling,
            dtype=dtype,
        )

        self.attn_mqa = RadixAttention(
            num_heads=num_heads,
            head_dim=kv_lora_rank + qk_rope_head_dim,
            scaling=self.qk_head_dim**-0.5,
            num_kv_heads=1,
            v_head_dim=kv_lora_rank,
            layer_id=layer_id,
        )

        self.use_absorbed = use_absorbed
        # Absorbed-MLA fused projections, populated by post_load_weights() after
        # weight loading: w_uk[R, n_h, D_k] folds W_UK into Q, w_uv[R, n_h, D_v]
        # folds W_UV into the output. Placeholders here so nnx tracks them in
        # the model state tree; sharded on the head dim like kv_b_proj.weight.
        if use_absorbed:
            uk_axes = (None, "tensor", None)
            self.w_uk = nnx.Param(
                jnp.zeros(
                    (kv_lora_rank, num_heads, qk_nope_head_dim),
                    dtype=dtype,
                    out_sharding=P(*uk_axes),
                )
            )
            self.w_uv = nnx.Param(
                jnp.zeros(
                    (kv_lora_rank, num_heads, v_head_dim),
                    dtype=dtype,
                    out_sharding=P(*uk_axes),
                )
            )
        # Non-absorbed (MHA) RadixAttention: every Q head has its own K/V,
        # head_dim is the concatenated nope+rope dim. We pad V to the same
        # head_dim before calling FlashAttention (RPA v3 requires k/v share
        # head_dim) and slice back to v_head_dim after.
        self.attn_mha = RadixAttention(
            num_heads=num_heads,
            head_dim=self.qk_head_dim,
            scaling=self.qk_head_dim**-0.5,
            num_kv_heads=num_heads,
            v_head_dim=self.qk_head_dim,
            layer_id=layer_id,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, jax.Array]:
        """MLA forward. Two paths, picked at server start via --attention-backend:

        - `fa` (default, absorbed): fold W_UK into Q, attend latent c_kv via
          MLA kernel (MQA). Cache stores latent. See docs/design/MLA.md §3.9.
        - `fa_mha` (non-absorbed): kv_b_proj decompresses to per-head K/V, run
          FlashAttention. ~70x more KV memory; for A/B and short-context only.
        """
        # Shared front-half: project Q, project KV-A latent, apply RoPE.
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
        q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)

        if self.use_absorbed:
            attn_output, kv_fused = self._forward_mqa(
                q_nope, q_rope, compressed, k_rope, forward_batch, token_to_kv_pool
            )
        else:
            attn_output, kv_fused = self._forward_mha(
                q_nope, q_rope, compressed, k_rope, forward_batch, token_to_kv_pool
            )

        output, _ = self.o_proj(attn_output)
        return output, kv_fused

    def post_load_weights(self):
        """Split kv_b_proj.weight into absorbed-MLA folded projections.

        kv_b_proj.weight has layout [kv_lora_rank, n_h * (qk_nope+v_head_dim)],
        head-major with [nope, v] within each head block. Absorbed MLA folds
        W_UK into Q (pre-attn) and W_UV into the output (post-attn), so once
        split we can drop kv_b_proj entirely. Mirrors sglang's
        deepseek_weight_loader.post_process(). No-op for non-absorbed path.
        """
        if not self.use_absorbed:
            return
        w_kv = self.kv_b_proj.weight.value.reshape(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        self.w_uk.value = w_kv[:, :, : self.qk_nope_head_dim]
        self.w_uv.value = w_kv[:, :, self.qk_nope_head_dim :]
        self.kv_b_proj = None

    def _forward_mqa(
        self,
        q_nope: jax.Array,
        q_rope: jax.Array,
        compressed: jax.Array,
        k_rope: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, jax.Array]:
        # ql_nope[t, h, r] = sum_d q_nope[t, h, d] * w_uk[r, h, d]
        ql_nope = jnp.einsum("thd,rhd->thr", q_nope, self.w_uk.value)

        # Latent K/V are a single shared head — pack into [T, 1, *] for MQA.
        c_kv_3d = compressed[:, None, :]

        o_latent, kv_fused = self.attn_mqa(
            ql_nope,
            c_kv_3d,
            c_kv_3d,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
            q_rope=q_rope,
            k_rope=k_rope,
        )

        # o_v[t, h, d] = sum_r o_latent[t, h, r] * w_uv[r, h, d]
        o_v = jnp.einsum("thr,rhd->thd", o_latent, self.w_uv.value)
        attn_output = o_v.reshape(-1, self.num_heads * self.v_head_dim)
        return attn_output, kv_fused

    def _forward_mha(
        self,
        q_nope: jax.Array,
        q_rope: jax.Array,
        compressed: jax.Array,
        k_rope: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, jax.Array]:
        # Decompress latent c_kv into per-head K-nope and V via kv_b_proj.
        kv, _ = self.kv_b_proj(compressed)
        kv = kv.reshape(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = jnp.split(kv, [self.qk_nope_head_dim], axis=-1)

        # k_rope is shared across heads — broadcast with the head-sharded spec
        # so the downstream concat keeps "tensor" sharding on the head axis.
        k_rope = jnp.broadcast_to(
            k_rope,
            (k_rope.shape[0], self.num_heads, self.qk_rope_head_dim),
            out_sharding=P(None, "tensor", None),
        )

        q = jnp.concatenate([q_nope, q_rope], axis=-1)
        k = jnp.concatenate([k_nope, k_rope], axis=-1)

        # RPA v3 requires k.shape[-1] == v.shape[-1]; attn_mha is configured with
        # v_head_dim=qk_head_dim, so pad V on the head dim and strip the pad off
        # the output below.
        v = jnp.pad(v, ((0, 0), (0, 0), (0, self.qk_head_dim - self.v_head_dim)))

        attn_output, kv_fused = self.attn_mha(
            q, k, v, forward_batch=forward_batch, token_to_kv_pool=token_to_kv_pool
        )

        # Strip V padding: o_proj expects num_heads * v_head_dim.
        attn_output = attn_output.reshape(-1, self.num_heads, self.qk_head_dim)
        attn_output = attn_output[:, :, : self.v_head_dim].reshape(
            -1, self.num_heads * self.v_head_dim
        )
        return attn_output, kv_fused


class DeepseekV3DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size

        rope_theta = getattr(config, "rope_theta", 10000.0)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 163840)

        # MLA attention
        q_lora_rank = getattr(config, "q_lora_rank", None)
        kv_lora_rank = getattr(config, "kv_lora_rank", 512)
        qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 128)
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 64)
        v_head_dim = getattr(config, "v_head_dim", 128)

        self.self_attn = DeepseekV3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            mesh=mesh,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rope_interleave=True,
            max_position_embeddings=max_position_embeddings,
            dtype=dtype,
            use_absorbed=getattr(config, "use_absorbed_mla", True),
        )

        # Adjust attention scaling for YaRN mscale
        if rope_scaling is not None:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", 0)
            if mscale_all_dim:
                scaling_factor = rope_scaling["factor"]
                mscale = _deepseek_yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.self_attn.attn_mqa.scaling *= mscale * mscale
                self.self_attn.attn_mha.scaling *= mscale * mscale

        # MLP: MoE or dense depending on layer index
        n_routed_experts = getattr(config, "n_routed_experts", None)
        first_k_dense_replace = getattr(config, "first_k_dense_replace", 0)
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)

        is_moe = (
            n_routed_experts is not None
            and layer_id >= first_k_dense_replace
            and layer_id % moe_layer_freq == 0
        )
        self.is_moe_layer = is_moe

        if not is_moe:
            self.mlp = DeepseekV3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                mesh=mesh,
                dtype=dtype,
            )
            self.moe_gate = None
        else:
            self._init_moe(config, mesh, layer_id, dtype)

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

    def _init_moe(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int,
        dtype: jnp.dtype,
    ):
        n_routed_experts = config.n_routed_experts
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 8)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 2048)
        n_group = getattr(config, "n_group", 1)
        topk_group = getattr(config, "topk_group", 1)
        norm_topk_prob = getattr(config, "norm_topk_prob", True)
        routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        scoring_func = getattr(config, "scoring_func", "sigmoid")
        topk_method = getattr(config, "topk_method", "noaux_tc")
        ep_size = getattr(config, "ep_size", 1)
        moe_backend = getattr(config, "moe_backend", MoEBackend.EPMOE)
        n_shared_experts = getattr(config, "n_shared_experts", None)

        # Gate
        self.moe_gate = GateLogit(
            input_size=config.hidden_size,
            num_experts=n_routed_experts,
            enable_expert_bias=(topk_method == "noaux_tc"),
            weight_dtype=dtype,
            score_func=scoring_func,
        )

        # TopK routing
        self.topk = TopK(
            topk=num_experts_per_tok,
            renormalize=norm_topk_prob,
            num_expert_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            layer_id=layer_id,
        )

        # Expert computation
        self.use_fused = moe_backend == MoEBackend.FUSED

        if self.use_fused:
            self.mlp = FusedEPMoE(
                hidden_size=config.hidden_size,
                num_experts=n_routed_experts,
                num_experts_per_tok=num_experts_per_tok,
                intermediate_dim=moe_intermediate_size,
                mesh=mesh,
                ep_size=ep_size,
                weight_dtype=dtype,
                dtype=dtype,
                layer_id=layer_id,
                renormalize_topk_logits=norm_topk_prob,
                routed_scaling_factor=routed_scaling_factor,
                use_grouped_topk=n_group > 0,
                num_groups=n_group,
                top_k_groups=topk_group,
                num_shared_experts=(n_shared_experts or 0),
                moe_shared_expert_intermediate_size=moe_intermediate_size,
                quantization_config=getattr(config, "quantization_config", None),
            )
        else:
            self.mlp = EPMoE(
                hidden_size=config.hidden_size,
                num_experts=n_routed_experts,
                num_experts_per_tok=num_experts_per_tok,
                intermediate_dim=moe_intermediate_size,
                mesh=mesh,
                ep_size=ep_size,
                weight_dtype=dtype,
                dtype=dtype,
                layer_id=layer_id,
                quantization_config=getattr(config, "quantization_config", None),
            )

        # Shared experts (non-fused: separate MLP; fused: built into FusedEPMoE)
        if n_shared_experts and n_shared_experts > 0 and not self.use_fused:
            self.shared_experts = DeepseekV3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=moe_intermediate_size * n_shared_experts,
                mesh=mesh,
                dtype=dtype,
            )
        else:
            self.shared_experts = None

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: jax.Array | None = None,
        dispatch_info: ExpertLocationMetadata | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None]:
        # Pre-norm with fused residual
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        # MLA attention
        hidden_states, kv_fused = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )

        # Post-attention residual + norm
        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP (MoE or dense)
        if self.is_moe_layer:
            if self.shared_experts is not None:
                shared_output = self.shared_experts(hidden_states)
            else:
                shared_output = None

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


class DeepseekV3Model(nnx.Module):
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
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )

        self.layers = nnx.data(
            [
                DeepseekV3DecoderLayer(
                    config=config,
                    mesh=mesh,
                    layer_id=i,
                    dtype=dtype,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="norm",
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, list, list]:
        hidden_states = self.embed_tokens(forward_batch.input_ids)
        residual = None
        layers_kv_fused = []
        layers_topk_ids = []

        for layer in self.layers:
            hidden_states, residual, kv_fused, topk_ids = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
                dispatch_info=forward_batch.expert_location_metadata,
            )
            layers_kv_fused.append(kv_fused)
            layers_topk_ids.append(topk_ids)

        if residual is not None:
            hidden_states += residual

        hidden_states = self.norm(hidden_states)
        return hidden_states, layers_kv_fused, layers_topk_ids


class DeepseekV3ForCausalLM(nnx.Module):
    @classmethod
    def patch_model_config(cls, mc: ModelConfig) -> None:
        # MLA: Q/K head dim is qk_nope + qk_rope; V head dim differs (v_head_dim).
        # Override the generic head_dim so the attention backend and KV pool
        # allocate MLA-shaped buffers.
        mc.attention_arch = AttentionArch.MLA
        qk_nope = getattr(mc.hf_text_config, "qk_nope_head_dim", 0)
        qk_rope = getattr(mc.hf_text_config, "qk_rope_head_dim", 0)
        if qk_nope and qk_rope:
            mc.head_dim = qk_nope + qk_rope

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        self.model = DeepseekV3Model(config, mesh=mesh, dtype=dtype)

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
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, layers_kv_fused, layers_topk_ids = self.model(
            forward_batch, token_to_kv_pool
        )
        output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        return output, layers_kv_fused, True, layers_topk_ids

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_weight_mappings(model_config)
        loader.load_weights_from_safetensors(weight_mappings)
        # Absorbed-MLA path: pre-split kv_b_proj into w_uk/w_uv and drop the
        # original projection (sglang parity, deepseek_weight_loader.post_process).
        for layer in self.model.layers:
            layer.self_attn.post_load_weights()
        logger.info("DeepSeek V3 weights loaded successfully!")

    def _create_weight_mappings(self, model_config: ModelConfig) -> dict:
        quant_config = getattr(model_config, "quantization_config", None)
        is_static_quant = quant_config is not None and quant_config.is_static_checkpoint

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
            "lm_head.weight": WeightMapping(
                target_path="lm_head.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
        }

        n_routed_experts = getattr(self.config, "n_routed_experts", None)
        first_k_dense_replace = getattr(self.config, "first_k_dense_replace", 0)
        moe_layer_freq = getattr(self.config, "moe_layer_freq", 1)
        moe_backend = getattr(self.config, "moe_backend", "epmoe")
        use_fused = moe_backend == "fused"

        for layer_idx in range(self.config.num_hidden_layers):
            is_moe = (
                n_routed_experts is not None
                and layer_idx >= first_k_dense_replace
                and layer_idx % moe_layer_freq == 0
            )
            layer_mappings = self._create_layer_mappings(
                layer_idx, is_moe, moe_backend, use_fused, is_static_quant
            )
            mappings.update(layer_mappings)

        return mappings

    def _create_layer_mappings(
        self,
        layer_idx: int,
        is_moe: bool,
        moe_backend: str,
        use_fused: bool,
        is_static_quant: bool = False,
    ) -> dict:
        prefix = f"model.layers.{layer_idx}"
        target = f"model.layers.{layer_idx}"
        mappings: dict = {}

        def add_linear(hf_prefix: str, target_prefix: str, sharding_std: tuple):
            # HF weights are `[out, in]`.
            #   Unquantized: loaded into LinearBase.weight `[in, out]` so
            #   `transpose=True` and sharding is the LinearBase kernel_axes.
            #   Static FP8: loaded into QuantizedLinear.weight_q `[out, in]`
            #   directly; sharding is kernel_axes swapped. Also register the
            #   `weight_scale_inv` sidecar into `weight_scale`.
            if not is_static_quant:
                mappings[f"{hf_prefix}.weight"] = WeightMapping(
                    target_path=f"{target_prefix}.weight",
                    sharding=sharding_std,
                    transpose=True,
                )
                return
            sharding_quant = (sharding_std[1], sharding_std[0])
            mappings[f"{hf_prefix}.weight"] = WeightMapping(
                target_path=f"{target_prefix}.weight_q",
                sharding=sharding_quant,
                transpose=False,
            )
            # HF block scale is `[out_blocks, in_blocks]`; WeightLoader's
            # _maybe_expand_linear_block_scale expands to `[in_blocks, 1, n_out]`.
            # QuantizedLinear.__call__ expects sharding P(kernel_axes[0], None,
            # kernel_axes[1]) on the 3D result, which maps back to the 2D
            # checkpoint as sharding_quant = (out_blocks_axis, in_blocks_axis):
            #   col-parallel: ("tensor", None) → 3D axis 2 sharded.
            #   row-parallel: (None, "tensor") → 3D axis 0 sharded.
            mappings[f"{hf_prefix}.weight_scale_inv"] = WeightMapping(
                target_path=f"{target_prefix}.weight_scale",
                sharding=sharding_quant,
                transpose=False,
            )

        # Layer norms
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

        # MLA attention. HF 'kv_a_proj_with_mqa' -> JAX 'kv_a_proj'.
        ap = f"{prefix}.self_attn"
        tp = f"{target}.self_attn"
        q_lora_rank = getattr(self.config, "q_lora_rank", None)
        if q_lora_rank is None:
            add_linear(f"{ap}.q_proj", f"{tp}.q_proj", (None, "tensor"))
        else:
            add_linear(f"{ap}.q_a_proj", f"{tp}.q_a_proj", (None, None))
            mappings[f"{ap}.q_a_layernorm.weight"] = WeightMapping(
                target_path=f"{tp}.q_a_layernorm.scale",
                sharding=(None,),
                transpose=False,
            )
            add_linear(f"{ap}.q_b_proj", f"{tp}.q_b_proj", (None, "tensor"))
        add_linear(f"{ap}.kv_a_proj_with_mqa", f"{tp}.kv_a_proj", (None, None))
        mappings[f"{ap}.kv_a_layernorm.weight"] = WeightMapping(
            target_path=f"{tp}.kv_a_layernorm.scale",
            sharding=(None,),
            transpose=False,
        )
        add_linear(f"{ap}.kv_b_proj", f"{tp}.kv_b_proj", (None, "tensor"))
        add_linear(f"{ap}.o_proj", f"{tp}.o_proj", ("tensor", None))

        if not is_moe:
            # Dense MLP (first K layers)
            for proj, sharding in [
                ("gate_proj", (None, "tensor")),
                ("up_proj", (None, "tensor")),
                ("down_proj", ("tensor", None)),
            ]:
                add_linear(f"{prefix}.mlp.{proj}", f"{target}.mlp.{proj}", sharding)
            return mappings

        # MoE gate (router) — NOT quantized in HF FP8 checkpoint.
        mappings[f"{prefix}.mlp.gate.weight"] = WeightMapping(
            target_path=f"{target}.moe_gate.kernel",
            sharding=(None, None),
            transpose=True,
        )
        if getattr(self.config, "topk_method", "noaux_tc") == "noaux_tc":
            mappings[f"{prefix}.mlp.gate.e_score_correction_bias"] = WeightMapping(
                target_path=f"{target}.moe_gate.bias",
                sharding=(None,),
                transpose=False,
            )

        # Expert weights (aggregated loading via create_moe_weights_mapping)
        from sgl_jax.srt.eplb.expert_location import get_global_expert_location_metadata

        metadata = get_global_expert_location_metadata()
        phy_to_log = None
        if metadata is not None:
            physical_to_logical_map = np.array(jax.device_get(metadata.physical_to_logical_map))
            phy_to_log = physical_to_logical_map[layer_idx]

        moe_mappings = create_moe_weights_mapping(
            prefix=prefix,
            target_prefix=target,
            num_experts=self.config.n_routed_experts,
            expert_type_names=("gate_proj", "up_proj", "down_proj"),
            moe_backend=moe_backend,
            physical_to_logical_map=phy_to_log,
        )
        mappings.update(moe_mappings)

        # Routed expert weight-scale sidecars (static FP8, non-fused only).
        # Fused MoE static-FP8 placeholder shapes are (1,) today — loading
        # block scales would need a dedicated fix in fused_moe.py. Skip.
        #
        # For each expert weight group (emitted by create_moe_weights_mapping
        # as `__MOE_EXPERTS__<target_base>`) register a parallel scale group
        # whose HF keys are the per-expert `*.weight_scale_inv` tensors and
        # whose target is `<target_base>_scale` (e.g. `wi_0_scale`). After
        # stacking, WeightLoader's _maybe_convert_epmoe_scale_for_kernel
        # converts the `[E, out_blocks, in_blocks]` layout into the
        # kernel-ready `[E, k_blocks, 1, n_out]` expected by GMM.
        if is_static_quant and not use_fused:
            for moe_key, wm in moe_mappings.items():
                if not moe_key.startswith("__MOE_EXPERTS__"):
                    continue
                target_base = wm.target_path[0]
                expert_scale_keys = [
                    k.replace(".weight", ".weight_scale_inv") for k in wm.target_path[1:]
                ]
                scale_target = f"{target_base}_scale"
                # Stacked checkpoint scale is `[E, out_blocks, in_blocks]`. Load
                # replicated on the block dims; _maybe_convert_epmoe_scale_for_kernel
                # expands via jnp.take, which fails if the gathered axis is
                # tensor-sharded (ambiguous output sharding). The converter reshards
                # to model_param.value.sharding at the end.
                mappings[f"__MOE_EXPERTS__{scale_target}"] = WeightMapping(
                    target_path=[scale_target] + expert_scale_keys,
                    sharding=("expert", None, None),
                    transpose=False,
                    physical_to_logical_map=wm.physical_to_logical_map,
                )

        # Shared expert weights
        n_shared_experts = getattr(self.config, "n_shared_experts", None)
        if n_shared_experts and n_shared_experts > 0:
            if use_fused:
                # Fused backend stores shared-expert weights as bare nnx.Param
                # (no QuantizedLinear swap). Static-FP8 scale mappings for the
                # fused path are TODO (requires fixed placeholder shapes in
                # FusedEPMoE.quantize_weights(is_static=True)).
                for hf_name, target_name in [
                    ("gate_proj", "w1_shared"),
                    ("up_proj", "w3_shared"),
                    ("down_proj", "w2_shared"),
                ]:
                    mappings[f"{prefix}.mlp.shared_experts.{hf_name}.weight"] = WeightMapping(
                        target_path=f"{target}.mlp.{target_name}",
                        sharding=(None, None),
                        transpose=True,
                    )
            else:
                for proj, sharding in [
                    ("gate_proj", (None, "tensor")),
                    ("up_proj", (None, "tensor")),
                    ("down_proj", ("tensor", None)),
                ]:
                    add_linear(
                        f"{prefix}.mlp.shared_experts.{proj}",
                        f"{target}.shared_experts.{proj}",
                        sharding,
                    )

        return mappings


class DeepseekV2ForCausalLM(DeepseekV3ForCausalLM):
    """DeepSeek V2 shares the same MLA + MoE architecture as V3."""

    pass


EntryClass = [DeepseekV3ForCausalLM, DeepseekV2ForCausalLM]
