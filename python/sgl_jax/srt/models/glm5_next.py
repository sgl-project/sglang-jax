"""GLM-5.3-Flash (`glm5_next`) model for sglang-jax.

Architecture: 45 layers of `[3x KDA linear attention, 1x NoPE sparse MLA]`, a 288-expert MoE
(top-8 + 1 shared), and four-stream manifold-constrained hyper-connections (mHC).

This file composes three existing sglang-jax facilities rather than reimplementing them:

  * KDA layers  -> `KimiDeltaAttention` (models/kimi_linear.py), whose parameterization matches
                   GLM-5.3 exactly: `f_a/f_b` gate LoRA, `g_a/g_b` output gate, `b_proj` beta,
                   gated output RMSNorm, and `A_log`/`dt_bias`/`gate_lower_bound`.
  * DSA layers  -> `Glm5Attention` (models/glm5_moe.py), now parameterized for GLM-5.3's MLA dims
                   (`q_lora 1536`, `qk_nope 256`, `qk_rope 0`, `v 256`) and the K-pool indexer.
  * MoE         -> `GateLogit`/`TopK` + `FusedEPMoEV2` (routed and in-kernel shared experts); the
                   fused v2 kernel is the only MoE backend with GLM-5.3's swiglu clamp.

Per-layer routing (KDA -> `recurrent_state_pool`, DSA -> `token_to_kv_pool`) mirrors
`KimiDecoderLayer`; the DSA `DSAFusedCache` threading mirrors `Glm5Model`.

mHC (`hc_mult=4`) is implemented by `Glm5NextHC`, wrapping each sublayer; see the module docstring
there for the math and its provenance.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.glm5_next import Glm5NextConfig, get_glm5_next_config
from sgl_jax.srt.configs.model_config import ModelConfig, MoEBackend
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.model_loader.weights import WeightLoader, WeightSpec
from sgl_jax.srt.models.glm5_moe import (
    Glm5Attention,
    Glm5ForCausalLM,
    Glm5MLP,
    fused_v2_moe,
    moe_router,
)
from sgl_jax.srt.models.kimi_linear import KimiDeltaAttention, kda_attention_mappings

logger = logging.getLogger(__name__)


class Glm5NextHC(nnx.Module):
    """Four-stream manifold-constrained hyper-connection (mHC) around one sublayer.

    Math:

        flat   = rmsnorm_f32(streams.reshape(..., hc*D))            # no learned weight
        mix    = flat @ fn                                          # [..., (2+hc)*hc]
        pre_w, post_w, comb_w = split(mix)
        pre    = sigmoid(pre_w * scale[0] + base[:hc]) + hc_eps
        post   = 2 * sigmoid(post_w * scale[1] + base[hc:2hc])
        comb   = softmax(comb_w.reshape(hc,hc) * scale[2] + base[2hc:]) + hc_eps
        comb   = sinkhorn(comb, iters)
        h      = sum(pre * streams)                                 # collapse to [..., D]
        ... sublayer(h) -> y ...
        streams = post[..., None] * y[:, None, :] + comb @ streams  # write back

    `fn` is stored [hc*D, (2+hc)*hc] (JAX in/out); `base` [(2+hc)*hc]; `scale` [3]. A per-Sinkhorn
    normalisation is done with explicit slice adds (as in the reference) so the chain stays one
    fusion.
    """

    def __init__(
        self,
        hidden_size: int,
        hc_mult: int,
        hc_iters: int,
        hc_eps: float,
        rms_eps: float,
        mesh,
        dtype=jnp.bfloat16,
    ):
        self.hc = hc_mult
        self.hc_iters = hc_iters
        self.hc_eps = hc_eps
        self.eps = rms_eps
        n = (2 + hc_mult) * hc_mult
        sh = NamedSharding(mesh, P(None, None))
        sh1 = NamedSharding(mesh, P(None))
        self.fn = nnx.Param(jnp.zeros((hc_mult * hidden_size, n), dtype=dtype, out_sharding=sh))
        self.base = nnx.Param(jnp.zeros((n,), dtype=jnp.float32, out_sharding=sh1))
        self.scale = nnx.Param(jnp.zeros((3,), dtype=jnp.float32, out_sharding=sh1))

    def _mix(self, streams: jax.Array):
        """streams: [..., hc*D] (hc streams concatenated on the last axis)."""
        hc = self.hc
        lead = streams.shape[:-1]
        d = streams.shape[-1] // hc
        flat = streams.reshape(*lead, hc * d).astype(jnp.float32)
        flat = flat * jax.lax.rsqrt(jnp.mean(flat * flat, -1, keepdims=True) + self.eps)
        mix = flat @ self.fn.value.astype(jnp.float32)
        pre_w, post_w, comb_w = mix[..., :hc], mix[..., hc : 2 * hc], mix[..., 2 * hc :]
        base, scale = self.base.value, self.scale.value
        pre = jax.nn.sigmoid(pre_w * scale[0] + base[:hc]) + self.hc_eps
        post = 2.0 * jax.nn.sigmoid(post_w * scale[1] + base[hc : 2 * hc])
        comb = (
            jax.nn.softmax(
                comb_w.reshape(*lead, hc, hc) * scale[2] + base[2 * hc :].reshape(hc, hc), -1
            )
            + self.hc_eps
        )
        rsum = lambda c: sum(c[..., :, j : j + 1] for j in range(hc))
        csum = lambda c: sum(c[..., j : j + 1, :] for j in range(hc))
        comb = comb / (csum(comb) + self.hc_eps)
        for _ in range(self.hc_iters - 1):
            comb = comb / (rsum(comb) + self.hc_eps)
            comb = comb / (csum(comb) + self.hc_eps)
        streams_htd = streams.reshape(*lead, hc, d).astype(jnp.float32)
        collapsed = jnp.einsum("...h,...hd->...d", pre, streams_htd).astype(streams.dtype)
        return post, comb, collapsed

    def pre(self, streams: jax.Array):
        post, comb, collapsed = self._mix(streams)
        return collapsed, (post, comb)

    def post(self, y: jax.Array, state, streams: jax.Array):
        """y: [..., D] sublayer output. Returns the new [..., hc*D] streams."""
        post, comb = state
        hc = self.hc
        lead = streams.shape[:-1]
        d = streams.shape[-1] // hc
        streams_htd = streams.reshape(*lead, hc, d).astype(jnp.float32)
        y = y.astype(jnp.float32)
        out = post[..., None] * y[..., None, :] + jnp.einsum(
            "...ij,...id->...jd", comb, streams_htd
        )
        return out.reshape(*lead, hc * d).astype(streams.dtype)


class Glm5NextDecoderLayer(nnx.Module):
    def __init__(self, config: Glm5NextConfig, mesh, layer_id: int = 0, dtype=jnp.bfloat16):
        self.layer_id = layer_id
        self.is_kda = config.is_kda_layer(layer_id)
        self.swiglu_limit = config.swiglu_limit

        hc_args = (
            config.hidden_size,
            config.hc_mult,
            config.hc_sinkhorn_iters,
            config.hc_eps,
            config.rms_norm_eps,
        )
        self.hc_attn = Glm5NextHC(*hc_args, mesh, dtype)
        self.hc_ffn = Glm5NextHC(*hc_args, mesh, dtype)

        # attention
        if self.is_kda:
            self.self_attn = KimiDeltaAttention(
                config=config,
                layer_idx=layer_id,
                mesh=mesh,
                dtype=dtype,
                v_head_dim=config.linear_attn_config["head_dim"],
            )
        else:
            indexer_type = config.indexer_types[layer_id]
            self.self_attn = Glm5Attention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                max_position_embeddings=config.max_position_embeddings,
                mesh=mesh,
                rope_theta=config.rope_theta,
                rms_norm_eps=config.rms_norm_eps,
                use_qk_norm=config.use_qk_norm,
                layer_id=layer_id,
                dtype=dtype,
                use_absorbed=True,
                has_indexer=(indexer_type == "full"),
                indexer_type=indexer_type,
                use_dsa_sparse=True,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                index_head_dim=config.index_head_dim,
                index_n_heads=config.index_n_heads,
                index_kpool=config.index_kpool,
            )

        # norms
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

        # MLP
        if layer_id < config.first_k_dense_replace:
            self.mlp = Glm5MLP(
                config.hidden_size,
                config.intermediate_size,
                mesh,
                layer_id,
                dtype,
                use_fused=False,
                swiglu_limit=config.swiglu_limit,
            )
            self.is_moe_layer = False
        else:
            self._build_moe(config, mesh, layer_id, dtype)
            self.is_moe_layer = True

    def _build_moe(self, config: Glm5NextConfig, mesh, layer_id: int, dtype) -> None:
        """Routed + shared experts on the fused v2 kernel, the only one with the swiglu clamp."""
        self.moe_gate, self.topk = moe_router(config, mesh, layer_id)
        self.mlp = fused_v2_moe(config, mesh, layer_id, dtype, config.n_shared_experts)

    def __call__(
        self,
        positions,
        hidden_states,
        forward_batch,
        token_to_kv_pool,
        recurrent_state_pool,
        dispatch_info=None,
        dsa_topk_in=None,
        dsa_topk_pages_in=None,
    ):
        # ---- attention sublayer ----
        h, hc_state = self.hc_attn.pre(hidden_states)
        h = self.input_layernorm(h)

        if self.is_kda:
            attn_out, attn_state = self.self_attn(positions, h, forward_batch, recurrent_state_pool)
        else:
            attn_out, attn_state = self.self_attn(
                positions,
                h,
                forward_batch,
                token_to_kv_pool,
                dsa_topk_in=dsa_topk_in,
                dsa_topk_pages_in=dsa_topk_pages_in,
            )

        hidden_states = self.hc_attn.post(attn_out, hc_state, hidden_states)
        h, hc_state = self.hc_ffn.pre(hidden_states)
        h = self.post_attention_layernorm(h)

        # ---- MLP sublayer ----
        if self.is_moe_layer:
            router_logits = self.moe_gate(h)
            topk_weights, topk_ids = self.topk(
                router_logits, self.moe_gate.bias.value, dispatch_info=dispatch_info
            )
            # SwiGLU clamp on the routed and (in-kernel) shared experts.
            limit = self.swiglu_limit
            mlp_out = self.mlp(
                h, topk_weights, topk_ids, swiglu_limit=limit, shared_swiglu_limit=limit
            )
        else:
            mlp_out = self.mlp(h)
            topk_ids = None

        hidden_states = self.hc_ffn.post(mlp_out, hc_state, hidden_states)
        return hidden_states, attn_state, topk_ids


class Glm5NextModel(nnx.Module):
    def __init__(self, config: Glm5NextConfig, mesh, dtype=jnp.bfloat16):
        self.config = config
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
                Glm5NextDecoderLayer(config=config, layer_id=i, dtype=dtype, mesh=mesh)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, param_dtype=dtype, scope_name="norm"
        )

    def __call__(self, forward_batch: ForwardBatch, memory_pools):
        hc = self.config.hc_mult
        # mHC keeps hc parallel residual streams: broadcast the embedding across the hc axis.
        hidden_states = jnp.tile(self.embed_tokens(forward_batch.input_ids), (1, hc))
        layers_kv_fused, layers_idx_fused, layers_topk_ids = [], [], []
        layers_recurrent, layers_conv = [], []
        dsa_topk, dsa_topk_pages = None, None

        for layer in self.layers:
            hidden_states, attn_state, topk_ids = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                memory_pools.token_to_kv_pool,
                memory_pools.recurrent_state_pool,
                dispatch_info=forward_batch.expert_location_metadata,
                dsa_topk_in=dsa_topk,
                dsa_topk_pages_in=dsa_topk_pages,
            )
            if layer.is_kda:
                rec_buf, conv_buf_list = attn_state
                layers_recurrent.append(rec_buf)
                layers_conv.append(conv_buf_list)
            else:  # DSAFusedCache (dsa_sparse is required)
                layers_kv_fused.append(attn_state.kv)
                if attn_state.idx is not None:
                    layers_idx_fused.append(attn_state.idx)
                if attn_state.topk is not None:
                    dsa_topk = attn_state.topk
                if attn_state.topk_pages is not None:
                    dsa_topk_pages = attn_state.topk_pages
            layers_topk_ids.append(topk_ids)

        # Collapse the hc streams into one before the final norm: mean over the hc axis.
        hidden_states = hidden_states.reshape(hidden_states.shape[0], hc, -1).mean(axis=1)
        hidden_states = self.norm(hidden_states)
        return (
            hidden_states,
            layers_kv_fused,
            layers_idx_fused,
            (layers_recurrent, layers_conv),
            layers_topk_ids,
        )


class Glm5NextForConditionalGeneration(nnx.Module):
    """Serving class for `zai-org/GLM-5.3-Flash` (HF arch `Glm5NextForConditionalGeneration`)."""

    @classmethod
    def patch_model_config(cls, config: ModelConfig) -> None:
        from sgl_jax.srt.configs.model_config import AttentionArch

        text = config.hf_text_config
        config.attention_arch = AttentionArch.MLA
        config.head_dim = getattr(text, "qk_nope_head_dim", 256) + getattr(
            text, "qk_rope_head_dim", 0
        )
        if hasattr(config.hf_config, "head_dim"):
            config.hf_config.head_dim = config.head_dim
        qc = config.quantization_config
        if qc is not None and qc.is_static_checkpoint:
            # The checkpoint keeps the KDA short convs in BF16 (modules_to_not_convert names
            # them `self_attn.q_conv1d`); the model nests them under `self_attn.attn`.
            qc.ignored_layers = list(qc.ignored_layers or []) + [
                f"self_attn.attn.{name}_conv1d" for name in "qkv"
            ]

    def __init__(self, config, mesh, dtype=jnp.bfloat16):
        cfg = get_glm5_next_config(config)
        if cfg is None:
            raise ValueError(f"not a glm5_next config: {type(config).__name__}")
        # Runtime fields ModelConfig sets on the outer HF config.
        if getattr(config, "quantization_config", None) is not None:
            cfg.quantization_config = config.quantization_config
        cfg.moe_backend = getattr(config, "moe_backend", MoEBackend.EPMOE)
        if cfg.moe_backend != MoEBackend.FUSED_V2:
            raise ValueError("GLM-5.3-Flash currently supports only --moe-backend fused_v2")
        if not getattr(config, "use_dsa_sparse", False):
            raise ValueError("GLM-5.3-Flash requires --attention-backend dsa_sparse")
        cfg.ep_size = getattr(config, "ep_size", 1)
        self.config = cfg
        self.mesh = mesh
        self.dtype = dtype
        self.model = Glm5NextModel(cfg, dtype=dtype, mesh=mesh)
        if not cfg.tie_word_embeddings:
            self.lm_head = ParallelLMHead(
                cfg.vocab_size,
                cfg.hidden_size,
                dtype=dtype,
                param_dtype=dtype,
                mesh=mesh,
                enable_dp_lm_head=getattr(config, "enable_dp_lm_head", False),
            )
        self.logits_processor = LogitsProcessor(
            cfg.vocab_size,
            mesh=mesh,
            enable_dp_lm_head=getattr(config, "enable_dp_lm_head", False),
        )

    def __call__(self, forward_batch: ForwardBatch, memory_pools, logits_metadata: LogitsMetadata):
        hidden, kv_fused, idx_fused, recurrent, topk_ids = self.model(forward_batch, memory_pools)
        if not self.config.tie_word_embeddings:
            output = self.logits_processor(hidden, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden, self.model.embed_tokens, logits_metadata)
        kv_update = (kv_fused, idx_fused) if idx_fused else kv_fused
        return (
            output,
            {"token_to_kv_pool": kv_update, "recurrent_state_pool": recurrent},
            True,
            topk_ids,
        )

    def load_weights(self, model_config: ModelConfig):
        # Absorbed-MLA preparation runs as Glm5Attention's prepare_weight_loading hook.
        loader = WeightLoader(
            model=self, model_config=model_config, mesh=self.mesh, dtype=self.dtype
        )
        loader.load(self._create_weight_mappings(model_config))
        logger.info("GLM-5.3-Flash weights loaded.")

    def prepare_weight_loading(self, loader, mappings):
        # Block-wise FP8 shared experts are requantized per channel at load, as for GLM-5.2.
        return Glm5ForCausalLM.prepare_weight_loading(self, loader, mappings)

    def _create_weight_mappings(self, model_config: ModelConfig) -> dict:
        """HF checkpoint names -> sglang-jax params.

        Reuses GLM-5.2's DSA-layer and MoE mappings (`Glm5ForCausalLM._create_moe_layer_mappings`) for the
        `deepseek_sparse_attention` layers, and adds GLM-5.3's KDA projection mappings and mHC tensors.
        """
        cfg = self.config
        # The GLM-5.3-Flash checkpoint stores text tensors under `model.language_model.` (multimodal
        # wrapper); the model's own params are at `model.`. HF source keys use the former.
        HF = "model.language_model"
        mappings: dict = {
            f"{HF}.embed_tokens.weight": WeightSpec(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            f"{HF}.norm.weight": WeightSpec(
                target_path="model.norm.scale", sharding=(None,), transpose=False
            ),
        }
        if not cfg.tie_word_embeddings:
            mappings["lm_head.weight"] = self.lm_head.weight_mapping("lm_head.embedding")

        quant_config = model_config.quantization_config
        is_static_quant = quant_config is not None and quant_config.is_static_checkpoint
        # modules_to_not_convert lists BF16 modules by their model path (e.g. DSA kv_b_proj).
        unquantized = (
            frozenset(quant_config.ignored_layers or []) if is_static_quant else frozenset()
        )
        for i in range(cfg.num_hidden_layers):
            is_dense = i < cfg.first_k_dense_replace
            has_indexer = cfg.indexer_types[i] == "full"
            # Base mappings (norms + attention + MLP/MoE) via GLM-5.2's builder, which only reads
            # self.config. For KDA layers we then swap the MLA attention keys for KDA keys.
            layer_map = Glm5ForCausalLM._create_moe_layer_mappings(
                self,
                i,
                i,
                is_dense,
                is_static_quant,
                has_indexer,
                hf_prefix=f"{HF}.layers",
                unquantized=unquantized,
            )
            if cfg.is_kda_layer(i):
                layer_map = {k: v for k, v in layer_map.items() if ".self_attn." not in k}
                layer_map.update(
                    kda_attention_mappings(
                        f"{HF}.layers.{i}",
                        f"model.layers.{i}.self_attn",
                        cfg.linear_attn_config,
                        # Checkpoint A_log is 1-D [num_heads]; the param is [1, 1, H, 1].
                        a_log_shape=(1, 1, cfg.linear_attn_config["num_heads"], 1),
                    )
                )
            elif has_indexer:
                # GLM-5.3-Flash KPool indexer extra tensors (absent on GLM-5.1/5.2).
                hfp = f"{HF}.layers.{i}.self_attn.indexer"
                tgt = f"model.layers.{i}.self_attn.indexer"
                layer_map[f"{hfp}.index_kpool_compress_gate"] = WeightSpec(
                    target_path=f"{tgt}.index_kpool_compress_gate.weight",
                    sharding=(None, None),
                    transpose=True,
                )
                layer_map[f"{hfp}.index_kpool_compress_ape"] = WeightSpec(
                    target_path=f"{tgt}.kpool_ape", sharding=(None, None), transpose=False
                )
            mappings.update(layer_map)
            mappings.update(self._create_hc_mappings(i, HF))
        return mappings

    @staticmethod
    def _create_hc_mappings(i: int, HF: str) -> dict:
        """mHC tensors: hc_attn_{fn,base,scale}, hc_ffn_{fn,base,scale}."""
        p = f"{HF}.layers.{i}"
        out: dict = {}
        for site in ("attn", "ffn"):
            out[f"{p}.hc_{site}_fn"] = WeightSpec(
                target_path=f"model.layers.{i}.hc_{site}.fn", sharding=(None, None), transpose=True
            )
            out[f"{p}.hc_{site}_base"] = WeightSpec(
                target_path=f"model.layers.{i}.hc_{site}.base", sharding=(None,), transpose=False
            )
            out[f"{p}.hc_{site}_scale"] = WeightSpec(
                target_path=f"model.layers.{i}.hc_{site}.scale", sharding=(None,), transpose=False
            )
        return out

    def get_input_embeddings(self):
        return self.model.embed_tokens


EntryClass = [Glm5NextForConditionalGeneration]
