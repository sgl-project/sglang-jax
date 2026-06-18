"""MiniMax-M3 (text-only, dense-attention fallback).

MSA layers (3-59) use plain dense GQA here; mathematically identical to MSA
when seq_len <= sparse_topk_blocks * sparse_block_size = 2048. Index-branch
weights (`self_attn.index_*`) and vision/mtp weights are skipped at load time.
Ground truth: transformers `modular_minimax_m3_vl.py` (refs/ground-truth.md).
"""

import logging

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    get_global_expert_location_metadata,
)
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, RotaryEmbedding
from sgl_jax.srt.layers.layernorm import GemmaRMSNorm
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
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)

_HF_PREFIX = "language_model."
_SKIP_PREFIXES = ("vision_tower.", "multi_modal_projector.", "mtp.")


def _text_config(config: PretrainedConfig) -> PretrainedConfig:
    return getattr(config, "text_config", config)


def msa_block_topk(
    iq: jax.Array,
    ik_hist: jax.Array,
    seq_len: jax.Array,
    q_pos: jax.Array,
    *,
    block_size: int,
    topk: int,
    local_blocks: int,
) -> tuple[jax.Array, jax.Array]:
    """Pure MSA indexer: iq [H_idx, D], ik_hist [L_pad, D] (zero-padded past
    seq_len). Returns (topk_block_idx [topk], n_valid_blocks). Matches HF
    modular_minimax_m3_vl.py Indexer.forward (block max-pool over tokens then
    over heads, +inf boost on local blocks, -inf-padded topk left-aligned)."""
    n_blocks = ik_hist.shape[0] // block_size
    scores = jnp.einsum("hd,ld->hl", iq.astype(jnp.float32), ik_hist.astype(jnp.float32))
    valid = jnp.arange(ik_hist.shape[0]) < seq_len
    scores = jnp.where(valid[None, :], scores, -jnp.inf)
    block_scores = scores.reshape(iq.shape[0], n_blocks, block_size).max(-1).max(0)
    q_block = q_pos // block_size
    for j in range(local_blocks):
        block_scores = block_scores.at[jnp.maximum(q_block - j, 0)].set(jnp.inf)
    k = min(topk, n_blocks)
    topk_score, topk_idx = jax.lax.top_k(block_scores, k)
    n_valid = jnp.minimum((seq_len + block_size - 1) // block_size, k)
    topk_idx = jnp.where(topk_score > -jnp.inf, topk_idx, 0)
    if k < topk:
        topk_idx = jnp.pad(topk_idx, (0, topk - k))
    return topk_idx, n_valid


def swigluoai(gate: jax.Array, up: jax.Array, alpha: float, limit: float) -> jax.Array:
    gate = jnp.clip(gate, max=limit)
    up = jnp.clip(up, -limit, limit)
    return (up + 1.0) * gate * jax.nn.sigmoid(gate * alpha)


class MiniMaxM3MLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        swiglu_alpha: float,
        swiglu_limit: float,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype,
    ):
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_limit = swiglu_limit
        self.gate_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.up_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.down_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            use_bias=False,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        gate, _ = self.gate_proj(hidden_states)
        up, _ = self.up_proj(hidden_states)
        out, _ = self.down_proj(swigluoai(gate, up, self.swiglu_alpha, self.swiglu_limit))
        return out


class MiniMaxM3Attention(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int,
        dtype: jnp.dtype,
    ):
        self.mesh = mesh
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", hidden_size // num_heads)
        tp = mesh.shape.get("tensor", 1)
        replicas = (tp + num_kv_heads - 1) // num_kv_heads if tp > num_kv_heads else 1
        self.kv_head_num = num_kv_heads * replicas
        self.q_head_num = num_heads
        self.scaling = self.head_dim**-0.5

        self.q_norm = GemmaRMSNorm(self.head_dim, epsilon=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, epsilon=config.rms_norm_eps)

        self.q_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_heads * self.head_dim,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.k_proj = LinearBase(
            input_size=hidden_size,
            output_size=self.kv_head_num * self.head_dim,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.v_proj = LinearBase(
            input_size=hidden_size,
            output_size=self.kv_head_num * self.head_dim,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.o_proj = LinearBase(
            input_size=num_heads * self.head_dim,
            output_size=hidden_size,
            use_bias=False,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.rotary_emb = RotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=getattr(config, "rotary_dim", self.head_dim),
            max_position_embeddings=getattr(config, "max_position_embeddings", 1048576),
            base=getattr(config, "rope_theta", 5000000),
            is_neox_style=True,
            dtype=dtype,
        )
        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.kv_head_num,
            layer_id=layer_id,
        )

        sa = getattr(config, "sparse_attention_config", None) or {}
        self.is_sparse = bool(sa.get("sparse_attention_freq", [0] * (layer_id + 1))[layer_id])
        if self.is_sparse:
            self.index_n_heads = sa["sparse_num_index_heads"]
            self.index_head_dim = sa["sparse_index_dim"]
            self.topk_blocks = sa["sparse_topk_blocks"]
            self.block_size = sa["sparse_block_size"]
            self.local_blocks = sa.get("sparse_local_block", 1)
            self.index_q_proj = LinearBase(
                input_size=hidden_size,
                output_size=self.index_n_heads * self.index_head_dim,
                use_bias=False,
                kernel_axes=(None, None),
                params_dtype=dtype,
                mesh=mesh,
            )
            self.index_k_proj = LinearBase(
                input_size=hidden_size,
                output_size=self.index_head_dim,
                use_bias=False,
                kernel_axes=(None, None),
                params_dtype=dtype,
                mesh=mesh,
            )
            self.index_q_norm = GemmaRMSNorm(self.index_head_dim, epsilon=config.rms_norm_eps)
            self.index_k_norm = GemmaRMSNorm(self.index_head_dim, epsilon=config.rms_norm_eps)
            assert self.index_head_dim == self.head_dim, (
                "index RoPE reuses main rotary_emb (head_size=head_dim); "
                f"sparse_index_dim={self.index_head_dim} must equal head_dim={self.head_dim}"
            )

    def _compute_index_qk(
        self, hidden_states: jax.Array, positions: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Returns (iq [T, n_idx_heads, d_idx], ik [T, 1, d_idx]) post norm+RoPE."""
        rep = NamedSharding(self.mesh, P("data", None, None))
        iq, _ = self.index_q_proj(hidden_states)
        ik, _ = self.index_k_proj(hidden_states)
        iq = self.index_q_norm(
            iq.reshape(-1, self.index_n_heads, self.index_head_dim, out_sharding=rep)
        )
        ik = self.index_k_norm(ik.reshape(-1, 1, self.index_head_dim, out_sharding=rep))
        iq, ik = self.rotary_emb(positions, iq, ik)
        return iq, ik

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, jax.Array]:
        head_sharding = NamedSharding(self.mesh, P("data", "tensor"))
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        q = q.reshape(-1, self.q_head_num, self.head_dim, out_sharding=head_sharding)
        k = k.reshape(-1, self.kv_head_num, self.head_dim, out_sharding=head_sharding)
        v = v.reshape(-1, self.kv_head_num, self.head_dim, out_sharding=head_sharding)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)

        ik_upd = None
        if self.is_sparse and hasattr(token_to_kv_pool, "set_index_k_buffer"):
            iq, ik_new = self._compute_index_qk(hidden_states, positions)
            attn_output, kv_fused, ik_upd = self.attn(
                q,
                k,
                v,
                forward_batch,
                token_to_kv_pool,
                index_q=iq,
                index_k=ik_new,
                msa_topk=self.topk_blocks,
                msa_local_blocks=self.local_blocks,
            )
        else:
            attn_output, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool)
        output, _ = self.o_proj(attn_output)
        return output, kv_fused, ik_upd


class MiniMaxM3DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int,
        dtype: jnp.dtype,
    ):
        self.layer_id = layer_id
        self.is_moe = bool(config.moe_layer_freq[layer_id])
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        swiglu_alpha = getattr(config, "swiglu_alpha", 1.702)
        swiglu_limit = getattr(config, "swiglu_limit", 7.0)

        self.self_attn = MiniMaxM3Attention(config, mesh=mesh, layer_id=layer_id, dtype=dtype)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps
        )

        if not self.is_moe:
            self.mlp = MiniMaxM3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.dense_intermediate_size,
                swiglu_alpha=swiglu_alpha,
                swiglu_limit=swiglu_limit,
                mesh=mesh,
                dtype=dtype,
            )
            return

        num_experts = config.num_local_experts
        num_experts_per_tok = config.num_experts_per_tok
        self.use_fused = getattr(config, "moe_backend", "epmoe") == "fused"
        self.moe_gate = GateLogit(
            input_size=config.hidden_size,
            num_experts=num_experts,
            weight_dtype=dtype,
            enable_expert_bias=getattr(config, "use_routing_bias", True),
            score_func=getattr(config, "scoring_func", "sigmoid"),
        )
        self.topk = TopK(
            topk=num_experts_per_tok,
            renormalize=True,
            routed_scaling_factor=self.routed_scaling_factor,
            layer_id=layer_id,
        )
        if self.use_fused:
            self.block_sparse_moe = FusedEPMoE(
                hidden_size=config.hidden_size,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                intermediate_dim=config.intermediate_size,
                mesh=mesh,
                ep_size=config.ep_size,
                weight_dtype=dtype,
                dtype=dtype,
                layer_id=layer_id,
                activation="swigluoai",
                renormalize_topk_logits=True,
                routed_scaling_factor=self.routed_scaling_factor,
                num_shared_experts=config.n_shared_experts,
                moe_shared_expert_intermediate_size=config.shared_intermediate_size,
            )
            self.shared_experts = None
        else:
            self.block_sparse_moe = EPMoE(
                hidden_size=config.hidden_size,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                intermediate_dim=config.intermediate_size,
                mesh=mesh,
                ep_size=config.ep_size,
                weight_dtype=dtype,
                dtype=dtype,
                activation="swigluoai",
                swiglu_alpha=swiglu_alpha,
                swiglu_limit=swiglu_limit,
                layer_id=layer_id,
            )
            self.shared_experts = MiniMaxM3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_intermediate_size * config.n_shared_experts,
                swiglu_alpha=swiglu_alpha,
                swiglu_limit=swiglu_limit,
                mesh=mesh,
                dtype=dtype,
            )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: jax.Array | None,
        dispatch_info: ExpertLocationMetadata | None = None,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states, kv_fused, ik_upd = self.self_attn(
            positions, hidden_states, forward_batch, token_to_kv_pool
        )

        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if not self.is_moe:
            hidden_states = self.mlp(hidden_states)
            return hidden_states, residual, kv_fused, ik_upd, None

        router_logits = self.moe_gate(hidden_states)
        correction_bias = self.moe_gate.bias.value if self.moe_gate.bias is not None else None
        topk_weights, topk_ids = self.topk(
            router_logits, correction_bias, dispatch_info=dispatch_info
        )
        if self.use_fused:
            hidden_states = self.block_sparse_moe(hidden_states, topk_weights, topk_ids)
        else:
            shared_out = self.shared_experts(hidden_states)
            routed_out = self.block_sparse_moe(hidden_states, topk_weights, topk_ids)
            hidden_states = routed_out + shared_out

        return hidden_states, residual, kv_fused, ik_upd, jax.sharding.reshard(topk_ids, P(None))


class MiniMaxM3Model(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype,
    ):
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
                MiniMaxM3DecoderLayer(config, mesh=mesh, layer_id=i, dtype=dtype)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ):
        hidden_states = self.embed_tokens(forward_batch.input_ids)
        residual = None
        layers_kv_fused = []
        layers_ik = []
        layers_topk_ids = []
        for layer in self.layers:
            hidden_states, residual, kv_fused, ik_upd, topk_ids = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
                dispatch_info=forward_batch.expert_location_metadata,
            )
            layers_kv_fused.append(kv_fused)
            if ik_upd is not None:
                layers_ik.append(ik_upd)
            if topk_ids is not None:
                layers_topk_ids.append(topk_ids)

        hidden_states += residual
        hidden_states = self.norm(hidden_states)
        return hidden_states, layers_kv_fused, layers_ik, layers_topk_ids


class MiniMaxM3SparseForCausalLM(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = _text_config(config)
        # model_runner injects ep_size/moe_backend onto the *outer* hf_config;
        # propagate to text_config so DecoderLayer sees them.
        for k in ("ep_size", "moe_backend"):
            if hasattr(config, k):
                setattr(self.config, k, getattr(config, k))
        self.mesh = mesh
        self.dtype = dtype
        logger.info("MiniMaxM3SparseForCausalLM dtype=%s (dense-attn fallback)", dtype)
        self.model = MiniMaxM3Model(self.config, mesh=mesh, dtype=dtype)
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size, mesh=mesh)

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, layers_kv_fused, layers_ik, layers_topk_ids = self.model(
            forward_batch, memory_pools.token_to_kv_pool
        )
        output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        pool_updates = {"token_to_kv_pool": layers_kv_fused}
        if layers_ik:
            pool_updates["msa_index_k"] = layers_ik
        return output, pool_updates, True, layers_topk_ids

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self, model_config=model_config, mesh=self.mesh, dtype=self.dtype
        )
        # Mapping-driven loader: keys absent from the mapping (vision_tower.*,
        # multi_modal_projector.*, mtp.*) are silently skipped — see _SKIP_PREFIXES.
        loader.load_weights_from_safetensors(self._create_weight_mappings())
        logger.info("MiniMaxM3 weights loaded successfully!")

    def _create_weight_mappings(self) -> dict:
        cfg = self.config
        mappings = {
            f"{_HF_PREFIX}model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            f"{_HF_PREFIX}model.norm.weight": WeightMapping(
                target_path="model.norm.weight", sharding=(None,), transpose=False
            ),
            f"{_HF_PREFIX}lm_head.weight": WeightMapping(
                target_path="lm_head.embedding", sharding=("tensor", None), transpose=False
            ),
        }
        for i in range(cfg.num_hidden_layers):
            mappings.update(self._create_layer_mappings(i))
        return mappings

    def _create_layer_mappings(self, layer_idx: int) -> dict:
        cfg = self.config
        prefix = f"{_HF_PREFIX}model.layers.{layer_idx}"
        target = f"model.layers.{layer_idx}"
        is_moe = bool(cfg.moe_layer_freq[layer_idx])
        mappings: dict = {}

        def add_linear(hf: str, tgt: str, sharding: tuple, kv_head_padding: bool = False):
            mappings[f"{hf}.weight"] = WeightMapping(
                target_path=f"{tgt}.weight",
                sharding=sharding,
                transpose=True,
                kv_head_padding=kv_head_padding,
            )

        for ln in ("input_layernorm", "post_attention_layernorm"):
            mappings[f"{prefix}.{ln}.weight"] = WeightMapping(
                target_path=f"{target}.{ln}.weight", sharding=(None,), transpose=False
            )

        ap, tp = f"{prefix}.self_attn", f"{target}.self_attn"
        add_linear(f"{ap}.q_proj", f"{tp}.q_proj", (None, "tensor"))
        add_linear(f"{ap}.k_proj", f"{tp}.k_proj", (None, "tensor"), kv_head_padding=True)
        add_linear(f"{ap}.v_proj", f"{tp}.v_proj", (None, "tensor"), kv_head_padding=True)
        add_linear(f"{ap}.o_proj", f"{tp}.o_proj", ("tensor", None))
        for n in ("q_norm", "k_norm"):
            mappings[f"{ap}.{n}.weight"] = WeightMapping(
                target_path=f"{tp}.{n}.weight", sharding=(None,), transpose=False
            )

        sa = getattr(cfg, "sparse_attention_config", None) or {}
        if sa.get("sparse_attention_freq", [0] * (layer_idx + 1))[layer_idx]:
            add_linear(f"{ap}.index_q_proj", f"{tp}.index_q_proj", (None, None))
            add_linear(f"{ap}.index_k_proj", f"{tp}.index_k_proj", (None, None))
            for n in ("index_q_norm", "index_k_norm"):
                mappings[f"{ap}.{n}.weight"] = WeightMapping(
                    target_path=f"{tp}.{n}.weight", sharding=(None,), transpose=False
                )

        if not is_moe:
            mp, tm = f"{prefix}.mlp", f"{target}.mlp"
            add_linear(f"{mp}.gate_proj", f"{tm}.gate_proj", (None, "tensor"))
            add_linear(f"{mp}.up_proj", f"{tm}.up_proj", (None, "tensor"))
            add_linear(f"{mp}.down_proj", f"{tm}.down_proj", ("tensor", None))
            return mappings

        moe_backend = getattr(cfg, "moe_backend", "epmoe")
        use_fused = moe_backend == "fused"
        bp = f"{prefix}.block_sparse_moe"
        mappings[f"{bp}.gate.weight"] = WeightMapping(
            target_path=f"{target}.moe_gate.kernel", sharding=(None, None), transpose=True
        )
        mappings[f"{bp}.e_score_correction_bias"] = WeightMapping(
            target_path=f"{target}.moe_gate.bias", sharding=(None,), transpose=False
        )
        sp = f"{bp}.shared_experts"
        if use_fused:
            for hf_name, target_name in (
                ("gate_proj", "w1_shared"),
                ("up_proj", "w3_shared"),
                ("down_proj", "w2_shared"),
            ):
                mappings[f"{sp}.{hf_name}.weight"] = WeightMapping(
                    target_path=f"{target}.block_sparse_moe.{target_name}",
                    sharding=(None, None),
                    transpose=True,
                )
        else:
            ts = f"{target}.shared_experts"
            add_linear(f"{sp}.gate_proj", f"{ts}.gate_proj", (None, "tensor"))
            add_linear(f"{sp}.up_proj", f"{ts}.up_proj", (None, "tensor"))
            add_linear(f"{sp}.down_proj", f"{ts}.down_proj", ("tensor", None))

        metadata = get_global_expert_location_metadata()
        phy_to_log = None
        if metadata is not None:
            phy_to_log = np.array(jax.device_get(metadata.physical_to_logical_map))[layer_idx]

        mappings.update(
            create_moe_weights_mapping(
                prefix=prefix,
                target_prefix=target,
                num_experts=cfg.num_local_experts,
                expert_type_names=("w1", "w3", "w2"),
                moe_backend=moe_backend,
                moe_path="block_sparse_moe",
                source_expert_pattern="experts.{i}",
                physical_to_logical_map=phy_to_log,
            )
        )
        return mappings


class MiniMaxM3SparseForConditionalGeneration(MiniMaxM3SparseForCausalLM):
    """Alias for the top-level VL architecture name in config.json. Vision
    inputs are not supported; this routes text-only requests to the LM path."""


EntryClass = [MiniMaxM3SparseForCausalLM, MiniMaxM3SparseForConditionalGeneration]
