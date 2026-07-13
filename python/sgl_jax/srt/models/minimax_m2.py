"""Inference-only MiniMax-M2 (M2/M2.5/M2.7) model.

Architecture: standard GQA + per-layer QK RMSNorm + partial RoPE + sigmoid-gated
MoE with e_score_correction_bias. All layers are MoE (no dense replace). Official
checkpoint is FP8 block-wise 128x128 (same format as DeepSeek-V3).

Reference: https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/main/modeling_minimax_m2.py
"""

import logging

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import lax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.eplb.expert_location import ExpertLocationMetadata
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, RotaryEmbedding
from sgl_jax.srt.layers.fused_moe import FusedEPMoE
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import EPMoE, GateLogit, TopK, create_moe_weights_mapping
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.profiling_utils import named_scope
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


class MiniMaxM2QKNorm(nnx.Module):
    """RMSNorm over the full ``num_heads * head_dim`` flat projection.

    MiniMax-M2 uses ``qk_norm_type="per_layer"``: variance is computed over all
    heads jointly (not per-head). With the head axis tensor-sharded, GSPMD
    inserts the cross-shard all-reduce for the ``mean(axis=-1)`` automatically.
    """

    def __init__(
        self,
        num_features: int,
        epsilon: float,
        param_dtype: jnp.dtype,
        kernel_axes: tuple[str | None, ...] = ("tensor",),
    ):
        self.epsilon = epsilon
        self.scale = nnx.Param(
            jnp.ones(
                (num_features,),
                dtype=param_dtype,
                out_sharding=P(*kernel_axes),
            )
        )

    @named_scope
    def __call__(self, x: jax.Array) -> jax.Array:
        orig_dtype = x.dtype
        x_f32 = jnp.asarray(x, jnp.float32)
        var = jnp.mean(lax.square(x_f32), axis=-1, keepdims=True)
        y = x_f32 * lax.rsqrt(var + self.epsilon)
        return (jnp.asarray(self.scale, jnp.float32) * y).astype(orig_dtype)


class MiniMaxM2Attention(nnx.Module):
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
        # FP8 static checkpoint: from_linear derives the QuantizedLinear
        # placeholder shapes from LinearBase.output_size *before* load-time
        # kv_head_padding, so size k/v_proj for the padded head count up front.
        # Replica count must match model_config.get_num_kv_head_replicas.
        tp = mesh.shape.get("tensor", 1)
        replicas = (tp + num_kv_heads - 1) // num_kv_heads if tp > num_kv_heads else 1
        self.kv_head_num = num_kv_heads * replicas
        self.q_head_num = num_heads
        self.scaling = self.head_dim**-0.5

        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        if self.use_qk_norm:
            self.q_norm = MiniMaxM2QKNorm(
                num_heads * self.head_dim,
                epsilon=config.rms_norm_eps,
                param_dtype=dtype,
            )
            self.k_norm = MiniMaxM2QKNorm(
                self.kv_head_num * self.head_dim,
                epsilon=config.rms_norm_eps,
                param_dtype=dtype,
            )

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
            max_position_embeddings=getattr(config, "max_position_embeddings", 196608),
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

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, jax.Array]:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        head_sharding = NamedSharding(self.mesh, P("data", "tensor"))
        q = q.reshape(-1, self.q_head_num, self.head_dim, out_sharding=head_sharding)
        k = k.reshape(-1, self.kv_head_num, self.head_dim, out_sharding=head_sharding)
        v = v.reshape(-1, self.kv_head_num, self.head_dim, out_sharding=head_sharding)

        q, k = self.rotary_emb(positions, q, k)
        attn_output, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool)

        output, _ = self.o_proj(attn_output)
        return output, kv_fused


class MiniMaxM2DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int,
        dtype: jnp.dtype,
    ):
        self.layer_id = layer_id
        self.self_attn = MiniMaxM2Attention(config, mesh=mesh, layer_id=layer_id, dtype=dtype)

        num_experts = config.num_local_experts
        num_experts_per_tok = config.num_experts_per_tok
        self.moe_backend = getattr(config, "moe_backend", "epmoe")
        self.use_fused = self.moe_backend == "fused"

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
            layer_id=layer_id,
            mesh=mesh,
        )

        moe_cls = FusedEPMoE if self.use_fused else EPMoE
        self.block_sparse_moe = moe_cls(
            hidden_size=config.hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            intermediate_dim=config.intermediate_size,
            mesh=mesh,
            ep_size=config.ep_size,
            weight_dtype=dtype,
            dtype=dtype,
            layer_id=layer_id,
            quantization_config=getattr(config, "quantization_config", None),
            **({"activation": "silu", "renormalize_topk_logits": True} if self.use_fused else {}),
        )

        self.input_layernorm = RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, param_dtype=dtype
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, param_dtype=dtype
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

        hidden_states, kv_fused = self.self_attn(
            positions, hidden_states, forward_batch, token_to_kv_pool
        )

        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        router_logits = self.moe_gate(hidden_states)
        correction_bias = self.moe_gate.bias.value if self.moe_gate.bias is not None else None
        topk_weights, topk_ids = self.topk(
            router_logits, correction_bias, dispatch_info=dispatch_info
        )
        if self.use_fused:
            mask = forward_batch.get_token_valid_mask(hidden_states.shape[0])
            topk_ids = jnp.where(mask[:, None], topk_ids, -1)
        hidden_states = self.block_sparse_moe(hidden_states, topk_weights, topk_ids)

        return hidden_states, residual, kv_fused, jax.sharding.reshard(topk_ids, P(None))


class MiniMaxM2Model(nnx.Module):
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
                MiniMaxM2DecoderLayer(config, mesh=mesh, layer_id=i, dtype=dtype)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps, param_dtype=dtype)

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ):
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

        hidden_states += residual
        hidden_states = self.norm(hidden_states)
        return hidden_states, layers_kv_fused, layers_topk_ids


class MiniMaxM2ForCausalLM(nnx.Module):
    @classmethod
    def patch_model_config(cls, mc: ModelConfig) -> None:
        # k/v_proj per-shard out_dim = head_dim = 128 == block_size_out (single
        # N-block) for any tp >= num_kv_heads. The narrow-N guard in the
        # block-wise kernel is conservative; allow it and rely on GSM8K sanity
        # to catch real accuracy collapse. Fallback: add k_proj/v_proj to
        # ignored_layers to dequant to BF16 (~372MB total).
        if mc.quantization_config is not None and mc.quantization_config.is_static_checkpoint:
            mc.quantization_config.allow_narrow_n_blockwise = True

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        logger.info("MiniMaxM2ForCausalLM dtype=%s", dtype)
        self.model = MiniMaxM2Model(config, mesh=mesh, dtype=dtype)
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
        hidden_states, layers_kv_fused, layers_topk_ids = self.model(
            forward_batch, memory_pools.token_to_kv_pool
        )
        output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        return output, {"token_to_kv_pool": layers_kv_fused}, True, layers_topk_ids

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self, model_config=model_config, mesh=self.mesh, dtype=self.dtype
        )
        loader.load_weights_from_safetensors(self._create_weight_mappings(model_config))
        self._maybe_pad_k_norm_scale()
        logger.info("MiniMaxM2 weights loaded successfully!")

    def _maybe_pad_k_norm_scale(self) -> None:
        """_apply_kv_head_padding only matches "k_proj"/"v_proj" hf_keys, so the
        per-layer k_norm scale (1D, num_kv_heads*head_dim) is loaded unpadded
        even with kv_head_padding=True. Tile per-head to match the padded k."""
        a0 = self.model.layers[0].self_attn
        if not a0.use_qk_norm or a0.k_norm.scale.shape[0] == a0.kv_head_num * a0.head_dim:
            return
        hd = a0.head_dim
        nkv = a0.k_norm.scale.shape[0] // hd
        ratio = a0.kv_head_num // nkv
        spec = NamedSharding(self.mesh, P("tensor"))
        for layer in self.model.layers:
            kn = layer.self_attn.k_norm
            ks = np.repeat(np.asarray(kn.scale.value).reshape(nkv, hd), ratio, axis=0).reshape(-1)
            kn.scale.value = jax.device_put(ks, spec)
        logger.info("Padded k_norm.scale %d→%d (×%d heads)", nkv * hd, a0.kv_head_num * hd, ratio)

    def _create_weight_mappings(self, model_config: ModelConfig) -> dict:
        quant_config = getattr(model_config, "quantization_config", None)
        is_static_quant = quant_config is not None and quant_config.is_static_checkpoint
        moe_backend = getattr(self.config, "moe_backend", "epmoe")
        use_fused = moe_backend == "fused"

        mappings = {
            "model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.norm.weight": WeightMapping(
                target_path="model.norm.scale", sharding=(None,), transpose=False
            ),
            "lm_head.weight": WeightMapping(
                target_path="lm_head.embedding", sharding=("tensor", None), transpose=False
            ),
        }
        for i in range(self.config.num_hidden_layers):
            mappings.update(self._create_layer_mappings(i, is_static_quant, moe_backend, use_fused))
        return mappings

    def _create_layer_mappings(
        self, layer_idx: int, is_static_quant: bool, moe_backend: str, use_fused: bool
    ) -> dict:
        prefix = target = f"model.layers.{layer_idx}"
        mappings: dict = {}

        def add_linear(hf: str, tgt: str, sharding_std: tuple, kv_head_padding: bool = False):
            if not is_static_quant:
                mappings[f"{hf}.weight"] = WeightMapping(
                    target_path=f"{tgt}.weight",
                    sharding=sharding_std,
                    transpose=True,
                    kv_head_padding=kv_head_padding,
                )
                return
            sharding_quant = (sharding_std[1], sharding_std[0])
            mappings[f"{hf}.weight"] = WeightMapping(
                target_path=f"{tgt}.weight_q",
                sharding=sharding_quant,
                transpose=False,
                kv_head_padding=kv_head_padding,
            )
            mappings[f"{hf}.weight_scale_inv"] = WeightMapping(
                target_path=f"{tgt}.weight_scale",
                sharding=sharding_quant,
                transpose=False,
                kv_head_padding=kv_head_padding,
            )

        for ln in ("input_layernorm", "post_attention_layernorm"):
            mappings[f"{prefix}.{ln}.weight"] = WeightMapping(
                target_path=f"{target}.{ln}.scale", sharding=(None,), transpose=False
            )

        ap, tp = f"{prefix}.self_attn", f"{target}.self_attn"
        add_linear(f"{ap}.q_proj", f"{tp}.q_proj", (None, "tensor"))
        add_linear(f"{ap}.k_proj", f"{tp}.k_proj", (None, "tensor"), kv_head_padding=True)
        add_linear(f"{ap}.v_proj", f"{tp}.v_proj", (None, "tensor"), kv_head_padding=True)
        add_linear(f"{ap}.o_proj", f"{tp}.o_proj", ("tensor", None))
        if getattr(self.config, "use_qk_norm", False):
            mappings[f"{ap}.q_norm.weight"] = WeightMapping(
                target_path=f"{tp}.q_norm.scale", sharding=("tensor",), transpose=False
            )
            # Load replicated so _maybe_pad_k_norm_scale can np.asarray it on
            # every host (sharded multi-host arrays are non-addressable).
            mappings[f"{ap}.k_norm.weight"] = WeightMapping(
                target_path=f"{tp}.k_norm.scale",
                sharding=(None,),
                transpose=False,
            )

        mappings[f"{prefix}.block_sparse_moe.gate.weight"] = WeightMapping(
            target_path=f"{target}.moe_gate.kernel", sharding=(None, None), transpose=True
        )
        if getattr(self.config, "use_routing_bias", True):
            mappings[f"{prefix}.block_sparse_moe.e_score_correction_bias"] = WeightMapping(
                target_path=f"{target}.moe_gate.bias", sharding=(None,), transpose=False
            )

        from sgl_jax.srt.eplb.expert_location import get_global_expert_location_metadata

        metadata = get_global_expert_location_metadata()
        phy_to_log = None
        if metadata is not None:
            phy_to_log = np.array(jax.device_get(metadata.physical_to_logical_map))[layer_idx]

        moe_mappings = create_moe_weights_mapping(
            prefix=prefix,
            target_prefix=target,
            num_experts=self.config.num_local_experts,
            expert_type_names=("w1", "w3", "w2"),
            moe_backend=moe_backend,
            moe_path="block_sparse_moe",
            source_expert_pattern="experts.{i}",
            physical_to_logical_map=phy_to_log,
        )
        mappings.update(moe_mappings)

        if is_static_quant and not use_fused:
            for moe_key, wm in moe_mappings.items():
                if not moe_key.startswith("__MOE_EXPERTS__"):
                    continue
                target_base = wm.target_path[0]
                expert_scale_keys = [
                    k.replace(".weight", ".weight_scale_inv") for k in wm.target_path[1:]
                ]
                mappings[f"__MOE_EXPERTS__{target_base}_scale"] = WeightMapping(
                    target_path=[f"{target_base}_scale"] + expert_scale_keys,
                    sharding=("expert", None, None),
                    transpose=False,
                    physical_to_logical_map=wm.physical_to_logical_map,
                )

        return mappings


EntryClass = [MiniMaxM2ForCausalLM]
