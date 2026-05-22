"""MiMo-V2.5-Pro multi-token-prediction (NextN/MTP) draft model.

One instance loads exactly one of the 3 MTP layers from
``model_mtp.safetensors`` (selected via ``config.mtp_layer_idx``); the
``MultiLayerDraftWorker`` (#1053 P1-4) creates one instance per layer. The
MTP block is a SWA-attention + dense-MLP decoder (not MoE), with the same
fused-QKV per-shard FP8 layout as the V2.5-Pro target.
"""

import logging
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.mimo_v2_flash import MiMoV2Attention, MiMoV2MLP
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


class MiMoV2NextNDecoderLayer(nnx.Module):
    """Single MTP decoder block: SWA attention + dense MLP (never MoE)."""

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if isinstance(rope_scaling, dict) and rope_scaling.get("rope_type") == "default":
            rope_scaling = None
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)

        self.self_attn = MiMoV2Attention(
            hidden_size=config.hidden_size,
            num_heads=config.swa_num_attention_heads,
            num_kv_heads=config.swa_num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=getattr(config, "swa_rope_theta", rope_theta),
            rope_scaling=rope_scaling,
            head_dim=config.swa_head_dim,
            v_head_dim=getattr(config, "swa_v_head_dim", None),
            sliding_window_size=getattr(config, "sliding_window_size", None),
            attention_sink_bias=getattr(config, "add_swa_attention_sink_bias", False),
            partial_rotary_factor=getattr(config, "partial_rotary_factor", 1.0),
            attention_value_scale=getattr(config, "attention_value_scale", None),
            layer_id=layer_id,
            dtype=dtype,
            mesh=mesh,
        )
        self.mlp = MiMoV2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            layer_id=layer_id,
            dtype=dtype,
            mesh=mesh,
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size, epsilon=config.layernorm_epsilon, param_dtype=dtype
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, epsilon=config.layernorm_epsilon, param_dtype=dtype
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, None]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states = hidden_states + residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states, kv_fused = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )

        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual, kv_fused, None


class MiMoV2ModelNextN(nnx.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            kernel_axes=("tensor", None),
            param_dtype=dtype,
            mesh=mesh,
        )
        self.enorm = RMSNorm(
            config.hidden_size, epsilon=config.layernorm_epsilon, param_dtype=dtype
        )
        self.hnorm = RMSNorm(
            config.hidden_size, epsilon=config.layernorm_epsilon, param_dtype=dtype
        )
        self.eh_proj = LinearBase(
            input_size=2 * config.hidden_size,
            output_size=config.hidden_size,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.mtp_block = MiMoV2NextNDecoderLayer(config, mesh=mesh, layer_id=0, dtype=dtype)
        self.final_layernorm = RMSNorm(
            config.hidden_size, epsilon=config.layernorm_epsilon, param_dtype=dtype
        )

    def __call__(
        self, forward_batch: ForwardBatch, token_to_kv_pool: KVCache
    ) -> tuple[jax.Array, list[jax.Array]]:
        embed = self.embed_tokens(forward_batch.input_ids)
        hidden_in = forward_batch.spec_info.hidden_states
        emb_sh = jax.typeof(embed).sharding
        if isinstance(emb_sh, jax.sharding.NamedSharding):
            hidden_in = jax.sharding.reshard(hidden_in, emb_sh)
        hidden_states, _ = self.eh_proj(
            jnp.concatenate((self.enorm(embed), self.hnorm(hidden_in)), axis=-1)
        )
        hidden_states, residual, kv_fused, _ = self.mtp_block(
            forward_batch.positions, hidden_states, forward_batch, token_to_kv_pool, None
        )
        hidden_states = self.final_layernorm(hidden_states + residual)
        return hidden_states, [kv_fused]


class MiMoV2MTPForCausalLM(nnx.Module):

    load_lm_head_from_target = True

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        self.mtp_layer_idx = getattr(config, "mtp_layer_idx", 0)
        self.model = MiMoV2ModelNextN(config, mesh=mesh, dtype=dtype)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=self.mesh)
        self._fused_qkv_buffers: dict[int, dict] = {}
        self._uses_fused_mtp_qkv = False
        self.hot_token_ids = None

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, layers_kv_fused = self.model(forward_batch, memory_pools.token_to_kv_pool)
        output = self.logits_processor(
            hidden_states, self.lm_head, logits_metadata, aux_hidden_states=None
        )
        return output, layers_kv_fused, True, None

    def load_weights(self, model_config: ModelConfig):
        self.loader = WeightLoader(
            model=self, model_config=model_config, mesh=self.mesh, dtype=self.dtype
        )
        mappings = self._create_weight_mappings()
        self.loader.load_weights_from_safetensors(mappings)

        if self.loader.is_static_quant:
            attn = self.model.mtp_block.self_attn
            head_dim, v_head_dim = attn.head_dim, attn.v_head_dim
            if self._uses_fused_mtp_qkv:
                # dequant_fused_qkv reads full-attn config fields; the MTP block uses
                # SWA dims, so derive the split config from the actual layer.
                mtp_qkv_config = SimpleNamespace(
                    head_dim=head_dim,
                    v_head_dim=v_head_dim,
                    num_attention_heads=attn.q_head_num,
                    num_key_value_heads=attn.k_head_num,
                )
                self.loader.dequant_fused_qkv(
                    self._fused_qkv_buffers, [self.model.mtp_block], mtp_qkv_config
                )
            else:
                self.loader.dequant_fp8_layers(
                    [self.model.mtp_block],
                    specs=[
                        ("self_attn.q_proj", head_dim),
                        ("self_attn.k_proj", head_dim),
                        ("self_attn.v_proj", v_head_dim),
                    ],
                )
            self.loader.dequant_fp8_layers(
                [self.model.mtp_block],
                specs=[
                    ("mlp.gate_proj", None),
                    ("mlp.up_proj", None),
                    ("mlp.down_proj", None),
                ],
            )
            self.loader.replicate_kv_heads(
                [self.model.mtp_block],
                specs=[("self_attn.k_proj", head_dim), ("self_attn.v_proj", v_head_dim)],
                target_kv_heads_fn=lambda attn: attn.k_head_num,
            )
        logger.info(
            "MiMoV2 MTP layer %d weights loaded (fused-qkv FP8=%s)",
            self.mtp_layer_idx,
            self.loader.is_static_quant,
        )

    def _create_weight_mappings(self) -> dict[str, WeightMapping]:
        idx = self.mtp_layer_idx
        prefix = f"model.mtp.layers.{idx}"
        block = "model.mtp_block"
        is_fp8 = self.loader.is_static_quant

        mappings: dict[str, WeightMapping] = {
            f"{prefix}.enorm.weight": WeightMapping(
                target_path="model.enorm.scale", sharding=(None,), transpose=False
            ),
            f"{prefix}.hnorm.weight": WeightMapping(
                target_path="model.hnorm.scale", sharding=(None,), transpose=False
            ),
            f"{prefix}.eh_proj.weight": WeightMapping(
                target_path="model.eh_proj.weight", sharding=(None, None), transpose=True
            ),
            f"{prefix}.final_layernorm.weight": WeightMapping(
                target_path="model.final_layernorm.scale", sharding=(None,), transpose=False
            ),
            f"{prefix}.input_layernorm.weight": WeightMapping(
                target_path=f"{block}.input_layernorm.scale", sharding=(None,), transpose=False
            ),
            f"{prefix}.pre_mlp_layernorm.weight": WeightMapping(
                target_path=f"{block}.post_attention_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.self_attn.o_proj.weight": WeightMapping(
                target_path=f"{block}.self_attn.o_proj.weight",
                sharding=("tensor", None),
                transpose=True,
                head_dim_padding=True,
            ),
            f"{prefix}.self_attn.attention_sink_bias": WeightMapping(
                target_path=f"{block}.self_attn.attention_sink_bias",
                sharding=("tensor",),
                transpose=False,
            ),
        }

        qkv_key = f"{prefix}.self_attn.qkv_proj"
        has_fused_qkv = (
            self.loader.has_weight_on_disk(f"{qkv_key}.weight")
            if hasattr(self.loader, "has_weight_on_disk")
            else True
        )
        self._uses_fused_mtp_qkv = has_fused_qkv

        if has_fused_qkv and is_fp8 and not self.loader.is_quant_ignored(qkv_key):
            mappings[f"{qkv_key}.weight"] = WeightMapping(
                target_path="__FUSED_QKV_WEIGHT__0", sharding=(None, None), transpose=False
            )
            mappings[f"{qkv_key}.weight_scale_inv"] = WeightMapping(
                target_path="__FUSED_QKV_SCALE__0", sharding=(None, None), transpose=False
            )
        elif has_fused_qkv:
            mappings[f"{qkv_key}.weight"] = WeightMapping(
                target_path=[
                    f"{block}.self_attn.q_proj.weight",
                    f"{block}.self_attn.k_proj.weight",
                    f"{block}.self_attn.v_proj.weight",
                ],
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=False,
                kv_head_padding=True,
            )
        else:
            for proj, head_dim_padding in [
                ("q_proj", True),
                ("k_proj", True),
                ("v_proj", False),
            ]:
                hf_key = f"{prefix}.self_attn.{proj}"
                ignored = self.loader.is_quant_ignored(hf_key)
                weight_suffix = "weight" if (not is_fp8 or ignored) else "weight_q"
                mappings[f"{hf_key}.weight"] = WeightMapping(
                    target_path=f"{block}.self_attn.{proj}.{weight_suffix}",
                    sharding=(None, "tensor"),
                    transpose=True,
                    head_dim_padding=head_dim_padding,
                    kv_head_padding=not is_fp8,
                )
                if is_fp8 and not ignored:
                    mappings[f"{hf_key}.weight_scale_inv"] = WeightMapping(
                        target_path=f"{block}.self_attn.{proj}.weight_scale",
                        sharding=(None, None),
                        transpose=False,
                    )

        for proj, sharding in [
            ("gate_proj", (None, "tensor")),
            ("up_proj", (None, "tensor")),
            ("down_proj", ("tensor", None)),
        ]:
            hf_key = f"{prefix}.mlp.{proj}"
            suffix = "weight_q" if is_fp8 else "weight"
            mappings[f"{hf_key}.weight"] = WeightMapping(
                target_path=f"{block}.mlp.{proj}.{suffix}",
                sharding=sharding,
                transpose=True,
            )
            if is_fp8:
                mappings[f"{hf_key}.weight_scale_inv"] = WeightMapping(
                    target_path=f"{block}.mlp.{proj}.weight_scale",
                    sharding=(None, None),
                    transpose=False,
                )

        return mappings

    def get_embed_and_head(self):
        return self.model.embed_tokens.embedding.value, self.lm_head.embedding.value

    def set_embed_and_head(self, embed: jax.Array, head: jax.Array) -> None:
        self.model.embed_tokens.embedding.value = embed
        self.lm_head.embedding.value = head

    def set_embed(self, embed: jax.Array) -> None:
        self.model.embed_tokens.embedding.value = embed


EntryClass = MiMoV2MTPForCausalLM
