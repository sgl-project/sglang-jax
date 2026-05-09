"""MiMo-V2-Flash / V2.5 / V2.5-Pro MTP (NextN) draft model.

The V2 family ships a 3-layer MTP head under ``model.mtp.layers.{0,1,2}``.
Each layer is a single SWA-attention block (window=128, K head_dim=192,
V head_dim=128, attention sink) followed by a *dense* FFN (no MoE). This
module loads layer 0 only and exposes it as a standard NEXTN draft model
for ``eagle_worker``; multi-layer EAGLE is left as a follow-up.

Weight key convention (from HF safetensors)::

    model.mtp.layers.0.enorm.weight                # token-side norm
    model.mtp.layers.0.hnorm.weight                # hidden-side norm
    model.mtp.layers.0.eh_proj.weight              # 2*h -> h
    model.mtp.layers.0.input_layernorm.weight
    model.mtp.layers.0.self_attn.{q,k,v,o}_proj.weight[_scale_inv]
    model.mtp.layers.0.self_attn.attention_sink_bias
    model.mtp.layers.0.pre_mlp_layernorm.weight
    model.mtp.layers.0.mlp.{gate,up,down}_proj.weight[_scale_inv]
    model.mtp.layers.0.final_layernorm.weight
"""

from __future__ import annotations

import logging

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


class MiMoV2FlashMTPLayer(nnx.Module):
    """One MTP block: SWA attention + dense MLP, with the V2 enorm/hnorm/eh_proj prelude."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
        mesh: jax.sharding.Mesh | None = None,
    ):
        self.layer_id = layer_id
        eps = getattr(config, "rms_norm_eps", getattr(config, "layernorm_epsilon", 1e-6))

        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            kernel_axes=("tensor", None),
            param_dtype=dtype,
            mesh=mesh,
        )
        self.enorm = RMSNorm(config.hidden_size, epsilon=eps)
        self.hnorm = RMSNorm(config.hidden_size, epsilon=eps)
        self.eh_proj = LinearBase(
            input_size=config.hidden_size * 2,
            output_size=config.hidden_size,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
        )

        self.input_layernorm = RMSNorm(config.hidden_size, epsilon=eps)
        self.self_attn = MiMoV2Attention(
            hidden_size=config.hidden_size,
            num_heads=config.swa_num_attention_heads,
            num_kv_heads=config.swa_num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            mesh=mesh,
            rope_theta=getattr(config, "swa_rope_theta", config.rope_theta),
            rope_scaling=getattr(config, "rope_scaling", None),
            head_dim=config.swa_head_dim,
            v_head_dim=config.swa_v_head_dim,
            sliding_window_size=getattr(config, "sliding_window_size", config.sliding_window),
            attention_sink_bias=getattr(config, "add_swa_attention_sink_bias", False),
            partial_rotary_factor=getattr(config, "partial_rotary_factor", 1.0),
            attention_value_scale=getattr(config, "attention_value_scale", None),
            layer_id=layer_id,
            dtype=dtype,
        )
        self.pre_mlp_layernorm = RMSNorm(config.hidden_size, epsilon=eps)
        self.mlp = MiMoV2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            mesh=mesh,
            layer_id=layer_id,
            dtype=dtype,
        )
        self.final_layernorm = RMSNorm(config.hidden_size, epsilon=eps)

    def __call__(self, forward_batch: ForwardBatch, token_to_kv_pool: KVCache):
        token_embeds = self.embed_tokens(forward_batch.input_ids)
        token_mask = (forward_batch.positions != 0).astype(token_embeds.dtype)[:, None]
        token_embeds = token_embeds * token_mask

        hidden_states, _ = self.eh_proj(
            jnp.concatenate(
                (
                    self.enorm(token_embeds),
                    self.hnorm(forward_batch.spec_info.hidden_states),
                ),
                axis=-1,
            )
        )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, kv_fused = self.self_attn(
            forward_batch.positions, hidden_states, forward_batch, token_to_kv_pool
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.pre_mlp_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states, [kv_fused]


class MiMoV2FlashMTPForCausalLM(nnx.Module):
    """NEXTN draft model wrapper for MiMo-V2-Flash / V2.5 / V2.5-Pro."""

    load_lm_head_from_target = True

    def __init__(
        self,
        config: PretrainedConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        mesh: jax.sharding.Mesh | None = None,
    ):
        self.config = config
        self.dtype = dtype
        self.mesh = mesh
        self.model = MiMoV2FlashMTPLayer(config, dtype=dtype, mesh=mesh)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
        )
        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=mesh)

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        kv_pool = memory_pools.token_to_kv_pool
        hidden_states, layers_kv_fused = self.model(forward_batch, kv_pool)
        if not getattr(self.config, "tie_word_embeddings", False):
            output = self.logits_processor(
                hidden_states, self.lm_head, logits_metadata, aux_hidden_states=None
            )
        else:
            output = self.logits_processor(
                hidden_states,
                self.model.embed_tokens,
                logits_metadata,
                aux_hidden_states=None,
            )
        return output, layers_kv_fused, True, []

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self, model_config=model_config, mesh=self.mesh, dtype=self.dtype
        )
        loader.load_weights_from_safetensors(self._create_weight_mappings())
        logger.info("MiMo-V2 MTP draft weights loaded successfully!")

    def _create_weight_mappings(self) -> dict:
        prefix = "model.mtp.layers.0"
        m: dict[str, WeightMapping] = {
            "model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "lm_head.weight": WeightMapping(
                target_path="lm_head.embedding",
                sharding=(None, None),
                transpose=False,
            ),
            f"{prefix}.enorm.weight": WeightMapping(
                target_path="model.enorm.scale", sharding=(None,), transpose=False
            ),
            f"{prefix}.hnorm.weight": WeightMapping(
                target_path="model.hnorm.scale", sharding=(None,), transpose=False
            ),
            f"{prefix}.eh_proj.weight": WeightMapping(
                target_path="model.eh_proj.weight",
                sharding=(None, None),
                transpose=True,
            ),
            f"{prefix}.input_layernorm.weight": WeightMapping(
                target_path="model.input_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.pre_mlp_layernorm.weight": WeightMapping(
                target_path="model.pre_mlp_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.final_layernorm.weight": WeightMapping(
                target_path="model.final_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.self_attn.attention_sink_bias": WeightMapping(
                target_path="model.self_attn.attention_sink_bias",
                sharding=("tensor",),
                transpose=False,
            ),
        }
        is_fp8 = getattr(self.config, "quantization_config", None) is not None
        # q/k/v_proj + mlp are FP8 (have weight_scale_inv) → QuantizedLinear (weight_q/weight_scale).
        # o_proj + eh_proj are bf16 → ignored_layers → plain LinearBase (weight).
        attn_projs = (
            ("q_proj", (None, "tensor"), False, True),
            ("k_proj", (None, "tensor"), True, True),
            ("v_proj", (None, "tensor"), True, True),
            ("o_proj", ("tensor", None), False, False),
        )
        for name, axes, kv_pad, fp8 in attn_projs:
            wsuf = "weight_q" if (is_fp8 and fp8) else "weight"
            m[f"{prefix}.self_attn.{name}.weight"] = WeightMapping(
                target_path=f"model.self_attn.{name}.{wsuf}",
                sharding=axes,
                transpose=True,
                kv_head_padding=kv_pad,
            )
            if is_fp8 and fp8:
                m[f"{prefix}.self_attn.{name}.weight_scale_inv"] = WeightMapping(
                    target_path=f"model.self_attn.{name}.weight_scale",
                    sharding=(None, None),
                    transpose=False,
                )
        for name, axes in (
            ("gate_proj", (None, "tensor")),
            ("up_proj", (None, "tensor")),
            ("down_proj", ("tensor", None)),
        ):
            wsuf = "weight_q" if is_fp8 else "weight"
            m[f"{prefix}.mlp.{name}.weight"] = WeightMapping(
                target_path=f"model.mlp.{name}.{wsuf}",
                sharding=axes,
                transpose=True,
            )
            if is_fp8:
                m[f"{prefix}.mlp.{name}.weight_scale_inv"] = WeightMapping(
                    target_path=f"model.mlp.{name}.weight_scale",
                    sharding=(None, None),
                    transpose=False,
                )
        return m

    def get_embed_and_head(self):
        return (
            self.model.embed_tokens.embedding.value,
            self.lm_head.embedding.value,
        )

    def set_embed_and_head(
        self,
        embed_weight: jax.Array | None = None,
        head_weight: jax.Array | None = None,
    ) -> None:
        # embed/lm_head are loaded from the same checkpoint as the target;
        # avoid re-assigning here to keep the nnx.Param sharding intact.
        pass


EntryClass = MiMoV2FlashMTPForCausalLM
