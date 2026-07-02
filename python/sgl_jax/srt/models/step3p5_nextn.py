"""Step-3.5-Flash multi-token-prediction (NextN/MTP) draft model.

One instance loads exactly one of the 3 MTP layers appended after the main
decoder (checkpoint keys ``model.layers.{45,46,47}.*``, selected via
``config.mtp_layer_idx``); the ``MultiLayerDraftWorker`` creates one instance
per layer. Every MTP layer is a sliding-window-attention + dense-MLP decoder
(never MoE), with the same per-head QK-norm, per-layer partial RoPE and
head-wise attention gate as the Step-3.5 target model — so the MTP block reuses
``Step3p5DecoderLayer`` unchanged, driven by a draft-local config shim whose
index-0 per-layer entries mirror the real MTP layer.

Differs from ``mimo_v2_nextn.py`` in three ways dictated by the real
checkpoint index:
  * norms are ``GemmaRMSNorm`` (param ``.weight``, not ``RMSNorm``'s ``.scale``);
  * the LM head is loaded from the checkpoint's own
    ``...transformer.shared_head.output.weight`` (``load_lm_head_from_target``
    is False — only the token embedding is injected from the target);
  * weights are BF16 (no FP8 scales), q/k/v are stored separately (not fused).
"""

from __future__ import annotations

import copy
import logging

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
from sgl_jax.srt.layers.layernorm import GemmaRMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.step3p5 import Step3p5DecoderLayer
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


def _mtp_block_config(config: PretrainedConfig, mtp_abs_layer: int) -> PretrainedConfig:
    """Draft-local config whose index-0 per-layer entries equal the real MTP layer.

    The MTP block reuses ``Step3p5DecoderLayer``, which derives its layer type,
    RoPE theta and partial-rotary factor from the per-layer lists indexed by
    ``layer_id``. The block must run with ``layer_id=0`` so it indexes slot 0 of
    the draft's single-layer KV pool, so we build a shim whose slot-0 entries are
    the real MTP layer's values (layers 45/46/47 — all sliding, theta 1e4,
    partial 1.0, no SwiGLU clamp, not MoE). ``attention_other_setting``,
    ``sliding_window``, ``rms_norm_eps`` etc. are scalar/dict fields kept as-is.
    """
    shim = copy.deepcopy(config)
    shim.num_hidden_layers = 1

    lt = config.layer_types
    shim.layer_types = [lt[mtp_abs_layer]] if lt else ["sliding_attention"]

    if isinstance(config.rope_theta, list):
        shim.rope_theta = [float(config.rope_theta[mtp_abs_layer])]

    prf = config.partial_rotary_factors
    shim.partial_rotary_factors = [float(prf[mtp_abs_layer])] if prf else None

    sl = config.swiglu_limits
    shim.swiglu_limits = [sl[mtp_abs_layer] if sl else 0.0]
    sls = config.swiglu_limits_shared
    shim.swiglu_limits_shared = [sls[mtp_abs_layer] if sls else 0.0]

    # Empty MoE enum → the single block is a dense MLP (matches MTP layers).
    shim.moe_layers_enum = []
    return shim


class Step3p5ModelNextN(nnx.Module):
    """embed + enorm/hnorm + eh_proj + one Step3p5DecoderLayer + shared_head norm."""

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        attn_impl: str = "flash",
    ) -> None:
        mtp_layer_idx = int(getattr(config, "mtp_layer_idx", 0))
        # Absolute checkpoint layer index (45 + idx). num_hidden_layers may already
        # be overridden to 1 for the draft, so recover the base count from the
        # full-length per-layer lists rather than from num_hidden_layers.
        base_layers = (
            len(config.layer_types) - int(getattr(config, "num_nextn_predict_layers", 0))
            if config.layer_types
            else int(getattr(config, "num_hidden_layers", 45))
        )
        self.mtp_abs_layer = base_layers + mtp_layer_idx

        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )
        self.enorm = GemmaRMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, add_unit_offset=True
        )
        self.hnorm = GemmaRMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, add_unit_offset=True
        )
        self.eh_proj = LinearBase(
            input_size=2 * config.hidden_size,
            output_size=config.hidden_size,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="eh_proj",
        )
        # Reuse the verified Step3p5DecoderLayer at layer_id=0 (KV slot 0) with a
        # shim config that makes slot 0 == the real MTP (sliding) layer.
        block_config = _mtp_block_config(config, self.mtp_abs_layer)
        self.mtp_block = Step3p5DecoderLayer(
            config=block_config,
            layer_id=0,
            mesh=mesh,
            dtype=dtype,
            attn_impl=attn_impl,
        )
        # `shared_head.norm` in the checkpoint — final norm before the LM head.
        self.shared_head_norm = GemmaRMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, add_unit_offset=True
        )

    def __call__(
        self, forward_batch: ForwardBatch, token_to_kv_pool: KVCache
    ) -> tuple[jax.Array, list[jax.Array], list[jax.Array]]:
        embed = self.embed_tokens(forward_batch.input_ids)
        hidden_in = forward_batch.spec_info.hidden_states
        emb_sh = jax.typeof(embed).sharding
        if isinstance(emb_sh, NamedSharding):
            hidden_in = jax.sharding.reshard(hidden_in, emb_sh)
        hidden_states, _ = self.eh_proj(
            jnp.concatenate((self.enorm(embed), self.hnorm(hidden_in)), axis=-1)
        )
        hidden_states, residual, kv_fused, _cb_flags, _topk_ids = self.mtp_block(
            forward_batch.positions, hidden_states, forward_batch, token_to_kv_pool, None
        )
        # Chain: the pre-norm hidden (hidden + residual, before shared_head.norm)
        # is what the multi-layer draft worker feeds into the next MTP layer.
        # Mirror llama_eagle3: return it as an aux hidden state so the logits
        # processor captures it as ``logits_output.hidden_states``, while the LM
        # head logits use the post-norm hidden.
        pre_norm = hidden_states + residual
        hidden_to_logits = self.shared_head_norm(pre_norm)
        return hidden_to_logits, [pre_norm], [kv_fused]


class Step3p5MTPForCausalLM(nnx.Module):
    """Step-3.5-Flash MTP draft model (own LM head; BF16; separate q/k/v)."""

    # MTP carries its own `shared_head.output` LM head — only the token
    # embedding is injected from the target (set_embed).
    load_lm_head_from_target = False

    # Step-3.5-Flash MTP is chain-style (upstream sglang enables chain only for
    # Step3p5MTP): each MTP layer consumes the previous layer's pre-norm hidden,
    # not the target hidden. `capture_aux_hidden_states` makes the logits
    # processor store that pre-norm hidden as `logits_output.hidden_states`
    # (sglang-jax's EAGLE3 mechanism); `chain_mtp_hidden_states` tells the
    # multi-layer draft worker to feed it forward between layers.
    capture_aux_hidden_states = True
    chain_mtp_hidden_states = True

    @classmethod
    def patch_model_config(cls, mc: ModelConfig) -> None:
        # Pin head_dim=128 for KV pool / attention backend sizing (mirror base).
        mc.head_dim = 128

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
        attn_impl: str = "flash",
    ) -> None:
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        self.mtp_layer_idx = int(getattr(config, "mtp_layer_idx", 0))
        self.model = Step3p5ModelNextN(config, mesh=mesh, dtype=dtype, attn_impl=attn_impl)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=mesh)
        logger.info(
            "Step3p5MTPForCausalLM dtype=%s attn_impl=%s mtp_layer_idx=%d (abs layer %d)",
            dtype,
            attn_impl,
            self.mtp_layer_idx,
            self.model.mtp_abs_layer,
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        hidden_to_logits, aux_hidden_states, layers_kv_fused = self.model(
            forward_batch, memory_pools.token_to_kv_pool
        )
        # Logits use the post-norm hidden; the pre-norm aux hidden is captured as
        # logits_output.hidden_states for chain propagation (see class docstring).
        output = self.logits_processor(
            hidden_to_logits,
            self.lm_head,
            logits_metadata,
            aux_hidden_states=aux_hidden_states if self.capture_aux_hidden_states else None,
        )
        return output, layers_kv_fused, True, None

    def load_weights(self, model_config: ModelConfig) -> None:
        self.loader = WeightLoader(
            model=self, model_config=model_config, mesh=self.mesh, dtype=self.dtype
        )
        mappings = self._create_weight_mappings()
        self.loader.load_weights_from_safetensors(mappings)
        logger.info(
            "Step3p5 MTP layer %d weights loaded (abs layer %d)",
            self.mtp_layer_idx,
            self.model.mtp_abs_layer,
        )

    def _create_weight_mappings(self) -> dict[str, WeightMapping]:
        prefix = f"model.layers.{self.model.mtp_abs_layer}"
        block = "model.mtp_block"

        mappings: dict[str, WeightMapping] = {
            # MTP-specific: embed+hidden norms and their fusion projection.
            f"{prefix}.enorm.weight": WeightMapping(
                target_path="model.enorm.weight", sharding=(None,), transpose=False
            ),
            f"{prefix}.hnorm.weight": WeightMapping(
                target_path="model.hnorm.weight", sharding=(None,), transpose=False
            ),
            f"{prefix}.eh_proj.weight": WeightMapping(
                target_path="model.eh_proj.weight", sharding=(None, None), transpose=True
            ),
            # shared_head: final norm + own LM head (nested under `.transformer.`).
            f"{prefix}.transformer.shared_head.norm.weight": WeightMapping(
                target_path="model.shared_head_norm.weight", sharding=(None,), transpose=False
            ),
            f"{prefix}.transformer.shared_head.output.weight": WeightMapping(
                target_path="lm_head.embedding", sharding=("tensor", None), transpose=False
            ),
            # Decoder-block norms.
            f"{prefix}.input_layernorm.weight": WeightMapping(
                target_path=f"{block}.input_layernorm.weight", sharding=(None,), transpose=False
            ),
            f"{prefix}.post_attention_layernorm.weight": WeightMapping(
                target_path=f"{block}.post_attention_layernorm.weight",
                sharding=(None,),
                transpose=False,
            ),
        }

        # Attention projections + per-head QK-norm (mirror base Step3p5 layer map).
        ap = f"{prefix}.self_attn"
        tp = f"{block}.self_attn"
        mappings[f"{ap}.q_proj.weight"] = WeightMapping(
            target_path=f"{tp}.q_proj.weight",
            sharding=(None, "tensor"),
            transpose=True,
            kv_head_padding=False,
        )
        mappings[f"{ap}.k_proj.weight"] = WeightMapping(
            target_path=f"{tp}.k_proj.weight",
            sharding=(None, "tensor"),
            transpose=True,
            kv_head_padding=True,
        )
        mappings[f"{ap}.v_proj.weight"] = WeightMapping(
            target_path=f"{tp}.v_proj.weight",
            sharding=(None, "tensor"),
            transpose=True,
            kv_head_padding=True,
        )
        mappings[f"{ap}.o_proj.weight"] = WeightMapping(
            target_path=f"{tp}.o_proj.weight",
            sharding=("tensor", None),
            transpose=True,
            kv_head_padding=False,
        )
        mappings[f"{ap}.g_proj.weight"] = WeightMapping(
            target_path=f"{tp}.g_proj.weight",
            sharding=(None, "tensor"),
            transpose=True,
            kv_head_padding=False,
        )
        mappings[f"{ap}.q_norm.weight"] = WeightMapping(
            target_path=f"{tp}.q_norm.weight", sharding=(None,), transpose=False
        )
        mappings[f"{ap}.k_norm.weight"] = WeightMapping(
            target_path=f"{tp}.k_norm.weight", sharding=(None,), transpose=False
        )

        # Dense MLP.
        for proj, sharding in [
            ("gate_proj", (None, "tensor")),
            ("up_proj", (None, "tensor")),
            ("down_proj", ("tensor", None)),
        ]:
            mappings[f"{prefix}.mlp.{proj}.weight"] = WeightMapping(
                target_path=f"{block}.mlp.{proj}.weight",
                sharding=sharding,
                transpose=True,
            )

        return mappings

    def get_embed_and_head(self):
        return self.model.embed_tokens.embedding.value, self.lm_head.embedding.value

    def set_embed_and_head(self, embed: jax.Array, head: jax.Array) -> None:
        self.model.embed_tokens.embedding.value = embed
        self.lm_head.embedding.value = head

    def set_embed(self, embed: jax.Array) -> None:
        self.model.embed_tokens.embedding.value = embed


EntryClass = [Step3p5MTPForCausalLM]
