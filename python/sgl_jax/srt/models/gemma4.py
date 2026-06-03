"""Gemma4 text decoder for sglang-jax.

31B dense variant only (no PLE / no Shared-KV / no MoE). Multimodal vision
encoder lives separately under ``srt/multimodal/models/gemma4/``; this file
loads ``model.language_model.*`` weights from the
``Gemma4ForConditionalGeneration`` checkpoint and ignores the rest.

Key deltas vs Gemma2:
- Heterogeneous attention: full layers use ``global_head_dim`` /
  ``num_global_key_value_heads`` and proportional RoPE; sliding layers use the
  base ``head_dim`` / ``num_key_value_heads`` with default RoPE.
- ``attention_k_eq_v``: full layers ship no ``v_proj`` checkpoint; we alias
  ``k_proj`` weights into ``v_proj`` at load time.
- Q/K RMSNorm + scale-free V RMSNorm; attention scaling is 1.0.
- Standard RMSNorm (``x * w``) everywhere, not Gemma2's ``x * (1 + w)``.
- Per-layer learned scalar multiplied onto the residual sum.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, get_rope
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)

_SLIDING = "sliding_attention"
_FULL = "full_attention"
_HF_PREFIX = "model.language_model"


def _layer_type(config: PretrainedConfig, layer_id: int) -> str:
    return config.layer_types[layer_id]


class Gemma4MLP(nnx.Module):
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
        )
        self.up_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
        )
        self.down_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            kernel_axes=("tensor", None),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        a1, _ = self.gate_proj(hidden_states)
        a2, _ = self.up_proj(hidden_states)
        intermediate = a2 * jax.nn.gelu(a1, approximate=True)
        output, _ = self.down_proj(intermediate)
        return output


class Gemma4Attention(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        layer_type = _layer_type(config, layer_id)
        is_sliding = layer_type == _SLIDING

        self.layer_id = layer_id
        self.num_heads = config.num_attention_heads
        self.head_dim = (
            getattr(config, "swa_head_dim", config.head_dim) if is_sliding else config.head_dim
        )
        self.num_kv_heads = (
            getattr(config, "swa_num_key_value_heads", config.num_key_value_heads)
            if is_sliding
            else config.num_key_value_heads
        )
        hidden_size = config.hidden_size

        self.q_proj = LinearBase(
            input_size=hidden_size,
            output_size=self.num_heads * self.head_dim,
            kernel_axes=(None, "tensor"),
            use_bias=config.attention_bias,
            params_dtype=dtype,
            mesh=mesh,
        )
        self.k_proj = LinearBase(
            input_size=hidden_size,
            output_size=self.num_kv_heads * self.head_dim,
            kernel_axes=(None, "tensor"),
            use_bias=config.attention_bias,
            params_dtype=dtype,
            mesh=mesh,
        )
        self.k_eq_v = not is_sliding and getattr(config, "attention_k_eq_v", False)
        if not self.k_eq_v:
            self.v_proj = LinearBase(
                input_size=hidden_size,
                output_size=self.num_kv_heads * self.head_dim,
                kernel_axes=(None, "tensor"),
                use_bias=config.attention_bias,
                params_dtype=dtype,
                mesh=mesh,
            )
        self.o_proj = LinearBase(
            input_size=self.num_heads * self.head_dim,
            output_size=hidden_size,
            kernel_axes=("tensor", None),
            use_bias=config.attention_bias,
            params_dtype=dtype,
            mesh=mesh,
        )

        self.q_norm = RMSNorm(self.head_dim, epsilon=config.rms_norm_eps, dtype=dtype)
        self.k_norm = RMSNorm(self.head_dim, epsilon=config.rms_norm_eps, dtype=dtype)
        self.v_norm = RMSNorm(
            self.head_dim, epsilon=config.rms_norm_eps, use_scale=False, dtype=dtype
        )

        rope_params = dict(config.rope_parameters.get(layer_type, {"rope_type": "default"}))
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=rope_params.get("rope_theta", 10000.0),
            rope_scaling=rope_params,
            is_neox_style=True,
            dtype=dtype,
        )

        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=1.0,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=config.sliding_window if is_sliding else 0,
            logit_cap=0.0,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, jax.Array]:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v = k if self.k_eq_v else self.v_proj(hidden_states)[0]

        q = q.reshape(-1, self.num_heads, self.head_dim)
        k = k.reshape(-1, self.num_kv_heads, self.head_dim)
        v = v.reshape(-1, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        q, k = self.rotary_emb(forward_batch.positions, q, k)

        attn_output, kv_fused = self.attn(
            q, k, v, forward_batch=forward_batch, token_to_kv_pool=token_to_kv_pool
        )
        output, _ = self.o_proj(attn_output)
        return output, kv_fused


class Gemma4DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        eps = config.rms_norm_eps

        self.self_attn = Gemma4Attention(config, layer_id, mesh=mesh, dtype=dtype)
        self.mlp = Gemma4MLP(config.hidden_size, config.intermediate_size, mesh=mesh, dtype=dtype)

        self.input_layernorm = RMSNorm(config.hidden_size, epsilon=eps, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, epsilon=eps, dtype=dtype)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, epsilon=eps, dtype=dtype)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, epsilon=eps, dtype=dtype)

        self.layer_scalar = nnx.Param(jnp.ones((1,), dtype=dtype))

    def __call__(
        self,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, jax.Array]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, kv_fused = self.self_attn(hidden_states, forward_batch, token_to_kv_pool)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = (residual + hidden_states) * jnp.asarray(
            self.layer_scalar, hidden_states.dtype
        )
        return hidden_states, kv_fused


class Gemma4TextModel(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.hidden_size = config.hidden_size
        self.embed_tokens = Embed(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )
        self.layers = nnx.data(
            [
                Gemma4DecoderLayer(config, layer_id=i, dtype=dtype, mesh=mesh)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps, dtype=dtype)

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, list[jax.Array]]:
        if (
            forward_batch.input_embedding is not None
            and forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
        ):
            hidden_states = forward_batch.input_embedding
        else:
            hidden_states = self.embed_tokens(forward_batch.input_ids)
            hidden_states *= jnp.array([self.hidden_size**0.5], dtype=hidden_states.dtype)

        layers_kv_fused: list[jax.Array] = []
        for layer in self.layers:
            hidden_states, kv_fused = layer(hidden_states, forward_batch, token_to_kv_pool)
            layers_kv_fused.append(kv_fused)

        hidden_states = self.norm(hidden_states)
        return hidden_states, layers_kv_fused


class Gemma4ForConditionalGeneration(nnx.Module):
    """Phase-1 text-only entry. Vision tower added in Phase 2."""

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        text_config = getattr(config, "text_config", config)
        self.text_config = text_config
        self.language_model = Gemma4TextModel(text_config, dtype=dtype, mesh=mesh)
        self.logits_processor = LogitsProcessor(
            text_config.vocab_size,
            soft_cap=getattr(text_config, "final_logit_softcapping", None),
            mesh=mesh,
        )

    # Expose ``model.layers`` so model_runner's hybrid-SWA layer scan works.
    @property
    def model(self):
        return self.language_model

    @classmethod
    def patch_model_config(cls, model_config: ModelConfig) -> None:
        """Remap HF Gemma4 attention dims to sglang's swa_* / base convention.

        HF: base attrs describe sliding layers, ``global_*`` describe full layers.
        sglang: base attrs describe full layers, ``swa_*`` describe sliding layers.

        Mutates ``hf_text_config`` in place (matching the existing sglang pattern
        in ``model_config.get_total_num_kv_heads_with_replication``). The
        ``_gemma4_patched`` guard makes this idempotent across the multiple
        ``ModelConfig`` constructions that share one ``hf_config`` object.
        """
        text_cfg = model_config.hf_text_config
        global_head_dim = getattr(text_cfg, "global_head_dim", None)
        global_kv = getattr(text_cfg, "num_global_key_value_heads", None)
        if global_head_dim is None and global_kv is None:
            return
        if getattr(text_cfg, "_gemma4_patched", False):
            # ModelConfig is constructed multiple times per process sharing the
            # same hf_config; the remap below is not idempotent.
            model_config.head_dim = text_cfg.head_dim
            return
        text_cfg._gemma4_patched = True

        text_cfg.swa_head_dim = text_cfg.head_dim
        text_cfg.swa_num_key_value_heads = text_cfg.num_key_value_heads
        if global_head_dim is not None:
            text_cfg.head_dim = global_head_dim
        if global_kv is not None:
            text_cfg.num_key_value_heads = global_kv

        # Mirror onto top-level hf_config for callers that read it directly
        # (e.g. model_runner_kv_cache_mixin reads swa_num_key_value_heads there).
        for attr in ("swa_head_dim", "swa_num_key_value_heads", "head_dim", "num_key_value_heads"):
            setattr(model_config.hf_config, attr, getattr(text_cfg, attr))

        model_config.head_dim = text_cfg.head_dim

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        kv_pool = memory_pools.token_to_kv_pool
        hidden_states, layers_kv_fused = self.language_model(forward_batch, kv_pool)
        output = self.logits_processor(
            hidden_states, self.language_model.embed_tokens, logits_metadata
        )
        return output, {"token_to_kv_pool": layers_kv_fused}, True, None

    # ------------------------------------------------------------------ #
    # Weight loading
    # ------------------------------------------------------------------ #

    def load_weights(self, model_config: ModelConfig) -> None:
        loader = WeightLoader(
            model=self, model_config=model_config, mesh=self.mesh, dtype=self.dtype
        )
        mappings = self._create_weight_mappings()
        loader.load_weights_from_safetensors(mappings)
        logger.info("Gemma4 language_model weights loaded (vision tower skipped).")

    def _k_eq_v_layers(self) -> set[int]:
        if not getattr(self.text_config, "attention_k_eq_v", False):
            return set()
        return {
            i
            for i, t in enumerate(self.text_config.layer_types)
            if t == _FULL
        }

    def _create_weight_mappings(self) -> dict[str, WeightMapping]:
        mappings: dict[str, WeightMapping] = {
            f"{_HF_PREFIX}.embed_tokens.weight": WeightMapping(
                target_path="language_model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            f"{_HF_PREFIX}.norm.weight": WeightMapping(
                target_path="language_model.norm.scale",
                sharding=(None,),
                transpose=False,
            ),
        }
        k_eq_v = self._k_eq_v_layers()
        for layer_id in range(self.text_config.num_hidden_layers):
            mappings.update(self._create_layer_mappings(layer_id, layer_id in k_eq_v))
        return mappings

    def _create_layer_mappings(self, layer_id: int, k_eq_v: bool) -> dict[str, WeightMapping]:
        src = f"{_HF_PREFIX}.layers.{layer_id}"
        dst = f"language_model.layers.{layer_id}"

        m: dict[str, WeightMapping] = {
            f"{src}.layer_scalar": WeightMapping(
                target_path=f"{dst}.layer_scalar", sharding=(None,), transpose=False
            ),
        }
        for ln in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            m[f"{src}.{ln}.weight"] = WeightMapping(
                target_path=f"{dst}.{ln}.scale", sharding=(None,), transpose=False
            )

        m[f"{src}.self_attn.q_norm.weight"] = WeightMapping(
            target_path=f"{dst}.self_attn.q_norm.scale", sharding=(None,), transpose=False
        )
        m[f"{src}.self_attn.k_norm.weight"] = WeightMapping(
            target_path=f"{dst}.self_attn.k_norm.scale", sharding=(None,), transpose=False
        )

        m[f"{src}.self_attn.q_proj.weight"] = WeightMapping(
            target_path=f"{dst}.self_attn.q_proj.weight",
            sharding=(None, "tensor"),
            transpose=True,
        )
        m[f"{src}.self_attn.o_proj.weight"] = WeightMapping(
            target_path=f"{dst}.self_attn.o_proj.weight",
            sharding=("tensor", None),
            transpose=True,
        )
        # Only full-attention layers need kv_head_padding (4 heads < TP=8).
        # Sliding layers have 16 heads ≥ any practical TP. WeightLoader's
        # padding uses model_config.head_dim (= full head_dim after patch),
        # so applying it to sliding (head_dim=256) would mis-reshape.
        m[f"{src}.self_attn.k_proj.weight"] = WeightMapping(
            target_path=f"{dst}.self_attn.k_proj.weight",
            sharding=(None, "tensor"),
            transpose=True,
            kv_head_padding=k_eq_v,
        )
        if not k_eq_v:
            m[f"{src}.self_attn.v_proj.weight"] = WeightMapping(
                target_path=f"{dst}.self_attn.v_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            )

        for proj, axes in (
            ("gate_proj", (None, "tensor")),
            ("up_proj", (None, "tensor")),
            ("down_proj", ("tensor", None)),
        ):
            m[f"{src}.mlp.{proj}.weight"] = WeightMapping(
                target_path=f"{dst}.mlp.{proj}.weight",
                sharding=axes,
                transpose=True,
            )
        return m


EntryClass = Gemma4ForConditionalGeneration
