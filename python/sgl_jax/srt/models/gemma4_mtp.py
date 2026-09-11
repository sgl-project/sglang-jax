"""Gemma4 MTP (Multi-Token Prediction) draft model for speculative decoding.

Q-only attention layers that read K/V from the target verifier model's KV
cache. Ported from the vllm-project/tpu-inference JAX-native implementation
(PRs #2751, #2771) and the SGLang GPU ``Gemma4AssistantForCausalLM``.

Key design (matches SGLang GPU):
- Token embedding uses the *target* model's embedding (backbone_hidden_size-dim),
  stored as ``_target_embed_weight``. Draft's own ``embed_tokens``
  (hidden_size-dim) is only used via tied lm_head.
- ``pre_projection(2*bb_dim → hidden_size)``, ``post_projection(hidden_size → bb_dim)``.
- Attention is Q-only — K/V come from target KV cache via layer_id redirect.
"""

from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, get_rope
from sgl_jax.srt.layers.layernorm import GemmaRMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.profiling_utils import named_scope
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)

init_fn = nnx.initializers.uniform()


# ---------------------------------------------------------------------------
# Q-only Attention (K/V from target model's shared KV cache)
# ---------------------------------------------------------------------------


class Gemma4MTPAttention(nnx.Module):
    """Q-only attention for Gemma4 MTP draft layers.

    Only projects Q (and optionally applies Q-norm + RoPE). K/V are dummy
    zeros — the real K/V are read from the target model's KV cache via
    RadixAttention with a redirected layer_id (set by the draft worker's
    KV-share map). The Pallas kernel writes dummy K/V to draft-specific
    page slots (allocated via EagleDraftInput), separate from target pages.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        max_position_embeddings: int,
        attention_bias: bool,
        dtype: jnp.dtype,
        mesh: jax.sharding.Mesh,
    ):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)

        self.layer_type = "full_attention"
        if hasattr(config, "layer_types") and layer_id < len(config.layer_types):
            self.layer_type = config.layer_types[layer_id]

        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = getattr(config, "sliding_window", 0) if self.is_sliding else 0

        # Per-layer-type RoPE parameters
        rope_parameters = getattr(config, "rope_parameters", {})
        rope_params = dict(rope_parameters.get(self.layer_type, {}))
        if not self.is_sliding:
            if "rope_type" not in rope_params:
                rope_params["rope_type"] = "proportional"
            if "partial_rotary_factor" not in rope_params:
                rope_params["partial_rotary_factor"] = 0.25
            rope_theta = rope_params.get("rope_theta", getattr(config, "rope_theta", 1000000.0))
        else:
            if "rope_type" not in rope_params:
                rope_params["rope_type"] = "default"
            if "partial_rotary_factor" not in rope_params:
                rope_params["partial_rotary_factor"] = 1.0
            rope_theta = rope_params.get(
                "rope_theta", getattr(config, "rope_local_base_freq", 10000.0)
            )

        if not self.is_sliding:
            self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        else:
            self.head_dim = getattr(
                config,
                "swa_head_dim",
                getattr(config, "head_dim", self.hidden_size // self.num_heads),
            )

        use_k_eq_v = (not self.is_sliding) and getattr(config, "attention_k_eq_v", False)
        if use_k_eq_v:
            self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)
        else:
            self.num_kv_heads = getattr(
                config,
                "swa_num_key_value_heads",
                getattr(config, "num_key_value_heads", self.num_heads),
            )

        self.q_head_num = self.num_heads
        self.kv_head_num = self.num_kv_heads
        self.mesh = mesh

        self.q_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.num_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="q_proj",
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, epsilon=rms_norm_eps, add_unit_offset=False)
        self.o_proj = LinearBase(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            use_bias=attention_bias,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="o_proj",
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            rope_scaling=rope_params,
            dtype=dtype,
        )

        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=1.0,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=self.sliding_window,
        )

    @named_scope
    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, jax.Array]:
        q, _ = self.q_proj(hidden_states)
        q = q.reshape(
            -1,
            self.q_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )
        q = self.q_norm(q)
        dummy_k = jnp.zeros_like(q)
        q, _ = self.rotary_emb(positions, q, dummy_k)

        # Dummy K/V — real K/V from target KV cache via redirected layer_id.
        # Name the sharding: a bare jnp.zeros() is REPLICATED under an explicit
        # mesh, whereas real projected K/V (and q above) are head-sharded on
        # "tensor". The attention backend's shard_map in_specs expect the latter,
        # so leaving these replicated forces an implicit reshard on every draft
        # step at best, and a sharding error at worst.
        kv_sharding = NamedSharding(self.mesh, P("data", "tensor", None))
        dummy_k = jnp.zeros(
            (q.shape[0], self.kv_head_num, self.head_dim),
            dtype=q.dtype,
            out_sharding=kv_sharding,
        )
        dummy_v = jnp.zeros(
            (q.shape[0], self.kv_head_num, self.head_dim),
            dtype=q.dtype,
            out_sharding=kv_sharding,
        )

        attn_output, kv_fused = self.attn(q, dummy_k, dummy_v, forward_batch, token_to_kv_pool)
        output, _ = self.o_proj(attn_output)
        return output, kv_fused


# ---------------------------------------------------------------------------
# MTP Decoder Layer
# ---------------------------------------------------------------------------


class Gemma4MTPDecoderLayer(nnx.Module):
    """Dense MTP decoder layer (no MoE)."""

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.mesh = mesh
        max_position_embeddings = getattr(config, "max_position_embeddings", 256000)
        attention_bias = getattr(config, "attention_bias", False)
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)

        self.layer_type = "full_attention"
        if hasattr(config, "layer_types") and layer_id < len(config.layer_types):
            self.layer_type = config.layer_types[layer_id]

        self.layer_scalar = nnx.Param(jnp.ones((1,), dtype=dtype))
        self.input_layernorm = GemmaRMSNorm(
            config.hidden_size,
            epsilon=rms_norm_eps,
            add_unit_offset=False,
        )
        self.self_attn = Gemma4MTPAttention(
            config=config,
            layer_id=layer_id,
            max_position_embeddings=max_position_embeddings,
            attention_bias=attention_bias,
            dtype=dtype,
            mesh=mesh,
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size,
            epsilon=rms_norm_eps,
            add_unit_offset=False,
        )
        self.pre_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size,
            epsilon=rms_norm_eps,
            add_unit_offset=False,
        )
        self.post_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size,
            epsilon=rms_norm_eps,
            add_unit_offset=False,
        )

        from sgl_jax.srt.models.gemma4 import Gemma4MLP

        self.mlp = Gemma4MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            layer_id=layer_id,
            dtype=dtype,
            mesh=mesh,
        )

    @named_scope
    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, kv_fused = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )
        attn_output = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + attn_output
        residual = hidden_states

        mlp_input = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(mlp_input)
        mlp_output = self.post_feedforward_layernorm(hidden_states)
        outputs = residual + mlp_output
        outputs = outputs * self.layer_scalar.value
        return outputs, kv_fused, [], None


# ---------------------------------------------------------------------------
# Schema-driven per-layer weight specs
# ---------------------------------------------------------------------------

_LAYER_WEIGHT_SPECS: list[tuple[str, tuple, bool]] = [
    ("layer_scalar", (None,), False),
    ("input_layernorm.weight", (None,), False),
    ("post_attention_layernorm.weight", (None,), False),
    ("pre_feedforward_layernorm.weight", (None,), False),
    ("post_feedforward_layernorm.weight", (None,), False),
    ("self_attn.q_proj.weight", (None, "tensor"), True),
    ("self_attn.q_norm.weight", (None,), False),
    ("self_attn.o_proj.weight", ("tensor", None), True),
    ("mlp.gate_proj.weight", (None, "tensor"), True),
    ("mlp.up_proj.weight", (None, "tensor"), True),
    ("mlp.down_proj.weight", ("tensor", None), True),
]


# ---------------------------------------------------------------------------
# Gemma4AssistantForCausalLM — main MTP draft model
# ---------------------------------------------------------------------------


class Gemma4AssistantForCausalLM(nnx.Module):
    """Gemma4 MTP (frozen-KV) draft model for speculative decoding.

    Matches the HF checkpoint ``"architectures": ["Gemma4AssistantForCausalLM"]``
    and the upstream SGLang GPU ``Gemma4AssistantForCausalLM`` pattern:

    - Token embedding uses the *target* model's embedding (backbone_hidden_size-dim),
      stored as ``_target_embed_weight``.
    - Draft's own ``embed_tokens`` (hidden_size-dim) is only used via tied lm_head.
    - ``set_embed_and_head(embed, head)`` stores the target embedding, discards head.
    - Attention is Q-only — K/V come from target KV cache via redirected layer_id.
    - ``pre_projection(2*bb_dim → hidden_size)``, ``post_projection(hidden_size → bb_dim)``.
    """

    # The draft worker's shared-embed path calls set_embed_and_head() only when
    # this is set, and falls back to set_embed() otherwise -- which is a no-op
    # here. Without this flag _target_embed_weight never binds and the first
    # forward raises "target embedding is not bound". Same convention as
    # mimo_v2_nextn.py.
    load_lm_head_from_target = True

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        super().__init__()
        self.mesh = mesh
        self.config = getattr(config, "text_config", config)
        self.dtype = dtype

        full_config = config
        backbone_hidden_size = getattr(full_config, "backbone_hidden_size", self.config.hidden_size)
        num_mtp_layers = getattr(self.config, "num_hidden_layers", 4)
        self.num_mtp_layers = num_mtp_layers
        self.backbone_hidden_size = backbone_hidden_size
        self.normalizer = backbone_hidden_size**0.5

        # -- Core MTP layers (inlined from MultiTokenPredictor) -----------------

        # Draft's own embed_tokens (hidden_size-dim) — for tied lm_head only.
        # Token embedding in forward uses target's embedding instead.
        self.embed_tokens = Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            param_dtype=dtype,
            dtype=dtype,
        )

        self.pre_projection = LinearBase(
            input_size=2 * backbone_hidden_size,
            output_size=self.config.hidden_size,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            mesh=mesh,
            scope_name="pre_projection",
        )
        self.post_projection = LinearBase(
            input_size=self.config.hidden_size,
            output_size=backbone_hidden_size,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            mesh=mesh,
            scope_name="post_projection",
        )

        self.layers = nnx.data(
            [
                Gemma4MTPDecoderLayer(config=self.config, mesh=mesh, layer_id=i, dtype=dtype)
                for i in range(num_mtp_layers)
            ]
        )

        self.norm = GemmaRMSNorm(
            self.config.hidden_size,
            epsilon=getattr(self.config, "rms_norm_eps", 1e-6),
            add_unit_offset=False,
        )

        # -- Output -------------------------------------------------------------

        self.final_logit_softcapping = getattr(self.config, "final_logit_softcapping", None)

        # Draft's lm_head: hidden_size → vocab.
        # When tie_word_embeddings=True (default for Gemma4 MTP), lm_head
        # aliases embed_tokens (hidden_size-dim).
        if not getattr(self.config, "tie_word_embeddings", True):
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                dtype=dtype,
                param_dtype=dtype,
                kernel_axes=("tensor", None),
            )

        # -- Sparse masked (centroid-based) embedder (inlined) ------------------

        use_ordered = getattr(full_config, "use_ordered_embeddings", False)
        if use_ordered:
            num_centroids = getattr(full_config, "num_centroids", 2048)
            top_k = getattr(full_config, "centroid_intermediate_top_k", 32)
            self.masked_num_centroids = num_centroids
            self.masked_centroid_top_k = top_k
            self.masked_vocab_size_per_centroid = self.config.vocab_size // num_centroids
            self.masked_num_selected = top_k * self.masked_vocab_size_per_centroid
            self.masked_centroids = LinearBase(
                input_size=self.config.hidden_size,
                output_size=num_centroids,
                use_bias=False,
                params_dtype=dtype,
                kernel_axes=(None, "tensor"),
                mesh=mesh,
                scope_name="masked_centroids",
            )
            self.masked_token_ordering = nnx.Param(
                jnp.zeros((self.config.vocab_size,), dtype=jnp.int32),
                eager_sharding=False,
            )
        else:
            self.masked_centroids = None

        self.logits_processor = LogitsProcessor(
            self.config.vocab_size,
            soft_cap=getattr(self.config, "final_logit_softcapping", None),
            mesh=self.mesh,
        )
        self.capture_aux_hidden_states = False

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
        if not loader.dummy_mode:
            weight_info = loader._scan_weight_info()
            weight_mappings = {k: v for k, v in weight_mappings.items() if k in weight_info}
        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("Gemma4 MTP weights loaded successfully!")

    def _create_weight_mappings(self) -> dict:
        mappings = {}

        # Draft's own embed_tokens (hidden_size-dim, loaded from checkpoint)
        mappings["model.embed_tokens.weight"] = WeightMapping(
            target_path="embed_tokens.embedding",
            sharding=("tensor", None),
            transpose=False,
        )

        # lm_head: tied to embed_tokens by default
        if not hasattr(self, "lm_head") or self.lm_head is None:
            mappings["lm_head.weight"] = WeightMapping(
                target_path="embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            )
        else:
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding",
                sharding=("tensor", None),
                transpose=False,
            )

        # Pre/post projections.
        #
        # NOTE THE MISSING "model." PREFIX. In google/gemma-4-*-it-assistant these
        # two tensors sit at the TOP LEVEL of the checkpoint -- verified against
        # the safetensors header:
        #     pre_projection.weight        [1024, 10752]   (hidden, 2*backbone)
        #     post_projection.weight       [5376, 1024]    (backbone, hidden)
        #     model.embed_tokens.weight    [262144, 1024]  <- these DO carry it
        #     model.norm.weight            [1024]
        # Mapping them as "model.pre_projection.weight" matched nothing, so both
        # stayed ShapeDtypeStruct placeholders, model_runner swapped them for empty
        # arrays (the "replaced 2 ShapeDtypeStruct state placeholders" log line),
        # and the first forward died in dot_general with a (0,) contracting
        # dimension. A silent no-match is the dangerous part: nothing warns, and
        # the loader still reports "weights loaded successfully".
        mappings["pre_projection.weight"] = WeightMapping(
            target_path="pre_projection.weight",
            sharding=(None, "tensor"),
            transpose=True,
        )
        mappings["post_projection.weight"] = WeightMapping(
            target_path="post_projection.weight",
            sharding=(None, "tensor"),
            transpose=True,
        )
        mappings["model.norm.weight"] = WeightMapping(
            target_path="norm.weight",
            sharding=(None,),
            transpose=False,
        )

        # Centroid masked embedder (inlined attributes)
        if self.masked_centroids is not None:
            mappings["masked_embedding.centroids.weight"] = WeightMapping(
                target_path="masked_centroids.weight",
                sharding=(None, "tensor"),
                transpose=True,
            )
            mappings["masked_embedding.token_ordering"] = WeightMapping(
                target_path="masked_token_ordering",
                sharding=(None,),
                transpose=False,
            )

        # Per-layer mappings (schema-driven)
        attn_bias = getattr(self.config, "attention_bias", False)
        for layer_idx in range(self.num_mtp_layers):
            mappings.update(self._create_layer_mappings(layer_idx, attn_bias))

        # "mtp." prefix (native checkpoint layout)
        for k, v in list(mappings.items()):
            if k.startswith("model."):
                mappings[k.replace("model.", "mtp.", 1)] = v

        # "language_model.model." prefix (multimodal layout)
        for k, v in list(mappings.items()):
            if k.startswith("model."):
                mappings[f"language_model.{k}"] = v

        return mappings

    def _create_layer_mappings(self, layer_idx: int, attn_bias: bool) -> dict:
        hf_prefix = f"model.layers.{layer_idx}"
        target_prefix = f"layers.{layer_idx}"

        mappings = {}
        for suffix, sharding, transpose in _LAYER_WEIGHT_SPECS:
            mappings[f"{hf_prefix}.{suffix}"] = WeightMapping(
                target_path=f"{target_prefix}.{suffix}",
                sharding=sharding,
                transpose=transpose,
            )

        if attn_bias:
            for proj in ("q_proj", "o_proj"):
                mappings[f"{hf_prefix}.self_attn.{proj}.bias"] = WeightMapping(
                    target_path=f"{target_prefix}.self_attn.{proj}.bias",
                    sharding=(None,),
                    transpose=False,
                )
        return mappings

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _embed_input_ids(self, input_ids: jax.Array) -> jax.Array:
        """Embed tokens using the stored TARGET embedding (backbone_hidden_size-dim)."""
        if not hasattr(self, "_target_embed_weight") or self._target_embed_weight is None:
            raise RuntimeError(
                "Gemma4 MTP target embedding is not bound. "
                "Call set_embed_and_head() before forward."
            )
        embed = self._target_embed_weight
        embed = embed.value if hasattr(embed, "value") else embed

        # The target's embedding is VOCAB-sharded -- gemma4.py builds it with
        # kernel_axes=("tensor", None), so this array is [vocab@tensor, features].
        # Gathering arbitrary rows from it therefore needs a collective, and under
        # an explicit-sharding mesh JAX refuses to guess the output spec: plain
        # `embed[input_ids]` raises ShardingTypeError. Name the spec instead, with
        # exactly the expression Embed.__call__ uses on this same array, so the
        # draft's embeddings are laid out identically to the target's.
        output_pspec = P("data", *([None] * (input_ids.ndim - 1)), None)
        token_embed = embed.at[input_ids].get(out_sharding=NamedSharding(self.mesh, output_pspec))
        return token_embed * self.normalizer

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: Any,
        logits_metadata: LogitsMetadata,
    ):
        kv_pool = memory_pools.token_to_kv_pool
        target_hidden_states = forward_batch.spec_info.hidden_states
        positions = forward_batch.positions

        # Embed with TARGET embedding (backbone_hidden_size-dim) + target hidden
        inputs_embeds = self._embed_input_ids(forward_batch.input_ids)

        # These two arrive with different specs: the embedding gather produces
        # P("data", None) (the repo's convention for [tokens, features], as in
        # sampler.py), while the captured target hidden states come back
        # replicated as P(None, None). jnp.concatenate requires its operands to
        # agree, so under an explicit-sharding mesh the mismatch is an error
        # rather than an implicit resharding. Move the replicated one onto the
        # sharded spec -- that direction is a local slice, whereas gathering the
        # embeddings up to replicated would cost an all-gather every draft step.
        target_hidden_states = jax.sharding.reshard(
            target_hidden_states, NamedSharding(self.mesh, P("data", None))
        )
        combined = jnp.concatenate([inputs_embeds, target_hidden_states], axis=-1)
        hidden_states, _ = self.pre_projection(combined)

        # Run all MTP layers. The per-layer fused KV the attention kernel returns
        # is DISCARDED on purpose — see the frozen-KV note on the return below.
        for layer in self.layers:
            hidden_states, _, _, _ = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                token_to_kv_pool=kv_pool,
            )

        draft_hidden = self.norm(hidden_states)

        # Compute logits from the hidden_size-dim tensor (lm_head's input space).
        if hasattr(self, "lm_head") and self.lm_head is not None:
            output = self.logits_processor(draft_hidden, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(draft_hidden, self.embed_tokens, logits_metadata)

        # The captured hidden is fed back as the NEXT draft step's
        # spec_info.hidden_states (eagle_draft_worker.draft_forward), where it is
        # concatenated with a backbone-dim embedding into pre_projection(2*bb_dim).
        # So it has to leave here in backbone space, not hidden space. Projecting
        # after the logits processor's row selection is equivalent to projecting
        # before it (post_projection is per-token linear) and touches fewer rows.
        if output.hidden_states is not None:
            output.hidden_states, _ = self.post_projection(output.hidden_states)

        # Frozen KV: this draft never owns KV state. Its attention reads the
        # TARGET's cache (RadixAttention.layer_id is redirected by the draft
        # worker) and its dummy zero K/V must never reach the pool — the write
        # back in MemoryPools.replace_all is POSITIONAL, so returning the draft's
        # own per-layer arrays would land them in target layers 0..N-1 and destroy
        # verified KV. memory_pools is a donated jit argument, so we cannot return
        # nothing either; pass the target's buffers straight through, unmodified.
        return output, {"token_to_kv_pool": kv_pool.get_all_fused_kv_buffers()}, [], None

    # ------------------------------------------------------------------
    # Embed / head sharing with target model
    # ------------------------------------------------------------------

    def get_embed_and_head(self):
        """Return (target_embed_weight, lm_head_weight).

        ``target_embed_weight`` is the TARGET embedding (bb-dim), set via
        ``set_embed_and_head()``. ``lm_head_weight`` is the draft's own.
        """
        target_embed = getattr(self, "_target_embed_weight", None)
        if target_embed is None:
            target_embed = self.embed_tokens.embedding
        head = (
            self.lm_head.embedding
            if hasattr(self, "lm_head") and self.lm_head is not None
            else self.embed_tokens.embedding
        )
        return target_embed, head

    def set_embed(self, _embed_weight):
        """No-op: single-embed set not used by Gemma4 MTP."""
        pass

    def set_embed_and_head(self, embed_weight, head_weight):
        """Store target embedding for token embed; discard target head.

        Matches SGLang GPU: ``set_embed_and_head(embed, head)`` stores
        ``embed`` as the target embedding (backbone_hidden_size-dim) and
        ignores ``head`` — the assistant keeps its own lm_head.
        """
        del head_weight
        self._target_embed_weight = embed_weight

    # ------------------------------------------------------------------
    # Logit computation (with optional masked embedder)
    # ------------------------------------------------------------------

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        """Standalone logit computation — NOT used by ``__call__``.

        ``__call__`` deliberately goes through ``self.logits_processor`` (dense
        full-vocab head) even when ``use_ordered_embeddings`` loaded the centroid
        weights. The masked path is an *approximation*, not a different id space:
        ``masked_token_ordering`` stores token ids grouped into centroid clusters
        and ``_select_and_score`` indexes an unpermuted ``lm_head`` by those ids,
        so the dense path is strictly more accurate, never wrong. It scores
        ``centroid_intermediate_top_k * (vocab / num_centroids)`` candidates via a
        large per-token gather, which on TPU is usually slower than the dense
        matmul it replaces — the tradeoff that makes it a win on GPU does not
        obviously carry over.

        Enabling it in ``__call__`` is therefore a benchmark-gated optimization,
        not a fix. Kept here (and unit-tested) so that switch is a small change.
        """
        if self.masked_centroids is not None:
            return self._compute_logits_masked(hidden_states)

        if hasattr(self, "lm_head") and self.lm_head is not None:
            # ParallelLMHead weight is (vocab_size, hidden_size); transpose for
            # (hidden_states) @ (hidden_size, vocab_size) → (tokens, vocab_size).
            logits = hidden_states @ self.lm_head.embedding.value.T
        else:
            # Embed.embedding is (vocab_size, hidden_size); .decode() handles
            # the transpose internally.
            logits = self.embed_tokens.decode(hidden_states)

        if self.final_logit_softcapping is not None:
            logits = jnp.tanh(logits / self.final_logit_softcapping) * self.final_logit_softcapping
        return logits

    def _get_full_lm_head_weight(self) -> jax.Array:
        if hasattr(self, "lm_head") and self.lm_head is not None:
            return self.lm_head.embedding.value
        return self.embed_tokens.embedding.value

    # -- Inlined MaskedEmbedder methods -----------------------------------------

    def _select_and_score(
        self,
        hidden_states: jax.Array,
        lm_head_weight: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        num_tokens = hidden_states.shape[0]
        if lm_head_weight.shape == (self.config.hidden_size, self.config.vocab_size):
            lm_head_weight = lm_head_weight.T
        elif lm_head_weight.shape != (self.config.vocab_size, self.config.hidden_size):
            raise ValueError(f"Unexpected lm_head_weight shape: {lm_head_weight.shape}")
        centroid_logits, _ = self.masked_centroids(hidden_states)
        # The centroid projection shards its output vocabulary over the tensor
        # axis.  Top-k needs to compare every centroid for each token, so the
        # selected axis must be replicated before invoking lax.top_k.  This is
        # the same explicit reshard used by the generic speculative top-k
        # helper; leaving the output tensor-sharded fails under explicit
        # multi-device sharding (and can select only local centroids).
        centroid_logits = jax.sharding.reshard(centroid_logits, NamedSharding(self.mesh, P()))
        _, top_k_indices = jax.lax.top_k(centroid_logits, k=self.masked_centroid_top_k)
        clusters = self.masked_token_ordering.value.reshape(
            self.masked_num_centroids, self.masked_vocab_size_per_centroid
        )
        # Both gathers pick arbitrary rows by token id. Under an explicit-sharding
        # mesh jnp.take cannot infer an output spec for that (and lm_head_weight is
        # vocab-sharded, so it needs a collective), which is why plain jnp.take
        # raises ShardingTypeError. Name the output spec so JAX inserts the
        # all-gather — the cost that makes this path a poor trade on TPU and the
        # reason compute_logits() is not wired into __call__.
        replicated = NamedSharding(self.mesh, P())
        selected = clusters.at[top_k_indices].get(out_sharding=replicated)
        selected_flat = selected.reshape(num_tokens, self.masked_num_selected)
        embeddings = lm_head_weight.at[selected_flat].get(out_sharding=replicated)
        logits = jnp.einsum("td,tsd->ts", hidden_states, embeddings)
        return logits, selected_flat

    def _compute_logits_masked(self, hidden_states: jax.Array) -> jax.Array:
        logits, indices = self._select_and_score(hidden_states, self._get_full_lm_head_weight())
        num_tokens = hidden_states.shape[0]
        output = jnp.full(
            (num_tokens, self.config.vocab_size),
            jnp.finfo(hidden_states.dtype).min,
            dtype=hidden_states.dtype,
        )
        row_indices = jnp.arange(num_tokens)[:, None]
        return output.at[row_indices, indices].set(logits)

    def _get_top_tokens_masked(
        self, hidden_states: jax.Array, lm_head_weight: jax.Array
    ) -> jax.Array:
        logits, indices = self._select_and_score(hidden_states, lm_head_weight)
        best_idx = jnp.argmax(logits, axis=-1, keepdims=True)
        return jnp.take_along_axis(indices, best_idx, axis=-1).squeeze(-1)


# ---------------------------------------------------------------------------
# Unified assistant alias (12B model — text path identical)
# ---------------------------------------------------------------------------


class Gemma4UnifiedAssistantForCausalLM(Gemma4AssistantForCausalLM):
    """Alias matching the HF unified assistant checkpoint arch name.

    The 12B-it-unified-assistant checkpoint declares
    ``"architectures": ["Gemma4UnifiedAssistantForCausalLM"]`` with
    ``model_type: "gemma4_unified_assistant"``. The text transformer path
    is identical to the non-unified assistant — this alias lets the
    ``ModelRegistry`` resolve it directly.
    """


EntryClass = [Gemma4AssistantForCausalLM, Gemma4UnifiedAssistantForCausalLM]
