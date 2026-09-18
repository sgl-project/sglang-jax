"""Unit tests for the Gemma4 MTP draft model.

Hub-free, mirroring ``test_qwen3_5`` / ``test_mimo_v2_nextn``: the config is
built in-process from the real ``google/gemma-4-12B-it-assistant``
``config.json`` scalars, with the real value noted beside each shrunk one (no
``from_pretrained`` / ``hf_hub_download``, so it never skips on an offline
runner). Dims are shrunk to keep module construction cheap on CPU; every
structural value and every *relationship* between values is verbatim, because
those are what the model actually branches on.

Two relationships in particular must survive shrinking, since collapsing either
one hides a real defect:
  * ``backbone_hidden_size != hidden_size`` -- equal values make a missing
    post_projection dimensionally invisible.
  * ``global_head_dim != head_dim`` -- equal values hide which one a layer picks.

Re-snapshot from the checkpoint if a future revision renames or adds keys.

Run on CPU with simulated devices:
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \\
      python -m pytest python/sgl_jax/test/models/test_gemma4_mtp.py -v
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.configs.gemma4 import Gemma4UnifiedAssistantConfig
from sgl_jax.srt.models.gemma4_mtp import (
    Gemma4AssistantForCausalLM,
    Gemma4MTPAttention,
    Gemma4MTPDecoderLayer,
    Gemma4UnifiedAssistantForCausalLM,
)
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

# Single-device mesh with explicit axis types for unit tests.
MESH = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])

# Real layer_types of the 12B assistant: three local layers then one global.
# Verbatim -- layer 0 is sliding and layer 3 is full, which is what the
# per-layer-type head_dim / KV-head / RoPE branches key off.
REAL_LAYER_TYPES = ["sliding_attention"] * 3 + ["full_attention"]
SLIDING_LAYER_ID = 0
FULL_LAYER_ID = 3

# text_config. HF names the *sliding* variant unprefixed and the full-attention
# variant with a `global_` prefix; Gemma4Config remaps them to swa_head_dim /
# head_dim, so building through that class exercises the remap too.
_TEXT_CONFIG = {
    "hidden_size": 128,  # real 1024
    "intermediate_size": 256,  # real 8192
    "num_attention_heads": 4,  # real 16
    "head_dim": 128,  # real 256  (sliding; 128-aligned, see merge_kv)
    "global_head_dim": 256,  # real 512  (full; kept 2x the sliding one)
    "num_key_value_heads": 2,  # real 8    (sliding)
    "num_global_key_value_heads": 1,  # real 1    (full) -- verbatim
    "vocab_size": 256,  # real 262144
    "max_position_embeddings": 4096,  # real 262144
    "num_hidden_layers": 4,  # verbatim
    "num_kv_shared_layers": 4,  # verbatim -- every draft layer is KV-shared
    "layer_types": REAL_LAYER_TYPES,  # verbatim
    "sliding_window": 1024,  # verbatim
    "attention_k_eq_v": True,  # verbatim
    "attention_bias": False,  # verbatim
    "rms_norm_eps": 1e-6,  # verbatim
    "tie_word_embeddings": True,  # verbatim
    "final_logit_softcapping": None,  # verbatim
    "rope_parameters": {  # verbatim
        "full_attention": {
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
            "rope_type": "proportional",
        },
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
    },
}

# Top-level keys (siblings of text_config in the real file).
_TOP_CONFIG = {
    "backbone_hidden_size": 384,  # real 3840 = the target's hidden_size
    "use_ordered_embeddings": False,  # verbatim
    "num_centroids": 16,  # real 2048
    "centroid_intermediate_top_k": 4,  # real 32
    "tie_word_embeddings": True,  # verbatim
}

# tie_word_embeddings is excluded: it appears at BOTH levels in the real file,
# and the model reads it off text_config, so it must not route top-level only.
_TOP_LEVEL_KEYS = frozenset(_TOP_CONFIG) - {"tie_word_embeddings"}

BACKBONE_HIDDEN = _TOP_CONFIG["backbone_hidden_size"]
DRAFT_HIDDEN = _TEXT_CONFIG["hidden_size"]
VOCAB = _TEXT_CONFIG["vocab_size"]


def _make_config(**overrides) -> Gemma4UnifiedAssistantConfig:
    """Real 12B assistant config with dims shrunk; overrides route by key."""
    text = dict(_TEXT_CONFIG)
    top = dict(_TOP_CONFIG)
    for key, value in overrides.items():
        if key in _TOP_LEVEL_KEYS:
            top[key] = value
        else:
            text[key] = value
    # Present at both levels in the real file; keep them in sync.
    if "tie_word_embeddings" in overrides:
        top["tie_word_embeddings"] = overrides["tie_word_embeddings"]
    return Gemma4UnifiedAssistantConfig(text_config=text, **top)


def _text_config(**overrides):
    """The remapped text_config, for constructing a layer/attention directly."""
    return _make_config(**overrides).text_config


class TestConfigRemap:
    """HF names the sliding variant unprefixed and the full one `global_`.

    Gemma4Config swaps them into (head_dim = full, swa_head_dim = sliding).
    Getting that backwards is silent -- attention still builds, just with the
    wrong head width -- so it is pinned against the real checkpoint's values.
    """

    def test_global_becomes_head_dim(self):
        tc = _text_config()
        assert tc.head_dim == _TEXT_CONFIG["global_head_dim"]  # full
        assert tc.swa_head_dim == _TEXT_CONFIG["head_dim"]  # sliding

    def test_global_becomes_num_key_value_heads(self):
        tc = _text_config()
        assert tc.num_key_value_heads == _TEXT_CONFIG["num_global_key_value_heads"]
        assert tc.swa_num_key_value_heads == _TEXT_CONFIG["num_key_value_heads"]

    def test_remap_is_not_applied_twice(self):
        """_gemma4_remapped guards it; a second pass would swap them back."""
        cfg = _make_config()
        first = (cfg.text_config.head_dim, cfg.text_config.swa_head_dim)
        Gemma4UnifiedAssistantConfig(text_config=cfg.text_config)
        assert (cfg.text_config.head_dim, cfg.text_config.swa_head_dim) == first

    def test_the_two_head_dims_differ(self):
        """Guard the fixture itself: equal values would make the swap untestable."""
        assert _TEXT_CONFIG["head_dim"] != _TEXT_CONFIG["global_head_dim"]


class TestPerLayerTypeGeometry:
    """Each layer picks head_dim / KV heads by its own attention type."""

    def _attn(self, layer_id):
        with jax.set_mesh(MESH):
            return Gemma4MTPAttention(
                config=_text_config(),
                layer_id=layer_id,
                max_position_embeddings=4096,
                attention_bias=False,
                dtype=jnp.bfloat16,
                mesh=MESH,
            )

    def test_full_layer_uses_global_geometry(self):
        attn = self._attn(FULL_LAYER_ID)
        assert attn.head_dim == _TEXT_CONFIG["global_head_dim"]
        # attention_k_eq_v is true, so a full layer takes the global KV count.
        assert attn.num_kv_heads == _TEXT_CONFIG["num_global_key_value_heads"]

    def test_sliding_layer_uses_local_geometry(self):
        attn = self._attn(SLIDING_LAYER_ID)
        assert attn.head_dim == _TEXT_CONFIG["head_dim"]
        assert attn.num_kv_heads == _TEXT_CONFIG["num_key_value_heads"]

    def test_layers_of_different_types_disagree(self):
        """The whole point of the KV-share map: a draft layer may only read a
        target layer of its own attention type."""
        full, sliding = self._attn(FULL_LAYER_ID), self._attn(SLIDING_LAYER_ID)
        assert (full.head_dim, full.num_kv_heads) != (
            sliding.head_dim,
            sliding.num_kv_heads,
        )

    def test_rope_differs_per_layer_type(self):
        """Real rope_parameters: full is 'proportional' at 1e6, sliding
        'default' at 1e4. A single shared RoPE would be wrong for one of them."""
        full, sliding = self._attn(FULL_LAYER_ID), self._attn(SLIDING_LAYER_ID)
        assert full.rotary_emb is not sliding.rotary_emb
        assert full.head_dim != sliding.head_dim


class TestGemma4MTPAttention:
    """Tests for the Q-only MTP attention layer."""

    def test_construction_full_attention(self):
        config = _text_config()
        with jax.set_mesh(MESH):
            attn = Gemma4MTPAttention(
                config=config,
                layer_id=FULL_LAYER_ID,
                max_position_embeddings=4096,
                attention_bias=False,
                dtype=jnp.bfloat16,
                mesh=MESH,
            )
        assert attn.is_sliding is False
        assert attn.layer_type == "full_attention"
        assert attn.sliding_window == 0

    def test_construction_sliding_attention(self):
        config = _text_config()
        with jax.set_mesh(MESH):
            attn = Gemma4MTPAttention(
                config=config,
                layer_id=SLIDING_LAYER_ID,
                max_position_embeddings=4096,
                attention_bias=False,
                dtype=jnp.bfloat16,
                mesh=MESH,
            )
        assert attn.is_sliding is True
        assert attn.layer_type == "sliding_attention"
        assert attn.sliding_window == 1024


class TestGemma4MTPDecoderLayer:
    """Tests for the MTP decoder layer."""

    def test_construction(self):
        config = _text_config()
        with jax.set_mesh(MESH):
            layer = Gemma4MTPDecoderLayer(
                config=config,
                mesh=MESH,
                layer_id=0,
                dtype=jnp.bfloat16,
            )
        assert layer.layer_id == 0
        assert layer.self_attn is not None
        assert layer.mlp is not None


class TestGemma4AssistantForCausalLM:
    """Tests for the top-level Gemma4 assistant (MTP) model."""

    def test_construction(self):
        config = _make_config()
        with jax.set_mesh(MESH):
            model = Gemma4AssistantForCausalLM(
                config=config,
                mesh=MESH,
                dtype=jnp.bfloat16,
            )
        assert model.num_mtp_layers == 4
        assert len(model.layers) == 4
        assert model.normalizer == BACKBONE_HIDDEN**0.5
        # tie_word_embeddings=True → lm_head is not created
        assert not hasattr(model, "lm_head") or model.lm_head is None

    def test_construction_with_separate_lm_head(self):
        config = _make_config(tie_word_embeddings=False)
        with jax.set_mesh(MESH):
            model = Gemma4AssistantForCausalLM(
                config=config,
                mesh=MESH,
                dtype=jnp.bfloat16,
            )
        assert model.lm_head is not None

    def test_construction_with_ordered_embeddings(self):
        config = _make_config(
            use_ordered_embeddings=True,
            num_centroids=16,
            centroid_intermediate_top_k=4,
        )
        with jax.set_mesh(MESH):
            model = Gemma4AssistantForCausalLM(
                config=config,
                mesh=MESH,
                dtype=jnp.bfloat16,
            )
        assert model.masked_centroids is not None
        assert model.masked_num_centroids == 16
        assert model.masked_vocab_size_per_centroid == VOCAB // 16

    def test_get_embed_and_head(self):
        config = _make_config(tie_word_embeddings=False)
        with jax.set_mesh(MESH):
            model = Gemma4AssistantForCausalLM(
                config=config,
                mesh=MESH,
                dtype=jnp.bfloat16,
            )
        embed, head = model.get_embed_and_head()
        assert embed is not None
        assert head is not None

    def test_set_embed(self):
        config = _make_config()
        with jax.set_mesh(MESH):
            model = Gemma4AssistantForCausalLM(
                config=config,
                mesh=MESH,
                dtype=jnp.bfloat16,
            )
        new_embed = nnx.Param(jnp.ones((VOCAB, BACKBONE_HIDDEN), dtype=jnp.bfloat16))
        model.set_embed(new_embed)  # no-op, shouldn't crash

    def test_set_embed_and_head(self):
        config = _make_config()
        with jax.set_mesh(MESH):
            model = Gemma4AssistantForCausalLM(
                config=config,
                mesh=MESH,
                dtype=jnp.bfloat16,
            )
        target_embed = nnx.Param(jnp.ones((VOCAB, BACKBONE_HIDDEN), dtype=jnp.bfloat16))
        target_head = nnx.Param(jnp.ones((VOCAB, BACKBONE_HIDDEN), dtype=jnp.bfloat16))
        model.set_embed_and_head(target_embed, target_head)
        assert model._target_embed_weight is target_embed

    def test_checkpoint_keys_match_the_real_assistant_layout(self):
        """Pin the exact HF key names, which are NOT uniformly prefixed.

        Read off the safetensors header of google/gemma-4-31B-it-assistant:

            pre_projection.weight          [1024, 10752]   <- top level
            post_projection.weight         [5376, 1024]    <- top level
            model.embed_tokens.weight      [262144, 1024]  <- model. prefix
            model.norm.weight              [1024]          <- model. prefix

        A mapping key that matches nothing fails SILENTLY: the parameter stays a
        ShapeDtypeStruct placeholder, model_runner swaps it for an empty array,
        the model reports "loaded successfully", and the first forward dies in
        dot_general with a (0,) contracting dimension. Nothing else in the suite
        catches that, because module shapes are correct either way.
        """
        config = _make_config()
        with jax.set_mesh(MESH):
            model = Gemma4AssistantForCausalLM(config=config, mesh=MESH, dtype=jnp.bfloat16)
        mappings = model._create_weight_mappings()

        for key in ("pre_projection.weight", "post_projection.weight"):
            assert key in mappings, f"{key} is top-level in the checkpoint"
            assert (
                f"model.{key}" not in mappings
            ), f"model.{key} matches no tensor in the checkpoint"

        for key in ("model.embed_tokens.weight", "model.norm.weight"):
            assert key in mappings, f"{key} carries the model. prefix in the checkpoint"

    def test_weight_mappings_exist(self):
        config = _make_config()
        with jax.set_mesh(MESH):
            model = Gemma4AssistantForCausalLM(
                config=config,
                mesh=MESH,
                dtype=jnp.bfloat16,
            )
        mappings = model._create_weight_mappings()
        assert len(mappings) > 0
        # All 4 layers have mappings (HF keys use model.layers. prefix)
        for layer_idx in range(4):
            prefix = f"model.layers.{layer_idx}"
            assert any(
                k.startswith(prefix) for k in mappings
            ), f"Missing weight mappings for layer {layer_idx}"
        # mtp. prefix aliases exist
        assert any(k.startswith("mtp.") for k in mappings)
        # Target paths use the inlined (flat) structure
        target_paths = {v.target_path for v in mappings.values()}
        assert any(
            p.startswith("layers.0.") for p in target_paths
        ), "Expected target_paths to use flat 'layers.N.' prefix"

    def test_softcapping(self):
        config = _make_config(final_logit_softcapping=50.0)
        with jax.set_mesh(MESH):
            model = Gemma4AssistantForCausalLM(
                config=config,
                mesh=MESH,
                dtype=jnp.bfloat16,
            )
        assert model.final_logit_softcapping == 50.0

    def test_compute_logits_masked(self):
        config = _make_config(
            use_ordered_embeddings=True,
            num_centroids=16,
            centroid_intermediate_top_k=4,
            tie_word_embeddings=False,
        )
        with jax.set_mesh(MESH):
            model = Gemma4AssistantForCausalLM(
                config=config,
                mesh=MESH,
                dtype=jnp.float32,
            )
            assert model.masked_centroids is not None

            @jax.jit
            def _compute(m, h):
                return m.compute_logits(h)

            hidden = jnp.ones((1, DRAFT_HIDDEN), dtype=jnp.float32)
            logits = _compute(model, hidden)
            assert logits.shape == (1, VOCAB)

    def test_compute_logits_lm_head(self):
        config = _make_config(
            tie_word_embeddings=False,
            use_ordered_embeddings=False,
        )
        with jax.set_mesh(MESH):
            model = Gemma4AssistantForCausalLM(
                config=config,
                mesh=MESH,
                dtype=jnp.float32,
            )

            @jax.jit
            def _compute(m, h):
                return m.compute_logits(h)

            hidden = jnp.ones((1, DRAFT_HIDDEN), dtype=jnp.float32)
            logits = _compute(model, hidden)
            assert logits.shape == (1, VOCAB)

    def test_select_and_score_shapes(self):
        config = _make_config(
            use_ordered_embeddings=True,
            num_centroids=16,
            centroid_intermediate_top_k=4,
        )
        with jax.set_mesh(MESH):
            model = Gemma4AssistantForCausalLM(
                config=config,
                mesh=MESH,
                dtype=jnp.float32,
            )

            @jax.jit
            def _compute(m, h, w):
                return m._select_and_score(h, w)

            bs = 2
            hidden = jnp.ones((bs, DRAFT_HIDDEN), dtype=jnp.float32)
            lm_head_weight = jnp.ones((VOCAB, DRAFT_HIDDEN), dtype=jnp.float32)
            logits, indices = _compute(model, hidden, lm_head_weight)
            assert logits.shape == (bs, 4 * (VOCAB // 16))
            assert indices.shape == (bs, 4 * (VOCAB // 16))


class TestProjectionContract:
    """The draft's hidden lives in two spaces and the projections move between them.

    ``pre_projection`` takes [target_embed(bb) ; target_hidden(bb)] -> hidden_size.
    ``post_projection`` takes hidden_size -> bb, because the captured hidden is fed
    back as the NEXT draft step's ``spec_info.hidden_states`` and re-enters
    ``pre_projection``. Every earlier test used backbone_hidden_size == hidden_size,
    which makes a missing post_projection invisible; these do not.
    """

    BB = 128
    HIDDEN = 256

    def _model(self, **overrides):
        config = _make_config(
            hidden_size=self.HIDDEN,
            backbone_hidden_size=self.BB,
            **overrides,
        )
        with jax.set_mesh(MESH):
            return Gemma4AssistantForCausalLM(config=config, mesh=MESH, dtype=jnp.float32)

    def test_pre_projection_consumes_two_backbone_vectors(self):
        model = self._model()
        assert model.pre_projection.weight.value.shape == (2 * self.BB, self.HIDDEN)

    def test_post_projection_returns_to_backbone_space(self):
        model = self._model()
        assert model.post_projection.weight.value.shape == (self.HIDDEN, self.BB)

    def test_post_projection_round_trip_shape(self):
        """A draft hidden must come out of post_projection sized for pre_projection."""
        model = self._model()
        with jax.set_mesh(MESH):
            draft_hidden = jnp.ones((3, self.HIDDEN), dtype=jnp.float32)
            projected, _ = model.post_projection(draft_hidden)
        assert projected.shape == (3, self.BB)

        # The fed-back hidden concatenated with a backbone-dim embedding must be
        # exactly what pre_projection accepts — this is the invariant that broke
        # when post_projection was never called.
        embed = jnp.ones((3, self.BB), dtype=jnp.float32)
        combined = jnp.concatenate([embed, projected], axis=-1)
        assert combined.shape[-1] == model.pre_projection.weight.value.shape[0]

    def test_normalizer_uses_backbone_dim(self):
        """Token embedding comes from the TARGET, so it scales by sqrt(bb)."""
        model = self._model()
        assert model.normalizer == self.BB**0.5


class TestSharedEmbedWiring:
    """The draft worker binds the target's embedding through a shared helper.

    That helper calls set_embed_and_head() only when the model declares
    load_lm_head_from_target, and set_embed() otherwise -- which is a no-op
    here. Without the flag nothing binds and the first forward raises. Nothing
    else in the suite exercises the flag, so it is pinned directly.
    """

    def test_declares_load_lm_head_from_target(self):
        assert Gemma4AssistantForCausalLM.load_lm_head_from_target is True
        assert Gemma4UnifiedAssistantForCausalLM.load_lm_head_from_target is True

    def test_worker_shared_embed_path_actually_binds(self):
        """Replays the branch the draft worker takes, rather than trusting it."""
        with jax.set_mesh(MESH):
            model = Gemma4AssistantForCausalLM(config=_make_config(), mesh=MESH, dtype=jnp.bfloat16)
        embed = nnx.Param(jnp.ones((VOCAB, BACKBONE_HIDDEN), dtype=jnp.bfloat16))
        head = nnx.Param(jnp.ones((VOCAB, DRAFT_HIDDEN), dtype=jnp.bfloat16))

        if getattr(model, "load_lm_head_from_target", False):
            model.set_embed_and_head(embed, head)
        else:
            model.set_embed(embed)

        assert getattr(model, "_target_embed_weight", None) is embed

    def test_set_embed_alone_is_insufficient(self):
        """Why the flag matters: the single-arg setter binds nothing."""
        with jax.set_mesh(MESH):
            model = Gemma4AssistantForCausalLM(config=_make_config(), mesh=MESH, dtype=jnp.bfloat16)
        model.set_embed(nnx.Param(jnp.ones((VOCAB, BACKBONE_HIDDEN), dtype=jnp.bfloat16)))
        assert getattr(model, "_target_embed_weight", None) is None


class TestGemma4UnifiedAssistantForCausalLM:
    """Tests for the unified assistant pass-through alias."""

    def test_is_subclass(self):
        assert issubclass(Gemma4UnifiedAssistantForCausalLM, Gemma4AssistantForCausalLM)

    def test_construction(self):
        config = _make_config()
        with jax.set_mesh(MESH):
            model = Gemma4UnifiedAssistantForCausalLM(
                config=config,
                mesh=MESH,
                dtype=jnp.bfloat16,
            )
        assert isinstance(model, Gemma4AssistantForCausalLM)
        assert model.num_mtp_layers == 4
