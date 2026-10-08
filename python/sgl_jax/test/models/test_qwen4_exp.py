"""Qwen3.8-Flash-Next assembly, on CPU.

Shape and structure only -- numbers need the real checkpoint. What is pinned
here is the four things the backbone does differently from Qwen3.5, each of
which is silent if wrong: which layers get which block, that every block is
wrapped in its own hyper connection, that the streams widen exactly once, and
that the mapping table names parameters the model actually has.
"""

from __future__ import annotations

import importlib.util
import os
import types
import unittest
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import AxisType, Mesh

from sgl_jax.srt.configs.model_config import ModelConfig, MoEBackend
from sgl_jax.srt.configs.qwen4_exp import Qwen4ExpConfig
from sgl_jax.srt.layers.attention.qsa_sparse_backend import QSAFusedCache
from sgl_jax.srt.layers.embeddings import MRotaryEmbedding, RotaryEmbedding
from sgl_jax.srt.layers.fused_moe import FusedEPMoE
from sgl_jax.srt.models.qwen3_5 import (
    Qwen3_5GatedDeltaNet,
    _create_qwen3_5_weight_mappings,
)
from sgl_jax.srt.models.qwen4_exp import (
    Qwen4ExpAttention,
    Qwen4ExpDecoderLayer,
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpModel,
    _create_qwen4_exp_weight_mappings,
)
from sgl_jax.test.test_utils import CustomTestCase

NUM_LAYERS = 8
INTERVAL = 4
PLE_LAYER_1BASED = 2


def _mesh():
    return Mesh(
        np.array(jax.devices())[:1].reshape(1, 1),
        axis_names=("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


# The section widths a real checkpoint ships, scaled to this test's head_dim:
# they have to sum to rotary_dim // 2.
MROPE_SECTION = [3, 3, 2]


def _config(*, num_layers=NUM_LAYERS, ple=False, mrope=False, **overrides):
    text = dict(
        num_hidden_layers=num_layers,
        full_attention_interval=INTERVAL,
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        intermediate_size=512,
        vocab_size=512,
        num_experts=8,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=256,
        num_experts_per_tok=2,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        indexer_head_dim=128,
        indexer_n_heads=4,
        indexer_kv_heads=1,
    )
    if ple:
        text["ple_layer_ids"] = [PLE_LAYER_1BASED]
    if mrope:
        # A real checkpoint nests RoPE under rope_parameters; the config
        # class flattens it into rope_scaling.
        text["rope_parameters"] = dict(
            rope_type="default",
            mrope_section=MROPE_SECTION,
            mrope_interleaved=True,
            rope_theta=10000000,
            partial_rotary_factor=0.25,
        )
    text.update(overrides)
    return Qwen4ExpConfig(text_config=text)


def _model(cfg, mesh):
    with jax.set_mesh(mesh):
        return Qwen4ExpModel(cfg, mesh)


class TestBackboneStructure(CustomTestCase):
    def test_layer_types_follow_the_interval(self):
        """Every fourth layer is full attention and the rest are GDN."""
        cfg = _config()
        model = _model(cfg, _mesh())

        full = [i for i, layer in enumerate(model.layers) if layer.is_full_attn]
        self.assertEqual(full, list(cfg.text_config.full_attention_layer_ids))
        self.assertEqual(len(full), NUM_LAYERS // INTERVAL)
        self.assertTrue(all(layer.ple is None for layer in model.layers))

    @unittest.skipUnless(
        importlib.util.find_spec("sgl_jax.srt.layers.ngram_embedding"),
        "the N-gram embedding module is not in the tree yet",
    )
    def test_the_ngram_layer_is_a_gdn_layer(self):
        """``ple_layer_ids`` is 1-based, matching the checkpoint's numbering,
        and the layer it names is an ordinary GDN layer that happens to carry
        the module -- not a type of its own."""
        model = _model(_config(ple=True), _mesh())
        ple = [i for i, layer in enumerate(model.layers) if layer.ple is not None]
        self.assertEqual(ple, [PLE_LAYER_1BASED - 1])
        self.assertFalse(model.layers[ple[0]].is_full_attn)

    def test_every_block_gets_its_own_hyper_connection(self):
        """Two per layer, because a layer has an attention slot and an MLP slot
        whichever family fills the first one."""
        model = _model(_config(), _mesh())
        for i, layer in enumerate(model.layers):
            self.assertTrue(layer.attn_hyper_connection.use_combine, f"layer {i}")
            self.assertTrue(layer.mlp_hyper_connection.use_combine, f"layer {i}")

    def test_the_hyper_connections_are_the_only_normalization(self):
        """Qwen3.5's three RMSNorms are absorbed into the mix, so neither the
        layers nor the model may still hold one."""
        model = _model(_config(), _mesh())
        self.assertFalse(hasattr(model, "norm"))
        for i, layer in enumerate(model.layers):
            self.assertFalse(hasattr(layer, "input_layernorm"), f"layer {i}")
            self.assertFalse(hasattr(layer, "post_attention_layernorm"), f"layer {i}")

        # The mixer reads the streams down and never writes back, so it builds
        # no injection weights.
        self.assertFalse(model.hyper_connection_mixer.use_combine)

    def test_the_streams_widen_once(self):
        """The embedding is one stream wide and the layers carry hc_count of
        them; widening is idempotent so only the first layer pays it."""
        cfg = _config()
        text = cfg.text_config
        model = _model(cfg, _mesh())
        narrow = jnp.zeros((3, text.hidden_size))
        wide = model.layers[0]._to_streams(narrow)

        self.assertEqual(wide.shape, (3, text.hc_count * text.hidden_size))
        self.assertEqual(model.layers[1]._to_streams(wide).shape, wide.shape)
        # Repeated, not zero-padded: every stream starts as the embedding.
        ones = jnp.ones((3, text.hidden_size))
        np.testing.assert_array_equal(
            np.asarray(model.layers[0]._to_streams(ones)), np.ones((3, wide.shape[-1]))
        )
        with self.assertRaises(ValueError):
            model.layers[0]._to_streams(jnp.zeros((3, text.hidden_size + 1)))


class TestGatedDeltaNet(CustomTestCase):
    def test_the_output_gate_follows_the_config(self):
        """Flash-Next gates the GDN output with sigmoid(z) where Qwen3.5 uses
        silu(z), and output_gate_type says which. Either passes a shape test,
        so the values are compared."""
        mesh = _mesh()
        T = 4
        for gate_type, act, other in (
            ("sigmoid", jax.nn.sigmoid, jax.nn.silu),
            ("swish", jax.nn.silu, jax.nn.sigmoid),
        ):
            with self.subTest(gate_type):
                cfg = _config(output_gate_type=gate_type)
                n_v = cfg.text_config.linear_num_value_heads
                d_v = cfg.text_config.linear_value_head_dim
                with jax.set_mesh(mesh):
                    gdn = Qwen3_5GatedDeltaNet(cfg, mesh, 0)
                    core = jax.random.normal(jax.random.key(0), (T, n_v, d_v))
                    z = jax.random.normal(jax.random.key(1), (T, n_v * d_v))
                    got = np.asarray(gdn._norm_gate(core, z), np.float32)
                    normed = gdn.norm(core).reshape(T, n_v * d_v)
                    want = np.asarray(normed * act(z), np.float32)
                    wrong = np.asarray(normed * other(z), np.float32)
                np.testing.assert_allclose(got, want, rtol=2e-2, atol=1e-2)
                self.assertFalse(np.allclose(got, wrong, rtol=2e-2, atol=1e-2))


class TestRotary(CustomTestCase):
    def test_the_attention_follows_the_checkpoint_into_mrope(self):
        """A checkpoint that ships mrope_section gets the multimodal rotary.
        Reading only the flat ``rope_scaling`` key says None for both, because
        the real layout nests under ``rope_parameters``."""
        mesh = _mesh()
        with jax.set_mesh(mesh):
            with_mrope = Qwen4ExpAttention(_config(mrope=True), mesh, layer_id=INTERVAL - 1)
            without = Qwen4ExpAttention(_config(), mesh, layer_id=INTERVAL - 1)

        self.assertIsInstance(with_mrope.rotary_emb, MRotaryEmbedding)
        self.assertIsInstance(without.rotary_emb, RotaryEmbedding)
        self.assertNotIsInstance(without.rotary_emb, MRotaryEmbedding)

    def test_the_indexer_gets_a_rotary_sized_to_its_own_slice(self):
        """``QSAIndexer`` hands the rotary only the leading rotary_dim of its
        narrower head, so it needs head_size == rotary_dim -- the attention's
        own rotary is head_dim wide and would be rejected. The indexer's
        positions are group indices, so it is never the multimodal variant."""
        cfg = _config(mrope=True)
        text = cfg.text_config
        mesh = _mesh()
        with jax.set_mesh(mesh):
            attn = Qwen4ExpAttention(cfg, mesh, layer_id=INTERVAL - 1)

        rotary_dim = int(text.head_dim * float(text.partial_rotary_factor))
        self.assertEqual(attn.indexer_rotary_emb.head_size, rotary_dim)
        self.assertNotIsInstance(attn.indexer_rotary_emb, MRotaryEmbedding)
        self.assertEqual(attn.rotary_emb.head_size, text.head_dim)

    def test_the_layer_hands_the_backend_what_it_needs(self):
        """Run the layer against a backend that records its kwargs. A QSA
        backend gets the indexer's projections plus the indexer and its rotary,
        which it runs per rank -- also when it sits inside the hybrid wrapper,
        as Flash-Next's does. Any other backend gets none of them.

        Under mrope the layer is handed three rows of positions, and the
        indexer's plain rotary must read the scalar ones instead, which only a
        real projection through the real call would notice."""
        cfg = _config(mrope=True)
        text = cfg.text_config
        mesh = _mesh()
        with jax.set_mesh(mesh):
            attn = Qwen4ExpAttention(cfg, mesh, layer_id=INTERVAL - 1)
        tokens = 4
        width = text.num_attention_heads * text.head_dim

        def run(full_slot, *, wrapped=False):
            seen = {}

            def backend(q, k, v, layer, forward_batch, pool, **kwargs):
                seen.update(kwargs)
                return jnp.zeros((tokens, width)), "kv"

            if wrapped:
                # The hybrid wrapper routes full-attention layers to full_attn_backend.
                backend.full_attn_backend = types.SimpleNamespace(full_slot=full_slot)
            else:
                backend.full_slot = full_slot
            forward_batch = types.SimpleNamespace(
                attn_backend=backend, positions=jnp.arange(tokens, dtype=jnp.int32)
            )
            mrope_positions = jnp.tile(jnp.arange(tokens, dtype=jnp.int32), (3, 1))
            with jax.set_mesh(mesh):
                attn(mrope_positions, jnp.zeros((tokens, text.hidden_size)), forward_batch, None)
            return seen

        seen = run({attn.layer_id: 0})
        self.assertEqual(sorted(seen), ["indexer", "indexer_k", "indexer_q", "indexer_rotary_emb"])
        self.assertIs(seen["indexer"], attn.indexer)
        self.assertIs(seen["indexer_rotary_emb"], attn.indexer_rotary_emb)
        self.assertEqual(
            seen["indexer_q"].shape, (tokens, text.indexer_n_heads, text.indexer_head_dim)
        )
        self.assertEqual(seen["indexer_k"].shape, (tokens, text.indexer_head_dim))

        self.assertEqual(sorted(run({attn.layer_id: 0}, wrapped=True)), sorted(seen))
        self.assertEqual(run(None), {})


class TestPoolUpdates(CustomTestCase):
    def test_the_kv_pool_update_splits_qsa_state(self):
        """A QSA backend returns each full layer's KV cache bundled with its
        compressed indexer cache and ring, and the pool takes them back as one
        (kv, compressed, ring) tuple; any other backend's caches go back as a
        plain list. The layers are stand-ins returning labelled states, so
        this checks the bookkeeping, not the arithmetic."""
        cfg = _config()
        mesh = _mesh()
        model = _model(cfg, mesh)
        full = cfg.text_config.full_attention_layer_ids
        tokens = 3

        def run(qsa):
            def layer_call(self, positions, hidden, forward_batch, pools, dispatch_info=None):
                i = self.layer_id
                if not self.is_full_attn:
                    return hidden, (f"rec{i}", [f"conv{i}"]), None, None
                state = QSAFusedCache(f"kv{i}", f"compressed{i}", f"ring{i}") if qsa else f"kv{i}"
                return hidden, state, None, None

            forward_batch = types.SimpleNamespace(
                forward_mode=types.SimpleNamespace(
                    is_extend_or_draft_extend_or_mixed=lambda: False
                ),
                input_ids=jnp.zeros((tokens,), jnp.int32),
                mrope_positions=None,
                positions=jnp.arange(tokens, dtype=jnp.int32),
                expert_location_metadata=None,
            )
            mixer = type(model.hyper_connection_mixer)
            with (
                mock.patch.object(Qwen4ExpDecoderLayer, "__call__", layer_call),
                mock.patch.object(mixer, "mix", lambda self, hidden: (hidden, None)),
                jax.set_mesh(mesh),
            ):
                return model(forward_batch, None)[1]

        self.assertEqual(
            run(qsa=True),
            (
                [f"kv{i}" for i in full],
                [f"compressed{i}" for i in full],
                [f"ring{i}" for i in full],
            ),
        )
        self.assertEqual(run(qsa=False), [f"kv{i}" for i in full])


# The released config.json, reduced to what differs from Qwen4ExpTextConfig's
# defaults (which are the released backbone) and matters for construction.
RELEASED_TEXT = dict(
    num_experts=512,
    num_experts_per_tok=10,
    moe_intermediate_size=640,
    shared_expert_intermediate_size=640,
    indexer_budget=2048,
    indexer_compress_ratio=4,
    indexer_head_dim=128,
    indexer_n_heads=4,
    indexer_kv_heads=1,
    ple_layer_ids=[2],
    rope_parameters=dict(
        rope_type="default",
        mrope_section=[11, 11, 10],
        mrope_interleaved=True,
        rope_theta=10000000,
        partial_rotary_factor=0.25,
    ),
)
# Carried along as a plain dict and never read: the model is text-only.
RELEASED_VISION = dict(
    depth=27, hidden_size=1152, num_heads=16, patch_size=16, spatial_merge_size=2
)


class TestReleasedConfig(CustomTestCase):
    def test_the_released_config_builds_text_only(self):
        """The released dimensions reach paths the small configs do not: the
        vision sub-config, mRoPE, 512 experts at an intermediate size that is
        not a multiple of 512, and an untied head over the full vocabulary.
        Built abstractly, so nothing is allocated."""
        text = dict(RELEASED_TEXT)
        if not importlib.util.find_spec("sgl_jax.srt.layers.ngram_embedding"):
            text["ple_layer_ids"] = []  # the N-gram module is not in the tree yet
        cfg = Qwen4ExpConfig(text_config=text, vision_config=RELEASED_VISION)
        mesh = _mesh()
        with jax.set_mesh(mesh):
            model = nnx.eval_shape(lambda: Qwen4ExpForConditionalGeneration(cfg, mesh))

        self.assertEqual(len(model.language_model.model.layers), 48)
        self.assertIsNone(model.visual)
        self.assertEqual(model.get_multimodal_encode_funcs(), {})

    def test_the_server_default_moe_backend_resolves_to_the_fused_kernel(self):
        """The MoE block builds FusedEPMoE whatever --moe-backend says, so the
        server default must resolve to fused for this architecture; otherwise
        the fused kernel's batch-size rules are not checked at startup."""
        model = _model(_config(), _mesh())
        self.assertTrue(any(isinstance(m, FusedEPMoE) for _, m in nnx.iter_graph(model)))

        text = dict(RELEASED_TEXT, ple_layer_ids=[])
        cfg = Qwen4ExpConfig(
            text_config=text, architectures=[Qwen4ExpForConditionalGeneration.__name__]
        )
        model_config = ModelConfig("unused", hf_config=cfg, moe_backend="epmoe")
        self.assertEqual(model_config.moe_backend, MoEBackend.FUSED)


def _mapping_head(config):
    from sgl_jax.srt.layers.embeddings import ParallelLMHead

    if config.tie_word_embeddings:
        return None
    return nnx.eval_shape(
        lambda: ParallelLMHead(config.text_config.vocab_size, config.text_config.hidden_size)
    )


class TestWeightMappings(CustomTestCase):
    def test_new_entries_name_parameters_the_model_has(self):
        """The hyper connections and the indexer are what this table adds over
        Qwen3.5's. A target that resolves to nothing loads nothing and raises
        nothing, so check it against the built module tree."""
        cfg = _config()
        model = _model(cfg, _mesh())
        params = {
            ".".join(str(p) for p in path)
            for path, _ in nnx.to_flat_state(nnx.state(model, nnx.Param))
        }

        mappings, _, _ = _create_qwen4_exp_weight_mappings(cfg, _mapping_head(cfg))
        prefix = "language_model.model."
        added = [
            m.target_path
            for k, m in mappings.items()
            if "hyper_connection" in k or ".indexer." in k
        ]
        self.assertTrue(added)
        for target in added:
            self.assertTrue(target.startswith(prefix), target)
            self.assertIn(target[len(prefix) :], params, f"{target} names no parameter")

    def test_without_the_ngram_layer_its_tensors_are_skipped(self):
        """With no layer building the N-gram module, its tensors are still in
        the checkpoint, and the load summary rejects any key it cannot place."""
        cfg = _config()
        mappings, visual_skip, mtp_skip = _create_qwen4_exp_weight_mappings(cfg, _mapping_head(cfg))
        weight_info = dict.fromkeys(mappings, [])
        weight_info[f"model.language_model.layers.{PLE_LAYER_1BASED - 1}.ple.key_proj.weight"] = []
        Qwen4ExpForConditionalGeneration._log_load_summary(
            mappings, weight_info, visual_skip, mtp_skip
        )

    def test_the_absorbed_norms_are_gone_and_no_added_entry_shares_a_target(self):
        """Leaving Qwen3.5's norms in would point at modules that no longer
        exist; an added entry sharing a target would silently load it twice.
        Qwen3.5's own table points several sources at one fused target on
        purpose, and its loader groups them, so only the added entries are
        checked."""
        cfg = _config()
        head = _mapping_head(cfg)
        mappings, _, _ = _create_qwen4_exp_weight_mappings(cfg, head)
        base, _, _ = _create_qwen3_5_weight_mappings(cfg, head)

        self.assertNotIn("model.language_model.norm.weight", mappings)
        for key in mappings:
            self.assertNotIn("input_layernorm", key)
            self.assertNotIn("post_attention_layernorm", key)

        def targets(entries):
            flat = []
            for mapping in entries:
                target = mapping.target_path
                flat.extend(target if isinstance(target, list) else [target])
            return flat

        added = targets(m for k, m in mappings.items() if k not in base)
        self.assertTrue(added)
        self.assertEqual(len(set(added)), len(added))
        self.assertEqual(set(added) & set(targets(base.values())), set())


if __name__ == "__main__":
    unittest.main()
