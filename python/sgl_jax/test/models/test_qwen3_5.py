"""Qwen3.5 (MoE 35B-A3B + dense 27B/2B) CPU component tests.

Runs in the ``unit-test-cpu`` suite (``run_suite.py``). Hub-free, mirroring
``test_mimo_v2_nextn`` / ``test_kimi_k25_weight_mapping``: configs are built
in-process from the real ``config.json`` scalars + RoPE block, and weight-mapping
coverage is checked against the checkpoint key set *reconstructed* from per-layer
templates snapshotted from the real ``model.safetensors.index.json`` (no
``from_pretrained`` / ``hf_hub_download``, so it never skips on an offline runner).
Dims are shrunk to keep module construction cheap on CPU.

The fused GDN/MoE kernels and numerical accuracy are gated separately by the TPU
smoke (``test/srt/test_qwen3_5_models.py``) + the e2e MMLU-Pro / GPQA evals.

Snapshots derive from Qwen3.5-{35B-A3B, 27B, 2B}; the per-layer block layout is
identical across the family. Re-snapshot from the index if a future revision adds
or renames tensors.

Run:
    python -m pytest python/sgl_jax/test/models/test_qwen3_5.py -q
"""

from __future__ import annotations

import os
import re
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.qwen3_5 import Qwen3_5DenseConfig, Qwen3_5HybridConfig
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

_mesh = create_device_mesh(
    ici_parallelism=[1, 1], dcn_parallelism=[1, 1], devices=[jax.devices()[0]]
)
jax.sharding.set_mesh(_mesh)


class _LinearRecorder:
    """Minimal LinearBase stand-in that records the per-call output target."""

    def __init__(self, output):
        self.output = output
        self.out_shardings = []

    def __call__(self, *args, out_sharding=None, **kwargs):
        del args, kwargs
        self.out_shardings.append(out_sharding)
        output = self.output
        if out_sharding is not None:
            output = jax.sharding.reshard(output, out_sharding)
        return output, None


class _ArrayRecorder:
    """Array-returning stand-in for MLP, router, and expert modules."""

    def __init__(self, output):
        self.output = output
        self.out_shardings = []

    def __call__(self, *args, out_sharding=None, **kwargs):
        del args, kwargs
        self.out_shardings.append(out_sharding)
        output = self.output
        if out_sharding is not None:
            output = jax.sharding.reshard(output, out_sharding)
        return output


class _PairRecorder:
    """Pair-returning stand-in for attention modules."""

    def __init__(self, output, state):
        self.output = output
        self.state = state
        self.out_shardings = []
        self.args = []

    def __call__(self, *args, out_sharding=None, **kwargs):
        del kwargs
        self.args.append(args)
        self.out_shardings.append(out_sharding)
        output = self.output
        if out_sharding is not None:
            output = jax.sharding.reshard(output, out_sharding)
        return output, self.state


class _Identity:
    def __call__(self, value, *args, **kwargs):
        del args, kwargs
        return value


class _RotaryIdentity:
    def __call__(self, positions, q, k):
        del positions
        return q, k


class _TopKRecorder:
    def __init__(self, weights, ids):
        self.weights = weights
        self.ids = ids
        self.routing_shardings = []

    def __call__(self, router_logits, dispatch_info=None, routing_sharding=None):
        del router_logits, dispatch_info
        self.routing_shardings.append(routing_sharding)
        return self.weights, self.ids


class _TokenMaskRecorder:
    def __init__(self, mask):
        self.mask = mask
        self.out_sharding = None

    def get_token_valid_mask(self, num_tokens, *, out_sharding=None):
        if num_tokens != self.mask.shape[0]:
            raise AssertionError(f"Expected {self.mask.shape[0]} tokens, got {num_tokens}.")
        self.out_sharding = out_sharding
        mask = self.mask
        if out_sharding is not None:
            mask = jax.sharding.reshard(mask, out_sharding)
        return mask


# RoPE block as the real config.json ships it (nested ``rope_parameters``, HF 5.x);
# values shrunk but structurally identical so the config class's flatten path runs.
_ROPE_PARAMETERS = {
    "rope_type": "default",
    "mrope_section": [6, 5, 5],  # sums to rotary_dim // 2 = (head_dim * partial) // 2
    "mrope_interleaved": True,
    "rope_theta": 12345,
    "partial_rotary_factor": 0.5,
}


def _make_config(*, num_layers: int, is_moe: bool, tie: bool = False):
    """Build a Qwen3.5 config with the real layout but small dims.

    ``full_attention_interval=4`` (real schedule) and ``num_layers`` drive the
    weight-mapping coverage; the small head/expert dims keep module construction
    cheap. MoE vs dense is keyed off ``num_experts`` (the ``is_moe`` discriminator).
    """
    text = dict(
        vocab_size=256,
        hidden_size=256,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        full_attention_interval=4,
        rms_norm_eps=1e-6,
        rope_parameters=dict(_ROPE_PARAMETERS),
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
    )
    if is_moe:
        text.update(
            num_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=128,
            shared_expert_intermediate_size=128,
        )
        return Qwen3_5HybridConfig(tie_word_embeddings=tie, text_config=text)
    text.update(intermediate_size=128)
    return Qwen3_5DenseConfig(tie_word_embeddings=tie, text_config=text)


# Per-layer HF safetensors key suffixes, snapshotted from the real
# model.safetensors.index.json (verified: the reconstruction below reproduces the
# real text-key set exactly — 693 for 35B-A3B, 851 for 27B, 320 for 2B).
_NORM_KEYS = ("input_layernorm.weight", "post_attention_layernorm.weight")
_GDN_KEYS = (
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.conv1d.weight",
    "linear_attn.A_log",
    "linear_attn.dt_bias",
    "linear_attn.norm.weight",
    "linear_attn.out_proj.weight",
)
_FULL_ATTN_KEYS = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
)
_DENSE_FFN_KEYS = ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")
_MOE_FFN_KEYS = (
    "mlp.gate.weight",
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
    "mlp.shared_expert.gate_proj.weight",
    "mlp.shared_expert.up_proj.weight",
    "mlp.shared_expert.down_proj.weight",
    "mlp.shared_expert_gate.weight",
)


def _expected_ckpt_keys(num_layers: int, is_moe: bool, tie: bool):
    """Reconstruct the HF source-key set the checkpoint ships (= the keys the
    weight mapping must cover). Independent of the production mapping code, so
    comparing the two is a real bidirectional coverage check."""
    keys = {
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
    }
    if not tie:  # tied variants reuse the embedding; no lm_head.weight on disk
        keys.add("lm_head.weight")
    ffn = _MOE_FFN_KEYS if is_moe else _DENSE_FFN_KEYS
    for i in range(num_layers):
        attn = _FULL_ATTN_KEYS if (i + 1) % 4 == 0 else _GDN_KEYS
        for suffix in _NORM_KEYS + attn + ffn:
            keys.add(f"model.language_model.layers.{i}.{suffix}")
    return keys


class TestQwen3_5(unittest.TestCase):
    """MoE 35B-A3B backbone: config plumbing, module wiring/shapes, mapping coverage."""

    @classmethod
    def setUpClass(cls):
        cls.mesh = _mesh
        cls.cfg = _make_config(num_layers=4, is_moe=True)

    # --- config: nested rope_parameters flattened to rope_scaling/theta/partial ---
    def test_config_flattens_rope_parameters(self):
        tc = self.cfg.text_config
        self.assertEqual(type(self.cfg).__name__, "Qwen3_5HybridConfig")
        self.assertEqual(self.cfg.model_type, "qwen3_5_moe")
        # Subset-only: newer transformers versions inject extra fields
        # (rope_theta, partial_rotary_factor) into rope_scaling. Assert just the
        # fields the model actually consumes.
        keys = ("rope_type", "mrope_section", "mrope_interleaved")
        self.assertEqual(
            {k: tc.rope_scaling[k] for k in keys},
            {
                "rope_type": "default",
                "mrope_section": [6, 5, 5],
                "mrope_interleaved": True,
            },
        )
        self.assertEqual(tc.rope_theta, 12345)
        self.assertEqual(tc.partial_rotary_factor, 0.5)
        self.assertFalse(self.cfg.tie_word_embeddings)

    # --- config: is_moe discriminator + FFN field exclusivity ---
    def test_is_moe_discriminator(self):
        moe = _make_config(num_layers=4, is_moe=True).text_config
        self.assertTrue(moe.is_moe)
        self.assertIsNotNone(moe.num_experts)
        self.assertIsNone(moe.intermediate_size)

        dense = _make_config(num_layers=4, is_moe=False).text_config
        self.assertFalse(dense.is_moe)
        self.assertIsNone(dense.num_experts)
        self.assertIsNone(dense.num_experts_per_tok)
        self.assertIsNotNone(dense.intermediate_size)

    # --- layer schedule derives from full_attention_interval ---
    def test_layer_types_matches_full_attention_interval(self):
        tc = self.cfg.text_config
        interval = int(tc.full_attention_interval)
        derived = [
            "full_attention" if ((i + 1) % interval == 0) else "linear_attention"
            for i in range(tc.num_hidden_layers)
        ]
        self.assertEqual(derived, list(tc.layer_types))

    # --- decoder layer picks full vs GDN attention from layer_types ---
    def test_decoder_layer_attention_type_from_layer_types(self):
        from sgl_jax.srt.models.qwen3_5 import (
            Qwen3_5Attention,
            Qwen3_5DecoderLayer,
            Qwen3_5GatedDeltaNet,
        )

        full_ids = self.cfg.text_config.full_attention_layer_ids
        self.assertIn(3, full_ids)  # interval=4 -> full at 3, 7, ...
        self.assertNotIn(0, full_ids)

        full_layer = Qwen3_5DecoderLayer(self.cfg, self.mesh, layer_id=full_ids[0])
        gdn_layer = Qwen3_5DecoderLayer(self.cfg, self.mesh, layer_id=0)
        self.assertTrue(full_layer.is_full_attn)
        self.assertIsInstance(full_layer.self_attn, Qwen3_5Attention)
        self.assertFalse(gdn_layer.is_full_attn)
        self.assertIsInstance(gdn_layer.self_attn, Qwen3_5GatedDeltaNet)

    # --- full-attn output-gate layout: q_proj emits [q | gate] per head ---
    def test_q_proj_output_gate_layout(self):
        from sgl_jax.srt.models.qwen3_5 import Qwen3_5Attention

        attn = Qwen3_5Attention(self.cfg, self.mesh, layer_id=3)
        self.assertTrue(attn.attn_output_gate)
        T = 4
        x = jnp.zeros((T, self.cfg.text_config.hidden_size), dtype=jnp.bfloat16)
        q_raw, _ = attn.q_proj(x)
        self.assertEqual(q_raw.shape, (T, attn.num_heads * 2 * attn.head_dim))
        q_gate = q_raw.reshape(T, attn.num_heads, 2 * attn.head_dim)
        self.assertEqual(q_gate[..., : attn.head_dim].shape, (T, attn.num_heads, attn.head_dim))
        self.assertEqual(q_gate[..., attn.head_dim :].shape, (T, attn.num_heads, attn.head_dim))

    # --- partial M-RoPE wiring: module reads mrope_section/rotary_dim from config ---
    def test_rope_wiring(self):
        from sgl_jax.srt.layers.embeddings import MRotaryEmbedding
        from sgl_jax.srt.models.qwen3_5 import Qwen3_5Attention

        attn = Qwen3_5Attention(self.cfg, self.mesh, layer_id=3)
        tc = self.cfg.text_config
        self.assertIsInstance(attn.rotary_emb, MRotaryEmbedding)
        self.assertEqual(list(attn.rotary_emb.mrope_section), [6, 5, 5])
        self.assertEqual(attn.rotary_emb.rotary_dim, int(tc.head_dim * tc.partial_rotary_factor))
        self.assertEqual(attn.rotary_emb.base, tc.rope_theta)

    # --- GDN fused-projection shapes ---
    def test_gdn_projection_shapes(self):
        from sgl_jax.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet

        tc = self.cfg.text_config
        gdn = Qwen3_5GatedDeltaNet(self.cfg, self.mesh, layer_id=0)
        key_dim = tc.linear_num_key_heads * tc.linear_key_head_dim
        value_dim = tc.linear_num_value_heads * tc.linear_value_head_dim
        self.assertEqual(
            gdn.in_proj_qkvz.weight.value.shape,
            (tc.hidden_size, 2 * key_dim + 2 * value_dim),
        )
        self.assertEqual(
            gdn.in_proj_ba.weight.value.shape,
            (tc.hidden_size, 2 * tc.linear_num_value_heads),
        )
        self.assertEqual(
            gdn.conv1d.weight.value.shape,
            (2 * key_dim + value_dim, tc.linear_conv_kernel_dim),
        )
        self.assertEqual(gdn.A_log.value.shape, (tc.linear_num_value_heads,))
        self.assertEqual(gdn.dt_bias.value.shape, (tc.linear_num_value_heads,))

    # --- GDN output gate is norm-before-gate SILU (not sigmoid) ---
    def test_gdn_output_norm_gate_is_silu(self):
        """GDN output gate is RMSNorm(core) * silu(z) — silu, NOT sigmoid.

        A silu->sigmoid swap passes shape tests but tanks accuracy. CPU-safe.
        """
        from sgl_jax.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet

        tc = self.cfg.text_config
        gdn = Qwen3_5GatedDeltaNet(self.cfg, self.mesh, layer_id=0)
        n_v, d_v = tc.linear_num_value_heads, tc.linear_value_head_dim
        value_dim = n_v * d_v
        T = 4
        core = jax.random.normal(jax.random.key(0), (T, n_v, d_v), dtype=jnp.bfloat16)
        z = jax.random.normal(jax.random.key(1), (T, value_dim), dtype=jnp.bfloat16)

        got = gdn._norm_gate(core, z)
        normed = gdn.norm(core).reshape(T, value_dim)
        expected_silu = normed * jax.nn.silu(z)
        expected_sigmoid = normed * jax.nn.sigmoid(z)

        self.assertEqual(got.shape, (T, value_dim))
        np.testing.assert_allclose(
            np.asarray(got, dtype=np.float32),
            np.asarray(expected_silu, dtype=np.float32),
            rtol=2e-2,
            atol=1e-2,
        )
        # silu and sigmoid must differ here, else the match above is vacuous.
        self.assertFalse(
            np.allclose(
                np.asarray(expected_silu, dtype=np.float32),
                np.asarray(expected_sigmoid, dtype=np.float32),
                rtol=2e-2,
                atol=1e-2,
            )
        )

    # --- MoE block wiring + sigmoid-gated shared expert ---
    def test_moe_block_shared_expert_gate(self):
        from sgl_jax.srt.models.qwen3_5 import Qwen3_5MoeBlock

        tc = self.cfg.text_config
        block = Qwen3_5MoeBlock(self.cfg, self.mesh, layer_id=0)
        self.assertIs(block.topk.mesh, self.mesh)
        self.assertEqual(
            block.experts.w1.value.shape,
            (tc.num_experts, tc.hidden_size, tc.moe_intermediate_size),
        )
        self.assertEqual(
            block.experts.w3.value.shape,
            (tc.num_experts, tc.hidden_size, tc.moe_intermediate_size),
        )
        self.assertEqual(
            block.experts.w2.value.shape,
            (tc.num_experts, tc.moe_intermediate_size, tc.hidden_size),
        )
        T = 4
        x = jax.random.normal(jax.random.key(0), (T, tc.hidden_size), dtype=jnp.bfloat16)
        gate_logit, _ = block.shared_expert_gate(x)
        self.assertEqual(gate_logit.shape, (T, 1))
        shared = block.shared_experts(x)
        self.assertEqual(shared.shape, (T, tc.hidden_size))
        gated = jax.nn.sigmoid(gate_logit) * shared
        self.assertEqual(gated.shape, (T, tc.hidden_size))

    # --- weight-mapping coverage vs the reconstructed 35B-A3B (MoE) key set ---
    def test_weight_mapping_covers_moe_keys(self):
        from sgl_jax.srt.models.qwen3_5 import _create_qwen3_5_weight_mappings

        cfg = _make_config(num_layers=40, is_moe=True)  # 35B-A3B layer count
        mapping, _, _ = _create_qwen3_5_weight_mappings(cfg)
        expected = _expected_ckpt_keys(num_layers=40, is_moe=True, tie=False)
        self.assertEqual(len(expected), 693)  # 692 text keys + lm_head.weight
        self.assertEqual(set(mapping), expected)

    # --- vision tower + MTP head are skipped by prefix, LM keys are not ---
    def test_visual_and_mtp_keys_are_skipped(self):
        from sgl_jax.srt.models.qwen3_5 import _create_qwen3_5_weight_mappings

        _, visual_skip, mtp_skip = _create_qwen3_5_weight_mappings(self.cfg)
        self.assertTrue(
            any(re.match(p, "model.visual.blocks.0.attn.qkv.weight") for p in visual_skip)
        )
        self.assertTrue(any(re.match(p, "mtp.layers.0.input_layernorm.weight") for p in mtp_skip))
        lm_key = "model.language_model.layers.0.input_layernorm.weight"
        self.assertFalse(any(re.match(p, lm_key) for p in visual_skip + mtp_skip))

    # --- expert capture builds the real capturer for MoE (counterpart to the
    #     dense router-less guard in TestQwen3_5Dense) ---
    def test_routed_experts_capturer_builds_for_moe(self):
        from sgl_jax.srt.layers.routed_experts_capturer import (
            RoutedExpertsCapturer,
            _RoutedExpertsCapturerReal,
        )

        # Qwen3.5 MoE keeps num_experts / top-k under text_config (the nested-config
        # path the helper must read). Enabling capture builds the real capturer and
        # sizes the per-layer host buffer from the resolved top-k.
        tc = self.cfg.text_config
        moe_mc = SimpleNamespace(hf_config=self.cfg, hf_text_config=tc)
        cap = RoutedExpertsCapturer.create(
            mesh=self.mesh,
            enable=True,
            model_config=moe_mc,
            num_tokens=1,
            max_padding=1,
            ep_size=1,
        )
        self.assertIsInstance(cap, _RoutedExpertsCapturerReal)
        self.assertEqual(cap.num_experts_per_tok, tc.num_experts_per_tok)
        self.assertEqual(cap.host_buffer.shape, (tc.num_hidden_layers, 1, tc.num_experts_per_tok))

        # A MoE model that instead exposes the count on the ROOT config under an
        # alias (n_routed_experts / num_local_experts) must still build the real
        # capturer via the helper's root fallback, not trip the router-less guard.
        # (Qwen3.5 itself uses text_config, above.)
        for total_alias in ("n_routed_experts", "num_local_experts"):
            root_mc = SimpleNamespace(
                hf_text_config=SimpleNamespace(num_hidden_layers=2),
                hf_config=SimpleNamespace(**{total_alias: 64, "num_experts_per_tok": 4}),
            )
            cap = RoutedExpertsCapturer.create(
                mesh=self.mesh,
                enable=True,
                model_config=root_mc,
                num_tokens=1,
                max_padding=1,
                ep_size=1,
            )
            self.assertIsInstance(cap, _RoutedExpertsCapturerReal)
            self.assertEqual(cap.host_buffer.shape, (2, 1, 4))


class TestQwen3_5SequenceParallelWiring(unittest.TestCase):
    """Hub-free call-contract tests for Qwen3.5 sequence-parallel wiring.

    The recorders deliberately stop before RadixAttention, GDN, and fused-MoE
    kernels. These tests answer only whether the root flag selects a target and
    whether that exact target reaches every row-parallel consumer.
    """

    @classmethod
    def setUpClass(cls):
        cls.mesh = _mesh

    def _target(self):
        return NamedSharding(self.mesh, P(("data", "tensor"), None))

    def test_decoder_reads_root_sequence_parallel_flag_default_and_enabled(self):
        from sgl_jax.srt.models.qwen3_5 import Qwen3_5DecoderLayer

        default_cfg = _make_config(num_layers=4, is_moe=False)
        default_layer = Qwen3_5DecoderLayer(default_cfg, self.mesh, layer_id=3)
        self.assertFalse(default_layer.enable_sequence_parallel)

        enabled_cfg = _make_config(num_layers=4, is_moe=False)
        enabled_cfg.enable_sequence_parallel = True
        enabled_layer = Qwen3_5DecoderLayer(enabled_cfg, self.mesh, layer_id=3)
        self.assertTrue(enabled_layer.enable_sequence_parallel)

    def test_qwen2_moe_mlp_threads_target_to_down_projection(self):
        from sgl_jax.srt.models.qwen2_moe import Qwen2MoeMLP

        tokens, hidden, intermediate = 4, 32, 64
        mlp = Qwen2MoeMLP(hidden, intermediate, self.mesh)
        down_proj = _LinearRecorder(jnp.zeros((tokens, hidden), dtype=jnp.bfloat16))
        mlp.down_proj = down_proj
        hidden_states = jnp.zeros((tokens, hidden), dtype=jnp.bfloat16)
        target = self._target()

        output = mlp(hidden_states, out_sharding=target)
        self.assertIs(down_proj.out_shardings[-1], target)
        self.assertEqual(output.sharding.spec, target.spec)

        mlp(hidden_states)
        self.assertIsNone(down_proj.out_shardings[-1])

    def test_full_attention_threads_target_to_o_projection(self):
        from sgl_jax.srt.models.qwen3_5 import Qwen3_5Attention

        cfg = _make_config(num_layers=4, is_moe=False)
        tc = cfg.text_config
        tokens = 4
        attn = Qwen3_5Attention(cfg, self.mesh, layer_id=3)
        attn.q_proj = _LinearRecorder(
            jnp.zeros(
                (tokens, attn.num_heads * 2 * attn.head_dim),
                dtype=jnp.bfloat16,
            )
        )
        attn.k_proj = _LinearRecorder(
            jnp.zeros((tokens, attn.num_kv_heads * attn.head_dim), dtype=jnp.bfloat16)
        )
        attn.v_proj = _LinearRecorder(
            jnp.zeros((tokens, attn.num_kv_heads * attn.head_dim), dtype=jnp.bfloat16)
        )
        attn.q_norm = _Identity()
        attn.k_norm = _Identity()
        attn.rotary_emb = _RotaryIdentity()
        attn.attn = _PairRecorder(
            jnp.zeros((tokens, attn.num_heads * attn.head_dim), dtype=jnp.bfloat16),
            "kv-state",
        )
        o_proj = _LinearRecorder(jnp.zeros((tokens, tc.hidden_size), dtype=jnp.bfloat16))
        attn.o_proj = o_proj
        target = self._target()

        output, state = attn(
            jnp.arange(tokens),
            jnp.zeros((tokens, tc.hidden_size), dtype=jnp.bfloat16),
            object(),
            object(),
            out_sharding=target,
        )

        self.assertIs(o_proj.out_shardings[-1], target)
        self.assertEqual(output.sharding.spec, target.spec)
        self.assertEqual(state, "kv-state")

    def test_gdn_threads_target_to_out_projection(self):
        from sgl_jax.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet

        cfg = _make_config(num_layers=4, is_moe=False)
        tc = cfg.text_config
        tokens = 4
        gdn = Qwen3_5GatedDeltaNet(cfg, self.mesh, layer_id=0)
        qkvz_width = 2 * gdn.key_dim + 2 * gdn.value_dim
        gdn.in_proj_qkvz = _LinearRecorder(jnp.zeros((tokens, qkvz_width), dtype=jnp.bfloat16))
        gdn.in_proj_ba = _LinearRecorder(
            jnp.zeros((tokens, 2 * gdn.num_v_heads), dtype=jnp.bfloat16)
        )
        gdn._shard_dt = _Identity()
        gdn._norm_gate = _Identity()
        gdn.self_attn = _PairRecorder(
            jnp.zeros((tokens, gdn.value_dim), dtype=jnp.bfloat16),
            "recurrent-state",
        )
        out_proj = _LinearRecorder(jnp.zeros((tokens, tc.hidden_size), dtype=jnp.bfloat16))
        gdn.out_proj = out_proj
        target = self._target()

        output, state = gdn(
            jnp.arange(tokens),
            jnp.zeros((tokens, tc.hidden_size), dtype=jnp.bfloat16),
            object(),
            object(),
            out_sharding=target,
        )

        self.assertIs(out_proj.out_shardings[-1], target)
        self.assertEqual(output.sharding.spec, target.spec)
        self.assertEqual(state, "recurrent-state")

    def test_moe_threads_target_to_shared_routed_and_mask_paths(self):
        from sgl_jax.srt.models.qwen3_5 import Qwen3_5MoeBlock

        cfg = _make_config(num_layers=4, is_moe=True)
        tc = cfg.text_config
        tokens = 4
        hidden_states = jnp.zeros((tokens, tc.hidden_size), dtype=jnp.bfloat16)
        block = Qwen3_5MoeBlock(cfg, self.mesh, layer_id=0)
        shared = _ArrayRecorder(jnp.zeros_like(hidden_states))
        shared_gate = _LinearRecorder(jnp.zeros((tokens, 1), dtype=jnp.bfloat16))
        experts = _ArrayRecorder(jnp.zeros_like(hidden_states))
        self.assertIs(block.topk.mesh, self.mesh)
        block.shared_experts = shared
        block.shared_expert_gate = shared_gate
        block.moe_gate = _ArrayRecorder(jnp.zeros((tokens, tc.num_experts), dtype=jnp.float32))
        raw_ids = jnp.arange(tokens * tc.num_experts_per_tok, dtype=jnp.int32).reshape(
            tokens, tc.num_experts_per_tok
        )
        topk = _TopKRecorder(
            jnp.ones((tokens, tc.num_experts_per_tok), dtype=jnp.bfloat16),
            raw_ids,
        )
        block.topk = topk
        block.experts = experts
        forward_batch = _TokenMaskRecorder(jnp.array([True, False, True, False]))
        target = self._target()

        output, topk_ids = block(
            hidden_states,
            forward_batch,
            dispatch_info=object(),
            out_sharding=target,
        )

        self.assertIs(shared.out_shardings[-1], target)
        self.assertIs(shared_gate.out_shardings[-1], target)
        self.assertIs(topk.routing_shardings[-1], target)
        self.assertIs(experts.out_shardings[-1], target)
        self.assertEqual(output.sharding.spec, target.spec)
        self.assertIsNotNone(forward_batch.out_sharding)
        self.assertEqual(forward_batch.out_sharding.spec, P(("data", "tensor")))
        self.assertTrue(
            all(axis is None for axis in topk_ids.sharding.spec),
            msg=f"topk_ids must remain replicated, got {topk_ids.sharding.spec}",
        )
        expected_ids = np.asarray(raw_ids).copy()
        expected_ids[[1, 3], :] = -1
        np.testing.assert_array_equal(np.asarray(topk_ids), expected_ids)

    def test_decoder_threads_one_target_to_attention_mlp_and_residual(self):
        from sgl_jax.srt.models.qwen3_5 import Qwen3_5DecoderLayer

        cfg = _make_config(num_layers=4, is_moe=False)
        cfg.enable_sequence_parallel = True
        tc = cfg.text_config
        tokens = 4
        target = self._target()
        memory_pools = SimpleNamespace(
            token_to_kv_pool=object(),
            recurrent_state_pool=object(),
        )
        cases = (
            (3, memory_pools.token_to_kv_pool, "full-attention-state"),
            (0, memory_pools.recurrent_state_pool, "gdn-state"),
        )

        for layer_id, expected_pool, expected_state in cases:
            with self.subTest(layer_id=layer_id):
                hidden_states = jnp.zeros((tokens, tc.hidden_size), dtype=jnp.bfloat16)
                layer = Qwen3_5DecoderLayer(cfg, self.mesh, layer_id=layer_id)
                layer.input_layernorm = _Identity()
                layer.post_attention_layernorm = _Identity()
                attention = _PairRecorder(jnp.zeros_like(hidden_states), expected_state)
                mlp = _ArrayRecorder(jnp.zeros_like(hidden_states))
                layer.self_attn = attention
                layer.mlp = mlp

                with mock.patch(
                    "sgl_jax.srt.models.qwen3_5.make_reduce_sharding",
                    return_value=target,
                ) as make_target:
                    output, residual, state, topk_ids = layer(
                        jnp.arange(tokens),
                        hidden_states,
                        object(),
                        memory_pools,
                    )

                args, kwargs = make_target.call_args
                self.assertIs(args[0], hidden_states)
                self.assertIs(args[1], self.mesh)
                self.assertEqual(kwargs, {"enable_sp": True})
                self.assertIs(attention.args[-1][3], expected_pool)
                self.assertIs(attention.out_shardings[-1], target)
                self.assertIs(mlp.out_shardings[-1], target)
                self.assertEqual(output.sharding.spec, target.spec)
                self.assertEqual(residual.sharding.spec, target.spec)
                self.assertEqual(state, expected_state)
                self.assertIsNone(topk_ids)


class TestQwen3_5Dense(unittest.TestCase):
    """Dense 27B/2B: SwiGLU FFN dispatch, mapping coverage, tied-embedding variant."""

    @classmethod
    def setUpClass(cls):
        cls.mesh = _mesh
        cls.cfg = _make_config(num_layers=4, is_moe=False)

    # --- dense config: router-less, model_type/class, FFN field ---
    def test_dense_config(self):
        tc = self.cfg.text_config
        self.assertEqual(type(self.cfg).__name__, "Qwen3_5DenseConfig")
        self.assertEqual(self.cfg.model_type, "qwen3_5")
        self.assertFalse(tc.is_moe)
        self.assertIsNone(tc.num_experts)
        self.assertIsNone(tc.num_experts_per_tok)
        self.assertIsNotNone(tc.intermediate_size)
        self.assertFalse(self.cfg.tie_word_embeddings)

    # --- decoder layer dispatches the dense SwiGLU MLP ---
    def test_decoder_dispatches_dense_mlp(self):
        from sgl_jax.srt.models.qwen2_moe import Qwen2MoeMLP
        from sgl_jax.srt.models.qwen3_5 import Qwen3_5DecoderLayer

        layer = Qwen3_5DecoderLayer(self.cfg, self.mesh, layer_id=0)  # GDN layer
        self.assertFalse(layer.is_moe)
        self.assertIsInstance(layer.mlp, Qwen2MoeMLP)

    # --- dense MLP forward (the only dense forward exercised on CPU) ---
    def test_dense_mlp_shape(self):
        from sgl_jax.srt.models.qwen2_moe import Qwen2MoeMLP

        hidden, inter, T = 32, 64, 4
        mlp = Qwen2MoeMLP(hidden_size=hidden, intermediate_size=inter, mesh=self.mesh)
        x = jax.random.normal(jax.random.key(0), (T, hidden), dtype=jnp.bfloat16)
        out = mlp(x)
        self.assertEqual(out.shape, (T, hidden))

    # --- weight mapping: reconstructed 27B (dense) key set; no expert keys ---
    def test_weight_mapping_covers_dense_keys(self):
        from sgl_jax.srt.models.qwen3_5 import _create_qwen3_5_weight_mappings

        cfg = _make_config(num_layers=64, is_moe=False)  # 27B layer count
        mapping, _, _ = _create_qwen3_5_weight_mappings(cfg)
        expected = _expected_ckpt_keys(num_layers=64, is_moe=False, tie=False)
        self.assertEqual(len(expected), 851)  # 850 text keys + lm_head.weight
        self.assertEqual(set(mapping), expected)
        self.assertIn("lm_head.weight", mapping)  # 27B is untied
        self.assertFalse(
            any("experts" in k or "shared_expert" in k for k in mapping),
            "dense mapping must not reference MoE expert keys",
        )

    # --- tied variant (2B) omits the lm_head.weight mapping ---
    def test_tied_variant_omits_lm_head_mapping(self):
        from sgl_jax.srt.models.qwen3_5 import _create_qwen3_5_weight_mappings

        cfg = _make_config(num_layers=24, is_moe=False, tie=True)  # 2B layer count
        self.assertTrue(cfg.tie_word_embeddings)
        mapping, _, _ = _create_qwen3_5_weight_mappings(cfg)
        self.assertNotIn("lm_head.weight", mapping)
        self.assertEqual(set(mapping), _expected_ckpt_keys(num_layers=24, is_moe=False, tie=True))

    # --- router-less capturer guard (shared infra; dense regression scenario) ---
    def test_dense_routed_experts_guard(self):
        from sgl_jax.srt.layers.routed_experts_capturer import (
            RoutedExpertsCapturer,
            _RoutedExpertsCapturerNoop,
        )

        router_less = SimpleNamespace(
            hf_text_config=SimpleNamespace(
                num_experts=None, num_experts_per_tok=None, num_hidden_layers=4
            )
        )
        # Flags off -> noop capturer, no allocation.
        cap = RoutedExpertsCapturer.create(
            mesh=self.mesh,
            enable=False,
            model_config=router_less,
            num_tokens=1,
            max_padding=1,
            ep_size=1,
        )
        self.assertIsInstance(cap, _RoutedExpertsCapturerNoop)
        # Any capture flag on -> clear error before NumPy shape allocation.
        for flag in ("enable", "enable_balance_debug", "enable_dist_recorder"):
            kwargs = dict(enable=False, enable_balance_debug=False, enable_dist_recorder=False)
            kwargs[flag] = True
            with self.assertRaises(ValueError):
                RoutedExpertsCapturer.create(
                    mesh=self.mesh,
                    model_config=router_less,
                    num_tokens=1,
                    max_padding=1,
                    ep_size=1,
                    **kwargs,
                )


if __name__ == "__main__":
    unittest.main()
