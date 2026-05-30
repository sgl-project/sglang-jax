"""Qwen3.5-35B-A3B consolidated CPU test (RFC §4.2).

Uses the local config-only fixture (config.json + safetensors index, no blobs).
CPU-only: validates config aliasing, layer schedule, module wiring/shapes,
partial M-RoPE construction, and weight-mapping coverage against the real
safetensors key set. Full numerical correctness is validated by the TPU smoke
(``test/srt/test_qwen3_5_models.py``), not here — the fused MoE / GDN kernels
do not run on CPU.
"""

from __future__ import annotations

import json
import os
import re
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
from transformers import AutoConfig

import sgl_jax.srt.hf_transformers_utils  # noqa: F401  (registers Qwen3_5HybridConfig)
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

CKPT_FIXTURE = os.environ.get(
    "QWEN3_5_FIXTURE",
    "/Users/yuhao/Desktop/Dev/Primatrix/_Delivery/202605A_sgl_jax_qwen3p5/"
    "Qwen3p5_all_model_configs/35B_A3B/",
)

_mesh = create_device_mesh(
    ici_parallelism=[1, 1], dcn_parallelism=[1, 1], devices=[jax.devices()[0]]
)
jax.sharding.set_mesh(_mesh)


class TestQwen3_5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = AutoConfig.from_pretrained(CKPT_FIXTURE, trust_remote_code=False)
        cls.mesh = _mesh

    # --- 1: config RoPE alias ---
    def test_config_alias_surfaces_rope_fields(self):
        tc = self.cfg.text_config
        self.assertEqual(type(self.cfg).__name__, "Qwen3_5HybridConfig")
        self.assertEqual(self.cfg.model_type, "qwen3_5_moe")
        self.assertEqual(
            tc.rope_scaling,
            {"rope_type": "default", "mrope_section": [11, 11, 10], "mrope_interleaved": True},
        )
        self.assertEqual(tc.rope_theta, 10000000)
        self.assertEqual(tc.partial_rotary_factor, 0.25)
        self.assertFalse(self.cfg.tie_word_embeddings)
        self.assertEqual(tc.head_dim, 256)

    # --- 2: layer schedule ---
    def test_layer_types_matches_full_attention_interval(self):
        tc = self.cfg.text_config
        interval = int(tc.full_attention_interval)
        derived = [
            "full_attention" if ((i + 1) % interval == 0) else "linear_attention"
            for i in range(tc.num_hidden_layers)
        ]
        self.assertEqual(derived, list(tc.layer_types))

    # --- 3: full-attn output-gate layout ---
    def test_q_proj_output_gate_layout(self):
        from sgl_jax.srt.models.qwen3_5 import Qwen3_5Attention

        attn = Qwen3_5Attention(self.cfg, self.mesh, layer_id=3)
        self.assertTrue(attn.attn_output_gate)
        T = 4
        x = jnp.zeros((T, self.cfg.text_config.hidden_size), dtype=jnp.bfloat16)
        q_raw, _ = attn.q_proj(x)
        self.assertEqual(q_raw.shape, (T, attn.num_heads * 2 * attn.head_dim))
        # per-head [q | gate] split
        q_gate = q_raw.reshape(T, attn.num_heads, 2 * attn.head_dim)
        self.assertEqual(q_gate[..., : attn.head_dim].shape, (T, attn.num_heads, attn.head_dim))
        self.assertEqual(q_gate[..., attn.head_dim :].shape, (T, attn.num_heads, attn.head_dim))

    # --- 4: partial M-RoPE wiring ---
    def test_rope_wiring(self):
        from sgl_jax.srt.layers.embeddings import MRotaryEmbedding
        from sgl_jax.srt.models.qwen3_5 import Qwen3_5Attention

        attn = Qwen3_5Attention(self.cfg, self.mesh, layer_id=3)
        self.assertIsInstance(attn.rotary_emb, MRotaryEmbedding)
        self.assertEqual(list(attn.rotary_emb.mrope_section), [11, 11, 10])
        self.assertEqual(attn.rotary_emb.rotary_dim, 64)
        self.assertEqual(attn.rotary_emb.base, 10000000)

    # --- 5: GDN fused-projection shapes ---
    def test_gdn_projection_shapes(self):
        from sgl_jax.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet

        tc = self.cfg.text_config
        gdn = Qwen3_5GatedDeltaNet(self.cfg, self.mesh, layer_id=0)
        key_dim = tc.linear_num_key_heads * tc.linear_key_head_dim
        value_dim = tc.linear_num_value_heads * tc.linear_value_head_dim
        self.assertEqual(
            gdn.in_proj_qkvz.weight.value.shape, (tc.hidden_size, 2 * key_dim + 2 * value_dim)
        )
        self.assertEqual(
            gdn.in_proj_ba.weight.value.shape, (tc.hidden_size, 2 * tc.linear_num_value_heads)
        )
        self.assertEqual(
            gdn.conv1d.weight.value.shape, (2 * key_dim + value_dim, tc.linear_conv_kernel_dim)
        )
        self.assertEqual(gdn.A_log.value.shape, (tc.linear_num_value_heads,))
        self.assertEqual(gdn.dt_bias.value.shape, (tc.linear_num_value_heads,))

    # --- 6: weight-mapping coverage vs the real safetensors index ---
    def test_weight_mapping_covers_ckpt_keys(self):
        from sgl_jax.srt.models.qwen3_5 import _create_qwen3_5_weight_mappings

        idx_path = Path(CKPT_FIXTURE) / "model.safetensors.index.json"
        with open(idx_path) as f:
            ckpt_keys = set(json.load(f)["weight_map"].keys())

        mapping, visual_skip, mtp_skip = _create_qwen3_5_weight_mappings(self.cfg)

        text_keys = {k for k in ckpt_keys if k.startswith("model.language_model.")}
        self.assertEqual(len(text_keys), 692)
        for k in text_keys:
            self.assertIn(k, mapping, f"unmapped text key: {k}")
        self.assertIn("lm_head.weight", mapping)

        visual_keys = {k for k in ckpt_keys if k.startswith("model.visual.")}
        self.assertEqual(len(visual_keys), 333)
        for k in visual_keys:
            self.assertTrue(
                any(re.match(p, k) for p in visual_skip), f"visual key not skipped: {k}"
            )

        mtp_keys = {k for k in ckpt_keys if k.startswith("mtp.")}
        self.assertEqual(len(mtp_keys), 785)
        for k in mtp_keys:
            self.assertTrue(any(re.match(p, k) for p in mtp_skip), f"mtp key not skipped: {k}")

        # No mapping key should reference a non-existent ckpt key (apart from the
        # synthetic lm_head/embeds, which are real).
        for src in mapping:
            self.assertIn(src, ckpt_keys, f"mapping references missing ckpt key: {src}")

    # --- 7: MoE block wiring + sigmoid-gated shared expert ---
    def test_moe_block_shared_expert_gate(self):
        from sgl_jax.srt.models.qwen3_5 import Qwen3_5MoeBlock

        tc = self.cfg.text_config
        block = Qwen3_5MoeBlock(self.cfg, self.mesh, layer_id=0)
        # Routed expert param shapes (FusedEPMoE w1/w3 = [E, hidden, inter], w2 = [E, inter, hidden]).
        self.assertEqual(
            block.experts.w1.value.shape, (tc.num_experts, tc.hidden_size, tc.moe_intermediate_size)
        )
        self.assertEqual(
            block.experts.w3.value.shape, (tc.num_experts, tc.hidden_size, tc.moe_intermediate_size)
        )
        self.assertEqual(
            block.experts.w2.value.shape, (tc.num_experts, tc.moe_intermediate_size, tc.hidden_size)
        )
        # Shared path (plain matmuls — CPU-safe). sigmoid(gate)*shared_expert(x).
        T = 4
        x = jax.random.normal(jax.random.key(0), (T, tc.hidden_size), dtype=jnp.bfloat16)
        gate_logit, _ = block.shared_expert_gate(x)
        self.assertEqual(gate_logit.shape, (T, 1))
        shared = block.shared_experts(x)
        self.assertEqual(shared.shape, (T, tc.hidden_size))
        gated = jax.nn.sigmoid(gate_logit) * shared
        self.assertEqual(gated.shape, (T, tc.hidden_size))


if __name__ == "__main__":
    unittest.main()
