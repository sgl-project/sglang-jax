"""CPU test: MTP-layer ``modules_to_not_convert`` translation for GlmMoeDsaForCausalLMNextN."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from sgl_jax.srt.models.glm5_moe import (
    _checkpoint_modules_to_not_convert,
    nextn_ignored_layers,
)


class TestNextnIgnoredLayers(unittest.TestCase):
    def test_translates_mtp_layer_entries(self):
        ignored = [
            "model.layers.3.mlp.gate",
            "model.layers.78.eh_proj",
            "model.layers.78.mlp.gate",
            "model.layers.78.self_attn.indexers_proj",
            "model.layers.78.shared_head.norm",
            "model.layers.78.hnorm",
            "lm_head",
        ]
        out = nextn_ignored_layers(ignored, 78)
        for ig in ignored:
            self.assertIn(ig, out)
        self.assertIn("eh_proj", out)
        self.assertIn("hnorm", out)
        self.assertIn("shared_head.norm", out)
        self.assertIn("mtp_block.mlp.gate", out)
        self.assertIn("mtp_block.self_attn.indexers_proj", out)
        self.assertNotIn(
            "mtp_block.mlp.gate", nextn_ignored_layers(["model.layers.3.mlp.gate"], 78)
        )

    def test_merges_checkpoint_entries_for_the_mtp_layer_only(self):
        # A user yaml (--quantization-config-path) starts from an empty
        # ignored list, so the MTP layer's bf16 modules must come from the
        # checkpoint's modules_to_not_convert; other layers' entries stay out.
        hf = [
            "model.layers.78.eh_proj",
            "model.layers.78.self_attn.indexers_proj",
            "model.layers.3.mlp.gate",
            "lm_head",
        ]
        out = nextn_ignored_layers(["indexer.weights_proj"], 78, hf_ignored=hf)
        self.assertIn("eh_proj", out)
        self.assertIn("mtp_block.self_attn.indexers_proj", out)
        self.assertIn("indexer.weights_proj", out)
        self.assertNotIn("model.layers.3.mlp.gate", out)
        self.assertNotIn("lm_head", out)
        self.assertEqual(nextn_ignored_layers(["eh_proj"], 78, hf_ignored=None), ["eh_proj"])

    def test_checkpoint_list_survives_the_hf_config_overwrite(self):
        # ModelConfig replaces hf_config.quantization_config with the unified
        # QuantizationConfig object; the raw checkpoint dict is kept on
        # hf_quantization_config_raw and must be preferred.
        obj = SimpleNamespace(is_static_checkpoint=True)  # not a dict
        mc = SimpleNamespace(
            hf_quantization_config_raw={"modules_to_not_convert": ["model.layers.78.eh_proj"]},
            hf_config=SimpleNamespace(quantization_config=obj),
        )
        self.assertEqual(_checkpoint_modules_to_not_convert(mc), ["model.layers.78.eh_proj"])
        mc2 = SimpleNamespace(hf_config=SimpleNamespace(quantization_config=obj))
        self.assertIsNone(_checkpoint_modules_to_not_convert(mc2))
        mc3 = SimpleNamespace(
            hf_config=SimpleNamespace(quantization_config={"ignored_layers": ["lm_head"]})
        )
        self.assertEqual(_checkpoint_modules_to_not_convert(mc3), ["lm_head"])

    def test_checkpoint_list_falls_back_to_config_json(self):
        # hf_config is lru-cached across the target and draft ModelConfigs, so
        # the draft may only ever see the overwritten object: read config.json.
        obj = SimpleNamespace(is_static_checkpoint=True)
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w") as f:
                json.dump(
                    {
                        "quantization_config": {
                            "modules_to_not_convert": ["model.layers.78.eh_proj"]
                        }
                    },
                    f,
                )
            mc = SimpleNamespace(model_path=tmp, hf_config=SimpleNamespace(quantization_config=obj))
            self.assertEqual(_checkpoint_modules_to_not_convert(mc), ["model.layers.78.eh_proj"])
        mc_missing = SimpleNamespace(
            model_path="/nonexistent/dir", hf_config=SimpleNamespace(quantization_config=obj)
        )
        self.assertIsNone(_checkpoint_modules_to_not_convert(mc_missing))

    def test_idempotent_and_none_safe(self):
        self.assertEqual(nextn_ignored_layers(None, 78), [])
        once = nextn_ignored_layers(["model.layers.78.eh_proj"], 78)
        self.assertEqual(nextn_ignored_layers(once, 78), once)


if __name__ == "__main__":
    unittest.main()
