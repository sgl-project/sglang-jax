"""CPU test: MTP-layer ``modules_to_not_convert`` translation for GlmMoeDsaForCausalLMNextN."""

import unittest

from sgl_jax.srt.models.glm5_moe import nextn_ignored_layers


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

    def test_idempotent_and_none_safe(self):
        self.assertEqual(nextn_ignored_layers(None, 78), [])
        once = nextn_ignored_layers(["model.layers.78.eh_proj"], 78)
        self.assertEqual(nextn_ignored_layers(once, 78), once)


if __name__ == "__main__":
    unittest.main()
