import unittest

import yaml

from sgl_jax.srt.multimodal.models.static_configs.yaml_registry import (
    StageConfigRegistry,
    get_stage_config_path,
)


class TestMiMoV25StageRegistry(unittest.TestCase):
    def test_mimo_v25_exact_hf_repo_maps_to_stage_config(self):
        path = get_stage_config_path("XiaomiMiMo/MiMo-V2.5")

        self.assertTrue(path.endswith("mimo_v2_5_stage_config.yaml"))

    def test_mimo_v25_local_basename_maps_to_stage_config(self):
        path = get_stage_config_path("/models/checkpoints/MiMo-V2.5")

        self.assertTrue(path.endswith("mimo_v2_5_stage_config.yaml"))

    def test_mimo_v25_keyword_fallback_is_case_insensitive(self):
        for model_path in (
            "/models/MiMo-V2.5-local-snapshot",
            "/models/mimo-v2.5-local-snapshot",
        ):
            with self.subTest(model_path=model_path):
                path = StageConfigRegistry.get_yaml_path(model_path)
                self.assertEqual(path.name, "mimo_v2_5_stage_config.yaml")

    def test_mimo_v25_text_only_variants_do_not_route_to_omni(self):
        # MiMo-V2.5-Pro / -Flash are text-only and must NOT match the omni config
        # via the broad substring (review D4-2 / D5-1).
        for model_path in (
            "/models/MiMo-V2.5-Pro",
            "/models/mimo-v2.5-pro",
            "/models/MiMo-V2.5-Flash",
        ):
            with self.subTest(model_path=model_path), self.assertRaises(ValueError):
                StageConfigRegistry.get_yaml_path(model_path)

    def test_mimo_v25_stage_config_uses_embedding_and_ar_input_embedding(self):
        path = get_stage_config_path("XiaomiMiMo/MiMo-V2.5")

        with open(path) as handle:
            config = yaml.safe_load(handle)

        self.assertEqual(config["model_arch"], "MiMo-V2.5")
        stages = config["stage_args"]
        self.assertEqual(stages[0]["scheduler"], "embedding")
        self.assertEqual(stages[0]["model_class"], "MiMoV2_5Embedding")
        self.assertEqual(stages[1]["scheduler"], "auto_regressive")
        self.assertEqual(stages[1]["model_class"], "MiMoV2ForCausalLM")
        self.assertTrue(stages[1]["precompile_params"]["input_embedding"])
        self.assertFalse(stages[1]["precompile_params"]["deepstack_visual_embedding"])
        self.assertFalse(stages[1]["precompile_params"]["mrope"])


if __name__ == "__main__":
    unittest.main()
