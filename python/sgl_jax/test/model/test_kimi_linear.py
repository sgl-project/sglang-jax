import json
import os
import unittest

import jax

from sgl_jax.srt.configs.load_config import LoadConfig, LoadFormat
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.model_loader import get_model_loader
from sgl_jax.srt.models.kimi_linear import KimiLinearForCausalLM
from sgl_jax.test.test_utils import create_device_mesh

MODEL_PATH = "/Kimi-Linear-48B-A3B-Instruct"


class TestKimiLinear(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        devices = jax.devices("tpu")
        cls.mesh = create_device_mesh(
            ici_parallelism=[-1, len(devices)],
            dcn_parallelism=[1, 1],
            devices=devices[: len(devices)],
        )

        model_loader = get_model_loader(
            load_config=LoadConfig(
                model_class=KimiLinearForCausalLM,
                load_format=LoadFormat.JAX,
            ),
            mesh=cls.mesh,
        )
        cfg = ModelConfig(model_path=MODEL_PATH)
        cfg.hf_config.ep_size = 4
        cls.model: KimiLinearForCausalLM = model_loader.load_model(model_config=cfg)
        cls.config = cfg

    def test_kimi_linear_weight_mapping(self):
        """Verify that model loads without error and has expected structure."""
        causal_lm = self.model
        inner_model = causal_lm.model
        self.assertIsNotNone(inner_model)
        self.assertIsNotNone(inner_model.embed_tokens)
        self.assertIsNotNone(inner_model.norm)
        self.assertIsNotNone(inner_model.layers)
        self.assertGreater(len(inner_model.layers), 0)
        print(f"Model loaded: {len(inner_model.layers)} layers")

    def test_weight_mapping_coverage(self):
        """
        Verify every HF safetensor key has a mapping.
        """
        index_path = os.path.join(MODEL_PATH, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        hf_keys = set(index["weight_map"].keys())

        weight_mappings = self.model._create_weight_mappings()
        mapped_keys = set()
        for key, mapping in weight_mappings.items():
            if key.startswith("__MOE_EXPERTS__"):
                if isinstance(mapping.target_path, list):
                    for expert_key in mapping.target_path[1:]:
                        mapped_keys.add(expert_key)
            else:
                mapped_keys.add(key)

        unmapped = set()
        for key in hf_keys:
            if key not in mapped_keys:
                unmapped.add(key)

        if unmapped:
            print(f"\nUnmapped HF keys ({len(unmapped)}):")
            for key in sorted(unmapped):
                print(f"  {key}")

        self.assertEqual(len(unmapped), 0, f"{len(unmapped)} HF weight keys have no mapping")


if __name__ == "__main__":
    unittest.main(verbosity=2)
