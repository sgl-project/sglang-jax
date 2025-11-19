import unittest
import uuid

import openai
import requests

from sgl_jax.srt.configs.load_config import LoadConfig
from sgl_jax.srt.hf_transformers_utils import get_config
from sgl_jax.srt.lora.lora import ChunkedSgmvLoRABackend, LoRAAdapter
from sgl_jax.srt.lora.lora_config import LoRAConfig
from sgl_jax.test.test_utils import CustomTestCase

LORA_MODEL = "trl-lib/Qwen3-4B-LoRA"
LORA_BASE_MODEL = "Qwen/Qwen3-4B"


class TestLora(CustomTestCase):
    def test_adapter(self):
        config = LoRAConfig(LORA_MODEL)
        base_hf_config = get_config(LORA_BASE_MODEL, trust_remote_code=True, revision=None)
        load_config = LoadConfig()
        lora_adapter = LoRAAdapter(
            uuid.uuid4(), config, base_hf_config, load_config, ChunkedSgmvLoRABackend()
        )
        lora_adapter.initialize_weights()
        self.assertEqual(lora_adapter.scaling, 1, "lora_alpha, r is not right")
        self.assertEqual(len(lora_adapter.layers), 36, "layers is not right")
        for i in range(36):
            ## dict_keys(['base_model.model.model.layers.22.self_attn.q_proj.lora_A.weight',
            # 'base_model.model.model.layers.22.self_attn.q_proj.lora_B.weight',
            # 'base_model.model.model.layers.22.self_attn.v_proj.lora_A.weight',
            # 'base_model.model.model.layers.22.self_attn.v_proj.lora_B.weight'])
            self.assertEqual(
                len(lora_adapter.layers[i].weights), 4, "weights per layer is not right"
            )


if __name__ == "__main__":
    unittest.main()
