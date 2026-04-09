# Copyright 2026 SII Team
# Licensed under the Apache License, Version 2.0

"""Test alignment between JAX CLIPTextModel implementation and PyTorch HuggingFace model."""

import logging
import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import CLIPTextModel as HFTextModel

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.multimodal.models.encoders.clip import CLIPTextModel as JAXTextModel
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/clip-vit-large-patch14"


def _compare(output1, output2, name, threshold=1e-3):
    np1 = (
        output1.detach().float().cpu().numpy()
        if isinstance(output1, torch.Tensor)
        else np.array(output1, dtype=np.float32)
    )
    np2 = (
        output2.detach().float().cpu().numpy()
        if isinstance(output2, torch.Tensor)
        else np.array(output2, dtype=np.float32)
    )
    mae = np.abs(np1 - np2).mean()
    max_diff = np.abs(np1 - np2).max()
    passed = mae < threshold
    status = "PASS" if passed else "FAIL"
    logger.info("%s %s: MAE=%.2e, Max=%.2e", status, name, mae, max_diff)
    return passed, mae


class TestCLIPTextModel(unittest.TestCase):
    model_name = DEFAULT_MODEL

    @classmethod
    def setUpClass(cls):
        jax.config.update("jax_default_matmul_precision", "highest")
        devices = jax.devices()
        tp = min(len(devices), 4)
        cls.mesh = create_device_mesh(
            ici_parallelism=[1, tp],
            dcn_parallelism=[1, 1],
            devices=devices[:tp],
            use_explicit_sharding=True,
        )

        cls.hf_model = HFTextModel.from_pretrained(
            cls.model_name, dtype=torch.float32, attn_implementation="eager"
        ).eval()

        if not os.path.isdir(cls.model_name):
            local_path = snapshot_download(
                cls.model_name, allow_patterns=["*.safetensors", "*.json"]
            )
        else:
            local_path = cls.model_name

        model_config = ModelConfig(model_path=local_path, dtype="float32")
        with jax.set_mesh(cls.mesh):
            cls.jax_model = JAXTextModel(cls.hf_model.config, cls.mesh, jnp.float32)
            cls.jax_model.load_weights(model_config)

    def test_weight_mapping(self):
        with jax.set_mesh(self.mesh):
            jax_m = JAXTextModel(self.hf_model.config, self.mesh, jnp.float32)
        jax_mappings = jax_m._weight_mappings()
        hf_state_dict = self.hf_model.state_dict()
        missing = [k for k in jax_mappings if k not in hf_state_dict]
        self.assertEqual(missing, [], f"JAX mapping keys missing in HF: {missing}")

    def test_single_seq_without_attn_mask(self):
        """Mimics the real FLUX.1 calling path — expects perfect alignment."""
        input_ids = torch.tensor(
            [[49406, 320, 2242, 1794, 2102, 22456, 49407, 49407, 49407, 49407,
              49407, 49407, 49407, 49407, 49407, 49407]],
            dtype=torch.long,
        )

        with torch.no_grad():
            hf_out = self.hf_model(input_ids=input_ids, attention_mask=None)

        with jax.set_mesh(self.mesh):
            jax_out = self.jax_model(
                input_ids=jnp.array(input_ids.numpy(), dtype=jnp.int32),
                attention_mask=None,
            )

        passed_hs, _ = _compare(
            hf_out.last_hidden_state[0], jax_out.last_hidden_state[0], "Hidden State"
        )
        passed_pool, _ = _compare(
            hf_out.pooler_output[0], jax_out.pooler_output[0], "Pooler Output"
        )
        self.assertTrue(passed_hs, "Hidden state mismatch")
        self.assertTrue(passed_pool, "Pooler output mismatch")


if __name__ == "__main__":
    unittest.main()
