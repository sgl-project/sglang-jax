# Copyright 2026 SII Team
# Licensed under the Apache License, Version 2.0

"""Test alignment between JAX T5EncoderModel implementation and PyTorch HuggingFace model."""

import logging
import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from transformers import T5EncoderModel as HFEncoderModel

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.multimodal.models.encoders.t5 import T5EncoderModel
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "google-t5/t5-small"
THRESHOLD = 1e-3


def _compare(output1, output2, name, threshold=THRESHOLD):
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


class TestT5Encoder(unittest.TestCase):
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
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)

        cls.hf_model = HFEncoderModel.from_pretrained(
            cls.model_name, torch_dtype=torch.float32, attn_implementation="eager"
        ).eval()

        if not os.path.isdir(cls.model_name):
            local_path = snapshot_download(
                cls.model_name,
                allow_patterns=["*.safetensors", "*.json", "model.safetensors.index.json"],
            )
        else:
            local_path = cls.model_name

        model_config = ModelConfig(model_path=local_path, dtype="float32")
        with jax.set_mesh(cls.mesh):
            cls.jax_model = T5EncoderModel(cls.hf_model.config, cls.mesh, jnp.float32)
            cls.jax_model.load_weights(model_config)

    def test_weight_mapping(self):
        jax_mappings = self.jax_model._weight_mappings()
        hf_state_dict = self.hf_model.state_dict()
        missing = [k for k in jax_mappings if k not in hf_state_dict]
        self.assertEqual(missing, [], f"JAX mapping keys missing in HF: {missing}")

    def test_single_sequence(self):
        texts = ["Hello world, this is a test for standard T5."]
        inp = self.tokenizer(texts, return_tensors="pt", padding=True)
        hf_ids, hf_mask = inp.input_ids, inp.attention_mask

        with torch.no_grad():
            hf_h = self.hf_model(input_ids=hf_ids, attention_mask=hf_mask).last_hidden_state

        with jax.set_mesh(self.mesh):
            jax_out = self.jax_model(
                input_ids=jnp.array(hf_ids.numpy(), dtype=jnp.int32),
                attention_mask=jnp.array(hf_mask.numpy(), dtype=jnp.int32),
            )

        passed, mae = _compare(hf_h[0], jax_out.last_hidden_state[0], "Encoder Output")
        self.assertTrue(passed, f"Encoder output mismatch: MAE={mae:.2e}")

    def test_batch_encoding(self):
        texts = [
            "Short text.",
            "A bit longer sentence.",
            "This is an even longer sentence for padding testing.",
        ]
        inp = self.tokenizer(texts, return_tensors="pt", padding=True)
        hf_ids, hf_mask = inp.input_ids, inp.attention_mask

        with torch.no_grad():
            hf_out = self.hf_model(
                input_ids=hf_ids, attention_mask=hf_mask
            ).last_hidden_state

        with jax.set_mesh(self.mesh):
            jax_out = self.jax_model(
                input_ids=jnp.array(hf_ids.numpy(), dtype=jnp.int32),
                attention_mask=jnp.array(hf_mask.numpy(), dtype=jnp.int32),
            ).last_hidden_state

        for i in range(len(texts)):
            actual_len = hf_mask[i].sum().item()
            passed, mae = _compare(
                hf_out[i, :actual_len], jax_out[i, :actual_len], f"Batch Seq {i + 1}"
            )
            self.assertTrue(passed, f"Batch seq {i + 1} mismatch: MAE={mae:.2e}")


if __name__ == "__main__":
    unittest.main()
