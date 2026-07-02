"""Alignment test: JAX Gemma4VisionEncoder vs HF Gemma4VisionModel.

CPU-only (vision tower ~570M). Fixed 280-token budget (2520 patches).
"""

import json
import os
import unittest

import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from safetensors import safe_open
from transformers import AutoConfig, Gemma4VisionModel

from sgl_jax.srt.multimodal.models.gemma4.vision_encoder import (
    Gemma4VisionEncoder,
    vision_weight_mappings,
)
from sgl_jax.test.test_utils import GEMMA4_31B_IT, CustomTestCase

os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _set_by_path(state, path, value):
    parts = path.split(".")
    node = state
    for p in parts[:-1]:
        node = node[int(p)] if p.isdigit() else node[p]
    leaf = parts[-1]
    cur = node[int(leaf)] if leaf.isdigit() else node[leaf]
    cur.value = value.astype(cur.value.dtype)


def _load_vision_weights(model_dir, num_layers):
    with open(f"{model_dir}/model.safetensors.index.json") as f:
        index = json.load(f)["weight_map"]
    mappings = vision_weight_mappings(num_layers)
    by_file: dict[str, list[str]] = {}
    for k in mappings:
        by_file.setdefault(index[k], []).append(k)
    tensors = {}
    for fname, keys in by_file.items():
        # framework="pt": HF state_dict consumes torch tensors directly; the
        # JAX side converts via .float().numpy() (np can't hold bf16 -> torch).
        with safe_open(f"{model_dir}/{fname}", framework="pt") as f:
            for k in keys:
                tensors[k] = f.get_tensor(k)
    return tensors, mappings


class TestGemma4VisionAlignment(CustomTestCase):
    COS_THRESHOLD = 0.999

    def test_vision_encoder_vs_hf(self):
        cfg = AutoConfig.from_pretrained(GEMMA4_31B_IT)
        vcfg = cfg.vision_config

        # 672x960 → 42x60 patches = 2520 → pool 3x3 → 280
        rng = np.random.default_rng(42)
        pv = rng.uniform(0, 1, size=(1, 2520, 768)).astype(np.float32)
        yy, xx = np.meshgrid(np.arange(42), np.arange(60), indexing="ij")
        pos = np.stack([xx.ravel(), yy.ravel()], axis=-1)[None].astype(np.int32)

        # HF reference
        hf = Gemma4VisionModel(vcfg).to(torch.bfloat16).eval()
        tensors, mappings = _load_vision_weights(GEMMA4_31B_IT, vcfg.num_hidden_layers)
        sd = {k.removeprefix("model.vision_tower."): v for k, v in tensors.items()}
        hf.load_state_dict(sd, strict=True)
        with torch.no_grad():
            hf_out = (
                hf(
                    pixel_values=torch.from_numpy(pv).to(torch.bfloat16),
                    pixel_position_ids=torch.from_numpy(pos).long(),
                )
                .last_hidden_state.float()
                .numpy()
            )

        # JAX
        enc = Gemma4VisionEncoder(vcfg, dtype=jnp.bfloat16)
        state = nnx.state(enc)
        for k, dst in mappings.items():
            w = tensors[k].float().numpy()
            if dst.endswith(".kernel"):
                w = w.T
            _set_by_path(state, dst, jnp.asarray(w))
        nnx.update(enc, state)
        jax_out, _ = enc(jnp.asarray(pv), jnp.asarray(pos))
        jax_out = np.asarray(jax_out.astype(jnp.float32))[0]

        self.assertEqual(hf_out.shape, jax_out.shape)
        cos = float(
            (hf_out * jax_out).sum() / (np.linalg.norm(hf_out) * np.linalg.norm(jax_out) + 1e-8)
        )
        mae = float(np.abs(hf_out - jax_out).mean())
        print(f"cos={cos:.6f} mae={mae:.5f}")
        self.assertGreater(cos, self.COS_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
