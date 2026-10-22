"""CPU: the GLM NextN draft under a static FP8 block yaml keeps ``eh_proj`` in
bf16 and dummy-loads (the gp9/gp9b boot failure: ``eh_proj.weight is not a
valid param path`` once quantize_model turned it into a QuantizedLinear)."""

import importlib.util
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from safetensors.numpy import save_file

import sgl_jax
from sgl_jax.srt.configs.quantization_config import QuantizationConfig
from sgl_jax.srt.layers.linear import LinearBase, QuantizedLinear
from sgl_jax.srt.model_loader.weights.loader import WeightLoader
from sgl_jax.srt.models.glm5_moe import (
    GlmMoeDsaForCausalLMNextN,
    _checkpoint_modules_to_not_convert,
    nextn_ignored_layers,
)
from sgl_jax.srt.utils.quantization.quantization_utils import apply_quantization

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "nextn_wm_helpers", os.path.join(_HERE, "test_glm5_nextn_weight_mappings.py")
)
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
_tiny_config, _mesh = _helpers._tiny_config, _helpers._mesh

YAML = os.path.join(
    os.path.dirname(sgl_jax.__file__),
    "srt/utils/quantization/configs/fp8_deepseek_v3.yaml",
)


def _draft_model_config(tmp, cfg, *, with_checkpoint_list: bool):
    n = cfg.num_hidden_layers
    hf = {
        "quant_method": "fp8",
        "modules_to_not_convert": [
            f"model.layers.{n}.eh_proj",
            f"model.layers.{n}.mlp.gate",
            f"model.layers.{n}.self_attn.indexers_proj",
            "lm_head",
        ],
    }
    with open(os.path.join(tmp, "config.json"), "w") as f:
        json.dump({"quantization_config": hf}, f)
    save_file({"placeholder": np.zeros((1,), np.float32)}, os.path.join(tmp, "model.safetensors"))
    qc = QuantizationConfig.from_path(YAML)
    # hf_config is the shared, already-overwritten object the draft sees.
    mc = SimpleNamespace(model_path=tmp, hf_config=cfg, quantization_config=qc, _dummy_mode=True)
    base = list(qc.ignored_layers or []) + ["indexer.weights_proj"]
    hf_ignored = _checkpoint_modules_to_not_convert(mc) if with_checkpoint_list else None
    qc.ignored_layers = nextn_ignored_layers(base, n, hf_ignored=hf_ignored)
    return mc


class TestNextnStaticQuantDummyLoad(unittest.TestCase):
    def test_without_checkpoint_list_eh_proj_is_quantized(self):
        cfg = _tiny_config(
            hidden_size=128, intermediate_size=128, moe_intermediate_size=128
        )  # block-128 static yaml
        mesh = _mesh()
        with tempfile.TemporaryDirectory() as tmp:
            mc = _draft_model_config(tmp, cfg, with_checkpoint_list=False)
            with jax.set_mesh(mesh):
                model = nnx.eval_shape(
                    lambda: GlmMoeDsaForCausalLMNextN(cfg, mesh=mesh, dtype=jnp.bfloat16)
                )
                model = apply_quantization(mc, model, is_static_input=True)
                self.assertIsInstance(model.eh_proj, QuantizedLinear)

    def test_with_checkpoint_list_eh_proj_stays_bf16_and_dummy_loads(self):
        cfg = _tiny_config(
            hidden_size=128, intermediate_size=128, moe_intermediate_size=128
        )  # block-128 static yaml
        mesh = _mesh()
        with tempfile.TemporaryDirectory() as tmp:
            mc = _draft_model_config(tmp, cfg, with_checkpoint_list=True)
            self.assertIn("eh_proj", mc.quantization_config.ignored_layers)
            with jax.set_mesh(mesh):
                model = nnx.eval_shape(
                    lambda: GlmMoeDsaForCausalLMNextN(cfg, mesh=mesh, dtype=jnp.bfloat16)
                )
                model = apply_quantization(mc, model, is_static_input=True)
                self.assertIsInstance(model.eh_proj, LinearBase)
                self.assertNotIsInstance(model.eh_proj, QuantizedLinear)
                self.assertIsInstance(model.mtp_block.self_attn.o_proj, QuantizedLinear)
                loader = WeightLoader(model, mc, mesh, dtype=jnp.bfloat16)
                mappings = GlmMoeDsaForCausalLMNextN._create_weight_mappings(mc)
                for _, spec in mappings.items():
                    targets = (
                        (spec.target_path,)
                        if isinstance(spec.target_path, str)
                        else spec.target_path
                    )
                    for target in targets:
                        loader._get_param(model, target)
                model.load_weights(mc)


if __name__ == "__main__":
    unittest.main()
