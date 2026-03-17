# Copyright 2026 SII Team
# Licensed under the Apache License, Version 2.0

"""Test alignment between JAX CLIPVisionModel implementation and PyTorch HuggingFace model."""

import argparse
import logging
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import CLIPVisionModel as HFVisionModel

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.multimodal.models.encoders.clip import CLIPVisionModel as JAXVisionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DTYPE = {
    "float32": (torch.float32, jnp.float32),
    "bfloat16": (torch.bfloat16, jnp.bfloat16),
}

# =============================================================================
# Utilities (复用相同逻辑)
# =============================================================================


def compare(output1, output2, name, threshold=1e-3):
    np1 = output1.detach().float().cpu().numpy() if isinstance(output1, torch.Tensor) else np.array(output1, dtype=np.float32)
    np2 = output2.detach().float().cpu().numpy() if isinstance(output2, torch.Tensor) else np.array(output2, dtype=np.float32)

    mae = np.abs(np1 - np2).mean()
    max_diff = np.abs(np1 - np2).max()

    passed = mae < threshold
    status = "✅" if passed else "❌"
    logger.info("%s %s: MAE=%s, Max=%s", status, name, f"{mae:.2e}", f"{max_diff:.2e}")
    return passed, mae


def set_param(model, path, val):
    parts = path.split(".")
    param = model
    for p in parts:
        param = param[int(p)] if p.isdigit() else getattr(param, p)
    param[...] = jnp.array(val, dtype=param[...].dtype)


def manual_load_weights(jax_model, hf_model):
    from collections import defaultdict
    mappings = jax_model._weight_mappings()
    pt_state = hf_model.state_dict()

    target_to_src = defaultdict(list)
    for src_k, m in mappings.items():
        if src_k in pt_state:
            target_to_src[m.target_path].append((src_k, m))

    loaded = 0
    for target_path, src_list in target_to_src.items():
        try:
            # 现在全是 1对1 映射，直接取 src_list[0] 即可，不再需要 else 分支！
            src_k, m = src_list[0]
            w = pt_state[src_k].detach().float().cpu().numpy()

            if "patch_embedding.kernel" in target_path and len(w.shape) == 4:
                # PyTorch: [O, I, H, W] -> JAX: [H, W, I, O]
                w = np.transpose(w, (2, 3, 1, 0))
            else:
                w = w.T if m.transpose else w

            set_param(jax_model, target_path, w)
            loaded += 1
        except Exception as e:
            logger.error(f"Failed to load {target_path}: {e}")

    if loaded < len(mappings) * 0.5:
        raise RuntimeError(f"Weight loading failed: only {loaded}/{len(mappings)} weights loaded")


def load_models(model_name, mesh, precision):
    pt_dtype, jax_dtype = DTYPE[precision]
    hf = HFVisionModel.from_pretrained(model_name, dtype=pt_dtype, attn_implementation="eager").eval()
    config = hf.config
    model_config = ModelConfig(model_path=model_name, dtype=precision)

    with jax.set_mesh(mesh):
        jax_m = JAXVisionModel(config, mesh, jax_dtype)
        try:
            jax_m.load_weights(model_config)
        except Exception:
            manual_load_weights(jax_m, hf)

    return hf, jax_m, config

# =============================================================================
# Tests
# =============================================================================


def test_vision_single(model_name, mesh, precision):
    logger.info("\n%s\nTest: CLIP Vision Encoder (Single Image)\n%s", "=" * 60, "=" * 60)
    hf, jax_m, config = load_models(model_name, mesh, precision)

    # Use random tensor to purely test model forward math equivalent
    pixel_values = torch.randn(1, config.num_channels, config.image_size, config.image_size, dtype=DTYPE[precision][0])

    with torch.no_grad():
        hf_out = hf(pixel_values=pixel_values).last_hidden_state
        # We manually apply this operation to the HF output here to align them for testing:
        hf_out = hf.vision_model.post_layernorm(hf_out)

    with jax.set_mesh(mesh):
        jax_out = jax_m(jnp.array(pixel_values.numpy(), dtype=DTYPE[precision][1]))

    passed, mae = compare(hf_out[0], jax_out[0], "Vision Encoder Output")
    return passed, mae


def test_vision_hidden_states(model_name, mesh, precision):
    logger.info("\n%s\nTest: CLIP Vision Encoder (Hidden States / VLM extract hook)\n%s", "=" * 60, "=" * 60)
    hf, jax_m, config = load_models(model_name, mesh, precision)

    pixel_values = torch.randn(2, config.num_channels, config.image_size, config.image_size, dtype=DTYPE[precision][0])

    with torch.no_grad():
        # HF: output_hidden_states returns all intermediate features (before post_layernorm!)
        hf_out = hf(pixel_values=pixel_values, output_hidden_states=True)

    with jax.set_mesh(mesh):
        # JAX: We matched the un-normed intermediate output hook
        _, jax_hiddens = jax_m(jnp.array(pixel_values.numpy(), dtype=DTYPE[precision][1]), output_hidden_states=True)

    # VLM traditionally extracts from layer -2
    hf_target_layer = hf_out.hidden_states[-2]
    jax_target_layer = jax_hiddens[-2]

    passed, mae = compare(hf_target_layer, jax_target_layer, "Vision Hidden State [-2]")
    return passed, mae


TESTS = {
    "vision_single": test_vision_single,
    "vision_hidden_states": test_vision_hidden_states,
}


def main():
    import traceback
    from sgl_jax.srt.utils.mesh_utils import create_device_mesh

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="openai/clip-vit-base-patch32", help="Model path")
    parser.add_argument("--test_type", default="all", choices=["all"] + list(TESTS.keys()))
    parser.add_argument("--tp_size", type=int, default=None)
    parser.add_argument("--precision", default="float32", choices=["float32", "bfloat16"])
    args = parser.parse_args()

    precision_map = {"float32": "highest", "bfloat16": "default"}
    jax.config.update("jax_default_matmul_precision", precision_map[args.precision])

    devices = jax.devices()
    tp = args.tp_size or min(len(devices), 4)
    mesh = create_device_mesh(ici_parallelism=[1, tp], dcn_parallelism=[1, 1], devices=devices[:tp], use_explicit_sharding=True)

    tests = list(TESTS.keys()) if args.test_type == "all" else [args.test_type]

    results = []
    t0 = time.time()
    for name in tests:
        try:
            passed, mae = TESTS[name](args.model_path, mesh, args.precision)
            results.append((name, passed, mae))
        except Exception as e:
            logger.error("%s: %s", name, e)
            traceback.print_exc()
            results.append((name, False, float("inf")))

    logger.info("\n%s\nSUMMARY\n%s", "=" * 60, "=" * 60)
    for name, passed, mae in results:
        logger.info("%s %s: MAE=%s", "✅" if passed else "❌", name, f"{mae:.2e}")

    all_pass = all(r[1] for r in results)
    logger.info("\n%s (%.1fs)", "✅ All PASSED" if all_pass else "❌ Some FAILED", time.time() - t0)
    return all_pass


if __name__ == "__main__":
    main()
