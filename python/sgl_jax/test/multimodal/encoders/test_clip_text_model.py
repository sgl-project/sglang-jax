# Copyright 2026 SII Team
# Licensed under the Apache License, Version 2.0

"""Test alignment between JAX CLIPTextModel implementation and PyTorch HuggingFace model."""

import argparse
import logging
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from transformers import CLIPTextModel as HFTextModel

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.multimodal.models.encoders.clip import CLIPTextModel as JAXTextModel

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DTYPE = {
    "float32": (torch.float32, jnp.float32),
    "bfloat16": (torch.bfloat16, jnp.bfloat16),
}

# =============================================================================
# Utilities
# =============================================================================


def compare(output1, output2, name, threshold=1e-3):
    """Compare two outputs (can be PyTorch, JAX, or numpy)."""
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
    status = "✅" if passed else "❌"
    logger.info("%s %s: MAE=%s, Max=%s", status, name, f"{mae:.2e}", f"{max_diff:.2e}")
    return passed, mae


def load_models(model_name, mesh, precision):
    pt_dtype, jax_dtype = DTYPE[precision]
    hf = HFTextModel.from_pretrained(model_name, dtype=pt_dtype, attn_implementation="eager").eval()
    config = hf.config

    if not os.path.isdir(model_name):
        local_model_path = snapshot_download(model_name, allow_patterns=["*.safetensors", "*.json"])
    else:
        local_model_path = model_name

    model_config = ModelConfig(model_path=local_model_path, dtype=precision)

    with jax.set_mesh(mesh):
        jax_m = JAXTextModel(config, mesh, jax_dtype)
        jax_m.load_weights(model_config)

    return hf, jax_m, config


# =============================================================================
# Tests
# =============================================================================


def test_weight_mapping(model_name, mesh, tokenizer, precision):
    logger.info("\n%s\nTest: CLIP Text Encoder (Weight Mapping)\n%s", "=" * 60, "=" * 60)

    pt_dtype, jax_dtype = DTYPE[precision]
    hf = HFTextModel.from_pretrained(model_name, dtype=pt_dtype, attn_implementation="eager").eval()

    with jax.set_mesh(mesh):
        jax_m = JAXTextModel(hf.config, mesh, jax_dtype)

    hf_state_dict = hf.state_dict()
    jax_mappings = jax_m._weight_mappings()

    missing_in_hf = []
    for src_key in jax_mappings:
        if src_key not in hf_state_dict:
            missing_in_hf.append(src_key)

    if missing_in_hf:
        logger.error("❌ Found %d JAX mapping keys missing in HF model:", len(missing_in_hf))
        for k in missing_in_hf[:10]:
            logger.error("  - %s", k)
        if len(missing_in_hf) > 10:
            logger.error("  ... and %d more.", len(missing_in_hf) - 10)
        return False, float("inf")

    logger.info(
        "✅ All %d JAX weight mappings successfully found in HuggingFace state_dict.",
        len(jax_mappings),
    )

    return True, 0.0


def test_text_single(model_name, mesh, tokenizer, precision):
    logger.info("\n%s\nTest: CLIP Text Encoder (Single Seq)\n%s", "=" * 60, "=" * 60)
    hf, jax_m, _ = load_models(model_name, mesh, precision)

    text = "A photo of a fast car driving on the highway."
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    with torch.no_grad():
        hf_out = hf(input_ids=input_ids, attention_mask=attention_mask)

    with jax.set_mesh(mesh):
        jax_out = jax_m(
            input_ids=jnp.array(input_ids.numpy(), dtype=jnp.int32),
            attention_mask=jnp.array(attention_mask.numpy(), dtype=jnp.int32),
        )

    passed_hs, mae_hs = compare(
        hf_out.last_hidden_state[0], jax_out.last_hidden_state[0], "Text Hidden State"
    )
    passed_pool, mae_pool = compare(
        hf_out.pooler_output[0], jax_out.pooler_output[0], "Text Pooler Output"
    )

    return passed_hs and passed_pool, max(mae_hs, mae_pool)


def test_text_batch(model_name, mesh, tokenizer, precision):
    logger.info("\n%s\nTest: CLIP Text Encoder (Batch)\n%s", "=" * 60, "=" * 60)
    hf, jax_m, _ = load_models(model_name, mesh, precision)

    texts = [
        "A photo of a fast car.",
        "An image of a cat sleeping on a sofa.",
        "The quick brown fox jumps over the lazy dog.",
    ]
    inputs = tokenizer(texts, padding="max_length", max_length=20, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    with torch.no_grad():
        hf_out = hf(input_ids=input_ids, attention_mask=attention_mask)

    with jax.set_mesh(mesh):
        jax_out = jax_m(
            input_ids=jnp.array(input_ids.numpy(), dtype=jnp.int32),
            attention_mask=jnp.array(attention_mask.numpy(), dtype=jnp.int32),
        )

    all_pass = True
    total_mae = 0.0
    for i in range(len(texts)):
        p_hs, m_hs = compare(
            hf_out.last_hidden_state[i], jax_out.last_hidden_state[i], f"Batch Seq {i+1} [Hidden]"
        )
        p_pool, m_pool = compare(
            hf_out.pooler_output[i], jax_out.pooler_output[i], f"Batch Seq {i+1} [Pooler]"
        )

        all_pass &= p_hs and p_pool
        total_mae += (m_hs + m_pool) / 2.0

    return all_pass, total_mae / len(texts)


TESTS = {
    "weight_mapping": test_weight_mapping,
    "text_single": test_text_single,
    "text_batch": test_text_batch,
}


def main():
    import traceback

    from sgl_jax.srt.utils.mesh_utils import create_device_mesh

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="openai/clip-vit-large-patch14", help="Model path")
    parser.add_argument("--test_type", default="all", choices=["all"] + list(TESTS.keys()))
    parser.add_argument("--tp_size", type=int, default=None)
    parser.add_argument("--precision", default="float32", choices=["float32", "bfloat16"])
    args = parser.parse_args()

    precision_map = {"float32": "highest", "bfloat16": "default"}
    jax.config.update("jax_default_matmul_precision", precision_map[args.precision])

    devices = jax.devices()
    tp = args.tp_size or min(len(devices), 4)
    mesh = create_device_mesh(
        ici_parallelism=[1, tp],
        dcn_parallelism=[1, 1],
        devices=devices[:tp],
        use_explicit_sharding=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tests = list(TESTS.keys()) if args.test_type == "all" else [args.test_type]

    results = []
    t0 = time.time()
    for name in tests:
        try:
            passed, mae = TESTS[name](args.model_path, mesh, tokenizer, args.precision)
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
