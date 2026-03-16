# Copyright 2026 SII Team
# Licensed under the Apache License, Version 2.0

"""Test alignment between JAX UMT5Encoder implementation and PyTorch HuggingFace model."""

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
from transformers import UMT5EncoderModel as HFEncoderModel

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.multimodal.models.encoders.t5 import UMT5EncoderModel

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


def hf_batch(texts, tokenizer):
    """Create HuggingFace padded batch."""
    inp = tokenizer(texts, return_tensors="pt", padding=True)
    return inp.input_ids, inp.attention_mask


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
    logger.info("%s %s: MAE=%.2e, Max=%.2e", status, name, mae, max_diff)
    return passed, mae


def load_models(model_name, mesh, precision):
    pt_dtype, jax_dtype = DTYPE[precision]

    hf = HFEncoderModel.from_pretrained(
        model_name, torch_dtype=pt_dtype, attn_implementation="eager"
    ).eval()
    config = hf.config

    if not os.path.isdir(model_name):
        local_model_path = snapshot_download(
            model_name, allow_patterns=["*.safetensors", "*.json", "model.safetensors.index.json"]
        )
    else:
        local_model_path = model_name

    model_config = ModelConfig(model_path=local_model_path, dtype=precision)

    with jax.set_mesh(mesh):
        jax_m = UMT5EncoderModel(config, mesh, jax_dtype)
        jax_m.load_weights(model_config)

    return hf, jax_m, config


# =============================================================================
# Tests
# =============================================================================


def test_weight_mapping(model_name, mesh, tokenizer, precision):
    """Verify that all keys required by JAX mappings exist in the HF model."""
    logger.info("\n%s\nTest: Weight Mapping\n%s", "=" * 60, "=" * 60)
    hf_m, jax_m, _ = load_models(model_name, mesh, precision)

    jax_mappings = jax_m._weight_mappings()
    hf_state_dict = hf_m.state_dict()

    missing_in_hf = []

    for jax_hf_key in jax_mappings:
        if jax_hf_key not in hf_state_dict:
            missing_in_hf.append(jax_hf_key)

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


def test_encoder(model_name, mesh, tokenizer, precision):
    logger.info("\n%s\nTest: UMT5 Encoder (Single Seq)\n%s", "=" * 60, "=" * 60)
    hf, jax_m, _ = load_models(model_name, mesh, precision)

    texts = ["Hello world, this is a test for standard UMT5.", "Machine learning is fascinating."]
    hf_ids, hf_mask = hf_batch(texts, tokenizer)

    with torch.no_grad():
        hf_h = hf(input_ids=hf_ids, attention_mask=hf_mask).last_hidden_state

    with jax.set_mesh(mesh):
        jax_output = jax_m(
            input_ids=jnp.array(hf_ids.numpy(), dtype=jnp.int32),
            attention_mask=jnp.array(hf_mask.numpy(), dtype=jnp.int32),
        )
        jax_h = jax_output.last_hidden_state

    if jax_output.attention_mask is None:
        logger.error("❌ UMT5EncoderModel failed to return attention_mask!")
        return False, float("inf")

    passed, mae = compare(hf_h[0], jax_h[0], "Encoder Output")
    return passed, mae


def test_encoder_batch(model_name, mesh, tokenizer, precision):
    logger.info("\n%s\nTest: UMT5 Encoder (Batch/Continuous)\n%s", "=" * 60, "=" * 60)
    hf, jax_m, _ = load_models(model_name, mesh, precision)

    texts = [
        "Short text.",
        "A bit longer sentence.",
        "This is an even longer sentence for padding testing.",
    ]
    hf_ids, hf_mask = hf_batch(texts, tokenizer)

    with torch.no_grad():
        hf_out = hf(input_ids=hf_ids, attention_mask=hf_mask).last_hidden_state

    with jax.set_mesh(mesh):
        jax_output = jax_m(
            input_ids=jnp.array(hf_ids.numpy(), dtype=jnp.int32),
            attention_mask=jnp.array(hf_mask.numpy(), dtype=jnp.int32),
        )
        jax_out = jax_output.last_hidden_state

    all_pass = True
    total_mae = 0.0

    for i in range(len(texts)):
        actual_len = hf_mask[i].sum().item()
        passed, mae = compare(hf_out[i, :actual_len], jax_out[i, :actual_len], f"Batch Seq {i + 1}")
        all_pass &= passed
        total_mae += mae

    return all_pass, total_mae / len(texts)


TESTS = {
    "weight_mapping": test_weight_mapping,
    "encoder": test_encoder,
    "encoder_batch": test_encoder_batch,
}


def main():
    import traceback

    from sgl_jax.srt.utils.mesh_utils import create_device_mesh

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", default="shkna1368/umt5-small-finetuned-umt5-poemV1", help="Model path"
    )
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
        status = "✅" if passed else "❌"
        logger.info("%s %s: MAE=%.2e", status, name, mae)

    all_pass = all(r[1] for r in results)
    status_msg = "✅ All PASSED" if all_pass else "❌ Some FAILED"
    logger.info("\n%s (%.1fs)", status_msg, time.time() - t0)
    return all_pass


if __name__ == "__main__":
    main()
