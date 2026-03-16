# Copyright 2026 SII Team
# Licensed under the Apache License, Version 2.0

"""Test alignment between JAX T5Encoder implementation and PyTorch HuggingFace model."""

import argparse
import logging
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from transformers import AutoTokenizer, T5Config
from transformers import T5EncoderModel as HFEncoderModel

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.multimodal.models.encoders.t5 import T5EncoderModel

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
    mappings = jax_model._weight_mappings()
    pt_state = hf_model.state_dict()
    loaded = 0
    for k, m in mappings.items():
        if k in pt_state:
            try:
                w = pt_state[k].detach().float().cpu().numpy()
                set_param(jax_model, m.target_path, w.T if m.transpose else w)
                loaded += 1
            except Exception:
                pass
    if loaded < len(mappings) * 0.5:
        raise RuntimeError(f"Weight loading failed: only {loaded}/{len(mappings)} weights loaded")


def load_models(model_name, mesh, precision):
    pt_dtype, jax_dtype = DTYPE[precision]
    config = T5Config.from_pretrained(model_name, dtype=pt_dtype)
    model_config = ModelConfig(model_path=model_name, dtype=precision)

    hf = HFEncoderModel.from_pretrained(model_name, dtype=pt_dtype, attn_implementation="eager").eval()

    with jax.set_mesh(mesh):
        jax_m = T5EncoderModel(config, mesh, jax_dtype)
        try:
            jax_m.load_weights(model_config)
        except Exception:
            manual_load_weights(jax_m, hf)

    return hf, jax_m, config

# =============================================================================
# Tests
# =============================================================================


def test_encoder(model_name, mesh, tokenizer, precision):
    logger.info("\n%s\nTest: T5 Encoder (Single Seq)\n%s", "=" * 60, "=" * 60)
    hf, jax_m, _ = load_models(model_name, mesh, precision)

    texts = ["Hello world, this is a test for standard T5.", "Machine learning is fascinating."]
    hf_ids, hf_mask = hf_batch(texts, tokenizer)

    with torch.no_grad():
        hf_h = hf(input_ids=hf_ids, attention_mask=hf_mask).last_hidden_state

    with jax.set_mesh(mesh):
        jax_out = jax_m(jnp.array(hf_ids.numpy(), dtype=jnp.int32))

    passed, mae = compare(hf_h[0], jax_out[0], "Encoder Output")
    return passed, mae


def test_encoder_batch(model_name, mesh, tokenizer, precision):
    logger.info("\n%s\nTest: T5 Encoder (Batch/Continuous)\n%s", "=" * 60, "=" * 60)
    hf, jax_m, _ = load_models(model_name, mesh, precision)

    texts = ["Short text.", "A bit longer sentence.", "This is an even longer sentence for padding testing."]
    hf_ids, hf_mask = hf_batch(texts, tokenizer)

    with torch.no_grad():
        hf_out = hf(input_ids=hf_ids).last_hidden_state

    with jax.set_mesh(mesh):
        jax_out = jax_m(jnp.array(hf_ids.numpy(), dtype=jnp.int32))

    all_pass = True
    total_mae = 0.0

    for i in range(len(texts)):
        passed, mae = compare(hf_out[i], jax_out[i], f"Batch Seq {i+1}")
        all_pass &= passed
        total_mae += mae

    return all_pass, total_mae / len(texts)


TESTS = {
    "encoder": test_encoder,
    "encoder_batch": test_encoder_batch,
}


def main():
    import traceback
    from sgl_jax.srt.utils.mesh_utils import create_device_mesh

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="google-t5/t5-small", help="Model path")
    parser.add_argument("--test_type", default="all", choices=["all"] + list(TESTS.keys()))
    parser.add_argument("--tp_size", type=int, default=None)
    parser.add_argument("--precision", default="float32", choices=["float32", "bfloat16"])
    args = parser.parse_args()

    precision_map = {"float32": "highest", "bfloat16": "default"}
    jax.config.update("jax_default_matmul_precision", precision_map[args.precision])

    devices = jax.devices()
    tp = args.tp_size or min(len(devices), 4)
    mesh = create_device_mesh(ici_parallelism=[1, tp], dcn_parallelism=[1, 1], devices=devices[:tp], use_explicit_sharding=True)

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