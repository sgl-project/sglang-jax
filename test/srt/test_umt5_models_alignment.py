# Copyright 2026 SII Team
# Licensed under the Apache License, Version 2.0

"""Test alignment between JAX UMT5 implementation and PyTorch HuggingFace model."""

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
from transformers import AutoTokenizer, UMT5Config
from transformers import UMT5ForConditionalGeneration as HFGenModel
from transformers import UMT5Model as HFModel

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sgl_jax.srt.configs.model_config import ModelConfig  # noqa: E402
from sgl_jax.srt.model_executor.forward_batch_info import (  # noqa: E402
    ForwardBatch,
    ForwardMode,
)
from sgl_jax.srt.models.umt5 import UMT5EncoderModel  # noqa: E402
from sgl_jax.srt.models.umt5 import UMT5ForConditionalGeneration as UMT5Gen
from sgl_jax.srt.models.umt5 import UMT5Model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DTYPE = {
    "float32": (torch.float32, jnp.float32),
    "bfloat16": (torch.bfloat16, jnp.bfloat16),
}


# =============================================================================
# Utilities
# =============================================================================


def sglang_batch(texts, tokenizer):
    """Create SGLang ForwardBatch."""
    encs = [tokenizer.encode(t, add_special_tokens=True) for t in texts]
    seq_lens = jnp.array([len(e) for e in encs], dtype=jnp.int32)
    ids = jnp.array([t for e in encs for t in e], dtype=jnp.int32)
    return ForwardBatch(
        bid=0,
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(texts),
        input_ids=ids,
        seq_lens=seq_lens,
        extend_seq_lens=seq_lens,
        req_pool_indices=jnp.arange(len(texts), dtype=jnp.int32),
        out_cache_loc=jnp.zeros(ids.shape[0], dtype=jnp.int32),
        positions=jnp.arange(ids.shape[0], dtype=jnp.int32),
    )


def hf_batch(texts, tokenizer):
    """Create HuggingFace padded batch."""
    inp = tokenizer(texts, return_tensors="pt", padding=True)
    return inp.input_ids, inp.attention_mask


def create_enc_dec_batch(sources, targets, tokenizer):
    """Create encoder-decoder batch for JAX model."""
    batch = sglang_batch(sources, tokenizer)
    tgt_encs = [tokenizer.encode(t, add_special_tokens=True) for t in targets]
    tgt_ids = jnp.array([t for e in tgt_encs for t in e], dtype=jnp.int32)
    tgt_lens = [len(e) for e in tgt_encs]
    batch.decoder_input_ids = tgt_ids
    batch.decoder_seq_lens = jnp.array(tgt_lens, dtype=jnp.int32)
    return batch, tgt_lens


def log_output_stats(output, prefix):
    """Log output statistics."""
    if isinstance(output, torch.Tensor):
        arr = output.detach().float().cpu().numpy()
    else:
        arr = np.array(output, dtype=np.float32)

    logger.info("%s shape: %s", prefix, arr.shape)
    logger.info("%s sample [0, :5]: %s", prefix, arr.reshape(-1, arr.shape[-1])[0, :5])
    logger.info(
        "%s stats: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
        prefix,
        arr.mean(),
        arr.std(),
        arr.min(),
        arr.max(),
    )


def compare(output1, output2, name, threshold=1e-3):
    """Compare two outputs (can be PyTorch, JAX, or numpy)."""
    if isinstance(output1, torch.Tensor):
        np1 = output1.detach().float().cpu().numpy()
    else:
        np1 = np.array(output1, dtype=np.float32)

    if isinstance(output2, torch.Tensor):
        np2 = output2.detach().float().cpu().numpy()
    else:
        np2 = np.array(output2, dtype=np.float32)

    mae = np.abs(np1 - np2).mean()
    max_diff = np.abs(np1 - np2).max()

    passed = mae < threshold
    status = "✅" if passed else "❌"
    logger.info("%s %s: MAE=%s, Max=%s", status, name, f"{mae:.2e}", f"{max_diff:.2e}")

    return passed, mae


def compare_batch_sequences(pt_outs, jax_out, seq_lens, name_prefix, threshold=1e-3):
    """Compare batch sequences between PyTorch and JAX outputs."""
    all_pass, total_mae, offset = True, 0.0, 0
    for i, (pt_seq, seq_len) in enumerate(zip(pt_outs, seq_lens)):
        jax_seq = jax_out[offset : offset + seq_len]
        jax_seq_reshaped = (
            jax_seq.reshape(1, -1, jax_seq.shape[-1]) if jax_seq.ndim == 2 else jax_seq
        )
        passed, mae = compare(pt_seq, jax_seq_reshaped, f"{name_prefix} Seq{i+1}", threshold)
        all_pass &= passed
        total_mae += mae
        offset += seq_len
    return all_pass, total_mae / len(pt_outs) if len(pt_outs) > 0 else 0.0


def set_param(model, path, val):
    """Set parameter value in JAX model by path."""
    parts = path.split(".")
    param = model
    for p in parts:
        if p.isdigit():
            param = param[int(p)]
        else:
            param = getattr(param, p)

    if not isinstance(param, nnx.Variable):
        raise ValueError(f"Expected nnx.Variable at {path}, got {type(param)}")

    param[...] = jnp.array(val, dtype=param[...].dtype)


def manual_load_weights(jax_model, hf_model):
    """Fallback manual weight loading from HF model."""
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


def load_models(model_name, mesh, precision, model_type="base"):
    """Load HF and JAX models."""
    pt_dtype, jax_dtype = DTYPE[precision]
    config = UMT5Config.from_pretrained(model_name, dtype=pt_dtype)
    model_config = ModelConfig(model_path=model_name, dtype=precision)

    # Load HF model
    if model_type == "gen":
        hf = HFGenModel.from_pretrained(
            model_name, dtype=pt_dtype, attn_implementation="eager"
        ).eval()
    else:
        hf = HFModel.from_pretrained(model_name, dtype=pt_dtype, attn_implementation="eager").eval()

    # Load JAX model
    with jax.set_mesh(mesh):
        if model_type == "gen":
            jax_m = UMT5Gen(config, mesh, jax_dtype)
        elif model_type == "encoder":
            jax_m = UMT5EncoderModel(config, mesh, jax_dtype)
        else:
            jax_m = UMT5Model(config, mesh, jax_dtype)

        try:
            jax_m.load_weights(model_config)
        except Exception:
            manual_load_weights(jax_m, hf)

    return hf, jax_m, config


# =============================================================================
# Tests
# =============================================================================


def test_encoder(model_name, mesh, tokenizer, precision):
    """Test encoder outputs."""
    logger.info("\n%s\nTest: Encoder\n%s", "=" * 60, "=" * 60)
    hf, jax_m, _ = load_models(model_name, mesh, precision, "encoder")

    texts = ["Hello world", "Machine learning"]
    batch = sglang_batch(texts, tokenizer)
    hf_ids, hf_mask = hf_batch(texts, tokenizer)

    with torch.no_grad():
        hf_h = hf.encoder(input_ids=hf_ids, attention_mask=hf_mask).last_hidden_state

    with jax.set_mesh(mesh):
        jax_out, _, _ = jax_m(batch)

    # Compare first sequence
    seq_len = batch.seq_lens[0].item()
    passed, mae = compare(hf_h[0, :seq_len], jax_out.hidden_states[:seq_len], "Encoder Output")
    return passed, mae


def test_encoder_batch(model_name, mesh, tokenizer, precision):
    """Test encoder with batch sequences."""
    logger.info("\n%s\nTest: Encoder (Batch)\n%s", "=" * 60, "=" * 60)
    hf, jax_m, _ = load_models(model_name, mesh, precision, "encoder")

    texts = ["Short", "A bit longer sentence", "This is an even longer sentence for testing"]

    # PyTorch: process each separately
    pt_outs = []
    for text in texts:
        ids, mask = hf_batch([text], tokenizer)
        with torch.no_grad():
            pt_out = hf.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
            pt_outs.append(pt_out[0])

    # JAX: single batch
    batch = sglang_batch(texts, tokenizer)
    with jax.set_mesh(mesh):
        jax_out, _, _ = jax_m(batch)

    return compare_batch_sequences(
        pt_outs, jax_out.hidden_states, batch.seq_lens.tolist(), "HF vs JAX"
    )


def test_encoder_decoder(model_name, mesh, tokenizer, precision):
    """Test full encoder-decoder model (basic)."""
    logger.info("\n%s\nTest: Encoder-Decoder (Basic)\n%s", "=" * 60, "=" * 60)
    hf, jax_m, _ = load_models(model_name, mesh, precision, "base")

    texts = ["Translate to German: Hello", "Translate to French: World"]
    batch = sglang_batch(texts, tokenizer)

    logger.info("Testing: %s", texts)

    # PyTorch: use input_ids as decoder_input_ids for basic test
    with torch.no_grad():
        hf_ids, hf_mask = hf_batch(texts, tokenizer)
        hf_out = hf(
            input_ids=hf_ids,
            attention_mask=hf_mask,
            decoder_input_ids=hf_ids,
            decoder_attention_mask=hf_mask,
        ).last_hidden_state

    # JAX
    with jax.set_mesh(mesh):
        jax_out = jax_m(batch)

    # Compare first sequence (unpadded)
    seq_len = batch.seq_lens[0].item()
    hf_seq = hf_out[0, :seq_len]
    jax_seq = jax_out[:seq_len]

    logger.info("\n--- Output Statistics ---")
    log_output_stats(hf_seq, "  PyTorch HF")
    log_output_stats(jax_seq, "  JAX")

    passed, mae = compare(hf_seq, jax_seq, "HF vs JAX")
    return passed, mae


def test_encoder_decoder_batch(model_name, mesh, tokenizer, precision):
    """Test encoder-decoder with separate source/target."""
    logger.info("\n%s\nTest: Encoder-Decoder (Batch)\n%s", "=" * 60, "=" * 60)
    hf, jax_m, _ = load_models(model_name, mesh, precision, "base")

    sources = ["Hello world", "Machine learning is great"]
    targets = ["Hallo Welt", "Maschinelles Lernen"]
    pairs = list(zip(sources, targets))

    # PyTorch: process each pair separately
    pt_outs = []
    for src, tgt in pairs:
        src_ids, src_mask = hf_batch([src], tokenizer)
        tgt_ids, tgt_mask = hf_batch([tgt], tokenizer)
        with torch.no_grad():
            pt_out = hf(
                input_ids=src_ids,
                attention_mask=src_mask,
                decoder_input_ids=tgt_ids,
                decoder_attention_mask=tgt_mask,
            ).last_hidden_state
        pt_outs.append(pt_out[0])

    # JAX: single batched forward
    batch, tgt_lens = create_enc_dec_batch(sources, targets, tokenizer)
    logger.info(
        "Batch: %d pairs, encoder: %d tokens, decoder: %d tokens",
        len(pairs),
        batch.input_ids.shape[0],
        sum(tgt_lens),
    )

    with jax.set_mesh(mesh):
        jax_out = jax_m(batch)

    # Log outputs
    logger.info("\n--- Output Statistics ---")
    for i, (pt_seq, seq_len) in enumerate(zip(pt_outs, tgt_lens)):
        logger.info("Pair %d:", i + 1)
        log_output_stats(pt_seq, "  PyTorch HF")

        offset = sum(tgt_lens[:i])
        jax_seq = jax_out[offset : offset + seq_len]
        log_output_stats(jax_seq, "  JAX")

    return compare_batch_sequences(pt_outs, jax_out, tgt_lens, "HF vs JAX", threshold=2e-3)


def greedy_generate(model, tokenizer, src_text, max_length=20, is_hf=True, config=None):
    """Greedy generation (autoregressive)."""
    if is_hf:
        src_ids, src_mask = hf_batch([src_text], tokenizer)
        with torch.no_grad():
            output = model.generate(
                input_ids=src_ids,
                attention_mask=src_mask,
                max_length=max_length,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        return output[0].cpu().numpy()
    else:
        # JAX generation with separate encoder/decoder
        src_batch = sglang_batch([src_text], tokenizer)

        # Encoder forward (only once)
        enc = model.shared(src_batch.input_ids)
        enc_out = model.encoder(enc, src_batch, None, True)
        enc_seq_lens = (
            src_batch.extend_seq_lens
            if hasattr(src_batch, "extend_seq_lens") and src_batch.extend_seq_lens is not None
            else src_batch.seq_lens
        )

        # Decoder generation (loop)
        dec_ids = [config.decoder_start_token_id if config else 0]
        for _ in range(max_length):
            dec_batch = ForwardBatch(
                bid=0,
                forward_mode=ForwardMode.EXTEND,
                batch_size=1,
                input_ids=jnp.array(dec_ids, dtype=jnp.int32),
                seq_lens=jnp.array([len(dec_ids)], dtype=jnp.int32),
                extend_seq_lens=jnp.array([len(dec_ids)], dtype=jnp.int32),
                req_pool_indices=jnp.array([0], dtype=jnp.int32),
                out_cache_loc=jnp.zeros(len(dec_ids), dtype=jnp.int32),
                positions=jnp.arange(len(dec_ids), dtype=jnp.int32),
            )
            dec_batch.encoder_hidden_states = enc_out
            dec_batch.encoder_seq_lens = enc_seq_lens

            logits, _, _ = model(dec_batch)
            next_tok = int(jnp.argmax(logits[-1]))
            dec_ids.append(next_tok)

            if next_tok == (config.eos_token_id if config else tokenizer.eos_token_id):
                break

        return np.array(dec_ids)


def test_generation(model_name, mesh, tokenizer, precision):
    """Test autoregressive generation."""
    logger.info("\n%s\nTest: Generation\n%s", "=" * 60, "=" * 60)
    hf, jax_m, config = load_models(model_name, mesh, precision, "gen")

    src = "Translate to German: Thank you"

    # Generate
    pt_tokens = greedy_generate(hf, tokenizer, src, max_length=20, is_hf=True, config=config)
    with jax.set_mesh(mesh):
        jax_tokens = greedy_generate(
            jax_m, tokenizer, src, max_length=20, is_hf=False, config=config
        )

    # Decode
    pt_text = tokenizer.decode(pt_tokens, skip_special_tokens=True)
    jax_text = tokenizer.decode(jax_tokens, skip_special_tokens=True)

    logger.info("Generated Tokens - HF: %s, JAX: %s", pt_tokens.tolist(), jax_tokens.tolist())
    logger.info("Generated Text - HF: %s, JAX: %s", pt_text, jax_text)

    # Token accuracy
    min_len = min(len(pt_tokens), len(jax_tokens))
    accuracy = (pt_tokens[:min_len] == jax_tokens[:min_len]).mean()
    logger.info("Token Accuracy: %s%%", f"{accuracy * 100:.2f}")

    passed = accuracy >= 0.95
    return passed, 0.0 if passed else 1.0


def test_generation_batch(model_name, mesh, tokenizer, precision):
    """Test generation with batch."""
    logger.info("\n%s\nTest: Generation (Batch)\n%s", "=" * 60, "=" * 60)
    hf, jax_m, config = load_models(model_name, mesh, precision, "gen")

    pairs = [
        ("Translate to German: Hello", "Hallo"),
        ("Translate to German: Thank you", "Danke"),
        ("Translate to German: Good morning", "Guten Morgen"),
    ]
    sources, targets = zip(*pairs)

    # PyTorch: process each pair separately
    pt_logits_list = []
    for src, tgt in pairs:
        src_ids, src_mask = hf_batch([src], tokenizer)
        tgt_ids, tgt_mask = hf_batch([tgt], tokenizer)
        pt_logits = hf(
            input_ids=src_ids,
            attention_mask=src_mask,
            decoder_input_ids=tgt_ids,
            decoder_attention_mask=tgt_mask,
        ).logits
        pt_logits_list.append(pt_logits[0])

    # JAX: single batched forward
    batch, tgt_lens = create_enc_dec_batch(sources, targets, tokenizer)
    with jax.set_mesh(mesh):
        jax_logits, _, _ = jax_m(batch)

    # Compare logits
    all_passed, total_mae = compare_batch_sequences(
        pt_logits_list, jax_logits, tgt_lens, "HF vs JAX", threshold=2e-3
    )

    # Check token accuracy and decode text
    logger.info("\n--- Token Accuracy & Generated Text ---")
    offset = 0
    total_acc = 0
    for i, seq_len in enumerate(tgt_lens):
        pt_seq = pt_logits_list[i]
        jax_seq = jax_logits[offset : offset + seq_len]

        pt_pred = (
            pt_seq.argmax(-1).cpu().numpy()
            if torch.is_tensor(pt_seq)
            else np.array(pt_seq.argmax(-1))
        )
        jax_pred = np.array(jax_seq.argmax(-1))

        accuracy = (pt_pred == jax_pred).mean()
        logger.info("Pair %d - Accuracy: %s%%", i + 1, f"{accuracy:.2%}")
        logger.info("  Predicted tokens (HF): %s", pt_pred.tolist())
        logger.info("  Predicted tokens (JAX): %s", jax_pred.tolist())

        # Decode text
        pt_text = tokenizer.decode(pt_pred, skip_special_tokens=True)
        jax_text = tokenizer.decode(jax_pred, skip_special_tokens=True)

        logger.info("  Expected:    %s", targets[i])
        logger.info("  PyTorch HF:  %s", pt_text)
        logger.info("  JAX:         %s", jax_text)

        total_acc += accuracy
        offset += seq_len

    avg_acc = total_acc / len(tgt_lens)
    logger.info("Average Accuracy: %s%%", f"{avg_acc:.2%}")

    return all_passed and avg_acc >= 0.95, total_mae


# =============================================================================
# Test Registry & Main
# =============================================================================

TESTS = {
    "encoder": test_encoder,
    "encoder_batch": test_encoder_batch,
    "encoder_decoder": test_encoder_decoder,
    "encoder_decoder_batch": test_encoder_decoder_batch,
    "generation": test_generation,
    "generation_batch": test_generation_batch,
}


def main():
    import traceback

    from sgl_jax.srt.utils.mesh_utils import create_device_mesh

    parser = argparse.ArgumentParser(description="UMT5 alignment tests (JAX vs PyTorch)")
    parser.add_argument("--model_path", default="google/umt5-base", help="Model path or HF name")
    parser.add_argument(
        "--test_type", default="all", choices=["all"] + list(TESTS.keys()), help="Test type to run"
    )
    parser.add_argument("--tp_size", type=int, default=None, help="Tensor parallelism size")
    parser.add_argument(
        "--precision", default="float32", choices=["float32", "bfloat16"], help="Model precision"
    )
    args = parser.parse_args()

    # Configure JAX matmul precision
    precision_map = {"float32": "highest", "bfloat16": "default"}
    jax.config.update("jax_default_matmul_precision", precision_map[args.precision])

    logger.info(
        "Model: %s, Test: %s, Precision: %s", args.model_path, args.test_type, args.precision
    )

    # Create mesh
    devices = jax.devices()
    tp = args.tp_size or min(len(devices), 4)
    mesh = create_device_mesh(
        ici_parallelism=[1, tp],
        dcn_parallelism=[1, 1],
        devices=devices[:tp],
        use_explicit_sharding=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Select tests
    tests = list(TESTS.keys()) if args.test_type == "all" else [args.test_type]

    # Run tests
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

    # Summary
    logger.info("\n%s\nSUMMARY\n%s", "=" * 60, "=" * 60)
    for name, passed, mae in results:
        logger.info("%s %s: MAE=%s", "✅" if passed else "❌", name, f"{mae:.2e}")

    all_pass = all(r[1] for r in results)
    logger.info("\n%s (%.1fs)", "✅ All PASSED" if all_pass else "❌ Some FAILED", time.time() - t0)
    return all_pass


if __name__ == "__main__":
    main()
