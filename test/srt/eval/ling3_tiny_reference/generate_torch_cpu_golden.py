#!/usr/bin/env python3
"""Generate Ling3 Tiny logits/token goldens with the official HF model on CPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch_cpu_ops import install_cpu_reference_ops
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(model_path: str, revision: str):
    kwargs = dict(
        revision=revision,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            **kwargs,
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            **kwargs,
        )


def _make_reduced_config(model_path: str, revision: str):
    config = AutoConfig.from_pretrained(model_path, revision=revision, trust_remote_code=True)
    overrides = {
        "vocab_size": 64,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 8,
        "v_head_dim": 8,
        "qk_head_dim": 8,
        "qk_nope_head_dim": 4,
        "qk_rope_head_dim": 4,
        "rotary_dim": 4,
        "q_lora_rank": 8,
        "kv_lora_rank": 8,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "num_shared_experts": 1,
        "moe_intermediate_size": 16,
        "moe_shared_expert_intermediate_size": 16,
        "n_group": 2,
        "topk_group": 1,
        "layer_group_size": 4,
        "max_position_embeddings": 128,
    }
    for key, value in overrides.items():
        setattr(config, key, value)
    config._attn_implementation = "eager"
    return config


def run_cpu_op_self_test(model_path: str, revision: str) -> None:
    config = _make_reduced_config(model_path, revision)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).to(
        device="cpu", dtype=torch.bfloat16
    )
    install_cpu_reference_ops(model)
    model.eval()
    input_ids = torch.tensor([[1, 5, 9, 3, 7, 2, 4, 6]], dtype=torch.long)
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True, return_dict=True)
        if output.logits.shape != (1, input_ids.shape[1], config.vocab_size):
            raise AssertionError(f"Unexpected self-test logits shape: {output.logits.shape}")
        if not torch.isfinite(output.logits.float()).all():
            raise AssertionError("CPU reference op self-test produced non-finite logits")
        decode = model(
            input_ids=torch.tensor([[8]], dtype=torch.long),
            attention_mask=torch.ones((1, input_ids.shape[1] + 1), dtype=torch.long),
            past_key_values=output.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        if decode.logits.shape != (1, 1, config.vocab_size):
            raise AssertionError(f"Unexpected decode logits shape: {decode.logits.shape}")
        if not torch.isfinite(decode.logits.float()).all():
            raise AssertionError("CPU reference op decode self-test produced non-finite logits")
    print("CPU reference op self-test passed", flush=True)


def _render_input_ids(tokenizer, case: dict, enable_thinking: bool) -> torch.Tensor:
    tokenized = tokenizer.apply_chat_template(
        case["messages"],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        return_tensors="pt",
    )
    if not isinstance(tokenized, torch.Tensor):
        tokenized = torch.tensor([tokenized], dtype=torch.long)
    if tokenized.ndim == 1:
        tokenized = tokenized.unsqueeze(0)
    return tokenized.to(device="cpu", dtype=torch.long)


def _generate_case(
    model,
    tokenizer,
    case: dict,
    enable_thinking: bool,
    max_new_tokens: int,
    top_k: int,
) -> dict[str, np.ndarray]:
    input_ids = _render_input_ids(tokenizer, case, enable_thinking)
    attention_mask = torch.ones_like(input_ids)
    current_ids = input_ids
    past_key_values = None
    greedy_ids = []
    topk_ids = []
    topk_logprobs = []
    first_token_logits = None
    prefill_hidden_states = None

    with torch.inference_mode():
        for step in range(max_new_tokens):
            output = model(
                input_ids=current_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=step == 0,
                return_dict=True,
            )
            logits = output.logits[0, -1].float()
            if step == 0:
                first_token_logits = logits.cpu().numpy()
                prefill_hidden_states = torch.stack(
                    [state[0, -1].float().cpu() for state in output.hidden_states]
                ).numpy()

            logprobs = torch.log_softmax(logits, dim=-1)
            values, indices = torch.topk(logprobs, k=top_k)
            topk_ids.append(indices.cpu().numpy())
            topk_logprobs.append(values.cpu().numpy())

            next_token = int(torch.argmax(logits).item())
            greedy_ids.append(next_token)
            past_key_values = output.past_key_values
            if next_token == tokenizer.eos_token_id:
                break
            current_ids = torch.tensor([[next_token]], dtype=torch.long)
            attention_mask = torch.cat(
                (attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype)), dim=1
            )

    return {
        "input_ids": input_ids[0].cpu().numpy().astype(np.int32),
        "first_token_logits": first_token_logits.astype(np.float32),
        "prefill_hidden_states": prefill_hidden_states.astype(np.float32),
        "step_topk_ids": np.stack(topk_ids).astype(np.int32),
        "step_topk_logprobs": np.stack(topk_logprobs).astype(np.float32),
        "greedy_token_ids": np.asarray(greedy_ids, dtype=np.int32),
    }


def parse_args() -> argparse.Namespace:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--prompts", type=Path, default=directory / "prompts.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--self-test-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_spec = json.loads(args.prompts.read_text(encoding="utf-8"))
    model_path = args.model_path or prompt_spec["model_id"]
    revision = prompt_spec["revision"]
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(max(1, min(4, args.threads)))

    run_cpu_op_self_test(model_path, revision)
    if args.self_test_only:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision, trust_remote_code=True)
    load_started = time.perf_counter()
    model = _load_model(model_path, revision)
    patches = install_cpu_reference_ops(model)
    model.eval()
    load_seconds = time.perf_counter() - load_started

    modeling_module = importlib.import_module(model.__class__.__module__)
    modeling_source = Path(inspect.getsourcefile(modeling_module) or "")
    artifacts = []
    for case in prompt_spec["cases"]:
        started = time.perf_counter()
        arrays = _generate_case(
            model,
            tokenizer,
            case,
            bool(prompt_spec["enable_thinking"]),
            int(prompt_spec["max_new_tokens"]),
            int(prompt_spec["top_k"]),
        )
        path = args.output_dir / f"{case['name']}.npz"
        np.savez_compressed(path, **arrays)
        record = {
            "name": case["name"],
            "artifact": path.name,
            "sha256": _sha256(path),
            "input_tokens": int(arrays["input_ids"].shape[0]),
            "generated_tokens": int(arrays["greedy_token_ids"].shape[0]),
            "elapsed_seconds": time.perf_counter() - started,
        }
        artifacts.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    manifest = {
        "schema_version": 1,
        "golden": "official_hf_model_with_pure_torch_cpu_fla_primitives",
        "model_id": prompt_spec["model_id"],
        "model_path": model_path,
        "revision": revision,
        "enable_thinking": prompt_spec["enable_thinking"],
        "max_new_tokens": prompt_spec["max_new_tokens"],
        "top_k": prompt_spec["top_k"],
        "modeling_source": str(modeling_source),
        "modeling_source_sha256": _sha256(modeling_source),
        "cpu_op_replacements": patches,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "load_seconds": load_seconds,
        "artifacts": artifacts,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
