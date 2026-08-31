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
from torch_cpu_ops import install_cpu_reference_ops, kda_cpu_reference
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or "unknown"


def _load_model(model_path: str, revision: str):
    kwargs = dict(
        revision=revision,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
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
    config = AutoConfig.from_pretrained(
        model_path, revision=revision, trust_remote_code=True
    )
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


def _check_kda_against_fla_naive() -> None:
    """Check the CPU shim against FLA's own unfused PyTorch recurrence."""

    from fla.ops.kda.naive import naive_recurrent_kda

    generator = torch.Generator(device="cpu").manual_seed(20260831)
    shape = (1, 5, 2, 4)
    q = torch.randn(shape, generator=generator)
    k = torch.randn(shape, generator=generator)
    v = torch.randn((1, 5, 2, 3), generator=generator)
    raw_gate = torch.randn(shape, generator=generator)
    raw_beta = torch.randn((1, 5, 2), generator=generator)
    a_log = torch.randn((2,), generator=generator)
    dt_bias = torch.randn((2, 4), generator=generator)
    initial_state = torch.randn((1, 2, 4, 3), generator=generator)
    lower_bound = -5.0

    q_normalized = q * torch.rsqrt(q.square().sum(dim=-1, keepdim=True) + 1e-6)
    k_normalized = k * torch.rsqrt(k.square().sum(dim=-1, keepdim=True) + 1e-6)
    gate = lower_bound * torch.sigmoid(
        a_log.exp().reshape(2, 1) * (raw_gate + dt_bias.reshape(2, 4))
    )
    beta = torch.sigmoid(raw_beta)
    expected_output, expected_state = naive_recurrent_kda(
        q_normalized,
        k_normalized,
        v,
        gate,
        beta,
        initial_state=initial_state,
        output_final_state=True,
    )
    actual_output, actual_state = kda_cpu_reference(
        q,
        k,
        v,
        raw_gate,
        raw_beta,
        A_log=a_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=lower_bound,
    )
    torch.testing.assert_close(actual_output, expected_output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_state, expected_state, rtol=1e-5, atol=1e-5)


def run_cpu_op_self_test(model_path: str, revision: str) -> None:
    _check_kda_against_fla_naive()
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
            raise AssertionError(
                f"Unexpected self-test logits shape: {output.logits.shape}"
            )
        if not torch.isfinite(output.logits.float()).all():
            raise AssertionError(
                "CPU reference op self-test produced non-finite logits"
            )
        decode = model(
            input_ids=torch.tensor([[8]], dtype=torch.long),
            attention_mask=torch.ones((1, input_ids.shape[1] + 1), dtype=torch.long),
            past_key_values=output.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        if decode.logits.shape != (1, 1, config.vocab_size):
            raise AssertionError(
                f"Unexpected decode logits shape: {decode.logits.shape}"
            )
        if not torch.isfinite(decode.logits.float()).all():
            raise AssertionError(
                "CPU reference op decode self-test produced non-finite logits"
            )
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


def _install_prefill_capture_hooks(model):
    captures: dict[str, dict[int, torch.Tensor]] = {
        "attention_input": {},
        "attention_output": {},
        "moe_input": {},
        "mlp_output": {},
        "router_raw_logits": {},
        "router_scores": {},
        "router_topk_ids": {},
        "router_topk_weights": {},
    }
    handles = []

    def capture_hidden(name: str, layer_index: int):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            captures[name][layer_index] = (
                value.reshape(-1, value.shape[-1])[-1].detach().float().cpu()
            )

        return hook

    def capture_router(layer_index: int):
        def hook(_module, _inputs, output):
            topk_ids, topk_weights, raw_logits = output
            raw_logits = (
                raw_logits.reshape(-1, raw_logits.shape[-1])[-1].detach().float().cpu()
            )
            captures["router_raw_logits"][layer_index] = raw_logits
            captures["router_scores"][layer_index] = torch.sigmoid(raw_logits)
            captures["router_topk_ids"][layer_index] = (
                topk_ids.reshape(-1, topk_ids.shape[-1])[-1].detach().cpu()
            )
            captures["router_topk_weights"][layer_index] = (
                topk_weights.reshape(-1, topk_weights.shape[-1])[-1]
                .detach()
                .float()
                .cpu()
            )

        return hook

    layers = list(model.model.layers[: model.config.num_hidden_layers])
    for layer_index, layer in enumerate(layers):
        handles.append(
            layer.input_layernorm.register_forward_hook(
                capture_hidden("attention_input", layer_index)
            )
        )
        handles.append(
            layer.attention.register_forward_hook(
                capture_hidden("attention_output", layer_index)
            )
        )
        handles.append(
            layer.post_attention_layernorm.register_forward_hook(
                capture_hidden("moe_input", layer_index)
            )
        )
        handles.append(
            layer.mlp.register_forward_hook(capture_hidden("mlp_output", layer_index))
        )
        gate = getattr(layer.mlp, "gate", None)
        if gate is not None:
            handles.append(gate.register_forward_hook(capture_router(layer_index)))
    return layers, captures, handles


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
    prefill_components = None
    capture_layers, component_captures, capture_handles = (
        _install_prefill_capture_hooks(model)
    )

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
                for handle in capture_handles:
                    handle.remove()
                first_token_logits = logits.cpu().numpy()
                prefill_hidden_states = torch.stack(
                    [state[0, -1].float().cpu() for state in output.hidden_states]
                ).numpy()
                num_layers = len(capture_layers)
                topk = int(model.config.num_experts_per_tok)
                prefill_components = {
                    name: torch.stack(
                        [component_captures[name][index] for index in range(num_layers)]
                    ).numpy()
                    for name in (
                        "attention_input",
                        "attention_output",
                        "moe_input",
                        "mlp_output",
                    )
                }
                num_experts = int(model.config.num_experts)
                for name in ("router_raw_logits", "router_scores"):
                    prefill_components[name] = torch.stack(
                        [
                            component_captures[name].get(
                                index,
                                torch.full(
                                    (num_experts,), torch.nan, dtype=torch.float32
                                ),
                            )
                            for index in range(num_layers)
                        ]
                    ).numpy()
                prefill_components["router_topk_ids"] = torch.stack(
                    [
                        component_captures["router_topk_ids"].get(
                            index, torch.full((topk,), -1, dtype=torch.long)
                        )
                        for index in range(num_layers)
                    ]
                ).numpy()
                prefill_components["router_topk_weights"] = torch.stack(
                    [
                        component_captures["router_topk_weights"].get(
                            index, torch.full((topk,), torch.nan, dtype=torch.float32)
                        )
                        for index in range(num_layers)
                    ]
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

    result = {
        "input_ids": input_ids[0].cpu().numpy().astype(np.int32),
        "first_token_logits": first_token_logits.astype(np.float32),
        "prefill_hidden_states": prefill_hidden_states.astype(np.float32),
        "step_topk_ids": np.stack(topk_ids).astype(np.int32),
        "step_topk_logprobs": np.stack(topk_logprobs).astype(np.float32),
        "greedy_token_ids": np.asarray(greedy_ids, dtype=np.int32),
    }
    for name, values in prefill_components.items():
        dtype = np.int32 if name == "router_topk_ids" else np.float32
        result[f"prefill_{name}"] = values.astype(dtype)
    return result


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
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, revision=revision, trust_remote_code=True
    )
    load_started = time.perf_counter()
    model = _load_model(model_path, revision)
    if model.config._attn_implementation != "eager":
        raise RuntimeError(
            "Torch CPU golden requires a causal eager attention mask, got "
            f"{model.config._attn_implementation!r}"
        )
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
        "attn_implementation": model.config._attn_implementation,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "aten_cpu_capability": os.environ.get("ATEN_CPU_CAPABILITY"),
        "torch_num_threads": torch.get_num_threads(),
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
