"""Tiny-random-model parity: sgl_jax Engine (CPU) vs HF transformers (CPU, fp32).

Builds a randomly initialised HF model from a small config, saves it, then
compares greedy continuations and per-token logprobs.

usage: python cpu_parity.py <arch> <outdir>
"""

import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ARCH, OUT = sys.argv[1], sys.argv[2]

TINY = {
    "glm4": dict(
        base="zai-org/GLM-4-9B-0414",
        over=dict(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=128,
        ),
    ),
    "gemma3": dict(
        base="google/gemma-3-1b-it",
        over=dict(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=6,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=128,
            sliding_window=16,
        ),
    ),
    # 4B text tower: global layers use linear RoPE scaling (factor 8)
    "gemma3lin": dict(
        base="google/gemma-3-4b-it",
        text_config=True,
        tokenizer="google/gemma-3-1b-it",
        over=dict(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=6,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=128,
            sliding_window=16,
            vocab_size=262144,
        ),
    ),
}[ARCH]

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "In a distant galaxy, a robot named",
    # long enough to cross the gemma3 tiny sliding window
    "Tool result: " + " ".join(f"item{i}" for i in range(40)) + ". Summary:",
]
N_NEW = 16
LOGP_TOL = float(os.environ.get("LOGP_TOL", "1e-4"))  # fp32 CPU


def build():
    cfg = AutoConfig.from_pretrained(TINY["base"])
    if TINY.get("text_config"):
        cfg = cfg.text_config
        cfg.architectures = ["Gemma3ForCausalLM"]
    for k, v in TINY["over"].items():
        setattr(cfg, k, v)
    if hasattr(cfg, "layer_types") and cfg.layer_types is not None:
        pat = getattr(cfg, "sliding_window_pattern", 6)
        cfg.layer_types = [
            "full_attention" if (i + 1) % pat == 0 else "sliding_attention"
            for i in range(cfg.num_hidden_layers)
        ]
    cfg.torch_dtype = "float32"
    cfg.dtype = "float32"
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    # random init with larger norm weights so layers actually matter
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "norm" in n:
                p.normal_(0.0, 0.3)
            elif "q_proj.weight" in n or "k_proj.weight" in n:
                # sharpen attention so positional (RoPE) errors change argmax
                p.normal_(0.0, 0.25)
            elif p.dim() == 1:
                p.normal_(0.0, 0.05)
    model.save_pretrained(OUT, safe_serialization=True)
    AutoTokenizer.from_pretrained(TINY.get("tokenizer", TINY["base"])).save_pretrained(OUT)
    return model


def hf_ref(model, tok):
    res = []
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids
        out = model.generate(
            ids,
            max_new_tokens=N_NEW,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
        new = out.sequences[0, ids.shape[1] :].tolist()
        lps = [
            torch.log_softmax(s[0].float(), -1)[t].item() for s, t in zip(out.scores, new)
        ]
        res.append(dict(input_ids=ids[0].tolist(), tokens=new, logprobs=lps))
    return res


def sgl(ref):
    from sgl_jax.srt.entrypoints.engine import Engine

    engine = Engine(
        model_path=OUT,
        tp_size=1,
        device="cpu",
        dtype="float32",
        random_seed=3,
        mem_fraction_static=0.2,
        chunked_prefill_size=128,
        precompile_bs_paddings=[4],
        precompile_token_paddings=[128],
        max_running_requests=4,
        skip_server_warmup=True,
        attention_backend="fa",
        page_size=16,
        log_requests=False,
        disable_overlap_schedule=True,
        watchdog_timeout=3000,
        grammar_backend="none",
        # CPU uses the native attention backend, which lacks the hybrid SWA pool.
        disable_hybrid_swa_memory=True,
    )
    outs = engine.generate(
        input_ids=[r["input_ids"] for r in ref],
        sampling_params={"max_new_tokens": N_NEW, "temperature": 0.0},
        return_logprob=True,
    )
    engine.shutdown()
    res = []
    for o in outs:
        meta = o["meta_info"]
        lp = [x[0] for x in meta["output_token_logprobs"]]
        toks = [x[1] for x in meta["output_token_logprobs"]]
        res.append(dict(tokens=toks, logprobs=lp))
    return res


if __name__ == "__main__":
    model = build()
    tok = AutoTokenizer.from_pretrained(OUT)
    ref = hf_ref(model.eval(), tok)
    del model
    got = sgl(ref)
    report = []
    all_match = True
    for i, (r, g) in enumerate(zip(ref, got)):
        n = min(len(r["tokens"]), len(g["tokens"]))
        first_div = next((j for j in range(n) if r["tokens"][j] != g["tokens"][j]), None)
        upto = first_div if first_div is not None else n
        diff = (
            float(np.max(np.abs(np.array(r["logprobs"][:upto]) - np.array(g["logprobs"][:upto]))))
            if upto
            else 0.0
        )
        match = (
            first_div is None and len(r["tokens"]) == len(g["tokens"]) and diff < LOGP_TOL
        )
        all_match &= match
        report.append(
            dict(prompt=i, greedy_match=match, first_divergence=first_div, max_logprob_diff=diff)
        )
        print(f"prompt {i}: match={match} first_div={first_div} max|dlogp|={diff:.2e}")
    json.dump(dict(arch=ARCH, all_match=all_match, prompts=report), open(OUT + "/parity.json", "w"), indent=1)
    print("ALL_MATCH" if all_match else "MISMATCH")
