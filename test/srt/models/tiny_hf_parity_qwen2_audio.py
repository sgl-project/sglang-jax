"""Tiny-random Qwen2-Audio parity: sgl_jax Engine (CPU) vs HF transformers (CPU, fp32).

usage: python cpu_parity_audio.py <outdir>
"""

import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch
from transformers import AutoConfig, AutoProcessor, Qwen2AudioForConditionalGeneration

OUT = sys.argv[1]
BASE = "Qwen/Qwen2-Audio-7B-Instruct"
N_NEW = 12
LOGP_TOL = float(os.environ.get("LOGP_TOL", "1e-4"))
SR = 16000


def clips():
    rng = np.random.default_rng(0)
    t = np.arange(int(3.3 * SR)) / SR
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    noise = (0.1 * rng.standard_normal(int(7.1 * SR))).astype(np.float32)
    return tone, noise


def requests(proc):
    tone, noise = clips()

    def chat(n, text):
        content = [{"type": "audio", "audio_url": "x"}] * n + [{"type": "text", "text": text}]
        return proc.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False
        )

    return [
        (chat(1, "What do you hear?"), [tone]),
        (chat(1, "Transcribe."), [noise]),
        (chat(2, "Compare the two clips."), [tone, noise]),
        (chat(0, "Say hello."), []),
    ]


def build():
    cfg = AutoConfig.from_pretrained(BASE)
    t, a = cfg.text_config, cfg.audio_config
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = 256, 512, 2
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 2, 1, 128
    if getattr(t, "layer_types", None):
        t.layer_types = t.layer_types[: t.num_hidden_layers]
    a.d_model, a.encoder_ffn_dim, a.encoder_layers, a.encoder_attention_heads = 64, 128, 2, 2
    cfg.dtype = "float32"
    torch.manual_seed(0)
    model = Qwen2AudioForConditionalGeneration(cfg).float().eval()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.dim() == 1:
                p.normal_(0.0, 0.05) if "norm" not in n else p.normal_(1.0, 0.1)
            else:
                p.normal_(0.0, 0.08)
        for lyr in model.model.language_model.layers:
            lyr.self_attn.q_proj.weight.normal_(0.0, 0.25)
            lyr.self_attn.k_proj.weight.normal_(0.0, 0.25)
    model.save_pretrained(OUT, safe_serialization=True)
    AutoProcessor.from_pretrained(BASE).save_pretrained(OUT)
    return model


def hf_ref(model, proc):
    res = []
    for text, auds in requests(proc):
        kw = dict(text=[text], return_tensors="pt")
        if auds:
            kw.update(audio=auds, sampling_rate=SR)
        enc = proc(**kw)
        out = model.generate(
            **enc, max_new_tokens=N_NEW, do_sample=False, output_scores=True, return_dict_in_generate=True
        )
        new = out.sequences[0, enc["input_ids"].shape[1]:].tolist()
        lps = [torch.log_softmax(s[0].float(), -1)[t].item() for s, t in zip(out.scores, new)]
        res.append(dict(tokens=new, logprobs=lps, n_input=enc["input_ids"].shape[1]))
    return res


def sgl(proc):
    from sgl_jax.srt.entrypoints.engine import Engine

    engine = Engine(
        model_path=OUT,
        tp_size=1,
        device="cpu",
        dtype="float32",
        random_seed=3,
        mem_fraction_static=0.45,
        chunked_prefill_size=1024,
        precompile_bs_paddings=[4],
        precompile_token_paddings=[1024],
        max_running_requests=4,
        skip_server_warmup=True,
        attention_backend="fa",
        page_size=16,
        log_requests=False,
        disable_overlap_schedule=True,
        watchdog_timeout=3000,
        grammar_backend="none",
        max_total_tokens=8192,
    )
    res = []
    for text, auds in requests(proc):
        out = engine.generate(
            prompt=text,
            audio_data=[(a, SR) for a in auds] or None,
            sampling_params={"max_new_tokens": N_NEW, "temperature": 0.0},
            return_logprob=True,
        )
        meta = out["meta_info"]
        res.append(
            dict(
                tokens=[x[1] for x in meta["output_token_logprobs"]],
                logprobs=[x[0] for x in meta["output_token_logprobs"]],
                n_input=meta.get("prompt_tokens"),
            )
        )
    engine.shutdown()
    return res


if __name__ == "__main__":
    model = build()
    proc = AutoProcessor.from_pretrained(OUT)
    ref = hf_ref(model, proc)
    del model
    got = sgl(proc)
    all_match, report = True, []
    for i, (r, g) in enumerate(zip(ref, got)):
        n = min(len(r["tokens"]), len(g["tokens"]))
        first_div = next((j for j in range(n) if r["tokens"][j] != g["tokens"][j]), None)
        upto = n if first_div is None else first_div
        diff = float(np.max(np.abs(np.subtract(r["logprobs"][:upto], g["logprobs"][:upto])))) if upto else 0.0
        match = first_div is None and len(r["tokens"]) == len(g["tokens"]) and diff < LOGP_TOL
        all_match &= match
        report.append(dict(request=i, greedy_match=match, first_divergence=first_div,
                           max_logprob_diff=diff, prompt_tokens=(r["n_input"], g["n_input"])))
        print(f"request {i}: match={match} first_div={first_div} max|dlogp|={diff:.2e} "
              f"prompt_tokens hf={r['n_input']} sgl={g['n_input']}")
    json.dump(dict(arch="qwen2_audio", all_match=all_match, requests=report),
              open(os.path.join(OUT, "parity.json"), "w"), indent=1)
    print("ALL_MATCH" if all_match else "MISMATCH")
