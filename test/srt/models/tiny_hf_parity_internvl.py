"""Tiny-random InternVL parity: sgl_jax Engine (CPU) vs HF transformers (CPU, fp32).

usage: python cpu_parity_vlm.py <outdir>
"""

import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, InternVLForConditionalGeneration

OUT = sys.argv[1]
BASE = "OpenGVLab/InternVL3-2B-hf"
N_NEW = 12
LOGP_TOL = float(os.environ.get("LOGP_TOL", "1e-4"))


def images():
    rng = np.random.default_rng(0)
    # 600x900 -> several crops + thumbnail; 300x300 -> single crop
    a = Image.fromarray((rng.random((600, 900, 3)) * 255).astype("uint8"))
    yy, xx = np.mgrid[0:300, 0:300]
    b = Image.fromarray(np.stack([xx % 256, yy % 256, (xx + yy) % 256], -1).astype("uint8"))
    return a, b


def requests(proc):
    a, b = images()
    def chat(n_img, text):
        content = [{"type": "image"}] * n_img + [{"type": "text", "text": text}]
        return proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
    return [
        (chat(1, "Describe this image."), [a]),
        (chat(1, "What is in the picture?"), [b]),
        (chat(2, "Compare the two images."), [a, b]),
        (chat(0, "Say hello."), []),
    ]


def build():
    cfg = AutoConfig.from_pretrained(BASE)
    t, v = cfg.text_config, cfg.vision_config
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = 256, 512, 2
    t.num_attention_heads, t.num_key_value_heads = 2, 1
    t.head_dim = 128
    if getattr(t, "layer_types", None):
        t.layer_types = t.layer_types[: t.num_hidden_layers]
    v.hidden_size, v.intermediate_size, v.num_hidden_layers, v.num_attention_heads = 64, 128, 2, 2
    cfg.dtype = "float32"
    torch.manual_seed(0)
    model = InternVLForConditionalGeneration(cfg).float().eval()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lambda" in n:
                p.fill_(1.0)
            elif p.dim() == 1:
                p.normal_(0.0, 0.05)
            elif "position_embeddings" in n or "cls_token" in n:
                p.normal_(0.0, 0.5)
            else:
                p.normal_(0.0, 0.08)
        # make the LM attend to image tokens strongly so vision errors move logits
        for lyr in model.model.language_model.layers:
            lyr.self_attn.q_proj.weight.normal_(0.0, 0.25)
            lyr.self_attn.k_proj.weight.normal_(0.0, 0.25)
    model.save_pretrained(OUT, safe_serialization=True)
    AutoProcessor.from_pretrained(BASE).save_pretrained(OUT)
    return model


def hf_ref(model, proc):
    res = []
    for text, imgs in requests(proc):
        enc = proc(text=text, images=imgs or None, return_tensors="pt")
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
        mem_fraction_static=0.2,
        chunked_prefill_size=4096,
        precompile_bs_paddings=[4],
        precompile_token_paddings=[4096],
        max_running_requests=4,
        skip_server_warmup=True,
        attention_backend="fa",
        page_size=16,
        log_requests=False,
        disable_overlap_schedule=True,
        watchdog_timeout=3000,
        grammar_backend="none",
        disable_precompile=True,
        # CPU native attention scans the whole KV pool; keep it small.
        max_total_tokens=8192,
    )
    res = []
    for text, imgs in requests(proc):
        out = engine.generate(
            prompt=text,
            image_data=imgs or None,
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
    json.dump(dict(arch="internvl", all_match=all_match, requests=report),
              open(os.path.join(OUT, "parity.json"), "w"), indent=1)
    print("ALL_MATCH" if all_match else "MISMATCH")
