"""HF reference dump on the REAL Step-3.5-Flash weights (run on the 8xH100 box).

Produces the reference side of two comparisons:
  A. end-to-end next-token distribution per prompt (full model, bf16) — the
     comprehensive real-weight check (decision-level: argmax + top-k).
  B. per-layer hidden states (full model, bf16) — a growth-curve so any HF<->jax
     divergence is localized to a layer.

Both run from ONE bf16 full-model forward (196B fits on 8xH100 = 640GB; fp32 would
be 784GB and does NOT fit — full-model fp32 is intentionally not attempted). The
tight fp32 per-layer check is a separate isolated-layer script.

Usage (on the GPU box):
    pip install torch transformers accelerate safetensors
    python dump_hf.py --ckpt <CHECKPOINT_PATH> --out hf_dump --topk 20 \
        --layers 0 1 4 43 44          # which layers to save hidden states for
    # smoke first: add --num-prompts 1

Notes / risks:
  - device_map="auto" must shard the custom MoE correctly; if it errors on the
    288-expert layers, pass --device-map balanced_low_0 or a manual map.
  - trust_remote_code loads modeling_step3p5.py from the checkpoint dir (or set
    --hf-src to a dir containing modeling_step3p5.py / configuration_step3p5.py).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to real Step-3.5-Flash checkpoint dir")
    ap.add_argument("--out", default="hf_dump", help="output dir")
    ap.add_argument("--inputs", default=None, help="inputs.json (default: alongside this script)")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--num-prompts", type=int, default=None, help="limit prompts (smoke)")
    ap.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=[0, 1, 4, 43, 44],
        help="layer indices to save hidden states for (B)",
    )
    ap.add_argument("--device-map", default="auto")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    inputs_path = args.inputs or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "inputs.json"
    )
    with open(inputs_path) as f:
        input_ids = json.load(f)["input_ids"]
    if args.num_prompts:
        input_ids = input_ids[: args.num_prompts]
    print(f"[hf] {len(input_ids)} prompts, ckpt={args.ckpt}", flush=True)

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()
    emb_dev = model.get_input_embeddings().weight.device

    # A: next-token top-k per prompt. B: per-layer hidden states (mean over layers req'd).
    a_out = []  # [{"prompt": i, "topk": [[tid, logprob], ...]}]
    # B accumulates the per-layer hidden state at the LAST position for each prompt,
    # stacked -> [num_prompts, hidden] per requested layer.
    b_hidden = {k: [] for k in args.layers}

    for i, ids in enumerate(input_ids):
        t = torch.tensor([ids], dtype=torch.long, device=emb_dev)
        with torch.no_grad():
            out = model(input_ids=t, output_hidden_states=True, use_cache=False)
        logits = out.logits[0, -1].float()  # [vocab], next-token after the prompt
        logprobs = torch.log_softmax(logits, dim=-1)
        vals, idx = torch.topk(logprobs, args.topk)
        a_out.append({"prompt": i, "topk": [[int(j), float(v)] for j, v in zip(idx, vals)]})
        # hidden_states: tuple len num_layers+1; [l] is the OUTPUT of layer l-1
        # (index 0 = embeddings). We save the output of layer k = hidden_states[k+1].
        for k in args.layers:
            h = out.hidden_states[k + 1][0, -1].float().cpu().numpy()  # [hidden]
            b_hidden[k].append(h)
        if (i + 1) % 8 == 0:
            print(f"[hf] {i + 1}/{len(input_ids)}", flush=True)

    with open(os.path.join(args.out, "A_nexttok.json"), "w") as f:
        json.dump({"topk": args.topk, "data": a_out}, f)
    for k, lst in b_hidden.items():
        np.save(os.path.join(args.out, f"B_layer{k}.npy"), np.stack(lst))
    print(f"[hf] wrote A_nexttok.json + B_layer*.npy to {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
