"""Offline compare of HF (reference) vs sglang-jax dumps. No accelerator needed.

A (end-to-end, decision-level): per prompt, does the next-token argmax agree?
  what is the top-k set overlap? This is the real-weight correctness signal —
  if HF and my implementation pick the same next token on the same real weights
  across many prompts, the full forward is equivalent at the decision level.
  (bf16 on both sides + different kernels => exact logprobs differ; argmax / top-k
  overlap is the criterion, like flash==naive.)

B (per-layer, if both sides dumped B_layer*.npy): per-layer max relative error of
  the last-position hidden state -> a growth curve that localizes any divergence.

Usage:
    python compare.py --hf hf_dump --jax jax_dump
"""

import argparse
import glob
import json
import os

import numpy as np


def _load_A(d):
    with open(os.path.join(d, "A_nexttok.json")) as f:
        data = json.load(f)["data"]
    return {r["prompt"]: r["topk"] for r in data}


def compare_A(hf_dir, jax_dir):
    hf, jx = _load_A(hf_dir), _load_A(jax_dir)
    common = sorted(set(hf) & set(jx))
    if not common:
        print("A: no common prompts")
        return
    # jax dump may be argmax-only (1 token/prompt) — the generation path avoids the
    # multi-host logprob bug; argmax agreement is still the main criterion.
    argmax_only = all(len(jx[p]) == 1 for p in common)
    argmax_agree = 0
    jx_in_hf_topk = 0
    overlaps = []
    logprob_absdiffs = []
    for p in common:
        hf_tk, jx_tk = hf[p], jx[p]
        hf_top1 = max(hf_tk, key=lambda x: x[1])[0]
        jx_top1 = max(jx_tk, key=lambda x: x[1])[0]
        argmax_agree += int(hf_top1 == jx_top1)
        hf_set = {t for t, _ in hf_tk}
        jx_in_hf_topk += int(jx_top1 in hf_set)
        if not argmax_only:
            jx_set = {t for t, _ in jx_tk}
            overlaps.append(len(hf_set & jx_set) / max(len(hf_set), 1))
            hf_lp, jx_lp = dict(hf_tk), dict(jx_tk)
            for t in hf_set & jx_set:
                logprob_absdiffs.append(abs(hf_lp[t] - jx_lp[t]))
    n = len(common)
    print("=== A: end-to-end next-token (HF vs jax, real weights) ===")
    print(f" prompts compared      : {n}" + ("   [jax: argmax-only]" if argmax_only else ""))
    print(f" argmax agreement      : {argmax_agree}/{n} = {argmax_agree / n:.4f}")
    print(f" jax argmax in HF top-k: {jx_in_hf_topk}/{n} = {jx_in_hf_topk / n:.4f}")
    if not argmax_only:
        print(f" mean top-k overlap    : {np.mean(overlaps):.4f}")
        if logprob_absdiffs:
            a = np.array(logprob_absdiffs)
            print(f" logprob |diff| on shared tokens: mean={a.mean():.4e} max={a.max():.4e}")
    print(
        " criterion: argmax agreement ~1.0 (a few borderline flips OK = bf16 noise);"
        " low agreement => real implementation divergence."
    )


def compare_B(hf_dir, jax_dir):
    hf_files = sorted(glob.glob(os.path.join(hf_dir, "B_layer*.npy")))
    if not hf_files:
        return
    print("\n=== B: per-layer last-position hidden (growth curve) ===")
    print(" layer |   max_rel_err | verdict")
    for hf_f in hf_files:
        name = os.path.basename(hf_f)
        jx_f = os.path.join(jax_dir, name)
        if not os.path.exists(jx_f):
            print(f" {name}: jax dump missing (skip)")
            continue
        h = np.load(hf_f).astype(np.float64)
        j = np.load(jx_f).astype(np.float64)
        rel = np.max(np.abs(h - j)) / (np.max(np.abs(h)) + 1e-9)
        k = name.replace("B_layer", "").replace(".npy", "")
        print(f" {k:>5} | {rel:>12.4e} | {'ok' if rel < 2e-2 else 'CHECK'}")
    print(" bf16 dumps => ~1e-2 expected; a single layer with a big jump localizes a bug.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--jax", required=True)
    args = ap.parse_args()
    compare_A(args.hf, args.jax)
    compare_B(args.hf, args.jax)
