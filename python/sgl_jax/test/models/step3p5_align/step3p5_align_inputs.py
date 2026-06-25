"""Shared fixed inputs for the real-weight HF<->sglang-jax alignment.

Both the HF reference dump (GPU box) and the sglang-jax dump (TPU) load the SAME
input_ids from inputs.json so the comparison is apples-to-apples. We send raw
token ids (no tokenizer) so there is zero tokenizer/template skew between sides —
the point is implementation EQUIVALENCE, not output quality, so seeded-random
prompts are fine.

Run once to (re)generate inputs.json:
    python step3p5_align_inputs.py --vocab 128896 --num-prompts 64 --len 24
"""

import argparse
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS_JSON = os.path.join(_HERE, "inputs.json")


def generate(vocab: int, num_prompts: int, length: int, seed: int = 1234) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    # Keep ids well inside vocab; avoid 0 (often bos/pad) as the very first token.
    return [[int(x) for x in rng.integers(1, vocab, size=length)] for _ in range(num_prompts)]


def load() -> list[list[int]]:
    with open(INPUTS_JSON) as f:
        return json.load(f)["input_ids"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=128896)
    ap.add_argument("--num-prompts", type=int, default=64)
    ap.add_argument("--len", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    data = generate(args.vocab, args.num_prompts, args.len, args.seed)
    with open(INPUTS_JSON, "w") as f:
        json.dump({"vocab": args.vocab, "seed": args.seed, "input_ids": data}, f)
    print(f"wrote {len(data)} prompts (len {args.len}) -> {INPUTS_JSON}")
