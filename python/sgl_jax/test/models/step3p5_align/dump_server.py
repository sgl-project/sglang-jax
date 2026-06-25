"""sglang-jax side (A): next-token top-k per prompt via the RUNNING server.

Hits /generate with the same input_ids as dump_hf.py and records the next-token
top-k logprob distribution (decision-level). Pairs with dump_hf.py's A_nexttok.json.

This uses the already-running real-weight server (no extra 196B load). The server
returns top-k logprobs for the first generated token = the next-token distribution
after the prompt — exactly what HF's logits[-1] gives.

Usage:
    STEP35_BASE_URL=http://<node0>:30000 python dump_server.py --out jax_dump --topk 20
    # smoke: --num-prompts 1
"""

import argparse
import json
import os
import sys

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="jax_dump")
    ap.add_argument(
        "--base-url", default=os.environ.get("STEP35_BASE_URL", "http://127.0.0.1:30000")
    )
    ap.add_argument("--inputs", default=None)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--num-prompts", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    inputs_path = args.inputs or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "inputs.json"
    )
    with open(inputs_path) as f:
        input_ids = json.load(f)["input_ids"]
    if args.num_prompts:
        input_ids = input_ids[: args.num_prompts]
    print(f"[jax] {len(input_ids)} prompts via {args.base_url}", flush=True)

    a_out = []
    for i, ids in enumerate(input_ids):
        resp = requests.post(
            f"{args.base_url}/generate",
            json={
                "input_ids": ids,
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                "return_logprob": True,
                "top_logprobs_num": args.topk,
            },
            timeout=120,
        )
        resp.raise_for_status()
        meta = resp.json()["meta_info"]
        # output_top_logprobs[0] = top-k for the first generated token (next-token).
        top = meta["output_top_logprobs"][0]  # [[logprob, token_id, ...], ...]
        a_out.append({"prompt": i, "topk": [[int(tid), float(lp)] for lp, tid, *_ in top]})
        if (i + 1) % 8 == 0:
            print(f"[jax] {i + 1}/{len(input_ids)}", flush=True)

    with open(os.path.join(args.out, "A_nexttok.json"), "w") as f:
        json.dump({"topk": args.topk, "data": a_out}, f)
    print(f"[jax] wrote A_nexttok.json to {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
