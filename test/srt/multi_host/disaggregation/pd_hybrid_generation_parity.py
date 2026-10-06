"""End-to-end generation parity: PD-disaggregated vs monolithic serving.

Manual-only (not wired into CI). RFC "PD disaggregation for hybrid models"
§5.2 item 2. Sends the same greedy requests to:

* ``--baseline-url``: a monolithic (non-disaggregated) server, and
* ``--pd-url``: the PD router in front of 1 prefill + 1 decode server,

both serving the same hybrid model (e.g. Qwen3.5 / Kimi-Linear) with the same
TP/DP layout, then checks that

* output token ids match exactly, and
* per-token output logprobs match within ``--logprob-atol`` (default 1e-5).

The prompts include lengths around page boundaries (L-1 divisible by the page
size), which exercise the "P skips the last prompt token" path.

Example:

    python test/srt/multi_host/disaggregation/pd_hybrid_generation_parity.py \\
        --baseline-url http://<mono-host>:30000 --pd-url http://<router-host>:8000 \\
        --model-path Qwen/Qwen3.5-... --page-size 64

Exit 0 on full parity; 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

DEFAULT_PROMPTS = [
    "The capital of France is",
    "Write a haiku about linear attention.",
    "List the first ten prime numbers separated by commas:",
    "Explain prefill/decode disaggregation in one paragraph.",
]


def _post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url.rstrip("/") + "/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _token_prompts(model_path: str | None, page_size: int) -> list[list[int]]:
    """Token-id prompts of length L with (L-1) % page_size == 0 and neighbours."""

    if not model_path:
        return []
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base = tok.encode(" ".join(["state space models are recurrent"] * 200))
    out = []
    for n_pages in (1, 2):
        aligned = n_pages * page_size + 1  # L-1 is a page multiple
        for length in (aligned - 1, aligned, aligned + 1):
            out.append(base[:length])
    out.append(base[:1])  # single-token prompt: P must not skip it
    return out


def _generate(url: str, prompt, args) -> tuple[list[int], list[float]]:
    body = {
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": args.max_new_tokens,
            "ignore_eos": True,
        },
        "return_logprob": True,
    }
    body["input_ids" if isinstance(prompt, list) else "text"] = prompt
    out = _post(url, body, args.timeout)
    meta = out["meta_info"]
    # Each entry is (logprob, token_id, token_text | None).
    entries = meta["output_token_logprobs"]
    ids = [int(e[1]) for e in entries]
    logprobs = [float(e[0]) for e in entries]
    return ids, logprobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-url", required=True)
    ap.add_argument("--pd-url", required=True)
    ap.add_argument("--model-path", default=None, help="enables page-boundary token prompts")
    ap.add_argument("--page-size", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--logprob-atol", type=float, default=1e-5)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    prompts = list(DEFAULT_PROMPTS) + _token_prompts(args.model_path, args.page_size)
    failures = 0
    for i, prompt in enumerate(prompts):
        label = f"#{i} len={len(prompt)}{' ids' if isinstance(prompt, list) else ' chars'}"
        base_ids, base_lp = _generate(args.baseline_url, prompt, args)
        pd_ids, pd_lp = _generate(args.pd_url, prompt, args)
        if base_ids != pd_ids:
            first = next(
                (k for k, (a, b) in enumerate(zip(base_ids, pd_ids)) if a != b),
                min(len(base_ids), len(pd_ids)),
            )
            print(f"[FAIL] {label}: token mismatch at position {first}", flush=True)
            print(f"       baseline={base_ids}\n       pd      ={pd_ids}", flush=True)
            failures += 1
            continue
        max_diff = max((abs(a - b) for a, b in zip(base_lp, pd_lp)), default=0.0)
        if max_diff > args.logprob_atol:
            print(
                f"[FAIL] {label}: tokens match, max |logprob diff|={max_diff:.3e} "
                f"> {args.logprob_atol:.0e}",
                flush=True,
            )
            failures += 1
            continue
        print(f"[PASS] {label}: {len(pd_ids)} tokens, max |logprob diff|={max_diff:.3e}")

    total = len(prompts)
    print(f"{total - failures}/{total} prompts at parity", flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
