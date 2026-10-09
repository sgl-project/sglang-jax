"""End-to-end generation parity: PD-disaggregated vs monolithic serving.

Manual-only (not wired into CI). RFC "PD disaggregation for hybrid models"
§5.2 item 2. Sends the same greedy requests to:

* ``--baseline-url``: a monolithic (non-disaggregated) server, and
* ``--pd-url``: the PD router in front of 1 prefill + 1 decode server,

both serving the same hybrid model (e.g. Qwen3.5 / Kimi-Linear) with the same
TP/DP layout, then checks that

* output token ids match exactly, and
* per-token output logprobs match within ``--logprob-atol`` (default 2e-2,
  i.e. two bf16 rounding steps near 0; PD re-runs the last prompt token as a
  separate 1-token extend, so logprobs on a rounding boundary can flip a step).

The prompts include lengths around page boundaries (L-1 divisible by the page
size), which exercise the "P skips the last prompt token" path.

The transfer engine is chosen when the PD servers are launched
(``--disaggregation-use-raiden`` selects Raiden, otherwise jax_transfer). Before
sending prompts, the script asks the router's ``/get_server_info`` which engine
each prefill/decode server runs and prints it; ``--expect-engine`` turns a
mismatch into a failure so a Raiden run cannot silently test jax_transfer.

Example:

    python test/srt/multi_host/disaggregation/pd_hybrid_generation_parity.py \\
        --baseline-url http://<mono-host>:30000 --pd-url http://<router-host>:8000 \\
        --model-path Qwen/Qwen3.5-... --page-size 64 --expect-engine raiden

[engine] prefill: disaggregation_mode=prefill engine=raiden
[engine] decode: disaggregation_mode=decode engine=raiden
[transformers] PyTorch was not found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
[PASS] #0 len=24 chars: 32 tokens, max |logprob diff|=0.000e+00, 0/32 positions differ
[PASS] #1 len=37 chars: 32 tokens, max |logprob diff|=0.000e+00, 0/32 positions differ
[PASS] #2 len=53 chars: 32 tokens, max |logprob diff|=0.000e+00, 0/32 positions differ
[PASS] #3 len=55 chars: 32 tokens, max |logprob diff|=0.000e+00, 0/32 positions differ
[PASS] #4 len=128 ids: 32 tokens, max |logprob diff|=7.721e-03, 1/32 positions differ
[PASS] #5 len=129 ids: 32 tokens, max |logprob diff|=7.721e-03, 2/32 positions differ
[PASS] #6 len=130 ids: 32 tokens, max |logprob diff|=7.721e-03, 1/32 positions differ
[PASS] #7 len=256 ids: 32 tokens, max |logprob diff|=0.000e+00, 0/32 positions differ
[PASS] #8 len=257 ids: 32 tokens, max |logprob diff|=0.000e+00, 0/32 positions differ
[PASS] #9 len=258 ids: 32 tokens, max |logprob diff|=0.000e+00, 0/32 positions differ
[PASS] #10 len=1 ids: 32 tokens, max |logprob diff|=0.000e+00, 0/32 positions differ
11/11 prompts at parity
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


def _check_pd_engine(pd_url: str, expected: str | None, timeout: float) -> str | None:
    """Print each PD server's role/engine; return an error string on a mismatch."""

    with urllib.request.urlopen(pd_url.rstrip("/") + "/get_server_info", timeout=timeout) as resp:
        info = json.loads(resp.read())
    servers = [("prefill", s) for s in info.get("prefill") or []] + [
        ("decode", s) for s in info.get("decode") or []
    ]
    if not servers:
        return f"{pd_url}/get_server_info lists no prefill/decode servers; is it the PD router?"
    errors = []
    for role, server in servers:
        engine = "raiden" if server.get("disaggregation_use_raiden") else "jax_transfer"
        mode = server.get("disaggregation_mode")
        print(f"[engine] {role}: disaggregation_mode={mode} engine={engine}", flush=True)
        if mode != role:
            errors.append(f"{role} server reports disaggregation_mode={mode!r}")
        if expected is not None and engine != expected:
            errors.append(f"{role} server runs {engine}, expected {expected}")
    return "; ".join(errors) or None


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
    ap.add_argument(
        "--logprob-atol",
        type=float,
        default=2e-2,
        help="max per-token |logprob diff|. Logprobs are bf16-quantized (~7.8e-3 "
        "steps near 0), and PD re-runs the last prompt token as a separate "
        "1-token extend, so values on a rounding boundary can flip one step. "
        "The default allows two steps; token ids must still match exactly.",
    )
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument(
        "--expect-engine",
        choices=["jax_transfer", "raiden"],
        default=None,
        help="fail unless every PD server behind --pd-url uses this KV transfer engine",
    )
    args = ap.parse_args()

    err = _check_pd_engine(args.pd_url, args.expect_engine, args.timeout)
    if err:
        print(f"[FAIL] engine check: {err}", flush=True)
        return 1

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
        diffs = [abs(a - b) for a, b in zip(base_lp, pd_lp)]
        n_diff = sum(d > 0 for d in diffs)
        max_k = max(range(len(diffs)), key=diffs.__getitem__, default=None)
        max_diff = diffs[max_k] if max_k is not None else 0.0
        summary = f"max |logprob diff|={max_diff:.3e}, {n_diff}/{len(diffs)} positions differ"
        if max_diff > args.logprob_atol:
            # Absolute position = prompt tokens + output index (known for id prompts).
            where = f"output index {max_k}"
            if isinstance(prompt, list):
                where += f" (absolute position {len(prompt) + max_k})"
            print(
                f"[FAIL] {label}: tokens match, {summary} > {args.logprob_atol:.0e}",
                flush=True,
            )
            print(
                f"       at {where}: token={pd_ids[max_k]} "
                f"baseline={base_lp[max_k]:.6f} pd={pd_lp[max_k]:.6f}",
                flush=True,
            )
            failures += 1
            continue
        print(f"[PASS] {label}: {len(pd_ids)} tokens, {summary}", flush=True)

    total = len(prompts)
    print(f"{total - failures}/{total} prompts at parity", flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
