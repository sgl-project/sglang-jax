"""Manual Raiden PD serving check against an overlap-off token reference.

Run against an idle P/D pair with ChunkCache. Record a reference with overlap
OFF, restart the same configuration with overlap ON, then verify twice to
exercise page reuse. Service launch and accelerator allocation are external.
"""

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer

QUEUE_FIELDS = (
    "waiting_queue_size",
    "running_batch_size",
    "pending_dp_reqs_size",
    "disagg_prefill_queue_size",
    "disagg_prealloc_queue_size",
    "disagg_transfer_queue_size",
    "req_to_token_pool_used",
)


def idle(urls, baseline=None):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        states = []
        for url in urls:
            response = requests.get(f"{url}/get_server_info", timeout=30)
            response.raise_for_status()
            states.extend(response.json()["internal_states"])
        assert states, "missing scheduler states"
        capacity = [(s["available_kv_tokens"], s["req_to_token_pool_available"]) for s in states]
        if all(not s[key] for s in states for key in QUEUE_FIELDS):
            if baseline is None or capacity == baseline:
                return states, capacity
        time.sleep(0.2)
    raise AssertionError(f"P/D queues or KV capacity did not recover: {states}")


def generate(url, case):
    previous = []
    done = False
    with requests.post(f"{url}/generate", json=case, stream=True, timeout=600) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1024):
            if not line.startswith(b"data:"):
                continue
            raw = line[5:].strip()
            if raw == b"[DONE]":
                done = True
                break
            event = json.loads(raw)
            assert "error" not in event, event
            meta = event["meta_info"]
            finish = meta.get("finish_reason")
            assert not finish or finish.get("type") != "abort", event
            logprobs = meta.get("output_token_logprobs", [])
            tokens = [item[1] for item in logprobs]
            assert tokens[: len(previous)] == previous, "non-monotonic output"
            assert len(tokens) == meta["completion_tokens"], "missing/duplicated logprobs"
            previous = tokens
    assert done, "stream ended without DONE"
    assert len(previous) == case["sampling_params"]["max_new_tokens"], "short output"
    return previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30020")
    parser.add_argument("--prefill-url", default="http://127.0.0.1:30000")
    parser.add_argument("--decode-url", default="http://127.0.0.1:30010")
    parser.add_argument("--model", required=True, help="Tokenizer path; use the served model")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--record-reference", action="store_true")
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    assert args.concurrency > 0
    urls = [args.prefill_url, args.decode_url]
    states, capacity = idle(urls)
    assert all(s["disable_radix_cache"] for s in states), "requires ChunkCache"
    assert all(
        s["disaggregation_overlap_enabled"] == (not args.record_reference) for s in states
    ), "record with both roles OFF; verify with both ON"
    if args.record_reference:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        cases = []
        for length in (128, 129, 4096, 4097):
            for variant in range(2):
                seed = tokenizer.encode(f"Explain this sequence carefully: {variant}. ")
                assert seed
                cases.append(
                    {
                        "input_ids": (seed * (length // len(seed) + 1))[:length],
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 128,
                            "ignore_eos": True,
                        },
                        "stream": True,
                        "return_logprob": True,
                        "logprob_start_len": -1,
                    }
                )
        expected = [generate(args.url, case) for case in cases]
        idle(urls, capacity)
        args.reference.write_text(
            json.dumps({"model": args.model, "cases": cases, "tokens": expected}, indent=2)
        )
    else:
        reference = json.loads(args.reference.read_text())
        assert reference["model"] == args.model, "reference model mismatch"
        cases, expected = reference["cases"], reference["tokens"]
        assert cases and len(cases) == len(expected)
        for _ in range(2):
            work = list(range(len(cases))) * 2
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                results = pool.map(lambda i: generate(args.url, cases[i]), work)
                for i, actual in zip(work, results):
                    assert actual == expected[i], f"token mismatch for case {i}"
            idle(urls, capacity)
    print("PASS: token equality, complete streams, logprob alignment, and KV/slot recovery")


if __name__ == "__main__":
    main()
