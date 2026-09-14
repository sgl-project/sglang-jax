"""Record a bounded cache-pressure sequence or compare two recorded runs.

Start servers separately with matching model/scheduler settings and a 2048-token
KV pool. Record JAX and Raiden runs, then compare them. This checks serving output;
use raiden_hicache_native.py for direct transfer and allocation assertions.
"""

import argparse
import json
import random
from pathlib import Path

import requests


def compare(actual_path, reference_path):
    actual = json.loads(Path(actual_path).read_text())
    reference = json.loads(Path(reference_path).read_text())
    assert len(actual) == len(reference) == 13
    for a, b in zip(actual, reference):
        assert a["index"] == b["index"]
        assert a["response"]["text"] == b["response"]["text"]
        assert (
            a["response"]["meta_info"]["output_token_logprobs"]
            == b["response"]["meta_info"]["output_token_logprobs"]
        )
    print("SERVING_EXACT_MATCH_PASS", len(actual), flush=True)


def record(url, output):
    rng = random.Random(42)
    prompts = [[rng.randrange(1000, 10000) for _ in range(640)] for _ in range(5)]
    requests.post(url + "/flush_cache", timeout=60).raise_for_status()
    rows = []
    for index in [0, 0, 1, 1, 2, 2, 3, 3, 0, 4, 4, 1, 0]:
        response = requests.post(
            url + "/generate",
            json={
                "input_ids": prompts[index],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 16,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "logprob_start_len": -1,
            },
            timeout=240,
        )
        response.raise_for_status()
        result = response.json()
        assert result["meta_info"]["completion_tokens"] == 16
        rows.append({"index": index, "response": result})
        Path(output).write_text(json.dumps(rows, indent=2))
    assert any(row["response"]["meta_info"].get("cached_tokens", 0) for row in rows)
    requests.post(url + "/flush_cache", timeout=60).raise_for_status()
    print("SERVING_CLIENT_PASS", len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Recorded run JSON")
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--reference", help="Compare --out with this recorded JAX run")
    args = parser.parse_args()
    if args.reference:
        compare(args.out, args.reference)
    else:
        record(args.url.rstrip("/"), args.out)
