"""Latency/throughput sweep against the V4.1 JAX server (/v1/completions, raw token-length-controlled prompts).

Usage: python bench_v41.py [port] [cases]   cases like "128x128@1,128x128@4,1024x256@4" = prompt_tokens x output_tokens @ concurrency
Each case sends 2*concurrency requests (ignore_eos is emulated by stop-free random text prompts and max_tokens; we report actual tokens).
"""

import concurrent.futures as cf
import json
import random
import statistics
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
CASES = (
    sys.argv[2]
    if len(sys.argv) > 2
    else "128x128@1,128x128@2,128x128@4,1024x128@1,1024x128@4,4096x128@1"
)
URL = f"http://localhost:{PORT}/v1/completions"
WORDS = "the of and to in is was for on that with as by at from his her an are were which this be or have had not but".split()


def prompt(n_tok, seed):
    r = random.Random(seed)
    # ~1 token per common word; prefix asks to continue so the model rarely stops early
    return "Write a very long story. " + " ".join(r.choice(WORDS) for _ in range(max(1, n_tok - 8)))


def one(args):
    p, n_out, seed = args
    body = json.dumps({"prompt": p, "max_tokens": n_out, "temperature": 0.7, "seed": seed}).encode()
    t = time.time()
    r = json.load(
        urllib.request.urlopen(
            urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=3600
        )
    )
    return r["usage"], r["timings"], time.time() - t


print(
    f"{'case':<16}{'reqs':>5}{'in_tok':>8}{'out_tok':>8}{'TTFT p50 s':>12}{'TPOT p50 ms':>13}{'e2e p50 s':>11}{'out tok/s aggr':>16}"
)
for case in CASES.split(","):
    io, c = case.split("@")
    n_in, n_out = map(int, io.split("x"))
    c = int(c)
    reqs = [(prompt(n_in, 1000 * c + i), n_out, i) for i in range(2 * c)]
    t0 = time.time()

    def staggered(a):
        time.sleep(
            a[2] % c * 0.03
        )  # spread the initial connections (older servers had a 5-deep accept backlog)
        return one(a)

    with cf.ThreadPoolExecutor(c) as ex:
        res = list(ex.map(staggered, reqs))
    wall = time.time() - t0
    ttft = [r[1]["ttft_s"] for r in res]
    tpot = [
        (r[1]["total_s"] - r[1]["ttft_s"]) / max(1, r[0]["completion_tokens"] - 1) * 1000
        for r in res
    ]
    out = sum(r[0]["completion_tokens"] for r in res)
    inn = statistics.mean(r[0]["prompt_tokens"] for r in res)
    print(
        f"{case:<16}{len(res):>5}{inn:>8.0f}{out/len(res):>8.0f}{statistics.median(ttft):>12.2f}{statistics.median(tpot):>13.1f}"
        f"{statistics.median([r[2] for r in res]):>11.1f}{out/wall:>16.1f}",
        flush=True,
    )
