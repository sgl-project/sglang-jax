"""Steady-state load test for the V4.1 JAX server (closed loop: N concurrent users, each issuing requests back to back for DURATION s).

Usage: python loadtest_v41.py <port> <users> <duration_s> <mean_in_tokens> <mean_out_tokens> [chip_hour_usd=2.70] [chips=32] [warmup_s=40]
Prompts/outputs are drawn from a lognormal around the given means (sigma 0.5) so the batch is heterogeneous. Throughput is measured
only over requests that START after the warm-up window and COMPLETE before the deadline-drain, i.e. in steady state.
Reports aggregate output/input tokens/s, TTFT and TPOT p50/p95, errors, and cost per 1M output / total tokens."""

import json
import math
import random
import statistics
import sys
import threading
import time
import urllib.request

PORT, USERS, DUR, MIN, MOUT = (
    int(sys.argv[1]),
    int(sys.argv[2]),
    float(sys.argv[3]),
    int(sys.argv[4]),
    int(sys.argv[5]),
)
CHIP_USD = float(sys.argv[6]) if len(sys.argv) > 6 else 2.70
CHIPS = int(sys.argv[7]) if len(sys.argv) > 7 else 32
WARM = float(sys.argv[8]) if len(sys.argv) > 8 else 40.0
URL = f"http://localhost:{PORT}/v1/completions"
WORDS = "the of and to in is was for on that with as by at from his her an are were which this be or have had not but market river engine garden window paper silver mountain harvest signal bridge lantern compass velvet orchard thunder marble canvas harbor ember falcon meadow quartz saddle timber violet whisper".split()
t_start = time.time()
deadline = t_start + DUR
lock = threading.Lock()
done = []
errors = [0]


def ln(mean, rng):
    s = 0.5
    mu = math.log(mean) - s * s / 2
    return max(8, int(rng.lognormvariate(mu, s)))


def worker(uid):
    rng = random.Random(1000 + uid)
    while time.time() < deadline:
        n_in, n_out = ln(MIN, rng), ln(MOUT, rng)
        prompt = "Write a long story. " + " ".join(
            rng.choice(WORDS) for _ in range(max(1, n_in - 6))
        )
        body = json.dumps(
            {
                "prompt": prompt,
                "max_tokens": n_out,
                "temperature": 0.7,
                "seed": rng.randrange(1 << 30),
            }
        ).encode()
        t0 = time.time()
        try:
            r = json.load(
                urllib.request.urlopen(
                    urllib.request.Request(URL, body, {"Content-Type": "application/json"}),
                    timeout=1800,
                )
            )
        except Exception:
            with lock:
                errors[0] += 1
            time.sleep(0.5)
            continue
        t1 = time.time()
        with lock:
            done.append(
                (
                    t0,
                    t1,
                    r["usage"]["prompt_tokens"],
                    r["usage"]["completion_tokens"],
                    r["timings"]["ttft_s"],
                    r["timings"]["total_s"],
                )
            )


ths = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(USERS)]
for i, th in enumerate(ths):
    th.start()
    time.sleep(0.02)  # spread connections
for th in ths:
    th.join(timeout=max(1, deadline - time.time()) + 600)
steady = [d for d in done if d[0] >= t_start + WARM]
if not steady:
    print("no steady-state requests completed")
    sys.exit(1)
w0 = min(d[0] for d in steady)
w1 = max(d[1] for d in steady)
span = max(1e-6, w1 - w0)
out_tok = sum(d[3] for d in steady)
in_tok = sum(d[2] for d in steady)
ttft = sorted(d[4] for d in steady)
tpot = sorted((d[5] - d[4]) / max(1, d[3] - 1) * 1000 for d in steady if d[3] > 4)
pct = lambda a, q: a[min(len(a) - 1, int(q * len(a)))]
out_ps = out_tok / span
tot_ps = (out_tok + in_tok) / span
usd_hr = CHIP_USD * CHIPS
print(
    f"users={USERS} mean_in={MIN} mean_out={MOUT} | steady window {span:.0f}s, {len(steady)} requests, errors={errors[0]}"
)
print(f"  output {out_ps:8.1f} tok/s | input {in_tok/span:8.1f} tok/s | total {tot_ps:8.1f} tok/s")
print(
    f"  TTFT p50 {pct(ttft,.5):.2f}s p95 {pct(ttft,.95):.2f}s | TPOT p50 {pct(tpot,.5):.1f} ms p95 {pct(tpot,.95):.1f} ms | per-user decode {1000/max(1e-9,statistics.median(tpot)):.1f} tok/s"
)
print(
    f"  cost @ ${usd_hr:.1f}/h ({CHIPS} chips x ${CHIP_USD}): ${usd_hr / (out_ps * 0.0036):.2f} per 1M output tokens | ${usd_hr / (tot_ps * 0.0036):.2f} per 1M total (in+out) tokens"
)
