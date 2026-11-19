"""Needle-in-a-haystack against the V4.1 server: hide a code in N tokens of non-repeating filler and ask for it.
Usage: python needle_v41.py <port> <n_tokens> [depth 0..1]   (prints prompt tokens, timings, answer, PASS/FAIL)"""

import json
import random
import sys
import time
import urllib.request

PORT, N = int(sys.argv[1]), int(sys.argv[2])
DEPTH = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
rng = random.Random(N)
W = "market river engine garden window paper silver mountain harvest signal bridge lantern compass velvet orchard thunder marble canvas harbor ember falcon meadow quartz saddle timber violet whisper".split()
code = str(rng.randrange(10**7, 10**8))
needle = f"Important note: the vault access code is {code}. Remember it."


def sentence():
    return f"On day {rng.randrange(1, 9999)} the {rng.choice(W)} {rng.choice(W)} reported {rng.randrange(10, 99999)} units near the {rng.choice(W)} {rng.choice(W)}."


# ~ 11 tokens per sentence; aim for N tokens
n_sent = N // 11
at = int(n_sent * DEPTH)
doc = " ".join(needle if i == at else sentence() for i in range(n_sent))
msgs = [
    {
        "role": "user",
        "content": doc
        + "\n\nWhat is the vault access code mentioned in the document above? Answer with just the number.",
    }
]
body = json.dumps(
    {"messages": msgs, "thinking": False, "max_tokens": 24, "temperature": 0}
).encode()
t = time.time()
r = json.load(
    urllib.request.urlopen(
        urllib.request.Request(
            f"http://localhost:{PORT}/v1/chat/completions",
            body,
            {"Content-Type": "application/json"},
        ),
        timeout=7200,
    )
)
ans = r["choices"][0]["message"]["content"]
print(
    f"target_tokens={N} prompt_tokens={r['usage']['prompt_tokens']} depth={DEPTH} ttft={r['timings']['ttft_s']:.1f}s total={r['timings']['total_s']:.1f}s wall={time.time()-t:.1f}s"
)
print(
    f"expected {code} | answer {ans!r} | {'PASS' if code in str(ans).replace(',', '').replace(' ', '') else 'FAIL'}"
)
