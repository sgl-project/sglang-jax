"""compute_ngram_ids: base revision vs working tree (fused).

Loads the host-side hash section out of both copies of ngram_embedding.py and
execs it standalone, so what is measured is the shipped source, not a
transcription of it. No jax/flax needed -- that section is pure numpy.

    python benchmark/kernels/ngram/bench_ngram_hash.py [base-rev]
"""

from __future__ import annotations

import json
import pathlib
import statistics
import subprocess
import sys
import time
import types

import numpy as np

REPO = str(pathlib.Path(__file__).resolve().parents[3])
REL = "python/sgl_jax/srt/layers/ngram_embedding.py"
BASE_REV = sys.argv[1] if len(sys.argv) > 1 else "a1c7923"  # last commit before the fusion


def _load(src: str, name: str):
    """Exec the pure-numpy hash section (constants .. compute_ngram_ids)."""
    body = src[src.index("_MASK64 = ") : src.index("# --- device layer ---")]
    mod = types.ModuleType(name)
    sys.modules[name] = mod  # dataclasses resolves annotations through sys.modules
    exec(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\nimport numpy as np\n" + body,
        mod.__dict__,
    )
    return mod


def load_versions():
    """`BASE_REV` is a git rev, or a path to a copy of the file for checkouts
    that have no .git (the TPU VMs are rsync'd, not cloned)."""
    tree = pathlib.Path(REPO, REL).read_text()
    if pathlib.Path(BASE_REV).is_file():
        base = pathlib.Path(BASE_REV).read_text()
    else:
        base = subprocess.run(
            ["git", "-C", REPO, "show", f"{BASE_REV}:{REL}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    return _load(base, "base"), _load(tree, "tree")


# --- independent reference (from test_ngram_embedding._ref_ngram_ids) -------
def ref_ngram_ids(input_ids, cu_seqlens, context, params):
    ctx_len = params.ngram_context_len
    out = np.zeros((len(input_ids), params.ngram_heads), dtype=np.int64)
    for b in range(len(cu_seqlens) - 1):
        start, end = int(cu_seqlens[b]), int(cu_seqlens[b + 1])
        history = list(context[b]) + list(input_ids[start:end])
        for pos in range(end - start):
            here = ctx_len + pos
            tokens, crossed = [int(history[here])], False
            for shift in range(1, ctx_len + 1):
                tok = int(history[here - shift])
                if crossed:
                    tok = params.eos_token_id
                crossed = crossed or tok == params.eos_token_id
                tokens.append(tok)
            for head in range(params.ngram_heads):
                order = head // params.heads_per_ngram + 2
                mixed = 0
                for i in range(order):
                    mixed ^= tokens[i] * int(params.multipliers[i])
                out[start + pos, head] = mixed % int(params.sizes[head]) + int(params.offsets[head])
    return out


# --- shapes -----------------------------------------------------------------
# Qwen4Exp released checkpoint: ngram_size=3, heads_per_ngram=8 -> HEADS=16,
# 16 primes above 20,000,000 -> 320,001,446 rows.
CFG = dict(
    ngram_size=3,
    heads_per_ngram=8,
    vocab_size=151936,
    ngram_vocab_size_base=20_000_000,
    eos_token_id=151643,
)

CASES = [
    ("decode B=1", 1, 1),
    ("decode B=64", 64, 64),
    ("decode B=256", 256, 256),
    ("decode B=512", 512, 512),
    ("prefill T=512", 512, 2),
    ("prefill T=2048", 2048, 4),
    ("prefill T=8192", 8192, 4),
    ("prefill T=32768", 32768, 8),
]


def make(T, B, params, rng, eos_rate=0.0):
    lens = np.full(B, T // B, np.int64)
    lens[: T % B] += 1
    cu = np.zeros(B + 1, np.int64)
    np.cumsum(lens, out=cu[1:])
    ids = rng.integers(0, CFG["vocab_size"], T, dtype=np.int64)
    if eos_rate:
        ids[rng.random(T) < eos_rate] = params.eos_token_id
    ctx = rng.integers(0, CFG["vocab_size"], (B, params.ngram_context_len), dtype=np.int64)
    return ids, cu, ctx


def bench(fn, reps=200, warm=20):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)


def main():
    base, tree = load_versions()
    params = tree.build_hash_params(**CFG)
    bparams = base.build_hash_params(**CFG)
    assert params.total_vocab_size == 320_001_446, params.total_vocab_size
    rng = np.random.default_rng(0)

    # correctness: both against each other and against the python reference
    print("correctness")
    for name, T, B in CASES:
        for eos_rate in (0.0, 0.05, 0.5):
            a = make(T, B, params, rng, eos_rate)
            want = base.compute_ngram_ids(*a, bparams)
            got = tree.compute_ngram_ids(*a, params)
            assert np.array_equal(want, got), f"{name} eos={eos_rate}"
            assert got.dtype == np.int32
            if T <= 2048:
                assert np.array_equal(got, ref_ngram_ids(*a, params)), f"ref {name}"
    # chunked prefill: splitting a sequence must not move any token's ids
    ids = rng.integers(0, CFG["vocab_size"], 64, dtype=np.int64)
    ids[ids == params.eos_token_id] = params.eos_token_id + 1
    pad = np.full((1, 2), params.eos_token_id, np.int64)
    whole = tree.compute_ngram_ids(ids, np.array([0, 64]), pad, params)
    first = tree.compute_ngram_ids(ids[:20], np.array([0, 20]), pad, params)
    second = tree.compute_ngram_ids(ids[20:], np.array([0, 44]), ids[18:20][None, :], params)
    assert np.array_equal(first, whole[:20]) and np.array_equal(second, whole[20:])
    # T == B but ragged (a 0-token and a 3-token request): the decode fast
    # path must NOT fire here, so check it still matches the base revision.
    r_ids = np.arange(4, dtype=np.int64) + 7
    r_cu = np.array([0, 0, 1, 4, 4])
    r_ctx = rng.integers(0, CFG["vocab_size"], (4, 2), dtype=np.int64)
    assert np.array_equal(
        tree.compute_ngram_ids(r_ids, r_cu, r_ctx, params),
        base.compute_ngram_ids(r_ids, r_cu, r_ctx, bparams),
    )
    assert np.array_equal(
        tree.compute_ngram_ids(r_ids, r_cu, r_ctx, params),
        ref_ngram_ids(r_ids, r_cu, r_ctx, params),
    )
    print("  matches the base revision, the python reference, chunk splits, and ragged T==B\n")

    print(f"{'case':<17}{'base ms':>10}{'fused ms':>10}{'speedup':>9}")
    rows = []
    for name, T, B in CASES:
        a = make(T, B, params, rng)
        m0 = bench(lambda a=a: base.compute_ngram_ids(*a, bparams))
        m1 = bench(lambda a=a: tree.compute_ngram_ids(*a, params))
        rows.append(
            {
                "case": name,
                "tokens": T,
                "reqs": B,
                "base_ms": round(m0, 4),
                "fused_ms": round(m1, 4),
                "speedup": round(m0 / m1, 3),
            }
        )
        print(f"{name:<17}{m0:>10.4f}{m1:>10.4f}{m0 / m1:>8.2f}x")
    with open(pathlib.Path(__file__).with_suffix(".json"), "w") as f:
        json.dump(
            {"numpy": np.__version__, "python": sys.version.split()[0], "rows": rows},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
