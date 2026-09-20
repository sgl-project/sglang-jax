"""Per-change ablation for the n-gram hash fusion (see README.md).

Carries the intermediate version -- prefix XOR in [T], but still reducing in
int64 -- so the README's per-change table can be re-derived. The shipped
function is the one in the working tree; this file only exists to attribute
the speedup to each change.

    python benchmark/kernels/ngram/ablate_hash_fusion.py
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from bench_ngram_hash import CFG, bench, load_versions, make  # noqa: E402

base, tree = load_versions()
bp = base.build_hash_params(**CFG)
tp = tree.build_hash_params(**CFG)


def prefix_xor_int64_reduce(input_ids, cu, context, p):
    """Change 1 only: mix in [T], reduce still signed."""
    input_ids = np.asarray(input_ids, np.int64).reshape(-1)
    cu = np.asarray(cu, np.int64)
    context = np.asarray(context, np.int64)
    L = p.ngram_context_len
    T = input_ids.shape[0]
    B = cu.shape[0] - 1
    hpn = p.heads_per_ngram
    sizes = p.sizes.reshape(L, hpn)
    offs = p.offsets.reshape(L, hpn)
    t = np.arange(T, dtype=np.int64)
    req = np.clip(np.searchsorted(cu, t, side="right") - 1, 0, B - 1)
    pos = t - cu[req]
    rolling = input_ids * p.multipliers[0]
    ids = np.empty((T, L, hpn), np.int32)
    res = np.empty((T, hpn), np.int64)
    crossed = np.zeros(T, bool)
    for s in range(1, L + 1):
        step = input_ids[np.clip(t - s, 0, T - 1)]
        col = np.clip(L - s + pos, 0, L - 1)
        tok = np.where(pos >= s, step, context[req, col])
        np.copyto(tok, p.eos_token_id, where=crossed)
        crossed |= tok == p.eos_token_id
        rolling ^= tok * p.multipliers[s]
        np.mod(rolling[:, None], sizes[s - 1][None, :], out=res)
        res += offs[s - 1][None, :]
        ids[:, s - 1, :] = res
    return ids.reshape(T, L * hpn)


rng = np.random.default_rng(0)
print(f"{'case':<17}{'base':>9}{'+[T] xor':>10}{'+u64':>9}{'+decode fp':>12}")
for name, T, B in [
    ("decode B=256", 256, 256),
    ("prefill T=2048", 2048, 4),
    ("prefill T=8192", 8192, 4),
    ("prefill T=32768", 32768, 8),
]:
    a = make(T, B, tp, rng)
    r0 = base.compute_ngram_ids(*a, bp)
    assert np.array_equal(r0, prefix_xor_int64_reduce(*a, tp)) and np.array_equal(
        r0, tree.compute_ngram_ids(*a, tp)
    )
    m0 = bench(lambda a=a: base.compute_ngram_ids(*a, bp))
    m1 = bench(lambda a=a: prefix_xor_int64_reduce(*a, tp))
    m2 = bench(lambda a=a: tree.compute_ngram_ids(*a, tp))
    print(f"{name:<17}{m0:>9.4f}{m1:>10.4f}{'':>9}{m2:>12.4f}")
    print(f"{'  speedup':<17}{'1.00x':>9}{m0/m1:>9.2f}x{'':>9}{m0/m2:>11.2f}x")
