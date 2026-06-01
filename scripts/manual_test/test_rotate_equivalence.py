#!/usr/bin/env python3
"""Equivalence test: fused device rotate == original host _rotate_ids.

Validates the fix for the partial-accept / padding-row divergence:
- fused used jnp.roll (wrap-around) -> last column got s0 instead of s3
- fused rotated padding rows too (original skips el==0)

Pure algorithm test on a single CPU device (no cluster / no TPU / no server):
  JAX_PLATFORMS=cpu python ../scripts/manual_test/test_rotate_equivalence.py

Covers: full / partial / mixed accept, padding reqs, dp>1, multi-layer chain.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.speculative.draft_extend_fused import _device_rotate_input_ids
from sgl_jax.srt.speculative.multi_layer_draft_worker import MultiLayerDraftWorker


def original_rotate(input_ids, ext_lens, sel_pos, new_tokens, dp, per_dp_bs):
    """Run the REAL MultiLayerDraftWorker._rotate_ids on a mock mwb."""
    mwb = SimpleNamespace(
        dp_size=dp,
        per_dp_bs_size=per_dp_bs,
        input_ids=input_ids.copy(),
        extend_seq_lens=ext_lens,
    )
    MultiLayerDraftWorker._rotate_ids(mwb, new_tokens, sel_pos)
    return mwb.input_ids


def fused_rotate(input_ids, ext_lens, sel_pos, new_tokens):
    """Run the REAL _device_rotate_input_ids (production fused code)."""
    out = _device_rotate_input_ids(
        jnp.asarray(input_ids),
        jnp.asarray(ext_lens),
        jnp.asarray(sel_pos),
        jnp.asarray(new_tokens),
    )
    return np.asarray(jax.device_get(out)).astype(np.int32)


def buggy_rotate(input_ids, ext_lens, sel_pos, new_tokens):
    """Negative control: the OLD buggy impl (jnp.roll wrap-around, no padding
    skip). Must DIVERGE from original on partial-accept / padding to prove the
    test actually discriminates."""
    bs = ext_lens.shape[0]
    tpr = input_ids.shape[0] // bs
    ids_2d = jnp.asarray(input_ids).reshape(bs, tpr)
    shifted = jnp.roll(ids_2d, -1, axis=1)
    shifted = shifted.at[jnp.arange(bs), jnp.asarray(sel_pos)].set(jnp.asarray(new_tokens))
    return np.asarray(shifted.reshape(-1)).astype(np.int32)


def run_case(name, bs, tokens_per_req, dp, real_bs_per_dp, accepts, num_layers=3):
    per_dp_bs = bs // dp
    ext_lens = np.zeros(bs, dtype=np.int32)
    for r in range(dp):
        for j in range(real_bs_per_dp):
            ext_lens[r * per_dp_bs + j] = tokens_per_req

    # distinct values so any divergence is visible
    base_ids = np.arange(bs * tokens_per_req, dtype=np.int32) + 100
    sel_pos = np.clip(np.asarray(accepts) - 1, 0, None).astype(np.int64)

    cur_o = base_ids.copy()
    cur_f = base_ids.copy()
    ok = True
    for layer in range(num_layers - 1):
        new_tokens = np.arange(bs, dtype=np.int32) + 1000 * (layer + 1)
        o = original_rotate(cur_o, ext_lens, sel_pos, new_tokens, dp, per_dp_bs)
        f = fused_rotate(cur_f, ext_lens, sel_pos, new_tokens)
        if not np.array_equal(o, f):
            ok = False
            diff = np.where(o != f)[0]
            print(f"  [{name}] layer {layer} DIFF at {len(diff)} positions {diff[:12].tolist()}")
            print(f"    orig  = {o[diff[:12]].tolist()}")
            print(f"    fused = {f[diff[:12]].tolist()}")
        cur_o, cur_f = o, f
    print(
        f"  [{name}] {'PASS' if ok else 'FAIL'}  (ext_lens={ext_lens.tolist()}, accepts={accepts})"
    )
    return ok


CASES = [
    # name, bs, tokens_per_req, dp, real_bs_per_dp, accepts
    ("dp1_full_nopad", 4, 4, 1, 4, [4, 4, 4, 4]),
    ("dp1_partial_nopad", 4, 4, 1, 4, [2, 1, 3, 4]),
    ("dp1_allpartial", 4, 4, 1, 4, [1, 1, 2, 2]),
    ("dp2_full_pad", 8, 4, 2, 3, [4, 4, 4, 0, 4, 4, 4, 0]),
    ("dp2_partial_pad", 8, 4, 2, 3, [2, 3, 1, 0, 4, 2, 3, 0]),
    ("dp4_mixed_pad", 16, 4, 4, 3, [2, 3, 1, 0, 4, 2, 3, 0, 1, 4, 2, 0, 3, 1, 4, 0]),
    # bs=64-like: real_bs_per_dp=9 of 16, lots of partial + padding
    (
        "dp4_bs64_like",
        64,
        4,
        4,
        9,
        sum([[((j % 4) + 1) for j in range(9)] + [0] * 7 for _ in range(4)], []),
    ),
]


def main():
    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")

    # Negative control: prove the test discriminates — the OLD buggy rotate
    # MUST diverge from original on a partial-accept + padding case.
    bs, tpr, dp, per_dp_bs = 8, 4, 2, 4
    ext = np.array([4, 4, 4, 0, 4, 4, 4, 0], dtype=np.int32)
    acc = [2, 3, 1, 0, 4, 2, 3, 0]
    sel = np.clip(np.array(acc) - 1, 0, None).astype(np.int64)
    ids = np.arange(bs * tpr, dtype=np.int32) + 100
    nt = np.arange(bs, dtype=np.int32) + 1000
    o = original_rotate(ids, ext, sel, nt, dp, per_dp_bs)
    bug = buggy_rotate(ids, ext, sel, nt)
    if np.array_equal(o, bug):
        print(
            "  [neg-control] UNEXPECTED: buggy rotate matches original — test is NOT discriminating!"
        )
        raise SystemExit(2)
    print(
        f"  [neg-control] OK: buggy rotate diverges from original at "
        f"{int((o != bug).sum())} positions (test discriminates)"
    )

    results = [run_case(*c) for c in CASES]
    print()
    if all(results):
        print("ALL PASS — fused rotate == original _rotate_ids")
    else:
        print(f"FAILED {results.count(False)}/{len(results)} cases")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
