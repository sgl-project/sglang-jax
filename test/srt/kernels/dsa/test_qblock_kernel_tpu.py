"""TPU parity: blocked (query-batched) sparse-MLA prefill kernel vs the
deployed per-query kernel and a numpy masked-softmax oracle, compiled (no
interpret). TPU-only (same convention as test_streamindex_topk.py).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.dsa.sparse_mla_prefill import (
    sparse_mla_attention,
    units_to_token_ids,
)
from sgl_jax.srt.kernels.dsa.sparse_mla_prefill_qblock import (
    sparse_mla_attention_qblock,
)

if jax.default_backend() != "tpu":
    pytest.skip("qblock TPU parity requires TPU", allow_module_level=True)

_SCALE = 1.0 / (192.0**0.5)
_KV_LORA = 512


def _oracle(q, kv, indices, positions, *, read_block):
    B, S, H, Dk = q.shape
    T = kv.shape[1]
    tok = np.asarray(units_to_token_ids(jnp.asarray(indices), read_block))
    q = np.asarray(q, np.float32)
    kv = np.asarray(kv, np.float32)
    pos = np.asarray(positions)
    out = np.zeros((B, S, H, _KV_LORA), np.float32)
    for b in range(B):
        for s in range(S):
            ids = tok[b, s]
            valid = (ids >= 0) & (ids <= pos[b, s]) & (ids < T)
            ids_safe = np.where(valid, ids, 0)
            k_sel = kv[b, ids_safe]
            logits = (q[b, s] @ k_sel.T) * _SCALE
            logits = np.where(valid[None, :], logits, -np.inf)
            m = logits.max(-1, keepdims=True)
            m = np.where(np.isneginf(m), 0.0, m)
            p = np.exp(logits - m)
            p = np.where(valid[None, :], p, 0.0)
            denom = p.sum(-1, keepdims=True)
            out[b, s] = (p / np.where(denom == 0.0, 1.0, denom)) @ k_sel[:, :_KV_LORA]
    return out


def _case(*, B, S, H, K, pages, dtype, page_size=128, seed=0):
    rng = np.random.default_rng(seed)
    Dk = _KV_LORA + 64
    T = page_size * pages
    q = jnp.asarray(rng.standard_normal((B, S, H, Dk)) * 0.1, dtype)
    kv = jnp.asarray(rng.standard_normal((B, T, Dk)) * 0.1, dtype)
    positions = np.zeros((B, S), np.int32)
    for b in range(B):
        positions[b] = np.linspace(page_size // 2, T - 1, S).astype(np.int32)
    idx = np.full((B, S, K), -1, np.int32)
    for b in range(B):
        for s in range(S):
            hi = int(positions[b, s] // page_size + 1)
            n = min(K, hi)
            ch = rng.choice(hi, size=n, replace=False)
            if 0 not in ch:
                ch[0] = 0
            idx[b, s, :n] = ch
    return q, kv, jnp.asarray(idx), jnp.asarray(positions), page_size


@pytest.mark.parametrize(
    "B, S, H, K, pages, qb, dtype, tol",
    [
        # deployment shape: H=4/device, QB=64 (QBH=256), bf16
        (1, 512, 4, 8, 16, 64, jnp.bfloat16, 3e-2),
        # fp32 tight-tolerance correctness anchor
        (1, 128, 4, 4, 8, 64, jnp.float32, 2e-3),
        # QBH exactly 128, multi-batch
        (2, 96, 8, 4, 8, 16, jnp.float32, 2e-3),
        # ragged tail block (S % QB != 0)
        (1, 200, 4, 6, 12, 64, jnp.bfloat16, 3e-2),
    ],
)
def test_qblock_tpu_parity(B, S, H, K, pages, qb, dtype, tol):
    q, kv, idx, pos, ps = _case(B=B, S=S, H=H, K=K, pages=pages, dtype=dtype, seed=B + S)
    common = dict(kv_lora_rank=_KV_LORA, read_block=ps, sm_scale=_SCALE)
    out_qb = np.asarray(sparse_mla_attention_qblock(q, kv, idx, pos, query_block=qb, **common))
    out_cur = np.asarray(sparse_mla_attention(q, kv, idx, pos, block_units=K, **common))
    ref = _oracle(q, kv, idx, pos, read_block=ps)
    e_qb = np.abs(out_qb - ref).max()
    e_cur = np.abs(out_cur - ref).max()
    e_x = np.abs(out_qb - out_cur).max()
    print(
        f"[tpu qblock {dtype.__name__} S={S}] |qb-ref|={e_qb:.3e} |cur-ref|={e_cur:.3e} |qb-cur|={e_x:.3e}"
    )
    assert e_qb < tol, f"qblock vs oracle: {e_qb}"
    assert e_x < tol, f"qblock vs deployed kernel: {e_x}"


# ── packed-ragged TPU parity: qblock ragged vs deployed ragged wrapper ──────

from sgl_jax.srt.kernels.dsa.sparse_mla_prefill import prefill_write_and_attend_ragged
from sgl_jax.srt.kernels.dsa.sparse_mla_prefill_qblock import (
    prefill_write_and_attend_ragged_qblock,
)

_PS = 128
_ROPE = 64
_DK_PAD = 640


@pytest.mark.parametrize(
    "seq_lens_list, qb, tol",
    [
        ([640, 1024, 1, 200], 48, 3e-2),  # varlen, blocks straddle requests
        ([1024, 1024], 64, 3e-2),  # aligned blocks
    ],
)
def test_ragged_qblock_tpu(seq_lens_list, qb, tol):
    rng = np.random.default_rng(17)
    H = 4
    K = 6
    num_seqs = len(seq_lens_list)
    pages_per_seq = int(np.ceil(max(seq_lens_list) / _PS))
    S_pad = pages_per_seq * _PS
    total = num_seqs * S_pad
    num_pages = num_seqs * pages_per_seq
    phys = rng.permutation(num_pages).reshape(num_seqs, pages_per_seq).astype(np.int32)

    ql = np.zeros((total, H, _KV_LORA), np.float32)
    qpe = np.zeros((total, H, _ROPE), np.float32)
    kvc = np.zeros((total, _KV_LORA), np.float32)
    kpe = np.zeros((total, _ROPE), np.float32)
    positions = np.zeros(total, np.int32)
    loc = np.full(total, -1, np.int32)
    tp = np.full((total, K), -1, np.int32)
    real = np.zeros(total, bool)
    for i, L in enumerate(seq_lens_list):
        base = i * S_pad
        ql[base : base + L] = rng.standard_normal((L, H, _KV_LORA)) * 0.1
        qpe[base : base + L] = rng.standard_normal((L, H, _ROPE)) * 0.1
        kvc[base : base + L] = rng.standard_normal((L, _KV_LORA)) * 0.1
        kpe[base : base + L] = rng.standard_normal((L, _ROPE)) * 0.1
        for j in range(L):
            t = base + j
            real[t] = True
            positions[t] = j
            loc[t] = phys[i, j // _PS] * _PS + (j % _PS)
            hi = j // _PS + 1
            n = min(K, hi)
            ch = rng.choice(hi, size=n, replace=False)
            if 0 not in ch:
                ch[0] = 0
            tp[t, :n] = ch

    args = (
        jnp.asarray(ql, jnp.bfloat16),
        jnp.asarray(qpe, jnp.bfloat16),
        jnp.asarray(kvc, jnp.bfloat16),
        jnp.asarray(kpe, jnp.bfloat16),
        jnp.zeros((num_pages, _PS // 2, 2, _DK_PAD), jnp.bfloat16),
        jnp.asarray(tp),
        jnp.asarray(positions),
        jnp.asarray(loc),
        jnp.asarray(seq_lens_list, jnp.int32),
        jnp.asarray(np.arange(num_seqs + 1) * S_pad, jnp.int32),
        jnp.asarray(np.arange(num_seqs + 1) * pages_per_seq * _PS, jnp.int32),
        jnp.asarray(phys.reshape(-1)),
    )
    kw = dict(kv_lora_rank=_KV_LORA, page_size=_PS, sm_scale=_SCALE)
    o_cur, cache_cur = prefill_write_and_attend_ragged(*args, **kw)
    o_qb, cache_qb = prefill_write_and_attend_ragged_qblock(*args, query_block=qb, **kw)
    d_o = np.abs(np.asarray(o_qb)[real] - np.asarray(o_cur)[real]).max()
    d_c = np.abs(
        np.asarray(cache_qb, dtype=np.float32) - np.asarray(cache_cur, dtype=np.float32)
    ).max()
    print(f"[tpu ragged qblock qb={qb}] |o_qb-o_cur|={d_o:.3e} |cache diff|={d_c:.3e}")
    assert d_c == 0.0, "self-write must be identical"
    assert d_o < tol, f"ragged qblock vs deployed drifted: {d_o}"


# ── pallas write-back vs XLA scatter (bit-identical contract) ────────────────

from sgl_jax.srt.kernels.dsa.sparse_mla_prefill_qblock import paged_write_back


@pytest.mark.parametrize(
    "case",
    [
        # single-seq page-aligned chunk (the 110k shape): one giant run
        dict(T=512, pages=6, locs="contig", seed=0),
        # multi-request, non-adjacent physical pages: run per page
        dict(T=384, pages=8, locs="scattered_pages", seed=1),
        # odd offsets: run starts mid-word (kv_packing phase mismatch paths)
        dict(T=131, pages=4, locs="odd", seed=2),
        # padded tail: loc == -1 must be dropped (canary)
        dict(T=256, pages=4, locs="padded", seed=3),
    ],
)
def test_paged_write_back_parity(case):
    rng = np.random.default_rng(case["seed"])
    ps, pk, D = 128, 2, 640
    T, Pn = case["T"], case["pages"]
    row = jnp.asarray(rng.standard_normal((T, D)) * 0.1, jnp.bfloat16)
    cache = jnp.asarray(rng.standard_normal((Pn, ps // pk, pk, D)) * 0.1, jnp.bfloat16)

    if case["locs"] == "contig":
        loc = np.arange(T, dtype=np.int32) + ps  # starts at page 1, aligned
    elif case["locs"] == "scattered_pages":
        pages = rng.permutation(Pn)[: -(-T // ps)]
        loc = np.concatenate(
            [p * ps + np.arange(min(ps, T - i * ps)) for i, p in enumerate(pages)]
        ).astype(np.int32)
    elif case["locs"] == "odd":
        loc = np.arange(T, dtype=np.int32) + ps + 1  # word-phase mismatch start
    else:  # padded
        loc = np.concatenate([np.arange(T - 64) + ps, np.full(64, -1)]).astype(np.int32)
    loc = jnp.asarray(loc)

    # oracle: the XLA flat scatter
    flat = cache.reshape(Pn * ps, D)
    want = flat.at[loc].set(row, mode="drop", wrap_negative_indices=False).reshape(cache.shape)
    got = paged_write_back(cache, row, loc, page_size=ps)
    np.testing.assert_array_equal(
        np.asarray(got, dtype=np.float32), np.asarray(want, dtype=np.float32)
    )
