"""Parity: blocked (query-batched) sparse-MLA prefill kernel vs the deployed
per-query kernel AND a masked-softmax numpy reference (flat KV, v1 scope).

Runs on CPU via ``interpret=True``.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_platform_name", "cpu")

_HERE = os.path.dirname(__file__)


def _load(name):
    path = os.path.normpath(
        os.path.join(_HERE, f"../../../../python/sgl_jax/srt/kernels/dsa/{name}.py")
    )
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


smp = _load("sparse_mla_prefill")
qb_mod = _load("sparse_mla_prefill_qblock")

sparse_mla_attention = smp.sparse_mla_attention
units_to_token_ids = smp.units_to_token_ids
sparse_mla_attention_qblock = qb_mod.sparse_mla_attention_qblock


def _reference(q, kv, indices, positions, *, read_block, kv_lora_rank, sm_scale):
    """Masked-softmax sparse MLA over the selected units, fp32 (same oracle as
    test_sparse_mla_prefill_parity)."""
    B, S, H, Dk = q.shape
    T = kv.shape[1]
    Dv = kv_lora_rank
    tok = np.asarray(units_to_token_ids(jnp.asarray(indices), read_block))
    q = np.asarray(q, np.float32)
    kv = np.asarray(kv, np.float32)
    pos = np.asarray(positions)
    out = np.zeros((B, S, H, Dv), np.float32)
    for b in range(B):
        for s in range(S):
            ids = tok[b, s]
            valid = (ids >= 0) & (ids <= pos[b, s]) & (ids < T)
            ids_safe = np.where(valid, ids, 0)
            k_sel = kv[b, ids_safe]
            logits = (q[b, s] @ k_sel.T) * sm_scale
            logits = np.where(valid[None, :], logits, -np.inf)
            m = logits.max(-1, keepdims=True)
            m = np.where(np.isneginf(m), 0.0, m)
            p = np.exp(logits - m)
            p = np.where(valid[None, :], p, 0.0)
            denom = p.sum(-1, keepdims=True)
            p = p / np.where(denom == 0.0, 1.0, denom)
            out[b, s] = p @ k_sel[:, :Dv]
    return out


def _make_case(*, B, S, H, K, pages, page_size=128, seed=0, empty_rows=()):
    rng = np.random.default_rng(seed)
    kv_lora_rank, rope = 512, 64
    Dk = kv_lora_rank + rope
    T = page_size * pages
    q = jnp.asarray(rng.standard_normal((B, S, H, Dk)) * 0.1, jnp.float32)
    kv = jnp.asarray(rng.standard_normal((B, T, Dk)) * 0.1, jnp.float32)
    positions = np.zeros((B, S), np.int32)
    for b in range(B):
        positions[b] = np.linspace(page_size // 2, T - 1, S).astype(np.int32)
    idx = np.full((B, S, K), -1, np.int32)
    for b in range(B):
        for s in range(S):
            hi = int(positions[b, s] // page_size + 1)
            n = min(K, hi)
            choices = rng.choice(hi, size=n, replace=False)
            if 0 not in choices:
                choices[0] = 0
            idx[b, s, :n] = choices
    for t in empty_rows:  # fully -1 rows (padding-like queries)
        idx[:, t, :] = -1
    return dict(
        q=q,
        kv=kv,
        indices=jnp.asarray(idx),
        positions=jnp.asarray(positions),
        kv_lora_rank=kv_lora_rank,
        page_size=page_size,
        sm_scale=1.0 / (192.0**0.5),
    )


def _run_both(c, *, query_block):
    common = dict(
        kv_lora_rank=c["kv_lora_rank"],
        read_block=c["page_size"],
        sm_scale=c["sm_scale"],
        interpret=True,
    )
    ref = _reference(
        c["q"],
        c["kv"],
        c["indices"],
        c["positions"],
        read_block=c["page_size"],
        kv_lora_rank=c["kv_lora_rank"],
        sm_scale=c["sm_scale"],
    )
    out_qb = np.asarray(
        sparse_mla_attention_qblock(
            c["q"], c["kv"], c["indices"], c["positions"], query_block=query_block, **common
        )
    )
    out_cur = np.asarray(
        sparse_mla_attention(
            c["q"],
            c["kv"],
            c["indices"],
            c["positions"],
            block_units=c["indices"].shape[2],
            **common,
        )
    )
    e_qb = np.abs(out_qb - ref).max()
    e_cur = np.abs(out_cur - ref).max()
    e_x = np.abs(out_qb - out_cur).max()
    print(f"[qblock] |qb-ref|={e_qb:.3e} |cur-ref|={e_cur:.3e} |qb-cur|={e_x:.3e}")
    assert e_qb < 2e-3, f"qblock vs reference parity failed: {e_qb}"
    assert e_x < 2e-3, f"qblock vs deployed kernel drifted: {e_x}"


def test_deployment_shape_qb64_h4():
    # deployment-like: H=4 heads/device, QB=64 (QBH=256), 6 pages, 2 blocks
    c = _make_case(B=1, S=128, H=4, K=3, pages=6, seed=0)
    _run_both(c, query_block=64)


def test_out_dtype_bf16_matches_f32_cast():
    """out_dtype=bf16 stores the f32-normalised result cast on store: bit-identical to the
    f32 output cast afterwards (the TSA backend's first consumer casts to bf16 anyway)."""
    c = _make_case(B=1, S=128, H=4, K=3, pages=6, seed=5)
    common = dict(
        kv_lora_rank=c["kv_lora_rank"],
        read_block=c["page_size"],
        sm_scale=c["sm_scale"],
        interpret=True,
        query_block=64,
    )
    o32 = sparse_mla_attention_qblock(c["q"], c["kv"], c["indices"], c["positions"], **common)
    o16 = sparse_mla_attention_qblock(
        c["q"], c["kv"], c["indices"], c["positions"], out_dtype=jnp.bfloat16, **common
    )
    assert o32.dtype == jnp.float32 and o16.dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(o32.astype(jnp.bfloat16)), np.asarray(o16))


def test_qbh_exact_128():
    # QB*H == 128 exactly (no lane padding), B=2
    c = _make_case(B=2, S=48, H=8, K=3, pages=4, seed=1)
    _run_both(c, query_block=16)


def test_ragged_tail_block():
    # S not a multiple of QB: trailing block is partially padded
    c = _make_case(B=1, S=45, H=4, K=2, pages=4, seed=2)
    _run_both(c, query_block=32)


def test_empty_and_padded_queries():
    # some queries select nothing (all -1): their output must be 0 and must not
    # poison neighbours in the same block
    c = _make_case(B=1, S=40, H=4, K=2, pages=4, seed=3, empty_rows=(5, 6, 39))
    _run_both(c, query_block=32)
    out = np.asarray(
        sparse_mla_attention_qblock(
            c["q"],
            c["kv"],
            c["indices"],
            c["positions"],
            kv_lora_rank=c["kv_lora_rank"],
            read_block=c["page_size"],
            query_block=32,
            sm_scale=c["sm_scale"],
            interpret=True,
        )
    )
    assert np.abs(out[:, [5, 6, 39]]).max() == 0.0, "empty-selection rows must be zero"


def test_small_u_max_padding():
    # u_max smaller than U_pad tiling forces -1 padded unit slots beyond counts
    c = _make_case(B=1, S=64, H=4, K=2, pages=3, seed=4)
    common = dict(
        kv_lora_rank=c["kv_lora_rank"],
        read_block=c["page_size"],
        sm_scale=c["sm_scale"],
        interpret=True,
    )
    ref = _reference(
        c["q"],
        c["kv"],
        c["indices"],
        c["positions"],
        read_block=c["page_size"],
        kv_lora_rank=c["kv_lora_rank"],
        sm_scale=c["sm_scale"],
    )
    out = np.asarray(
        sparse_mla_attention_qblock(
            c["q"], c["kv"], c["indices"], c["positions"], query_block=64, u_max=3, **common
        )
    )
    err = np.abs(out - ref).max()
    print(f"[qblock u_max=pages] |qb-ref|={err:.3e}")
    assert err < 2e-3


if __name__ == "__main__":
    test_deployment_shape_qb64_h4()
    test_qbh_exact_128()
    test_ragged_tail_block()
    test_empty_and_padded_queries()
    test_small_u_max_padding()
    print("QBLOCK PARITY OK")


# ── packed-ragged parity: qblock ragged vs the deployed ragged wrapper ──────

prefill_write_and_attend_ragged = smp.prefill_write_and_attend_ragged
prefill_write_and_attend_ragged_qblock = qb_mod.prefill_write_and_attend_ragged_qblock

_PS = 128
_KV_LORA, _ROPE = 512, 64
_DK_PAD = 640
_H8 = 8
_SC = 1.0 / (192.0**0.5)


def _ragged_case(seq_lens_list, K, seed=0):
    """Static-shape packed batch: per-request query block padded to S_pad,
    physical pages a permutation (so bases/loc are genuinely exercised)."""
    rng = np.random.default_rng(seed)
    num_seqs = len(seq_lens_list)
    pages_per_seq = int(np.ceil(max(seq_lens_list) / _PS))
    S_pad = pages_per_seq * _PS
    total = num_seqs * S_pad
    num_pages = num_seqs * pages_per_seq
    phys = rng.permutation(num_pages).reshape(num_seqs, pages_per_seq).astype(np.int32)

    ql = np.zeros((total, _H8, _KV_LORA), np.float32)
    qpe = np.zeros((total, _H8, _ROPE), np.float32)
    kvc = np.zeros((total, _KV_LORA), np.float32)
    kpe = np.zeros((total, _ROPE), np.float32)
    positions = np.zeros(total, np.int32)
    loc = np.full(total, -1, np.int32)
    tp = np.full((total, K), -1, np.int32)
    real = np.zeros(total, bool)
    for i, L in enumerate(seq_lens_list):
        base = i * S_pad
        ql[base : base + L] = rng.standard_normal((L, _H8, _KV_LORA)) * 0.1
        qpe[base : base + L] = rng.standard_normal((L, _H8, _ROPE)) * 0.1
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
    return dict(
        ql=jnp.asarray(ql),
        qpe=jnp.asarray(qpe),
        kvc=jnp.asarray(kvc),
        kpe=jnp.asarray(kpe),
        cache=jnp.zeros((num_pages, _PS, 1, _DK_PAD), jnp.float32),
        topk_pages=jnp.asarray(tp),
        positions=jnp.asarray(positions),
        loc=jnp.asarray(loc),
        seq_lens=jnp.asarray(seq_lens_list, jnp.int32),
        cu_q_lens=jnp.asarray(np.arange(num_seqs + 1) * S_pad, jnp.int32),
        cu_kv_lens=jnp.asarray(np.arange(num_seqs + 1) * pages_per_seq * _PS, jnp.int32),
        page_indices=jnp.asarray(phys.reshape(-1)),
        real=real,
    )


@pytest.mark.parametrize(
    "seq_lens_list, qb",
    [
        # varlen incl. a 1-token (decode-like) row; QB=48 does not divide the
        # 256-token per-request padding => blocks straddle request boundaries
        ([200, 512, 1, 129], 48),
        # uniform blocks aligned to requests
        ([256, 256], 64),
    ],
)
def test_ragged_qblock_vs_deployed(seq_lens_list, qb):
    c = _ragged_case(seq_lens_list, K=4, seed=11)
    args = (
        c["ql"],
        c["qpe"],
        c["kvc"],
        c["kpe"],
        c["cache"],
        c["topk_pages"],
        c["positions"],
        c["loc"],
        c["seq_lens"],
        c["cu_q_lens"],
        c["cu_kv_lens"],
        c["page_indices"],
    )
    kw = dict(kv_lora_rank=_KV_LORA, page_size=_PS, sm_scale=_SC, interpret=True)
    o_cur, cache_cur = prefill_write_and_attend_ragged(*args, **kw)
    o_qb, cache_qb = prefill_write_and_attend_ragged_qblock(*args, query_block=qb, **kw)
    real = c["real"]
    d_o = np.abs(np.asarray(o_qb)[real] - np.asarray(o_cur)[real]).max()
    d_c = np.abs(np.asarray(cache_qb) - np.asarray(cache_cur)).max()
    print(f"[ragged qblock qb={qb}] |o_qb-o_cur|={d_o:.3e} |cache diff|={d_c:.3e}")
    assert d_c == 0.0, "self-write must be identical"
    assert d_o < 2e-3, f"ragged qblock vs deployed ragged drifted: {d_o}"


# ── TSA (token-sharded attention): sharded slices vs one full-chunk call ────

prefill_write_and_attend_ragged_qblock_tsa = qb_mod.prefill_write_and_attend_ragged_qblock_tsa


@pytest.mark.parametrize(
    "seq_lens_list, shards, qb",
    [
        # varlen incl. 1-token row; shard boundaries straddle request
        # boundaries AND qblock boundaries
        ([200, 512, 1, 129], 4, 16),
        # short seqs: shards dominated by padded rows (tail-group coverage)
        ([130, 70], 4, 16),
    ],
)
def test_tsa_sharded_vs_full(seq_lens_list, shards, qb):
    c = _ragged_case(seq_lens_list, K=4, seed=13)
    kw = dict(kv_lora_rank=_KV_LORA, page_size=_PS, sm_scale=_SC, interpret=True)
    o_full, cache_full = prefill_write_and_attend_ragged_qblock(
        c["ql"],
        c["qpe"],
        c["kvc"],
        c["kpe"],
        c["cache"],
        c["topk_pages"],
        c["positions"],
        c["loc"],
        c["seq_lens"],
        c["cu_q_lens"],
        c["cu_kv_lens"],
        c["page_indices"],
        query_block=qb,
        **kw,
    )
    T = c["ql"].shape[0]
    assert T % shards == 0
    n = T // shards
    outs, d_c = [], 0.0
    for s in range(shards):
        sl = slice(s * n, (s + 1) * n)
        o_s, cache_s = prefill_write_and_attend_ragged_qblock_tsa(
            c["ql"][sl],
            c["qpe"][sl],
            c["kvc"],
            c["kpe"],
            c["cache"],
            c["topk_pages"][sl],
            c["positions"][sl],
            c["loc"],
            c["seq_lens"],
            c["cu_q_lens"],
            c["cu_kv_lens"],
            c["page_indices"],
            query_block=qb,
            q_token_offset=s * n,
            **kw,
        )
        outs.append(np.asarray(o_s))
        d_c = max(d_c, np.abs(np.asarray(cache_s) - np.asarray(cache_full)).max())
    o_tsa = np.concatenate(outs, 0)
    real = c["real"]
    d_o = np.abs(o_tsa[real] - np.asarray(o_full)[real]).max()
    print(f"[tsa shards={shards} qb={qb}] |o_tsa-o_full|={d_o:.3e} |cache diff|={d_c:.3e}")
    assert d_c == 0.0, "every shard must self-write the identical full-chunk cache"
    assert d_o < 2e-3, f"TSA sharded attend drifted from full-chunk qblock: {d_o}"


@pytest.mark.parametrize(
    "seq_lens_list, shards, qb",
    [
        ([200, 512, 1, 129], 4, 16),  # 1024 rows = 16 blocks x 4 shards: blocks straddle requests
        ([130, 70], 4, 16),  # 512 rows = 8 blocks x 4 shards: tail blocks mostly padding
    ],
)
def test_tsa_zigzag_vs_full(seq_lens_list, shards, qb):
    """Block-strided (zigzag) token assignment: shard s owns query blocks s, s+shards, ...
    Every block is the same 16 consecutive tokens as in the full-chunk call, so the
    attend output must be bit-identical regardless of which shard ran it."""
    c = _ragged_case(seq_lens_list, K=4, seed=13)
    kw = dict(kv_lora_rank=_KV_LORA, page_size=_PS, sm_scale=_SC, interpret=True)
    o_full, cache_full = prefill_write_and_attend_ragged_qblock(
        c["ql"],
        c["qpe"],
        c["kvc"],
        c["kpe"],
        c["cache"],
        c["topk_pages"],
        c["positions"],
        c["loc"],
        c["seq_lens"],
        c["cu_q_lens"],
        c["cu_kv_lens"],
        c["page_indices"],
        query_block=qb,
        **kw,
    )
    T = c["ql"].shape[0]
    assert T % (shards * qb) == 0
    nb = T // qb
    o_zz = np.zeros_like(np.asarray(o_full))
    for s in range(shards):
        blocks = np.arange(s, nb, shards)
        idx = (blocks[:, None] * qb + np.arange(qb)[None, :]).reshape(-1).astype(np.int32)
        o_s, cache_s = prefill_write_and_attend_ragged_qblock_tsa(
            c["ql"][idx],
            c["qpe"][idx],
            c["kvc"],
            c["kpe"],
            c["cache"],
            c["topk_pages"][idx],
            c["positions"][idx],
            c["loc"],
            c["seq_lens"],
            c["cu_q_lens"],
            c["cu_kv_lens"],
            c["page_indices"],
            query_block=qb,
            q_token_index=jnp.asarray(idx),
            **kw,
        )
        assert np.abs(np.asarray(cache_s) - np.asarray(cache_full)).max() == 0.0
        o_zz[idx] = np.asarray(o_s)
    real = c["real"]
    d_o = np.abs(o_zz[real] - np.asarray(o_full)[real]).max()
    print(f"[tsa zigzag shards={shards} qb={qb}] |o_zz-o_full|={d_o:.3e}")
    assert (
        d_o == 0.0
    ), f"zigzag shard assignment must be bit-identical to the full-chunk qblock: {d_o}"


def test_tsa_zigzag_index_matches_offset():
    """q_token_index with a contiguous arange must reproduce q_token_offset exactly."""
    c = _ragged_case([200, 512, 1, 129], K=4, seed=13)
    kw = dict(kv_lora_rank=_KV_LORA, page_size=_PS, sm_scale=_SC, interpret=True)
    T = c["ql"].shape[0]
    n = T // 4
    sl = slice(n, 2 * n)
    args = (
        c["ql"][sl],
        c["qpe"][sl],
        c["kvc"],
        c["kpe"],
        c["cache"],
        c["topk_pages"][sl],
        c["positions"][sl],
        c["loc"],
        c["seq_lens"],
        c["cu_q_lens"],
        c["cu_kv_lens"],
        c["page_indices"],
    )
    o_a, _ = prefill_write_and_attend_ragged_qblock_tsa(
        *args, query_block=16, q_token_offset=n, **kw
    )
    o_b, _ = prefill_write_and_attend_ragged_qblock_tsa(
        *args, query_block=16, q_token_index=jnp.arange(n, 2 * n, dtype=jnp.int32), **kw
    )
    assert np.abs(np.asarray(o_a) - np.asarray(o_b)).max() == 0.0


# ── argument validation: ragged page-size guard, u_max truncation warning ───


def test_ragged_rejects_non16_page_size():
    # RB == page_size == 24 passes the equality check but must hit the %16
    # guard (same contract as the per-query paged kernel) before any layout
    # work — the arrays are never touched, so dummies suffice.
    S = 8
    q = jnp.zeros((1, S, 2, 576), jnp.float32)
    kv = jnp.zeros((4, 24, 1, _DK_PAD), jnp.float32)
    idx = jnp.full((1, S, 2), -1, jnp.int32)
    pos = jnp.zeros((1, S), jnp.int32)
    with pytest.raises(ValueError, match="read_block % 16 == 0"):
        sparse_mla_attention_qblock(
            q,
            kv,
            idx,
            pos,
            read_block=24,
            sm_scale=_SC,
            page_size=24,
            q_seq_id=jnp.zeros((S,), jnp.int32),
            seq_lens=jnp.asarray([S], jnp.int32),
            cu_kv_lens=jnp.asarray([0, 96], jnp.int32),
            page_indices=jnp.asarray([0, 1, 2, 3], jnp.int32),
        )


def test_umax_reduced_warns_once(caplog):
    # explicit u_max below the lossless bound min(QB*K, num_units) must log a
    # one-time warning; the default (None) must stay silent.
    qb_mod._warn_umax_truncation.cache_clear()
    c = _make_case(B=1, S=64, H=4, K=2, pages=4, seed=5)
    common = dict(
        kv_lora_rank=c["kv_lora_rank"],
        read_block=c["page_size"],
        query_block=64,
        sm_scale=c["sm_scale"],
        interpret=True,
    )
    with caplog.at_level("WARNING", logger=qb_mod.__name__):
        sparse_mla_attention_qblock(
            c["q"], c["kv"], c["indices"], c["positions"], u_max=3, **common
        )
        sparse_mla_attention_qblock(
            c["q"], c["kv"], c["indices"], c["positions"], u_max=3, **common
        )
    hits = [r for r in caplog.records if "u_max" in r.getMessage()]
    assert len(hits) == 1, f"expected exactly one truncation warning, got {len(hits)}"

    caplog.clear()
    with caplog.at_level("WARNING", logger=qb_mod.__name__):
        sparse_mla_attention_qblock(c["q"], c["kv"], c["indices"], c["positions"], **common)
    assert not [
        r for r in caplog.records if "u_max" in r.getMessage()
    ], "default u_max must not warn"


def test_head_major_rows_bit_identical():
    """head_major=True (q [B,H,S,Dk], rows h*QB+q) is a pure row permutation of the
    token-major kernel: outputs must be bit-identical (interpret)."""
    c = _make_case(B=1, S=200, H=4, K=3, pages=8, seed=11, empty_rows=(7, 199))
    common = dict(
        kv_lora_rank=c["kv_lora_rank"],
        read_block=c["page_size"],
        sm_scale=c["sm_scale"],
        interpret=True,
        query_block=32,  # 32 * 4 = 128 rows: no lane padding
    )
    o_tm = sparse_mla_attention_qblock(c["q"], c["kv"], c["indices"], c["positions"], **common)
    o_hm = sparse_mla_attention_qblock(
        jnp.transpose(c["q"], (0, 2, 1, 3)),
        c["kv"],
        c["indices"],
        c["positions"],
        head_major=True,
        **common,
    )
    assert o_hm.shape == (1, 4, 200, c["kv_lora_rank"])
    np.testing.assert_array_equal(np.asarray(jnp.transpose(o_hm, (0, 2, 1, 3))), np.asarray(o_tm))


def test_head_major_rejects_lane_padded_rows():
    c = _make_case(B=1, S=64, H=4, K=2, pages=4, seed=12)
    with pytest.raises(ValueError, match="head_major"):
        sparse_mla_attention_qblock(
            jnp.transpose(c["q"], (0, 2, 1, 3)),
            c["kv"],
            c["indices"],
            c["positions"],
            kv_lora_rank=c["kv_lora_rank"],
            read_block=c["page_size"],
            sm_scale=c["sm_scale"],
            interpret=True,
            query_block=16,  # 16 * 4 = 64 rows -> would need padding to 128
            head_major=True,
        )


def test_tsa_head_major_matches_token_major():
    """TSA wrapper: head-major q/o (the absorb-dot natural layouts) equals the token-major
    path bit for bit, cache write included; the a2a/zigzag-sharded row slice is the same."""
    c = _ragged_case([200, 512, 1, 129], K=4, seed=21)
    H = c["ql"].shape[1]
    qb = 128 // H if 128 % H == 0 else 128
    kw = dict(kv_lora_rank=_KV_LORA, page_size=_PS, sm_scale=_SC, interpret=True, query_block=qb)
    T = c["ql"].shape[0]
    n = T // 4
    sl = slice(n, 2 * n)  # an interior shard: block boundaries straddle requests
    args = (
        c["kvc"],
        c["kpe"],
        c["cache"],
        c["topk_pages"][sl],
        c["positions"][sl],
        c["loc"],
        c["seq_lens"],
        c["cu_q_lens"],
        c["cu_kv_lens"],
        c["page_indices"],
    )
    o_tm, cache_tm = prefill_write_and_attend_ragged_qblock_tsa(
        c["ql"][sl], c["qpe"][sl], *args, q_token_offset=n, **kw
    )
    o_hm, cache_hm = prefill_write_and_attend_ragged_qblock_tsa(
        jnp.transpose(c["ql"][sl], (1, 0, 2)),
        jnp.transpose(c["qpe"][sl], (1, 0, 2)),
        *args,
        q_token_offset=n,
        head_major=True,
        **kw,
    )
    assert o_hm.shape == (H, n, _KV_LORA)
    np.testing.assert_array_equal(np.asarray(cache_hm), np.asarray(cache_tm))
    np.testing.assert_array_equal(np.asarray(jnp.transpose(o_hm, (1, 0, 2))), np.asarray(o_tm))


def _assert_same_schedule(got, ref):
    """Bit-identical on the CPU backend; one f32 ulp on TPU hosts (see test_row_groups_two_bit_identical)."""
    if jax.default_backend() == "cpu":
        np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
    else:
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-6, atol=0)


def test_row_groups_two_bit_identical(monkeypatch):
    """ROW_GROUPS=2 issues both groups' QK before any PV; rows are independent so the
    result equals the single-group schedule (interpret), head-major and token-major, VMEM
    flash state on and off. Bit-identical on the CPU backend; on a TPU host the interpreter
    accumulates the 8-page units of the two groups in a different f32 order and the results
    differ by one ulp (max rel 2.4e-7 on v7x, the TPU test pod), so there the comparison is
    a 1-ulp allclose."""
    c = _make_case(B=1, S=200, H=4, K=3, pages=8, seed=17, empty_rows=(3, 150))
    common = dict(
        kv_lora_rank=c["kv_lora_rank"],
        read_block=c["page_size"],
        sm_scale=c["sm_scale"],
        interpret=True,
        query_block=64,  # 64 * 4 = 256 rows -> two 128-lane groups
    )
    monkeypatch.setattr(qb_mod, "_ROW_GROUPS", 1)
    ref = sparse_mla_attention_qblock(c["q"], c["kv"], c["indices"], c["positions"], **common)
    monkeypatch.setattr(qb_mod, "_ROW_GROUPS", 2)
    got = sparse_mla_attention_qblock(c["q"], c["kv"], c["indices"], c["positions"], **common)
    _assert_same_schedule(got, ref)
    got_hm = sparse_mla_attention_qblock(
        jnp.transpose(c["q"], (0, 2, 1, 3)),
        c["kv"],
        c["indices"],
        c["positions"],
        head_major=True,
        **common,
    )
    _assert_same_schedule(jnp.transpose(got_hm, (0, 2, 1, 3)), ref)
    monkeypatch.setattr(
        qb_mod, "_ACC_SCRATCH", True
    )  # production form: flash state in VMEM scratch
    got2 = sparse_mla_attention_qblock(c["q"], c["kv"], c["indices"], c["positions"], **common)
    if jax.default_backend() == "cpu":
        _assert_same_schedule(got2, ref)
    else:
        np.testing.assert_allclose(np.asarray(got2), np.asarray(ref), rtol=1e-6, atol=0)


def test_row_groups_falls_back_when_rows_not_divisible(monkeypatch):
    # 32 * 4 = 128 rows cannot split into two 128-lane groups -> single group, same result
    c = _make_case(B=1, S=64, H=4, K=2, pages=4, seed=18)
    common = dict(
        kv_lora_rank=c["kv_lora_rank"],
        read_block=c["page_size"],
        sm_scale=c["sm_scale"],
        interpret=True,
        query_block=32,
    )
    monkeypatch.setattr(qb_mod, "_ROW_GROUPS", 1)
    ref = sparse_mla_attention_qblock(c["q"], c["kv"], c["indices"], c["positions"], **common)
    monkeypatch.setattr(qb_mod, "_ROW_GROUPS", 2)
    got = sparse_mla_attention_qblock(c["q"], c["kv"], c["indices"], c["positions"], **common)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
