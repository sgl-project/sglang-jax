"""Parity: fused sparse-MLA prefill kernel vs a masked-softmax reference.

Validates ``sparse_mla_attention`` (poc kernel, placed at
``sgl_jax.srt.kernels.dsa.sparse_mla_prefill``) at **page-level granularity**
(``read_block == page_size``), which is exactly how PR1 wires it into the DSA
backend's EXTEND path: the indexer's ``topk_pages`` (seq-local page ids) become
the kernel's per-query unit ids.

Two modes are checked against the same reference:
  * flat  : ``kv`` is a flat [B, T, Dk] latent buffer.
  * paged : ``kv`` is the packed 4D MLA cache + per-seq page table.

Runs on CPU via ``interpret=True`` (no TPU needed).
"""

from __future__ import annotations

import importlib.util
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_platform_name", "cpu")

# Import the kernel module directly from its file (it only depends on jax/pallas,
# so we don't need the full sgl_jax package installed for this parity check).
_HERE = os.path.dirname(__file__)
_KERNEL_PATH = os.path.normpath(
    os.path.join(_HERE, "../../../../python/sgl_jax/srt/kernels/dsa/sparse_mla_prefill.py")
)
_spec = importlib.util.spec_from_file_location("sparse_mla_prefill", _KERNEL_PATH)
smp = importlib.util.module_from_spec(_spec)
sys.modules["sparse_mla_prefill"] = smp
_spec.loader.exec_module(smp)

sparse_mla_attention = smp.sparse_mla_attention
units_to_token_ids = smp.units_to_token_ids
flat_to_paged_cache = smp.flat_to_paged_cache
prefill_write_and_attend = smp.prefill_write_and_attend
prefill_write_and_attend_ragged = smp.prefill_write_and_attend_ragged


def _reference(q, kv, indices, positions, *, read_block, kv_lora_rank, sm_scale):
    """Masked-softmax sparse MLA over the selected units, in fp32.

    q:   [B, S, H, Dk]   kv: [B, T, Dk]   indices: [B, S, K] unit ids
    positions: [B, S]    returns [B, S, H, kv_lora_rank]
    """
    B, S, H, Dk = q.shape
    T = kv.shape[1]
    Dv = kv_lora_rank
    tok = np.asarray(units_to_token_ids(jnp.asarray(indices), read_block))  # [B,S,K*RB]
    q = np.asarray(q, np.float32)
    kv = np.asarray(kv, np.float32)
    pos = np.asarray(positions)
    out = np.zeros((B, S, H, Dv), np.float32)
    for b in range(B):
        for s in range(S):
            ids = tok[b, s]
            valid = (ids >= 0) & (ids <= pos[b, s]) & (ids < T)
            ids_safe = np.where(valid, ids, 0)
            k_sel = kv[b, ids_safe]  # [N, Dk]
            logits = (q[b, s] @ k_sel.T) * sm_scale  # [H, N]
            logits = np.where(valid[None, :], logits, -np.inf)
            m = logits.max(-1, keepdims=True)
            m = np.where(np.isneginf(m), 0.0, m)
            p = np.exp(logits - m)
            p = np.where(valid[None, :], p, 0.0)
            denom = p.sum(-1, keepdims=True)
            denom = np.where(denom == 0.0, 1.0, denom)
            p = p / denom
            out[b, s] = p @ k_sel[:, :Dv]  # [H, Dv]
    return out


def _make_case(seed=0):
    rng = np.random.default_rng(seed)
    B, S, H = 2, 6, 8
    kv_lora_rank, rope = 512, 64
    Dk = kv_lora_rank + rope
    page_size = 128
    pages_per_seq = 4
    T = page_size * pages_per_seq
    K = 3  # selected pages per query (page-level => RB=page_size)

    q = jnp.asarray(rng.standard_normal((B, S, H, Dk)) * 0.1, jnp.float32)
    kv = jnp.asarray(rng.standard_normal((B, T, Dk)) * 0.1, jnp.float32)

    # query positions: spread across the context so causal bounds vary per token
    positions = np.zeros((B, S), np.int32)
    for b in range(B):
        positions[b] = np.linspace(page_size, T - 1, S).astype(np.int32)
    positions = jnp.asarray(positions)

    # selection: page 0 (always causally valid) + random distinct pages, padded
    # with -1 when fewer than K pages are causally reachable (exercises the
    # kernel's topk-padding path — real indexer output is -1-padded).
    idx = np.full((B, S, K), -1, np.int32)
    for b in range(B):
        for s in range(S):
            hi = int(positions[b, s] // page_size + 1)  # causally-reachable pages
            n = min(K, hi)
            choices = rng.choice(hi, size=n, replace=False)
            if 0 not in choices:
                choices[0] = 0  # guarantee ≥1 valid key
            idx[b, s, :n] = choices
    indices = jnp.asarray(idx)
    return dict(
        q=q,
        kv=kv,
        indices=indices,
        positions=positions,
        kv_lora_rank=kv_lora_rank,
        page_size=page_size,
        T=T,
        K=K,
        sm_scale=1.0 / (Dk**0.5),
    )


def _run(mode="flat", seed=0):
    c = _make_case(seed)
    ref = _reference(
        c["q"],
        c["kv"],
        c["indices"],
        c["positions"],
        read_block=c["page_size"],
        kv_lora_rank=c["kv_lora_rank"],
        sm_scale=c["sm_scale"],
    )
    if mode == "flat":
        out = sparse_mla_attention(
            c["q"],
            c["kv"],
            c["indices"],
            c["positions"],
            kv_lora_rank=c["kv_lora_rank"],
            read_block=c["page_size"],
            block_units=c["K"],
            sm_scale=c["sm_scale"],
            interpret=True,
        )
    else:
        cache, page_table = flat_to_paged_cache(c["kv"], c["page_size"], kv_packing=2)
        # pad feature dim to Dk_pad (kernel does this for q internally; cache must match)
        Dk = c["q"].shape[-1]
        Dk_pad = ((Dk + 127) // 128) * 128
        if cache.shape[-1] != Dk_pad:
            cache = jnp.pad(cache, ((0, 0), (0, 0), (0, 0), (0, Dk_pad - Dk)))
        out = sparse_mla_attention(
            c["q"],
            cache,
            c["indices"],
            c["positions"],
            kv_lora_rank=c["kv_lora_rank"],
            read_block=c["page_size"],
            block_units=c["K"],
            sm_scale=c["sm_scale"],
            interpret=True,
            page_table=page_table,
            page_size=c["page_size"],
            seq_len=c["T"],
        )
    out = np.asarray(out)
    err = np.abs(out - ref)
    print(f"[{mode}] max|err|={err.max():.3e}  mean|err|={err.mean():.3e}  shape={out.shape}")
    assert err.max() < 2e-3, f"{mode}: parity failed, max err {err.max()}"
    return err.max()


def _run_write_attend(seed=0):
    """End-to-end: self-write latent into a paged cache, then page-level sparse
    attend — vs a masked-softmax reference over the written latent. Single-seq."""
    rng = np.random.default_rng(seed)
    T, H = 512, 8
    kv_lora_rank, rope = 512, 64
    Dk_pad = 640
    page_size = 128
    pages = T // page_size  # single request occupies pages 0..pages-1
    K = 3
    scale = 1.0 / ((kv_lora_rank + rope) ** 0.5)

    ql = jnp.asarray(rng.standard_normal((T, H, kv_lora_rank)) * 0.1, jnp.float32)
    qpe = jnp.asarray(rng.standard_normal((T, H, rope)) * 0.1, jnp.float32)
    kvc = jnp.asarray(rng.standard_normal((T, kv_lora_rank)) * 0.1, jnp.float32)
    kpe = jnp.asarray(rng.standard_normal((T, rope)) * 0.1, jnp.float32)
    # empty fp32 paged cache [P, ps//pk, pk, Dk_pad] with pk=1
    cache = jnp.zeros((pages, page_size, 1, Dk_pad), jnp.float32)
    loc = jnp.arange(T, dtype=jnp.int32)  # token t -> physical slot t
    positions = jnp.arange(T, dtype=jnp.int32)

    tp = np.full((T, K), -1, np.int32)
    for t in range(T):
        hi = t // page_size + 1
        n = min(K, hi)
        ch = rng.choice(hi, size=n, replace=False)
        if 0 not in ch:
            ch[0] = 0
        tp[t, :n] = ch
    topk_pages = jnp.asarray(tp)

    o, cache_new = prefill_write_and_attend(
        ql,
        qpe,
        kvc,
        kpe,
        cache,
        topk_pages,
        positions,
        loc,
        kv_lora_rank=kv_lora_rank,
        page_size=page_size,
        sm_scale=scale,
        interpret=True,
    )
    o = np.asarray(o)

    # reference over the written latent [T, 576]
    latent = np.concatenate([np.asarray(kvc), np.asarray(kpe)], axis=-1)  # [T, 576]
    q_full = np.concatenate([np.asarray(ql), np.asarray(qpe)], axis=-1)  # [T, H, 576]
    tok = np.asarray(units_to_token_ids(topk_pages, page_size)).reshape(T, -1)  # [T, K*ps]
    ref = np.zeros((T, H, kv_lora_rank), np.float32)
    for t in range(T):
        ids = tok[t]
        valid = (ids >= 0) & (ids <= t) & (ids < T)
        ids_safe = np.where(valid, ids, 0)
        k_sel = latent[ids_safe]
        logits = (q_full[t] @ k_sel.T) * scale
        logits = np.where(valid[None, :], logits, -np.inf)
        m = logits.max(-1, keepdims=True)
        m = np.where(np.isneginf(m), 0.0, m)
        p = np.exp(logits - m)
        p = np.where(valid[None, :], p, 0.0)
        denom = p.sum(-1, keepdims=True)
        ref[t] = (p / np.where(denom == 0.0, 1.0, denom)) @ k_sel[:, :kv_lora_rank]

    err = np.abs(o - ref)
    # also verify the self-write landed: cache row for token t == [kvc|kpe|pad]
    flat = np.asarray(cache_new).reshape(pages * page_size, Dk_pad)
    w_err = np.abs(flat[:T, :kv_lora_rank] - np.asarray(kvc)).max()
    print(f"[write+attend] max|err|={err.max():.3e}  self-write max|err|={w_err:.3e}")
    assert err.max() < 2e-3, f"attend parity failed: {err.max()}"
    assert w_err < 1e-6, f"self-write failed: {w_err}"
    return err.max()


def _run_write_attend_canary(seed=0):
    """Regression: padded tokens carry out_cache_loc == -1 and MUST be dropped from
    the self-write, NOT wrapped into the last physical slot (jax .at[].set(mode='drop')
    still wraps negatives). Place a canary in the final slot (owned by no real token)
    and a -1 loc for the padded tail; the canary must survive."""
    rng = np.random.default_rng(seed)
    T_real, T_pad, H = 256, 64, 8
    T = T_real + T_pad
    kv_lora_rank, rope = 512, 64
    Dk_pad = 640
    page_size = 128
    pages = 3  # real request uses pages 0..1; page 2 holds the canary
    K = 3
    scale = 1.0 / ((kv_lora_rank + rope) ** 0.5)

    ql = jnp.asarray(rng.standard_normal((T, H, kv_lora_rank)) * 0.1, jnp.float32)
    qpe = jnp.asarray(rng.standard_normal((T, H, rope)) * 0.1, jnp.float32)
    kvc = jnp.asarray(rng.standard_normal((T, kv_lora_rank)) * 0.1, jnp.float32)
    kpe = jnp.asarray(rng.standard_normal((T, rope)) * 0.1, jnp.float32)

    CANARY = 12345.0
    cache = np.zeros((pages, page_size, 1, Dk_pad), np.float32)
    cache[pages - 1, page_size - 1, 0, :] = CANARY  # very last physical slot
    cache = jnp.asarray(cache)

    # real tokens t -> slot t (pages 0..1); padded tail -> out_cache_loc == -1
    loc = jnp.asarray(np.concatenate([np.arange(T_real), np.full(T_pad, -1)]).astype(np.int32))
    positions = jnp.asarray(np.concatenate([np.arange(T_real), np.zeros(T_pad)]).astype(np.int32))

    tp = np.full((T, K), -1, np.int32)
    for t in range(T_real):
        hi = t // page_size + 1
        n = min(K, hi)
        ch = rng.choice(hi, size=n, replace=False)
        if 0 not in ch:
            ch[0] = 0
        tp[t, :n] = ch
    topk_pages = jnp.asarray(tp)

    _, cache_new = prefill_write_and_attend(
        ql,
        qpe,
        kvc,
        kpe,
        cache,
        topk_pages,
        positions,
        loc,
        kv_lora_rank=kv_lora_rank,
        page_size=page_size,
        sm_scale=scale,
        interpret=True,
    )
    flat = np.asarray(cache_new).reshape(pages * page_size, Dk_pad)
    canary_after = flat[pages * page_size - 1]
    assert np.allclose(
        canary_after, CANARY
    ), f"padded -1 loc corrupted the final KV slot: {canary_after[:3]} != {CANARY}"
    w_err = np.abs(flat[:T_real, :kv_lora_rank] - np.asarray(kvc)[:T_real]).max()
    assert w_err < 1e-6, f"self-write regressed: {w_err}"
    print(f"[write+attend canary] final-slot canary preserved; self-write max|err|={w_err:.3e}")


# ─────────────────────────────────────────────────────────────────────────────
# Packed-ragged (A3 batching) harness + invariant gates G0–G6.
#
# Models the static-shape serving batch: ``num_seqs`` requests packed along the
# token axis, each request's query block padded to a uniform ``S_pad`` tokens and
# its KV padded to a uniform ``pages_per_seq`` pages (so ``cu_kv_lens[i]//ps ==
# i*pages_per_seq`` and ``page_indices`` has ``num_seqs*pages_per_seq`` slots —
# exactly what ``streamindex_page_topk_ref`` / ``sparse_mla_ref`` / ``_scatter_paged``
# assume). Physical pages are a *permutation* so the per-request ``base`` and the
# self-write ``loc`` are genuinely exercised (identity would hide base bugs).
# ─────────────────────────────────────────────────────────────────────────────

_H = 8
_KV_LORA = 512
_ROPE = 64
_DK_PAD = 640
_PS = 128
_SCALE = 1.0 / ((_KV_LORA + _ROPE) ** 0.5)


def _seq_inputs(rng, L, K):
    """Self-contained inputs for ONE request of length ``L`` (single-shot, prefix 0):
    query, latent to write, and seq-local causal page-topk. Independent of packing
    position — the atom for the cross-run identity gates (G1/G2/G3)."""
    ql = np.asarray(rng.standard_normal((L, _H, _KV_LORA)) * 0.1, np.float32)
    qpe = np.asarray(rng.standard_normal((L, _H, _ROPE)) * 0.1, np.float32)
    kvc = np.asarray(rng.standard_normal((L, _KV_LORA)) * 0.1, np.float32)
    kpe = np.asarray(rng.standard_normal((L, _ROPE)) * 0.1, np.float32)
    tp = np.full((L, K), -1, np.int32)
    for j in range(L):
        hi = j // _PS + 1  # causally-reachable seq-local pages
        n = min(K, hi)
        ch = rng.choice(hi, size=n, replace=False)
        if 0 not in ch:
            ch[0] = 0
        tp[j, :n] = ch
    return dict(ql=ql, qpe=qpe, kvc=kvc, kpe=kpe, tp=tp, L=int(L))


def _assemble(seqs, seed_pages=0, page_perm=True):
    """Pack a list of per-request ``_seq_inputs`` into the static-shape batch:
    uniform ``S_pad`` query block and ``pages_per_seq`` KV pages per request,
    physical pages a (optional) permutation. Returns a case dict."""
    num_seqs = len(seqs)
    K = seqs[0]["tp"].shape[1]
    pages_per_seq = int(np.ceil(max(s["L"] for s in seqs) / _PS))
    S_pad = pages_per_seq * _PS
    total = num_seqs * S_pad
    num_pages = num_seqs * pages_per_seq

    prng = np.random.default_rng(seed_pages)
    perm = prng.permutation(num_pages) if page_perm else np.arange(num_pages)
    phys = perm.reshape(num_seqs, pages_per_seq).astype(np.int32)

    ql = np.zeros((total, _H, _KV_LORA), np.float32)
    qpe = np.zeros((total, _H, _ROPE), np.float32)
    kvc = np.zeros((total, _KV_LORA), np.float32)
    kpe = np.zeros((total, _ROPE), np.float32)
    positions = np.zeros(total, np.int32)
    loc = np.full(total, -1, np.int32)
    tp = np.full((total, K), -1, np.int32)
    real = np.zeros(total, bool)
    for i, s in enumerate(seqs):
        L = s["L"]
        base = i * S_pad
        ql[base : base + L] = s["ql"]
        qpe[base : base + L] = s["qpe"]
        kvc[base : base + L] = s["kvc"]
        kpe[base : base + L] = s["kpe"]
        tp[base : base + L] = s["tp"]
        for j in range(L):
            t = base + j
            real[t] = True
            positions[t] = j
            loc[t] = phys[i, j // _PS] * _PS + (j % _PS)

    cache = np.zeros((num_pages, _PS, 1, _DK_PAD), np.float32)
    seq_lens = np.asarray([s["L"] for s in seqs], np.int32)
    cu_q_lens = np.arange(num_seqs + 1, dtype=np.int32) * S_pad
    cu_kv_lens = np.arange(num_seqs + 1, dtype=np.int32) * (pages_per_seq * _PS)
    page_indices = phys.reshape(-1)

    # G6 metadata invariants (host-side, fail fast before the kernel).
    assert cu_q_lens[-1] == total == len(loc)
    assert page_indices.shape[0] == num_seqs * pages_per_seq
    assert ((tp >= -1) & (tp < pages_per_seq)).all(), "topk page id out of [-1, pages_per_seq)"

    return dict(
        ql=jnp.asarray(ql),
        qpe=jnp.asarray(qpe),
        kvc=jnp.asarray(kvc),
        kpe=jnp.asarray(kpe),
        cache=jnp.asarray(cache),
        topk_pages=jnp.asarray(tp),
        positions=jnp.asarray(positions),
        loc=jnp.asarray(loc),
        seq_lens=jnp.asarray(seq_lens),
        cu_q_lens=jnp.asarray(cu_q_lens),
        cu_kv_lens=jnp.asarray(cu_kv_lens),
        page_indices=jnp.asarray(page_indices),
        seq_lens_list=[s["L"] for s in seqs],
        S_pad=S_pad,
        pages_per_seq=pages_per_seq,
        real=real,
        _kvc=kvc,
        _kpe=kpe,
        _ql=ql,
        _qpe=qpe,
        _tp=tp,
    )


def _make_ragged_batch(seq_lens_list, K, seed=0, page_perm=True):
    rng = np.random.default_rng(seed)
    seqs = [_seq_inputs(rng, L, K) for L in seq_lens_list]
    return _assemble(seqs, seed_pages=seed, page_perm=page_perm)


def _ragged_reference(c):
    """Masked-softmax oracle: each request is an independent single-shot prefill
    over its own written latent (seq-local page ids => seq-local token ids)."""
    S_pad = c["S_pad"]
    total = c["_ql"].shape[0]
    out = np.zeros((total, _H, _KV_LORA), np.float32)
    latent = np.concatenate([c["_kvc"], c["_kpe"]], axis=-1)  # [total, 576]
    q_full = np.concatenate([c["_ql"], c["_qpe"]], axis=-1)  # [total, H, 576]
    tok = np.asarray(units_to_token_ids(jnp.asarray(c["_tp"]), _PS)).reshape(
        total, -1
    )  # [total, K*ps]
    for i, L in enumerate(c["seq_lens_list"]):
        base = i * S_pad
        for j in range(L):
            t = base + j
            ids = tok[t]  # seq-local token ids
            valid = (ids >= 0) & (ids <= j) & (ids < L)
            ids_safe = np.where(valid, base + ids, base)  # into global latent
            k_sel = latent[ids_safe]
            logits = (q_full[t] @ k_sel.T) * _SCALE
            logits = np.where(valid[None, :], logits, -np.inf)
            m = logits.max(-1, keepdims=True)
            m = np.where(np.isneginf(m), 0.0, m)
            p = np.exp(logits - m)
            p = np.where(valid[None, :], p, 0.0)
            denom = p.sum(-1, keepdims=True)
            out[t] = (p / np.where(denom == 0.0, 1.0, denom)) @ k_sel[:, :_KV_LORA]
    return out


def _run_case(c):
    o, cache_new = prefill_write_and_attend_ragged(
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
        kv_lora_rank=_KV_LORA,
        page_size=_PS,
        sm_scale=_SCALE,
        interpret=True,
    )
    return np.asarray(o), np.asarray(cache_new)


def _run_ragged(seq_lens_list, K, seed=0, page_perm=True):
    c = _make_ragged_batch(seq_lens_list, K, seed=seed, page_perm=page_perm)
    o, cache_new = _run_case(c)
    return o, cache_new, c


def _seq_slice(c, i):
    """Global token indices of request i's REAL tokens."""
    base = i * c["S_pad"]
    L = c["seq_lens_list"][i]
    return np.arange(base, base + L)


def test_ragged_parity_varlen():
    """Ragged multi-request prefill vs masked-softmax oracle (varying seq_lens)."""
    o, _, c = _run_ragged([384, 200, 512, 129], K=4, seed=1)
    ref = _ragged_reference(c)
    real = c["real"]
    err = np.abs(o[real] - ref[real])
    print(f"[ragged varlen] max|err|={err.max():.3e}  seqs={c['seq_lens_list']}  shape={o.shape}")
    assert err.max() < 2e-3, f"ragged parity failed: {err.max()}"


def test_ragged_self_write_and_canary():
    """G5: self-write landed for real tokens; padded (-1 loc) tokens touch nothing."""
    o, cache_new, c = _run_ragged([300, 128, 400], K=4, seed=2)
    flat = cache_new.reshape(-1, _DK_PAD)
    loc = np.asarray(c["loc"])
    kvc = c["_kvc"]
    real = c["real"]
    # every real token's latent landed at its physical slot
    w_err = np.abs(flat[loc[real], :_KV_LORA] - kvc[real]).max()
    assert w_err < 1e-6, f"ragged self-write failed: {w_err}"
    # no physical slot outside the union of real locs was written (canary: all
    # such slots must remain zero — the initial cache).
    written = set(loc[real].tolist())
    all_slots = set(range(flat.shape[0]))
    untouched = np.array(sorted(all_slots - written), dtype=np.int64)
    assert np.abs(flat[untouched]).max() < 1e-6, "padded/-1 write leaked into an unused slot"
    print(f"[ragged self-write] max|err|={w_err:.3e}  untouched_slots={len(untouched)}")


def test_gate_G0_single_seq_equivalence():
    """G0: the ragged path with num_seqs==1 matches both the oracle and the
    existing single-sequence prefill_write_and_attend on identical inputs."""
    L, K = 512, 3
    o_r, _, c = _run_ragged([L], K=K, seed=3, page_perm=False)  # identity pages
    ref = _ragged_reference(c)
    real = c["real"]
    assert np.abs(o_r[real] - ref[real]).max() < 2e-3

    # same inputs through the single-seq wrapper (contiguous loc, pages 0..P-1).
    o_s, _ = prefill_write_and_attend(
        c["ql"][:L],
        c["qpe"][:L],
        c["kvc"][:L],
        c["kpe"][:L],
        jnp.zeros((c["pages_per_seq"], _PS, 1, _DK_PAD), jnp.float32),
        c["topk_pages"][:L],
        c["positions"][:L],
        jnp.arange(L, dtype=jnp.int32),
        kv_lora_rank=_KV_LORA,
        page_size=_PS,
        sm_scale=_SCALE,
        interpret=True,
    )
    d = np.abs(o_r[:L] - np.asarray(o_s)).max()
    print(f"[G0 single-seq] ragged-vs-oracle & ragged-vs-single|err|={d:.3e}")
    assert d < 2e-3, f"G0: ragged(num_seqs==1) != single-seq path: {d}"


def test_gate_G1_batch_of_identical():
    """G1: N *identical* requests → each request's output block is bit-for-bit the
    same (same inputs, only the physical page numbering differs per slot, so this
    also checks the per-request base indexing)."""
    L, K = 384, 4
    rng = np.random.default_rng(7)
    a = _seq_inputs(rng, L, K)
    c = _assemble([a, a, a], seed_pages=7, page_perm=True)
    o, _ = _run_case(c)
    b0, b1, b2 = (o[_seq_slice(c, i)] for i in range(3))
    d = max(np.abs(b1 - b0).max(), np.abs(b2 - b0).max())
    print(f"[G1] batch-of-identical inter-block max|err|={d:.3e}")
    assert d < 1e-5, f"G1: identical requests produced different outputs: {d}"


def test_gate_G2_cross_seq_no_bleed():
    """G2: request A's output is identical whether run alone or batched beside an
    unrelated request B (strongest catch for a wrong per-request page base /
    causal frame — one sequence reading another's KV)."""
    K = 4
    a = _seq_inputs(np.random.default_rng(11), 300, K)  # fixed inputs for A
    b = _seq_inputs(np.random.default_rng(22), 220, K)  # unrelated B
    o_pair, _ = _run_case(_assemble([a, b], seed_pages=1, page_perm=True))
    c_alone = _assemble([a], seed_pages=99, page_perm=False)
    o_alone, _ = _run_case(c_alone)
    LA = a["L"]
    d = np.abs(o_pair[:LA] - o_alone[:LA]).max()
    print(f"[G2] no-bleed A alone-vs-paired max|err|={d:.3e}")
    assert d < 2e-3, f"G2: batching an unrelated seq changed A: {d}"


def test_gate_G3_permutation_invariance():
    """G3: permuting request order permutes the output blocks correspondingly —
    A's output is the same in [A,B] and [B,A]."""
    K = 4
    a = _seq_inputs(np.random.default_rng(31), 256, K)
    b = _seq_inputs(np.random.default_rng(41), 384, K)
    o_ab, _ = _run_case(_assemble([a, b], seed_pages=3, page_perm=True))
    c_ba = _assemble([b, a], seed_pages=5, page_perm=True)
    o_ba, _ = _run_case(c_ba)
    LA, LB = a["L"], b["L"]
    S_pad = c_ba["S_pad"]
    a_in_ab = o_ab[:LA]
    a_in_ba = o_ba[S_pad : S_pad + LA]  # A is the 2nd request in [B, A]
    b_in_ab = o_ab[S_pad : S_pad + LB]
    b_in_ba = o_ba[:LB]
    da = np.abs(a_in_ab - a_in_ba).max()
    db = np.abs(b_in_ab - b_in_ba).max()
    print(f"[G3] permutation invariance A|err|={da:.3e} B|err|={db:.3e}")
    assert max(da, db) < 2e-3, f"G3: output depends on packing position: A={da} B={db}"


def test_gate_G4_dense_equals_sparse_superset():
    """G4: when K >= pages_per_seq (every reachable page selected), the sparse
    output equals full causal MLA over each request's written latent."""
    seq_lens = [384, 200, 512]
    pages_per_seq = int(np.ceil(max(seq_lens) / _PS))
    o, _, c = _run_ragged(seq_lens, K=pages_per_seq, seed=5)
    # full (dense) reference: attend ALL causal tokens (not just selected pages).
    latent = np.concatenate([c["_kvc"], c["_kpe"]], axis=-1)
    q_full = np.concatenate([c["_ql"], c["_qpe"]], axis=-1)
    S_pad = c["S_pad"]
    dense = np.zeros_like(o)
    for i, L in enumerate(seq_lens):
        base = i * S_pad
        lat = latent[base : base + L]
        for j in range(L):
            t = base + j
            k_sel = lat[: j + 1]  # all causal tokens
            logits = (q_full[t] @ k_sel.T) * _SCALE
            p = np.exp(logits - logits.max(-1, keepdims=True))
            p = p / p.sum(-1, keepdims=True)
            dense[t] = p @ k_sel[:, :_KV_LORA]
    real = c["real"]
    d = np.abs(o[real] - dense[real]).max()
    print(f"[G4] sparse(all pages) vs dense causal max|err|={d:.3e}")
    assert d < 2e-3, f"G4: full-selection sparse != dense: {d}"


def test_gate_A1_prefix_equivalence():
    """A1 (radix/prefix caching): a prompt prefilled single-shot must produce the
    same suffix outputs as prefilling it as [cached prefix] + [extend suffix].

    Prefix path: pre-write the prefix latent into the cache, then run the ragged
    wrapper over ONLY the suffix query tokens with *absolute* positions and the full
    ``seq_lens`` causal bound. The extend tokens must match the single-shot run
    exactly (the kernel already threads prefix pages via page_indices + seq_lens)."""
    Lfull, K = 512, 4
    Lp = 256  # cached prefix length
    Le = Lfull - Lp
    pps = int(np.ceil(Lfull / _PS))

    rng = np.random.default_rng(23)
    seq = _seq_inputs(rng, Lfull, K)  # ql/qpe/kvc/kpe/tp for the full prompt
    # single-shot reference (identity pages ⇒ physical slot == token id)
    c_full = _assemble([seq], seed_pages=0, page_perm=False)
    o_full, _ = _run_case(c_full)

    # prefix cache: write tokens 0..Lp-1 at their physical slots (== token id)
    latent = np.concatenate([seq["kvc"], seq["kpe"]], axis=-1)  # [Lfull, 576]
    cache_prefix = np.zeros((pps * _PS, _DK_PAD), np.float32)
    cache_prefix[:Lp, : _KV_LORA + _ROPE] = latent[:Lp]
    cache_prefix = jnp.asarray(cache_prefix.reshape(pps, _PS, 1, _DK_PAD))

    o_suf, _ = prefill_write_and_attend_ragged(
        jnp.asarray(seq["ql"][Lp:]),
        jnp.asarray(seq["qpe"][Lp:]),
        jnp.asarray(seq["kvc"][Lp:]),
        jnp.asarray(seq["kpe"][Lp:]),
        cache_prefix,
        jnp.asarray(seq["tp"][Lp:]),  # seq-local page ids over the FULL kv
        jnp.arange(Lp, Lfull, dtype=jnp.int32),  # ABSOLUTE positions
        jnp.arange(Lp, Lfull, dtype=jnp.int32),  # physical slots for the suffix
        jnp.asarray([Lfull], jnp.int32),  # full causal bound (prefix+extend)
        jnp.asarray([0, Le], jnp.int32),  # cu_q_lens over the extend tokens
        jnp.asarray([0, pps * _PS], jnp.int32),
        jnp.arange(pps, dtype=jnp.int32),
        kv_lora_rank=_KV_LORA,
        page_size=_PS,
        sm_scale=_SCALE,
        interpret=True,
    )
    o_suf = np.asarray(o_suf)
    d = np.abs(o_suf - o_full[Lp:Lfull]).max()
    print(f"[A1 prefix-equiv] suffix single-shot-vs-cached max|err|={d:.3e} (Lp={Lp} Le={Le})")
    assert d < 2e-3, f"A1: cached-prefix+extend != single-shot on the suffix: {d}"


def test_gate_A2_chunked_equivalence():
    """A2 (chunked prefill): prefilling a prompt in N sequential chunks must give
    the same per-token outputs as a single-shot prefill. Each chunk is an extend
    with the growing prefix as its cache — the same kernel machinery A1 validates,
    applied repeatedly against a persistent cache (the IndexShare carry is intra-
    pass, so no cross-chunk state is needed; full layers rescore the full kv)."""
    Lfull, K = 512, 4
    chunks = [(0, 160), (160, 320), (320, 512)]  # ragged chunk sizes
    pps = int(np.ceil(Lfull / _PS))

    rng = np.random.default_rng(29)
    seq = _seq_inputs(rng, Lfull, K)
    o_full, _ = _run_case(_assemble([seq], seed_pages=0, page_perm=False))

    latent = np.concatenate([seq["kvc"], seq["kpe"]], axis=-1)
    cache = jnp.zeros((pps, _PS, 1, _DK_PAD), jnp.float32)
    worst = 0.0
    for a, b in chunks:
        Lc = b - a
        o_c, cache = prefill_write_and_attend_ragged(
            jnp.asarray(seq["ql"][a:b]),
            jnp.asarray(seq["qpe"][a:b]),
            jnp.asarray(seq["kvc"][a:b]),
            jnp.asarray(seq["kpe"][a:b]),
            cache,
            jnp.asarray(seq["tp"][a:b]),
            jnp.arange(a, b, dtype=jnp.int32),  # absolute positions
            jnp.arange(a, b, dtype=jnp.int32),  # physical slots (== token id)
            jnp.asarray([b], jnp.int32),  # causal bound = kv seen so far
            jnp.asarray([0, Lc], jnp.int32),
            jnp.asarray([0, pps * _PS], jnp.int32),
            jnp.arange(pps, dtype=jnp.int32),
            kv_lora_rank=_KV_LORA,
            page_size=_PS,
            sm_scale=_SCALE,
            interpret=True,
        )
        d = np.abs(np.asarray(o_c) - o_full[a:b]).max()
        worst = max(worst, float(d))
    print(f"[A2 chunked-equiv] worst chunk-vs-single-shot max|err|={worst:.3e} chunks={chunks}")
    assert worst < 2e-3, f"A2: N-chunk prefill != single-shot: {worst}"


def test_parity_flat():
    _run("flat")


def test_parity_paged():
    _run("paged")


def test_prefill_write_and_attend():
    _run_write_attend()


def test_prefill_write_and_attend_padded_loc_canary():
    _run_write_attend_canary()


if __name__ == "__main__":
    for seed in range(3):
        _run("flat", seed)
        _run("paged", seed)
    for seed in range(3):
        _run_write_attend(seed)
    _run_write_attend_canary()
    # packed-ragged (A3) parity + invariant gates
    test_ragged_parity_varlen()
    test_ragged_self_write_and_canary()
    test_gate_G0_single_seq_equivalence()
    test_gate_G1_batch_of_identical()
    test_gate_G2_cross_seq_no_bleed()
    test_gate_G3_permutation_invariance()
    test_gate_G4_dense_equals_sparse_superset()
    test_gate_A1_prefix_equivalence()
    test_gate_A2_chunked_equivalence()
    print("PARITY OK")
