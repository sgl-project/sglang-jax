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
    print("PARITY OK")
