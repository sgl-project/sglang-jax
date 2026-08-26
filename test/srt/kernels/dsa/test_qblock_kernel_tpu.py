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
