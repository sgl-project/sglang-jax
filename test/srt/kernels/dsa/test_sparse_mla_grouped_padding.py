"""Grouped (``group_queries=True``) page-level sparse MLA must hand the mixed
kernel ``kv_len >= q_len`` for every pseudo-sequence, including the batch-size
padding groups (``kv_lens == 0``).

The mixed kernel sizes its KV loop by ``cdiv(kv_len, bkv_sz)``; ``kv_len == 0``
with ``G`` query rows skips that loop entirely, so the query block DMA is started
but never waited (a trailing padded group leaves the DMA semaphore non-zero at
kernel exit) and the next sequence's prefetch is never issued (an inner padded
group hangs the core). The padding groups must therefore look like ``G``
all-new tokens on the sentinel page, mirroring the per-token path where padded
tokens get ``kv_len == 1``.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sgl_jax.srt.kernels.dsa.sparse_mla as sparse_mla_mod

PAGE = 8
PPS = 8  # pages per seq
K = 3  # topk positions per token
KPM = 4  # k_pages_max
G = 4
H, LKV, R = 4, 128, 64
TOTAL_PAGES = 1 + 4 * PPS + 1  # 0 unused, 4 seqs, sentinel last


def _meta(base_lens):
    """Verify-batch metadata (mla_backend.get_forward_metadata for TARGET_VERIFY):
    valid requests get G query rows, trailing bs-padding requests get 0."""
    S = len(base_lens)
    valid = np.asarray(base_lens) > 0
    ext = np.where(valid, G, 0).astype(np.int32)
    seq_lens = np.where(valid, np.asarray(base_lens) + G, 0).astype(np.int32)
    cu_q = np.concatenate([[0], np.cumsum(ext)]).astype(np.int32)
    cu_kv = (np.arange(S + 1) * PPS * PAGE).astype(np.int32)
    page_indices = (1 + np.arange(S * PPS)).astype(np.int32)
    dist = np.array([0, 0, int(valid.sum())], np.int32)
    return seq_lens, cu_q, cu_kv, page_indices, dist


def _run_grouped(base_lens, group_queries=True):
    S = len(base_lens)
    T = S * G
    seq_lens, cu_q, cu_kv, page_indices, dist = _meta(base_lens)
    key = jax.random.PRNGKey(0)
    ql = jax.random.normal(key, (T, H, LKV), jnp.bfloat16)
    qpe = jnp.zeros((T, H, R), jnp.bfloat16)
    nkv = jnp.zeros((T, LKV), jnp.bfloat16)
    nkpe = jnp.zeros((T, R), jnp.bfloat16)
    cache = jnp.zeros((TOTAL_PAGES, PAGE // 2, 2, LKV + 128), jnp.bfloat16)
    # topk: valid tokens pick the first K positions of their request, padded -1
    topk = np.full((T, K), -1, np.int32)
    for s in range(S):
        if seq_lens[s] > 0:
            topk[s * G : (s + 1) * G] = np.arange(K)[None, :]
    captured = {}

    def fake_kernel(ql_, qpe_, nkv_, nkpe_, cache_, kv_lens, pi, cu_q_, cu_kv_, dist_, **kw):
        captured.update(
            kv_lens=np.asarray(kv_lens),
            page_indices=np.asarray(pi).reshape(-1, KPM),
            cu_q=np.asarray(cu_q_),
            dist=np.asarray(dist_),
            kw=kw,
        )
        return ql_, cache_

    with (
        jax.disable_jit(),
        mock.patch.object(sparse_mla_mod, "mla_ragged_paged_attention", fake_kernel),
    ):
        sparse_mla_mod.sparse_mla_page_level(
            ql,
            qpe,
            nkv,
            nkpe,
            cache,
            jnp.asarray(seq_lens),
            jnp.asarray(topk),
            jnp.asarray(page_indices),
            jnp.asarray(cu_q),
            jnp.asarray(cu_kv),
            jnp.asarray(dist),
            None,
            sm_scale=1.0,
            page_size=PAGE,
            pages_per_seq=PPS,
            kv_lora_rank=LKV,
            k_pages_max=KPM,
            page_share_group=G if group_queries else 1,
            group_queries=group_queries,
        )
    return captured


@pytest.mark.parametrize("base_lens", [[20, 37, 0, 0], [20, 37, 45, 0], [20, 0, 0, 0]])
def test_padded_groups_get_kv_len_g_on_sentinel_page(base_lens):
    cap = _run_grouped(base_lens)
    S = len(base_lens)
    valid = np.asarray(base_lens) > 0
    sentinel = TOTAL_PAGES - 1
    assert cap["cu_q"].tolist() == (np.arange(S + 1) * G).tolist()
    assert cap["dist"].tolist() == [0, 0, S]
    # Every pseudo-sequence has at least G positions (the G new tokens).
    assert (cap["kv_lens"] >= G).all(), cap["kv_lens"]
    # Padded groups: exactly G all-new tokens, all pages on the sentinel.
    assert (cap["kv_lens"][~valid] == G).all(), cap["kv_lens"]
    assert (cap["page_indices"][~valid] == sentinel).all()
    # Valid groups keep real pages and more than the G new positions.
    assert (cap["kv_lens"][valid] > G).all()
    assert (cap["page_indices"][valid][:, 0] != sentinel).all()


def test_valid_groups_unchanged_by_padding():
    dense = _run_grouped([20, 37])
    padded = _run_grouped([20, 37, 0, 0])
    np.testing.assert_array_equal(dense["kv_lens"], padded["kv_lens"][:2])
    np.testing.assert_array_equal(dense["page_indices"], padded["page_indices"][:2])


def test_grouped_requests_static_mixed_q_len_g():
    # G query rows per pseudo-sequence: the mixed kernel must size its query
    # block by G instead of the tuned-table block (bq 256 for a 4-row block
    # multiplies the flash-attention work by 64x and hides the KV-read win).
    assert _run_grouped([20, 37, 0, 0])["kw"]["mixed_static_q_len"] == G
    assert _run_grouped([20, 37, 0, 0], group_queries=False)["kw"].get("mixed_static_q_len") is None
