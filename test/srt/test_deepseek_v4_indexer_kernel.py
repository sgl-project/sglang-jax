"""CSA indexer through ``kernels/dsa/streamindex_topk`` (completed-groups mask + exact
SparseCore selection) versus the native-JAX ``csa_indexer_topk_ref`` reference.

The kernel reads the paged indexer cache directly; the reference reads the gathered
``[E, D]`` key array. Both must select the same compressed entries for the same
gathered-row ordering that ``dispatch.read_tables`` produces.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.layers.attention.dsv4.decode import select_decode_entries
from sgl_jax.srt.layers.attention.dsv4.indexer import (
    INVALID_ENTRY,
    csa_indexer_topk_kernel,
    kernel_read_layout,
    resolve_indexer_backend,
)
from sgl_jax.srt.layers.attention.dsv4.ref.indexer import csa_indexer_topk_ref

RATIO, PAGE_SIZE, CPS = 4, 128, 32  # CPS = compressed entries per page
H, D, K = 64, 128, 512


def _synthetic_batch(requests, *, seed=0, padded_queries=0):
    """requests: list of (prefix_len, q_len). Each request owns contiguous compressed
    pages starting after page 0 (the padding page), like C1's allocator."""
    rng = np.random.default_rng(seed)
    seq_lens = np.array([p + q for p, q in requests], np.int32)
    q_lens = np.array([q for _, q in requests], np.int32)
    cu_q = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    T = int(cu_q[-1]) + padded_queries
    # page ownership
    page_cursor, rows, req_ids, entry_ids, first_page = 1, [], [], [], []
    for r, s in enumerate(seq_lens):
        complete = int(s) // RATIO
        n_pages = -(-complete // CPS) if complete else 0
        first_page.append(page_cursor)
        entries = np.arange(complete)
        rows.append(page_cursor * CPS + entries)
        req_ids.append(np.full(complete, r))
        entry_ids.append(entries)
        page_cursor += n_pages
    total_pages = page_cursor + 1
    tables = dict(
        compressed_rows=np.concatenate(rows).astype(np.int32),
        compressed_request_ids=np.concatenate(req_ids).astype(np.int32),
        compressed_entry_ids=np.concatenate(entry_ids).astype(np.int32),
    )
    query_request_ids = np.concatenate(
        [np.full(q, r) for r, (_, q) in enumerate(requests)] + [np.full(padded_queries, -1)]
    ).astype(np.int32)
    query_positions = np.concatenate(
        [np.arange(p, p + q) for p, q in requests] + [np.zeros(padded_queries)]
    ).astype(np.int32)
    valid = np.concatenate([np.ones(int(cu_q[-1]), bool), np.zeros(padded_queries, bool)])
    q = rng.standard_normal((T, H, D), np.float32)
    w = rng.standard_normal((T, H), np.float32)
    buffer = rng.standard_normal((total_pages * CPS, D), np.float32)
    return dict(
        q=jnp.asarray(q, jnp.bfloat16),
        weights=jnp.asarray(w),
        indexer_buffer=jnp.asarray(buffer, jnp.bfloat16),
        seq_lens=jnp.asarray(seq_lens),
        q_lens=jnp.asarray(q_lens),
        cu_q_lens=jnp.asarray(cu_q),
        query_request_ids=jnp.asarray(query_request_ids),
        query_positions=jnp.asarray(query_positions),
        valid_token_mask=jnp.asarray(valid),
        tables={k: jnp.asarray(v) for k, v in tables.items()},
        first_page=np.array(first_page),
    )


def test_resolve_indexer_backend(monkeypatch):
    monkeypatch.delenv("DSV4_INDEXER_BACKEND", raising=False)
    expected = "kernel" if jax.default_backend() == "tpu" else "reference"
    assert resolve_indexer_backend("auto") == expected
    assert resolve_indexer_backend("reference") == "reference"
    monkeypatch.setenv("DSV4_INDEXER_BACKEND", "reference")
    assert resolve_indexer_backend("auto") == "reference"
    with pytest.raises(ValueError):
        resolve_indexer_backend("pallas")


def test_kernel_read_layout_matches_tables():
    b = _synthetic_batch([(300, 100), (0, 257), (1000, 1), (0, 0)])
    pages, offsets = kernel_read_layout(
        b["tables"]["compressed_rows"],
        b["seq_lens"],
        b["q_lens"],
        ratio=RATIO,
        compressed_page_size=CPS,
    )
    pages, offsets = np.asarray(pages), np.asarray(offsets)
    counts = np.where(np.asarray(b["q_lens"]) > 0, np.asarray(b["seq_lens"]) // RATIO, 0)
    np.testing.assert_array_equal(offsets, np.cumsum(counts) - counts)
    assert pages.shape[0] == 4 and pages.shape[1] * CPS >= counts.max()
    for r, c in enumerate(counts):
        n_pages = -(-c // CPS)
        np.testing.assert_array_equal(pages[r, :n_pages], b["first_page"][r] + np.arange(n_pages))


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="Pallas kernel needs TPU")
@pytest.mark.parametrize("padded_queries", [0, 7])
def test_kernel_selection_matches_reference(padded_queries):
    b = _synthetic_batch([(300, 100), (0, 257), (1000, 1), (0, 0)], padded_queries=padded_queries)
    keys = jnp.take(b["indexer_buffer"], b["tables"]["compressed_rows"], axis=0)
    reference = np.asarray(
        csa_indexer_topk_ref(
            b["q"],
            b["weights"],
            keys,
            b["query_positions"],
            b["query_request_ids"],
            b["tables"]["compressed_request_ids"],
            b["valid_token_mask"],
            entry_group_ids=b["tables"]["compressed_entry_ids"],
            k=K,
            ratio=RATIO,
        )
    )
    got = np.asarray(
        csa_indexer_topk_kernel(
            b["q"],
            b["weights"],
            b["indexer_buffer"],
            compressed_rows=b["tables"]["compressed_rows"],
            seq_lens=b["seq_lens"],
            q_lens=b["q_lens"],
            cu_q_lens=b["cu_q_lens"],
            query_request_ids=b["query_request_ids"],
            valid_token_mask=b["valid_token_mask"],
            k=K,
            ratio=RATIO,
            compressed_page_size=CPS,
        )
    )
    assert got.shape == reference.shape == (b["q"].shape[0], K)
    # Contract: -1 packed at the tail, padded queries select nothing.
    assert np.all(np.diff((got < 0).astype(int), axis=1) >= 0)
    assert np.all(got[~np.asarray(b["valid_token_mask"])] == INVALID_ENTRY)
    # Legality: same request and completed group, judged by the gathered-row tables.
    creq = np.asarray(b["tables"]["compressed_request_ids"])
    cent = np.asarray(b["tables"]["compressed_entry_ids"])
    qreq, qpos = np.asarray(b["query_request_ids"]), np.asarray(b["query_positions"])
    for t in np.flatnonzero(np.asarray(b["valid_token_mask"])):
        sel = got[t][got[t] >= 0]
        assert np.all(creq[sel] == qreq[t]), f"query {t} selected another request's entry"
        assert np.all(cent[sel] < (qpos[t] + 1) // RATIO), f"query {t} selected an incomplete group"
        assert len(set(sel.tolist())) == len(sel)
        ref = set(reference[t][reference[t] >= 0].tolist())
        # Same count of valid selections; near-tie flips from bf16 accumulation order are
        # tolerated but must stay rare.
        assert len(sel) == len(ref)
        if ref:
            assert len(ref & set(sel.tolist())) / len(ref) >= 0.98


def test_select_decode_entries_marks_invalid_as_minus_one():
    # The Pallas decode scorer marks entries at or beyond each request's length with
    # finfo.min; the selector must never return them and must report -1 for them.
    neg = jnp.finfo(jnp.float32).min
    lengths = jnp.asarray([3000, 10, 0], jnp.int32)
    scores = jnp.asarray(np.random.default_rng(1).standard_normal((3, 4096), np.float32))
    scores = jnp.where(jnp.arange(4096)[None, :] < lengths[:, None], scores, neg)
    selected, valid = select_decode_entries(scores, lengths, take=64)
    selected, valid = np.asarray(selected), np.asarray(valid)
    assert valid.sum(1).tolist() == [64, 10, 0]
    assert np.all(selected[valid] < np.asarray(lengths)[:, None].repeat(64, 1)[valid])
    exact = np.asarray(jax.lax.top_k(scores, 64)[1])
    assert set(selected[0].tolist()) == set(exact[0].tolist())
