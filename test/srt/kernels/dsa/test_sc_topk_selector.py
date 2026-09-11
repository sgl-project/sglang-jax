"""SparseCore top-k selector for the DSA indexer (``streamindex_topk`` exit stage).

Parity against exact ``jax.lax.top_k`` on the score matrix the indexer kernel
produces, including the ``-inf`` columns the kernel leaves for masked / never-written
entries and the ``E < k`` case that must yield trailing ``-1``.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sgl_jax.srt.kernels.dsa.streamindex_topk as streamindex_topk_mod
from sgl_jax.srt.kernels.dsa.streamindex_topk import (
    SC_TOPK_MIN_ENTRIES,
    _sc_padded_width,
    sc_topk_available,
    select_topk_indices,
    should_use_sc_topk,
    streamindex_topk,
)

if jax.default_backend() != "tpu":
    pytest.skip("SparseCore top-k selector requires TPU", allow_module_level=True)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

K = 512


def _exact_indices(scores, k):
    vals, idxs = jax.lax.top_k(scores, k)
    return np.where(np.asarray(vals) == -np.inf, -1, np.asarray(idxs))


def _set_rows_equal(a, b):
    return all(set(x[x >= 0].tolist()) == set(y[y >= 0].tolist()) for x, y in zip(a, b))


def _trailing_minus_one(a):
    flags = (a < 0).astype(np.int32)
    return bool(np.all(np.diff(flags, axis=1) >= 0))


class _FakeSparseCore:
    """v6e geometry (pltpu.get_tpu_info().sparse_core on TPU v6e)."""

    num_cores = 2
    num_subcores = 16
    num_lanes = 8
    vmem_capacity_bytes = 262144


def test_should_use_sc_topk_thresholds(monkeypatch):
    """Pure policy: small rows fall back to XLA, large rows go to SparseCore, rows whose
    per-subcore key+value slice exceeds half of VMEM fall back."""
    monkeypatch.setattr(streamindex_topk_mod, "_sparse_core_info", lambda: _FakeSparseCore())
    assert not should_use_sc_topk(num_entries=256, batch=8)
    assert not should_use_sc_topk(num_entries=SC_TOPK_MIN_ENTRIES - 1, batch=64)
    assert should_use_sc_topk(num_entries=SC_TOPK_MIN_ENTRIES, batch=1)
    # 262144 entries / 16 subcores * 8 B = 128 KiB = exactly half of v6e VMEM.
    assert should_use_sc_topk(num_entries=262144, batch=2048)
    assert not should_use_sc_topk(num_entries=262144 + 16 * 8, batch=1)
    assert not should_use_sc_topk(num_entries=1 << 26, batch=1)


def test_should_use_sc_topk_false_without_sparse_core(monkeypatch):
    monkeypatch.setattr(streamindex_topk_mod, "_sparse_core_info", lambda: None)
    assert not should_use_sc_topk(num_entries=262144, batch=64)
    assert not sc_topk_available()
    with pytest.raises(ValueError, match="requires a TPU with a SparseCore"):
        select_topk_indices(jnp.zeros((1, 8192), jnp.float32), K, backend="sc")


def test_sc_padded_width_matches_kernel_constraints():
    info = _FakeSparseCore()
    # Below the cooperative threshold: one tile, whole vectors of num_lanes.
    assert _sc_padded_width(4096, info) == 4096
    assert _sc_padded_width(4090, info) == 4096
    # Above it: num_subcores tiles, each a whole number of vectors.
    assert _sc_padded_width(4097, info) == 4224  # next multiple of 16 * 8
    assert _sc_padded_width(25000, info) == 25088
    assert _sc_padded_width(262144, info) == 262144


def test_auto_backend_routes_by_row_width(monkeypatch):
    """auto -> XLA for short rows, SparseCore for long rows (both selectors observed)."""
    calls = []
    orig_sc, orig_xla = streamindex_topk_mod._sc_select, streamindex_topk_mod._xla_select
    monkeypatch.setattr(
        streamindex_topk_mod, "_sc_select", lambda s, k: (calls.append("sc"), orig_sc(s, k))[1]
    )
    monkeypatch.setattr(
        streamindex_topk_mod, "_xla_select", lambda s, k: (calls.append("xla"), orig_xla(s, k))[1]
    )
    select_topk_indices(jnp.zeros((1, 256), jnp.float32), K, backend="auto")
    assert calls == ["xla"]
    if sc_topk_available():
        select_topk_indices(jnp.zeros((1, 262144), jnp.float32), K, backend="auto")
        assert calls == ["xla", "sc"]


@pytest.mark.skipif(not sc_topk_available(), reason="no SparseCore top-k on this chip")
@pytest.mark.parametrize("batch,num_entries", [(64, 262144), (3, 25000), (1, 8192)])
def test_sc_selector_matches_exact_topk(batch, num_entries):
    """Random f32 scores with a block of -inf columns: SC result == exact top-k as a set,
    values descending, -1 only at the tail. num_entries=25000 is not a multiple of 256, so
    it exercises the padding path."""
    rng = np.random.default_rng(0)
    s = rng.standard_normal((batch, num_entries), np.float32)
    s[:, num_entries - 1000 :] = -np.inf  # tail the kernel never wrote / causal-masked
    scores = jnp.asarray(s)
    got = np.asarray(select_topk_indices(scores, K, backend="sc"))
    exact = _exact_indices(scores, K)
    assert got.shape == (batch, K)
    assert _set_rows_equal(got, exact)
    assert _trailing_minus_one(got)
    assert not np.any(got >= num_entries - 1000), "a -inf column was selected"


@pytest.mark.skipif(not sc_topk_available(), reason="no SparseCore top-k on this chip")
def test_sc_selector_fewer_valid_than_k_pads_with_minus_one():
    scores = (
        jnp.full((4, 8192), -jnp.inf, jnp.float32)
        .at[:, :300]
        .set(jnp.asarray(np.random.default_rng(1).standard_normal((4, 300), np.float32)))
    )
    got = np.asarray(select_topk_indices(scores, K, backend="sc"))
    assert np.all((got >= 0).sum(1) == 300)
    assert _trailing_minus_one(got)
    assert _set_rows_equal(got, _exact_indices(scores, K))


@pytest.mark.skipif(not sc_topk_available(), reason="no SparseCore top-k on this chip")
def test_sc_selector_edge_rows():
    """E == k (no -1), all -inf (all -1) and 600 tied maxima (exact set, no early -1)."""
    exact_k = jnp.asarray(np.random.default_rng(5).standard_normal((4, K), np.float32))
    got = np.asarray(select_topk_indices(exact_k, K, backend="sc"))
    assert np.all(got >= 0) and _set_rows_equal(got, _exact_indices(exact_k, K))

    all_inf = jnp.full((2, 8192), -jnp.inf, jnp.float32)
    assert np.all(np.asarray(select_topk_indices(all_inf, K, backend="sc")) == -1)

    tied = jnp.zeros((1, 8192), jnp.float32).at[0, :600].set(1.0)
    got = np.asarray(select_topk_indices(tied, K, backend="sc"))
    assert np.all(got >= 0) and np.all(got[0] < 600) and len(set(got[0].tolist())) == K


def test_xla_selector_unchanged():
    """backend='xla' is the pre-existing approx_max_k(recall_target=1.0) path."""
    scores = jnp.asarray(np.random.default_rng(2).standard_normal((8, 4096), np.float32))
    got = np.asarray(select_topk_indices(scores, K, backend="xla"))
    assert _set_rows_equal(got, _exact_indices(scores, K)) and _trailing_minus_one(got)


def _indexer_case(B, T_per_seq, ctx, *, ratio=4, page=128, H=8, D=128, seed=3):
    """Decode/prefill batch over a bf16 indexer cache; ctx in tokens, entries = ctx // ratio."""
    dev = jax.devices()[0]
    rng = np.random.default_rng(seed)
    T = B * T_per_seq
    entries = ctx // ratio
    pps = -(-entries // page)
    total_pages = B * pps + 1
    q = jax.device_put(jnp.asarray(rng.standard_normal((T, H, D), np.float32), jnp.bfloat16), dev)
    w = jax.device_put(jnp.asarray(rng.standard_normal((T, H), np.float32)), dev)
    cache = jax.device_put(
        jax.random.normal(jax.random.key(seed), (total_pages, page // 2, 2, D), jnp.bfloat16), dev
    )
    nd = B if T_per_seq == 1 else 0
    return dict(
        q=q,
        indexer_weights=w,
        cache_kv=cache,
        seq_lens=jnp.full((B,), ctx, jnp.int32),
        page_indices=jnp.asarray(np.arange(1, B * pps + 1, dtype=np.int32)),
        cu_q_lens=jnp.asarray(np.arange(0, T + 1, T_per_seq, dtype=np.int32)),
        distribution=jnp.asarray((nd, nd, B), jnp.int32),
    )


@pytest.mark.skipif(not sc_topk_available(), reason="no SparseCore top-k on this chip")
@pytest.mark.parametrize(
    "B,T_per_seq,ctx",
    [(8, 1, 1 << 20), (1, 256, 1 << 18), (8, 1, 100_000), (8, 1, 1024)],
)
def test_streamindex_topk_sc_matches_xla_end_to_end(B, T_per_seq, ctx):
    """The kernel's own scores through both selector backends give identical index sets;
    ctx=1024 (256 entries < k) also checks the trailing -1 contract."""
    args = _indexer_case(B, T_per_seq, ctx)
    outs = {}
    for backend in ("xla", "sc", "auto"):
        outs[backend] = np.asarray(
            streamindex_topk(
                **args,
                k=K,
                compression_ratio=4,
                num_kv_pages_per_block=64,
                num_queries_per_block=(1, 64, 64),
                topk_backend=backend,
            )
        )
    assert _set_rows_equal(outs["xla"], outs["sc"])
    assert _set_rows_equal(outs["xla"], outs["auto"])
    assert _trailing_minus_one(outs["sc"])
    if ctx == 1024:
        assert np.all((outs["sc"] >= 0).sum(1) == 256)
