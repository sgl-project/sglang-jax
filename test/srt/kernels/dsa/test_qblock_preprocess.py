"""CPU parity for the query-block preprocessing of the blocked sparse-MLA
prefill kernel (P1: attend query-batching).

The preprocessing turns per-query page selections ``topk_units [T, K]`` into
per-query-block tables:

* ``blk_units  [nQB, U_max]``  — sorted union of the block's selected units, -1 pad
* ``blk_member [nQB, QB, U_max]`` int8 — query-to-union-slot membership bitmap
* ``blk_counts [nQB]``        — TRUE unique count (uncapped; > U_max ⇒ overflow,
  the caller must gate on it before trusting the tables)

Parity contract (vs the per-query kernel semantics): for every query ``t`` whose
block did not overflow, ``{ blk_units[b, u] : blk_member[b, t - b*QB, u] == 1 }``
must equal ``{ x in topk_units[t] : x >= 0 }`` — i.e. the -inf membership bias in
the blocked kernel reproduces the per-query selection exactly.

Pure jnp — runs on CPU; TPU shape/e2e checks live with the kernel tests.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.dsa.sparse_mla_prefill_qblock import build_block_unit_tables


def _oracle(topk: np.ndarray, qb: int, u_max: int):
    """Reference in plain python sets."""
    T, K = topk.shape
    n_blk = -(-T // qb)
    pad = n_blk * qb - T
    tk = np.pad(topk, ((0, pad), (0, 0)), constant_values=-1).reshape(n_blk, qb, K)
    units = np.full((n_blk, u_max), -1, np.int32)
    member = np.zeros((n_blk, qb, u_max), np.int8)
    counts = np.zeros((n_blk,), np.int32)
    for b in range(n_blk):
        uniq = sorted({int(x) for x in tk[b].ravel() if x >= 0})
        counts[b] = len(uniq)
        keep = uniq[:u_max]
        units[b, : len(keep)] = keep
        col = {u: j for j, u in enumerate(keep)}
        for q in range(qb):
            for x in tk[b, q]:
                if int(x) in col:
                    member[b, q, col[int(x)]] = 1
    return units, member, counts


def _check(topk: np.ndarray, qb: int, u_max: int):
    units, member, counts = jax.jit(
        build_block_unit_tables, static_argnames=("query_block", "u_max")
    )(jnp.asarray(topk, jnp.int32), query_block=qb, u_max=u_max)
    ounits, omember, ocounts = _oracle(topk, qb, u_max)
    np.testing.assert_array_equal(np.asarray(counts), ocounts)
    np.testing.assert_array_equal(np.asarray(units), ounits)
    np.testing.assert_array_equal(np.asarray(member), omember)
    # per-query set reconstruction (the actual kernel-facing contract)
    T = topk.shape[0]
    u_np, m_np = np.asarray(units), np.asarray(member)
    for t in range(T):
        b, q = divmod(t, qb)
        if ocounts[b] > u_max:
            continue  # overflowed block: caller must gate, no contract
        got = {int(u_np[b, j]) for j in np.nonzero(m_np[b, q])[0]}
        want = {int(x) for x in topk[t] if x >= 0}
        assert got == want, f"query {t}: {got} != {want}"


def _local_topk(rng, T, K, num_units, spread):
    """Locality-flavoured selection: pages near the query's own page + sinks."""
    base = (np.arange(T) * num_units // max(T, 1))[:, None]
    jitter = rng.integers(-spread, spread + 1, size=(T, K))
    tk = np.clip(base + jitter, 0, num_units - 1)
    tk[:, 0] = 0  # sink page for everyone (heavy overlap)
    return tk.astype(np.int32)


def test_parity_110k_shape():
    # 110k deployment shape: 8192-query chunk, K=32 pages, 880-page pool.
    rng = np.random.default_rng(0)
    tk = _local_topk(rng, T=8192, K=32, num_units=880, spread=40)
    _check(tk, qb=64, u_max=880)


def test_parity_uniform_random():
    # no locality at all — worst-case unions
    rng = np.random.default_rng(1)
    tk = rng.integers(0, 96, size=(256, 8)).astype(np.int32)
    _check(tk, qb=32, u_max=96)


@pytest.mark.parametrize("T", [1, 63, 64, 65, 130])
def test_block_boundaries(T):
    rng = np.random.default_rng(2)
    tk = rng.integers(0, 40, size=(T, 4)).astype(np.int32)
    _check(tk, qb=64, u_max=40)


def test_padded_and_empty_rows():
    rng = np.random.default_rng(3)
    tk = rng.integers(0, 40, size=(96, 4)).astype(np.int32)
    tk[10] = -1  # fully padded query (e.g. beyond seq end)
    tk[20, 2:] = -1  # partially padded topk row
    tk[64:] = -1  # whole second block empty
    _check(tk, qb=64, u_max=40)


def test_duplicates_within_query():
    tk = np.array([[3, 3, 7, 7], [7, 3, 3, 3], [-1, 5, 5, -1]], np.int32)
    _check(tk, qb=2, u_max=8)


def test_overflow_reported_uncapped():
    # true union (16) exceeds u_max (8): counts must report the TRUE size so the
    # caller can gate; retained prefix = first u_max sorted uniques.
    tk = np.arange(16, dtype=np.int32).reshape(4, 4)  # one block, 16 uniques
    units, member, counts = build_block_unit_tables(jnp.asarray(tk), query_block=4, u_max=8)
    assert int(counts[0]) == 16
    np.testing.assert_array_equal(np.asarray(units[0]), np.arange(8))
    ounits, omember, _ = _oracle(tk, 4, 8)
    np.testing.assert_array_equal(np.asarray(member), omember)


@pytest.mark.parametrize("num_seqs", [8, 32])  # concurrency envelope: cc=8 / cc=32
def test_packed_multirequest_global_keys(num_seqs):
    # Ragged/packed form (cc>1): requests packed on the token axis, seq-local page
    # ids lifted to global keys by each request's page-table base. Query blocks
    # straddle request boundaries — membership must still be exact per query.
    rng = np.random.default_rng(4)
    K, pages_per_seq = 8, 24
    q_lens = rng.integers(3, 90, size=num_seqs)
    rows = []
    for rid in range(num_seqs):
        base = rid * pages_per_seq
        tk = rng.integers(0, pages_per_seq, size=(q_lens[rid], K)).astype(np.int32)
        tk[tk % 5 == 0] = -1  # sprinkle padding
        rows.append(np.where(tk >= 0, tk + base, -1))
    tk_packed = np.concatenate(rows, axis=0)
    _check(tk_packed, qb=64, u_max=num_seqs * pages_per_seq)


def test_build_write_runs_table():
    # run-table builder for the pallas write-back: word/token split + overflow
    from sgl_jax.srt.kernels.dsa.sparse_mla_prefill_qblock import _build_write_runs

    pk = 2
    # aligned contiguous run + gap(-1) + odd-phase run
    loc = np.array([4, 5, 6, 7, -1, -1, 11, 12, 13], np.int32)  # src 0..3 / 6..8
    tbl, n_raw = _build_write_runs(jnp.asarray(loc), kv_packing=pk, r_cap=8)
    tbl = np.asarray(tbl)
    assert int(n_raw) == 2
    live = tbl[tbl[:, 3] > 0]
    # reconstruct per-token writes from the table and compare with the oracle
    writes = {}
    for kind, src, dst, n in live.tolist():
        for i in range(n):
            if kind == 0:  # word run
                for w in range(pk):
                    writes[(dst + i) * pk + w] = (src + i) * pk + w
            else:
                writes[dst + i] = src + i
    want = {int(l): t for t, l in enumerate(loc) if l >= 0}
    assert writes == want, (writes, want)
