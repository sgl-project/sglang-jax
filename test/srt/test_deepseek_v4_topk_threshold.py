"""Bisection top-k threshold (interpret) == exact top-k membership from lax.top_k."""

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.dsv4.topk_threshold import (
    score_key,
    topk_membership_mask,
    topk_threshold,
)
from sgl_jax.srt.layers.attention.dsv4.ref.topk_threshold import topk_threshold_ref


def _reference_mask(scores, k):
    """Membership of the exact top-k (ties: every entry equal to the k-th value)."""
    s = np.asarray(scores, np.float32)
    out = np.zeros(s.shape, bool)
    for t in range(s.shape[0]):
        finite = np.isfinite(s[t])
        if finite.sum() == 0:
            continue
        vals = np.sort(s[t][finite])[::-1]
        kth = vals[min(k, vals.size) - 1]
        out[t] = finite & (s[t] >= kth)
    return out


def test_score_key_is_monotone():
    x = np.array([-np.inf, -3.5, -1e-30, -0.0, 0.0, 1e-30, 2.0, np.inf], np.float32)
    keys = np.asarray(score_key(jnp.asarray(x)))
    assert np.all(np.diff(keys.astype(np.int64)) >= 0)
    assert keys[3] <= keys[4]  # -0.0 orders no later than +0.0


def test_threshold_matches_reference():
    rng = np.random.default_rng(0)
    for T, E, k in ((16, 256, 8), (40, 1024, 512), (9, 384, 64)):
        s = rng.standard_normal((T, E)).astype(np.float32) * 3
        s[rng.random((T, E)) < 0.3] = -np.inf  # illegal entries
        s[0] = -np.inf  # a row with nothing legal
        s[1, : k // 2] = 1.0  # fewer legal than k
        s[1, k // 2 :] = -np.inf
        s[2, :] = 0.25  # all ties
        s[3, 5] = s[3, 6]  # a tie at some rank
        want = _reference_mask(s, k)
        got = np.asarray(topk_membership_mask(jnp.asarray(s), k, interpret=True))
        np.testing.assert_array_equal(got, want)
        thr_k = np.asarray(topk_threshold(jnp.asarray(s), k, interpret=True))
        thr_x = np.asarray(topk_threshold_ref(jnp.asarray(s), k))
        np.testing.assert_array_equal(thr_k, thr_x)


def test_padding_rows_and_lanes():
    rng = np.random.default_rng(1)
    T, E, k = 13, 200, 16  # E not a lane multiple, T not a sublane multiple
    s = rng.standard_normal((T, E)).astype(np.float32)
    want = _reference_mask(s, k)
    got = np.asarray(topk_membership_mask(jnp.asarray(s), k, interpret=True))
    np.testing.assert_array_equal(got, want)
    assert got.sum(1).min() == k


def _membership_case(rng, T, Ep, k, q_lens, counts):
    """Scores in request-local columns for a request-major token layout."""
    from sgl_jax.srt.layers.attention.dsv4.indexer import INVALID_ENTRY  # noqa: F401

    req_ids = np.concatenate([np.full(n, r, np.int32) for r, n in enumerate(q_lens)])
    assert req_ids.size == T
    scores = np.full((T, Ep), -np.inf, np.float32)
    for t in range(T):
        r = req_ids[t]
        scores[t, : counts[r]] = rng.standard_normal(counts[r])
    offsets = np.cumsum(counts) - np.asarray(counts)
    valid = np.ones(T, bool)
    return (
        scores,
        offsets.astype(np.int32),
        np.asarray(q_lens, np.int32),
        req_ids,
        valid,
        sum(counts),
    )


def test_membership_from_scores_single_and_multi_request():
    from sgl_jax.srt.kernels.dsa.streamindex_topk import select_topk_indices
    from sgl_jax.srt.layers.attention.dsv4.attention import packed_membership
    from sgl_jax.srt.layers.attention.dsv4.indexer import membership_from_scores

    rng = np.random.default_rng(4)
    k = 16
    # one active request among four slots (slot 2), as in a padded bs bucket
    s, off, ql, rid, valid, E = _membership_case(rng, 24, 256, k, [0, 0, 24, 0], [0, 0, 200, 0])
    got = np.asarray(
        membership_from_scores(
            jnp.asarray(s),
            jnp.asarray(off),
            q_lens=jnp.asarray(ql),
            query_request_ids=jnp.asarray(rid),
            valid_token_mask=jnp.asarray(valid),
            k=k,
            num_entries=E,
        )
    )
    want = _reference_mask(s[:, :E], k)
    np.testing.assert_array_equal(got, want)
    assert got.sum(1).min() == k
    # two active requests: index path with offsets
    s, off, ql, rid, valid, E = _membership_case(rng, 24, 256, k, [10, 0, 14, 0], [100, 0, 60, 0])
    got = np.asarray(
        membership_from_scores(
            jnp.asarray(s),
            jnp.asarray(off),
            q_lens=jnp.asarray(ql),
            query_request_ids=jnp.asarray(rid),
            valid_token_mask=jnp.asarray(valid),
            k=k,
            num_entries=E,
        )
    )
    sel = np.asarray(select_topk_indices(jnp.asarray(s), k, backend="xla"))
    sel = np.where(sel >= 0, sel + off[rid][:, None], -1)
    want = np.asarray(packed_membership(jnp.asarray(sel), E))
    np.testing.assert_array_equal(got, want)
    # and it equals the exact per-request membership
    ref = np.zeros((24, E), bool)
    for t in range(24):
        r = rid[t]
        row = s[t, : (100 if r == 0 else 60)]
        kth = np.sort(row)[::-1][k - 1]
        ref[t, off[r] : off[r] + row.size] = row >= kth
    np.testing.assert_array_equal(got, ref)


def test_membership_from_scores_under_explicit_mesh():
    from jax.sharding import AxisType, NamedSharding
    from jax.sharding import PartitionSpec as P

    from sgl_jax.srt.layers.attention.dsv4.indexer import membership_from_scores

    rng = np.random.default_rng(5)
    k = 8
    s, off, ql, rid, valid, E = _membership_case(rng, 16, 128, k, [0, 16], [0, 100])
    mesh = jax.make_mesh(
        (1, 1), ("data", "tensor"), axis_types=(AxisType.Explicit, AxisType.Explicit)
    )

    # The server runs the CSA path inside a shard_map region (pallas_call needs
    # manual axes under explicit meshes); mirror that here.
    def body(a):
        return membership_from_scores(
            a,
            jnp.asarray(off),
            q_lens=jnp.asarray(ql),
            query_request_ids=jnp.asarray(rid),
            valid_token_mask=jnp.asarray(valid),
            k=k,
            num_entries=E,
        )

    with jax.set_mesh(mesh):
        ss = jax.device_put(jnp.asarray(s), NamedSharding(mesh, P("data", None)))
        f = jax.jit(
            jax.shard_map(
                body,
                mesh=mesh,
                in_specs=P("data", None),
                out_specs=P("data", None),
                check_vma=False,
            )
        )
        got = np.asarray(f(ss))
    np.testing.assert_array_equal(got, _reference_mask(s[:, :E], k))
