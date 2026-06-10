"""CPU mock test for device-chain state timing (Wave 3d v2).

Verifies the host-side allocate_lens / seq_lens bookkeeping under the
chain event_loop ordering (bump → prepare_for_decode → unbump → finalize)
produces correct per-round ext (= accept, no cumulative drift) and that
the optimistic alloc upper bound covers the real verify KV-write range.
Does NOT exercise the fused JIT (device-only); that's TPU-gated.
"""
import numpy as np
import pytest


NDT = 4


class MockReqInfo:
    def __init__(self, seq_lens, alloc_lens):
        self.seq_lens = np.asarray(seq_lens, dtype=np.int64)
        self.allocate_lens = np.asarray(alloc_lens, dtype=np.int64)
        self.reqs = list(range(len(seq_lens)))


def chain_round(info: MockReqInfo, accept_prev: np.ndarray):
    """One chain loop iteration: finalize_{k-1} → bump → prep_k → unbump.

    Returns (ext_allocated, new_r_optimistic, verify_write_range_real).
    """
    # finalize_{k-1}: seq_lens += accept (real)
    info.seq_lens = info.seq_lens + accept_prev  # real_{k-1}
    real_seq = info.seq_lens.copy()

    # bump (optimistic for in-flight round k)
    info.seq_lens = info.seq_lens + NDT

    # prepare_for_decode (uses bumped seq_lens)
    old_r = info.allocate_lens
    new_r = info.seq_lens + NDT - 1
    ext = new_r - old_r
    info.allocate_lens = new_r

    # unbump
    info.seq_lens = info.seq_lens - NDT

    # round k verify writes [real_seq : real_seq+ndt] (device-side, real seq_lens)
    verify_hi = real_seq + NDT
    return ext, new_r, verify_hi


def test_chain_ext_equals_accept_no_drift():
    """ext_r per round must equal accept_{k-1} (real), not cumulative."""
    # initial: post-prefill, first decode round already prepped (alloc_0 optimistic)
    seq0 = np.array([512, 600, 700, 512])  # real_{-1} (post-prefill)
    # first prep (round 0) used bump: alloc_0 = seq0 + 2*ndt - 1
    alloc0 = seq0 + 2 * NDT - 1
    info = MockReqInfo(seq0, alloc0)

    accepts = [
        np.array([3, 4, 1, 2]),
        np.array([4, 2, 3, 4]),
        np.array([1, 1, 4, 3]),
        np.array([2, 3, 2, 1]),
    ]
    for k, acc in enumerate(accepts):
        ext, new_r, verify_hi = chain_round(info, acc)
        # ext must equal accept (no drift)
        np.testing.assert_array_equal(ext, acc), f"round {k}: ext={ext} != accept={acc}"
        # alloc upper bound must cover verify write range [real:real+ndt]
        assert np.all(verify_hi - 1 <= new_r), (
            f"round {k}: verify_hi-1={verify_hi-1} > alloc new_r={new_r} (under-alloc)"
        )
        # alloc must not over-shoot by more than ndt (bounded over-alloc)
        assert np.all(new_r - (verify_hi - 1) <= NDT), (
            f"round {k}: over-alloc {new_r - (verify_hi-1)} > ndt"
        )


def test_chain_first_round_from_prefill():
    """First decode round (no prev chain): alloc_0 from non-optimistic old_r.

    Prefill sets allocate_lens = prefill_len (not optimistic). First chain
    prep bumps seq_lens but old_r = prefill_len → ext = 2*ndt-1 (one-time
    over-alloc), covered by drain release.
    """
    seq0 = np.array([512, 600])
    alloc_prefill = seq0.copy()  # prefill: allocate_lens = prefill_len
    info = MockReqInfo(seq0, alloc_prefill)

    # first chain round: accept_prev = 1 (prefill output 1 token, but seq_lens
    # already includes it? assume seq0 is post-prefill-output, accept_prev=0 here)
    # Actually finalize for prefill round writes seq_lens += 1 (extend output).
    ext, new_r, verify_hi = chain_round(info, np.array([1, 1]))
    # ext = (seq0+1+2ndt-1) - seq0 = 2ndt → one-time over-alloc of (2ndt - accept)
    assert np.all(ext == 2 * NDT), f"first-round ext={ext}"
    assert np.all(verify_hi - 1 <= new_r)

    # second round onwards: ext = accept
    ext2, new_r2, _ = chain_round(info, np.array([3, 2]))
    np.testing.assert_array_equal(ext2, [3, 2])


def test_chain_batch_shrink_shape_consistent():
    """After a req finishes (filter), spec_info.allocate_lens shape must match reqs.

    chain mode: finalize does NOT write spec_info; prep maintains it. filter_batch
    must shrink both reqs and allocate_lens consistently.
    """
    info = MockReqInfo([512, 600, 700], np.array([512, 600, 700]) + 2 * NDT - 1)
    chain_round(info, np.array([3, 4, 2]))
    # req[1] finishes → filter
    keep = [0, 2]
    info.seq_lens = info.seq_lens[keep]
    info.allocate_lens = info.allocate_lens[keep]
    info.reqs = [info.reqs[i] for i in keep]
    # next round must not assert shape mismatch
    assert info.allocate_lens.shape[0] == len(info.reqs) == 2
    ext, _, _ = chain_round(info, np.array([2, 3]))
    np.testing.assert_array_equal(ext, [2, 3])


def test_chain_finalize_must_not_overwrite_alloc():
    """Regression for step2 crash: if finalize writes spec_info (with stale
    allocate_lens from round k-1), next prep's old_r is wrong → ext drifts.
    This test documents why finalize must skip spec_info under chain.
    """
    info = MockReqInfo([512], [512 + 2 * NDT - 1])
    chain_round(info, np.array([3]))  # alloc_1 = 512+3 + 2ndt-1
    alloc_1 = info.allocate_lens.copy()

    # WRONG: finalize overwrites with stale alloc_0
    stale_alloc_0 = np.array([512 + 2 * NDT - 1])
    info.allocate_lens = stale_alloc_0  # ← what step2 did

    # next prep: old_r = stale_alloc_0, new_r = (512+3+4 +ndt) + ndt-1
    info.seq_lens = info.seq_lens + 4  # finalize accept=4
    info.seq_lens = info.seq_lens + NDT  # bump
    new_r = info.seq_lens + NDT - 1
    ext_wrong = new_r - stale_alloc_0
    ext_right = new_r - alloc_1
    # wrong ext = accept_0 + accept_1 = 7 (cumulative); right = accept_1 = 4
    assert ext_wrong[0] == 7 and ext_right[0] == 4, (ext_wrong, ext_right)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
