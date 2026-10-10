import numpy as np
import pytest

from sgl_jax.srt.layers.dcp.layout import (
    get_dcp_lens,
    owner,
    physical_index,
    validate_dcp_mesh,
    virtual_index,
    virtual_page_size,
)
from sgl_jax.srt.layers.dcp.merge import merge_dcp_attention


def test_dcp_size_1_is_identity():
    lens = np.array([0, 1, 17, 2048], dtype=np.int32)
    np.testing.assert_array_equal(get_dcp_lens(lens, 1, 0), lens)
    np.testing.assert_array_equal(physical_index(np.arange(8), 1), np.arange(8))
    np.testing.assert_array_equal(owner(np.arange(8), 1), np.zeros(8, dtype=int))
    assert virtual_page_size(64, 1) == 64


def test_get_dcp_lens_17_over_16():
    # 17 // 16 + (rank < 1) → rank 0 has 2, ranks 1..15 have 1.
    for rank in range(16):
        got = int(get_dcp_lens(17, 16, rank))
        assert got == (2 if rank == 0 else 1), rank


def test_virtual_page_balanced_no_holes():
    page_size, dcp_size = 64, 16
    vpage = virtual_page_size(page_size, dcp_size)
    assert vpage == 1024
    tokens = np.arange(vpage)
    seen = {r: [] for r in range(dcp_size)}
    for v in tokens:
        seen[int(owner(v, dcp_size))].append(int(physical_index(v, dcp_size)))
    for rank, slots in seen.items():
        assert slots == list(range(page_size)), rank
    # Union of physical slots is 0..63 on every rank; owners partition the virtual page.
    owners = owner(tokens, dcp_size)
    assert set(owners.tolist()) == set(range(dcp_size))
    assert len(owners) == vpage


def test_block_interleave_puts_one_dsa_page_on_one_rank():
    """The property Slice 7 rests on: at ``interleave == page_size`` a DSA page
    (128 consecutive virtual tokens) is entirely on rank ``d % dcp`` as the whole
    physical page ``d // dcp`` — so the page-granular kernels can read it as-is."""
    page_size, dcp_size = 128, 16
    for d in range(64):
        v = np.arange(d * page_size, (d + 1) * page_size)
        owners = owner(v, dcp_size, page_size)
        assert set(owners.tolist()) == {d % dcp_size}, d
        phys = physical_index(v, dcp_size, page_size)
        # one contiguous physical page, in order
        expected = (d // dcp_size) * page_size + np.arange(page_size)
        np.testing.assert_array_equal(phys, expected)


def test_block_interleave_is_a_bijection_and_balanced():
    page_size, dcp_size = 128, 16
    vpage = virtual_page_size(page_size, dcp_size)
    assert vpage == 2048  # unchanged by interleave: the allocator is untouched
    tokens = np.arange(4 * vpage)
    seen = {r: [] for r in range(dcp_size)}
    for v in tokens:
        seen[int(owner(v, dcp_size, page_size))].append(int(physical_index(v, dcp_size, page_size)))
    for rank, slots in seen.items():
        # every rank gets an equal, hole-free share of physical slots
        assert slots == sorted(slots), rank
        assert slots == list(range(len(tokens) // dcp_size)), rank
    # round-trip through virtual_index
    for v in tokens[::37]:
        r = int(owner(v, dcp_size, page_size))
        p = int(physical_index(v, dcp_size, page_size))
        assert int(virtual_index(p, dcp_size, r, page_size)) == int(v)


def test_interleave_1_is_unchanged():
    """I=1 must reproduce the token-striped CUDA rule exactly."""
    v = np.arange(1000)
    for dcp in (1, 2, 16):
        np.testing.assert_array_equal(owner(v, dcp, 1), v % dcp)
        np.testing.assert_array_equal(physical_index(v, dcp, 1), v // dcp)
        for lens in (0, 1, 17, 2048):
            for r in range(dcp):
                assert int(get_dcp_lens(lens, dcp, r, interleave=1)) == int(
                    get_dcp_lens(lens, dcp, r)
                )


def test_get_dcp_lens_counts_owned_blocks():
    page_size, dcp_size = 128, 16
    vpage = page_size * dcp_size
    for lens in (0, 1, 127, 128, 129, 2048, 2049, 7001, 8192):
        counts = [
            int(get_dcp_lens(lens, dcp_size, r, interleave=page_size)) for r in range(dcp_size)
        ]
        # partition: the per-rank counts must sum to the total
        assert sum(counts) == lens, (lens, counts)
        # and match a brute-force count of the owner rule
        v = np.arange(lens)
        brute = [int((owner(v, dcp_size, page_size) == r).sum()) for r in range(dcp_size)]
        assert counts == brute, (lens, counts, brute)
        # full virtual pages are exactly balanced
        if lens % vpage == 0:
            assert len(set(counts)) == 1, (lens, counts)


def test_validate_dcp_mesh():
    validate_dcp_mesh(tp_size=16, dp_size=1, dcp_size=1)
    validate_dcp_mesh(tp_size=16, dp_size=1, dcp_size=16)
    with pytest.raises(ValueError, match="must divide attention_tp"):
        validate_dcp_mesh(tp_size=16, dp_size=1, dcp_size=3)
    with pytest.raises(ValueError, match="must divide attention_tp"):
        validate_dcp_mesh(tp_size=16, dp_size=16, dcp_size=16)


def _softmax_attn(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """q [B,D], k/v [S,D] → (out [B,D], lse [B]) natural log."""
    scores = q @ k.T
    m = np.max(scores, axis=-1, keepdims=True)
    ex = np.exp(scores - m)
    z = np.sum(ex, axis=-1, keepdims=True)
    out = (ex / z) @ v
    lse = (m.squeeze(-1) + np.log(z.squeeze(-1))).astype(np.float32)
    return out.astype(np.float32), lse


def test_merge_matches_full_softmax():
    rng = np.random.default_rng(0)
    b, s, d, n = 3, 16, 8, 2
    q = rng.normal(size=(b, d)).astype(np.float32)
    k = rng.normal(size=(s, d)).astype(np.float32)
    v = rng.normal(size=(s, d)).astype(np.float32)
    full_o, full_lse = _softmax_attn(q, k, v)

    partial_o = []
    partial_lse = []
    for r in range(n):
        idx = np.arange(r, s, n)
        o_r, lse_r = _softmax_attn(q, k[idx], v[idx])
        partial_o.append(o_r)
        partial_lse.append(lse_r)
    merged_o, merged_lse = merge_dcp_attention(
        np.stack(partial_o, axis=0), np.stack(partial_lse, axis=0)
    )
    np.testing.assert_allclose(merged_o, full_o, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(merged_lse, full_lse, rtol=1e-5, atol=1e-5)


def test_merge_empty_shard_is_zero_weight():
    rng = np.random.default_rng(1)
    b, d = 2, 4
    o0 = rng.normal(size=(b, d)).astype(np.float32)
    lse0 = rng.normal(size=(b,)).astype(np.float32)
    o1 = np.zeros((b, d), dtype=np.float32)
    lse1 = np.full((b,), -np.inf, dtype=np.float32)
    merged_o, merged_lse = merge_dcp_attention(
        np.stack([o0, o1], axis=0), np.stack([lse0, lse1], axis=0)
    )
    np.testing.assert_allclose(merged_o, o0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(merged_lse, lse0, rtol=1e-6, atol=1e-6)
    assert np.isfinite(merged_o).all()
    assert np.isfinite(merged_lse).all()


def test_merge_jax_matches_numpy():
    rng = np.random.default_rng(2)
    b, h, d, n = 2, 3, 8, 4
    partial_o = rng.normal(size=(n, b, h, d)).astype(np.float32)
    partial_lse = rng.normal(size=(n, b, h)).astype(np.float32)
    partial_lse[1] = -np.inf
    partial_o[1] = 0
    np_o, np_lse = merge_dcp_attention(partial_o, partial_lse)
    from sgl_jax.srt.layers.dcp.merge import merge_dcp_attention_jax

    jax_o, jax_lse = merge_dcp_attention_jax(partial_o, partial_lse)
    np.testing.assert_allclose(np.asarray(jax_o), np_o, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(jax_lse), np_lse, rtol=1e-5, atol=1e-5)
