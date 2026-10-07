"""Slice 3 W0: owner-only prefill scatter into a mocked per-rank pool."""

from __future__ import annotations

import numpy as np
import pytest

from sgl_jax.srt.layers.dcp.layout import physical_page_indices, physical_write_loc


def _scatter_owned(
    virtual_loc: np.ndarray,
    values: np.ndarray,
    dcp_size: int,
    dcp_rank: int,
    n_physical: int,
) -> np.ndarray:
    """Numpy stand-in for ``flat.at[physical_write_loc].set(..., drop -1)``."""
    dest = np.zeros((n_physical,) + values.shape[1:], dtype=values.dtype)
    write_loc = physical_write_loc(virtual_loc, dcp_size, dcp_rank)
    for i, slot in enumerate(write_loc):
        if slot >= 0:
            dest[int(slot)] = values[i]
    return dest


def test_physical_write_loc_jax_matches_numpy():
    jnp = pytest.importorskip("jax.numpy")
    from sgl_jax.srt.layers.dcp.write import physical_write_loc_jax

    loc = np.array([-1, 0, 1, 2, 7], dtype=np.int32)
    np.testing.assert_array_equal(np.asarray(physical_write_loc_jax(jnp.asarray(loc), 1, 0)), loc)
    for rank in (0, 1):
        np.testing.assert_array_equal(
            np.asarray(physical_write_loc_jax(jnp.asarray(loc), 2, rank)),
            physical_write_loc(loc, 2, rank),
        )


def test_physical_positions_are_the_exact_causal_bound():
    """The attend kernels mask ``physical_slot <= positions``. Under DCP that is
    only correct if ``physical_positions_jax`` yields a bound whose slot set is
    *exactly* the owned virtual tokens at or before the query — check it against
    brute force for every rank, both interleaves, including the empty-shard case.
    """
    import jax.numpy as jnp

    from sgl_jax.srt.layers.dcp.layout import owner, physical_index
    from sgl_jax.srt.layers.dcp.write import physical_positions_jax

    dcp, page = 4, 8
    for interleave in (1, page):
        for rank in range(dcp):
            pos = np.arange(0, 6 * page, dtype=np.int32)
            bound = np.asarray(physical_positions_jax(jnp.asarray(pos), dcp, rank, interleave))
            for i, p in enumerate(pos):
                v = np.arange(p + 1)
                expect = sorted(
                    int(physical_index(x, dcp, interleave))
                    for x in v
                    if owner(x, dcp, interleave) == rank
                )
                got = list(range(int(bound[i]) + 1))
                assert got == expect, (interleave, rank, int(p), got, expect)
            # a rank owning nothing yet must get -1 so no key passes the mask
            if rank > 0:
                assert (
                    int(physical_positions_jax(jnp.array([0], np.int32), dcp, rank, interleave)[0])
                    == -1
                )


def test_physical_write_loc_jax_matches_numpy_block_interleaved():
    import jax.numpy as jnp

    from sgl_jax.srt.layers.dcp.write import physical_write_loc_jax

    page_size, dcp_size = 128, 16
    loc = np.concatenate([np.arange(-1, 4097), np.full(5, -1)]).astype(np.int32)
    for rank in range(dcp_size):
        np.testing.assert_array_equal(
            np.asarray(physical_write_loc_jax(jnp.asarray(loc), dcp_size, rank, page_size)),
            physical_write_loc(loc, dcp_size, rank, page_size),
        )
    # every non-padded token is written by exactly one rank, to a distinct slot
    written = {}
    for rank in range(dcp_size):
        phys = np.asarray(physical_write_loc(loc, dcp_size, rank, page_size))
        for i, p in enumerate(phys):
            if p >= 0:
                assert (rank, int(p)) not in written
                written[(rank, int(p))] = i
    assert len(written) == int((loc >= 0).sum())


def test_physical_write_loc_dcp1_is_identity():
    loc = np.array([-1, 0, 1, 7], dtype=np.int32)
    np.testing.assert_array_equal(physical_write_loc(loc, 1, 0), loc)
    assert physical_write_loc(5, 1, 0) == 5
    assert physical_write_loc(-1, 1, 0) == -1


def test_w0_8_token_prompt_n2_stripes_even_odd():
    # Virtual tokens 0..7, distinctive payload = 10 * v. Rank 0 owns even,
    # rank 1 owns odd; physical slots are v//2.
    dcp_size = 2
    loc = np.arange(8, dtype=np.int32)
    values = (10 * loc).reshape(8, 1)
    rank0 = _scatter_owned(loc, values, dcp_size, 0, n_physical=4)
    rank1 = _scatter_owned(loc, values, dcp_size, 1, n_physical=4)
    np.testing.assert_array_equal(rank0.ravel(), [0, 20, 40, 60])
    np.testing.assert_array_equal(rank1.ravel(), [10, 30, 50, 70])


def test_w0_padding_and_foreign_tokens_are_dropped():
    loc = np.array([-1, 0, 1, 2], dtype=np.int32)
    values = np.array([[9], [10], [11], [12]], dtype=np.int32)
    rank0 = _scatter_owned(loc, values, dcp_size=2, dcp_rank=0, n_physical=4)
    # Only virtual 0 and 2 (phys 0 and 1). Pad and odd token stay zero.
    np.testing.assert_array_equal(rank0.ravel(), [10, 12, 0, 0])


def test_physical_page_indices_dcp1_matches_current_stride():
    page_size = 64
    loc = np.arange(page_size, page_size + 128, dtype=np.int32)
    got = physical_page_indices(loc, page_size, dcp_size=1)
    expect = loc[::page_size] // page_size
    np.testing.assert_array_equal(got, expect)


def test_physical_page_indices_one_virtual_page():
    page_size, dcp_size = 64, 16
    # First allocated virtual page is slots 1024..2047 after the reserved page.
    loc = np.arange(1024, 2048, dtype=np.int32)
    pages = physical_page_indices(loc, page_size, dcp_size)
    # virtual 1024 → physical 64 → physical page 1 (page 0 is the reserved page).
    np.testing.assert_array_equal(pages, np.array([1], dtype=np.int32))
