"""Unit tests for the req-pool-slot future-token map (overlap scheduling).

Regression for the padded-bs ring wraparound: a burst of prefill batches
advanced the old ring cursor by the PADDED batch size and overwrote
outstanding placeholders before the owning request's first decode resolved
them. The map is now indexed by req_pool_idx + 1; these tests pin the
set/resolve round trip, padding-row drop, and wraparound immunity.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.managers.utils import future_slot_indices, set_future_token_ids


@pytest.fixture
def mesh():
    return jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",))


def _map(size=18):
    return jnp.zeros((size,), dtype=jnp.int32)


def test_set_resolve_round_trip(mesh):
    fmap = _map()
    seq_lens = np.array([5, 3, 9, 0], dtype=np.int32)  # last row is padding
    req_pool = np.array([7, 2, 11, 0], dtype=np.int32)
    slots = future_slot_indices(seq_lens, req_pool, fmap.shape[0])
    tokens = jnp.array([101, 202, 303, 999], dtype=jnp.int32)

    fmap = set_future_token_ids(fmap, slots, tokens, mesh)

    # resolve_future_token_ids (unchanged by this fix) computes
    # map[-placeholder] for negative ids; verify that lookup directly.
    placeholders = np.where(seq_lens > 0, -slots, 0).astype(np.int32)
    fmap_np = np.asarray(fmap)
    np.testing.assert_array_equal(fmap_np[-placeholders[:3]], [101, 202, 303])
    # padding row keeps a non-negative id (0) -> resolve passes it through
    assert placeholders[3] == 0


def test_padding_rows_do_not_clobber_slot_zero(mesh):
    """Old failure shape: padding rows carrying req_pool_idx fill value 0
    must not overwrite the request that legitimately owns pool idx 0."""
    fmap = _map()
    # real request at pool idx 0 writes its pending token
    slots_a = future_slot_indices(
        np.array([4], dtype=np.int32), np.array([0], dtype=np.int32), fmap.shape[0]
    )
    fmap = set_future_token_ids(fmap, slots_a, jnp.array([555], dtype=jnp.int32), mesh)
    # another batch: one real row (pool idx 3) + three padding rows (fill 0)
    seq_lens = np.array([6, 0, 0, 0], dtype=np.int32)
    req_pool = np.array([3, 0, 0, 0], dtype=np.int32)
    slots_b = future_slot_indices(seq_lens, req_pool, fmap.shape[0])
    fmap = set_future_token_ids(fmap, slots_b, jnp.array([777, -1, -1, -1], dtype=jnp.int32), mesh)

    assert int(fmap[1]) == 555  # pool idx 0 slot untouched by padding rows
    assert int(fmap[4]) == 777


def test_prefill_burst_does_not_wrap(mesh):
    """The ring-cursor design corrupted this exact scenario: many padded
    prefill batches between a request's prefill and its first decode."""
    fmap = _map(size=34)
    # request R prefills first: pool idx 5, pending token 4242
    slots_r = future_slot_indices(
        np.array([7], dtype=np.int32), np.array([5], dtype=np.int32), fmap.shape[0]
    )
    fmap = set_future_token_ids(fmap, slots_r, jnp.array([4242], dtype=jnp.int32), mesh)
    # 50 subsequent padded prefill batches for OTHER requests
    rng = np.random.default_rng(0)
    for step in range(50):
        pool_idx = int(rng.integers(6, 32))
        seq_lens = np.full(8, 0, dtype=np.int32)  # padded batch of 8, 1 real row
        req_pool = np.zeros(8, dtype=np.int32)
        seq_lens[0], req_pool[0] = 3, pool_idx
        slots = future_slot_indices(seq_lens, req_pool, fmap.shape[0])
        toks = jnp.full((8,), 10_000 + step, dtype=jnp.int32)
        fmap = set_future_token_ids(fmap, slots, toks, mesh)
    # R's slot (pool idx 5 -> map index 6) must still hold its own token
    assert int(np.asarray(fmap)[6]) == 4242
