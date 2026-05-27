"""Tests for the v3 tuned block-size table and lookup."""

import jax.numpy as jnp
import pytest

from sgl_jax.srt.kernels.ragged_paged_attention import tuned_block_sizes_v3
from sgl_jax.srt.kernels.ragged_paged_attention.tuned_block_sizes_v3 import (
    get_tuned_block_sizes_v3,
)


@pytest.fixture(autouse=True)
def _clean_table():
    """Snapshot/restore the module-level table around each test."""
    snapshot = {k: dict(v) for k, v in tuned_block_sizes_v3.TUNED_BLOCK_SIZES_V3.items()}
    yield
    tuned_block_sizes_v3.TUNED_BLOCK_SIZES_V3.clear()
    tuned_block_sizes_v3.TUNED_BLOCK_SIZES_V3.update(snapshot)


def test_empty_table_returns_none():
    assert get_tuned_block_sizes_v3("d", jnp.bfloat16, jnp.bfloat16, 32, 1, 128, 256, 64) is None


def test_lookup_hit_returns_tuple(monkeypatch):
    from sgl_jax.srt.kernels.ragged_paged_attention import tuned_block_sizes_v3 as mod

    def fake_tpu_version():
        return 7

    monkeypatch.setattr(mod, "get_tpu_version", fake_tpu_version)

    # Patch get_simplified_key to return a deterministic device label.
    def fake_simplified_key(page_size, q_dtype, kv_dtype, q_h, kv_h, hd, mnt):
        return ("TPU v7", "bfloat16", "bfloat16", q_h, kv_h, hd, page_size, mnt)

    monkeypatch.setattr(mod, "get_simplified_key", fake_simplified_key)

    # Non-SWA entry (sliding_window=None in the key)
    mod.TUNED_BLOCK_SIZES_V3["TPU v7"][("d", None, "bfloat16", "bfloat16", 32, 1, 128, 256, 64)] = (
        1,
        4096,
        1,
        4096,
    )

    assert mod.get_tuned_block_sizes_v3("d", jnp.bfloat16, jnp.bfloat16, 32, 1, 128, 256, 64) == (
        1,
        4096,
        1,
        4096,
    )


def test_sliding_window_separates_buckets(monkeypatch):
    """SWA layers must NOT hit non-SWA entries (different best bkv)."""
    from sgl_jax.srt.kernels.ragged_paged_attention import tuned_block_sizes_v3 as mod

    monkeypatch.setattr(mod, "get_tpu_version", lambda: 7)
    monkeypatch.setattr(
        mod,
        "get_simplified_key",
        lambda ps, q_dt, kv_dt, q_h, kv_h, hd, mnt: (
            "TPU v7",
            "bfloat16",
            "bfloat16",
            q_h,
            kv_h,
            hd,
            ps,
            mnt,
        ),
    )

    # Add a non-SWA entry only
    mod.TUNED_BLOCK_SIZES_V3["TPU v7"][("d", None, "bfloat16", "bfloat16", 32, 2, 256, 256, 64)] = (
        1,
        2048,
        1,
        2048,
    )

    # Non-SWA lookup hits
    assert mod.get_tuned_block_sizes_v3("d", jnp.bfloat16, jnp.bfloat16, 32, 2, 256, 256, 64) == (
        1,
        2048,
        1,
        2048,
    )
    # SWA lookup misses (bucket isn't there yet)
    assert (
        mod.get_tuned_block_sizes_v3(
            "d", jnp.bfloat16, jnp.bfloat16, 32, 2, 256, 256, 64, sliding_window=128
        )
        is None
    )

    # Add SWA entry with different best
    mod.TUNED_BLOCK_SIZES_V3["TPU v7"][("d", 128, "bfloat16", "bfloat16", 32, 2, 256, 256, 64)] = (
        1,
        256,
        1,
        256,
    )
    assert mod.get_tuned_block_sizes_v3(
        "d", jnp.bfloat16, jnp.bfloat16, 32, 2, 256, 256, 64, sliding_window=128
    ) == (1, 256, 1, 256)


def test_invalid_stage_raises():
    with pytest.raises(ValueError):
        get_tuned_block_sizes_v3("x", jnp.bfloat16, jnp.bfloat16, 32, 1, 128, 256, 64)


def test_unknown_device_returns_none(monkeypatch):
    from sgl_jax.srt.kernels.ragged_paged_attention import tuned_block_sizes_v3 as mod

    monkeypatch.setattr(mod, "get_tpu_version", lambda: 7)
    monkeypatch.setattr(
        mod,
        "get_simplified_key",
        lambda *args, **kwargs: ("TPU vUNKNOWN", "bfloat16", "bfloat16", 32, 1, 128, 256, 64),
    )
    assert (
        mod.get_tuned_block_sizes_v3("d", jnp.bfloat16, jnp.bfloat16, 32, 1, 128, 256, 64) is None
    )


def test_pre_v5_returns_none(monkeypatch):
    from sgl_jax.srt.kernels.ragged_paged_attention import tuned_block_sizes_v3 as mod

    monkeypatch.setattr(mod, "get_tpu_version", lambda: 4)
    assert (
        mod.get_tuned_block_sizes_v3("d", jnp.bfloat16, jnp.bfloat16, 32, 1, 128, 256, 64) is None
    )
