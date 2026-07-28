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
    mod.TUNED_BLOCK_SIZES_V3["TPU v7"].clear()

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


def test_target_verify_uses_tokens_per_seq_bucket(monkeypatch):
    """Target verify must not share a key with ordinary MIXED traffic."""
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
    mod.TUNED_BLOCK_SIZES_V3["TPU v7"].clear()
    mod.TUNED_BLOCK_SIZES_V3["TPU v7"][
        ("v", None, 4, "bfloat16", "bfloat16", 16, 1, 256, 256, 128)
    ] = (4, 2048, 4, 2048)

    assert mod.get_tuned_block_sizes_v3(
        "v",
        jnp.bfloat16,
        jnp.bfloat16,
        16,
        1,
        256,
        256,
        128,
        tokens_per_seq=4,
    ) == (4, 2048, 4, 2048)
    assert (
        mod.get_tuned_block_sizes_v3(
            "v",
            jnp.bfloat16,
            jnp.bfloat16,
            16,
            1,
            256,
            256,
            128,
            tokens_per_seq=8,
        )
        is None
    )
    assert (
        mod.get_tuned_block_sizes_v3(
            "m",
            jnp.bfloat16,
            jnp.bfloat16,
            16,
            1,
            256,
            256,
            128,
        )
        is None
    )


def test_target_verify_requires_tokens_per_seq():
    with pytest.raises(ValueError, match="tokens_per_seq"):
        get_tuned_block_sizes_v3(
            "v",
            jnp.bfloat16,
            jnp.bfloat16,
            16,
            1,
            256,
            256,
            128,
        )


@pytest.mark.parametrize(
    ("sliding_window", "tokens_per_seq", "max_num_tokens", "expected"),
    [
        (None, 4, 16, (4, 2048, 4, 2048)),
        (None, 4, 32, (4, 2048, 4, 2048)),
        (None, 4, 64, (4, 2048, 4, 2048)),
        (None, 4, 128, (4, 2048, 4, 2048)),
        (None, 4, 256, (4, 2048, 4, 2048)),
        (None, 2, 128, (2, 2048, 2, 2048)),
        (None, 8, 128, (8, 2048, 8, 2048)),
        (128, 4, 16, (4, 256, 4, 256)),
        (128, 4, 32, (4, 256, 4, 256)),
        (128, 4, 64, (4, 256, 4, 256)),
        (128, 4, 128, (4, 256, 4, 256)),
        (128, 4, 256, (4, 256, 4, 256)),
        (128, 2, 128, (2, 256, 2, 256)),
        (128, 8, 128, (8, 256, 8, 256)),
    ],
)
def test_mimo_target_verify_tuned_matrix(
    monkeypatch,
    sliding_window,
    tokens_per_seq,
    max_num_tokens,
    expected,
):
    """The measured full/SWA matrix must route to its distinct winners."""
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

    assert (
        mod.get_tuned_block_sizes_v3(
            "v",
            jnp.bfloat16,
            jnp.bfloat16,
            16,
            1,
            256,
            256,
            max_num_tokens,
            sliding_window=sliding_window,
            tokens_per_seq=tokens_per_seq,
        )
        == expected
    )


def test_invalid_stage_raises():
    with pytest.raises(ValueError):
        get_tuned_block_sizes_v3("x", jnp.bfloat16, jnp.bfloat16, 32, 1, 128, 256, 64)


def test_unknown_device_returns_none(monkeypatch):
    from sgl_jax.srt.kernels.ragged_paged_attention import tuned_block_sizes_v3 as mod

    monkeypatch.setattr(mod, "get_tpu_version", lambda: 7)
    monkeypatch.setattr(
        mod,
        "get_simplified_key",
        lambda *args, **kwargs: (
            "TPU vUNKNOWN",
            "bfloat16",
            "bfloat16",
            32,
            1,
            128,
            256,
            64,
        ),
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
