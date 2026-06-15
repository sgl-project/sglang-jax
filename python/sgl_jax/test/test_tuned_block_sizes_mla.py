"""Tests for the MLA v2 tuned block-size table and lookup."""

import jax.numpy as jnp
import pytest

from sgl_jax.srt.kernels.mla.v2 import tuned_block_sizes
from sgl_jax.srt.kernels.mla.v2.tuned_block_sizes import get_tuned_block_sizes_mla


@pytest.fixture(autouse=True)
def _clean_state():
    """Snapshot/restore the module-level table and warned-misses set."""
    table_snapshot = {k: dict(v) for k, v in tuned_block_sizes.TUNED_BLOCK_SIZES_MLA.items()}
    warned_snapshot = set(tuned_block_sizes._WARNED_MISSES)
    yield
    tuned_block_sizes.TUNED_BLOCK_SIZES_MLA.clear()
    tuned_block_sizes.TUNED_BLOCK_SIZES_MLA.update(table_snapshot)
    tuned_block_sizes._WARNED_MISSES.clear()
    tuned_block_sizes._WARNED_MISSES.update(warned_snapshot)


def _patch_env(monkeypatch, *, tpu_version: int = 7, device: str = "TPU v7"):
    monkeypatch.setattr(tuned_block_sizes, "get_tpu_version", lambda: tpu_version)
    monkeypatch.setattr(tuned_block_sizes, "get_device_name", lambda: device)


def test_empty_table_returns_none(monkeypatch):
    _patch_env(monkeypatch)
    assert (
        get_tuned_block_sizes_mla("decode", jnp.bfloat16, jnp.bfloat16, 8, 512, 64, 256, 128)
        is None
    )
    assert (
        get_tuned_block_sizes_mla("mixed", jnp.bfloat16, jnp.bfloat16, 8, 512, 64, 256, 512) is None
    )


def test_decode_hit_returns_3tuple(monkeypatch):
    _patch_env(monkeypatch)
    tuned_block_sizes.TUNED_BLOCK_SIZES_MLA["TPU v7"][
        ("decode", "bfloat16", "bfloat16", 8, 512, 64, 256, 128)
    ] = (3, 2, 8)
    assert get_tuned_block_sizes_mla(
        "decode", jnp.bfloat16, jnp.bfloat16, 8, 512, 64, 256, 128
    ) == (3, 2, 8)


def test_mixed_hit_returns_2tuple(monkeypatch):
    _patch_env(monkeypatch)
    tuned_block_sizes.TUNED_BLOCK_SIZES_MLA["TPU v7"][
        ("mixed", "bfloat16", "bfloat16", 8, 512, 64, 256, 512)
    ] = (4, 32)
    assert get_tuned_block_sizes_mla("mixed", jnp.bfloat16, jnp.bfloat16, 8, 512, 64, 256, 512) == (
        4,
        32,
    )


def test_decode_and_mixed_buckets_are_separate(monkeypatch):
    """Same shape under different case_label must not collide."""
    _patch_env(monkeypatch)
    tuned_block_sizes.TUNED_BLOCK_SIZES_MLA["TPU v7"][
        ("decode", "bfloat16", "bfloat16", 8, 512, 64, 256, 128)
    ] = (3, 2, 8)

    # decode hits
    assert get_tuned_block_sizes_mla(
        "decode", jnp.bfloat16, jnp.bfloat16, 8, 512, 64, 256, 128
    ) == (3, 2, 8)
    # mixed misses (no entry for "mixed" at this shape)
    assert (
        get_tuned_block_sizes_mla("mixed", jnp.bfloat16, jnp.bfloat16, 8, 512, 64, 256, 128) is None
    )


def test_key_normalization_pow2_buckets(monkeypatch):
    """num_q_heads, page_size, max_num_tokens are bucketed by next_power_of_2."""
    _patch_env(monkeypatch)
    # Stored under pow2-128 key; lookup with mnt=100 → next_pow2=128 → hits.
    tuned_block_sizes.TUNED_BLOCK_SIZES_MLA["TPU v7"][
        ("decode", "bfloat16", "bfloat16", 8, 512, 64, 256, 128)
    ] = (3, 2, 8)
    assert get_tuned_block_sizes_mla(
        "decode", jnp.bfloat16, jnp.bfloat16, 8, 512, 64, 256, 100
    ) == (3, 2, 8)
    # mnt=200 → next_pow2=256 → no entry → None
    assert (
        get_tuned_block_sizes_mla("decode", jnp.bfloat16, jnp.bfloat16, 8, 512, 64, 256, 200)
        is None
    )


def test_invalid_case_label_raises():
    with pytest.raises(ValueError):
        get_tuned_block_sizes_mla("prefill", jnp.bfloat16, jnp.bfloat16, 8, 512, 64, 256, 128)


def test_unknown_device_returns_none(monkeypatch):
    _patch_env(monkeypatch, device="TPU vUNKNOWN")
    assert (
        get_tuned_block_sizes_mla("decode", jnp.bfloat16, jnp.bfloat16, 8, 512, 64, 256, 128)
        is None
    )


def test_pre_v5_returns_none(monkeypatch):
    _patch_env(monkeypatch, tpu_version=4)
    assert (
        get_tuned_block_sizes_mla("decode", jnp.bfloat16, jnp.bfloat16, 8, 512, 64, 256, 128)
        is None
    )


def test_warned_misses_is_one_shot_per_key(monkeypatch, caplog):
    """Each unique miss-key logs INFO at most once."""
    _patch_env(monkeypatch)
    import logging

    # Bind caplog to the module's logger explicitly — relying on root-level
    # propagation breaks if anyone in the stack disables it.
    caplog.set_level(logging.INFO, logger=tuned_block_sizes.logger.name)

    # Trigger 3 misses, only the first should log.
    get_tuned_block_sizes_mla("decode", jnp.bfloat16, jnp.bfloat16, 8, 512, 64, 256, 128)
    get_tuned_block_sizes_mla("decode", jnp.bfloat16, jnp.bfloat16, 8, 512, 64, 256, 128)
    get_tuned_block_sizes_mla("decode", jnp.bfloat16, jnp.bfloat16, 8, 512, 64, 256, 128)
    miss_records = [r for r in caplog.records if "MLA tuned-block-size LOOKUP MISS" in r.message]
    assert len(miss_records) == 1, f"expected 1 miss log, got {len(miss_records)}"

    # Different key → another log
    get_tuned_block_sizes_mla("mixed", jnp.bfloat16, jnp.bfloat16, 8, 512, 64, 256, 512)
    miss_records = [r for r in caplog.records if "MLA tuned-block-size LOOKUP MISS" in r.message]
    assert len(miss_records) == 2, f"expected 2 miss logs, got {len(miss_records)}"
