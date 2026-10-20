"""dsa_dense_arm_flags: the host-side dense-arm rule shared by ForwardBatch.init_new and the fused-prefill
target batch (exact-dense arm): pure EXTEND only, per-sequence max_seq <= DSA_PREFILL_DENSE_EXACT_MAX_CONTEXT
(default 2048), bound clamped to the model's index_topk when given."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from sgl_jax.srt.model_executor.forward_batch_info import (
    ForwardMode,
    dsa_dense_arm_flags,
)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("DSA_PREFILL_DENSE_EXACT_MAX_CONTEXT", "2048")
    return monkeypatch


def test_single_sequence_1k_takes_exact(env):
    assert dsa_dense_arm_flags(ForwardMode.EXTEND, np.array([1024]), 1024) is True


def test_verify_and_draft_extend_never_dense(env):
    for mode in (ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND, ForwardMode.DECODE):
        assert dsa_dense_arm_flags(mode, np.array([1024]), 1024) is False


def test_sequence_over_bound_is_sparse(env):
    assert dsa_dense_arm_flags(ForwardMode.EXTEND, np.array([1024, 4096]), 5120) is False


def test_disabled_when_bound_zero(env):
    env.setenv("DSA_PREFILL_DENSE_EXACT_MAX_CONTEXT", "0")
    assert dsa_dense_arm_flags(ForwardMode.EXTEND, np.array([1024]), 1024) is False


def test_device_seq_lens_never_synced(env):
    # verify / draft-extend batches carry device arrays; the helper must not read them (per-step D2H sync)
    import jax.numpy as jnp

    dev = jnp.array([1024], dtype=jnp.int32)
    assert dsa_dense_arm_flags(ForwardMode.EXTEND, dev, 1024) is False
    assert dsa_dense_arm_flags(ForwardMode.TARGET_VERIFY, dev, 4) is False


def test_default_bound_without_env(monkeypatch):
    # registered default (environ.py): 2048
    monkeypatch.delenv("DSA_PREFILL_DENSE_EXACT_MAX_CONTEXT", raising=False)
    assert dsa_dense_arm_flags(ForwardMode.EXTEND, np.array([2048]), 2048) is True
    assert dsa_dense_arm_flags(ForwardMode.EXTEND, np.array([2049]), 2049) is False


def test_bound_clamped_to_index_topk(env):
    # index_topk below the env bound: sequences longer than index_topk stay sparse (same rule as the backend)
    assert dsa_dense_arm_flags(ForwardMode.EXTEND, np.array([1024]), 1024, index_topk=512) is False
    assert dsa_dense_arm_flags(ForwardMode.EXTEND, np.array([512]), 512, index_topk=512) is True
    assert dsa_dense_arm_flags(ForwardMode.EXTEND, np.array([1024]), 1024, index_topk=4096) is True
