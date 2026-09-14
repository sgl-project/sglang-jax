"""Unit coverage for Frozen-KV's dedicated device-first verify handoff."""

from __future__ import annotations

import types
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.speculative.eagle_info import EagleVerifyInput
from sgl_jax.srt.speculative.frozen_kv_mtp_worker import (
    FrozenKvMtpDraftWorker,
    FrozenKvMtpWorker,
    _build_frozen_kv_fused_draft_extend,
    _frozen_kv_verify_and_publish,
    _select_frozen_kv_proposal_start,
)
from sgl_jax.srt.speculative.relay_buffer import SpecSeedRelayBuffers


def test_frozen_kv_precompiles_target_prefill_seed_publication():
    """Startup warms seed publication without restoring assistant prefill."""
    worker = FrozenKvMtpWorker.__new__(FrozenKvMtpWorker)
    assert worker.supports_non_fused_spec_prefill_precompile() is True


def test_fused_verify_publishes_selected_target_seed_in_same_jit():
    """Acceptance, row selection, and relay scatter share one executable."""
    draft_tokens = jnp.asarray([7, 10, 11, 12, 8, 30, 31, 32], dtype=jnp.int32)
    target_top1 = jnp.asarray([10, 11, 99, 13, 88, 30, 31, 32], dtype=jnp.int32)
    logits = jnp.full((8, 128), -10.0, dtype=jnp.float32)
    logits = logits.at[jnp.arange(8), target_top1].set(10.0)
    hidden = jnp.arange(16, dtype=jnp.float32).reshape(8, 2)
    positions = jnp.arange(8, dtype=jnp.int32)
    capacity = 8
    buffers = SpecSeedRelayBuffers(
        token_ids=jnp.zeros((1, capacity), dtype=jnp.int32),
        draft_token_ids=jnp.zeros((1, capacity), dtype=jnp.int32),
        hidden_states=jnp.zeros((1, capacity, 2), dtype=jnp.float32),
        is_target_seed=jnp.zeros((1, capacity), dtype=bool),
    )

    run = jax.jit(
        partial(
            _frozen_kv_verify_and_publish,
            draft_token_num=4,
            dp_size=1,
        )
    )
    _, _, _, predict, accept_lens, updated = run(
        draft_tokens,
        logits,
        hidden,
        positions,
        buffers,
        jnp.asarray([2, 5], dtype=jnp.int32),
        jnp.asarray([True, True]),
    )

    np.testing.assert_array_equal(np.asarray(accept_lens), [3, 1])
    np.testing.assert_array_equal(np.asarray(updated.token_ids)[0, [2, 5]], [99, 88])
    np.testing.assert_array_equal(np.asarray(updated.draft_token_ids)[0, [2, 5]], [99, 88])
    np.testing.assert_array_equal(np.asarray(updated.hidden_states)[0, [2, 5]], [[4, 5], [8, 9]])
    np.testing.assert_array_equal(np.asarray(updated.is_target_seed)[0, [2, 5]], [True, True])
    np.testing.assert_array_equal(np.asarray(predict), target_top1)


def test_fused_draft_selects_seed_output_only_for_live_rows():
    """Inactive padded rows retain relay values while live rows consume seeds."""
    relay_token = jnp.asarray([10, 20, 30], dtype=jnp.int32)
    relay_hidden = jnp.asarray([[1, 2], [3, 4], [5, 6]], dtype=jnp.float32)
    seed_logits = jnp.full((3, 64), -10.0, dtype=jnp.float32)
    seed_logits = seed_logits.at[jnp.arange(3), jnp.asarray([11, 22, 33])].set(10.0)
    seed_hidden = relay_hidden + 100

    run = jax.jit(_select_frozen_kv_proposal_start)
    token, hidden = run(
        relay_token,
        relay_hidden,
        jnp.asarray([True, False, True]),
        seed_logits,
        seed_hidden,
    )

    np.testing.assert_array_equal(np.asarray(token), [11, 20, 33])
    np.testing.assert_array_equal(np.asarray(hidden), [[101, 102], [3, 4], [105, 106]])


def test_fused_draft_builder_has_stable_profile_name():
    """The TPU trace exposes the reviewer-requested top-level dispatch name."""
    assert _build_frozen_kv_fused_draft_extend(3).__name__ == "draft_extend_fused"


def test_relay_padding_preserves_dp_segments():
    """Generic bucket padding must not move rank-one rows into rank zero."""
    value = np.asarray([10, 11, 20, 21], dtype=np.int32)
    padded = FrozenKvMtpDraftWorker._pad_relay_rows_for_bucket(
        value,
        padded_bs=8,
        dp_size=2,
        fill_value=-1,
    )
    np.testing.assert_array_equal(padded, [10, 11, -1, -1, 20, 21, -1, -1])


def test_decode_precompile_uses_the_same_relay_descriptor_as_runtime():
    """Startup must compile the fused route instead of its legacy fallback."""
    calls = []

    class _Draft:
        def _publish_seed_relay(self, **kwargs):
            calls.append(kwargs)

        def new_draft_input(self, **kwargs):
            return kwargs

    worker = FrozenKvMtpWorker.__new__(FrozenKvMtpWorker)
    worker._draft_worker = _Draft()
    batch = types.SimpleNamespace(
        logits_indices_selector=np.asarray([0, 2], dtype=np.int32),
        req_pool_indices=np.asarray([4, -1, 7], dtype=np.int32),
        seq_lens=np.asarray([10, 0, 20], dtype=np.int32),
    )
    spec_info = types.SimpleNamespace(
        verified_id=np.asarray([100, 0, 200], dtype=np.int32),
        topk_index=np.asarray([[101], [0], [201]], dtype=np.int32),
        hidden_states=np.asarray([[1, 2], [0, 0], [3, 4]], dtype=np.float32),
        allocate_lens=np.asarray([16, 0, 32], dtype=np.int32),
    )

    descriptor = worker.prepare_spec_decode_precompile_state(batch, spec_info)

    np.testing.assert_array_equal(calls[0]["verified_id"], [100, 200])
    np.testing.assert_array_equal(calls[0]["draft_token_ids"], [101, 201])
    np.testing.assert_array_equal(descriptor["future_indices"], [4, 7])
    np.testing.assert_array_equal(descriptor["allocate_lens"], [16, 32])
    np.testing.assert_array_equal(descriptor["new_seq_lens"], [10, 20])
    np.testing.assert_array_equal(descriptor["relay_seed_mask"], [True, True])


class _FakeDraftWorker:
    seed_relay_buffers = object()


class _FakeVerifyInput(EagleVerifyInput):
    draft_token = jnp.arange(8, dtype=jnp.int32)
    custom_mask = jnp.ones((1,), dtype=bool)
    positions = jnp.arange(8, dtype=jnp.int32)
    spec_steps = 3
    draft_token_num = 4

    def sample_device(self, model_worker_batch, logits_output, rng, mesh):
        del model_worker_batch, logits_output, rng, mesh
        # Two padded slots, four candidate rows per slot. Only slot zero is live.
        return (
            jnp.arange(8, dtype=jnp.int32),
            jnp.arange(100, 108, dtype=jnp.int32),
            jnp.array([3, 0], dtype=jnp.int32),
            jnp.array([[0, 1, 2, -1], [-1, -1, -1, -1]], dtype=jnp.int32),
        )


def test_dedicated_verify_rejects_non_greedy_fallback():
    """Unsupported sampling must never re-enter generic EAGLE verification."""
    draft = _FakeDraftWorker()
    verify_input = _FakeVerifyInput.__new__(_FakeVerifyInput)
    model_worker_batch = types.SimpleNamespace(
        spec_info_padded=verify_input,
        seq_lens=np.array([10, 0], dtype=np.int32),
        positions=jnp.arange(8, dtype=jnp.int32),
        logits_indices_selector=np.array([0], dtype=np.int32),
        req_pool_indices=np.array([17, -1], dtype=np.int32),
        sampling_info=types.SimpleNamespace(is_all_greedy=False),
        bid=3,
    )
    target_worker = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(
            attn_backend=types.SimpleNamespace(get_eagle_forward_metadata=lambda _mwb: "metadata")
        ),
        forward_batch_generation=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unsupported verify must fail before target forward")
        ),
    )
    worker = FrozenKvMtpWorker.__new__(FrozenKvMtpWorker)
    worker.server_args = types.SimpleNamespace(disable_overlap_schedule=True)
    worker._target_worker = target_worker
    worker._draft_worker = draft
    worker.mesh = object()
    worker.page_size = 128
    worker.speculative_num_steps = 3
    worker.speculative_num_draft_tokens = 4

    with pytest.raises(RuntimeError, match="fused greedy linear verify path"):
        worker.verify(model_worker_batch, jnp.array([20, 0], dtype=jnp.int32))


def test_dedicated_verify_rejects_missing_fused_runner_contract():
    """A fake/partial runner must not silently select the removed fallback."""

    class _NativeChainVerifyInput(_FakeVerifyInput):
        custom_mask = None
        draft_token_num = 4
        spec_steps = 3
        draft_token = jnp.asarray([7, 10, 11, 12, 8, 30, 31, 32], dtype=jnp.int32)

    draft = _FakeDraftWorker()
    verify_input = _NativeChainVerifyInput.__new__(_NativeChainVerifyInput)
    model_worker_batch = types.SimpleNamespace(
        spec_info_padded=verify_input,
        seq_lens=np.array([10, 20], dtype=np.int32),
        positions=jnp.arange(8, dtype=jnp.int32),
        logits_indices_selector=np.array([0, 1], dtype=np.int32),
        req_pool_indices=np.array([17, 19], dtype=np.int32),
        sampling_info=types.SimpleNamespace(is_all_greedy=True),
        bid=3,
    )
    target_worker = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(
            attn_backend=types.SimpleNamespace(get_eagle_forward_metadata=lambda _mwb: "metadata")
        ),
        forward_batch_generation=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("missing fused contract must fail before target forward")
        ),
    )
    worker = FrozenKvMtpWorker.__new__(FrozenKvMtpWorker)
    worker.server_args = types.SimpleNamespace(disable_overlap_schedule=True)
    worker._target_worker = target_worker
    worker._draft_worker = draft
    worker.mesh = object()
    worker.page_size = 128
    worker.speculative_num_steps = 3
    worker.speculative_num_draft_tokens = 4

    with pytest.raises(RuntimeError, match="fused greedy linear verify path"):
        worker.verify(model_worker_batch, jnp.array([20, 30], dtype=jnp.int32))
