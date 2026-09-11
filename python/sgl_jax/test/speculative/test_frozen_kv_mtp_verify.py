"""Unit coverage for Frozen-KV's dedicated device-first verify handoff."""

from __future__ import annotations

import types

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.speculative.eagle_info import EagleVerifyInput
from sgl_jax.srt.speculative.frozen_kv_mtp_worker import (
    FrozenKvMtpDraftWorker,
    FrozenKvMtpWorker,
)


class _FakeNextDraftInput:
    def _validate_non_overlap_state(self):
        return None


class _FakeDraftWorker:
    _compact_request_rows = staticmethod(FrozenKvMtpDraftWorker._compact_request_rows)

    def __init__(self, calls):
        self.calls = calls
        self.draft_model_runner = types.SimpleNamespace(rngs=object())

    def publish_seed_after_verify_device(self, **kwargs):
        self.calls.append(("publish", kwargs))

    def new_draft_input(self, **kwargs):
        self.calls.append(("descriptor", kwargs))
        return _FakeNextDraftInput()


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


def test_dedicated_verify_publishes_device_seed_before_host_descriptor(monkeypatch):
    """The Frozen path must not use generic EAGLE's host sample handoff."""
    import sgl_jax.srt.speculative.frozen_kv_mtp_worker as frozen_module

    # The real operation is an explicit layout conversion. CPU unit coverage
    # needs only its data-flow contract, not a physical TPU mesh.
    monkeypatch.setattr(frozen_module, "replicate_to_mesh", lambda _mesh, *xs: xs)

    calls = []
    draft = _FakeDraftWorker(calls)
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
    logits_output = types.SimpleNamespace(
        next_token_logits=jnp.arange(8 * 3, dtype=jnp.float32).reshape(8, 3),
        hidden_states=jnp.arange(8 * 2, dtype=jnp.float32).reshape(8, 2),
    )
    target_worker = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(
            attn_backend=types.SimpleNamespace(get_eagle_forward_metadata=lambda _mwb: "metadata")
        ),
        forward_batch_generation=lambda _mwb, **_kwargs: (logits_output, None, 0),
    )
    worker = FrozenKvMtpWorker.__new__(FrozenKvMtpWorker)
    worker.server_args = types.SimpleNamespace(disable_overlap_schedule=True)
    worker._target_worker = target_worker
    worker._draft_worker = draft
    worker.mesh = object()
    worker.page_size = 128
    worker.speculative_num_steps = 3
    worker.speculative_num_draft_tokens = 4

    result = worker.verify(model_worker_batch, jnp.array([20, 0], dtype=jnp.int32))

    assert [name for name, _ in calls] == ["publish", "descriptor"]
    published = calls[0][1]
    assert isinstance(published["accept_lengths"], type(jnp.array(0)))
    np.testing.assert_array_equal(published["accept_lengths"], np.array([3, 0]))
    descriptor = calls[1][1]
    np.testing.assert_array_equal(descriptor["future_indices"], np.array([17], dtype=np.int32))
    np.testing.assert_array_equal(descriptor["allocate_lens"], np.array([20], dtype=np.int32))
    np.testing.assert_array_equal(descriptor["new_seq_lens"], np.array([13], dtype=np.int32))
    np.testing.assert_array_equal(descriptor["accept_length_cpu"], np.array([3], dtype=np.int32))
    np.testing.assert_array_equal(descriptor["relay_seed_mask"], np.array([True]))
    np.testing.assert_array_equal(result.accept_lens, np.array([3, 0], dtype=np.int32))
    np.testing.assert_array_equal(result.next_token_ids, np.arange(8, dtype=np.int32))


def test_dedicated_verify_uses_native_chain_logits_without_replication(monkeypatch):
    """Frozen top-k-one verify must follow DFlash's native-sharding pattern."""
    import sgl_jax.srt.speculative.frozen_kv_mtp_worker as frozen_module

    def fail_if_replicated(*_args, **_kwargs):
        raise AssertionError("native Frozen-KV chain verify must not replicate target output")

    monkeypatch.setattr(frozen_module, "replicate_to_mesh", fail_if_replicated)

    class _NativeChainVerifyInput(_FakeVerifyInput):
        custom_mask = None
        draft_token_num = 4
        spec_steps = 3
        draft_token = jnp.asarray([7, 10, 11, 12, 8, 30, 31, 32], dtype=jnp.int32)

        def sample_device(self, *_args, **_kwargs):
            raise AssertionError("native Frozen-KV chain must not enter generic tree sampling")

    calls = []
    draft = _FakeDraftWorker(calls)
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
    target_top1 = jnp.asarray([10, 11, 99, 13, 88, 30, 31, 32], dtype=jnp.int32)
    logits = jnp.full((8, 128), -10.0, dtype=jnp.float32)
    logits = logits.at[jnp.arange(8), target_top1].set(10.0)
    logits_output = types.SimpleNamespace(
        next_token_logits=logits,
        hidden_states=jnp.arange(16, dtype=jnp.float32).reshape(8, 2),
    )
    target_hidden_before_scheduler_gather = logits_output.hidden_states
    target_worker = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(
            attn_backend=types.SimpleNamespace(get_eagle_forward_metadata=lambda _mwb: "metadata")
        ),
        forward_batch_generation=lambda _mwb, **_kwargs: (logits_output, None, 0),
    )
    worker = FrozenKvMtpWorker.__new__(FrozenKvMtpWorker)
    worker.server_args = types.SimpleNamespace(disable_overlap_schedule=True)
    worker._target_worker = target_worker
    worker._draft_worker = draft
    worker.mesh = object()
    worker.page_size = 128
    worker.speculative_num_steps = 3
    worker.speculative_num_draft_tokens = 4

    result = worker.verify(model_worker_batch, jnp.array([20, 30], dtype=jnp.int32))

    assert [name for name, _ in calls] == ["publish", "descriptor"]
    published = calls[0][1]
    np.testing.assert_array_equal(published["verified_tokens"], target_top1)
    np.testing.assert_array_equal(published["accept_lengths"], [3, 1])
    # Target hidden rows are still the original target-forward object; no
    # host-orchestrated P() replica is introduced before seed selection.
    assert published["target_hidden"] is target_hidden_before_scheduler_gather
    np.testing.assert_array_equal(result.next_token_ids, target_top1)
    np.testing.assert_array_equal(result.accept_lens, [3, 1])
