from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.speculative.draft_extend_fused import (
    GreedyVerifyPostprocessOutput,
    _greedy_verify_postprocess_jit,
)


def test_greedy_verify_postprocess_safe_index_matches_host_logic():
    logits = jnp.arange(8 * 3, dtype=jnp.float32).reshape(8, 3)
    hidden = jnp.arange(8 * 5, dtype=jnp.float32).reshape(8, 5)
    positions = jnp.arange(8, dtype=jnp.int32) + 100
    seq_lens = jnp.array([10, 20], dtype=jnp.int32)
    accept_index = jnp.array([0, 1, -1, -1, 4, 5, 6, -1], dtype=jnp.int32)
    accept_length = jnp.array([2, 3], dtype=jnp.int32)
    verified_id = jnp.array([11, 12, 0, 0, 21, 22, 23, 0], dtype=jnp.int32)

    out = _greedy_verify_postprocess_jit(
        logits,
        hidden,
        positions,
        seq_lens,
        accept_index,
        accept_length,
        verified_id,
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
    )

    safe_index = np.array([0, 1, 3, 3, 4, 5, 6, 7], dtype=np.int32)
    np.testing.assert_array_equal(np.asarray(out.next_token_logits), np.asarray(logits)[safe_index])
    np.testing.assert_array_equal(np.asarray(out.hidden_states), np.asarray(hidden)[safe_index])
    np.testing.assert_array_equal(np.asarray(out.positions), np.asarray(positions)[safe_index])
    np.testing.assert_array_equal(np.asarray(out.new_seq_lens), np.array([12, 23], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(out.select_index), np.array([1, 6], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(out.verified_id), np.asarray(verified_id))
    np.testing.assert_array_equal(np.asarray(out.accept_lens), np.asarray(accept_length))


class _SamplingInfo:
    is_all_greedy = True


class _Batch:
    sampling_info = _SamplingInfo()
    speculative_eagle_topk = 1
    speculative_num_steps = 3
    speculative_num_draft_tokens = 4

    def __init__(self, bs):
        self.seq_lens = np.ones((bs,), dtype=np.int32)


def test_fused_greedy_decode_predicate_accepts_only_fixed_bucket():
    from sgl_jax.srt.speculative.base_worker import _can_use_fused_greedy_decode_step3

    assert _can_use_fused_greedy_decode_step3(_Batch(32))
    assert not _can_use_fused_greedy_decode_step3(_Batch(16))

    batch = _Batch(32)
    batch.sampling_info = _SamplingInfo()
    batch.sampling_info.is_all_greedy = False
    assert not _can_use_fused_greedy_decode_step3(batch)

    batch = _Batch(32)
    batch.speculative_num_steps = 2
    assert not _can_use_fused_greedy_decode_step3(batch)


def test_multi_layer_draft_worker_routes_fixed_greedy_path(monkeypatch):
    from sgl_jax.srt.speculative import draft_extend_fused
    from sgl_jax.srt.speculative.multi_layer_draft_worker import MultiLayerDraftWorker

    calls = []

    def fake_step3(worker, model_worker_batch, batch_output):
        calls.append("step3")

    def fake_fallback(worker, model_worker_batch, batch_output):
        calls.append("fallback")

    monkeypatch.setattr(draft_extend_fused, "draft_extend_for_decode_fused_step3", fake_step3)
    monkeypatch.setattr(draft_extend_fused, "draft_extend_for_decode_fused", fake_fallback)

    worker = object.__new__(MultiLayerDraftWorker)
    batch = type("Batch", (), {"use_fused_greedy_decode_step3": True})()
    worker.draft_extend_for_decode(batch, object())

    assert calls == ["step3"]


def test_step3_entrypoint_applies_postprocess_before_fallback(monkeypatch):
    from sgl_jax.srt.speculative import draft_extend_fused

    post = GreedyVerifyPostprocessOutput(
        next_token_logits=jnp.ones((8, 3), dtype=jnp.float32),
        hidden_states=jnp.ones((8, 5), dtype=jnp.float32) * 2,
        positions=jnp.arange(8, dtype=jnp.int32) + 200,
        new_seq_lens=jnp.array([12, 23], dtype=jnp.int32),
        select_index=jnp.array([1, 6], dtype=jnp.int32),
        verified_id=jnp.arange(8, dtype=jnp.int32) + 10,
        accept_lens=jnp.array([2, 3], dtype=jnp.int32),
    )
    calls = []

    def fake_postprocess(*args, **kwargs):
        calls.append(("postprocess", kwargs))
        return post

    def fake_fallback(worker, model_worker_batch, batch_output):
        calls.append(("fallback", None))
        assert batch_output.logits_output.next_token_logits is post.next_token_logits
        assert batch_output.logits_output.hidden_states is post.hidden_states
        assert batch_output.next_draft_input.hidden_states is post.hidden_states
        assert batch_output.next_draft_input.verified_id is post.verified_id
        assert batch_output.accept_lens is post.accept_lens
        assert model_worker_batch.positions is post.positions

    monkeypatch.setattr(draft_extend_fused, "_greedy_verify_postprocess_jit", fake_postprocess)
    monkeypatch.setattr(draft_extend_fused, "draft_extend_for_decode_fused", fake_fallback)

    draft_worker = SimpleNamespace(speculative_num_steps=3, speculative_num_draft_tokens=4)
    model_worker_batch = SimpleNamespace(
        positions=jnp.arange(8, dtype=jnp.int32),
        seq_lens=jnp.array([10, 20], dtype=jnp.int32),
    )
    batch_output = SimpleNamespace(
        logits_output=SimpleNamespace(
            next_token_logits=jnp.zeros((8, 3), dtype=jnp.float32),
            hidden_states=jnp.zeros((8, 5), dtype=jnp.float32),
        ),
        next_draft_input=SimpleNamespace(
            accept_index=jnp.array([0, 1, -1, -1, 4, 5, 6, -1], dtype=jnp.int32),
            verified_id=jnp.array([11, 12, 0, 0, 21, 22, 23, 0], dtype=jnp.int32),
            new_seq_lens=None,
        ),
        accept_lens=jnp.array([2, 3], dtype=jnp.int32),
    )

    draft_extend_fused.draft_extend_for_decode_fused_step3(
        draft_worker, model_worker_batch, batch_output
    )

    assert calls == [
        (
            "postprocess",
            {"speculative_num_steps": 3, "speculative_num_draft_tokens": 4},
        ),
        ("fallback", None),
    ]
    assert batch_output.next_draft_input.new_seq_lens is post.new_seq_lens
