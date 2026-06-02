from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.speculative.draft_extend_fused import (
    _greedy_step3_prepare_draft_inputs,
    _greedy_verify_postprocess_jit,
    _replicate_for_host_output,
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


def test_greedy_step3_prepare_draft_inputs_matches_safe_index_logic():
    hidden = jnp.arange(8 * 5, dtype=jnp.float32).reshape(8, 5)
    positions = jnp.arange(8, dtype=jnp.int32) + 100
    seq_lens = jnp.array([10, 20], dtype=jnp.int32)
    accept_index = jnp.array([0, 1, -1, -1, 4, 5, 6, -1], dtype=jnp.int32)
    accept_length = jnp.array([2, 3], dtype=jnp.int32)
    verified_id = jnp.array([11, 12, 0, 0, 21, 22, 23, 0], dtype=jnp.int32)

    out = _greedy_step3_prepare_draft_inputs(
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
    np.testing.assert_array_equal(np.asarray(out.hidden_states), np.asarray(hidden)[safe_index])
    np.testing.assert_array_equal(np.asarray(out.positions), np.asarray(positions)[safe_index])
    np.testing.assert_array_equal(np.asarray(out.new_seq_lens), np.array([12, 23], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(out.select_index), np.array([1, 6], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(out.verified_id), np.asarray(verified_id))
    np.testing.assert_array_equal(np.asarray(out.accept_lens), np.asarray(accept_length))
    np.testing.assert_array_equal(np.asarray(out.sel_pos), np.array([1, 2], dtype=np.int32))


def test_greedy_step3_prepare_draft_inputs_preserves_position_data_sharding():
    devices = np.asarray(jax.devices())
    if devices.size < 4:
        pytest.skip("requires at least 4 JAX devices for a 2x2 mesh")

    mesh = Mesh(
        devices[:4].reshape(2, 2),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    data_sharding = NamedSharding(mesh, P("data"))
    replicated_sharding = NamedSharding(mesh, P())
    hidden = jax.device_put(jnp.arange(8 * 5, dtype=jnp.float32).reshape(8, 5), replicated_sharding)
    positions = jax.device_put(jnp.arange(8, dtype=jnp.int32) + 100, data_sharding)
    seq_lens = jax.device_put(jnp.array([10, 20], dtype=jnp.int32), data_sharding)
    accept_index = jax.device_put(
        jnp.array([0, 1, -1, -1, 4, 5, 6, -1], dtype=jnp.int32),
        replicated_sharding,
    )
    accept_length = jax.device_put(jnp.array([2, 3], dtype=jnp.int32), data_sharding)
    verified_id = jax.device_put(
        jnp.array([11, 12, 0, 0, 21, 22, 23, 0], dtype=jnp.int32),
        replicated_sharding,
    )

    @jax.jit
    def prepare(hidden, positions, seq_lens, accept_index, accept_length, verified_id):
        return _greedy_step3_prepare_draft_inputs(
            hidden,
            positions,
            seq_lens,
            accept_index,
            accept_length,
            verified_id,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
        )

    with jax.set_mesh(mesh):
        out = prepare(hidden, positions, seq_lens, accept_index, accept_length, verified_id)

    assert out.positions.sharding == data_sharding


def test_replicate_for_host_output_reshards_data_sharded_array():
    devices = np.asarray(jax.devices())
    if devices.size < 4:
        pytest.skip("requires at least 4 JAX devices for a 2x2 mesh")

    mesh = Mesh(
        devices[:4].reshape(2, 2),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    data_sharding = NamedSharding(mesh, P("data"))
    replicated_sharding = NamedSharding(mesh, P())
    value = jax.device_put(jnp.array([12, 23], dtype=jnp.int32), data_sharding)

    @jax.jit
    def replicate(value):
        return _replicate_for_host_output(value, replicated_sharding)

    with jax.set_mesh(mesh):
        out = replicate(value)

    assert out.sharding.is_fully_replicated
    np.testing.assert_array_equal(np.asarray(out), np.array([12, 23], dtype=np.int32))


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


def test_multi_layer_draft_forward_topk1_uses_linear_token_list_for_host_inputs(monkeypatch):
    from sgl_jax.srt.speculative import multi_layer_draft_worker
    from sgl_jax.srt.speculative.multi_layer_draft_worker import MultiLayerDraftWorker

    def fail_select_top_k_tokens(*args, **kwargs):
        raise AssertionError("topk=1 should bypass generic tree-list construction")

    monkeypatch.setattr(multi_layer_draft_worker, "select_top_k_tokens", fail_select_top_k_tokens)

    worker = object.__new__(MultiLayerDraftWorker)
    worker.topk = 1
    worker.speculative_num_steps = 3
    batch = SimpleNamespace(
        seq_lens=np.array([5, 7], dtype=np.int32),
        spec_info_padded=SimpleNamespace(
            topk_p=np.ones((2, 3, 1), dtype=np.float32),
            topk_index=np.array([[[11], [12], [13]], [[21], [22], [23]]], dtype=np.int32),
            hidden_states=np.ones((2, 4), dtype=np.float32),
        ),
    )

    score_list, token_list, parents_list = worker.draft_forward(batch)

    assert score_list is None
    assert parents_list is None
    assert isinstance(token_list, np.ndarray)
    np.testing.assert_array_equal(
        token_list,
        np.array([[11, 12, 13], [21, 22, 23]], dtype=np.int32),
    )


def test_multi_layer_draft_forward_topk1_preserves_device_inputs(monkeypatch):
    from sgl_jax.srt.speculative import multi_layer_draft_worker
    from sgl_jax.srt.speculative.multi_layer_draft_worker import MultiLayerDraftWorker

    def fail_select_top_k_tokens(*args, **kwargs):
        raise AssertionError("topk=1 should bypass generic tree-list construction")

    monkeypatch.setattr(multi_layer_draft_worker, "select_top_k_tokens", fail_select_top_k_tokens)

    worker = object.__new__(MultiLayerDraftWorker)
    worker.topk = 1
    worker.speculative_num_steps = 3
    batch = SimpleNamespace(
        seq_lens=jnp.array([5, 7], dtype=jnp.int32),
        spec_info_padded=SimpleNamespace(
            topk_p=jnp.ones((2, 3, 1), dtype=jnp.float32),
            topk_index=jnp.array([[[11], [12], [13]], [[21], [22], [23]]], dtype=jnp.int32),
            hidden_states=jnp.ones((2, 4), dtype=jnp.float32),
        ),
    )

    score_list, token_list, parents_list = worker.draft_forward(batch)

    assert score_list is None
    assert parents_list is None
    assert isinstance(token_list, jax.Array)
    np.testing.assert_array_equal(
        np.asarray(token_list),
        np.array([[11, 12, 13], [21, 22, 23]], dtype=np.int32),
    )


def test_step3_entrypoint_does_not_split_postprocess_before_fused_extend(monkeypatch):
    from sgl_jax.srt.speculative import draft_extend_fused

    calls = []

    def fail_if_split_postprocess(*args, **kwargs):
        raise AssertionError("Step3 must fuse verify postprocess into draft-extend JIT")

    def fake_fused_step3_impl(worker, model_worker_batch, batch_output):
        calls.append(("fused_step3_impl", None))
        assert batch_output.next_draft_input.safe_index is safe_index
        assert batch_output.next_draft_input.accept_index is accept_index
        assert batch_output.accept_lens is accept_lens

    monkeypatch.setattr(
        draft_extend_fused, "_greedy_verify_postprocess_jit", fail_if_split_postprocess
    )
    monkeypatch.setattr(
        draft_extend_fused,
        "_draft_extend_for_decode_fused_step3_impl",
        fake_fused_step3_impl,
        raising=False,
    )

    draft_worker = SimpleNamespace(speculative_num_steps=3, speculative_num_draft_tokens=4)
    accept_index = jnp.array([0, 1, -1, -1, 4, 5, 6, -1], dtype=jnp.int32)
    safe_index = jnp.array([0, 1, 3, 3, 4, 5, 6, 7], dtype=jnp.int32)
    accept_lens = jnp.array([2, 3], dtype=jnp.int32)
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
            accept_index=accept_index,
            safe_index=safe_index,
            verified_id=jnp.array([11, 12, 0, 0, 21, 22, 23, 0], dtype=jnp.int32),
            new_seq_lens=None,
        ),
        accept_lens=accept_lens,
    )

    draft_extend_fused.draft_extend_for_decode_fused_step3(
        draft_worker, model_worker_batch, batch_output
    )

    assert calls == [("fused_step3_impl", None)]
