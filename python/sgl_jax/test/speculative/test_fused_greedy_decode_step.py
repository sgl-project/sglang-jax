import inspect
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.speculative.draft_extend_fused import (
    _greedy_sample_and_prepare_draft_inputs,
    _greedy_step3_prepare_draft_inputs,
    _replicate_for_host_output,
)


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


def test_greedy_sample_and_prepare_draft_inputs_calls_verify_inside(monkeypatch):
    from sgl_jax.srt.speculative import draft_extend_fused

    calls = []

    def fake_verify_tree_greedy(**kwargs):
        calls.append(kwargs)
        return (
            jnp.array([[0, 1, -1, -1], [4, 5, 6, -1]], dtype=jnp.int32),
            jnp.array([2, 3], dtype=jnp.int32),
            jnp.array(
                [
                    [11, 12, 0, 0, 99],
                    [21, 22, 23, 0, 88],
                ],
                dtype=jnp.int32,
            ),
        )

    monkeypatch.setattr(draft_extend_fused, "verify_tree_greedy", fake_verify_tree_greedy)

    hidden = jnp.arange(8 * 5, dtype=jnp.float32).reshape(8, 5)
    positions = jnp.arange(8, dtype=jnp.int32) + 100
    seq_lens = jnp.array([10, 20], dtype=jnp.int32)
    draft_tokens = jnp.array([11, 12, 13, 14, 21, 22, 23, 24], dtype=jnp.int32)
    retrive_index = jnp.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=jnp.int32)
    retrive_next_token = jnp.array([[1, 2, 3, -1], [1, 2, 3, -1]], dtype=jnp.int32)
    retrive_next_sibling = jnp.full((2, 4), -1, dtype=jnp.int32)
    next_token_logits = jnp.zeros((8, 32), dtype=jnp.float32)

    prepared = _greedy_sample_and_prepare_draft_inputs(
        target_hidden=hidden,
        positions=positions,
        seq_lens=seq_lens,
        draft_tokens=draft_tokens,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        next_token_logits=next_token_logits,
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
    )

    assert len(calls) == 1
    assert calls[0]["draft_tokens"] is draft_tokens
    assert calls[0]["next_token_logits"] is next_token_logits
    np.testing.assert_array_equal(
        np.asarray(prepared.accept_lens),
        np.array([3, 4], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(prepared.select_index),
        np.array([2, 7], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(prepared.verified_id),
        np.array([11, 12, 0, 0, 99, 21, 22, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(prepared.hidden_states),
        np.asarray(hidden)[np.array([0, 1, 3, 3, 4, 5, 6, 7], dtype=np.int32)],
    )


def test_greedy_chain_sample_and_prepare_from_predict_matches_fixed_chain_semantics():
    from sgl_jax.srt.speculative.draft_extend_fused import (
        _greedy_sample_and_prepare_draft_inputs_chain_from_predict,
    )

    hidden = jnp.arange(8 * 5, dtype=jnp.float32).reshape(8, 5)
    positions = jnp.arange(8, dtype=jnp.int32) + 100
    seq_lens = jnp.array([10, 20], dtype=jnp.int32)
    draft_tokens = jnp.array([10, 11, 12, 13, 20, 21, 22, 23], dtype=jnp.int32)
    target_predict = jnp.array([11, 12, 99, 0, 99, 0, 0, 0], dtype=jnp.int32)

    prepared = _greedy_sample_and_prepare_draft_inputs_chain_from_predict(
        target_hidden=hidden,
        positions=positions,
        seq_lens=seq_lens,
        draft_tokens=draft_tokens,
        target_predict=target_predict,
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
    )

    np.testing.assert_array_equal(
        np.asarray(prepared.accept_lens),
        np.array([3, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(prepared.select_index),
        np.array([2, 4], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(prepared.verified_id),
        np.array([11, 12, 99, 0, 99, 0, 0, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(prepared.predict[:8]),
        np.array([11, 12, 99, 0, 99, 0, 0, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(prepared.hidden_states),
        np.asarray(hidden)[np.array([0, 1, 2, 3, 4, 7, 7, 7], dtype=np.int32)],
    )


def test_step3_logits_metadata_can_skip_placeholder_accept_lens(monkeypatch):
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
    from sgl_jax.srt.speculative import draft_extend_fused

    devices = np.asarray(jax.devices())
    mesh = Mesh(devices[:1], ("data",))
    placeholder_accept_lens = jnp.array([1, 1], dtype=jnp.int32)
    converted = []

    def fake_device_array_preserve_device(value, sharding):
        converted.append(value)
        return value

    monkeypatch.setattr(
        draft_extend_fused,
        "_device_array_preserve_device",
        fake_device_array_preserve_device,
    )

    batch = SimpleNamespace(
        forward_mode=ForwardMode.DRAFT_EXTEND,
        capture_hidden_mode=None,
        extend_seq_lens=jnp.array([4, 4], dtype=jnp.int32),
        logits_indices=jnp.array([3, 7], dtype=jnp.int32),
        spec_info_padded=SimpleNamespace(accept_length=placeholder_accept_lens),
        input_embedding=None,
        deepstack_visual_embedding=None,
        lora_scalings=None,
        extend_input_logprob_token_ids=None,
    )

    metadata = draft_extend_fused._logits_metadata_from_model_worker_batch_preserve_device(
        batch,
        mesh,
        include_accept_lens=False,
    )

    assert metadata.accept_lens is None
    assert all(value is not placeholder_accept_lens for value in converted)


def test_fused_greedy_materialize_keeps_scheduler_d2h_in_one_boundary():
    from sgl_jax.srt.speculative.draft_extend_fused import (
        _materialize_fused_greedy_batch_output_for_scheduler,
    )

    bs = 2
    width = 4
    hidden_size = 3
    total_bs = 3
    batch_output = SimpleNamespace(
        logits_output=None,
        next_draft_input=SimpleNamespace(),
        allocate_lens=np.array([4, 4], dtype=np.int32),
        accept_lens=None,
        next_token_ids=None,
    )
    predict = jnp.array(
        [
            [10, 11, 12, 13, 14],
            [20, 21, 22, 23, 24],
            [30, 31, 32, 33, 34],
        ],
        dtype=jnp.int32,
    )
    topk_index_stacked = jnp.arange(total_bs * 3, dtype=jnp.int32).reshape(total_bs, 3, 1)

    out = _materialize_fused_greedy_batch_output_for_scheduler(
        batch_output=batch_output,
        selector=np.array([2, 0], dtype=np.int32),
        real_bs=bs,
        layer0_hidden=jnp.ones((total_bs, hidden_size), dtype=jnp.float32),
        topk_index_stacked=topk_index_stacked,
        accept_lens_device=jnp.array([3, 4, 2], dtype=jnp.int32),
        new_seq_lens_device=jnp.array([13, 24, 35], dtype=jnp.int32),
        predict_device=predict,
        target_logits=jnp.ones((total_bs * width, 8), dtype=jnp.float32),
        target_hidden=jnp.ones((total_bs * width, hidden_size), dtype=jnp.float32),
    )

    assert isinstance(out.next_token_ids, np.ndarray)
    np.testing.assert_array_equal(out.next_token_ids, np.asarray(predict))
    assert isinstance(out.accept_lens, np.ndarray)
    np.testing.assert_array_equal(out.accept_lens, np.array([3, 4, 2], dtype=np.int32))
    np.testing.assert_array_equal(
        out.next_draft_input.verified_id, np.array([31, 12], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        out.next_draft_input.topk_index, np.asarray(topk_index_stacked)[[2, 0]]
    )
    np.testing.assert_array_equal(
        out.next_draft_input.topk_p,
        np.ones(np.asarray(topk_index_stacked)[[2, 0]].shape, dtype=np.float32),
    )


def test_fused_greedy_verify_round_bypasses_split_verify_and_step3(monkeypatch):
    from sgl_jax.srt.speculative import draft_extend_fused

    devices = np.asarray(jax.devices())
    mesh = Mesh(devices[:1], ("data",))
    calls = []

    class _SpecInfo:
        draft_token = jnp.arange(8, dtype=jnp.int32)
        retrive_index = jnp.arange(8, dtype=jnp.int32).reshape(2, 4)
        retrive_next_token = jnp.full((2, 4), -1, dtype=jnp.int32)
        retrive_next_sibling = jnp.full((2, 4), -1, dtype=jnp.int32)

        def prepare_for_verify(self, model_worker_batch, page_size, target_worker):
            calls.append(("prepare_for_verify", page_size))

    class _AttnBackend:
        def get_eagle_forward_metadata(self, batch):
            calls.append("target_metadata")
            return "target-forward-metadata"

    class _MemoryPools:
        def replace_all(self, updates):
            calls.append(("replace", updates))

    target_mr = SimpleNamespace(
        attn_backend=_AttnBackend(),
        memory_pools=_MemoryPools(),
        model_state_leaves=("target-leaf",),
        _model_def="target-def",
        _model_state_def="target-state-def",
    )
    target_worker = SimpleNamespace(
        model_runner=target_mr,
        model_config=SimpleNamespace(hidden_size=3, dtype=jnp.bfloat16, vocab_size=8),
        forward_batch_generation=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("must not call split target forward")
        ),
    )
    draft_mr = SimpleNamespace(
        attn_backend=SimpleNamespace(forward_metadata="draft-forward-metadata"),
        memory_pools=_MemoryPools(),
        model_state_leaves=("draft-leaf",),
        _model_def="draft-def",
        _model_state_def="draft-state-def",
    )
    draft_worker = SimpleNamespace(
        _workers=[SimpleNamespace(model_runner=draft_mr)],
        draft_model_runner=draft_mr,
        mesh=mesh,
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        topk=1,
        draft_extend_for_decode=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("must not call split draft_extend")
        ),
    )
    spec_worker = SimpleNamespace(
        target_worker=target_worker,
        draft_worker=draft_worker,
        mesh=mesh,
        page_size=64,
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        verify=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("must not call split verify")
        ),
    )
    batch = SimpleNamespace(
        spec_info_padded=_SpecInfo(),
        seq_lens=np.array([10, 20], dtype=np.int32),
        logits_indices_selector=np.array([0, 1], dtype=np.int32),
        real_bs=2,
        bid=jnp.array([0], dtype=jnp.int32),
        input_ids=jnp.arange(8, dtype=jnp.int32),
    )

    target_fb = SimpleNamespace(bid=None)
    draft_fb = SimpleNamespace(bid=None)

    def fake_forward_batch_init_new_preserve_device(batch_arg, model_runner):
        return target_fb if model_runner is target_mr else draft_fb

    def fake_prepare_verify_placeholders(draft_worker_arg, batch_arg):
        calls.append("prepare_topk1_verify_placeholders")
        return (
            jnp.array([101, 201], dtype=jnp.int32),
            jnp.array([[102, 103, 104], [202, 203, 204]], dtype=jnp.int32),
        )

    def fake_prepare_step3(draft_worker_arg, batch_arg, batch_output):
        calls.append("prepare_step3")
        return batch_arg, "draft-logits-metadata"

    def fake_fused_jit(*args, **kwargs):
        calls.append(("fused_jit", args[0], args[7]))
        return (
            jnp.ones((2, 3), dtype=jnp.float32),
            jnp.ones((2, 3, 1), dtype=jnp.int32),
            "target-pool-updates",
            ("draft-pool-updates",),
            jnp.array([3, 3], dtype=jnp.int32),
            jnp.array([13, 23], dtype=jnp.int32),
            jnp.ones((2, 5), dtype=jnp.int32),
            jnp.ones((8, 8), dtype=jnp.float32),
            jnp.ones((8, 3), dtype=jnp.float32),
        )

    def fake_materialize(**kwargs):
        calls.append("materialize_batch_output")
        batch_output = kwargs["batch_output"]
        batch_output.accept_lens = np.asarray(kwargs["accept_lens_device"])
        batch_output.next_draft_input.new_seq_lens = np.asarray(kwargs["new_seq_lens_device"])[
            kwargs["selector"]
        ]
        return batch_output

    monkeypatch.setattr(
        draft_extend_fused,
        "_forward_batch_init_new_preserve_device",
        fake_forward_batch_init_new_preserve_device,
    )
    monkeypatch.setattr(
        draft_extend_fused,
        "_prepare_topk1_verify_placeholders_from_draft_state",
        fake_prepare_verify_placeholders,
    )
    monkeypatch.setattr(
        draft_extend_fused,
        "_logits_metadata_from_model_worker_batch_preserve_device",
        lambda batch_arg, mesh_arg, **kwargs: "target-logits-metadata",
    )
    monkeypatch.setattr(
        draft_extend_fused,
        "_prepare_step3_model_worker_batch_for_draft_extend",
        fake_prepare_step3,
    )
    monkeypatch.setattr(
        draft_extend_fused,
        "_build_fused_greedy_verify_step3_jit",
        lambda num_layers, topk: fake_fused_jit,
        raising=False,
    )
    monkeypatch.setattr(
        draft_extend_fused,
        "_materialize_fused_greedy_batch_output_for_scheduler",
        fake_materialize,
        raising=False,
    )
    monkeypatch.setattr(
        draft_extend_fused.jnp,
        "zeros",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("fused host placeholder must use np.zeros")
        ),
    )
    monkeypatch.setattr(
        draft_extend_fused.jnp,
        "ones_like",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("fused host placeholder accept_lens must use np.ones")
        ),
    )

    out = draft_extend_fused.fused_greedy_verify_and_draft_extend_for_decode(
        spec_worker,
        batch,
        np.array([10, 20], dtype=np.int32),
    )

    assert "prepare_topk1_verify_placeholders" in calls
    assert ("prepare_for_verify", 64) in calls
    assert "prepare_step3" in calls
    assert any(call[0] == "fused_jit" for call in calls if isinstance(call, tuple))
    assert "materialize_batch_output" in calls
    np.testing.assert_array_equal(np.asarray(out.accept_lens), np.array([3, 3]))
    np.testing.assert_array_equal(out.next_draft_input.new_seq_lens, np.array([13, 23]))


def test_fused_greedy_verify_round_builds_topk1_verify_inputs_inside_jit(monkeypatch):
    from sgl_jax.srt.speculative import draft_extend_fused

    devices = np.asarray(jax.devices())
    mesh = Mesh(devices[:1], ("data",))
    calls = []
    previous_verified_id = np.array([101, 201], dtype=np.int32)
    previous_topk_index = np.array([[[102], [103], [104]], [[202], [203], [204]]], dtype=np.int32)

    class _SpecInfo:
        verified_id = previous_verified_id
        topk_index = previous_topk_index

        def prepare_for_verify(self, model_worker_batch, page_size, target_worker):
            calls.append(("prepare_for_verify", page_size))

    class _ForwardMode:
        def is_idle(self):
            return False

    class _AttnBackend:
        def get_eagle_forward_metadata(self, batch):
            calls.append("target_metadata")
            return "target-forward-metadata"

    class _MemoryPools:
        def replace_all(self, updates):
            calls.append(("replace", updates))

    target_mr = SimpleNamespace(
        attn_backend=_AttnBackend(),
        memory_pools=_MemoryPools(),
        model_state_leaves=("target-leaf",),
        _model_def="target-def",
        _model_state_def="target-state-def",
    )
    target_worker = SimpleNamespace(
        model_runner=target_mr,
        model_config=SimpleNamespace(hidden_size=3, dtype=jnp.bfloat16, vocab_size=8),
    )
    draft_mr = SimpleNamespace(
        attn_backend=SimpleNamespace(forward_metadata="draft-forward-metadata"),
        memory_pools=_MemoryPools(),
        model_state_leaves=("draft-leaf",),
        _model_def="draft-def",
        _model_state_def="draft-state-def",
    )

    def padding_for_decode(batch_arg):
        calls.append("padding_for_decode")

    draft_worker = SimpleNamespace(
        _workers=[SimpleNamespace(model_runner=draft_mr)],
        draft_model_runner=draft_mr,
        mesh=mesh,
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        topk=1,
        padding_for_decode=padding_for_decode,
    )
    spec_worker = SimpleNamespace(
        target_worker=target_worker,
        draft_worker=draft_worker,
        mesh=mesh,
        page_size=64,
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
    )
    batch = SimpleNamespace(
        spec_info_padded=_SpecInfo(),
        seq_lens=np.array([10, 20], dtype=np.int32),
        seq_lens_sum=np.array(30, dtype=np.int32),
        logits_indices_selector=np.array([0, 1], dtype=np.int32),
        real_bs=2,
        bid=jnp.array([0], dtype=jnp.int32),
        input_ids=jnp.arange(8, dtype=jnp.int32),
        forward_mode=_ForwardMode(),
    )
    target_fb = SimpleNamespace(bid=None)
    draft_fb = SimpleNamespace(bid=None)

    def fake_forward_batch_init_new_preserve_device(batch_arg, model_runner):
        return target_fb if model_runner is target_mr else draft_fb

    def fake_prepare_step3(draft_worker_arg, batch_arg, batch_output):
        calls.append("prepare_step3")
        return batch_arg, "draft-logits-metadata"

    def fake_fused_jit(*args, **kwargs):
        calls.append(("fused_jit", args[-2], args[-1]))
        np.testing.assert_array_equal(np.asarray(args[-2]), previous_verified_id)
        np.testing.assert_array_equal(np.asarray(args[-1]), previous_topk_index[:, :, 0])
        return (
            jnp.ones((2, 3), dtype=jnp.float32),
            jnp.ones((2, 3, 1), dtype=jnp.int32),
            "target-pool-updates",
            ("draft-pool-updates",),
            jnp.array([3, 3], dtype=jnp.int32),
            jnp.array([13, 23], dtype=jnp.int32),
            jnp.ones((2, 5), dtype=jnp.int32),
            jnp.ones((8, 8), dtype=jnp.float32),
            jnp.ones((8, 3), dtype=jnp.float32),
        )

    def fake_materialize(**kwargs):
        calls.append("materialize_batch_output")
        batch_output = kwargs["batch_output"]
        batch_output.accept_lens = np.asarray(kwargs["accept_lens_device"])
        return batch_output

    assert not hasattr(draft_extend_fused, "_prepare_topk1_verify_inputs_from_draft_state")
    monkeypatch.setattr(
        draft_extend_fused,
        "_forward_batch_init_new_preserve_device",
        fake_forward_batch_init_new_preserve_device,
    )
    monkeypatch.setattr(
        draft_extend_fused,
        "_logits_metadata_from_model_worker_batch_preserve_device",
        lambda batch_arg, mesh_arg, **kwargs: "target-logits-metadata",
    )
    monkeypatch.setattr(
        draft_extend_fused,
        "_prepare_step3_model_worker_batch_for_draft_extend",
        fake_prepare_step3,
    )
    monkeypatch.setattr(
        draft_extend_fused,
        "_build_fused_greedy_verify_step3_jit",
        lambda num_layers, topk: fake_fused_jit,
        raising=False,
    )
    monkeypatch.setattr(
        draft_extend_fused,
        "_materialize_fused_greedy_batch_output_for_scheduler",
        fake_materialize,
        raising=False,
    )

    out = draft_extend_fused.fused_greedy_verify_and_draft_extend_for_decode(
        spec_worker,
        batch,
        np.array([10, 20], dtype=np.int32),
    )

    assert "padding_for_decode" in calls
    assert any(call[0] == "fused_jit" for call in calls if isinstance(call, tuple))
    np.testing.assert_array_equal(np.asarray(out.accept_lens), np.array([3, 3]))


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


def test_fused_greedy_jit_does_not_reshard_model_outputs_before_compute():
    from sgl_jax.srt.speculative import draft_extend_fused

    source = inspect.getsource(draft_extend_fused._build_fused_greedy_verify_step3_jit)

    assert "jax.sharding.reshard(target_output.next_token_logits" not in source
    assert "jax.sharding.reshard(target_output.hidden_states" not in source
    assert "jax.sharding.reshard(output.next_token_logits" not in source
    assert "jax.sharding.reshard(output.hidden_states" not in source


def test_fused_greedy_jit_only_returns_large_target_outputs_when_requested():
    from sgl_jax.srt.speculative import draft_extend_fused

    source = inspect.getsource(draft_extend_fused._build_fused_greedy_verify_step3_jit)

    assert "return_target_logits" in source
    assert "return_target_hidden" in source
    assert "if return_target_logits:" in source
    assert "if return_target_hidden:" in source


def test_topk1_index_helper_does_not_reshard_per_layer_outputs():
    from sgl_jax.srt.speculative import draft_extend_fused

    source = inspect.getsource(draft_extend_fused._topk1_index_from_logits)

    assert "jax.sharding.reshard" not in source


def test_fused_greedy_jit_does_not_return_topk1_probs_to_host():
    from sgl_jax.srt.speculative import draft_extend_fused

    jit_source = inspect.getsource(draft_extend_fused._build_fused_greedy_verify_step3_jit)
    materialize_source = inspect.getsource(
        draft_extend_fused._materialize_fused_greedy_batch_output_for_scheduler
    )

    assert "all_topk_p" not in jit_source
    assert "stacked_p" not in jit_source
    assert "topk_p_stacked" not in materialize_source
    assert "np.ones(topk_index.shape" in materialize_source


def test_fused_greedy_jit_does_not_return_selected_verified_id_to_host():
    from sgl_jax.srt.speculative import draft_extend_fused

    jit_source = inspect.getsource(draft_extend_fused._build_fused_greedy_verify_step3_jit)
    materialize_source = inspect.getsource(
        draft_extend_fused._materialize_fused_greedy_batch_output_for_scheduler
    )

    assert "selected_verified_id" not in jit_source
    assert "selected_verified_id_device" not in materialize_source
    assert "predict[selector, accept_lens[selector] - 1]" in materialize_source


def test_fused_greedy_jit_does_not_replicate_target_predict_before_verify():
    from sgl_jax.srt.speculative import draft_extend_fused

    source = inspect.getsource(draft_extend_fused._build_fused_greedy_verify_step3_jit)

    assert "target_predict = jax.sharding.reshard" not in source
    assert "verify_tree_greedy_pallas_call" not in source


def test_fused_greedy_module_does_not_import_pallas_verify_for_fixed_chain():
    from sgl_jax.srt.speculative import draft_extend_fused

    source = inspect.getsource(draft_extend_fused)

    assert "verify_tree_greedy_pallas_call" not in source


def test_fused_greedy_selects_next_draft_rows_before_host_materialize():
    from sgl_jax.srt.speculative import draft_extend_fused

    jit_source = inspect.getsource(draft_extend_fused._build_fused_greedy_verify_step3_jit)
    materialize_source = inspect.getsource(
        draft_extend_fused._materialize_fused_greedy_batch_output_for_scheduler
    )

    assert "selected_layer0_hidden" in jit_source
    assert "selected_verified_id" not in jit_source
    assert "select_index_device" not in materialize_source
    assert "np.asarray(layer0_hidden)[select_index]" not in materialize_source
    assert "np.asarray(verified_id_device)[select_index]" not in materialize_source


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


def test_fused_greedy_decode_predicate_uses_worker_config_before_padding():
    from sgl_jax.srt.speculative.base_worker import _can_use_fused_greedy_decode_step3

    batch = SimpleNamespace(
        sampling_info=_SamplingInfo(),
        seq_lens=np.ones((32,), dtype=np.int32),
        speculative_eagle_topk=0,
        speculative_num_steps=0,
        speculative_num_draft_tokens=0,
    )
    draft_worker = SimpleNamespace(
        topk=1,
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
    )

    assert _can_use_fused_greedy_decode_step3(batch, draft_worker)

    batch.speculative_num_steps = 2
    assert not _can_use_fused_greedy_decode_step3(batch, draft_worker)


def test_multi_layer_draft_worker_rejects_direct_fixed_greedy_step3_route(monkeypatch):
    from sgl_jax.srt.speculative import draft_extend_fused
    from sgl_jax.srt.speculative.multi_layer_draft_worker import MultiLayerDraftWorker

    calls = []

    def fake_fallback(worker, model_worker_batch, batch_output):
        calls.append("fallback")

    monkeypatch.setattr(draft_extend_fused, "draft_extend_for_decode_fused", fake_fallback)

    worker = object.__new__(MultiLayerDraftWorker)
    batch = type("Batch", (), {"use_fused_greedy_decode_step3": True})()

    with pytest.raises(RuntimeError, match="Fixed greedy decode must use"):
        worker.draft_extend_for_decode(batch, object())

    assert calls == []


def test_base_spec_worker_routes_fixed_greedy_decode_to_fused_round(monkeypatch):
    from sgl_jax.srt.speculative import base_worker, draft_extend_fused
    from sgl_jax.srt.speculative.base_worker import BaseSpecWorker

    calls = []

    class _ForwardMode:
        def is_extend(self):
            return False

    class _DraftWorker:
        def draft(self, model_worker_batch):
            raise AssertionError("fixed greedy route should not call standalone draft")

        def draft_extend_for_decode(self, model_worker_batch, batch_output):
            raise AssertionError("fixed greedy route should not call separate draft_extend")

    def fail_verify(self, model_worker_batch, cur_allocate_lens):
        raise AssertionError("fixed greedy route should not call separate verify")

    def fake_fused_round(spec_worker, model_worker_batch, cur_allocate_lens):
        calls.append(("fused_round", tuple(cur_allocate_lens.tolist())))
        return "batch-output"

    monkeypatch.setattr(
        base_worker, "_can_use_fused_greedy_decode_step3", lambda batch, draft_worker: True
    )
    monkeypatch.setattr(BaseSpecWorker, "verify", fail_verify)
    monkeypatch.setattr(
        draft_extend_fused,
        "fused_greedy_verify_and_draft_extend_for_decode",
        fake_fused_round,
        raising=False,
    )

    worker = object.__new__(BaseSpecWorker)
    worker._draft_worker = _DraftWorker()
    batch = SimpleNamespace(
        forward_mode=_ForwardMode(),
        logits_indices_selector=np.array([0, 2], dtype=np.int32),
        spec_info_padded=SimpleNamespace(
            allocate_lens=np.array([10, 11, 12, 13], dtype=np.int32),
        ),
    )

    out = worker.forward_batch_speculative_generation(batch)

    assert out == "batch-output"
    assert calls == [("fused_round", (10, 12))]
    assert batch.use_fused_greedy_decode_step3 is True


def test_base_spec_worker_verify_rejects_direct_fixed_greedy_route():
    from sgl_jax.srt.speculative.base_worker import BaseSpecWorker

    worker = object.__new__(BaseSpecWorker)
    batch = SimpleNamespace(use_fused_greedy_decode_step3=True)

    with pytest.raises(RuntimeError, match="must enter the whole-round fused path"):
        worker.verify(batch, np.array([1], dtype=np.int32))


def test_prepare_topk1_verify_placeholders_normalizes_host_ids_without_jnp_zeros(
    monkeypatch,
):
    from sgl_jax.srt.speculative import draft_extend_fused
    from sgl_jax.srt.speculative.draft_extend_fused import (
        _prepare_topk1_verify_placeholders_from_draft_state,
    )
    from sgl_jax.srt.speculative.eagle_util import EagleVerifyInput

    calls = []

    def padding_for_decode(batch):
        calls.append("padding_for_decode")

    def fail_jnp_zeros(*args, **kwargs):
        raise AssertionError("host placeholders must not launch device zeros")

    monkeypatch.setattr(draft_extend_fused.jnp, "zeros", fail_jnp_zeros)

    draft_worker = SimpleNamespace(
        padding_for_decode=padding_for_decode,
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        topk=1,
    )
    batch = SimpleNamespace(
        seq_lens=np.array([8, 12], dtype=np.int32),
        seq_lens_sum=np.array(20, dtype=np.int32),
        spec_info_padded=SimpleNamespace(
            verified_id=np.array([101, 201], dtype=np.int64),
            topk_index=np.array([[[102], [103], [104]], [[202], [203], [204]]], dtype=np.int64),
        ),
    )

    previous_verified_id, previous_token_list = _prepare_topk1_verify_placeholders_from_draft_state(
        draft_worker, batch
    )

    assert calls == ["padding_for_decode"]
    assert isinstance(batch.spec_info_padded, EagleVerifyInput)
    assert previous_verified_id.dtype == np.int32
    assert previous_token_list.dtype == np.int32
    np.testing.assert_array_equal(previous_verified_id, np.array([101, 201], dtype=np.int32))
    np.testing.assert_array_equal(
        previous_token_list,
        np.array([[102, 103, 104], [202, 203, 204]], dtype=np.int32),
    )
    assert batch.spec_info_padded.draft_token.dtype == np.int32
    assert batch.spec_info_padded.draft_token.shape == (8,)


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


def test_eagle_draft_topk1_uses_host_chain_builder_for_host_inputs(monkeypatch):
    from sgl_jax.srt.speculative import eagle_draft_worker
    from sgl_jax.srt.speculative.eagle_draft_worker import EagleDraftWorker

    packed = np.array(
        [
            [101, 102, 103, 104, 201, 202, 203, 204],
            [7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, -1, 1, 2, 3, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=np.int32,
    )
    calls = []

    def fake_host_builder(verified_id, token_list, seq_lens, num_verify_tokens, batch_size):
        calls.append((verified_id, token_list, seq_lens, num_verify_tokens, batch_size))
        return packed

    def fail_device_builder(*args, **kwargs):
        raise AssertionError("host arrays should not launch the device chain builder")

    monkeypatch.setattr(eagle_draft_worker, "build_chain_verify_inputs", fake_host_builder)
    monkeypatch.setattr(eagle_draft_worker, "build_chain_verify_inputs_device", fail_device_builder)

    worker = SimpleNamespace(
        topk=1,
        speculative_num_draft_tokens=4,
        speculative_num_steps=3,
        mesh=None,
        padding_for_decode=lambda batch: None,
        draft_forward=lambda batch: (
            None,
            np.array([[102, 103, 104], [202, 203, 204]], dtype=np.int32),
            None,
        ),
    )
    batch = SimpleNamespace(
        seq_lens=np.array([8, 12], dtype=np.int32),
        seq_lens_sum=np.array(20, dtype=np.int32),
        spec_info_padded=SimpleNamespace(verified_id=np.array([101, 201], dtype=np.int32)),
    )

    EagleDraftWorker.draft(worker, batch)

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][2], np.array([7, 11], dtype=np.int32))
    np.testing.assert_array_equal(batch.spec_info_padded.draft_token, packed[0])
    np.testing.assert_array_equal(
        batch.spec_info_padded.retrive_index,
        np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32),
    )


def test_eagle_draft_topk1_keeps_device_chain_builder_for_device_inputs(monkeypatch):
    from sgl_jax.srt.speculative import eagle_draft_worker
    from sgl_jax.srt.speculative.eagle_draft_worker import EagleDraftWorker

    packed = jnp.array(
        [
            [101, 102, 103, 104, 201, 202, 203, 204],
            [7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, -1, 1, 2, 3, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=jnp.int32,
    )
    calls = []

    def fail_host_builder(*args, **kwargs):
        raise AssertionError("device arrays should keep the device chain builder available")

    def fake_device_builder(verified_id, token_list, seq_lens, num_verify_tokens, batch_size):
        calls.append((verified_id, token_list, seq_lens, num_verify_tokens, batch_size))
        return packed

    monkeypatch.setattr(eagle_draft_worker, "build_chain_verify_inputs", fail_host_builder)
    monkeypatch.setattr(eagle_draft_worker, "build_chain_verify_inputs_device", fake_device_builder)

    worker = SimpleNamespace(
        topk=1,
        speculative_num_draft_tokens=4,
        speculative_num_steps=3,
        mesh=Mesh(np.asarray(jax.devices())[:1], ("data",)),
        padding_for_decode=lambda batch: None,
        draft_forward=lambda batch: (
            None,
            jnp.array([[102, 103, 104], [202, 203, 204]], dtype=jnp.int32),
            None,
        ),
    )
    batch = SimpleNamespace(
        seq_lens=jnp.array([8, 12], dtype=jnp.int32),
        seq_lens_sum=jnp.array(20, dtype=jnp.int32),
        spec_info_padded=SimpleNamespace(verified_id=jnp.array([101, 201], dtype=jnp.int32)),
    )

    EagleDraftWorker.draft(worker, batch)

    assert len(calls) == 1
    np.testing.assert_array_equal(np.asarray(calls[0][2]), np.array([7, 11], dtype=np.int32))
    assert isinstance(batch.spec_info_padded.draft_token, jax.Array)


def test_obsolete_step3_entrypoint_is_removed():
    from sgl_jax.srt.speculative import draft_extend_fused

    assert not hasattr(draft_extend_fused, "draft_extend_for_decode_fused_step3")
    assert not hasattr(draft_extend_fused, "_draft_extend_for_decode_fused_step3_impl")
    assert not hasattr(draft_extend_fused, "_build_fused_greedy_step3_draft_extend_jit")
