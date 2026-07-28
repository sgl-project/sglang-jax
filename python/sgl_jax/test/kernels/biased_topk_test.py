"""Equivalence tests for the sort-free biased top-k Pallas kernel."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _reference(router_logits, correction_bias, *, topk):
    scores_for_choice = router_logits.astype(jnp.float32) + correction_bias.astype(jnp.float32)
    topk_ids = jax.lax.top_k(scores_for_choice, topk)[1]
    topk_weights = jnp.take_along_axis(router_logits.astype(jnp.float32), topk_ids, axis=1)
    return topk_weights, topk_ids


def test_mimo_shape_matches_reference():
    from sgl_jax.srt.kernels.biased_topk import biased_topk_pallas

    tokens, experts, topk = 256, 384, 8
    logits = jax.nn.sigmoid(
        jax.random.normal(jax.random.key(0), (tokens, experts), dtype=jnp.float32)
    )
    bias = jax.random.normal(jax.random.key(1), (experts,), dtype=jnp.float32) * 0.1

    expected_weights, expected_ids = _reference(logits, bias, topk=topk)
    actual_weights, actual_ids = biased_topk_pallas(
        logits,
        bias,
        topk=topk,
        block_tokens=256,
        interpret=True,
    )

    np.testing.assert_array_equal(np.asarray(actual_ids), np.asarray(expected_ids))
    np.testing.assert_allclose(
        np.asarray(actual_weights),
        np.asarray(expected_weights),
        rtol=0,
        atol=1e-6,
    )


def test_auto_block_tokens_rejects_unsafe_large_token_shape():
    from sgl_jax.srt.kernels.biased_topk import biased_topk_pallas

    tokens, experts = 2050, 128
    logits = jnp.zeros((tokens, experts), dtype=jnp.float32)
    bias = jnp.zeros((experts,), dtype=jnp.float32)

    with pytest.raises(ValueError, match="no VMEM-safe block_tokens"):
        biased_topk_pallas(
            logits,
            bias,
            topk=8,
            block_tokens="auto",
            interpret=True,
        )


@pytest.mark.parametrize("experts", [128, 256, 384, 512])
@pytest.mark.parametrize("topk", [1, 2, 8, 16])
def test_generic_static_shapes_match_reference(experts, topk):
    from sgl_jax.srt.kernels.biased_topk import biased_topk_pallas

    tokens = 128
    logits = jax.nn.sigmoid(
        jax.random.normal(jax.random.key(experts + topk), (tokens, experts), dtype=jnp.float32)
    )
    bias = jax.random.normal(jax.random.key(topk), (experts,), dtype=jnp.float32) * 0.25

    expected_weights, expected_ids = _reference(logits, bias, topk=topk)
    actual_weights, actual_ids = biased_topk_pallas(
        logits,
        bias,
        topk=topk,
        block_tokens=tokens,
        interpret=True,
    )

    assert actual_weights.shape == (tokens, topk)
    assert actual_ids.shape == (tokens, topk)
    assert actual_weights.dtype == jnp.float32
    assert actual_ids.dtype == jnp.int32
    np.testing.assert_array_equal(np.asarray(actual_ids), np.asarray(expected_ids))
    np.testing.assert_allclose(
        np.asarray(actual_weights),
        np.asarray(expected_weights),
        rtol=0,
        atol=1e-6,
    )


def test_flat_ties_choose_lowest_expert_ids():
    from sgl_jax.srt.kernels.biased_topk import biased_topk_pallas

    tokens, experts, topk = 64, 384, 8
    logits = jnp.full((tokens, experts), 0.5, dtype=jnp.float32)
    bias = jnp.zeros((experts,), dtype=jnp.float32)

    actual_weights, actual_ids = biased_topk_pallas(
        logits,
        bias,
        topk=topk,
        block_tokens=tokens,
        interpret=True,
    )

    np.testing.assert_array_equal(
        np.asarray(actual_ids),
        np.tile(np.arange(topk, dtype=np.int32), (tokens, 1)),
    )
    np.testing.assert_array_equal(
        np.asarray(actual_weights),
        np.full((tokens, topk), 0.5, dtype=np.float32),
    )


def test_post_bias_tie_returns_pre_bias_weights_in_stable_order():
    from sgl_jax.srt.kernels.biased_topk import biased_topk_pallas

    tokens, experts = 64, 128
    logits = jnp.zeros((tokens, experts), dtype=jnp.float32)
    logits = logits.at[:, 3].set(0.4)
    logits = logits.at[:, 5].set(0.3)
    bias = jnp.full((experts,), -1.0, dtype=jnp.float32)
    bias = bias.at[3].set(0.1)
    bias = bias.at[5].set(0.2)

    actual_weights, actual_ids = biased_topk_pallas(
        logits,
        bias,
        topk=2,
        block_tokens=tokens,
        interpret=True,
    )

    np.testing.assert_array_equal(
        np.asarray(actual_ids),
        np.tile(np.array([3, 5], dtype=np.int32), (tokens, 1)),
    )
    np.testing.assert_allclose(
        np.asarray(actual_weights),
        np.tile(np.array([0.4, 0.3], dtype=np.float32), (tokens, 1)),
        rtol=0,
        atol=1e-6,
    )


@pytest.mark.parametrize("bias_mode", ["zero", "negative"])
def test_zero_and_negative_bias_match_reference(bias_mode):
    from sgl_jax.srt.kernels.biased_topk import biased_topk_pallas

    tokens, experts, topk = 256, 384, 8
    logits = jax.nn.sigmoid(
        jax.random.normal(jax.random.key(21), (tokens, experts), dtype=jnp.float32)
    )
    if bias_mode == "zero":
        bias = jnp.zeros((experts,), dtype=jnp.float32)
    else:
        bias = -jax.random.uniform(
            jax.random.key(22),
            (experts,),
            dtype=jnp.float32,
        )

    expected_weights, expected_ids = _reference(logits, bias, topk=topk)
    actual_weights, actual_ids = biased_topk_pallas(
        logits,
        bias,
        topk=topk,
        block_tokens=128,
        interpret=True,
    )

    np.testing.assert_array_equal(np.asarray(actual_ids), np.asarray(expected_ids))
    np.testing.assert_allclose(
        np.asarray(actual_weights),
        np.asarray(expected_weights),
        rtol=0,
        atol=1e-6,
    )


def test_topk_shared_flag_dispatches_plain_biased_routing(monkeypatch):
    from sgl_jax.srt.eplb.expert_location import (
        get_global_server_args,
        set_global_server_args,
    )
    from sgl_jax.srt.layers import gate

    tokens, experts, topk = 128, 128, 8
    logits = jax.nn.sigmoid(
        jax.random.normal(jax.random.key(7), (tokens, experts), dtype=jnp.float32)
    )
    bias = jax.random.normal(jax.random.key(8), (experts,), dtype=jnp.float32) * 0.1
    expected_weights, expected_ids = _reference(logits, bias, topk=topk)
    called = False

    def fake_biased_topk(router_logits, correction_bias, *, topk, **_):
        nonlocal called
        called = True
        return _reference(router_logits, correction_bias, topk=topk)

    monkeypatch.setattr(gate, "biased_topk_pallas", fake_biased_topk)
    previous_server_args = get_global_server_args()
    set_global_server_args(SimpleNamespace(enable_grouped_topk_kernel=True, device="tpu"))
    try:
        actual_weights, actual_ids = gate.TopK(
            topk=topk,
            renormalize=False,
        )(logits, correction_bias=bias)
    finally:
        set_global_server_args(previous_server_args)

    assert called
    np.testing.assert_array_equal(np.asarray(actual_ids), np.asarray(expected_ids))
    np.testing.assert_allclose(
        np.asarray(actual_weights),
        np.asarray(expected_weights),
        rtol=0,
        atol=0,
    )


def test_topk_shared_flag_dispatches_grouped_biased_routing(monkeypatch):
    from sgl_jax.srt.eplb.expert_location import (
        get_global_server_args,
        set_global_server_args,
    )
    from sgl_jax.srt.layers import gate

    tokens, experts, topk = 128, 128, 8
    logits = jax.nn.sigmoid(
        jax.random.normal(jax.random.key(9), (tokens, experts), dtype=jnp.float32)
    )
    bias = jax.random.normal(jax.random.key(10), (experts,), dtype=jnp.float32) * 0.1
    module = gate.TopK(
        topk=topk,
        renormalize=False,
        num_expert_group=8,
        topk_group=4,
    )
    expected_weights, expected_ids = module._biased_grouped_topk_jax(logits, bias)
    called = False

    def fake_grouped_topk(router_logits, correction_bias, **_):
        nonlocal called
        called = True
        return module._biased_grouped_topk_jax(router_logits, correction_bias)

    monkeypatch.setattr(gate, "grouped_topk_pallas", fake_grouped_topk)
    previous_server_args = get_global_server_args()
    set_global_server_args(SimpleNamespace(enable_grouped_topk_kernel=True, device="tpu"))
    try:
        actual_weights, actual_ids = module(logits, correction_bias=bias)
    finally:
        set_global_server_args(previous_server_args)

    assert called
    np.testing.assert_array_equal(np.asarray(actual_ids), np.asarray(expected_ids))
    np.testing.assert_allclose(
        np.asarray(actual_weights),
        np.asarray(expected_weights),
        rtol=0,
        atol=0,
    )


def test_topk_flag_off_keeps_jax_path(monkeypatch):
    from sgl_jax.srt.eplb.expert_location import (
        get_global_server_args,
        set_global_server_args,
    )
    from sgl_jax.srt.layers import gate

    logits = jnp.arange(128, dtype=jnp.float32)[None, :]
    bias = jnp.zeros((128,), dtype=jnp.float32)

    def fail_if_called(*_, **__):
        raise AssertionError("Pallas kernel must not run when the flag is disabled")

    monkeypatch.setattr(gate, "biased_topk_pallas", fail_if_called)
    previous_server_args = get_global_server_args()
    set_global_server_args(SimpleNamespace(enable_grouped_topk_kernel=False, device="tpu"))
    try:
        actual_weights, actual_ids = gate.TopK(
            topk=8,
            renormalize=False,
        )(logits, correction_bias=bias)
    finally:
        set_global_server_args(previous_server_args)

    expected_weights, expected_ids = _reference(logits, bias, topk=8)
    np.testing.assert_array_equal(np.asarray(actual_ids), np.asarray(expected_ids))
    np.testing.assert_array_equal(np.asarray(actual_weights), np.asarray(expected_weights))


def test_topk_kernel_path_preserves_normalization_and_scaling(monkeypatch):
    from sgl_jax.srt.eplb.expert_location import (
        get_global_server_args,
        set_global_server_args,
    )
    from sgl_jax.srt.layers.gate import TopK

    monkeypatch.setenv("PALLAS_INTERPRET", "1")
    tokens, experts, topk = 64, 128, 8
    logits = jax.nn.sigmoid(
        jax.random.normal(jax.random.key(11), (tokens, experts), dtype=jnp.float32)
    )
    bias = jax.random.normal(jax.random.key(12), (experts,), dtype=jnp.float32) * 0.1
    raw_weights, expected_ids = _reference(logits, bias, topk=topk)
    expected_weights = raw_weights / raw_weights.sum(axis=-1, keepdims=True) * 1.75

    previous_server_args = get_global_server_args()
    set_global_server_args(SimpleNamespace(enable_grouped_topk_kernel=True, device="tpu"))
    try:
        actual_weights, actual_ids = TopK(
            topk=topk,
            renormalize=True,
            routed_scaling_factor=1.75,
        )(logits, correction_bias=bias)
    finally:
        set_global_server_args(previous_server_args)

    np.testing.assert_array_equal(np.asarray(actual_ids), np.asarray(expected_ids))
    np.testing.assert_allclose(
        np.asarray(actual_weights),
        np.asarray(expected_weights),
        rtol=0,
        atol=1e-6,
    )


def test_topk_unsafe_large_token_shape_falls_back_to_jax():
    from sgl_jax.srt.eplb.expert_location import (
        get_global_server_args,
        set_global_server_args,
    )
    from sgl_jax.srt.layers.gate import TopK

    tokens, experts, topk = 2050, 128, 8
    logits = jax.nn.sigmoid(
        jax.random.normal(jax.random.key(13), (tokens, experts), dtype=jnp.float32)
    )
    bias = jax.random.normal(jax.random.key(14), (experts,), dtype=jnp.float32) * 0.1
    expected_weights, expected_ids = _reference(logits, bias, topk=topk)

    previous_server_args = get_global_server_args()
    set_global_server_args(SimpleNamespace(enable_grouped_topk_kernel=True, device="tpu"))
    try:
        actual_weights, actual_ids = TopK(
            topk=topk,
            renormalize=False,
        )(logits, correction_bias=bias)
    finally:
        set_global_server_args(previous_server_args)

    np.testing.assert_array_equal(np.asarray(actual_ids), np.asarray(expected_ids))
    np.testing.assert_array_equal(np.asarray(actual_weights), np.asarray(expected_weights))


def test_v7_tuned_block_sizes_cover_mimo_sweep(monkeypatch):
    from sgl_jax.srt.kernels.biased_topk import tuned_block_sizes

    monkeypatch.setattr(tuned_block_sizes, "_device_name", lambda: "TPU v7")
    expected = {
        64: 64,
        128: 128,
        256: 256,
        512: 512,
        1024: 512,
        2048: 512,
        4096: 1024,
        8192: 1024,
        16384: 1024,
        32768: 1024,
    }

    for tokens, block_tokens in expected.items():
        assert tuned_block_sizes.get_tuned_bt(tokens, 384, 8) == block_tokens
    assert tuned_block_sizes.get_tuned_bt(1024, 256, 8) is None


def test_v6e_tuned_block_sizes_cover_mimo_sweep(monkeypatch):
    from sgl_jax.srt.kernels.biased_topk import tuned_block_sizes

    monkeypatch.setattr(tuned_block_sizes, "_device_name", lambda: "TPU v6e")
    expected = {
        64: 64,
        128: 128,
        256: 256,
        512: 256,
        1024: 256,
        2048: 1024,
        4096: 1024,
        8192: 1024,
        16384: 1024,
        32768: 1024,
    }

    for tokens, block_tokens in expected.items():
        assert tuned_block_sizes.get_tuned_bt(tokens, 384, 8) == block_tokens
    assert tuned_block_sizes.get_tuned_bt(1024, 256, 8) is None


def test_topk_kernel_path_preserves_logical_to_physical_mapping(monkeypatch):
    from sgl_jax.srt.eplb.expert_location import (
        ExpertLocationMetadata,
        get_global_server_args,
        set_global_server_args,
    )
    from sgl_jax.srt.layers.gate import TopK

    monkeypatch.setenv("PALLAS_INTERPRET", "1")
    tokens, experts, topk = 64, 128, 8
    logits = jnp.tile(jnp.arange(experts, dtype=jnp.float32), (tokens, 1))
    bias = jnp.zeros((experts,), dtype=jnp.float32)
    _, logical_ids = _reference(logits, bias, topk=topk)
    logical_to_physical = np.arange(experts - 1, -1, -1, dtype=np.int32)[None, :]
    dispatch_info = ExpertLocationMetadata(
        ep_dispatch_algorithm="static",
        logical_to_rank_dispatch_physical_map=logical_to_physical,
        logical_to_all_physical_map=logical_to_physical[..., None],
        logical_to_all_physical_map_num_valid=np.ones((1, experts), dtype=np.int32),
        physical_to_logical_map=logical_to_physical,
        num_physical_experts=experts,
    )

    previous_server_args = get_global_server_args()
    set_global_server_args(SimpleNamespace(enable_grouped_topk_kernel=True, device="tpu"))
    try:
        _, physical_ids = TopK(
            topk=topk,
            renormalize=False,
        )(logits, correction_bias=bias, dispatch_info=dispatch_info)
    finally:
        set_global_server_args(previous_server_args)

    expected_physical_ids = logical_to_physical[0][np.asarray(logical_ids)]
    np.testing.assert_array_equal(
        np.asarray(physical_ids),
        expected_physical_ids,
    )
