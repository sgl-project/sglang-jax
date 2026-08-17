"""State-pool contract tests for the fused chunk-parallel GDN adapter."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sgl_jax.srt.kernels.gdn.fused_chunk_parallel_adapter as adapter
import sgl_jax.srt.layers.attention.linear.gdn_backend as gdn_backend
from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    LinearRecurrentAttnBackendMetadata,
)

N_KQ = 1
N_V = 2
D_K = 1
D_V = 1
KERNEL_SIZE = 3
DIM = 2 * N_KQ * D_K + N_V * D_V


def _mesh():
    return jax.sharding.Mesh(jax.devices()[:1], ("tensor",))


def test_track_indices_are_validated_per_dp_rank():
    track_indices = jnp.asarray([2, 3, 2, 3], dtype=jnp.int32)
    state_indices = jnp.asarray([1, 4, 1, 4], dtype=jnp.int32)
    track_mask = jnp.ones((4,), dtype=jnp.bool_)

    assert (
        adapter._validate_track_indices_per_dp(
            track_indices,
            state_indices,
            pool_size=5,
            dp=2,
            track_mask=track_mask,
        )
        is None
    )


def test_track_indices_still_reject_duplicates_within_a_dp_rank():
    with pytest.raises(ValueError, match="duplicate checkpoint"):
        adapter._validate_track_indices_per_dp(
            jnp.asarray([2, 2, 2, 3], dtype=jnp.int32),
            jnp.asarray([1, 4, 1, 4], dtype=jnp.int32),
            pool_size=5,
            dp=2,
            track_mask=jnp.ones((4,), dtype=jnp.bool_),
        )


def _inputs():
    mixed_qkv = jnp.asarray(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
        ],
        dtype=jnp.bfloat16,
    )
    b = jnp.arange(6, dtype=jnp.bfloat16).reshape(3, N_V)
    a = b + jnp.asarray(10, dtype=jnp.bfloat16)
    conv_state = jnp.arange(8 * DIM * (KERNEL_SIZE - 1), dtype=jnp.bfloat16).reshape(
        8, DIM, KERNEL_SIZE - 1
    )
    recurrent_state = jnp.arange(8 * N_V * D_K * D_V, dtype=jnp.float32).reshape(8, N_V, D_K, D_V)
    conv_weight = jnp.arange(DIM * KERNEL_SIZE, dtype=jnp.bfloat16).reshape(DIM, KERNEL_SIZE)
    a_log = jnp.asarray([0.25, 0.5], dtype=jnp.float32)
    dt_bias = jnp.asarray([0.75, 1.0], dtype=jnp.float32)
    cu_seqlens = jnp.asarray([0, 1, 3, 3], dtype=jnp.int32)
    state_indices = jnp.asarray([1, 2, 3], dtype=jnp.int32)
    has_initial_state = jnp.asarray([False, True, True])
    seq_lens = jnp.asarray([1, 7, 5], dtype=jnp.int32)
    return (
        mixed_qkv,
        b,
        a,
        conv_state,
        recurrent_state,
        conv_weight,
        a_log,
        dt_bias,
        cu_seqlens,
        state_indices,
        has_initial_state,
        seq_lens,
    )


def _call(*, track_indices, vendor):
    inputs = _inputs()
    original = getattr(adapter, "_fused_chunk_parallel_kernel", None)
    adapter._fused_chunk_parallel_kernel = vendor
    try:
        entrypoint = getattr(
            adapter,
            "_fused_chunk_parallel_prefill_local",
            adapter.fused_chunk_parallel_prefill,
        )
        result = entrypoint(
            *inputs[:10],
            track_indices,
            *inputs[10:],
            n_kq=N_KQ,
            n_v=N_V,
            d_k=D_K,
            d_v=D_V,
            kernel_size=KERNEL_SIZE,
        )
    finally:
        if original is None:
            del adapter._fused_chunk_parallel_kernel
        else:
            adapter._fused_chunk_parallel_kernel = original
    return inputs, result


def test_adapter_converts_layout_metadata_and_preserves_pool_contract():
    captured = {}

    def fake_vendor(
        qkv,
        b,
        a,
        conv_state,
        recurrent_state,
        conv_weight,
        conv_bias,
        a_log,
        dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        seq_lens,
        *,
        n_kq,
        n_v,
        d_k,
        d_v,
        kernel_size,
        **kwargs,
    ):
        del kwargs
        captured.update(locals())
        query_lens = query_start_loc[1:] - query_start_loc[:-1]

        # Deterministically emulate the vendor's full-pool result. Deliberately
        # corrupt slot 0 and an unused slot: the adapter must consume only the
        # active running-slot values from the vendor result.
        new_conv = conv_state.at[0].set(-100).at[7].set(-700)
        new_recurrent = recurrent_state.at[0].set(-1000).at[7].set(-7000)
        for row in range(state_indices.shape[0]):
            idx = state_indices[row]
            has_tokens = query_lens[row] > 0
            conv_value = conv_state[idx] + (row + 1) * 10
            recurrent_value = recurrent_state[idx] + (row + 1) * 100
            new_conv = new_conv.at[idx].set(jnp.where(has_tokens, conv_value, conv_state[idx]))
            new_recurrent = new_recurrent.at[idx].set(
                jnp.where(has_tokens, recurrent_value, recurrent_state[idx])
            )

        output = jnp.arange(qkv.shape[0] * n_v * d_v, dtype=jnp.float32).reshape(
            qkv.shape[0], n_v * d_v
        )
        return (new_conv, new_recurrent), output

    track_indices = jnp.asarray([4, 5, 6], dtype=jnp.int32)
    inputs, (output, new_conv, new_recurrent) = _call(
        track_indices=track_indices, vendor=fake_vendor
    )
    (
        mixed_qkv,
        b,
        a,
        conv_state,
        recurrent_state,
        conv_weight,
        a_log,
        dt_bias,
        cu_seqlens,
        state_indices,
        _,
        seq_lens,
    ) = inputs

    # The incoming local tensor is already one TP stripe [q_rank|k_rank|v_rank].
    np.testing.assert_array_equal(captured["qkv"], mixed_qkv)
    np.testing.assert_array_equal(captured["b"], b)
    np.testing.assert_array_equal(captured["a"], a)
    np.testing.assert_array_equal(captured["conv_weight"], conv_weight[:, None, :])
    assert captured["conv_bias"] is None
    np.testing.assert_array_equal(captured["a_log"], a_log)
    np.testing.assert_array_equal(captured["dt_bias"], dt_bias)
    np.testing.assert_array_equal(captured["query_start_loc"], cu_seqlens)
    np.testing.assert_array_equal(captured["state_indices"], state_indices)
    np.testing.assert_array_equal(captured["distribution"], np.asarray([0, 0, 3]))
    np.testing.assert_array_equal(captured["seq_lens"], seq_lens)
    assert (
        captured["n_kq"],
        captured["n_v"],
        captured["d_k"],
        captured["d_v"],
        captured["kernel_size"],
    ) == (N_KQ, N_V, D_K, D_V, KERNEL_SIZE)

    # Conv pool is transposed for the vendor. Fresh request 0 is masked to
    # zero; continuing and zero-length request states are gathered unchanged.
    vendor_conv = np.asarray(captured["conv_state"])
    vendor_recurrent = np.asarray(captured["recurrent_state"])
    assert vendor_conv.shape == (8, KERNEL_SIZE - 1, DIM)
    np.testing.assert_array_equal(vendor_conv[1], np.zeros((2, DIM)))
    np.testing.assert_array_equal(vendor_conv[2], np.asarray(conv_state[2]).T)
    np.testing.assert_array_equal(vendor_conv[3], np.asarray(conv_state[3]).T)
    np.testing.assert_array_equal(vendor_recurrent[1], np.zeros((N_V, D_K, D_V)))
    np.testing.assert_array_equal(vendor_recurrent[2], recurrent_state[2])
    np.testing.assert_array_equal(vendor_recurrent[3], recurrent_state[3])

    # All three vendor returns are consumed and restored to the SGL-JAX ABI.
    assert output.shape == (3, N_V, D_V)
    assert output.dtype == mixed_qkv.dtype
    np.testing.assert_array_equal(
        output.reshape(3, -1), np.arange(6, dtype=np.float32).reshape(3, 2)
    )
    assert new_conv.dtype == conv_state.dtype
    assert new_recurrent.dtype == recurrent_state.dtype

    expected_conv_1 = np.full_like(np.asarray(conv_state[1]), 10)
    expected_conv_2 = np.asarray(conv_state[2]) + 20
    expected_rec_1 = np.full_like(np.asarray(recurrent_state[1]), 100)
    expected_rec_2 = np.asarray(recurrent_state[2]) + 200
    np.testing.assert_array_equal(new_conv[1], expected_conv_1)
    np.testing.assert_array_equal(new_conv[2], expected_conv_2)
    np.testing.assert_array_equal(new_recurrent[1], expected_rec_1)
    np.testing.assert_array_equal(new_recurrent[2], expected_rec_2)

    # Checkpoints receive the same final state as running slots. The
    # zero-length request snapshots its unchanged gathered initial state.
    np.testing.assert_array_equal(new_conv[4], expected_conv_1)
    np.testing.assert_array_equal(new_conv[5], expected_conv_2)
    np.testing.assert_array_equal(new_conv[6], conv_state[3])
    np.testing.assert_array_equal(new_recurrent[4], expected_rec_1)
    np.testing.assert_array_equal(new_recurrent[5], expected_rec_2)
    np.testing.assert_array_equal(new_recurrent[6], recurrent_state[3])

    # Dummy slot 0 and unused running slot 3/unused pool contents survive.
    np.testing.assert_array_equal(new_conv[0], conv_state[0])
    np.testing.assert_array_equal(new_recurrent[0], recurrent_state[0])
    np.testing.assert_array_equal(new_conv[3], conv_state[3])
    np.testing.assert_array_equal(new_recurrent[3], recurrent_state[3])
    np.testing.assert_array_equal(new_conv[7], conv_state[7])
    np.testing.assert_array_equal(new_recurrent[7], recurrent_state[7])


@pytest.mark.parametrize(
    ("track_indices", "match"),
    [
        ([4, 4, 0], "duplicate"),
        ([4, 8, 0], "out of range"),
        ([1, 5, 0], "running state"),
    ],
)
def test_debug_validator_rejects_invalid_track_indices(track_indices, match):
    with pytest.raises(ValueError, match=match):
        adapter._validate_track_indices(
            jnp.asarray(track_indices, dtype=jnp.int32),
            jnp.asarray([1, 2, 3], dtype=jnp.int32),
            pool_size=8,
        )


def test_local_adapter_does_not_repeat_track_validation(monkeypatch):
    def vendor(qkv, b, a, conv_state, recurrent_state, *args, **kwargs):
        del b, a, args, kwargs
        output = jnp.zeros((qkv.shape[0], N_V * D_V), dtype=qkv.dtype)
        return (conv_state, recurrent_state), output

    monkeypatch.setattr(
        adapter,
        "_validate_track_indices",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("local shard must not repeat track validation")
        ),
    )

    _call(track_indices=jnp.asarray([4, 5, 6], dtype=jnp.int32), vendor=vendor)


def test_forward_extend_rejects_active_dummy_track_before_vendor(monkeypatch):
    monkeypatch.setenv(adapter._TRACK_VALIDATION_ENV, "1")
    monkeypatch.setattr(
        gdn_backend,
        "validate_fused_chunk_parallel_capability",
        lambda **_: None,
    )

    def vendor(*args, **kwargs):
        del args, kwargs
        raise AssertionError("vendor must not run for invalid active track metadata")

    monkeypatch.setattr(adapter, "_fused_chunk_parallel_kernel", vendor)
    backend = gdn_backend.GDNAttnBackend(
        num_k_heads=N_KQ,
        num_v_heads=N_V,
        head_k_dim=D_K,
        head_v_dim=D_V,
        conv_kernel_size=KERNEL_SIZE,
        mesh=_mesh(),
        dtype=jnp.bfloat16,
        prefill_impl="fused_chunk_parallel",
    )
    backend.forward_metadata = LinearRecurrentAttnBackendMetadata(
        cu_q_lens=jnp.asarray([0, 1], dtype=jnp.int32),
        recurrent_indices=jnp.asarray([1], dtype=jnp.int32),
        has_initial_state=jnp.asarray([False]),
        recurrent_track_indices=jnp.asarray([0], dtype=jnp.int32),
        recurrent_track_mask=jnp.asarray([True]),
    )

    with pytest.raises(ValueError, match="active track.*dummy"):
        backend.forward_extend(
            jnp.ones((1, DIM), dtype=jnp.bfloat16),
            jnp.zeros((3, DIM, KERNEL_SIZE - 1), dtype=jnp.bfloat16),
            jnp.zeros((3, N_V, D_K, D_V), dtype=jnp.float32),
            jnp.zeros((1, N_V), dtype=jnp.bfloat16),
            jnp.zeros((1, N_V), dtype=jnp.bfloat16),
            jnp.zeros((DIM, KERNEL_SIZE), dtype=jnp.bfloat16),
            jnp.zeros((N_V,), dtype=jnp.float32),
            jnp.zeros((N_V,), dtype=jnp.float32),
            seq_lens=jnp.asarray([1], dtype=jnp.int32),
        )


def test_prefill_rejects_track_outside_dp_local_pool_before_shard_map(monkeypatch):
    monkeypatch.setenv(adapter._TRACK_VALIDATION_ENV, "1")
    metadata = LinearRecurrentAttnBackendMetadata(
        cu_q_lens=jnp.asarray([0, 1, 2], dtype=jnp.int32),
        recurrent_indices=jnp.asarray([1, 1], dtype=jnp.int32),
        has_initial_state=jnp.asarray([True, True]),
        # Global pool size is 8, but DP=2 makes the rank-local size 4.
        recurrent_track_indices=jnp.asarray([4, 2], dtype=jnp.int32),
        recurrent_track_mask=jnp.asarray([True, True]),
    )
    backend = SimpleNamespace(
        forward_metadata=metadata,
        mesh=SimpleNamespace(shape={"data": 2, "tensor": 1}),
        num_k_heads=N_KQ,
        num_v_heads=N_V,
        head_k_dim=D_K,
        head_v_dim=D_V,
        conv_kernel_size=KERNEL_SIZE,
    )
    monkeypatch.setattr(
        adapter.jax,
        "shard_map",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("shard_map must not run for invalid local track metadata")
        ),
    )

    with pytest.raises(ValueError, match="out of range"):
        adapter.fused_chunk_parallel_prefill(
            backend,
            jnp.ones((2, DIM), dtype=jnp.bfloat16),
            jnp.zeros((8, DIM, KERNEL_SIZE - 1), dtype=jnp.bfloat16),
            jnp.zeros((8, N_V, D_K, D_V), dtype=jnp.float32),
            jnp.zeros((2, N_V), dtype=jnp.bfloat16),
            jnp.zeros((2, N_V), dtype=jnp.bfloat16),
            jnp.zeros((DIM, KERNEL_SIZE), dtype=jnp.bfloat16),
            jnp.zeros((N_V,), dtype=jnp.float32),
            jnp.zeros((N_V,), dtype=jnp.float32),
            jnp.asarray([1, 1], dtype=jnp.int32),
        )


def test_jitted_invalid_track_does_not_execute_vendor(monkeypatch):
    monkeypatch.setenv(adapter._TRACK_VALIDATION_ENV, "1")
    vendor_calls = []

    def vendor(qkv, b, a, conv_state, recurrent_state, *args, **kwargs):
        del b, a, args, kwargs
        jax.debug.callback(lambda: vendor_calls.append("called"))
        return (conv_state, recurrent_state), jnp.zeros((qkv.shape[0], N_V * D_V), dtype=qkv.dtype)

    monkeypatch.setattr(adapter, "_fused_chunk_parallel_kernel", vendor)
    mesh = _mesh()

    @jax.jit
    def run(track_indices):
        backend = SimpleNamespace(
            forward_metadata=LinearRecurrentAttnBackendMetadata(
                cu_q_lens=jnp.asarray([0, 1], dtype=jnp.int32),
                recurrent_indices=jnp.asarray([1], dtype=jnp.int32),
                has_initial_state=jnp.asarray([True]),
                recurrent_track_indices=track_indices,
                recurrent_track_mask=jnp.asarray([True]),
            ),
            mesh=mesh,
            num_k_heads=N_KQ,
            num_v_heads=N_V,
            head_k_dim=D_K,
            head_v_dim=D_V,
            conv_kernel_size=KERNEL_SIZE,
        )
        return adapter.fused_chunk_parallel_prefill(
            backend,
            jnp.ones((1, DIM), dtype=jnp.bfloat16),
            jnp.zeros((3, DIM, KERNEL_SIZE - 1), dtype=jnp.bfloat16),
            jnp.zeros((3, N_V, D_K, D_V), dtype=jnp.float32),
            jnp.zeros((1, N_V), dtype=jnp.bfloat16),
            jnp.zeros((1, N_V), dtype=jnp.bfloat16),
            jnp.zeros((DIM, KERNEL_SIZE), dtype=jnp.bfloat16),
            jnp.zeros((N_V,), dtype=jnp.float32),
            jnp.zeros((N_V,), dtype=jnp.float32),
            jnp.asarray([1], dtype=jnp.int32),
        )

    with jax.sharding.set_mesh(mesh), pytest.raises(Exception, match="active track.*dummy"):
        jax.block_until_ready(run(jnp.asarray([0], dtype=jnp.int32)))
    assert vendor_calls == []


@dataclass
class _ForwardFixture:
    output_value: float = 5.0
    conv_value: float = 7.0
    recurrent_value: float = 9.0

    def vendor(self, qkv, b, a, conv_state, recurrent_state, *args, **kwargs):
        del b, a, args, kwargs
        new_conv = conv_state.at[1].set(self.conv_value)
        new_recurrent = recurrent_state.at[1].set(self.recurrent_value)
        output = jnp.full((qkv.shape[0], N_V * D_V), self.output_value)
        return (new_conv, new_recurrent), output


def test_forward_extend_executes_the_callable_frozen_at_initialization(monkeypatch):
    monkeypatch.setattr(
        gdn_backend,
        "validate_fused_chunk_parallel_capability",
        lambda **_: None,
    )
    fixture = _ForwardFixture()
    monkeypatch.setattr(adapter, "_fused_chunk_parallel_kernel", fixture.vendor, raising=False)
    monkeypatch.setattr(
        adapter,
        "_validate_track_indices_per_dp",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("production prefill must not build debug validation")
        ),
    )
    backend = gdn_backend.GDNAttnBackend(
        num_k_heads=N_KQ,
        num_v_heads=N_V,
        head_k_dim=D_K,
        head_v_dim=D_V,
        conv_kernel_size=KERNEL_SIZE,
        mesh=_mesh(),
        dtype=jnp.bfloat16,
        prefill_impl="fused_chunk_parallel",
    )
    # Mutating both selector state and the backend module's symbol after
    # construction must not replace the initialized callable.
    backend.prefill_impl = "chunked_jax"
    monkeypatch.setattr(
        gdn_backend,
        "fused_chunk_parallel_prefill",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("not frozen")),
    )

    with jax.sharding.set_mesh(backend.mesh):
        backend.forward_metadata = LinearRecurrentAttnBackendMetadata(
            cu_q_lens=jnp.asarray([0, 2], dtype=jnp.int32),
            recurrent_indices=jnp.asarray([1], dtype=jnp.int32),
            has_initial_state=jnp.asarray([False]),
            recurrent_track_indices=jnp.asarray([2], dtype=jnp.int32),
            recurrent_track_mask=jnp.asarray([True]),
        )
        output, new_conv, new_recurrent = backend.forward_extend(
            jnp.ones((2, DIM), dtype=jnp.bfloat16),
            jnp.zeros((3, DIM, KERNEL_SIZE - 1), dtype=jnp.bfloat16),
            jnp.zeros((3, N_V, D_K, D_V), dtype=jnp.float32),
            jnp.zeros((2, N_V), dtype=jnp.bfloat16),
            jnp.zeros((2, N_V), dtype=jnp.bfloat16),
            jnp.zeros((DIM, KERNEL_SIZE), dtype=jnp.bfloat16),
            jnp.zeros((N_V,), dtype=jnp.float32),
            jnp.zeros((N_V,), dtype=jnp.float32),
            seq_lens=jnp.asarray([2], dtype=jnp.int32),
        )
        output = np.asarray(output)
        new_conv = np.asarray(new_conv)
        new_recurrent = np.asarray(new_recurrent)

    np.testing.assert_array_equal(output, fixture.output_value)
    np.testing.assert_array_equal(new_conv[1], fixture.conv_value)
    np.testing.assert_array_equal(new_recurrent[1], fixture.recurrent_value)
    np.testing.assert_array_equal(new_conv[2], fixture.conv_value)
    np.testing.assert_array_equal(new_recurrent[2], fixture.recurrent_value)
