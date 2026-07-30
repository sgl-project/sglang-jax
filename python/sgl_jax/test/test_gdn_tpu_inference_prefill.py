"""Numerical contract tests for the real TPU-Inference v3 GDN prefill."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sgl_jax.srt.kernels.gdn.tpu_inference_adapter as adapter
from sgl_jax.srt.kernels.gdn.gated_delta import (
    decode_gated_delta_rule_ref,
    jax_causal_conv1d_prefill,
    jax_causal_conv1d_update,
    ragged_gated_delta_rule_ref,
)
from sgl_jax.srt.kernels.gdn.tpu_inference_v3 import fused_conv1d_gdn

pytestmark = pytest.mark.skipif(
    not any(device.platform == "tpu" for device in jax.local_devices()),
    reason="the fused TPU-Inference v3 DMA/state contract requires real TPU hardware",
)


N_KQ = 1
N_V = 2
D_K = 128
D_V = 128
KERNEL_SIZE = 4
DIM = 2 * N_KQ * D_K + N_V * D_V
LENGTHS = (1, 63, 64, 65, 127, 128, 129, 0)
NUM_REQUESTS = len(LENGTHS)
POOL_SIZE = 2 * NUM_REQUESTS + 2
STATE_INDICES = np.arange(1, NUM_REQUESTS + 1, dtype=np.int32)
TRACK_INDICES = np.arange(NUM_REQUESTS + 1, 2 * NUM_REQUESTS + 1, dtype=np.int32)
HAS_INITIAL_STATE = np.asarray([False, True, False, True, True, False, True, True], dtype=np.bool_)


@dataclass(frozen=True)
class _Fixture:
    mixed_qkv: jax.Array
    b: jax.Array
    a: jax.Array
    conv_state: jax.Array
    recurrent_state: jax.Array
    conv_weight: jax.Array
    a_log: jax.Array
    dt_bias: jax.Array
    cu_seqlens: jax.Array
    state_indices: jax.Array
    track_indices: jax.Array
    has_initial_state: jax.Array
    seq_lens: jax.Array


def _random_bf16(rng: np.random.Generator, shape, scale=0.1):
    return jnp.asarray(rng.standard_normal(shape) * scale, dtype=jnp.bfloat16)


def _make_fixture(seed: int = 311) -> _Fixture:
    rng = np.random.default_rng(seed)
    total_tokens = sum(LENGTHS)
    cu_seqlens = np.concatenate(([0], np.cumsum(LENGTHS))).astype(np.int32)
    # Continuing requests must have total length greater than this query chunk.
    seq_lens = np.asarray(LENGTHS, dtype=np.int32) + HAS_INITIAL_STATE.astype(np.int32) * 17
    return _Fixture(
        mixed_qkv=_random_bf16(rng, (total_tokens, DIM)),
        b=_random_bf16(rng, (total_tokens, N_V)),
        a=_random_bf16(rng, (total_tokens, N_V)),
        conv_state=_random_bf16(rng, (POOL_SIZE, DIM, KERNEL_SIZE - 1), scale=0.03),
        recurrent_state=jnp.asarray(
            rng.standard_normal((POOL_SIZE, N_V, D_K, D_V)) * 0.01,
            dtype=jnp.float32,
        ),
        conv_weight=_random_bf16(rng, (DIM, KERNEL_SIZE), scale=0.03),
        a_log=jnp.asarray(rng.uniform(-1.0, 0.0, (N_V,)), dtype=jnp.float32),
        dt_bias=jnp.asarray(rng.uniform(-0.5, 0.5, (N_V,)), dtype=jnp.float32),
        cu_seqlens=jnp.asarray(cu_seqlens),
        state_indices=jnp.asarray(STATE_INDICES),
        track_indices=jnp.asarray(TRACK_INDICES),
        has_initial_state=jnp.asarray(HAS_INITIAL_STATE),
        seq_lens=jnp.asarray(seq_lens),
    )


def _reference_prefill(f: _Fixture):
    conv_out_t, new_conv = jax_causal_conv1d_prefill(
        f.mixed_qkv.T,
        f.conv_weight,
        cu_seqlens=f.cu_seqlens,
        conv_state=f.conv_state,
        state_indices=f.state_indices,
        has_initial_state=f.has_initial_state,
        activation="silu",
        track_indices=f.track_indices,
        track_mask=jnp.ones((NUM_REQUESTS,), dtype=jnp.bool_),
    )
    new_recurrent, output = ragged_gated_delta_rule_ref(
        conv_out_t.T,
        f.b,
        f.a,
        f.recurrent_state,
        f.a_log,
        f.dt_bias,
        f.cu_seqlens,
        f.state_indices,
        f.has_initial_state,
        n_kq=N_KQ,
        n_v=N_V,
        d_k=D_K,
        d_v=D_V,
        track_indices=f.track_indices,
        track_mask=jnp.ones((NUM_REQUESTS,), dtype=jnp.bool_),
    )
    return output, new_conv, new_recurrent


def _vendor_prefill(monkeypatch, f: _Fixture):
    calls = []

    def counted_vendor(*args, **kwargs):
        calls.append("actual-v3")
        return fused_conv1d_gdn(*args, **kwargs)

    monkeypatch.setattr(adapter, "_vendor_fused_conv1d_gdn", counted_vendor)
    result = adapter.fused_conv1d_gdn_prefill(
        f.mixed_qkv,
        f.b,
        f.a,
        f.conv_state,
        f.recurrent_state,
        f.conv_weight,
        f.a_log,
        f.dt_bias,
        f.cu_seqlens,
        f.state_indices,
        f.track_indices,
        f.has_initial_state,
        f.seq_lens,
        n_kq=N_KQ,
        n_v=N_V,
        d_k=D_K,
        d_v=D_V,
        kernel_size=KERNEL_SIZE,
    )
    return calls, jax.block_until_ready(result)


def _assert_unchanged(actual, expected):
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_real_vendor_matches_reference_for_packed_ragged_prefill(monkeypatch):
    fixture = _make_fixture()
    snapshots = tuple(
        np.asarray(value).copy()
        for value in (
            fixture.mixed_qkv,
            fixture.b,
            fixture.a,
            fixture.conv_state,
            fixture.recurrent_state,
            fixture.conv_weight,
            fixture.a_log,
            fixture.dt_bias,
            fixture.cu_seqlens,
            fixture.state_indices,
            fixture.track_indices,
            fixture.has_initial_state,
            fixture.seq_lens,
        )
    )

    expected = _reference_prefill(fixture)
    calls, actual = _vendor_prefill(monkeypatch, fixture)

    assert calls == ["actual-v3"]
    for name, actual_value, expected_value in zip(
        ("output", "conv_state", "recurrent_state"),
        actual,
        expected,
        strict=True,
    ):
        assert np.isfinite(np.asarray(actual_value)).all()
        np.testing.assert_allclose(
            actual_value,
            expected_value,
            rtol=2e-2,
            atol=5e-2,
            err_msg=name,
        )

    # Dummy slot 0, the final unused slot, and every checkpoint must remain
    # isolated while running and track slots receive identical final states.
    _, actual_conv, actual_recurrent = actual
    for slot in (0, POOL_SIZE - 1):
        _assert_unchanged(actual_conv[slot], fixture.conv_state[slot])
        _assert_unchanged(actual_recurrent[slot], fixture.recurrent_state[slot])
    for running, track in zip(STATE_INDICES, TRACK_INDICES, strict=True):
        np.testing.assert_allclose(actual_conv[running], actual_conv[track])
        np.testing.assert_allclose(actual_recurrent[running], actual_recurrent[track])

    for value, snapshot in zip(
        (
            fixture.mixed_qkv,
            fixture.b,
            fixture.a,
            fixture.conv_state,
            fixture.recurrent_state,
            fixture.conv_weight,
            fixture.a_log,
            fixture.dt_bias,
            fixture.cu_seqlens,
            fixture.state_indices,
            fixture.track_indices,
            fixture.has_initial_state,
            fixture.seq_lens,
        ),
        snapshots,
        strict=True,
    ):
        _assert_unchanged(value, snapshot)


def test_real_vendor_prefill_preserves_reference_decode_continuity(monkeypatch):
    fixture = _make_fixture(seed=312)
    expected_prefill = _reference_prefill(fixture)
    calls, actual_prefill = _vendor_prefill(monkeypatch, fixture)
    assert calls == ["actual-v3"]

    rng = np.random.default_rng(313)
    next_qkv = _random_bf16(rng, (NUM_REQUESTS, DIM))
    next_b = _random_bf16(rng, (NUM_REQUESTS, N_V))
    next_a = _random_bf16(rng, (NUM_REQUESTS, N_V))
    decode_initial = jnp.ones((NUM_REQUESTS,), dtype=jnp.bool_)

    def decode(prefill):
        _, conv_pool, recurrent_pool = prefill
        conv_out, next_conv = jax_causal_conv1d_update(
            next_qkv,
            conv_pool,
            fixture.state_indices,
            fixture.conv_weight,
            activation="silu",
            has_initial_state=decode_initial,
        )
        next_recurrent, output = decode_gated_delta_rule_ref(
            conv_out,
            next_b,
            next_a,
            recurrent_pool,
            fixture.a_log,
            fixture.dt_bias,
            fixture.state_indices,
            n_kq=N_KQ,
            n_v=N_V,
            d_k=D_K,
            d_v=D_V,
            has_initial_state=decode_initial,
        )
        return output, next_conv, next_recurrent

    actual_decode = jax.block_until_ready(decode(actual_prefill))
    expected_decode = jax.block_until_ready(decode(expected_prefill))
    for name, actual_value, expected_value in zip(
        ("decode_output", "decode_conv_state", "decode_recurrent_state"),
        actual_decode,
        expected_decode,
        strict=True,
    ):
        assert np.isfinite(np.asarray(actual_value)).all()
        np.testing.assert_allclose(
            actual_value,
            expected_value,
            rtol=2e-2,
            atol=5e-2,
            err_msg=name,
        )
