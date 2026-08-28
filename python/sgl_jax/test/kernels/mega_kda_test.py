"""Accuracy and packed-layout coverage for the inference-only Mega KDA kernel."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.kda.mega_kda import (
    is_mega_kda_layout_supported,
    kda_forward_packed,
)
from sgl_jax.srt.kernels.kda.naive import naive_recurrent_kda

K = V = 128
LOWER_BOUND = -5.0
SCALE = K**-0.5


def _normalize(value: jax.Array) -> jax.Array:
    value = value.astype(jnp.float32)
    value *= jax.lax.rsqrt(jnp.sum(value * value, axis=-1, keepdims=True) + 1e-6)
    return value.astype(jnp.bfloat16)


def _case(heads: int, lengths: tuple[int, ...], seed: int = 1550):
    rng = np.random.default_rng(seed + heads)
    tokens = sum(lengths)
    shape = (1, tokens, heads, K)

    def bf16_normal(scale: float = 1.0):
        values = rng.standard_normal(shape, dtype=np.float32) * scale
        return jnp.asarray(values, dtype=jnp.bfloat16)

    return {
        "q": bf16_normal(),
        "k": bf16_normal(),
        "v": bf16_normal(1.5),
        "g": jnp.asarray(
            rng.uniform(-4.5, 4.5, shape).astype(np.float32),
            dtype=jnp.bfloat16,
        ),
        "beta": jnp.asarray(
            rng.uniform(0.05, 0.95, shape[:-1]).astype(np.float32),
            dtype=jnp.bfloat16,
        ),
        "a_log": jnp.asarray(rng.uniform(0.2, 3.0, (heads,)).astype(np.float32)),
        "dt_bias": jnp.asarray(rng.uniform(-8.0, -1.5, (heads, K)).astype(np.float32)),
        "initial_state": jnp.asarray(
            rng.standard_normal(
                (len(lengths), heads, K, V),
                dtype=np.float32,
            )
            * 0.1
        ),
        "cu_seqlens": jnp.asarray(
            [0, *np.cumsum(lengths, dtype=np.int32)],
            dtype=jnp.int32,
        ),
        "lengths": lengths,
    }


def _reference(arrays, lower_bound: float | None = LOWER_BOUND):
    q = _normalize(arrays["q"])
    k = _normalize(arrays["k"])
    gate_input = arrays["g"].astype(jnp.float32) + arrays["dt_bias"][None, None, :, :]
    gate_scale = jnp.exp(arrays["a_log"])[None, None, :, None]
    if lower_bound is None:
        activated_g = -gate_scale * jax.nn.softplus(gate_input)
    else:
        activated_g = lower_bound * jax.nn.sigmoid(gate_scale * gate_input)
    outputs = []
    states = []
    offset = 0
    for segment, length in enumerate(arrays["lengths"]):
        output, state = naive_recurrent_kda(
            q[:, offset : offset + length],
            k[:, offset : offset + length],
            arrays["v"][:, offset : offset + length],
            activated_g[:, offset : offset + length],
            arrays["beta"][:, offset : offset + length],
            scale=SCALE,
            initial_state=arrays["initial_state"][segment : segment + 1],
            output_final_state=True,
        )
        outputs.append(output)
        states.append(state[0])
        offset += length
    return jnp.concatenate(outputs, axis=1), jnp.stack(states)


def _mega(arrays, lower_bound: float | None = LOWER_BOUND):
    return kda_forward_packed(
        arrays["q"],
        arrays["k"],
        arrays["v"],
        arrays["g"],
        arrays["beta"],
        cu_seqlens=arrays["cu_seqlens"],
        A_log=arrays["a_log"],
        dt_bias=arrays["dt_bias"],
        scale=SCALE,
        initial_state=arrays["initial_state"],
        lower_bound=lower_bound,
    )


@pytest.mark.parametrize(
    ("heads", "lengths"),
    [
        (8, (61,)),
        (12, (64, 64)),
        (16, (64, 64)),
        (24, (64, 64)),
    ],
)
def test_mega_kda_matches_naive(heads: int, lengths: tuple[int, ...]):
    arrays = _case(heads, lengths)
    reference_output, reference_state = _reference(arrays)
    mega_output, mega_state = _mega(arrays)
    jax.block_until_ready((reference_output, reference_state, mega_output, mega_state))

    np.testing.assert_allclose(
        np.asarray(mega_output, dtype=np.float32),
        np.asarray(reference_output, dtype=np.float32),
        rtol=5e-2,
        atol=5e-2,
    )
    np.testing.assert_allclose(
        np.asarray(mega_state, dtype=np.float32),
        np.asarray(reference_state, dtype=np.float32),
        rtol=5e-2,
        atol=5e-2,
    )


def test_bounded_boundary_preserves_nonzero_initial_state():
    """A segment starting inside a tile must use its own recurrent state."""
    arrays = _case(8, (37, 91), seed=1557)
    arrays["g"] = jnp.zeros_like(arrays["g"])
    arrays["a_log"] = jnp.zeros_like(arrays["a_log"])
    arrays["dt_bias"] = jnp.zeros_like(arrays["dt_bias"])
    arrays["initial_state"] = jnp.zeros_like(arrays["initial_state"])
    arrays["initial_state"] = (
        arrays["initial_state"]
        .at[1]
        .set(jnp.broadcast_to(jnp.eye(K, V, dtype=jnp.float32), (8, K, V)))
    )

    reference_output, _ = _reference(arrays)
    mega_output, _ = _mega(arrays)
    jax.block_until_ready((reference_output, mega_output))

    first_b_token = arrays["lengths"][0]
    reference = np.asarray(reference_output[:, first_b_token], dtype=np.float32)
    actual = np.asarray(mega_output[:, first_b_token], dtype=np.float32)
    assert np.max(np.abs(reference)) > 1e-4
    np.testing.assert_allclose(actual, reference, rtol=2e-2, atol=2e-4)


@pytest.mark.parametrize(
    ("heads", "lengths"),
    [
        (8, (61,)),
        (8, (63, 65)),
        (16, (64, 64)),
    ],
)
def test_unbounded_mega_kda_matches_naive(heads: int, lengths: tuple[int, ...]):
    arrays = _case(heads, lengths, seed=1554)
    arrays["beta"] = arrays["beta"].astype(jnp.float32)
    reference_output, reference_state = _reference(arrays, lower_bound=None)
    mega_output, mega_state = _mega(arrays, lower_bound=None)
    jax.block_until_ready((reference_output, reference_state, mega_output, mega_state))

    np.testing.assert_allclose(
        np.asarray(mega_output, dtype=np.float32),
        np.asarray(reference_output, dtype=np.float32),
        rtol=5e-2,
        atol=5e-2,
    )
    np.testing.assert_allclose(
        np.asarray(mega_state, dtype=np.float32),
        np.asarray(reference_state, dtype=np.float32),
        rtol=5e-2,
        atol=5e-2,
    )


def test_unbounded_mega_kda_handles_extreme_gate_decay():
    arrays = _case(8, (64,), seed=1555)
    arrays["beta"] = arrays["beta"].astype(jnp.float32)
    arrays["g"] = jnp.full_like(arrays["g"], 4.0)
    arrays["dt_bias"] = jnp.ones_like(arrays["dt_bias"])
    arrays["a_log"] = jnp.full_like(arrays["a_log"], np.log(4.0))
    reference_output, reference_state = _reference(arrays, lower_bound=None)
    mega_output, mega_state = _mega(arrays, lower_bound=None)
    jax.block_until_ready((reference_output, reference_state, mega_output, mega_state))

    assert np.isfinite(np.asarray(mega_output)).all()
    assert np.isfinite(np.asarray(mega_state)).all()
    np.testing.assert_allclose(
        np.asarray(mega_output, dtype=np.float32),
        np.asarray(reference_output, dtype=np.float32),
        rtol=5e-2,
        atol=5e-2,
    )
    np.testing.assert_allclose(
        np.asarray(mega_state, dtype=np.float32),
        np.asarray(reference_state, dtype=np.float32),
        rtol=5e-2,
        atol=5e-2,
    )


def test_mega_kda_state_chaining_matches_single_prefill():
    arrays = _case(8, (128,), seed=1552)
    full_output, full_state = _mega(arrays)

    first = dict(arrays)
    for name in ("q", "k", "v", "g", "beta"):
        first[name] = arrays[name][:, :64]
    first["lengths"] = (64,)
    first["cu_seqlens"] = jnp.asarray([0, 64], dtype=jnp.int32)
    first_output, first_state = _mega(first)

    second = dict(arrays)
    for name in ("q", "k", "v", "g", "beta"):
        second[name] = arrays[name][:, 64:]
    second["lengths"] = (64,)
    second["cu_seqlens"] = jnp.asarray([0, 64], dtype=jnp.int32)
    second["initial_state"] = first_state
    second_output, second_state = _mega(second)

    split_output = jnp.concatenate([first_output, second_output], axis=1)
    jax.block_until_ready((full_output, full_state, split_output, second_state))
    np.testing.assert_allclose(
        np.asarray(split_output, dtype=np.float32),
        np.asarray(full_output, dtype=np.float32),
        rtol=2e-2,
        atol=1e-2,
    )
    np.testing.assert_allclose(
        np.asarray(second_state, dtype=np.float32),
        np.asarray(full_state, dtype=np.float32),
        rtol=2e-2,
        atol=1e-2,
    )


def test_unbounded_mega_kda_state_chaining_matches_single_prefill():
    arrays = _case(8, (128,), seed=1556)
    arrays["beta"] = arrays["beta"].astype(jnp.float32)
    full_output, full_state = _mega(arrays, lower_bound=None)

    first = dict(arrays)
    for name in ("q", "k", "v", "g", "beta"):
        first[name] = arrays[name][:, :64]
    first["lengths"] = (64,)
    first["cu_seqlens"] = jnp.asarray([0, 64], dtype=jnp.int32)
    first_output, first_state = _mega(first, lower_bound=None)

    second = dict(arrays)
    for name in ("q", "k", "v", "g", "beta"):
        second[name] = arrays[name][:, 64:]
    second["lengths"] = (64,)
    second["cu_seqlens"] = jnp.asarray([0, 64], dtype=jnp.int32)
    second["initial_state"] = first_state
    second_output, second_state = _mega(second, lower_bound=None)

    split_output = jnp.concatenate([first_output, second_output], axis=1)
    jax.block_until_ready((full_output, full_state, split_output, second_state))
    np.testing.assert_allclose(
        np.asarray(split_output, dtype=np.float32),
        np.asarray(full_output, dtype=np.float32),
        rtol=2e-2,
        atol=1e-2,
    )
    np.testing.assert_allclose(
        np.asarray(second_state, dtype=np.float32),
        np.asarray(full_state, dtype=np.float32),
        rtol=2e-2,
        atol=1e-2,
    )


@pytest.mark.parametrize(
    ("cu_seqlens", "expected"),
    [
        ([0, 64], True),
        ([0, 63, 128], True),
        ([0, 21, 42, 64], False),
        ([0, 0, 32, 64], True),
    ],
)
def test_mega_kda_layout_guard(cu_seqlens: list[int], expected: bool):
    actual = is_mega_kda_layout_supported(
        jnp.asarray(cu_seqlens, dtype=jnp.int32),
        num_tokens=128,
    )
    assert bool(actual) is expected
