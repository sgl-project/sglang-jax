"""TPU parity tests for the opt-in PLE fusion (including the donated pool)."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.ngram_conv import ngram_conv_prefill, ngram_conv_update
from sgl_jax.srt.kernels.ngram_fused import ngram_decode_pallas, ngram_extend_pallas

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="TPU Pallas lowering")


def _reference(
    key, query, value, nk, nq, nc, weight, pool, slots, init, cu=None, *, hidden, dilation
):
    def norm(x, w):
        grouped = x.astype(jnp.float32).reshape(x.shape[0], -1, hidden)
        y = grouped * jax.lax.rsqrt(jnp.mean(grouped**2, -1, keepdims=True) + 1e-6)
        return (y.reshape(x.shape) * (1 + w.astype(jnp.float32))).astype(x.dtype)

    k = norm(key, nk).astype(jnp.float32).reshape(query.shape[0], -1, hidden)
    q = norm(query, nq).astype(jnp.float32).reshape(k.shape)
    dot = jnp.sum(k * q, -1, keepdims=True) / np.sqrt(hidden)
    gate = jax.nn.sigmoid(jnp.sign(dot) * jnp.sqrt(jnp.maximum(jnp.abs(dot), 1e-6)))
    u = (gate * value[:, None, :].astype(jnp.float32)).reshape(query.shape).astype(query.dtype)
    if cu is None:
        y, state = ngram_conv_update(
            norm(u, nc),
            pool,
            slots,
            weight,
            has_initial_state=init,
            dilation=dilation,
        )
    else:
        y, state = ngram_conv_prefill(
            norm(u, nc).T,
            weight,
            cu_seqlens=cu,
            conv_state=pool,
            state_indices=slots,
            has_initial_state=init,
            dilation=dilation,
        )
        y = y.T
    return u + y, state


def _inputs(batch, hidden, dtype, dilation=3, kernel=4):
    rng = np.random.default_rng(12)
    channels = 2 * hidden

    def rand(shape, scale=1):
        return jnp.asarray(rng.normal(size=shape) * scale, dtype)

    slots = np.arange(1, batch + 1, dtype=np.int32)[::-1].copy()
    # Repeated slot zero must not race or be modified, even with valid state.
    slots[::5] = 0
    return [
        rand((batch, channels)),
        rand((batch, channels)),
        rand((batch, hidden)),
        rand((channels,), 0.1).astype(jnp.float32),
        rand((channels,), 0.1).astype(jnp.float32),
        rand((channels,), 0.1).astype(jnp.float32),
        rand((channels, kernel), 0.1),
        rand((batch + 5, channels, (kernel - 1) * dilation)),
        jnp.asarray(slots),
        jnp.asarray(np.arange(batch) % 3 != 0),
    ]


@pytest.mark.parametrize(
    "batch,hidden,dtype,key_dtype",
    [
        (1, 128, jnp.float32, jnp.float32),
        (9, 128, jnp.bfloat16, jnp.bfloat16),
        (16, 2560, jnp.bfloat16, jnp.float32),
    ],
)
def test_decode_parity(batch, hidden, dtype, key_dtype):
    args = _inputs(batch, hidden, dtype)
    args[0] = args[0].astype(key_dtype)
    reference = jax.jit(functools.partial(_reference, hidden=hidden, dilation=3))
    fused = jax.jit(functools.partial(ngram_decode_pallas, hidden_size=hidden), donate_argnums=(7,))
    want = jax.block_until_ready(reference(*args))
    original_pool = np.asarray(args[7]).copy()
    got = jax.block_until_ready(fused(*args))
    for actual, expected in zip(got, want, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual).astype(float),
            np.asarray(expected).astype(float),
            atol=0.025 if dtype == jnp.bfloat16 else 2e-5,
            rtol=0.025 if dtype == jnp.bfloat16 else 2e-5,
        )
    untouched = np.setdiff1d(np.arange(batch + 5), np.asarray(args[8])[np.asarray(args[8]) != 0])
    np.testing.assert_array_equal(np.asarray(got[1])[untouched], original_pool[untouched])
    # Reuse a donated pool across calls, as serving does.
    next_args = args[:7] + [got[1]] + args[8:]
    next_want = jax.block_until_ready(reference(*next_args))
    next_got = jax.block_until_ready(fused(*next_args))
    for actual, expected in zip(next_got, next_want, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual).astype(float),
            np.asarray(expected).astype(float),
            atol=0.025 if dtype == jnp.bfloat16 else 2e-5,
            rtol=0.025 if dtype == jnp.bfloat16 else 2e-5,
        )


@pytest.mark.parametrize("dilation,kernel", [(1, 2), (2, 3), (3, 4)])
def test_prefill_then_decode_zero_gate_and_fresh_state(dilation, kernel):
    hidden = 128
    args = (
        _inputs(7, hidden, jnp.float32, dilation, kernel)[:7]
        + _inputs(2, hidden, jnp.float32, dilation, kernel)[7:]
    )
    args[1] = jnp.zeros_like(args[1])  # dot == 0, so the trained gate is exactly 0.5.
    args[8] = jnp.array([1, 2], jnp.int32)
    args[9] = jnp.array([False, True])
    cu = jnp.array([0, 2, 7], jnp.int32)
    reference = jax.jit(functools.partial(_reference, hidden=hidden, dilation=dilation))
    want = jax.block_until_ready(reference(*args, cu))
    got = jax.block_until_ready(
        jax.jit(
            functools.partial(ngram_extend_pallas, hidden_size=hidden, dilation=dilation),
            donate_argnums=(7,),
        )(*args, cu)
    )
    for actual, expected in zip(got, want, strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=2e-5, rtol=2e-5)
    next_args = (
        _inputs(2, hidden, jnp.float32, dilation, kernel)[:3]
        + args[3:7]
        + [got[1], args[8], jnp.ones(2, bool)]
    )
    next_want = jax.block_until_ready(reference(*next_args))
    next_got = jax.block_until_ready(
        jax.jit(
            functools.partial(ngram_decode_pallas, hidden_size=hidden, dilation=dilation),
            donate_argnums=(7,),
        )(*next_args)
    )
    for actual, expected in zip(next_got, next_want, strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize(
    "lengths,hidden,dtype",
    [([1, 0, 7, 34, 2, 1], 128, jnp.float32), ([19, 35, 2], 2560, jnp.bfloat16)],
)
def test_extend_parity(lengths, hidden, dtype):
    args = _inputs(sum(lengths), hidden, dtype)[:7] + _inputs(len(lengths), hidden, dtype)[7:]
    args[0] = args[0].astype(jnp.float32)
    args.append(jnp.asarray(np.r_[0, np.cumsum(lengths)], jnp.int32))
    want = jax.jit(functools.partial(_reference, hidden=hidden, dilation=3))(*args)
    jax.block_until_ready(want)
    original_pool = np.asarray(args[7]).copy()
    got = jax.jit(functools.partial(ngram_extend_pallas, hidden_size=hidden), donate_argnums=(7,))(
        *args
    )
    jax.block_until_ready(got)
    for actual, expected in zip(got, want, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual).astype(float),
            np.asarray(expected).astype(float),
            atol=0.025 if dtype == jnp.bfloat16 else 2e-5,
            rtol=0.025 if dtype == jnp.bfloat16 else 2e-5,
        )
    untouched = np.setdiff1d(
        np.arange(len(lengths) + 5), np.asarray(args[8])[np.asarray(args[8]) != 0]
    )
    np.testing.assert_array_equal(np.asarray(got[1])[untouched], original_pool[untouched])
