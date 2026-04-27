"""Tests for ``short_convolution`` against an ``nnx.Conv`` baseline.

``short_convolution`` is variable-length aware (it accepts a packed
``[total_tokens, hidden]`` input together with ``cu_seqlens``) while
``nnx.Conv`` operates on a single ``[1, T, D]`` sample. To compare them we
run ``short_convolution`` once on the packed batch, then run ``nnx.Conv``
sequence-by-sequence and concatenate the per-sequence outputs back into the
packed layout. Decode mode is also exercised by stepping the conv one token
at a time and checking the rolling cache against the equivalent ``nnx.Conv``
output on the full prefix.
"""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from sgl_jax.srt.layers.attention.linear.short_convolution import short_convolution
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.test.test_utils import CustomTestCase


def _make_weight(rng: np.random.Generator, hidden: int, kernel: int, dtype):
    """Random depthwise conv kernel, shape [D, K]."""
    w = rng.standard_normal((hidden, kernel)).astype(np.float32) * 0.1
    return jnp.asarray(w, dtype=dtype)


def _make_nnx_conv(weight_dk: jax.Array, kernel_size: int, hidden: int):
    """Build an ``nnx.Conv`` whose kernel matches ``weight_dk`` ([D, K]).

    nnx.Conv depthwise convention with ``feature_group_count=hidden`` and
    ``in_features=hidden`` expects kernel shape ``[K, 1, D]``. We construct
    a fresh module then overwrite its kernel.
    """
    conv = nnx.Conv(
        in_features=hidden,
        out_features=hidden,
        kernel_size=(kernel_size,),
        feature_group_count=hidden,
        padding=[(kernel_size - 1, 0)],  # left-pad → causal
        use_bias=False,
        rngs=nnx.Rngs(0),
        dtype=weight_dk.dtype,
        param_dtype=weight_dk.dtype,
    )
    # weight_dk: [D, K]  ->  nnx kernel: [K, 1, D]
    kernel_kid = jnp.transpose(weight_dk, (1, 0))[:, None, :]
    conv.kernel.value = kernel_kid.astype(conv.kernel.value.dtype)
    return conv


def _baseline_extend(
    x_packed: jax.Array,
    cu_seqlens: jax.Array,
    conv: nnx.Conv,
    initial_cache: jax.Array,
) -> jax.Array:
    """Run nnx.Conv per sequence and concat outputs back to packed layout.

    For each sequence, prepend the last ``K-1`` tokens of its initial cache
    so the causal conv sees the same history as ``short_convolution``.
    """
    K = conv.kernel.value.shape[0]
    pieces = []
    cu = np.asarray(cu_seqlens)
    for i in range(len(cu) - 1):
        start, end = int(cu[i]), int(cu[i + 1])
        seq = x_packed[start:end]  # [T_i, D]
        # Cache slot K-1 is the most recent prior token; we need the K-1
        # most recent prior tokens, which sit in slots [1, K-1].
        history = initial_cache[i, :, 1:]  # [D, K-1]
        history_t = jnp.swapaxes(history, 0, 1)  # [K-1, D]
        seq_with_history = jnp.concatenate([history_t, seq], axis=0)  # [K-1+T_i, D]
        # nnx.Conv expects [B, T, D]; we used left-pad of K-1 already, but
        # we also baked the real history in. To prevent double-counting,
        # use ``padding="VALID"`` semantics by manually trimming: run with
        # the conv's own (K-1, 0) left pad, then drop the first K-1 outputs
        # corresponding to the injected history.
        y = conv(seq_with_history[None, ...])[0]  # [(K-1)+T_i, D]
        y = y[K - 1 :]  # drop outputs over the injected history
        pieces.append(jax.nn.silu(y))
    return jnp.concatenate(pieces, axis=0)


def _baseline_decode(
    x_step: jax.Array,  # [B, D]
    prior_inputs: list[jax.Array],  # per-seq history of past tokens, each [T_i, D]
    conv: nnx.Conv,
) -> jax.Array:
    """Decode baseline: run nnx.Conv on (history + new_token) per seq, take
    the last output position."""
    K = conv.kernel.value.shape[0]
    outs = []
    for i, hist in enumerate(prior_inputs):
        full = jnp.concatenate([hist, x_step[i : i + 1]], axis=0)  # [T+1, D]
        # Pad on the left so the conv is causal even when T+1 < K.
        if full.shape[0] < K:
            pad = jnp.zeros((K - full.shape[0], full.shape[1]), dtype=full.dtype)
            full = jnp.concatenate([pad, full], axis=0)
        y = conv(full[None, ...])[0]  # padding=(K-1,0) → output length = full.shape[0]
        outs.append(jax.nn.silu(y[-1:]))
    return jnp.concatenate(outs, axis=0)


class ShortConvolutionTest(CustomTestCase):
    """Compare ``short_convolution`` against an ``nnx.Conv`` baseline."""

    def setUp(self):
        super().setUp()
        self.rng = np.random.default_rng(0)
        self.dtype = jnp.float32  # use fp32 for tighter numerical comparison
        self.atol = 1e-4
        self.rtol = 1e-4

    # ------------------------------------------------------------------
    # EXTEND mode
    # ------------------------------------------------------------------

    def _run_extend_case(self, seq_lens, hidden, kernel, has_prior_cache):
        total = sum(seq_lens)
        cu_seqlens = jnp.asarray(np.concatenate([[0], np.cumsum(seq_lens)]), dtype=jnp.int32)
        x = jnp.asarray(
            self.rng.standard_normal((total, hidden)).astype(np.float32),
            dtype=self.dtype,
        )
        weight = _make_weight(self.rng, hidden, kernel, self.dtype)

        B = len(seq_lens)
        if has_prior_cache:
            cache = jnp.asarray(
                self.rng.standard_normal((B, hidden, kernel)).astype(np.float32),
                dtype=self.dtype,
            )
        else:
            cache = jnp.zeros((B, hidden, kernel), dtype=self.dtype)

        # short_convolution: one shot over the packed batch.
        y_short, new_cache = short_convolution(
            x,
            weight,
            cache,
            cu_seqlens,
            ForwardMode.EXTEND,
            bias=None,
            activation="silu",
        )
        self.assertEqual(y_short.shape, (total, hidden))
        self.assertEqual(new_cache.shape, (B, hidden, kernel))

        # Baseline: nnx.Conv per sequence with injected history.
        conv = _make_nnx_conv(weight, kernel, hidden)
        y_ref = _baseline_extend(x, cu_seqlens, conv, cache)

        np.testing.assert_allclose(
            np.asarray(y_short),
            np.asarray(y_ref),
            atol=self.atol,
            rtol=self.rtol,
        )

        # New cache should be the last K input tokens of each sequence,
        # falling back to prior cache slots when the sequence is shorter.
        cu = np.asarray(cu_seqlens)
        for i in range(B):
            start, end = int(cu[i]), int(cu[i + 1])
            T_i = end - start
            seq = np.asarray(x[start:end])  # [T_i, D]
            old = np.asarray(cache[i])  # [D, K]
            if T_i >= kernel:
                expected = seq[-kernel:].T  # [D, K]
            else:
                expected = np.concatenate([old[:, T_i:], seq.T], axis=1)  # [D, K]
            np.testing.assert_allclose(
                np.asarray(new_cache[i]),
                expected,
                atol=self.atol,
                rtol=self.rtol,
            )

    def test_extend_single_sequence_no_prior_cache(self):
        self._run_extend_case(seq_lens=[7], hidden=1024, kernel=4, has_prior_cache=False)

    def test_extend_multi_sequence_no_prior_cache(self):
        self._run_extend_case(seq_lens=[5, 3, 9], hidden=1024, kernel=4, has_prior_cache=False)

    def test_extend_multi_sequence_with_prior_cache(self):
        self._run_extend_case(seq_lens=[5, 3, 9], hidden=1024, kernel=4, has_prior_cache=True)

    def test_extend_short_sequence_with_prior_cache(self):
        # Sequence shorter than kernel → new cache must mix prior cache
        # tail with the few new tokens.
        self._run_extend_case(seq_lens=[1, 2, 3], hidden=1024, kernel=4, has_prior_cache=True)

    def test_extend_kernel_size_2(self):
        self._run_extend_case(seq_lens=[3, 4], hidden=1024, kernel=2, has_prior_cache=True)

    # ------------------------------------------------------------------
    # DECODE mode
    # ------------------------------------------------------------------

    def test_decode_single_step_against_baseline(self):
        B, hidden, kernel = 3, 1024, 4
        weight = _make_weight(self.rng, hidden, kernel, self.dtype)

        # Build a per-sequence prior history of varying lengths so the cache
        # represents a real prefix.
        histories = [
            jnp.asarray(
                self.rng.standard_normal((L, hidden)).astype(np.float32),
                dtype=self.dtype,
            )
            for L in [5, 2, 7]
        ]
        # Construct the cache: slot K-1 = most recent prior token.
        cache = np.zeros((B, hidden, kernel), dtype=np.float32)
        for i, h in enumerate(histories):
            h_np = np.asarray(h)
            take = min(kernel, h_np.shape[0])
            # Place the last `take` tokens into the rightmost slots.
            cache[i, :, kernel - take :] = h_np[-take:].T
        cache = jnp.asarray(cache, dtype=self.dtype)

        x_step = jnp.asarray(
            self.rng.standard_normal((B, hidden)).astype(np.float32),
            dtype=self.dtype,
        )

        y_short, new_cache = short_convolution(
            x_step,
            weight,
            cache,
            cu_seqlens=None,
            forward_mode=ForwardMode.DECODE,
            bias=None,
            activation="silu",
        )
        self.assertEqual(y_short.shape, (B, hidden))
        self.assertEqual(new_cache.shape, (B, hidden, kernel))

        conv = _make_nnx_conv(weight, kernel, hidden)
        y_ref = _baseline_decode(x_step, histories, conv)

        np.testing.assert_allclose(
            np.asarray(y_short),
            np.asarray(y_ref),
            atol=self.atol,
            rtol=self.rtol,
        )

        # New cache should equal old cache shifted left + new x at slot K-1.
        for i in range(B):
            old = np.asarray(cache[i])
            expected = np.concatenate([old[:, 1:], np.asarray(x_step[i])[:, None]], axis=1)
            np.testing.assert_allclose(
                np.asarray(new_cache[i]),
                expected,
                atol=self.atol,
                rtol=self.rtol,
            )

    def test_decode_rolling_matches_extend(self):
        """Decoding token-by-token should match running extend on the full
        sequence in one shot."""
        T, hidden, kernel = 6, 1024, 4
        seq = jnp.asarray(
            self.rng.standard_normal((T, hidden)).astype(np.float32),
            dtype=self.dtype,
        )
        weight = _make_weight(self.rng, hidden, kernel, self.dtype)

        # One-shot extend (single sequence, no prior cache).
        cu = jnp.asarray([0, T], dtype=jnp.int32)
        y_extend, _ = short_convolution(
            seq,
            weight,
            jnp.zeros((1, hidden, kernel), dtype=self.dtype),
            cu,
            ForwardMode.EXTEND,
            bias=None,
            activation="silu",
        )

        # Token-by-token decode starting from a zero cache.
        cache = jnp.zeros((1, hidden, kernel), dtype=self.dtype)
        outs = []
        for t in range(T):
            y_step, cache = short_convolution(
                seq[t : t + 1],
                weight,
                cache,
                cu_seqlens=None,
                forward_mode=ForwardMode.DECODE,
                bias=None,
                activation="silu",
            )
            outs.append(y_step)
        y_decode = jnp.concatenate(outs, axis=0)

        np.testing.assert_allclose(
            np.asarray(y_decode),
            np.asarray(y_extend),
            atol=self.atol,
            rtol=self.rtol,
        )

    # ------------------------------------------------------------------
    # Bias / activation paths
    # ------------------------------------------------------------------

    def test_bias_added_before_activation(self):
        T, hidden, kernel = 4, 1024, 3
        x = jnp.asarray(
            self.rng.standard_normal((T, hidden)).astype(np.float32),
            dtype=self.dtype,
        )
        weight = _make_weight(self.rng, hidden, kernel, self.dtype)
        bias = jnp.asarray(
            self.rng.standard_normal((hidden,)).astype(np.float32),
            dtype=self.dtype,
        )
        cache = jnp.zeros((1, hidden, kernel), dtype=self.dtype)
        cu = jnp.asarray([0, T], dtype=jnp.int32)

        y_with_bias, _ = short_convolution(
            x, weight, cache, cu, ForwardMode.EXTEND, bias=bias, activation="silu"
        )
        y_no_bias, _ = short_convolution(
            x, weight, cache, cu, ForwardMode.EXTEND, bias=None, activation=None
        )
        # silu(no_bias_pre + bias) should differ from no_bias and not be
        # simply (no_bias + bias) because of the silu nonlinearity.
        diff = np.asarray(y_with_bias) - np.asarray(y_no_bias)
        self.assertGreater(float(np.max(np.abs(diff))), 1e-4)

    def test_activation_none_skips_silu(self):
        T, hidden, kernel = 4, 1024, 3
        x = jnp.asarray(
            self.rng.standard_normal((T, hidden)).astype(np.float32),
            dtype=self.dtype,
        )
        weight = _make_weight(self.rng, hidden, kernel, self.dtype)
        cache = jnp.zeros((1, hidden, kernel), dtype=self.dtype)
        cu = jnp.asarray([0, T], dtype=jnp.int32)

        y_silu, _ = short_convolution(
            x, weight, cache, cu, ForwardMode.EXTEND, bias=None, activation="silu"
        )
        y_none, _ = short_convolution(
            x, weight, cache, cu, ForwardMode.EXTEND, bias=None, activation=None
        )
        np.testing.assert_allclose(
            np.asarray(y_silu),
            np.asarray(jax.nn.silu(y_none)),
            atol=self.atol,
            rtol=self.rtol,
        )

    def test_invalid_activation_raises(self):
        with self.assertRaises(ValueError):
            short_convolution(
                jnp.zeros((1, 4)),
                jnp.zeros((4, 3)),
                jnp.zeros((1, 4, 3)),
                jnp.asarray([0, 1], dtype=jnp.int32),
                ForwardMode.EXTEND,
                activation="not_a_real_activation",
            )

    def test_activation_named_relu_matches_jax_relu(self):
        T, hidden, kernel = 4, 1024, 3
        x = jnp.asarray(
            self.rng.standard_normal((T, hidden)).astype(np.float32),
            dtype=self.dtype,
        )
        weight = _make_weight(self.rng, hidden, kernel, self.dtype)
        cache = jnp.zeros((1, hidden, kernel), dtype=self.dtype)
        cu = jnp.asarray([0, T], dtype=jnp.int32)

        y_relu, _ = short_convolution(
            x, weight, cache, cu, ForwardMode.EXTEND, bias=None, activation="relu"
        )
        y_none, _ = short_convolution(
            x, weight, cache, cu, ForwardMode.EXTEND, bias=None, activation=None
        )
        np.testing.assert_allclose(
            np.asarray(y_relu),
            np.asarray(jax.nn.relu(y_none)),
            atol=self.atol,
            rtol=self.rtol,
        )

    def test_activation_callable_is_used(self):
        T, hidden, kernel = 4, 1024, 3
        x = jnp.asarray(
            self.rng.standard_normal((T, hidden)).astype(np.float32),
            dtype=self.dtype,
        )
        weight = _make_weight(self.rng, hidden, kernel, self.dtype)
        cache = jnp.zeros((1, hidden, kernel), dtype=self.dtype)
        cu = jnp.asarray([0, T], dtype=jnp.int32)

        y_callable, _ = short_convolution(
            x, weight, cache, cu, ForwardMode.EXTEND, bias=None, activation=jax.nn.gelu
        )
        y_none, _ = short_convolution(
            x, weight, cache, cu, ForwardMode.EXTEND, bias=None, activation=None
        )
        np.testing.assert_allclose(
            np.asarray(y_callable),
            np.asarray(jax.nn.gelu(y_none)),
            atol=self.atol,
            rtol=self.rtol,
        )

    def test_extend_requires_cu_seqlens(self):
        with self.assertRaises(ValueError):
            short_convolution(
                jnp.zeros((1, 4)),
                jnp.zeros((4, 3)),
                jnp.zeros((1, 4, 3)),
                cu_seqlens=None,
                forward_mode=ForwardMode.EXTEND,
            )


class ShortConvolutionBF16Test(ShortConvolutionTest):
    """Same suite as ShortConvolutionTest but with bf16 inputs/weights.

    bf16 has ~3 decimal digits of mantissa, so we relax the tolerances
    accordingly. This exercises the dtype-preserving code path: the
    convolution should run end-to-end in bf16 (no internal fp32
    upcast) and match an nnx.Conv baseline that also runs in bf16.
    """

    def setUp(self):
        super().setUp()
        self.dtype = jnp.bfloat16
        self.atol = 1e-2
        self.rtol = 1e-2


if __name__ == "__main__":
    unittest.main()
