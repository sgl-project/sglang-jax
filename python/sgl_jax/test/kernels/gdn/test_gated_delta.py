"""Unit tests for the primitives + kernels in :mod:`gated_delta`.

Covers ``_l2norm``, ``_gated_delta_step``, ``jax_causal_conv1d_prefill``,
``jax_causal_conv1d_update``, and ``decode_gated_delta_rule_ref``. The
ragged kernel is covered separately in ``test_ragged_gated_delta_rule_ref.py``.

Run with:
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \\
        python -m pytest test/srt/test_gated_delta.py -v
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.gdn import (
    decode_gated_delta_rule_ref,
    jax_causal_conv1d_prefill,
    jax_causal_conv1d_update,
    ragged_gated_delta_rule_ref,
)
from sgl_jax.srt.kernels.gdn.gated_delta import _gated_delta_step, _l2norm


class L2NormTest(unittest.TestCase):
    def test_unit_vectors(self):
        """3-4-5 triangle: ||(3,4)|| = 5, normalized = (0.6, 0.8)."""
        x = jnp.array([[3.0, 4.0]], dtype=jnp.float32)
        y = _l2norm(x)
        np.testing.assert_allclose(y, [[0.6, 0.8]], atol=1e-6)

    def test_normalizes_along_last_axis_only(self):
        """Each row gets its own norm; 2D shape preserved."""
        x = jnp.array([[1.0, 0.0], [0.0, 5.0], [3.0, 4.0]], dtype=jnp.float32)
        y = _l2norm(x)
        # Each row's L2 norm is ~1 (not ||x.flatten()||).
        np.testing.assert_allclose(jnp.linalg.norm(y, axis=-1), [1.0, 1.0, 1.0], atol=1e-5)

    def test_zero_vector_eps_safe(self):
        """Eps prevents NaN on the all-zeros input."""
        y = _l2norm(jnp.zeros((1, 4), dtype=jnp.float32))
        self.assertTrue(bool(jnp.all(jnp.isfinite(y))))


class GatedDeltaStepTest(unittest.TestCase):
    """Math sanity for the leading-dim-agnostic recurrence primitive."""

    def test_zero_state_no_decay_full_beta(self):
        """With S=0, β=1, g=0: new_state = k⊗v, out = (q·k)·v."""
        H, K, V = 2, 3, 4
        rng = jax.random.split(jax.random.key(0), 3)
        q = jax.random.normal(rng[0], (H, K), dtype=jnp.float32)
        k = jax.random.normal(rng[1], (H, K), dtype=jnp.float32)
        v = jax.random.normal(rng[2], (H, V), dtype=jnp.float32)
        g = jnp.zeros((H,), dtype=jnp.float32)
        beta = jnp.ones((H,), dtype=jnp.float32)
        state = jnp.zeros((H, K, V), dtype=jnp.float32)

        new_state, out = _gated_delta_step(state, q, k, v, g, beta)

        # k ⊗ v
        expected_state = k[..., None] * v[..., None, :]
        np.testing.assert_allclose(new_state, expected_state, atol=1e-5)

        # out = (state @ q) summed over K with new state
        # = ((q·k) · v) per head
        qk = (q * k).sum(axis=-1, keepdims=True)  # [H, 1]
        expected_out = qk * v
        np.testing.assert_allclose(out, expected_out, atol=1e-5)

    def test_full_decay_drops_state(self):
        """g = -inf (decay = 0) means prior state is forgotten — only the
        new k⊗(β·v) survives."""
        H, K, V = 1, 2, 3
        rng = jax.random.split(jax.random.key(1), 4)
        q = jax.random.normal(rng[0], (H, K))
        k = jax.random.normal(rng[1], (H, K))
        v = jax.random.normal(rng[2], (H, V))
        beta = jnp.full((H,), 0.5)
        # Plant non-zero prior state, then decay it away.
        state = jax.random.normal(rng[3], (H, K, V))

        # exp(-100) ≈ 0
        g = jnp.full((H,), -100.0)
        new_state, _ = _gated_delta_step(state, q, k, v, g, beta)
        # Result should be ~ k ⊗ (β·v).
        expected = k[..., None] * (beta[..., None] * v)[..., None, :]
        np.testing.assert_allclose(new_state, expected, atol=1e-5)

    def test_leading_dim_agnostic_batched_equals_unbatched(self):
        """Calling with [B, H, ...] batched should equal calling per-element
        with [H, ...] and stacking — the function is leading-dim-agnostic."""
        B, H, K, V = 3, 2, 4, 6
        rng = jax.random.split(jax.random.key(2), 6)
        state = jax.random.normal(rng[0], (B, H, K, V))
        q = jax.random.normal(rng[1], (B, H, K))
        k = jax.random.normal(rng[2], (B, H, K))
        v = jax.random.normal(rng[3], (B, H, V))
        g = jax.random.normal(rng[4], (B, H)) * 0.1
        beta = jax.random.normal(rng[5], (B, H)) * 0.5

        # Batched call.
        ns_batch, out_batch = _gated_delta_step(state, q, k, v, g, beta)

        # Per-element calls.
        ns_each, out_each = [], []
        for i in range(B):
            ns, o = _gated_delta_step(state[i], q[i], k[i], v[i], g[i], beta[i])
            ns_each.append(ns)
            out_each.append(o)
        ns_stack = jnp.stack(ns_each, axis=0)
        out_stack = jnp.stack(out_each, axis=0)

        np.testing.assert_allclose(ns_batch, ns_stack, atol=1e-5)
        np.testing.assert_allclose(out_batch, out_stack, atol=1e-5)


class CausalConv1dUpdateTest(unittest.TestCase):
    def test_matches_window_dot(self):
        """y = window · weight (per-channel) where window = [state, x_new]."""
        B, D, K = 2, 3, 4
        rng = jax.random.split(jax.random.key(10), 3)
        x = jax.random.normal(rng[0], (B, D), dtype=jnp.float32)
        state = jax.random.normal(rng[1], (B, D, K - 1), dtype=jnp.float32)
        weight = jax.random.normal(rng[2], (D, K), dtype=jnp.float32)

        y, new_state = jax_causal_conv1d_update(x, state, weight, bias=None)

        # Reference: window = [state | x_new], y = sum(window * weight) over K.
        window = jnp.concatenate([state, x[..., None]], axis=-1)  # [B, D, K]
        expected_y = (window * weight[None]).sum(axis=-1)
        np.testing.assert_allclose(y, expected_y, atol=1e-5)
        np.testing.assert_allclose(new_state, window[..., 1:], atol=0)

    def test_silu_activation(self):
        """activation='silu' applies SiLU on top of the linear output."""
        B, D, K = 1, 2, 3
        rng = jax.random.split(jax.random.key(11), 3)
        x = jax.random.normal(rng[0], (B, D))
        state = jax.random.normal(rng[1], (B, D, K - 1))
        weight = jax.random.normal(rng[2], (D, K))

        y_lin, _ = jax_causal_conv1d_update(x, state, weight, bias=None)
        y_silu, _ = jax_causal_conv1d_update(x, state, weight, bias=None, activation="silu")
        np.testing.assert_allclose(y_silu, jax.nn.silu(y_lin), atol=1e-5)


class CausalConv1dPrefillTest(unittest.TestCase):
    """The depthwise causal conv1d. Boundary handling is the subtle bit."""

    def _naive_conv(self, x, weight, init_left=None):
        """Reference: per-channel causal conv with optional left-pad state.

        x: [D, T], weight: [D, K], init_left: [D, K-1] or None (zero pad).
        Returns y: [D, T].
        """
        D, T = x.shape
        K = weight.shape[1]
        left = jnp.zeros((D, K - 1), dtype=x.dtype) if init_left is None else init_left
        padded = jnp.concatenate([left, x], axis=-1)  # [D, T+K-1]
        out = []
        for t in range(T):
            window = padded[:, t : t + K]  # [D, K]
            out.append((window * weight).sum(axis=-1))
        return jnp.stack(out, axis=-1)  # [D, T]

    def test_single_request_no_state_matches_naive(self):
        D, K = 3, 3
        T = 5
        rng = jax.random.split(jax.random.key(20), 2)
        x = jax.random.normal(rng[0], (D, T), dtype=jnp.float32)
        weight = jax.random.normal(rng[1], (D, K), dtype=jnp.float32)

        y, final = jax_causal_conv1d_prefill(
            x,
            weight,
            bias=None,
            cu_seqlens=jnp.array([0, T], dtype=jnp.int32),
            conv_state=None,
            state_indices=None,
        )
        np.testing.assert_allclose(y, self._naive_conv(x, weight), atol=1e-5)
        # Final state should be the last K-1 tokens of x.
        np.testing.assert_allclose(final[0], x[:, -(K - 1) :], atol=1e-5)

    def test_initial_state_carried_in(self):
        """First K-1 outputs should mix in conv_state contents."""
        D, K = 2, 3
        T = 4
        rng = jax.random.split(jax.random.key(21), 3)
        x = jax.random.normal(rng[0], (D, T))
        weight = jax.random.normal(rng[1], (D, K))
        # Slot 0 = null block (zeros), slot 1 carries non-zero state.
        prior = jax.random.normal(rng[2], (D, K - 1))
        conv_state = jnp.zeros((2, D, K - 1)).at[1].set(prior)

        y, final = jax_causal_conv1d_prefill(
            x,
            weight,
            bias=None,
            cu_seqlens=jnp.array([0, T], dtype=jnp.int32),
            conv_state=conv_state,
            state_indices=jnp.array([1], dtype=jnp.int32),
        )
        np.testing.assert_allclose(y, self._naive_conv(x, weight, init_left=prior), atol=1e-5)
        # Final state still equals the last K-1 of x (request is long enough).
        np.testing.assert_allclose(final[0], x[:, -(K - 1) :], atol=1e-5)

    def test_multi_request_boundary_isolation(self):
        """Token 0 of request 1 must NOT see any token from request 0."""
        D, K = 1, 3
        # Two requests of length 3 each, packed into x of length 6.
        # Set channel 0 of req0 to all 100 (poison) and req1 to all 1.
        x = jnp.zeros((D, 6))
        x = x.at[0, :3].set(100.0)
        x = x.at[0, 3:].set(1.0)
        weight = jnp.ones((D, K))  # straight sum of window

        y, _ = jax_causal_conv1d_prefill(
            x,
            weight,
            bias=None,
            cu_seqlens=jnp.array([0, 3, 6], dtype=jnp.int32),
            conv_state=None,
            state_indices=None,
        )
        # Req1's token 0 (global idx 3) should sum [0, 0, 1] = 1, NOT 100+1+1.
        # Req1's token 1 should sum [0, 1, 1] = 2.
        # Req1's token 2 should sum [1, 1, 1] = 3.
        np.testing.assert_allclose(y[0, 3:], [1.0, 2.0, 3.0], atol=1e-5)
        # Req0 unaffected: 100, 200, 300.
        np.testing.assert_allclose(y[0, :3], [100.0, 200.0, 300.0], atol=1e-5)

    def test_short_request_left_padded_from_state(self):
        """A request shorter than K-1 has its final_state assembled from
        (state-left-pad) + (whatever real tokens it has)."""
        D, K = 1, 4  # K-1 = 3
        T = 2  # request shorter than K-1
        x = jnp.array([[10.0, 20.0]])
        weight = jnp.zeros((D, K))  # output unused, focus on final_state
        prior = jnp.array([[7.0, 8.0, 9.0]])  # state[0,1,2] = newest at idx 2

        conv_state = jnp.zeros((1, D, K - 1)).at[0].set(prior)
        _, final = jax_causal_conv1d_prefill(
            x,
            weight,
            bias=None,
            cu_seqlens=jnp.array([0, T], dtype=jnp.int32),
            conv_state=conv_state,
            state_indices=jnp.array([0], dtype=jnp.int32),
        )
        # State holds the K-1=3 most recent tokens BEFORE this batch, newest
        # at index K-2=2. Before this batch the (logical) tail is
        # ..., prior[0]=7, prior[1]=8, prior[2]=9. After appending 10, 20
        # the new most-recent K-1=3 logical tokens are [9, 10, 20].
        np.testing.assert_allclose(final[0, 0], [9.0, 10.0, 20.0], atol=1e-5)

    def test_short_request_left_padded_with_zeros_when_no_state(self):
        """Same as above but with no prior state — the K-1-T missing
        lookback positions zero-pad rather than pulling from a slot."""
        D, K = 1, 4  # K-1 = 3 left-pad slots needed
        T = 2
        x = jnp.array([[10.0, 20.0]])
        weight = jnp.zeros((D, K))

        _, final = jax_causal_conv1d_prefill(
            x,
            weight,
            bias=None,
            cu_seqlens=jnp.array([0, T], dtype=jnp.int32),
            conv_state=None,
            state_indices=None,
        )
        # Logical stream is [pad=0, 10, 20]; last K-1=3 = [0, 10, 20].
        np.testing.assert_allclose(final[0, 0], [0.0, 10.0, 20.0], atol=1e-5)

    def test_kernel_size_1_no_lookback(self):
        """K=1 means depthwise per-token multiply with no temporal mixing.
        ``final_state`` has shape ``(B, D, 0)`` since there's no state to keep."""
        D, K = 2, 1
        T = 4
        rng = jax.random.split(jax.random.key(22), 2)
        x = jax.random.normal(rng[0], (D, T))
        weight = jax.random.normal(rng[1], (D, K))

        y, final = jax_causal_conv1d_prefill(
            x,
            weight,
            bias=None,
            cu_seqlens=jnp.array([0, T], dtype=jnp.int32),
            conv_state=None,
            state_indices=None,
        )
        # y[d, t] = x[d, t] * weight[d, 0].
        np.testing.assert_allclose(y, x * weight, atol=1e-5)
        self.assertEqual(final.shape, (1, D, 0))


class DecodeGatedDeltaRuleRefTest(unittest.TestCase):
    """The decode kernel is the parallel-across-B specialisation of the
    ragged kernel. It must be numerically equivalent to running the
    ragged kernel with cu_seqlens = arange(B+1)."""

    def test_matches_ragged_with_singleton_seqs(self):
        """decode_gated_delta_rule_ref(...) ==
        ragged_gated_delta_rule_ref(cu_seqlens=arange(B+1), has_initial_state=True)."""
        n_kq, n_v, d_k, d_v = 1, 2, 4, 8
        B = 5
        conv_dim = 2 * n_kq * d_k + n_v * d_v
        rng = jax.random.split(jax.random.key(30), 6)
        mq = jax.random.normal(rng[0], (B, conv_dim), dtype=jnp.bfloat16) * 0.3
        b = jax.random.normal(rng[1], (B, n_v), dtype=jnp.bfloat16) * 0.5
        a = jax.random.normal(rng[2], (B, n_v), dtype=jnp.bfloat16) * 0.5
        A_log = jax.random.normal(rng[3], (n_v,)) * 0.3
        dt_bias = jax.random.normal(rng[4], (n_v,)) * 0.3
        rec = jax.random.normal(rng[5], (B + 1, n_v, d_k, d_v), dtype=jnp.float32) * 0.05
        si = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32)

        nr_d, out_d = decode_gated_delta_rule_ref(
            mq,
            b,
            a,
            rec,
            A_log,
            dt_bias,
            si,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
        )
        nr_r, out_r = ragged_gated_delta_rule_ref(
            mq,
            b,
            a,
            rec,
            A_log,
            dt_bias,
            cu_seqlens=jnp.arange(B + 1, dtype=jnp.int32),
            state_indices=si,
            has_initial_state=jnp.ones((B,), dtype=jnp.bool_),
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
        )
        np.testing.assert_allclose(out_d, out_r, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(nr_d, nr_r, atol=1e-4, rtol=1e-4)

    def test_output_shapes(self):
        """Per-request outputs at expected shapes/dtypes."""
        n_kq, n_v, d_k, d_v = 2, 4, 8, 16
        B = 3
        conv_dim = 2 * n_kq * d_k + n_v * d_v
        mq = jnp.ones((B, conv_dim), dtype=jnp.bfloat16) * 0.1
        b = jnp.zeros((B, n_v), dtype=jnp.bfloat16)
        a = jnp.zeros((B, n_v), dtype=jnp.bfloat16)
        A_log = jnp.zeros((n_v,), dtype=jnp.float32)
        dt_bias = jnp.zeros((n_v,), dtype=jnp.float32)
        rec = jnp.zeros((B + 1, n_v, d_k, d_v), dtype=jnp.float32)
        si = jnp.array([1, 2, 3], dtype=jnp.int32)

        new_rec, out = decode_gated_delta_rule_ref(
            mq,
            b,
            a,
            rec,
            A_log,
            dt_bias,
            si,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
        )
        self.assertEqual(out.shape, (B, n_v, d_v))
        self.assertEqual(out.dtype, jnp.bfloat16)
        self.assertEqual(new_rec.shape, (B, n_v, d_k, d_v))
        self.assertEqual(new_rec.dtype, jnp.float32)

    def test_gqa_matches_ragged_with_singletons(self):
        """GQA (n_v > n_kq, v_per_k > 1): decode kernel still equals the
        ragged kernel with cu_seqlens=arange(B+1). Q/K head expansion must
        happen inside both impls for them to agree."""
        n_kq, n_v, d_k, d_v = 2, 4, 8, 8
        B = 3
        conv_dim = 2 * n_kq * d_k + n_v * d_v
        rng = jax.random.split(jax.random.key(31), 6)
        mq = jax.random.normal(rng[0], (B, conv_dim), dtype=jnp.bfloat16) * 0.3
        b = jax.random.normal(rng[1], (B, n_v), dtype=jnp.bfloat16) * 0.5
        a = jax.random.normal(rng[2], (B, n_v), dtype=jnp.bfloat16) * 0.5
        A_log = jax.random.normal(rng[3], (n_v,)) * 0.3
        dt_bias = jax.random.normal(rng[4], (n_v,)) * 0.3
        rec = jax.random.normal(rng[5], (B + 1, n_v, d_k, d_v), dtype=jnp.float32) * 0.05
        si = jnp.array([1, 2, 3], dtype=jnp.int32)

        nr_d, out_d = decode_gated_delta_rule_ref(
            mq,
            b,
            a,
            rec,
            A_log,
            dt_bias,
            si,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
        )
        nr_r, out_r = ragged_gated_delta_rule_ref(
            mq,
            b,
            a,
            rec,
            A_log,
            dt_bias,
            cu_seqlens=jnp.arange(B + 1, dtype=jnp.int32),
            state_indices=si,
            has_initial_state=jnp.ones((B,), dtype=jnp.bool_),
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
        )
        np.testing.assert_allclose(out_d, out_r, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(nr_d, nr_r, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
