"""Unit tests for ``ragged_gated_delta_rule_ref``.

These tests exist primarily as a debugging aid — each case isolates one
piece of the kernel's behaviour (gating math, ragged batching, initial-state
masking, padding, GQA, decode equivalence). When a test fails, the
``_python_reference`` function runs the same recurrence in straight Python
so you can step through token-by-token and compare against the kernel.

Run with:
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \\
        python -m pytest test/srt/test_ragged_gated_delta_rule_ref.py -v
"""

from __future__ import annotations

import os
import unittest

# Force CPU + 8 fake devices before importing JAX so explicit-axis meshes work.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.gdn import ragged_gated_delta_rule_ref

pytestmark = pytest.mark.cpu_only

# ---------------------------------------------------------------------------
# Helpers: a straight-Python reference for one request and a small fixture.
# ---------------------------------------------------------------------------


def _l2norm(x, eps=1e-6):
    x = x.astype(jnp.float32)
    return x / jnp.sqrt((x * x).sum(axis=-1, keepdims=True) + eps)


def _python_reference(
    mixed_qkv,  # [T, 2*n_kq*d_k + n_v*d_v]
    b,
    a,  # [T, n_v], [T, n_v]
    initial_state,  # [n_v, d_k, d_v] fp32
    A_log,
    dt_bias,  # [n_v]
    n_kq,
    n_v,
    d_k,
    d_v,
):
    """Token-by-token reference for a SINGLE request.

    Mirrors the kernel's math line for line, but with a Python ``for`` loop
    instead of ``lax.scan`` and no ragged-batch indexing — useful as an
    independent oracle when a test fails.
    """
    T = mixed_qkv.shape[0]
    key_dim = n_kq * d_k
    q = mixed_qkv[:, :key_dim]
    k = mixed_qkv[:, key_dim : 2 * key_dim]
    v = mixed_qkv[:, 2 * key_dim :]

    repeat = n_v // n_kq
    A = jnp.exp(A_log.astype(jnp.float32))
    scale = d_k**-0.5

    state = initial_state.astype(jnp.float32)
    outs = []
    for t in range(T):
        q_h = q[t].reshape(n_kq, d_k)
        k_h = k[t].reshape(n_kq, d_k)
        v_h = v[t].reshape(n_v, d_v)
        if repeat > 1:
            q_h = jnp.repeat(q_h, repeat, axis=0)
            k_h = jnp.repeat(k_h, repeat, axis=0)
        q_h = _l2norm(q_h) * scale
        k_h = _l2norm(k_h)
        v_h = v_h.astype(jnp.float32)
        beta = jax.nn.sigmoid(b[t].astype(jnp.float32))
        g = -A * jax.nn.softplus(a[t].astype(jnp.float32) + dt_bias.astype(jnp.float32))

        decay = jnp.exp(g)[:, None, None]
        state = state * decay
        kv_mem = (state * k_h[..., None]).sum(axis=-2)
        delta = (v_h - kv_mem) * beta[:, None]
        state = state + k_h[..., None] * delta[..., None, :]
        out = (state * q_h[..., None]).sum(axis=-2)  # [n_v, d_v]
        outs.append(out)
    output = jnp.stack(outs, axis=0)  # [T, n_v, d_v]
    return state, output.astype(mixed_qkv.dtype)


def _make_inputs(seed, total_tokens, n_kq, n_v, d_k, d_v, dtype=jnp.bfloat16):
    """Random fixture: returns (mixed_qkv, b, a, A_log, dt_bias)."""
    keys = jax.random.split(jax.random.key(seed), 5)
    conv_dim = 2 * n_kq * d_k + n_v * d_v
    mixed_qkv = jax.random.normal(keys[0], (total_tokens, conv_dim), dtype=dtype) * 0.5
    b = jax.random.normal(keys[1], (total_tokens, n_v), dtype=dtype) * 0.5
    a = jax.random.normal(keys[2], (total_tokens, n_v), dtype=dtype) * 0.5
    A_log = jax.random.normal(keys[3], (n_v,), dtype=jnp.float32) * 0.3
    dt_bias = jax.random.normal(keys[4], (n_v,), dtype=jnp.float32) * 0.3
    return mixed_qkv, b, a, A_log, dt_bias


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class RaggedGatedDeltaRuleRefTest(unittest.TestCase):
    # All tests share these shapes — small enough to inspect by hand.
    n_kq, n_v, d_k, d_v = 1, 2, 4, 8
    NUM_BLOCKS = 4  # state table size; slot 0 is the null block.

    # --- Case 1: single request, fresh prefill -----------------------------
    def test_single_request_matches_python_reference(self):
        """B=1, has_initial_state=False — the simplest case.

        The kernel and the Python reference should produce bit-identical
        outputs (same math, same dtypes) modulo bf16 rounding from the
        cast on the way out.
        """
        T = 6
        mixed_qkv, b, a, A_log, dt_bias = _make_inputs(
            0, T, self.n_kq, self.n_v, self.d_k, self.d_v
        )
        rec = jnp.zeros((self.NUM_BLOCKS, self.n_v, self.d_k, self.d_v), dtype=jnp.float32)

        new_rec, out = ragged_gated_delta_rule_ref(
            mixed_qkv,
            b,
            a,
            rec,
            A_log,
            dt_bias,
            cu_seqlens=jnp.array([0, T], dtype=jnp.int32),
            state_indices=jnp.array([1], dtype=jnp.int32),
            has_initial_state=jnp.array([False], dtype=jnp.bool_),
            n_kq=self.n_kq,
            n_v=self.n_v,
            d_k=self.d_k,
            d_v=self.d_v,
        )

        ref_state, ref_out = _python_reference(
            mixed_qkv,
            b,
            a,
            jnp.zeros((self.n_v, self.d_k, self.d_v), dtype=jnp.float32),
            A_log,
            dt_bias,
            self.n_kq,
            self.n_v,
            self.d_k,
            self.d_v,
        )

        np.testing.assert_allclose(out, ref_out, atol=1e-3, rtol=1e-3)
        # `new_rec` is the full pool table (kernel scatters internally).
        # state_indices=[1] → the updated slot is index 1.
        np.testing.assert_allclose(new_rec[1], ref_state, atol=1e-4, rtol=1e-4)

    # --- Case 2: with initial state ----------------------------------------
    def test_initial_state_is_picked_up(self):
        """has_initial_state=True — the kernel should gather slot's prior state.

        Plant a non-zero state in slot 2, run with has_initial_state=True
        vs False, expect different outputs (False zeroes it).
        """
        T = 4
        mixed_qkv, b, a, A_log, dt_bias = _make_inputs(
            1, T, self.n_kq, self.n_v, self.d_k, self.d_v
        )
        rec = jnp.zeros((self.NUM_BLOCKS, self.n_v, self.d_k, self.d_v), dtype=jnp.float32)
        # Plant a deterministic prior state in slot 2.
        prior = jax.random.normal(
            jax.random.key(99), (self.n_v, self.d_k, self.d_v), dtype=jnp.float32
        )
        rec = rec.at[2].set(prior)

        common = dict(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            A_log=A_log,
            dt_bias=dt_bias,
            cu_seqlens=jnp.array([0, T], dtype=jnp.int32),
            state_indices=jnp.array([2], dtype=jnp.int32),
            n_kq=self.n_kq,
            n_v=self.n_v,
            d_k=self.d_k,
            d_v=self.d_v,
        )
        # With has_initial_state=True: prior should flow in.
        _, out_with = ragged_gated_delta_rule_ref(
            recurrent_state=rec,
            has_initial_state=jnp.array([True]),
            **common,
        )
        ref_with = _python_reference(
            mixed_qkv,
            b,
            a,
            prior,
            A_log,
            dt_bias,
            self.n_kq,
            self.n_v,
            self.d_k,
            self.d_v,
        )[1]
        np.testing.assert_allclose(out_with, ref_with, atol=1e-3, rtol=1e-3)

        # With has_initial_state=False: prior should be ignored.
        _, out_without = ragged_gated_delta_rule_ref(
            recurrent_state=rec,
            has_initial_state=jnp.array([False]),
            **common,
        )
        ref_without = _python_reference(
            mixed_qkv,
            b,
            a,
            jnp.zeros_like(prior),
            A_log,
            dt_bias,
            self.n_kq,
            self.n_v,
            self.d_k,
            self.d_v,
        )[1]
        np.testing.assert_allclose(out_without, ref_without, atol=1e-3, rtol=1e-3)

        # Sanity: the two should not coincide for non-zero prior.
        self.assertFalse(jnp.allclose(out_with, out_without, atol=1e-3))

    # --- Case 3: ragged batching independence ------------------------------
    def test_ragged_batching_matches_per_request_runs(self):
        """Two reqs of different lengths in one packed batch produce the
        same outputs as running each request independently.

        If this test passes, sequence boundaries are clean: no token from
        req-0 ever leaks into req-1's recurrent state and vice versa.
        """
        lens = [3, 5]
        T = sum(lens)
        mixed_qkv, b, a, A_log, dt_bias = _make_inputs(
            2, T, self.n_kq, self.n_v, self.d_k, self.d_v
        )
        rec = jnp.zeros((self.NUM_BLOCKS, self.n_v, self.d_k, self.d_v), dtype=jnp.float32)
        cu = jnp.array([0, lens[0], lens[0] + lens[1]], dtype=jnp.int32)

        new_rec, out = ragged_gated_delta_rule_ref(
            mixed_qkv,
            b,
            a,
            rec,
            A_log,
            dt_bias,
            cu_seqlens=cu,
            state_indices=jnp.array([1, 2], dtype=jnp.int32),
            has_initial_state=jnp.array([False, False]),
            n_kq=self.n_kq,
            n_v=self.n_v,
            d_k=self.d_k,
            d_v=self.d_v,
        )

        # Run each request through the Python reference and stack.
        zero_state = jnp.zeros((self.n_v, self.d_k, self.d_v), dtype=jnp.float32)
        ref_state_0, ref_out_0 = _python_reference(
            mixed_qkv[: lens[0]],
            b[: lens[0]],
            a[: lens[0]],
            zero_state,
            A_log,
            dt_bias,
            self.n_kq,
            self.n_v,
            self.d_k,
            self.d_v,
        )
        ref_state_1, ref_out_1 = _python_reference(
            mixed_qkv[lens[0] :],
            b[lens[0] :],
            a[lens[0] :],
            zero_state,
            A_log,
            dt_bias,
            self.n_kq,
            self.n_v,
            self.d_k,
            self.d_v,
        )
        ref_out = jnp.concatenate([ref_out_0, ref_out_1], axis=0)
        np.testing.assert_allclose(out, ref_out, atol=1e-3, rtol=1e-3)
        # `new_rec` is the full pool table; pluck the per-request slots
        # (state_indices=[1, 2]).
        np.testing.assert_allclose(new_rec[1], ref_state_0, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(new_rec[2], ref_state_1, atol=1e-4, rtol=1e-4)

    # --- Case 4: padding tokens are ignored --------------------------------
    def test_padding_tokens_do_not_mutate_state(self):
        """Tokens beyond cu_seqlens[-1] are padding; their writes are masked
        off by ``valid_mask``, so the per-seq new state should equal the
        no-padding result.
        """
        real_T = 4
        pad = 3
        T = real_T + pad
        mixed_qkv, b, a, A_log, dt_bias = _make_inputs(
            3, T, self.n_kq, self.n_v, self.d_k, self.d_v
        )
        rec = jnp.zeros((self.NUM_BLOCKS, self.n_v, self.d_k, self.d_v), dtype=jnp.float32)

        new_rec_padded, _ = ragged_gated_delta_rule_ref(
            mixed_qkv,
            b,
            a,
            rec,
            A_log,
            dt_bias,
            cu_seqlens=jnp.array([0, real_T], dtype=jnp.int32),  # last_valid_loc = 4
            state_indices=jnp.array([1], dtype=jnp.int32),
            has_initial_state=jnp.array([False]),
            n_kq=self.n_kq,
            n_v=self.n_v,
            d_k=self.d_k,
            d_v=self.d_v,
        )
        new_rec_clean, _ = ragged_gated_delta_rule_ref(
            mixed_qkv[:real_T],
            b[:real_T],
            a[:real_T],
            rec,
            A_log,
            dt_bias,
            cu_seqlens=jnp.array([0, real_T], dtype=jnp.int32),
            state_indices=jnp.array([1], dtype=jnp.int32),
            has_initial_state=jnp.array([False]),
            n_kq=self.n_kq,
            n_v=self.n_v,
            d_k=self.d_k,
            d_v=self.d_v,
        )
        np.testing.assert_allclose(new_rec_padded[0], new_rec_clean[0], atol=1e-5, rtol=1e-5)

    # --- Case 5: GQA expansion (n_v > n_kq) --------------------------------
    def test_gqa_expansion_matches_pre_repeated(self):
        """When n_v > n_kq, the kernel repeats Q/K to n_v heads internally.

        Run with (n_kq=2, n_v=4) and compare against running with q/k
        manually pre-repeated and (n_kq=4, n_v=4). Outputs should match.
        """
        n_kq, n_v, d_k, d_v = 2, 4, 4, 6
        repeat = n_v // n_kq
        T = 5
        key_dim = n_kq * d_k
        keys = jax.random.split(jax.random.key(4), 5)
        mixed_qkv_in = (
            jax.random.normal(keys[0], (T, 2 * key_dim + n_v * d_v), dtype=jnp.bfloat16) * 0.5
        )
        b = jax.random.normal(keys[1], (T, n_v), dtype=jnp.bfloat16) * 0.5
        a = jax.random.normal(keys[2], (T, n_v), dtype=jnp.bfloat16) * 0.5
        A_log = jax.random.normal(keys[3], (n_v,), dtype=jnp.float32) * 0.3
        dt_bias = jax.random.normal(keys[4], (n_v,), dtype=jnp.float32) * 0.3
        rec = jnp.zeros((2, n_v, d_k, d_v), dtype=jnp.float32)

        new_rec_a, out_a = ragged_gated_delta_rule_ref(
            mixed_qkv_in,
            b,
            a,
            rec,
            A_log,
            dt_bias,
            cu_seqlens=jnp.array([0, T], dtype=jnp.int32),
            state_indices=jnp.array([1], dtype=jnp.int32),
            has_initial_state=jnp.array([False]),
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
        )

        # Now build the equivalent (n_kq=n_v) input: repeat q and k across
        # heads, leave v unchanged.
        q = mixed_qkv_in[:, :key_dim].reshape(T, n_kq, d_k)
        k = mixed_qkv_in[:, key_dim : 2 * key_dim].reshape(T, n_kq, d_k)
        v = mixed_qkv_in[:, 2 * key_dim :]
        q_rep = jnp.repeat(q, repeat, axis=1).reshape(T, n_v * d_k)
        k_rep = jnp.repeat(k, repeat, axis=1).reshape(T, n_v * d_k)
        mixed_qkv_eq = jnp.concatenate([q_rep, k_rep, v], axis=-1)
        new_rec_b, out_b = ragged_gated_delta_rule_ref(
            mixed_qkv_eq,
            b,
            a,
            rec,
            A_log,
            dt_bias,
            cu_seqlens=jnp.array([0, T], dtype=jnp.int32),
            state_indices=jnp.array([1], dtype=jnp.int32),
            has_initial_state=jnp.array([False]),
            n_kq=n_v,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
        )
        np.testing.assert_allclose(out_a, out_b, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(new_rec_a, new_rec_b, atol=1e-4, rtol=1e-4)

    # --- Case 6: prefill-then-decode equivalence ---------------------------
    def test_prefill_then_decode_equals_full_prefill(self):
        """Run T tokens as one prefill, vs the same T tokens broken into a
        prefill (T-1 tokens) followed by a single decode step. The final
        state and last output should match.

        This is the contract decode mode relies on: continuing a sequence
        token-by-token from a saved state must produce the same trajectory
        as scanning the whole thing in one go.
        """
        T = 6
        mixed_qkv, b, a, A_log, dt_bias = _make_inputs(
            5, T, self.n_kq, self.n_v, self.d_k, self.d_v
        )
        rec = jnp.zeros((self.NUM_BLOCKS, self.n_v, self.d_k, self.d_v), dtype=jnp.float32)

        # Path A: full prefill.
        new_rec_full, out_full = ragged_gated_delta_rule_ref(
            mixed_qkv,
            b,
            a,
            rec,
            A_log,
            dt_bias,
            cu_seqlens=jnp.array([0, T], dtype=jnp.int32),
            state_indices=jnp.array([1], dtype=jnp.int32),
            has_initial_state=jnp.array([False]),
            n_kq=self.n_kq,
            n_v=self.n_v,
            d_k=self.d_k,
            d_v=self.d_v,
        )

        # Path B: prefill T-1 tokens, then one decode step.
        new_rec_pref, _ = ragged_gated_delta_rule_ref(
            mixed_qkv[: T - 1],
            b[: T - 1],
            a[: T - 1],
            rec,
            A_log,
            dt_bias,
            cu_seqlens=jnp.array([0, T - 1], dtype=jnp.int32),
            state_indices=jnp.array([1], dtype=jnp.int32),
            has_initial_state=jnp.array([False]),
            n_kq=self.n_kq,
            n_v=self.n_v,
            d_k=self.d_k,
            d_v=self.d_v,
        )
        # `new_rec_pref` is already the full pool table with slot 1 updated —
        # pass it straight in as the state table for the decode step.
        new_rec_dec, out_dec = ragged_gated_delta_rule_ref(
            mixed_qkv[T - 1 :],
            b[T - 1 :],
            a[T - 1 :],
            new_rec_pref,
            A_log,
            dt_bias,
            cu_seqlens=jnp.array([0, 1], dtype=jnp.int32),
            state_indices=jnp.array([1], dtype=jnp.int32),
            has_initial_state=jnp.array([True]),  # continuation
            n_kq=self.n_kq,
            n_v=self.n_v,
            d_k=self.d_k,
            d_v=self.d_v,
        )

        # Last-token output and final state should coincide; per-request slot
        # is index 1 in the full pool table.
        np.testing.assert_allclose(out_full[-1], out_dec[0], atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(new_rec_full[1], new_rec_dec[1], atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
