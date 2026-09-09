"""Correctness and donation coverage for the recurrent Pallas CoW clone."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.h0_clone import clone_slots_inplace


def _slow_clone(buffer, src, dst):
    """S5a reference: immutable gather/scatter over the whole buffer."""
    payload_dims = (1,) * (buffer.ndim - 1)
    value = jnp.where((src == 0).reshape((-1,) + payload_dims), buffer[dst], buffer[src])
    return buffer.at[dst].set(value)


class TestRecurrentH0Clone:
    def test_temporal_and_conv_match_slow_path(self):
        temporal = jnp.arange(6 * 2 * 4 * 4, dtype=jnp.float32).reshape(6, 2, 4, 4)
        conv = jnp.arange(6 * 12 * 3, dtype=jnp.bfloat16).reshape(6, 12, 3)
        src = jnp.array([1, 0, 4], dtype=jnp.int32)
        dst = jnp.array([2, 3, 5], dtype=jnp.int32)

        actual_temporal = jax.jit(clone_slots_inplace)(temporal, src, dst)
        actual_conv = jax.jit(clone_slots_inplace)(conv, src, dst)

        np.testing.assert_array_equal(actual_temporal, _slow_clone(temporal, src, dst))
        np.testing.assert_array_equal(actual_conv, _slow_clone(conv, src, dst))

    def test_donated_buffer_can_feed_the_next_clone(self):
        @partial(jax.jit, donate_argnums=(0,))
        def clone_and_consume(buffer, src, dst):
            cloned = clone_slots_inplace(buffer, src, dst)
            return cloned, jnp.sum(cloned[dst])

        src = jnp.array([1], dtype=jnp.int32)
        dst = jnp.array([2], dtype=jnp.int32)
        first_input = jnp.arange(4 * 2 * 8, dtype=jnp.float32).reshape(4, 2, 8)
        expected_first = _slow_clone(first_input, src, dst)
        first, first_sum = clone_and_consume(first_input, src, dst)
        np.testing.assert_array_equal(first, expected_first)

        second_src = jnp.array([2], dtype=jnp.int32)
        second_dst = jnp.array([3], dtype=jnp.int32)
        second, second_sum = clone_and_consume(first, second_src, second_dst)
        expected_second = _slow_clone(expected_first, second_src, second_dst)
        np.testing.assert_array_equal(second, expected_second)
        assert float(first_sum) > 0.0
        assert float(second_sum) > 0.0
