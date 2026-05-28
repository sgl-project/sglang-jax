"""Kernel-level unit tests for ``decode_simple_gla_fused``.

These tests exercise the fused DECODE Pallas kernel directly (no shard_map,
no full backend pipeline) against the JAX reference path
``fused_recurrent_simple_gla``. Integration with the LightningAttn backend
is covered separately by ``test/layers/test_lightning_backend.py``.

Run locally on Mac (no TPU):
    PALLAS_INTERPRET=1 pytest python/sgl_jax/test/kernels/simple_gla_fused_test.py -v

Run on TPU:
    pytest python/sgl_jax/test/kernels/simple_gla_fused_test.py -v
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Reference path (the kernel we are replacing).
from sgl_jax.srt.kernels.simple_gla.simple_gla import fused_recurrent_simple_gla

# Fused twin under test.
from sgl_jax.srt.kernels.simple_gla.simple_gla_fused import decode_simple_gla_fused

# K = V (simple GLA constraint); BK = BV = 128 statically in Pallas kernel.
_DEFAULT_DTYPE = jnp.float32
_ATOL = 1e-3
_RTOL = 1e-3


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────
def _make_g_gamma(num_heads: int, seed: int = 0) -> jax.Array:
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(-0.1, -0.01, size=(num_heads,)), dtype=jnp.float32)


def _make_buffer(
    total_slots: int,
    num_heads: int,
    head_dim: int,
    seed: int = 1,
    dtype: jnp.dtype = _DEFAULT_DTYPE,
) -> jax.Array:
    rng = np.random.default_rng(seed)
    buf = rng.normal(size=(total_slots, num_heads, head_dim, head_dim))
    # Slot 0 is the dummy — must be zero by RecurrentStatePool convention.
    buf[0] = 0
    return jnp.asarray(buf, dtype=dtype)


def _make_qkv(num_seqs, num_heads, head_dim, seed, dtype=_DEFAULT_DTYPE, scale=0.1):
    rng = np.random.default_rng(seed)
    q = jnp.asarray(rng.normal(size=(num_seqs, num_heads, head_dim)), dtype=dtype) * scale
    k = jnp.asarray(rng.normal(size=(num_seqs, num_heads, head_dim)), dtype=dtype) * scale
    v = jnp.asarray(rng.normal(size=(num_seqs, num_heads, head_dim)), dtype=dtype) * scale
    return q, k, v


def _baseline_decode(q, k, v, buf, indices, has_init, gamma):
    """Reproduce the old decode path: gather → fused_recurrent_simple_gla → scatter."""
    h0 = buf[indices]
    if has_init is not None:
        h0 = jnp.where(has_init[:, None, None, None], h0, 0.0)

    q_d = q[:, None, :, :]
    k_d = k[:, None, :, :]
    v_d = v[:, None, :, :]
    o_d, ht = fused_recurrent_simple_gla(
        q_d,
        k_d,
        v_d,
        g_gamma=gamma,
        initial_state=h0,
        output_final_state=True,
        scale=None,
    )
    keep_mask = (indices == 0).reshape(-1, 1, 1, 1)
    safe_val = jnp.where(keep_mask, buf[indices], ht)
    new_buf = buf.at[indices].set(safe_val)
    return o_d[:, 0, :, :], new_buf


def _run_pair(q, k, v, buf, indices, has_init, gamma):
    """Run reference + fused on snapshot copies; return (o_ref, buf_ref, o_fused, buf_fused)."""
    # Fused donates ``recurrent_buffer``; run baseline first on a fresh copy
    # so it can still read the original buf.
    buf_snapshot = jnp.asarray(np.asarray(buf))
    o_ref, buf_ref = _baseline_decode(q, k, v, buf_snapshot, indices, has_init, gamma)
    o_fused, buf_fused = decode_simple_gla_fused(
        q,
        k,
        v,
        recurrent_buffer=buf,
        recurrent_indices=indices,
        has_initial_state=has_init,
        g_gamma=gamma,
        scale=None,
    )
    return o_ref, buf_ref, o_fused, buf_fused


# ─────────────────────────────────────────────────────────────────────────
# Equivalence tests vs reference path
# ─────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "num_seqs,num_heads,head_dim",
    [
        (4, 2, 128),  # original tiny case
        (4, 8, 128),  # H=8 matches Ling-2.6-flash per-shard
        (1, 8, 128),  # single-token decode (smallest grid)
        (16, 8, 128),  # mid batch
        (32, 8, 128),  # production decode batch (128 concurrency / 4 dp shards)
    ],
)
def test_decode_basic_equivalence(num_seqs, num_heads, head_dim):
    """Fused kernel output matches reference for varying (N, H) at K=V=128."""
    total_slots = num_seqs + 1
    indices = jnp.arange(1, num_seqs + 1, dtype=jnp.int32)
    has_init = jnp.full((num_seqs,), True, dtype=jnp.bool_)

    q, k, v = _make_qkv(num_seqs, num_heads, head_dim, seed=50)
    buf = _make_buffer(total_slots, num_heads, head_dim, seed=51)
    gamma = _make_g_gamma(num_heads, seed=52)

    o_ref, buf_ref, o_fused, buf_fused = _run_pair(q, k, v, buf, indices, has_init, gamma)
    np.testing.assert_allclose(np.asarray(o_fused), np.asarray(o_ref), atol=_ATOL, rtol=_RTOL)
    np.testing.assert_allclose(np.asarray(buf_fused), np.asarray(buf_ref), atol=_ATOL, rtol=_RTOL)


def test_decode_dummy_slot_skipped():
    """``pool_idx == 0`` (dummy slot) must leave slot 0 contents bit-equal."""
    H, D = 8, 128
    num_seqs = 3
    total_slots = num_seqs + 1
    indices = jnp.asarray([1, 0, 2], dtype=jnp.int32)  # middle one targets dummy slot
    has_init = jnp.asarray([True, True, True], dtype=jnp.bool_)

    q, k, v = _make_qkv(num_seqs, H, D, seed=60)
    buf = _make_buffer(total_slots, H, D, seed=61)
    gamma = _make_g_gamma(H, seed=62)
    buf_orig = np.asarray(buf)  # snapshot for post-call asserts on buf[0]

    o_ref, buf_ref, o_fused, buf_fused = _run_pair(q, k, v, buf, indices, has_init, gamma)

    np.testing.assert_allclose(np.asarray(o_fused), np.asarray(o_ref), atol=_ATOL, rtol=_RTOL)
    # Dummy slot 0 must be bit-identical to the pre-call buffer.
    np.testing.assert_array_equal(np.asarray(buf_fused[0]), buf_orig[0])
    # Other slots: kernel result matches reference.
    np.testing.assert_allclose(
        np.asarray(buf_fused[1]), np.asarray(buf_ref[1]), atol=_ATOL, rtol=_RTOL
    )
    np.testing.assert_allclose(
        np.asarray(buf_fused[2]), np.asarray(buf_ref[2]), atol=_ATOL, rtol=_RTOL
    )


def test_decode_has_initial_state_false():
    """``has_initial_state == False`` ⇒ treat gathered state as zero."""
    H, D = 8, 128
    num_seqs = 4
    total_slots = num_seqs + 1
    indices = jnp.asarray([1, 2, 3, 4], dtype=jnp.int32)
    has_init = jnp.asarray([False, False, False, False], dtype=jnp.bool_)

    q, k, v = _make_qkv(num_seqs, H, D, seed=70)
    buf = _make_buffer(total_slots, H, D, seed=71)
    gamma = _make_g_gamma(H, seed=72)

    o_ref, buf_ref, o_fused, buf_fused = _run_pair(q, k, v, buf, indices, has_init, gamma)
    np.testing.assert_allclose(np.asarray(o_fused), np.asarray(o_ref), atol=_ATOL, rtol=_RTOL)
    np.testing.assert_allclose(np.asarray(buf_fused), np.asarray(buf_ref), atol=_ATOL, rtol=_RTOL)


def test_decode_mixed_init_and_dummy():
    """Mixed has_init / pool_idx == 0 within one batch — the most realistic case
    that hits all three conditional branches simultaneously (warm seq with state,
    cold seq with no state, dummy padding slot)."""
    H, D = 8, 128
    num_seqs = 6
    total_slots = num_seqs + 2
    # mix:  warm-with-state, dummy, cold-no-state, warm-with-state, dummy, cold-no-state
    indices = jnp.asarray([1, 0, 3, 4, 0, 6], dtype=jnp.int32)
    has_init = jnp.asarray([True, True, False, True, False, False], dtype=jnp.bool_)

    q, k, v = _make_qkv(num_seqs, H, D, seed=80)
    buf = _make_buffer(total_slots, H, D, seed=81)
    gamma = _make_g_gamma(H, seed=82)
    buf_orig = np.asarray(buf)

    o_ref, buf_ref, o_fused, buf_fused = _run_pair(q, k, v, buf, indices, has_init, gamma)

    np.testing.assert_allclose(np.asarray(o_fused), np.asarray(o_ref), atol=_ATOL, rtol=_RTOL)
    # Dummy slot still zero.
    np.testing.assert_array_equal(np.asarray(buf_fused[0]), buf_orig[0])
    # All non-touched slots (2, 5, 7) untouched.
    for untouched in [2, 5, 7]:
        np.testing.assert_array_equal(np.asarray(buf_fused[untouched]), buf_orig[untouched])
    # All touched non-dummy slots match reference.
    for touched in [1, 3, 4, 6]:
        np.testing.assert_allclose(
            np.asarray(buf_fused[touched]),
            np.asarray(buf_ref[touched]),
            atol=_ATOL,
            rtol=_RTOL,
        )


# ─────────────────────────────────────────────────────────────────────────
# Behavioral tests (buffer corruption, dummy invariant, determinism, propagation)
# ─────────────────────────────────────────────────────────────────────────
def test_decode_untouched_slots_unchanged():
    """Slots not in ``recurrent_indices`` must be byte-identical after the call."""
    H, D = 8, 128
    total_slots = 16
    # Only touch slots 1, 4, 9 — leave 0 (dummy), 2-3, 5-8, 10-15 untouched.
    indices = jnp.asarray([1, 4, 9], dtype=jnp.int32)
    has_init = jnp.asarray([True, True, True], dtype=jnp.bool_)
    num_seqs = indices.shape[0]

    q, k, v = _make_qkv(num_seqs, H, D, seed=90)
    buf = _make_buffer(total_slots, H, D, seed=91)
    gamma = _make_g_gamma(H, seed=92)
    buf_orig = np.asarray(buf)

    _o, buf_fused = decode_simple_gla_fused(
        q,
        k,
        v,
        recurrent_buffer=buf,
        recurrent_indices=indices,
        has_initial_state=has_init,
        g_gamma=gamma,
        scale=None,
    )
    touched = {1, 4, 9}
    for s in range(total_slots):
        if s in touched:
            continue
        np.testing.assert_array_equal(
            np.asarray(buf_fused[s]),
            buf_orig[s],
            err_msg=f"slot {s} (untouched) was modified",
        )


def test_decode_deterministic():
    """Same inputs across two calls produce bit-identical outputs (donated input
    means we feed a fresh copy on the second call)."""
    H, D = 8, 128
    num_seqs = 8
    total_slots = num_seqs + 1
    indices = jnp.arange(1, num_seqs + 1, dtype=jnp.int32)
    has_init = jnp.full((num_seqs,), True, dtype=jnp.bool_)

    q, k, v = _make_qkv(num_seqs, H, D, seed=100)
    buf_np = np.asarray(_make_buffer(total_slots, H, D, seed=101))
    gamma = _make_g_gamma(H, seed=102)

    o1, buf1 = decode_simple_gla_fused(
        q,
        k,
        v,
        recurrent_buffer=jnp.asarray(buf_np),
        recurrent_indices=indices,
        has_initial_state=has_init,
        g_gamma=gamma,
        scale=None,
    )
    o2, buf2 = decode_simple_gla_fused(
        q,
        k,
        v,
        recurrent_buffer=jnp.asarray(buf_np),
        recurrent_indices=indices,
        has_initial_state=has_init,
        g_gamma=gamma,
        scale=None,
    )
    np.testing.assert_array_equal(np.asarray(o1), np.asarray(o2))
    np.testing.assert_array_equal(np.asarray(buf1), np.asarray(buf2))


def test_decode_state_propagation_multistep():
    """Chain N successive decode calls and verify the state evolves the same way
    in fused vs reference (catches scatter ordering / aliasing bugs)."""
    H, D = 8, 128
    num_seqs = 4
    n_steps = 5
    total_slots = num_seqs + 1
    indices = jnp.arange(1, num_seqs + 1, dtype=jnp.int32)

    # First step: cold (has_init=False). Subsequent steps: warm.
    buf_ref = _make_buffer(total_slots, H, D, seed=110)
    buf_fused = jnp.asarray(np.asarray(buf_ref))
    gamma = _make_g_gamma(H, seed=111)

    for step in range(n_steps):
        has_init = jnp.full((num_seqs,), step > 0, dtype=jnp.bool_)
        q, k, v = _make_qkv(num_seqs, H, D, seed=120 + step)

        # Reference branch
        o_ref, buf_ref_new = _baseline_decode(q, k, v, buf_ref, indices, has_init, gamma)
        buf_ref = buf_ref_new

        # Fused branch (donates its buf; pass fresh copy each step)
        buf_fused_in = jnp.asarray(np.asarray(buf_fused))
        o_fused, buf_fused = decode_simple_gla_fused(
            q,
            k,
            v,
            recurrent_buffer=buf_fused_in,
            recurrent_indices=indices,
            has_initial_state=has_init,
            g_gamma=gamma,
            scale=None,
        )

        np.testing.assert_allclose(
            np.asarray(o_fused),
            np.asarray(o_ref),
            atol=_ATOL,
            rtol=_RTOL,
            err_msg=f"output mismatch at step {step}",
        )
        np.testing.assert_allclose(
            np.asarray(buf_fused),
            np.asarray(buf_ref),
            atol=_ATOL,
            rtol=_RTOL,
            err_msg=f"buffer mismatch at step {step}",
        )


def test_decode_repeated_dummy_keeps_slot0_zero():
    """Repeated calls with some dummy-slot targets must not pollute slot 0."""
    H, D = 8, 128
    total_slots = 8
    gamma = _make_g_gamma(H, seed=130)
    buf = _make_buffer(total_slots, H, D, seed=131)

    for step in range(4):
        # Every step routes one of the 4 seqs to slot 0.
        indices = jnp.asarray([1, 0, 3, 5], dtype=jnp.int32)
        has_init = jnp.full((4,), True, dtype=jnp.bool_)
        q, k, v = _make_qkv(4, H, D, seed=140 + step)
        _o, buf = decode_simple_gla_fused(
            q,
            k,
            v,
            recurrent_buffer=buf,
            recurrent_indices=indices,
            has_initial_state=has_init,
            g_gamma=gamma,
            scale=None,
        )
        # Slot 0 must remain exactly zero across every step.
        np.testing.assert_array_equal(
            np.asarray(buf[0]),
            np.zeros((H, D, D), dtype=np.asarray(buf).dtype),
            err_msg=f"slot 0 polluted after step {step}",
        )
