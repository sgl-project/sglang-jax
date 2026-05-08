"""GLA shard correctness tests at production-like dimensions.

Verifies LightningAttnBackend (decode + extend) against a pure-numpy
GLA reference implementation using per-shard dimensions matching
Ling-2.6-flash production (H_local=4, K=128).

Production: H=64, K=128, TP=16 → H_local=4
Test:       H=16, K=128, TP=4  → H_local=4  (same per-shard shape)

Run with: pytest python/sgl_jax/test/layers/test_gla_shard.py -v
"""

import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    MockRecurrentStatePool,
)
from sgl_jax.srt.layers.attention.linear.lightning_backend import LightningAttnBackend
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])

try:
    from sgl_jax.srt.kernels.simple_gla.simple_gla import fused_recurrent_simple_gla

    HAS_SIMPLE_GLA = True
except ImportError:
    HAS_SIMPLE_GLA = False

_HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
_N_DEVICES = jax.device_count()

requires_simple_gla = pytest.mark.skipif(
    not HAS_SIMPLE_GLA, reason="simple_gla kernel not available"
)
requires_tpu = pytest.mark.skipif(not _HAS_TPU, reason="chunk kernel requires TPU")

_H = 16
_K = 128
_CHUNK_SIZE = 64

TOKEN_BUCKETS = [1 << i for i in range(6, 15)]  # [64, ..., 16384]
BS_BUCKETS = [1 << i for i in range(0, 9)]  # [1, ..., 256]


# ---------------------------------------------------------------------------
# Numpy GLA reference (float64)
# ---------------------------------------------------------------------------


def numpy_gla_recurrent(q, k, v, g_gamma, h0=None, scale=None):
    """Pure-numpy GLA recurrence in float64.

    Args:
        q, k, v: [B, T, H, K] float arrays
        g_gamma: [H] negative log-decay per head
        h0: [B, H, K, K] initial state or None (zeros)
        scale: float or None (defaults to K^-0.5)
    Returns:
        output: [B, T, H, K] float32
        final_state: [B, H, K, K] float32
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K**-0.5
    gamma = np.exp(g_gamma)

    h = np.zeros((B, H, K, V), dtype=np.float64) if h0 is None else h0.astype(np.float64).copy()

    outputs = []
    for t in range(T):
        h = gamma[None, :, None, None] * h
        h = h + np.einsum("bhk,bhv->bhkv", k[:, t], v[:, t])
        o = np.einsum("bhk,bhkv->bhv", q[:, t], h) * scale
        outputs.append(o)

    output = np.stack(outputs, axis=1)
    return output.astype(np.float32), h.astype(np.float32)


# ---------------------------------------------------------------------------
# Padding helpers (mirrors control-plane logic)
# ---------------------------------------------------------------------------


def _pad_to_bucket(n, buckets):
    for b in buckets:
        if n <= b:
            return b
    return ((n + buckets[-1] - 1) // buckets[-1]) * buckets[-1]


def _make_slopes(H, layer_id=5, num_layers=80):
    """Generate ALiBi slopes matching BailingMoeV2_5LinearAttention."""

    def _get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    def _build_slope_tensor(n):
        if math.log2(n).is_integer():
            return _get_slopes_power_of_2(n)
        closest = 2 ** math.floor(math.log2(n))
        return (
            _get_slopes_power_of_2(closest) + _build_slope_tensor(2 * closest)[0::2][: n - closest]
        )

    base = np.array(_build_slope_tensor(H), dtype=np.float32)
    slope = -base * (1 - (layer_id - 1) / (num_layers - 1) + 1e-5)
    return slope


def _build_extend_batch(extend_seq_lens_real):
    """Build a padded batch matching control-plane rules.

    Returns:
        padded_seq_lens: np.array with BS-padded extend_seq_lens (zeros appended)
        input_ids: np.array of shape [T_outer] (zeros for padding)
        T_outer: total token-padded length
    """
    real = np.array(extend_seq_lens_real, dtype=np.int32)
    total_tokens = int(real.sum())
    T_outer = _pad_to_bucket(total_tokens, TOKEN_BUCKETS)

    bs = len(real)
    bs_padded = _pad_to_bucket(bs, BS_BUCKETS)
    padded_seq_lens = np.zeros(bs_padded, dtype=np.int32)
    padded_seq_lens[:bs] = real

    input_ids = np.zeros(T_outer, dtype=np.int32)
    return padded_seq_lens, input_ids, T_outer


def _build_extend_qkv(extend_seq_lens_real, H, K, rng):
    """Generate q/k/v with padding, plus per-request numpy references.

    Returns:
        q, k, v: np.array [T_outer, H, K] float32 (padding positions are zero)
        padded_seq_lens, input_ids, T_outer: from _build_extend_batch
        per_request_qkv: list of (q_i, k_i, v_i) each [1, T_i, H, K]
    """
    padded_seq_lens, input_ids, T_outer = _build_extend_batch(extend_seq_lens_real)

    q = np.zeros((T_outer, H, K), dtype=np.float32)
    k = np.zeros((T_outer, H, K), dtype=np.float32)
    v = np.zeros((T_outer, H, K), dtype=np.float32)

    per_request_qkv = []
    offset = 0
    for seq_len in extend_seq_lens_real:
        qi = rng.standard_normal((1, seq_len, H, K)).astype(np.float32) * 0.1
        ki = rng.standard_normal((1, seq_len, H, K)).astype(np.float32) * 0.1
        vi = rng.standard_normal((1, seq_len, H, K)).astype(np.float32) * 0.1
        q[offset : offset + seq_len] = qi[0]
        k[offset : offset + seq_len] = ki[0]
        v[offset : offset + seq_len] = vi[0]
        per_request_qkv.append((qi, ki, vi))
        offset += seq_len

    return q, k, v, padded_seq_lens, input_ids, T_outer, per_request_qkv


# ---------------------------------------------------------------------------
# Backend test helpers
# ---------------------------------------------------------------------------


def _make_mock_pool(layer_id, recurrent_state, recurrent_indices=None):
    B = recurrent_state.shape[0]
    if recurrent_indices is None:
        recurrent_indices = np.arange(1, B + 1, dtype=np.int32)
    N_plus_1 = int(max(recurrent_indices)) + 1
    buf = jnp.zeros((N_plus_1,) + recurrent_state.shape[1:], dtype=recurrent_state.dtype)
    buf = buf.at[jnp.array(recurrent_indices)].set(recurrent_state)
    return MockRecurrentStatePool(layer_caches={layer_id: (buf, [])}), recurrent_indices


def _extract_state(pool_updates, recurrent_indices):
    new_ssm_full, conv_list = pool_updates
    return new_ssm_full[jnp.array(recurrent_indices)]


def _setup_backend(backend, forward_mode, recurrent_indices, extend_seq_lens=None, input_ids=None):
    batch = SimpleNamespace(forward_mode=forward_mode, recurrent_indices=recurrent_indices)
    if forward_mode == ForwardMode.DECODE:
        batch.seq_lens = np.ones(len(recurrent_indices), dtype=np.int32)
    elif forward_mode == ForwardMode.EXTEND:
        batch.extend_seq_lens = np.asarray(extend_seq_lens, dtype=np.int32)
        batch.seq_lens = np.asarray(extend_seq_lens, dtype=np.int32)
        batch.input_ids = np.asarray(input_ids, dtype=np.int32)
    metadata = backend.get_forward_metadata(batch)
    backend.forward_metadata = metadata
    return metadata


def _make_fake_layer(layer_id=5, slope=None, H=_H):
    if slope is None:
        slope = jnp.array(_make_slopes(H), dtype=jnp.float32)
    return SimpleNamespace(
        layer_id=layer_id,
        slope=slope,
        mesh=mesh,
        num_heads=H,
        head_dim=_K,
    )


# ---------------------------------------------------------------------------
# Class 1: TestDecodeSharded
# ---------------------------------------------------------------------------


@requires_simple_gla
class TestDecodeSharded:
    """Decode precision: fused_recurrent_simple_gla vs numpy reference at H=16, K=128."""

    def test_decode_single_request(self):
        B, H, K = 1, _H, _K
        rng = np.random.default_rng(42)
        g_gamma = _make_slopes(H).astype(np.float32)

        q_np = rng.standard_normal((B, 1, H, K)).astype(np.float32) * 0.1
        k_np = rng.standard_normal((B, 1, H, K)).astype(np.float32) * 0.1
        v_np = rng.standard_normal((B, 1, H, K)).astype(np.float32) * 0.1
        h0_np = rng.standard_normal((B, H, K, K)).astype(np.float32) * 0.01

        ref_out, ref_state = numpy_gla_recurrent(q_np, k_np, v_np, g_gamma, h0=h0_np)

        out_jax, state_jax = fused_recurrent_simple_gla(
            jnp.array(q_np),
            jnp.array(k_np),
            jnp.array(v_np),
            g_gamma=jnp.array(g_gamma),
            initial_state=jnp.array(h0_np),
            output_final_state=True,
            scale=None,
        )

        np.testing.assert_allclose(np.array(out_jax), ref_out, atol=1e-3)
        np.testing.assert_allclose(np.array(state_jax), ref_state, atol=1e-3)

    def test_decode_multi_step(self):
        """32-step autoregressive decode, B=4."""
        B, H, K, steps = 4, _H, _K, 32
        rng = np.random.default_rng(200)
        g_gamma = _make_slopes(H).astype(np.float32)

        h_np = rng.standard_normal((B, H, K, K)).astype(np.float32) * 0.01
        h_jax = jnp.array(h_np)

        h_ref = h_np.copy()
        for t in range(steps):
            q_np = rng.standard_normal((B, 1, H, K)).astype(np.float32) * 0.1
            k_np = rng.standard_normal((B, 1, H, K)).astype(np.float32) * 0.1
            v_np = rng.standard_normal((B, 1, H, K)).astype(np.float32) * 0.1

            _, h_ref = numpy_gla_recurrent(q_np, k_np, v_np, g_gamma, h0=h_ref)

            _, h_jax = fused_recurrent_simple_gla(
                jnp.array(q_np),
                jnp.array(k_np),
                jnp.array(v_np),
                g_gamma=jnp.array(g_gamma),
                initial_state=h_jax,
                output_final_state=True,
                scale=None,
            )

        np.testing.assert_allclose(
            np.array(h_jax),
            h_ref,
            atol=1e-2,
            err_msg="Multi-step decode (32 steps) state diverged",
        )


# ---------------------------------------------------------------------------
# Class 2: TestExtendSharded
# ---------------------------------------------------------------------------


@requires_simple_gla
@requires_tpu
class TestExtendSharded:
    """Extend precision: chunk kernel via LightningAttnBackend vs numpy reference."""

    def _run_extend_precision_test(self, extend_seq_lens_real, rng_seed=42, h0_np=None):
        """Common test body: backend extend vs per-request numpy reference."""
        layer_id = 5
        H, K = _H, _K
        rng = np.random.default_rng(rng_seed)
        g_gamma = _make_slopes(H)

        q_np, k_np, v_np, padded_seq_lens, input_ids, T_outer, per_req = _build_extend_qkv(
            extend_seq_lens_real, H, K, rng
        )

        B_real = len(extend_seq_lens_real)
        B_padded = len(padded_seq_lens)

        if h0_np is None:
            h0_np = np.zeros((B_real, H, K, K), dtype=np.float32)
        h0_padded = np.zeros((B_padded, H, K, K), dtype=np.float32)
        h0_padded[:B_real] = h0_np

        with jax.set_mesh(mesh):
            backend = LightningAttnBackend(mesh=mesh)
            rec_indices = np.arange(1, B_padded + 1, dtype=np.int32)
            pool, _ = _make_mock_pool(layer_id, jnp.array(h0_padded), rec_indices)
            _setup_backend(
                backend,
                ForwardMode.EXTEND,
                rec_indices,
                extend_seq_lens=padded_seq_lens,
                input_ids=input_ids,
            )
            layer = _make_fake_layer(layer_id=layer_id, H=H)
            fb = SimpleNamespace(forward_mode=ForwardMode.EXTEND)

            output, pool_updates = backend(
                jnp.array(q_np),
                jnp.array(k_np),
                jnp.array(v_np),
                layer=layer,
                forward_batch=fb,
                recurrent_state_pool=pool,
            )
            output_np = np.array(output)
            states_np = np.array(_extract_state(pool_updates, rec_indices))

        cs = _CHUNK_SIZE
        ref_outputs = []
        ref_states = []
        offset = 0
        for i, (qi, ki, vi) in enumerate(per_req):
            seq_len = extend_seq_lens_real[i]
            aligned_len = ((seq_len + cs - 1) // cs) * cs
            # Pad to chunk-aligned length so numpy sees the same decay steps as the kernel
            qi_pad = np.zeros((1, aligned_len, H, K), dtype=np.float32)
            ki_pad = np.zeros((1, aligned_len, H, K), dtype=np.float32)
            vi_pad = np.zeros((1, aligned_len, H, K), dtype=np.float32)
            qi_pad[:, :seq_len] = qi
            ki_pad[:, :seq_len] = ki
            vi_pad[:, :seq_len] = vi
            h0_i = h0_np[i : i + 1]
            ref_out_i, ref_state_i = numpy_gla_recurrent(qi_pad, ki_pad, vi_pad, g_gamma, h0=h0_i)
            ref_outputs.append(ref_out_i[0, :seq_len])
            ref_states.append(ref_state_i[0])

        offset = 0
        for i, seq_len in enumerate(extend_seq_lens_real):
            actual_out = output_np[offset : offset + seq_len]
            ref_out = ref_outputs[i].reshape(seq_len, H * K)
            np.testing.assert_allclose(
                actual_out,
                ref_out,
                atol=5e-2,
                err_msg=f"Extend output mismatch for request {i} (seq_len={seq_len})",
            )
            offset += seq_len

        for i in range(B_real):
            np.testing.assert_allclose(
                states_np[i],
                ref_states[i],
                atol=5e-2,
                err_msg=f"Extend state mismatch for request {i}",
            )

    def test_extend_single_short(self):
        self._run_extend_precision_test([100], rng_seed=1001)

    def test_extend_multi_request_variable(self):
        self._run_extend_precision_test([256, 100, 512], rng_seed=1005)

    def test_extend_with_nonzero_initial_state(self):
        rng = np.random.default_rng(1009)
        h0 = rng.standard_normal((1, _H, _K, _K)).astype(np.float32) * 0.01
        self._run_extend_precision_test([4096], rng_seed=1009, h0_np=h0)


# ---------------------------------------------------------------------------
# Class 3: TestEndToEndBackend
# ---------------------------------------------------------------------------


@requires_simple_gla
class TestEndToEndBackend:
    """End-to-end backend flow: pool state roundtrip, extend→decode continuity."""

    @requires_tpu
    def test_extend_then_decode(self):
        """Extend 4096 tokens then decode 32 steps — verify state continuity."""
        layer_id = 5
        H, K = _H, _K
        seq_len = 4096
        decode_steps = 32
        rng = np.random.default_rng(3001)
        g_gamma = _make_slopes(H)

        q_ext_np = rng.standard_normal((1, seq_len, H, K)).astype(np.float32) * 0.1
        k_ext_np = rng.standard_normal((1, seq_len, H, K)).astype(np.float32) * 0.1
        v_ext_np = rng.standard_normal((1, seq_len, H, K)).astype(np.float32) * 0.1
        h0_np = np.zeros((1, H, K, K), dtype=np.float32)

        _, ref_state = numpy_gla_recurrent(q_ext_np, k_ext_np, v_ext_np, g_gamma, h0=h0_np)

        with jax.set_mesh(mesh):
            backend = LightningAttnBackend(mesh=mesh)
            padded_seq_lens, input_ids, T_outer = _build_extend_batch([seq_len])
            B_padded = len(padded_seq_lens)

            rec_indices = np.arange(1, B_padded + 1, dtype=np.int32)
            h0_padded = np.zeros((B_padded, H, K, K), dtype=np.float32)
            pool, _ = _make_mock_pool(layer_id, jnp.array(h0_padded), rec_indices)

            _setup_backend(
                backend,
                ForwardMode.EXTEND,
                rec_indices,
                extend_seq_lens=padded_seq_lens,
                input_ids=input_ids,
            )
            layer = _make_fake_layer(layer_id=layer_id, H=H)
            fb_ext = SimpleNamespace(forward_mode=ForwardMode.EXTEND)

            q_ext = jnp.zeros((T_outer, H, K), dtype=jnp.float32)
            k_ext = jnp.zeros((T_outer, H, K), dtype=jnp.float32)
            v_ext = jnp.zeros((T_outer, H, K), dtype=jnp.float32)
            q_ext = q_ext.at[:seq_len].set(jnp.array(q_ext_np[0]))
            k_ext = k_ext.at[:seq_len].set(jnp.array(k_ext_np[0]))
            v_ext = v_ext.at[:seq_len].set(jnp.array(v_ext_np[0]))

            _, pool_updates = backend(
                q_ext,
                k_ext,
                v_ext,
                layer=layer,
                forward_batch=fb_ext,
                recurrent_state_pool=pool,
            )
            extend_state = np.array(_extract_state(pool_updates, rec_indices))

        np.testing.assert_allclose(
            extend_state[0],
            ref_state[0],
            atol=5e-2,
            err_msg="Extend state != numpy reference before decode",
        )

        h_ref = ref_state.copy()
        h_jax = jnp.array(extend_state)
        for step in range(decode_steps):
            q_d = rng.standard_normal((1, 1, H, K)).astype(np.float32) * 0.1
            k_d = rng.standard_normal((1, 1, H, K)).astype(np.float32) * 0.1
            v_d = rng.standard_normal((1, 1, H, K)).astype(np.float32) * 0.1

            _, h_ref = numpy_gla_recurrent(q_d, k_d, v_d, g_gamma, h0=h_ref)

            _, h_jax = fused_recurrent_simple_gla(
                jnp.array(q_d),
                jnp.array(k_d),
                jnp.array(v_d),
                g_gamma=jnp.array(g_gamma),
                initial_state=h_jax[:1],
                output_final_state=True,
                scale=None,
            )

        np.testing.assert_allclose(
            np.array(h_jax),
            h_ref,
            atol=5e-2,
            err_msg="State diverged after extend + 32 decode steps",
        )
