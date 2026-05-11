"""Unit tests for :class:`Qwen3_5GatedDeltaNet`.

The class wraps :class:`GDNAttnBackend` with the HF Qwen3-5 projection
structure (six independent linears: q/k/v/z/b/a), a gated GemmaRMSNorm,
and an ``out_proj``. The backend + kernels are covered separately; this
file exercises the glue: param shapes, the RMS gate math, and the
end-to-end shape/dtype contract.

Run with:
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \\
        python -m pytest test/srt/test_qwen3_5_gated_delta_net.py -v
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.linear.qwen3_5_gated_delta_net import (
    Qwen3_5GatedDeltaNet,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mesh():
    devices = mesh_utils.create_device_mesh((8,))[:1].reshape((1, 1))
    return Mesh(
        devices, ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def _make_config(hidden_size=64, n_kq=1, n_v=2, d_k=4, d_v=8, K=3, eps=1e-6):
    return SimpleNamespace(
        hidden_size=hidden_size,
        linear_num_value_heads=n_v,
        linear_num_key_heads=n_kq,
        linear_key_head_dim=d_k,
        linear_value_head_dim=d_v,
        linear_conv_kernel_dim=K,
        rms_norm_eps=eps,
    )


class _FakeForwardMode:
    def __init__(self, decode: bool):
        self._decode = decode

    def is_decode(self):
        return self._decode


class _FakeGDNMetadata:
    def __init__(self, cu_seqlens):
        self.cu_seqlens = cu_seqlens


class _FakeForwardBatch:
    def __init__(self, is_decode, mamba_cache_indices, cu_seqlens=None, extend_prefix_lens=None):
        self.forward_mode = _FakeForwardMode(is_decode)
        self.mamba_cache_indices = mamba_cache_indices
        self.gdn_metadata = _FakeGDNMetadata(cu_seqlens)
        self.extend_prefix_lens = extend_prefix_lens


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class Qwen3_5GatedDeltaNetInitTest(unittest.TestCase):
    def test_projection_shapes_match_hf_layout(self):
        """``in_proj_qkv`` is a merged Q/K/V GEMM; z/b/a stay separate. Param
        shapes match the HF Qwen3-5 checkpoint structure."""
        mesh = _make_mesh()
        with jax.set_mesh(mesh):
            cfg = _make_config(hidden_size=32, n_kq=2, n_v=4, d_k=8, d_v=8, K=3)
            layer = Qwen3_5GatedDeltaNet(cfg, layer_id=0, mamba_layer_id=0, mesh=mesh)
            key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
            value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
            qkv_total = 2 * key_dim + value_dim

            # Fused Q/K/V projection: single weight tensor of size 2*key+value.
            self.assertEqual(layer.in_proj_qkv.weight.value.shape, (cfg.hidden_size, qkv_total))
            self.assertEqual(layer.in_proj_qkv.output_sizes, [key_dim, key_dim, value_dim])
            # z/b/a stay separate.
            self.assertEqual(layer.in_proj_z.weight.value.shape, (cfg.hidden_size, value_dim))
            self.assertEqual(layer.in_proj_b.weight.value.shape, (cfg.hidden_size, cfg.linear_num_value_heads))
            self.assertEqual(layer.in_proj_a.weight.value.shape, (cfg.hidden_size, cfg.linear_num_value_heads))
            self.assertEqual(layer.out_proj.weight.value.shape, (value_dim, cfg.hidden_size))
            self.assertEqual(layer.attention.conv1d_weight.value.shape,
                             (layer.conv_dim, cfg.linear_conv_kernel_dim))
            self.assertEqual(layer.attention.A_log.value.shape, (cfg.linear_num_value_heads,))
            self.assertEqual(layer.attention.dt_bias.value.shape, (cfg.linear_num_value_heads,))
            self.assertEqual(layer.rms_scale.value.shape, (cfg.linear_value_head_dim,))


class Qwen3_5GatedDeltaNetRmsGateTest(unittest.TestCase):
    """``_rms_gate`` = ``RMSNorm(core)·γ * silu(z)``."""

    def test_matches_explicit_formula(self):
        mesh = _make_mesh()
        with jax.set_mesh(mesh):
            cfg = _make_config(n_v=2, d_v=4, eps=1e-6)
            layer = Qwen3_5GatedDeltaNet(cfg, 0, 0, mesh)
            gamma = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
            layer.rms_scale = nnx.Param(gamma)

            T, n_v, d_v = 3, 2, 4
            rng = jax.random.split(jax.random.key(7), 2)
            core = jax.random.normal(rng[0], (T, n_v, d_v), dtype=jnp.bfloat16) * 0.5
            z = jax.random.normal(rng[1], (T, n_v, d_v), dtype=jnp.bfloat16) * 0.5

            got = layer._rms_gate(core, z)

            # Reference RMS over the last axis in fp32.
            xf = core.astype(jnp.float32)
            rms = jnp.sqrt((xf * xf).mean(axis=-1, keepdims=True) + 1e-6)
            ref = (xf / rms) * gamma * jax.nn.silu(z.astype(jnp.float32))
            ref = ref.astype(core.dtype)
            np.testing.assert_allclose(got, ref, atol=1e-2, rtol=1e-2)


class Qwen3_5GatedDeltaNetEndToEndTest(unittest.TestCase):
    """End-to-end shape/dtype/finite contract.

    Numerical correctness lives in ``test_gated_delta.py`` and
    ``test_ragged_gated_delta_rule_ref.py`` (which exercise the kernels
    directly without a mesh). Here we just confirm the layer wires
    projections → conv → recurrence → RMS gate → out_proj without
    sharding errors and the output shape matches ``hidden_size``.
    """

    def _run_layer(self, is_decode):
        mesh = _make_mesh()
        with jax.set_mesh(mesh):
            cfg = _make_config(hidden_size=32, n_kq=1, n_v=2, d_k=4, d_v=8, K=3)
            layer = Qwen3_5GatedDeltaNet(cfg, 0, 0, mesh)
            conv_dim = layer.conv_dim
            layer.attention.conv1d_weight = nnx.Param(
                jax.random.normal(jax.random.key(0), (conv_dim, cfg.linear_conv_kernel_dim), dtype=jnp.bfloat16) * 0.05
            )

            # 3 reqs of length [3, 2] for extend, B=3 for decode.
            if is_decode:
                T = 3
                B = 3
                cu_seqlens = None
                extend_prefix_lens = None
            else:
                T = 5
                B = 2
                cu_seqlens = jnp.array([0, 3, 5], dtype=jnp.int32)
                extend_prefix_lens = jnp.array([0, 0], dtype=jnp.int32)

            hidden = jax.random.normal(jax.random.key(1), (T, cfg.hidden_size), dtype=jnp.bfloat16) * 0.3
            conv_state = jnp.zeros(
                (B + 1, conv_dim, cfg.linear_conv_kernel_dim - 1), dtype=jnp.bfloat16,
                out_sharding=NamedSharding(mesh, P(None, "tensor", None)),
            )
            rec_state = jnp.zeros(
                (B + 1, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim),
                dtype=jnp.float32,
                out_sharding=NamedSharding(mesh, P(None, "tensor", None, None)),
            )
            fb = _FakeForwardBatch(
                is_decode=is_decode,
                mamba_cache_indices=jnp.arange(1, B + 1, dtype=jnp.int32),
                cu_seqlens=cu_seqlens,
                extend_prefix_lens=extend_prefix_lens,
            )
            return layer(hidden, fb, conv_state, rec_state), B, T, cfg, conv_dim

    def test_decode_path(self):
        (out, new_conv, new_rec), B, T, cfg, conv_dim = self._run_layer(is_decode=True)
        self.assertEqual(out.shape, (T, cfg.hidden_size))
        self.assertEqual(out.dtype, jnp.bfloat16)
        self.assertEqual(new_conv.shape, (B, conv_dim, cfg.linear_conv_kernel_dim - 1))
        self.assertEqual(new_rec.shape,
                         (B, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim))
        self.assertTrue(bool(jnp.all(jnp.isfinite(out))))

    def test_extend_path(self):
        (out, new_conv, new_rec), B, T, cfg, conv_dim = self._run_layer(is_decode=False)
        self.assertEqual(out.shape, (T, cfg.hidden_size))
        self.assertEqual(out.dtype, jnp.bfloat16)
        self.assertEqual(new_conv.shape, (B, conv_dim, cfg.linear_conv_kernel_dim - 1))
        self.assertEqual(new_rec.shape,
                         (B, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim))
        self.assertTrue(bool(jnp.all(jnp.isfinite(out))))


if __name__ == "__main__":
    unittest.main()
