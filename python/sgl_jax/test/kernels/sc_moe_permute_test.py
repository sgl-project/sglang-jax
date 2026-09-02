"""Tests for the SparseCore MoE permute/unpermute kernels (TPU with SparseCore only)."""

import os

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.sparse_core.moe_permute import (
    reference_combine,
    sc_combine,
    sc_dispatch_gather,
    sparse_core_available,
)
from sgl_jax.srt.layers.moe import EPMoE
from sgl_jax.test.test_utils import create_device_mesh

jax.config.parse_flags_with_absl()

HIDDEN = 6144
NUM_EXPERTS = 256
LOCAL_EXPERTS = 16
TOPK = 8


def _routing(T, num_experts, topk, seed):
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((T, num_experts)).astype(np.float32)
    topk_ids = np.argsort(-logits, axis=1)[:, :topk]
    w = np.take_along_axis(logits, topk_ids, 1)
    w = np.exp(w)
    w /= w.sum(1, keepdims=True)
    flat = topk_ids.ravel()
    sorted_sel = np.argsort(flat, kind="stable")
    revert = np.empty_like(sorted_sel)
    revert[sorted_sel] = np.arange(sorted_sel.size)
    group_sizes = np.bincount(flat, minlength=num_experts)
    return topk_ids, w.astype(np.float32), sorted_sel, revert, group_sizes


def _cos(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


class ScMoePermuteKernelTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        if not sparse_core_available():
            self.skipTest("requires a TPU with SparseCore")

    @parameterized.parameters(64, 1024, 4096, 8192)
    def test_dispatch_gather_matches_xla_on_local_rows(self, T):
        rng = np.random.default_rng(1)
        topk_ids, _, sorted_sel, _, group_sizes = _routing(T, NUM_EXPERTS, TOPK, seed=T)
        token_indices = (sorted_sel // TOPK).astype(np.int32)
        # local shard = experts [LOCAL_EXPERTS, 2*LOCAL_EXPERTS): a non-zero offset exercises `start`.
        offsets = np.concatenate([[0], np.cumsum(group_sizes)])
        start, end = int(offsets[LOCAL_EXPERTS]), int(offsets[2 * LOCAL_EXPERTS])
        hidden = jnp.asarray(rng.standard_normal((T, HIDDEN)).astype(np.float32)).astype(
            jnp.bfloat16
        )
        tok = jnp.asarray(token_indices)
        ref = hidden[tok]
        out = sc_dispatch_gather(
            hidden, tok, jnp.asarray(start, jnp.int32), jnp.asarray(end, jnp.int32)
        )
        self.assertEqual(out.shape, ref.shape)
        self.assertEqual(out.dtype, ref.dtype)
        np.testing.assert_array_equal(
            np.asarray(out[start:end].astype(jnp.float32)),
            np.asarray(ref[start:end].astype(jnp.float32)),
        )

    @parameterized.product(T=(64, 1024, 4096, 8192), weight_dtype=(jnp.float32, jnp.bfloat16))
    def test_combine_matches_reference_and_masks_nonlocal_rows(self, T, weight_dtype):
        rng = np.random.default_rng(2)
        topk_ids, w, sorted_sel, revert, group_sizes = _routing(T, NUM_EXPERTS, TOPK, seed=T + 7)
        flat = topk_ids.ravel()
        local = (flat >= LOCAL_EXPERTS) & (flat < 2 * LOCAL_EXPERTS)
        offsets = np.concatenate([[0], np.cumsum(group_sizes)])
        start, end = int(offsets[LOCAL_EXPERTS]), int(offsets[2 * LOCAL_EXPERTS])
        # gmm output sorted by expert. Production zero-fills non-local rows; here we
        # deliberately put garbage there so the test proves the mask is honoured.
        x = rng.standard_normal((T * TOPK, HIDDEN)).astype(np.float32) * 1e3
        x[start:end] = rng.standard_normal((end - start, HIDDEN)).astype(np.float32)
        x = jnp.asarray(x).astype(jnp.bfloat16)
        x_clean = x.at[:start].set(0).at[end:].set(0)
        weights = jnp.asarray(w).astype(weight_dtype)
        revert_j = jnp.asarray(revert.astype(np.int32))
        mask = jnp.asarray(local)

        ref = reference_combine(x_clean, revert_j, weights, TOPK)
        out = sc_combine(x, revert_j, weights, mask, TOPK)
        self.assertEqual(out.shape, (T, HIDDEN))
        self.assertEqual(out.dtype, jnp.bfloat16)
        ref32 = np.asarray(ref.astype(jnp.float32))
        out32 = np.asarray(out.astype(jnp.float32))
        self.assertGreater(_cos(ref32, out32), 0.9999)
        np.testing.assert_allclose(out32, ref32, rtol=2e-2, atol=2e-2)


class EPMoEScPermuteTest(absltest.TestCase):
    """Whole-layer parity: EPMoE with the SparseCore permute path vs the XLA path."""

    def setUp(self):
        super().setUp()
        if not sparse_core_available():
            self.skipTest("requires a TPU with SparseCore")

    def test_epmoe_output_parity(self):
        ndev = len(jax.devices())
        mesh = create_device_mesh(ici_parallelism=[1, ndev], dcn_parallelism=[1, 1])
        hidden, inter, num_experts, topk, T = 1024, 512, 4 * ndev, 4, 8192
        rng = np.random.default_rng(3)
        topk_ids, w, *_ = _routing(T, num_experts, topk, seed=11)

        with jax.set_mesh(mesh):
            common = dict(
                hidden_size=hidden,
                num_experts=num_experts,
                num_experts_per_tok=topk,
                ep_size=ndev,
                mesh=mesh,
                intermediate_dim=inter,
            )
            layer_xla = EPMoE(use_sc_permute=False, **common)
            layer_sc = EPMoE(use_sc_permute=True, **common)
            for name in ("wi_0", "wi_1", "wo"):
                getattr(layer_sc, name).value = getattr(layer_xla, name).value
            hs = jax.device_put(
                jnp.asarray(rng.standard_normal((T, hidden)).astype(np.float32)).astype(
                    jnp.bfloat16
                ),
                jax.sharding.NamedSharding(mesh, P(None, None)),
            )
            tw = jax.device_put(jnp.asarray(w), jax.sharding.NamedSharding(mesh, P(None, None)))
            ti = jax.device_put(
                jnp.asarray(topk_ids.astype(np.int32)),
                jax.sharding.NamedSharding(mesh, P(None, None)),
            )
            self.assertTrue(layer_sc.use_sc_permute)
            self.assertFalse(layer_xla.use_sc_permute)
            out_xla = layer_xla(hs, tw, ti)
            out_sc = layer_sc(hs, tw, ti)

        a = np.asarray(out_xla.astype(jnp.float32))
        b = np.asarray(out_sc.astype(jnp.float32))
        self.assertEqual(a.shape, b.shape)
        # Outputs are O(1e4..1e5) with heavy cancellation across top_k terms, so
        # per-element rtol is meaningless; bound the error relative to the output
        # scale instead. (A float64 emulation puts the SC path *closer* to exact
        # than the XLA einsum, see ~/GLM/sc-moe-permute/parity_probe.log.)
        scale = np.abs(a).max()
        self.assertGreater(_cos(a, b), 0.9999)
        self.assertLess(np.abs(a - b).max(), 2e-2 * scale)
        self.assertLess(np.abs(a - b).mean(), 5e-3 * np.abs(a).mean())

    def test_env_flag_default_off(self):
        ndev = len(jax.devices())
        mesh = create_device_mesh(ici_parallelism=[1, ndev], dcn_parallelism=[1, 1])
        with jax.set_mesh(mesh):
            layer = EPMoE(
                hidden_size=256,
                num_experts=ndev,
                num_experts_per_tok=1,
                ep_size=ndev,
                mesh=mesh,
                intermediate_dim=128,
            )
        self.assertFalse(layer.use_sc_permute)
        self.assertNotIn("SGL_JAX_MOE_SC_PERMUTE", os.environ)


if __name__ == "__main__":
    absltest.main()
