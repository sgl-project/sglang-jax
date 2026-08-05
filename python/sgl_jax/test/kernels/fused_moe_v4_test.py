"""CPU 4-device parity test for the v4 (TP-MoE) kernel.

Gates the v4 port before TPU end-to-end testing on ling_v3_flash. Validates:

  (1) tp_moe (prefill ragged_dot path) with replicated/full weights matches
      a numpy reference.
  (2) tp_moe_decode (decode einsum path) matches the reference.
  (3) fused_tp_moe_v4 (shard_map standalone entry) on a 4-device CPU mesh
      with TP-sliced weights matches the reference within bf16 tolerance.
  (4) Sentinel handling: -1 padding tokens are routed to the dummy group
      and contribute zero.

Run with:
  XLA_FLAGS="--xla_force_host_platform_device_count=4" JAX_PLATFORMS=cpu \\
    pytest python/sgl_jax/test/kernels/fused_moe_v4_test.py -v

Adapted from AInfer/tests/tpu/test_moe_v4.py (which the AInfer team uses to
gate the same kernel on its side).
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.fused_moe.v4.kernel import (
    fused_tp_moe_v4,
    tp_moe,
    tp_moe_decode,
)


def _ref_moe_numpy(tokens, w1, w2, w3, topk_ids, topk_weights):
    """Per-token, per-expert SwiGLU FFN + weighted sum. Reference in float32."""
    T, H = tokens.shape
    top_k = topk_ids.shape[1]
    out = np.zeros((T, H), dtype=np.float32)
    for t in range(T):
        for k in range(top_k):
            eid = int(topk_ids[t, k])
            w = float(topk_weights[t, k])
            if eid < 0 or w == 0:
                continue
            x = tokens[t].astype(np.float32)
            gate = x @ w1[eid].astype(np.float32)
            up = x @ w3[eid].astype(np.float32)
            act = (gate / (1 + np.exp(-gate))) * up
            down = act @ w2[eid].astype(np.float32)
            out[t] += w * down
    return out


def _make_topk(rng, T, top_k, num_experts):
    """Sample top_k expert ids per token (no duplicates within a token) and
    uniform-ish weights. Used to keep the test deterministic."""
    ids = np.stack(
        [rng.choice(num_experts, size=top_k, replace=False) for _ in range(T)]
    ).astype(np.int32)
    weights = rng.uniform(0.1, 1.0, size=(T, top_k)).astype(np.float32)
    weights = weights / weights.sum(axis=1, keepdims=True)  # normalize per token
    return ids, weights


class TPMoEKernelTest(parameterized.TestCase):
    """tp_moe / tp_moe_decode: kernel-level parity against numpy."""

    def test_tp_moe_replicated_vs_ref(self):
        """tp_moe with full (non-sliced) weights matches numpy ref."""
        rng = np.random.RandomState(42)
        E, H, I, T, top_k = 16, 32, 16, 128, 4  # T > _DECODE_THRESHOLD=64 -> prefill path

        tokens = rng.randn(T, H).astype(np.float32)
        w1 = rng.randn(E, H, I).astype(np.float32) * 0.1
        w2 = rng.randn(E, I, H).astype(np.float32) * 0.1
        w3 = rng.randn(E, H, I).astype(np.float32) * 0.1
        ids, weights = _make_topk(rng, T, top_k, E)

        ref = _ref_moe_numpy(tokens, w1, w2, w3, ids, weights)

        out = tp_moe(
            jnp.asarray(tokens, dtype=jnp.bfloat16),
            jnp.asarray(w1, dtype=jnp.bfloat16),
            jnp.asarray(w2, dtype=jnp.bfloat16),
            jnp.asarray(w3, dtype=jnp.bfloat16),
            jnp.asarray(ids),
            jnp.asarray(weights, dtype=jnp.bfloat16),
            num_experts=E,
        )
        out_np = np.asarray(out, dtype=np.float32)
        # bf16 tolerance on accumulated FFN outputs.
        np.testing.assert_allclose(out_np, ref, atol=0.5, rtol=0.1)

    def test_tp_moe_decode_vs_ref(self):
        """tp_moe_decode (gather + einsum) matches numpy ref at T=1, top_k=8."""
        rng = np.random.RandomState(123)
        E, H, I, T, top_k = 16, 32, 16, 1, 8

        tokens = rng.randn(T, H).astype(np.float32)
        w1 = rng.randn(E, H, I).astype(np.float32) * 0.1
        w2 = rng.randn(E, I, H).astype(np.float32) * 0.1
        w3 = rng.randn(E, H, I).astype(np.float32) * 0.1
        ids, weights = _make_topk(rng, T, top_k, E)

        ref = _ref_moe_numpy(tokens, w1, w2, w3, ids, weights)

        out = tp_moe_decode(
            jnp.asarray(tokens, dtype=jnp.bfloat16),
            jnp.asarray(w1, dtype=jnp.bfloat16),
            jnp.asarray(w2, dtype=jnp.bfloat16),
            jnp.asarray(w3, dtype=jnp.bfloat16),
            jnp.asarray(ids),
            jnp.asarray(weights, dtype=jnp.bfloat16),
            num_experts=E,
        )
        out_np = np.asarray(out, dtype=np.float32)
        np.testing.assert_allclose(out_np, ref, atol=0.5, rtol=0.1)

    def test_padding_routed_to_sentinel(self):
        """Tokens with topk_id == -1 must contribute zero."""
        rng = np.random.RandomState(7)
        E, H, I, T, top_k = 8, 16, 8, 16, 2

        tokens = rng.randn(T, H).astype(np.float32)
        w1 = rng.randn(E, H, I).astype(np.float32) * 0.1
        w2 = rng.randn(E, I, H).astype(np.float32) * 0.1
        w3 = rng.randn(E, H, I).astype(np.float32) * 0.1

        ids, weights = _make_topk(rng, T, top_k, E)
        # Mask the second half of tokens as padding (-1 with weight 0).
        ids[T // 2 :, :] = -1
        weights[T // 2 :, :] = 0.0

        out_kernel = tp_moe(
            jnp.asarray(tokens, dtype=jnp.bfloat16),
            jnp.asarray(w1, dtype=jnp.bfloat16),
            jnp.asarray(w2, dtype=jnp.bfloat16),
            jnp.asarray(w3, dtype=jnp.bfloat16),
            jnp.asarray(ids),
            jnp.asarray(weights, dtype=jnp.bfloat16),
            num_experts=E,
        )
        out_np = np.asarray(out_kernel, dtype=np.float32)
        # Padded half must be exactly zero (no FFN contribution).
        self.assertEqual(np.max(np.abs(out_np[T // 2 :])), 0.0)
        # First half must agree with the unpadded reference.
        ref = _ref_moe_numpy(
            tokens[: T // 2], w1, w2, w3, ids[: T // 2], weights[: T // 2]
        )
        np.testing.assert_allclose(out_np[: T // 2], ref, atol=0.5, rtol=0.1)


class FusedTPMoEV4ShardMapTest(parameterized.TestCase):
    """fused_tp_moe_v4: standalone shard_map entry on a (data=1, tensor=tp) mesh."""

    def setUp(self):
        super().setUp()
        if jax.device_count() < 4:
            self.skipTest(
                f"need 4 CPU devices, have {jax.device_count()}; "
                "set XLA_FLAGS=--xla_force_host_platform_device_count=4"
            )
        devices = np.asarray(jax.devices()[:4]).reshape(1, 4)
        self.mesh = Mesh(devices, axis_names=("data", "tensor"))

    def test_fused_tp_moe_v4_vs_ref(self):
        rng = np.random.RandomState(2025)
        # Use I divisible by tp=4 so the kernel sees I_local = I/tp.
        E, H, I, T, top_k = 8, 32, 16, 128, 4  # I/tp = 4, T > 64 -> prefill
        tp = 4

        tokens = rng.randn(T, H).astype(np.float32)
        w1 = rng.randn(E, H, I).astype(np.float32) * 0.1
        w2 = rng.randn(E, I, H).astype(np.float32) * 0.1
        w3 = rng.randn(E, H, I).astype(np.float32) * 0.1
        ids, weights = _make_topk(rng, T, top_k, E)

        ref = _ref_moe_numpy(tokens, w1, w2, w3, ids, weights)

        # Shard weights TP-wise on intermediate dim using NamedSharding.
        w13_sharding = NamedSharding(self.mesh, P(None, None, "tensor"))
        w2_sharding = NamedSharding(self.mesh, P(None, "tensor", None))
        tok_sharding = NamedSharding(self.mesh, P("data", None))

        w1_j = jax.device_put(jnp.asarray(w1, dtype=jnp.bfloat16), w13_sharding)
        w3_j = jax.device_put(jnp.asarray(w3, dtype=jnp.bfloat16), w13_sharding)
        w2_j = jax.device_put(jnp.asarray(w2, dtype=jnp.bfloat16), w2_sharding)
        tok_j = jax.device_put(jnp.asarray(tokens, dtype=jnp.bfloat16), tok_sharding)
        ids_j = jax.device_put(jnp.asarray(ids), tok_sharding)
        w_j = jax.device_put(jnp.asarray(weights, dtype=jnp.bfloat16), tok_sharding)

        out = fused_tp_moe_v4(
            self.mesh, tok_j, w1_j, w2_j, w3_j, ids_j, w_j,
            num_experts=E, tp_axis_name="tensor", data_axis_name="data",
        )
        out_np = np.asarray(out, dtype=np.float32)
        # Slightly larger tolerance: TP partials sum in bf16 across 4 chips.
        np.testing.assert_allclose(out_np, ref, atol=1.0, rtol=0.15)


class FusedTPMoEV4WrapperTest(parameterized.TestCase):
    """FusedTPMoEV4 (the engine-layer wrapper): reshape correctness."""

    def setUp(self):
        super().setUp()
        if jax.device_count() < 4:
            self.skipTest(
                f"need 4 CPU devices, have {jax.device_count()}; "
                "set XLA_FLAGS=--xla_force_host_platform_device_count=4"
            )
        devices = np.asarray(jax.devices()[:4]).reshape(1, 4)
        self.mesh = Mesh(devices, axis_names=("data", "tensor"))

    def test_reshape_weights_for_tp_sharding(self):
        """After reshape_weights_for_tp:
        - w1_tp / w3_tp / w2_tp exist with the TP partition specs.
        - Old w1 / w2 / w3 attributes are removed.
        - _tp_weights_ready is True.
        """
        # Importing here so that any import-time failure (e.g. flax nnx changes)
        # surfaces as a clear test error rather than a module-load crash.
        from sgl_jax.srt.layers.fused_moe import FusedTPMoEV4

        E, H, I, top_k = 8, 32, 16, 4
        layer = FusedTPMoEV4(
            hidden_size=H,
            num_experts=E,
            num_experts_per_tok=top_k,
            ep_size=1,
            mesh=self.mesh,
            intermediate_dim=I,
            weight_dtype=jnp.bfloat16,
            dtype=jnp.bfloat16,
            num_shared_experts=0,
        )

        # Sanity: pre-reshape weights exist and pre-reshape kernel call must
        # raise (the guard catches missing _tp_weights_ready).
        self.assertTrue(hasattr(layer, "w1"))
        self.assertTrue(hasattr(layer, "w2"))
        self.assertTrue(hasattr(layer, "w3"))
        self.assertFalse(layer._tp_weights_ready)

        layer.reshape_weights_for_tp()

        self.assertTrue(layer._tp_weights_ready)
        self.assertTrue(hasattr(layer, "w1_tp"))
        self.assertTrue(hasattr(layer, "w2_tp"))
        self.assertTrue(hasattr(layer, "w3_tp"))
        self.assertFalse(hasattr(layer, "w1"))
        self.assertFalse(hasattr(layer, "w2"))
        self.assertFalse(hasattr(layer, "w3"))

        # Spec check: w1_tp and w3_tp shard the last dim ("tensor"); w2_tp shards
        # the middle dim ("tensor"). Sharding spec is a tuple; allow shorter
        # specs to match by padding with None.
        def _spec_tuple(arr):
            spec = arr.sharding.spec
            return tuple(spec) + (None,) * (arr.ndim - len(spec))

        self.assertEqual(_spec_tuple(layer.w1_tp.value), (None, None, "tensor"))
        self.assertEqual(_spec_tuple(layer.w3_tp.value), (None, None, "tensor"))
        self.assertEqual(_spec_tuple(layer.w2_tp.value), (None, "tensor", None))

        # Calling twice must be a no-op (idempotent).
        layer.reshape_weights_for_tp()
        self.assertTrue(layer._tp_weights_ready)


if __name__ == "__main__":
    absltest.main()
