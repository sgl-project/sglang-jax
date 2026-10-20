"""EPMoE.dispatch_stage + experts_stage == EPMoE.__call__ (CPU, 4 host devices, ep=4).

The two-stage call is the prefill micro-batch PIN=4 path (SparseCore-stream ordering of the
two windows' MoE all-reduces); it must be value-identical to the single call, with and without
a combine pin, and the anchor must be a replicated f32[1, 1] on the model mesh."""

import os
import unittest

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.moe import EPMoE


def _mesh():
    devs = np.array(jax.devices()).reshape(len(jax.devices()), 1)
    return Mesh(
        devs,
        axis_names=("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )


class EPMoETwoStageTest(unittest.TestCase):
    def test_two_stage_matches_single_call(self):
        mesh = _mesh()
        ep = len(jax.devices())
        H, I, E, K, T = 256, 512, 2 * ep, 2, 64
        with jax.set_mesh(mesh):
            moe = EPMoE(
                hidden_size=H,
                num_experts=E,
                num_experts_per_tok=K,
                ep_size=ep,
                mesh=mesh,
                intermediate_dim=I,
                quantization_config=None,
            )
        key = jax.random.PRNGKey(0)
        k0, k1, k2, kx, kt = jax.random.split(key, 5)
        from flax import nnx

        with jax.set_mesh(moe.moe_mesh):
            moe.wi_0 = nnx.Param(
                jax.random.normal(k0, (E, H, I), jnp.bfloat16) * 0.05,
                out_sharding=P("expert", None, "tensor"),
            )
            moe.wi_1 = nnx.Param(
                jax.random.normal(k1, (E, H, I), jnp.bfloat16) * 0.05,
                out_sharding=P("expert", None, "tensor"),
            )
            moe.wo = nnx.Param(
                jax.random.normal(k2, (E, I, H), jnp.bfloat16) * 0.05,
                out_sharding=P("expert", "tensor", None),
            )
        x = jax.random.normal(kx, (T, H), jnp.bfloat16)
        topk_ids = jax.random.randint(kt, (T, K), 0, E, jnp.int32)
        topk_w = jnp.full((T, K), 1.0 / K, jnp.bfloat16)
        with jax.set_mesh(mesh):
            ref = moe(x, topk_w, topk_ids)
            state, anchor = moe.dispatch_stage(x, topk_ids)
            out = moe.experts_stage(state, topk_w)
            pin = jnp.asarray([[3.0]], jnp.bfloat16)
            out_pinned = moe.experts_stage(state, topk_w, combine_pin=pin)
            out_nan_pin = moe.experts_stage(state, topk_w, combine_pin=jnp.asarray([[jnp.nan]]))
        self.assertEqual(anchor.shape, (1, 1))
        self.assertEqual(anchor.dtype, jnp.float32)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))
        np.testing.assert_array_equal(np.asarray(out_pinned), np.asarray(ref))
        np.testing.assert_array_equal(np.asarray(out_nan_pin), np.asarray(ref))
        # anchor is a real data edge on the gathered rows (one of x's elements)
        self.assertTrue(np.isfinite(np.asarray(anchor)).all())


if __name__ == "__main__":
    unittest.main()
