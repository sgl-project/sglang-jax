"""One all_to_all over q_nope ++ q_rope equals the two separate all_to_alls (pure data movement)."""

import os
import unittest

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import jax.numpy as jnp
import numpy as np
from jax import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P


class TsaA2aMergeQTest(unittest.TestCase):
    def test_merged_equals_separate(self):
        devs = np.array(jax.devices()[:4])
        mesh = Mesh(devs, ("tensor",))
        T, Htp, Dn, Dr = 64, 2, 192, 64  # per-device q [T, H/tp, D]
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        qn = jax.random.normal(k1, (T * 4, Htp, Dn), jnp.float32).astype(jnp.bfloat16)
        qr = jax.random.normal(k2, (T * 4, Htp, Dr), jnp.float32).astype(jnp.bfloat16)

        def separate(a, b):
            at = jax.lax.all_to_all(a, "tensor", split_axis=0, concat_axis=1, tiled=True)
            bt = jax.lax.all_to_all(b, "tensor", split_axis=0, concat_axis=1, tiled=True)
            return at, bt

        def merged(a, b):
            ct = jax.lax.all_to_all(
                jnp.concatenate([a, b], axis=-1), "tensor", split_axis=0, concat_axis=1, tiled=True
            )
            return ct[..., : a.shape[-1]], ct[..., a.shape[-1] :]

        spec = P("tensor", None, None)
        run = lambda f: shard_map(
            f, mesh=mesh, in_specs=(spec, spec), out_specs=(spec, spec), check_vma=False
        )
        a1, b1 = run(separate)(qn, qr)
        a2, b2 = run(merged)(qn, qr)
        np.testing.assert_array_equal(
            np.asarray(a1).view(np.uint16), np.asarray(a2).view(np.uint16)
        )
        np.testing.assert_array_equal(
            np.asarray(b1).view(np.uint16), np.asarray(b2).view(np.uint16)
        )
        self.assertEqual(a2.shape, a1.shape)


if __name__ == "__main__":
    unittest.main()
