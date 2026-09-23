"""CHECK report of the chained draft pool on an explicit dp=2 CPU mesh.

The report runs inside the fused draft-extend JIT: the step-0 slot lookup
gathers data-sharded metadata (must be an auto-sharded region) and the result
must be arrays only (JIT outputs). This exercises _chain_pool_report exactly as
the fused function calls it.
"""

import os

if "--xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
    )
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.speculative.draft_extend_fused import (
    _chain_pool_leaf_names,
    _chain_pool_report,
)

PS, PK, D = 128, 2, 16
BS, N = 2, 4  # requests x draft tokens per request (T = 8)
PAGES_PER_REQ = 4


def _mesh(dp):
    devices = np.array(jax.devices()[:4]).reshape(dp, 4 // dp)
    return jax.sharding.Mesh(
        devices,
        axis_names=("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )


class ChainPoolCheckTest(unittest.TestCase):
    def _run(self, dp):
        mesh = _mesh(dp)
        bs = dp  # one request per dp rank; per-rank cu arrays are [0, N] / [0, pages*PS]
        lens = [200, 300][:bs]
        pi_np = np.arange(bs * PAGES_PER_REQ, dtype=np.int32) + 1
        with jax.set_mesh(mesh):
            data = NamedSharding(mesh, P("data"))
            seq_lens = jax.device_put(jnp.array(lens, jnp.int32), data)
            cu_q = jax.device_put(jnp.tile(jnp.array([0, N], jnp.int32), bs), data)
            cu_kv = jax.device_put(
                jnp.tile(jnp.array([0, PAGES_PER_REQ * PS], jnp.int32), bs), data
            )
            page_indices = jax.device_put(jnp.asarray(pi_np), data)
            n_pages = bs * PAGES_PER_REQ + 2
            pool = jax.device_put(
                jnp.zeros((n_pages, PS // PK, PK, D), jnp.bfloat16),
                NamedSharding(mesh, P(None, None, None, "tensor")),
            )
            like = jax.device_put(jnp.zeros((bs * N,), jnp.int32), data)
            # expected step-0 window slots (numpy): request s writes kv index seq_len - N + i
            # at page_indices[s*pages + idx // PS] * PS + idx % PS
            exp = []
            for s_, L in enumerate(lens):
                for i in range(N):
                    idx = L - N + i
                    exp.append(int(pi_np[s_ * PAGES_PER_REQ + idx // PS]) * PS + idx % PS)
            # vN differs from v0 at one window slot of request 0 and one fresh slot (window end + 1)
            changed = [exp[1], exp[N - 1] + 1]
            flat = pool.reshape(-1, D)
            vN_flat = flat.at[jnp.array(changed)].set(jnp.ones((D,), jnp.bfloat16))
            v0 = {"token_to_kv_pool": [pool]}
            vN = {"token_to_kv_pool": [vN_flat.reshape(pool.shape)]}

            @jax.jit
            def run(v0, vN, seq_lens, cu_q, cu_kv, page_indices, like):
                md_ = SimpleNamespace(
                    seq_lens=seq_lens, cu_q_lens=cu_q, cu_kv_lens=cu_kv, page_indices=page_indices
                )
                return _chain_pool_report(v0, vN, md_, bs * N, PS, like)

            counts, firsts, loc0 = run(v0, vN, seq_lens, cu_q, cu_kv, page_indices, like)
            for a in (counts, firsts, loc0):
                self.assertIsInstance(a, jax.Array)  # arrays only, no Python objects
            self.assertEqual(int(counts[0]), 2)
            rows = [int(x) for x in np.asarray(firsts[0]) if x >= 0]
            self.assertEqual(sorted(rows), sorted(changed))
            self.assertEqual(
                _chain_pool_leaf_names(vN), [f"['token_to_kv_pool'][0] shape={tuple(pool.shape)}"]
            )
            if dp == 1:
                # the slot lookup is exact on a single data rank (tp16 dp=1 is the production case)
                np.testing.assert_array_equal(np.asarray(loc0), np.array(exp, np.int32))

    def test_dp1(self):
        self._run(1)

    def test_dp2(self):
        self._run(2)


if __name__ == "__main__":
    unittest.main()
