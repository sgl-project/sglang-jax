"""gmm_v2 skip-prefix metadata scan: bit-exact vs the full scan for every EP rank, and timing.

Run on a TPU host. Shapes mirror GLM-5.2 int4 experts (E=256 total, 32 local, k=6144, n=2048)."""

import os
import time
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm_v2 import gmm_v2

E, LOCAL, K, N = 256, 32, 6144, 2048
ITERS = int(os.environ.get("GMM_SKIP_PREFIX_ITERS", "50"))


@unittest.skipIf(jax.devices()[0].platform != "tpu", "TPU kernel")
class TestGmmV2SkipPrefix(unittest.TestCase):
    def _case(self, m_rows, rank, seed=0):
        rng = np.random.default_rng(seed)
        # Global token->expert counts (all 256 experts). The rank's LOCAL groups get the same
        # size pattern for every rank (so per-call work is rank-invariant and the timing isolates
        # the metadata scan over the non-local prefix); the other groups share the remaining rows.
        local = rng.multinomial(m_rows // 2, np.ones(LOCAL) / LOCAL).astype(np.int32)
        others = rng.multinomial(m_rows - m_rows // 2, np.ones(E - LOCAL) / (E - LOCAL)).astype(
            np.int32
        )
        counts = np.concatenate([others[: rank * LOCAL], local, others[rank * LOCAL :]]).astype(
            np.int32
        )
        assert counts.sum() == m_rows and counts.shape == (E,)
        lhs = jnp.asarray(rng.standard_normal((m_rows, K)), jnp.bfloat16)
        rhs = jnp.asarray(rng.integers(-8, 8, (LOCAL, K, N)), jnp.int8)
        scale = jnp.asarray(rng.uniform(0.5, 1.5, (LOCAL, 1, 1, N)), jnp.float32)
        gs = jnp.asarray(counts)
        off = jnp.asarray([rank * LOCAL], jnp.int32)
        f_full = jax.jit(
            lambda a, w, sc, g, o: gmm_v2(
                a,
                w,
                g,
                rhs_scale=sc,
                group_offset=o,
                preferred_element_type=jnp.bfloat16,
                skip_prefix_groups=False,
            )
        )
        f_skip = jax.jit(
            lambda a, w, sc, g, o: gmm_v2(
                a,
                w,
                g,
                rhs_scale=sc,
                group_offset=o,
                preferred_element_type=jnp.bfloat16,
                skip_prefix_groups=True,
            )
        )
        y0 = f_full(lhs, rhs, scale, gs, off)
        y1 = f_skip(lhs, rhs, scale, gs, off)
        same = bool(jnp.all(y0 == y1))
        self.assertTrue(same, f"m={m_rows} rank={rank}: skip-prefix output differs from full scan")
        if ITERS:
            res = []
            for f in (f_full, f_skip):
                f(lhs, rhs, scale, gs, off).block_until_ready()
                t0 = time.perf_counter()
                for _ in range(ITERS):
                    y = f(lhs, rhs, scale, gs, off)
                y.block_until_ready()
                res.append((time.perf_counter() - t0) / ITERS * 1e6)
            print(
                f"GMM-SKIP m={m_rows} rank={rank}: full {res[0]:.1f} us/call, skip {res[1]:.1f} us/call, exact={same}"
            )

    def test_ranks_bs1(self):
        for rank in (0, 3, 7):
            self._case(32, rank)

    def test_ranks_cc8(self):
        for rank in (0, 3, 7):
            self._case(256, rank)

    def test_ranks_prefill(self):
        for rank in (0, 7):
            self._case(4096, rank, seed=1)


if __name__ == "__main__":
    unittest.main()
