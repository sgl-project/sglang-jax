"""Split-flat one-shot all-reduce vs jax.lax.psum on the production mesh shapes.

Runs on a real 8-device TPU host (v7x 2x2x1).  Set ONESHOT_AR_INTERPRET=1 to run the
Pallas interpreter on 8 host CPU devices instead (XLA_FLAGS=--xla_force_host_platform_device_count=8;
slow, and cross-device semaphores are only approximated there).
"""

import os
import time
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.collectives.oneshot_allreduce import ALL_REDUCE_KERNELS

INTERPRET = os.environ.get("ONESHOT_AR_INTERPRET", "0") == "1"
MODES = os.environ.get("ONESHOT_AR_MODES", "split_flat").split(",")
TIMED_ITERS = int(os.environ.get("ONESHOT_AR_TIMED_ITERS", "0" if INTERPRET else "50"))


@unittest.skipIf(len(jax.devices()) < 8, "needs 8 devices")
class TestOneshotAllReduce(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(
            "\nDEVICES",
            [
                (d.id, getattr(d, "coords", None), getattr(d, "core_on_chip", None))
                for d in jax.devices()[:8]
            ],
        )
        try:
            from sgl_jax.srt.utils.mesh_utils import create_device_mesh

            m = create_device_mesh(ici_parallelism=[1, 8], dcn_parallelism=[1, 1])
            print(
                "PROD-MESH-ORDER",
                [
                    (d.id, getattr(d, "coords", None), getattr(d, "core_on_chip", None))
                    for d in m.devices.flat
                ],
            )
        except Exception as e:  # noqa: BLE001
            print("PROD-MESH-ORDER unavailable:", e)

    def _run(self, mesh, axes, reduce_axis, rows, cols, tag):
        for mode in MODES:
            self._run_mode(
                ALL_REDUCE_KERNELS[mode], mesh, axes, reduce_axis, rows, cols, f"{tag}-{mode}"
            )

    def _run_mode(self, ar_fn, mesh, axes, reduce_axis, rows, cols, tag):
        n = mesh.shape[reduce_axis]
        spec = P(reduce_axis, None)  # rows sharded over the reduce axis, columns whole
        x = jnp.asarray(
            np.random.default_rng(0).standard_normal((n * rows, cols)) * 0.5, dtype=jnp.bfloat16
        )
        x = jax.device_put(x, jax.sharding.NamedSharding(mesh, spec))

        def body(v):
            return ar_fn(v, reduce_axis, mesh_axis_names=axes, num_devices=n, interpret=INTERPRET)

        f_new = jax.jit(shard_map(body, mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False))
        f_ref = jax.jit(
            shard_map(
                lambda v: jax.lax.psum(v, reduce_axis),
                mesh=mesh,
                in_specs=spec,
                out_specs=spec,
                check_vma=False,
            )
        )
        got = f_new(x)
        ref = f_ref(x)
        g = np.asarray(got).astype(np.float32)
        r = np.asarray(ref).astype(np.float32)
        # exact oracle: shards summed in f32 on the host, then rounded once to bf16
        xs = np.asarray(x).astype(np.float32).reshape(n, rows, cols)
        exact = xs.sum(axis=0)
        exact_bf16 = np.asarray(jnp.asarray(exact, dtype=jnp.bfloat16)).astype(np.float32)
        # one bf16 ulp at the oracle's binade, absolute floor 2^-14
        ulp = np.maximum(
            2.0 ** np.floor(np.log2(np.maximum(np.abs(exact), 2.0**-14))) * 2.0**-7, 2.0**-14
        )
        per_dev = g.reshape(n, rows, cols)
        err_new = np.abs(per_dev[0] - exact) / ulp
        err_ref = np.abs(r.reshape(n, rows, cols)[0] - exact) / ulp
        print(
            f"ONESHOT-AR {tag} err/ulp vs f32 oracle: oneshot max={err_new.max():.2f} mean={err_new.mean():.3f} "
            f"| psum max={err_ref.max():.2f} mean={err_ref.mean():.3f} | oneshot==round(exact): "
            f"{np.mean(per_dev[0] == exact_bf16):.4f} | psum==round(exact): {np.mean(r.reshape(n, rows, cols)[0] == exact_bf16):.4f}"
        )
        # Gate: no worse than the production XLA psum (bf16 exchange rounds intermediates in both).
        self.assertLessEqual(
            float(err_new.mean()),
            float(err_ref.mean()) * 1.1 + 0.02,
            f"{tag}: mean error worse than psum",
        )
        self.assertLessEqual(
            float(err_new.max()),
            max(float(err_ref.max()) * 2.0, 4.0),
            f"{tag}: max error worse than psum",
        )
        self.assertTrue(np.all(per_dev == per_dev[0]), f"{tag}: devices disagree")
        # repeated calls must keep working (barrier/ack release between instances)
        for _ in range(3):
            g2 = np.asarray(f_new(x)).astype(np.float32)
            self.assertTrue(np.all(g2 == g), f"{tag}: result changed across calls")
        if TIMED_ITERS:
            k = TIMED_ITERS

            def chain(fn):
                def run(v):
                    return jax.lax.fori_loop(0, k, lambda _, a: (fn(a) * 0.125).astype(a.dtype), v)

                return jax.jit(
                    shard_map(run, mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False)
                )

            for fn, name in (
                (
                    lambda v: ar_fn(
                        v, reduce_axis, mesh_axis_names=axes, num_devices=n, interpret=INTERPRET
                    ),
                    "oneshot",
                ),
                (lambda v: jax.lax.psum(v, reduce_axis), "psum"),
            ):
                f = chain(fn)
                for _ in range(3):
                    f(got).block_until_ready()
                best = 1e9
                for _ in range(5):
                    t0 = time.perf_counter()
                    f(got).block_until_ready()
                    best = min(best, (time.perf_counter() - t0) / k * 1e6)
                print(
                    f"ONESHOT-AR {tag} {name} rows={rows} cols={cols} KB={rows*cols*2/1024:.0f} us_per_call={best:.2f} (chain of {k} in one jit)"
                )

    def test_1d_mesh_48kb(self):
        devs = np.array(jax.devices()[:8])
        self._run(Mesh(devs, ("x",)), ("x",), "x", 4, 6144, "1d-48KB")

    def test_prod_mesh_48kb(self):
        devs = np.array(jax.devices()[:8]).reshape(1, 8)
        self._run(
            Mesh(devs, ("data", "tensor")), ("data", "tensor"), "tensor", 4, 6144, "prod-48KB"
        )

    def test_prod_mesh_384kb(self):
        devs = np.array(jax.devices()[:8]).reshape(1, 8)
        self._run(
            Mesh(devs, ("data", "tensor")), ("data", "tensor"), "tensor", 32, 6144, "prod-384KB"
        )

    def test_prod_mesh_1536kb(self):
        devs = np.array(jax.devices()[:8]).reshape(1, 8)
        self._run(
            Mesh(devs, ("data", "tensor")), ("data", "tensor"), "tensor", 128, 6144, "prod-1536KB"
        )


if __name__ == "__main__":
    unittest.main()
