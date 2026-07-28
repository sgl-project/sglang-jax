import unittest

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from sgl_jax.srt.utils.jax_utils import device_array


def _make_mesh():
    return Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("data", "tensor"))


class TestDeviceArrayShardingReuse(unittest.TestCase):
    """device_array must hand out a canonical NamedSharding per (mesh, spec).

    jaxlib's PjitFunctionCache fast-path compares input shardings by object
    pointer; a fresh NamedSharding per call defeats it and forces the pjit
    python slow path on every step (issue #1452).
    """

    def test_sharding_is_pointer_stable_across_calls(self):
        mesh = _make_mesh()
        data = np.arange(8, dtype=np.int32)
        (a1,) = device_array((data,), sharding=NamedSharding(mesh, PartitionSpec("data")))
        (a2,) = device_array((data,), sharding=NamedSharding(mesh, PartitionSpec("data")))
        self.assertIs(a1.sharding, a2.sharding)

    def test_sharding_is_pointer_stable_across_equal_meshes(self):
        # Equal-but-distinct Mesh objects must also canonicalize to one object.
        data = np.arange(8, dtype=np.int32)
        (a1,) = device_array((data,), sharding=NamedSharding(_make_mesh(), PartitionSpec("data")))
        (a2,) = device_array((data,), sharding=NamedSharding(_make_mesh(), PartitionSpec("data")))
        self.assertIs(a1.sharding, a2.sharding)

    def test_distinct_specs_get_distinct_shardings(self):
        mesh = _make_mesh()
        data = np.arange(8, dtype=np.int32)
        (a1,) = device_array((data,), sharding=NamedSharding(mesh, PartitionSpec("data")))
        (a2,) = device_array((data,), sharding=NamedSharding(mesh, PartitionSpec(None)))
        self.assertEqual(a1.sharding.spec, PartitionSpec("data"))
        self.assertEqual(a2.sharding.spec, PartitionSpec(None))

    def test_values_unchanged(self):
        mesh = _make_mesh()
        data = np.arange(8, dtype=np.int32)
        (arr,) = device_array((data,), sharding=NamedSharding(mesh, PartitionSpec("data")))
        np.testing.assert_array_equal(np.asarray(arr), data)

    def test_none_sharding_path_unchanged(self):
        arr = device_array(np.arange(4, dtype=np.int32))
        np.testing.assert_array_equal(np.asarray(arr), np.arange(4, dtype=np.int32))


if __name__ == "__main__":
    unittest.main()


class TestShapeDtypeStructDefeatsCppCache(unittest.TestCase):
    """A jax.ShapeDtypeStruct leaf among jit args fails ComputeCallSignature
    in jaxlib and forces the python dispatch path on every call (#1452).
    ModelRunner replaces such placeholders with zero-length arrays at init."""

    def test_sds_arg_misses_every_call_and_arrays_do_not(self):
        import jax._src.test_util as jtu

        a = jax.device_put(np.zeros((4,), np.float32), jax.devices()[0])
        sds = jax.ShapeDtypeStruct((4,), np.float32)
        f = jax.jit(lambda xs: len(xs))
        f([a, a])
        with jtu.count_pjit_cpp_cache_miss() as c:
            f([a, a])
        self.assertEqual(c(), 0)
        f([a, sds])
        misses = []
        for _ in range(3):
            with jtu.count_pjit_cpp_cache_miss() as c:
                f([a, sds])
            misses.append(c())
        self.assertEqual(misses, [1, 1, 1])
        # replacement with a real (even zero-length) array restores the fastpath
        fixed = jax.device_put(np.zeros((0,), np.float32), jax.devices()[0])
        f([a, fixed])
        with jtu.count_pjit_cpp_cache_miss() as c:
            f([a, fixed])
        self.assertEqual(c(), 0)
