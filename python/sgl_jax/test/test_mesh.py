import unittest

from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.test.test_utils import CustomTestCase


class TestMesh(CustomTestCase):
    def test_mesh_with_no_device_indexes(self):
        mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
        self.assertEqual(mesh.shape.get("data"), 1, "dp should be 1")
        self.assertEqual(mesh.shape.get("tensor"), 4, "tp should be 1")

    def test_mesh_with_device_indexes(self):
        mesh = create_device_mesh(
            ici_parallelism=[1, -1], dcn_parallelism=[1, 1], device_indexes=[0, 1]
        )
        self.assertEqual(mesh.shape.get("data"), 1, "dp should be 1")
        self.assertEqual(mesh.shape.get("tensor"), 2, "tp should be 1")

    def test_mesh_with_duplicated_device_indexes(self):
        mesh = create_device_mesh(
            ici_parallelism=[1, -1], dcn_parallelism=[1, 1], device_indexes=[0, 1, 0, 1]
        )
        self.assertEqual(mesh.shape.get("data"), 1, "dp should be 1")
        self.assertEqual(mesh.shape.get("tensor"), 2, "tp should be 1")

    def test_mesh_with_large_device_indexes(self):
        try:
            _ = create_device_mesh(
                ici_parallelism=[1, -1], dcn_parallelism=[1, 1], device_indexes=[0, 4]
            )
        except Exception as e:
            self.assertTrue(
                isinstance(e, RuntimeError), "the device indexes have exceeded the len of devices"
            )


if __name__ == "__main__":
    unittest.main()
