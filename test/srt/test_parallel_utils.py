import unittest
from unittest.mock import patch

from sgl_jax.srt.utils.parallel_utils import should_scatter


class TestShouldScatter(unittest.TestCase):
    def setUp(self):
        self.patcher = patch(
            "sgl_jax.srt.utils.parallel_utils.global_config.tpu_scatter_min_local_size",
            128,
        )
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_single_device_never_scatters(self):
        self.assertFalse(should_scatter(dim_size=8192, num_devices=1))
        self.assertFalse(should_scatter(dim_size=8192, num_devices=0))

    def test_below_min_local_size_does_not_scatter(self):
        # 4 * 128 = 512 is the minimum; 256 is below.
        self.assertFalse(should_scatter(dim_size=256, num_devices=4))

    def test_exactly_min_local_size_scatters(self):
        self.assertTrue(should_scatter(dim_size=512, num_devices=4))

    def test_above_min_local_size_scatters(self):
        self.assertTrue(should_scatter(dim_size=2048, num_devices=4))

    def test_not_divisible_does_not_scatter(self):
        # Above threshold (4 * 128 = 512) but 513 % 4 != 0.
        self.assertFalse(should_scatter(dim_size=513, num_devices=4))

    def test_respects_configured_min_local_size(self):
        with patch(
            "sgl_jax.srt.utils.parallel_utils.global_config.tpu_scatter_min_local_size",
            64,
        ):
            # 4 * 64 = 256, so 256 now qualifies.
            self.assertTrue(should_scatter(dim_size=256, num_devices=4))
            self.assertFalse(should_scatter(dim_size=128, num_devices=4))


if __name__ == "__main__":
    unittest.main()
