import unittest
from unittest.mock import MagicMock, patch

from sgl_jax.srt.kernels.ragged_paged_attention.util import get_tpu_version


class TestKernelUtils(unittest.TestCase):

    @patch("jax.devices")
    def test_get_tpu_version(self, mock_jax_devices):
        # Test TPU v5
        mock_device = MagicMock()
        mock_device.device_kind = "TPU v5"
        mock_jax_devices.return_value = [mock_device]
        self.assertEqual(get_tpu_version(), 5)

        # Test TPU v4 lite
        mock_device.device_kind = "TPU v4 lite"
        mock_jax_devices.return_value = [mock_device]
        self.assertEqual(get_tpu_version(), 4)

        # Test TPU v5 lite
        mock_device.device_kind = "TPU v5 lite"
        mock_jax_devices.return_value = [mock_device]
        self.assertEqual(get_tpu_version(), 5)

        # Test TPU v6
        mock_device.device_kind = "TPU v6 lite"
        mock_jax_devices.return_value = [mock_device]
        self.assertEqual(get_tpu_version(), 6)

        # Test TPU7x
        mock_device.device_kind = "TPU7x"
        mock_jax_devices.return_value = [mock_device]
        self.assertEqual(get_tpu_version(), 7)

        # Test CPU
        mock_device.device_kind = "CPU"
        mock_jax_devices.return_value = [mock_device]
        self.assertEqual(get_tpu_version(), -1)

        # Test GPU
        mock_device.device_kind = "NVIDIA H100"
        mock_jax_devices.return_value = [mock_device]
        self.assertEqual(get_tpu_version(), -1)


if __name__ == "__main__":
    unittest.main()
