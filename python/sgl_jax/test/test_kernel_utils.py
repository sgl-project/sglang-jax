import unittest
from unittest.mock import MagicMock, patch

import jax.numpy as jnp

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3 import (
    _get_span_int_dtype,
)
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

    def test_get_span_int_dtype_requires_safe_capacity(self):
        self.assertEqual(
            _get_span_int_dtype(
                jnp.bfloat16,
                tpu_version=6,
                use_causal_mask=True,
                pages_per_seq=255,
                page_size=128,
            ),
            jnp.int16,
        )
        self.assertEqual(
            _get_span_int_dtype(
                jnp.bfloat16,
                tpu_version=6,
                use_causal_mask=True,
                pages_per_seq=256,
                page_size=128,
            ),
            jnp.int32,
        )

    def test_get_span_int_dtype_keeps_existing_guards(self):
        for q_dtype, tpu_version, use_causal_mask in (
            (jnp.float32, 6, True),
            (jnp.bfloat16, 5, True),
            (jnp.bfloat16, 6, False),
        ):
            self.assertEqual(
                _get_span_int_dtype(
                    q_dtype,
                    tpu_version=tpu_version,
                    use_causal_mask=use_causal_mask,
                    pages_per_seq=255,
                    page_size=128,
                ),
                jnp.int32,
            )


if __name__ == "__main__":
    unittest.main()
