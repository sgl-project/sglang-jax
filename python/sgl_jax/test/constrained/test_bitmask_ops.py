import ctypes
import unittest

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.constrained.bitmask_ops import (
    allocate_token_bitmask,
    apply_token_bitmask,
    fill_token_bitmask,
    unpack_bitmask,
)


class MockLLInterpreter:
    """A minimal mock of LLInterpreter that writes an int32 mask pattern.

    It fills the provided pointer with precomputed int32 values that represent
    allowed token indices via bit positions.
    """

    def __init__(self, values: np.ndarray):
        assert values.dtype == np.int32
        self.values = values

    def unsafe_compute_mask_ptr(self, ptr: int, nbytes: int):
        expected_nbytes = self.values.size * 4
        assert nbytes == expected_nbytes
        buf = (ctypes.c_int32 * self.values.size).from_address(ptr)
        for i in range(self.values.size):
            buf[i] = int(self.values[i])


class BitmaskOpsTest(unittest.TestCase):
    def test_allocate_token_bitmask(self):
        batch_size = 2
        vocab_size = 33  # requires 2 int32 slots
        mask = allocate_token_bitmask(batch_size, vocab_size)
        self.assertEqual(mask.shape, (2, 2))
        self.assertEqual(mask.dtype, np.int32)
        self.assertTrue(np.all(mask == 0))

    def test_fill_and_unpack_bitmask(self):
        # Prepare a 2-row bitmask; vocab_size=80 -> 3 int32 values per row
        batch_size = 2
        vocab_size = 80
        mask = allocate_token_bitmask(batch_size, vocab_size)

        # Allow token indices spread across three int32s, including edge bits
        # int 0: 0, 5, 31
        # int 1: 32, 39, 63
        # int 2: 64, 70, 79
        allowed = [0, 5, 31, 32, 39, 63, 64, 70, 79]
        val0 = (1 << 0) | (1 << 5) | (1 << 31)
        val1 = (1 << 0) | (1 << (39 - 32)) | (1 << (63 - 32))  # indices 32, 39, 63
        val2 = (1 << 0) | (1 << (70 - 64)) | (1 << (79 - 64))  # indices 64, 70, 79
        values = np.array([val0, val1, val2], dtype=np.int64).astype(np.int32)

        matcher = MockLLInterpreter(values)
        # Fill only row 1; row 0 remains zeros
        fill_token_bitmask(matcher, mask, batch_idx=1)

        # Row 0 should be all zeros
        np.testing.assert_array_equal(mask[0], np.zeros_like(mask[0]))
        # Row 1 should match the pattern
        np.testing.assert_array_equal(mask[1], values)

        # Unpack and verify booleans
        unpacked = unpack_bitmask(jnp.array(mask))  # shape [2, num_int32 * 32]
        self.assertEqual(unpacked.shape[0], 2)
        num_bits = unpacked.shape[1]
        # Check row 0: all False
        self.assertFalse(bool(unpacked[0].any()))
        # Check row 1: allowed indices are True; others False
        row1 = np.array(unpacked[1])  # to numpy for easy indexing
        for i in range(num_bits):
            if i in allowed:
                self.assertTrue(row1[i])
            else:
                self.assertFalse(row1[i])

    def test_apply_token_bitmask(self):
        # vocab_size=80 -> 3 int32s per row
        batch_size = 2
        vocab_size = 80
        mask = allocate_token_bitmask(batch_size, vocab_size)

        # Allow only a few indices for row 1
        allowed = [0, 5, 31, 32, 39, 63, 64, 70, 79]
        val0 = (1 << 0) | (1 << 5) | (1 << 31)
        val1 = (1 << 0) | (1 << (39 - 32)) | (1 << (63 - 32))
        val2 = (1 << 0) | (1 << (70 - 64)) | (1 << (79 - 64))
        mask[1, :] = np.array([val0, val1, val2], dtype=np.int64).astype(np.int32)

        logits = jnp.arange(batch_size * vocab_size, dtype=jnp.float32).reshape(
            batch_size, vocab_size
        )

        masked = apply_token_bitmask(logits, jnp.array(mask))
        masked_np = np.array(masked)

        # Row 0 should be entirely -inf (no allowed tokens)
        self.assertTrue(np.isneginf(masked_np[0]).all())

        # Row 1: allowed indices keep original values, others -inf
        for i in range(vocab_size):
            if i in allowed:
                self.assertTrue(
                    np.isfinite(masked_np[1, i])
                    and np.isclose(masked_np[1, i], float(vocab_size + i))
                )
            else:
                self.assertTrue(np.isneginf(masked_np[1, i]))


if __name__ == "__main__":
    unittest.main()
