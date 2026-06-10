"""Tests for mm_core.pad_value (pure int math; runnable on any interpreter)."""

import unittest

from sgl_jax.srt.mm_core.pad_value import (
    MM_PAD_SHIFT_VALUE,
    derive_pad_value,
    sanity_check_mm_pad_shift_value,
)

_PAD_MOD = 1 << 30


class TestPadValue(unittest.TestCase):
    def test_in_range(self):
        for h in (0, 1, 2**24, 2**30, 2**31, 2**63 - 1):
            pv = derive_pad_value(h)
            self.assertGreaterEqual(pv, MM_PAD_SHIFT_VALUE)
            self.assertLess(pv, MM_PAD_SHIFT_VALUE + _PAD_MOD)

    def test_above_realistic_vocab(self):
        # Any realistic LLM vocab (<= a few hundred k) sits below the pad_value floor,
        # so a pad_value can never collide with a real token id.
        self.assertGreater(derive_pad_value(0), 300_000)

    def test_deterministic(self):
        self.assertEqual(derive_pad_value(123456789), derive_pad_value(123456789))

    def test_distinct_hashes_usually_distinct(self):
        self.assertNotEqual(derive_pad_value(1), derive_pad_value(2))

    def test_sanity_check(self):
        sanity_check_mm_pad_shift_value(256_000)  # ok, no raise
        sanity_check_mm_pad_shift_value(MM_PAD_SHIFT_VALUE)  # boundary ok
        with self.assertRaises(ValueError):
            sanity_check_mm_pad_shift_value(MM_PAD_SHIFT_VALUE + 1)


if __name__ == "__main__":
    unittest.main()
