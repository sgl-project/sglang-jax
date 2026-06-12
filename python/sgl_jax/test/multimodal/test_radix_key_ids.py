"""Scheme-B radix cache key logic (Req.radix_key_ids / radix_fill_ids), review M-4.

Guards the invariant that a multimodal request's padded radix key (cache_input_ids) is
length-aligned with the clean origin_input_ids, and that a mismatch fails LOUDLY instead of
silently falling back to origin_input_ids (which would collide different-media requests on a
shared text prefix -> wrong KV reuse, cross-request answer bleed). The hard per-request abort
lives at intake (Scheduler.handle_generate_request); these tests cover the property tripwires.

The radix logic depends only on plain-list attributes, so we exercise it on a minimal carrier
without constructing a full Req (which needs sampling params, tokenizer, etc.).
"""

import unittest

from sgl_jax.srt.managers.schedule_batch import Req


class _Carrier:
    """Holds just the attributes Req.radix_key_ids / radix_fill_ids read."""

    def __init__(self, cache_input_ids, origin_input_ids, fill_ids, output_ids):
        self.cache_input_ids = cache_input_ids
        self.origin_input_ids = origin_input_ids
        self.fill_ids = fill_ids
        self.output_ids = output_ids


def _key(c):
    return Req.radix_key_ids.fget(c)  # property -> call its getter on the carrier


def _fill(c):
    return Req.radix_fill_ids(c)  # plain method


class TestRadixKeyIds(unittest.TestCase):
    def test_text_request_is_byte_identical(self):
        # cache_input_ids None (text req): key == origin_input_ids, fill == fill_ids (as-is path).
        c = _Carrier(None, [1, 2, 3], [1, 2, 3], [])
        self.assertEqual(_key(c), [1, 2, 3])
        self.assertEqual(_fill(c), [1, 2, 3])

    def test_mm_request_uses_cache_key(self):
        # length-matched mm req: the padded copy is the radix key (per-image pad_value signature).
        c = _Carrier([10, 11, 12], [1, 2, 3], [1, 2, 3], [])
        self.assertEqual(_key(c), [10, 11, 12])
        self.assertEqual(_fill(c), [10, 11, 12])

    def test_mm_request_chunked_keeps_output_tail(self):
        # chunked/decoded: prompt portion from cache_input_ids, output tail from fill_ids.
        c = _Carrier([10, 11, 12], [1, 2, 3], [1, 2, 3, 99], [99])  # n_prompt = 4 - 1 = 3
        self.assertEqual(_fill(c), [10, 11, 12, 99])

    def test_length_mismatch_raises_not_silent_fallback(self):
        # The M-4 bug: a mismatched cache key must NOT silently fall back to origin_input_ids.
        bad = _Carrier([10, 11], [1, 2, 3], [1, 2, 3], [])
        with self.assertRaises(AssertionError) as cm_key:
            _key(bad)
        self.assertIn("M-4", str(cm_key.exception))
        with self.assertRaises(AssertionError) as cm_fill:
            _fill(bad)
        self.assertIn("M-4", str(cm_fill.exception))


if __name__ == "__main__":
    unittest.main()
