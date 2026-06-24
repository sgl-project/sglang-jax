"""Recurrent state sizing config + validation (#1380).

Covers the pure ``_recurrent_slot_factor`` helper (slots per concurrent request)
and the ``ServerArgs`` static validation / normalization for the new
``--recurrent-track-interval`` knob gated by ``--enable-recurrent-extra-buffer``. No
server launch: direct ``ServerArgs(...)`` construction and a plain namespace for
the helper.
"""

import unittest
from types import SimpleNamespace

from sgl_jax.srt.model_executor.model_runner_kv_cache_mixin import (
    _enforce_recurrent_state_server_constraints,
    _recurrent_slot_factor,
)
from sgl_jax.srt.server_args import ServerArgs


def _factor_args(
    *,
    disable_radix_cache=False,
    enable_unified_radix_tree=False,
    enable_recurrent_extra_buffer=False,
):
    return SimpleNamespace(
        disable_radix_cache=disable_radix_cache,
        enable_unified_radix_tree=enable_unified_radix_tree,
        enable_recurrent_extra_buffer=enable_recurrent_extra_buffer,
    )


class TestRecurrentSlotFactor(unittest.TestCase):
    """One running slot plus headroom: 1 legacy, 2 radix, 3 extra-buffer."""

    def test_legacy_disabled_radix_is_one(self):
        self.assertEqual(_recurrent_slot_factor(_factor_args(disable_radix_cache=True)), 1)

    def test_plain_no_radix_is_one(self):
        self.assertEqual(_recurrent_slot_factor(_factor_args()), 1)

    def test_unified_radix_without_extra_buffer_is_two(self):
        self.assertEqual(_recurrent_slot_factor(_factor_args(enable_unified_radix_tree=True)), 2)

    def test_extra_buffer_is_three(self):
        self.assertEqual(
            _recurrent_slot_factor(
                _factor_args(
                    enable_unified_radix_tree=True,
                    enable_recurrent_extra_buffer=True,
                )
            ),
            3,
        )

    def test_extra_buffer_with_disabled_radix_is_one(self):
        # disable_radix_cache wins (legacy 1:1 path); extra-buffer is invalid
        # there and rejected by server_args / constraints elsewhere.
        self.assertEqual(
            _recurrent_slot_factor(
                _factor_args(disable_radix_cache=True, enable_recurrent_extra_buffer=True)
            ),
            1,
        )


def _server_args(**overrides):
    kwargs = dict(
        model_path="dummy",
        enable_unified_radix_tree=True,
        enable_recurrent_extra_buffer=True,
        page_size=128,
    )
    kwargs.update(overrides)
    return ServerArgs(**kwargs)


class TestExtraBufferStaticValidation(unittest.TestCase):
    """``__post_init__`` static checks gated by ``enable_recurrent_extra_buffer``."""

    def test_extra_buffer_requires_page_size_gt_1(self):
        with self.assertRaises((ValueError, AssertionError)):
            _server_args(page_size=1)

    def test_track_interval_none_defaults_to_chunked_prefill_size(self):
        # Default snapshot interval = chunked_prefill_size (NOT page_size), so the
        # recurrent feature never force-splits prefill below the chunk size (which
        # would inherit the chunk-size-sensitive accuracy penalty).
        sa = _server_args(recurrent_track_interval=None, page_size=256, chunked_prefill_size=512)
        self.assertEqual(sa.recurrent_track_interval, 512)

    def test_track_interval_none_falls_back_to_page_size_when_chunked_disabled(self):
        sa = _server_args(recurrent_track_interval=None, page_size=256, chunked_prefill_size=-1)
        self.assertEqual(sa.recurrent_track_interval, 256)

    def test_track_interval_above_chunked_prefill_allowed(self):
        # interval > chunk budget is ALLOWED (warns): a chunk never reaches a
        # snapshot boundary, so sub-interval prompts cache nothing, but the
        # request still progresses (the zero-cache-len chunk skip advances
        # prefix_indices -- no stall). It must resolve to the requested value.
        sa = _server_args(recurrent_track_interval=1024, chunked_prefill_size=512)
        self.assertEqual(sa.recurrent_track_interval, 1024)

    def test_track_interval_below_chunked_prefill_allowed(self):
        # Smaller interval is allowed (finer cache granularity) but warns about
        # the accuracy cost; it must still resolve to the requested value.
        sa = _server_args(recurrent_track_interval=128, chunked_prefill_size=512)
        self.assertEqual(sa.recurrent_track_interval, 128)

    def test_track_interval_zero_rejected(self):
        with self.assertRaises((ValueError, AssertionError)):
            _server_args(recurrent_track_interval=0)

    def test_track_interval_not_divisible_by_page_size_rejected(self):
        with self.assertRaises((ValueError, AssertionError)):
            _server_args(recurrent_track_interval=200, page_size=128)

    def test_default_interval_rejects_non_page_aligned_chunk(self):
        # The interval auto-derives from chunked_prefill_size, so a non-page-aligned
        # chunk must fail with a message naming chunked_prefill_size (not the
        # auto-derived interval), even though check_server_args runs later.
        with self.assertRaises(ValueError) as cm:
            _server_args(recurrent_track_interval=None, chunked_prefill_size=1000, page_size=128)
        self.assertIn("chunked-prefill-size", str(cm.exception))

    def test_extra_buffer_with_speculative_rejected(self):
        with self.assertRaises((ValueError, AssertionError)):
            _server_args(speculative_algorithm="EAGLE")

    def test_extra_buffer_with_mixed_chunk_rejected(self):
        with self.assertRaises((ValueError, AssertionError)):
            _server_args(enable_mixed_chunk=True)

    def test_no_extra_buffer_skips_static_checks(self):
        # Page size 1 + speculative are fine when extra-buffer is off.
        sa = ServerArgs(
            model_path="dummy",
            page_size=1,
            enable_recurrent_extra_buffer=False,
        )
        self.assertIsNone(sa.recurrent_track_interval)


class TestRecurrentStateConstraints(unittest.TestCase):
    """``_enforce_recurrent_state_server_constraints`` model-dependent checks."""

    def test_extra_buffer_incompatible_with_disable_radix_cache(self):
        sa = _server_args(disable_radix_cache=True)
        with self.assertRaises(AssertionError):
            _enforce_recurrent_state_server_constraints(sa)

    def test_extra_buffer_requires_unified_radix_tree(self):
        sa = _server_args(enable_unified_radix_tree=False)
        with self.assertRaises(AssertionError):
            _enforce_recurrent_state_server_constraints(sa)

    def test_extra_buffer_unified_radix_passes(self):
        sa = _server_args()
        _enforce_recurrent_state_server_constraints(sa)  # no raise

    def test_extra_buffer_rejected_for_lightning(self):
        # GLA/Lightning has no extra-buffer support; reject at init even though
        # the radix/page constraints are otherwise satisfied.
        sa = _server_args()
        with self.assertRaises(AssertionError):
            _enforce_recurrent_state_server_constraints(sa, is_lightning=True)

    def test_lightning_without_extra_buffer_passes(self):
        # A Lightning model on the non-extra-buffer recurrent radix path is fine.
        sa = ServerArgs(
            model_path="dummy",
            enable_unified_radix_tree=True,
            page_size=1,
        )
        _enforce_recurrent_state_server_constraints(sa, is_lightning=True)  # no raise

    def test_pr1_unified_radix_requires_page_size_1(self):
        sa = ServerArgs(
            model_path="dummy",
            enable_unified_radix_tree=True,
            enable_recurrent_extra_buffer=False,
            page_size=128,
        )
        with self.assertRaises(AssertionError):
            _enforce_recurrent_state_server_constraints(sa)

    def test_pr1_unified_radix_page_size_1_passes(self):
        sa = ServerArgs(
            model_path="dummy",
            enable_unified_radix_tree=True,
            page_size=1,
        )
        _enforce_recurrent_state_server_constraints(sa)  # no raise

    def test_legacy_disable_radix_passes(self):
        sa = ServerArgs(model_path="dummy", disable_radix_cache=True)
        _enforce_recurrent_state_server_constraints(sa)  # early return, no raise


if __name__ == "__main__":
    unittest.main()
