"""Pure sizing helpers (reservation_slots_per_request / request_owned_slots) and the
ServerArgs validation for --recurrent-track-interval; no server launch."""

import unittest
from types import SimpleNamespace

from sgl_jax.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
    _enforce_recurrent_state_server_constraints,
    recurrent_admission_blocked,
    request_owned_slots,
    reservation_slots_per_request,
)
from sgl_jax.srt.server_args import ServerArgs

_MIXIN_LOGGER = "sgl_jax.srt.model_executor.model_runner_kv_cache_mixin"


def _factor_args(
    *,
    disable_radix_cache=False,
    enable_unified_radix_tree=False,
    enable_recurrent_extra_buffer=False,
    disable_overlap_schedule=False,
):
    return SimpleNamespace(
        disable_radix_cache=disable_radix_cache,
        enable_unified_radix_tree=enable_unified_radix_tree,
        enable_recurrent_extra_buffer=enable_recurrent_extra_buffer,
        disable_overlap_schedule=disable_overlap_schedule,
    )


class TestReservationSlotsPerRequest(unittest.TestCase):
    """Back-cap reservation: 1 legacy, 2 radix, 3 extra-buffer. The surplus over
    request_owned_slots is the implicit per-req snapshot headroom."""

    def test_legacy_disabled_radix_is_one(self):
        self.assertEqual(reservation_slots_per_request(_factor_args(disable_radix_cache=True)), 1)

    def test_plain_no_radix_is_one(self):
        self.assertEqual(reservation_slots_per_request(_factor_args()), 1)

    def test_unified_radix_without_extra_buffer_is_two(self):
        self.assertEqual(
            reservation_slots_per_request(_factor_args(enable_unified_radix_tree=True)), 2
        )

    def test_extra_buffer_is_three(self):
        self.assertEqual(
            reservation_slots_per_request(
                _factor_args(
                    enable_unified_radix_tree=True,
                    enable_recurrent_extra_buffer=True,
                )
            ),
            3,
        )

    def test_extra_buffer_reservation_unaffected_by_overlap(self):
        # The reservation factor stays 3 regardless of overlap; only consumption
        # (request_owned_slots) drops to 2 overlap-off, leaving 1 headroom slot/req.
        self.assertEqual(
            reservation_slots_per_request(
                _factor_args(
                    enable_unified_radix_tree=True,
                    enable_recurrent_extra_buffer=True,
                    disable_overlap_schedule=True,
                )
            ),
            3,
        )

    def test_extra_buffer_with_disabled_radix_is_one(self):
        # disable_radix_cache wins (legacy 1:1 path); extra-buffer is invalid
        # there and rejected by server_args / constraints elsewhere.
        self.assertEqual(
            reservation_slots_per_request(
                _factor_args(disable_radix_cache=True, enable_recurrent_extra_buffer=True)
            ),
            1,
        )


class TestRequestOwnedSlots(unittest.TestCase):
    """Actual per-req consumption: 1 running + ping-pong track slots."""

    def test_legacy_disabled_radix_is_one(self):
        self.assertEqual(request_owned_slots(_factor_args(disable_radix_cache=True)), 1)

    def test_plain_no_radix_is_one(self):
        self.assertEqual(request_owned_slots(_factor_args()), 1)

    def test_unified_radix_without_extra_buffer_consumes_one(self):
        # page=1 base path consumes a single running slot (reservation is 2 -> the
        # surplus is headroom). Consumption != reservation here.
        self.assertEqual(request_owned_slots(_factor_args(enable_unified_radix_tree=True)), 1)

    def test_extra_buffer_overlap_on_is_three(self):
        self.assertEqual(
            request_owned_slots(
                _factor_args(
                    enable_unified_radix_tree=True,
                    enable_recurrent_extra_buffer=True,
                )
            ),
            3,
        )

    def test_extra_buffer_overlap_off_is_two(self):
        # --disable-overlap-schedule needs only 1 ping-pong track slot.
        self.assertEqual(
            request_owned_slots(
                _factor_args(
                    enable_unified_radix_tree=True,
                    enable_recurrent_extra_buffer=True,
                    disable_overlap_schedule=True,
                )
            ),
            2,
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

    def test_unified_radix_without_extra_buffer_requires_page_size_1(self):
        sa = ServerArgs(
            model_path="dummy",
            enable_unified_radix_tree=True,
            enable_recurrent_extra_buffer=False,
            page_size=128,
        )
        with self.assertRaises(AssertionError):
            _enforce_recurrent_state_server_constraints(sa)

    def test_unified_radix_without_extra_buffer_page_size_1_passes(self):
        sa = ServerArgs(
            model_path="dummy",
            enable_unified_radix_tree=True,
            page_size=1,
        )
        _enforce_recurrent_state_server_constraints(sa)  # no raise

    def test_legacy_disable_radix_passes(self):
        sa = ServerArgs(model_path="dummy", disable_radix_cache=True)
        _enforce_recurrent_state_server_constraints(sa)  # early return, no raise

    def test_hicache_rejected_on_recurrent_radix(self):
        # Recurrent state is device-only (never backed up), so the host tier can
        # never serve a hit on a recurrent tree; reject the combo at init.
        sa = _server_args(hicache_storage="none")
        with self.assertRaises(AssertionError):
            _enforce_recurrent_state_server_constraints(sa)

    def test_hicache_rejected_on_page1_recurrent_radix(self):
        sa = ServerArgs(
            model_path="dummy",
            enable_unified_radix_tree=True,
            page_size=1,
            hicache_storage="none",
        )
        with self.assertRaises(AssertionError):
            _enforce_recurrent_state_server_constraints(sa)

    def test_hicache_disable_passes(self):
        sa = _server_args(hicache_storage="disable")
        _enforce_recurrent_state_server_constraints(sa)  # no raise


class TestResolveMaxNumReqsHeadroom(unittest.TestCase):
    """``_resolve_max_num_reqs`` back-cap: the auto/ratio path enforces snapshot
    headroom; the explicit path keeps the operator's max_running and only warns."""

    def _runner(self, *, max_recurrent_state_size, dp_size, user_supplied, **sa_over):
        sa = SimpleNamespace(
            disable_radix_cache=False,
            enable_unified_radix_tree=True,
            enable_recurrent_extra_buffer=True,
            disable_overlap_schedule=True,
            max_recurrent_state_size=max_recurrent_state_size,
            max_num_reqs=None,
        )
        sa.__dict__.update(sa_over)
        return SimpleNamespace(
            is_draft_worker=False,
            spec_algorithm=None,
            server_args=sa,
            linear_recurrent_config=object(),
            dp_size=dp_size,
            recurrent_size_user_supplied=user_supplied,
        )

    def _resolve(self, runner, requested):
        return ModelRunnerKVCacheMixin._resolve_max_num_reqs(runner, requested)

    def test_auto_path_reserves_headroom(self):
        # size 192 / dp 4 -> 48 slots/rank; overlap-off consumes 2/req.
        # headroom = ceil(0.25*48)=12; running/rank = (48-12)//2 = 18 -> 72 global.
        runner = self._runner(max_recurrent_state_size=192, dp_size=4, user_supplied=False)
        result = self._resolve(runner, 1000)
        self.assertEqual(result, 72)
        slots_per_rank = 192 // 4
        owned = request_owned_slots(runner.server_args)
        headroom = slots_per_rank - (result // 4) * owned
        self.assertGreaterEqual(headroom, 12)

    def test_auto_path_running_floor_is_at_least_dp(self):
        # Tiny pool still yields >=1 running slot/rank (no zero/negative cap).
        runner = self._runner(max_recurrent_state_size=8, dp_size=4, user_supplied=False)
        result = self._resolve(runner, 1000)
        self.assertGreaterEqual(result, 4)

    def test_explicit_path_keeps_reservation_cap_no_warn(self):
        # Explicit 192/dp4, overlap-off: cap = 192//3 = 64; owned 2 < reservation 3
        # leaves real headroom 192 - 64*2 = 64 -> no warning.
        runner = self._runner(max_recurrent_state_size=192, dp_size=4, user_supplied=True)
        import logging

        logger = logging.getLogger(_MIXIN_LOGGER)
        with self.assertNoLogs(logger, level="WARNING"):
            result = self._resolve(runner, 1000)
        self.assertEqual(result, 64)

    def test_explicit_path_overlap_on_zero_headroom_warns(self):
        # Overlap-ON: owned == reservation == 3, so 192//3 = 64 running consumes the
        # whole pool -> zero headroom -> warn (but max_running is NOT reduced).
        runner = self._runner(
            max_recurrent_state_size=192,
            dp_size=4,
            user_supplied=True,
            disable_overlap_schedule=False,
        )
        with self.assertLogs(_MIXIN_LOGGER, level="WARNING"):
            result = self._resolve(runner, 1000)
        self.assertEqual(result, 64)

    def test_undersized_pool_below_owned_fails_fast_auto(self):
        # Overlap-ON extra-buffer: owned = 3. size 8 / dp 4 -> 2 slots/rank < 3 ->
        # the pool cannot fit even one running request. Fail fast instead of
        # advertising >=1 running/rank the pool can never allocate (which would
        # leave the rank marked full forever).
        runner = self._runner(
            max_recurrent_state_size=8,
            dp_size=4,
            user_supplied=False,
            disable_overlap_schedule=False,
        )
        with self.assertRaises(ValueError):
            self._resolve(runner, 1000)

    def test_undersized_pool_below_owned_fails_fast_explicit(self):
        # Same fail-fast on the explicit --max-recurrent-state-size path.
        runner = self._runner(
            max_recurrent_state_size=8,
            dp_size=4,
            user_supplied=True,
            disable_overlap_schedule=False,
        )
        with self.assertRaises(ValueError):
            self._resolve(runner, 1000)

    def test_explicit_reservation_cap_floored_to_zero_fails_fast(self):
        # Overlap-off owned=2 passes the per-slot guard (slots/rank 2 >= 2), but
        # the reservation divide + dp floor lands at 0 (8//3=2 -> 2-2%4=0):
        # max_running=0 must fail fast, not start a server that admits nothing.
        runner = self._runner(max_recurrent_state_size=8, dp_size=4, user_supplied=True)
        with self.assertRaises(ValueError):
            self._resolve(runner, 1000)

    def test_slots_equal_owned_does_not_fail(self):
        # Boundary: slots/rank == owned fits exactly one req/rank (zero snapshot
        # headroom), so it must NOT fail fast. Overlap-off owned=2, size 8/dp4 -> 2.
        runner = self._runner(max_recurrent_state_size=8, dp_size=4, user_supplied=False)
        self.assertGreaterEqual(self._resolve(runner, 1000), 4)


class TestRecurrentAdmissionBlocked(unittest.TestCase):
    """The scheduler admission backpressure predicate. ``keeps_locked`` discounts
    a snapshot a cross-request prefix hit will keep protected (locked, hence
    non-evictable) while the req still needs request_owned_slots fresh slots."""

    def test_cold_admission_at_exact_capacity_admits(self):
        # No prefix hit (keeps_locked=0): free+evictable == demand -> admit.
        self.assertFalse(recurrent_admission_blocked(free=0, evictable=3, demand=3))

    def test_prefix_hit_at_exact_capacity_defers(self):
        # Hit locks 1 of the 3 evictable snapshots -> only 2 reclaimable for 3
        # fresh slots -> must defer (else alloc_req_slots evicts too few and raises).
        self.assertTrue(recurrent_admission_blocked(free=0, evictable=3, demand=3, keeps_locked=1))

    def test_prefix_hit_with_one_free_slot_admits(self):
        # One free slot covers the locked snapshot's shortfall.
        self.assertFalse(recurrent_admission_blocked(free=1, evictable=3, demand=3, keeps_locked=1))

    def test_obvious_oversubscription_blocks(self):
        self.assertTrue(recurrent_admission_blocked(free=0, evictable=1, demand=3))


if __name__ == "__main__":
    unittest.main()
