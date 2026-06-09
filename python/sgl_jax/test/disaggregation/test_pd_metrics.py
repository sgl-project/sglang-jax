"""Tests for PD Prometheus metrics helpers (no-op safe, alloc/free gauge)."""

from __future__ import annotations

from sgl_jax.srt.disaggregation.common import metrics
from sgl_jax.srt.disaggregation.common.metrics import (
    PD_STATE_TRANSITION_TOTAL,
    PD_TRANSFER_FAILURES_TOTAL,
    PD_TRANSFER_INFLIGHT,
    host_pool_alloc,
    host_pool_free,
    time_phase,
)


class TestNoOpSafety:
    def test_counter_inc_does_not_raise(self):
        PD_STATE_TRANSITION_TOTAL.labels(
            from_state="BOOTSTRAPPING", to_state="WAITING_FOR_INPUT", role="prefill"
        ).inc()

    def test_failures_counter_does_not_raise(self):
        PD_TRANSFER_FAILURES_TOTAL.labels(reason="timeout", role="decode").inc()

    def test_inflight_gauge_inc_dec(self):
        g = PD_TRANSFER_INFLIGHT.labels(role="prefill")
        g.inc()
        g.dec()


class TestTimePhase:
    def test_returns_context_manager(self):
        with time_phase("transfer", "prefill"):
            pass

    def test_explicit_enter_exit(self):
        timer = time_phase("forward", "decode")
        timer.__enter__()
        timer.__exit__(None, None, None)

    def test_exit_without_enter_is_safe(self):
        timer = time_phase("forward", "decode")
        # _start is None -> no observe, no raise
        timer.__exit__(None, None, None)


class TestHostPoolGauge:
    def test_alloc_then_free_returns_to_zero(self):
        name = "test_pool_alloc_free"
        host_pool_alloc(name, 3)
        assert metrics._pool_in_use[name] == 3
        host_pool_free(name, 3)
        assert metrics._pool_in_use[name] == 0

    def test_free_never_goes_negative(self):
        name = "test_pool_negative"
        host_pool_free(name, 5)
        assert metrics._pool_in_use[name] == 0

    def test_default_count_is_one(self):
        name = "test_pool_default"
        host_pool_alloc(name)
        host_pool_alloc(name)
        assert metrics._pool_in_use[name] == 2
        host_pool_free(name)
        assert metrics._pool_in_use[name] == 1
