"""PD metrics module.

The real prometheus_client may or may not be installed depending on
the deployment image. ``metrics.py`` must not crash either way and
must accept every label combo described in the RFC schema. These
tests are CPU-only — they neither bind a transfer server nor open
sockets.
"""

from __future__ import annotations

from sgl_jax.srt.disaggregation import metrics as M


def test_state_transition_counter_accepts_labels():
    # All three label dimensions must be non-empty strings.
    M.PD_STATE_TRANSITION_TOTAL.labels(
        from_state="bootstrapping",
        to_state="waiting_for_input",
        role="prefill",
    ).inc()


def test_transfer_bytes_counter_directions():
    for direction in ("d2h", "h2d", "net"):
        for role in ("prefill", "decode"):
            M.PD_TRANSFER_BYTES_TOTAL.labels(direction=direction, role=role).inc(1024)


def test_transfer_duration_phases():
    for phase in ("bootstrap", "pull", "ack"):
        for role in ("prefill", "decode"):
            with M.time_phase(phase, role):
                pass


def test_transfer_inflight_inc_dec():
    M.PD_TRANSFER_INFLIGHT.labels(role="prefill").inc()
    M.PD_TRANSFER_INFLIGHT.labels(role="prefill").dec()


def test_transfer_failures_reasons():
    for reason in (
        "timeout",
        "peer_crash",
        "network",
        "auth",
        "bootstrap_lookup",
        "receiver_init",
        "shutdown",
        "other",
    ):
        M.PD_TRANSFER_FAILURES_TOTAL.labels(reason=reason, role="decode").inc()


def test_host_pool_helpers_balance():
    M.host_pool_alloc("test-pool", 3)
    M.host_pool_alloc("test-pool", 1)
    M.host_pool_free("test-pool", 2)
    M.host_pool_free("test-pool", 2)
    # Should not go negative; double-free guard is in the helper.
    M.host_pool_free("test-pool", 5)


def test_bootstrap_registry_gauge_set():
    M.PD_BOOTSTRAP_REGISTRY_SIZE.set(0)
    M.PD_BOOTSTRAP_REGISTRY_SIZE.set(7)
    M.PD_BOOTSTRAP_REGISTRY_SIZE.set(0)


def test_is_prometheus_available_is_bool():
    assert isinstance(M.is_prometheus_available(), bool)


def test_noop_when_prom_missing(monkeypatch):
    """Ensure callers can no-op without crashing even when the
    real prometheus_client isn't installed. We simulate the absence
    by monkey-patching ``inc`` / ``observe`` / ``set`` to raise; the
    helpers must swallow."""

    class _BoomMetric:
        def labels(self, **_):
            return self

        def inc(self, _amount=1.0):
            raise RuntimeError("boom")

        def set(self, _value):
            raise RuntimeError("boom")

        def observe(self, _value):
            raise RuntimeError("boom")

    monkeypatch.setattr(M, "PD_HOST_POOL_USED_BUFFERS", _BoomMetric())
    M.host_pool_alloc("boom-pool", 1)
    M.host_pool_free("boom-pool", 1)
