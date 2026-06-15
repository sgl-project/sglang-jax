"""Tests for PD metrics: counters/labels and Prometheus exposition."""

from __future__ import annotations

import importlib
import os
import pytest
import sgl_jax.srt.utils.common_utils as cu

from sgl_jax.srt.disaggregation.bootstrap import build_app
from sgl_jax.srt.disaggregation.common import metrics
from sgl_jax.srt.disaggregation.common.metrics import (
    PD_STATE_TRANSITION_TOTAL,
    PD_TRANSFER_FAILURES_TOTAL,
    PD_TRANSFER_INFLIGHT,
    host_pool_alloc,
    host_pool_free,
    time_phase,
)


# ---- from test_pd_metrics.py ----

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

# ---- from test_pd_metrics_exposition.py ----

_HAS_PROM = importlib.util.find_spec("prometheus_client") is not None

@pytest.fixture
def clean_multiproc_env():
    saved = os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
    yield
    os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
    if saved is not None:
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = saved

def test_set_prometheus_multiproc_dir_sets_env(clean_multiproc_env):
    """Creates a tempdir and points PROMETHEUS_MULTIPROC_DIR at it."""
    cu.set_prometheus_multiproc_dir()
    path = os.environ["PROMETHEUS_MULTIPROC_DIR"]
    assert os.path.isdir(path)

def test_set_prometheus_multiproc_dir_reuses_existing(clean_multiproc_env):
    """An existing PROMETHEUS_MULTIPROC_DIR is preserved (subprocesses must
    inherit the parent's dir, not clobber it)."""
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = "/tmp/preset-prom-dir"
    cu.set_prometheus_multiproc_dir()
    assert os.environ["PROMETHEUS_MULTIPROC_DIR"] == "/tmp/preset-prom-dir"

@pytest.mark.skipif(not _HAS_PROM, reason="prometheus_client not installed")
def test_add_prometheus_middleware_mounts_metrics(clean_multiproc_env):
    """add_prometheus_middleware appends a /metrics Mount route."""
    from fastapi import FastAPI

    # MultiProcessCollector requires the dir to exist.
    cu.set_prometheus_multiproc_dir()
    app = FastAPI()
    before = len(app.routes)
    cu.add_prometheus_middleware(app)
    assert len(app.routes) == before + 1
    paths = [getattr(r, "path", None) for r in app.routes]
    assert "/metrics" in paths

def test_bootstrap_metrics_route_matches_availability():
    """Bootstrap exposes a single-process /metrics only when prometheus_client
    is importable; otherwise the route is silently absent."""
    app, _registry = build_app()
    paths = [getattr(r, "path", None) for r in app.routes]
    if _HAS_PROM:
        assert "/metrics" in paths
    else:
        assert "/metrics" not in paths

@pytest.mark.skipif(not _HAS_PROM, reason="prometheus_client not installed")
def test_bootstrap_metrics_returns_registry_size():
    """When prometheus is available the bootstrap /metrics returns 200 and
    carries the pd_bootstrap_registry_size series."""
    from fastapi.testclient import TestClient

    app, _registry = build_app()
    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "pd_bootstrap_registry_size" in resp.text
