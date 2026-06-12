"""Tests for PD prometheus /metrics exposition.

These tests must pass whether or not ``prometheus_client`` is installed:
the multiprocess helpers are exercised directly, and the prometheus-dependent
paths are skipped when the dependency is absent.
"""

import importlib
import os

import pytest

import sgl_jax.srt.utils.common_utils as cu
from sgl_jax.srt.disaggregation.bootstrap import build_app

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
