"""Wrapper accepts pytree payloads (list of arrays), not just single arrays."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from sgl_jax.srt.disaggregation.jax_transfer.wrapper import JaxTransferWrapper


def test_pull_requires_every_leaf_sharded():
    w = JaxTransferWrapper("127.0.0.1", 0)
    # not started -> sharding validation must happen first and reject a None-sharded leaf
    bad = [jax.ShapeDtypeStruct((2, 2), jnp.float32, sharding=None)]
    with pytest.raises(ValueError):
        w.pull("u", bad, remote_addr="127.0.0.1:1")


def test_nbytes_sums_over_leaves(monkeypatch):
    recorded = {}

    class _Stub:
        def await_pull(self, _uuid_int, data):
            recorded["leaves"] = jax.tree.leaves(data)

    class _FakeMetricChild:
        def inc(self, value):
            recorded["inc"] = value

    class _FakeMetric:
        def labels(self, **kwargs):
            recorded["labels"] = kwargs
            return _FakeMetricChild()

    # register_pull does a local ``from ...common.metrics import
    # PD_TRANSFER_BYTES_TOTAL`` on every call, so patch the symbol at its
    # source module — that is the name the import binds to.
    import sgl_jax.srt.disaggregation.common.metrics as metrics_mod

    monkeypatch.setattr(metrics_mod, "PD_TRANSFER_BYTES_TOTAL", _FakeMetric())

    w = JaxTransferWrapper("127.0.0.1", 0)
    w._server = _Stub()
    w._started = True
    arrs = [jnp.ones((4,), jnp.float32), jnp.ones((8,), jnp.float32)]
    w.register_pull("u1", arrs)
    assert len(recorded["leaves"]) == 2
    # 4 * float32 (4 bytes) = 16, 8 * float32 = 32, summed = 48.
    assert recorded["inc"] == 48
