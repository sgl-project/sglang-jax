"""CPU checks for probe guards and per-layer/page evidence, not TPU transfers."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hybrid_hicache_transfer_probe import _pages, _verify_component, run_transfer_probe
from jax.sharding import AxisType, Mesh


def test_page_evidence_checks_every_layer_and_page():
    pages = [
        jnp.arange(48, dtype=jnp.float32).reshape(3, 2, 2, 2, 2),
        jnp.arange(48, dtype=jnp.float32).reshape(3, 2, 2, 2, 2) + 100,
    ]
    pool = SimpleNamespace(
        page_size=2,
        dp_size=1,
        kv_buffer=pages,
        kv_partition_axis="tensor",
        mesh=Mesh(
            np.asarray(jax.devices()[:1]).reshape(1, 1),
            ("data", "tensor"),
            axis_types=(AxisType.Explicit, AxisType.Explicit),
        ),
    )
    indices = np.array([2, 3, 4, 5], dtype=np.int32)
    expected = [np.asarray(buf)[1:3].copy() for buf in pages]
    rows = _verify_component(pool, indices, 0, expected, [7, 13])
    assert [(r["model_layer"], r["device_page"]) for r in rows] == [
        (7, 1),
        (7, 2),
        (13, 1),
        (13, 2),
    ]
    expected[1][1, 0, 0, 0, 0] = -1
    with pytest.raises(AssertionError):
        _verify_component(pool, indices, 0, expected, [7, 13])


def test_page_contract_rejects_partial_and_noncontiguous_page():
    with pytest.raises(AssertionError):
        _pages([2, 3, 4], 2)
    with pytest.raises(AssertionError):
        _pages([2, 4], 2)


def test_probe_requires_explicit_opt_in_before_accessing_scheduler():
    with pytest.raises(ValueError, match="destructive_opt_in"):
        run_transfer_probe(None)
