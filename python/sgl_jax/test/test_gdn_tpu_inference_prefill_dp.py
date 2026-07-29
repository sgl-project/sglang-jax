"""Distributed runtime contract for TPU-Inference v3 GDN correctness runs."""

from __future__ import annotations

import jax
import numpy as np
import pytest
from jax.experimental import multihost_utils

pytestmark = pytest.mark.skipif(
    not any(device.platform == "tpu" for device in jax.local_devices()),
    reason="the distributed GDN contract requires real TPU hardware",
)


def test_all_hosts_form_unique_process_and_device_partitions():
    """Every host must participate before numerical state evidence is accepted."""
    identity = np.asarray(
        [jax.process_index(), jax.local_device_count()],
        dtype=np.int32,
    )
    gathered = np.asarray(multihost_utils.process_allgather(identity))

    assert sorted(gathered[:, 0].tolist()) == list(range(jax.process_count()))
    assert np.all(gathered[:, 1] == jax.local_device_count())
    assert jax.device_count() == jax.process_count() * jax.local_device_count()
