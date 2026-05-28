"""Contract tests for ``JaxTransferWrapper`` on CPU.

The underlying ``jax.experimental.transfer`` is shimmed out via
``sys.modules`` so these tests run on any JAX install without a TPU
backend. (On CPU-only jaxlib the real ``jax.experimental.transfer``
fails to import because ``jaxlib._jax`` lacks ``TransferConnection``.)
We only assert the wrapper-level contract (sharding check, idempotent
``start``, JAX version log, singleton enforcement); cross-pod behavior
is covered by the manual byte round-trip script.
"""

from __future__ import annotations

import logging
import sys
import types
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.disaggregation import jax_transfer_wrapper as jtw_mod
from sgl_jax.srt.disaggregation.jax_transfer_wrapper import (
    JaxTransferWrapper,
    _uuid_to_int,
    get_or_create_wrapper,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    jtw_mod._reset_singleton_for_test()
    yield
    jtw_mod._reset_singleton_for_test()


def _device_sharding() -> NamedSharding:
    devices = jax.local_devices()
    mesh = Mesh(np.asarray(devices).reshape(len(devices)), axis_names=("x",))
    return NamedSharding(mesh, P("x"))


def _shim_transfer_module(fake_server):
    """Inject a fake ``jax.experimental.transfer`` into ``sys.modules`` so
    the wrapper's ``from jax.experimental.transfer import
    start_transfer_server`` resolves to a mock without triggering the
    real (CPU-broken) module import.
    """

    fake_mod = types.ModuleType("jax.experimental.transfer")
    fake_mod.start_transfer_server = mock.MagicMock(return_value=fake_server)
    return mock.patch.dict(sys.modules, {"jax.experimental.transfer": fake_mod})


def test_pull_rejects_spec_without_sharding():
    wrapper = JaxTransferWrapper("127.0.0.1", 31000)
    spec_no_sharding = jax.ShapeDtypeStruct((4,), jnp.bfloat16)
    assert spec_no_sharding.sharding is None
    with pytest.raises(ValueError, match="sharding"):
        wrapper.pull("req-0", spec_no_sharding, remote_addr="1.2.3.4:1")


def test_pull_requires_remote_addr():
    fake_server = mock.MagicMock()
    fake_server.connect.return_value = mock.MagicMock()
    with (
        _shim_transfer_module(fake_server),
        mock.patch.object(jtw_mod.jax, "local_devices", return_value=[mock.MagicMock()]),
    ):
        wrapper = JaxTransferWrapper("127.0.0.1", 31001)
        wrapper.start()
    spec = jax.ShapeDtypeStruct((4,), jnp.bfloat16, sharding=_device_sharding())
    with pytest.raises(ValueError, match="remote_addr"):
        wrapper.pull("req-0", spec, remote_addr=None)


def test_start_is_idempotent_and_logs_jax_version(caplog):
    fake_server = mock.MagicMock()
    with (
        _shim_transfer_module(fake_server) as patched_modules,
        mock.patch.object(jtw_mod.jax, "local_devices", return_value=[mock.MagicMock()]),
    ):
        wrapper = JaxTransferWrapper("127.0.0.1", 31002, channel_number=2)
        with caplog.at_level(logging.INFO, logger=jtw_mod.logger.name):
            s1 = wrapper.start()
            s2 = wrapper.start()
            s3 = wrapper.start()
        mock_start = sys.modules["jax.experimental.transfer"].start_transfer_server
        del patched_modules

    assert s1 is fake_server
    assert s2 is fake_server
    assert s3 is fake_server
    assert mock_start.call_count == 1
    assert wrapper.is_started

    matching_records = [r for r in caplog.records if "JaxTransferWrapper started" in r.getMessage()]
    assert len(matching_records) == 1, [r.getMessage() for r in caplog.records]
    msg = matching_records[0].getMessage()
    assert "jax_version=" in msg
    assert "channel_number=2" in msg


def test_register_pull_before_start_raises():
    wrapper = JaxTransferWrapper("127.0.0.1", 31003)
    with pytest.raises(RuntimeError, match="start"):
        wrapper.register_pull("req-0", jnp.zeros((4,), jnp.bfloat16))


def test_pull_before_start_raises():
    wrapper = JaxTransferWrapper("127.0.0.1", 31004)
    spec = jax.ShapeDtypeStruct((4,), jnp.bfloat16, sharding=_device_sharding())
    with pytest.raises(RuntimeError, match="start"):
        wrapper.pull("req-0", spec, remote_addr="1.2.3.4:1")


def test_register_pull_keeps_data_alive_until_release():
    fake_server = mock.MagicMock()
    with (
        _shim_transfer_module(fake_server),
        mock.patch.object(jtw_mod.jax, "local_devices", return_value=[mock.MagicMock()]),
    ):
        wrapper = JaxTransferWrapper("127.0.0.1", 31005)
        wrapper.start()

    arr = jnp.arange(4, dtype=jnp.bfloat16)
    wrapper.register_pull("req-A", arr)
    assert "req-A" in wrapper._pending
    fake_server.await_pull.assert_called_once()
    called_uuid = fake_server.await_pull.call_args.args[0]
    assert called_uuid == _uuid_to_int("req-A")

    wrapper.release("req-A")
    assert "req-A" not in wrapper._pending


def test_register_pull_rejects_duplicate_uuid():
    fake_server = mock.MagicMock()
    with (
        _shim_transfer_module(fake_server),
        mock.patch.object(jtw_mod.jax, "local_devices", return_value=[mock.MagicMock()]),
    ):
        wrapper = JaxTransferWrapper("127.0.0.1", 31006)
        wrapper.start()

    arr1 = jnp.arange(4, dtype=jnp.bfloat16)
    arr2 = jnp.arange(8, dtype=jnp.bfloat16)
    wrapper.register_pull("dup", arr1)
    with pytest.raises(RuntimeError, match="already registered"):
        wrapper.register_pull("dup", arr2)
    # First registration is intact; second was rejected before touching state.
    assert wrapper._pending["dup"] is arr1
    assert fake_server.await_pull.call_count == 1

    # After release the same uuid is reusable.
    wrapper.release("dup")
    wrapper.register_pull("dup", arr2)
    assert wrapper._pending["dup"] is arr2
    assert fake_server.await_pull.call_count == 2


def test_singleton_rejects_rebinding():
    w1 = get_or_create_wrapper("10.0.0.1", 31010, channel_number=1)
    w2 = get_or_create_wrapper("10.0.0.1", 31010, channel_number=1)
    assert w1 is w2

    with pytest.raises(RuntimeError, match="rebind"):
        get_or_create_wrapper("10.0.0.2", 31010)
    with pytest.raises(RuntimeError, match="rebind"):
        get_or_create_wrapper("10.0.0.1", 31011)
    with pytest.raises(RuntimeError, match="channel_number"):
        get_or_create_wrapper("10.0.0.1", 31010, channel_number=4)


def test_uuid_str_to_int_is_stable_and_in_range():
    assert _uuid_to_int("req-0") == _uuid_to_int("req-0")
    assert _uuid_to_int("req-0") != _uuid_to_int("req-1")
    for s in ("a", "abc", "req-12345", "🦫"):
        v = _uuid_to_int(s)
        assert 0 <= v < (1 << 32)
