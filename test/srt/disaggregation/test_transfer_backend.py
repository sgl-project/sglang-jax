"""Unit tests for ``TransferBackend`` ABC and ``JaxTransferBackend``."""

from __future__ import annotations

from unittest import mock

import pytest

from sgl_jax.srt.disaggregation.transport.backend import (
    JaxTransferBackend,
    TransferBackend,
)


def test_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        TransferBackend()  # type: ignore[abstract]


def test_jax_backend_delegates_register_pull():
    wrapper = mock.MagicMock()
    backend = JaxTransferBackend(wrapper)
    data = mock.MagicMock()
    backend.register_pull("uuid-1", data)
    wrapper.register_pull.assert_called_once_with("uuid-1", data)


def test_jax_backend_delegates_pull():
    wrapper = mock.MagicMock()
    backend = JaxTransferBackend(wrapper)
    spec = mock.MagicMock()
    backend.pull("uuid-2", spec, remote_addr="10.0.0.1:30000")
    wrapper.pull.assert_called_once_with(
        "uuid-2", spec, remote_addr="10.0.0.1:30000"
    )


def test_jax_backend_delegates_release():
    wrapper = mock.MagicMock()
    backend = JaxTransferBackend(wrapper)
    backend.release("uuid-3")
    wrapper.release.assert_called_once_with("uuid-3")


def test_jax_backend_exposes_wrapper():
    wrapper = mock.MagicMock()
    backend = JaxTransferBackend(wrapper)
    assert backend.wrapper is wrapper


def test_jax_backend_is_transfer_backend():
    wrapper = mock.MagicMock()
    backend = JaxTransferBackend(wrapper)
    assert isinstance(backend, TransferBackend)
