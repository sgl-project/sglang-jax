"""Transport backend abstraction.

A ``TransferBackend`` publishes, pulls, and releases tensor arrays
without knowing what they represent (KV cache, embeddings, recurrent
state, etc.).  ``JaxTransferBackend`` delegates to
:class:`JaxTransferWrapper` for the ``jax.experimental.transfer`` API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax

from sgl_jax.srt.disaggregation.jax_transfer_wrapper import JaxTransferWrapper


class TransferBackend(ABC):
    """Leaf-level tensor publish / pull / release."""

    @abstractmethod
    def register_pull(self, uuid: str, data: jax.Array) -> None:
        """Make ``data`` available for remote pull under ``uuid``."""

    @abstractmethod
    def pull(
        self,
        uuid: str,
        spec: jax.ShapeDtypeStruct,
        *,
        remote_addr: str,
    ) -> jax.Array:
        """Pull a tensor from ``remote_addr`` keyed by ``uuid``."""

    @abstractmethod
    def release(self, uuid: str) -> None:
        """Drop the internal reference for ``uuid``."""


class JaxTransferBackend(TransferBackend):
    """Thin wrapper around :class:`JaxTransferWrapper`."""

    def __init__(self, wrapper: JaxTransferWrapper) -> None:
        self._wrapper = wrapper

    @property
    def wrapper(self) -> JaxTransferWrapper:
        return self._wrapper

    def register_pull(self, uuid: str, data: jax.Array) -> None:
        self._wrapper.register_pull(uuid, data)

    def pull(
        self,
        uuid: str,
        spec: jax.ShapeDtypeStruct,
        *,
        remote_addr: str,
    ) -> jax.Array:
        return self._wrapper.pull(uuid, spec, remote_addr=remote_addr)

    def release(self, uuid: str) -> None:
        self._wrapper.release(uuid)
