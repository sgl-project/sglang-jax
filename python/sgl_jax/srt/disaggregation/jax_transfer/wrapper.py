"""Process-level wrapper over ``jax.experimental.transfer``."""

from __future__ import annotations

import logging
import threading
import zlib
from typing import Any

import jax

logger = logging.getLogger(__name__)

_GLOBAL_LOCK = threading.Lock()
_GLOBAL_WRAPPER: JaxTransferWrapper | None = None


def _uuid_to_int(uuid: str) -> int:
    return zlib.crc32(uuid.encode("utf-8")) & 0xFFFFFFFF


class JaxTransferWrapper:
    def __init__(self, host_ip: str, port: int, channel_number: int = 1) -> None:
        self._host_ip = host_ip
        self._port = port
        self._channel_number = channel_number
        self._init_lock = threading.Lock()
        self._server: Any | None = None
        self._pending_lock = threading.Lock()
        self._pending: dict[str, Any] = {}
        self._links_lock = threading.Lock()
        self._links: dict[str, Any] = {}

    @property
    def host_ip(self) -> str:
        return self._host_ip

    @property
    def port(self) -> int:
        return self._port

    @property
    def channel_number(self) -> int:
        return self._channel_number

    @property
    def is_started(self) -> bool:
        return self._server is not None

    @property
    def server(self) -> Any:
        return self._server

    def start(self) -> Any:
        if self._server is not None:
            return self._server
        with self._init_lock:
            if self._server is not None:
                return self._server
            from jax.experimental.transfer import start_transfer_server

            server_addr = f"{self._host_ip}:{self._port}"
            self._server = start_transfer_server(
                jax.local_devices()[0].client,
                server_addr,
                [f"{self._host_ip}:0"],
                max_num_parallel_copies=self._channel_number,
                transfer_size=64 * 1024 * 1024,
                use_raw_buffers=False,
            )
            logger.info(
                "JaxTransferWrapper started at %s (channel_number=%d, jax_version=%s)",
                server_addr,
                self._channel_number,
                jax.__version__,
            )
        return self._server

    def register_pull(self, uuid: str, data: Any) -> None:
        if self._server is None:
            raise RuntimeError("JaxTransferWrapper.start() must be called before register_pull()")
        sharding = getattr(data, "sharding", None)
        if sharding is not None and not data.is_fully_addressable:
            raise ValueError(
                f"register_pull received a non-local array spanning "
                f"{len(sharding.device_set)} devices"
            )
        with self._pending_lock:
            if uuid in self._pending:
                raise RuntimeError(f"uuid {uuid!r} is already registered")
            # Registration and the retained reference must become visible
            # atomically with respect to release from the ack thread.
            self._server.await_pull(_uuid_to_int(uuid), data)
            self._pending[uuid] = data
        try:
            from sgl_jax.srt.disaggregation.common.metrics import PD_TRANSFER_BYTES_TOTAL

            PD_TRANSFER_BYTES_TOTAL.labels(direction="net", role="prefill").inc(
                int(sum(int(leaf.nbytes) for leaf in jax.tree.leaves(data)))
            )
        except Exception:  # noqa: BLE001
            pass

    def pull(self, uuid: str, spec: Any, remote_addr: str | None = None) -> Any:
        for leaf in jax.tree.leaves(spec):
            if getattr(leaf, "sharding", None) is None:
                raise ValueError("JAX transfer requires sharding on every result spec")
        if self._server is None:
            raise RuntimeError("JaxTransferWrapper.start() must be called before pull()")
        if remote_addr is None:
            raise ValueError("remote_addr is required")
        return self._connect(remote_addr).pull(_uuid_to_int(uuid), spec)

    def release(self, uuid: str) -> None:
        with self._pending_lock:
            self._pending.pop(uuid, None)

    def _connect(self, remote_addr: str) -> Any:
        with self._links_lock:
            link = self._links.get(remote_addr)
            if link is None:
                link = self._server.connect(remote_addr)
                self._links[remote_addr] = link
            return link


def get_or_create_wrapper(
    host_ip: str,
    port: int,
    channel_number: int = 1,
) -> JaxTransferWrapper:
    global _GLOBAL_WRAPPER
    with _GLOBAL_LOCK:
        if _GLOBAL_WRAPPER is None:
            _GLOBAL_WRAPPER = JaxTransferWrapper(host_ip, port, channel_number)
            return _GLOBAL_WRAPPER
        existing = _GLOBAL_WRAPPER
        if (existing.host_ip, existing.port) != (host_ip, port):
            raise RuntimeError(
                f"JaxTransferWrapper cannot rebind from "
                f"{existing.host_ip}:{existing.port} to {host_ip}:{port}"
            )
        if existing.channel_number != channel_number:
            raise RuntimeError(
                f"JaxTransferWrapper already uses channel_number={existing.channel_number}"
            )
        return existing


def _reset_singleton_for_test() -> None:
    global _GLOBAL_WRAPPER
    with _GLOBAL_LOCK:
        _GLOBAL_WRAPPER = None
