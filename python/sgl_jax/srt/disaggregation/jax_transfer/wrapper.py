"""Process-level wrapper over ``jax.experimental.transfer``.

The wrapper enforces:
  * one transfer server per process (host:port unique), guarded by a
    module-level lock
  * idempotent ``start()``: repeat calls return the same server
  * a ``ValueError`` raised eagerly when ``pull(spec)`` is given a
    ``ShapeDtypeStruct`` without sharding (the underlying API would
    otherwise blow up deep inside with ``AttributeError`` on
    ``sharding.device_set``)
  * a small reference-keeping book (``_pending``) so that arrays passed
    to ``register_pull`` survive until ``release()``; the underlying
    ``await_pull`` only registers and returns immediately.

The wrapper does NOT lock to a specific JAX version. The contract tests
in ``test_jax_transfer_wrapper`` will fail loudly if a future JAX
release breaks any of the above assumptions.

Multi-peer note: the RFC sketch lists ``pull(uuid, spec)`` only, but a
process-level wrapper that fans out to multiple peers needs to know
*which* peer to pull from. We extend the signature with an explicit
``remote_addr`` argument and cache one ``link`` per remote.
"""

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
    """Stable mapping from a public ``str`` uuid to the 32-bit int that the
    underlying JAX transfer API expects. ``zlib.crc32`` is deterministic
    across processes and Python versions, which is what we need for
    cross-pod pull/register pairing.

    32 bits gives a birthday-bound collision risk at roughly 65k
    concurrent uuids. That is acceptable for a bounded number of
    in-flight transfers, but a wider hash would reduce the risk further.
    The :py:meth:`JaxTransferWrapper.register_pull` duplicate check
    catches repeated string uuids, but it cannot detect a ``crc32``
    collision between two distinct strings.
    """

    return zlib.crc32(uuid.encode("utf-8")) & 0xFFFFFFFF


class JaxTransferWrapper:
    """Process-level wrapper. Use :func:`get_or_create_wrapper` to obtain
    the singleton; constructing this class directly only stores config —
    nothing happens until ``start()``.
    """

    def __init__(
        self,
        host_ip: str,
        port: int,
        channel_number: int = 1,
    ) -> None:
        self._host_ip = host_ip
        self._port = port
        self._channel_number = channel_number
        self._init_lock = threading.Lock()
        self._server: Any | None = None
        self._started = False
        # ``_pending`` is mutated from ``register_pull`` and from the
        # side-channel ack path, so access is serialized with a lock.
        self._pending_lock = threading.Lock()
        self._pending: dict[str, jax.Array] = {}
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
        return self._started

    @property
    def server(self) -> Any:
        return self._server

    def start(self) -> Any:
        """Idempotent. Returns the underlying transfer server.

        Logs the installed JAX version on first start so the contract
        tests can detect API drift across upgrades.
        """

        if self._started:
            return self._server
        with self._init_lock:
            if self._started:
                return self._server
            from jax.experimental.transfer import start_transfer_server

            client = jax.local_devices()[0].client
            server_addr = f"{self._host_ip}:{self._port}"
            transport_addr = f"{self._host_ip}:0"
            self._server = start_transfer_server(
                client,
                server_addr,
                [transport_addr],
                max_num_parallel_copies=self._channel_number,
                transfer_size=64 * 1024 * 1024,
                use_raw_buffers=False,
            )
            self._started = True
            logger.info(
                "JaxTransferWrapper started at %s " "(channel_number=%d, jax_version=%s)",
                server_addr,
                self._channel_number,
                jax.__version__,
            )
        return self._server

    def register_pull(self, uuid: str, data: jax.Array) -> None:
        """Register ``data`` for a future remote pull keyed by ``uuid``.

        Non-blocking: returns as soon as the underlying API has registered
        the buffer. Caller must keep ``data`` alive (the wrapper holds a
        reference internally) until ``release(uuid)`` is called or, in a
        properly wired stack, until the out-of-band ack signals the remote
        has finished pulling.

        Raises ``RuntimeError`` if ``uuid`` is already registered. The
        wrapper does not silently overwrite — the caller must call
        ``release(uuid)`` first if a re-registration is intended.

        Thread-safe with respect to :meth:`release` from the ZMQ
        listener thread.
        """

        if not self._started:
            raise RuntimeError(
                "JaxTransferWrapper.start() must be called before " "register_pull()"
            )
        with self._pending_lock:
            if uuid in self._pending:
                raise RuntimeError(
                    f"uuid {uuid!r} is already registered; call "
                    f"release({uuid!r}) first if you intend to "
                    f"re-register"
                )
            # Call await_pull while holding the lock so we never race
            # a release for the same uuid between server registration
            # and bookkeeping update.
            self._server.await_pull(_uuid_to_int(uuid), data)
            self._pending[uuid] = data
        try:
            from sgl_jax.srt.disaggregation.common.metrics import PD_TRANSFER_BYTES_TOTAL

            PD_TRANSFER_BYTES_TOTAL.labels(direction="net", role="prefill").inc(int(data.nbytes))
        except Exception:  # noqa: BLE001
            pass

    def pull(
        self,
        uuid: str,
        spec: jax.ShapeDtypeStruct,
        remote_addr: str | None = None,
    ) -> jax.Array:
        """Pull a previously registered buffer from ``remote_addr``.

        ``spec.sharding`` MUST be set; the underlying JAX transfer API
        requires it and would otherwise fail deep inside with a
        ``NoneType has no attribute 'device_set'`` error.
        """

        if spec.sharding is None:
            raise ValueError(
                "JaxTransferWrapper.pull requires spec.sharding; "
                "jax.experimental.transfer needs an explicit sharding "
                "for every ShapeDtypeStruct."
            )
        if not self._started:
            raise RuntimeError("JaxTransferWrapper.start() must be called before pull()")
        if remote_addr is None:
            raise ValueError(
                "JaxTransferWrapper.pull requires remote_addr; the "
                "process-level wrapper supports multiple peers."
            )
        link = self._connect(remote_addr)
        return link.pull(_uuid_to_int(uuid), [spec])[0]

    def release(self, uuid: str) -> None:
        """Drop the wrapper's reference to a previously registered buffer.

        After release the underlying array can be garbage collected as
        long as the caller holds no other references. Called from the
        ZMQ listener thread after the decoder acknowledges the pull.

        Thread-safe with respect to :meth:`register_pull` from the main
        thread.
        """

        with self._pending_lock:
            self._pending.pop(uuid, None)

    def _connect(self, remote_addr: str) -> Any:
        if remote_addr in self._links:
            return self._links[remote_addr]
        link = self._server.connect(remote_addr)
        self._links[remote_addr] = link
        return link


def get_or_create_wrapper(
    host_ip: str,
    port: int,
    channel_number: int = 1,
) -> JaxTransferWrapper:
    """Return the process-level wrapper, creating it on first call.

    Subsequent calls with mismatched ``host_ip``/``port``/
    ``channel_number`` raise ``RuntimeError`` — the process is bound to a
    single transfer server.
    """

    global _GLOBAL_WRAPPER
    with _GLOBAL_LOCK:
        if _GLOBAL_WRAPPER is None:
            _GLOBAL_WRAPPER = JaxTransferWrapper(host_ip, port, channel_number)
            return _GLOBAL_WRAPPER
        existing = _GLOBAL_WRAPPER
        if (existing.host_ip, existing.port) != (host_ip, port):
            raise RuntimeError(
                f"JaxTransferWrapper already bound to "
                f"{existing.host_ip}:{existing.port}, cannot rebind to "
                f"{host_ip}:{port}"
            )
        if existing.channel_number != channel_number:
            raise RuntimeError(
                f"JaxTransferWrapper already created with "
                f"channel_number={existing.channel_number}, cannot "
                f"change to {channel_number}"
            )
        return existing


def _reset_singleton_for_test() -> None:
    """Test-only: clear the module-level singleton between cases."""

    global _GLOBAL_WRAPPER
    with _GLOBAL_LOCK:
        _GLOBAL_WRAPPER = None
