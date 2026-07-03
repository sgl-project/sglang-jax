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

``pull()`` takes an explicit ``remote_addr`` and caches one ``link`` per
remote peer.
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
    cross-pod pull/register pairing. 32-bit collision risk is acceptable
    for the bounded number of in-flight transfers.
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
        self._pending: dict[str, Any] = {}
        # ``_links`` is created and used only on the background pull worker
        # thread (lazy connect inside ``pull``). The lock is kept as a cheap
        # guard in case a future caller pulls from another thread.
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

    def register_pull(self, uuid: str, data: Any) -> None:
        """Register ``data`` for a future remote pull keyed by ``uuid``.

        ``data`` may be a single ``jax.Array`` or any pytree of arrays
        (e.g. a per-layer ``list`` of KV buffers). The underlying
        ``await_pull`` flattens/unflattens internally, so the pytree is
        passed through unchanged.

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
        sharding = getattr(data, "sharding", None)
        if sharding is not None and not data.is_fully_addressable:
            raise ValueError(
                f"register_pull: array spans {len(sharding.device_set)} "
                f"devices but only {jax.local_device_count()} are local. "
                f"await_pull only registers process-local shards; pass the "
                f"per-host shard (see prefill._global_to_local_shard)."
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
            from sgl_jax.srt.disaggregation.common.metrics import (
                PD_TRANSFER_BYTES_TOTAL,
            )

            PD_TRANSFER_BYTES_TOTAL.labels(direction="net", role="prefill").inc(
                int(sum(int(leaf.nbytes) for leaf in jax.tree.leaves(data)))
            )
        except Exception:  # noqa: BLE001
            pass

    def pull(
        self,
        uuid: str,
        spec: Any,
        remote_addr: str | None = None,
    ) -> Any:
        """Pull a previously registered buffer from ``remote_addr``.

        ``spec`` may be a single ``jax.ShapeDtypeStruct`` or any pytree of
        them (e.g. a per-layer ``list``). Every leaf's ``sharding`` MUST be
        set; the underlying JAX transfer API requires it and would
        otherwise fail deep inside with a ``NoneType has no attribute
        'device_set'`` error.
        """

        for leaf in jax.tree.leaves(spec):
            if getattr(leaf, "sharding", None) is None:
                raise ValueError(
                    "JaxTransferWrapper.pull requires sharding on every leaf; "
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
        return link.pull(_uuid_to_int(uuid), spec)

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
        with self._links_lock:
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


# =====================================================================
# raiden data plane (Phase 0): tpu-raiden TransferEngine wrapper
# =====================================================================
#
# The raiden path replaces the four ``jax.experimental.transfer`` calls above
# with tpu-raiden's ``KVCacheManager`` (repurposed as a TransferEngine). Unlike
# path-A — which registers per-request host/HBM *buffers* keyed by a crc32 uuid
# and pulls them whole — raiden references the *device KV pool tensors directly*
# and moves them at *block* granularity:
#
#   * producer(P): ``register_read(req_id, uuid, block_ids)`` marks the device
#     blocks (= sgl-jax pages) of a request as readable.
#   * consumer(D): ``start_read(req_id, uuid, remote_endpoint, remote_block_ids,
#     local_block_ids)`` asynchronously pulls those blocks straight into D's
#     device KV pool blocks (no Pallas write-back needed).
#   * both sides poll ``poll_stats() -> (done_sending, done_recving,
#     failed_recving)`` — lists of *req_id strings* (NOT uuids). Completion is
#     ``req_id in done_*``; this replaces the ZMQ pull-done ack side channel.
#
# The control plane is raiden's own TCP socket (``local_control_port``); the
# endpoint descriptors are read back from ``get_local_endpoints()`` and
# advertised over bootstrap so D can reach P.

_GLOBAL_RAIDEN_LOCK = threading.Lock()
_GLOBAL_RAIDEN_WRAPPER: RaidenTransferWrapper | None = None


class RaidenTransferWrapper:
    """Process-level wrapper over tpu-raiden's ``KVCacheManager``.

    One engine per process, constructed lazily in :meth:`start` from the
    device KV pool's per-layer tensors. The engine *references* those tensors
    (does not own them), so the pool must already be created and sharded before
    ``start()`` is called.

    Thread-safety: ``register_read`` / ``start_read`` / ``poll_stats`` are thin
    pass-throughs to the compiled engine; the module keeps no per-request book
    (raiden tracks state internally, keyed by ``req_id``). ``start()`` is
    idempotent and guarded.
    """

    def __init__(
        self,
        host_ip: str,
        control_port: int = 0,
        *,
        parallelism: int = 1,
    ) -> None:
        self._host_ip = host_ip
        self._control_port = control_port
        self._parallelism = max(1, int(parallelism))
        self._init_lock = threading.Lock()
        self._engine: Any | None = None
        self._engine_full: Any | None = None
        self._engine_swa: Any | None = None
        self._started = False
        self._is_hybrid_swa = False
        self._endpoints: list[Any] | None = None
        self._endpoints_swa: list[Any] | None = None

    @property
    def host_ip(self) -> str:
        return self._host_ip

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def is_hybrid_swa(self) -> bool:
        return self._is_hybrid_swa

    @property
    def engine(self) -> Any:
        return self._engine

    @property
    def engine_full(self) -> Any:
        return self._engine_full or self._engine

    @property
    def engine_swa(self) -> Any | None:
        return self._engine_swa

    @property
    def endpoints(self) -> list[Any] | None:
        """raiden endpoint descriptors (from ``get_local_endpoints()``).

        These are what D passes as ``remote_endpoint`` to ``start_read`` and
        what P advertises via bootstrap. Available only after :meth:`start`.
        """

        return self._endpoints

    @property
    def endpoints_swa(self) -> list[Any] | None:
        """SWA engine endpoint descriptors (hybrid SWA models only)."""
        return self._endpoints_swa

    def start(
        self,
        kv_caches: list[Any],
        *,
        max_blocks: int,
        num_slots: int,
        timeout_s: float = 120.0,
        kv_caches_swa: list[Any] | None = None,
    ) -> Any:
        """Idempotent. Construct one (or two) KVCacheManager over ``kv_caches``.

        ``kv_caches`` is the device full-attention pool's per-layer tensor list.
        ``kv_caches_swa`` (optional) is the SWA pool's per-layer tensor list for
        hybrid (SWA+full) models. When both are given, two independent raiden
        engines are created — one per pool — because raiden broadcasts the same
        block_ids to all layers in an engine, but full and SWA layers use
        different page numbering.

        ``max_blocks`` / ``num_slots`` / ``timeout_s`` apply to both engines.
        """

        if self._started:
            return self._engine_full if self._is_hybrid_swa else self._engine
        with self._init_lock:
            if self._started:
                return self._engine_full if self._is_hybrid_swa else self._engine
            from tpu_raiden.api.jax.kv_cache_manager import KVCacheManager

            if not kv_caches:
                raise ValueError("RaidenTransferWrapper.start requires kv_caches")
            self._is_hybrid_swa = bool(kv_caches_swa)
            self._engine = KVCacheManager(
                kv_caches=list(kv_caches),
                local_control_port=self._control_port,
                max_blocks=int(max_blocks),
                num_slots=int(num_slots),
                timeout_s=float(timeout_s),
                parallelism=self._parallelism,
                unsafe_skip_buffer_lock=True,
            )
            self._endpoints = self._engine.get_local_endpoints()
            self._engine_full = self._engine  # alias for clarity in dual-engine mode

            if self._is_hybrid_swa:
                # SWA engine gets its own control port (0 = kernel picks).
                self._engine_swa = KVCacheManager(
                    kv_caches=list(kv_caches_swa),
                    local_control_port=0,
                    max_blocks=int(max_blocks),
                    num_slots=int(num_slots),
                    timeout_s=float(timeout_s),
                    parallelism=self._parallelism,
                    unsafe_skip_buffer_lock=True,
                )
                self._endpoints_swa = self._engine_swa.get_local_endpoints()
            else:
                self._engine_swa = None
                self._endpoints_swa = None

            self._started = True
            logger.info(
                "RaidenTransferWrapper started host=%s control_port=%s is_hybrid_swa=%s "
                "endpoints=%s%s (jax_version=%s)",
                self._host_ip,
                self._control_port,
                self._is_hybrid_swa,
                self._endpoints,
                f" endpoints_swa={self._endpoints_swa}" if self._is_hybrid_swa else "",
                jax.__version__,
            )
        return self._engine_full if self._is_hybrid_swa else self._engine

    def register_read(
        self,
        req_id: str,
        uuid: int,
        block_ids: list[int],
        *,
        swa_block_ids: list[int] | None = None,
    ) -> bool:
        """Producer: mark ``block_ids`` of ``req_id`` readable.

        For hybrid SWA models, ``swa_block_ids`` registers the SWA-layer blocks
        on the SWA engine under the same uuid (the engines are independent so
        there is no uuid collision). Returns the full-engine result (SWA
        registration is best-effort logged).
        """

        if not self._started:
            raise RuntimeError("RaidenTransferWrapper.start() must be called before register_read()")
        result = bool(self._engine_full.register_read(req_id, uuid, list(block_ids)))
        if self._is_hybrid_swa and swa_block_ids:
            self._engine_swa.register_read(req_id, uuid, list(swa_block_ids))
        return result

    def start_read(
        self,
        req_id: str,
        uuid: int,
        remote_endpoint: Any,
        remote_block_ids: list[int],
        local_block_ids: list[int],
        *,
        parallelism: int = 1,
        swa_remote_endpoint: Any = None,
        swa_remote_block_ids: list[int] | None = None,
        swa_local_block_ids: list[int] | None = None,
    ) -> None:
        """Consumer: asynchronously pull blocks from producer into local pool.

        For hybrid SWA models, ``swa_*`` arguments drive a parallel start_read
        on the SWA engine with the SWA-pool page indices.
        """

        if not self._started:
            raise RuntimeError("RaidenTransferWrapper.start() must be called before start_read()")
        self._engine_full.start_read(
            req_id,
            uuid,
            remote_endpoint,
            list(remote_block_ids),
            list(local_block_ids),
            parallelism,
        )
        if self._is_hybrid_swa and swa_remote_endpoint is not None:
            self._engine_swa.start_read(
                req_id,
                uuid,
                swa_remote_endpoint,
                list(swa_remote_block_ids or []),
                list(swa_local_block_ids or []),
                parallelism,
            )

    def poll_stats(self) -> tuple[list[str], list[str], list[str]]:
        """Non-blocking poll: ``(done_sending, done_recving, failed_recving)``
        as lists of *req_id strings*. Aggregated across both engines when hybrid
        SWA is active."""

        if not self._started:
            raise RuntimeError("RaidenTransferWrapper.start() must be called before poll_stats()")
        done_s, done_r, failed_r = self._engine_full.poll_stats()
        if self._is_hybrid_swa:
            ds, dr, fr = self._engine_swa.poll_stats()
            done_s = list(set(done_s) | set(ds))
            done_r = list(set(done_r) | set(dr))
            failed_r = list(set(failed_r) | set(fr))
        return done_s, done_r, failed_r


def get_or_create_raiden_wrapper(
    host_ip: str,
    control_port: int = 0,
    *,
    parallelism: int = 1,
) -> RaidenTransferWrapper:
    """Return the process-level raiden wrapper, creating it on first call.

    Only stores config; the engine is constructed on :meth:`start` once the KV
    pool tensors are available.
    """

    global _GLOBAL_RAIDEN_WRAPPER
    with _GLOBAL_RAIDEN_LOCK:
        if _GLOBAL_RAIDEN_WRAPPER is None:
            _GLOBAL_RAIDEN_WRAPPER = RaidenTransferWrapper(
                host_ip, control_port, parallelism=parallelism
            )
        return _GLOBAL_RAIDEN_WRAPPER


def _reset_raiden_singleton_for_test() -> None:
    """Test-only: clear the module-level raiden singleton between cases."""

    global _GLOBAL_RAIDEN_WRAPPER
    with _GLOBAL_RAIDEN_LOCK:
        _GLOBAL_RAIDEN_WRAPPER = None
