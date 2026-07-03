"""PD bootstrap: P↔D rendezvous service + client."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass, field

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)


HEARTBEAT_TTL_SECONDS = 30.0
# Beat at ~TTL/3 so a single missed beat doesn't evict the entry.
HEARTBEAT_INTERVAL_SECONDS = HEARTBEAT_TTL_SECONDS / 3.0

# PD wire protocol version. Bump when ``PrefillInfo``
# or any of the 4 endpoint payloads change shape.
PROTOCOL_VERSION: int = 1
MIN_COMPATIBLE_VERSION: int = 1


def _set_registry_size(n: int) -> None:
    """Mirror the registry's current count into Prometheus. Best-effort —
    metrics must never crash the registration path."""

    try:
        from sgl_jax.srt.disaggregation.common.metrics import PD_BOOTSTRAP_REGISTRY_SIZE

        PD_BOOTSTRAP_REGISTRY_SIZE.set(n)
    except Exception:  # noqa: BLE001
        logger.debug("metrics emit failed", exc_info=True)


@dataclass
class PrefillInfo:
    """One prefill worker's connection info as seen by decode workers."""

    bootstrap_key: str
    host: str
    transfer_port: int
    side_channel_port: int
    tp_rank: int = 0
    tp_size: int = 1
    system_dp_rank: int = 0
    jax_process_index: int = 0
    jax_process_count: int = 1
    protocol_version: int = PROTOCOL_VERSION
    # KV layout. Decode must match these or the transferred KV would be
    # silently misinterpreted. Defaults (0 / "") mean "not reported" so a
    # peer predating these fields skips the check.
    page_size: int = 0
    kv_dtype: str = ""
    # raiden control-plane endpoint (Phase 0 raiden data plane). Empty means
    # "not a raiden peer" (path-A). ``local_control_port`` is the resolved TCP
    # port raiden's KVCacheManager listens on; ``raiden_endpoints_json`` is the
    # JSON-encoded ``get_local_endpoints()`` descriptor list that decode passes
    # verbatim to ``start_read(remote_endpoint=...)``.
    local_control_port: int = 0
    raiden_endpoints_json: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _reject_if_below_protocol_floor(info: dict[str, object]) -> None:
    """Raise ``RuntimeError`` if a prefill peer reports a protocol version
    below ``MIN_COMPATIBLE_VERSION``. Shared by the per-request lookup and
    the decode-side :class:`PrefillInfoCache` so both reject stale peers
    identically."""

    peer_version = int(info.get("protocol_version", 0))
    if peer_version < MIN_COMPATIBLE_VERSION:
        raise RuntimeError(
            f"prefill peer {info.get('bootstrap_key')!r} reports "
            f"protocol_version={peer_version} below "
            f"MIN_COMPATIBLE_VERSION={MIN_COMPATIBLE_VERSION}; "
            "rolling upgrade must finish before this peer can be "
            "used"
        )


def resolve_kv_dtype_name(dtype: object) -> str:
    """Canonical dtype name for KV-layout compatibility advertising.

    Both peers must publish the dtype actually used by their initialized KV
    pool — not the CLI literal (often ``"auto"``) — otherwise two peers with
    different resolved dtypes but both configured ``auto`` would pass the
    compatibility check, and an ``auto`` peer could reject a compatible one.
    """

    if dtype is None:
        return ""
    try:
        import jax.numpy as jnp

        return str(jnp.dtype(dtype).name)
    except Exception:
        return str(dtype)


def check_prefill_compat(
    info: dict[str, object],
    *,
    local_page_size: int,
    local_kv_dtype: str,
) -> None:
    """Raise ``ValueError`` if the prefill peer's KV layout is incompatible.

    Backward-compatible: a field is only checked when both sides report it
    (peer value truthy and local value truthy), so older peers and
    not-yet-initialized decoders never trigger a false rejection.
    """

    peer_page_size = int(info.get("page_size", 0) or 0)
    if peer_page_size and local_page_size and peer_page_size != local_page_size:
        raise ValueError(
            f"prefill peer {info.get('bootstrap_key')!r} uses "
            f"page_size={peer_page_size} but this decode uses "
            f"page_size={local_page_size}; KV layout incompatible"
        )
    peer_kv_dtype = str(info.get("kv_dtype", "") or "")
    if peer_kv_dtype and local_kv_dtype and peer_kv_dtype != local_kv_dtype:
        raise ValueError(
            f"prefill peer {info.get('bootstrap_key')!r} uses "
            f"kv_dtype={peer_kv_dtype!r} but this decode uses "
            f"kv_dtype={local_kv_dtype!r}; KV layout incompatible"
        )


class RegisterPrefillRequest(BaseModel):
    bootstrap_key: str
    host: str
    transfer_port: int
    side_channel_port: int
    tp_rank: int = 0
    tp_size: int = 1
    system_dp_rank: int = 0
    jax_process_index: int = 0
    jax_process_count: int = 1
    protocol_version: int = PROTOCOL_VERSION
    page_size: int = 0
    kv_dtype: str = ""
    local_control_port: int = 0
    raiden_endpoints_json: str = ""


class RegisterTransferRequest(BaseModel):
    """Per-request block metadata P publishes so D can pull with raiden.

    raiden's ``start_read`` needs the producer's device block ids
    (``remote_block_ids``). Those are per-request and only P knows them (the
    paged allocator picks non-contiguous blocks), so they cannot be derived by
    D deterministically — P registers them here keyed by ``bootstrap_room`` and
    D fetches them at admission. This is the raiden analogue of path-A's
    crc32(uuid) pairing (which only needed a shared int, not the block layout).
    """

    bootstrap_room: int
    transfer_id: str
    remote_block_ids: list[int]
    # Endpoint + control port are also on PrefillInfo, but a decode that already
    # resolved its peer can skip the second lookup by reading them here.
    local_control_port: int = 0
    raiden_endpoints_json: str = ""
    # Chunked KV transfer: P publishes one entry per chunk (each an independent
    # raiden uuid, since register_read is overwrite-per-uuid not cumulative).
    # ``chunk_index`` is 0..N-1; ``num_chunks`` is set to N on the FINAL chunk
    # (0 means "more chunks coming"), so D learns the total once P is done.
    # ``chunk_page_offset`` is the sequence-relative page index where this chunk
    # starts, so D can slice the matching pages of its whole-prompt kv_indices
    # without having to reproduce P's chunk boundaries.
    chunk_index: int = 0
    num_chunks: int = 0
    chunk_page_offset: int = 0
    # SWA hybrid-attention fields (empty for non-SWA models).
    swa_block_ids: list[int] = []
    swa_raiden_endpoints_json: str = ""


class HeartbeatRequest(BaseModel):
    bootstrap_key: str


class UnregisterPrefillRequest(BaseModel):
    bootstrap_key: str


@dataclass
class _Registry:
    """In-memory state for the FastAPI app. Carries its own lock."""

    prefills: dict[str, PrefillInfo] = field(default_factory=dict)
    last_seen: dict[str, float] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    ttl_seconds: float = HEARTBEAT_TTL_SECONDS
    clock: Callable[[], float] = time.monotonic
    # raiden per-request block metadata, keyed by bootstrap_room. Written by P
    # (register_read side) and read by D at admission + on every poll tick.
    # Chunked transfer publishes one entry per chunk, so the value is an inner
    # dict keyed by chunk_index. Accumulated (not overwritten) so D can learn
    # about new chunks as P produces them; a room is popped explicitly by D once
    # the whole transfer is done (or by TTL) to keep the table bounded.
    transfers: dict[int, dict[int, dict[str, object]]] = field(default_factory=dict)
    # Per room: N once P has published its FINAL chunk (num_chunks>0); absent
    # until then so D keeps polling instead of assuming the transfer is complete.
    transfer_num_chunks: dict[int, int] = field(default_factory=dict)

    def now(self) -> float:
        return self.clock()  # type: ignore[no-any-return]

    def register(self, info: PrefillInfo) -> None:
        with self.lock:
            self.prefills[info.bootstrap_key] = info
            self.last_seen[info.bootstrap_key] = self.now()
            current = len(self.prefills)
        _set_registry_size(current)

    def heartbeat(self, key: str) -> bool:
        with self.lock:
            if key not in self.prefills:
                return False
            self.last_seen[key] = self.now()
            return True

    def unregister(self, key: str) -> bool:
        with self.lock:
            removed = self.prefills.pop(key, None)
            self.last_seen.pop(key, None)
            current = len(self.prefills)
        if removed is not None:
            _set_registry_size(current)
        return removed is not None

    def _evict_stale_locked(self) -> None:
        cutoff = self.now() - self.ttl_seconds
        stale = [k for k, t in self.last_seen.items() if t < cutoff]
        if not stale:
            return
        for k in stale:
            self.prefills.pop(k, None)
            self.last_seen.pop(k, None)
        _set_registry_size(len(self.prefills))

    def list_all(self) -> list[PrefillInfo]:
        with self.lock:
            self._evict_stale_locked()
            return list(self.prefills.values())

    def pick_for_room(
        self, bootstrap_room: int, dp_rank: int | None = None
    ) -> PrefillInfo | None:
        with self.lock:
            self._evict_stale_locked()
            if not self.prefills:
                return None
            candidates = {
                k: v
                for k, v in self.prefills.items()
                if dp_rank is None or v.system_dp_rank == dp_rank
            }
            if not candidates:
                return None
            keys = sorted(candidates.keys())
            chosen = keys[bootstrap_room % len(keys)]
            return candidates[chosen]

    def register_transfer(self, info: dict[str, object]) -> None:
        with self.lock:
            room = int(info["bootstrap_room"])
            chunk_index = int(info.get("chunk_index", 0) or 0)
            self.transfers.setdefault(room, {})[chunk_index] = info
            num_chunks = int(info.get("num_chunks", 0) or 0)
            if num_chunks > 0:
                self.transfer_num_chunks[room] = num_chunks

    def get_transfer_chunks(self, bootstrap_room: int) -> dict[str, object] | None:
        """Read (without consuming) all chunk metadata P has published for
        ``bootstrap_room``.

        Returns ``{"chunks": {chunk_index: info, ...}, "num_chunks": N}`` where
        ``num_chunks`` is 0 until P publishes its final chunk. D polls this every
        tick to discover newly-published chunks; it pops the room via
        :meth:`pop_room` once the whole transfer is done. Returns None if P has
        not registered any chunk yet (D defers + retries).
        """
        with self.lock:
            room = int(bootstrap_room)
            chunks = self.transfers.get(room)
            if not chunks:
                return None
            return {
                "chunks": dict(chunks),
                "num_chunks": self.transfer_num_chunks.get(room, 0),
            }

    def pop_room(self, bootstrap_room: int) -> None:
        """Drop all chunk metadata for ``bootstrap_room`` (D calls this on
        SUCCESS/failure so the table stays bounded)."""
        with self.lock:
            room = int(bootstrap_room)
            self.transfers.pop(room, None)
            self.transfer_num_chunks.pop(room, None)


def build_app(
    registry: _Registry | None = None,
    *,
    shared_secret: str | None = None,
) -> tuple[FastAPI, _Registry]:
    """Build the FastAPI app + return its registry.

    Exposed for testing (pass a registry with a mock clock) and for
    :class:`BootstrapServer` (which manages its own).

    ``shared_secret`` enables Bearer auth on every
    endpoint except ``/health``.
    """

    if registry is None:
        registry = _Registry()

    app = FastAPI()

    if shared_secret is not None:
        from sgl_jax.srt.disaggregation.pd_auth import verify_bearer

        @app.middleware("http")
        async def _auth_mw(request: Request, call_next):
            # /health stays open so liveness / readiness probes from
            # kubelet (which don't share the secret) keep working.
            if request.url.path != "/health" and not verify_bearer(
                shared_secret, request.headers.get("authorization")
            ):
                from fastapi.responses import JSONResponse

                return JSONResponse(
                    status_code=401,
                    content={"detail": "unauthorized"},
                )
            return await call_next(request)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/register_prefill")
    def register_prefill(req: RegisterPrefillRequest) -> dict[str, str]:
        info = PrefillInfo(**req.model_dump())
        registry.register(info)
        return {"status": "registered"}

    @app.post("/heartbeat")
    def heartbeat(req: HeartbeatRequest) -> dict[str, str]:
        if not registry.heartbeat(req.bootstrap_key):
            raise HTTPException(
                status_code=404,
                detail=f"unknown bootstrap_key {req.bootstrap_key!r}",
            )
        return {"status": "ok"}

    @app.post("/unregister_prefill")
    def unregister_prefill(req: UnregisterPrefillRequest) -> dict[str, str]:
        registry.unregister(req.bootstrap_key)
        return {"status": "unregistered"}

    @app.get("/list_prefills")
    def list_prefills() -> dict[str, list[dict[str, object]]]:
        return {"prefills": [p.to_dict() for p in registry.list_all()]}

    @app.get("/get_prefill_info")
    def get_prefill_info(
        bootstrap_room: int, dp_rank: int | None = None
    ) -> dict[str, object]:
        info = registry.pick_for_room(bootstrap_room, dp_rank=dp_rank)
        if info is None:
            raise HTTPException(
                status_code=503,
                detail="no prefill workers registered",
            )
        return info.to_dict()

    @app.post("/register_transfer")
    def register_transfer(req: RegisterTransferRequest) -> dict[str, str]:
        registry.register_transfer(req.model_dump())
        return {"status": "registered"}

    @app.get("/get_transfer_info")
    def get_transfer_info(bootstrap_room: int) -> dict[str, object]:
        info = registry.get_transfer_chunks(bootstrap_room)
        if info is None:
            # Not registered yet: 404 lets the decode side treat it as
            # "defer + retry" (never abort) rather than a hard error.
            raise HTTPException(
                status_code=404,
                detail=f"no transfer info for bootstrap_room={bootstrap_room}",
            )
        return info

    @app.post("/pop_transfer")
    def pop_transfer(bootstrap_room: int) -> dict[str, str]:
        registry.pop_room(bootstrap_room)
        return {"status": "popped"}

    # Bootstrap runs as a standalone single process and does NOT inherit
    # PROMETHEUS_MULTIPROC_DIR, so it exposes its own default-registry
    # /metrics (single-process) carrying pd_bootstrap_registry_size.
    with suppress(ImportError):
        from fastapi.responses import Response
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        @app.get("/metrics")
        def metrics() -> Response:
            return Response(
                content=generate_latest(),
                media_type=CONTENT_TYPE_LATEST,
            )

    return app, registry


class BootstrapServer:
    """Runs the bootstrap FastAPI app in a background uvicorn thread."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8998,
        log_level: str = "warning",
        *,
        shared_secret: str | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._log_level = log_level
        self.app, self.registry = build_app(shared_secret=shared_secret)
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._started = False

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def is_started(self) -> bool:
        return self._started

    def start(self) -> None:
        if self._started:
            return
        config = uvicorn.Config(
            self.app,
            host=self._host,
            port=self._port,
            log_level=self._log_level,
            access_log=False,
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        # uvicorn.Server.run() installs signal handlers, which can only
        # be done from the main thread. Tell it not to.
        self._server.install_signal_handlers = False
        self._thread = threading.Thread(
            target=self._server.run,
            name=f"BootstrapServer-{self._port}",
            daemon=True,
        )
        self._thread.start()
        self._wait_until_ready(timeout_s=10.0)
        self._started = True
        logger.info("BootstrapServer started at %s:%d", self._host, self._port)

    def stop(self) -> None:
        if not self._started:
            return
        assert self._server is not None
        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning(
                    "BootstrapServer at %s:%d did not stop within 5s",
                    self._host,
                    self._port,
                )
            self._thread = None
        self._server = None
        self._started = False

    def _wait_until_ready(self, timeout_s: float) -> None:
        url = f"http://127.0.0.1:{self._port}/health"
        deadline = time.monotonic() + timeout_s
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            try:
                r = httpx.get(url, timeout=0.5)
                if r.status_code == 200:
                    return
            except Exception as e:  # noqa: BLE001
                last_err = e
            time.sleep(0.05)
        raise TimeoutError(
            f"BootstrapServer did not become ready within {timeout_s}s " f"(last error: {last_err})"
        )


class BootstrapClient:
    """Stateless HTTP client for the bootstrap server."""

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 5.0,
        register_retries: int = 30,
        register_retry_delay_s: float = 1.0,
        *,
        shared_secret: str | None = None,
    ) -> None:
        if base_url.endswith("/"):
            base_url = base_url[:-1]
        self._base_url = base_url
        self._timeout_s = timeout_s
        self._register_retries = register_retries
        self._register_retry_delay_s = register_retry_delay_s
        self._shared_secret = shared_secret
        # Reuse one thread-safe ``httpx.Client`` (and its connection pool + SSL
        # context) across calls; building a throwaway client per call rebuilds
        # the SSL context every time, which under load blocked the decode event
        # loop inside ``get_prefill_info``.
        self._client = httpx.Client(timeout=timeout_s)

    @property
    def base_url(self) -> str:
        return self._base_url

    def _headers(self) -> dict[str, str]:
        from sgl_jax.srt.disaggregation.pd_auth import bearer_header

        return bearer_header(self._shared_secret)

    def health(self) -> bool:
        r = self._client.get(f"{self._base_url}/health", timeout=self._timeout_s)
        return r.status_code == 200

    def register_prefill(
        self,
        bootstrap_key: str,
        host: str,
        transfer_port: int,
        side_channel_port: int,
        *,
        tp_rank: int = 0,
        tp_size: int = 1,
        system_dp_rank: int = 0,
        jax_process_index: int = 0,
        jax_process_count: int = 1,
        protocol_version: int = PROTOCOL_VERSION,
        page_size: int = 0,
        kv_dtype: str = "",
        local_control_port: int = 0,
        raiden_endpoints_json: str = "",
    ) -> None:
        payload = {
            "bootstrap_key": bootstrap_key,
            "host": host,
            "transfer_port": transfer_port,
            "side_channel_port": side_channel_port,
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "system_dp_rank": system_dp_rank,
            "jax_process_index": jax_process_index,
            "jax_process_count": jax_process_count,
            "protocol_version": protocol_version,
            "page_size": page_size,
            "kv_dtype": kv_dtype,
            "local_control_port": local_control_port,
            "raiden_endpoints_json": raiden_endpoints_json,
        }
        last_err: Exception | None = None
        for attempt in range(self._register_retries):
            try:
                r = self._client.post(
                    f"{self._base_url}/register_prefill",
                    json=payload,
                    timeout=self._timeout_s,
                    headers=self._headers(),
                )
                r.raise_for_status()
                return
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_err = e
                if attempt + 1 < self._register_retries:
                    logger.warning(
                        "bootstrap register_prefill attempt %d/%d "
                        "failed (%s); retrying in %.1fs",
                        attempt + 1,
                        self._register_retries,
                        e,
                        self._register_retry_delay_s,
                    )
                    time.sleep(self._register_retry_delay_s)
        raise RuntimeError(
            f"bootstrap register_prefill failed after "
            f"{self._register_retries} attempts: {last_err}"
        )

    def heartbeat(self, bootstrap_key: str) -> None:
        r = self._client.post(
            f"{self._base_url}/heartbeat",
            json={"bootstrap_key": bootstrap_key},
            timeout=self._timeout_s,
            headers=self._headers(),
        )
        r.raise_for_status()

    def unregister_prefill(self, bootstrap_key: str) -> None:
        r = self._client.post(
            f"{self._base_url}/unregister_prefill",
            json={"bootstrap_key": bootstrap_key},
            timeout=self._timeout_s,
            headers=self._headers(),
        )
        r.raise_for_status()

    def list_prefills(self) -> list[dict[str, object]]:
        r = self._client.get(
            f"{self._base_url}/list_prefills",
            timeout=self._timeout_s,
            headers=self._headers(),
        )
        r.raise_for_status()
        return r.json()["prefills"]

    def get_prefill_info(
        self, bootstrap_room: int, dp_rank: int | None = None
    ) -> dict[str, object]:
        params: dict[str, object] = {"bootstrap_room": bootstrap_room}
        if dp_rank is not None:
            params["dp_rank"] = dp_rank
        r = self._client.get(
            f"{self._base_url}/get_prefill_info",
            params=params,
            timeout=self._timeout_s,
            headers=self._headers(),
        )
        r.raise_for_status()
        info = r.json()
        # Reject peers below the supported protocol floor.
        _reject_if_below_protocol_floor(info)
        return info

    def register_transfer(
        self,
        bootstrap_room: int,
        transfer_id: str,
        remote_block_ids: list[int],
        *,
        local_control_port: int = 0,
        raiden_endpoints_json: str = "",
        chunk_index: int = 0,
        num_chunks: int = 0,
        chunk_page_offset: int = 0,
        swa_block_ids: list[int] | None = None,
        swa_raiden_endpoints_json: str = "",
    ) -> None:
        """P: publish per-chunk block metadata for raiden pull (keyed by room +
        chunk_index). Best-effort with the shared client timeout; the caller
        treats a failure as a transfer failure for that request. ``num_chunks``
        is set to N only on the final chunk (0 means more chunks coming).

        For hybrid SWA models, ``swa_block_ids`` and ``swa_raiden_endpoints_json``
        carry the SWA-pool counterpart metadata."""

        payload = {
            "bootstrap_room": bootstrap_room,
            "transfer_id": transfer_id,
            "remote_block_ids": list(remote_block_ids),
            "local_control_port": local_control_port,
            "raiden_endpoints_json": raiden_endpoints_json,
            "chunk_index": chunk_index,
            "num_chunks": num_chunks,
            "chunk_page_offset": chunk_page_offset,
            "swa_block_ids": list(swa_block_ids or []),
            "swa_raiden_endpoints_json": swa_raiden_endpoints_json,
        }
        r = self._client.post(
            f"{self._base_url}/register_transfer",
            json=payload,
            timeout=self._timeout_s,
            headers=self._headers(),
        )
        r.raise_for_status()

    def get_transfer_info(self, bootstrap_room: int) -> dict[str, object] | None:
        """D: fetch (without consuming) all chunk metadata P published for
        ``bootstrap_room``. Returns ``{"chunks": {int: info}, "num_chunks": N}``
        with int chunk-index keys, or None if nothing registered yet (404) so
        the caller defers + retries rather than aborting."""

        r = self._client.get(
            f"{self._base_url}/get_transfer_info",
            params={"bootstrap_room": bootstrap_room},
            timeout=self._timeout_s,
            headers=self._headers(),
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        body = r.json()
        # JSON turns the inner dict's int keys into strings; normalize back.
        raw_chunks = body.get("chunks", {}) or {}
        chunks = {int(k): v for k, v in raw_chunks.items()}
        return {"chunks": chunks, "num_chunks": int(body.get("num_chunks", 0) or 0)}

    def pop_transfer(self, bootstrap_room: int) -> None:
        """D: drop the room's chunk metadata once the whole transfer is done
        (or failed), keeping the bootstrap table bounded. Best-effort."""

        r = self._client.post(
            f"{self._base_url}/pop_transfer",
            params={"bootstrap_room": bootstrap_room},
            timeout=self._timeout_s,
            headers=self._headers(),
        )
        r.raise_for_status()


class PrefillInfoCache:
    """Decode-side cache of the prefill registry.

    Resolves per-room selection LOCALLY — mirroring the server's
    ``_Registry.pick_for_room`` (``sorted(keys)[room % len]``) — so a warm
    cache serves requests with zero network round-trips. The full registry is
    refreshed via ``list_prefills`` at most once per ``refresh_interval_s`` to
    evict dead/unregistered prefills. If the registry is empty the room returns
    ``None`` so the caller defers and retries next tick (never abort).
    """

    def __init__(
        self,
        client: BootstrapClient,
        *,
        refresh_interval_s: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = client
        self._refresh_interval_s = refresh_interval_s
        self._clock = clock
        self._lock = threading.Lock()
        self._by_key: dict[str, dict[str, object]] = {}
        self._sorted_keys: list[str] = []
        # -inf so the very first lookup always refreshes regardless of clock.
        self._last_refresh: float = float("-inf")
        # Transient-failure bookkeeping (throttled logging).
        self._refresh_failures = 0
        self._last_fail_log: float = float("-inf")

    def _refresh_locked(self) -> None:
        prefills = self._client.list_prefills()
        by_key = {str(p["bootstrap_key"]): p for p in prefills}
        self._by_key = by_key
        self._sorted_keys = sorted(by_key)
        self._last_refresh = self._clock()

    def _pick_locked(
        self, bootstrap_room: int, dp_rank: int | None = None
    ) -> dict[str, object] | None:
        if not self._sorted_keys:
            return None
        if dp_rank is not None:
            candidates = [
                k
                for k in self._sorted_keys
                if int(self._by_key[k].get("system_dp_rank", 0)) == dp_rank
            ]
            if not candidates:
                return None
            chosen = candidates[bootstrap_room % len(candidates)]
        else:
            chosen = self._sorted_keys[bootstrap_room % len(self._sorted_keys)]
        return self._by_key[chosen]

    def pick_for_room(
        self, bootstrap_room: int, dp_rank: int | None = None
    ) -> dict[str, object] | None:
        """Return prefill info for ``bootstrap_room``, or ``None`` if no
        prefill is registered yet (caller should defer + retry).

        Refreshes the warm cache whenever ``refresh_interval_s`` has elapsed
        (rate-limited), regardless of hit/miss, so a prefill that has died or
        unregistered is evicted instead of being served from a stale entry.
        Within the interval, a hit returns from cache with zero network I/O.
        Raises ``RuntimeError`` if the chosen peer is below the protocol floor.
        """

        with self._lock:
            now = self._clock()
            if now - self._last_refresh >= self._refresh_interval_s:
                try:
                    self._refresh_locked()
                except Exception as exc:  # noqa: BLE001
                    # A transient bootstrap-server failure (timeout / 5xx /
                    # connection reset) must NOT propagate: the decode intake
                    # catches any exception here as an abort, which would
                    # violate the never-abort contract over a momentary blip.
                    # Back off for the interval (so we neither hammer a down
                    # server nor re-block the event loop every tick) and serve
                    # from the existing — possibly stale — cache. An empty
                    # cache returns None below, so the caller defers + retries.
                    self._last_refresh = now
                    self._refresh_failures += 1
                    if now - self._last_fail_log >= 5.0:
                        self._last_fail_log = now
                        logger.warning(
                            "PrefillInfoCache refresh failed (%d total); serving "
                            "%d cached prefill(s): %s",
                            self._refresh_failures,
                            len(self._sorted_keys),
                            exc,
                        )
            info = self._pick_locked(bootstrap_room, dp_rank=dp_rank)
        if info is None:
            return None
        _reject_if_below_protocol_floor(info)
        return info


class HeartbeatDaemon:
    """Background thread that heartbeats the bootstrap server on
    behalf of a prefill engine. Without this, the registration falls
    off the registry after ``HEARTBEAT_TTL_SECONDS``.
    """

    def __init__(
        self,
        client: BootstrapClient,
        bootstrap_keys: list[str],
        interval_s: float = HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        self._client = client
        self._bootstrap_keys = bootstrap_keys
        self._interval_s = interval_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name=f"BootstrapHeartbeat-{self._bootstrap_keys[0]}-...",
            daemon=True,
        )
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_s + 1.0)
            self._thread = None
        self._started = False

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            for key in self._bootstrap_keys:
                if self._stop_event.is_set():
                    break
                try:
                    self._client.heartbeat(key)
                except Exception:
                    logger.warning(
                        "bootstrap heartbeat for %s failed; will retry",
                        key,
                        exc_info=True,
                    )
            self._stop_event.wait(self._interval_s)
