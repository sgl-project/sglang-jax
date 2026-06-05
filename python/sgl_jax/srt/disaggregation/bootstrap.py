"""PD bootstrap: P↔D rendezvous service + client."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
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
        from sgl_jax.srt.disaggregation.common.metrics import (
            PD_BOOTSTRAP_REGISTRY_SIZE,
        )

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
    protocol_version: int = PROTOCOL_VERSION

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class RegisterPrefillRequest(BaseModel):
    bootstrap_key: str
    host: str
    transfer_port: int
    side_channel_port: int
    tp_rank: int = 0
    tp_size: int = 1
    system_dp_rank: int = 0
    protocol_version: int = PROTOCOL_VERSION


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
        stale = [
            k for k, t in self.last_seen.items() if t < cutoff
        ]
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

    def pick_for_room(self, bootstrap_room: int) -> PrefillInfo | None:
        with self.lock:
            self._evict_stale_locked()
            if not self.prefills:
                return None
            keys = sorted(self.prefills.keys())
            chosen = keys[bootstrap_room % len(keys)]
            return self.prefills[chosen]


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
    def get_prefill_info(bootstrap_room: int) -> dict[str, object]:
        info = registry.pick_for_room(bootstrap_room)
        if info is None:
            raise HTTPException(
                status_code=503,
                detail="no prefill workers registered",
            )
        return info.to_dict()

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
        logger.info(
            "BootstrapServer started at %s:%d", self._host, self._port
        )

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
                    self._host, self._port,
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
            f"BootstrapServer did not become ready within {timeout_s}s "
            f"(last error: {last_err})"
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

    @property
    def base_url(self) -> str:
        return self._base_url

    def _headers(self) -> dict[str, str]:
        from sgl_jax.srt.disaggregation.pd_auth import bearer_header

        return bearer_header(self._shared_secret)

    def health(self) -> bool:
        r = httpx.get(
            f"{self._base_url}/health", timeout=self._timeout_s
        )
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
        protocol_version: int = PROTOCOL_VERSION,
    ) -> None:
        payload = {
            "bootstrap_key": bootstrap_key,
            "host": host,
            "transfer_port": transfer_port,
            "side_channel_port": side_channel_port,
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "system_dp_rank": system_dp_rank,
            "protocol_version": protocol_version,
        }
        last_err: Exception | None = None
        for attempt in range(self._register_retries):
            try:
                r = httpx.post(
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
                        attempt + 1, self._register_retries, e,
                        self._register_retry_delay_s,
                    )
                    time.sleep(self._register_retry_delay_s)
        raise RuntimeError(
            f"bootstrap register_prefill failed after "
            f"{self._register_retries} attempts: {last_err}"
        )

    def heartbeat(self, bootstrap_key: str) -> None:
        r = httpx.post(
            f"{self._base_url}/heartbeat",
            json={"bootstrap_key": bootstrap_key},
            timeout=self._timeout_s,
            headers=self._headers(),
        )
        r.raise_for_status()

    def unregister_prefill(self, bootstrap_key: str) -> None:
        r = httpx.post(
            f"{self._base_url}/unregister_prefill",
            json={"bootstrap_key": bootstrap_key},
            timeout=self._timeout_s,
            headers=self._headers(),
        )
        r.raise_for_status()

    def list_prefills(self) -> list[dict[str, object]]:
        r = httpx.get(
            f"{self._base_url}/list_prefills",
            timeout=self._timeout_s,
            headers=self._headers(),
        )
        r.raise_for_status()
        return r.json()["prefills"]

    def get_prefill_info(self, bootstrap_room: int) -> dict[str, object]:
        r = httpx.get(
            f"{self._base_url}/get_prefill_info",
            params={"bootstrap_room": bootstrap_room},
            timeout=self._timeout_s,
            headers=self._headers(),
        )
        r.raise_for_status()
        info = r.json()
        # Reject peers below the supported protocol floor.
        peer_version = int(info.get("protocol_version", 0))
        if peer_version < MIN_COMPATIBLE_VERSION:
            raise RuntimeError(
                f"prefill peer {info.get('bootstrap_key')!r} reports "
                f"protocol_version={peer_version} below "
                f"MIN_COMPATIBLE_VERSION={MIN_COMPATIBLE_VERSION}; "
                "rolling upgrade must finish before this peer can be "
                "used"
            )
        return info


class HeartbeatDaemon:
    """Background thread that heartbeats the bootstrap server on
    behalf of a prefill engine. Without this, the registration falls
    off the registry after ``HEARTBEAT_TTL_SECONDS``.
    """

    def __init__(
        self,
        client: BootstrapClient,
        bootstrap_key: str,
        interval_s: float = HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        self._client = client
        self._bootstrap_key = bootstrap_key
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
            name=f"BootstrapHeartbeat-{self._bootstrap_key}",
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
            try:
                self._client.heartbeat(self._bootstrap_key)
            except Exception:
                logger.warning(
                    "bootstrap heartbeat for %s failed; will retry",
                    self._bootstrap_key,
                    exc_info=True,
                )
            self._stop_event.wait(self._interval_s)
