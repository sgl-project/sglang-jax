"""D→P pull-done side channel.

Decouples the prefill side from blocking on transfer completion. The
Stage 0 sender optimistically advanced to ``SUCCESS`` right after
``register_pull`` returned; in Stage 1 the sender waits for an explicit
ack from the decoder before transitioning. The transport is ZMQ
``DEALER → ROUTER`` carrying ``msgpack({"uuid": bytes})``.

Threading model:
  * P side: ROUTER socket bound in :meth:`start`. A daemon listener
    thread recv-loops, parses each frame, looks up the callback under
    a lock, and invokes it. ``register_callback`` from the main thread
    and ``pop`` from the listener thread share ``_callbacks`` under the
    same lock — the read-then-decide-then-pop sequence on the listener
    side is atomic.
  * D side: A fresh DEALER socket per ``send_done`` call (no pool, no
    persistent connection). The send is fire-and-forget — if the P
    side has already torn down or hasn't bound yet, the ack is lost.
    Recovery from lost acks is intentionally out of scope (Stage 4
    hardening adds end-to-end timeouts).

Process-wide ZMQ context: ``zmq.Context.instance()``. Tests spinning
up multiple notifiers on different ports share the context safely.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from collections import OrderedDict
from typing import Callable, Dict, Optional, Set

import msgpack
import zmq

logger = logging.getLogger(__name__)


PullDoneCallback = Callable[[bytes], None]


@dataclasses.dataclass(frozen=True)
class RetiredTransferInfo:
    state: str
    reason: str
    retired_at: float


class ZmqPullNotifier:
    """Backend-local ZMQ notifier for ``pull-done`` events.

    Lives next to ``JaxTransferKVManager`` in
    ``srt/disaggregation/jax_transfer/`` because multi-host coordination
    of the side channel is the backend's responsibility. Each PD
    process creates one of these and shares it between sender and
    receiver paths.
    """

    def __init__(
        self,
        role: str,
        host: str,
        port: int,
        *,
        shared_secret: Optional[str] = None,
    ) -> None:
        if role not in ("prefill", "decode"):
            raise ValueError(
                f"role must be 'prefill' or 'decode', got {role!r}"
            )
        self._role = role
        self._host = host
        self._port = port
        self._shared_secret = shared_secret
        self._ctx = zmq.Context.instance()
        self._router: Optional[zmq.Socket] = None
        self._stop_event = threading.Event()
        self._listener_thread: Optional[threading.Thread] = None
        self._callbacks_lock = threading.Lock()
        self._callbacks: Dict[bytes, PullDoneCallback] = {}
        self._dispatching: Set[bytes] = set()
        self._retired: OrderedDict[bytes, RetiredTransferInfo] = (
            OrderedDict()
        )
        self._max_retired = 4096
        self._started = False

    @property
    def role(self) -> str:
        return self._role

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
        """Idempotent. On P, binds the ROUTER and spawns the listener
        thread. On D, this is a no-op — DEALER sockets are created
        per-send.
        """

        if self._started:
            return
        if self._role == "prefill":
            self._router = self._ctx.socket(zmq.ROUTER)
            # Without LINGER=0, closing the socket while messages are
            # still pending would block. We don't care about queued
            # outbound frames at shutdown.
            self._router.setsockopt(zmq.LINGER, 0)
            self._router.bind(f"tcp://{self._host}:{self._port}")
            self._stop_event.clear()
            self._listener_thread = threading.Thread(
                target=self._listen_loop,
                name=f"ZmqPullNotifier-listener-{self._port}",
                daemon=True,
            )
            self._listener_thread.start()
        self._started = True
        logger.info(
            "ZmqPullNotifier started role=%s addr=%s:%d",
            self._role, self._host, self._port,
        )

    def stop(self) -> None:
        """Stop the listener and tear down the ROUTER. Idempotent."""

        if not self._started:
            return
        self._stop_event.set()
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=3.0)
            if self._listener_thread.is_alive():
                # The listener is stuck in poll/recv. Closing the
                # ROUTER socket from this thread is undefined behavior
                # because pyzmq sockets are not thread-safe, but we
                # log loudly and continue — leaving the listener
                # running would leak the socket and the port. Stage 1
                # review I3.
                logger.warning(
                    "ZmqPullNotifier listener at port %d did not stop "
                    "within 3s; closing socket from main thread, which "
                    "may produce a spurious ZMQError from the listener.",
                    self._port,
                )
            self._listener_thread = None
        if self._router is not None:
            self._router.close(linger=0)
            self._router = None
        self._started = False

    # ------------------------------------------------------------------
    # P side
    # ------------------------------------------------------------------

    def register_callback(
        self, uuid: bytes, cb: PullDoneCallback
    ) -> None:
        """Register ``cb`` to fire when an ack for ``uuid`` arrives.

        Raises ``RuntimeError`` if the same uuid is already registered;
        callers must explicitly cancel before re-registering to avoid
        silent overwrites under concurrent register/listener access.
        """

        if self._role != "prefill":
            raise RuntimeError(
                "register_callback is only valid on prefill notifiers"
            )
        if not self._started:
            raise RuntimeError("call start() before register_callback")
        with self._callbacks_lock:
            self._dispatching.discard(uuid)
            self._retired.pop(uuid, None)
            if uuid in self._callbacks:
                raise RuntimeError(
                    f"uuid={uuid!r} already has a pending callback"
                )
            self._callbacks[uuid] = cb

    def unregister_callback(self, uuid: bytes) -> Optional[PullDoneCallback]:
        with self._callbacks_lock:
            return self._callbacks.pop(uuid, None)

    def pending_count(self) -> int:
        with self._callbacks_lock:
            return len(self._callbacks)

    def mark_retired(
        self, uuid: bytes, *, state: str, reason: str
    ) -> None:
        """Remember that ``uuid`` reached a terminal state.

        This keeps a small bounded record after the live callback is
        gone so the listener can distinguish benign late acks from
        truly unknown uuids.
        """

        info = RetiredTransferInfo(
            state=state,
            reason=reason,
            retired_at=time.monotonic(),
        )
        with self._callbacks_lock:
            self._dispatching.discard(uuid)
            self._retired[uuid] = info
            self._retired.move_to_end(uuid)
            while len(self._retired) > self._max_retired:
                self._retired.popitem(last=False)

    # ------------------------------------------------------------------
    # D side
    # ------------------------------------------------------------------

    def send_done(
        self, uuid: bytes, target_host: str, target_port: int
    ) -> None:
        """D→P: tell the prefill side ``uuid`` has been pulled.

        Fire-and-forget. Uses a fresh DEALER socket per call. The
        wrapper holds the ZMQ context, so socket creation is cheap.

        When ``shared_secret`` is set the message carries an
        ``hmac`` field; the prefill listener rejects messages without
        a valid tag (Stage 4 H-C).
        """

        if self._role != "decode":
            raise RuntimeError(
                "send_done is only valid on decode notifiers"
            )
        if not self._started:
            raise RuntimeError("call start() before send_done")
        payload = {"uuid": uuid}
        if self._shared_secret is not None:
            from sgl_jax.srt.disaggregation.pd_auth import compute_tag

            payload["hmac"] = compute_tag(self._shared_secret, uuid)
        sock = self._ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.LINGER, 1000)
        try:
            sock.connect(f"tcp://{target_host}:{target_port}")
            sock.send(msgpack.packb(payload, use_bin_type=True))
        finally:
            sock.close()

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _listen_loop(self) -> None:
        assert self._router is not None
        poller = zmq.Poller()
        poller.register(self._router, zmq.POLLIN)
        while not self._stop_event.is_set():
            events = dict(poller.poll(timeout=100))
            if self._router not in events:
                continue
            try:
                frames = self._router.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                continue
            # ROUTER prefix: [identity, payload]
            if len(frames) < 2:
                logger.warning(
                    "ZmqPullNotifier got malformed frame "
                    "(len=%d), dropping", len(frames),
                )
                continue
            payload = frames[-1]
            try:
                msg = msgpack.unpackb(payload, raw=True)
                uuid = msg[b"uuid"]
            except (msgpack.UnpackException, KeyError, TypeError) as e:
                logger.warning(
                    "ZmqPullNotifier failed to decode payload: %s", e
                )
                continue
            if self._shared_secret is not None:
                from sgl_jax.srt.disaggregation.pd_auth import verify_tag

                candidate = msg.get(b"hmac") if isinstance(msg, dict) else None
                if not verify_tag(self._shared_secret, uuid, candidate):
                    logger.warning(
                        "ZmqPullNotifier dropping uuid=%r with "
                        "missing/invalid HMAC", uuid,
                    )
                    try:
                        from sgl_jax.srt.disaggregation.metrics import (
                            PD_TRANSFER_FAILURES_TOTAL,
                        )

                        PD_TRANSFER_FAILURES_TOTAL.labels(
                            reason="auth", role="prefill"
                        ).inc()
                    except Exception:  # noqa: BLE001
                        pass
                    continue
            dispatching = False
            retired: Optional[RetiredTransferInfo] = None
            with self._callbacks_lock:
                cb = self._callbacks.pop(uuid, None)
                if cb is not None:
                    self._dispatching.add(uuid)
                else:
                    dispatching = uuid in self._dispatching
                    retired = self._retired.get(uuid)
                    if retired is not None:
                        self._retired.move_to_end(uuid)
            if cb is None:
                if dispatching:
                    logger.info(
                        "ZmqPullNotifier received duplicate ack for "
                        "uuid=%r while callback dispatch is in flight; "
                        "dropping",
                        uuid,
                    )
                    continue
                if retired is not None:
                    logger.info(
                        "ZmqPullNotifier received late ack for retired "
                        "transfer uuid=%r state=%s reason=%s; dropping",
                        uuid,
                        retired.state,
                        retired.reason,
                    )
                    continue
                logger.warning(
                    "ZmqPullNotifier received uuid=%r with no "
                    "registered callback; dropping",
                    uuid,
                )
                continue
            try:
                cb(uuid)
            except Exception:
                logger.exception(
                    "ZmqPullNotifier callback for uuid=%r raised", uuid
                )
            finally:
                with self._callbacks_lock:
                    self._dispatching.discard(uuid)
