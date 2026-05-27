"""Unit tests for :class:`ZmqPullNotifier`.

Run on CPU — pure userspace ZMQ over localhost. Each test allocates a
fresh port from the OS to avoid cross-test collision.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest

from sgl_jax.srt.disaggregation.jax_transfer.zmq_notifier import (
    ZmqPullNotifier,
)


def _free_port() -> int:
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def p_notifier():
    port = _free_port()
    n = ZmqPullNotifier("prefill", "127.0.0.1", port)
    n.start()
    yield n
    n.stop()


@pytest.fixture
def d_notifier():
    n = ZmqPullNotifier("decode", "127.0.0.1", _free_port())
    n.start()
    yield n
    n.stop()


def _wait_for(predicate, timeout_s: float = 2.0):
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_role_validation():
    with pytest.raises(ValueError, match="role"):
        ZmqPullNotifier("foo", "127.0.0.1", 12345)


def test_register_callback_only_on_prefill(d_notifier):
    with pytest.raises(RuntimeError, match="prefill"):
        d_notifier.register_callback(b"req", lambda u: None)


def test_send_done_only_on_decode(p_notifier):
    with pytest.raises(RuntimeError, match="decode"):
        p_notifier.send_done(b"req", "127.0.0.1", p_notifier.port)


def test_register_callback_before_start_raises():
    n = ZmqPullNotifier("prefill", "127.0.0.1", _free_port())
    with pytest.raises(RuntimeError, match="start"):
        n.register_callback(b"x", lambda u: None)


def test_send_done_before_start_raises():
    n = ZmqPullNotifier("decode", "127.0.0.1", _free_port())
    with pytest.raises(RuntimeError, match="start"):
        n.send_done(b"x", "127.0.0.1", 1)


def test_start_is_idempotent(p_notifier):
    p_notifier.start()
    p_notifier.start()
    assert p_notifier.is_started


def test_single_callback_fires(p_notifier, d_notifier):
    received: list[bytes] = []
    event = threading.Event()

    def cb(uuid: bytes) -> None:
        received.append(uuid)
        event.set()

    p_notifier.register_callback(b"req-A", cb)
    d_notifier.send_done(b"req-A", "127.0.0.1", p_notifier.port)
    assert event.wait(timeout=2.0)
    assert received == [b"req-A"]
    assert p_notifier.pending_count() == 0


def test_unregistered_uuid_does_not_crash_listener(p_notifier, d_notifier):
    d_notifier.send_done(b"nobody", "127.0.0.1", p_notifier.port)
    # Listener should remain alive — register a fresh callback and
    # confirm it still fires.
    time.sleep(0.1)
    received = threading.Event()
    p_notifier.register_callback(b"after", lambda u: received.set())
    d_notifier.send_done(b"after", "127.0.0.1", p_notifier.port)
    assert received.wait(timeout=2.0)


def test_duplicate_register_raises(p_notifier):
    p_notifier.register_callback(b"dup", lambda u: None)
    with pytest.raises(RuntimeError, match="already"):
        p_notifier.register_callback(b"dup", lambda u: None)


def test_unregister_returns_callback(p_notifier):
    cb = lambda u: None  # noqa: E731
    p_notifier.register_callback(b"x", cb)
    assert p_notifier.unregister_callback(b"x") is cb
    assert p_notifier.unregister_callback(b"x") is None


def test_concurrent_100_acks(p_notifier, d_notifier):
    n = 100
    seen: list[bytes] = []
    seen_lock = threading.Lock()
    done = threading.Event()

    def make_cb():
        def cb(uuid: bytes) -> None:
            with seen_lock:
                seen.append(uuid)
                if len(seen) == n:
                    done.set()
        return cb

    uuids = [f"req-{i:03d}".encode("utf-8") for i in range(n)]
    for u in uuids:
        p_notifier.register_callback(u, make_cb())

    def sender(u: bytes) -> None:
        d_notifier.send_done(u, "127.0.0.1", p_notifier.port)

    threads = [threading.Thread(target=sender, args=(u,)) for u in uuids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert done.wait(timeout=5.0), f"only saw {len(seen)} / {n} acks"
    assert sorted(seen) == sorted(uuids)
    assert p_notifier.pending_count() == 0


def test_stop_is_idempotent_and_releases_socket():
    n = ZmqPullNotifier("prefill", "127.0.0.1", _free_port())
    n.start()
    n.stop()
    n.stop()  # second stop is a no-op
    assert not n.is_started
