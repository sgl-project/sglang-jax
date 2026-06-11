"""Decode pull is dispatched inline and never blocks the event loop.

The decode scheduler is single-threaded. ``JaxTransferKVReceiver.poll()``
dispatches ``wrapper.pull`` inline — jax's transfer ``pull`` returns a
future immediately, so the dispatch is non-blocking and no per-request
background thread is needed. These tests pin that contract:

  * ``init()`` pre-connects the link to the remote peer so the latency of
    the first ``server.connect`` never lands inside ``poll()``.
  * the first ``poll()`` transitions WAITING_FOR_INPUT -> TRANSFERRING,
    dispatches the pull once, and stores the futures synchronously.
  * a later ``poll()`` drives ack -> SUCCESS once every future is ready.
  * a reaper-driven ``fail()`` wins the terminal state; a subsequent
    ``poll()`` stays FAILED and never sends an ack.
  * ``poll()`` spawns no threads.
"""
from __future__ import annotations

import threading

from sgl_jax.srt.disaggregation.base.kv_manager import KVPoll
from sgl_jax.srt.disaggregation.jax_transfer.conn import (
    JaxTransferKVReceiver,
    PMetadata,
)


class _Leaf:
    """Stand-in for a pulled jax.Array leaf; a custom object is a pytree leaf.

    Readiness is read live from the owning wrapper so a test can flip it
    after the pull has been dispatched.
    """

    def __init__(self, wrapper: "_Wrapper"):
        self._wrapper = wrapper

    def is_ready(self) -> bool:
        return self._wrapper.ready


class _Wrapper:
    def __init__(self, *, raise_exc: bool = False, ready: bool = True):
        self._raise = raise_exc
        self.ready = ready
        self.calls = 0
        self.connected: list[str] = []

    def connect(self, remote_addr):
        self.connected.append(remote_addr)

    def pull(self, uuid, spec, remote_addr=None):
        self.calls += 1
        if self._raise:
            raise RuntimeError("pull boom")
        return _Leaf(self)


class _Notifier:
    def __init__(self):
        self.sent = []

    def send_done(self, uuid_bytes, host, port):
        self.sent.append((uuid_bytes, host, port))


class _FakeMgr:
    def __init__(self, *, raise_exc=False, ready=True):
        self._wrapper = _Wrapper(raise_exc=raise_exc, ready=ready)
        self._notifier = _Notifier()
        self.terminal = []
        self.pruned = []

    @property
    def wrapper(self):
        return self._wrapper

    @property
    def zmq_notifier(self):
        return self._notifier

    def record_terminal(self, req_id, *, role, transfer_id, state, reason):
        self.terminal.append((req_id, role, state, reason))

    def _prune_receiver(self, req_id):
        self.pruned.append(req_id)


def _pmeta():
    return PMetadata(
        remote_addr="1.2.3.4:5000",
        uuid="uuid-1",
        specs={"k": object()},
        p_side_channel_host="1.2.3.4",
        p_side_channel_port=5001,
    )


def _make_receiver(**kw):
    mgr = _FakeMgr(**kw)
    recv = JaxTransferKVReceiver(mgr, "req-a")
    recv.init(_pmeta())
    return mgr, recv


def test_init_preconnects():
    mgr, recv = _make_receiver()
    assert mgr.wrapper.connected == ["1.2.3.4:5000"]
    assert recv.state == KVPoll.WAITING_FOR_INPUT


def test_poll_dispatches_inline_then_success():
    mgr, recv = _make_receiver()

    # First poll dispatches the pull once and stores the futures synchronously.
    assert recv.poll() == KVPoll.TRANSFERRING
    assert mgr.wrapper.calls == 1
    assert recv.result is not None

    # Next poll drives ack -> SUCCESS once every future is ready.
    assert recv.poll() == KVPoll.SUCCESS
    assert mgr.wrapper.calls == 1  # not re-dispatched
    assert mgr._notifier.sent
    assert ("req-a", "decode", KVPoll.SUCCESS, "ack_send") in mgr.terminal


def test_poll_waits_for_ready():
    mgr, recv = _make_receiver(ready=False)

    assert recv.poll() == KVPoll.TRANSFERRING
    # Futures not ready yet -> stays TRANSFERRING, no ack.
    assert recv.poll() == KVPoll.TRANSFERRING
    assert mgr._notifier.sent == []

    mgr.wrapper.ready = True
    assert recv.poll() == KVPoll.SUCCESS
    assert mgr._notifier.sent


def test_pull_exception_transitions_failed():
    mgr, recv = _make_receiver(raise_exc=True)

    assert recv.poll() == KVPoll.FAILED
    assert ("req-a", "decode", KVPoll.FAILED, "pull_init") in mgr.terminal
    assert "req-a" in mgr.pruned
    assert recv.result is None


def test_reaper_fail_then_poll_stays_failed():
    mgr, recv = _make_receiver(ready=False)

    assert recv.poll() == KVPoll.TRANSFERRING

    # Reaper times out the in-flight transfer.
    recv.fail(reason="timeout")
    assert recv.state == KVPoll.FAILED
    assert ("req-a", "decode", KVPoll.FAILED, "timeout") in mgr.terminal

    # Even if the futures become ready later, poll stays FAILED and no ack
    # is ever sent for a failed transfer.
    mgr.wrapper.ready = True
    assert recv.poll() == KVPoll.FAILED
    assert mgr._notifier.sent == []


def test_poll_never_spawns_thread():
    mgr, recv = _make_receiver()

    before = threading.active_count()
    recv.poll()
    recv.poll()
    assert threading.active_count() == before
