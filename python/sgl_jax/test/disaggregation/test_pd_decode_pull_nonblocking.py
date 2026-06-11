"""Decode pull never blocks the single-threaded decode event loop.

On TPU ``JaxTransferWrapper.pull`` (``link.pull``) is synchronous — it
blocks until the transfer completes. Dispatching it inline inside
``poll()`` would freeze the decode event loop, so the blocking pull is
handed to a single long-lived background worker owned by the manager.
These tests pin that contract:

  * ``init()`` only records metadata and arms the receiver. It does NOT
    connect — the transfer link is a native handle that must be created and
    used on the same thread, so the worker connects lazily inside its own
    pull (see ``test_init_does_not_connect`` / ``test_worker_connects_then_pulls``).
  * the first ``poll()`` transitions WAITING_FOR_INPUT -> TRANSFERRING and
    *enqueues* the receiver — it does NOT pull. ``poll()`` spawns no thread
    and stays non-blocking.
  * the background worker (driven explicitly in these tests via
    ``_run_pull``) performs the blocking pull and stores the results.
  * a later ``poll()`` drives ack -> SUCCESS once every leaf is ready.
  * a reaper ``fail()`` that wins before the worker stores its results
    keeps the terminal state — late results are dropped, never resurrected.
  * the manager owns exactly one persistent worker thread that drains the
    queue.
"""
from __future__ import annotations

import threading

from sgl_jax.srt.disaggregation.base.kv_manager import KVPoll
from sgl_jax.srt.disaggregation.jax_transfer.conn import (
    JaxTransferKVManager,
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
        # Mirror the real wrapper: ``pull`` lazily connects (and caches) the
        # link on the calling thread before fetching.
        if remote_addr is not None and remote_addr not in self.connected:
            self.connected.append(remote_addr)
        if self._raise:
            raise RuntimeError("pull boom")
        return _Leaf(self)


class _Notifier:
    def __init__(self):
        self.sent = []

    def send_done(self, uuid_bytes, host, port):
        self.sent.append((uuid_bytes, host, port))


class _FakeMgr:
    """Stands in for the manager. ``enqueue_pull`` records the receiver but
    does NOT run it — tests drive the worker explicitly via ``_run_pull`` so
    the ordering between worker completion and reaper ``fail()`` is fully
    controllable.
    """

    def __init__(self, *, raise_exc=False, ready=True):
        self._wrapper = _Wrapper(raise_exc=raise_exc, ready=ready)
        self._notifier = _Notifier()
        self.terminal = []
        self.pruned = []
        self.enqueued: list[JaxTransferKVReceiver] = []

    @property
    def wrapper(self):
        return self._wrapper

    @property
    def zmq_notifier(self):
        return self._notifier

    def enqueue_pull(self, receiver):
        self.enqueued.append(receiver)

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


def test_init_does_not_connect():
    mgr, recv = _make_receiver()
    # init must NOT touch the link: connect+pull stay on the worker thread.
    assert mgr.wrapper.connected == []
    assert recv.state == KVPoll.WAITING_FOR_INPUT


def test_poll_enqueues_without_pulling():
    mgr, recv = _make_receiver()

    # First poll transitions to TRANSFERRING and enqueues — it must NOT pull.
    assert recv.poll() == KVPoll.TRANSFERRING
    assert mgr.enqueued == [recv]
    assert mgr.wrapper.calls == 0
    assert recv.result is None

    # Polling again before the worker runs stays TRANSFERRING, no ack.
    assert recv.poll() == KVPoll.TRANSFERRING
    assert mgr._notifier.sent == []


def test_worker_pull_then_success():
    mgr, recv = _make_receiver()
    assert recv.poll() == KVPoll.TRANSFERRING
    # Enqueue does not connect; the link is created on the worker thread.
    assert mgr.wrapper.connected == []

    # The background worker performs the blocking pull and stores results.
    recv._run_pull()
    assert mgr.wrapper.calls == 1
    assert mgr.wrapper.connected == ["1.2.3.4:5000"]
    assert recv.result is not None

    # Next poll drives ack -> SUCCESS once every leaf is ready.
    assert recv.poll() == KVPoll.SUCCESS
    assert mgr.wrapper.calls == 1  # not re-pulled
    assert mgr._notifier.sent
    assert ("req-a", "decode", KVPoll.SUCCESS, "ack_send") in mgr.terminal


def test_poll_waits_for_ready():
    mgr, recv = _make_receiver(ready=False)

    assert recv.poll() == KVPoll.TRANSFERRING
    recv._run_pull()
    # Results stored but leaves not ready yet -> stays TRANSFERRING, no ack.
    assert recv.poll() == KVPoll.TRANSFERRING
    assert mgr._notifier.sent == []

    mgr.wrapper.ready = True
    assert recv.poll() == KVPoll.SUCCESS
    assert mgr._notifier.sent


def test_pull_exception_transitions_failed():
    mgr, recv = _make_receiver(raise_exc=True)

    assert recv.poll() == KVPoll.TRANSFERRING
    # The worker hits the exception and drives the terminal transition.
    recv._run_pull()
    assert recv.state == KVPoll.FAILED
    assert ("req-a", "decode", KVPoll.FAILED, "pull_init") in mgr.terminal
    assert "req-a" in mgr.pruned
    assert recv.result is None

    # A subsequent poll stays FAILED.
    assert recv.poll() == KVPoll.FAILED


def test_reaper_fail_then_worker_drops_results():
    mgr, recv = _make_receiver()

    assert recv.poll() == KVPoll.TRANSFERRING

    # Reaper times out the in-flight transfer before the worker finishes.
    recv.fail(reason="timeout")
    assert recv.state == KVPoll.FAILED
    assert ("req-a", "decode", KVPoll.FAILED, "timeout") in mgr.terminal

    # The worker completes late: results must be dropped, not stored, and the
    # terminal state must not be resurrected.
    recv._run_pull()
    assert recv.result is None
    assert recv.state == KVPoll.FAILED

    # Even if leaves are ready, poll stays FAILED and no ack is ever sent.
    mgr.wrapper.ready = True
    assert recv.poll() == KVPoll.FAILED
    assert mgr._notifier.sent == []


def test_poll_never_spawns_thread():
    mgr, recv = _make_receiver()

    before = threading.active_count()
    recv.poll()
    recv.poll()
    assert threading.active_count() == before


def test_manager_owns_one_persistent_pull_worker():
    """The manager starts exactly one long-lived worker that drains the
    queue and runs each receiver's blocking pull off the event loop."""

    mgr = JaxTransferKVManager(wrapper=object(), zmq_notifier=object())

    worker_threads = [
        t for t in threading.enumerate() if t.name == "jax-kv-pull-worker"
    ]
    assert len(worker_threads) == 1
    assert worker_threads[0].daemon

    ran = threading.Event()

    class _Stub:
        def _run_pull(self):
            ran.set()

    mgr.enqueue_pull(_Stub())
    assert ran.wait(timeout=5.0)
