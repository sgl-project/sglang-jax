"""Decode pull runs off the event loop: poll() never blocks on _pull_flat.

The decode scheduler is single-threaded. ``JaxTransferKVReceiver.poll()``
used to call the blocking ``wrapper.pull`` inline, so one stuck pull froze
the whole scheduler. These tests pin the contract that poll() dispatches
the pull to a background thread and returns ``TRANSFERRING`` immediately,
and that a reaper-driven ``fail()`` during an in-flight pull is race-safe
(late results are dropped, never overwriting a terminal state).
"""
from __future__ import annotations

import threading

from sgl_jax.srt.disaggregation.base.kv_manager import KVPoll
from sgl_jax.srt.disaggregation.jax_transfer.conn import (
    JaxTransferKVReceiver,
    PMetadata,
)


class _Leaf:
    """Stand-in for a pulled jax.Array leaf; a custom object is a pytree leaf."""

    def is_ready(self) -> bool:
        return True


class _Wrapper:
    def __init__(self, *, block: bool = False, raise_exc: bool = False):
        self._block = block
        self._raise = raise_exc
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def pull(self, uuid, spec, remote_addr=None):
        self.calls += 1
        self.entered.set()
        if self._block:
            self.release.wait(timeout=5.0)
        if self._raise:
            raise RuntimeError("pull boom")
        return _Leaf()


class _Notifier:
    def __init__(self):
        self.sent = []

    def send_done(self, uuid_bytes, host, port):
        self.sent.append((uuid_bytes, host, port))


class _FakeMgr:
    def __init__(self, *, block=False, raise_exc=False):
        self._wrapper = _Wrapper(block=block, raise_exc=raise_exc)
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


def test_poll_does_not_block():
    mgr, recv = _make_receiver(block=True)

    # First poll dispatches the pull and returns immediately even though the
    # pull is still blocked inside the background thread.
    assert recv.poll() == KVPoll.TRANSFERRING
    assert mgr.wrapper.entered.wait(timeout=2.0)
    assert recv.result is None  # pull has not returned yet

    # Still TRANSFERRING while the background pull is wedged.
    assert recv.poll() == KVPoll.TRANSFERRING

    # Let the pull finish, then poll drives ack -> SUCCESS.
    mgr.wrapper.release.set()
    recv._pull_thread.join(timeout=5.0)
    assert recv.poll() == KVPoll.SUCCESS
    assert mgr._notifier.sent
    assert ("req-a", "decode", KVPoll.SUCCESS, "ack_send") in mgr.terminal


def test_pull_exception_transitions_failed():
    mgr, recv = _make_receiver(raise_exc=True)

    assert recv.poll() == KVPoll.TRANSFERRING
    recv._pull_thread.join(timeout=5.0)

    assert recv.poll() == KVPoll.FAILED
    assert ("req-a", "decode", KVPoll.FAILED, "pull_init") in mgr.terminal
    assert "req-a" in mgr.pruned
    assert recv.result is None


def test_reaper_fail_during_transfer_is_safe():
    mgr, recv = _make_receiver(block=True)

    assert recv.poll() == KVPoll.TRANSFERRING
    assert mgr.wrapper.entered.wait(timeout=2.0)

    # Reaper times out the in-flight pull while the background thread is wedged.
    recv.fail(reason="timeout")
    assert recv.state == KVPoll.FAILED
    assert ("req-a", "decode", KVPoll.FAILED, "timeout") in mgr.terminal

    # Pull eventually returns; _run_pull must drop the late result, not
    # resurrect a terminal request.
    mgr.wrapper.release.set()
    recv._pull_thread.join(timeout=5.0)
    assert recv.result is None
    assert recv.state == KVPoll.FAILED


def test_late_success_after_fail_stays_failed():
    mgr, recv = _make_receiver(block=True)

    assert recv.poll() == KVPoll.TRANSFERRING
    assert mgr.wrapper.entered.wait(timeout=2.0)
    recv.fail(reason="timeout")

    mgr.wrapper.release.set()
    recv._pull_thread.join(timeout=5.0)

    # No ack is ever sent for a failed transfer, and poll stays FAILED.
    assert recv.poll() == KVPoll.FAILED
    assert mgr._notifier.sent == []
