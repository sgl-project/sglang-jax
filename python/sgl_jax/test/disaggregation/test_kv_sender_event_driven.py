"""Sender event-driven state-machine test.

Wires a real :class:`ZmqPullNotifier` pair (P + D in-process) to a
mocked :class:`JaxTransferWrapper`, verifying the sender transitions
from ``TRANSFERRING`` to ``SUCCESS`` only after the decoder sends the
ack — and that the wrapper's ``release`` + the manager's lifecycle
prune both fire exactly once.

Path A (D2H staging) is exercised via a real :class:`QueueHostKVPool`
(allocated on CPU jax). Path B (direct from HBM) is exercised with a
plain ``jnp.array`` payload.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec

from sgl_jax.srt.disaggregation.base.kv_manager import KVPoll
from sgl_jax.srt.disaggregation.jax_transfer.conn import JaxTransferKVManager
from sgl_jax.srt.disaggregation.jax_transfer.zmq_notifier import ZmqPullNotifier
from sgl_jax.srt.mem_cache.host_kv_pool import QueueHostKVPool


def _free_port() -> int:
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_until(predicate, timeout_s: float = 2.0) -> bool:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def _mock_wrapper():
    """Wrapper substitute that records register_pull / release without
    touching the real jax.experimental.transfer API.
    """

    w = mock.MagicMock()
    w.is_started = True
    w._pending = {}

    def register_pull(uuid, data):
        w._pending[uuid] = data

    def release(uuid):
        w._pending.pop(uuid, None)

    w.register_pull.side_effect = register_pull
    w.release.side_effect = release
    return w


def _make_host_pool(pool_size=4, max_tokens=8) -> QueueHostKVPool:
    devices = jax.local_devices()
    mesh = Mesh(np.asarray(devices[:1]).reshape(1), axis_names=("x",))
    return QueueHostKVPool(
        pool_size=pool_size,
        max_tokens_per_buffer=max_tokens,
        layer_num=1,
        kv_head_per_rank=1,
        head_dim=4,
        dtype=jnp.float32,
        mesh=mesh,
        partition_spec=PartitionSpec(),
    )


@pytest.fixture
def notifiers():
    p_port = _free_port()
    p = ZmqPullNotifier("prefill", "127.0.0.1", p_port)
    p.start()
    d = ZmqPullNotifier("decode", "127.0.0.1", _free_port())
    d.start()
    yield p, d
    d.stop()
    p.stop()


def test_path_b_sender_transitions_only_after_ack(notifiers):
    p_notifier, d_notifier = notifiers
    wrapper = _mock_wrapper()
    mgr = JaxTransferKVManager(wrapper, p_notifier)

    sender = mgr.create_sender("req-PATH-B")
    sender.init(kv_indices=None)
    payload = jnp.arange(4, dtype=jnp.float32)
    sender.attach_payload(payload, use_d2h_staging=False)
    sender.send()

    assert sender.poll() == KVPoll.TRANSFERRING
    assert "req-PATH-B" in wrapper._pending
    # No ack yet — sender stays TRANSFERRING.
    time.sleep(0.05)
    assert sender.poll() == KVPoll.TRANSFERRING

    d_notifier.send_done(b"req-PATH-B", "127.0.0.1", p_notifier.port)
    assert _wait_until(lambda: sender.poll() == KVPoll.SUCCESS)
    assert "req-PATH-B" not in wrapper._pending
    assert wrapper.release.call_count == 1
    # Sender is pruned from the manager after SUCCESS.
    assert "req-PATH-B" not in mgr._senders
    record = mgr.get_terminal_record("req-PATH-B", role="prefill")
    assert record is not None
    assert record.state == KVPoll.SUCCESS
    assert record.reason == "ack"


def test_path_a_sender_releases_host_buffer_on_ack(notifiers):
    p_notifier, d_notifier = notifiers
    wrapper = _mock_wrapper()
    pool = _make_host_pool(pool_size=2, max_tokens=8)
    initial_available = pool.available_size()
    mgr = JaxTransferKVManager(wrapper, p_notifier, host_pool=pool)

    sender = mgr.create_sender("req-PATH-A")
    sender.init(kv_indices=None)
    device_kv = jnp.ones((4, 1, 1, 4), dtype=jnp.float32)
    sender.attach_payload(device_kv, use_d2h_staging=True)
    sender.send()

    assert sender.poll() == KVPoll.TRANSFERRING
    # Host pool checked out one buffer.
    assert pool.available_size() == initial_available - 1

    d_notifier.send_done(b"req-PATH-A", "127.0.0.1", p_notifier.port)
    assert _wait_until(lambda: sender.poll() == KVPoll.SUCCESS)
    # Host buffer returned.
    assert pool.available_size() == initial_available
    assert "req-PATH-A" not in mgr._senders


def test_path_a_requires_host_pool(notifiers):
    p_notifier, _ = notifiers
    wrapper = _mock_wrapper()
    mgr = JaxTransferKVManager(wrapper, p_notifier)  # no host_pool

    sender = mgr.create_sender("req-no-pool")
    sender.init(kv_indices=None)
    payload = jnp.ones((4, 1, 1, 4), dtype=jnp.float32)
    sender.attach_payload(payload, use_d2h_staging=True)
    with pytest.raises(RuntimeError, match="host_pool"):
        sender.send()


def test_sender_fail_cancels_pending_callback(notifiers):
    p_notifier, _ = notifiers
    wrapper = _mock_wrapper()
    mgr = JaxTransferKVManager(wrapper, p_notifier)

    sender = mgr.create_sender("req-fail")
    sender.init(kv_indices=None)
    sender.attach_payload(jnp.zeros(4, dtype=jnp.float32), use_d2h_staging=False)
    sender.send()
    assert sender.poll() == KVPoll.TRANSFERRING
    assert p_notifier.pending_count() == 1

    sender.fail()
    assert sender.poll() == KVPoll.FAILED
    assert p_notifier.pending_count() == 0
    assert "req-fail" not in mgr._senders
    # release should have been called.
    assert wrapper.release.call_count == 1


def test_send_without_attach_payload_raises(notifiers):
    p_notifier, _ = notifiers
    wrapper = _mock_wrapper()
    mgr = JaxTransferKVManager(wrapper, p_notifier)

    sender = mgr.create_sender("req-bare")
    sender.init(kv_indices=None)
    with pytest.raises(RuntimeError, match="payload"):
        sender.send()


def test_attach_payload_rejects_double_attach(notifiers):
    p_notifier, _ = notifiers
    wrapper = _mock_wrapper()
    mgr = JaxTransferKVManager(wrapper, p_notifier)

    sender = mgr.create_sender("req-double")
    sender.init(kv_indices=None)
    payload = jnp.zeros(4, dtype=jnp.float32)
    sender.attach_payload(payload, use_d2h_staging=False)
    with pytest.raises(RuntimeError, match="already attached"):
        sender.attach_payload(payload, use_d2h_staging=False)


# Regression tests for the sender/ack races. Both use a wrapper
# substitute whose ``register_pull`` blocks on a barrier so we can
# deterministically drive the listener-vs-main race.


def _barrier_wrapper(barrier_event: threading.Event):
    w = mock.MagicMock()
    w.is_started = True
    w._pending = {}

    def register_pull(uuid, data):
        # Hold for a moment so the test thread can fire the ack
        # before ``send`` returns and transitions the sender to
        # TRANSFERRING.
        barrier_event.wait(timeout=5.0)
        w._pending[uuid] = data

    def release(uuid):
        w._pending.pop(uuid, None)

    w.register_pull.side_effect = register_pull
    w.release.side_effect = release
    return w


def test_send_ack_race_safe(notifiers):
    """An ack arriving mid-handoff must not wedge the sender.

    ``send()`` holds ``_state_lock`` around callback registration,
    producer handoff, and the state transition. The listener's
    ``_on_ack`` blocks on that lock until ``send()`` finishes, then
    transitions the sender to SUCCESS.
    """

    p_notifier, d_notifier = notifiers
    barrier = threading.Event()
    wrapper = _barrier_wrapper(barrier)
    mgr = JaxTransferKVManager(wrapper, p_notifier)

    sender = mgr.create_sender("req-race")
    sender.init(kv_indices=None)
    sender.attach_payload(jnp.zeros(4, dtype=jnp.float32), use_d2h_staging=False)

    # Run ``send`` on a background thread so the main thread can fire
    # the ack while ``send`` is blocked inside ``producer_handoff``.
    send_done = threading.Event()

    def run_send():
        sender.send()
        send_done.set()

    send_thread = threading.Thread(target=run_send)
    send_thread.start()

    # Wait for ``send`` to be inside ``register_pull`` (callback is
    # registered first under the state lock; producer_handoff then
    # blocks on the barrier).
    assert _wait_until(lambda: p_notifier.pending_count() == 1)
    # Fire the ack now, before ``send`` releases the state lock.
    d_notifier.send_done(b"req-race", "127.0.0.1", p_notifier.port)
    # Listener thread pops the callback and tries to acquire
    # ``_state_lock``; it must block because ``send`` is still
    # holding it.
    assert _wait_until(lambda: p_notifier.pending_count() == 0)

    # Release ``send``'s barrier; it finishes the transition to
    # TRANSFERRING, releases the lock, and ``_on_ack`` proceeds.
    barrier.set()
    assert send_done.wait(timeout=3.0)
    assert _wait_until(lambda: sender.poll() == KVPoll.SUCCESS)
    assert wrapper.release.call_count == 1
    assert "req-race" not in mgr._senders


def test_fail_owns_cleanup_when_callback_still_registered(notifiers):
    """If ``fail()`` wins the callback race, it owns cleanup.

    When the listener has not popped the callback yet,
    ``unregister_callback()`` returns it and ``fail()`` must run
    ``on_done()`` exactly once.
    """

    p_notifier, _ = notifiers
    wrapper = _mock_wrapper()
    pool = _make_host_pool(pool_size=2, max_tokens=8)
    mgr = JaxTransferKVManager(wrapper, p_notifier, host_pool=pool)

    sender = mgr.create_sender("req-fail-owns-cleanup")
    sender.init(kv_indices=None)
    device_kv = jnp.ones((4, 1, 1, 4), dtype=jnp.float32)
    sender.attach_payload(device_kv, use_d2h_staging=True)
    sender.send()
    in_use_initial = pool.total_size() - pool.available_size()
    assert in_use_initial == 1

    sender.fail()
    # ``fail`` claimed cleanup → buffer returned, wrapper released.
    assert sender.poll() == KVPoll.FAILED
    assert pool.available_size() == pool.total_size()
    assert wrapper.release.call_count == 1
    assert "req-C2a" not in mgr._senders


def test_fail_after_listener_popped_skips_cleanup(notifiers):
    """If the listener already owns the callback, ``fail()`` skips cleanup.

    Once the listener pops the callback, the in-flight ``_on_ack`` path
    must be the only owner of ``on_done()``.
    """

    p_notifier, _ = notifiers
    wrapper = _mock_wrapper()
    pool = _make_host_pool(pool_size=2, max_tokens=8)
    mgr = JaxTransferKVManager(wrapper, p_notifier, host_pool=pool)

    sender = mgr.create_sender("req-ack-owns-cleanup")
    sender.init(kv_indices=None)
    device_kv = jnp.ones((4, 1, 1, 4), dtype=jnp.float32)
    sender.attach_payload(device_kv, use_d2h_staging=True)
    sender.send()

    # Listener thread pops the callback (simulated by manual pop —
    # same dict op the listener uses internally).
    cb = p_notifier.unregister_callback(b"req-ack-owns-cleanup")
    assert cb is not None

    # ``fail`` runs while the popped callback is still in flight.
    sender.fail()
    assert sender.poll() == KVPoll.FAILED
    # Buffer is NOT yet returned because ``fail`` ceded cleanup.
    assert pool.available_size() == pool.total_size() - 1
    # ``wrapper.release`` has NOT been called by fail either.
    assert wrapper.release.call_count == 0

    # Now run the popped callback as the listener would have done.
    cb(b"req-ack-owns-cleanup")
    # ``_on_ack`` runs cleanup exactly once.
    assert pool.available_size() == pool.total_size()
    assert wrapper.release.call_count == 1


def test_late_ack_from_old_transfer_id_does_not_complete_reused_req(notifiers):
    """Lifecycle regression: if a logical request id is reused for a
    new transfer attempt, a stale ack for the OLD transfer id must not
    complete the new sender.
    """

    p_notifier, d_notifier = notifiers
    wrapper = _mock_wrapper()
    mgr = JaxTransferKVManager(wrapper, p_notifier)

    sender1 = mgr.create_sender("req-reuse")
    sender1.init(kv_indices=None, transfer_id="req-reuse#old")
    sender1.attach_payload(jnp.arange(4, dtype=jnp.float32), use_d2h_staging=False)
    sender1.send()
    sender1.fail(reason="test")

    sender2 = mgr.create_sender("req-reuse")
    sender2.init(kv_indices=None, transfer_id="req-reuse#new")
    sender2.attach_payload(
        jnp.arange(4, dtype=jnp.float32) + 100,
        use_d2h_staging=False,
    )
    sender2.send()

    # A stale ack for the OLD transfer attempt must not terminate the
    # current sender.
    d_notifier.send_done(b"req-reuse#old", "127.0.0.1", p_notifier.port)
    time.sleep(0.05)
    assert sender2.poll() == KVPoll.TRANSFERRING
    assert "req-reuse#new" in wrapper._pending

    # The matching ack still completes the current transfer.
    d_notifier.send_done(b"req-reuse#new", "127.0.0.1", p_notifier.port)
    assert _wait_until(lambda: sender2.poll() == KVPoll.SUCCESS)
    assert "req-reuse#new" not in wrapper._pending


def test_late_ack_after_success_is_classified_as_retired(notifiers, caplog):
    p_notifier, d_notifier = notifiers
    wrapper = _mock_wrapper()
    mgr = JaxTransferKVManager(wrapper, p_notifier)

    sender = mgr.create_sender("req-late-success")
    sender.init(kv_indices=None)
    sender.attach_payload(jnp.arange(4, dtype=jnp.float32), use_d2h_staging=False)
    sender.send()

    d_notifier.send_done(b"req-late-success", "127.0.0.1", p_notifier.port)
    assert _wait_until(lambda: sender.poll() == KVPoll.SUCCESS)

    with caplog.at_level(
        logging.INFO,
        logger="sgl_jax.srt.disaggregation.jax_transfer.zmq_notifier",
    ):
        d_notifier.send_done(b"req-late-success", "127.0.0.1", p_notifier.port)
        assert _wait_until(
            lambda: any(
                "retired transfer" in rec.getMessage() and "req-late-success" in rec.getMessage()
                for rec in caplog.records
            )
        )

    assert not any(
        "no registered callback" in rec.getMessage() and "req-late-success" in rec.getMessage()
        for rec in caplog.records
    )


def test_late_ack_after_fail_is_classified_as_retired(notifiers, caplog):
    p_notifier, d_notifier = notifiers
    wrapper = _mock_wrapper()
    mgr = JaxTransferKVManager(wrapper, p_notifier)

    sender = mgr.create_sender("req-late-fail")
    sender.init(kv_indices=None)
    sender.attach_payload(jnp.arange(4, dtype=jnp.float32), use_d2h_staging=False)
    sender.send()
    sender.fail(reason="test")

    with caplog.at_level(
        logging.INFO,
        logger="sgl_jax.srt.disaggregation.jax_transfer.zmq_notifier",
    ):
        d_notifier.send_done(b"req-late-fail", "127.0.0.1", p_notifier.port)
        assert _wait_until(
            lambda: any(
                "retired transfer" in rec.getMessage() and "req-late-fail" in rec.getMessage()
                for rec in caplog.records
            )
        )

    assert not any(
        "no registered callback" in rec.getMessage() and "req-late-fail" in rec.getMessage()
        for rec in caplog.records
    )


def test_new_sender_attempt_clears_old_terminal_record(notifiers):
    p_notifier, d_notifier = notifiers
    wrapper = _mock_wrapper()
    mgr = JaxTransferKVManager(wrapper, p_notifier)

    sender1 = mgr.create_sender("req-retry")
    sender1.init(kv_indices=None)
    sender1.attach_payload(jnp.arange(4, dtype=jnp.float32), use_d2h_staging=False)
    sender1.send()
    d_notifier.send_done(b"req-retry", "127.0.0.1", p_notifier.port)
    assert _wait_until(lambda: sender1.poll() == KVPoll.SUCCESS)
    assert mgr.get_terminal_record("req-retry", role="prefill") is not None

    sender2 = mgr.create_sender("req-retry")
    assert mgr.get_terminal_record("req-retry", role="prefill") is None
    sender2.init(kv_indices=None)


def test_sender_abort_failure_exception_and_clear(notifiers):
    p_notifier, _ = notifiers
    wrapper = _mock_wrapper()
    mgr = JaxTransferKVManager(wrapper, p_notifier)

    sender = mgr.create_sender("req-abort")
    sender.init(kv_indices=None)
    sender.attach_payload(jnp.arange(4, dtype=jnp.float32), use_d2h_staging=False)
    sender.send()
    sender.abort()

    assert sender.poll() == KVPoll.FAILED
    record = mgr.get_terminal_record("req-abort", role="prefill")
    assert record is not None
    assert record.state == KVPoll.FAILED
    assert record.reason == "abort"

    with pytest.raises(RuntimeError, match="abort"):
        sender.failure_exception()

    sender.clear()
    assert mgr.get_terminal_record("req-abort", role="prefill") is None
    sender.clear()
