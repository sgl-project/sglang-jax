"""Tests for JaxTransferKVManager, sender event-driven lifecycle, and orphan reaper.

Sender tests wire a real :class:`ZmqPullNotifier` pair (P + D in-process) to a
mocked :class:`JaxTransferWrapper`, verifying the sender transitions
from ``TRANSFERRING`` to ``SUCCESS`` only after the decoder sends the
ack — and that the wrapper's ``release`` + the manager's lifecycle
prune both fire exactly once.

Reaper tests drive ``JaxTransferKVManager.reap_once(now)`` with a hand-stepped
clock so the tests stay deterministic.
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
from sgl_jax.srt.disaggregation.common.zmq_notifier import ZmqPullNotifier
from sgl_jax.srt.disaggregation.jax_transfer.conn import JaxTransferKVManager
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


# ------------------------------------------------------------------
# Sender event-driven lifecycle (from test_kv_sender_event_driven)
# ------------------------------------------------------------------


def test_path_b_sender_transitions_only_after_ack(notifiers):
    p_notifier, d_notifier = notifiers
    wrapper = _mock_wrapper()
    mgr = JaxTransferKVManager(wrapper, p_notifier)

    sender = mgr.create_sender("req-PATH-B")
    sender.init(kv_indices=None)
    payload = {"kv": jnp.arange(4, dtype=jnp.float32)}
    sender.attach_payload(payload, use_d2h_staging=False)
    sender.send()

    assert sender.poll() == KVPoll.TRANSFERRING
    assert "req-PATH-B:kv" in wrapper._pending
    # No ack yet — sender stays TRANSFERRING.
    time.sleep(0.05)
    assert sender.poll() == KVPoll.TRANSFERRING

    d_notifier.send_done(b"req-PATH-B", "127.0.0.1", p_notifier.port)
    assert _wait_until(lambda: sender.poll() == KVPoll.SUCCESS)
    assert "req-PATH-B:kv" not in wrapper._pending
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
    sender.attach_payload({"kv": device_kv}, use_d2h_staging=True)
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
    sender.attach_payload({"kv": payload}, use_d2h_staging=True)
    with pytest.raises(RuntimeError, match="host_pool"):
        sender.send()


def test_sender_fail_cancels_pending_callback(notifiers):
    p_notifier, _ = notifiers
    wrapper = _mock_wrapper()
    mgr = JaxTransferKVManager(wrapper, p_notifier)

    sender = mgr.create_sender("req-fail")
    sender.init(kv_indices=None)
    sender.attach_payload({"kv": jnp.zeros(4, dtype=jnp.float32)}, use_d2h_staging=False)
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
    payload = {"kv": jnp.zeros(4, dtype=jnp.float32)}
    sender.attach_payload(payload, use_d2h_staging=False)
    with pytest.raises(RuntimeError, match="already attached"):
        sender.attach_payload(payload, use_d2h_staging=False)


# Regression tests for the sender/ack races.


def _barrier_wrapper(barrier_event: threading.Event):
    w = mock.MagicMock()
    w.is_started = True
    w._pending = {}

    def register_pull(uuid, data):
        barrier_event.wait(timeout=5.0)
        w._pending[uuid] = data

    def release(uuid):
        w._pending.pop(uuid, None)

    w.register_pull.side_effect = register_pull
    w.release.side_effect = release
    return w


def test_send_ack_race_safe(notifiers):
    p_notifier, d_notifier = notifiers
    barrier = threading.Event()
    wrapper = _barrier_wrapper(barrier)
    mgr = JaxTransferKVManager(wrapper, p_notifier)

    sender = mgr.create_sender("req-race")
    sender.init(kv_indices=None)
    sender.attach_payload({"kv": jnp.zeros(4, dtype=jnp.float32)}, use_d2h_staging=False)

    send_done = threading.Event()

    def run_send():
        sender.send()
        send_done.set()

    send_thread = threading.Thread(target=run_send)
    send_thread.start()

    assert _wait_until(lambda: p_notifier.pending_count() == 1)
    d_notifier.send_done(b"req-race", "127.0.0.1", p_notifier.port)
    assert _wait_until(lambda: p_notifier.pending_count() == 0)

    barrier.set()
    assert send_done.wait(timeout=3.0)
    assert _wait_until(lambda: sender.poll() == KVPoll.SUCCESS)
    assert wrapper.release.call_count == 1
    assert "req-race" not in mgr._senders


def test_fail_owns_cleanup_when_callback_still_registered(notifiers):
    p_notifier, _ = notifiers
    wrapper = _mock_wrapper()
    pool = _make_host_pool(pool_size=2, max_tokens=8)
    mgr = JaxTransferKVManager(wrapper, p_notifier, host_pool=pool)

    sender = mgr.create_sender("req-fail-owns-cleanup")
    sender.init(kv_indices=None)
    device_kv = jnp.ones((4, 1, 1, 4), dtype=jnp.float32)
    sender.attach_payload({"kv": device_kv}, use_d2h_staging=True)
    sender.send()
    in_use_initial = pool.total_size() - pool.available_size()
    assert in_use_initial == 1

    sender.fail()
    assert sender.poll() == KVPoll.FAILED
    assert pool.available_size() == pool.total_size()
    assert wrapper.release.call_count == 1
    assert "req-fail-owns-cleanup" not in mgr._senders


def test_fail_after_listener_popped_skips_cleanup(notifiers):
    p_notifier, _ = notifiers
    wrapper = _mock_wrapper()
    pool = _make_host_pool(pool_size=2, max_tokens=8)
    mgr = JaxTransferKVManager(wrapper, p_notifier, host_pool=pool)

    sender = mgr.create_sender("req-ack-owns-cleanup")
    sender.init(kv_indices=None)
    device_kv = jnp.ones((4, 1, 1, 4), dtype=jnp.float32)
    sender.attach_payload({"kv": device_kv}, use_d2h_staging=True)
    sender.send()

    cb = p_notifier.unregister_callback(b"req-ack-owns-cleanup")
    assert cb is not None

    sender.fail()
    assert sender.poll() == KVPoll.FAILED
    assert pool.available_size() == pool.total_size() - 1
    assert wrapper.release.call_count == 0

    cb(b"req-ack-owns-cleanup")
    assert pool.available_size() == pool.total_size()
    assert wrapper.release.call_count == 1


def test_late_ack_from_old_transfer_id_does_not_complete_reused_req(notifiers):
    p_notifier, d_notifier = notifiers
    wrapper = _mock_wrapper()
    mgr = JaxTransferKVManager(wrapper, p_notifier)

    sender1 = mgr.create_sender("req-reuse")
    sender1.init(kv_indices=None, transfer_id="req-reuse#old")
    sender1.attach_payload({"kv": jnp.arange(4, dtype=jnp.float32)}, use_d2h_staging=False)
    sender1.send()
    sender1.fail(reason="test")

    sender2 = mgr.create_sender("req-reuse")
    sender2.init(kv_indices=None, transfer_id="req-reuse#new")
    sender2.attach_payload(
        {"kv": jnp.arange(4, dtype=jnp.float32) + 100},
        use_d2h_staging=False,
    )
    sender2.send()

    d_notifier.send_done(b"req-reuse#old", "127.0.0.1", p_notifier.port)
    time.sleep(0.05)
    assert sender2.poll() == KVPoll.TRANSFERRING
    assert "req-reuse#new:kv" in wrapper._pending

    d_notifier.send_done(b"req-reuse#new", "127.0.0.1", p_notifier.port)
    assert _wait_until(lambda: sender2.poll() == KVPoll.SUCCESS)
    assert "req-reuse#new:kv" not in wrapper._pending


def test_late_ack_after_success_is_classified_as_retired(notifiers, caplog):
    p_notifier, d_notifier = notifiers
    wrapper = _mock_wrapper()
    mgr = JaxTransferKVManager(wrapper, p_notifier)

    sender = mgr.create_sender("req-late-success")
    sender.init(kv_indices=None)
    sender.attach_payload({"kv": jnp.arange(4, dtype=jnp.float32)}, use_d2h_staging=False)
    sender.send()

    d_notifier.send_done(b"req-late-success", "127.0.0.1", p_notifier.port)
    assert _wait_until(lambda: sender.poll() == KVPoll.SUCCESS)

    with caplog.at_level(
        logging.INFO,
        logger="sgl_jax.srt.disaggregation.common.zmq_notifier",
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
    sender.attach_payload({"kv": jnp.arange(4, dtype=jnp.float32)}, use_d2h_staging=False)
    sender.send()
    sender.fail(reason="test")

    with caplog.at_level(
        logging.INFO,
        logger="sgl_jax.srt.disaggregation.common.zmq_notifier",
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
    sender1.attach_payload({"kv": jnp.arange(4, dtype=jnp.float32)}, use_d2h_staging=False)
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
    sender.attach_payload({"kv": jnp.arange(4, dtype=jnp.float32)}, use_d2h_staging=False)
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


def test_multi_entry_payload_registers_all_sub_uuids(notifiers):
    p_notifier, d_notifier = notifiers
    wrapper = _mock_wrapper()
    mgr = JaxTransferKVManager(wrapper, p_notifier)

    sender = mgr.create_sender("req-multi")
    sender.init(kv_indices=None)
    sender.attach_payload(
        {
            "kv": jnp.zeros(4, dtype=jnp.float32),
            "temporal": jnp.ones(8, dtype=jnp.float32),
        },
        use_d2h_staging=False,
    )
    sender.send()

    assert sender.poll() == KVPoll.TRANSFERRING
    assert "req-multi:kv" in wrapper._pending
    assert "req-multi:temporal" in wrapper._pending
    assert wrapper.register_pull.call_count == 2

    d_notifier.send_done(b"req-multi", "127.0.0.1", p_notifier.port)
    assert _wait_until(lambda: sender.poll() == KVPoll.SUCCESS)
    assert "req-multi:kv" not in wrapper._pending
    assert "req-multi:temporal" not in wrapper._pending
    assert wrapper.release.call_count == 2


def test_multi_entry_fail_releases_all_sub_uuids(notifiers):
    p_notifier, _ = notifiers
    wrapper = _mock_wrapper()
    mgr = JaxTransferKVManager(wrapper, p_notifier)

    sender = mgr.create_sender("req-multi-fail")
    sender.init(kv_indices=None)
    sender.attach_payload(
        {
            "kv": jnp.zeros(4, dtype=jnp.float32),
            "conv": jnp.ones(2, dtype=jnp.float32),
            "temporal": jnp.ones(8, dtype=jnp.float32),
        },
        use_d2h_staging=False,
    )
    sender.send()

    assert sender.poll() == KVPoll.TRANSFERRING
    assert wrapper.register_pull.call_count == 3

    sender.fail()
    assert sender.poll() == KVPoll.FAILED
    assert wrapper.release.call_count == 3
    assert len(wrapper._pending) == 0


def test_empty_payload_rejected(notifiers):
    p_notifier, _ = notifiers
    wrapper = _mock_wrapper()
    mgr = JaxTransferKVManager(wrapper, p_notifier)

    sender = mgr.create_sender("req-empty")
    sender.init(kv_indices=None)
    with pytest.raises(ValueError, match="non-empty"):
        sender.attach_payload({}, use_d2h_staging=False)


def test_entry_name_with_colon_rejected(notifiers):
    p_notifier, _ = notifiers
    wrapper = _mock_wrapper()
    mgr = JaxTransferKVManager(wrapper, p_notifier)

    sender = mgr.create_sender("req-colon")
    sender.init(kv_indices=None)
    sender.attach_payload(
        {"bad:name": jnp.zeros(4, dtype=jnp.float32)},
        use_d2h_staging=False,
    )
    with pytest.raises(ValueError, match="must not contain ':'"):
        sender.send()


# ------------------------------------------------------------------
# Orphan reaper and per-phase timeout (from test_orphan_reaper)
# ------------------------------------------------------------------


def _make_mgr(ack_timeout=10.0, pull_timeout=5.0):
    wrapper = mock.MagicMock()
    notifier = mock.MagicMock()
    notifier.unregister_callback.return_value = None
    return JaxTransferKVManager(
        wrapper,
        notifier,
        host_pool=None,
        ack_timeout_seconds=ack_timeout,
        pull_timeout_seconds=pull_timeout,
        reaper_interval_seconds=60.0,
    )


def test_reap_once_does_nothing_when_no_inflight():
    mgr = _make_mgr()
    s, r = mgr.reap_once(now=1000.0)
    assert s == [] and r == []


def test_reap_once_skips_senders_in_bootstrapping():
    mgr = _make_mgr()
    sender = mgr.create_sender("req-1")
    s, r = mgr.reap_once(now=10000.0)
    assert s == [] and r == []
    assert sender.state == KVPoll.BOOTSTRAPPING


def test_reap_once_force_fails_orphan_sender(monkeypatch):
    mgr = _make_mgr(ack_timeout=10.0)
    sender = mgr.create_sender("req-orphan")
    sender._transition_to(KVPoll.WAITING_FOR_INPUT)
    sender._transition_to(KVPoll.TRANSFERRING)
    sender._transfer_started_at = 0.0
    sender._status = mock.MagicMock()

    s, r = mgr.reap_once(now=10.0 + 0.001)
    assert s == ["req-orphan"]
    assert r == []
    assert sender.state == KVPoll.FAILED
    record = mgr.get_terminal_record("req-orphan", role="prefill")
    assert record is not None
    assert record.state == KVPoll.FAILED
    assert record.reason == "timeout"


def test_reap_once_force_fails_orphan_receiver():
    mgr = _make_mgr(pull_timeout=5.0)
    receiver = mgr.create_receiver("req-stuck")
    receiver._transition_to(KVPoll.WAITING_FOR_INPUT)
    receiver._transition_to(KVPoll.TRANSFERRING)
    receiver._transfer_started_at = 100.0

    s, r = mgr.reap_once(now=105.001)
    assert s == []
    assert r == ["req-stuck"]
    assert receiver.state == KVPoll.FAILED
    record = mgr.get_terminal_record("req-stuck", role="decode")
    assert record is not None
    assert record.state == KVPoll.FAILED
    assert record.reason == "timeout"


def test_reap_once_below_threshold_is_no_op():
    mgr = _make_mgr(ack_timeout=10.0, pull_timeout=5.0)
    sender = mgr.create_sender("s1")
    sender._transition_to(KVPoll.WAITING_FOR_INPUT)
    sender._transition_to(KVPoll.TRANSFERRING)
    sender._transfer_started_at = 0.0
    sender._status = mock.MagicMock()

    receiver = mgr.create_receiver("r1")
    receiver._transition_to(KVPoll.WAITING_FOR_INPUT)
    receiver._transition_to(KVPoll.TRANSFERRING)
    receiver._transfer_started_at = 0.0

    s, r = mgr.reap_once(now=4.0)
    assert s == []
    assert r == []


def test_reaper_disabled_timeouts():
    mgr = _make_mgr(ack_timeout=0.0, pull_timeout=0.0)
    sender = mgr.create_sender("s1")
    sender._transition_to(KVPoll.WAITING_FOR_INPUT)
    sender._transition_to(KVPoll.TRANSFERRING)
    sender._transfer_started_at = 0.0

    s, r = mgr.reap_once(now=1_000_000.0)
    assert s == [] and r == []


def test_inflight_count_reflects_active():
    mgr = _make_mgr()
    assert mgr.inflight_count() == (0, 0)
    mgr.create_sender("a")
    mgr.create_receiver("b")
    mgr.create_receiver("c")
    assert mgr.inflight_count() == (1, 2)


def test_graceful_shutdown_drains_when_clean():
    mgr = _make_mgr()
    aborted_s, aborted_r = mgr.graceful_shutdown(drain_timeout_seconds=0.1)
    assert (aborted_s, aborted_r) == (0, 0)


def test_graceful_shutdown_force_fails_after_timeout():
    mgr = _make_mgr()
    sender = mgr.create_sender("s1")
    sender._transition_to(KVPoll.WAITING_FOR_INPUT)
    sender._transition_to(KVPoll.TRANSFERRING)
    sender._transfer_started_at = 0.0
    sender._status = mock.MagicMock()

    aborted_s, aborted_r = mgr.graceful_shutdown(drain_timeout_seconds=0.05)
    assert aborted_s == 1
    assert sender.state == KVPoll.FAILED
    record = mgr.get_terminal_record("s1", role="prefill")
    assert record is not None
    assert record.state == KVPoll.FAILED
    assert record.reason == "shutdown"


def test_receiver_fail_is_lock_protected_against_concurrent_poll():
    mgr = _make_mgr()
    receiver = mgr.create_receiver("r-race")
    receiver._metadata = mock.MagicMock()
    receiver._metadata.uuid = "r-race"
    receiver._metadata.p_side_channel_host = "127.0.0.1"
    receiver._metadata.p_side_channel_port = 9601
    receiver._metadata.remote_addr = "127.0.0.1:30001"
    receiver._metadata.spec = mock.MagicMock()

    receiver._transition_to(KVPoll.WAITING_FOR_INPUT)
    receiver._transition_to(KVPoll.TRANSFERRING)
    receiver._transfer_started_at = 0.0
    fake_arr = mock.MagicMock()
    fake_arr.is_ready.return_value = True
    receiver._result = fake_arr

    receiver.fail(reason="timeout")
    assert receiver.state == KVPoll.FAILED

    final = receiver.poll()
    assert final == KVPoll.FAILED


def test_receiver_abort_failure_exception_and_clear():
    mgr = _make_mgr()
    receiver = mgr.create_receiver("r-abort")

    receiver.abort()
    assert receiver.state == KVPoll.FAILED
    record = mgr.get_terminal_record("r-abort", role="decode")
    assert record is not None
    assert record.state == KVPoll.FAILED
    assert record.reason == "abort"

    with pytest.raises(RuntimeError, match="abort"):
        receiver.failure_exception()

    receiver.clear()
    assert mgr.get_terminal_record("r-abort", role="decode") is None
    receiver.clear()
