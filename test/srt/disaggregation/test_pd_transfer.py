"""Transfer-stack tests: JaxTransferWrapper contract, KVPoll/CommonKVManager,
and ZmqPullNotifier.

The underlying ``jax.experimental.transfer`` is shimmed out via
``sys.modules`` so the wrapper tests run on any JAX install without a TPU
backend. (On CPU-only jaxlib the real ``jax.experimental.transfer``
fails to import because ``jaxlib._jax`` lacks ``TransferConnection``.)
We only assert the wrapper-level contract; cross-pod behavior is covered
by the manual byte round-trip script. The ZmqPullNotifier tests are pure
userspace ZMQ over localhost; each allocates a fresh OS port.
"""

from __future__ import annotations

import logging
import socket
import sys
import threading
import time
import types
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.disaggregation.base.kv_manager import (
    LEGAL_TRANSITIONS,
    KVPoll,
    StateHolder,
    is_legal_transition,
)
from sgl_jax.srt.disaggregation.common.core import (
    CommonKVManager,
    TerminalTransferRecord,
)
from sgl_jax.srt.disaggregation.common.zmq_notifier import ZmqPullNotifier
from sgl_jax.srt.disaggregation.jax_transfer import wrapper as jtw_mod
from sgl_jax.srt.disaggregation.jax_transfer.wrapper import (
    JaxTransferWrapper,
    _uuid_to_int,
    get_or_create_wrapper,
)

# ==================================================================
# JaxTransferWrapper contract (from test_wrapper)
# ==================================================================


@pytest.fixture(autouse=True)
def _reset_singleton():
    jtw_mod._reset_singleton_for_test()
    yield
    jtw_mod._reset_singleton_for_test()


def _device_sharding() -> NamedSharding:
    devices = jax.local_devices()
    mesh = Mesh(np.asarray(devices).reshape(len(devices)), axis_names=("x",))
    return NamedSharding(mesh, P("x"))


def _shim_transfer_module(fake_server):
    """Inject a fake ``jax.experimental.transfer`` into ``sys.modules`` so
    the wrapper's ``from jax.experimental.transfer import
    start_transfer_server`` resolves to a mock without triggering the
    real (CPU-broken) module import.
    """

    fake_mod = types.ModuleType("jax.experimental.transfer")
    fake_mod.start_transfer_server = mock.MagicMock(return_value=fake_server)
    return mock.patch.dict(sys.modules, {"jax.experimental.transfer": fake_mod})


def test_pull_rejects_spec_without_sharding():
    wrapper = JaxTransferWrapper("127.0.0.1", 31000)
    spec_no_sharding = jax.ShapeDtypeStruct((4,), jnp.bfloat16)
    assert spec_no_sharding.sharding is None
    with pytest.raises(ValueError, match="sharding"):
        wrapper.pull("req-0", spec_no_sharding, remote_addr="1.2.3.4:1")


def test_pull_requires_remote_addr():
    fake_server = mock.MagicMock()
    fake_server.connect.return_value = mock.MagicMock()
    with (
        _shim_transfer_module(fake_server),
        mock.patch.object(jtw_mod.jax, "local_devices", return_value=[mock.MagicMock()]),
    ):
        wrapper = JaxTransferWrapper("127.0.0.1", 31001)
        wrapper.start()
    spec = jax.ShapeDtypeStruct((4,), jnp.bfloat16, sharding=_device_sharding())
    with pytest.raises(ValueError, match="remote_addr"):
        wrapper.pull("req-0", spec, remote_addr=None)


def test_start_is_idempotent_and_logs_jax_version(caplog):
    fake_server = mock.MagicMock()
    with (
        _shim_transfer_module(fake_server) as patched_modules,
        mock.patch.object(jtw_mod.jax, "local_devices", return_value=[mock.MagicMock()]),
    ):
        wrapper = JaxTransferWrapper("127.0.0.1", 31002, channel_number=2)
        with caplog.at_level(logging.INFO, logger=jtw_mod.logger.name):
            s1 = wrapper.start()
            s2 = wrapper.start()
            s3 = wrapper.start()
        mock_start = sys.modules["jax.experimental.transfer"].start_transfer_server
        del patched_modules

    assert s1 is fake_server
    assert s2 is fake_server
    assert s3 is fake_server
    assert mock_start.call_count == 1
    assert wrapper.is_started

    matching_records = [r for r in caplog.records if "JaxTransferWrapper started" in r.getMessage()]
    assert len(matching_records) == 1, [r.getMessage() for r in caplog.records]
    msg = matching_records[0].getMessage()
    assert "jax_version=" in msg
    assert "channel_number=2" in msg


def test_register_pull_before_start_raises():
    wrapper = JaxTransferWrapper("127.0.0.1", 31003)
    with pytest.raises(RuntimeError, match="start"):
        wrapper.register_pull("req-0", jnp.zeros((4,), jnp.bfloat16))


def test_pull_before_start_raises():
    wrapper = JaxTransferWrapper("127.0.0.1", 31004)
    spec = jax.ShapeDtypeStruct((4,), jnp.bfloat16, sharding=_device_sharding())
    with pytest.raises(RuntimeError, match="start"):
        wrapper.pull("req-0", spec, remote_addr="1.2.3.4:1")


def test_register_pull_keeps_data_alive_until_release():
    fake_server = mock.MagicMock()
    with (
        _shim_transfer_module(fake_server),
        mock.patch.object(jtw_mod.jax, "local_devices", return_value=[mock.MagicMock()]),
    ):
        wrapper = JaxTransferWrapper("127.0.0.1", 31005)
        wrapper.start()

    arr = jnp.arange(4, dtype=jnp.bfloat16)
    wrapper.register_pull("req-A", arr)
    assert "req-A" in wrapper._pending
    fake_server.await_pull.assert_called_once()
    called_uuid = fake_server.await_pull.call_args.args[0]
    assert called_uuid == _uuid_to_int("req-A")

    wrapper.release("req-A")
    assert "req-A" not in wrapper._pending


def test_register_pull_rejects_duplicate_uuid():
    fake_server = mock.MagicMock()
    with (
        _shim_transfer_module(fake_server),
        mock.patch.object(jtw_mod.jax, "local_devices", return_value=[mock.MagicMock()]),
    ):
        wrapper = JaxTransferWrapper("127.0.0.1", 31006)
        wrapper.start()

    arr1 = jnp.arange(4, dtype=jnp.bfloat16)
    arr2 = jnp.arange(8, dtype=jnp.bfloat16)
    wrapper.register_pull("dup", arr1)
    with pytest.raises(RuntimeError, match="already registered"):
        wrapper.register_pull("dup", arr2)
    # First registration is intact; second was rejected before touching state.
    assert wrapper._pending["dup"] is arr1
    assert fake_server.await_pull.call_count == 1

    # After release the same uuid is reusable.
    wrapper.release("dup")
    wrapper.register_pull("dup", arr2)
    assert wrapper._pending["dup"] is arr2
    assert fake_server.await_pull.call_count == 2


def test_singleton_rejects_rebinding():
    w1 = get_or_create_wrapper("10.0.0.1", 31010, channel_number=1)
    w2 = get_or_create_wrapper("10.0.0.1", 31010, channel_number=1)
    assert w1 is w2

    with pytest.raises(RuntimeError, match="rebind"):
        get_or_create_wrapper("10.0.0.2", 31010)
    with pytest.raises(RuntimeError, match="rebind"):
        get_or_create_wrapper("10.0.0.1", 31011)
    with pytest.raises(RuntimeError, match="channel_number"):
        get_or_create_wrapper("10.0.0.1", 31010, channel_number=4)


def test_uuid_str_to_int_is_stable_and_in_range():
    assert _uuid_to_int("req-0") == _uuid_to_int("req-0")
    assert _uuid_to_int("req-0") != _uuid_to_int("req-1")
    for s in ("a", "abc", "req-12345", "\U0001f9ab"):
        v = _uuid_to_int(s)
        assert 0 <= v < (1 << 32)


# ==================================================================
# KVPoll state machine (from test_transport)
# ==================================================================


def test_legal_transitions_set_matches_rfc():
    assert (
        frozenset(
            {
                (KVPoll.BOOTSTRAPPING, KVPoll.WAITING_FOR_INPUT),
                (KVPoll.WAITING_FOR_INPUT, KVPoll.TRANSFERRING),
                (KVPoll.TRANSFERRING, KVPoll.SUCCESS),
                (KVPoll.BOOTSTRAPPING, KVPoll.FAILED),
                (KVPoll.WAITING_FOR_INPUT, KVPoll.FAILED),
                (KVPoll.TRANSFERRING, KVPoll.FAILED),
            }
        )
        == LEGAL_TRANSITIONS
    )


@pytest.mark.parametrize(
    ("current", "next_state"),
    sorted(LEGAL_TRANSITIONS, key=lambda pair: (pair[0].value, pair[1].value)),
)
def test_legal_transitions_succeed(current: KVPoll, next_state: KVPoll):
    holder = StateHolder(initial=current)
    assert is_legal_transition(current, next_state) is True
    holder._transition_to(next_state)
    assert holder.state == next_state


@pytest.mark.parametrize(
    ("current", "next_state"),
    [
        (KVPoll.BOOTSTRAPPING, KVPoll.TRANSFERRING),
        (KVPoll.BOOTSTRAPPING, KVPoll.SUCCESS),
        (KVPoll.WAITING_FOR_INPUT, KVPoll.SUCCESS),
        (KVPoll.TRANSFERRING, KVPoll.WAITING_FOR_INPUT),
    ],
)
def test_representative_illegal_transitions_raise(current: KVPoll, next_state: KVPoll):
    holder = StateHolder(initial=current)
    assert is_legal_transition(current, next_state) is False
    with pytest.raises(ValueError, match="illegal KVPoll transition"):
        holder._transition_to(next_state)
    assert holder.state == current


@pytest.mark.parametrize("terminal", [KVPoll.SUCCESS, KVPoll.FAILED])
@pytest.mark.parametrize("target", [KVPoll.BOOTSTRAPPING, KVPoll.WAITING_FOR_INPUT])
def test_terminal_states_reject_outgoing_transitions(terminal: KVPoll, target: KVPoll):
    holder = StateHolder(initial=terminal)
    with pytest.raises(ValueError, match="illegal KVPoll transition"):
        holder._transition_to(target)
    assert holder.state == terminal


def test_self_loops_are_illegal():
    for state in KVPoll:
        holder = StateHolder(initial=state)
        with pytest.raises(ValueError):
            holder._transition_to(state)


# ==================================================================
# CommonKVManager (from test_transport)
# ==================================================================


class _MockParticipant:
    """Minimal duck-type stand-in for a sender or receiver."""

    def __init__(self, *, transfer_started_at: float | None = None):
        self.transfer_started_at = transfer_started_at
        self.failed = False
        self.fail_reason: str | None = None

    def fail(self, *, reason: str = "test") -> None:
        self.failed = True
        self.fail_reason = reason


class _TestKVManager(CommonKVManager):
    """Concrete subclass for testing (CommonKVManager is abstract)."""

    def create_sender(self, req_id):
        raise NotImplementedError

    def create_receiver(self, req_id):
        raise NotImplementedError


def _make_core(ack_timeout=10.0, pull_timeout=5.0):
    return _TestKVManager(
        ack_timeout_seconds=ack_timeout,
        pull_timeout_seconds=pull_timeout,
        reaper_interval_seconds=60.0,
    )


# Registry


def test_register_and_prune_sender():
    core = _make_core()
    p = _MockParticipant()
    core.register_sender("req-1", p)
    assert core.inflight_count() == (1, 0)
    core._prune_sender("req-1")
    assert core.inflight_count() == (0, 0)


def test_register_and_prune_receiver():
    core = _make_core()
    p = _MockParticipant()
    core.register_receiver("req-1", p)
    assert core.inflight_count() == (0, 1)
    core._prune_receiver("req-1")
    assert core.inflight_count() == (0, 0)


def test_duplicate_sender_raises():
    core = _make_core()
    core.register_sender("dup", _MockParticipant())
    with pytest.raises(ValueError, match="already exists"):
        core.register_sender("dup", _MockParticipant())


def test_duplicate_receiver_raises():
    core = _make_core()
    core.register_receiver("dup", _MockParticipant())
    with pytest.raises(ValueError, match="already exists"):
        core.register_receiver("dup", _MockParticipant())


def test_prune_nonexistent_is_no_op():
    core = _make_core()
    core._prune_sender("ghost")
    core._prune_receiver("ghost")


# Terminal records


def test_record_and_get_terminal():
    core = _make_core()
    core.record_terminal(
        "req-1",
        role="prefill",
        transfer_id="tid-1",
        state=KVPoll.SUCCESS,
        reason="ack",
    )
    rec = core.get_terminal_record("req-1", role="prefill")
    assert rec is not None
    assert isinstance(rec, TerminalTransferRecord)
    assert rec.req_id == "req-1"
    assert rec.state == KVPoll.SUCCESS
    assert rec.reason == "ack"


def test_clear_terminal_record():
    core = _make_core()
    core.record_terminal(
        "req-1",
        role="decode",
        transfer_id="tid-1",
        state=KVPoll.FAILED,
        reason="timeout",
    )
    core._clear_terminal_record("req-1", role="decode")
    assert core.get_terminal_record("req-1", role="decode") is None


def test_terminal_records_bounded_eviction():
    core = _make_core()
    core._max_terminal_records = 3
    for i in range(5):
        core.record_terminal(
            f"req-{i}",
            role="prefill",
            transfer_id=f"tid-{i}",
            state=KVPoll.SUCCESS,
            reason="ack",
        )
    assert core.get_terminal_record("req-0", role="prefill") is None
    assert core.get_terminal_record("req-1", role="prefill") is None
    assert core.get_terminal_record("req-2", role="prefill") is not None
    assert core.get_terminal_record("req-4", role="prefill") is not None


def test_register_sender_clears_old_terminal():
    core = _make_core()
    core.record_terminal(
        "req-1",
        role="prefill",
        transfer_id="tid-old",
        state=KVPoll.FAILED,
        reason="old",
    )
    assert core.get_terminal_record("req-1", role="prefill") is not None
    core.register_sender("req-1", _MockParticipant())
    assert core.get_terminal_record("req-1", role="prefill") is None


# Reaper


def test_reap_once_no_inflight():
    core = _make_core()
    s, r = core.reap_once(now=1000.0)
    assert s == [] and r == []


def test_reap_once_skips_not_started():
    core = _make_core(ack_timeout=10.0)
    core.register_sender("s1", _MockParticipant(transfer_started_at=None))
    s, r = core.reap_once(now=10000.0)
    assert s == []


def test_reap_once_force_fails_sender():
    core = _make_core(ack_timeout=10.0)
    p = _MockParticipant(transfer_started_at=0.0)
    core.register_sender("s1", p)
    s, r = core.reap_once(now=10.001)
    assert s == ["s1"]
    assert p.failed
    assert p.fail_reason == "timeout"


def test_reap_once_force_fails_receiver():
    core = _make_core(pull_timeout=5.0)
    p = _MockParticipant(transfer_started_at=100.0)
    core.register_receiver("r1", p)
    s, r = core.reap_once(now=105.001)
    assert r == ["r1"]
    assert p.failed


def test_reap_once_below_threshold():
    core = _make_core(ack_timeout=10.0, pull_timeout=5.0)
    core.register_sender("s1", _MockParticipant(transfer_started_at=0.0))
    core.register_receiver("r1", _MockParticipant(transfer_started_at=0.0))
    s, r = core.reap_once(now=4.0)
    assert s == [] and r == []


def test_disabled_timeouts():
    core = _make_core(ack_timeout=0.0, pull_timeout=0.0)
    core.register_sender("s1", _MockParticipant(transfer_started_at=0.0))
    s, r = core.reap_once(now=1_000_000.0)
    assert s == [] and r == []


# Inflight count


def test_inflight_count():
    core = _make_core()
    assert core.inflight_count() == (0, 0)
    core.register_sender("a", _MockParticipant())
    core.register_receiver("b", _MockParticipant())
    core.register_receiver("c", _MockParticipant())
    assert core.inflight_count() == (1, 2)


# Graceful shutdown


def test_graceful_shutdown_no_inflight():
    core = _make_core()
    aborted_s, aborted_r = core.graceful_shutdown(drain_timeout_seconds=0.05)
    assert (aborted_s, aborted_r) == (0, 0)


def test_graceful_shutdown_force_fails_stragglers():
    core = _make_core()
    p1 = _MockParticipant(transfer_started_at=0.0)
    p2 = _MockParticipant(transfer_started_at=0.0)
    core.register_sender("s1", p1)
    core.register_receiver("r1", p2)
    aborted_s, aborted_r = core.graceful_shutdown(drain_timeout_seconds=0.05)
    assert aborted_s == 1
    assert aborted_r == 1
    assert p1.failed and p1.fail_reason == "shutdown"
    assert p2.failed and p2.fail_reason == "shutdown"


# ==================================================================
# ZmqPullNotifier (from test_zmq_pull_notifier)
# ==================================================================


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
    assert len(p_notifier._callbacks) == 0


def test_single_callback_fires_with_shared_secret():
    shared_secret = "pd-secret"
    p_notifier = ZmqPullNotifier("prefill", "127.0.0.1", _free_port(), shared_secret=shared_secret)
    d_notifier = ZmqPullNotifier("decode", "127.0.0.1", _free_port(), shared_secret=shared_secret)
    p_notifier.start()
    d_notifier.start()
    try:
        received: list[bytes] = []
        event = threading.Event()

        def cb(uuid: bytes) -> None:
            received.append(uuid)
            event.set()

        p_notifier.register_callback(b"req-auth", cb)
        d_notifier.send_done(b"req-auth", "127.0.0.1", p_notifier.port)
        assert event.wait(timeout=2.0)
        assert received == [b"req-auth"]
        assert len(p_notifier._callbacks) == 0
    finally:
        d_notifier.stop()
        p_notifier.stop()


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


def test_concurrent_32_acks(p_notifier, d_notifier):
    n = 32
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

    uuids = [f"req-{i:03d}".encode() for i in range(n)]
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
    assert len(p_notifier._callbacks) == 0


def test_mark_retired_dedup(p_notifier, d_notifier):
    received: list[bytes] = []
    event = threading.Event()
    uuid = b"dedup-uuid"

    p_notifier.register_callback(uuid, lambda u: (received.append(u), event.set()))
    d_notifier.send_done(uuid, "127.0.0.1", p_notifier.port)
    assert event.wait(timeout=2.0)
    assert len(received) == 1

    p_notifier.mark_retired(uuid, state="SUCCESS", reason="ack")

    d_notifier.send_done(uuid, "127.0.0.1", p_notifier.port)
    time.sleep(0.3)
    assert len(received) == 1


def test_stop_is_idempotent_and_releases_socket():
    n = ZmqPullNotifier("prefill", "127.0.0.1", _free_port())
    n.start()
    n.stop()
    n.stop()  # second stop is a no-op
    assert not n.is_started
