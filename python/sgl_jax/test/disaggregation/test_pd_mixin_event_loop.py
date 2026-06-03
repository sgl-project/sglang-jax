"""Drive the PD Decode Mixin through one event-loop tick with a mock
scheduler. These tests cover Stage 2 review C1/C3/C4/I6/I7: PD reqs
must be removed from ``waiting_queue`` after intake (so they don't
get treated as normal prefill), KV-pull failures must release the
``req_to_token_pool`` slot without calling ``cache_finished_req``,
and resource leaks on bootstrap-lookup / receiver-init failure must
not happen.
"""

from __future__ import annotations

from http import HTTPStatus
from types import SimpleNamespace
from typing import Any
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.disaggregation.base.kv_manager import KVPoll
from sgl_jax.srt.disaggregation.decode import (
    DecodeBookkeeping,
    DecodePreallocQueue,
    DecodeTransferQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sgl_jax.srt.disaggregation.prefill import (
    PrefillBootstrapQueue,
    SchedulerDisaggregationPrefillMixin,
)
from sgl_jax.srt.managers.schedule_batch import FINISH_ABORT, FINISH_LENGTH


def _fake_req(rid: str, *, bootstrap_room: int | None = 42, req_pool_idx=7):
    return SimpleNamespace(
        rid=rid,
        bootstrap_room=bootstrap_room,
        disagg_transfer_id=rid,
        req_pool_idx=req_pool_idx,
        fill_ids=[1, 2, 3],
        origin_input_ids=[1, 2, 3],
        output_ids=[],
        finished_reason=None,
        finished_output=False,
        return_logprob=False,
        return_output_logprob_only=False,
        prefix_indices=[],
        last_matched_prefix_len=0,
    )


class _MockReqToTokenPool:
    def __init__(self):
        self.freed: list[int] = []

    def free(self, idx: int) -> None:
        self.freed.append(idx)


class _MockDecodeScheduler(SchedulerDisaggregationDecodeMixin):
    """Just enough of the Scheduler surface for the Mixin to drive."""

    def __init__(self):
        self.waiting_queue: list[Any] = []
        self.req_to_token_pool = _MockReqToTokenPool()
        self.disagg_prealloc_queue = DecodePreallocQueue()
        self.disagg_transfer_queue = DecodeTransferQueue()
        self.debug_pulls: list[Any] = []
        # Bootstrap / manager / receiver are wired by tests below.

    def process_input_requests(self, recv_reqs):
        # Mimic the real scheduler: build (well, alias) a Req per
        # recv_req and append to waiting_queue.
        for r in recv_reqs:
            self.waiting_queue.append(r)

    # Overrides for the model-specific hooks.
    def _build_kv_spec_for_req(self, req):
        return mock.MagicMock(name="spec")

    def _write_kv_to_pool(self, req, kv_indices, kv):
        self._wrote = (req.rid, kv_indices, kv)

    def _maybe_log_decode_pull_debug(self, req, kv):
        self.debug_pulls.append((req.rid, kv))


def _mock_receiver(initial_state=KVPoll.WAITING_FOR_INPUT):
    r = mock.MagicMock()
    r._state = initial_state
    r.poll.side_effect = lambda: r._state
    r.init.return_value = None
    r.result = mock.MagicMock(name="result_kv")
    return r


def test_pd_req_removed_from_waiting_queue_at_intake():
    """C1 regression: a PD req must not remain on waiting_queue
    after process_input_requests_disagg_decode returns, otherwise
    the scheduler's normal get_next_batch_to_run will try to prefill
    it on the decode side and defeat PD.
    """

    sched = _MockDecodeScheduler()
    sched.disagg_bootstrap_client = mock.MagicMock()
    sched.disagg_bootstrap_client.get_prefill_info.return_value = {
        "host": "10.0.0.1",
        "transfer_port": 30001,
        "side_channel_port": 9600,
    }
    sched.disagg_kv_manager = mock.MagicMock()
    sched.disagg_kv_manager.create_receiver.return_value = _mock_receiver()

    pd_req = _fake_req("r-pd", bootstrap_room=42)
    non_pd_req = _fake_req("r-other", bootstrap_room=None)

    sched.process_input_requests_disagg_decode([pd_req, non_pd_req])

    # PD req must have been pulled out.
    assert pd_req not in sched.waiting_queue
    # Non-PD req must stay.
    assert non_pd_req in sched.waiting_queue
    # PD req must be tracked in the prealloc queue.
    assert len(sched.disagg_prealloc_queue) == 1


def test_decode_intake_uses_transfer_id_for_receiver_metadata():
    sched = _MockDecodeScheduler()
    sched.disagg_bootstrap_client = mock.MagicMock()
    sched.disagg_bootstrap_client.get_prefill_info.return_value = {
        "host": "10.0.0.1",
        "transfer_port": 30001,
        "side_channel_port": 9600,
    }
    receiver = _mock_receiver()
    sched.disagg_kv_manager = mock.MagicMock()
    sched.disagg_kv_manager.create_receiver.return_value = receiver

    pd_req = _fake_req("r-wire", bootstrap_room=42)
    pd_req.disagg_transfer_id = "wire-uuid-42"

    sched.process_input_requests_disagg_decode([pd_req])

    (metadata,), _ = receiver.init.call_args
    assert metadata.uuid == "wire-uuid-42"


def test_bootstrap_lookup_failure_releases_resources_and_doesnt_leak():
    """C4 regression: a failed bootstrap lookup must release the
    ``req_to_token_pool`` slot AND not leave the req in
    waiting_queue. Previously the except block only logged.
    """

    sched = _MockDecodeScheduler()
    sched.disagg_bootstrap_client = mock.MagicMock()
    sched.disagg_bootstrap_client.get_prefill_info.side_effect = (
        RuntimeError("bootstrap unreachable")
    )
    sched.disagg_kv_manager = mock.MagicMock()

    pd_req = _fake_req("r-fail", bootstrap_room=42, req_pool_idx=3)
    sched.process_input_requests_disagg_decode([pd_req])

    assert pd_req not in sched.waiting_queue
    assert len(sched.disagg_prealloc_queue) == 0
    # Resources released.
    assert 3 in sched.req_to_token_pool.freed


def test_receiver_init_failure_releases_kv_indices():
    """C4 regression, complementary: a failed receiver init must
    release any KV indices already allocated and the
    ``req_to_token_pool`` slot.
    """

    sched = _MockDecodeScheduler()
    sched.disagg_bootstrap_client = mock.MagicMock()
    sched.disagg_bootstrap_client.get_prefill_info.return_value = {
        "host": "10.0.0.1",
        "transfer_port": 30001,
        "side_channel_port": 9600,
    }
    sched.disagg_kv_manager = mock.MagicMock()
    # Receiver init raises.
    bad_receiver = mock.MagicMock()
    bad_receiver.init.side_effect = RuntimeError("boom")
    sched.disagg_kv_manager.create_receiver.return_value = bad_receiver

    released_indices = []

    def _prealloc(req):
        return "kv-idx-7"

    def _release(idx):
        released_indices.append(idx)

    sched._prealloc_decode_kv_indices = _prealloc
    sched._release_decode_kv_indices = _release

    pd_req = _fake_req("r-init-fail", bootstrap_room=42, req_pool_idx=9)
    sched.process_input_requests_disagg_decode([pd_req])

    assert pd_req not in sched.waiting_queue
    assert len(sched.disagg_prealloc_queue) == 0
    assert released_indices == ["kv-idx-7"]
    assert 9 in sched.req_to_token_pool.freed


def test_process_decode_queue_success_path_enqueues_for_decode():
    """Happy path: receiver SUCCESS → KV written → req re-enqueued
    on waiting_queue for the standard decode loop.
    """

    sched = _MockDecodeScheduler()
    receiver = _mock_receiver(initial_state=KVPoll.SUCCESS)
    pd_req = _fake_req("r-ok", bootstrap_room=42)
    sched.disagg_prealloc_queue.add(
        DecodeBookkeeping(
            req_id="r-ok", req=pd_req, receiver=receiver,
            kv_indices="kv-idx-A", started=True,
        )
    )

    sched.process_decode_queue()

    assert hasattr(sched, "_wrote")
    result_kv = receiver.result["kv"]
    assert sched._wrote == ("r-ok", "kv-idx-A", result_kv)
    assert sched.debug_pulls == [("r-ok", result_kv)]
    assert pd_req in sched.waiting_queue


def test_process_decode_queue_failure_path_releases_resources():
    """C3 regression: receiver FAILED must NOT call
    ``cache_finished_req`` (which would assume the req went through
    prefill). Default impl frees the req_to_token_pool slot.
    """

    sched = _MockDecodeScheduler()
    receiver = _mock_receiver(initial_state=KVPoll.FAILED)
    released = []

    pd_req = _fake_req("r-bad", bootstrap_room=42, req_pool_idx=11)
    sched.disagg_prealloc_queue.add(
        DecodeBookkeeping(
            req_id="r-bad", req=pd_req, receiver=receiver,
            kv_indices="kv-idx-X", started=True,
        )
    )
    sched._release_decode_kv_indices = released.append

    sched.process_decode_queue()

    assert released == ["kv-idx-X"]
    assert 11 in sched.req_to_token_pool.freed
    # cache_finished_req was never called — would have raised because
    # the mock scheduler doesn't define it.
    assert pd_req not in sched.waiting_queue


def test_extract_pd_reqs_preserves_non_pd_order():
    """I6 regression: targeted extraction must not reorder non-PD
    reqs in the waiting queue.
    """

    sched = _MockDecodeScheduler()
    a = _fake_req("a", bootstrap_room=None)
    b = _fake_req("b", bootstrap_room=42)
    c = _fake_req("c", bootstrap_room=None)
    d = _fake_req("d", bootstrap_room=43)
    sched.waiting_queue.extend([a, b, c, d])

    pd = sched._extract_pd_reqs_from_waiting_queue({"b", "d"})
    assert pd == [b, d]
    assert sched.waiting_queue == [a, c]


# --- Prefill Mixin sanity (covers PrefillBootstrapQueue interaction) ---


class _MockPrefillScheduler(SchedulerDisaggregationPrefillMixin):
    def __init__(self):
        self.disagg_prefill_queue = PrefillBootstrapQueue()
        self.released_reqs: list[Any] = []
        self.debug_extracts: list[Any] = []
        self.streamed: list[Any] = []
        self.sampling_done: list[Any] = []

    def _release_prefill_req_resources(self, req):
        self.released_reqs.append(req)

    def _maybe_log_prefill_extract_debug(self, req, kv, **meta):
        self.debug_extracts.append((req.rid, kv, meta))

    def stream_output(
        self,
        reqs,
        return_logprob,
        return_output_logprob_only,
        skip_req=None,
        cache_miss_count=None,
    ):
        self.streamed.append(
            (
                list(reqs),
                return_logprob,
                return_output_logprob_only,
                skip_req,
                cache_miss_count,
            )
        )

    def set_next_batch_sampling_info_done(self, batch):
        self.sampling_done.append(batch)


def test_send_kv_chunk_drains_terminal_senders_and_releases():
    sched = _MockPrefillScheduler()
    sender = mock.MagicMock()
    sender.poll.side_effect = lambda: KVPoll.SUCCESS
    pd_req = _fake_req("p-1", bootstrap_room=1)

    sched.disagg_prefill_queue.add(
        pd_req.rid, sender,
        on_terminal=lambda req=pd_req: sched._release_prefill_req_resources(req),
    )

    sched.send_kv_chunk()
    assert sched.released_reqs == [pd_req]
    assert len(sched.disagg_prefill_queue) == 0


def test_process_prefill_chunk_calls_debug_hook_after_extract():
    sched = _MockPrefillScheduler()
    sched.disagg_use_d2h_staging = False
    sched._extract_req_kv = mock.MagicMock(return_value="device-kv")

    sender = mock.MagicMock()
    sched.disagg_kv_manager = mock.MagicMock()
    sched.disagg_kv_manager.create_sender.return_value = sender

    req = _fake_req("p-debug", bootstrap_room=9)
    batch = SimpleNamespace(reqs=[req])

    sched.process_prefill_chunk(batch, result=mock.sentinel.result)

    assert sched.debug_extracts
    rid, kv, meta = sched.debug_extracts[0]
    assert rid == "p-debug"
    assert kv == "device-kv"
    assert "use_d2h_staging" in meta
    assert not sched.streamed
    assert sched.sampling_done == [batch]


def test_process_prefill_chunk_uses_transfer_id_for_sender():
    sched = _MockPrefillScheduler()
    sched.disagg_use_d2h_staging = False
    sched._extract_req_kv = mock.MagicMock(return_value="device-kv")

    sender = mock.MagicMock()
    sched.disagg_kv_manager = mock.MagicMock()
    sched.disagg_kv_manager.create_sender.return_value = sender

    req = _fake_req("p-wire", bootstrap_room=9)
    req.disagg_transfer_id = "wire-prefill-9"
    batch = SimpleNamespace(reqs=[req])

    sched.process_prefill_chunk(batch, result=mock.sentinel.result)

    sender.init.assert_called_once_with(
        kv_indices=None,
        transfer_id="wire-prefill-9",
    )


def test_process_prefill_chunk_skips_standard_prefill_output_for_pd_req():
    sched = _MockPrefillScheduler()
    sched.disagg_use_d2h_staging = False
    sched._extract_req_kv = mock.MagicMock(return_value="device-kv")
    sender = mock.MagicMock()
    sched.disagg_kv_manager = mock.MagicMock()
    sched.disagg_kv_manager.create_sender.return_value = sender

    req = _fake_req("p-only", bootstrap_room=7)
    req.return_logprob = False
    req.return_output_logprob_only = False
    batch = SimpleNamespace(reqs=[req])
    result = SimpleNamespace(next_token_ids=[123])

    sched.process_prefill_chunk(batch, result=result)

    assert req.output_ids == []
    assert req.finished_reason is None
    assert not sched.streamed


def test_send_kv_chunk_success_streams_empty_completion_for_prefill_only_req():
    sched = _MockPrefillScheduler()
    sender = mock.MagicMock()
    sender.poll.return_value = KVPoll.SUCCESS
    sender.clear.return_value = None

    req = _fake_req("p-done", bootstrap_room=1)
    req.return_logprob = False
    req.return_output_logprob_only = False
    req.finished_reason = None
    req.output_ids = []
    req.finished_output = False

    sched.disagg_prefill_queue.add(
        req.rid,
        sender,
        on_terminal=lambda req=req, snd=sender: sched._on_prefill_transfer_terminal(
            req, snd
        ),
    )

    sched.send_kv_chunk()

    assert isinstance(req.finished_reason, FINISH_LENGTH)
    assert req.finished_reason.length == 0
    assert sched.streamed
    streamed_reqs, _, _, _, _ = sched.streamed[0]
    assert streamed_reqs == [req]
    assert sender.clear.call_count == 1
    assert sched.released_reqs == [req]


def test_send_kv_chunk_failed_streams_abort_for_prefill_only_req():
    sched = _MockPrefillScheduler()
    sender = mock.MagicMock()
    sender.poll.return_value = KVPoll.FAILED
    sender.clear.return_value = None
    sender.failure_exception.side_effect = RuntimeError("sender failed")

    req = _fake_req("p-fail", bootstrap_room=1)
    req.return_logprob = False
    req.return_output_logprob_only = False
    req.finished_reason = None
    req.output_ids = []
    req.finished_output = False

    sched.disagg_prefill_queue.add(
        req.rid,
        sender,
        on_terminal=lambda req=req, snd=sender: sched._on_prefill_transfer_terminal(
            req, snd
        ),
    )

    sched.send_kv_chunk()

    assert isinstance(req.finished_reason, FINISH_ABORT)
    assert req.finished_reason.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert "sender failed" in req.finished_reason.message
    assert sched.streamed
    streamed_reqs, _, _, _, _ = sched.streamed[0]
    assert streamed_reqs == [req]
    assert sender.clear.call_count == 1
    assert sched.released_reqs == [req]


def test_prefill_event_loop_skips_idle_memory_checks_for_pd_queue():
    order = []

    class _IdlePrefillScheduler(SchedulerDisaggregationPrefillMixin):
        def __init__(self):
            self._comm_backend = None
            self._engine_paused = False
            self.disagg_prefill_queue = PrefillBootstrapQueue()
            self.last_batch = None
            self.new_token_ratio = None
            self.init_new_token_ratio = 1.0

        def recv_requests(self):
            return []

        def select_dp_for_request(self, recv_reqs):
            return recv_reqs

        def process_input_requests(self, recv_reqs):
            return None

        def get_next_batch_to_run(self):
            return None

        def send_kv_chunk(self):
            order.append("send")
            raise RuntimeError("stop-loop")

        def check_memory(self):
            order.append("check")

        def check_tree_cache(self):
            order.append("tree")

    sched = _IdlePrefillScheduler()
    with pytest.raises(RuntimeError, match="stop-loop"):
        sched.event_loop_normal_disagg_prefill()

    assert order == ["send"]


def test_verify_decode_writeback_debug_executes_when_enabled(monkeypatch):
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    from sgl_jax.srt.disaggregation import prefill as prefill_mod

    monkeypatch.setenv("SGL_JAX_PD_DEBUG_KV", "1")

    sched = _MockDecodeScheduler()
    mesh = Mesh(np.asarray(jax.local_devices()[:1]).reshape(1), ("x",))
    sharding = NamedSharding(mesh, PartitionSpec(None, None, None))
    buf = jax.device_put(jnp.zeros((1, 2, 1), dtype=jnp.float32), sharding)

    class _Pool:
        def __init__(self):
            self.start_layer = 0
            self.layer_num = 1
            self.kv_sharding = sharding
            self.mesh = mesh

        def get_kv_buffer(self, _layer_id):
            return buf

    monkeypatch.setattr(
        prefill_mod,
        "_jit_gather_all_layers",
        lambda buffers, _page_ids, _out_sharding: buffers,
    )

    kv = jnp.stack([buf], axis=0)
    req = _fake_req("r-writeback", bootstrap_room=1)

    sched._maybe_verify_decode_writeback_debug(
        req,
        _Pool(),
        np.asarray([0], dtype=np.int32),
        kv,
    )


def test_write_kv_to_pool_keeps_last_real_page_when_bucket_is_padded():
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    mesh = Mesh(np.asarray(jax.local_devices()[:1]).reshape(1), ("x",))
    sharding = NamedSharding(mesh, PartitionSpec(None, None, None))
    kv_buffer = jax.device_put(
        jnp.zeros((8, 2, 1), dtype=jnp.float32), sharding
    )

    class _Pool:
        def __init__(self):
            self.page_size = 2
            self.mesh = mesh
            self.kv_sharding = sharding
            self.kv_buffer = [kv_buffer]
            self.start_layer = 0
            self.layer_num = 1
            self.dtype = jnp.float32

        def get_kv_buffer(self, layer_id):
            return self.kv_buffer[layer_id]

    class _Allocator:
        def __init__(self, pool):
            self._pool = pool

        def get_kvcache(self):
            return self._pool

    class _WritebackScheduler(SchedulerDisaggregationDecodeMixin):
        def __init__(self, pool):
            self.token_to_kv_pool_allocator = _Allocator(pool)

        def _maybe_verify_decode_writeback_debug(self, req, kv_pool, page_ids_padded, kv):
            return None

    pool = _Pool()
    sched = _WritebackScheduler(pool)
    req = _fake_req("r-padded", bootstrap_room=1)
    req.origin_input_ids = [1, 2, 3, 4, 5]
    req.output_ids = []

    # page_size=2, seqlen=5 -> num_pages=3 real pages, but kv arrives in
    # bucket-4 shape. The last row is stale padded data and must NOT
    # overwrite the last real page.
    kv = jnp.asarray(
        [
            [
                [[1.0], [1.0]],
                [[2.0], [2.0]],
                [[3.0], [3.0]],
                [[9.0], [9.0]],
            ]
        ],
        dtype=jnp.float32,
    )
    kv_indices = np.asarray([8, 9, 10, 11, 12, 13], dtype=np.int32)

    sched._write_kv_to_pool(req, kv_indices, kv)

    host = np.asarray(jax.device_get(pool.kv_buffer[0]))
    assert host[4].tolist() == [[1.0], [1.0]]
    assert host[5].tolist() == [[2.0], [2.0]]
    assert host[6].tolist() == [[3.0], [3.0]]


def test_prefill_extract_failure_skips_send():
    """If _extract_req_kv raises, no sender is created and the queue is empty."""

    class _FailExtractScheduler(SchedulerDisaggregationPrefillMixin):
        def __init__(self):
            self.disagg_prefill_queue = PrefillBootstrapQueue()
            self.disagg_use_d2h_staging = False
            self._extract_called = False

        def set_next_batch_sampling_info_done(self, batch):
            pass

        def _extract_req_kv(self, req):
            self._extract_called = True
            raise RuntimeError("simulated extract failure")

        def _maybe_log_prefill_extract_debug(self, req, kv, **meta):
            pass

    sched = _FailExtractScheduler()
    sched.disagg_kv_manager = mock.MagicMock()
    req = _fake_req("r-fail-extract", bootstrap_room=1)
    batch = SimpleNamespace(reqs=[req], next_batch_sampling_info=None)

    sched.process_prefill_chunk(batch, None)

    assert sched._extract_called
    assert len(sched.disagg_prefill_queue) == 0
    sched.disagg_kv_manager.create_sender.assert_not_called()


def test_decode_event_loop_idle_structure():
    """Decode event loop: when no batch and no recv_reqs, it still calls
    process_decode_queue each tick.
    """

    class _IdleDecodeScheduler(SchedulerDisaggregationDecodeMixin):
        def __init__(self):
            from sgl_jax.srt.disaggregation.decode import (
                DecodePreallocQueue,
                DecodeTransferQueue,
            )

            self.disagg_prealloc_queue = DecodePreallocQueue()
            self.disagg_transfer_queue = DecodeTransferQueue()
            self.waiting_queue = []
            self._engine_paused = False
            self._comm_backend = None
            self.cur_batch = None
            self.last_batch = None
            self.new_token_ratio = 0.3
            self.init_new_token_ratio = 0.3
            self._ticks = 0

        def recv_requests(self):
            self._ticks += 1
            if self._ticks > 2:
                raise StopIteration("done")
            return []

        def select_dp_for_request(self, reqs):
            return reqs

        def process_input_requests(self, recv_reqs):
            pass

        def get_next_batch_to_run(self):
            return None

    sched = _IdleDecodeScheduler()

    with pytest.raises(StopIteration):
        sched.event_loop_normal_disagg_decode()

    assert sched._ticks == 3
    assert sched.cur_batch is None
