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
from pathlib import Path
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


def test_pd_mixins_use_explicit_scheduler_contracts_for_first_party_fields():
    """Keep PD mixins close to upstream style.

    The mixins are Scheduler-only extensions, so they should type ``self``
    against Scheduler under TYPE_CHECKING instead of hiding missing
    attributes with type-ignore comments. First-party request fields should
    also be accessed directly; getattr is reserved for capability probes or
    truly heterogeneous inputs.
    """

    root = Path(__file__).resolve().parents[3]
    sources = {
        "prefill": root / "python" / "sgl_jax" / "srt" / "disaggregation" / "prefill.py",
        "decode": root / "python" / "sgl_jax" / "srt" / "disaggregation" / "decode.py",
    }
    for source in sources.values():
        text = source.read_text()
        assert "type: ignore" not in text

    prefill_text = sources["prefill"].read_text()
    decode_text = sources["decode"].read_text()
    for bad in (
        'getattr(req, "bootstrap_room"',
        'getattr(req, "disagg_transfer_id"',
        'getattr(req, "return_logprob"',
        'getattr(req, "return_output_logprob_only"',
    ):
        assert bad not in prefill_text
        assert bad not in decode_text


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

    def free(self, req) -> None:
        self.freed.append(req.req_pool_idx)
        req.req_pool_idx = None


class _MockDecodeScheduler(SchedulerDisaggregationDecodeMixin):
    """Just enough of the Scheduler surface for the Mixin to drive."""

    def __init__(self):
        self.waiting_queue: list[Any] = []
        self.req_to_token_pool = _MockReqToTokenPool()
        self.disagg_prealloc_queue = DecodePreallocQueue()
        self.disagg_transfer_queue = DecodeTransferQueue()
        self.debug_pulls: list[Any] = []
        self._comm_backend = None
        self.send_to_tokenizer = mock.MagicMock()
        self.token_to_kv_pool_allocator = None
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
    sched.disagg_bootstrap_client.get_prefill_info.side_effect = RuntimeError(
        "bootstrap unreachable"
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
            req_id="r-ok",
            req=pd_req,
            receiver=receiver,
            kv_indices="kv-idx-A",
            started=True,
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
            req_id="r-bad",
            req=pd_req,
            receiver=receiver,
            kv_indices="kv-idx-X",
            started=True,
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
        pd_req.rid,
        sender,
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
        on_terminal=lambda req=req, snd=sender: sched._on_prefill_transfer_terminal(req, snd),
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
        on_terminal=lambda req=req, snd=sender: sched._on_prefill_transfer_terminal(req, snd),
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
    kv_buffer = jax.device_put(jnp.zeros((8, 2, 1), dtype=jnp.float32), sharding)

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


def test_prefill_extract_failure_aborts_and_releases():
    """If _extract_req_kv raises, the req gets FINISH_ABORT and resources are released."""

    class _FailExtractScheduler(SchedulerDisaggregationPrefillMixin):
        def __init__(self):
            self.disagg_prefill_queue = PrefillBootstrapQueue()
            self.disagg_use_d2h_staging = False
            self._extract_called = False
            self.released_reqs: list[Any] = []
            self.streamed: list[Any] = []

        def set_next_batch_sampling_info_done(self, batch):
            pass

        def _extract_req_kv(self, req):
            self._extract_called = True
            raise RuntimeError("simulated extract failure")

        def _maybe_log_prefill_extract_debug(self, req, kv, **meta):
            pass

        def _release_prefill_req_resources(self, req):
            self.released_reqs.append(req)

        def stream_output(
            self,
            reqs,
            return_logprob,
            return_output_logprob_only,
            skip_req=None,
            cache_miss_count=None,
        ):
            self.streamed.append(list(reqs))

    sched = _FailExtractScheduler()
    sched.disagg_kv_manager = mock.MagicMock()
    req = _fake_req("r-fail-extract", bootstrap_room=1)
    req.return_logprob = False
    req.return_output_logprob_only = False
    batch = SimpleNamespace(reqs=[req], next_batch_sampling_info=None)

    sched.process_prefill_chunk(batch, None)

    assert sched._extract_called
    assert len(sched.disagg_prefill_queue) == 0
    sched.disagg_kv_manager.create_sender.assert_not_called()
    assert isinstance(req.finished_reason, FINISH_ABORT)
    assert "simulated extract failure" in req.finished_reason.message
    assert sched.streamed == [[req]]
    assert sched.released_reqs == [req]


def test_prefill_sender_init_failure_aborts_and_releases():
    """If sender.init raises, the req gets FINISH_ABORT and resources are released."""

    sched = _MockPrefillScheduler()
    sched.disagg_use_d2h_staging = False
    sched._extract_req_kv = mock.MagicMock(return_value="device-kv")

    bad_sender = mock.MagicMock()
    bad_sender.init.side_effect = RuntimeError("sender init exploded")
    sched.disagg_kv_manager = mock.MagicMock()
    sched.disagg_kv_manager.create_sender.return_value = bad_sender

    req = _fake_req("p-sender-fail", bootstrap_room=7)
    req.return_logprob = False
    req.return_output_logprob_only = False
    batch = SimpleNamespace(reqs=[req])

    sched.process_prefill_chunk(batch, None)

    assert isinstance(req.finished_reason, FINISH_ABORT)
    assert "sender init exploded" in req.finished_reason.message
    assert sched.streamed
    assert sched.released_reqs == [req]
    assert len(sched.disagg_prefill_queue) == 0
    assert bad_sender.abort.called
    assert bad_sender.clear.called


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


# --- Decode failure path: AbortReq sent to tokenizer ---


def test_bootstrap_lookup_failure_sends_abort_to_tokenizer():
    sched = _MockDecodeScheduler()
    sched.disagg_bootstrap_client = mock.MagicMock()
    sched.disagg_bootstrap_client.get_prefill_info.side_effect = RuntimeError(
        "bootstrap unreachable"
    )
    sched.disagg_kv_manager = mock.MagicMock()

    pd_req = _fake_req("r-abort-boot", bootstrap_room=42, req_pool_idx=3)
    sched.process_input_requests_disagg_decode([pd_req])

    from sgl_jax.srt.managers.io_struct import AbortReq

    sched.send_to_tokenizer.send_pyobj.assert_called_once()
    abort_msg = sched.send_to_tokenizer.send_pyobj.call_args[0][0]
    assert isinstance(abort_msg, AbortReq)
    assert abort_msg.rid == "r-abort-boot"


def test_receiver_init_failure_sends_abort_to_tokenizer():
    sched = _MockDecodeScheduler()
    sched.disagg_bootstrap_client = mock.MagicMock()
    sched.disagg_bootstrap_client.get_prefill_info.return_value = {
        "host": "10.0.0.1",
        "transfer_port": 30001,
        "side_channel_port": 9600,
    }
    bad_receiver = mock.MagicMock()
    bad_receiver.init.side_effect = RuntimeError("boom")
    sched.disagg_kv_manager = mock.MagicMock()
    sched.disagg_kv_manager.create_receiver.return_value = bad_receiver
    sched._prealloc_decode_kv_indices = lambda req: "kv-idx"
    sched._release_decode_kv_indices = mock.MagicMock()

    pd_req = _fake_req("r-abort-init", bootstrap_room=42, req_pool_idx=5)
    sched.process_input_requests_disagg_decode([pd_req])

    from sgl_jax.srt.managers.io_struct import AbortReq

    sched.send_to_tokenizer.send_pyobj.assert_called_once()
    abort_msg = sched.send_to_tokenizer.send_pyobj.call_args[0][0]
    assert isinstance(abort_msg, AbortReq)
    assert abort_msg.rid == "r-abort-init"


def test_kv_writeback_failure_sends_abort_to_tokenizer():
    sched = _MockDecodeScheduler()

    def _bad_write(req, kv_indices, kv):
        raise RuntimeError("writeback exploded")

    sched._write_kv_to_pool = _bad_write
    sched._release_decode_kv_indices = mock.MagicMock()

    receiver = _mock_receiver(initial_state=KVPoll.SUCCESS)
    pd_req = _fake_req("r-abort-wb", bootstrap_room=42, req_pool_idx=8)
    sched.disagg_prealloc_queue.add(
        DecodeBookkeeping(
            req_id="r-abort-wb",
            req=pd_req,
            receiver=receiver,
            kv_indices="kv-idx-W",
            started=True,
        )
    )

    sched.process_decode_queue()

    from sgl_jax.srt.managers.io_struct import AbortReq

    sched.send_to_tokenizer.send_pyobj.assert_called_once()
    abort_msg = sched.send_to_tokenizer.send_pyobj.call_args[0][0]
    assert isinstance(abort_msg, AbortReq)
    assert abort_msg.rid == "r-abort-wb"


def test_receiver_terminal_failed_sends_abort_to_tokenizer():
    sched = _MockDecodeScheduler()
    sched._release_decode_kv_indices = mock.MagicMock()

    receiver = _mock_receiver(initial_state=KVPoll.FAILED)
    pd_req = _fake_req("r-abort-fail", bootstrap_room=42, req_pool_idx=10)
    sched.disagg_prealloc_queue.add(
        DecodeBookkeeping(
            req_id="r-abort-fail",
            req=pd_req,
            receiver=receiver,
            kv_indices="kv-idx-F",
            started=True,
        )
    )

    sched.process_decode_queue()

    from sgl_jax.srt.managers.io_struct import AbortReq

    sched.send_to_tokenizer.send_pyobj.assert_called_once()
    abort_msg = sched.send_to_tokenizer.send_pyobj.call_args[0][0]
    assert isinstance(abort_msg, AbortReq)
    assert abort_msg.rid == "r-abort-fail"


# --- dispatch_scheduler_event_loop routing ---


def test_dispatch_scheduler_event_loop_routes_correctly():
    from sgl_jax.srt.disaggregation.runtime import dispatch_scheduler_event_loop

    for mode, overlap, expected_method in [
        ("null", False, "event_loop_normal"),
        ("null", True, "event_loop_overlap"),
        ("prefill", False, "event_loop_normal_disagg_prefill"),
        ("decode", False, "event_loop_normal_disagg_decode"),
    ]:
        scheduler = mock.MagicMock()
        scheduler.enable_overlap = overlap
        server_args = SimpleNamespace(disaggregation_mode=mode)

        dispatch_scheduler_event_loop(scheduler, server_args)

        getattr(scheduler, expected_method).assert_called_once()


# --- abort_matching / flush / internal_state ---


def test_decode_prealloc_queue_abort_matching():
    q = DecodePreallocQueue()
    e1 = DecodeBookkeeping(req_id="r-1", req=_fake_req("r-1"))
    e2 = DecodeBookkeeping(req_id="r-2", req=_fake_req("r-2"))
    e3 = DecodeBookkeeping(req_id="s-1", req=_fake_req("s-1"))
    q.add(e1)
    q.add(e2)
    q.add(e3)

    aborted = q.abort_matching("r-", abort_all=False)
    assert {e.req_id for e in aborted} == {"r-1", "r-2"}
    assert len(q) == 1


def test_decode_transfer_queue_abort_matching():
    q = DecodeTransferQueue()
    e1 = DecodeBookkeeping(req_id="x-a", req=_fake_req("x-a"))
    e2 = DecodeBookkeeping(req_id="y-b", req=_fake_req("y-b"))
    q.add(e1)
    q.add(e2)

    aborted = q.abort_matching("", abort_all=True)
    assert len(aborted) == 2
    assert len(q) == 0


def test_prefill_queue_abort_matching():
    q = PrefillBootstrapQueue()
    sender = mock.MagicMock()
    q.add("p-1", sender)
    q.add("p-2", sender)
    q.add("q-1", sender)

    aborted = q.abort_matching("p-", abort_all=False)
    assert {e.req_id for e in aborted} == {"p-1", "p-2"}
    assert len(q) == 1


def test_prefill_queue_abort_calls_on_terminal():
    """abort_request must invoke on_terminal for entries popped from
    the prefill queue, otherwise the request has no terminal response
    and resources leak.
    """

    sched = _MockPrefillScheduler()
    sender = mock.MagicMock()
    sender.poll.return_value = KVPoll.WAITING_FOR_INPUT

    req = _fake_req("p-abort-1", bootstrap_room=1)
    req.return_logprob = False
    req.return_output_logprob_only = False

    sched.disagg_prefill_queue.add(
        req.rid,
        sender,
        on_terminal=lambda req=req, snd=sender: sched._on_prefill_transfer_terminal(req, snd),
    )

    # Provide minimal Scheduler attributes so abort_request can run.
    sched.waiting_queue = []
    sched.grammar_queue = []
    sched.running_batch = SimpleNamespace(reqs_info=[])
    sched.cur_batch = None
    sched._comm_backend = None
    sched.send_to_tokenizer = mock.MagicMock()
    sched.disagg_prealloc_queue = None
    sched.disagg_transfer_queue = None

    from sgl_jax.srt.managers.scheduler import Scheduler

    recv_req = SimpleNamespace(rid="p-abort-1", abort_all=False)
    Scheduler.abort_request(sched, recv_req)

    assert sender.abort.called
    assert sched.released_reqs == [req]
    assert len(sched.disagg_prefill_queue) == 0


# --- disagg_transfer_id contract ---


def test_different_rids_same_transfer_id_uses_shared_id():
    """P and D have different rids but share the same transfer_id.
    Both sides must use the shared transfer_id, not their own rid.
    """

    # --- Prefill side ---
    p_sched = _MockPrefillScheduler()
    p_sched.disagg_use_d2h_staging = False
    p_sched._extract_req_kv = mock.MagicMock(return_value="kv-data")
    p_sender = mock.MagicMock()
    p_sched.disagg_kv_manager = mock.MagicMock()
    p_sched.disagg_kv_manager.create_sender.return_value = p_sender

    p_req = _fake_req("p-req-1", bootstrap_room=7)
    p_req.disagg_transfer_id = "shared-xfer-42"
    p_batch = SimpleNamespace(reqs=[p_req])
    p_sched.process_prefill_chunk(p_batch, None)

    p_sender.init.assert_called_once_with(kv_indices=None, transfer_id="shared-xfer-42")

    # --- Decode side ---
    d_sched = _MockDecodeScheduler()
    d_sched.disagg_bootstrap_client = mock.MagicMock()
    d_sched.disagg_bootstrap_client.get_prefill_info.return_value = {
        "host": "10.0.0.1",
        "transfer_port": 30001,
        "side_channel_port": 9600,
    }
    d_receiver = _mock_receiver()
    d_sched.disagg_kv_manager = mock.MagicMock()
    d_sched.disagg_kv_manager.create_receiver.return_value = d_receiver

    d_req = _fake_req("d-req-1", bootstrap_room=7)
    d_req.disagg_transfer_id = "shared-xfer-42"
    d_sched.process_input_requests_disagg_decode([d_req])

    (metadata,), _ = d_receiver.init.call_args
    assert metadata.uuid == "shared-xfer-42"


# ---- shutdown handler (merged from test_disagg_shutdown_handler.py) ----------


def test_disagg_shutdown_handler_unregisters_and_drains():
    from sgl_jax.srt.disaggregation.runtime import _make_disagg_shutdown

    scheduler = mock.MagicMock()
    scheduler.disagg_bootstrap_key = "bkey"
    scheduler.disagg_bootstrap_client = mock.MagicMock()
    scheduler.disagg_heartbeat = mock.MagicMock()
    scheduler.disagg_kv_manager = mock.MagicMock()

    fn = _make_disagg_shutdown(scheduler, "prefill")
    fn()
    scheduler.disagg_bootstrap_client.unregister_prefill.assert_called_once_with("bkey")
    scheduler.disagg_heartbeat.stop.assert_called_once()
    scheduler.disagg_kv_manager.graceful_shutdown.assert_called_once()
    scheduler.disagg_kv_manager.zmq_notifier.stop.assert_called_once()
    # Idempotent.
    fn()
    scheduler.disagg_bootstrap_client.unregister_prefill.assert_called_once()


# ---- debug utils (merged from test_pd_debug_utils.py) -----------------------


def test_kv_debug_enabled_honors_flag_and_req_filter(monkeypatch):
    from sgl_jax.srt.disaggregation.debug_utils import kv_debug_enabled

    monkeypatch.delenv("SGL_JAX_PD_DEBUG_KV", raising=False)
    monkeypatch.delenv("SGL_JAX_PD_DEBUG_REQ_ID", raising=False)
    assert not kv_debug_enabled("req-1")

    monkeypatch.setenv("SGL_JAX_PD_DEBUG_KV", "1")
    assert kv_debug_enabled("req-1")

    monkeypatch.setenv("SGL_JAX_PD_DEBUG_REQ_ID", "bucket16")
    assert kv_debug_enabled("probe-bucket16-req")
    assert not kv_debug_enabled("probe-bucket8-req")


def test_build_kv_debug_snapshot_captures_global_and_page_digests():
    from sgl_jax.srt.disaggregation.debug_utils import build_kv_debug_snapshot

    kv = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
    snapshot = build_kv_debug_snapshot(kv)

    assert snapshot.shape == (2, 3, 4)
    assert snapshot.dtype == "int16"
    assert len(snapshot.global_digest) == 16
    assert len(snapshot.page_digests) == 2
    assert len(snapshot.page_digests[0]) == 3
    assert snapshot.page_digests[0][0] != snapshot.page_digests[0][1]
    assert snapshot.sample_page_digests(max_layers=1, max_pages=2) == (
        (snapshot.page_digests[0][0], snapshot.page_digests[0][1]),
    )


def test_compare_kv_debug_snapshots_finds_first_mismatch():
    from sgl_jax.srt.disaggregation.debug_utils import (
        build_kv_debug_snapshot,
        count_kv_debug_mismatches,
        find_first_kv_debug_mismatch,
    )

    base = np.arange(2 * 2 * 3, dtype=np.int16).reshape(2, 2, 3)
    modified = base.copy()
    modified[1, 0, 2] += 7

    left = build_kv_debug_snapshot(base)
    right = build_kv_debug_snapshot(modified)

    assert count_kv_debug_mismatches(left, right) == 1
    assert find_first_kv_debug_mismatch(left, right) == (1, 0)


def test_build_kv_debug_snapshot_rejects_inputs_without_layer_and_page_axes():
    from sgl_jax.srt.disaggregation.debug_utils import build_kv_debug_snapshot

    with pytest.raises(ValueError, match="at least 2 dims"):
        build_kv_debug_snapshot(np.arange(4, dtype=np.int16))
