# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import math
from types import SimpleNamespace

import jax
import numpy as np
import pytest
import zmq
from jax.sharding import Mesh

import sgl_jax.srt.managers.scheduler as scheduler_module
from sgl_jax.srt.layers.logits_processor import LogitsProcessor
from sgl_jax.srt.layers.sampler import (
    get_token_ids_logprobs as sampler_get_token_ids_logprobs,
)
from sgl_jax.srt.managers.io_struct import (
    GenerateReqInput,
    ReleaseScoringCacheReqInput,
    ScoreFromCacheReqInput,
    ScoreFromCacheReqOutput,
    TokenizedGenerateReqInput,
)
from sgl_jax.srt.managers.scheduler import Scheduler
from sgl_jax.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sgl_jax.srt.managers.tokenizer_manager import ReqState, TokenizerManager


class _FakeCreateTokenizedManager:
    _create_tokenized_object = TokenizerManager._create_tokenized_object

    def __init__(self):
        self.preferred_sampling_params = None
        self.tokenizer = None
        self.model_config = SimpleNamespace(vocab_size=32000)


class _FakePrefillManager:
    _prefill_and_cache = TokenizerManager._prefill_and_cache

    def __init__(self):
        self.seen_requests = []

    async def generate_request(self, req, request=None):
        del request
        self.seen_requests.append(req)
        yield {"meta_info": {"id": req.rid}}


class _FakeExtendManager:
    _batched_extend_score = TokenizerManager._batched_extend_score
    _batched_extend_score_with_metrics = TokenizerManager._batched_extend_score_with_metrics

    def __init__(self):
        self.seen_request = None

    async def generate_request(self, req, request=None):
        del request
        self.seen_request = req
        yield [
            {
                "index": 0,
                "meta_info": {
                    "output_token_ids_logprobs": [
                        [
                            (math.log(0.9), 10, None),
                            (math.log(0.1), 20, None),
                        ]
                    ]
                },
            },
            {
                "index": 1,
                "meta_info": {
                    "output_token_ids_logprobs": [
                        [
                            (math.log(0.2), 10, None),
                            (math.log(0.8), 20, None),
                        ]
                    ]
                },
            },
        ]


class _FakeExtendMissingLogprobsManager:
    _batched_extend_score = TokenizerManager._batched_extend_score
    _batched_extend_score_with_metrics = TokenizerManager._batched_extend_score_with_metrics

    async def generate_request(self, req, request=None):
        del req, request
        yield [{"index": 0, "meta_info": {"output_token_ids_logprobs": []}}]


class _FakeReleaseCacheCommunicator:
    def __init__(self, outputs=None, delay_s: float = 0.0, raise_exc: Exception | None = None):
        self.outputs = outputs
        self.delay_s = delay_s
        self.raise_exc = raise_exc

    async def __call__(self, req, timeout=None):
        del req, timeout
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.outputs


class _FakeReleaseCacheManager:
    _release_cache = TokenizerManager._release_cache

    def __init__(self, communicator, timeout_s: float):
        self.release_scoring_cache_communicator = communicator
        self.server_args = SimpleNamespace(multi_item_prefill_extend_cache_timeout=timeout_s)

    def auto_create_handle_loop(self):
        return None


class _NoopSender:
    def __init__(self):
        self.calls = []

    def send_pyobj(self, obj):
        self.calls.append(obj)


class _FakeSchedulerLivenessManager:
    _send_one_request = TokenizerManager._send_one_request
    _send_batch_requests = TokenizerManager._send_batch_requests
    _wait_one_response = TokenizerManager._wait_one_response
    _build_scheduler_unavailable_message = TokenizerManager._build_scheduler_unavailable_message
    _fail_pending_requests = TokenizerManager._fail_pending_requests
    _mark_scheduler_unavailable = TokenizerManager._mark_scheduler_unavailable
    _check_scheduler_health = TokenizerManager._check_scheduler_health
    _raise_if_scheduler_unavailable = TokenizerManager._raise_if_scheduler_unavailable

    def __init__(self):
        self.wait_timeout = 0.01
        self.scheduler_pids = [4321]
        self.scheduler_unavailable_error = None
        self.health_check_failed = False
        self.rid_to_state = {}
        self.send_to_scheduler = _NoopSender()
        self.log_requests = False

    @staticmethod
    def _is_process_alive(pid: int) -> bool:
        del pid
        return False


class _FakeBatchRequestContainer:
    def __init__(self, requests: list[SimpleNamespace], stream: bool = False):
        self._requests = requests
        self.batch_size = len(requests)
        self.stream = stream
        self.parallel_sample_num = 1

    def __getitem__(self, index: int) -> SimpleNamespace:
        return self._requests[index]


class _FakeBatchHandleManager:
    _handle_batch_request = TokenizerManager._handle_batch_request

    def __init__(self, enable_batch_send: bool, enable_batch_encode: bool):
        self.server_args = SimpleNamespace(
            enable_tokenizer_batch_encode=enable_batch_encode,
            enable_tokenizer_batch_send=enable_batch_send,
        )
        self.sent_single = []
        self.sent_batch = []

    def _validate_batch_tokenization_constraints(self, batch_size, obj):
        del batch_size, obj
        return None

    async def _batch_tokenize_and_process(self, batch_size: int, obj):
        del obj
        return [SimpleNamespace(tokenized_idx=i) for i in range(batch_size)]

    async def _tokenize_one_request(self, obj):
        return SimpleNamespace(tokenized_single=obj.rid)

    def _send_one_request(self, obj, tokenized_obj, created_time=None):
        self.sent_single.append((obj.rid, tokenized_obj))
        return ReqState([], True, asyncio.Event(), obj, created_time=created_time)

    def _send_batch_requests(self, objs, tokenized_objs, created_time=None):
        self.sent_batch.append(([obj.rid for obj in objs], tokenized_objs))
        return [ReqState([], True, asyncio.Event(), obj, created_time=created_time) for obj in objs]

    async def _wait_one_response(self, obj, state, request=None):
        del state, request
        yield {"meta_info": {"id": obj.rid}, "text": "", "index": 0}


class _FakeIngressSocket:
    def __init__(self, payloads: list):
        self.payloads = list(payloads)

    def recv_pyobj(self, flags=None):
        del flags
        if self.payloads:
            return self.payloads.pop(0)
        raise zmq.ZMQError()


class _FakeSchedulerIngress:
    recv_requests = Scheduler.recv_requests

    def __init__(self, tokenizer_payloads: list, rpc_payloads: list):
        self.node_rank = 0
        self.nnodes = 1
        self.recv_from_tokenizer = _FakeIngressSocket(tokenizer_payloads)
        self.recv_from_rpc = _FakeIngressSocket(rpc_payloads)
        self.ingress_recv_calls = 0
        self.ingress_nonempty_calls = 0
        self.ingress_max_batch_size = 0
        self.ingress_tokenizer_frames = 0
        self.ingress_rpc_frames = 0
        self.ingress_tokenizer_messages = 0
        self.ingress_rpc_messages = 0
        self.ingress_batch_size_histogram = {
            "eq_0": 0,
            "eq_1": 0,
            "2_to_4": 0,
            "5_to_16": 0,
            "gt_16": 0,
        }
        self.ingress_score_paths = {
            "tokenizer_multi_item_packed": 0,
            "tokenizer_cache_for_scoring": 0,
            "tokenizer_extend_from_cache": 0,
            "rpc_score_from_cache_v2": 0,
            "rpc_release_scoring_cache": 0,
        }
        self.ingress_score_path_frames = {
            "tokenizer_multi_item_packed": 0,
            "tokenizer_cache_for_scoring": 0,
            "tokenizer_extend_from_cache": 0,
            "rpc_score_from_cache_v2": 0,
            "rpc_release_scoring_cache": 0,
        }


def test_create_tokenized_object_keeps_prefill_extend_fields():
    manager = _FakeCreateTokenizedManager()
    req = GenerateReqInput(
        rid="req-1",
        input_ids=[1, 2, 3],
        sampling_params={"max_new_tokens": 0},
        return_logprob=True,
        token_ids_logprob=[10, 20],
        cache_for_scoring=True,
        extend_from_cache="cache-handle-1",
        is_single=True,
    )
    req.normalize_batch_and_arguments()

    tokenized = manager._create_tokenized_object(req, input_text=None, input_ids=req.input_ids)

    assert tokenized.cache_for_scoring is True
    assert tokenized.extend_from_cache == "cache-handle-1"


def test_prefill_and_cache_uses_single_request_and_stable_handle():
    manager = _FakePrefillManager()
    handle = asyncio.run(manager._prefill_and_cache([11, 12, 13]))

    assert len(manager.seen_requests) == 1
    req = manager.seen_requests[0]
    assert req.input_ids == [11, 12, 13]
    assert req.is_single is True
    assert req.cache_for_scoring is True
    assert isinstance(req.rid, str)
    assert handle == req.rid


def test_batched_extend_score_passes_cache_handle_and_scores_items():
    manager = _FakeExtendManager()
    scores = asyncio.run(
        manager._batched_extend_score(
            cache_handle="cache-handle-xyz",
            items=[[1], [2]],
            label_token_ids=[10, 20],
            apply_softmax=False,
        )
    )

    assert manager.seen_request is not None
    assert manager.seen_request.extend_from_cache == "cache-handle-xyz"
    assert manager.seen_request.input_ids == [[1], [2]]
    assert manager.seen_request.return_logprob is True
    assert manager.seen_request.return_output_logprob_only is False
    assert manager.seen_request.token_ids_logprob == [10, 20]
    assert manager.seen_request.logprob_start_len is None
    assert len(scores) == 2
    assert scores[0] == pytest.approx([0.9, 0.1])
    assert scores[1] == pytest.approx([0.2, 0.8])


def test_batched_extend_score_raises_when_output_logprobs_missing():
    manager = _FakeExtendMissingLogprobsManager()
    with pytest.raises(RuntimeError, match="output_token_ids_logprobs is empty"):
        asyncio.run(
            manager._batched_extend_score(
                cache_handle="cache-handle-xyz",
                items=[[1]],
                label_token_ids=[10, 20],
                apply_softmax=False,
            )
        )


def test_release_cache_returns_true_on_success():
    manager = _FakeReleaseCacheManager(
        communicator=_FakeReleaseCacheCommunicator(
            outputs=[SimpleNamespace(success=True, released_items=1, error_msg="")]
        ),
        timeout_s=1.0,
    )
    assert asyncio.run(manager._release_cache("cache-handle-ok")) is True


def test_release_cache_times_out_and_returns_false():
    manager = _FakeReleaseCacheManager(
        communicator=_FakeReleaseCacheCommunicator(outputs=[], delay_s=0.2),
        timeout_s=0.01,
    )
    assert asyncio.run(manager._release_cache("cache-handle-timeout")) is False


class _FakeReqToTokenPool:
    def __init__(self, available_size: int):
        self._available_size = available_size

    def available_size(self) -> int:
        return self._available_size


class _FakeRunningBatch:
    def __init__(self):
        self.batch_is_full = False
        self.reqs = []

    def is_empty(self) -> bool:
        return len(self.reqs) == 0


class _FakeSchedulerCacheOps:
    _unpack_scoring_cache_entry = Scheduler._unpack_scoring_cache_entry
    _release_scoring_cache_entry = Scheduler._release_scoring_cache_entry
    _touch_scoring_cache_entry = Scheduler._touch_scoring_cache_entry
    _evict_expired_scoring_cache_nodes = Scheduler._evict_expired_scoring_cache_nodes
    _resolve_extend_from_cache = Scheduler._resolve_extend_from_cache
    _record_scoring_cache_lookup = Scheduler._record_scoring_cache_lookup
    _record_scoring_cache_handle_released = Scheduler._record_scoring_cache_handle_released

    def __init__(self, timeout_s: float):
        self.scoring_cache_timeout = timeout_s
        self._last_scoring_cache_gc = 0.0
        self.scoring_cache_nodes = {}
        self.scoring_cache_lookup_queries = 0
        self.scoring_cache_lookup_hits = 0
        self.scoring_cache_lookup_misses = 0
        self.scoring_cache_lookup_by_path = {
            "extend": {"queries": 0, "hits": 0, "misses": 0},
            "score_from_cache_v2": {"queries": 0, "hits": 0, "misses": 0},
        }
        self.scoring_cache_handles_released = 0
        self.scoring_cache_handles_released_manual = 0
        self.scoring_cache_handles_released_expired = 0
        self.scoring_cache_handles_released_other = 0
        self.scoring_cache_handles_missing_node = 0
        self.tree_cache = SimpleNamespace(dec_lock_ref=lambda *args, **kwargs: None)


def test_resolve_extend_from_cache_missing_handle_returns_error():
    scheduler = _FakeSchedulerCacheOps(timeout_s=60.0)
    recv_req = SimpleNamespace(
        rid="req-missing",
        input_ids=[101, 102],
        extra_key=None,
        extend_from_cache="missing-handle",
        sampling_params=SimpleNamespace(max_new_tokens=0),
    )

    cached_prefix_ctx, err = scheduler._resolve_extend_from_cache(recv_req)

    assert cached_prefix_ctx is None
    assert "Missing scoring cache handle" in err
    assert recv_req.input_ids == [101, 102]


def test_resolve_extend_from_cache_merges_prefix_and_suffix():
    scheduler = _FakeSchedulerCacheOps(timeout_s=0.0)
    scheduler.scoring_cache_nodes["cache-1"] = (
        "node",
        None,
        [1, 2, 3],
        np.array([0, 1, 2], dtype=np.int32),
        "extra-key",
        0.0,
    )
    recv_req = SimpleNamespace(
        rid="req-hit",
        input_ids=[11, 12],
        extra_key=None,
        extend_from_cache="cache-1",
        sampling_params=SimpleNamespace(max_new_tokens=0),
    )

    cached_prefix_ctx, err = scheduler._resolve_extend_from_cache(recv_req)

    assert err is None
    assert cached_prefix_ctx is not None
    assert recv_req.input_ids == [1, 2, 3, 11, 12]
    assert recv_req.extra_key == "extra-key"


def test_evict_expired_scoring_cache_nodes_removes_stale_entries():
    scheduler = _FakeSchedulerCacheOps(timeout_s=10.0)
    scheduler.scoring_cache_nodes["cache-stale"] = (
        None,
        None,
        [1, 2, 3],
        np.array([0, 1, 2], dtype=np.int32),
        None,
        0.0,
    )

    removed = scheduler._evict_expired_scoring_cache_nodes(now=20.0)
    assert removed == 1
    assert "cache-stale" not in scheduler.scoring_cache_nodes


def test_scheduler_req_slot_exhaustion_does_not_stick_batch_full():
    """Scheduler should defer prefill when req slots are exhausted without deadlocking future rounds."""
    scheduler = SimpleNamespace(
        grammar_queue=[],
        move_ready_grammar_requests=lambda: None,
        running_batch=_FakeRunningBatch(),
        waiting_queue=[object()],
        chunked_req=None,
        max_running_requests=8,
        req_to_token_pool=_FakeReqToTokenPool(available_size=0),
    )

    new_batch = Scheduler.get_new_batch_prefill(scheduler)
    assert new_batch is None
    assert scheduler.running_batch.batch_is_full is True

    # Simulate a later scheduler round after pressure is relieved.
    scheduler.req_to_token_pool._available_size = 1
    scheduler.waiting_queue = []
    new_batch = Scheduler.get_new_batch_prefill(scheduler)

    assert new_batch is None
    # Important for liveness: soft throttle clears when the running batch is idle.
    assert scheduler.running_batch.batch_is_full is False


def test_wait_one_response_fails_fast_when_scheduler_dies():
    manager = _FakeSchedulerLivenessManager()
    req_obj = SimpleNamespace(stream=False, rid="rid-1")
    state = ReqState([], False, asyncio.Event(), req_obj, created_time=0.0)
    manager.rid_to_state["rid-1"] = state

    async def _await_next():
        gen = manager._wait_one_response(req_obj, state, request=None)
        return await gen.__anext__()

    with pytest.raises(ValueError, match="Scheduler subprocess is unavailable"):
        asyncio.run(_await_next())

    assert manager.health_check_failed is True
    assert manager.scheduler_unavailable_error is not None
    assert state.finished is True
    assert state.event.is_set()


def test_send_one_request_fails_fast_when_scheduler_unavailable():
    manager = _FakeSchedulerLivenessManager()
    manager.scheduler_unavailable_error = "Scheduler subprocess is unavailable. Please restart."
    req = GenerateReqInput(
        rid="rid-2",
        input_ids=[1, 2, 3],
        sampling_params={"max_new_tokens": 0},
        is_single=True,
    )
    req.normalize_batch_and_arguments()

    with pytest.raises(ValueError, match="Scheduler subprocess is unavailable"):
        manager._send_one_request(req, tokenized_obj=SimpleNamespace(), created_time=0.0)

    assert manager.send_to_scheduler.calls == []


def test_send_batch_requests_sends_single_payload_and_tracks_all_states():
    manager = _FakeSchedulerLivenessManager()
    manager.scheduler_pids = []
    reqs = [SimpleNamespace(rid="rid-a"), SimpleNamespace(rid="rid-b")]
    tokenized_objs = [SimpleNamespace(tok=1), SimpleNamespace(tok=2)]

    states = manager._send_batch_requests(reqs, tokenized_objs, created_time=1.0)

    assert len(states) == 2
    assert set(manager.rid_to_state.keys()) == {"rid-a", "rid-b"}
    assert len(manager.send_to_scheduler.calls) == 1
    assert manager.send_to_scheduler.calls[0] == tokenized_objs


def test_send_batch_requests_raises_on_length_mismatch():
    manager = _FakeSchedulerLivenessManager()
    manager.scheduler_pids = []
    with pytest.raises(ValueError, match="same length"):
        manager._send_batch_requests([SimpleNamespace(rid="rid-a")], [], created_time=0.0)


def test_handle_batch_request_uses_single_send_when_batch_send_enabled():
    manager = _FakeBatchHandleManager(enable_batch_send=True, enable_batch_encode=True)
    obj = _FakeBatchRequestContainer(
        [SimpleNamespace(rid="rid-1"), SimpleNamespace(rid="rid-2")],
        stream=False,
    )

    async def _collect():
        outputs = []
        async for out in manager._handle_batch_request(obj, request=None, created_time=0.0):
            outputs.append(out)
        return outputs

    outputs = asyncio.run(_collect())
    assert len(outputs) == 1
    assert len(outputs[0]) == 2
    assert len(manager.sent_batch) == 1
    assert manager.sent_single == []


def test_handle_batch_request_uses_per_request_send_when_batch_send_disabled():
    manager = _FakeBatchHandleManager(enable_batch_send=False, enable_batch_encode=True)
    obj = _FakeBatchRequestContainer(
        [SimpleNamespace(rid="rid-1"), SimpleNamespace(rid="rid-2")],
        stream=False,
    )

    async def _collect():
        outputs = []
        async for out in manager._handle_batch_request(obj, request=None, created_time=0.0):
            outputs.append(out)
        return outputs

    outputs = asyncio.run(_collect())
    assert len(outputs) == 1
    assert len(outputs[0]) == 2
    assert manager.sent_batch == []
    assert len(manager.sent_single) == 2


def test_handle_batch_request_uses_single_send_without_batch_encode():
    manager = _FakeBatchHandleManager(enable_batch_send=True, enable_batch_encode=False)
    obj = _FakeBatchRequestContainer(
        [SimpleNamespace(rid="rid-1"), SimpleNamespace(rid="rid-2")],
        stream=False,
    )

    async def _collect():
        outputs = []
        async for out in manager._handle_batch_request(obj, request=None, created_time=0.0):
            outputs.append(out)
        return outputs

    outputs = asyncio.run(_collect())
    assert len(outputs) == 1
    assert len(outputs[0]) == 2
    assert len(manager.sent_batch) == 1
    assert manager.sent_single == []


def test_scheduler_recv_requests_unpacks_list_payload_into_logical_batch():
    tokenizer_payload = [
        TokenizedGenerateReqInput(
            rid="tok-1",
            input_ids=[1, 2],
            sampling_params={},
            cache_for_scoring=True,
            is_multi_item_scoring=True,
        ),
        TokenizedGenerateReqInput(
            rid="tok-2",
            input_ids=[1, 3],
            sampling_params={},
            extend_from_cache="cache-handle-1",
        ),
    ]
    rpc_payload = [
        ScoreFromCacheReqInput(
            rid="rpc-1",
            cache_handle="cache-handle-1",
            items_2d=[[7, 8]],
            label_token_ids=[198],
        ),
        ReleaseScoringCacheReqInput(rid="rpc-2"),
    ]
    scheduler = _FakeSchedulerIngress(
        tokenizer_payloads=[tokenizer_payload],
        rpc_payloads=[rpc_payload],
    )

    recv_reqs = scheduler.recv_requests()

    assert len(recv_reqs) == 4
    assert scheduler.ingress_tokenizer_frames == 1
    assert scheduler.ingress_rpc_frames == 1
    assert scheduler.ingress_tokenizer_messages == 2
    assert scheduler.ingress_rpc_messages == 2
    assert scheduler.ingress_nonempty_calls == 1
    assert scheduler.ingress_max_batch_size == 4
    assert scheduler.ingress_batch_size_histogram["2_to_4"] == 1
    assert scheduler.ingress_score_paths["tokenizer_multi_item_packed"] == 1
    assert scheduler.ingress_score_paths["tokenizer_cache_for_scoring"] == 1
    assert scheduler.ingress_score_paths["tokenizer_extend_from_cache"] == 1
    assert scheduler.ingress_score_paths["rpc_score_from_cache_v2"] == 1
    assert scheduler.ingress_score_paths["rpc_release_scoring_cache"] == 1
    assert scheduler.ingress_score_path_frames["tokenizer_multi_item_packed"] == 1
    assert scheduler.ingress_score_path_frames["tokenizer_cache_for_scoring"] == 1
    assert scheduler.ingress_score_path_frames["tokenizer_extend_from_cache"] == 1
    assert scheduler.ingress_score_path_frames["rpc_score_from_cache_v2"] == 1
    assert scheduler.ingress_score_path_frames["rpc_release_scoring_cache"] == 1


def test_scheduler_recv_requests_counts_score_control_reqs_on_tokenizer_socket():
    tokenizer_payload = [
        ScoreFromCacheReqInput(
            rid="score-1",
            cache_handle="cache-handle-2",
            items_2d=[[7, 8]],
            label_token_ids=[198],
        ),
        ReleaseScoringCacheReqInput(rid="release-1"),
    ]
    scheduler = _FakeSchedulerIngress(
        tokenizer_payloads=[tokenizer_payload],
        rpc_payloads=[],
    )

    recv_reqs = scheduler.recv_requests()

    assert len(recv_reqs) == 2
    assert scheduler.ingress_tokenizer_frames == 1
    assert scheduler.ingress_rpc_frames == 0
    assert scheduler.ingress_tokenizer_messages == 2
    assert scheduler.ingress_rpc_messages == 0
    assert scheduler.ingress_score_paths["rpc_score_from_cache_v2"] == 1
    assert scheduler.ingress_score_paths["rpc_release_scoring_cache"] == 1
    assert scheduler.ingress_score_path_frames["rpc_score_from_cache_v2"] == 1
    assert scheduler.ingress_score_path_frames["rpc_release_scoring_cache"] == 1


def test_mark_scheduler_unavailable_aborts_all_pending_requests():
    manager = _FakeSchedulerLivenessManager()
    state1 = ReqState([], False, asyncio.Event(), SimpleNamespace(stream=False), created_time=0.0)
    state2 = ReqState([], False, asyncio.Event(), SimpleNamespace(stream=False), created_time=0.0)
    manager.rid_to_state["rid-a"] = state1
    manager.rid_to_state["rid-b"] = state2

    manager._mark_scheduler_unavailable(
        "Scheduler subprocess is unavailable (dead pid(s): 4321). Please restart the server."
    )

    assert manager.health_check_failed is True
    assert manager.rid_to_state == {}
    for state in (state1, state2):
        assert state.finished is True
        assert state.event.is_set()
        finish_reason = state.out_list[-1]["meta_info"]["finish_reason"]
        assert finish_reason["type"] == "abort"
        assert "Scheduler subprocess is unavailable" in finish_reason["message"]


def test_token_ids_logprobs_handles_ragged_prefill_lengths():
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    all_logprobs = jax.numpy.array(
        [
            [0.10, 0.20, 0.70],
            [0.15, 0.25, 0.60],
            [0.20, 0.30, 0.50],
        ],
        dtype=jax.numpy.float32,
    )
    logits_metadata = SimpleNamespace(
        token_ids_logprobs=[[0, 2], [0, 2], [0, 2]],
        extend_logprob_pruned_lens_cpu=[0, 2, 1],
    )

    vals, idxs = LogitsProcessor.get_token_ids_logprobs(all_logprobs, logits_metadata, mesh)

    assert vals.shape == (3, 2, 2)
    assert idxs.shape == (3, 2, 2)
    np.testing.assert_array_equal(np.array(idxs[0]), np.array([[-1, -1], [-1, -1]]))
    np.testing.assert_allclose(np.array(vals[1]), np.array([[0.10, 0.70], [0.15, 0.60]]))
    np.testing.assert_allclose(np.array(vals[2][0]), np.array([0.20, 0.50]))
    np.testing.assert_array_equal(np.array(idxs[2][1]), np.array([-1, -1]))


def test_input_logprob_slicing_handles_nested_lists():
    req = SimpleNamespace(
        is_multi_item_scoring=False,
        multi_item_scoring_delimiter=None,
        input_token_logprobs=[],
        temp_input_top_logprobs_val=[],
        temp_input_top_logprobs_idx=[],
        temp_input_token_ids_logprobs_val=[],
        temp_input_token_ids_logprobs_idx=[],
        input_token_logprobs_val=None,
        top_logprobs_num=2,
        token_ids_logprob=[101, 202],
    )
    output = SimpleNamespace(
        input_token_logprobs=[0.1, 0.2, 0.3, 0.4],
        input_top_logprobs_val=[
            [[0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.7, 0.3, 0.0], [0.6, 0.4, 0.0]]
        ],
        input_top_logprobs_idx=[[[11, 12, -1], [21, 22, -1], [31, 32, -1], [41, 42, -1]]],
        input_token_ids_logprobs_val=[
            [[0.6, 0.4, 0.0], [0.7, 0.3, 0.0], [0.8, 0.2, 0.0], [0.9, 0.1, 0.0]]
        ],
        input_token_ids_logprobs_idx=[
            [[101, 202, -1], [101, 202, -1], [101, 202, -1], [101, 202, -1]]
        ],
    )

    SchedulerOutputProcessorMixin.add_input_logprob_return_values(
        self=SimpleNamespace(),
        i=0,
        req=req,
        output=output,
        logprob_pt=0,
        num_input_logprobs=2,
        last_prefill_chunk=False,
    )

    assert req.input_token_logprobs == [0.1, 0.2]
    assert req.temp_input_top_logprobs_val == [[[0.9, 0.1], [0.8, 0.2]]]
    assert req.temp_input_top_logprobs_idx == [[[11, 12], [21, 22]]]
    assert req.temp_input_token_ids_logprobs_val == [[[0.6, 0.4], [0.7, 0.3]]]
    assert req.temp_input_token_ids_logprobs_idx == [[[101, 202], [101, 202]]]


def test_sampler_token_ids_logprobs_handles_none_entries():
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    logprobs = jax.numpy.array(
        [
            [0.20, 0.30, 0.50],
            [0.60, 0.10, 0.30],
        ],
        dtype=jax.numpy.float32,
    )

    vals, idxs = sampler_get_token_ids_logprobs(logprobs, [None, [0, 2]], mesh)

    assert vals.shape == (2, 2)
    assert idxs.shape == (2, 2)
    np.testing.assert_array_equal(np.array(idxs[0]), np.array([-1, -1]))
    np.testing.assert_allclose(np.array(vals[1]), np.array([0.60, 0.30]))
    np.testing.assert_array_equal(np.array(idxs[1]), np.array([0, 2]))


class _FakeScorePrefillExtendManager:
    score_prefill_extend = TokenizerManager.score_prefill_extend
    _record_score_fastpath_fallback = TokenizerManager._record_score_fastpath_fallback

    def __init__(
        self, fastpath_enabled: bool, fastpath_output: ScoreFromCacheReqOutput | Exception
    ):
        self.server_args = SimpleNamespace(
            multi_item_extend_batch_size=64,
            multi_item_enable_score_from_cache_v2=fastpath_enabled,
            multi_item_score_fastpath_log_metrics=True,
        )
        self.fastpath_output = fastpath_output
        self.prefill_calls = 0
        self.fastpath_calls = 0
        self.baseline_calls: list[list[list[int]]] = []
        self.logged_metrics = []
        self.score_fastpath_attempted = 0
        self.score_fastpath_succeeded = 0
        self.score_fastpath_fallback = 0
        self.score_fastpath_fallback_reasons = {}

    async def _prefill_and_cache(self, query_tokens: list[int]) -> str:
        del query_tokens
        self.prefill_calls += 1
        return "cache-handle"

    async def _score_from_cache_fastpath_v2(
        self,
        cache_handle: str,
        items: list[list[int]],
        label_token_ids: list[int],
        apply_softmax: bool,
    ) -> ScoreFromCacheReqOutput:
        del cache_handle, items, label_token_ids, apply_softmax
        self.fastpath_calls += 1
        if isinstance(self.fastpath_output, Exception):
            raise self.fastpath_output
        return self.fastpath_output

    async def _batched_extend_score_with_metrics(
        self,
        cache_handle: str,
        items: list[list[int]],
        label_token_ids: list[int],
        apply_softmax: bool,
    ) -> tuple[list[list[float]], dict[str, float | int]]:
        del cache_handle, label_token_ids, apply_softmax
        self.baseline_calls.append(items)
        scores = [[float(item[0]), float(item[0]) + 0.1] for item in items]
        metrics = {
            "dispatch_count": 1,
            "queue_wait_s": 0.001,
            "device_compute_s": 0.002,
            "host_orchestration_s": 0.003,
            "lifecycle_requests_sent": len(items),
            "lifecycle_results_received": len(items),
        }
        return scores, metrics

    async def _release_cache(self, cache_handle: str) -> bool:
        del cache_handle
        return True

    def _maybe_log_score_path_metrics(self, metrics: dict):
        self.logged_metrics.append(metrics)


class _FakeSchedulerScoreFromCacheV2:
    score_from_cache_v2 = Scheduler.score_from_cache_v2
    _score_from_cache_v2_validate_items = Scheduler._score_from_cache_v2_validate_items
    _score_from_cache_v2_fallback_output = Scheduler._score_from_cache_v2_fallback_output
    _record_score_from_cache_v2_fallback = Scheduler._record_score_from_cache_v2_fallback
    _record_score_from_cache_v2_timing = Scheduler._record_score_from_cache_v2_timing
    _record_scoring_cache_lookup = Scheduler._record_scoring_cache_lookup
    _scoring_cache_metrics_snapshot = Scheduler._scoring_cache_metrics_snapshot
    _estimate_score_from_cache_v2_words = Scheduler._estimate_score_from_cache_v2_words
    _touch_scoring_cache_entry = Scheduler._touch_scoring_cache_entry
    _unpack_scoring_cache_entry = Scheduler._unpack_scoring_cache_entry

    def __init__(self):
        self.enable_overlap = False
        self.server_args = SimpleNamespace(
            multi_item_score_from_cache_v2_items_per_step=64,
            multi_item_score_label_only_logprob=False,
            allow_auto_truncate=False,
            max_running_requests=1024,
            device="tpu",
        )
        self.req_to_token_pool = SimpleNamespace(available_size=lambda: 1024)
        self.model_config = SimpleNamespace(hf_eos_token_id={2}, vocab_size=32000)
        self.max_req_len = 32768
        self.max_req_input_len = 32768
        self.scoring_cache_nodes = {
            "cache-ok": (
                "node",
                None,
                [101] * 2000,
                np.arange(2000, dtype=np.int32),
                None,
                0.0,
            )
        }
        self.scoring_cache_timeout = 0.0
        self._last_scoring_cache_gc = 0.0
        self.score_from_cache_v2_attempted = 0
        self.score_from_cache_v2_succeeded = 0
        self.score_from_cache_v2_fallback = 0
        self.score_from_cache_v2_fallback_reasons = {}
        self.score_from_cache_v2_queue_wait_s_total = 0.0
        self.score_from_cache_v2_device_compute_s_total = 0.0
        self.score_from_cache_v2_host_orchestration_s_total = 0.0
        self.score_from_cache_v2_queue_wait_s_max = 0.0
        self.score_from_cache_v2_device_compute_s_max = 0.0
        self.score_from_cache_v2_host_orchestration_s_max = 0.0
        self.scoring_cache_lookup_queries = 0
        self.scoring_cache_lookup_hits = 0
        self.scoring_cache_lookup_misses = 0
        self.scoring_cache_lookup_by_path = {
            "extend": {"queries": 0, "hits": 0, "misses": 0},
            "score_from_cache_v2": {"queries": 0, "hits": 0, "misses": 0},
        }
        self.scoring_cache_handles_created = 0
        self.scoring_cache_handles_released = 0
        self.scoring_cache_handles_released_manual = 0
        self.scoring_cache_handles_released_expired = 0
        self.scoring_cache_handles_released_other = 0
        self.scoring_cache_handles_missing_node = 0
        self.chunk_calls = []
        self.label_only_chunk_calls = []
        self.fail_next_chunk = False
        self.force_estimated_words = None

    def _evict_expired_scoring_cache_nodes(self):
        return 0

    def _run_score_from_cache_v2_chunk(
        self,
        cache_handle: str,
        chunk_items: list[list[int]],
        label_token_ids: list[int],
        apply_softmax: bool,
        cached_last_node,
        cached_prefix_indices,
        prefix_ids: list[int],
        cached_extra_key: str | None,
    ) -> tuple[list[list[float]], float, float]:
        del (
            cache_handle,
            label_token_ids,
            apply_softmax,
            cached_last_node,
            cached_prefix_indices,
            prefix_ids,
            cached_extra_key,
        )
        self.chunk_calls.append([item[0] for item in chunk_items])
        if self.fail_next_chunk:
            self.fail_next_chunk = False
            raise RuntimeError("synthetic chunk failure")
        return (
            [[float(item[0]), float(item[0]) + 1.0] for item in chunk_items],
            0.01,
            0.02,
        )

    def _estimate_score_from_cache_v2_words(self, prefix_len: int, items: list[list[int]]) -> int:
        if self.force_estimated_words is not None:
            return self.force_estimated_words
        return Scheduler._estimate_score_from_cache_v2_words(prefix_len, items)

    def _run_score_from_cache_v2_chunk_label_only(
        self,
        cache_handle: str,
        chunk_items: list[list[int]],
        label_token_ids: list[int],
        label_token_ids_arr,
        apply_softmax: bool,
        cached_last_node,
        cached_prefix_indices,
        prefix_ids: list[int],
        cached_extra_key: str | None,
    ) -> tuple[list[list[float]], float, float]:
        del (
            cache_handle,
            label_token_ids,
            label_token_ids_arr,
            apply_softmax,
            cached_last_node,
            cached_prefix_indices,
            prefix_ids,
            cached_extra_key,
        )
        self.label_only_chunk_calls.append([item[0] for item in chunk_items])
        return (
            [[float(item[0]), float(item[0]) + 0.5] for item in chunk_items],
            0.02,
            0.01,
        )


def _parity_metrics(
    baseline_scores: list[list[float]],
    fastpath_scores: list[list[float]],
) -> tuple[float, float]:
    diffs = []
    for base_row, fast_row in zip(baseline_scores, fastpath_scores):
        diffs.extend(abs(a - b) for a, b in zip(base_row, fast_row))
    return max(diffs), sum(diffs) / len(diffs)


def test_score_prefill_extend_fastpath_v2_500x20_order_and_count():
    expected_scores = [[float(i), float(i) + 0.5] for i in range(500)]
    manager = _FakeScorePrefillExtendManager(
        fastpath_enabled=True,
        fastpath_output=ScoreFromCacheReqOutput(
            success=True,
            scores=expected_scores,
            dispatch_count=8,
            queue_wait_s=0.01,
            device_compute_s=0.2,
            host_orchestration_s=0.05,
        ),
    )
    query_tokens = [11] * 2000
    items = [[i] * 20 for i in range(500)]

    scores = asyncio.run(
        manager.score_prefill_extend(
            query_tokens=query_tokens,
            item_tokens_list=items,
            label_token_ids=[9454, 2753],
            apply_softmax=False,
        )
    )

    assert len(scores) == 500
    assert scores == expected_scores
    assert manager.prefill_calls == 1
    assert manager.fastpath_calls == 1
    assert manager.baseline_calls == []
    assert manager.score_fastpath_attempted == 1
    assert manager.score_fastpath_succeeded == 1
    assert manager.score_fastpath_fallback == 0


def test_score_prefill_extend_fastpath_exception_falls_back_and_recovers():
    manager = _FakeScorePrefillExtendManager(
        fastpath_enabled=True,
        fastpath_output=RuntimeError("synthetic fastpath communicator failure"),
    )
    query_tokens = [7] * 2000
    items = [[i] * 20 for i in range(500)]

    scores = asyncio.run(
        manager.score_prefill_extend(
            query_tokens=query_tokens,
            item_tokens_list=items,
            label_token_ids=[9454, 2753],
            apply_softmax=False,
        )
    )

    assert len(scores) == 500
    assert manager.fastpath_calls == 1
    assert len(manager.baseline_calls) > 0
    assert manager.score_fastpath_attempted == 1
    assert manager.score_fastpath_succeeded == 0
    assert manager.score_fastpath_fallback == 1
    assert manager.score_fastpath_fallback_reasons.get("runtime_exception") == 1

    # Recovery sanity: a second request still succeeds.
    scores_2 = asyncio.run(
        manager.score_prefill_extend(
            query_tokens=query_tokens,
            item_tokens_list=items[:10],
            label_token_ids=[9454, 2753],
            apply_softmax=False,
        )
    )
    assert len(scores_2) == 10


def test_score_from_cache_v2_chunk_loop_dispatches_multiple_steps():
    scheduler = _FakeSchedulerScoreFromCacheV2()
    items = [[i] * 20 for i in range(150)]
    out = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            cache_handle="cache-ok",
            items_2d=items,
            label_token_ids=[9454, 2753],
            items_per_step=64,
            apply_softmax=False,
        )
    )

    assert out.success is True
    assert out.dispatch_count == 3
    assert len(out.scores) == 150
    assert out.scores[0] == [0.0, 1.0]
    assert out.scores[-1] == [149.0, 150.0]
    assert len(scheduler.chunk_calls) == 3
    assert scheduler.score_from_cache_v2_succeeded == 1


def test_score_from_cache_v2_caps_items_per_step_by_req_slots():
    scheduler = _FakeSchedulerScoreFromCacheV2()
    scheduler.server_args.max_running_requests = 24
    scheduler.req_to_token_pool = SimpleNamespace(available_size=lambda: 25)
    items = [[i] * 20 for i in range(50)]
    out = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            cache_handle="cache-ok",
            items_2d=items,
            label_token_ids=[9454, 2753],
            items_per_step=64,
            apply_softmax=False,
        )
    )

    assert out.success is True
    assert out.dispatch_count == 3
    assert [len(chunk) for chunk in scheduler.chunk_calls] == [24, 24, 2]


def test_score_from_cache_v2_reqpool_oversubscribe_flag_uses_available_slots(monkeypatch):
    scheduler = _FakeSchedulerScoreFromCacheV2()
    scheduler.server_args.max_running_requests = 24
    scheduler.req_to_token_pool = SimpleNamespace(available_size=lambda: 25)
    monkeypatch.setattr(
        scheduler_module,
        "SCORE_V2_ALLOW_REQPOOL_OVERSUBSCRIBE",
        True,
    )
    items = [[i] * 20 for i in range(50)]
    out = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            cache_handle="cache-ok",
            items_2d=items,
            label_token_ids=[9454, 2753],
            items_per_step=64,
            apply_softmax=False,
        )
    )

    assert out.success is True
    assert out.dispatch_count == 2
    assert [len(chunk) for chunk in scheduler.chunk_calls] == [25, 25]


def test_score_from_cache_v2_size_guard_fallback():
    scheduler = _FakeSchedulerScoreFromCacheV2()
    scheduler.force_estimated_words = np.iinfo(np.int32).max
    out = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            cache_handle="cache-ok",
            items_2d=[[1] * 20 for _ in range(4)],
            label_token_ids=[9454, 2753],
            items_per_step=4,
            apply_softmax=False,
        )
    )

    assert out.success is False
    assert out.fallback_reason == "size_guard"
    assert scheduler.score_from_cache_v2_fallback == 1
    assert scheduler.score_from_cache_v2_fallback_reasons.get("size_guard") == 1


def test_score_from_cache_v2_runtime_exception_does_not_poison_future_requests():
    scheduler = _FakeSchedulerScoreFromCacheV2()
    scheduler.fail_next_chunk = True
    first = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            cache_handle="cache-ok",
            items_2d=[[1] * 20 for _ in range(8)],
            label_token_ids=[9454, 2753],
            items_per_step=4,
            apply_softmax=False,
        )
    )
    assert first.success is False
    assert first.fallback_reason == "runtime_exception"

    second = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            cache_handle="cache-ok",
            items_2d=[[2] * 20 for _ in range(8)],
            label_token_ids=[9454, 2753],
            items_per_step=4,
            apply_softmax=False,
        )
    )
    assert second.success is True
    assert len(second.scores) == 8


def test_score_from_cache_v2_label_only_uses_dedicated_chunk_runner():
    scheduler = _FakeSchedulerScoreFromCacheV2()
    scheduler.server_args.multi_item_score_label_only_logprob = True
    items = [[i] * 20 for i in range(10)]
    out = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            cache_handle="cache-ok",
            items_2d=items,
            label_token_ids=[9454, 2753],
            items_per_step=4,
            apply_softmax=False,
        )
    )

    assert out.success is True
    assert out.dispatch_count == 3
    assert [len(chunk) for chunk in scheduler.label_only_chunk_calls] == [4, 4, 2]
    assert scheduler.chunk_calls == []


def test_score_from_cache_v2_label_only_rejects_unsupported_backend():
    scheduler = _FakeSchedulerScoreFromCacheV2()
    scheduler.server_args.multi_item_score_label_only_logprob = True
    scheduler.server_args.device = "metal"
    out = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            cache_handle="cache-ok",
            items_2d=[[1] * 20 for _ in range(4)],
            label_token_ids=[9454, 2753],
            items_per_step=4,
            apply_softmax=False,
        )
    )

    assert out.success is False
    assert out.fallback_reason == "unsupported_backend"


def test_score_from_cache_v2_updates_scoring_cache_lookup_counters_on_hit():
    scheduler = _FakeSchedulerScoreFromCacheV2()
    out = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            cache_handle="cache-ok",
            items_2d=[[1] * 20 for _ in range(4)],
            label_token_ids=[9454, 2753],
            items_per_step=4,
            apply_softmax=False,
        )
    )

    assert out.success is True
    metrics = scheduler._scoring_cache_metrics_snapshot()
    assert metrics["lookup_queries"] == 1
    assert metrics["lookup_hits"] == 1
    assert metrics["lookup_misses"] == 0
    assert metrics["lookup_by_path"]["score_from_cache_v2"]["queries"] == 1
    assert metrics["lookup_by_path"]["score_from_cache_v2"]["hits"] == 1
    assert metrics["lookup_hit_rate"] == 1.0


def test_score_from_cache_v2_updates_scoring_cache_lookup_counters_on_miss():
    scheduler = _FakeSchedulerScoreFromCacheV2()
    out = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            cache_handle="cache-missing",
            items_2d=[[1] * 20 for _ in range(4)],
            label_token_ids=[9454, 2753],
            items_per_step=4,
            apply_softmax=False,
        )
    )

    assert out.success is False
    assert out.fallback_reason == "missing_cache_handle"
    metrics = scheduler._scoring_cache_metrics_snapshot()
    assert metrics["lookup_queries"] == 1
    assert metrics["lookup_hits"] == 0
    assert metrics["lookup_misses"] == 1
    assert metrics["lookup_by_path"]["score_from_cache_v2"]["queries"] == 1
    assert metrics["lookup_by_path"]["score_from_cache_v2"]["misses"] == 1
    assert metrics["lookup_hit_rate"] == 0.0


def test_score_from_cache_v2_timing_counters_are_recorded():
    scheduler = _FakeSchedulerScoreFromCacheV2()
    out = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            cache_handle="cache-ok",
            items_2d=[[1] * 20 for _ in range(4)],
            label_token_ids=[9454, 2753],
            items_per_step=4,
            apply_softmax=False,
        )
    )

    assert out.success is True
    assert scheduler.score_from_cache_v2_attempted == 1
    assert scheduler.score_from_cache_v2_succeeded == 1
    assert scheduler.score_from_cache_v2_queue_wait_s_total >= 0.0
    assert scheduler.score_from_cache_v2_device_compute_s_total == pytest.approx(0.01)
    assert scheduler.score_from_cache_v2_host_orchestration_s_total == pytest.approx(0.02)
    assert scheduler.score_from_cache_v2_device_compute_s_max == pytest.approx(0.01)
    assert scheduler.score_from_cache_v2_host_orchestration_s_max == pytest.approx(0.02)

    before_queue_wait = scheduler.score_from_cache_v2_queue_wait_s_total
    miss_out = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            cache_handle="cache-missing",
            items_2d=[[1] * 20 for _ in range(4)],
            label_token_ids=[9454, 2753],
            items_per_step=4,
            apply_softmax=False,
        )
    )

    assert miss_out.success is False
    assert scheduler.score_from_cache_v2_attempted == 2
    assert scheduler.score_from_cache_v2_fallback == 1
    assert scheduler.score_from_cache_v2_queue_wait_s_total >= before_queue_wait
    # Missing-cache fallback records zero compute overhead.
    assert scheduler.score_from_cache_v2_device_compute_s_total == pytest.approx(0.01)
    assert scheduler.score_from_cache_v2_host_orchestration_s_total == pytest.approx(0.02)


def test_score_from_cache_v2_parity_metric_threshold():
    baseline_scores = [[0.1, 0.9], [0.3, 0.7], [0.8, 0.2]]
    fastpath_scores = [[0.1000004, 0.8999996], [0.3000001, 0.6999999], [0.8, 0.2]]
    max_abs_diff, mean_abs_diff = _parity_metrics(baseline_scores, fastpath_scores)
    assert max_abs_diff < 1e-3
    assert mean_abs_diff < 5e-4
