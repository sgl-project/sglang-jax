import asyncio
import math
from types import SimpleNamespace

import numpy as np
import pytest

from sgl_jax.srt.managers.io_struct import GenerateReqInput
from sgl_jax.srt.managers.scheduler import Scheduler
from sgl_jax.srt.managers.tokenizer_manager import TokenizerManager


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

    def __init__(self, timeout_s: float):
        self.scoring_cache_timeout = timeout_s
        self._last_scoring_cache_gc = 0.0
        self.scoring_cache_nodes = {}
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
