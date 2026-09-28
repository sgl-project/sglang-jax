import threading
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
from sgl_jax.srt.managers import scheduler_output_processor_mixin as output_processor
from sgl_jax.srt.managers.schedule_batch import (
    FINISH_ABORT,
    ModelWorkerSamplingInfo,
    Req,
)
from sgl_jax.srt.managers.scheduler import GenerationBatchResult, Scheduler
from sgl_jax.srt.managers.tp_worker_overlap_thread import ModelWorkerClient
from sgl_jax.srt.managers.tp_worker_overlap_v2 import ModelWorkerOverlap
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.sampling.sampling_params import SamplingParams


@pytest.fixture
def submission_worker():
    worker = object.__new__(ModelWorkerOverlap)
    worker._executor = ThreadPoolExecutor(max_workers=1)
    worker._last_submission = None
    try:
        yield worker
    finally:
        worker.shutdown()


def test_submit_returns_before_forward_and_preserves_relay_order(submission_worker):
    worker = submission_worker
    entered = threading.Event()
    release = threading.Event()
    calls = []
    owner = threading.get_ident()
    host_cache_loc = np.array([3, 4], dtype=np.int32)
    batch = SimpleNamespace(
        sampling_info=SimpleNamespace(grammars=None), cache_loc=host_cache_loc[:]
    )

    def forward(batch, metadata):
        assert threading.get_ident() != owner
        entered.set()
        assert release.wait(5)
        np.testing.assert_array_equal(batch.cache_loc, [3, 4])
        calls.append("forward")
        return SimpleNamespace(batch=batch)

    def sample(context):
        assert threading.get_ident() != owner
        calls.append("sample")
        return "result"

    worker._launch_forward = forward
    worker._launch_sample = sample
    try:
        first = worker.launch_forward(batch)
        assert entered.wait(5)
        host_cache_loc[:] = 99
        result = worker.launch_sample(first)
        following = worker.launch_forward(batch)
        assert not first.future.done()
        assert not result.done()
        assert not following.future.done()
    finally:
        release.set()
    assert result.result(timeout=5) == "result"
    following.future.result(timeout=5)
    following.wait()
    assert calls == ["forward", "sample", "forward"]


@pytest.mark.parametrize("failing_stage", ["forward", "sample"])
def test_failed_submission_stops_following_forward_and_unblocks_barrier(
    submission_worker, failing_stage
):
    worker = submission_worker
    calls = []
    batch = SimpleNamespace(sampling_info=SimpleNamespace(grammars=None), cache_loc=np.zeros(1))

    def forward(batch, metadata):
        calls.append("forward")
        if failing_stage == "forward":
            raise RuntimeError("submission failed")
        return SimpleNamespace(batch=batch)

    def sample(context):
        calls.append("sample")
        raise RuntimeError("submission failed")

    worker._launch_forward = forward
    worker._launch_sample = sample
    first = worker.launch_forward(batch)
    result = worker.launch_sample(first)
    following = worker.launch_forward(batch)
    with pytest.raises(RuntimeError, match="submission failed"):
        result.result(timeout=5)
    with pytest.raises(RuntimeError, match="submission failed"):
        following.future.result(timeout=5)
    with pytest.raises(RuntimeError, match="submission failed"):
        following.wait()
    assert calls == (["forward"] if failing_stage == "forward" else ["forward", "sample"])


def test_grammar_mask_is_snapshotted_by_scheduler_before_async_sample(submission_worker):
    worker = submission_worker
    release = threading.Event()
    owner = threading.get_ident()

    class Grammar:
        finished = False
        allowed = 1

        def allocate_vocab_mask(self, **kwargs):
            assert threading.get_ident() == owner
            return np.zeros((1, 1), dtype=np.int32)

        def is_terminated(self):
            return False

        def fill_vocab_mask(self, mask, index):
            assert threading.get_ident() == owner
            mask[index] = self.allowed

    grammar = Grammar()
    batch = SimpleNamespace(
        cache_loc=np.zeros(1),
        sampling_info=ModelWorkerSamplingInfo(
            temperatures=np.ones((1, 1)),
            top_ps=np.ones(1),
            top_ks=np.ones(1),
            min_ps=np.zeros(1),
            vocab_size=32,
            grammars=[grammar],
        ),
    )

    def forward(batch, metadata):
        assert release.wait(5)
        return SimpleNamespace(batch=batch)

    worker._launch_forward = forward
    worker._launch_sample = lambda context: context.batch.sampling_info.vocab_mask.copy()
    try:
        submission = worker.launch_forward(batch)
        result = worker.launch_sample(submission)
        grammar.allowed = 2
    finally:
        release.set()
    np.testing.assert_array_equal(result.result(timeout=5), [[1]])


def test_scheduler_can_prepare_placeholder_outputs_before_sampling_finishes(submission_worker):
    worker = submission_worker
    release = threading.Event()
    worker_batch = SimpleNamespace(
        sampling_info=SimpleNamespace(grammars=None),
        seq_lens=np.ones(4),
        cache_loc=np.zeros(1),
        bid=42,
    )
    batch = SimpleNamespace(return_logprob=False)

    def forward(batch, metadata):
        assert release.wait(5)
        return SimpleNamespace(batch=batch)

    worker._launch_forward = forward
    worker._launch_sample = lambda context: ("logits", [11, 0, 22, 0], 7)
    worker.resolve_last_batch_result = lambda logits, ids, batch, misses: (
        logits,
        ids,
        misses,
    )
    scheduler = object.__new__(Scheduler)
    scheduler.tp_worker = worker
    scheduler.enable_overlap_v2 = True
    placeholders = []
    scheduler._extract_dp_output_ids = lambda ids, wb, batch: placeholders.append(ids)
    try:
        submission = worker.launch_forward(worker_batch)
        result = scheduler._launch_batch_sample(batch, submission)
        assert isinstance(result, GenerationBatchResult)
        assert not result.launch_result.done()
        assert result.next_token_ids is None
        np.testing.assert_array_equal(placeholders[0], [0, 0, 0, 0])
    finally:
        release.set()
    assert scheduler._resolve_overlap_v2_result(result) == ("logits", [11, 0, 22, 0], 7)
    assert result.launch_result is None


def _make_result_processor(monkeypatch, *, prefill, terminal):
    reqs = [
        Req(
            rid=f"rank-{rank}",
            origin_input_text="",
            origin_input_ids=[1],
            sampling_params=SamplingParams(max_new_tokens=8),
            dp_rank=rank,
            eos_token_ids={2},
            vocab_size=100,
            return_output_logprob_only=True,
        )
        for rank in range(2)
    ]
    grammar_tokens = []
    for rank, req in enumerate(reqs):
        req.req_pool_idx = rank + 1
        req.grammar = SimpleNamespace(
            finished=False,
            is_terminated=lambda: False,
            accept_token=lambda token, rank=rank: grammar_tokens.append((rank, token)),
        )
    if terminal == "abort":
        reqs[0].to_finish = FINISH_ABORT()
    tokens = [2 if terminal == "eos" else 5, 6]
    batch = SimpleNamespace(
        dp_size=2,
        per_dp_bs_size=1,
        forward_mode=ForwardMode.EXTEND if prefill else ForwardMode.DECODE,
        reqs_info=[SimpleNamespace(reqs=[req], decoding_reqs=[]) for req in reqs],
        return_hidden_states=False,
        return_logprob=False,
        return_output_logprob_only=True,
        batch_size=lambda: 2,
        next_batch_sampling_info=None,
    )
    result = GenerationBatchResult(
        logits_output=SimpleNamespace(
            next_token_logprobs=[-0.1, -0.2], input_token_logprobs=None, hidden_states=None
        ),
        next_token_ids=tokens,
        extend_input_len_per_req=None,
        extend_logprob_start_len_per_req=None,
        bid=42,
        cache_miss_count=0,
    )
    scheduler = object.__new__(Scheduler)
    scheduler.enable_overlap = scheduler.enable_overlap_v2 = True
    scheduler.spec_algorithm = None
    scheduler.is_generation = True
    scheduler.pd = ""
    scheduler.num_generated_tokens = scheduler.forward_ct_decode = 0
    scheduler.server_args = SimpleNamespace(decode_log_interval=100)
    scheduler._resolve_overlap_v2_result = lambda result: (
        result.logits_output,
        result.next_token_ids,
        result.cache_miss_count,
    )
    resources = []
    scheduler.tree_cache = SimpleNamespace(
        cache_unfinished_req=lambda req: resources.append(("cache", req.rid))
    )
    scheduler.maybe_collect_routed_experts = lambda req: resources.append(("experts", req.rid))
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(
        free_group_begin=lambda: None, free_group_end=lambda: None
    )
    monkeypatch.setattr(
        output_processor,
        "_complete_precision_trace",
        lambda req: resources.append(("trace", req.rid)),
    )
    monkeypatch.setattr(
        output_processor,
        "release_kv_cache",
        lambda req, *args, **kwargs: resources.append(("release", req.rid)),
    )
    streamed = []
    scheduler.stream_output = lambda *args, **kwargs: streamed.append(True)
    return scheduler, batch, result, reqs, grammar_tokens, resources, streamed


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("terminal", ["eos", "abort"])
@pytest.mark.parametrize("submission_fails", [False, True])
def test_cpu_results_precede_resource_barrier(monkeypatch, prefill, terminal, submission_fails):
    scheduler, batch, result, reqs, grammar, resources, streamed = _make_result_processor(
        monkeypatch, prefill=prefill, terminal=terminal
    )
    reached_barrier = threading.Event()
    submitted = Future()

    def wait():
        reached_barrier.set()
        submitted.result(timeout=5)

    with ThreadPoolExecutor(max_workers=1) as executor:
        processing = executor.submit(
            scheduler.process_batch_result, batch, result, SimpleNamespace(wait=wait)
        )
        try:
            assert reached_barrier.wait(5)
            # A terminal request on the first DP rank must not delay CPU work
            # on the second rank, but shared resources and outputs stay fenced.
            assert [req.output_ids for req in reqs] == [[result.next_token_ids[0]], [6]]
            assert [req.output_token_logprobs_val for req in reqs] == [[-0.1], [-0.2]]
            assert grammar == [(0, result.next_token_ids[0]), (1, 6)]
            assert reqs[0].finished() and not reqs[1].finished()
            assert not processing.done()
            assert resources == streamed == []
        finally:
            if submission_fails:
                submitted.set_exception(RuntimeError("submission failed"))
            else:
                submitted.set_result(None)
        if submission_fails:
            with pytest.raises(RuntimeError, match="submission failed"):
                processing.result(timeout=5)
            assert resources == streamed == []
        else:
            processing.result(timeout=5)
            assert resources == [
                ("experts", "rank-0"),
                ("trace", "rank-0"),
                ("release", "rank-0"),
            ] + ([("cache", "rank-1")] if prefill else [])
            assert streamed == [True]


def test_live_decode_does_not_wait_for_current_submission(monkeypatch):
    scheduler, batch, result, reqs, grammar, resources, streamed = _make_result_processor(
        monkeypatch, prefill=False, terminal=None
    )

    def unexpected_wait():
        pytest.fail("CPU-only result processing waited for current forward submission")

    scheduler.process_batch_result(batch, result, SimpleNamespace(wait=unexpected_wait))
    assert [req.output_ids for req in reqs] == [[5], [6]]
    assert grammar == [(0, 5), (1, 6)]
    assert resources == []
    assert streamed == [True]


def test_chunk_abort_keeps_resources_owned_until_after_cpu_bookkeeping(monkeypatch):
    scheduler, batch, result, reqs, grammar, resources, streamed = _make_result_processor(
        monkeypatch, prefill=True, terminal="abort"
    )
    reqs[0].is_chunked = 1
    scheduler.chunked_reqs = [reqs[0], None]
    scheduler._pending_chunked_abort_reqs = [reqs[0], None]
    scheduler._release_prefill_host_buffer = lambda req: resources.append(("host", req.rid))

    def wait():
        assert reqs[0].is_chunked == 0
        assert not reqs[0].finished()
        assert scheduler.chunked_reqs[0] is reqs[0]
        assert scheduler._pending_chunked_abort_reqs[0] is reqs[0]
        assert reqs[1].output_ids == [6]
        assert reqs[1].output_token_logprobs_val == [-0.2]
        assert grammar == [(1, 6)]
        assert resources == streamed == []

    scheduler.process_batch_result(batch, result, SimpleNamespace(wait=wait))
    assert reqs[0].finished()
    assert scheduler.chunked_reqs == scheduler._pending_chunked_abort_reqs == [None, None]
    assert resources == [
        ("trace", "rank-0"),
        ("host", "rank-0"),
        ("release", "rank-0"),
        ("cache", "rank-1"),
    ]
    assert streamed == [True]


def test_sample_enqueues_before_barrier_and_next_round_waits():
    scheduler = object.__new__(Scheduler)
    calls = []
    waited = False
    batch = SimpleNamespace(copy=lambda: batch)

    def wait():
        nonlocal waited
        assert calls == ["receive", "input", "prepare", "forward", "result", "sample"]
        calls.append("barrier")
        waited = True

    def receive():
        if waited:
            raise StopIteration
        calls.append("receive")
        return []

    scheduler._comm_backend = None
    scheduler.recv_requests = receive
    scheduler.select_dp_for_request = lambda reqs: reqs
    scheduler.process_input_requests = lambda reqs: calls.append("input")
    scheduler._engine_paused = False
    scheduler.get_next_batch_to_run = lambda: (calls.append("prepare"), batch)[1]
    scheduler._pending_h2d = []
    scheduler._launch_batch_forward = lambda batch: (
        calls.append("forward"),
        SimpleNamespace(wait=wait),
    )[1]
    scheduler.last_batch = object()

    def process(batch, result, barrier):
        calls.append("result")
        assert not waited

    # The loop owns the deque; seed one previous result at the forward boundary.
    forward = scheduler._launch_batch_forward

    def launch(batch):
        scheduler.result_queue.append((scheduler.last_batch, object()))
        return forward(batch)

    scheduler._launch_batch_forward = launch
    scheduler.process_batch_result = process
    scheduler._launch_batch_sample = lambda batch, context: calls.append("sample")
    with pytest.raises(StopIteration):
        scheduler._event_loop_overlap_v2()
    assert calls[-1] == "barrier"


def _make_logits_output():
    return LogitsProcessorOutput(
        next_token_logits=jnp.zeros((4, 8), dtype=jnp.float32),
        hidden_states=jnp.arange(8, dtype=jnp.float32).reshape(4, 2),
        next_token_logprobs=jnp.array([-0.1, -1.0, -0.2, -2.0], dtype=jnp.float32),
        input_token_logprobs=jnp.array([-0.3, -0.4, -0.5], dtype=jnp.float32),
        next_token_top_logprobs_val=jnp.array(
            [
                [-0.1, -0.2, -0.3],
                [-1.0, -1.1, -1.2],
                [-0.4, -0.5, -0.6],
                [-2.0, -2.1, -2.2],
            ],
            dtype=jnp.float32,
        ),
        next_token_top_logprobs_idx=jnp.array(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
            dtype=jnp.int32,
        ),
        next_token_token_ids_logprobs_val=jnp.arange(24, dtype=jnp.float32).reshape(4, 6),
    )


def _make_batch():
    return SimpleNamespace(
        return_logprob=True,
        return_output_logprob_only=False,
        logits_indices_selector=np.array([0, 2], dtype=np.int32),
        top_logprobs_nums=[2, 0, 1, 0],
        token_ids_logprobs=[[1, 3], None, [2], None],
    )


def _resolve_with_legacy_path(logits_output, next_token_ids, batch):
    worker = object.__new__(ModelWorkerOverlap)
    worker._materialize_logprobs_to_host(
        logits_output,
        batch,
        batch.logits_indices_selector,
    )

    client = object.__new__(ModelWorkerClient)
    client.output_queue = Queue()
    client.output_queue.put((None, logits_output, next_token_ids, 7))
    launch_done = threading.Event()
    launch_done.set()
    return client.resolve_last_batch_result(launch_done)


def _resolve_with_v2_path(logits_output, next_token_ids, batch):
    worker = object.__new__(ModelWorkerOverlap)
    return worker.resolve_last_batch_result(
        logits_output,
        next_token_ids,
        batch,
        7,
    )


def test_v2_resolver_matches_legacy_host_output_contract():
    next_token_ids = jnp.array([11, 0, 22, 0], dtype=jnp.int32)
    legacy_logits, legacy_ids, legacy_misses = _resolve_with_legacy_path(
        _make_logits_output(), next_token_ids, _make_batch()
    )
    v2_logits, v2_ids, v2_misses = _resolve_with_v2_path(
        _make_logits_output(), next_token_ids, _make_batch()
    )

    assert v2_ids == legacy_ids
    assert v2_misses == legacy_misses
    assert isinstance(v2_logits.next_token_logprobs, list)
    assert isinstance(v2_logits.input_token_logprobs, list)
    np.testing.assert_allclose(
        v2_logits.next_token_logprobs,
        legacy_logits.next_token_logprobs,
    )
    np.testing.assert_allclose(
        v2_logits.input_token_logprobs,
        legacy_logits.input_token_logprobs,
    )
    assert v2_logits.next_token_top_logprobs_val == legacy_logits.next_token_top_logprobs_val
    assert v2_logits.next_token_top_logprobs_idx == legacy_logits.next_token_top_logprobs_idx
    assert (
        v2_logits.next_token_token_ids_logprobs_val
        == legacy_logits.next_token_token_ids_logprobs_val
    )
    assert (
        v2_logits.next_token_token_ids_logprobs_idx
        == legacy_logits.next_token_token_ids_logprobs_idx
    )
    np.testing.assert_array_equal(v2_logits.hidden_states, legacy_logits.hidden_states)
