import dataclasses
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from sgl_jax.srt.managers.schedule_batch import (
    ModelWorkerBatch,
    ModelWorkerSamplingInfo,
)
from sgl_jax.srt.managers.scheduler import GenerationBatchResult, Scheduler
from sgl_jax.srt.managers.tp_worker_overlap_v2 import ModelWorkerOverlap
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.speculative.base_worker import BaseSpecWorker
from sgl_jax.srt.speculative.eagle_info import EagleDraftInput
from sgl_jax.srt.speculative.overlap_utils import resolve_spec_decode_token_ids
from sgl_jax.srt.speculative.overlap_v2 import (
    SpeculativePlan,
    execute_speculative_batch,
    pack_verified_tokens,
    snapshot_speculative_batch,
)
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

ALGORITHMS = ["EAGLE", "EAGLE3", "NEXTN", "DFLASH", "DSPARK"]


@pytest.mark.parametrize("grammar,penalty", [(False, False), (True, False), (False, True)])
def test_sampling_retirement_reads_real_per_rank_state(grammar, penalty):
    scheduler = object.__new__(Scheduler)
    sampling = SamplingBatchInfo.generate_for_precompile(1, 32)
    assert not hasattr(sampling, "grammars")
    sampling.penalizer_orchestrator = SimpleNamespace(is_required=penalty)
    info = SimpleNamespace(
        reqs=[SimpleNamespace(grammar=object() if grammar else None)],
        sampling_info=sampling,
    )
    scheduler.last_batch = SimpleNamespace(reqs_info=[info])
    assert scheduler._spec_sampling_needs_retirement() == (grammar or penalty)


def worker_batch(mode=ForwardMode.DECODE):
    values = {
        field.name: None
        for field in dataclasses.fields(ModelWorkerBatch)
        if field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING
    }
    values.update(
        bid=42,
        forward_mode=mode,
        input_ids=np.array([1, 0, 2, 0]),
        seq_lens=np.array([10, 0, 20, 0]),
        positions=np.array([9, 0, 19, 0]),
        out_cache_loc=np.arange(4),
        req_pool_indices=np.arange(4),
        cache_loc=np.arange(8).reshape(4, 2),
        real_bs=2,
        real_bs_per_dp=[1, 1],
        dp_size=2,
        per_dp_bs_size=2,
        logits_indices_selector=np.array([0, 2]),
        return_logprob=False,
        return_output_logprob_only=False,
        sampling_info=ModelWorkerSamplingInfo.generate_for_precompile(4, 32),
    )
    return ModelWorkerBatch(**values)


@pytest.fixture
def executor():
    worker = object.__new__(ModelWorkerOverlap)
    worker._executor = ThreadPoolExecutor(max_workers=1)
    worker._last_submission = None
    worker.get_precompile_paddings = lambda: ([], [], [])
    try:
        yield worker
    finally:
        worker.shutdown()


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_all_algorithms_admitted_by_v2(monkeypatch, algorithm):
    monkeypatch.setenv("SGLANG_JAX_OVERLAP_V2", "1")
    args = ServerArgs(
        model_path="target",
        speculative_algorithm=algorithm,
        speculative_draft_model_path="draft",
        speculative_eagle_topk=1 if algorithm in ("DFLASH", "DSPARK") else 2,
        speculative_num_steps=1 if algorithm in ("DFLASH", "DSPARK") else 3,
        speculative_num_draft_tokens=8,
        attention_backend="fa",
        grammar_backend="none",
    )
    args.check_server_args()
    if algorithm in ("EAGLE", "EAGLE3", "NEXTN"):
        monkeypatch.delenv("SGLANG_JAX_OVERLAP_V2")
        with pytest.raises(ValueError, match="Speculative overlap scheduler"):
            args.check_server_args()


def test_snapshots_remove_request_state_and_keep_sampling_ready():
    owner = threading.get_ident()

    class Grammar:
        finished = False
        allowed = 7

        def allocate_vocab_mask(self, **kwargs):
            return np.zeros((4, 1), dtype=np.int32)

        def is_terminated(self):
            return False

        def fill_vocab_mask(self, mask, index):
            assert threading.get_ident() == owner
            mask[index] = self.allowed

    batch = worker_batch()
    grammar = Grammar()
    batch.sampling_info.grammars = [grammar] * 4
    batch.sampling_info.penalizer_orchestrator = object()
    batch.sampling_info.sampling_info_done = threading.Event()
    batch.spec_info_padded = EagleDraftInput(
        future_indices=np.array([1, 2]), allocate_lens=np.array([16, 24])
    )
    snapshot = snapshot_speculative_batch(batch)
    batch.cache_loc[:] = 99
    batch.seq_lens[:] = 99
    batch.sampling_info.temperatures[:] = 99
    batch.spec_info_padded.allocate_lens[:] = 99
    grammar.allowed = 99
    assert snapshot is not batch
    np.testing.assert_array_equal(snapshot.cache_loc, np.arange(8).reshape(4, 2))
    np.testing.assert_array_equal(snapshot.seq_lens, [10, 0, 20, 0])
    np.testing.assert_array_equal(snapshot.spec_info_padded.allocate_lens, [16, 24])
    np.testing.assert_array_equal(snapshot.sampling_info.vocab_mask, [[7]] * 4)
    assert snapshot.sampling_info.temperatures[0] != 99
    assert snapshot.sampling_info.grammars is None
    assert snapshot.sampling_info.penalizer_orchestrator is None
    assert snapshot.sampling_info.sampling_info_done is None
    assert snapshot.launch_done is None
    # The old helper must not recreate an unset event in the submission thread.
    spec_worker = object.__new__(BaseSpecWorker)
    BaseSpecWorker._prepare_overlap_sampling_info(spec_worker, snapshot)
    assert spec_worker.cur_sampling_info is snapshot.sampling_info
    assert snapshot.sampling_info.sampling_info_done is None


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("fused", [False, True])
def test_all_algorithms_use_v2_without_normal_sampling(executor, algorithm, fused):
    owner = threading.get_ident()
    entered = threading.Event()
    release = threading.Event()
    calls = []
    mwb = worker_batch()
    algorithm = SpeculativeAlgorithm[algorithm]
    decode_relay = algorithm.is_dflash_family() or (
        fused and (algorithm.is_eagle3() or algorithm.is_nextn())
    )
    output = GenerationBatchResult(None, np.arange(16), None, None, 42, 0)
    output.next_draft_input = EagleDraftInput(
        topk_p=np.ones((2, 1)),
        topk_index=np.ones((2, 1), dtype=np.int32),
        hidden_states=np.ones((2, 2)),
        verified_id=np.array([3, 4]),
        allocate_lens=np.array([16, 24]),
        future_indices=np.array([0, 2]) if decode_relay else None,
        new_seq_lens=np.array([12, 0, 23, 0]),
    )
    output.accept_lens = np.array([2, 0, 3, 0])

    def run(batch, relay):
        assert threading.get_ident() != owner
        assert batch is not mwb
        calls.append("relay" if relay else "generic")
        entered.set()
        assert release.wait(5)
        # Real generic execution changes this field during target verification.
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        return (output, output.next_draft_input.new_seq_lens) if relay else output

    draft_worker = SimpleNamespace(
        speculative_num_draft_tokens=4,
        _can_use_fused_spec_decode=fused and algorithm.is_nextn(),
        _can_use_fused_eagle3_verify=fused and algorithm.is_eagle3(),
        _can_use_fused_spec_prefill=lambda batch: False,
        forward_batch_speculative_generation=lambda batch: run(batch, False),
        forward_batch_speculative_decode_overlap=lambda batch: run(batch, True),
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        dp_size=2,
        return_logprob=False,
        return_output_logprob_only=False,
        reqs_info=[
            SimpleNamespace(
                reqs=[object()], seq_lens=np.array([n]), spec_info=None, decoding_reqs=[]
            )
            for n in (10, 20)
        ],
        get_spec_model_worker_batch=lambda *args, **kwargs: mwb,
    )
    scheduler = object.__new__(Scheduler)
    scheduler.spec_algorithm = algorithm
    scheduler.enable_overlap = scheduler.enable_overlap_v2 = True
    scheduler.tp_worker = executor
    scheduler.draft_worker = draft_worker
    scheduler.forward_ct = 0
    scheduler.page_size = 1
    scheduler.server_args = SimpleNamespace(enable_static_lora=False)
    scheduler._profile_batch_predicate = lambda batch: None
    executor.launch_sample = lambda *args: pytest.fail("spec sampled twice")
    try:
        submission = scheduler._launch_speculative_batch(batch)
        assert entered.wait(5)
        assert not submission.future.done()
        assert all(info.spec_info is None for info in batch.reqs_info)
    finally:
        release.set()
    result = scheduler._finish_speculative_batch(batch, submission)
    assert calls == ["relay" if decode_relay else "generic"]
    assert all(info.spec_info is not None for info in batch.reqs_info)
    assert [info.spec_info.new_seq_lens.tolist() for info in batch.reqs_info] == [[12], [23]]
    if not decode_relay:
        assert [info.seq_lens.tolist() for info in batch.reqs_info] == [[12], [23]]
    np.testing.assert_array_equal(result.accept_lens, [2, 0, 3, 0])
    assert result.launch_result is None


def test_prefill_then_decode_share_fifo_and_failed_step_stops_following(executor):
    calls = []
    release = threading.Event()

    def prefill(batch):
        assert release.wait(5)
        calls.append("prefill")
        return "prefill result"

    def decode(batch):
        calls.append("decode")
        raise RuntimeError("spec failed")

    worker = SimpleNamespace(
        _can_use_fused_spec_prefill=lambda batch: True,
        forward_batch_speculative_prefill_overlap=prefill,
        forward_batch_speculative_decode_overlap=decode,
    )
    try:
        first = executor.launch_speculative(
            worker, worker_batch(ForwardMode.EXTEND), SpeculativePlan(False, True)
        )
        second = executor.launch_speculative(worker, worker_batch(), SpeculativePlan(True, False))
        third = executor.launch_speculative(worker, worker_batch(), SpeculativePlan(True, False))
        assert not second.future.done()
    finally:
        release.set()
    assert first.future.result(timeout=5) == ("prefill result", None)
    for submission in (second, third):
        with pytest.raises(RuntimeError, match="spec failed"):
            submission.wait()
    assert calls == ["prefill", "decode"]


def test_tree_paths_are_packed_in_acceptance_order_with_dp_padding():
    # Accepted tree nodes need not be consecutive: [0, 3, 4] and [10, 12].
    predictions = np.arange(20) + 100
    indices = np.array([0, 3, 4, -1, -1, -1, 10, 12, -1, -1, -1, -1])
    verified = np.zeros(12, dtype=np.int32)
    verified[indices >= 0] = predictions[indices[indices >= 0]]
    accepts = np.array([3, 0, 2, 0])
    tokens = pack_verified_tokens(verified, accepts, 5)
    batch = SimpleNamespace(
        per_dp_bs_size=2,
        dp_size=2,
        reqs_info=[SimpleNamespace(reqs=[object()]), SimpleNamespace(reqs=[object()])],
    )
    result = SimpleNamespace(next_token_ids=tokens, accept_lens=accepts)
    resolved, lengths = resolve_spec_decode_token_ids(result, batch, 5)
    assert resolved == [[100, 103, 104], [], [110, 112], []]
    assert lengths == [3, 0, 2, 0]


@pytest.mark.parametrize("stateful", [False, True])
def test_spec_loop_retires_and_drains_without_sampling(stateful):
    scheduler = object.__new__(Scheduler)
    scheduler.spec_algorithm = SpeculativeAlgorithm.NEXTN
    scheduler._comm_backend = None
    scheduler._engine_paused = False
    scheduler._pending_h2d = []
    scheduler.last_batch = None
    calls = []
    batch = SimpleNamespace(copy=lambda: batch)
    remaining = iter([batch, batch, None])
    scheduler.recv_requests = lambda: []
    scheduler.select_dp_for_request = lambda reqs: reqs
    scheduler.process_input_requests = lambda reqs: None
    scheduler.get_next_batch_to_run = lambda: (calls.append("prepare"), next(remaining))[1]
    scheduler._spec_sampling_needs_retirement = (
        lambda: stateful and scheduler.last_batch is not None
    )
    scheduler._launch_batch_forward = lambda batch: pytest.fail("normal forward used")
    scheduler._launch_batch_sample = lambda *args: pytest.fail("normal sample used")
    scheduler._launch_speculative_batch = lambda batch: (
        calls.append("submit"),
        SimpleNamespace(wait=lambda: calls.append("barrier")),
    )[1]
    scheduler._finish_speculative_batch = lambda *args: calls.append("publish")
    scheduler.process_batch_result = lambda *args: calls.append("retire")
    scheduler.on_idle = lambda: None
    with pytest.raises(StopIteration):
        scheduler._event_loop_overlap_v2()
    assert len(scheduler.result_queue) == 0
    assert calls[:4] == ["prepare", "submit", "publish", "barrier"]
    assert calls[4:9] == (
        ["retire", "prepare", "submit", "publish", "barrier"]
        if stateful
        else ["prepare", "submit", "retire", "publish", "barrier"]
    )
    assert calls.count("retire") == 2


def test_generic_execution_preserves_original_decode_mode():
    batch = worker_batch()
    output = SimpleNamespace(next_draft_input=EagleDraftInput(new_seq_lens=np.array([12])))

    def run(batch):
        batch.forward_mode = ForwardMode.DRAFT_EXTEND
        return output

    _, lengths = execute_speculative_batch(
        SimpleNamespace(forward_batch_speculative_generation=run),
        batch,
        SpeculativePlan(False, False),
    )
    np.testing.assert_array_equal(lengths, [12])


def test_generic_verify_returns_tree_path_tokens(monkeypatch):
    from sgl_jax.srt.speculative import base_worker

    # Exercise the real verify wrapper; mock only the model and tree kernel.
    worker = object.__new__(BaseSpecWorker)
    worker.speculative_num_steps = 2
    worker.speculative_num_draft_tokens = 5
    worker.mesh = None
    worker.server_args = SimpleNamespace(disable_overlap_schedule=False)
    indices = np.array([0, 3, 4, -1, -1, -1, 10, 12, -1, -1, -1, -1])
    predictions = np.arange(20) + 100
    verified = np.zeros(12, dtype=np.int32)
    verified[indices >= 0] = predictions[indices[indices >= 0]]
    accepts = np.array([3, 0, 2, 0])
    mwb = worker_batch()
    mwb.spec_sampling_prepared = True
    mwb.spec_algorithm = SpeculativeAlgorithm.EAGLE
    mwb.positions = np.arange(20)

    def prepare(batch):
        batch.seq_lens -= 1
        batch.forward_mode = ForwardMode.TARGET_VERIFY

    mwb.spec_info_padded = SimpleNamespace(
        prepare_for_verify=prepare,
        sample=lambda *args: (predictions, verified, accepts, indices),
    )
    logits = SimpleNamespace(next_token_logits=np.ones((20, 2)), hidden_states=np.ones((20, 2)))
    worker._target_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            attn_backend=SimpleNamespace(get_eagle_forward_metadata=lambda batch: None)
        ),
        forward_batch_generation=lambda *args, **kwargs: (logits, None, 0),
    )
    worker._draft_worker = SimpleNamespace(draft_model_runner=SimpleNamespace(rngs=None))
    monkeypatch.setattr(base_worker, "replicate_to_mesh", lambda mesh, *arrays: arrays)
    output = worker.verify(mwb, np.array([16, 24]))
    np.testing.assert_array_equal(output.next_token_ids.reshape(4, 5)[0, :3], [100, 103, 104])
    np.testing.assert_array_equal(output.next_token_ids.reshape(4, 5)[2, :2], [110, 112])
    np.testing.assert_array_equal(output.next_draft_input.new_seq_lens, [13, 0, 22, 0])


def test_generic_precompile_does_not_enter_relay_only_path(monkeypatch):
    import jax
    from jax.sharding import Mesh

    from sgl_jax.srt.speculative.eagle_worker import EAGLEWorker

    monkeypatch.setattr(EagleDraftInput, "ALLOC_LEN_PER_DECODE", 5)
    worker = object.__new__(EAGLEWorker)
    worker._can_use_fused_spec_decode = worker._can_use_fused_eagle3_verify = False
    worker.spec_relay_buffers = None
    worker.init_spec_relay_buffers()
    assert worker.spec_relay_buffers is None
    worker.server_args = SimpleNamespace(dp_size=2, dtype="float32")
    worker.speculative_algorithm = SpeculativeAlgorithm.EAGLE
    worker.precompile_bs_paddings = [4]
    worker.mesh = Mesh(np.array(jax.devices()), ("data",))
    worker.page_size = 1
    worker.topk = 2
    worker.speculative_num_steps = 2
    worker.speculative_num_draft_tokens = 5

    def dummy(*args, **kwargs):
        batch = worker_batch()
        batch.sampling_info.is_all_greedy = True
        return batch

    worker._draft_worker = SimpleNamespace(
        max_req_len=16,
        model_config=SimpleNamespace(hidden_size=2),
        compilation_manager=SimpleNamespace(_make_dummy_batch=dummy),
    )
    calls = []
    worker.forward_batch_speculative_decode_overlap = lambda batch: pytest.fail(
        "relay-only precompile"
    )
    worker.forward_batch_speculative_generation = lambda batch: calls.append(batch)
    worker.precompile_spec_decode()
    assert len(calls) == 1
    assert calls[0].speculative_eagle_topk == 2


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("terminal", ["eos", "abort"])
def test_spec_resource_retirement_waits_after_cpu_work(monkeypatch, algorithm, prefill, terminal):
    from sgl_jax.test.test_tp_worker_overlap_v2 import _make_result_processor

    scheduler, batch, result, reqs, grammar, resources, streamed = _make_result_processor(
        monkeypatch, prefill=prefill, terminal=terminal
    )
    scheduler.spec_algorithm = SpeculativeAlgorithm[algorithm]
    scheduler.draft_worker = SimpleNamespace(speculative_num_draft_tokens=4)
    scheduler.accept_token = scheduler.draft_token = 0
    scheduler.cum_spec_accept_length = scheduler.cum_spec_accept_count = 0
    scheduler.spec_num_forward_ct = 0
    batch.return_output_logprob_only = False
    for req in reqs:
        req.return_output_logprob_only = False
    if not prefill:
        first_token = 2 if terminal == "eos" else 5
        result.next_token_ids = np.array([first_token, 0, 0, 0, 6, 7, 0, 0])
        result.accept_lens = np.array([1, 2])

    def wait():
        assert reqs[0].finished()
        assert reqs[1].output_ids == ([6] if prefill else [6, 7])
        assert resources == streamed == []

    scheduler.process_batch_result(batch, result, SimpleNamespace(wait=wait))
    assert ("release", "rank-0") in resources
    assert streamed == [True]
