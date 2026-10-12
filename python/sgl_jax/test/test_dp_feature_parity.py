"""CPU regressions for DP request/token padding across optional features."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.constrained.bitmask_ops import (
    allocate_token_bitmask,
    apply_token_bitmask,
)
from sgl_jax.srt.lora.backend.bgmv_backend import BgmvLoRABackend, bgmv_shrink
from sgl_jax.srt.managers.io_struct import GenerateReqInput
from sgl_jax.srt.managers.schedule_batch import Req, ScheduleBatch
from sgl_jax.srt.managers.scheduler_output_processor_mixin import _collect_hidden_states
from sgl_jax.srt.managers.tokenizer_manager import TokenizerManager
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.sampling.sampling_params import SamplingParams


def make_req(rid, *, hidden=False, lora=None):
    return Req(
        str(rid),
        None,
        [1, 2, 3, 4],
        SamplingParams(max_new_tokens=4),
        return_hidden_states=hidden,
        lora_id=lora,
    )


def make_batch(reqs, mode=ForwardMode.EXTEND):
    batch = ScheduleBatch.init_new(reqs, None, None, None, None, True, len(reqs))
    batch.forward_mode = mode
    return batch


def test_hidden_states_request_propagation():
    request = GenerateReqInput(input_ids=[[1, 2], [3, 4]], return_hidden_states=True)
    request.normalize_batch_and_arguments()
    manager = SimpleNamespace(
        preferred_sampling_params=None,
        tokenizer=None,
        model_config=SimpleNamespace(vocab_size=64),
        server_args=SimpleNamespace(disaggregation_mode="null"),
    )
    for i in range(2):
        item = request[i]
        tokenized = TokenizerManager._create_tokenized_object(manager, item, None, item.input_ids)
        assert tokenized.return_hidden_states
    batch = make_batch([[make_req(0)], [make_req(1, hidden=True)]])
    assert batch.return_hidden_states
    assert batch.copy().return_hidden_states


@pytest.mark.parametrize("counts", [[1, 1], [0, 2], [2, 0, 1, 1]])
def test_hidden_states_prefill_chunks_and_decode(counts):
    reqs = [[make_req(f"{r}-{i}", hidden=i == 0) for i in range(n)] for r, n in enumerate(counts)]
    batch = make_batch(reqs)
    for info in batch.reqs_info:
        info.extend_lens = [2] * len(info.reqs)
        info.prefix_lens = [0] * len(info.reqs)
        info.seq_lens = np.full(len(info.reqs), 2, np.int32)
    rows = np.arange(len(counts) * 8, dtype=np.float32).reshape(-1, 1)
    snapshot = batch.copy()
    _collect_hidden_states(snapshot, rows)
    for rank, info in enumerate(batch.reqs_info):
        for i, req in enumerate(info.reqs):
            assert req.hidden_states == (rows[rank * 8 : rank * 8 + 2].tolist() if i == 0 else [])
        info.prefix_lens = [2] * len(info.reqs)
        info.extend_lens = [1] * len(info.reqs)
        info.seq_lens[:] = 3
    # Output snapshots must not track subsequent overlap scheduler updates.
    assert all(np.all(info.seq_lens == 2) for info in snapshot.reqs_info)
    _collect_hidden_states(batch.copy(), rows + 100)
    # Re-prefill replaces an existing suffix; it does not duplicate rows.
    _collect_hidden_states(batch.copy(), rows + 100)
    batch.forward_mode = ForwardMode.DECODE
    for info in batch.reqs_info:
        info.seq_lens[:] = 4
    _collect_hidden_states(batch.copy(), rows + 200)
    for rank, info in enumerate(batch.reqs_info):
        if info.reqs:
            assert info.reqs[0].hidden_states == [
                [rank * 8],
                [rank * 8 + 1],
                [rank * 8 + 100],
                [rank * 8 + 200],
            ]
            for req in info.reqs[1:]:
                assert req.hidden_states == []


def test_hidden_states_cache_prefix_requires_captured_rows():
    req = make_req(0, hidden=True)
    req.fill_ids = [1, 2, 3, 4]
    assert req.adjust_max_prefix_ids() == []
    req.hidden_states = [[1.0], [2.0]]
    assert req.adjust_max_prefix_ids() == [1, 2]


@pytest.mark.parametrize("counts", [[1, 1], [0, 2], [2, 0, 1, 1]])
@pytest.mark.parametrize("mode", [ForwardMode.EXTEND, ForwardMode.DECODE])
def test_lora_adapter_and_token_slots(counts, mode):
    reqs = [[make_req(f"{r}-{i}", lora=str(r + 1)) for i in range(n)] for r, n in enumerate(counts)]
    batch = make_batch(reqs, mode)
    per_dp_bs, per_dp_tokens = 3, 8 if mode.is_extend() else 3
    ids = batch._merge_lora_ids(per_dp_bs, len(counts) * per_dp_bs, False)
    expected_ids = [
        str(r + 1) if i < n else "0" for r, n in enumerate(counts) for i in range(per_dp_bs)
    ]
    assert ids == expected_ids
    assert batch._merge_lora_ids(per_dp_bs, len(ids), True) == ["0"] * len(ids)
    lengths = [i + 1 if i < n else 0 for n in counts for i in range(per_dp_bs)]
    mwb = SimpleNamespace(
        forward_mode=mode,
        dp_size=len(counts),
        seq_lens=np.array(lengths),
        extend_seq_lens=np.array(lengths),
        input_ids=np.zeros(len(counts) * per_dp_tokens, np.int32),
    )
    weights = list(map(int, ids))
    ranks = [0] + [2] * len(counts)
    scalings = [0] + [r + 0.5 for r in range(len(counts))]
    BgmvLoRABackend.prepare_lora_batch(None, mwb, weights, ranks, scalings)
    expected = []
    for r, n in enumerate(counts):
        local = [r + 1] * (sum(range(1, n + 1)) if mode.is_extend() else n)
        expected.extend(local + [0] * (per_dp_tokens - len(local)))
    np.testing.assert_array_equal(mwb.lora_token_indices, expected)
    np.testing.assert_array_equal(mwb.lora_scalings, np.array(scalings)[expected])
    np.testing.assert_array_equal(mwb.lora_ranks, np.array(ranks)[expected])
    # Exercise the production adapter kernel on CPU with distinct weights.
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",))
    sharding = NamedSharding(mesh, P())
    x = jnp.ones((len(expected), 2), dtype=jnp.float32)
    w = jnp.broadcast_to(jnp.arange(len(counts) + 1)[:, None, None], (len(counts) + 1, 2, 2))
    actual = bgmv_shrink(
        x,
        w.astype(x.dtype),
        jnp.array(mwb.lora_token_indices),
        sharding,
        jnp.array(mwb.lora_scalings),
    )
    want = 2 * np.array(expected) * np.array(mwb.lora_scalings)
    np.testing.assert_allclose(actual, np.repeat(want[:, None], 2, axis=1))


class SingleTokenGrammar:
    finished = False

    def __init__(self, token):
        self.token = token
        self.mask = None

    def is_terminated(self):
        return False

    def allocate_vocab_mask(self, vocab_size, batch_size):
        if self.mask is None:
            self.mask = allocate_token_bitmask(batch_size, vocab_size)
        return self.mask

    def fill_vocab_mask(self, mask, idx):
        mask[idx, self.token // 32] = 1 << (self.token % 32)


@pytest.mark.parametrize("counts", [[1, 1], [0, 2], [2, 0, 1, 1]])
def test_grammar_masks_follow_dp_slots_and_preserve_unconstrained_rows(counts):
    reqs = [[make_req(f"{r}-{i}") for i in range(n)] for r, n in enumerate(counts)]
    for r, group in enumerate(reqs):
        if group:
            group[0].grammar = SingleTokenGrammar(r + 1)
    batch = make_batch(reqs)
    merged = batch._merge_sampling_info(3, len(counts) * 3)
    merged.vocab_size = 64
    merged.update_grammar_vocab_mask()
    original = merged.vocab_mask.copy()
    logits = np.tile(np.arange(64, dtype=np.float32), (len(counts) * 3, 1))
    masked = np.asarray(apply_token_bitmask(jnp.array(logits), jnp.array(merged.vocab_mask)))
    for rank, count in enumerate(counts):
        for slot in range(3):
            row = rank * 3 + slot
            if count and slot == 0:
                assert masked[row].argmax() == rank + 1
                assert np.isfinite(masked[row]).sum() == 1
            else:
                np.testing.assert_array_equal(masked[row], logits[row])
    # A later grammar step cannot overwrite a mask awaiting device transfer.
    for group in reqs:
        if group:
            group[0].grammar.token += 8
    previous = merged.vocab_mask
    merged.update_grammar_vocab_mask()
    np.testing.assert_array_equal(previous, original)
    assert not np.array_equal(previous, merged.vocab_mask)


def test_hidden_states_output_alignment_and_diffusion_compatibility():
    from sgl_jax.srt.managers.scheduler_output_processor_mixin import (
        SchedulerOutputProcessorMixin,
    )

    requests = [make_req(0, hidden=True), make_req(1), make_req(2, hidden=True)]
    for req in requests:
        req.stream = True
        req.output_ids = [5]
        req.hidden_states = [[float(i)] for i in range(4)] if req.return_hidden_states else []
    outputs = []
    scheduler = SimpleNamespace(
        stream_interval=1,
        skip_tokenizer_init=True,
        spec_algorithm=None,
        _comm_backend=None,
        send_to_detokenizer=SimpleNamespace(send_pyobj=outputs.append),
    )
    SchedulerOutputProcessorMixin.stream_output_generation(
        scheduler, requests, False, False, skip_reqs={id(requests[0])}
    )
    result = outputs[0]
    assert result.rids == ["1", "2"]
    assert result.output_hidden_states == [None, [[0.0], [1.0], [2.0], [3.0]]]
    assert result.output_hidden_states_for_mm == [None, [[[0.0], [1.0], [2.0], [3.0]]]]
    requests[2].hidden_states[0] = [99.0]
    assert result.output_hidden_states[1][0] == [0.0]


def test_hidden_states_skipped_requests_still_advance_token_offset():
    first = make_req(0, hidden=True)
    first.is_retracted = True
    second = make_req(1, hidden=True)
    batch = make_batch([[], [first, second]])
    batch.reqs_info[1].extend_lens = [3, 2]
    batch.reqs_info[1].prefix_lens = [0, 0]
    _collect_hidden_states(batch, np.arange(16, dtype=np.float32).reshape(-1, 1))
    assert first.hidden_states == []
    assert second.hidden_states == [[11.0], [12.0]]


def test_hidden_states_speculative_request_rejected_before_scheduling():
    from sgl_jax.srt.managers.io_struct import TokenizedGenerateReqInput
    from sgl_jax.srt.managers.scheduler import Scheduler
    from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

    outputs = []
    scheduler = SimpleNamespace(
        model_config=SimpleNamespace(hf_eos_token_id=None, vocab_size=64),
        tokenizer=None,
        spec_algorithm=SpeculativeAlgorithm.NEXTN,
        stream_output=lambda reqs, *args: outputs.extend(reqs),
    )
    request = TokenizedGenerateReqInput(
        rid="hidden-spec",
        input_ids=[1, 2],
        radix_input_ids=[1, 2],
        sampling_params=SamplingParams(max_new_tokens=2),
        return_hidden_states=True,
    )
    Scheduler.handle_generate_request(scheduler, request)
    assert len(outputs) == 1
    assert outputs[0].finished()
    assert "not supported with speculative decoding" in outputs[0].finished_reason.message
