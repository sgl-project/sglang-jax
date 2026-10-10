"""CPU integration checks for scheduled requests -> immutable model inputs."""

import copy
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from sgl_jax.srt.managers.schedule_batch import (
    ModelWorkerSamplingInfo,
    Req,
    ScheduleBatch,
    ScheduleReqsInfo,
)
from sgl_jax.srt.model_executor.batch_inputs import FORWARD_INPUT_NAMES
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.sampling.sampling_params import SamplingParams
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm


def make_batch(counts, mode, *, recurrent=False, logprob=False):
    infos = []
    for rank, count in enumerate(counts):
        lengths = [i + 2 if mode.is_extend() else 1 for i in range(count)]
        prefixes = [rank + 1] * count
        seqs = np.array(prefixes, np.int32) + lengths
        reqs = [
            Req(
                f"{rank}-{i}",
                None,
                [1, 2, 3],
                SamplingParams(max_new_tokens=4),
                lora_id=str(rank + 1),
            )
            for i in range(count)
        ]
        sampling = ModelWorkerSamplingInfo.generate_for_precompile_all_greedy(count, 32)
        sampling.temperatures[:] = rank + 0.5
        sampling.top_ks[:] = rank + 2
        info = ScheduleReqsInfo(
            reqs=reqs,
            seq_lens=seqs.astype(np.int32),
            req_pool_indices=np.arange(rank * 4, rank * 4 + count, dtype=np.int32),
            input_ids=np.arange(rank * 100, rank * 100 + sum(lengths), dtype=np.int32),
            out_cache_loc=np.arange(rank * 100 + 1, rank * 100 + 1 + sum(lengths), dtype=np.int32),
            prefix_lens=prefixes,
            extend_lens=lengths,
            extend_logprob_start_lens=[1] * count,
            extend_input_logprob_token_ids=np.arange(sum(n - 1 for n in lengths), dtype=np.int32),
            top_logprobs_nums=[2] * count,
            token_ids_logprobs=[[1, 3]] * count,
            sampling_info=sampling,
        )
        if recurrent:
            info.recurrent_indices = np.arange(rank * 10 + 1, rank * 10 + count + 1, dtype=np.int32)
            info.recurrent_cow_src_indices = np.full(count, rank + 5, np.int32)
            info.recurrent_track_indices = np.full(count, rank + 10, np.int32)
            info.recurrent_track_mask = np.ones(count, np.int32)
        infos.append(info)
    return ScheduleBatch(
        reqs_info=infos,
        dp_size=len(counts),
        forward_mode=mode,
        model_config=SimpleNamespace(
            is_in_model_multimodal=False, hf_config=SimpleNamespace(vision_encoder_parallel="data")
        ),
        req_to_token_pool=SimpleNamespace(
            req_to_token=np.arange(32 * 64, dtype=np.int32).reshape(32, 64),
            cache_loc_host_buf=np.zeros(len(counts) * 256, np.int32),
        ),
        spec_algorithm=SpeculativeAlgorithm.NONE,
        return_logprob=logprob,
    )


def build(batch):
    dp = batch.dp_size
    return batch.get_model_worker_batch(
        [dp * 8, dp * 16], [dp * 2, dp * 4], [dp * 128, dp * 256], page_size=4
    )


@pytest.mark.parametrize("counts", [[2], [0, 2], [2, 0, 1, 1]])
@pytest.mark.parametrize("mode", [ForwardMode.EXTEND, ForwardMode.MIXED, ForwardMode.DECODE])
def test_request_token_and_feature_layout(counts, mode):
    batch = make_batch(counts, mode, recurrent=True, logprob=True)
    result = build(batch)
    plan = result.layout
    assert plan.request_counts == tuple(counts)
    assert result.real_bs == sum(counts)
    assert result.real_input_ids_len == sum(len(i.input_ids) for i in batch.reqs_info)
    for rank, info in enumerate(batch.reqs_info):
        n = counts[rank]
        rs, ts = plan.request_slice(rank), plan.token_slice(rank)
        np.testing.assert_array_equal(result.input_ids[ts], info.input_ids)
        np.testing.assert_array_equal(result.out_cache_loc[ts], info.out_cache_loc)
        np.testing.assert_array_equal(result.seq_lens[rs], info.seq_lens)
        np.testing.assert_array_equal(result.req_pool_indices[rs], info.req_pool_indices)
        expected_positions = (
            np.concatenate(
                [
                    np.arange(p, p + q, dtype=np.int32)
                    for p, q in zip(info.prefix_lens, info.extend_lens)
                ]
            )
            if n
            else []
        )
        np.testing.assert_array_equal(result.positions[ts], expected_positions)
        rpad = slice(rs.stop, plan.request_slice(rank, padded=True).stop)
        tpad = slice(ts.stop, plan.token_slice(rank, padded=True).stop)
        assert np.all(result.seq_lens[rpad] == 0)
        assert np.all(result.req_pool_indices[rpad] == -1)
        assert np.all(result.input_ids[tpad] == 0)
        assert np.all(result.out_cache_loc[tpad] == -1)
        np.testing.assert_array_equal(result.recurrent_cow_src_indices[rs], np.full(n, rank + 5))
        assert info.recurrent_cow_src_indices is None
        assert info.recurrent_track_mask is None
        assert result.lora_ids[rs] == [str(rank + 1)] * n
        assert result.top_logprobs_nums[rs] == [2] * n
        np.testing.assert_array_equal(
            result.sampling_info.temperatures[rs, 0], np.full(n, rank + 0.5)
        )
        if mode.is_extend():
            np.testing.assert_array_equal(
                result.logits_indices[rs], np.cumsum(info.extend_lens) - 1
            )
    np.testing.assert_array_equal(
        result.seq_lens[result.logits_indices_selector],
        np.concatenate([i.seq_lens for i in batch.reqs_info]),
    )
    with pytest.raises(ValueError):
        result.input_ids[0] = 99


@pytest.mark.parametrize("dp", [1, 2, 4])
def test_direct_upload_and_inflight_snapshot(dp, monkeypatch):
    if len(jax.devices()) < dp:
        pytest.skip("Run with XLA_FLAGS=--xla_force_host_platform_device_count=4")
    counts = [1 if rank != 1 else 0 for rank in range(dp)]
    batch = make_batch(counts, ForwardMode.EXTEND, recurrent=True)
    first = build(batch)
    mesh = Mesh(np.array(jax.devices()[:dp]), ("data",))
    sharding = NamedSharding(mesh, PartitionSpec("data"))

    # No flattened host field is needed for the upload. In particular, no
    # per-field materialization or concatenate is allowed in this path.
    def unexpected_host_read(*args):
        raise AssertionError("upload materialized a host field")

    with monkeypatch.context() as m:
        m.setattr(first.inputs, "host", unexpected_host_read)
        uploaded = first.inputs.to_device(sharding)
    saved = {name: getattr(first, name) for name in FORWARD_INPUT_NAMES}
    for info in batch.reqs_info:
        info.input_ids[:] += 1000
        info.seq_lens[:] += 1
        info.prefix_lens = [p + 1 for p in info.prefix_lens]
    second = build(batch)
    for name, actual in zip(FORWARD_INPUT_NAMES, uploaded):
        if actual is not None:
            np.testing.assert_array_equal(np.asarray(actual), saved[name])
            assert actual.sharding.spec == PartitionSpec("data")
    assert not np.array_equal(first.input_ids, second.input_ids)
    # A shallow worker copy can replace inputs without changing the first batch.
    clone = copy.copy(first)
    clone.seq_lens = first.seq_lens + 10
    assert first.layout is not None and clone.layout is None
    np.testing.assert_array_equal(first.seq_lens, saved["seq_lens"])
    replaced = clone.inputs.to_device(sharding)
    np.testing.assert_array_equal(replaced[1], saved["seq_lens"] + 10)


@pytest.mark.parametrize("dp", [1, 2, 4])
def test_forward_batch_matches_reference_pack(dp):
    if len(jax.devices()) < dp:
        pytest.skip("Requires multiple CPU devices")
    from sgl_jax.srt.utils.jax_utils import packed_device_array

    mwb = build(make_batch([1] * dp, ForwardMode.DECODE, recurrent=True))
    mesh = Mesh(np.array(jax.devices()[:dp]), ("data",))
    runner = SimpleNamespace(
        mesh=mesh,
        model=SimpleNamespace(mrope_position_axes=0),
        attn_backend=None,
        model_config=SimpleNamespace(
            is_embedding=False, hf_config=SimpleNamespace(architectures=[])
        ),
    )
    fb = ForwardBatch.init_new(mwb, runner)
    reference = packed_device_array(
        tuple(getattr(mwb, n) for n in FORWARD_INPUT_NAMES),
        NamedSharding(mesh, PartitionSpec("data")),
    )
    for name, expected in zip(FORWARD_INPUT_NAMES, reference):
        actual = getattr(fb, name)
        if expected is None:
            assert actual is None
        else:
            np.testing.assert_array_equal(actual, expected)


def test_speculative_prefill_replaces_the_previous_snapshot():
    from sgl_jax.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput

    batch = build(make_batch([2, 0, 1, 1], ForwardMode.EXTEND))
    previous_inputs = batch.inputs
    old_ids = batch.input_ids.copy()
    old_lens = batch.seq_lens.copy()
    layout = batch.layout
    accepted = np.arange(batch.real_bs, dtype=np.int32) + 900
    draft = EagleDraftInput(verified_id=accepted)
    draft.prepare_for_extend_after_target_prefill(batch)
    np.testing.assert_array_equal(previous_inputs.host("input_ids"), old_ids)
    last = layout.sequences.query_starts + layout.sequences.query_lengths - 1
    offset = 0
    for rank, count in enumerate(layout.request_counts):
        slots = last[offset : offset + count] + layout.token_slice(rank).start
        np.testing.assert_array_equal(batch.input_ids[slots], accepted[offset : offset + count])
        offset += count
    verify = SimpleNamespace(
        draft_token=np.arange(16, dtype=np.int32), positions=np.arange(16, dtype=np.int32)
    )
    EagleVerifyInput.prepare_for_verify(verify, batch)
    expected = old_lens.copy()
    expected[batch.logits_indices_selector] -= 1
    np.testing.assert_array_equal(batch.seq_lens, expected)
    np.testing.assert_array_equal(previous_inputs.host("seq_lens"), old_lens)


def test_extend_warmup_uses_the_same_dp_token_segments():
    from sgl_jax.srt.model_executor.compilation_manager import CompilationManager

    manager = object.__new__(CompilationManager)
    manager.vocab_size = 32
    manager.capture_hidden_states = False
    manager.has_recurrent_state = False
    manager.supports_recurrent_cow = False
    manager.supports_recurrent_track = False
    dummy = manager._make_dummy_batch(8, 32, ForwardMode.EXTEND, 64, dp_size=4)
    plan = dummy.layout
    for rank in range(4):
        tokens = plan.token_slice(rank)
        padded = plan.token_slice(rank, padded=True)
        np.testing.assert_array_equal(dummy.input_ids[tokens], [1, 1])
        assert np.all(dummy.input_ids[tokens.stop : padded.stop] == 0)
        assert np.all(dummy.out_cache_loc[tokens.stop : padded.stop] == -1)
    assert plan.token_capacity != plan.request_capacity


def test_idle_batch_keeps_safe_padding():
    batch = build(make_batch([0, 0, 0, 0], ForwardMode.IDLE))
    assert batch.real_bs == batch.real_input_ids_len == 0
    assert batch.logits_indices_selector.size == 0
    assert np.all(batch.seq_lens == 0)
    assert np.all(batch.req_pool_indices == -1)
    assert np.all(batch.out_cache_loc == -1)


def test_staging_groups_dtypes_without_reusing_storage():
    from sgl_jax.srt.model_executor.batch_inputs import BatchInputBuffer

    if len(jax.devices()) < 2:
        pytest.skip("Requires multiple CPU devices")
    fields = {
        "input_ids": ((8,), np.int32, 0),
        "lora_scalings": ((4, 1), np.float32, 1),
    }
    first = BatchInputBuffer(2, fields)
    held_view = first.view("input_ids", 1)
    held_view[:] = [1, 2, 3, 4]
    snapshot = first.finish()
    with pytest.raises(ValueError):
        held_view[0] = 999
    second = BatchInputBuffer(2, fields)
    second.view("input_ids", 1)[:] = 0
    mesh = Mesh(np.array(jax.devices()[:2]), ("data",))
    uploaded = snapshot.to_device(NamedSharding(mesh, PartitionSpec("data")))
    np.testing.assert_array_equal(uploaded[0], [0, 0, 0, 0, 1, 2, 3, 4])
    np.testing.assert_array_equal(uploaded[7], np.ones((4, 1), np.float32))
