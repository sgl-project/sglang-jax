"""Capacity exhaustion must split encoder batches instead of failing valid requests."""

from concurrent.futures import CancelledError
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
import pytest

from sgl_jax.srt.disaggregation.encoder.errors import EncoderBatchCapacityError
from sgl_jax.srt.disaggregation.encoder.raiden_pool import create_encoder_pool
from sgl_jax.srt.disaggregation.encoder.raiden_transfer import (
    RaidenEncoderServerTransfer,
)
from sgl_jax.srt.disaggregation.encoder.runtime import EncoderRequest, EncoderRuntime


def _transfer(*, pages=8, page_size=2, lanes=1, slots=8):
    transfer = object.__new__(RaidenEncoderServerTransfer)
    transfer.num_lanes = lanes
    transfer._pages_per_lane = pages // lanes
    transfer._max_inflight = slots

    def pages_needed(tokens):
        if tokens > pages * page_size:
            raise ValueError("Encoder request exceeds the pool token capacity")
        return (tokens + page_size - 1) // page_size

    transfer.pool = NS(pages_needed=pages_needed)
    return transfer


@pytest.mark.parametrize(
    "counts,rows,kwargs",
    [
        ([6, 6], np.arange(12), {"pages": 5}),
        ([1, 1], np.arange(2), {"slots": 1}),
        ([6, 6], np.arange(12), {"pages": 4, "lanes": 2}),
        ([18], np.arange(18), {"pages": 8}),
    ],
)
def test_raiden_reports_unreservable_capacity_before_mutating_pool(counts, rows, kwargs):
    transfer = _transfer(**kwargs)
    with pytest.raises(EncoderBatchCapacityError):
        transfer.reserve_batch_sync(
            [str(i) for i in range(len(counts))], counts, output_indices=rows
        )


def _runtime(counts, *, capacity=8, failure=None):
    requests = [
        EncoderRequest(
            {"req_id": str(i), "part_idx": 0, "modality": "image"},
            Mock(done=lambda: False),
            inputs=count,
            token_count=count,
        )
        for i, count in enumerate(counts)
    ]
    layouts = []
    encoded = []
    staged = []
    delivered = []

    def build(inputs):
        layouts.append(tuple(inputs))
        return inputs, np.arange(sum(inputs), dtype=np.int32)

    def encode(inputs, *, modality):
        encoded.append(tuple(inputs))
        return tuple(inputs)

    def reserve(ids, tokens, *, output_indices, cancelled):
        if failure is not None:
            raise failure
        if sum(tokens) > capacity:
            raise EncoderBatchCapacityError("test capacity")
        assert len(output_indices) == sum(tokens)
        if cancelled():
            raise CancelledError()

    transfer = NS(
        reserve_batch_sync=reserve,
        stage_batch_sync=lambda ids, values: staged.append(tuple(ids)),
        release=Mock(),
    )
    runtime = EncoderRuntime(NS(build_batch=build, encode=encode), transfer)
    runtime._deliver = lambda reqs, results: delivered.extend(zip(reqs, results))
    runtime._encode_batch(requests)
    return runtime, requests, layouts, encoded, staged, delivered


def test_capacity_split_rebuilds_layout_and_preserves_every_request():
    runtime, requests, layouts, encoded, staged, delivered = _runtime([6, 6, 2])
    assert layouts == [(6, 6, 2), (6,), (6, 2)]
    assert encoded == [(6,), (6, 2)]
    assert staged == [(requests[0].transfer_id,), tuple(r.transfer_id for r in requests[1:])]
    assert not delivered
    batches = [runtime._transfer_queue.get_nowait() for _ in range(2)]
    assert [r for batch, _ in batches for r in batch] == requests


def test_one_oversized_request_does_not_fail_valid_siblings():
    runtime, requests, _, encoded, staged, delivered = _runtime([2, 10, 2])
    assert encoded == [(2,), (2,)]
    assert len(staged) == 2
    assert len(delivered) == 1
    assert delivered[0][0] is requests[1]
    assert isinstance(delivered[0][1], EncoderBatchCapacityError)
    assert [runtime._transfer_queue.get_nowait()[0][0] for _ in range(2)] == [
        requests[0],
        requests[2],
    ]


def test_other_backend_errors_are_not_retried_or_encoded():
    error = RuntimeError("transport unavailable")
    _, requests, layouts, encoded, staged, delivered = _runtime([2, 2], failure=error)
    assert layouts == [(2, 2)]
    assert not encoded and not staged
    assert delivered == [(request, error) for request in requests]


@pytest.mark.parametrize(
    "shards,tokens,pages",
    [
        (1, 128, 1),
        (1, 129, 2),
        (1, 32768, 256),
        (2, 128, 2),
        (2, 129, 2),
        (2, 32768, 256),
        (4, 128, 4),
        (4, 129, 4),
        (4, 32768, 256),
    ],
)
def test_pool_token_capacity_is_global_and_rounded_to_complete_shards(shards, tokens, pages):
    args = NS(encoder_transfer_max_tokens=tokens, encoder_max_batch_size=8)
    config = NS(hidden_size=8, dtype=np.float32, hf_config=NS())
    mesh = NS(size=shards, axis_names=("lane",))
    with (
        patch("sgl_jax.srt.disaggregation.encoder.raiden_pool.RaidenPool") as pool,
        patch("sgl_jax.srt.disaggregation.encoder.raiden_pool.jax.sharding.NamedSharding"),
    ):
        create_encoder_pool(args, config, mesh)
    assert pool.call_args.kwargs["capacity"] == pages
