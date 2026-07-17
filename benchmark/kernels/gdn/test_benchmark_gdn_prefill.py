"""Tests for I5's multi-process rank identity contract."""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.kernels.gdn import benchmark_gdn_prefill as benchmark


def _runtime(process_index: int) -> dict[str, object]:
    return {
        "process_count": 4,
        "device_count": 16,
        "local_device_count": 4,
        "process_index": process_index,
        "device_kind": "TPU v6 lite",
        "device_platforms": ["tpu"],
    }


def test_performance_runtime_accepts_nonidentity_complete_permutations(monkeypatch):
    gathered = np.asarray([[1, 0], [0, 1], [2, 2], [3, 3]], dtype=np.int32)
    monkeypatch.setattr(
        benchmark,
        "_gather_rank_identities",
        lambda requested_id, pjrt_process_index: gathered,
        raising=False,
    )

    identity = benchmark._validate_performance_runtime(_runtime(process_index=1), rank=0)

    assert identity["requested_id"] == 0
    assert identity["pjrt_process_index"] == 1
    assert identity["requested_ids"] == [0, 1, 2, 3]
    assert identity["pjrt_process_indices"] == [0, 1, 2, 3]


def test_rank_permutations_accept_nonidentity_mapping():
    identity = benchmark._validate_rank_permutations(
        requested_ids=[0, 1, 2, 3],
        pjrt_process_indices=[1, 0, 2, 3],
        process_count=4,
    )

    assert identity == {
        "requested_ids": [0, 1, 2, 3],
        "pjrt_process_indices": [0, 1, 2, 3],
    }


@pytest.mark.parametrize(
    ("field", "requested_ids", "pjrt_process_indices"),
    [
        ("requested IDs", [0, 0, 2, 3], [0, 1, 2, 3]),
        ("requested IDs", [0, 1, 2], [0, 1, 2, 3]),
        ("requested IDs", [0, 1, 2, 4], [0, 1, 2, 3]),
        ("PJRT process indices", [0, 1, 2, 3], [0, 0, 2, 3]),
        ("PJRT process indices", [0, 1, 2, 3], [0, 1, 2]),
        ("PJRT process indices", [0, 1, 2, 3], [0, 1, 2, 4]),
    ],
    ids=(
        "duplicate-requested",
        "missing-requested",
        "out-of-range-requested",
        "duplicate-pjrt",
        "missing-pjrt",
        "out-of-range-pjrt",
    ),
)
def test_rank_permutations_reject_invalid_sets(field, requested_ids, pjrt_process_indices):
    with pytest.raises(RuntimeError, match=field):
        benchmark._validate_rank_permutations(
            requested_ids=requested_ids,
            pjrt_process_indices=pjrt_process_indices,
            process_count=4,
        )


def test_artifact_paths_remain_keyed_by_requested_id():
    cache, profiler = benchmark._paths("chunkwise", rank=1)

    assert cache == "/tmp/beaver-324/jax-cache/i5/optimized/rank-1"
    assert profiler == "/tmp/beaver-324/profiler/i5/optimized/rank-1"
