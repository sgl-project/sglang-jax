"""Tests for I5's multi-process rank identity contract."""

from __future__ import annotations

import errno
import json
from pathlib import Path

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


def _performance_artifact(rank: int = 0) -> dict[str, object]:
    return {
        "schema_version": 1,
        "rank": rank,
        "evidence": {"classification": "performance"},
        "code_revision": "a" * 40,
        "lengths": [{"length": 4096, "median_ms": 1.0, "finite": True}],
    }


def test_publish_uses_local_staging_when_target_rejects_hardlinks(tmp_path, monkeypatch):
    staging = tmp_path / "local" / "rank-0.json"
    published = tmp_path / "data" / "rank-0.json"
    payload = _performance_artifact()
    benchmark._atomic_json(staging, payload)
    original_link = benchmark.os.link

    def reject_target_hardlinks(source, target):
        if Path(target).parent == published.parent:
            raise OSError(errno.EOPNOTSUPP, "hardlinks unsupported")
        return original_link(source, target)

    monkeypatch.setattr(benchmark.os, "link", reject_target_hardlinks)

    receipt = benchmark._publish_rank_artifact(staging, published, rank=0)

    marker = Path(f"{published}.complete")
    assert json.loads(published.read_text(encoding="utf-8")) == payload
    assert json.loads(marker.read_text(encoding="utf-8")) == receipt
    assert receipt["artifact_sha256"] == benchmark._sha256(published.read_bytes())
    assert receipt["artifact_size"] == published.stat().st_size


def test_publish_withholds_complete_marker_when_readback_sha_differs(tmp_path, monkeypatch):
    staging = tmp_path / "local" / "rank-0.json"
    published = tmp_path / "data" / "rank-0.json"
    benchmark._atomic_json(staging, _performance_artifact())
    original_read_bytes = benchmark.Path.read_bytes

    def corrupt_published_readback(path):
        data = original_read_bytes(path)
        return data + b"corrupt" if path == published else data

    monkeypatch.setattr(benchmark.Path, "read_bytes", corrupt_published_readback)

    with pytest.raises(RuntimeError, match="SHA256"):
        benchmark._publish_rank_artifact(staging, published, rank=0)

    assert published.exists()
    assert not Path(f"{published}.complete").exists()


def test_publish_rejects_non_finite_staged_json_before_copy(tmp_path):
    staging = tmp_path / "local" / "rank-0.json"
    published = tmp_path / "data" / "rank-0.json"
    staging.parent.mkdir()
    staging.write_text(
        '{"schema_version": 1, "rank": 0, "evidence": {"classification": '
        '"performance"}, "code_revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", '
        '"lengths": [{"median_ms": NaN}]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite"):
        benchmark._publish_rank_artifact(staging, published, rank=0)

    assert not published.exists()
    assert not Path(f"{published}.complete").exists()
