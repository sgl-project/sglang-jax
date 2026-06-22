"""Smoke tests for the grouped-topk profile comparison benchmark."""

import json

from benchmark.kernels.grouped_topk.profile_compare_training import run_comparison


def test_profile_compare_training_writes_metrics(tmp_path):
    output_path = tmp_path / "metrics.jsonl"

    rows = run_comparison(
        token_sizes=[16],
        e=32,
        n_group=8,
        topk_group=4,
        topk=4,
        output_path=output_path,
        trace_root=tmp_path / "traces",
        warmup=1,
        iters=1,
        block_tokens=16,
        interpret=True,
    )

    assert {row["variant"] for row in rows} == {
        "v1_fused",
        "v1_training_gather",
    }
    assert all(row["T"] == 16 and row["num_samples"] >= 1 for row in rows)
    assert all(row["scope"] for row in rows)

    written = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert written == rows
