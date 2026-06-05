from sgl_jax.bench_spec_overlap_tpu_mock import run_mock_tpu_overlap


def test_tpu_mock_harness_runs_with_tiny_cpu_shape():
    result = run_mock_tpu_overlap(
        strategy="target_future_relay",
        steps=2,
        global_batch=4,
        width=8,
        verify_loops=1,
        draft_loops=1,
        scheduler_ms=0.0,
        metadata_ms=0.0,
        profile_dir=None,
        initialize_distributed=False,
    )

    assert result["strategy"] == "target_future_relay"
    assert result["steps"] == 2
    assert result["process_index"] >= 0
    assert result["elapsed_ms"] >= 0.0


def test_tpu_mock_device_chain_scheduler_thread_runs_with_tiny_cpu_shape():
    result = run_mock_tpu_overlap(
        strategy="device_chain_scheduler_thread",
        steps=2,
        global_batch=4,
        width=8,
        verify_loops=1,
        draft_loops=1,
        scheduler_ms=0.0,
        metadata_ms=0.0,
        profile_dir=None,
        initialize_distributed=False,
    )

    assert result["strategy"] == "device_chain_scheduler_thread"
    assert result["steps"] == 2
    assert result["process_index"] >= 0
    assert result["elapsed_ms"] >= 0.0
