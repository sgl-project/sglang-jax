from sgl_jax.bench_spec_overlap_mock import (
    MockDurations,
    run_mock_spec_overlap,
    summarize_chrome_trace,
    write_chrome_trace,
)


def test_mock_current_schedule_reproduces_spec_overlap_bubble():
    durations = MockDurations()

    result = run_mock_spec_overlap("current", steps=5, durations=durations)

    assert len(result.verify_to_verify_gaps_ms) == 4
    assert min(result.verify_to_verify_idle_ms) > 8.0
    assert min(result.verify_to_verify_gaps_ms) > 10.0


def test_mock_future_relay_schedule_eliminates_tpu_idle_bubble():
    durations = MockDurations()

    result = run_mock_spec_overlap("target_future_relay", steps=5, durations=durations)

    assert max(result.verify_to_verify_idle_ms) == 0.0
    assert result.verify_to_verify_gaps_ms == [durations.draft_extend_tpu_ms] * 4


def test_mock_future_relay_keeps_scheduler_cpu_off_tpu_critical_path():
    durations = MockDurations()

    result = run_mock_spec_overlap("target_future_relay", steps=3, durations=durations)

    verify_starts = [
        event.start_ms
        for event in result.events
        if event.lane == "tpu" and event.name == "jit_fused_greedy_verify"
    ]
    scheduler_events = [
        event
        for event in result.events
        if event.lane == "scheduler" and event.name == "process_phase_a_and_batch"
    ]

    assert scheduler_events[0].end_ms > verify_starts[1]


def test_mock_same_batch_device_chain_eliminates_idle_without_scheduler_gate():
    durations = MockDurations()

    result = run_mock_spec_overlap("same_batch_device_chain", steps=5, durations=durations)

    assert result.verify_to_verify_gaps_ms == [durations.draft_extend_tpu_ms] * 4
    assert result.verify_to_verify_idle_ms == [0.0, 0.0, 0.0, 0.0]
    launch_events = [
        event
        for event in result.events
        if event.lane == "worker" and event.name == "launch_next_verify_from_same_batch_state"
    ]
    scheduler_events = [
        event
        for event in result.events
        if event.lane == "scheduler" and event.name == "process_phase_a_catch_up"
    ]
    assert launch_events
    assert scheduler_events
    assert scheduler_events[0].end_ms > launch_events[0].end_ms


def test_mock_trace_summary_reports_current_idle_and_target_no_idle(tmp_path):
    current = run_mock_spec_overlap("current", steps=5)
    target = run_mock_spec_overlap("target_future_relay", steps=5)
    current_trace = tmp_path / "current.json"
    target_trace = tmp_path / "target.json"
    write_chrome_trace(current, current_trace)
    write_chrome_trace(target, target_trace)

    current_summary = summarize_chrome_trace(current_trace)
    target_summary = summarize_chrome_trace(target_trace)

    assert min(current_summary["verify_to_verify_idle_ms"]) > 8.0
    assert target_summary["verify_to_verify_idle_ms"] == [0.0, 0.0, 0.0, 0.0]
