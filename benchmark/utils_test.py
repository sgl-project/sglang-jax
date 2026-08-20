from benchmark.utils import _extract_device_durations_by_pid_ms


def test_extract_device_durations_prefers_named_scope_call_done_markers():
    trace = {
        "traceEvents": [
            {
                "name": "fused_v2",
                "pid": 7,
                "dur": 999,
                "args": {},
            },
            {
                "name": "wrapped-call",
                "pid": 7,
                "ts": 2,
                "args": {
                    "tf_op": "jit(SGLANG_JAX_BENCH_1)/wrapped-call",
                    "device_duration_ps": "3000000000",
                },
            },
            {
                "name": "wrapped-call-done",
                "pid": 7,
                "ts": 2,
                "args": {
                    "tf_op": "jit(SGLANG_JAX_BENCH_1)/wrapped-call-done",
                    "device_duration_ps": "2000000000",
                },
            },
            {
                "name": "wrapped-call-done",
                "pid": 7,
                "ts": 1,
                "args": {
                    "tf_op": "jit(SGLANG_JAX_BENCH_0)/wrapped-call-done",
                    "device_duration_ps": "1000000000",
                },
            },
            {
                "name": "wrapped-call-done",
                "pid": 8,
                "ts": 1,
                "args": {
                    "tf_op": "jit(SGLANG_JAX_BENCH_0)/wrapped-call-done",
                    "device_duration_ps": "1500000000",
                },
            },
            {
                "name": "wrapped-call-done",
                "pid": 8,
                "ts": 2,
                "args": {
                    "tf_op": "jit(SGLANG_JAX_BENCH_1)/wrapped-call-done",
                    "device_duration_ps": "2500000000",
                },
            },
        ]
    }

    assert _extract_device_durations_by_pid_ms(trace, "fused_v2") == {
        7: [1.0, 2.0],
        8: [1.5, 2.5],
    }


def test_extract_device_durations_accepts_legacy_task_device_events():
    trace = {
        "traceEvents": [
            {
                "name": "host-marker",
                "pid": 1,
                "ts": 0,
                "dur": 99,
                "args": {"tf_op": "jit(SGLANG_JAX_BENCH_0)/host-marker"},
            },
            {
                "name": "fused_rs-step",
                "pid": 11,
                "ts": 1,
                "args": {"device_duration_ps": "4250000000"},
            }
        ]
    }

    assert _extract_device_durations_by_pid_ms(trace, "fused_rs") == {11: [4.25]}
