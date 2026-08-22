import importlib.util
import sys
import types
import unittest

if importlib.util.find_spec("jax") is None:
    # The parser is pure Python, while benchmark.utils imports JAX for the
    # profiler entrypoint.  Keep this unit test runnable in the lightweight
    # local environment without pretending to exercise profiler integration.
    sys.modules["jax"] = types.ModuleType("jax")

from benchmark.utils import _extract_trace_measurements


def _event(
    *,
    name: str,
    pid: int,
    ts: float,
    duration_ms: float,
    tf_op: str = "",
    hlo_category: str | None = None,
):
    args = {
        "device_duration_ps": duration_ms * 1e9,
        "tf_op": tf_op,
    }
    if hlo_category is not None:
        args["hlo_category"] = hlo_category
    return {"name": name, "pid": pid, "ts": ts, "args": args}


class TraceMeasurementTest(unittest.TestCase):
    def test_extracts_call_task_and_scoped_collectives_from_same_pid(self):
        events = []
        for iteration in range(2):
            base = iteration * 100.0
            events.extend(
                (
                    _event(
                        name="SGLANG_JAX_BENCH_call-done",
                        pid=7,
                        ts=base,
                        duration_ms=6.0 + iteration,
                        tf_op=f"SGLANG_JAX_BENCH_{iteration}",
                    ),
                    _event(
                        name="all-gather.1",
                        pid=7,
                        ts=base + 1,
                        duration_ms=1.0 + iteration,
                        tf_op="fused_rs_hidden_all_gather/all-gather",
                        hlo_category="all-gather",
                    ),
                    _event(
                        name="all-gather.2",
                        pid=7,
                        ts=base + 2,
                        duration_ms=0.1 + iteration,
                        tf_op="fused_rs_topk_ids_all_gather/all-gather",
                        hlo_category="all-gather",
                    ),
                    _event(
                        name="gmm_v2_fused_rs-test",
                        pid=7,
                        ts=base + 3,
                        duration_ms=4.0 + iteration,
                    ),
                )
            )

        # A second PID has fewer samples; existing benchmark semantics select
        # the representative PID with the most matching events.
        events.append(
            _event(
                name="gmm_v2_fused_rs-test",
                pid=8,
                ts=0,
                duration_ms=99.0,
            )
        )
        # Same HLO category but outside the named scope must not be attributed.
        events.append(
            _event(
                name="all-gather.other",
                pid=7,
                ts=5,
                duration_ms=88.0,
                tf_op="unrelated/all-gather",
                hlo_category="all-gather",
            )
        )

        measurements = _extract_trace_measurements(
            {"traceEvents": events},
            task=r"gmm_v2_fused_rs.*",
            stage_scopes={
                "hidden": ("fused_rs_hidden_all_gather", "all-gather"),
                "topk_ids": ("fused_rs_topk_ids_all_gather", "all-gather"),
            },
        )

        self.assertEqual(measurements["call_samples_ms"], [6.0, 7.0])
        self.assertEqual(measurements["task_samples_ms"], [4.0, 5.0])
        self.assertEqual(measurements["stage_samples_ms"]["hidden"], [1.0, 2.0])
        self.assertEqual(measurements["stage_samples_ms"]["topk_ids"], [0.1, 1.1])

    def test_missing_call_marker_is_not_replaced_by_task_duration(self):
        measurements = _extract_trace_measurements(
            {
                "traceEvents": [
                    _event(
                        name="gmm_v2_fused_rs-test",
                        pid=7,
                        ts=0,
                        duration_ms=4.0,
                    )
                ]
            },
            task=r"gmm_v2_fused_rs.*",
            stage_scopes={},
        )

        self.assertEqual(measurements["call_samples_ms"], [])
        self.assertEqual(measurements["task_samples_ms"], [4.0])


if __name__ == "__main__":
    unittest.main()
