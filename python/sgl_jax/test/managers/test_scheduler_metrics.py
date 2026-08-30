import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import Mock, call

from sgl_jax.srt.managers.scheduler_metrics_mixin import record_queue_wait_times


def _req(dp_rank, start, end=None):
    return SimpleNamespace(
        dp_rank=dp_rank,
        queue_time_start=start,
        queue_time_end=end,
    )


def test_record_queue_wait_times_groups_by_dp_rank():
    reqs = [_req(0, 8.0), _req(1, 9.0), _req(0, 9.5)]
    metric = Mock()

    record_queue_wait_times(reqs, metric, now=10.0)

    assert metric.labels.call_args_list == [
        call(dp_rank="0"),
        call(dp_rank="1"),
        call(dp_rank="0"),
    ]
    assert metric.labels.return_value.observe.call_args_list == [
        call(2.0),
        call(1.0),
        call(0.5),
    ]
    assert [req.queue_time_end for req in reqs] == [10.0, 10.0, 10.0]


def test_record_queue_wait_times_ignores_retracted_and_unstarted_requests():
    already_recorded = _req(0, 1.0, end=2.0)
    no_start = _req(1, None)
    metric = Mock()

    record_queue_wait_times([already_recorded, no_start], metric, now=10.0)

    metric.labels.assert_not_called()
    assert already_recorded.queue_time_end == 2.0
    assert no_start.queue_time_end is None


def test_queue_time_histogram_is_exported_in_single_and_spawned_modes(tmp_path):
    """Exercise fresh prometheus imports using both scheduler launch shapes."""
    script = textwrap.dedent(
        """
        import os
        import subprocess
        import sys

        mode, metrics_dir = sys.argv[1:]

        if mode == "single":
            # Match http_server: the scheduler module is imported before launch
            # configures Prometheus, then the metric is created in-process.
            from sgl_jax.srt.managers.scheduler_metrics_mixin import create_queue_time_histogram
            assert "prometheus_client" not in sys.modules

        os.environ["PROMETHEUS_MULTIPROC_DIR"] = metrics_dir

        if mode == "single":
            metric = create_queue_time_histogram()
            metric.labels(dp_rank="0").observe(0.25)
            from prometheus_client import values
            assert values.ValueClass.__name__ != "MutexValue"
        else:
            child = (
                "from sgl_jax.srt.managers.scheduler_metrics_mixin "
                "import create_queue_time_histogram; "
                "metric = create_queue_time_histogram(); "
                'metric.labels(dp_rank="1").observe(0.5)'
            )
            subprocess.run([sys.executable, "-c", child], check=True, env=os.environ)

        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sgl_jax.srt.utils.common_utils import add_prometheus_middleware

        app = FastAPI()
        add_prometheus_middleware(app)
        response = TestClient(app).get("/metrics")
        assert response.status_code == 200
        output = response.text
        expected_rank = "0" if mode == "single" else "1"
        assert f'sglang:queue_time_seconds_count{{dp_rank="{expected_rank}"}} 1.0' in output, output
        """
    )

    env = os.environ.copy()
    env.pop("PROMETHEUS_MULTIPROC_DIR", None)
    for mode in ("single", "spawned"):
        metrics_dir = tmp_path / mode
        metrics_dir.mkdir()
        subprocess.run(
            [sys.executable, "-c", script, mode, str(metrics_dir)],
            check=True,
            env=env,
        )
