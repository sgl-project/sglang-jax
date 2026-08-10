from __future__ import annotations

import importlib.util
import threading
from pathlib import Path
from unittest import mock

BENCHMARK_PATH = (
    Path(__file__).resolve().parents[2] / "benchmark" / "glm52" / "bench_dsa_cache_hit.py"
)
SPEC = importlib.util.spec_from_file_location("bench_dsa_cache_hit", BENCHMARK_PATH)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


def test_shared_prefix_inputs_share_only_the_prefix() -> None:
    prefixes, extended = BENCHMARK._make_inputs(4, 8, 3, prefix_mode="shared")

    assert prefixes == [prefixes[0]] * 4
    assert all(value[:8] == prefixes[0] for value in extended)
    assert len({tuple(value[8:]) for value in extended}) == 4
    assert len({value[8] for value in extended}) == 4
    assert all(len(set(value[8:])) > 1 for value in extended)


def test_independent_prefix_inputs_do_not_share_prefixes() -> None:
    prefixes, extended = BENCHMARK._make_inputs(4, 8, 3, prefix_mode="independent")

    assert len({tuple(value) for value in prefixes}) == 4
    assert len({value[0] for value in prefixes}) == 4
    assert all(value[:8] == prefixes[i] for i, value in enumerate(extended))


def test_random_inputs_are_reproducible_by_seed() -> None:
    first = BENCHMARK._make_inputs(4, 8, 8, prefix_mode="shared", random_seed=7)
    repeated = BENCHMARK._make_inputs(4, 8, 8, prefix_mode="shared", random_seed=7)
    different = BENCHMARK._make_inputs(4, 8, 8, prefix_mode="shared", random_seed=8)

    assert first == repeated
    assert first != different


def test_profile_wraps_only_the_measured_phase_api() -> None:
    start_response = mock.Mock()
    stop_response = mock.Mock()
    running_response = mock.Mock()
    running_response.json.return_value = {"status": "in_progress"}
    idle_response = mock.Mock()
    idle_response.json.return_value = {"status": "idle"}

    with (
        mock.patch.object(
            BENCHMARK.requests,
            "post",
            side_effect=(start_response, stop_response),
        ) as post,
        mock.patch.object(
            BENCHMARK.requests,
            "get",
            side_effect=(running_response, idle_response),
        ) as get,
    ):
        BENCHMARK._start_profile(
            "http://server",
            Path("/tmp/profile"),
            host_tracer_level=0,
            python_tracer_level=0,
        )
        BENCHMARK._stop_profile("http://server")

    assert post.call_args_list[0].args == ("http://server/start_profile",)
    assert post.call_args_list[0].kwargs["json"] == {
        "output_dir": "/tmp/profile",
        "host_tracer_level": 0,
        "python_tracer_level": 0,
    }
    assert post.call_args_list[1].args == ("http://server/stop_profile",)
    assert [call.kwargs["timeout"] for call in get.call_args_list] == [
        BENCHMARK.PROFILE_CONTROL_TIMEOUT_S,
        BENCHMARK.PROFILE_CONTROL_TIMEOUT_S,
    ]
    start_response.raise_for_status.assert_called_once_with()
    stop_response.raise_for_status.assert_called_once_with()
    running_response.raise_for_status.assert_called_once_with()
    idle_response.raise_for_status.assert_called_once_with()


def test_stage_profile_request_selects_prefill_and_decode() -> None:
    response = mock.Mock()

    with mock.patch.object(BENCHMARK.requests, "post", return_value=response) as post:
        BENCHMARK._start_profile(
            "http://server",
            Path("/tmp/profile"),
            host_tracer_level=0,
            python_tracer_level=0,
            num_steps=3,
            profile_by_stage=True,
            profile_stages=["prefill", "decode"],
        )

    assert post.call_args.kwargs["json"] == {
        "output_dir": "/tmp/profile",
        "host_tracer_level": 0,
        "python_tracer_level": 0,
        "num_steps": 3,
        "profile_by_stage": True,
        "profile_stages": ["prefill", "decode"],
    }


def test_stop_profile_is_a_noop_after_stage_profile_auto_completes() -> None:
    idle_response = mock.Mock()
    idle_response.json.return_value = {"status": "idle"}

    with (
        mock.patch.object(BENCHMARK.requests, "get", return_value=idle_response) as get,
        mock.patch.object(BENCHMARK.requests, "post") as post,
    ):
        BENCHMARK._stop_profile("http://server")

    post.assert_not_called()
    assert get.call_args.kwargs["timeout"] == BENCHMARK.PROFILE_CONTROL_TIMEOUT_S
    idle_response.raise_for_status.assert_called_once_with()


def test_parallel_single_requests_preserve_result_order() -> None:
    def fake_run_native_batch(base_url, input_ids, output_len, *, label):
        request_id = input_ids[0][0]
        return {
            "wall_s": float(request_id),
            "ttft_s": [request_id + 0.1],
            "request_to_headers_s": [request_id + 0.01],
            "headers_to_first_token_s": [request_id + 0.09],
            "request_start_unix_ns": [request_id * 100],
            "response_headers_unix_ns": [request_id * 100 + 10],
            "first_token_unix_ns": [request_id * 100 + 20],
            "decode_s": [request_id + 0.2],
            "cached_tokens": [request_id + 10],
            "completion_tokens": [request_id + 20],
        }

    with mock.patch.object(
        BENCHMARK, "_run_native_batch", side_effect=fake_run_native_batch
    ) as run_native_batch:
        result = BENCHMARK._run_parallel_single_requests(
            "http://server", [[1], [2], [3], [4]], 1, label="profile"
        )

    assert result["wall_s"] >= 0
    assert result["ttft_s"] == [1.1, 2.1, 3.1, 4.1]
    assert result["request_to_headers_s"] == [1.01, 2.01, 3.01, 4.01]
    assert result["headers_to_first_token_s"] == [1.09, 2.09, 3.09, 4.09]
    assert result["request_start_unix_ns"] == [100, 200, 300, 400]
    assert result["response_headers_unix_ns"] == [110, 210, 310, 410]
    assert result["first_token_unix_ns"] == [120, 220, 320, 420]
    assert result["decode_s"] == [1.2, 2.2, 3.2, 4.2]
    assert result["cached_tokens"] == [11, 12, 13, 14]
    assert result["completion_tokens"] == [21, 22, 23, 24]
    assert run_native_batch.call_count == 4
    assert all(len(call.args[1]) == 1 for call in run_native_batch.call_args_list)


def test_profile_admission_barrier_queues_all_requests_before_resume() -> None:
    pause_response = mock.Mock()
    pause_response.json.return_value = {"success": True}
    resume_response = mock.Mock()
    resume_response.json.return_value = {"success": True}
    server_info_response = mock.Mock()
    scheduler_states = iter(
        [
            [{"waiting_queue_size": 1}, {"waiting_queue_size": 1}],
            [{"waiting_queue_size": 3}, {"waiting_queue_size": 1}],
            [{"waiting_queue_size": 2}, {"waiting_queue_size": 2}],
        ]
    )
    server_info_response.json.side_effect = lambda: {"internal_states": next(scheduler_states)}
    resumed = threading.Event()

    def fake_post(url, **kwargs):
        assert url.endswith("/set_internal_state")
        if kwargs["json"]["state_data"]["engine_paused"]:
            return pause_response
        resumed.set()
        return resume_response

    def fake_run_parallel_single_requests(*args, **kwargs):
        assert resumed.wait(timeout=2)
        return {"wall_s": 1.0}

    on_admitted = mock.Mock()
    with (
        mock.patch.object(BENCHMARK.requests, "post", side_effect=fake_post) as post,
        mock.patch.object(BENCHMARK.requests, "get", return_value=server_info_response),
        mock.patch.object(BENCHMARK.time, "sleep") as sleep,
        mock.patch.object(
            BENCHMARK,
            "_run_parallel_single_requests",
            side_effect=fake_run_parallel_single_requests,
        ),
    ):
        result = BENCHMARK._run_native_batch_with_admission_barrier(
            "http://server",
            [[1], [2], [3], [4]],
            1,
            label="profile",
            on_admitted=on_admitted,
            profile_settle_s=0.25,
        )

    assert result == {"wall_s": 1.0}
    on_admitted.assert_called_once_with()
    assert post.call_args_list[0].args == ("http://server/set_internal_state",)
    assert post.call_args_list[0].kwargs["json"]["state_data"] == {"engine_paused": True}
    assert post.call_args_list[1].args == ("http://server/set_internal_state",)
    assert post.call_args_list[1].kwargs["json"]["state_data"] == {"engine_paused": False}
    pause_response.raise_for_status.assert_called_once_with()
    resume_response.raise_for_status.assert_called_once_with()
    assert server_info_response.raise_for_status.call_count == 3
    assert sleep.call_args_list[-1] == mock.call(0.25)


def test_variant_flag_can_label_radix_topk() -> None:
    parser_source = BENCHMARK_PATH.read_text()

    assert parser_source.count('"--variant"') == 1
    assert '"variant": args.variant' in parser_source
