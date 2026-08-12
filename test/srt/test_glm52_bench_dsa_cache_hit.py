from __future__ import annotations

import importlib.util
import threading
from pathlib import Path
from unittest import mock

import pytest

BENCHMARK_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmark"
    / "glm52"
    / "bench_dsa_cache_hit.py"
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


def test_unique_prefix_aliases_are_equivalent() -> None:
    independent = BENCHMARK._make_inputs(
        4, 8, 3, prefix_mode="independent", random_seed=11
    )
    unique = BENCHMARK._make_inputs(4, 8, 3, prefix_mode="unique", random_seed=11)
    unique_long = BENCHMARK._make_inputs(
        4, 8, 3, prefix_mode="unique-prefix", random_seed=11
    )

    assert independent == unique == unique_long


def test_warm_inputs_branch_away_from_measured_extensions() -> None:
    prefixes, extended = BENCHMARK._make_inputs(
        4,
        8,
        3,
        prefix_mode="shared",
        random_token_min=1000,
        random_token_max=2000,
    )
    warm_inputs = BENCHMARK._make_warm_inputs(
        prefixes,
        extended,
        warm_branch_token=1999,
    )

    assert all(warm[:-1] == prefix for warm, prefix in zip(warm_inputs, prefixes))
    assert all(warm[-1] == 1999 for warm in warm_inputs)
    assert all(
        warm[-1] != measured[len(prefix)]
        for warm, measured, prefix in zip(warm_inputs, extended, prefixes)
    )


def test_grouped_prefix_inputs_create_two_contiguous_cache_groups() -> None:
    prefixes, extended = BENCHMARK._make_inputs(
        8, 8, 3, prefix_mode="grouped", prefix_group_count=2
    )

    assert prefixes[:4] == [prefixes[0]] * 4
    assert prefixes[4:] == [prefixes[4]] * 4
    assert prefixes[0] != prefixes[4]
    assert all(value[:8] == prefixes[i] for i, value in enumerate(extended))
    layout = BENCHMARK._prefix_layout(prefixes)
    assert layout["unique_prefixes"] == 2
    assert sorted(group["requests"] for group in layout["prefix_groups"]) == [4, 4]


def test_grouped_prefix_inputs_require_balanced_groups() -> None:
    try:
        BENCHMARK._make_inputs(7, 8, 3, prefix_mode="grouped", prefix_group_count=2)
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("unbalanced grouped prefixes should fail")


def test_c32_shared_layout_requires_two_requests_per_dp() -> None:
    prefixes, _ = BENCHMARK._make_inputs(32, 8, 3, prefix_mode="shared")
    layout = BENCHMARK._build_workload_layout(
        prefixes,
        prefix_mode="shared-prefix",
        prefix_group_count=2,
        dp_size=16,
        expected_requests_per_dp=2,
    )

    assert layout["prefix_mode"] == "shared"
    assert layout["requests_per_dp"] == 2
    assert layout["cached_prefixes_per_dp"] == 1
    assert [len(batch) for batch in layout["warm_batches"]] == [16]


def test_c32_unique_layout_warms_two_prefixes_per_dp() -> None:
    prefixes, _ = BENCHMARK._make_inputs(32, 8, 3, prefix_mode="unique-prefix")
    layout = BENCHMARK._build_workload_layout(
        prefixes,
        prefix_mode="unique-prefix",
        prefix_group_count=2,
        dp_size=16,
        expected_requests_per_dp=2,
    )

    assert layout["prefix_mode"] == "unique"
    assert layout["prefix_layout"]["unique_prefixes"] == 32
    assert layout["cached_prefixes_per_dp"] == 2
    assert layout["warm_batches"] == [prefixes]


def test_c64_shared_layout_warms_one_prefix_per_dp() -> None:
    prefixes, _ = BENCHMARK._make_inputs(64, 8, 3, prefix_mode="shared-prefix")
    layout = BENCHMARK._build_workload_layout(
        prefixes,
        prefix_mode="shared-prefix",
        prefix_group_count=2,
        dp_size=32,
        expected_requests_per_dp=2,
    )

    assert layout["prefix_mode"] == "shared"
    assert layout["prefix_layout"]["unique_prefixes"] == 1
    assert layout["cached_prefixes_per_dp"] == 1
    assert [len(batch) for batch in layout["warm_batches"]] == [32]


def test_c64_unique_layout_preserves_round_robin_alignment() -> None:
    prefixes, _ = BENCHMARK._make_inputs(64, 8, 3, prefix_mode="unique")
    layout = BENCHMARK._build_workload_layout(
        prefixes,
        prefix_mode="unique",
        prefix_group_count=2,
        dp_size=32,
        expected_requests_per_dp=2,
    )

    assert layout["prefix_layout"]["unique_prefixes"] == 64
    assert layout["requests_per_dp"] == 2
    assert layout["cached_prefixes_per_dp"] == 2
    assert layout["warm_batches"] == [prefixes]


def test_c64_grouped_layout_installs_both_prefixes_on_every_dp() -> None:
    prefixes, _ = BENCHMARK._make_inputs(
        64, 8, 3, prefix_mode="grouped", prefix_group_count=2
    )
    layout = BENCHMARK._build_workload_layout(
        prefixes,
        prefix_mode="grouped",
        prefix_group_count=2,
        dp_size=32,
        expected_requests_per_dp=2,
    )

    assert layout["prefix_layout"]["unique_prefixes"] == 2
    assert layout["cached_prefixes_per_dp"] == 2
    assert [len(batch) for batch in layout["warm_batches"]] == [32, 32]


def test_concurrency_layout_rejects_unexpected_requests_per_dp() -> None:
    prefixes, _ = BENCHMARK._make_inputs(64, 8, 3, prefix_mode="shared")

    with pytest.raises(ValueError, match="concurrency invariant failed"):
        BENCHMARK._build_workload_layout(
            prefixes,
            prefix_mode="shared",
            prefix_group_count=2,
            dp_size=16,
            expected_requests_per_dp=2,
        )


def test_random_inputs_are_reproducible_by_seed() -> None:
    first = BENCHMARK._make_inputs(4, 8, 8, prefix_mode="shared", random_seed=7)
    repeated = BENCHMARK._make_inputs(4, 8, 8, prefix_mode="shared", random_seed=7)
    different = BENCHMARK._make_inputs(4, 8, 8, prefix_mode="shared", random_seed=8)

    assert first == repeated
    assert first != different


def test_throughput_metrics_keep_e2e_and_decode_scopes_separate() -> None:
    metrics = BENCHMARK._throughput_metrics(
        {
            "completion_tokens": [4, 4],
            "wall_s": 4.0,
            "decode_batch_wall_s": 2.0,
            "measurement_origin": "scheduler_release",
        }
    )

    assert metrics["output_throughput_tok_s"] == 2.0
    assert metrics["e2e_output_throughput_tok_s"] == 2.0
    assert metrics["e2e_output_tokens"] == 8
    assert metrics["decode_throughput_tok_s"] == 3.0
    assert metrics["decode_output_tokens"] == 6
    assert metrics["output_throughput_scope"] == (
        "scheduler_release_to_last_completion"
    )


def _server_info(
    *, dp_size: int, concurrency: int, max_prefill_tokens: int, per_dp_capacity: int
) -> dict:
    return {
        "dp_size": dp_size,
        "dp_schedule_policy": "round_robin",
        "disable_radix_cache": False,
        "page_size": 64,
        "max_prefill_tokens": max_prefill_tokens,
        "chunked_prefill_size": 2048,
        "max_running_requests": concurrency,
        "context_length": 135168,
        "internal_states": [
            {
                "engine_paused": False,
                "waiting_queue_size": 0,
                "waiting_queue_rids": [],
                "pending_dp_reqs_size": 0,
                "running_batch_size": 0,
                "running_batch_rids": [],
                "chunked_req_rids": [None] * dp_size,
                "memory_usage": {"token_capacity": per_dp_capacity * dp_size},
            }
        ],
    }


@pytest.mark.parametrize(
    (
        "dp_size",
        "concurrency",
        "cached_prefixes_per_dp",
        "max_prefill_tokens",
        "per_dp_capacity",
    ),
    [
        (16, 32, 1, 32768, 230000),
        (16, 32, 2, 32768, 300000),
        (32, 64, 1, 65536, 230000),
        (32, 64, 2, 65536, 426000),
    ],
)
def test_server_configuration_accepts_synchronized_c32_c64_shapes(
    dp_size,
    concurrency,
    cached_prefixes_per_dp,
    max_prefill_tokens,
    per_dp_capacity,
) -> None:
    evidence = BENCHMARK._validate_server_configuration(
        _server_info(
            dp_size=dp_size,
            concurrency=concurrency,
            max_prefill_tokens=max_prefill_tokens,
            per_dp_capacity=per_dp_capacity,
        ),
        concurrency=concurrency,
        dp_size=dp_size,
        requests_per_dp=2,
        cached_prefixes_per_dp=cached_prefixes_per_dp,
        prefix_len=131072,
        extend_len=1024,
        output_len=1024,
    )

    assert evidence["required_global_prefill_tokens"] == max_prefill_tokens
    assert evidence["required_per_dp_chunk_tokens"] == 2048
    assert evidence["requests_per_dp"] == 2


def test_server_configuration_rejects_c32_unique_prefix_capacity_shortfall() -> None:
    with pytest.raises(RuntimeError, match="per-DP token capacity"):
        BENCHMARK._validate_server_configuration(
            _server_info(
                dp_size=16,
                concurrency=32,
                max_prefill_tokens=32768,
                per_dp_capacity=230000,
            ),
            concurrency=32,
            dp_size=16,
            requests_per_dp=2,
            cached_prefixes_per_dp=2,
            prefix_len=131072,
            extend_len=1024,
            output_len=1024,
        )


def test_server_log_validation_requires_one_full_prefill_and_decode(tmp_path) -> None:
    server_log = tmp_path / "server.log"
    layout = [2, 2, 2, 2]
    server_log.write_text(
        "\n".join(
            [
                "Prefill batch. #new-seq: 8, #new-token: 32, "
                "#cached-token: 80, #running-req: 0, "
                f"#prefill per DP: {layout}, #running per DP: {[0] * 4}, "
                "#queue-req: 0,",
                "Decode batch. #running-req: 8, "
                f"#running-req per DP: {layout}, #queue-req: 0,",
            ]
        )
        + "\n"
    )

    evidence = BENCHMARK._validate_measured_server_log(
        server_log,
        start_offset=0,
        concurrency=8,
        dp_size=4,
        requests_per_dp=2,
        extend_len=4,
        expected_cached_tokens=80,
    )

    assert evidence["prefill_batch_count"] == 1
    assert evidence["prefill_per_dp"] == layout
    assert evidence["decode_per_dp"] == layout


def test_server_log_validation_rejects_partial_prefill_waves(tmp_path) -> None:
    server_log = tmp_path / "server.log"
    server_log.write_text(
        "\n".join(
            [
                "Prefill batch. #new-seq: 3, #new-token: 12, "
                "#cached-token: 30, #running-req: 0, "
                "#prefill per DP: [1, 1, 1, 0], #running per DP: [0, 0, 0, 0], "
                "#queue-req: 0,",
                "Prefill batch. #new-seq: 5, #new-token: 20, "
                "#cached-token: 50, #running-req: 3, "
                "#prefill per DP: [1, 1, 1, 2], #running per DP: [1, 1, 1, 0], "
                "#queue-req: 0,",
            ]
        )
        + "\n"
    )

    with pytest.raises(RuntimeError, match="exactly one scheduler batch"):
        BENCHMARK._validate_measured_server_log(
            server_log,
            start_offset=0,
            concurrency=8,
            dp_size=4,
            requests_per_dp=2,
            extend_len=4,
            expected_cached_tokens=80,
        )


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


def test_admission_snapshot_deduplicates_replicated_scheduler_states() -> None:
    expected = {"batch-0", "batch-1"}
    state = {
        "engine_paused": True,
        "waiting_queue_size": 2,
        "waiting_queue_rids": ["batch-0", "batch-1"],
        "pending_dp_reqs_size": 0,
        "running_batch_rids": [],
        "chunked_req_rids": [None, None],
    }

    snapshot = BENCHMARK._admission_snapshot(
        {"internal_states": [state, state.copy()]}, expected
    )

    assert snapshot["complete"] is True
    assert snapshot["waiting_unique_count"] == 2
    assert snapshot["state_waiting_sizes"] == [2, 2]


def test_admission_barrier_queues_one_ordered_native_batch_before_resume() -> None:
    pause_response = mock.Mock()
    pause_response.json.return_value = {"success": True}
    resume_response = mock.Mock()
    resume_response.json.return_value = {"success": True}
    server_info_response = mock.Mock()
    idle_state = {
        "engine_paused": False,
        "waiting_queue_size": 0,
        "waiting_queue_rids": [],
        "pending_dp_reqs_size": 0,
        "running_batch_size": 0,
        "running_batch_rids": [],
        "chunked_req_rids": [None, None],
    }
    scheduler_states = iter(
        [
            [idle_state],
            [
                {
                    **idle_state,
                    "engine_paused": True,
                    "waiting_queue_size": 1,
                    "waiting_queue_rids": ["profile-0"],
                },
                {
                    **idle_state,
                    "engine_paused": True,
                    "waiting_queue_size": 1,
                    "waiting_queue_rids": ["profile-1"],
                },
            ],
            [
                {
                    **idle_state,
                    "engine_paused": True,
                    "waiting_queue_size": 2,
                    "waiting_queue_rids": ["profile-0", "profile-2"],
                },
                {
                    **idle_state,
                    "engine_paused": True,
                    "waiting_queue_size": 2,
                    "waiting_queue_rids": ["profile-1", "profile-3"],
                },
            ],
        ]
    )
    server_info_response.json.side_effect = lambda: {
        "internal_states": next(scheduler_states)
    }
    resumed = threading.Event()

    def fake_post(url, **kwargs):
        assert url.endswith("/set_internal_state")
        if kwargs["json"]["state_data"]["engine_paused"]:
            return pause_response
        resumed.set()
        return resume_response

    def fake_run_native_batch(base_url, input_ids, output_len, *, label, timing_state):
        assert base_url == "http://server"
        assert input_ids == [[1], [2], [3], [4]]
        assert output_len == 1
        assert label == "profile"
        assert resumed.wait(timeout=2)
        return {"wall_s": 1.0}

    on_admitted = mock.Mock()
    with (
        mock.patch.object(BENCHMARK.requests, "post", side_effect=fake_post) as post,
        mock.patch.object(BENCHMARK.requests, "get", return_value=server_info_response),
        mock.patch.object(BENCHMARK.time, "sleep") as sleep,
        mock.patch.object(
            BENCHMARK,
            "_run_native_batch",
            side_effect=fake_run_native_batch,
        ) as run_native_batch,
    ):
        result = BENCHMARK._run_native_batch_with_admission_barrier(
            "http://server",
            [[1], [2], [3], [4]],
            1,
            label="profile",
            on_admitted=on_admitted,
            profile_settle_s=0.25,
        )

    assert result["wall_s"] == 1.0
    assert result["admission_evidence"]["ordered_native_batch"] is True
    assert result["admission_evidence"]["expected_request_count"] == 4
    assert result["admission_evidence"]["snapshot"]["waiting_unique_count"] == 4
    on_admitted.assert_called_once_with()
    run_native_batch.assert_called_once()
    assert post.call_args_list[0].args == ("http://server/set_internal_state",)
    assert post.call_args_list[0].kwargs["json"]["state_data"] == {
        "engine_paused": True
    }
    assert post.call_args_list[1].args == ("http://server/set_internal_state",)
    assert post.call_args_list[1].kwargs["json"]["state_data"] == {
        "engine_paused": False
    }
    pause_response.raise_for_status.assert_called_once_with()
    resume_response.raise_for_status.assert_called_once_with()
    assert server_info_response.raise_for_status.call_count == 3
    assert sleep.call_args_list[-1] == mock.call(0.25)


def test_variant_flag_can_label_radix_topk() -> None:
    parser_source = BENCHMARK_PATH.read_text()

    assert parser_source.count('"--variant"') == 1
    assert '"variant": args.variant' in parser_source
