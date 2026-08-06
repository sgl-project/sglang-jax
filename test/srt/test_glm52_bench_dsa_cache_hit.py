from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest import mock

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

    with mock.patch.object(
        BENCHMARK.requests,
        "post",
        side_effect=(start_response, stop_response),
    ) as post:
        with mock.patch.object(
            BENCHMARK.requests,
            "get",
            side_effect=(running_response, idle_response),
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

    with mock.patch.object(BENCHMARK.requests, "get", return_value=idle_response):
        with mock.patch.object(BENCHMARK.requests, "post") as post:
            BENCHMARK._stop_profile("http://server")

    post.assert_not_called()
    idle_response.raise_for_status.assert_called_once_with()


def test_variant_flag_can_label_radix_topk() -> None:
    parser_source = BENCHMARK_PATH.read_text()

    assert '"--variant"' in parser_source
    assert '"variant": args.variant' in parser_source
