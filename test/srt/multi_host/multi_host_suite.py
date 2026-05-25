from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PerfCase:
    name: str
    input_len: int
    output_len: int
    num_prompts: int
    max_concurrency: int
    request_rate: float = float("inf")
    seed: int = 42
    flush_cache: bool = False


@dataclass(frozen=True)
class AccuracyCase:
    name: str
    dataset: str
    model_id: str
    eval_batch_size: int = 32
    generation_config: dict[str, Any] = field(default_factory=dict)
    limit: int | None = None
    timeout: int | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    nnodes: int
    node_rank: int
    dist_init_addr: str
    host: str = "0.0.0.0"
    port: int = 30000


# --host / --port / --model-path are passed to popen_launch_server through
# dedicated kwargs and must not be set again via LaunchProfile.server_args.
_RUNTIME_MANAGED_SERVER_ARGS = frozenset(
    {
        "--model-path",
        "--tp-size",
        "--tensor-parallel-size",
        "--dp-size",
        "--data-parallel-size",
        "--ep-size",
        "--nnodes",
        "--node-rank",
        "--dist-init-addr",
        "--host",
        "--port",
    }
)


def _validate_user_server_args(server_args: tuple[str, ...]) -> None:
    for arg in server_args:
        name = arg.split("=", 1)[0]
        if name in _RUNTIME_MANAGED_SERVER_ARGS:
            raise ValueError(f"{name} is managed by the multi-host runtime")


@dataclass(frozen=True)
class ModelRun:
    launch_profile: str  # path to YAML profile; suite_runner resolves to absolute
    cases: list[PerfCase | AccuracyCase]


@dataclass(frozen=True)
class MultiHostSuite:
    name: str
    runs: list[ModelRun]


def dry_run_suite(suite: MultiHostSuite) -> dict:
    return {
        "suite": suite.name,
        "runs": [_dry_run_model_run(run) for run in suite.runs],
    }


def _dry_run_model_run(run: ModelRun) -> dict:
    from profile_loader import load_profile

    profile = load_profile(run.launch_profile)
    return {
        "name": profile.name,
        "target": profile.target,
        "launch_profile": run.launch_profile,
        "model_path": profile.model_path,
        "parallelism": {
            "tp_size": profile.tp_size,
            "dp_size": profile.dp_size,
            "ep_size": profile.ep_size,
        },
        "cases": [_dry_run_case(case) for case in run.cases],
    }


def _dry_run_case(case: PerfCase | AccuracyCase) -> dict:
    return {
        "name": case.name,
        "type": "perf" if isinstance(case, PerfCase) else "accuracy",
    }
