from dataclasses import dataclass, field
from typing import Any, Literal


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
    dry_run_result: Literal["success", "failed"] = "success"


@dataclass(frozen=True)
class AccuracyCase:
    name: str
    dataset: str
    model_id: str
    eval_batch_size: int = 32
    generation_config: dict[str, Any] = field(default_factory=dict)
    limit: int | None = None
    timeout: int | None = None
    dry_run_result: Literal["success", "failed"] = "success"


@dataclass(frozen=True)
class ModelRunConfig:
    model_path: str
    tp_size: int
    dp_size: int
    ep_size: int | None = None
    port: int = 30000
    server_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeConfig:
    nnodes: int
    node_rank: int
    dist_init_addr: str
    host: str = "0.0.0.0"
    port: int = 30000


# --host / --port / --model-path are passed to popen_launch_server through
# dedicated kwargs and must not be set again via ModelRunConfig.server_args.
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


def build_other_server_args(model_cfg: ModelRunConfig, runtime_cfg: RuntimeConfig) -> list[str]:
    """Build the ``other_args`` list for popen_launch_server.

    --model-path / --host / --port are passed via dedicated popen_launch_server
    kwargs and are intentionally absent here.
    """
    _validate_user_server_args(model_cfg.server_args)

    args = [
        "--tp-size",
        str(model_cfg.tp_size),
        "--dp-size",
        str(model_cfg.dp_size),
    ]
    if model_cfg.ep_size is not None:
        args.extend(["--ep-size", str(model_cfg.ep_size)])

    args.extend(
        [
            "--nnodes",
            str(runtime_cfg.nnodes),
            "--node-rank",
            str(runtime_cfg.node_rank),
            "--dist-init-addr",
            runtime_cfg.dist_init_addr,
        ]
    )
    args.extend(model_cfg.server_args)
    return args


@dataclass(frozen=True)
class ModelRun:
    name: str
    model: ModelRunConfig
    cases: list[PerfCase | AccuracyCase]


@dataclass(frozen=True)
class MultiHostSuite:
    name: str
    target: str
    runs: list[ModelRun]


def dry_run_suite(suite: MultiHostSuite) -> dict:
    runs = [_dry_run_model_run(run) for run in suite.runs]
    result = "failed" if any(run["result"] == "failed" for run in runs) else "success"

    return {
        "suite": suite.name,
        "target": suite.target,
        "result": result,
        "runs": runs,
    }


def _dry_run_model_run(run: ModelRun) -> dict:
    cases = [_dry_run_case(case) for case in run.cases]
    result = "failed" if any(case["result"] == "failed" for case in cases) else "success"

    return {
        "name": run.name,
        "model_path": run.model.model_path,
        "parallelism": {
            "tp_size": run.model.tp_size,
            "dp_size": run.model.dp_size,
            "ep_size": run.model.ep_size,
        },
        "result": result,
        "cases": cases,
    }


def _dry_run_case(case: PerfCase | AccuracyCase) -> dict:
    if isinstance(case, PerfCase):
        return {
            "name": case.name,
            "type": "perf",
            "result": case.dry_run_result,
        }

    return {
        "name": case.name,
        "type": "accuracy",
        "result": case.dry_run_result,
    }
