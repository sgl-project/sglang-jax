"""Host-neutral launch contract: how to start a server for a nightly case.

``LaunchProfile`` describes a server to launch (model path, parallelism,
extra server args) and is loaded from a YAML profile under a ``launch_profiles/``
directory. ``RuntimeConfig`` carries the per-launch runtime coordinates
(nnodes / node rank / dist-init address): single-host runners build a
``nnodes=1`` localhost config, while the multi-host runner derives one from the
GKE pod environment. ``build_other_server_args`` turns a profile + runtime into
the ``other_args`` list for ``popen_launch_server``.

Shared by the single-host runner (``test/srt/nightly/single_host/accuracy_case_runner.py``) and the
multi-host package (``test/srt/nightly/multi_host/``) so both read one profile schema.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

# --host / --port / --model-path and the parallelism flags are passed via
# dedicated popen_launch_server kwargs / LaunchProfile fields / RuntimeConfig and
# must not be set again through LaunchProfile.server_args.
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
            raise ValueError(f"{name} is managed by the launch runtime")


@dataclass(frozen=True)
class RuntimeConfig:
    nnodes: int
    node_rank: int
    dist_init_addr: str
    host: str = "0.0.0.0"
    port: int = 30000


class LaunchProfile(BaseModel):
    name: str
    target: str
    model_path: str
    tp_size: int = Field(gt=0)
    dp_size: int = Field(gt=0)
    ep_size: int | None = Field(default=None, gt=0)
    port: int = 30000
    server_args: tuple[str, ...] = ()
    # Strict precompile guard (SGLANG_JAX_ENABLE_CACHE_MISS_CHECK): crash the
    # server if a request hits an un-precompiled shape. Right for fixed-shape perf
    # sweeps; set False for client-driven benches with dynamic shapes (the
    # recurrent reuse-sweep / A/B vary prompt length, batch, and K).
    check_cache_miss: bool = True

    @field_validator("server_args")
    @classmethod
    def _no_runtime_managed(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        _validate_user_server_args(v)
        return v


def load_profile(profile_path: str | Path) -> LaunchProfile:
    data = yaml.safe_load(Path(profile_path).read_text())
    return LaunchProfile(**data)


def build_other_server_args(profile: LaunchProfile, runtime_cfg: RuntimeConfig) -> list[str]:
    """Build the ``other_args`` list for popen_launch_server.

    --model-path / --host / --port are passed via dedicated popen_launch_server
    kwargs and are intentionally absent here.
    """
    args = [
        "--tp-size",
        str(profile.tp_size),
        "--dp-size",
        str(profile.dp_size),
    ]
    if profile.ep_size is not None:
        args.extend(["--ep-size", str(profile.ep_size)])

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
    args.extend(profile.server_args)
    return args
