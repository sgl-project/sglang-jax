from pathlib import Path

import yaml
from multi_host_suite import RuntimeConfig, _validate_user_server_args
from pydantic import BaseModel, Field, field_validator


class LaunchProfile(BaseModel):
    name: str
    target: str
    model_path: str
    tp_size: int = Field(gt=0)
    dp_size: int = Field(gt=0)
    ep_size: int | None = Field(default=None, gt=0)
    port: int = 30000
    server_args: tuple[str, ...] = ()

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
