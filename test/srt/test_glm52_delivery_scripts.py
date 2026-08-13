from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DELIVERY = REPO_ROOT / "benchmark/glm52/delivery"


def _fake_python(tmp_path: Path) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    argv_path = tmp_path / "python-argv.txt"
    executable = bin_dir / "python3"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        ': "${FAKE_PYTHON_ARGV:?}"\n'
        'printf \'%s\\n\' "$@" > "$FAKE_PYTHON_ARGV"\n'
    )
    executable.chmod(0o755)
    return bin_dir, argv_path


def _fake_sgl_eval(tmp_path: Path) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    argv_path = tmp_path / "sgl-eval-argv.txt"
    executable = bin_dir / "sgl-eval"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        ': "${FAKE_SGL_EVAL_ARGV:?}"\n'
        'printf \'%s\\n\' "$@" > "$FAKE_SGL_EVAL_ARGV"\n'
    )
    executable.chmod(0o755)
    return bin_dir, argv_path


def _base_env(tmp_path: Path) -> tuple[dict[str, str], Path]:
    bin_dir, argv_path = _fake_python(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "FAKE_PYTHON_ARGV": str(argv_path),
            "GLM52_SERVER_LOG": "",
        }
    )
    return env, argv_path


@pytest.mark.parametrize(
    (
        "quantization",
        "physical_chips",
        "world",
        "parallel_size",
        "concurrency",
        "max_prefill_tokens",
        "mem_fraction",
    ),
    [
        ("blockwise", 8, 2, 16, 32, 32768, "0.83"),
        ("channelwise", 8, 2, 16, 32, 32768, "0.88"),
        ("blockwise", 16, 4, 32, 64, 65536, "0.83"),
        ("channelwise", 16, 4, 32, 64, 65536, "0.88"),
    ],
)
def test_serve_wrappers_pin_delivery_topology(
    tmp_path,
    quantization,
    physical_chips,
    world,
    parallel_size,
    concurrency,
    max_prefill_tokens,
    mem_fraction,
) -> None:
    env, argv_path = _base_env(tmp_path)
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}\n")
    (model_path / "model.safetensors.index.json").write_text('{"weight_map": {}}\n')
    env.update(
        {
            "WORLD": str(world),
            "RANK": "0",
            "MASTER_ADDR": "rank0",
            "MODEL_PATH": str(model_path),
            "GLM52_SKIP_TUNE_VALIDATION": "1",
            "GLM52_DVFS_P_STATE": "off",
        }
    )

    subprocess.run(
        [str(DELIVERY / "serve" / f"{quantization}_{physical_chips}chip.sh")],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )
    args = argv_path.read_text().splitlines()

    assert args[:3] == ["-m", "sgl_jax.launch_server", "--model-path"]
    assert args[3] == str(model_path)
    assert args[args.index("--tp-size") + 1] == str(parallel_size)
    assert args[args.index("--dp-size") + 1] == str(parallel_size)
    assert args[args.index("--ep-size") + 1] == str(parallel_size)
    assert args[args.index("--max-running-requests") + 1] == str(concurrency)
    assert args[args.index("--precompile-bs-paddings") + 1] == str(concurrency)
    assert args[args.index("--max-prefill-tokens") + 1] == str(max_prefill_tokens)
    assert args[args.index("--precompile-token-paddings") + 1] == str(
        max_prefill_tokens
    )
    assert args[args.index("--mem-fraction-static") + 1] == mem_fraction
    if quantization == "channelwise":
        assert "--quantization-config-path" in args
    else:
        assert "--quantization-config-path" not in args


@pytest.mark.parametrize(
    ("physical_chips", "concurrency", "dp_size", "prefix_mode"),
    [(8, 32, 16, "shared"), (16, 64, 32, "unique")],
)
def test_benchmark_wrappers_pin_concurrency_and_prefix_mode(
    tmp_path, physical_chips, concurrency, dp_size, prefix_mode
) -> None:
    env, argv_path = _base_env(tmp_path)
    server_log = tmp_path / "server.log"
    server_log.touch()
    output = tmp_path / "metrics.jsonl"
    env.update(
        {
            "SERVER_LOG": str(server_log),
            "OUTPUT": str(output),
            "QUANTIZATION": "channelwise",
        }
    )

    subprocess.run(
        [str(DELIVERY / "benchmark" / f"run_{physical_chips}chip.sh")],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )
    args = argv_path.read_text().splitlines()

    assert args[args.index("--concurrency") + 1] == str(concurrency)
    assert args[args.index("--dp-size") + 1] == str(dp_size)
    assert args[args.index("--expected-requests-per-dp") + 1] == "2"
    assert args[args.index("--prefix-mode") + 1] == prefix_mode


def test_serve_wrapper_rejects_wrong_world_before_model_load(tmp_path) -> None:
    env, _ = _base_env(tmp_path)
    env.update({"WORLD": "4", "RANK": "0", "MASTER_ADDR": "rank0"})

    result = subprocess.run(
        [str(DELIVERY / "serve/channelwise_8chip.sh")],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "expected WORLD=2" in result.stderr


@pytest.mark.parametrize(
    ("scope", "expected_examples"), [("quick", "200"), ("full", "1319")]
)
def test_gsm8k_eval_scope_controls_default_example_count(
    tmp_path: Path, scope: str, expected_examples: str
) -> None:
    bin_dir, argv_path = _fake_sgl_eval(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "FAKE_SGL_EVAL_ARGV": str(argv_path),
            "EVAL_SCOPE": scope,
            "OUT_ROOT": str(tmp_path / "output"),
        }
    )

    subprocess.run(
        [str(DELIVERY / "eval/run.sh"), "gsm8k"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )
    args = argv_path.read_text().splitlines()

    assert args[args.index("--num-examples") + 1] == expected_examples
    assert args[args.index("--out-dir") + 1].endswith(f"gsm8k/{scope}")


def test_falcon_manifests_form_deployment_by_scenario_matrix() -> None:
    manifests = sorted((DELIVERY / "falcon").glob("*/*chip/*.yaml"))
    assert len(manifests) == 12

    actual = set()
    for path in manifests:
        quantization, chip_dir, filename = path.relative_to(DELIVERY / "falcon").parts
        chips = int(chip_dir.removesuffix("chip"))
        scenario = filename.removesuffix(".yaml")
        actual.add((quantization, chips, scenario))

        manifest = yaml.safe_load(path.read_text())
        worker = manifest["role_to_task_spec"]["worker"]
        envs = worker["envs"]
        config = json.loads(manifest["config"])
        source = worker["sources"][0]
        assert envs["GLM52_QUANTIZATION"] == quantization
        assert int(envs["GLM52_PHYSICAL_CHIPS"]) == chips
        assert envs["RUN_MODE"] == scenario
        assert config["scenario"] == scenario
        assert config["source_commit"] == envs["SOURCE_COMMIT"] == source["commit"]
        assert re.fullmatch(r"[0-9a-f]{40}", source["commit"])
        assert set(source["commit"]) != {"0"}
        assert "benchmark/glm52/delivery/falcon/runner.sh" in worker["command"]
        assert "cp /tmp" not in worker["command"]
        assert worker["replica"] == (2 if chips == 8 else 4)
        assert worker["device_topo"] == ("2x2x2" if chips == 8 else "2x2x4")
        if scenario != "eval":
            assert config["concurrency"] == (32 if chips == 8 else 64)
            assert config["prefix_mode"] == ("shared" if chips == 8 else "unique")
        if scenario == "profile":
            assert manifest["exp_type"] == "PROFILING"
            assert manifest["artifact_type"] == "trace"
            assert manifest["profile"] is True
        elif scenario == "eval":
            assert envs["EVAL_DATASET"] == "gsm8k"
            assert envs["EVAL_SCOPE"] == "quick"
            assert config["num_examples"] == 200
            assert config["full_num_examples"] == 1319

    assert actual == {
        (quantization, chips, scenario)
        for quantization in ("blockwise", "channelwise")
        for chips in (8, 16)
        for scenario in ("benchmark", "profile", "eval")
    }
