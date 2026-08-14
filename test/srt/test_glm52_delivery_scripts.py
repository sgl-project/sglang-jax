from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DELIVERY = REPO_ROOT / "benchmark/glm52/delivery"


def _load_long_context_validator():
    path = DELIVERY / "validation/validate_openai_long_context.py"
    spec = importlib.util.spec_from_file_location("validate_openai_long_context", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_serve_wrapper_accepts_agent_eval_capacity_overrides(tmp_path: Path) -> None:
    env, argv_path = _base_env(tmp_path)
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}\n")
    (model_path / "model.safetensors.index.json").write_text('{"weight_map": {}}\n')
    env.update(
        {
            "WORLD": "2",
            "RANK": "0",
            "MASTER_ADDR": "rank0",
            "MODEL_PATH": str(model_path),
            "GLM52_SKIP_TUNE_VALIDATION": "1",
            "GLM52_DVFS_P_STATE": "off",
            "GLM52_CONTEXT_LENGTH": "202752",
            "GLM52_MAX_RUNNING_REQUESTS": "32",
            "GLM52_PRECOMPILE_BS_PADDING": "32",
            "MEM_FRACTION_STATIC": "0.89",
        }
    )

    subprocess.run(
        [str(DELIVERY / "serve/channelwise_8chip.sh")],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )
    args = argv_path.read_text().splitlines()
    assert args[args.index("--context-length") + 1] == "202752"
    assert args[args.index("--max-running-requests") + 1] == "32"
    assert args[args.index("--precompile-bs-paddings") + 1] == "32"
    assert args[args.index("--mem-fraction-static") + 1] == "0.89"


def test_serve_wrapper_rejects_max_running_below_fused_moe_minimum(
    tmp_path: Path,
) -> None:
    env, _ = _base_env(tmp_path)
    env.update(
        {
            "WORLD": "2",
            "RANK": "0",
            "MASTER_ADDR": "rank0",
            "GLM52_MAX_RUNNING_REQUESTS": "16",
        }
    )

    result = subprocess.run(
        [str(DELIVERY / "serve/channelwise_8chip.sh")],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "below the fused-MoE minimum 2 * EP=32" in result.stderr


def test_long_context_probe_builds_at_least_target_tokens() -> None:
    validator = _load_long_context_validator()

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            del kwargs
            return list(range(len(messages[0]["content"]) // 4 + 8))

    content, token_count = validator._build_prompt(FakeTokenizer(), 2_000)
    assert content.endswith("Reply with exactly OK.")
    assert token_count >= 2_000


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


def test_evalscope_trace_audit_separates_integrity_from_quality(tmp_path: Path) -> None:
    work_dir = tmp_path / "run"
    review_dir = work_dir / "reviews" / "model"
    review_dir.mkdir(parents=True)
    review_path = review_dir / "officeqa.jsonl"
    output = tmp_path / "audit.json"
    row = {
        "index": 7,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "reasoning", "reasoning": "I should inspect the files."},
                    {"type": "text", "text": ""},
                ],
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "bash",
                            "arguments": {"command": "grep needle file"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "[stderr]\nno match\n[exit 1]",
            },
        ],
        "agent_trace": {
            "framework": "native",
            "strategy": "function_calling",
            "environment": "local",
            "max_steps": 15,
            "events": [
                {"type": "model_generate", "timestamp": 1, "payload": {}},
                {
                    "type": "tool_call",
                    "timestamp": 2,
                    "payload": {"id": "call-1", "name": "bash"},
                },
                {"type": "tool_result", "timestamp": 3, "payload": {"id": "call-1"}},
                {"type": "submit", "timestamp": 4, "payload": {}},
            ],
        },
        "sample_score": {"score": {"accuracy": 0}},
    }
    review_path.write_text(json.dumps(row) + "\n")
    command = [
        "python3",
        str(DELIVERY / "validation/audit_evalscope_agent_trace.py"),
        "--work-dir",
        str(work_dir),
        "--expected-samples",
        "1",
        "--expected-max-steps",
        "15",
        "--require-tools",
        "--require-reasoning-separation",
        "--output",
        str(output),
    ]

    subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    report = json.loads(output.read_text())
    assert report["passed"] is True
    assert report["reasoning_part_count"] == 1
    assert report["nonzero_tool_exit_codes"] == {"1": 1}
    assert "exit 1 x1" in report["quality_issues"][0]

    row["messages"][0]["content"][1]["text"] = "</think> leaked"
    review_path.write_text(json.dumps(row) + "\n")
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert json.loads(output.read_text())["passed"] is False


def test_falcon_manifests_form_deployment_by_scenario_matrix() -> None:
    manifests = sorted(
        path
        for path in (DELIVERY / "falcon").glob("*/*chip/*.yaml")
        if path.stem in {"benchmark", "profile", "eval"}
    )
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


def test_officeqa_smoke_manifest_pins_agent_eval_runtime() -> None:
    path = DELIVERY / "falcon/channelwise/8chip/officeqa_smoke.yaml"
    manifest = yaml.safe_load(path.read_text())
    worker = manifest["role_to_task_spec"]["worker"]
    config = json.loads(manifest["config"])
    source = worker["sources"][0]

    assert worker["replica"] == 2
    assert worker["device_type"] == "v7x"
    assert worker["device_topo"] == "2x2x2"
    assert worker["envs"]["RUN_MODE"] == "agent_eval"
    assert worker["envs"]["EVALSCOPE_LIMIT"] == "16"
    assert worker["envs"]["EVALSCOPE_BATCH_SIZE"] == "16"
    assert worker["envs"]["EVALSCOPE_ENABLE_THINKING"] == "true"
    assert worker["envs"]["GLM52_DP_SCHEDULE_POLICY"] == "cache_aware"
    assert worker["envs"]["GLM52_CONTEXT_LENGTH"] == "202752"
    assert worker["envs"]["GLM52_MAX_RUNNING_REQUESTS"] == "32"
    assert worker["envs"]["MEM_FRACTION_STATIC"] == "0.89"
    assert (
        config["source_commit"] == worker["envs"]["SOURCE_COMMIT"] == source["commit"]
    )
    assert config["evalscope_commit"] == worker["envs"]["EVALSCOPE_COMMIT"]
    assert config["reasoning_preflight"] is True
    assert config["bash_output_policy"] == "unmodified"
    assert config["context_length"] == 202_752
    assert config["max_running_requests"] == 32
    assert config["mem_fraction_static"] == 0.89
    assert config["minimum_kv_headroom_tokens"] == 8192
    assert config["long_context_preflight_tokens"] == 150_000
    assert (
        config["quant_config"]
        == "fp8_glm52_static_per_channel_moe_w8a8_linear_w8a16.yaml"
    )
