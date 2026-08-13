from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import ml_dtypes
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

# safetensors' NumPy adapter still looks up the optional float8 dtype on the
# numpy module in some released versions. ml_dtypes owns that dtype today.
if not hasattr(np, "float8_e4m3fn"):
    np.float8_e4m3fn = ml_dtypes.float8_e4m3fn

REPO_ROOT = Path(__file__).resolve().parents[2]
CONVERTER_DIR = REPO_ROOT / "benchmark/glm52/delivery/convert"


def _write_toy_checkpoint(source: Path) -> dict[str, np.ndarray]:
    source.mkdir()
    tensors = {
        "model.embed_tokens.weight": np.arange(4, dtype=np.float32)
        .reshape(2, 2)
        .astype(ml_dtypes.bfloat16),
        "model.layers.0.self_attn.q_a_proj.weight": np.array(
            [[0.0, 0.0, 0.0, 0.0], [-2.0, -1.0, 1.0, 2.0]],
            dtype=ml_dtypes.bfloat16,
        ),
        "model.layers.0.self_attn.q_a_layernorm.weight": np.ones(
            2, dtype=ml_dtypes.bfloat16
        ),
        "model.layers.0.mlp.gate.weight": np.arange(8, dtype=np.float32)
        .reshape(2, 4)
        .astype(ml_dtypes.bfloat16),
        "model.layers.0.mlp.experts.0.down_proj.weight": np.array(
            [[-4.0, -3.0, 2.0, 1.0], [0.25, -0.5, 0.75, -1.0]],
            dtype=ml_dtypes.bfloat16,
        ),
        "model.layers.0.indexer.wq_b.weight": np.array(
            [[1.0, 2.0], [-3.0, 4.0]], dtype=ml_dtypes.bfloat16
        ),
        "model.layers.0.indexer.weights_proj.weight": np.arange(4, dtype=np.float32)
        .reshape(2, 2)
        .astype(ml_dtypes.bfloat16),
    }
    shard_names = [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    shard_tensors = [
        dict(list(tensors.items())[:4]),
        dict(list(tensors.items())[4:]),
    ]
    weight_map = {}
    for shard_name, shard in zip(shard_names, shard_tensors, strict=True):
        save_file(shard, source / shard_name, metadata={"format": "pt"})
        weight_map.update({name: shard_name for name in shard})

    total_size = sum(tensor.nbytes for tensor in tensors.values())
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            indent=2,
        )
        + "\n"
    )
    (source / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Glm4MoeForCausalLM"],
                "quantization_config": {"quant_method": "old"},
            },
            indent=2,
        )
        + "\n"
    )
    (source / "tokenizer_config.json").write_text('{"model_max_length": 135168}\n')
    (source / "_DOWNLOAD_COMPLETE").write_text("source complete\n")
    return tensors


def _run_wrapper(
    tmp_path: Path,
    source: Path,
    staging: Path,
    final: Path,
    run_id: str,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHON": sys.executable,
            "SOURCE_MODEL": str(source),
            "STAGING_MODEL": str(staging),
            "TARGET_MODEL": str(final),
            "LOCAL_ROOT": str(tmp_path / "local"),
            "ARTIFACT_ROOT": str(tmp_path / "artifacts"),
            "WORKERS": "2",
            "RUN_ID": run_id,
            "CHUNK_ELEMENTS": "4",
            "BARRIER_TIMEOUT": "30",
            "EXPECTED_SHARDS": "2",
            "EXPECTED_SELECTED_TENSORS": "3",
            "EXPECTED_WEIGHT_MAP_COUNT": "10",
        }
    )
    return subprocess.run(
        [str(CONVERTER_DIR / "run.sh")],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )


def test_channelwise_converter_is_resumable_and_preserves_quantization_policy(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    final = tmp_path / "final"
    tensors = _write_toy_checkpoint(source)

    first = _run_wrapper(tmp_path, source, staging, final, "first")
    assert first.returncode == 0, first.stdout + first.stderr

    marker = json.loads((final / "_DOWNLOAD_COMPLETE").read_text())
    assert marker["schema"] == "glm52-fp8-e4m3fn-output-channel-v1"
    assert marker["shard_count"] == 2
    assert marker["selected_tensors"] == 3
    assert marker["weight_map_count"] == 10
    assert not (final / "_CONVERSION_IN_PROGRESS").exists()

    index = json.loads((final / "model.safetensors.index.json").read_text())
    assert len(index["weight_map"]) == 10
    assert index["metadata"]["total_size"] == marker["tensor_data_bytes"]
    config = json.loads((final / "config.json").read_text())
    assert config["quantization_config"] == {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": None,
    }
    assert (final / "tokenizer_config.json").read_text() == (
        source / "tokenizer_config.json"
    ).read_text()

    selected = {
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
        "model.layers.0.indexer.wq_b.weight",
    }
    loaded = {}
    for shard_name in sorted(set(index["weight_map"].values())):
        with safe_open(final / shard_name, framework="np") as handle:
            loaded.update({name: handle.get_tensor(name) for name in handle.keys()})

    for name in selected:
        scale_name = name.removesuffix(".weight") + ".weight_scale_inv"
        quantized = loaded[name]
        scale = loaded[scale_name]
        assert quantized.dtype == np.dtype(ml_dtypes.float8_e4m3fn)
        assert scale.dtype == np.float32
        expected_scale = (
            np.max(np.abs(tensors[name].astype(np.float32)), axis=1) / 448.0
        )
        np.testing.assert_array_equal(scale, expected_scale)
        reconstructed = quantized.astype(np.float32) * scale[:, None]
        np.testing.assert_allclose(
            reconstructed,
            tensors[name].astype(np.float32),
            rtol=0.13,
            atol=float(expected_scale.max()),
        )
    assert np.all(loaded["model.layers.0.self_attn.q_a_proj.weight"][0] == 0)
    assert loaded["model.layers.0.self_attn.q_a_proj.weight_scale_inv"][
        0
    ] == np.float32(0)

    for name in set(tensors) - selected:
        assert name in loaded
        assert name.removesuffix(".weight") + ".weight_scale_inv" not in loaded
        np.testing.assert_array_equal(loaded[name], tensors[name])

    second = _run_wrapper(tmp_path, source, staging, final, "second")
    assert second.returncode == 0, second.stdout + second.stderr
    assert second.stdout.count("GLM52_CHANNELWISE_CONVERSION_ALREADY_COMPLETE") == 2


def test_wrapper_stops_peers_when_revision_expectation_fails(tmp_path: Path) -> None:
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    final = tmp_path / "final"
    _write_toy_checkpoint(source)

    env = os.environ.copy()
    env.update(
        {
            "PYTHON": sys.executable,
            "SOURCE_MODEL": str(source),
            "STAGING_MODEL": str(staging),
            "TARGET_MODEL": str(final),
            "LOCAL_ROOT": str(tmp_path / "local"),
            "ARTIFACT_ROOT": str(tmp_path / "artifacts"),
            "WORKERS": "2",
            "RUN_ID": "wrong-expectation",
            "BARRIER_TIMEOUT": "30",
            "EXPECTED_SHARDS": "2",
            "EXPECTED_SELECTED_TENSORS": "4",
            "EXPECTED_WEIGHT_MAP_COUNT": "11",
        }
    )
    result = subprocess.run(
        [str(CONVERTER_DIR / "run.sh")],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
    )

    assert result.returncode == 1
    assert "selected_tensors mismatch" in result.stderr
    assert "peers were stopped" in result.stderr
    assert not (final / "_DOWNLOAD_COMPLETE").exists()
