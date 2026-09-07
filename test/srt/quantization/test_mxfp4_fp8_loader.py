"""CPU tests for the load-time MXFP4 to resident FP8 conversion path."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from sgl_jax.srt.utils.quantization.mxfp4_fp8_loader import (
    Mxfp4ConversionError,
    convert_mxfp4_pair,
    convert_mxfp4_pair_from_reader,
)


def _pack_codes(codes: np.ndarray) -> np.ndarray:
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)


def test_reader_is_row_bounded_and_returns_existing_moe_layout() -> None:
    rows, columns = 5, 256
    codes = np.tile(np.arange(16, dtype=np.uint8), rows * columns // 16).reshape(rows, columns)
    packed = _pack_codes(codes)
    scales = np.full((rows, columns // 32), 127, dtype=np.uint8)
    weight_reads: list[int] = []
    scale_reads: list[int] = []

    def read_weight(row_slice: slice) -> np.ndarray:
        weight_reads.append(row_slice.stop - row_slice.start)
        return packed[row_slice]

    def read_scale(row_slice: slice) -> np.ndarray:
        scale_reads.append(row_slice.stop - row_slice.start)
        return scales[row_slice]

    converted = convert_mxfp4_pair_from_reader(
        weight_name="layers.0.ffn.experts.0.w1.weight",
        scale_name="layers.0.ffn.experts.0.w1.scale",
        weight_shape=packed.shape,
        scale_shape=scales.shape,
        weight_dtype="I8",
        scale_dtype="F8_E8M0",
        read_weight_rows=read_weight,
        read_scale_rows=read_scale,
        row_chunk_size=2,
    )

    assert max(weight_reads) <= 2
    assert max(scale_reads) <= 2
    assert converted.report.exact
    assert converted.report.measured_peak_python_bytes is None
    kernel_weight, kernel_scale = converted.as_moe_kernel_layout()
    assert kernel_weight.shape == (1, columns, rows)
    assert kernel_scale.shape == (1, 1, 1, rows)
    assert str(kernel_weight.dtype) == "float8_e4m3fn"
    assert kernel_scale.dtype == np.float32


def test_i8_storage_bits_are_preserved_and_names_are_exact() -> None:
    packed = np.asarray([[0xF1, 0x2E] * 16], dtype=np.uint8)
    scales = np.full((1, 2), 127, dtype=np.uint8)
    converted = convert_mxfp4_pair(
        packed.view(np.int8),
        scales.view(np.int8),
        weight_name="layers.0.ffn.experts.0.w2.weight",
        scale_name="layers.0.ffn.experts.0.w2.scale",
    )
    assert converted.report.exact
    with pytest.raises(Mxfp4ConversionError, match="pair mismatch"):
        convert_mxfp4_pair(
            packed,
            scales,
            weight_name="layers.0.ffn.experts.0.w2.weight",
            scale_name="layers.0.ffn.experts.0.w2.weight_scale",
        )
    with pytest.raises(Mxfp4ConversionError, match="I8/F8_E8M0"):
        convert_mxfp4_pair_from_reader(
            weight_name="layers.0.ffn.experts.0.w2.weight",
            scale_name="layers.0.ffn.experts.0.w2.scale",
            weight_shape=packed.shape,
            scale_shape=scales.shape,
            weight_dtype="F32",
            scale_dtype="F8_E8M0",
            read_weight_rows=lambda rows: packed[rows],
            read_scale_rows=lambda rows: scales[rows],
        )


def test_strict_mode_rejects_nonzero_fp8_underflow() -> None:
    # Two K32 groups with a 2**127 scale ratio cannot share one FP8 row scale
    # without erasing the small group. The default must fail closed.
    codes = np.full((1, 128), 2, dtype=np.uint8)  # E2M1 value 1.0
    packed = _pack_codes(codes)
    scales = np.asarray([[127, 0, 0, 0]], dtype=np.uint8)
    with pytest.raises(Mxfp4ConversionError, match="lossy"):
        convert_mxfp4_pair(
            packed,
            scales,
            weight_name="layers.0.ffn.experts.0.w3.weight",
            scale_name="layers.0.ffn.experts.0.w3.scale",
        )
    converted = convert_mxfp4_pair(
        packed,
        scales,
        weight_name="layers.0.ffn.experts.0.w3.weight",
        scale_name="layers.0.ffn.experts.0.w3.scale",
        strict=False,
    )
    assert converted.report.underflow_count > 0
    assert not converted.report.exact


def test_real_safetensors_06_f8_e8m0_roundtrip(tmp_path: Path) -> None:
    """Exercise the actual safetensors file format, including F8_E8M0 bytes."""

    ml_dtypes = pytest.importorskip("ml_dtypes")
    save_file = pytest.importorskip("safetensors.numpy").save_file
    weight_name = "layers.0.ffn.experts.0.w1.weight"
    scale_name = "layers.0.ffn.experts.0.w1.scale"
    codes = np.tile(np.arange(16, dtype=np.uint8), 16).reshape(1, 256)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8).view(np.int8)
    scales = np.full((1, 8), 127, dtype=np.uint8).view(ml_dtypes.float8_e8m0fnu)
    path = tmp_path / "model-00001.safetensors"
    save_file({weight_name: packed, scale_name: scales}, str(path))

    from sgl_jax.srt.utils.quantization.mxfp4_fp8_loader import (
        convert_mxfp4_pair_from_safetensors,
        inspect_mxfp4_checkpoint,
    )

    converted = convert_mxfp4_pair_from_safetensors(
        path,
        weight_name,
        scale_name,
        row_chunk_size=1,
    )
    assert converted.report.exact
    assert converted.scale_fp32.dtype == np.float32
    assert converted.as_moe_kernel_layout()[0].shape == (1, 256, 1)

    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {weight_name: path.name, scale_name: path.name}}))
    indexed_report = inspect_mxfp4_checkpoint(tmp_path, row_chunk_size=1)
    assert indexed_report["all_exact"]
    assert indexed_report["checkpoint_coverage"]["coverage_complete"]

    index.unlink()
    sample_report = inspect_mxfp4_checkpoint(tmp_path, row_chunk_size=1)
    assert sample_report["all_converted_tensors_exact"]
    assert not sample_report["all_exact"]
    assert not sample_report["checkpoint_coverage"]["coverage_complete"]


def test_real_sample_roundtrip_when_evidence_dir_is_provided() -> None:
    evidence_dir = os.environ.get("M02_REAL_SAMPLE_DIR")
    if not evidence_dir:
        pytest.skip("set M02_REAL_SAMPLE_DIR to run the captured DeepSeek-V4 sample")
    root = Path(evidence_dir)
    manifests = sorted(root.glob("real-*-sample-manifest.json"))
    assert manifests, f"no sample manifests in {root}"
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        records = {record["name"]: record for record in manifest["tensors"]}
        for weight_name, weight_record in records.items():
            if not weight_name.endswith(".weight"):
                continue
            scale_name = f"{weight_name[:-len('.weight')]}.scale"
            scale_record = records[scale_name]
            assert weight_record["declared_dtype"] == "I8"
            assert scale_record["declared_dtype"] == "F8_E8M0"
            packed = np.fromfile(root / weight_record["raw_file"], dtype=np.uint8).reshape(
                weight_record["shape"]
            )
            scales = np.fromfile(root / scale_record["raw_file"], dtype=np.uint8).reshape(
                scale_record["shape"]
            )
            converted = convert_mxfp4_pair(
                packed,
                scales,
                weight_name=weight_name,
                scale_name=scale_name,
                row_chunk_size=127,
            )
            assert converted.report.exact
            assert converted.report.underflow_count == 0
            assert converted.as_moe_kernel_layout()[0].shape[1:] == (
                converted.logical_shape[1],
                converted.logical_shape[0],
            )
