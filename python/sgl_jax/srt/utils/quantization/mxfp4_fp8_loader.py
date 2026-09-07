"""Bounded load-time conversion for DeepSeek-V4 MXFP4 expert tensors.

DeepSeek-V4 Flash checkpoints store expert weights as packed ``I8`` bytes and
their per-K32 scales as ``F8_E8M0`` bytes.  This module converts one exact
``*.weight``/``*.scale`` pair to resident E4M3FN FP8 with one FP32 power-of-two
scale per output row.  It intentionally does not change a global quantization
configuration or write a converted checkpoint.

The reader API accepts row callbacks so a loader can keep the decoded FP32
working set bounded.  ``convert_mxfp4_pair_from_safetensors`` is the direct
single-pair entry point; ``inspect_mxfp4_checkpoint`` is a report-only whole
checkpoint verifier for M0.2.  The returned kernel layout is the existing
fused MoE layout: weight ``[E,K,N]`` and scale ``[E,1,1,N]``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np

FP4_CODEBOOK = np.asarray(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)
FP8_DTYPE = ml_dtypes.float8_e4m3fn
FP8_MAX = np.float32(448.0)
E8M0_MIN_EXPONENT = -127
E8M0_INVALID_CODE = 255
M0_2_SCHEMA = "deepseek-v4-m02-load-time-mxfp4-fp8-v1"
MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024


class Mxfp4ConversionError(ValueError):
    """Raised when a source pair is malformed or cannot be losslessly promoted."""


def _dtype_name(dtype: Any) -> str:
    """Return a safetensors-style dtype name without importing safetensors."""

    return dtype if isinstance(dtype, str) else str(dtype)


def _as_u8_bytes(array: np.ndarray, *, name: str) -> np.ndarray:
    """Preserve source byte bits for signed I8 or float8-backed arrays."""

    value = np.asarray(array)
    if value.dtype == np.uint8:
        return np.ascontiguousarray(value)
    if value.dtype == np.int8:
        return np.ascontiguousarray(value.view(np.uint8))
    if value.dtype == np.dtype(ml_dtypes.float8_e8m0fnu):
        return np.ascontiguousarray(value.view(np.uint8))
    raise Mxfp4ConversionError(
        f"{name} must expose one-byte storage (uint8/int8/F8_E8M0), got {value.dtype}"
    )


def _validate_names(weight_name: str, scale_name: str) -> None:
    expected = f"{weight_name[:-len('.weight')]}.scale" if weight_name.endswith(".weight") else None
    if expected is None:
        raise Mxfp4ConversionError(f"MXFP4 weight name must end in .weight, got {weight_name!r}")
    if scale_name != expected:
        raise Mxfp4ConversionError(
            f"MXFP4 source pair mismatch: {weight_name!r} requires {expected!r}, "
            f"got {scale_name!r}"
        )


def _validate_shapes(
    packed_shape: Sequence[int],
    scale_shape: Sequence[int],
    *,
    weight_name: str,
    scale_name: str,
) -> tuple[int, int]:
    if len(packed_shape) != 2 or len(scale_shape) != 2:
        raise Mxfp4ConversionError(
            f"{weight_name}/{scale_name} must be rank-2, got "
            f"{tuple(packed_shape)}/{tuple(scale_shape)}"
        )
    rows, packed_columns = (int(packed_shape[0]), int(packed_shape[1]))
    scale_rows, scale_columns = (int(scale_shape[0]), int(scale_shape[1]))
    columns = packed_columns * 2
    if rows <= 0 or packed_columns <= 0 or columns % 32:
        raise Mxfp4ConversionError(
            f"{weight_name} shape={tuple(packed_shape)} implies invalid K={columns}; "
            "K must be a positive multiple of 32"
        )
    expected_scale_shape = (rows, columns // 32)
    if (scale_rows, scale_columns) != expected_scale_shape:
        raise Mxfp4ConversionError(
            f"{scale_name} shape={tuple(scale_shape)} does not match {weight_name} "
            f"shape={tuple(packed_shape)}; expected {expected_scale_shape}"
        )
    return rows, columns


def _decode_e8m0(scale_u8: np.ndarray, *, name: str) -> np.ndarray:
    if np.any(scale_u8 == E8M0_INVALID_CODE):
        raise Mxfp4ConversionError(f"{name} contains reserved F8_E8M0 code 255")
    exponents = scale_u8.astype(np.int16) + E8M0_MIN_EXPONENT
    decoded = np.ldexp(np.ones(scale_u8.shape, dtype=np.float32), exponents)
    if not np.all(np.isfinite(decoded)) or np.any(decoded <= 0):
        raise Mxfp4ConversionError(f"{name} has non-finite or non-positive F8_E8M0 scales")
    return decoded


def _decode_chunk(packed: np.ndarray, scale_e8m0: np.ndarray, *, scale_name: str) -> np.ndarray:
    packed_u8 = _as_u8_bytes(packed, name="packed MXFP4 weight")
    scale_u8 = _as_u8_bytes(scale_e8m0, name="F8_E8M0 scale")
    rows, packed_columns = packed_u8.shape
    columns = packed_columns * 2
    nibbles = np.empty((rows, columns), dtype=np.uint8)
    nibbles[:, 0::2] = packed_u8 & np.uint8(0x0F)
    nibbles[:, 1::2] = packed_u8 >> np.uint8(4)
    values = FP4_CODEBOOK[nibbles]
    scales = _decode_e8m0(scale_u8, name=scale_name)
    decoded = (values.reshape(rows, columns // 32, 32) * scales[:, :, None]).reshape(rows, columns)
    if not np.all(np.isfinite(decoded)):
        raise Mxfp4ConversionError(f"decoded MXFP4 pair {scale_name!r} is non-finite")
    return np.asarray(decoded, dtype=np.float32)


def _power2_scales(decoded: np.ndarray, *, weight_name: str) -> np.ndarray:
    absmax = np.max(np.abs(decoded), axis=1).astype(np.float64)
    ratios = absmax / float(FP8_MAX)
    scales = np.ones(decoded.shape[0], dtype=np.float32)
    nonzero = ratios > 0
    exponents = np.ceil(np.log2(ratios[nonzero])).astype(np.int64)
    if np.any(exponents < -149) or np.any(exponents > 127):
        raise Mxfp4ConversionError(
            f"{weight_name} requires a power-of-two FP32 scale outside the finite range"
        )
    if np.any(nonzero):
        scales[nonzero] = np.ldexp(
            np.ones(np.count_nonzero(nonzero), dtype=np.float32), exponents.astype(np.int32)
        )
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise Mxfp4ConversionError(f"{weight_name} produced invalid FP32 power-of-two scales")
    return scales


@dataclass(frozen=True)
class Mxfp4ConversionReport:
    """Exactness and memory accounting for one converted tensor."""

    element_count: int
    exact_match_count: int
    underflow_count: int
    max_abs_error: float
    rel_l2: float
    row_chunk_size: int
    source_weight_bytes: int
    source_scale_bytes: int
    resident_weight_bytes: int
    resident_scale_bytes: int
    source_weight_chunk_bytes: int
    source_scale_chunk_bytes: int
    quantized_fp8_chunk_bytes: int
    decoded_fp32_chunk_bytes: int
    fp32_working_bytes: int
    fp64_metric_working_bytes: int
    calculated_peak_host_bytes: int
    measured_peak_python_bytes: int | None = None

    @property
    def exact(self) -> bool:
        return self.exact_match_count == self.element_count and self.underflow_count == 0

    def as_dict(self) -> dict[str, int | float | bool | None]:
        return {
            "element_count": self.element_count,
            "exact_match_count": self.exact_match_count,
            "exact_match_fraction": self.exact_match_count / self.element_count,
            "underflow_count": self.underflow_count,
            "max_abs_error": self.max_abs_error,
            "rel_l2": self.rel_l2,
            "exact": self.exact,
            "row_chunk_size": self.row_chunk_size,
            "source_weight_bytes": self.source_weight_bytes,
            "source_scale_bytes": self.source_scale_bytes,
            "resident_weight_bytes": self.resident_weight_bytes,
            "resident_scale_bytes": self.resident_scale_bytes,
            "resident_output_bytes": self.resident_weight_bytes + self.resident_scale_bytes,
            "source_weight_chunk_bytes": self.source_weight_chunk_bytes,
            "source_scale_chunk_bytes": self.source_scale_chunk_bytes,
            "source_chunk_bytes": self.source_weight_chunk_bytes + self.source_scale_chunk_bytes,
            "quantized_fp8_chunk_bytes": self.quantized_fp8_chunk_bytes,
            "decoded_fp32_chunk_bytes": self.decoded_fp32_chunk_bytes,
            "fp32_working_bytes": self.fp32_working_bytes,
            "fp64_metric_working_bytes": self.fp64_metric_working_bytes,
            "calculated_peak_host_bytes": self.calculated_peak_host_bytes,
            "measured_peak_python_bytes": self.measured_peak_python_bytes,
        }


@dataclass(frozen=True)
class ResidentFp8Weight:
    """Resident E4M3FN bytes plus a per-output-channel FP32 scale."""

    weight_fp8: np.ndarray
    scale_fp32: np.ndarray
    weight_name: str
    scale_name: str
    report: Mxfp4ConversionReport

    def __post_init__(self) -> None:
        if self.weight_fp8.dtype != np.dtype(FP8_DTYPE) or self.weight_fp8.ndim != 2:
            raise Mxfp4ConversionError(
                f"resident weight must be rank-2 E4M3FN, got {self.weight_fp8.dtype}/"
                f"{self.weight_fp8.shape}"
            )
        if self.scale_fp32.dtype != np.float32 or self.scale_fp32.shape != (
            self.weight_fp8.shape[0],
        ):
            raise Mxfp4ConversionError(
                f"resident scale must be FP32 [N], got {self.scale_fp32.dtype}/"
                f"{self.scale_fp32.shape}"
            )

    @property
    def weight_u8(self) -> np.ndarray:
        """Return the exact E4M3FN storage bytes for JAX/safetensors handoff."""

        return np.ascontiguousarray(self.weight_fp8.view(np.uint8))

    @property
    def logical_shape(self) -> tuple[int, int]:
        return tuple(self.weight_fp8.shape)

    @property
    def storage_bytes(self) -> int:
        return int(self.weight_u8.nbytes + self.scale_fp32.nbytes)

    def dequantize(self) -> np.ndarray:
        return self.weight_fp8.astype(np.float32) * self.scale_fp32[:, None]

    def as_moe_kernel_layout(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``([1,K,N] FP8, [1,1,1,N] FP32)`` for the fused MoE kernel."""

        weight = np.ascontiguousarray(self.weight_fp8.T[None, :, :])
        scale = np.ascontiguousarray(self.scale_fp32[None, None, None, :])
        return weight, scale

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": M0_2_SCHEMA,
            "weight_name": self.weight_name,
            "scale_name": self.scale_name,
            "source_weight_dtype": "I8",
            "source_scale_dtype": "F8_E8M0",
            "resident_weight_dtype": "F8_E4M3FN",
            "resident_scale_dtype": "F32",
            "source_layout": "[N,K/2] + [N,K/32]",
            "kernel_layout": "[E,K,N] + [E,1,1,N]",
            "logical_shape": list(self.logical_shape),
            "scale_shape": list(self.scale_fp32.shape),
            "report": self.report.as_dict(),
        }


RowReader = Callable[[slice], np.ndarray]


@dataclass(frozen=True)
class _SafetensorsTensor:
    """Header entry used by the raw one-byte reader below."""

    path: Path
    name: str
    dtype: str
    shape: tuple[int, ...]
    byte_offset: int
    byte_size: int


def _read_safetensors_header(path: Path) -> dict[str, _SafetensorsTensor]:
    """Read only a safetensors header and byte offsets.

    ``safetensors~=0.6.1`` cannot expose ``F8_E8M0`` through its NumPy
    framework on the supported NumPy versions (it asks NumPy for a dtype that
    NumPy does not provide).  The file format is still unambiguous: both
    ``I8`` and ``F8_E8M0`` are one-byte row-major storage.  Reading validated
    byte ranges keeps the real loader usable without reinterpreting the
    exponent bytes as an unsupported NumPy dtype.
    """

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Mxfp4ConversionError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        file_size = path.stat().st_size
        if file_size < 8:
            raise Mxfp4ConversionError(f"truncated safetensors file {path}")
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise Mxfp4ConversionError(f"truncated safetensors header prefix in {path}")
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size > MAX_SAFETENSORS_HEADER_BYTES:
                raise Mxfp4ConversionError(
                    f"safetensors header in {path} is too large: {header_size} bytes"
                )
            if header_size > file_size - 8:
                raise Mxfp4ConversionError(
                    f"safetensors header in {path} exceeds file size: "
                    f"{header_size} > {file_size - 8}"
                )
            raw_header = stream.read(header_size)
            if len(raw_header) != header_size:
                raise Mxfp4ConversionError(f"truncated safetensors header in {path}")
        header = json.loads(raw_header, object_pairs_hook=reject_duplicate_keys)
    except (OSError, json.JSONDecodeError, struct.error) as exc:
        raise Mxfp4ConversionError(f"cannot read safetensors header {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise Mxfp4ConversionError(f"safetensors header in {path} is not an object")

    data_section_offset = 8 + header_size
    result: dict[str, _SafetensorsTensor] = {}
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(metadata, dict):
            raise Mxfp4ConversionError(f"invalid metadata for tensor {name!r} in {path}")
        dtype = _dtype_name(metadata.get("dtype"))
        shape = tuple(int(dim) for dim in metadata.get("shape", ()))
        offsets = metadata.get("data_offsets")
        data_bytes = file_size - data_section_offset
        if (
            not isinstance(offsets, (list, tuple))
            or len(offsets) != 2
            or any(int(offset) < 0 for offset in offsets)
        ):
            raise Mxfp4ConversionError(f"invalid data_offsets for tensor {name!r} in {path}")
        start, end = (int(offsets[0]), int(offsets[1]))
        if end < start or end > data_bytes:
            raise Mxfp4ConversionError(
                f"data_offsets for tensor {name!r} leave the data section in {path}: "
                f"[{start}, {end}) with {data_bytes} data bytes"
            )
        if name in result:
            raise Mxfp4ConversionError(f"duplicate tensor name {name!r} in {path}")
        result[name] = _SafetensorsTensor(
            path=path,
            name=name,
            dtype=dtype,
            shape=shape,
            byte_offset=data_section_offset + start,
            byte_size=end - start,
        )
    return result


def _read_one_byte_rows(
    stream: Any,
    tensor: _SafetensorsTensor,
    rows: slice,
) -> np.ndarray:
    if len(tensor.shape) != 2:
        raise Mxfp4ConversionError(f"{tensor.name} must be rank-2, got {tensor.shape}")
    start, stop, step = rows.indices(tensor.shape[0])
    if step != 1:
        raise Mxfp4ConversionError("safetensors row reads require a unit step")
    byte_count = (stop - start) * tensor.shape[1]
    byte_offset = tensor.byte_offset + start * tensor.shape[1]
    stream.seek(byte_offset)
    raw = stream.read(byte_count)
    if len(raw) != byte_count:
        raise Mxfp4ConversionError(
            f"short read for {tensor.name}: got {len(raw)} bytes, expected {byte_count}"
        )
    return np.frombuffer(raw, dtype=np.uint8).reshape(stop - start, tensor.shape[1])


def convert_mxfp4_pair_from_reader(
    *,
    weight_name: str,
    scale_name: str,
    weight_shape: Sequence[int],
    scale_shape: Sequence[int],
    weight_dtype: str,
    scale_dtype: str,
    read_weight_rows: RowReader,
    read_scale_rows: RowReader,
    row_chunk_size: int = 128,
    strict: bool = True,
) -> ResidentFp8Weight:
    """Convert one metadata-validated pair using bounded row reads.

    ``read_*_rows(slice(start, stop))`` must return the requested rank-2 rows.
    A strict conversion raises ``Mxfp4ConversionError`` on any FP8 rounding or
    nonzero-to-zero underflow; callers must opt into ``strict=False`` to inspect
    a lossy tensor rather than silently accepting it.
    """

    _validate_names(weight_name, scale_name)
    if _dtype_name(weight_dtype) != "I8" or _dtype_name(scale_dtype) != "F8_E8M0":
        raise Mxfp4ConversionError(
            f"{weight_name}/{scale_name} requires I8/F8_E8M0 metadata, got "
            f"{weight_dtype}/{scale_dtype}"
        )
    rows, columns = _validate_shapes(
        weight_shape,
        scale_shape,
        weight_name=weight_name,
        scale_name=scale_name,
    )
    if row_chunk_size <= 0:
        raise Mxfp4ConversionError(f"row_chunk_size must be positive, got {row_chunk_size}")
    row_chunk_size = min(int(row_chunk_size), rows)

    packed_columns = columns // 2
    output_bytes = np.empty((rows, columns), dtype=np.uint8)
    output_scale = np.empty(rows, dtype=np.float32)
    exact_count = 0
    underflow_count = 0
    max_abs_error = 0.0
    squared_error = 0.0
    squared_reference = 0.0

    for start in range(0, rows, row_chunk_size):
        stop = min(start + row_chunk_size, rows)
        row_slice = slice(start, stop)
        packed_chunk = _as_u8_bytes(read_weight_rows(row_slice), name=weight_name)
        scale_chunk = _as_u8_bytes(read_scale_rows(row_slice), name=scale_name)
        if packed_chunk.shape != (stop - start, packed_columns):
            raise Mxfp4ConversionError(
                f"reader returned {weight_name} shape={packed_chunk.shape}, expected "
                f"{(stop - start, packed_columns)}"
            )
        if scale_chunk.shape != (stop - start, columns // 32):
            raise Mxfp4ConversionError(
                f"reader returned {scale_name} shape={scale_chunk.shape}, expected "
                f"{(stop - start, columns // 32)}"
            )

        decoded = _decode_chunk(packed_chunk, scale_chunk, scale_name=scale_name)
        scales = _power2_scales(decoded, weight_name=weight_name)
        normalized = decoded / scales[:, None]
        quantized = np.clip(normalized, -FP8_MAX, FP8_MAX).astype(FP8_DTYPE)
        reconstructed = quantized.astype(np.float32) * scales[:, None]
        difference = reconstructed - decoded
        exact_count += int(np.count_nonzero(reconstructed == decoded))
        underflow_count += int(np.count_nonzero((decoded != 0) & (reconstructed == 0)))
        max_abs_error = max(max_abs_error, float(np.max(np.abs(difference))))
        squared_error += float(np.sum(np.square(difference, dtype=np.float64)))
        squared_reference += float(np.sum(np.square(decoded, dtype=np.float64)))
        output_bytes[row_slice] = quantized.view(np.uint8)
        output_scale[row_slice] = scales

    element_count = rows * columns
    report = Mxfp4ConversionReport(
        element_count=element_count,
        exact_match_count=exact_count,
        underflow_count=underflow_count,
        max_abs_error=max_abs_error,
        rel_l2=math.sqrt(squared_error)
        / (math.sqrt(squared_reference) if squared_reference else 1.0),
        row_chunk_size=row_chunk_size,
        source_weight_bytes=rows * packed_columns,
        source_scale_bytes=rows * (columns // 32),
        resident_weight_bytes=rows * columns,
        resident_scale_bytes=rows * 4,
        source_weight_chunk_bytes=row_chunk_size * packed_columns,
        source_scale_chunk_bytes=row_chunk_size * (columns // 32),
        quantized_fp8_chunk_bytes=row_chunk_size * columns,
        decoded_fp32_chunk_bytes=row_chunk_size * columns * 4,
        fp32_working_bytes=row_chunk_size * columns * 4 * 4,
        fp64_metric_working_bytes=row_chunk_size * columns * 8 * 2,
        # This is an allocation accounting calculation, not a process RSS or
        # HBM measurement. NumPy keeps the decoded, normalized, reconstructed,
        # difference, and metric work arrays live at overlapping points.
        calculated_peak_host_bytes=(
            rows * columns
            + rows * 4
            + row_chunk_size * packed_columns
            + row_chunk_size * (columns // 32)
            + row_chunk_size * columns
            + row_chunk_size * columns * 4 * 4
            + row_chunk_size * columns * 8 * 2
        ),
    )
    if strict and not report.exact:
        raise Mxfp4ConversionError(
            f"lossy MXFP4->FP8 conversion for {weight_name}: "
            f"exact={report.exact_match_count}/{report.element_count}, "
            f"underflow={report.underflow_count}, max_abs_error={report.max_abs_error}"
        )

    return ResidentFp8Weight(
        weight_fp8=output_bytes.view(FP8_DTYPE),
        scale_fp32=output_scale,
        weight_name=weight_name,
        scale_name=scale_name,
        report=report,
    )


def convert_mxfp4_pair(
    packed: np.ndarray,
    scale_e8m0: np.ndarray,
    *,
    weight_name: str,
    scale_name: str,
    row_chunk_size: int = 128,
    strict: bool = True,
) -> ResidentFp8Weight:
    """Convert already-read arrays through the same bounded conversion path."""

    packed_u8 = _as_u8_bytes(packed, name=weight_name)
    scale_u8 = _as_u8_bytes(scale_e8m0, name=scale_name)
    return convert_mxfp4_pair_from_reader(
        weight_name=weight_name,
        scale_name=scale_name,
        weight_shape=packed_u8.shape,
        scale_shape=scale_u8.shape,
        weight_dtype="I8",
        scale_dtype="F8_E8M0",
        read_weight_rows=lambda rows: packed_u8[rows],
        read_scale_rows=lambda rows: scale_u8[rows],
        row_chunk_size=row_chunk_size,
        strict=strict,
    )


def convert_mxfp4_pair_from_safetensors(
    weight_file: str | Path,
    weight_name: str,
    scale_name: str,
    *,
    scale_file: str | Path | None = None,
    row_chunk_size: int = 128,
    strict: bool = True,
) -> ResidentFp8Weight:
    """Read one real safetensors pair and convert it without writing output."""

    _validate_names(weight_name, scale_name)
    weight_path = Path(weight_file)
    scale_path = Path(scale_file if scale_file is not None else weight_file)
    weight_entries = _read_safetensors_header(weight_path)
    scale_entries = (
        weight_entries if scale_path == weight_path else _read_safetensors_header(scale_path)
    )
    try:
        weight_entry = weight_entries[weight_name]
        scale_entry = scale_entries[scale_name]
    except KeyError as exc:
        raise Mxfp4ConversionError(
            f"missing safetensors tensor {exc.args[0]!r} in "
            f"{weight_path if weight_path == scale_path else (weight_path, scale_path)}"
        ) from exc
    if weight_entry.dtype != "I8" or scale_entry.dtype != "F8_E8M0":
        raise Mxfp4ConversionError(
            f"{weight_name}/{scale_name} requires I8/F8_E8M0 metadata, got "
            f"{weight_entry.dtype}/{scale_entry.dtype}"
        )
    expected_weight_bytes = int(np.prod(weight_entry.shape, dtype=np.int64))
    expected_scale_bytes = int(np.prod(scale_entry.shape, dtype=np.int64))
    if (
        weight_entry.byte_size != expected_weight_bytes
        or scale_entry.byte_size != expected_scale_bytes
    ):
        raise Mxfp4ConversionError(
            f"one-byte metadata mismatch for {weight_name}/{scale_name}: "
            f"data bytes={weight_entry.byte_size}/{scale_entry.byte_size}, "
            f"shape products={expected_weight_bytes}/{expected_scale_bytes}"
        )

    with contextlib.ExitStack() as stack:
        streams: dict[Path, Any] = {}
        for path in (weight_path, scale_path):
            if path not in streams:
                try:
                    streams[path] = stack.enter_context(path.open("rb"))
                except OSError as exc:
                    raise Mxfp4ConversionError(
                        f"cannot open safetensors file {path}: {exc}"
                    ) from exc

        def read_weight_rows(rows: slice) -> np.ndarray:
            return _read_one_byte_rows(streams[weight_path], weight_entry, rows)

        def read_scale_rows(rows: slice) -> np.ndarray:
            return _read_one_byte_rows(streams[scale_path], scale_entry, rows)

        return convert_mxfp4_pair_from_reader(
            weight_name=weight_name,
            scale_name=scale_name,
            weight_shape=weight_entry.shape,
            scale_shape=scale_entry.shape,
            weight_dtype=weight_entry.dtype,
            scale_dtype=scale_entry.dtype,
            read_weight_rows=read_weight_rows,
            read_scale_rows=read_scale_rows,
            row_chunk_size=row_chunk_size,
            strict=strict,
        )


def _checkpoint_locations(
    checkpoint: Path,
) -> tuple[dict[str, tuple[Path, tuple[int, ...], str]], dict[str, Any]]:
    locations: dict[str, tuple[Path, tuple[int, ...], str]] = {}
    files = sorted(checkpoint.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no *.safetensors files found in {checkpoint}")
    for path in files:
        for name, tensor in _read_safetensors_header(path).items():
            if name in locations:
                raise Mxfp4ConversionError(
                    f"duplicate tensor name {name!r} in "
                    f"{locations[name][0].name} and {path.name}"
                )
            locations[name] = (path, tensor.shape, tensor.dtype)

    index_candidates = [checkpoint / "model.safetensors.index.json"]
    index_candidates.extend(sorted(checkpoint.glob("*.safetensors.index.json")))
    index_path = next((path for path in index_candidates if path.is_file()), None)
    if index_path is None:
        return locations, {
            "present": False,
            "valid": False,
            "coverage_complete": False,
            "mode": "glob_without_index",
            "path": None,
            "note": (
                "no safetensors index was found; globbed headers are a bounded sample/file "
                "scan and cannot establish full-checkpoint coverage"
            ),
        }

    try:
        index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Mxfp4ConversionError(f"cannot read safetensors index {index_path}: {exc}") from exc
    weight_map = index_payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise Mxfp4ConversionError(f"safetensors index {index_path} has no non-empty weight_map")

    checkpoint_files = {path.relative_to(checkpoint).as_posix() for path in files}
    indexed_files = {str(filename) for filename in weight_map.values()}
    missing_files = sorted(indexed_files - checkpoint_files)
    if missing_files:
        raise Mxfp4ConversionError(
            f"safetensors index references missing file(s): {missing_files[:8]}"
        )
    missing_headers = sorted(set(weight_map) - set(locations))
    if missing_headers:
        raise Mxfp4ConversionError(
            f"safetensors index references tensor(s) absent from headers: {missing_headers[:8]}"
        )
    wrong_files = sorted(
        name
        for name, filename in weight_map.items()
        if locations[name][0].relative_to(checkpoint).as_posix() != filename
    )
    if wrong_files:
        raise Mxfp4ConversionError(
            f"safetensors index/header file mismatch for tensor(s): {wrong_files[:8]}"
        )
    extra_headers = sorted(set(locations) - set(weight_map))
    if extra_headers:
        raise Mxfp4ConversionError(
            f"safetensors headers contain tensor(s) absent from index: {extra_headers[:8]}"
        )
    return locations, {
        "present": True,
        "valid": True,
        "coverage_complete": True,
        "mode": "indexed",
        "path": str(index_path),
        "weight_map_tensor_count": len(weight_map),
        "header_tensor_count": len(locations),
        "referenced_file_count": len(set(weight_map.values())),
    }


def inspect_mxfp4_checkpoint(
    checkpoint: str | Path,
    *,
    row_chunk_size: int = 128,
    strict: bool = True,
    measure_python_allocations: bool = False,
) -> dict[str, Any]:
    """Convert every discoverable MXFP4 pair and return a report only.

    No FP8 artifact is written.  ``measured_peak_python_bytes`` uses
    ``tracemalloc`` when requested and measures Python allocations only; it is
    intentionally reported separately from the calculated tensor-memory account.
    """

    checkpoint_path = Path(checkpoint)
    locations, coverage = _checkpoint_locations(checkpoint_path)
    pairs: list[tuple[str, str]] = []
    unpaired: list[str] = []
    for weight_name, (_, _, weight_dtype) in sorted(locations.items()):
        if not weight_name.endswith(".weight") or weight_dtype != "I8":
            continue
        scale_name = f"{weight_name[:-len('.weight')]}.scale"
        scale_location = locations.get(scale_name)
        if scale_location is None or scale_location[2] != "F8_E8M0":
            unpaired.append(weight_name)
            continue
        pairs.append((weight_name, scale_name))
    if strict and unpaired:
        raise Mxfp4ConversionError(
            f"found I8 weights without paired F8_E8M0 .scale tensors: {unpaired[:8]}"
        )
    if not pairs:
        raise Mxfp4ConversionError(
            f"no I8/F8_E8M0 MXFP4 pairs found in checkpoint {checkpoint_path}"
        )

    import tracemalloc

    records: list[dict[str, Any]] = []
    measured_peak = 0
    for weight_name, scale_name in pairs:
        weight_file = locations[weight_name][0]
        scale_file = locations[scale_name][0]
        if measure_python_allocations:
            tracemalloc.start()
        converted = None
        try:
            converted = convert_mxfp4_pair_from_safetensors(
                weight_file,
                weight_name,
                scale_name,
                scale_file=scale_file,
                row_chunk_size=row_chunk_size,
                strict=strict,
            )
            record = converted.metadata()
            if measure_python_allocations:
                _, peak = tracemalloc.get_traced_memory()
                measured_peak = max(measured_peak, peak)
                record["report"]["measured_peak_python_bytes"] = int(peak)
            records.append(record)
        finally:
            # Do not retain the previous tensor's resident FP8 output while
            # opening/converting the next tensor. The report keeps metadata
            # only; full FP8 output is deliberately not saved by this tool.
            if converted is not None:
                del converted
            if measure_python_allocations:
                tracemalloc.stop()

    calculated_peak = max(
        (int(record["report"]["calculated_peak_host_bytes"]) for record in records),
        default=0,
    )
    resident_bytes = sum(
        int(record["report"]["resident_weight_bytes"])
        + int(record["report"]["resident_scale_bytes"])
        for record in records
    )
    return {
        "schema": M0_2_SCHEMA,
        "checkpoint": str(checkpoint_path),
        "network_access": False,
        "row_chunk_size": row_chunk_size,
        "strict": strict,
        "pair_count": len(records),
        "unpaired_i8_weights": unpaired,
        "checkpoint_coverage": coverage,
        "all_converted_tensors_exact": all(record["report"]["exact"] for record in records),
        "all_exact": (
            bool(coverage["coverage_complete"])
            and not unpaired
            and all(record["report"]["exact"] for record in records)
        ),
        "resident_bytes_calculated": resident_bytes,
        "peak_host_bytes_calculated": calculated_peak,
        "peak_python_bytes_measured": measured_peak if measure_python_allocations else None,
        "measurement_note": (
            "calculated account includes resident output and overlapping raw/FP32/FP8 "
            "row-chunk arrays; "
            "optional measured value is tracemalloc Python allocation only, not RSS/HBM"
        ),
        "tensors": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--row-chunk-size", type=int, default=128)
    parser.add_argument("--allow-lossy", action="store_true")
    parser.add_argument("--measure-python-allocations", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = inspect_mxfp4_checkpoint(
        args.checkpoint,
        row_chunk_size=args.row_chunk_size,
        strict=not args.allow_lossy,
        measure_python_allocations=args.measure_python_allocations,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
