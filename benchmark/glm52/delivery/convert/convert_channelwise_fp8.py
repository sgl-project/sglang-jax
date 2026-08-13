#!/usr/bin/env python3
"""Convert a GLM-5.2 BF16 checkpoint to per-output-channel FP8.

The converter intentionally does not materialize a full shard in memory. It
parses safetensors headers itself, memory-maps one input shard, and quantizes a
bounded number of rows at a time. Converted shards retain their original file
names; every converted ``*.weight`` tensor gets a colocated
``*.weight_scale_inv`` FP32 tensor in the same shard.

Multiple processes may cooperate through ``--rank``/``--world``. A shared
staging directory is resumable at shard granularity. Publication to ``--final``
does not begin until every staging shard passes validation, and
``_DOWNLOAD_COMPLETE`` is written last.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import os
import re
import struct
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

import ml_dtypes
import numpy as np

SCHEMA = "glm52-fp8-e4m3fn-output-channel-v1"
FP8_DTYPE = "F8_E4M3"
FP8_MAX = np.float32(448.0)
HEADER_LIMIT_BYTES = 256 * 1024 * 1024
COPY_CHUNK_BYTES = 32 * 1024 * 1024
CONTROL_DIR_NAME = ".glm52-channelwise-conversion"

# Safetensors dtype names and their storage widths. Quantized tensors are
# restricted further below, but passthrough tensors may use any of these.
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}
QUANTIZABLE_NUMPY_DTYPES = {
    "BF16": np.dtype(ml_dtypes.bfloat16),
    "F16": np.dtype(np.float16),
    "F32": np.dtype(np.float32),
}

ATTENTION_WEIGHT_RE = re.compile(
    r"(?:^|\.)self_attn\."
    r"(?:q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)\.weight$"
)
INDEXER_WEIGHT_RE = re.compile(r"(?:^|\.)indexer\.(?:wq_b|wk)\.weight$")
MLP_WEIGHT_RE = re.compile(
    r"(?:^|\.)mlp\."
    r"(?:(?:experts\.\d+|shared_experts)\.)?"
    r"(?:gate_proj|up_proj|down_proj)\.weight$"
)


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class SafetensorsHeader:
    header_bytes: bytes
    data_start: int
    file_bytes: int
    metadata: dict[str, str] | None
    tensors: tuple[TensorInfo, ...]

    @property
    def tensor_map(self) -> dict[str, TensorInfo]:
        return {tensor.name: tensor for tensor in self.tensors}


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _product(shape: Iterable[int]) -> int:
    return math.prod(shape)


def _scale_name(weight_name: str) -> str:
    if not weight_name.endswith(".weight"):
        raise ValueError(f"not a weight key: {weight_name}")
    return f"{weight_name[: -len('.weight')]}.weight_scale_inv"


def should_quantize(name: str, dtype: str, shape: tuple[int, ...]) -> bool:
    """Return whether a source tensor belongs to the GLM-5.2 FP8 policy."""
    if dtype not in QUANTIZABLE_NUMPY_DTYPES or len(shape) != 2:
        return False
    return bool(
        ATTENTION_WEIGHT_RE.search(name)
        or INDEXER_WEIGHT_RE.search(name)
        or MLP_WEIGHT_RE.search(name)
    )


def _read_safetensors_header(path: Path) -> SafetensorsHeader:
    file_bytes = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated safetensors prefix: {path}")
        (header_len,) = struct.unpack("<Q", prefix)
        if header_len <= 0 or header_len > HEADER_LIMIT_BYTES:
            raise ValueError(f"invalid safetensors header length {header_len}: {path}")
        header_bytes = handle.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"truncated safetensors header: {path}")

    try:
        raw = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safetensors JSON header: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"safetensors header must be an object: {path}")

    data_start = 8 + header_len
    data_bytes = file_bytes - data_start
    metadata = raw.pop("__metadata__", None)
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError(f"invalid __metadata__ in {path}")

    tensors = []
    occupied = []
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"invalid tensor entry {name!r} in {path}")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if dtype not in DTYPE_BYTES:
            raise ValueError(f"unsupported dtype {dtype!r} for {name!r} in {path}")
        if not isinstance(shape, list) or not all(
            isinstance(dim, int) and dim >= 0 for dim in shape
        ):
            raise ValueError(f"invalid shape for {name!r} in {path}: {shape!r}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(offset, int) for offset in offsets)
        ):
            raise ValueError(f"invalid data_offsets for {name!r} in {path}")
        start, end = offsets
        expected = _product(shape) * DTYPE_BYTES[dtype]
        if start < 0 or end < start or end > data_bytes or end - start != expected:
            raise ValueError(
                f"invalid byte range for {name!r} in {path}: "
                f"offsets={offsets}, expected_bytes={expected}, data_bytes={data_bytes}"
            )
        tensors.append(TensorInfo(name, dtype, tuple(shape), start, end))
        occupied.append((start, end, name))

    occupied.sort()
    cursor = 0
    for start, end, name in occupied:
        if start != cursor:
            raise ValueError(
                f"non-contiguous or overlapping tensor data before {name!r} in {path}: "
                f"expected_start={cursor}, actual_start={start}"
            )
        cursor = end
    if cursor != data_bytes:
        raise ValueError(
            f"safetensors payload size mismatch in {path}: tensors={cursor}, data={data_bytes}"
        )

    tensors.sort(key=lambda tensor: tensor.start)
    return SafetensorsHeader(
        header_bytes=header_bytes,
        data_start=data_start,
        file_bytes=file_bytes,
        metadata=metadata,
        tensors=tuple(tensors),
    )


def _output_specs(
    source_header: SafetensorsHeader,
) -> tuple[list[tuple[str, str, tuple[int, ...]]], int]:
    specs = []
    selected = 0
    source_names = {tensor.name for tensor in source_header.tensors}
    for tensor in source_header.tensors:
        if should_quantize(tensor.name, tensor.dtype, tensor.shape):
            scale_name = _scale_name(tensor.name)
            if scale_name in source_names:
                raise ValueError(
                    f"source checkpoint already contains generated scale key {scale_name!r}"
                )
            specs.append((tensor.name, FP8_DTYPE, tensor.shape))
            specs.append((scale_name, "F32", (tensor.shape[0],)))
            selected += 1
        else:
            specs.append((tensor.name, tensor.dtype, tensor.shape))
    return specs, selected


def _encode_header(
    specs: list[tuple[str, str, tuple[int, ...]]],
    metadata: dict[str, str] | None,
) -> tuple[bytes, dict[str, TensorInfo]]:
    raw: dict[str, object] = {}
    cursor = 0
    output_map = {}
    for name, dtype, shape in specs:
        nbytes = _product(shape) * DTYPE_BYTES[dtype]
        raw[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [cursor, cursor + nbytes],
        }
        output_map[name] = TensorInfo(name, dtype, shape, cursor, cursor + nbytes)
        cursor += nbytes
    if metadata is not None:
        raw["__metadata__"] = metadata
    encoded = json.dumps(raw, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    padding = (-len(encoded)) % 8
    return encoded + b" " * padding, output_map


def _copy_range(
    source_map: mmap.mmap,
    start: int,
    end: int,
    output: BinaryIO,
) -> None:
    cursor = start
    while cursor < end:
        next_cursor = min(cursor + COPY_CHUNK_BYTES, end)
        output.write(source_map[cursor:next_cursor])
        cursor = next_cursor


def _quantize_tensor(
    source_map: mmap.mmap,
    absolute_start: int,
    tensor: TensorInfo,
    output: BinaryIO,
    chunk_elements: int,
) -> None:
    rows, columns = tensor.shape
    dtype = QUANTIZABLE_NUMPY_DTYPES[tensor.dtype]
    rows_per_chunk = max(1, chunk_elements // max(columns, 1))
    scales = np.empty(rows, dtype=np.float32)
    row_bytes = columns * dtype.itemsize

    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        chunk_rows = row_end - row_start
        offset = absolute_start + row_start * row_bytes
        source_view = np.frombuffer(
            source_map,
            dtype=dtype,
            count=chunk_rows * columns,
            offset=offset,
        ).reshape(chunk_rows, columns)
        chunk = source_view.astype(np.float32)
        scale = np.max(np.abs(chunk), axis=1).astype(np.float32) / FP8_MAX
        scale_safe = np.where(scale == 0, np.float32(1.0), scale)
        quantized = np.clip(
            chunk / scale_safe[:, None],
            -FP8_MAX,
            FP8_MAX,
        ).astype(ml_dtypes.float8_e4m3fn)
        output.write(quantized.tobytes(order="C"))
        scales[row_start:row_end] = scale
        del quantized, scale_safe, scale, chunk, source_view

    output.write(scales.tobytes(order="C"))


def convert_shard(
    source_path: Path,
    output_path: Path,
    chunk_elements: int,
) -> dict[str, object]:
    source_header = _read_safetensors_header(source_path)
    specs, selected = _output_specs(source_header)
    encoded_header, _ = _encode_header(specs, source_header.metadata)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("rb") as source, output_path.open("wb") as output:
        source_map = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            output.write(struct.pack("<Q", len(encoded_header)))
            output.write(encoded_header)
            for tensor in source_header.tensors:
                absolute_start = source_header.data_start + tensor.start
                if should_quantize(tensor.name, tensor.dtype, tensor.shape):
                    _quantize_tensor(
                        source_map,
                        absolute_start,
                        tensor,
                        output,
                        chunk_elements,
                    )
                else:
                    _copy_range(
                        source_map,
                        absolute_start,
                        source_header.data_start + tensor.end,
                        output,
                    )
            output.flush()
            os.fsync(output.fileno())
        finally:
            source_map.close()

    validation = validate_converted_shard(source_path, output_path)
    validation.update(
        {
            "selected_tensors": selected,
            "source_file_bytes": source_header.file_bytes,
            "source_header_sha256": hashlib.sha256(
                source_header.header_bytes
            ).hexdigest(),
        }
    )
    return validation


def validate_converted_shard(source_path: Path, output_path: Path) -> dict[str, object]:
    source_header = _read_safetensors_header(source_path)
    output_header = _read_safetensors_header(output_path)
    specs, selected = _output_specs(source_header)
    expected = {name: (dtype, shape) for name, dtype, shape in specs}
    actual = {
        tensor.name: (tensor.dtype, tensor.shape) for tensor in output_header.tensors
    }
    if actual != expected:
        missing = sorted(set(expected) - set(actual))[:10]
        unexpected = sorted(set(actual) - set(expected))[:10]
        mismatched = sorted(
            name
            for name in set(actual) & set(expected)
            if actual[name] != expected[name]
        )[:10]
        raise ValueError(
            f"converted shard validation failed for {output_path}: "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
        )
    return {
        "file_bytes": output_header.file_bytes,
        "tensor_data_bytes": sum(tensor.nbytes for tensor in output_header.tensors),
        "tensor_count": len(output_header.tensors),
        "selected_tensors": selected,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _copy_file_atomic(
    source: Path,
    destination: Path,
    run_id: str,
    *,
    calculate_sha256: bool = False,
) -> str | None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.part-{run_id}-{os.getpid()}-{uuid.uuid4().hex}"
    )
    digest = hashlib.sha256() if calculate_sha256 else None
    try:
        with source.open("rb") as src, temporary.open("wb") as dst:
            while chunk := src.read(COPY_CHUNK_BYTES):
                dst.write(chunk)
                if digest is not None:
                    digest.update(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return digest.hexdigest() if digest is not None else None


def _load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _wait_for_json(
    path: Path,
    predicate,
    timeout: int,
    description: str,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            if path.is_file():
                payload = _load_json(path)
                if predicate(payload):
                    return payload
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(2)
    suffix = f"; last_error={last_error}" if last_error else ""
    raise TimeoutError(f"timed out waiting for {description}: {path}{suffix}")


def _wait_for_ranks(
    control_dir: Path,
    prefix: str,
    world: int,
    run_id: str,
    timeout: int,
) -> list[dict[str, object]]:
    statuses = []
    for rank in range(world):
        path = control_dir / f"{prefix}-rank-{rank}.json"
        status = _wait_for_json(
            path,
            lambda payload: (
                payload.get("run_id") == run_id and payload.get("status") == "complete"
            ),
            timeout,
            f"{prefix} rank {rank}",
        )
        statuses.append(status)
    return statuses


def _load_source_index(source: Path) -> tuple[dict[str, object], dict[str, str]]:
    index_path = source / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing checkpoint index: {index_path}")
    index = _load_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in weight_map.items()
    ):
        raise ValueError(f"invalid weight_map in {index_path}")
    return index, dict(weight_map)


def _source_shards(source: Path, weight_map: dict[str, str]) -> list[str]:
    shards = sorted(set(weight_map.values()))
    if not shards:
        raise ValueError("source checkpoint index contains no shards")
    missing = [name for name in shards if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"source checkpoint is missing shards: {missing[:10]}")
    return shards


def _validate_expected_count(label: str, actual: int, expected: int | None) -> None:
    if expected is not None and actual != expected:
        raise ValueError(
            f"GLM-5.2 production checkpoint {label} mismatch: "
            f"actual={actual}, expected={expected}; verify the source revision "
            "or explicitly override the delivery wrapper expectation"
        )


def _preflight_source(
    source: Path,
    weight_map: dict[str, str],
    shards: list[str],
    expected_shards: int | None,
    expected_selected_tensors: int | None,
    expected_weight_map_count: int | None,
) -> dict[str, int]:
    indexed_by_shard: dict[str, set[str]] = {shard: set() for shard in shards}
    for name, shard in weight_map.items():
        if shard not in indexed_by_shard:
            raise ValueError(f"source index references unexpected shard {shard!r}")
        indexed_by_shard[shard].add(name)

    selected = 0
    for shard_name in shards:
        header = _read_safetensors_header(source / shard_name)
        actual_names = {tensor.name for tensor in header.tensors}
        if actual_names != indexed_by_shard[shard_name]:
            raise ValueError(f"source index/header key mismatch for {shard_name}")
        _, shard_selected = _output_specs(header)
        selected += shard_selected

    final_weight_map_count = len(weight_map) + selected
    _validate_expected_count("shard_count", len(shards), expected_shards)
    _validate_expected_count("selected_tensors", selected, expected_selected_tensors)
    _validate_expected_count(
        "weight_map_count", final_weight_map_count, expected_weight_map_count
    )
    return {
        "shard_count": len(shards),
        "selected_tensors": selected,
        "weight_map_count": final_weight_map_count,
    }


def _shard_manifest_path(control_dir: Path, shard_name: str) -> Path:
    return control_dir / "shards" / f"{shard_name}.json"


def _staging_shard_is_resumable(
    source_path: Path,
    staging_path: Path,
    manifest_path: Path,
) -> tuple[bool, dict[str, object] | None]:
    if not staging_path.is_file() or not manifest_path.is_file():
        return False, None
    try:
        manifest = _load_json(manifest_path)
        source_header = _read_safetensors_header(source_path)
        if (
            manifest.get("schema") != SCHEMA
            or manifest.get("shard") != source_path.name
            or manifest.get("source_file_bytes") != source_header.file_bytes
            or manifest.get("source_header_sha256")
            != hashlib.sha256(source_header.header_bytes).hexdigest()
            or manifest.get("output_file_bytes") != staging_path.stat().st_size
        ):
            return False, None
        validate_converted_shard(source_path, staging_path)
        if manifest.get("output_sha256") != _sha256_file(staging_path):
            return False, None
        return True, manifest
    except (OSError, ValueError, json.JSONDecodeError):
        return False, None


def _conversion_config(source_config: dict[str, object]) -> dict[str, object]:
    config = dict(source_config)
    config["quantization_config"] = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": None,
    }
    return config


def _prepare_staging_metadata(
    source: Path,
    staging: Path,
    control_dir: Path,
    weight_map: dict[str, str],
    shards: list[str],
    run_id: str,
    expected_shards: int | None,
    expected_selected_tensors: int | None,
    expected_weight_map_count: int | None,
) -> dict[str, object]:
    new_weight_map = dict(weight_map)
    selected = 0
    tensor_data_bytes = 0
    safetensors_file_bytes = 0

    for shard_name in shards:
        source_header = _read_safetensors_header(source / shard_name)
        output_header = _read_safetensors_header(staging / shard_name)
        safetensors_file_bytes += output_header.file_bytes
        tensor_data_bytes += sum(tensor.nbytes for tensor in output_header.tensors)
        for tensor in source_header.tensors:
            if should_quantize(tensor.name, tensor.dtype, tensor.shape):
                if weight_map.get(tensor.name) != shard_name:
                    raise ValueError(
                        f"source index maps {tensor.name!r} to "
                        f"{weight_map.get(tensor.name)!r}, expected {shard_name!r}"
                    )
                new_weight_map[_scale_name(tensor.name)] = shard_name
                selected += 1

    source_index, _ = _load_source_index(source)
    new_index = dict(source_index)
    metadata = dict(new_index.get("metadata") or {})
    metadata["total_size"] = tensor_data_bytes
    new_index["metadata"] = metadata
    new_index["weight_map"] = new_weight_map
    _write_json_atomic(staging / "model.safetensors.index.json", new_index)

    config_path = source / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing source config: {config_path}")
    _write_json_atomic(
        staging / "config.json", _conversion_config(_load_json(config_path))
    )

    shard_set = set(shards)
    excluded = {
        "config.json",
        "model.safetensors.index.json",
        "_DOWNLOAD_COMPLETE",
        "_CONVERSION_IN_PROGRESS",
    }
    for source_file in source.iterdir():
        if (
            not source_file.is_file()
            or source_file.name in shard_set
            or source_file.name in excluded
        ):
            continue
        _copy_file_atomic(source_file, staging / source_file.name, run_id)

    prepared = {
        "kind": "staging_prepared",
        "run_id": run_id,
        "schema": SCHEMA,
        "selected_tensors": selected,
        "shard_count": len(shards),
        "weight_map_count": len(new_weight_map),
        "tensor_data_bytes": tensor_data_bytes,
        "safetensors_file_bytes": safetensors_file_bytes,
        "updated_at": _utc_now(),
    }
    _validate_expected_count("shard_count", len(shards), expected_shards)
    _validate_expected_count("selected_tensors", selected, expected_selected_tensors)
    _validate_expected_count(
        "weight_map_count", len(new_weight_map), expected_weight_map_count
    )
    _write_json_atomic(control_dir / "staging-prepared.json", prepared)
    return prepared


def _validate_final_checkpoint(
    final: Path,
    shards: list[str],
    expected: dict[str, object],
) -> None:
    _, weight_map = _load_source_index(final)
    if len(weight_map) != expected["weight_map_count"]:
        raise ValueError(
            f"final weight_map count mismatch: {len(weight_map)} != "
            f"{expected['weight_map_count']}"
        )
    indexed_by_shard: dict[str, set[str]] = {shard: set() for shard in shards}
    for name, shard in weight_map.items():
        if shard not in indexed_by_shard:
            raise ValueError(f"final index references unexpected shard {shard!r}")
        indexed_by_shard[shard].add(name)

    selected = 0
    tensor_data_bytes = 0
    safetensors_file_bytes = 0
    for shard_name in shards:
        header = _read_safetensors_header(final / shard_name)
        actual_names = {tensor.name for tensor in header.tensors}
        if actual_names != indexed_by_shard[shard_name]:
            raise ValueError(f"final index/header key mismatch for {shard_name}")
        selected += sum(name.endswith(".weight_scale_inv") for name in actual_names)
        tensor_data_bytes += sum(tensor.nbytes for tensor in header.tensors)
        safetensors_file_bytes += header.file_bytes
    if selected != expected["selected_tensors"]:
        raise ValueError(
            f"final scale count mismatch: {selected} != {expected['selected_tensors']}"
        )
    if tensor_data_bytes != expected["tensor_data_bytes"]:
        raise ValueError(
            f"final tensor bytes mismatch: {tensor_data_bytes} != "
            f"{expected['tensor_data_bytes']}"
        )
    if safetensors_file_bytes != expected["safetensors_file_bytes"]:
        raise ValueError(
            f"final safetensors bytes mismatch: {safetensors_file_bytes} != "
            f"{expected['safetensors_file_bytes']}"
        )


def _write_artifact(artifact_dir: Path, name: str, payload: object) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(artifact_dir / name, payload)


def _ensure_safe_paths(source: Path, staging: Path, final: Path) -> None:
    resolved = [path.resolve() for path in (source, staging, final)]
    if len(set(resolved)) != 3:
        raise ValueError(f"source, staging, and final must be distinct: {resolved}")
    for path in resolved:
        if path == Path(path.anchor):
            raise ValueError(f"refusing to use filesystem root: {path}")


def _initialize_run(
    source: Path,
    staging: Path,
    final: Path,
    control_dir: Path,
    world: int,
    run_id: str,
) -> None:
    if final.exists():
        if not final.is_dir():
            raise FileExistsError(f"final path exists and is not a directory: {final}")
        if any(final.iterdir()):
            raise FileExistsError(
                "final directory already exists without a valid "
                f"_DOWNLOAD_COMPLETE marker: {final}; inspect it manually before retrying"
            )
    staging.mkdir(parents=True, exist_ok=True)
    control_dir.mkdir(parents=True, exist_ok=True)
    marker_path = staging / "_CONVERSION_IN_PROGRESS"
    if marker_path.is_file():
        previous = _load_json(marker_path)
        for key, expected in (
            ("schema", SCHEMA),
            ("source", str(source)),
            ("staging", str(staging)),
            ("final", str(final)),
        ):
            if previous.get(key) != expected:
                raise ValueError(
                    f"staging marker mismatch for {key}: "
                    f"{previous.get(key)!r} != {expected!r}"
                )
    for pattern in ("convert-rank-*.json", "publish-rank-*.json"):
        for stale in control_dir.glob(pattern):
            stale.unlink()
    for stale in ("staging-prepared.json", "publish-ready.json"):
        (control_dir / stale).unlink(missing_ok=True)
    marker = {
        "kind": "conversion_in_progress",
        "schema": SCHEMA,
        "source": str(source),
        "staging": str(staging),
        "final": str(final),
        "world": world,
        "run_id": run_id,
        "updated_at": _utc_now(),
    }
    _write_json_atomic(marker_path, marker)


def run(args: argparse.Namespace) -> None:
    source = Path(args.source).resolve()
    staging = Path(args.staging).resolve()
    final = Path(args.final).resolve()
    local_dir = Path(args.local_dir).resolve()
    artifact_dir = Path(args.artifact_dir).resolve()
    _ensure_safe_paths(source, staging, final)
    if not source.is_dir():
        raise FileNotFoundError(f"source checkpoint does not exist: {source}")
    if not 0 <= args.rank < args.world:
        raise ValueError(f"rank {args.rank} is outside [0, {args.world})")
    if args.chunk_elements <= 0:
        raise ValueError("chunk-elements must be positive")

    _, weight_map = _load_source_index(source)
    shards = _source_shards(source, weight_map)
    complete_marker = final / "_DOWNLOAD_COMPLETE"
    if complete_marker.is_file():
        summary = _load_json(complete_marker)
        if summary.get("schema") != SCHEMA or summary.get("status") != "complete":
            raise ValueError(f"invalid completion marker: {complete_marker}")
        for label, expected in (
            ("shard_count", args.expected_shards),
            ("selected_tensors", args.expected_selected_tensors),
            ("weight_map_count", args.expected_weight_map_count),
        ):
            if expected is not None and summary.get(label) != expected:
                raise ValueError(
                    f"existing final checkpoint {label} mismatch: "
                    f"{summary.get(label)} != {expected}"
                )
        _validate_final_checkpoint(final, shards, summary)
        _write_artifact(
            artifact_dir,
            f"no-op-rank-{args.rank}.json",
            {
                "kind": "already_complete",
                "rank": args.rank,
                "schema": SCHEMA,
                "final": str(final),
                "status": "complete",
                "updated_at": _utc_now(),
            },
        )
        print(f"GLM52_CHANNELWISE_CONVERSION_ALREADY_COMPLETE final={final}")
        return

    control_dir = staging / CONTROL_DIR_NAME
    run_id = args.run_id

    if args.rank == 0:
        preflight = _preflight_source(
            source,
            weight_map,
            shards,
            args.expected_shards,
            args.expected_selected_tensors,
            args.expected_weight_map_count,
        )
        print(
            "GLM52_CHANNELWISE_PREFLIGHT_COMPLETE "
            f"shards={preflight['shard_count']} "
            f"selected_tensors={preflight['selected_tensors']} "
            f"weight_map_count={preflight['weight_map_count']}"
        )
        _initialize_run(source, staging, final, control_dir, args.world, run_id)
    else:
        _wait_for_json(
            staging / "_CONVERSION_IN_PROGRESS",
            lambda payload: (
                payload.get("schema") == SCHEMA
                and payload.get("run_id") == run_id
                and payload.get("world") == args.world
            ),
            args.barrier_timeout,
            "rank-0 conversion initialization",
        )

    assigned = shards[args.rank :: args.world]
    converted = 0
    resumed = 0
    selected_tensors = 0
    output_bytes = 0
    local_dir.mkdir(parents=True, exist_ok=True)
    for position, shard_name in enumerate(assigned, start=1):
        source_path = source / shard_name
        staging_path = staging / shard_name
        manifest_path = _shard_manifest_path(control_dir, shard_name)
        resumable, manifest = _staging_shard_is_resumable(
            source_path, staging_path, manifest_path
        )
        if resumable:
            resumed += 1
            selected_tensors += int(manifest["selected_tensors"])
            output_bytes += int(manifest["output_file_bytes"])
            print(
                f"GLM52_CONVERT_RESUME rank={args.rank} shard={shard_name} "
                f"progress={position}/{len(assigned)}"
            )
            continue

        local_path = local_dir / f"{shard_name}.tmp-{run_id}-{args.rank}"
        local_path.unlink(missing_ok=True)
        print(
            f"GLM52_CONVERT_START rank={args.rank} shard={shard_name} "
            f"progress={position}/{len(assigned)}"
        )
        validation = convert_shard(source_path, local_path, args.chunk_elements)
        staging_sha256 = _copy_file_atomic(
            local_path, staging_path, run_id, calculate_sha256=True
        )
        assert staging_sha256 is not None
        manifest = {
            "kind": "converted_shard",
            "schema": SCHEMA,
            "run_id": run_id,
            "rank": args.rank,
            "shard": shard_name,
            "source_file_bytes": validation["source_file_bytes"],
            "source_header_sha256": validation["source_header_sha256"],
            "output_file_bytes": validation["file_bytes"],
            "output_sha256": staging_sha256,
            "selected_tensors": validation["selected_tensors"],
            "tensor_count": validation["tensor_count"],
            "tensor_data_bytes": validation["tensor_data_bytes"],
            "updated_at": _utc_now(),
        }
        _write_json_atomic(manifest_path, manifest)
        local_path.unlink(missing_ok=True)
        converted += 1
        selected_tensors += int(validation["selected_tensors"])
        output_bytes += int(validation["file_bytes"])
        print(
            f"GLM52_CONVERT_COMPLETE rank={args.rank} shard={shard_name} "
            f"bytes={validation['file_bytes']} sha256={staging_sha256}"
        )

    convert_status = {
        "kind": "rank_conversion_complete",
        "schema": SCHEMA,
        "run_id": run_id,
        "rank": args.rank,
        "assigned": len(assigned),
        "converted": converted,
        "resumed": resumed,
        "selected_tensors": selected_tensors,
        "output_bytes": output_bytes,
        "status": "complete",
        "updated_at": _utc_now(),
    }
    _write_json_atomic(control_dir / f"convert-rank-{args.rank}.json", convert_status)
    _write_artifact(artifact_dir, f"convert-rank-{args.rank}.json", convert_status)
    _wait_for_ranks(control_dir, "convert", args.world, run_id, args.barrier_timeout)

    if args.rank == 0:
        prepared = _prepare_staging_metadata(
            source,
            staging,
            control_dir,
            weight_map,
            shards,
            run_id,
            args.expected_shards,
            args.expected_selected_tensors,
            args.expected_weight_map_count,
        )
        if final.exists():
            if not final.is_dir():
                raise FileExistsError(
                    f"final path became a non-directory before publication: {final}"
                )
            if any(final.iterdir()):
                raise FileExistsError(
                    f"final directory became non-empty before publication: {final}"
                )
        final.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(
            final / "_CONVERSION_IN_PROGRESS",
            {
                "kind": "publication_in_progress",
                "schema": SCHEMA,
                "run_id": run_id,
                "source": str(source),
                "staging": str(staging),
                "final": str(final),
                "updated_at": _utc_now(),
            },
        )
        for staging_file in staging.iterdir():
            if not staging_file.is_file() or staging_file.name in set(shards) | {
                "_CONVERSION_IN_PROGRESS"
            }:
                continue
            _copy_file_atomic(staging_file, final / staging_file.name, run_id)
        _write_json_atomic(
            control_dir / "publish-ready.json",
            {
                "kind": "publish_ready",
                "schema": SCHEMA,
                "run_id": run_id,
                "status": "complete",
                "updated_at": _utc_now(),
            },
        )
    else:
        prepared = _wait_for_json(
            control_dir / "staging-prepared.json",
            lambda payload: payload.get("run_id") == run_id,
            args.barrier_timeout,
            "staging metadata preparation",
        )
        _wait_for_json(
            control_dir / "publish-ready.json",
            lambda payload: payload.get("run_id") == run_id,
            args.barrier_timeout,
            "final publication initialization",
        )

    published = 0
    published_bytes = 0
    for shard_name in assigned:
        manifest = _load_json(_shard_manifest_path(control_dir, shard_name))
        source_path = staging / shard_name
        destination = final / shard_name
        final_sha256 = _copy_file_atomic(
            source_path, destination, run_id, calculate_sha256=True
        )
        assert final_sha256 is not None
        if final_sha256 != manifest["output_sha256"]:
            raise ValueError(
                f"final checksum mismatch for {shard_name}: "
                f"{final_sha256} != {manifest['output_sha256']}"
            )
        published += 1
        published_bytes += destination.stat().st_size

    publish_status = {
        "kind": "rank_publish_complete",
        "schema": SCHEMA,
        "run_id": run_id,
        "rank": args.rank,
        "assigned": len(assigned),
        "published": published,
        "resumed": 0,
        "published_bytes": published_bytes,
        "status": "complete",
        "updated_at": _utc_now(),
    }
    _write_json_atomic(control_dir / f"publish-rank-{args.rank}.json", publish_status)
    _write_artifact(artifact_dir, f"publish-rank-{args.rank}.json", publish_status)

    rank_status = _wait_for_ranks(
        control_dir, "publish", args.world, run_id, args.barrier_timeout
    )
    if args.rank == 0:
        _validate_final_checkpoint(final, shards, prepared)
        summary = {
            "kind": "download_complete",
            "schema": SCHEMA,
            "run_id": run_id,
            "source": str(source),
            "staging": str(staging),
            "final": str(final),
            "shard_count": len(shards),
            "selected_tensors": prepared["selected_tensors"],
            "weight_map_count": prepared["weight_map_count"],
            "tensor_data_bytes": prepared["tensor_data_bytes"],
            "safetensors_file_bytes": prepared["safetensors_file_bytes"],
            "rank_status": rank_status,
            "status": "complete",
            "updated_at": _utc_now(),
        }
        (final / "_CONVERSION_IN_PROGRESS").unlink(missing_ok=True)
        _write_json_atomic(complete_marker, summary)
        _write_artifact(artifact_dir, "conversion-summary.json", summary)
        print(
            "GLM52_CHANNELWISE_CONVERSION_COMPLETE "
            f"final={final} shards={len(shards)} "
            f"selected_tensors={prepared['selected_tensors']}"
        )
    else:
        _wait_for_json(
            complete_marker,
            lambda payload: (
                payload.get("schema") == SCHEMA and payload.get("status") == "complete"
            ),
            args.barrier_timeout,
            "final _DOWNLOAD_COMPLETE marker",
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="Complete BF16 checkpoint directory"
    )
    parser.add_argument(
        "--staging", required=True, help="Shared resumable staging directory"
    )
    parser.add_argument(
        "--final", required=True, help="Final channel-wise checkpoint directory"
    )
    parser.add_argument(
        "--rank", type=int, default=0, help="Zero-based local worker rank"
    )
    parser.add_argument("--world", type=int, default=1, help="Number of local workers")
    parser.add_argument("--run-id", default=f"manual-{os.getpid()}")
    parser.add_argument(
        "--local-dir",
        default="/tmp/glm52-fp8-channelwise",
        help="Worker-local scratch directory; only one output shard is retained at a time",
    )
    parser.add_argument(
        "--artifact-dir",
        default="/tmp/glm52-fp8-channelwise-artifact",
        help="Worker-local JSON report directory",
    )
    parser.add_argument(
        "--chunk-elements",
        type=int,
        default=4 * 1024 * 1024,
        help="Maximum source elements converted to float32 per row chunk",
    )
    parser.add_argument(
        "--barrier-timeout",
        type=int,
        default=48 * 60 * 60,
        help="Shared-filesystem worker barrier timeout in seconds",
    )
    parser.add_argument(
        "--expected-shards",
        type=int,
        help="Optional exact source shard count for revision validation",
    )
    parser.add_argument(
        "--expected-selected-tensors",
        type=int,
        help="Optional exact number of weights selected for conversion",
    )
    parser.add_argument(
        "--expected-weight-map-count",
        type=int,
        help="Optional exact final index key count",
    )
    args = parser.parse_args(argv)
    if args.world <= 0:
        parser.error("--world must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_args(argv))
        return 0
    except Exception as exc:
        print(
            f"GLM52_CHANNELWISE_CONVERSION_FAILED type={type(exc).__name__} error={exc}",
            file=sys.stderr,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
