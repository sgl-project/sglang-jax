#!/usr/bin/env python3
"""Validate GLM-5.2 serving-profile runtime coverage from an XPlane file.

This intentionally uses a small protobuf wire reader instead of TensorFlow or
XProf Python packages, so the same gate can run in a Falcon workload, a Falcon
analysis, or on a downloaded profile artifact.
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


MODEL_MODULE_MARKER = "jit_jitted_run_model("
SAMPLER_MODULE_MARKER = "jit_jitted_sampler("
SET_FUTURE_MODULE_MARKER = "jit_set_future_token_ids("

# The C32 extend shape has two 1024-token sequences per DP rank. A full GLM-5.2
# IndexShare group contributes one instruction with each of these three shapes.
EXTEND_INDEXER_KERNELS = (
    "quantized_matmul_kernel_2048_256_1024",
    "quantized_matmul_kernel_512_1024_2048",
    "quantized_matmul_kernel_2048_256_512",
)


@dataclass(frozen=True)
class CoverageExpectation:
    dense_attention: int = 3
    sparse_attention: int = 75
    sparse_moe: int = 75
    full_indexer_groups: int = 19
    require_sampler: bool = True
    require_set_future: bool = True


@dataclass(frozen=True)
class ModuleEvent:
    name: str
    start_ps: int
    duration_ps: int

    @property
    def end_ps(self) -> int:
        return self.start_ps + self.duration_ps


@dataclass(frozen=True)
class ModelCoverage:
    module_name: str
    module_wall_ms: float
    dense_attention: int
    sparse_attention: int
    sparse_moe: int
    full_indexer_groups: int
    indexer_kernel_counts: dict[str, int]


def _varint(buf: memoryview, pos: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7


def _fields(buf: memoryview):
    pos = 0
    size = len(buf)
    while pos < size:
        tag, pos = _varint(buf, pos)
        number = tag >> 3
        wire = tag & 7
        if wire == 0:
            value, pos = _varint(buf, pos)
            yield number, wire, value
        elif wire == 1:
            yield number, wire, buf[pos : pos + 8]
            pos += 8
        elif wire == 2:
            length, pos = _varint(buf, pos)
            value = buf[pos : pos + length]
            pos += length
            yield number, wire, value
        elif wire == 5:
            yield number, wire, buf[pos : pos + 4]
            pos += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")


def _text(buf: memoryview) -> str:
    return bytes(buf).decode("utf-8", errors="replace")


def _metadata_entry(buf: memoryview) -> tuple[int | None, str, str]:
    key = None
    value = None
    for number, wire, item in _fields(buf):
        if number == 1 and wire == 0:
            key = item
        elif number == 2 and wire == 2:
            value = item

    name = ""
    display = ""
    if value is not None:
        for number, wire, item in _fields(value):
            if number == 2 and wire == 2:
                name = _text(item)
            elif number == 4 and wire == 2:
                display = _text(item)
    return key, name, display


def _event_header(buf: memoryview) -> tuple[int, int, int]:
    metadata_id = 0
    offset_ps = 0
    duration_ps = 0
    for number, wire, item in _fields(buf):
        if number == 1 and wire == 0:
            metadata_id = item
        elif number == 2 and wire == 0:
            offset_ps = item
        elif number == 3 and wire == 0:
            duration_ps = item
    return metadata_id, offset_ps, duration_ps


def _instruction_symbol(name: str, display: str) -> str:
    """Return a stable HLO instruction symbol without the operand text."""

    value = display or name
    match = re.match(r"\s*(%?[^\s=]+)", value)
    return match.group(1).lstrip("%") if match else value.strip()


def classify_symbols(symbols: Iterable[str]) -> dict[str, object]:
    """Count semantic GLM-5.2 operations from unique HLO symbols."""

    unique = set(symbols)
    indexer_counts = {
        kernel: sum(symbol.startswith(kernel) for symbol in unique)
        for kernel in EXTEND_INDEXER_KERNELS
    }
    return {
        "dense_attention": sum(symbol.startswith("MLA-m-") for symbol in unique),
        "sparse_attention": sum(
            symbol.startswith("dsa_tensor_core_attention") for symbol in unique
        ),
        "sparse_moe": sum(symbol.startswith("fused-moe-v2-") for symbol in unique),
        "full_indexer_groups": min(indexer_counts.values(), default=0),
        "indexer_kernel_counts": indexer_counts,
    }


def evaluate_coverage(
    coverage: ModelCoverage,
    module_names: Iterable[str],
    expectation: CoverageExpectation,
) -> list[str]:
    """Return completeness failures; an empty list means the trace passes."""

    failures = []
    for field in ("dense_attention", "sparse_moe"):
        actual = getattr(coverage, field)
        expected = getattr(expectation, field)
        if actual != expected:
            failures.append(f"{field}: expected {expected}, observed {actual}")

    # Pallas emits more than one runtime instruction for some DSA and Indexer
    # shapes. Their raw XLA-symbol counts are useful lower bounds, but are not
    # one-to-one layer counts. Fused MoE remains the exact sparse-layer guard.
    for field in ("sparse_attention", "full_indexer_groups"):
        actual = getattr(coverage, field)
        expected = getattr(expectation, field)
        if actual < expected:
            failures.append(f"{field}: expected at least {expected}, observed {actual}")

    names = tuple(module_names)
    model_positions = [
        index for index, name in enumerate(names) if MODEL_MODULE_MARKER in name
    ]
    model_position = model_positions[-1] if model_positions else -1
    sampler_positions = [
        index for index, name in enumerate(names) if SAMPLER_MODULE_MARKER in name
    ]
    sampler_position = next(
        (index for index in sampler_positions if index > model_position), None
    )
    if expectation.require_sampler and sampler_position is None:
        failures.append("sampler module is absent after the model module")
    set_future_position = next(
        (
            index
            for index, name in enumerate(names)
            if SET_FUTURE_MODULE_MARKER in name
            and index
            > (sampler_position if sampler_position is not None else model_position)
        ),
        None,
    )
    if expectation.require_set_future and set_future_position is None:
        failures.append("set_future_token_ids module is absent after model sampling")
    return failures


def _parse_device_plane(plane_buf: memoryview, device: str) -> dict[str, object] | None:
    plane_name = ""
    metadata = {}
    lines = []
    for number, wire, item in _fields(plane_buf):
        if number == 2 and wire == 2:
            plane_name = _text(item)
        elif number == 3 and wire == 2:
            lines.append(item)
        elif number == 4 and wire == 2:
            key, name, display = _metadata_entry(item)
            if key is not None:
                metadata[key] = (name, display)
    if plane_name != device:
        return None

    line_data = {}
    for line_buf in lines:
        line_name = ""
        timestamp_ps = 0
        event_bufs = []
        for number, wire, item in _fields(line_buf):
            if number == 2 and wire == 2:
                line_name = _text(item)
            elif number == 3 and wire == 0:
                timestamp_ps = item * 1000
            elif number == 4 and wire == 2:
                event_bufs.append(item)
        if line_name in ("XLA Modules", "XLA Ops"):
            line_data[line_name] = (timestamp_ps, event_bufs)
    if "XLA Modules" not in line_data or "XLA Ops" not in line_data:
        return None

    module_ts, module_bufs = line_data["XLA Modules"]
    all_modules = []
    model_modules = []
    for module_buf in module_bufs:
        metadata_id, offset_ps, duration_ps = _event_header(module_buf)
        name, display = metadata.get(metadata_id, (f"metadata:{metadata_id}", ""))
        combined = " ".join(value for value in (name, display) if value)
        module = ModuleEvent(display or name, module_ts + offset_ps, duration_ps)
        all_modules.append(module)
        if MODEL_MODULE_MARKER in combined:
            model_modules.append(module)
    all_modules.sort(key=lambda row: row.start_ps)
    model_modules.sort(key=lambda row: row.start_ps)

    symbols_by_model = [set() for _ in model_modules]
    starts = [row.start_ps for row in model_modules]
    ops_ts, op_bufs = line_data["XLA Ops"]
    for op_buf in op_bufs:
        metadata_id, offset_ps, _ = _event_header(op_buf)
        start_ps = ops_ts + offset_ps
        index = bisect.bisect_right(starts, start_ps) - 1
        if index < 0 or start_ps >= model_modules[index].end_ps:
            continue
        name, display = metadata.get(metadata_id, (f"metadata:{metadata_id}", ""))
        symbols_by_model[index].add(_instruction_symbol(name, display))

    coverages = []
    for module, symbols in zip(model_modules, symbols_by_model, strict=True):
        counts = classify_symbols(symbols)
        coverages.append(
            ModelCoverage(
                module_name=module.name,
                module_wall_ms=module.duration_ps / 1e9,
                **counts,
            )
        )

    relevant = [
        module
        for module in all_modules
        if any(
            marker in module.name
            for marker in (
                "jit_resolve_future_token_ids(",
                MODEL_MODULE_MARKER,
                SAMPLER_MODULE_MARKER,
                SET_FUTURE_MODULE_MARKER,
            )
        )
    ]
    device_span_ms = None
    if relevant:
        device_span_ms = (
            max(row.end_ps for row in relevant) - min(row.start_ps for row in relevant)
        ) / 1e9

    return {
        "device": plane_name,
        "module_sequence": [
            {
                "name": row.name,
                "start_ps": row.start_ps,
                "duration_ms": row.duration_ps / 1e9,
            }
            for row in all_modules
        ],
        "model_coverages": [asdict(row) for row in coverages],
        "forward_device_span_ms": device_span_ms,
    }


def analyze_xplane(path: Path, device: str) -> dict[str, object]:
    data = memoryview(path.read_bytes())
    devices = []
    for number, wire, item in _fields(data):
        if number == 1 and wire == 2:
            parsed = _parse_device_plane(item, device)
            if parsed is not None:
                devices.append(parsed)
    return {"path": str(path), "size_bytes": path.stat().st_size, "devices": devices}


def _select_files(root: Path, path_contains: str | None) -> list[Path]:
    candidates = sorted(root.rglob("*.xplane.pb"))
    if path_contains:
        candidates = [path for path in candidates if path_contains in str(path)]
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--path-contains")
    parser.add_argument("--device", default="/device:TPU:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-dense-attention", type=int, default=3)
    parser.add_argument("--expected-sparse-attention", type=int, default=75)
    parser.add_argument("--expected-sparse-moe", type=int, default=75)
    parser.add_argument("--expected-full-indexer-groups", type=int, default=19)
    parser.add_argument("--allow-missing-sampler", action="store_true")
    parser.add_argument("--allow-missing-set-future", action="store_true")
    args = parser.parse_args()

    expectation = CoverageExpectation(
        dense_attention=args.expected_dense_attention,
        sparse_attention=args.expected_sparse_attention,
        sparse_moe=args.expected_sparse_moe,
        full_indexer_groups=args.expected_full_indexer_groups,
        require_sampler=not args.allow_missing_sampler,
        require_set_future=not args.allow_missing_set_future,
    )
    files = _select_files(args.profile_root, args.path_contains)
    if not files:
        parser.error("no matching .xplane.pb files found")

    result = {"expectation": asdict(expectation), "files": [], "passed": True}
    for path in files:
        file_row = analyze_xplane(path, args.device)
        file_failures = []
        for device_row in file_row["devices"]:
            coverages = [ModelCoverage(**row) for row in device_row["model_coverages"]]
            if len(coverages) != 1:
                file_failures.append(
                    f"expected exactly one model module, observed {len(coverages)}"
                )
                continue
            module_names = [row["name"] for row in device_row["module_sequence"]]
            file_failures.extend(
                evaluate_coverage(coverages[0], module_names, expectation)
            )
        if not file_row["devices"]:
            file_failures.append(f"device plane {args.device!r} is absent")
        file_row["failures"] = file_failures
        file_row["passed"] = not file_failures
        result["files"].append(file_row)
        result["passed"] = result["passed"] and file_row["passed"]

    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
