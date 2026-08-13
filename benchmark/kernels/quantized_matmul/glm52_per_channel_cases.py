"""GLM-5.2 TP-local shapes for the per-channel quantized matmul benchmark."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

ANCHOR_M = (2, 4, 8, 16, 128, 1024)
FULL_M = (1, 2, 4, 8, 16, 32, 64, 128, 256, 1024)
MODES = ("w8a8", "w8a16")
IMPLEMENTATIONS = ("xla", "pallas_aligned")
SUITES = ("anchor", "full")


@dataclass(frozen=True)
class ProjectionShape:
    operation: str
    tp1: tuple[int, int]
    tp2: tuple[int, int]

    def for_tp(self, tp_degree: int) -> tuple[int, int]:
        if tp_degree == 1:
            return self.tp1
        if tp_degree == 2:
            return self.tp2
        raise ValueError(f"unsupported TP degree: {tp_degree}")


GLM52_PROJECTIONS = (
    ProjectionShape("q_a_proj", (2048, 6144), (2048, 6144)),
    ProjectionShape("q_b_proj", (16384, 2048), (8192, 2048)),
    ProjectionShape("kv_a_proj_with_mqa", (576, 6144), (576, 6144)),
    ProjectionShape("o_proj", (6144, 16384), (6144, 8192)),
    ProjectionShape("merged_gate_up_proj", (24576, 6144), (12288, 6144)),
    ProjectionShape("down_proj", (6144, 12288), (6144, 6144)),
    # Full DSA indexer projections are replicated rather than tensor-sharded.
    # They are included because C128/DP16 decode pads to local M=8 and the
    # production per-channel registry requires an exact entry for every
    # quantized linear reached by that graph.
    ProjectionShape("indexer_wq_b", (4096, 2048), (4096, 2048)),
    ProjectionShape("indexer_wk", (128, 6144), (128, 6144)),
)

FULL_SHAPES = (
    ("kv_a_proj_with_mqa", 1),
    ("q_a_proj", 1),
    ("q_b_proj", 2),
    ("o_proj", 2),
    ("merged_gate_up_proj", 2),
    ("down_proj", 2),
    ("indexer_wq_b", 1),
    ("indexer_wk", 1),
)


@dataclass(frozen=True, order=True)
class CaseAlias:
    operation: str
    tp_degree: int


@dataclass(frozen=True, order=True)
class CaseKey:
    m: int
    n: int
    k: int
    mode: str
    implementation: str
    variant: str = "baseline"


@dataclass(frozen=True)
class BenchmarkCase:
    key: CaseKey
    suites: tuple[str, ...]
    aliases: tuple[CaseAlias, ...]

    @property
    def primary_alias(self) -> CaseAlias:
        return self.aliases[0]

    @property
    def case_id(self) -> str:
        alias = self.primary_alias
        return (
            f"glm52_per_channel_{alias.operation}_tp{alias.tp_degree}"
            f"_m{self.key.m}_n{self.key.n}_k{self.key.k}"
            f"_{self.key.mode}_{self.key.implementation}_{self.key.variant}"
        )


@dataclass(frozen=True)
class _LogicalCase:
    suite: str
    alias: CaseAlias
    key: CaseKey


def _projection_map() -> dict[str, ProjectionShape]:
    return {projection.operation: projection for projection in GLM52_PROJECTIONS}


def _validate_choices(name: str, values: Sequence[str], supported: Sequence[str]) -> None:
    unknown = sorted(set(values) - set(supported))
    if unknown:
        raise ValueError(f"unknown {name}: {unknown}; supported={list(supported)}")


def _logical_cases(
    suites: Sequence[str],
    modes: Sequence[str],
    implementations: Sequence[str],
) -> Iterable[_LogicalCase]:
    projections = _projection_map()
    for suite in suites:
        if suite == "anchor":
            aliases = (
                CaseAlias(projection.operation, tp_degree)
                for projection in GLM52_PROJECTIONS
                for tp_degree in (1, 2)
            )
            m_values = ANCHOR_M
        elif suite == "full":
            aliases = (CaseAlias(operation, tp_degree) for operation, tp_degree in FULL_SHAPES)
            m_values = FULL_M
        else:
            raise AssertionError(f"validated unsupported suite: {suite}")

        for alias in aliases:
            n, k = projections[alias.operation].for_tp(alias.tp_degree)
            for m in m_values:
                for mode in modes:
                    for implementation in implementations:
                        yield _LogicalCase(
                            suite=suite,
                            alias=alias,
                            key=CaseKey(
                                m=m,
                                n=n,
                                k=k,
                                mode=mode,
                                implementation=implementation,
                            ),
                        )


def build_cases(
    *,
    suites: Sequence[str] = SUITES,
    operations: Sequence[str] | None = None,
    tp_degrees: Sequence[int] | None = None,
    m_values: Sequence[int] | None = None,
    modes: Sequence[str] = MODES,
    implementations: Sequence[str] = IMPLEMENTATIONS,
) -> list[BenchmarkCase]:
    """Build deduplicated physical cases while retaining logical op/TP aliases."""

    suites = tuple(suites)
    modes = tuple(modes)
    implementations = tuple(implementations)
    _validate_choices("suites", suites, SUITES)
    _validate_choices("modes", modes, MODES)
    _validate_choices("implementations", implementations, IMPLEMENTATIONS)
    if not suites or not modes or not implementations:
        raise ValueError("suites, modes, and implementations must be non-empty")

    supported_operations = tuple(projection.operation for projection in GLM52_PROJECTIONS)
    operation_filter = None if operations is None else set(operations)
    if operation_filter is not None:
        _validate_choices("operations", tuple(operation_filter), supported_operations)
    tp_filter = None if tp_degrees is None else set(tp_degrees)
    if tp_filter is not None:
        unknown_tp = sorted(tp_filter - {1, 2})
        if unknown_tp:
            raise ValueError(f"unsupported TP degrees: {unknown_tp}")
    m_filter = None if m_values is None else set(m_values)
    if m_filter is not None and (not m_filter or any(m <= 0 for m in m_filter)):
        raise ValueError("M filters must contain positive integers")

    grouped: dict[CaseKey, dict[str, set]] = {}
    for logical in _logical_cases(suites, modes, implementations):
        if operation_filter is not None and logical.alias.operation not in operation_filter:
            continue
        if tp_filter is not None and logical.alias.tp_degree not in tp_filter:
            continue
        if m_filter is not None and logical.key.m not in m_filter:
            continue
        entry = grouped.setdefault(logical.key, {"suites": set(), "aliases": set()})
        entry["suites"].add(logical.suite)
        entry["aliases"].add(logical.alias)

    return [
        BenchmarkCase(
            key=key,
            suites=tuple(sorted(metadata["suites"])),
            aliases=tuple(sorted(metadata["aliases"])),
        )
        for key, metadata in sorted(grouped.items())
    ]


def expected_case_counts() -> dict[str, int]:
    anchor = {case.key for case in build_cases(suites=("anchor",))}
    full = {case.key for case in build_cases(suites=("full",))}
    return {
        "anchor": len(anchor),
        "full": len(full),
        "overlap": len(anchor & full),
        "union": len(anchor | full),
    }
