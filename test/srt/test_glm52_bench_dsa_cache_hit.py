from __future__ import annotations

import importlib.util
from pathlib import Path

BENCHMARK_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmark"
    / "glm52"
    / "bench_dsa_cache_hit.py"
)
SPEC = importlib.util.spec_from_file_location("bench_dsa_cache_hit", BENCHMARK_PATH)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


def test_shared_prefix_inputs_share_only_the_prefix() -> None:
    prefixes, extended = BENCHMARK._make_inputs(4, 8, 3, prefix_mode="shared")

    assert prefixes == [prefixes[0]] * 4
    assert all(value[:8] == prefixes[0] for value in extended)
    assert len({tuple(value[8:]) for value in extended}) == 4
    assert len({value[8] for value in extended}) == 4
    assert all(len(set(value[8:])) > 1 for value in extended)


def test_independent_prefix_inputs_do_not_share_prefixes() -> None:
    prefixes, extended = BENCHMARK._make_inputs(4, 8, 3, prefix_mode="independent")

    assert len({tuple(value) for value in prefixes}) == 4
    assert len({value[0] for value in prefixes}) == 4
    assert all(value[:8] == prefixes[i] for i, value in enumerate(extended))


def test_random_inputs_are_reproducible_by_seed() -> None:
    first = BENCHMARK._make_inputs(4, 8, 8, prefix_mode="shared", random_seed=7)
    repeated = BENCHMARK._make_inputs(4, 8, 8, prefix_mode="shared", random_seed=7)
    different = BENCHMARK._make_inputs(4, 8, 8, prefix_mode="shared", random_seed=8)

    assert first == repeated
    assert first != different
