from benchmark.glm52.xplane_runtime_coverage import (
    EXTEND_INDEXER_KERNELS,
    CoverageExpectation,
    ModelCoverage,
    classify_symbols,
    evaluate_coverage,
)


def _coverage(symbols):
    return ModelCoverage(
        module_name="jit_jitted_run_model(...)...",
        module_wall_ms=3500.0,
        **classify_symbols(symbols),
    )


def test_complete_glm52_extend_coverage_passes():
    symbols = {f"MLA-m-shape.{index}" for index in range(1, 4)}
    symbols.update(f"dsa_tensor_core_attention.{index}" for index in range(1, 76))
    symbols.update(f"fused-moe-v2-shape.{index}" for index in range(1, 76))
    for kernel in EXTEND_INDEXER_KERNELS:
        symbols.update(f"{kernel}.{index}" for index in range(1, 20))

    failures = evaluate_coverage(
        _coverage(symbols),
        (
            "jit_jitted_run_model(...)...",
            "jit_jitted_sampler(...)...",
            "jit_set_future_token_ids(...)...",
        ),
        CoverageExpectation(),
    )

    assert failures == []


def test_partial_glm52_extend_coverage_is_rejected_with_semantic_counts():
    symbols = {f"MLA-m-shape.{index}" for index in range(1, 4)}
    symbols.update(f"dsa_tensor_core_attention.{index}" for index in range(1, 4))
    symbols.update(f"fused-moe-v2-shape.{index}" for index in range(1, 4))
    for kernel in EXTEND_INDEXER_KERNELS:
        symbols.add(f"{kernel}.1")

    failures = evaluate_coverage(
        _coverage(symbols),
        ("jit_jitted_run_model(...)...",),
        CoverageExpectation(),
    )

    assert failures == [
        "sparse_moe: expected 75, observed 3",
        "sparse_attention: expected at least 75, observed 3",
        "full_indexer_groups: expected at least 19, observed 1",
        "sampler module is absent after the model module",
        "set_future_token_ids module is absent after model sampling",
    ]
