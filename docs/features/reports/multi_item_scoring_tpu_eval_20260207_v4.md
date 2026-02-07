# Multi-Item Scoring TPU Evaluation (2026-02-07, v4)

Note: Broader cross-model validation is captured in
`docs/features/reports/multi_item_scoring_tpu_model_matrix_20260207_v1.md`.

## Environment

- Project: `sglang-jax-tests-1769450780`
- Zone: `us-east5-b`
- TPU: `v6e-1`
- Model: `Qwen/Qwen3-0.6B`
- Branch snapshot: `feat/multi-item-scoring`

Delta from v3:

1. default chunk size changed to `2` (`--multi-item-scoring-chunk-size`)
2. strict regression test now enforces zero changed-length drift

## Artifacts

- `docs/features/reports/multi_item_eval_results_20260207_v4.json`
- `docs/features/reports/serial_score_eval_results_20260207_v2.json`
- `docs/features/reports/multi_vs_serial_eval_results_20260207_v4.json`
- `docs/features/reports/jax_torch_parity_results_20260207_v4.json`
- `docs/features/reports/sgl_multi_server_20260207_v4.log`

## Results

## Correctness / Isolation

- Same-length mutation isolation max diff: `0.0`
- Changed-length mutation isolation max diff: `0.0`

Interpretation:

- The changed-length leakage observed in prior runs is removed for this benchmark setup.

## Semantics-Aware Equivalence

From `multi_vs_serial_eval_results_20260207_v4.json`:

- Query-only comparison (`query + item`): max diff `0.3387`
- Delimiter-aligned comparison (`query + delimiter + item`): max diff `0.0148`

Interpretation:

- Query-only mismatch remains expected due delimiter-visible multi-item semantics.
- Delimiter-aligned comparison remains close.

## JAX vs PyTorch Parity

From `jax_torch_parity_results_20260207_v4.json`:

- `jax_serial` vs torch serial (`query + item`): max diff `0.0041`
- `jax_multi` vs torch multi-semantic (`query + delimiter + item`): max diff `0.0115`

Interpretation:

- Serial parity remains strong.
- Multi semantic parity remains close with stable behavior.

## Performance (multi call vs serial one-item calls)

`speedup_vs_serial_p50 = serial_p50 / multi_p50`

- `N=1`: `0.926x`
- `N=8`: `3.451x`
- `N=32`: `4.689x`
- `N=64`: `5.324x`
- `N=128`: `5.298x`

Interpretation:

- Correctness-focused chunking reduced peak speedup versus v3, but multi-item still provides strong batch gains.

## CI Regression Coverage

- `test/srt/test_multi_item_regression.py::TestMultiItemRegression.test_multi_item_isolation_and_speed`
- Included in `performance-test-tpu-v6e-1` suite via `test/srt/run_suite.py`.
