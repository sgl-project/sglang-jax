# Multi-Item Scoring TPU Evaluation (2026-02-07, v3)

> Superseded by `docs/features/reports/multi_item_scoring_tpu_eval_20260207_v4.md`.

## Environment

- Project: `sglang-jax-tests-1769450780`
- Zone: `us-east5-b`
- TPU: `v6e-1`
- Model: `Qwen/Qwen3-0.6B`
- Branch snapshot: `feat/multi-item-scoring`

Delta from v2:

1. updated multi-item custom mask topology:
   - shared prefix is `query + first delimiter`
   - each item block is isolated from other item delimiters/blocks
2. added regression coverage and CI suite wiring:
   - `test/srt/test_multi_item_regression.py`
   - `test/srt/run_suite.py` (`performance-test-tpu-v6e-1`)

## Artifacts

- `docs/features/reports/multi_item_eval_results_20260207_v3.json`
- `docs/features/reports/serial_score_eval_results_20260207_v2.json`
- `docs/features/reports/multi_vs_serial_eval_results_20260207_v3.json`
- `docs/features/reports/jax_torch_parity_results_20260207_v3.json`

## Results

## Correctness / Isolation

- Same-length mutation isolation max diff: `0.0`
- Changed-length mutation isolation max diff: `0.0117`

Interpretation:

- Same-length invariance remains exact.
- Changed-length drift improved from v2 (`0.0363 -> 0.0117`) but is not fully zero.

## Semantics-Aware Equivalence

From `multi_vs_serial_eval_results_20260207_v3.json`:

- Query-only comparison (`query + item`): max diff `0.3387`
- Delimiter-aligned comparison (`query + delimiter + item`): max diff `0.0148`

Interpretation:

- Query-only mismatch remains expected due intentional delimiter-visible semantics.
- Delimiter-aligned comparison remains close.

## JAX vs PyTorch Parity

From `jax_torch_parity_results_20260207_v3.json`:

- `jax_serial` vs torch serial (`query + item`): max diff `0.0041`
- `jax_multi` vs torch multi-semantic (`query + delimiter + item`): max diff `0.0115`
- `jax_multi` vs torch serial (`query + item`): max diff `0.3402`

Interpretation:

- Serial parity remains strong.
- Multi semantic parity remains close, though slightly looser than v2 due mask change.

## Performance (multi call vs serial one-item calls)

`speedup_vs_serial_p50 = serial_p50 / multi_p50`

- `N=1`: `0.884x`
- `N=8`: `4.984x`
- `N=32`: `7.514x`
- `N=64`: `8.800x`
- `N=128`: `9.405x`

Interpretation:

- Multi-item remains strongly beneficial for batch scoring.
- Single-item path is still better through one-item calls.

## CI Regression Coverage

New TPU regression check:

- `test/srt/test_multi_item_regression.py::TestMultiItemRegression.test_multi_item_isolation_and_speed`

Current gates:

1. same-length isolation max diff must be `0.0`
2. changed-length isolation max diff must be `<= 0.02`
3. 32-item multi call must be at least `3.0x` faster than 32 serial one-item calls
