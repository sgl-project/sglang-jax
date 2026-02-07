# Multi-Item Scoring TPU Model Matrix (2026-02-07, v1)

## Environment

- Project: `sglang-jax-tests-1769450780`
- Zone: `us-east5-b`
- TPU: `v6e-1`
- Branch snapshot: `feat/multi-item-scoring`
- Attention backend: `fa`
- Multi-item chunk size: `2` (default in this branch)

## Scope

Broader validation matrix after the v4 correctness fix:

1. `Qwen/Qwen3-0.6B` (baseline from v4)
2. `Qwen/Qwen3-1.7B`
3. `Qwen/Qwen3-4B`

## Artifacts

### Qwen3-0.6B

- `docs/features/reports/multi_item_eval_results_20260207_v4.json`
- `docs/features/reports/serial_score_eval_results_20260207_v2.json`
- `docs/features/reports/multi_vs_serial_eval_results_20260207_v4.json`
- `docs/features/reports/jax_torch_parity_results_20260207_v4.json`

### Qwen3-1.7B

- `docs/features/reports/multi_item_eval_results_20260207_qwen3_1_7b_v2.json`
- `docs/features/reports/serial_score_eval_results_20260207_qwen3_1_7b_v2.json`
- `docs/features/reports/multi_vs_serial_eval_results_20260207_qwen3_1_7b_v2.json`
- `docs/features/reports/jax_torch_parity_results_20260207_qwen3_1_7b_v2.json`

### Qwen3-4B

- `docs/features/reports/multi_item_eval_results_20260207_qwen3_4b_v1.json`
- `docs/features/reports/serial_score_eval_results_20260207_qwen3_4b_v1.json`
- `docs/features/reports/multi_vs_serial_eval_results_20260207_qwen3_4b_v1.json`
- `docs/features/reports/jax_torch_parity_results_20260207_qwen3_4b_v1.json`

## Summary Table

| Model | Same-len isolation max | Changed-len isolation max | Delimiter-aligned equiv max | JAX serial vs torch serial max | JAX multi vs torch multi-sem max | Speedup N=8 | Speedup N=32 | Speedup N=128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 0.0000 | 0.0000 | 0.0148 | 0.0041 | 0.0115 | 3.45x | 4.69x | 5.30x |
| Qwen3-1.7B | 0.0000 | 0.0000 | 0.0150 | 0.0118 | 0.0050 | 3.22x | 4.27x | 5.22x |
| Qwen3-4B | 0.0000 | 0.0000 | 0.0119 | 0.0052 | 0.0041 | 3.07x | 3.85x | 4.29x |

## Notes

- Changed-length isolation drift is `0.0` across all validated models in this matrix.
- Delimiter-aligned JAX multi vs JAX serial equivalence remains tight (`<= 0.0150` max diff).
- Query-only equivalence remains model-dependent and is not the primary semantic target for this feature.
- Multi-item throughput remains strong for larger item counts (roughly `3.1x`-`5.3x` speedup vs serial one-item calls).

## Compatibility Observations

Two attempted Qwen2.5 models (`Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`) failed on first scoring request with:

- `ValueError: ... reshape((2, -1, 8, 128)) ... size of ref (1024)`

This is consistent with a fused-KV layout assumption mismatch in the current kernel path and is tracked as follow-up kernel compatibility work, not a multi-item masking correctness issue.

## Rollout Readiness

For Qwen3 targets on TPU v6e-1 with current feature gate and settings, results are rollout-ready:

- isolation correctness: pass
- semantic-aligned parity: pass
- throughput objective: pass
