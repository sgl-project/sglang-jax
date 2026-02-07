# Multi-Item Scoring TPU Evaluation (2026-02-07)

> Superseded by `docs/features/reports/multi_item_scoring_tpu_eval_20260207_v2.md`.

## Environment

- Project: `sglang-jax-tests-1769450780`
- Zone: `us-east5-b`
- TPU: `v6e-1`
- Model: `Qwen/Qwen3-0.6B`
- Branch snapshot: `feat/multi-item-scoring` (workspace copy on TPU VM)
- Server mode: FlashAttention backend (`--attention-backend fa`)

Artifacts:

- `docs/features/reports/multi_item_eval_results_20260207.json`
- `docs/features/reports/multi_vs_serial_eval_results_20260207.json`
- `docs/features/reports/jax_torch_parity_results_20260207.json`
- `docs/features/reports/sgl_multi_server_20260207.log`

## What Was Validated

1. Multi-item API path works for text and token modes.
2. Delimiter collision validation rejects unsafe inputs.
3. Isolation behavior under two mutations:
   - same-length mutation of one item
   - changed-length mutation of one item
4. Throughput/latency for `N in {1, 8, 32, 64, 128}`.
5. Parity:
   - JAX multi-item vs JAX serial one-item requests
   - JAX (multi + serial) vs PyTorch Transformers reference (one-item equivalent sequence)

## Key Results

## Correctness

- Same-length mutation isolation is exact:
  - `same_length_mutation_max_abs_diff = 0.0`
- Changed-length mutation causes drift in unrelated items:
  - `changed_length_mutation_max_abs_diff = 0.0770`

Interpretation:

- Attention masking isolates items correctly when token layout is length-stable.
- Length-dependent position coupling remains (items after the changed item can move in absolute position and change score).

## Parity

- JAX serial vs PyTorch reference:
  - `max_abs_diff_jax_serial_vs_torch = 0.0344`
- JAX multi vs PyTorch reference:
  - `max_abs_diff_jax_multi_vs_torch = 0.2525`
- JAX multi vs JAX serial:
  - `equiv_max_abs_diff = 0.2520`

Interpretation:

- Single-item behavior is reasonably close to PyTorch reference.
- Multi-item mode shows substantial deviation, consistent with length/position coupling.

## Performance (multi call vs serial one-item calls)

`speedup_vs_serial_p50 = serial_p50 / multi_p50`

- `N=1`: `0.995x`
- `N=8`: `1.688x` (multi faster)
- `N=32`: `0.554x` (multi slower)
- `N=64`: `0.279x` (multi slower)
- `N=128`: `0.134x` (multi much slower)

Interpretation:

- Current MVP helps at small N but regresses heavily at larger N on `v6e-1`.
- Main bottleneck is consistent with current custom mask construction/cost (`O(T^2)` dense path).

## Assessment

Current implementation should be treated as a feature-gated correctness MVP, not production-ready for large-N scoring workloads.

Blocking gaps before rollout:

1. Position-reset semantics per item block (to remove length-coupled drift and improve parity).
2. More efficient mask/dataflow path for large N (avoid dense `T^2` overhead).

## Recommended Next Technical Steps

1. Add per-item relative position reset in JAX path (FlashInfer-like token position semantics).
2. Replace dense flat custom mask path with a segmented/sparse representation in the attention pipeline.
3. Add CI/e2e parity tests with fixed tokenized fixtures:
   - same-length perturbation invariance
   - changed-length perturbation invariance target
   - JAX multi vs serial tolerance gates
4. Re-run this benchmark matrix on TPU after each major change and track regression budget.
