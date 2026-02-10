# Multi-Item Scoring Evaluation Plan

Evaluation reports and machine-readable artifacts live in the dev-scripts repo:

`sglang-jax-dev-scripts/evaluations/multi-item-scoring/`

Latest run reports:

- `multi_item_scoring_tpu_eval_20260207_v4.md`
- `multi_item_scoring_tpu_model_matrix_20260207_v1.md`

Latest machine-readable artifacts (per model):

**Qwen3-0.6B:**
- `multi_item_eval_results_20260207_v4.json`
- `serial_score_eval_results_20260207_v2.json`
- `multi_vs_serial_eval_results_20260207_v4.json`
- `jax_torch_parity_results_20260207_v4.json`

**Qwen3-1.7B:**
- `multi_item_eval_results_20260207_qwen3_1_7b_v2.json`
- `serial_score_eval_results_20260207_qwen3_1_7b_v2.json`
- `multi_vs_serial_eval_results_20260207_qwen3_1_7b_v2.json`
- `jax_torch_parity_results_20260207_qwen3_1_7b_v2.json`

**Qwen3-4B:**
- `multi_item_eval_results_20260207_qwen3_4b_v1.json`
- `serial_score_eval_results_20260207_qwen3_4b_v1.json`
- `multi_vs_serial_eval_results_20260207_qwen3_4b_v1.json`
- `jax_torch_parity_results_20260207_qwen3_4b_v1.json`

## Goals

Evaluate strategy choices with consistent criteria:

- correctness
- item-isolation guarantees
- latency/throughput
- compile behavior
- memory usage

## Strategies

Evaluate the same workloads across:

1. `custom_mask` (current implementation)
2. causal-only baseline (negative control, expected incorrect cross-item leakage)
3. future alternatives (for example procedural/sparse masks) if implemented

## Workload Matrix

Use fixed tokenized inputs for apples-to-apples comparisons.

- Item counts: `1, 8, 32, 64, 128`
- Query lengths: short/medium/long buckets
- Item lengths: mixed (including empty-item cases)
- Label set sizes: `2, 4, 8`

## Correctness Checks

1. **Single vs multi equivalence (JAX)**
   - Compare per-item boundary logprobs and final score vectors.
2. **Isolation check**
   - Perturb one item; unrelated item scores should remain stable.
3. **Delimiter edge cases**
   - Validate collision rejection and max-length guards.

## JAX vs PyTorch Parity

Use identical:

- model revision
- tokenizer revision
- tokenized request payloads
- label token ids

Compare:

1. raw boundary logprobs
2. final outputs for `apply_softmax=True`
3. final outputs for `apply_softmax=False`

Document intentional divergences separately (for example stricter validation).

## Performance and Compile Metrics

Track:

- P50/P95 latency
- throughput (items/s)
- first-request compile latency by token bucket
- number of new compile variants
- peak mask memory for each token bucket

## Reporting Format

Record each run with:

- strategy
- model/tokenizer revision
- workload id
- metric values
- pass/fail against thresholds

Keep outputs in machine-readable format (CSV/JSON) plus a short markdown summary.

## Repro Script Entry Points

For repeatable TPU runs, use:

1. `scripts/multi_item/evaluate_score_endpoint.py`
   - Collects isolation/performance metrics for one endpoint in `multi` or `serial` mode.
2. `scripts/multi_item/combine_multi_item_eval.py`
   - Combines multi+serial artifacts and computes PyTorch references.

### Important Reference Semantics

Report both comparisons:

1. **Query-only serial semantics** (`query + item`)
2. **Multi semantic-aligned reference** (`query + delimiter + item`)

The multi implementation intentionally inserts visible delimiters, so query-only parity is expected to differ.
