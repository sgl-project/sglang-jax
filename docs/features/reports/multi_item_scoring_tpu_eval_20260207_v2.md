# Multi-Item Scoring TPU Evaluation (2026-02-07, v2)

> Superseded by `docs/features/reports/multi_item_scoring_tpu_eval_20260207_v3.md`.

## Environment

- Project: `sglang-jax-tests-1769450780`
- Zone: `us-east5-b`
- TPU: `v6e-1`
- Model: `Qwen/Qwen3-0.6B`
- Branch snapshot: `feat/multi-item-scoring`

Key implementation deltas validated in this run:

1. delimiter-token validation uses `len(tokenizer)` (special-token-safe)
2. multi-item per-request chunking (`--multi-item-scoring-chunk-size`)
3. per-item extend-position reset
4. static custom-mask assembly cleanup
5. repeatable evaluation scripts under `scripts/multi_item/`

## Artifacts

- `docs/features/reports/multi_item_eval_results_20260207_v2.json`
- `docs/features/reports/serial_score_eval_results_20260207_v2.json`
- `docs/features/reports/multi_vs_serial_eval_results_20260207_v2.json`
- `docs/features/reports/jax_torch_parity_results_20260207_v2.json`
- `docs/features/reports/sgl_multi_server_20260207_v2.log`
- `docs/features/reports/sgl_single_server_20260207_v2.log`

## Repro Commands

### 1) Multi-item endpoint run

```bash
python -m sgl_jax.launch_server \
  --model-path Qwen/Qwen3-0.6B \
  --host 0.0.0.0 --port 30010 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tp-size 1 \
  --multi-item-scoring-delimiter 151643 \
  --multi-item-scoring-chunk-size 8 \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --attention-backend fa
```

```bash
PYTHONPATH=python python scripts/multi_item/evaluate_score_endpoint.py \
  --mode multi \
  --url http://127.0.0.1:30010/v1/score \
  --model Qwen/Qwen3-0.6B \
  --rounds 3 --warmup 1 --item-counts 1,8,32,64,128 \
  --output-json docs/features/reports/multi_item_eval_results_20260207_v2.json
```

### 2) Serial endpoint run

```bash
python -m sgl_jax.launch_server \
  --model-path Qwen/Qwen3-0.6B \
  --host 0.0.0.0 --port 30011 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tp-size 1 \
  --attention-backend fa
```

```bash
PYTHONPATH=python python scripts/multi_item/evaluate_score_endpoint.py \
  --mode serial \
  --url http://127.0.0.1:30011/v1/score \
  --model Qwen/Qwen3-0.6B \
  --rounds 3 --warmup 1 --item-counts 1,8,32,64,128 \
  --output-json docs/features/reports/serial_score_eval_results_20260207_v2.json
```

### 3) Combine + Torch parity

```bash
PYTHONPATH=python python scripts/multi_item/combine_multi_item_eval.py \
  --multi-json docs/features/reports/multi_item_eval_results_20260207_v2.json \
  --serial-json docs/features/reports/serial_score_eval_results_20260207_v2.json \
  --output-multi-vs-serial-json docs/features/reports/multi_vs_serial_eval_results_20260207_v2.json \
  --output-jax-torch-parity-json docs/features/reports/jax_torch_parity_results_20260207_v2.json \
  --torch-model Qwen/Qwen3-0.6B
```

## Results

## Correctness / Isolation

- Same-length mutation isolation max diff: `0.0`
- Changed-length mutation isolation max diff: `0.0363`

Interpretation:

- Position-reset update removed same-length leakage.
- Changed-length leakage is reduced vs previous run (`0.0770 -> 0.0363`) but not fully eliminated.

## Semantics-Aware Equivalence

From `multi_vs_serial_eval_results_20260207_v2.json`:

- Query-only comparison (`query + item`): max diff `0.3387`
- Delimiter-aligned comparison (`query + delimiter + item`): max diff `0.0148`

Interpretation:

- Large query-only gap is expected due visible delimiter semantics in multi-item mode.
- Once aligned to multi semantics, multi/serial outputs are close.

## JAX vs PyTorch Parity

From `jax_torch_parity_results_20260207_v2.json`:

- `jax_serial` vs torch serial (`query + item`): max diff `0.0041`
- `jax_multi` vs torch multi-semantic (`query + delimiter + item`): max diff `0.0034`
- `jax_multi` vs torch serial (`query + item`): max diff `0.3402`

Interpretation:

- Implementation parity is strong when compared under matching semantics.
- The large mismatch appears only when semantic formats are intentionally different.

## Performance (multi call vs serial one-item calls)

`speedup_vs_serial_p50 = serial_p50 / multi_p50`

- `N=1`: `0.890x`
- `N=8`: `4.887x`
- `N=32`: `7.380x`
- `N=64`: `8.695x`
- `N=128`: `9.305x`

Interpretation:

- Multi-item path now scales strongly for batch scoring on v6e-1.
- Single-item requests are slightly faster through the serial path (expected overhead tradeoff).
