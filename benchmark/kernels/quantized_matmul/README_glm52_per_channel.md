# GLM-5.2 per-channel quantized matmul benchmark

This benchmark compares the current shard-local XLA implementation with the
TPU-Inference-aligned Pallas kernel for FP8 per-output-channel weights. `TP=2`
selects the corresponding local matrix shape on one device; it does not run a
collective.

Primary latency is extracted from XProf `device_duration_ps`. Host wall time is
diagnostic only. Run each formal job in three fresh processes with distinct
`--process-run-id` values.

The completed v7x measurements and optimization conclusions are documented in
[RESULTS_glm52_per_channel_v7x.md](RESULTS_glm52_per_channel_v7x.md).

## Inspect the case matrix without JAX

```bash
python -m benchmark.kernels.quantized_matmul.bench_glm52_per_channel \
  --suite all --list-cases
```

Expected counts are anchor 200, full 240, overlap 120, and union 320.

## TPU smoke

```bash
python -m benchmark.kernels.quantized_matmul.bench_glm52_per_channel \
  --suite smoke \
  --weight-ring-count 1 \
  --samples 3 \
  --process-run-id smoke-1 \
  --dump-hlo-dir /tmp/glm52-per-channel/hlo \
  --output-jsonl /tmp/glm52-per-channel/smoke.jsonl
```

## Weight-ring calibration

Run the four calibration shapes separately for ring sizes 1, 4, and 16. The
example below selects one shape; repeat for `q_b_proj TP=2`, `o_proj TP=2`, and
`merged_gate_up_proj TP=2`.

```bash
python -m benchmark.kernels.quantized_matmul.bench_glm52_per_channel \
  --suite anchor \
  --operations kv_a_proj_with_mqa \
  --tp-degree 1 \
  --m 2 \
  --weight-ring-count 16 \
  --process-run-id ring16-1 \
  --output-jsonl /tmp/glm52-per-channel/ring16-run1.jsonl
```

Freeze ring 4 for the formal sweep when ring 4 and ring 16 p50 differ by at
most 2%; otherwise use the larger ring or label the result
`cache_plateau_not_reached`. The completed v7x calibration observed a maximum
absolute p50 delta of 0.449% and therefore used ring 4.

## Formal jobs

Split each independent run into eight jobs:

```text
{QKVO, Dense-MLP} × {TP1, TP2} × {W8A8, W8A16}
```

Example QKVO TP=2 W8A8 job:

```bash
python -m benchmark.kernels.quantized_matmul.bench_glm52_per_channel \
  --suite all \
  --operations q_a_proj q_b_proj kv_a_proj_with_mqa o_proj \
  --tp-degree 2 \
  --modes w8a8 \
  --weight-ring-count 4 \
  --warmup 10 \
  --samples 30 \
  --process-run-id run-1 \
  --trace-root /tmp/glm52-per-channel/xprof-run1 \
  --dump-hlo-dir /tmp/glm52-per-channel/hlo-run1 \
  --output-jsonl /tmp/glm52-per-channel/qkvo-tp2-w8a8-run1.jsonl
```

Run the same command in fresh processes with `run-2` and `run-3`. Dense MLP
uses `--operations merged_gate_up_proj down_proj`.

## Fixed profile set

```bash
python -m benchmark.kernels.quantized_matmul.bench_glm52_per_channel \
  --suite profiles --weight-ring-count 4 --process-run-id profile-1 \
  --output-jsonl /tmp/glm52-per-channel/profiles.jsonl
```

This selects 32 cases: four shape families, `M={2,1024}`, W8A8/W8A16, and
XLA/Pallas.

## Benchmark-only tuned-value sweep

Inject one Pallas tile without changing the production tuning table:

```bash
python -m benchmark.kernels.quantized_matmul.bench_glm52_per_channel \
  --suite anchor \
  --operations q_b_proj \
  --tp-degree 2 \
  --m 2 1024 \
  --modes w8a8 w8a16 \
  --implementations pallas_aligned \
  --tuned-value 2,1024,2048 \
  --weight-ring-count 4 \
  --output-jsonl /tmp/glm52-per-channel/tuned-bm2-bn1024-bk2048.jsonl
```

`--tuned-value BM,BN,BK` labels the case variant and records
`metadata_source=benchmark_cli_tuned_value_override`. It is deliberately
incompatible with the XLA implementation so a sweep cannot silently duplicate
an unchanged XLA baseline.

## Aggregate independent runs

```bash
python -m benchmark.kernels.quantized_matmul.analyze_glm52_per_channel \
  /tmp/glm52-per-channel/*-run1.jsonl \
  /tmp/glm52-per-channel/*-run2.jsonl \
  /tmp/glm52-per-channel/*-run3.jsonl \
  --expected-runs 3 \
  --strict \
  --output /tmp/glm52-per-channel/summary.json
```

The analyzer reports raw-sample aggregates, independent-run counts, CV issues,
Pallas/XLA speedups, W8A8/W8A16 control ratios, and ring 4/ring 16 plateaus.
W8A8/W8A16 is explicitly a control ratio, not a pure activation-quantization
cost measurement.
