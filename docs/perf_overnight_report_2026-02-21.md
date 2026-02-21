# Overnight Perf Report (2026-02-21)

## Scope

- TPU: `pr27-repro-0221220428` (`us-east5-a`, v6e-1)
- Model: `Qwen/Qwen3-0.6B`
- Path: `multi_item_enable_score_from_cache_v2=1`, `multi_item_score_label_only_logprob=1`
- Kernel lane: `SGLANG_RPA_KERNEL_V11=1`

## P0 Results

### Canary Sweep (items_per_step tuning first)

- `max_running_requests=20` sweep (`ips=16/20/24/32/48/64`) completed.
- `max_running_requests=24` sweep (`ips=16/20/24/32/48/64`) completed after stability gate.
- Stability gate outcome:
  - `score_from_cache_v2_metrics.fallback=0` on all sweep points.
  - Score-path fallback in logs: `0.0` for all points.

Top sweep points:

- `mr20_ips48`: `1701.59 items/s`, latency `p99=0.3171s`
- `mr20_ips24`: `1696.79 items/s`, latency `p99=0.2981s` (more stable tails)
- `mr24_ips48`: `1843.26 items/s`, latency `p99=0.2898s`
- `mr24_ips32`: `1839.96 items/s`, latency `p99=0.2847s` (better tail stability)

Artifacts:

- `/tmp/canary_mr20_summary.jsonl`
- `/tmp/canary_mr24_summary.jsonl`

### Single-Request Matrix (P95/P99 + host/device split + cache counters)

Config and key outputs:

- `mr20_ips24`
  - `1687.63 items/s`, latency `p99=0.2989s`
  - queue wait `p99=0.000831s`
  - host/device share `16.54% / 83.46%`
- `mr24_ips32`
  - `1837.31 items/s`, latency `p99=0.2763s`
  - queue wait `p99=0.000818s`
  - host/device share `16.86% / 83.14%`
- `mr24_ips48`
  - `1824.71 items/s`, latency `p99=0.2868s`
  - queue wait `p99=0.000835s`
  - host/device share `16.65% / 83.35%`
- `mr24_ips64`
  - `1850.12 items/s`, latency `p99=0.2855s`
  - queue wait `p99=0.000821s`
  - host/device share `17.20% / 82.80%`

Cache counters (new runtime counters) from `internal_state.scoring_cache_metrics`:

- all tested runs show:
  - `lookup_queries == lookup_hits`
  - `lookup_misses = 0`
  - `lookup_hit_rate = 1.0`
  - `handles_missing_node = 0`

Artifact:

- `/tmp/p0_single_matrix.jsonl`

### Concurrent Matrix (QPS + P95/P99 + host/device split + cache counters)

Shapes:

- high-throughput: `concurrency=4`, `items_per_request=200`
- control-path stress: `concurrency=12`, `items_per_request=20`

Highlights:

- `4x200` best:
  - `mr24_ips48`: `1749.70 items/s`, `qps=8.75`, request latency `p99=0.4646s`
- `12x20` best:
  - `mr20_ips24`: `798.76 items/s`, `qps=39.94`, request latency `p99=0.2992s`

Host/device split:

- `4x200`: host share about `15.45%` to `17.60%`
- `12x20`: host share about `15.98%` to `21.03%` (higher control-path pressure)

Fallback and cache counters:

- `score_from_cache_v2_metrics.fallback = 0` in all 6 runs.
- `scoring_cache_metrics.lookup_hit_rate = 1.0` in all 6 runs.

Artifact:

- `/tmp/p0_concurrency_matrix.jsonl`

## P1 Changes Implemented

### gc.freeze improvements

- Existing flag-gated freeze path retained:
  - `--enable-gc-freeze`
- Added rollback flag:
  - `--gc-freeze-rollback`
- Scheduler now logs:
  - `freeze_before`, `freeze_after`, and `gc_count`
- Files:
  - `python/sgl_jax/srt/server_args.py`
  - `python/sgl_jax/srt/managers/scheduler.py`
  - `test/srt/test_server_args_gc_freeze.py`

### Score-path communicator concurrency

- Already enabled in branch via correlated request-id matching for score fastpath RPC.
- Regression tests remain green for targeted cases.

### Runtime cache counters (vLLM-style query/hit/miss)

- Added explicit scoring-cache counters and internal-state export:
  - lookup queries/hits/misses (global + per path)
  - handle create/release counts
  - hit rate and active handles
- Files:
  - `python/sgl_jax/srt/managers/scheduler.py`
  - `python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py`
  - `test/srt/test_multi_item_prefill_extend_regression.py`

### Batch-integrity verification across scoring paths

- Added scheduler ingress counters/histograms and path classification:
  - `internal_state.ingress_metrics`
  - tokenizer vs rpc message totals
  - score-path message counters by mode
- Verified with a 3-mode path probe (`/tmp/p1_ingress_paths.log`):
  - `packed` path:
    - `tokenizer_multi_item_packed=4`
    - `tokenizer_cache_for_scoring=0`
    - `tokenizer_extend_from_cache=0`
  - `prefill_extend_baseline` path:
    - `tokenizer_cache_for_scoring=1`
    - `tokenizer_extend_from_cache=120`
    - batch histogram includes `gt_16` (expected fan-out behavior)
  - `fastpath_v2_label_only` path:
    - `tokenizer_cache_for_scoring=1`
    - `score_from_cache_v2_metrics.attempted=1`, `fallback=0`
- Notes:
  - In this runtime, fastpath-v2 request traffic appears on tokenizer ingress counters rather than rpc counters (`rpc_score_from_cache_v2=0`), which is now directly observable.

## Recommended Canary Contract

- Primary (throughput-focused): `mr24`, `ips=48`
- Stable tail alternative: `mr24`, `ips=32`
- Control-path heavy traffic alternative: `mr20`, `ips=24`

## Notes

- One stale TPU process from an earlier ad-hoc GC check was found and killed before final matrix runs.
- All benchmark matrices above were executed after cleanup on isolated scripts and captured to `/tmp/*.jsonl`.
