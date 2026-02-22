# Ranking Perf TODO (LinkedIn-Aligned Track)

## Scope

This plan intentionally stays close to the LinkedIn SGLang ranking post and focuses on four tracks only:

1. Batching path hardening.
2. Scoring-only execution path.
3. Prefix reuse/caching path.
4. Python runtime hardening.

## Out Of Scope (For Now)

- Sequence-classification reranker path.
- Sparse/sampled label-only objective changes.
- Dedicated new label-head kernel work.
- MixLM-style embedding token cache experiments.
- Large architecture changes beyond scheduler/runtime hardening.

## Baseline And Measurement

- Baseline branch: `feat/rpa-kernel-v11-hotshape` plus PR28 commits.
- Device/model baseline: TPU v6e-1 + `Qwen/Qwen3-0.6B`.
- Benchmark matrix must be captured for every change:
  - Single-request throughput (`items/s`).
  - Concurrent load: `QPS`, `P50/P95/P99`.
  - Host vs device split (`queue_wait_s`, `host_orchestration_s`, `device_compute_s`).
  - Cache counters (`lookup_queries`, `lookup_hits`, `lookup_misses`, fallback counts).

## Execution Plan

1. `P0` Stage 1: Batching Path (Tokenizer -> ZMQ -> Scheduler)
   - Status: `in progress`.
   - Tasks:
     - Ensure batch tokenization is used for all scoring ingress modes.
     - Preserve batch integrity across tokenizer send and scheduler receive.
     - Keep ingress counters and batch histogram enabled for every benchmark run.
   - Exit criteria:
     - No silent batch fragmentation in benchmark path.
     - Higher effective scheduler batch sizes at same load.

2. `P1` Stage 2: Scoring-Only Execution
   - Status: `in progress`.
   - Tasks:
     - Keep decode/sampling disabled for ranking runs.
     - Keep final-token scoring only.
     - Keep fastpath-v2 + label-only enabled in benchmark lane.
     - Track fallback rate as a first-class regression gate.
   - Exit criteria:
     - Fallback remains zero (or near-zero under canary load).
     - No correctness drift in score outputs.

3. `P1` Stage 3: Prefix Reuse/Caching
   - Status: `in progress`.
   - Tasks:
     - Keep prefill-once/extend-many scoring path healthy.
     - Keep cache query/hit/miss counters and handle lifecycle counters in reports.
     - Verify cache behavior under both single-request and concurrent canaries.
   - Exit criteria:
     - High cache hit ratio on expected shared-prefix workloads.
     - No cache-handle leak or missing-node growth under steady load.

4. `P1` Stage 4: Python Runtime Hardening
   - Status: `in progress`.
   - Tasks:
     - Keep `gc.freeze` flag-gated after warmup.
     - Keep rollback switch (`--gc-freeze-rollback`) for safe canary rollback.
     - Monitor freeze count and tail latency impact (`P95/P99`).
     - Continue `_Communicator` score-path concurrency with response correlation.
   - Exit criteria:
     - No throughput regression beyond noise.
     - Tail latency improved or unchanged.
     - Fast rollback path validated.

## Canary Policy

- Start with conservative canary config, then step up:
  - `max_running_requests=20`, then `24` only if stable.
  - Tune `multi_item_extend_batch_size` before changing other knobs.
- Promotion gates:
  - No sustained fallback increase.
  - No sustained queue-wait blowup.
  - `P99` does not regress beyond agreed threshold for that workload.

## Notes On P99 Target

- For very large single requests (for example 500-item rerank payloads), sub-100ms request-level `P99` is generally not realistic at current compute cost.
- Keep two latency targets:
  - Throughput lane (large batch payloads).
  - Interactive lane (smaller payloads with strict `P99` SLO).
