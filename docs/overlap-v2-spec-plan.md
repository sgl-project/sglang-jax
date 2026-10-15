# Unified overlap v2 for speculative decoding

## Baseline and scope

- Parent: PR #1715, `a3c0591fe0339a946cf4270ede44b7474635277d`.
- Branch: `codex/overlap-v2-spec`.
- With `SGLANG_JAX_OVERLAP_V2=1`, all implemented speculative algorithms use the v2 scheduler and its ordered executor: EAGLE, EAGLE3, NEXTN (including multi-layer MTP), DFLASH and DSPARK. No algorithm-specific fallback to the legacy scheduler.
- Preserve the legacy scheduler when the flag is disabled. PD remains outside this PR's scope. STANDALONE has no worker implementation in the parent commit; this change does not implement it.
- Preserve existing algorithm, attention-backend and request-feature constraints. In particular, this scheduler adaptation does not implement constrained speculative verification or change acceptance algorithms/kernels.

## Ownership and execution

“Single-threaded” refers to ownership, not literally one Python thread: the scheduler owns requests, allocation and batches; one FIFO executor owns device submission. JAX device execution remains asynchronous.

1. Remove the algorithm exclusion from v2 selection. Lift the legacy fused-shape restriction only for v2; keep backend constraints.
2. Split speculative execution into owner-side preparation, worker execution and owner-side publication. Snapshot mutable host arrays and sampling metadata; never enqueue a live ScheduleBatch or Req.
3. Submit the existing complete speculative step through the same executor as normal generation. Reuse fused NEXTN/EAGLE3 and DFLASH/DSPARK relay paths when applicable; otherwise execute the existing generic draft/verify/draft-extend path. Speculative verification already samples: never enqueue normal sampling afterward.
4. Return the output and next-round sequence metadata through a future. Publish per-rank draft state and lengths on the scheduler before preparing another batch. The future also provides the donation/error barrier.
5. Retire previous CPU output during current submission, deferring shared KV/cache release until the barrier. When sampling depends on mutable request state, retire the previous result before preparing the next snapshot, avoiding stale state and cyclic event waits.
6. Preserve DP padding, chunked prefill, EOS/abort, queue drain and failure propagation. Generic paths may still synchronize internally; moving them into v2 does not imply eliminating every device synchronization.

## Validation and delivery

- CPU tests: all algorithm dispatches, host snapshots, owner publication, FIFO/error propagation, no duplicate sampling, DP acceptance lengths, resource barriers and lifecycle behavior.
- Existing overlap-v2, relay and speculative scheduling regressions; formatting and diff checks.
- TPU follow-up: paired legacy/v2 serving runs from the same parent commit, runtime and weights. Qwen3-8B DFLASH and EAGLE3 GSM8K; MiMo-V2-Flash NEXTN AIME26; concurrency 32, DP2, greedy, plus short EOS/abort/reuse stress. EAGLE and DSPARK require their own model smoke runs. CPU tests are not model-level correctness or performance evidence.
- Keep plan, implementation and test evidence on this isolated branch. Do not push or open a PR unless requested.

## Implementation status — 2026-10-08

Implemented on the branch above. Enable with `SGLANG_JAX_OVERLAP_V2=1` and leave overlap scheduling enabled. Normal generation and all five implemented spec algorithms select the same v2 event loop and executor. The adapter also handles generic tree-token packing, DP-padded length compaction, and prevents generic configurations from entering relay-only precompilation.

Validation environment: Python 3.12, JAX 0.11.1, CPU. The existing normal-v2/relay baseline passed 24 tests before modification.

```bash
JAX_PLATFORMS=cpu PYTHONPATH=python .venv/bin/python -m pytest -q -rs \
  python/sgl_jax/test/test_tp_worker_overlap_v2.py \
  python/sgl_jax/test/test_speculative_overlap_v2.py \
  python/sgl_jax/test/test_tp_worker_overlap_thread.py \
  python/sgl_jax/test/test_relay_buffer.py \
  python/sgl_jax/test/test_scheduler_chunked_ownership.py \
  python/sgl_jax/test/test_spec_accept_metrics.py \
  python/sgl_jax/test/speculative/test_spec_dp_shapes.py \
  python/sgl_jax/test/speculative/test_spec_info.py \
  python/sgl_jax/test/speculative/test_dflash_worker.py \
  python/sgl_jax/test/speculative/test_dflash_info.py \
  python/sgl_jax/test/speculative/test_dflash_server_args.py \
  python/sgl_jax/test/speculative/test_draft_extend_fused.py
# 184 passed, 11 skipped, 2 subtests passed

JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  PYTHONPATH=python .venv/bin/python -m pytest -q \
  python/sgl_jax/test/speculative/test_draft_extend_fused.py
# 18 passed, including all 11 cases skipped on the single-device run
```

The new test file contains 43 cases and is registered in the CPU test suite. Black, isort, Ruff and `git diff --check` pass for the changed files. Serving/model forwards in adapter tests are mocked; the eight-device run uses simulated CPU devices, not TPUs. No TPU serving, accuracy, acceptance-length or performance run has been completed for this branch yet.

## DFLASH TPU validation — 2026-10-08

Reused the original Qwen3-8B DFLASH/GSM8K recipe: all 1319 samples, greedy, thinking off, concurrency 32, TP8/DP2, output limit 8192 on v7x-8. Actual scheduler state and worker class confirmed overlap v2.

- First attempt `exp-kp55j4bh1m` exposed a real-type mismatch: per-rank `SamplingBatchInfo` has no `grammars` field. The retirement check now reads grammar state from `Req`. Added three tests using the real sampling class; 69 focused tests passed. The new test file now has 46 cases.
- Successful rerun: `exp-3v6ogt0t87`, artifact `art-lem179vzyk`, export analysis `an-b3s6i42g2e`.
- Executed base `a3c0591fe0339a946cf4270ede44b7474635277d` plus patch SHA256 `20b42b987c4e211bd61d152b4b4f24d161c56b9a15438b418979a349b1fc2ce2`. Patch and per-file hashes were archived and validated before serving; subsequent documentation changes are not part of that executed snapshot.
- Result: 1231/1319, reported accuracy 93.33%, acc_len 368621/58958 = 6.2523, two output-limit truncations, observed concurrency 32. All 1319 effective request hashes match the historical baseline.
- Historical main `61b304ce8a09` result was 1232/1319, 93.40%, acc_len 6.1196, zero truncations. The accuracy/truncation thresholds are not met against that reference. Different base commits and minor dependency drift prevent attribution specifically to v2; a same-source legacy/v2 pair is still needed. The two truncated responses repeat answer text until 8192 tokens.
- This is model-level validation of DFLASH only. Other speculative algorithms and performance still require their own serving validation.

## EAGLE3 TPU validation — 2026-10-08

- Reused the historical Qwen3-8B EAGLE3/GSM8K recipe: 1319 samples, greedy, thinking off, concurrency 32, TP8/DP2, 8192 output tokens, v7x-8. Draft checkpoint conversion verified all 15 tensors bitwise.
- Experiment `exp-xeql7nhi0t` succeeded; artifact `art-zq2us29hy5`; export analysis `an-5651spk4u2`. Actual scheduler flag and `ModelWorkerOverlap` class confirmed v2.
- Executed base `a3c0591fe0339a946cf4270ede44b7474635277d` plus patch SHA256 `0ef5a7d6cd0fff04fcb96a5a9712ffe7a364beceda2432e871bff54b45da363e`. Runtime files match the successful DFlash snapshot; the patch hash differs because of documentation. This section was appended after that snapshot.
- Result: 1231/1319, reported accuracy 93.33%, acc_len 353455/128042 = 2.760461, zero truncations, peak concurrency 32. All 1319 effective request hashes match the historical baseline.
- Historical main reference `exp-vjydpr64iz`: 1229/1319, 93.18%, acc_len 2.758263, zero truncations. All three historical thresholds pass. This validates this EAGLE3 serving configuration; different base commits and minor dependency drift prevent a v2-only causal conclusion.

## Multi-layer NEXTN/MTP TPU validation — 2026-10-08

- Reused MiMo-V2-Flash/AIME26 128K recipe: all 30 samples, greedy, thinking enabled, concurrency limit 32, TP8/DP2/EP8, v7x-8, output cap 131072 and context 139264. Validated three MTP weight layers before serving.
- Experiment `exp-fxixx0yh6s` succeeded; artifact `art-pfvnl6ixw8`; export analysis `an-3jf2kh8ahg`. Actual scheduler flag and `ModelWorkerOverlap` class confirmed v2. Executed the same base and patch snapshot as the EAGLE3 experiment above.
- Result: 27/30, 90.00% accuracy, acc_len 972209/282042 = 3.447036, three truncations, peak concurrency 30 (dataset has 30 questions). All 30 effective input hashes match the historical reference.
- Historical main `exp-l7a74d7bwa`: 23/30, 76.67%, acc_len 3.632717, eight truncations. Accuracy and truncation gates pass; acc_len declined 5.11%, marginally failing the 5% threshold. Fewer long/truncated generations change the aggregate output distribution, so this does not isolate scheduler effects. Different base commits and minor runtime drift also prevent v2-only attribution.
- DFLASH, EAGLE3 and multi-layer NEXTN now have completed model-level v2 runs. Functional completion and historical regression thresholds are separate: EAGLE3 passes all three; DFLASH fails accuracy/truncation; MiMo fails the acceptance-length threshold. Other configurations and algorithms still require model-level coverage.
