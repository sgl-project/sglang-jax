# GDN state-pool copy fix and TPU acceptance

Date: 2026-09-21. Issue: #1667.

## Result and scope

The no-tracking fused chunk-parallel adapter now passes the original state pools to the kernel, encodes resets in sequence metadata, and materializes saved dummy slices before pool donation. On the issue's Qwen3.5-27B configuration, full recurrent-layer-pool copies fell from 48 to zero. The original startup OOM was reproduced, and the candidate completed startup and four generation requests without reducing capacity.

- Baseline: `1056e49ae2abea7e69a403644e26bc362e1deadb`.
- Branch/workspace: `codex/fix-gdn-state-pool-copy`, `/Users/feibo/.codex/worktrees/aaa4/sglang-jax`.
- Validated adapter SHA256: `37a79364e1d615834318765bd165bc17de96ca878f3035379d0177707a541b2c`.
- Validation benchmark SHA256: `f95b241bdd07de256774a56f3ea7cf7816117fceb1d22773552f5eda9fb7c50c`.
- Bounded v7x-8 validation completed before committing or opening the PR.

## Why the fix needs a barrier

The old adapter gathers and masks active states, scatters a temporary kernel pool, then scatters results into the original pool. Keeping that original pool alive prevents the kernel's declared aliases from avoiding whole-pool copies.

The no-tracking path now:

1. Passes the incoming recurrent pool directly to the existing vendor kernel.
2. Uses `where(has_initial_state, seq_lens, query_lens)` to retain both initial-state gates. The kernel reads initial state only when total length exceeds query length.
3. Saves only dummy slot 0 and places both pools and both saved slices behind one `optimization_barrier`. All subsequent consumers use the barrier outputs.
4. Clears fresh, non-dummy, zero-length slots with dropped out-of-range scatter updates, because zero-length sequences have no kernel tiles.
5. Restores saved dummy slots, including dummy requests with positive token counts during precompile.

The first candidate without the barrier passed all numerical checks but still had two full-pool copies. Its real TPU HLO fused the saved dummy slice into the post-kernel restore. Consequently the restore still consumed the original full pool across the aliased kernel call. Buffer assignment confirmed a separate 404,226,048-byte recurrent pool at capacity 1024. The shared barrier prevents this fusion; the second experiment verified that the copies disappeared.

Tracking retains its original implementation, including when tracking arrays exist with all masks false. No vendor kernel, request sizing, route selection, or numerical tolerance changed.

## Runtime and original-capacity configuration

- Single host, v7x-8: four physical chips / eight JAX devices, topology 2x2x1.
- Actual mesh verified from synchronized forwards: data=4, tensor=2.
- Python 3.12.12, JAX/jaxlib 0.11.1, libtpu 0.0.46.1.
- Qwen3.5-27B BF16; CLI TP=8/DP=4; context/max sequence 2048; max-running-requests 1024; mem-fraction-static 0.9; max-prefill-tokens 16384; chunked-prefill-size 4096; page size 128; radix disabled; vision DP; two multimodal workers; seed 0; fused chunk-parallel GDN.
- Effective request limit was 768. Both variants reached the same first EXTEND shape: bs=768, tokens=1024.

## Full-model acceptance

| Check | Baseline | Candidate |
| --- | --- | --- |
| Original-capacity startup | OOM at first EXTEND | Passed |
| Full recurrent-layer-pool copies in target optimized HLO | 48, including `copy.6367` | 0 |
| Target module default-memory buffer assignment | 110,153,047,616 B (102.59 GiB) | 90,570,170,368 B (84.35 GiB) |
| Original-capacity generation | Server never ready | Four requests passed |

Both target modules were identified by parameter metadata, not just module number: `forward_batch[15]` is input_embedding BF16[1024,5120], and `forward_batch[2]` is seq_lens S32[768], as defined by `ForwardBatch.tree_flatten`. Their artifact filename is `module_0240.jit_jitted_run_model.cl_974389722.*`.

The candidate completed all five original-capacity EXTEND paddings, including 16384 tokens, and its DECODE precompilation. Startup readiness took 597.34 seconds. Each of four fixed synthetic-token requests returned 16 output IDs, completion_tokens=16, a length/16 stop, and 16 finite generated-token logprobs. The synchronized bs768/tokens 1024 completion and DP4/TP2 mesh gates passed. Instrumentation recorded no errors.

The maximum recorded per-device allocator `peak_bytes_in_use` across candidate startup and requests was **88.236526966 GiB**, below the 94.74 GiB device limit. This is a runtime allocator peak, distinct from the 84.35 GiB static report for one compiled shape. Independent device maxima are not summed.

For paired correctness, baseline and candidate both ran capacity 256 / fraction 0.8 with otherwise matching settings. All four requests passed token-count/stop/finite-logprob checks. All generated token IDs matched exactly, and every request's maximum absolute generated-token logprob difference was **0.0**. Fixed tolerances were atol=rtol=1e-5 and were not relaxed.

## Single-layer acceptance

Global GDN heads were 16/48 with dimensions 128/128 and convolution kernel 4. The donated-pool benchmark used global tokens 1024, DP4/TP2, 40 synchronized timing samples after three warmups, and three steps per scenario.

| Pool capacity / batch | Median baseline -> candidate | Executable temporary estimate baseline -> candidate | Full-pool copies |
| --- | --- | --- | --- |
| 256 / 192 | 1.716 -> 1.113 ms | 273,700,736 -> 95,221,280 B | 2 -> 0 |
| 512 / 384 | 2.594 -> 1.402 ms | 528,083,520 -> 153,484,576 B | 2 -> 0 |
| 1024 / 768 | 4.610 -> 2.318 ms | 1,048,023,328 -> 304,479,552 B | 2 -> 0 |

All 45 scenario/step comparisons passed exact finite equality for outputs and entire state pools: fresh, resumed, mixed, dummy, and empty requests. Independent checks confirmed unchanged dummy/inactive slots. Repeated timing trajectories also matched exactly. The benchmark now fails if candidate optimized HLO retains a full recurrent-pool copy; the gate was checked against the real failing first-candidate HLO.

Do not multiply single-layer savings by the layer count to infer full-model peak memory. Executable `memory_analysis()` estimates and dump buffer assignments are separate compiler views. A substantial difference between those views in the first candidate could not be fully explained; it was not attributed to a hidden reserve. At capacity 1024 the final candidate's default-memory dump was 720,657,984 B, versus 721,080,160 B for the executable's net argument/output/alias/temp estimate. The latter excludes generated code and is not a runtime peak.

## Tests and review

- CPU suite: **37 passed, 6 hardware skips**. The skipped tests subsequently ran on real TPU.
- Real TPU suite: **6 passed** on each candidate; final candidate took 24.67 seconds.
- CPU state checks cover initial-state gating, fresh/resumed empty requests, all-empty batches, slot reuse, positive/repeated dummy slots, eager nested donation, jitted outer donation, and no-track -> snapshot -> no-track transitions.
- The boundary regression rejects the baseline's pre-kernel pool modification. It compares contents rather than Python identity because eager barriers may return different array objects; real TPU HLO establishes physical copy elimination.
- Independent read-only review found no blocker and independently ran 18 state-contract tests.
- Fixed lint versions passed: isort 5.13.2, Ruff 0.13.3, Black 24.10.0, codespell 2.4.1. No unrelated formatting or vendor edits.

Reproduce CPU checks:

```bash
PYTHONPATH=python JAX_PLATFORMS=cpu \
  /Users/feibo/.codex/worktrees/0fa6/sglang-jax/.venv/bin/python -m pytest -q -rs \
  python/sgl_jax/test/test_gdn_fused_chunk_parallel_state_contract.py \
  python/sgl_jax/test/test_gdn_prefill_dispatch.py \
  python/sgl_jax/test/kernels/test_gdn_fused_chunk_parallel_provenance.py \
  python/sgl_jax/test/test_gdn_fused_chunk_parallel_prefill.py \
  python/sgl_jax/test/test_gdn_fused_chunk_parallel_prefill_dp.py
```

The real-TPU A/B driver is `benchmark/kernels/gdn/validate_state_pool_copy.py`; its CLI requires explicit model head counts and a pristine baseline adapter.

## Evidence and boundaries

| Experiment | ID | Outcome |
| --- | --- | --- |
| First single-layer candidate | `exp-yiki7gkvu3` | Numerical pass; copy-elimination gate failed |
| Final single-layer candidate | `exp-ko6xzkpkpc` | Numerical, timing-trajectory and copy gates passed |
| Four full-model variants | `exp-d674zu6pm9` | Original-capacity and paired correctness gates passed; experiment succeeded |

Falcon artifacts include exact source hashes, package manifests, runtime probes, raw responses, memory events, HLO dumps, buffer assignments, and server logs. Export analyses: `an-jlru701v6v`, `an-2if5n6ddmw`, `an-vkkdl9a9sc`. Full-model artifact: `art-cbbhpndmd0`.

Local copies and replay manifests are in `/tmp/gdn1667-validation/`; the remote full-model artifact also contains its harness. Key local files include `v2-results.json`, `candidate-original-summary.json`, `target-hlo-evidence.json`, `target-shape-evidence.json`, and both original-capacity target memory reports. Full export analysis succeeded. Its only missing files are the nine expected readiness/request/response files for the baseline that OOMed before serving; there were no parse errors. Downloaded raw paired responses were independently checked for exact token/logprob equality, valid 16-token length stops and finite logprobs; their export hashes matched. The generic HLO exporter could not match token semantics automatically; target-module attribution above was verified separately using input_embedding/seq_lens metadata.

All three TPU experiments succeeded as executions (the first candidate still failed the subsequently applied copy objective). The full-model experiment completed at 03:07:01 UTC. The cluster snapshot at 03:08:54 UTC showed its pod Succeeded, no running allocation from this task, and no active lease for any of the three experiments. No serving process or TPU experiment remains running.

Limitations:

- Full-model paired comparison covers generated token IDs and their logprobs, not full-vocabulary logits or language-quality evaluation.
- Requests were sequential synthetic-token inputs, not a sustained high-concurrency load test. High-batch startup and single-layer state trajectories were validated separately.
- Tracking correctness has the existing real-TPU unit coverage; its pool-copy cost was not changed. No speculative decoding or image-request end-to-end acceptance was performed.
- Runtime hooks synchronize forwards to record completion and allocator data. No full-model throughput claim is made.
- Server termination signals in logs after successful requests are expected runner cleanup, not generation failures.
