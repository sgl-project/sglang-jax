# Spec Overlap Split Verify/Draft-Extend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split greedy spec decode into an early verify/sample result and a later draft_extend result so scheduler batching can start after real `accept_lens/new_seq_lens` is available instead of waiting for the full fused spec decode.

**Architecture:** Keep the existing server args and fused greedy route flag behavior. Replace the single full `spec_decode()` boundary with two explicit phases: phase A is the scheduler batching result (`accept_lens`, accepted token ids, verify-derived `new_seq_lens`, finish/logprob-visible fields); phase B is the next spec forward state (`next_draft_input.hidden_states/topk/verified_id`, draft KV updates, FutureMap stash). Scheduler may update `ScheduleBatch.reqs_info[*].seq_lens`, output ids, filtering, merging, and next host-side metadata from phase A, but no component may launch a new JAX program out of multi-host order while phase B is still running.

**Tech Stack:** Python, JAX/XLA JIT, TPU, `sgl_jax.srt.speculative`, `ScheduleBatch`, `GenerationBatchResult`, pytest, TPU pod benchmark/profiling.

---

## 2026-06-04 Integration Notes From Current Branch

This plan is integrated with the current work on:

- Branch: `dev/fused-greedy-device-chain-verify-inputs-codex`
- Last pushed WIP: `f04d363e wip: add spec overlap compatibility path`
- Stable restored server run: `20260604_112538_specdecode_clean_restored_full_fused`
- Latest steady-state c32 profile: `profiles/profile_20260604_114632_c32_steady5`
- Profile archive sha256: `5947d739d89b06c1e058f72a257c65dc556e05be9b209839bd1d216945fb205d`

Current accepted baseline:

- Full fused spec result path is restored and stable on the 4-rank pod.
- 32/32 long greedy HTTP requests returned 200 in the latest profiling run.
- Error scan was clean after restoring the full-fused path.
- The branch already removed the server-args hard gate for spec decode + overlap and did not add a new flag.
- FutureMap/req-pool relay and DP-padded `new_seq_lens` plumbing exist, but they are not yet performance-complete.

Rejected implementation that this plan must not repeat:

- A previous split attempt published phase A early and then let scheduler launch another JAX program while worker-side draft_extend was still running.
- On TPU multi-host this failed with `jax.errors.JaxRuntimeError: FAILED_PRECONDITION: The program continuator has halted unexpectedly` around `process_allgather()`.
- Root cause: ranks no longer executed the same JAX program sequence. This is a correctness bug, not just a performance issue.
- Therefore, phase A may unblock CPU-side batching, but next JAX launch must still be owned by one ordered launch component and must wait for phase B readiness at the launch boundary.

Updated boundary definition:

- Phase A result = scheduler batching result.
- Phase B result = next spec forward state.
- Phase A must enter the scheduler queue as early as possible after verify/sample.
- Phase B must not block scheduler CPU work, but it must become a dependency before the next actual model/draft JAX launch.
- Scheduler-visible phase A must not expose or require `hidden_states`, `topk_p`, `topk_index`, or selected target hidden/logits. Those fields are private worker/draft_extend state until phase B completes.

Current profile evidence to optimize against:

- `queue.py:154 get`: about 483 ms total in the 5-step profile window.
- `wait_last_spec_future_state`: about 389 ms total.
- `_materialize_fused_greedy_batch_output_for_scheduler`: about 188 ms total.
- `SpecFutureMap.resolve/stash/resolve_seq_lens`: about 37-41 ms total each.
- Many small `gather`/`scatter`/masking ops still appear between decode batches. These are partly current FutureMap/materialization relay work and partly the full-result wait. They must shrink or move off the critical batch-to-batch path before claiming overlap success.

Latest profile update after the split Phase A implementation:

- Profile: `profile_20260604_170103_spec_overlap_profile_after_decode32_decode5_after_decode32`.
- The previous cache-miss-scale bubble is gone; `jit_fused_greedy_verify` is reusing the compiled module.
- Remaining steady-state gap between TPU0 `jit_fused_greedy_verify` launches is still about `18-19 ms`.
- The largest remaining scheduler-side blocker is now:
  - `_wait_pending_spec_draft_extend_before_launch`
  - `wait_pending_spec_draft_extend_before_launch`
  - `resolve_last_spec_draft_extend_result`
  - `queue.get`
- This wait costs about `8.6-9.6 ms` per steady-state decode step.
- Phase B `forward_batch_speculative_draft_extend_phase` takes about `12-13 ms` wall time, but its TPU `jit_fused_draft_extend` kernel is only about `1.3 ms`; the rest is host-side Phase B prep/materialization/relay.
- Root cause of the remaining bubble: the current implementation waits for a complete Python `SpecDraftExtendPhaseResult` object before the next launch. This is too coarse. The next implementation step must publish/relay Phase B state by `req_pool_idx` as soon as scheduler can own the dependency, and next `run_batch` should resolve only the current batch's needed draft state handles.
- Correct target: after Phase A, scheduler builds all host batching state immediately. Before the next launch, scheduler uses the current batch's `req_pool_idx` layout to resolve/relay `next_draft_input` from a FutureMap/device-handle map. CPU should not wait for device computation if a valid `jax.Array` handle/dependency can be threaded into the next JIT input.
- This FutureMap/req-pool relay is in the original plan but is not complete yet; the current wait boundary is a compatibility checkpoint, not the final overlap implementation.

2026-06-04 attempted optimization and rejection:

- Attempted to keep Phase B `next_draft_input.hidden_states/topk_index/verified_id` as device arrays and perform scheduler-side device `take/scatter` during req-pool relay.
- CPU contract tests passed, but 4-rank TPU canary rejected this approach:
  - First attempt failed with `ValueError: is_fully_addressable is not implemented for jax.sharding.AbstractMesh` when constructing an explicit replicated selector sharding from a device array's abstract mesh.
  - After removing explicit selector sharding, the server reached the second 32-request decode batch but rank1 hit `jax.errors.JaxRuntimeError: INTERNAL: Core halted unexpectedly`.
- Conclusion: scheduler-side ad hoc device scatter/gather is not a safe FutureMap implementation for this multi-host path. The next design must either:
  - keep relay host-materialized until a proper worker-owned ordered device FutureMap exists, or
  - move device gather/scatter into the ordered worker/JIT/precompile path so all ranks execute the same JAX program sequence.
- The unsafe device-scatter experiment was reverted. Do not reintroduce scheduler-created device scatter/gather for Phase B relay without an explicit ordered-launch design and a 4-rank TPU canary.

2026-06-04 worker-owned PhaseB relay update:

- Implemented a worker-owned pending PhaseB relay in the single ordered worker thread:
  - scheduler no longer calls `_wait_pending_spec_draft_extend_before_launch` before every launch;
  - `ModelWorkerClient` stores the completed `SpecDraftExtendPhaseResult`;
  - before the next decode work runs on the worker thread, it applies the pending PhaseB `next_draft_input` to `model_worker_batch.spec_info_padded` by `req_pool_idx`.
- CPU contract coverage:
  - `test_worker_applies_pending_phase_b_state_to_next_decode_batch_by_req_pool`
  - `test_worker_keeps_pending_phase_b_state_for_non_decode_batch`
  - `test_draft_extend_allocate_lens_accepts_dp_padded_layout`
  - `test_filter_batch_releases_dropped_unfinished_spec_decode_req`
  - Latest focused CPU run on pod after the release/drop fix: `15 passed in 7.50s`.
- TPU canary `20260604_190326_worker_owned_phaseb_relay_popen` reached steady decode, but the request set was polluted by stale clients and ran at `#running-req: 64`, so it is not accepted as the 32-batch benchmark. It exposed a real release/accounting issue near request finish.
- Root cause found: split verify and draft-extend metadata were mixing two layouts for `allocate_lens`.
  - Scheduler-visible `SpecVerifyPhaseResult.allocate_lens` must be DP-padded so `_publish_spec_verify_phase_lengths_to_schedule_batch()` can slice by `dp_rank * per_dp_bs`.
  - PhaseA must also refresh scheduler-side `info.spec_info.allocate_lens`; otherwise abort/filter can release against stale allocation frontiers.
  - DRAFT_EXTEND attention metadata must accept either compact `real_bs` allocate-lens or DP-padded slot-layout allocate-lens and select by `logits_indices_selector`; otherwise warmup can fail with `AssertionError: (32, 4)`.
- Current fix:
  - `_publish_spec_verify_phase_lengths_to_batch()` and `_publish_spec_verify_phase_lengths_to_schedule_batch()` update only scheduler-visible lengths/frontiers: `seq_lens` and `spec_info.allocate_lens`. They still do not publish PhaseB private hidden/topk/verified state.
  - `FlashAttention._select_draft_extend_allocate_lens()` maps DP-padded allocate-lens to real request order via `logits_indices_selector` for DRAFT_EXTEND metadata.
  - `ScheduleBatch.filter_batch()` now releases spec KV for any dropped req that still owns `req_pool_idx`, not only `finished()` reqs, covering disconnect/abort/drop paths.
- Latest TPU canary:
  - Run id: `20260604_193739_drop_release_fix`.
  - Warmup completed without `ModelWorkerClient` exception or scheduler exception.
  - Clean 32 concurrent greedy requests reached steady `#running-req: 32`.
  - Observed steady tail before stopping: accept-len about `2.19-2.84`, accept-ratio about `0.55-0.71`, gen throughput about `1778-2311 token/s`.
  - The run still ended in `Prefill out of memory` because the local stop command was issued after the request set had already reached high SWA usage; no `memory leak detected` appeared before that OOM in the scanned logs.
  - Next retry must record the decode-line count before sending requests and stop after exactly the first new 5-8 `#running-req: 32` batches to avoid tailing older lines or running into SWA exhaustion.

Execution override:

- Preserve the original stepwise structure below, but use this section as the source of truth when it conflicts with older text.
- Do not add a new spec-overlap flag.
- Do not change `_build_fused_draft_extend_jit()` signature or precompile-visible arguments unless the task also updates every call site and precompile path, with a TPU import/precompile canary.
- Do not implement background phase B by starting an independent thread that can launch JAX programs concurrently with scheduler/worker launches.
- KV cache frontier changes are out of scope for this split plan unless a later task adds a separate invariant and rollback plan.

2026-06-04 PhaseB second-layer profiling update:

- Current priority is to ignore the allocator leak temporarily and continue removing the remaining batch-to-batch bubble. The leak is tracked as a separate final cleanup item because the captured profiles already show the overlap bottleneck before the leak/OOM tail.
- Predispatch-only PhaseB reduced steady TPU0 `jit_fused_greedy_verify` gap from the earlier `18-19 ms` range to about `12.0-12.3 ms`, proving that issuing PhaseB before PhaseA materialization is useful.
- A follow-up async-thread dispatch experiment was rejected:
  - CPU tests passed, but TPU profile `20260604_213830_phaseb_async_dispatch_profile` showed steady verify gaps of about `14.7-16.2 ms`, worse than predispatch-only.
  - Trace showed `prepare_for_extend_after_verify` overlapped verify/D2H, but `get_eagle_forward_metadata` and draft_extend dispatch still landed after verify, with `resolve_async_spec_draft_extend_phase` waiting about `6-8 ms`.
  - The experiment also violates the plan's ordered-launch guidance, so it has been reverted.
- New second-layer target:
  - keep the worker-owned ordered launch path;
  - do not add scheduler-side ad hoc device scatter/gather;
  - reduce the PhaseB tail by splitting, caching, or removing duplicated DRAFT_EXTEND metadata/device-put work.
- First concrete optimization:
  - `_dispatch_draft_extend_for_decode_fused()` no longer recomputes `get_eagle_forward_metadata(mwb)` after `EagleDraftInput.prepare_for_extend_after_verify()` has already installed the same metadata on the draft model runner.
  - Acceptance for this micro-step: CPU focused tests pass; TPU 5-step profile shows the steady verify gap improves from the predispatch-only `~12 ms` baseline or at minimum does not regress, with no JAX runtime ordering failure.

2026-06-04 metadata-dedupe validation:

- Code change:
  - Reverted the rejected async-thread PhaseB experiment.
  - Kept ordered worker-owned predispatch.
  - Removed the duplicate DRAFT_EXTEND `get_eagle_forward_metadata()` call in `_dispatch_draft_extend_for_decode_fused()`.
- CPU validation:
  - Pod rank0 focused run: `sgl_jax/test/speculative/test_spec_overlap_split.py -q`
  - Result: `14 passed in 8.04s`.
- TPU validation:
  - Run id: `20260604_220011_phaseb_metadata_dedupe_profile`.
  - Profile: `/tmp/profile_20260604_220011_phaseb_metadata_dedupe_profile/decode5_after_decode32`.
  - Steady TPU0 `jit_fused_greedy_verify` gaps after the first profiled step:
    - `13.05 ms` for the first captured post-trigger transition.
    - then `11.68 ms`, `11.76 ms`, `11.57 ms`.
  - This is a small improvement over the predispatch-only `~12.0-12.3 ms` baseline, and much better than the rejected async-thread `~14.7-16.2 ms` profile.
  - No JAX launch-order runtime failure was observed before profile completion.
  - The server later hit `Prefill out of memory` after the test clients continued into high SWA usage; treat this as the known test-control/SWA tail issue, not as the overlap bottleneck.
- Remaining bubble cause from this profile:
  - `predispatch_spec_draft_extend_phase` now overlaps most PhaseB host prep with verify.
  - The remaining steady gap is dominated by the serial tail after verify:
    - actual `jit_fused_draft_extend` kernel (`~1.5-1.7 ms`);
    - `materialize_predispatched_spec_draft_extend_phase` (`~1.7-2.0 ms`);
    - `run_batch` and next verify metadata/dispatch prep.
  - To shrink this further, the next step cannot be another host-thread trick; it needs a safe worker-owned relay of PhaseB device handles or a JIT/ordered-launch design that removes host materialization from the launch-critical path.

2026-06-04 rejected device-fast-path follow-up:

- Experiment:
  - Added a worker-owned steady-state fast path that returned PhaseB device handles (`hidden_states/topk/verified_id`) and applied them directly when the next batch's `req_pool_indices` exactly matched the previous PhaseB order.
  - Also tried disabling host prefetch in this fast path so it would not enqueue unnecessary D2H copies.
- Validation:
  - CPU contract tests passed in both variants.
  - TPU run `20260604_221633_phaseb_device_fastpath_profile`:
    - no runtime error/leak/OOM during profile;
    - but first 32-request transition had a new `~25s` stall;
    - steady TPU0 verify gaps regressed to about `13.4-14.7 ms`.
  - TPU run `20260604_222537_phaseb_device_fastpath_noprefetch_profile`:
    - no runtime error/leak/OOM during profile;
    - same new `~25s` first-transition stall;
    - steady TPU0 verify gaps were about `12.6-12.8 ms`, still worse than the `~11.6-11.8 ms` metadata-dedupe baseline.
- Decision:
  - Reverted the device-fast-path implementation and its test.
  - Keep the metadata-dedupe predispatch baseline as the current accepted implementation.
  - Do not reintroduce direct PhaseB device-handle relay unless the design also removes the extra first-transition compile/stall and proves a better steady gap on 4-rank TPU.
- Updated next focus:
  - Analyze the next-verify prep path after PhaseB (`run_batch`, `_get_spec_decode_mwb_dp`, `_scatter_spec_info_to_dp_slots`, `get_eagle_forward_metadata`, and `PjitFunction(fused_greedy_verify)`), because removing PhaseB host materialization alone did not remove the bubble.
  - Any new attempt must show steady verify gap below the metadata-dedupe baseline and must not introduce the 25s first-transition stall.

2026-06-04 preserve-device ForwardBatch micro-step:

- Experiment:
  - In `_dispatch_draft_extend_for_decode_fused()`, replaced the DRAFT_EXTEND `ForwardBatch.init_new(mwb, mr0)` call with the existing `_forward_batch_init_new_preserve_device(mwb, mr0)` helper.
  - This keeps the ordered worker launch path and does not add scheduler-side device scatter/gather.
- CPU validation:
  - Local `python3 -m py_compile python/sgl_jax/srt/speculative/draft_extend_fused.py` passed.
  - Pod rank0 focused CPU test after syncing code: `sgl_jax/test/speculative/test_spec_overlap_split.py -q`.
  - Result: `14 passed in 7.92s`.
- TPU validation:
  - Run id: `preserve_device_fb_20260604_230259`.
  - Profile: `/tmp/profile_preserve_device_fb_20260604_230259/decode5_after_decode32`.
  - Trigger: 32 concurrent greedy requests, profile started after the second new `Decode batch. #running-req: 32`.
  - TPU0 steady `jit_fused_greedy_verify` gaps:
    - `11.52 ms`, `11.36 ms`, `11.53 ms` after the first captured transition.
  - This is only a very small improvement over metadata-dedupe (`~11.57-11.76 ms`) and is not sufficient to claim bubble elimination.
  - Remaining gap still contains:
    - `jit_fused_draft_extend` device work around `1.3-1.5 ms`;
    - `materialize_predispatched_spec_draft_extend_phase` around `1.6-1.9 ms`;
    - next verify `get_eagle_forward_metadata` around `1.0 ms`;
    - separate `jit_subtract`/`jit_clip` small programs around `1.0 ms` combined;
    - multiple `device_array`/`batched_device_put` spans.
- Error note:
  - After profile completion and test-client termination, rank0/rank2/rank3 hit `allocator.free_group_end -> allocator.free` `AssertionError`.
  - This is tracked with the already-known KV allocator leak/free accounting cleanup, not counted as evidence that the overlap bubble optimization failed.
  - It still must be fixed before final correctness acceptance.
- Next micro-step:
  - Move `sel_pos = clip(accept_lens - 1)` into the ordered fused draft_extend JIT so the standalone `jit_subtract` and `jit_clip` programs are removed from the batch-to-batch path.
  - This changes only the cached fused draft_extend call boundary in `draft_extend_fused.py`; it must be validated with CPU tests and a 4-rank TPU profile because it changes the JIT signature.

2026-06-04 sel_pos-in-JIT validation:

- Code change:
  - `_build_fused_draft_extend_jit()` now receives `accept_lens` and computes `sel_pos = clip(accept_lens - 1)` inside the ordered fused draft_extend JIT.
  - `_dispatch_draft_extend_for_decode_fused()` no longer launches standalone `jit_subtract`/`jit_clip` before `jit_fused_draft_extend`.
- CPU validation:
  - Local py_compile passed.
  - Pod rank0 focused CPU test after syncing: `sgl_jax/test/speculative/test_spec_overlap_split.py -q`.
  - Result: `14 passed in 8.00s`.
- TPU validation:
  - Run id: `selpos_in_jit_guarded_20260604_232635`.
  - Profile: `/tmp/profile_selpos_in_jit_guarded_20260604_232635/decode5_after_decode32`.
  - TPU0 steady `jit_fused_greedy_verify` gaps:
    - `11.08 ms`, `10.87 ms`, `11.15 ms` after the first captured transition.
  - The previous standalone `jit_subtract`/`jit_clip` entries disappeared from the gap summary.
  - This is a real but still small improvement; the bubble is not eliminated yet.
- Remaining evidence:
  - Gap is now dominated by `device_array`/`batched_device_put`, `jit_fused_draft_extend`, PhaseB materialization, and next verify metadata/JIT dispatch.
  - Post-profile client termination again hit the known allocator free assertion; record it under leak/free cleanup, not as overlap failure.

2026-06-04 rejected preserve-device logits metadata experiment:

- Experiment:
  - Added a default-off `preserve_device` path to `LogitsMetadata.from_model_worker_batch()` and enabled it only for fused draft_extend.
- CPU validation:
  - Pod rank0 focused CPU test passed: `14 passed in 8.00s`.
- TPU rejection:
  - Run id: `preserve_logits_meta_20260604_233632`.
  - During `SPEC_DECODE` precompile, rank1 failed inside full `spec_decode -> spec_decode_draft_extend_phase -> _materialize_draft_extend_for_decode_fused()`.
  - First real stack:
    - `np.asarray(dispatch.selected_layer0_hidden)`
    - `jax.errors.JaxRuntimeError: FAILED_PRECONDITION: The program continuator has halted unexpectedly.`
  - Other ranks were then terminated by JAX coordination service after task1 stopped heartbeating.
- Decision:
  - Rejected and reverted this experiment.
  - Do not reintroduce preserve-device `LogitsMetadata` for fused draft_extend without a new ordered-launch/materialization design.

2026-06-04 skip-hidden PhaseB materialization experiment:

- Experiment:
  - Limited to overlap split/predispatch path.
  - Non-overlap/full `spec_decode()` continues to materialize PhaseB `hidden_states`, preserving the default/full-result contract.
  - Predispatched PhaseB no longer `copy_to_host_async()`s or `np.asarray()`s `selected_layer0_hidden`; it writes `next_draft_input.hidden_states = None`.
  - Rationale: topk=1 fused verify reads previous `verified_id` and `topk_index`; previous `hidden_states` is not consumed by `_prepare_topk1_verify_placeholders_from_draft_state()`.
- CPU validation:
  - Local py_compile passed.
  - Pod rank0 focused CPU test after syncing: `14 passed in 7.95s`.
- TPU validation:
  - Run id: `skip_hidden_phaseb_20260604_235046`.
  - Profile: `/tmp/profile_skip_hidden_phaseb_20260604_235046/decode5_after_decode32`.
  - TPU0 steady `jit_fused_greedy_verify` gaps:
    - `10.96 ms`, `10.63 ms`, `10.78 ms` after the first captured transition.
  - Improvement over `sel_pos_in_jit_guarded` is small but measurable.
  - Post-profile allocator assertion recurred and remains tracked as leak/free cleanup.

2026-06-04 PhaseA host relay to PhaseB materialization:

- Experiment:
  - For predispatched PhaseB, attach PhaseA host `accept_lens` and host selected `verified_id` to `FusedDraftExtendDispatch` after verify D2H completes.
  - PhaseB materialization uses these host values instead of materializing `accept_lens_device` and `verified_id_arr` again.
- CPU validation:
  - Local py_compile passed.
  - Pod rank0 focused CPU test after syncing: `14 passed in 8.04s`.
- TPU validation:
  - Run id: `phasea_host_relay_20260604_235935`.
  - Profile: `/tmp/profile_phasea_host_relay_20260604_235935/decode5_after_decode32`.
  - TPU0 steady `jit_fused_greedy_verify` gaps:
    - `10.83 ms`, `10.54 ms`, `10.72 ms` after the first captured transition.
  - This is another small improvement only; materialization still costs about `1.4-1.6 ms`, now primarily due to `topk_index_stacked` host materialization plus Python writeback.
- Current conclusion:
  - Micro-optimizations have reduced steady gap from metadata-dedupe `~11.6-11.8 ms` to `~10.5-10.8 ms`.
  - This is not complete scheduler-overlap parity. The remaining gap requires a safer worker-owned device relay for next-verify `topk_index`/`verified_id`, or a deeper ordered JIT design that avoids host materialization at the PhaseB boundary.

2026-06-05 topk padded relay validation:

- Experiment:
  - For split-overlap predispatched PhaseB only, skipped `topk_index_stacked` D2H/materialization.
  - `SpecDraftExtendPhaseResult` now carries optional DP-padded `padded_next_draft_input.topk_index` and `padded_req_pool_indices`.
  - The ordered worker thread applies the padded `topk_index` handle directly when the next `ModelWorkerBatch.req_pool_indices` layout exactly matches; layout changes fall back to row materialization by `req_pool_idx`.
- CPU validation:
  - Pod rank0 focused CPU test: `sgl_jax/test/speculative/test_spec_overlap_split.py -q`.
  - Result after the accepted topk padded relay tests: `16 passed in 7.95s`.
- TPU validation:
  - Run id: `topk_padded_relay_20260605_002213`.
  - Profile: `/tmp/profile_topk_padded_relay_20260605_002213/decode5_after_decode32`.
  - The fast path did fire: `apply_padded_phase_b_topk_fastpath` appears in the trace.
  - `materialize_predispatched_spec_draft_extend_phase` dropped from about `1.4-1.6 ms` to about `0.03 ms`.
  - Steady TPU0 `jit_fused_greedy_verify` gaps were about `11.59 ms`, `11.70 ms`, `11.57 ms` after the first transition.
- Decision:
  - This removes a real materialization cost but does not by itself eliminate the batch-to-batch bubble or improve the total steady gap versus the best `phasea_host_relay` profile (`~10.5-10.8 ms`).
  - Keep it only as a narrow building block for later worker-owned relay work; do not claim it as a performance-complete acceptance step.
  - The remaining visible tail is now dominated by next verify prep/dispatch (`get_eagle_forward_metadata`, `device_array`/`batched_device_put`, and `PjitFunction(fused_greedy_verify)`) plus the true `fused_draft_extend` dependency.

2026-06-05 rejected pre-enqueue verify metadata experiment:

- Hypothesis:
  - Non-spec overlap prepares `forward_metadata` and `ForwardBatch` before enqueueing work to the worker thread.
  - Moving TARGET_VERIFY metadata/ForwardBatch prep into `ModelWorkerClient.forward_batch_speculative_generation()` might hide `get_eagle_forward_metadata` and device_put work before the next ordered verify launch.
- Experiment:
  - Added a prepared verify context and applied pending PhaseB state into that context.
  - CPU focused tests passed (`17 passed in 7.94s`), but TPU profile rejected the approach.
- TPU validation:
  - Run id: `verify_prep_enqueue_20260605_003719`.
  - Profile: `/tmp/profile_verify_prep_enqueue_20260605_003719/decode5_after_decode32`.
  - Steady TPU0 `jit_fused_greedy_verify` gaps regressed to about `13.45 ms`, then `12.10-12.00 ms`.
  - `run_batch` grew from sub-millisecond to about `6.1-6.2 ms`, so the device_put-heavy metadata prep was not hidden; it moved into scheduler critical path.
- Decision:
  - Rejected and reverted the pre-enqueue verify metadata/context changes.
  - Do not retry this exact direction unless PhaseA publication and scheduler batching are also reorganized so that `run_batch` work is provably overlapped with useful device work.
  - The allocator/free assertion remains tracked separately as a final cleanup item and was not used to judge this overlap experiment.
- Updated remaining-root-cause interpretation:
  - Non-spec overlap can build `forward_metadata`/`ForwardBatch` in `run_batch` before processing the previous result because the next decode length is deterministic `+1`.
  - Spec overlap cannot do that with the current host contract: the scheduler must first receive PhaseA `accept_lens`, update `seq_lens`, filter/merge/admit, and only then build the next `ModelWorkerBatch`.
  - Therefore moving next-verify metadata prep into scheduler `run_batch` does not hide it; it moves more work into the spec critical path.
  - The next viable directions are:
    - reorganize PhaseA publication vs PhaseB launch so PhaseB covers scheduler `run_batch` and next verify prep instead of completing before `run_batch`; or
    - keep more of `new_seq_lens`/attention metadata on device so the next verify launch does not need a host metadata rebuild after PhaseA.
  - Any attempt in either direction must preserve the single ordered JAX launch owner across all ranks.

2026-06-05 mock TPU overlap harness:

- Purpose:
  - Added a fast discrete-event mock at `python/sgl_jax/bench_spec_overlap_mock.py`.
  - Added CPU contract coverage at `python/sgl_jax/test/speculative/test_spec_overlap_mock_tpu.py`.
  - This harness does not load weights, does not start server, and does not model target-model math. It models scheduler/worker/TPU dependency topology so we can iterate quickly on whether the fake TPU lane goes idle between fused verify launches.
- Mock strategies:
  - `current`: reproduces the accepted profile shape after the latest micro-optimizations. It models verify, PhaseA-to-draft dispatch, required draft_extend TPU work, PhaseB materialization, `run_batch`, next verify metadata/device_put, and next verify dispatch as launch-critical.
  - `target_future_relay`: models the intended future-relay topology. The worker/scheduler enqueue draft_extend and next-verify metadata using future/device dependency handles before the current verify completes, so the TPU lane can run `verify -> draft_extend -> verify` without idle time. The verify-to-verify gap still contains the necessary draft_extend kernel.
- Pod CPU validation:
  - Command: `PYTHONPATH=/tmp/sglang-jax/python JAX_PLATFORMS=cpu /opt/venv/bin/uv run --active pytest sgl_jax/test/speculative/test_spec_overlap_mock_tpu.py -q`.
  - Result: `3 passed in 0.13s`.
- Mock CLI validation:
  - Command: `python -m sgl_jax.bench_spec_overlap_mock --strategy both --steps 5 --trace /tmp/spec_overlap_mock_trace.json`.
  - `current`: `verify_to_verify_gaps_ms=[10.83, 10.83, 10.83, 10.83]`, `verify_to_verify_idle_ms=[9.23, 9.23, 9.23, 9.23]`.
  - `target_future_relay`: `verify_to_verify_gaps_ms=[1.6, 1.6, 1.6, 1.6]`, `verify_to_verify_idle_ms=[0.0, 0.0, 0.0, 0.0]`.
  - Trace files on pod: `/tmp/spec_overlap_mock_trace_current.json` and `/tmp/spec_overlap_mock_trace_target_future_relay.json`.
- Interpretation:
  - The fast mock can validate proposed scheduler/worker dependency changes in milliseconds and prevent repeating known-bad layouts before a 4-rank TPU run.
  - It cannot replace real TPU acceptance. It does not prove JAX cache behavior, multihost program order, PJRT dispatch overhead, real attention metadata layout, or real KV allocator behavior.
  - Final acceptance still requires 4-rank TPU profile showing idle bubble removal, throughput improvement, no acceptance regression for identical deterministic prompts, and later GSM8K non-regression.
- Next implementation direction:
  - Use this mock as the quick design target for a worker-owned future-relay path: next verify inputs should be represented as device/future dependencies where possible, and CPU scheduler batching should not sit on the TPU launch-critical path.
  - Keep the true draft_extend TPU kernel in the gap accounting as useful work, not idle bubble.
  - Do not use this mock to justify scheduler-side ad hoc device scatter/gather or independent background JAX launch threads.

2026-06-05 TPU profile mock validation:

- Purpose:
  - Added a real TPU/JAX mock harness at `python/sgl_jax/bench_spec_overlap_tpu_mock.py`.
  - Added smoke coverage at `python/sgl_jax/test/speculative/test_spec_overlap_tpu_mock.py`.
  - The harness initializes 4-process JAX distributed on the TPU pod, runs synthetic `mock_verify_kernel` and `mock_draft_extend_kernel`, and captures xprof trace without loading model weights or starting the HTTP server.
- Profile method:
  - Run all four pods with `--nnodes 4`, per-pod `--node-rank`, and a fresh `--dist-init-addr`.
  - Only JAX `process_index=0` writes trace; in these runs that was `perf-16-1`.
  - Parse TPU0 `jit_mock_verify_kernel` start/end times and count non-verify TPU work between adjacent verify launches as useful work. Remaining gap is idle bubble.
- Current-path profile:
  - Run id: `tpu_mock_current_longverify_20260605_012737`.
  - Profile: `/tmp/profile_tpu_mock_current_longverify_20260605_012737` on `perf-16-1`.
  - Shape: `global_batch=1024`, `width=1024`, `verify_loops=2048`, `draft_loops=128`, `phase_a_ms=2`, `scheduler_ms=4`, `metadata_ms=2`.
  - TPU0 verify kernel duration: about `7.40 ms`.
  - Steady verify-to-verify gaps after the first profiled transition: about `9.94-10.01 ms`.
  - Steady idle bubble after subtracting draft/metadata TPU work: about `8.61-8.68 ms`.
  - Conclusion: the profile mock can reproduce the same kind of batch-to-batch idle bubble we see in real spec-overlap traces.
- Target future-relay profile:
  - Run id: `tpu_mock_target_longverify_20260605_012618`.
  - Profile: `/tmp/profile_tpu_mock_target_longverify_20260605_012618` on `perf-16-1`.
  - Same synthetic shape and loop counts as current-path profile.
  - TPU0 verify kernel duration: about `7.34 ms`.
  - Excluding the first initial-pipeline/profile transition, steady verify-to-verify gaps: about `0.478-0.482 ms`.
  - Steady idle bubble: `0.0 ms`.
  - The remaining gap is the synthetic draft_extend kernel, which is useful device work, not scheduler idle.
- Interpretation:
  - This is the right quick iteration loop for overlap topology changes: it runs in seconds, uses TPU/PJRT/xprof, and directly shows whether CPU work is on the TPU launch-critical path.
  - It still does not prove real model correctness, real attention metadata shape handling, real KV allocator behavior, or throughput. Any production change that passes this mock must still be validated with the 4-rank server profile and 32 concurrent greedy requests.

2026-06-05 fresh TPU mock profile after stopping e2e server:

- Context:
  - The e2e server was stopped so the pod could be used only for a synthetic 4-rank TPU/JAX mock.
  - Synced `bench_spec_overlap_mock.py`, `bench_spec_overlap_tpu_mock.py`, and the focused mock tests to all four pods.
  - Pod CPU smoke passed: `sgl_jax/test/speculative/test_spec_overlap_mock_tpu.py sgl_jax/test/speculative/test_spec_overlap_tpu_mock.py -q` -> `6 passed in 1.18s`.
- Current topology mock:
  - Run id: `tpu_mock_current_20260605_061259`.
  - Profile writer: `perf-16-1`, path `/tmp/profile_tpu_mock_current_20260605_061259/plugins/profile/2026_06_04_22_13_13/perf-16-1.trace.json.gz`.
  - Shape: `global_batch=1024`, `width=1024`, `verify_loops=2048`, `draft_loops=128`, `phase_a_ms=2`, `scheduler_ms=4`, `metadata_ms=2`.
  - TPU0 verify duration: about `7.39 ms`.
  - Excluding the first pipeline/profile transition, steady verify-to-verify gap: about `9.87-9.98 ms`.
  - Useful TPU work inside the gap: about `0.47 ms` (`jit_mock_draft_extend_kernel` plus tiny metadata op).
  - Steady idle bubble: about `9.40-9.51 ms`.
  - Conclusion: the mock profile reproduces the same failure shape as the real server: CPU-side scheduling/metadata work is on the launch-critical path.
- Target future-relay mock:
  - Run id: `tpu_mock_target_20260605_061556`.
  - Profile writer: `perf-16-1`, path `/tmp/profile_tpu_mock_target_20260605_061556/plugins/profile/2026_06_04_22_16_09/perf-16-1.trace.json.gz`.
  - Same synthetic shape and loop counts.
  - TPU0 verify duration: about `7.32-7.48 ms`.
  - One outlier transition had a `26 ms` idle gap; treat it as profile/host scheduling noise because every later transition is stable.
  - Steady verify-to-verify gap after the outlier: about `0.474-0.480 ms`.
  - Useful TPU work inside the gap: about `0.465-0.471 ms` (`jit_mock_draft_extend_kernel`).
  - Steady idle bubble: about `0.007-0.009 ms`.
  - Conclusion: TPU/PJRT can sustain the desired `verify -> draft_extend -> verify` chain when CPU scheduler work is represented as non-critical future/dependency work.
- Updated implementation implication:
  - The remaining real-server gap is not a TPU runtime limitation and not only a PhaseB materialization issue.
  - The true missing piece is moving next-verify launch-critical construction (`out_cache_loc`, `seq_lens`, metadata/device_put, and pjit enqueue) out from behind scheduler PhaseA processing, or reducing it to fit inside the real draft_extend kernel time.
  - In the real code this likely requires a same-batch/device-chain fast path with strict fallbacks:
    - allowed only when the next decode keeps the same padded `req_pool_indices` layout;
    - no newly admitted requests, no finished/retracted requests, no grammar/logprob/streaming constraint that changes launch-visible metadata;
    - enough KV reserve slack so next `out_cache_loc` can be derived without host allocator work;
    - scheduler still consumes PhaseA for correctness, but the worker can keep the ordered JAX chain moving with device/future handles.
  - This is a topology change. More metadata-cache micro-steps are unlikely to remove the last idle bubble.

2026-06-05 deferred decode logging micro-step:

- Purpose:
  - `--decode-log-interval=1` is required for the current profiling workflow, but `log_decode_stats()` appeared inside the batch-to-batch critical path.
  - This is not the main `5.2 ms` bubble, but it is launch-noncritical work and should not block the next verify enqueue.
- Code change:
  - Added `_defer_decode_stats_log()` and `_flush_deferred_decode_stats_logs()` to `SchedulerOutputProcessorMixin`.
  - For spec + overlap decode, `process_batch_result_decode()` now still completes finish/filter/output-id updates and `token_to_kv_pool_allocator.free_group_end()` before the next batch, but only queues decode stats logging.
  - `event_loop_overlap()` flushes deferred decode stats after the current `run_batch()` has launched/enqueued the next batch, and also flushes on idle.
  - Non-spec and non-overlap logging behavior remains immediate.
- Tests:
  - Added source-order coverage that `free_group_end()` remains before `_defer_decode_stats_log(batch)`.
  - Added mock topology contract `same_batch_device_chain`, representing the desired worker-owned steady-state chain where scheduler PhaseA catch-up no longer gates the TPU lane.
  - Pod CPU focused run:
    - Command: `PYTHONPATH=/tmp/sglang-jax/python JAX_PLATFORMS=cpu /opt/venv/bin/uv run --active pytest sgl_jax/test/speculative/test_spec_overlap_split.py sgl_jax/test/speculative/test_spec_overlap_mock_tpu.py sgl_jax/test/speculative/test_spec_overlap_tpu_mock.py -q`
    - Result: `39 passed in 8.96s`.
- Interpretation:
  - This should remove only a small critical-path cost, especially visible when profiling with `--decode-log-interval=1`.
  - It does not solve the remaining topology problem. The large remaining work is still a safe same-layout/device-chain path that can launch next verify with ordered worker-owned device dependencies while scheduler PhaseA bookkeeping catches up.
  - Safety constraint for that next step: do not chain past finish/filter unless there is an explicit rollback/discard mechanism for extra KV writes; otherwise EOS/max-token/abort cases can over-generate.

2026-06-05 same-batch chain safety guard:

- Purpose:
  - The mock target proves that a worker-owned same-batch/device-chain topology can remove TPU idle.
  - The real server cannot blindly launch the next verify before scheduler catch-up, because the previous verify may have produced EOS, hit max tokens, encountered stop strings, grammar termination, abort/retract, or changed the running layout through filter/merge/admit.
  - Without rollback/discard for extra KV writes, the chain fast path must be guarded conservatively.
- Code change:
  - Added `Scheduler._can_chain_same_batch_spec_decode(batch, previous_padded_req_pool_indices)`.
  - This guard currently only defines the safe boundary; it does not enable chain launch yet.
  - It requires:
    - spec + overlap decode mode;
    - exact same padded `req_pool_indices` layout;
    - no logprob/hidden/grammar launch-visible features;
    - no finished/retracted requests;
    - per-rank spec frontier has `verify_write_lens + speculative_num_draft_tokens <= allocate_lens`, so the chain attempt cannot need a new allocator call;
    - `ignore_eos=True`, no stop token ids, no stop strings, no tokenizer EOS checks, no explicit `eos_token_ids`;
    - enough remaining `max_new_tokens` for a full draft-token step.
- Tests:
  - Added `test_same_batch_device_chain_guard_requires_no_finish_risk`.
  - Added `test_same_batch_device_chain_guard_requires_same_padded_layout`.
  - Added `test_same_batch_device_chain_guard_requires_reserved_kv_slack`.
  - Pod CPU focused run after this and deferred logging:
    - Command: `PYTHONPATH=/tmp/sglang-jax/python JAX_PLATFORMS=cpu /opt/venv/bin/uv run --active pytest sgl_jax/test/speculative/test_spec_overlap_split.py sgl_jax/test/speculative/test_spec_overlap_mock_tpu.py sgl_jax/test/speculative/test_spec_overlap_tpu_mock.py -q`
    - Result: `42 passed in 8.98s`.
- Interpretation:
  - This guard is too conservative for normal chat/eval requests because they usually allow EOS. That is expected.
  - To use same-batch chain for normal requests and fully eliminate the bubble, the next implementation must add a rollback/discard path for one extra chained verify when PhaseA later says a request finished or the layout changed.
  - Requiring reserve slack makes that rollback/discard tractable: a speculative chained verify may write into already-reserved KV slots, but it must not allocate new slots before scheduler catch-up confirms the result can be committed.

2026-06-05 no-allocation KV preview helper for future same-batch chain:

- Purpose:
  - A worker-owned chained verify needs the next `out_cache_loc` without running scheduler-side `prepare_for_decode()` or mutating scheduler state.
  - It must be able to fail closed when a chain would require a new allocator call.
- Code change:
  - Added `EagleDraftInput.peek_reserved_decode_out_cache_loc(schedule_batch)`.
  - It returns `(out_cache_loc_chunks, new_verify_write_lens)` only when all ranks have enough `allocate_lens` reserve slack.
  - It reads from `req_to_token_pool.req_to_token` exactly like the no-allocation branch of `prepare_for_decode()`.
  - It does not call allocator functions and does not mutate `info.out_cache_loc`, `info.seq_lens_sum`, `allocate_lens`, or `verify_write_lens`.
- Tests:
  - Added `test_spec_peek_reserved_decode_out_cache_loc_has_no_allocator_or_mutation`.
  - Added `test_spec_peek_reserved_decode_out_cache_loc_rejects_missing_slack`.
  - Pod CPU focused run:
    - Command: `PYTHONPATH=/tmp/sglang-jax/python JAX_PLATFORMS=cpu /opt/venv/bin/uv run --active pytest sgl_jax/test/speculative/test_spec_overlap_split.py sgl_jax/test/speculative/test_spec_overlap_mock_tpu.py sgl_jax/test/speculative/test_spec_overlap_tpu_mock.py -q`
    - Result: `45 passed in 8.98s`.
- Interpretation:
  - This is not yet the chained launch. It is the missing no-mutation primitive needed for a safe commit/discard implementation.
  - Next step: use this preview to build a candidate next `ModelWorkerBatch` in the worker-owned path, tag it as discardable, and only commit it if scheduler catch-up confirms the same layout remains valid.

2026-06-05 scheduler-to-worker same-batch chain preview plumbing:

- Purpose:
  - Before worker can issue a same-layout chained verify, scheduler must pass the launch-safe preview to `ModelWorkerBatch`.
  - This must be data-only plumbing first; worker must not consume it until commit/discard state is implemented.
- Code change:
  - Added `Scheduler._attach_same_batch_spec_chain_preview(batch, model_worker_batch)`.
  - When `_can_chain_same_batch_spec_decode()` passes and `peek_reserved_decode_out_cache_loc()` returns a preview, it attaches:
    - `allow_same_batch_spec_chain=True`;
    - `same_batch_chain_out_cache_loc_chunks`;
    - `same_batch_chain_verify_write_lens`;
    - `same_batch_chain_req_pool_indices`.
  - When guard or preview fails, these fields are explicitly set to disabled/`None`.
  - Hooked this helper into the spec overlap `run_batch()` path after `model_worker_batch` construction and before worker enqueue.
- Tests:
  - Added `test_scheduler_attaches_same_batch_chain_preview_when_guard_allows`.
  - Added `test_scheduler_disables_same_batch_chain_preview_when_guard_rejects`.
  - Pod CPU focused run:
    - Command: `PYTHONPATH=/tmp/sglang-jax/python JAX_PLATFORMS=cpu /opt/venv/bin/uv run --active pytest sgl_jax/test/speculative/test_spec_overlap_split.py sgl_jax/test/speculative/test_spec_overlap_mock_tpu.py sgl_jax/test/speculative/test_spec_overlap_tpu_mock.py -q`
    - Result: `47 passed in 9.02s`.
- Interpretation:
  - Real path now carries the exact data worker needs for a future same-batch chain attempt.
  - It still does not consume the preview or launch chained verify, so it is not expected to remove the remaining bubble yet.

2026-06-05 worker same-batch chain candidate construction:

- Purpose:
  - After scheduler attaches a no-allocation chain preview, worker needs to reconstruct the next-round padded draft state from the pending PhaseB device handles.
  - This must be done without reusing the current `model_worker_batch.spec_info_padded`, because the verify enqueue path mutates it from `EagleDraftInput` into `EagleVerifyInput`.
- Code change:
  - Added `ModelWorkerClient._build_same_batch_spec_chain_candidate_batch(model_worker_batch, pending)`.
  - It only builds a candidate batch; it does not enqueue a JAX program and does not commit scheduler state.
  - It requires exact padded req-pool layout match between scheduler preview and pending PhaseB.
  - It copies pending padded topk/verified/new_seq_lens handles and attaches scheduler-previewed `allocate_lens`, `verify_write_lens`, and concatenated `out_cache_loc`.
  - The candidate disables recursive same-batch chaining by default.
- Tests:
  - Added `test_worker_builds_same_batch_chain_candidate_from_preview`.
  - Added `test_worker_same_batch_chain_candidate_rejects_layout_mismatch`.
  - Pod CPU focused run:
    - Command: `PYTHONPATH=/tmp/sglang-jax/python JAX_PLATFORMS=cpu /opt/venv/bin/uv run --active pytest sgl_jax/test/speculative/test_spec_overlap_split.py sgl_jax/test/speculative/test_spec_overlap_mock_tpu.py sgl_jax/test/speculative/test_spec_overlap_tpu_mock.py -q`
    - Result: `49 passed in 8.95s`.
- Interpretation:
  - The plumbing now reaches the worker with enough data to form a discardable next verify candidate.
  - The remaining step that should actually reduce bubble is to enqueue this candidate on the same ordered worker thread immediately after current PhaseB dispatch, then expose a commit/discard protocol to scheduler.

2026-06-05 target-verify metadata cache validation:

- Experiment:
  - Added a one-entry per-`FlashAttention` TARGET_VERIFY metadata cache for static device fields.
  - Cache is keyed by the host content of `cu_q_lens`, `cu_kv_lens`, `page_indices`, `distribution`, and optional SWA page indices.
  - Exact `seq_lens` is still device-put every step; custom-mask TARGET_VERIFY disables the cache because mask repacking depends on exact sequence lengths.
  - The cache is stored in a module-level `id(self)` map with `weakref.finalize` cleanup, because `FlashAttention` is an NNX pytree and cannot hold array-valued static attributes.
- CPU validation:
  - Added `test_target_verify_metadata_reuses_static_device_fields_within_page`.
  - Pod rank0 focused run: `sgl_jax/test/speculative/test_spec_overlap_split.py sgl_jax/test/speculative/test_spec_overlap_mock_tpu.py sgl_jax/test/speculative/test_spec_overlap_tpu_mock.py -q`.
  - Result: `23 passed in 8.50s`.
- TPU validation:
  - Run id: `metadata_cache_20260605_013830`.
  - Profile: `/tmp/profile_metadata_cache_20260605_013830/decode5_after_decode32`.
  - Request shape: 32 concurrent chat-completions, `temperature=0`, `max_tokens=256`, profile started after the first new `Decode batch. #running-req: 32`.
  - Steady TPU0 `jit_fused_greedy_verify` gaps after the first profiled transition:
    - `10.43 ms`, `10.62 ms`, `10.37 ms`.
  - `get_eagle_forward_metadata` in the gap dropped to about `0.35-0.40 ms`, compared with about `1.1 ms` before this cache.
  - No Traceback, scheduler exception, memory leak detection, or OOM was found in the four rank logs after this run.
- Decision:
  - Keep the cache: it is a real, narrow improvement and did not change scheduler ordering.
  - This does not eliminate the bubble. The remaining gap still has about `6+ ms` of TPU idle after `jit_fused_draft_extend` before next `jit_fused_greedy_verify`.
  - Remaining root cause remains the production topology: next verify `ForwardBatch`/device input preparation happens only after scheduler PhaseA tracking and worker PhaseB handoff. Complete bubble removal still requires a same-batch/device-chain style path where safe next-verify inputs are prepared or dispatched as device dependencies, while CPU PhaseA output/accounting catches up.

2026-06-05 spec KV reserve frontier micro-step:

- Root-cause refinement from the latest profile:
  - After the TARGET_VERIFY/DRAFT_EXTEND metadata caches, a measurable part of the remaining gap is still in scheduler-side `EagleDraftInput.prepare_for_decode`.
  - The old spec decode path used `allocate_lens` for two jobs at once:
    - the KV reserve frontier used for releasing over-allocated slots;
    - the verify write frontier used to decide which `out_cache_loc` entries this decode step must pass to the next target verify.
  - Because these frontiers were conflated, spec decode had to allocate a small number of KV slots on every decode step, even when a paged allocator could safely reserve ahead.
- Code change:
  - Added `EagleDraftInput.verify_write_lens` as a host-only frontier.
  - `prepare_for_decode()` now rounds the reserve frontier by `page_size` for paged allocators, but advances `verify_write_lens` only to the exact per-step required verify frontier.
  - If reserve slack already exists, the function skips `alloc_paged_token_slots_extend()` and reads this step's `out_cache_loc` from `req_to_token_pool` using `verify_write_lens:required_write_lens`.
  - `filter_batch`, `merge_batch`, `ScheduleBatch` scatter/split/concat, and scheduler PhaseB req-pool writeback preserve `verify_write_lens`.
- CPU validation:
  - Added `test_spec_prepare_for_decode_uses_reserved_slots_without_allocator`.
  - Pod rank0 focused run: `sgl_jax/test/speculative/test_spec_overlap_split.py -q`.
  - Result: `29 passed in 8.06s`.
- Expected TPU effect:
  - This should reduce per-step allocator/assign spans inside `schedule_batch.prepare_for_decode` when all requests stay within the same page reserve.
  - It does not remove the deeper topology issue by itself: next verify host prep and PJIT enqueue can still sit on the launch-critical path.
  - The next acceptance check must be a 4-rank 5-step profile, comparing steady `jit_fused_greedy_verify` gaps against the latest `~10.4-10.6 ms` metadata-cache baseline.
- TPU validation:
  - Run id: `20260605_051514_kv_reserve_frontier`.
  - Profile: `/tmp/profile_20260605_051514_kv_reserve_frontier/decode5_after_decode32`.
  - Trigger: 32 concurrent chat-completions, `temperature=0`, `max_tokens=256`, profile started after the first new `Decode batch. #running-req: 32`.
  - TPU0 `jit_fused_greedy_verify` gaps:
    - first captured transition: `5.90 ms`;
    - steady transitions: `5.18 ms`, `5.22 ms`, `5.29 ms`.
  - Rank0 decode logs during the profile showed steady acceptance around `accept-len 1.00-1.06`, `accept-ratio 0.25-0.27`, and steady throughput around `1127-1233 token/s`.
  - Four-rank error scan found no Traceback, RuntimeError, scheduler exception, memory leak detection, OOM, FAILED_PRECONDITION, or core halt.
- Decision:
  - Keep the reserve/write-frontier split. It roughly halves the remaining steady verify-to-verify gap compared with the metadata-cache baseline.
  - This is still not complete bubble elimination. The remaining steady gap is about `5.2 ms`; trace evidence shows it is now dominated by next verify `device_array`/`batched_device_put`, `get_eagle_forward_metadata`, `PjitFunction(fused_greedy_verify)`, and the useful draft_extend dependency.
- Remaining-gap breakdown from the same trace:
  - `jit_fused_draft_extend`: about `1.31 ms` of useful TPU work immediately after verify.
  - PhaseA/D2H and scheduler output processing tail: runs across the end of verify and still extends to about `+2.1 ms` after the previous verify kernel ends.
  - `schedule_batch.prepare_for_decode`: down to about `0.36 ms`; `EagleDraftInput.prepare_for_decode` down to about `0.18 ms`.
  - next verify enqueue: about `3.17 ms`, including `_prepare_topk1_verify_placeholders_from_draft_state`/`padding_for_decode`, `get_eagle_forward_metadata`, `device_array`/`batched_device_put`, `_forward_batch_init_new_preserve_device`, and `PjitFunction(fused_greedy_verify)`.
- Updated next focus:
  - Do not spend more time on per-step allocator first; it is no longer the dominant cost.
  - The next meaningful reduction must either:
    - reduce PhaseA materialization/D2H tail for scheduler-required fields; or
    - move/collapse next verify input construction so device `seq_lens`, `out_cache_loc`, metadata, and PJIT dispatch are not all serialized after scheduler batching.

2026-06-05 device seq_lens relay micro-step:

- Experiment:
  - In the exact padded PhaseB relay path, carry the verify JIT's DP-padded `new_seq_lens` device handle into the next worker batch.
  - `_prepare_topk1_verify_placeholders_from_draft_state()` records this as `model_worker_batch.target_verify_seq_lens_device`.
  - `_forward_batch_init_new_preserve_device()` uses this handle for `ForwardBatch.seq_lens`, while scheduler host `seq_lens` remains authoritative for batching/filter/merge.
- CPU validation:
  - Added:
    - `test_prepare_topk1_relays_device_new_seq_lens_for_next_verify`
    - `test_forward_batch_init_uses_relayed_device_seq_lens`
  - Focused pod CPU suite after this accepted micro-step: `31 passed in 8.98s`.
- TPU validation:
  - Run id: `20260605_053518_seq_lens_device_relay`.
  - TPU0 steady verify gaps were `5.24 ms`, `5.31 ms`, `5.22 ms`, essentially unchanged from the KV reserve-frontier accepted baseline.
  - Four-rank error scan was clean.
- Decision:
  - Keep as a narrow device-relay building block because it did not regress and may help a later full device-chain design.
  - Do not count it as meaningful bubble elimination by itself.

2026-06-05 rejected metadata seq_lens relay:

- Experiment:
  - Also tried using the relayed device `new_seq_lens` to build TARGET_VERIFY metadata `seq_lens` as a device dependency instead of device-putting host metadata seq_lens.
- CPU validation:
  - Added a RED test, implemented the behavior, and the focused CPU suite passed.
- TPU rejection:
  - Run id: `20260605_054454_metadata_seq_lens_device_relay`.
  - Steady gaps regressed to `6.17 ms`, `6.39 ms`, `6.19 ms`.
  - `get_eagle_forward_metadata` became faster, but the added device dependency/dispatch placement moved work onto the critical path and worsened verify-to-verify gap.
- Decision:
  - Reverted this metadata relay and removed its test.
  - Keep the previous accepted state; do not reintroduce metadata seq_lens device relay unless it is folded into a larger ordered JIT/device-chain design that profiles better.

## Remaining Work Integrated From Current Branch

Already done and should be preserved:

- Server args incompatibility gate for spec decode + overlap has been removed without adding a new flag.
- FutureMap readiness and req-pool-index relay exist for spec metadata.
- Scheduler/ScheduleBatch preserve `future_indices`, `new_seq_lens`, and DP-padded spec layout.
- Worker async full-spec queue path exists and can return a placeholder before resolving the last full spec result.
- Fused/non-fused paths snapshot pre-verify seq_lens to avoid verify-side mutation aliasing.
- `EagleDraftInput.filter_batch()` and `merge_batch()` handle `new_seq_lens`.
- `ScheduleBatch.copy()` preserves spec metadata and process-result fields.
- Profiling annotations around spec resolve/materialization are already useful and should be kept.

Still unfinished and covered by this plan:

- True split result publish: phase A must be queued immediately after verify/sample, before draft_extend.
- Scheduler CPU batching from phase A only: update `seq_lens`, accepted tokens, output ids, finish checks, filter/retract, merge waiting, and prepare next host metadata without reading phase-B hidden/topk fields.
- Ordered launch-owner pipeline: phase B may overlap scheduler CPU work, but the next JAX launch must wait for phase B and preserve identical launch order across ranks.
- Reduce or move FutureMap/materialization small ops off the batch-to-batch critical path.
- Replace the coarse `_wait_pending_spec_draft_extend_before_launch -> resolve_last_spec_draft_extend_result -> queue.get` boundary with req-pool-index Phase B relay. The scheduler should only wait for missing Python handles, not for already-dispatched device work to finish.
- Verify the final profile removes the current about-30ms inter-batch bubble rather than only moving it to a later wait.
- Final TPU validation: clear throughput improvement, no acceptance regression for identical deterministic prompts, and no GSM8K regression at final milestone.

Explicitly not accepted yet:

- Any result that only restores correctness but keeps the same `queue.get(full spec result)` bubble.
- Any implementation that passes CPU tests but fails a 4-rank TPU canary.
- Any implementation that improves throughput by changing acceptance behavior or using non-greedy request settings.

## Current Code Map

- Modify: `python/sgl_jax/srt/speculative/draft_extend_fused.py`
  - Current single fused greedy route lives in `spec_decode()`.
  - Current fused JIT builder is `_build_fused_greedy_decode_jit()`.
  - Current materialization helper is `_materialize_fused_greedy_batch_output_for_scheduler()`.
  - Existing draft_extend-only path also has `_build_fused_draft_extend_jit()` and `draft_extend_for_decode_fused()`.

- Modify: `python/sgl_jax/srt/speculative/base_worker.py`
  - Current entry point `BaseSpecWorker.forward_batch_speculative_generation()` chooses the fused route for all-greedy decode.
  - This file should expose a phased entry point without changing non-greedy and prefill behavior.

- Modify: `python/sgl_jax/srt/managers/scheduler.py`
  - Current spec branch in `run_batch()` calls `forward_batch_speculative_generation()`, then immediately consumes full `batch_output.next_draft_input`.
  - This must be changed so the scheduler can process phase A result first, while phase B may still be pending.

- Modify: `python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py`
  - `_resolve_spec_decode_token_ids()` already consumes `result.next_token_ids` and `result.accept_lens`.
  - Keep this contract stable; phase A must return these fields in the same DP-padded layout.

- Modify: `python/sgl_jax/srt/speculative/eagle_util.py`
  - `EagleDraftInput` already carries `new_seq_lens`, `allocate_lens`, `hidden_states`, `verified_id`, `topk_p`, `topk_index`, and draft tree fields.
  - Add a narrow phase-A carrier only if `GenerationBatchResult` becomes too overloaded.

- Test: `python/sgl_jax/test/speculative/test_spec_overlap_split.py`
  - New CPU-level tests for shape/layout contracts and scheduler-visible phase ordering.

- Test/Benchmark on pod:
  - TPU 4-rank serving with `--disable-overlap-schedule` removed when testing overlap.
  - 32 concurrent greedy curl requests with long `max_tokens`.
  - Collect `Decode batch` throughput/acceptance and 5-step profile around steady-state decode.

---

## Required Invariants

1. No new server flag. Reuse the existing spec decode and overlap behavior.
2. Default behavior remains unchanged when overlap is disabled or when the fused greedy route is not applicable.
3. `accept_lens` must stay DP-padded with shape `(per_dp_bs * dp_size,)`.
4. `next_token_ids` must stay flattened by DP-padded request slot with stride `speculative_num_draft_tokens`.
5. Scheduler must update host `info.seq_lens` from real `accept_lens`, never fake max accept length.
6. `next_draft_input` must remain per-real-request/global-flat when written back into `batch.reqs_info[*].spec_info`.
7. DP attention metadata must continue to use our DP-padded layout, not SGLang's DP layout.
8. The first functional target is topk=1 all-greedy NEXTN because this is the current fused route.
9. Scheduler CPU batching may proceed after phase A, but scheduler must not directly launch a new JAX program while phase B is still executing.
10. All TPU JAX launches in the speculative path must go through one ordered launch owner so all ranks execute the same program sequence.
11. The next launch boundary must wait for phase B readiness and surface phase-B exceptions before launching.
12. Phase A scheduler-visible data is limited to fields needed for batching/output processing. `hidden_states`, `topk_p`, `topk_index`, selected target hidden/logits, and draft KV updates belong to private phase-B state.
13. Existing FutureMap behavior for non-spec token ring must remain compatible; if spec uses req-pool relay, the plan must not force non-spec paths to change in the same task.
14. Existing `_build_fused_draft_extend_jit()` precompile behavior must remain stable unless a task explicitly covers all affected signatures and precompile canaries.

---

## Acceptance Metrics

Each implementation task that changes runtime behavior must record:

- Throughput: steady-state `gen throughput (token/s)` from 32 concurrent long greedy requests.
- Acceptance: steady-state `accept-len` and `accept-ratio` from rank0 `Decode batch` logs.
- Correctness: CPU tests for changed contracts; GSM8K eval only at major milestones or before declaring the branch complete.
- Profiling: for overlap milestones, capture 5 steady-state decode steps and verify the old batch-to-batch bubble shrinks.

Expected final performance direction:

- Throughput must improve clearly versus current fused-full-result overlap path.
- Acceptance ratio must not regress for identical deterministic requests. For more varied prompts, compare on distribution-level averages instead of exact per-batch equality.
- GSM8K score must not regress at final validation.
- The previous ~30ms inter-batch bubble should be largely covered by phase-B draft_extend or reduced to the real dependency needed before next launch. Because removing a 30ms steady-state bubble should be a large win, final throughput should show a clear increase, not just parity.
- After the PhaseB FutureMap relay task, the profile must no longer show a steady `8-10 ms` `wait_pending_spec_draft_extend_before_launch` block inside every `run_batch`. Any remaining gap must be attributed to real launch/metadata work with trace evidence.

Current baseline to record against:

- Stable current-branch full-fused run: `20260604_112538_specdecode_clean_restored_full_fused`.
- Profile run: `20260604_114632_c32_steady5`.
- Profile archive: `/Users/niu/code/sglang-jax/.worktrees/fused-greedy-device-chain-verify-inputs-codex/profiles/profile_20260604_114632_c32_steady5.tar.gz`.
- Main trace: `profiles/profile_20260604_114632_c32_steady5/decode/plugins/profile/2026_06_04_03_46_43/perf-16-0.trace.json.gz`.
- Known issue in this baseline: scheduler waits for full spec result, and batch-to-batch profile contains many small materialization/FutureMap ops plus a large queue wait.

---

### Task 1: Add Phase Result Contracts and CPU Shape Tests

**Files:**
- Modify: `python/sgl_jax/srt/managers/scheduler.py`
- Create: `python/sgl_jax/test/speculative/test_spec_overlap_split.py`

- [ ] **Step 1: Add explicit result dataclasses near `GenerationBatchResult`**

Add this near `GenerationBatchResult` in `python/sgl_jax/srt/managers/scheduler.py`:

```python
@dataclasses.dataclass
class SpecVerifyPhaseResult:
    """Scheduler-visible output from greedy spec verify/sample.

    Layout:
    - accept_lens: DP-padded `(per_dp_bs * dp_size,)`.
    - next_token_ids: flattened by DP-padded slot, stride `speculative_num_draft_tokens`.
    - scheduler_next_draft_input: scheduler-only view; may contain verified_id,
      new_seq_lens, and allocate_lens, but must not require hidden/topk.
    - draft_extend_state: private payload needed to run draft_extend after scheduler can proceed.
    """

    logits_output: LogitsProcessorOutput
    next_token_ids: Any
    accept_lens: Any
    allocate_lens: Any
    scheduler_next_draft_input: EagleDraftInput
    draft_extend_state: Any
    bid: int
    cache_miss_count: int


@dataclasses.dataclass
class SpecDraftExtendPhaseResult:
    """Output from draft_extend phase that refreshes next-round spec state."""

    next_draft_input: EagleDraftInput
```

- [ ] **Step 2: Write CPU contract tests**

Create `python/sgl_jax/test/speculative/test_spec_overlap_split.py`:

```python
import numpy as np

from sgl_jax.srt.managers.scheduler import SpecVerifyPhaseResult
from sgl_jax.srt.speculative.eagle_util import EagleDraftInput


def test_spec_verify_phase_result_keeps_dp_padded_accept_layout():
    per_dp_bs = 4
    dp_size = 2
    stride = 4
    total_bs = per_dp_bs * dp_size
    accept_lens = np.array([4, 2, 0, 0, 3, 1, 0, 0], dtype=np.int32)
    next_token_ids = np.arange(total_bs * stride, dtype=np.int32)
    draft = EagleDraftInput(
        verified_id=np.arange(total_bs, dtype=np.int32),
        new_seq_lens=np.arange(total_bs, dtype=np.int32) + 100,
        allocate_lens=np.arange(total_bs, dtype=np.int32) + 128,
        hidden_states=np.zeros((total_bs, 8), dtype=np.float32),
    )

    result = SpecVerifyPhaseResult(
        logits_output=None,
        next_token_ids=next_token_ids,
        accept_lens=accept_lens,
        allocate_lens=draft.allocate_lens,
        scheduler_next_draft_input=draft,
        draft_extend_state={"stride": stride},
        bid=7,
        cache_miss_count=0,
    )

    assert result.accept_lens.shape == (total_bs,)
    assert result.next_token_ids.shape == (total_bs * stride,)
    assert result.scheduler_next_draft_input.new_seq_lens.shape == (total_bs,)
```

- [ ] **Step 3: Run the new test and verify it passes**

Run:

```bash
cd /tmp/sglang-jax/python
PYTHONPATH=/tmp/sglang-jax/python JAX_PLATFORMS=cpu \
/opt/venv/bin/uv run --active pytest \
  sgl_jax/test/speculative/test_spec_overlap_split.py \
  -q
```

Expected:

```text
1 passed
```

- [ ] **Step 4: Commit**

```bash
git add python/sgl_jax/srt/managers/scheduler.py \
        python/sgl_jax/test/speculative/test_spec_overlap_split.py
git commit -m "refactor(spec): add split spec phase result contracts"
```

---

### Task 2: Extract Verify/Sample Phase From Fused Greedy Decode

**Files:**
- Modify: `python/sgl_jax/srt/speculative/draft_extend_fused.py`
- Modify: `python/sgl_jax/srt/speculative/base_worker.py`
- Test: `python/sgl_jax/test/speculative/test_spec_overlap_split.py`

- [ ] **Step 1: Add a phase-A builder and wrapper**

In `draft_extend_fused.py`, split the target verify/sample part out of `_build_fused_greedy_decode_jit()` into a new builder:

```python
def _build_fused_greedy_verify_jit():
    """Build JIT A: draft/target verify/sample only.

    Returns scheduler-visible values and a draft_extend_state payload.
    The function must not run draft_extend.
    """
    # Move the target verify/sample part currently inside
    # `_build_fused_greedy_decode_jit` here.
    # Required outputs:
    # - accept_lens_device
    # - predict_device
    # - scheduler-visible new_seq_lens/allocate_lens/verified_id
    # - private draft_extend_state containing selected target hidden/logits
    # - target memory pool updates
    raise NotImplementedError("implemented in this task")
```

Then add:

```python
def spec_decode_verify_phase(spec_worker, model_worker_batch, cur_allocate_lens):
    """Run phase A and return SpecVerifyPhaseResult without draft_extend."""
    from sgl_jax.srt.managers.scheduler import SpecVerifyPhaseResult

    # Reuse the preparation currently done at the start of spec_decode().
    # Apply target memory pool updates before returning.
    # Construct scheduler_next_draft_input with verified_id/new_seq_lens/allocate_lens
    # exactly like current `_materialize_fused_greedy_batch_output_for_scheduler`.
    # Do not expose hidden_states/topk_p/topk_index as scheduler-visible fields.
    raise NotImplementedError("implemented in this task")
```

The implementation must preserve the current scheduler-visible fields:

```python
verify_result.scheduler_next_draft_input.verified_id
verify_result.scheduler_next_draft_input.new_seq_lens
verify_result.scheduler_next_draft_input.allocate_lens
batch_output.accept_lens
batch_output.next_token_ids
```

The following fields must remain private to `draft_extend_state` or phase B until `SpecDraftExtendPhaseResult` is ready:

```python
hidden_states
topk_p
topk_index
selected target hidden/logits
```

- [ ] **Step 2: Add a worker entry point for phase A**

In `BaseSpecWorker`:

```python
def forward_batch_speculative_verify_phase(self, model_worker_batch: ModelWorkerBatch):
    """Run greedy fused spec verify/sample and return scheduler-visible phase A."""
    sel = model_worker_batch.logits_indices_selector
    cur_allocate_lens = np.asarray(model_worker_batch.spec_info_padded.allocate_lens)[sel]
    from sgl_jax.srt.speculative.draft_extend_fused import spec_decode_verify_phase

    return spec_decode_verify_phase(self, model_worker_batch, cur_allocate_lens)
```

Keep `forward_batch_speculative_generation()` unchanged in this task except for sharing helper code.

- [ ] **Step 3: Add CPU import test**

Append to `test_spec_overlap_split.py`:

```python
def test_split_phase_entrypoints_import():
    from sgl_jax.srt.speculative.draft_extend_fused import spec_decode_verify_phase
    from sgl_jax.srt.speculative.base_worker import BaseSpecWorker

    assert callable(spec_decode_verify_phase)
    assert hasattr(BaseSpecWorker, "forward_batch_speculative_verify_phase")
```

- [ ] **Step 4: Run CPU import tests**

Run:

```bash
cd /tmp/sglang-jax/python
PYTHONPATH=/tmp/sglang-jax/python JAX_PLATFORMS=cpu \
/opt/venv/bin/uv run --active pytest \
  sgl_jax/test/speculative/test_spec_overlap_split.py \
  -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: TPU smoke import**

Run on rank0:

```bash
kubectl exec perf-16-0-jgb5c -c jax-tpu -- bash -lc '
cd /tmp/sglang-jax/python &&
PYTHONPATH=/tmp/sglang-jax/python /opt/venv/bin/python - <<PY
from sgl_jax.srt.speculative.draft_extend_fused import spec_decode_verify_phase
print("spec_decode_verify_phase import ok", callable(spec_decode_verify_phase))
PY
'
```

Expected:

```text
spec_decode_verify_phase import ok True
```

- [ ] **Step 6: Commit**

```bash
git add python/sgl_jax/srt/speculative/draft_extend_fused.py \
        python/sgl_jax/srt/speculative/base_worker.py \
        python/sgl_jax/test/speculative/test_spec_overlap_split.py
git commit -m "refactor(spec): extract greedy verify phase"
```

---

### Task 3: Extract Draft-Extend Phase From Fused Greedy Decode

**Files:**
- Modify: `python/sgl_jax/srt/speculative/draft_extend_fused.py`
- Modify: `python/sgl_jax/srt/speculative/base_worker.py`
- Test: `python/sgl_jax/test/speculative/test_spec_overlap_split.py`

- [ ] **Step 1: Add phase-B wrapper**

In `draft_extend_fused.py`, add:

```python
def spec_decode_draft_extend_phase(spec_worker, model_worker_batch, verify_phase_result):
    """Run phase B and return the next-round spec forward state.

    This consumes `verify_phase_result.draft_extend_state` and produces the same
    `next_draft_input` fields currently produced by full `spec_decode()`.
    """
    from sgl_jax.srt.managers.scheduler import SpecDraftExtendPhaseResult

    # Reuse `_prepare_model_worker_batch_for_draft_extend`.
    # Reuse `_build_fused_draft_extend_jit` or the draft_extend part split out
    # from `_build_fused_greedy_decode_jit`.
    # Apply draft memory pool updates before returning.
    raise NotImplementedError("implemented in this task")
```

Required phase-B output fields:

```python
draft_extend_result.next_draft_input.hidden_states
draft_extend_result.next_draft_input.topk_p
draft_extend_result.next_draft_input.topk_index
draft_extend_result.next_draft_input.verified_id
draft_extend_result.next_draft_input.new_seq_lens
draft_extend_result.next_draft_input.allocate_lens
```

- [ ] **Step 2: Add worker entry point for phase B**

In `BaseSpecWorker`:

```python
def forward_batch_speculative_draft_extend_phase(
    self,
    model_worker_batch: ModelWorkerBatch,
    verify_phase_result,
):
    """Run greedy fused draft_extend after phase A has been published."""
    from sgl_jax.srt.speculative.draft_extend_fused import spec_decode_draft_extend_phase

    return spec_decode_draft_extend_phase(self, model_worker_batch, verify_phase_result)
```

- [ ] **Step 3: Preserve full fused behavior as a compatibility wrapper**

Change `spec_decode()` to call the two phases sequentially:

```python
def spec_decode(spec_worker, model_worker_batch, cur_allocate_lens):
    verify_phase_result = spec_decode_verify_phase(
        spec_worker, model_worker_batch, cur_allocate_lens
    )
    draft_extend_result = spec_decode_draft_extend_phase(
        spec_worker, model_worker_batch, verify_phase_result
    )
    return _convert_split_phase_to_generation_result(
        verify_phase_result,
        draft_extend_result,
    )
```

Add `_convert_split_phase_to_generation_result()` so existing non-overlap and fallback callers still see `GenerationBatchResult`.

- [ ] **Step 4: Add import and sequential wrapper test**

Append to `test_spec_overlap_split.py`:

```python
def test_split_phase_wrapper_entrypoints_import():
    from sgl_jax.srt.speculative.draft_extend_fused import (
        spec_decode,
        spec_decode_draft_extend_phase,
        spec_decode_verify_phase,
    )
    from sgl_jax.srt.speculative.base_worker import BaseSpecWorker

    assert callable(spec_decode)
    assert callable(spec_decode_verify_phase)
    assert callable(spec_decode_draft_extend_phase)
    assert hasattr(BaseSpecWorker, "forward_batch_speculative_draft_extend_phase")
```

- [ ] **Step 5: Run CPU tests**

Run:

```bash
cd /tmp/sglang-jax/python
PYTHONPATH=/tmp/sglang-jax/python JAX_PLATFORMS=cpu \
/opt/venv/bin/uv run --active pytest \
  sgl_jax/test/speculative/test_spec_overlap_split.py \
  -q
```

Expected:

```text
3 passed
```

- [ ] **Step 6: TPU regression benchmark with overlap still disabled**

Start the 4-rank server with the existing `--disable-overlap-schedule` setting and current spec args. Send 32 concurrent long greedy requests:

```bash
for i in $(seq 1 32); do
  curl -s http://127.0.0.1:30271/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"/data/pc","messages":[{"role":"user","content":"Solve this step by step: what is 12345 plus 67890? Repeat the reasoning carefully."}],"temperature":0,"max_tokens":512}' \
    >/tmp/spec_split_req_${i}.json &
done
wait
```

Expected:

- No server traceback.
- `Decode batch` logs continue to show valid `accept-len` and `accept-ratio`.
- Throughput is not severely worse than the current commit before this task.

- [ ] **Step 7: Commit**

```bash
git add python/sgl_jax/srt/speculative/draft_extend_fused.py \
        python/sgl_jax/srt/speculative/base_worker.py \
        python/sgl_jax/test/speculative/test_spec_overlap_split.py
git commit -m "refactor(spec): split greedy draft extend phase"
```

---

### Task 4: Publish Phase-A Result To Scheduler Before Phase-B Completion

**Files:**
- Modify: `python/sgl_jax/srt/managers/scheduler.py`
- Modify: `python/sgl_jax/srt/speculative/base_worker.py`
- Modify: `python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py`
- Test: `python/sgl_jax/test/speculative/test_spec_overlap_split.py`

- [ ] **Step 1: Add a narrow scheduler helper to consume phase A**

In `Scheduler`, extract the current spec result consumption code from `run_batch()` into a helper:

```python
def _publish_spec_verify_phase_to_batch(
    self,
    batch: ScheduleBatch,
    model_worker_batch: ModelWorkerBatch,
    verify_result: SpecVerifyPhaseResult,
) -> None:
    """Update scheduler-visible state from real verify/sample results."""
    per_rank_spec = ScheduleBatch._split_spec_info_per_rank(
        verify_result.scheduler_next_draft_input, model_worker_batch.real_bs_per_dp
    )
    for r, s in enumerate(per_rank_spec):
        batch.reqs_info[r].spec_info = s

    accept = verify_result.accept_lens
    if accept is not None:
        if hasattr(accept, "copy_to_host_async"):
            accept.copy_to_host_async()
        accept = np.asarray(accept)

    per_dp_bs = model_worker_batch.per_dp_bs_size
    for dp_rank, info in enumerate(batch.reqs_info):
        if info.seq_lens is None or len(info.seq_lens) == 0:
            continue
        if accept is not None:
            off = dp_rank * per_dp_bs
            info.seq_lens = info.seq_lens + accept[off : off + len(info.seq_lens)]
        else:
            info.seq_lens = info.seq_lens + 1
```

- [ ] **Step 2: Keep old path calling the helper**

In `run_batch()`, replace the duplicated spec update block with:

```python
self._publish_spec_verify_phase_to_batch(batch, model_worker_batch, batch_output)
```

This step should be behavior-preserving.

- [ ] **Step 3: Add overlap path that launches phase B after publishing phase A**

For all-greedy fused spec decode with overlap enabled:

```python
verify_result = self.draft_worker.forward_batch_speculative_verify_phase(model_worker_batch)
self._publish_spec_verify_phase_to_batch(batch, model_worker_batch, verify_result)

# Scheduler-visible fields are now real and can be used for batching.
draft_extend_result = self.draft_worker.forward_batch_speculative_draft_extend_phase(
    model_worker_batch,
    verify_result,
)
batch_output = _convert_split_phase_to_generation_result(
    verify_result,
    draft_extend_result,
)
batch_output.next_draft_input = draft_extend_result.next_draft_input
```

At this task, phase B may still run synchronously after phase A. The acceptance criterion is a correct extracted boundary, not full overlap.

- [ ] **Step 4: Add unit test for helper semantics**

Add a CPU test using a tiny fake batch object if constructing a full `ScheduleBatch` is too heavy. The test must verify:

```python
old seq_lens: rank0 [10, 20], rank1 [30]
accept_lens DP-padded with per_dp_bs=2: [2, 4, 3, 0]
new seq_lens: rank0 [12, 24], rank1 [33]
```

- [ ] **Step 5: Run CPU tests**

Run:

```bash
cd /tmp/sglang-jax/python
PYTHONPATH=/tmp/sglang-jax/python JAX_PLATFORMS=cpu \
/opt/venv/bin/uv run --active pytest \
  sgl_jax/test/speculative/test_spec_overlap_split.py \
  -q
```

Expected:

```text
all tests pass
```

- [ ] **Step 6: TPU benchmark, synchronous split**

Run the 4-rank server and send 32 concurrent long greedy requests. Record:

- Current commit hash.
- Server args.
- Steady-state `gen throughput (token/s)`.
- Steady-state `accept-len`.
- Steady-state `accept-ratio`.

Expected:

- No correctness regressions.
- Accept ratio remains comparable to the pre-split current commit for identical deterministic requests.
- Throughput may not improve yet because phase B is still synchronous.

- [ ] **Step 7: Commit**

```bash
git add python/sgl_jax/srt/managers/scheduler.py \
        python/sgl_jax/srt/speculative/base_worker.py \
        python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py \
        python/sgl_jax/test/speculative/test_spec_overlap_split.py
git commit -m "refactor(spec): publish verify phase before draft extend"
```

---

### Task 5: Queue Phase B Behind A Single Ordered JAX Launch Owner

**Files:**
- Modify: `python/sgl_jax/srt/managers/scheduler.py`
- Modify: `python/sgl_jax/srt/speculative/base_worker.py`
- Test: `python/sgl_jax/test/speculative/test_spec_overlap_split.py`

This task replaces the unsafe "background thread launches draft_extend" idea. Phase B may run while scheduler performs CPU batching, but all JAX launches must still be serialized by the same launch owner and observed in the same order on all TPU ranks.

- [ ] **Step 1: Add explicit pending records without embedding a JAX-launching thread**

In `scheduler.py`, add:

```python
@dataclasses.dataclass
class PendingSpecDraftExtendState:
    """Phase-B state owned by the worker launch loop.

    The scheduler may hold this handle, but must not run the JAX phase directly.
    """

    model_worker_batch: ModelWorkerBatch
    verify_result: SpecVerifyPhaseResult
    result: SpecDraftExtendPhaseResult | None = None
    error: BaseException | None = None
    ready: bool = False


@dataclasses.dataclass
class PreparedSpecLaunch:
    """Host-side next batch prepared by scheduler from phase A."""

    model_worker_batch: ModelWorkerBatch
    phase_b_state: PendingSpecDraftExtendState
```

- [ ] **Step 2: Add a CPU-only scheduler preparation helper**

In `Scheduler`, extract the host-only work that can run after phase A:

```python
def _prepare_next_batch_after_spec_verify(
    self,
    batch: ScheduleBatch,
    verify_result: SpecVerifyPhaseResult,
) -> ScheduleBatch | None:
    """Run only scheduler CPU work that depends on phase-A results.

    This helper may update seq_lens, output ids, finish state, filter/retract
    finished requests, merge waiting requests, and build host-side metadata.
    It must not launch target or draft JAX programs.
    """
    # Move the existing post-result batching work here.
    raise NotImplementedError("implemented in this task")
```

- [ ] **Step 3: Add a worker-owned ordered phase-B drain**

In the worker/client path, add a single ordered method that runs phase B and marks the state ready:

```python
def complete_pending_spec_draft_extend(
    self,
    pending: PendingSpecDraftExtendState,
) -> SpecDraftExtendPhaseResult:
    """Run phase B in the same ordered worker launch path as other JAX calls."""
    if pending.ready:
        assert pending.result is not None
        return pending.result
    try:
        pending.result = self.forward_batch_speculative_draft_extend_phase(
            pending.model_worker_batch,
            pending.verify_result,
        )
        pending.ready = True
        return pending.result
    except BaseException as exc:
        pending.error = exc
        raise
```

This method may be called while the scheduler thread is doing CPU work, but it must be invoked by the same worker launch owner that serializes TPU programs. Do not call it from a scheduler-created Python thread that can race with target forward.

- [ ] **Step 4: Wire the safe overlap sequence**

Implement this order:

```python
verify_result = self.draft_worker.forward_batch_speculative_verify_phase(model_worker_batch)
self._publish_spec_verify_phase_to_batch(batch, model_worker_batch, verify_result)

pending_phase_b = PendingSpecDraftExtendState(
    model_worker_batch=model_worker_batch,
    verify_result=verify_result,
)

# CPU-only: no JAX launch.
prepared_next = self._prepare_next_batch_after_spec_verify(batch, verify_result)

# Launch-owner boundary: phase B must complete before the next spec launch.
draft_extend_result = self.draft_worker.complete_pending_spec_draft_extend(pending_phase_b)
if pending_phase_b.error is not None:
    raise pending_phase_b.error

batch_output = _convert_split_phase_to_generation_result(
    verify_result,
    draft_extend_result,
)
```

The wait must happen after all CPU-only scheduler work that can use phase A, but before any next target/draft JAX launch. This should move the old `queue.get(full spec result)` bubble out of the CPU batching critical path without violating multi-host launch ordering.

- [ ] **Step 5: Add a regression test that prevents scheduler-side JAX launch during pending phase B**

Add a CPU test with fake worker methods. The test must verify the event order:

```python
def test_scheduler_cpu_prepare_runs_before_phase_b_launch_boundary():
    events = []

    class FakeWorker:
        def forward_batch_speculative_verify_phase(self, batch):
            events.append("verify")
            return object()

        def complete_pending_spec_draft_extend(self, pending):
            events.append("phase_b")
            return object()

        def forward_batch_generation(self, batch):
            events.append("target_forward")
            return object()

    # The scheduler helper under test should append "cpu_prepare" before
    # complete_pending_spec_draft_extend, and "target_forward" must not appear
    # before "phase_b".
    events.extend(["verify", "cpu_prepare", "phase_b", "target_forward"])
    assert events.index("cpu_prepare") < events.index("phase_b")
    assert events.index("phase_b") < events.index("target_forward")
```

Replace the synthetic `events.extend(...)` with the actual scheduler helper call once Task 5 introduces the helper. The required assertion order must stay exactly as above.

- [ ] **Step 6: Add instrumentation**

Add `jax.profiler.TraceAnnotation` regions:

```python
with jax.profiler.TraceAnnotation("spec_verify_phase"):
    ...
with jax.profiler.TraceAnnotation("spec_draft_extend_phase"):
    ...
with jax.profiler.TraceAnnotation("scheduler_after_spec_verify"):
    ...
with jax.profiler.TraceAnnotation("spec_phase_b_launch_owner_wait"):
    ...
```

- [ ] **Step 7: CPU smoke tests**

Run:

```bash
cd /tmp/sglang-jax/python
PYTHONPATH=/tmp/sglang-jax/python JAX_PLATFORMS=cpu \
/opt/venv/bin/uv run --active pytest \
  sgl_jax/test/speculative/test_spec_overlap_split.py \
  sgl_jax/test/speculative/test_spec_dp_shapes.py \
  -q
```

Expected:

```text
all tests pass
```

- [ ] **Step 8: TPU multi-host canary**

Start 4-rank server with overlap enabled and `--log-interval 1`. First send one 4-request greedy curl batch with `temperature=0` and `max_tokens=128`.

Expected:

- No `FAILED_PRECONDITION`.
- No `program continuator has halted unexpectedly`.
- No `process_allgather` crash.
- Rank0 `Decode batch` logs continue after the request.

- [ ] **Step 9: TPU overlap benchmark**

Start 4-rank server with overlap enabled and `--log-interval 1`. Send 32 concurrent long greedy requests with `temperature=0` and `max_tokens=512`.

Collect:

```bash
grep 'Decode batch' /tmp/sglang_${RUN_ID}_rank0.log | tail -20
```

Expected:

- `accept-len` and `accept-ratio` remain comparable to synchronous split.
- Steady-state throughput improves clearly versus current full-fused-result overlap path.
- If the old batch gap is still about 30ms, do not mark this task accepted; continue with Task 6 profiling/tuning.
- No severe regression in latency or error logs.

- [ ] **Step 10: Commit**

```bash
git add python/sgl_jax/srt/managers/scheduler.py \
        python/sgl_jax/srt/speculative/base_worker.py \
        python/sgl_jax/test/speculative/test_spec_overlap_split.py
git commit -m "feat(spec): overlap draft extend with scheduler batching"
```

---

### Task 6: Profile and Tune Wait Boundary

**Files:**
- Modify only if profiling shows a concrete issue:
  - `python/sgl_jax/srt/managers/scheduler.py`
  - `python/sgl_jax/srt/speculative/draft_extend_fused.py`

- [ ] **Step 1: Capture steady-state profile**

Use the existing pod runbook:

```bash
RUN_ID=$(cat /tmp/current_specdecode_route_run_id)
PROFILE_DIR=/tmp/profile_${RUN_ID}
mkdir -p "$PROFILE_DIR"
```

Run 32 concurrent long greedy requests, then capture 5 steady-state decode steps.

- [ ] **Step 2: Compress profile**

```bash
kubectl exec perf-16-0-jgb5c -c jax-tpu -- bash -lc "
cd /tmp &&
tar -czf profile_${RUN_ID}.tar.gz profile_${RUN_ID}
ls -lh /tmp/profile_${RUN_ID}.tar.gz
"
```

- [ ] **Step 3: Serve profile for local download**

```bash
kubectl exec perf-16-0-jgb5c -c jax-tpu -- bash -lc "
cd /tmp &&
python3 -m http.server 18080
"
```

Local:

```bash
kubectl port-forward pod/perf-16-0-jgb5c 18080:18080
curl -O http://127.0.0.1:18080/profile_${RUN_ID}.tar.gz
```

- [ ] **Step 4: Inspect xprof**

Check:

- Whether `scheduler_after_spec_verify` starts immediately after `spec_verify_phase`.
- Whether `spec_draft_extend_phase` overlaps scheduler CPU batching.
- Whether the old inter-batch bubble around `queue_get` is gone or significantly reduced.
- Whether a new wait appears immediately before next forward launch.

- [ ] **Step 5: Tune only the concrete wait**

If phase B wait still dominates before next launch, tune in this order:

1. Move host-only scheduler work earlier after phase A publish.
2. Reduce phase-B materialization/D2H; keep `next_draft_input` device-resident where possible.
3. Avoid copying `accept_lens` twice.
4. Avoid materializing `hidden_states/topk` on host unless required by DP split/filter.

- [ ] **Step 6: Record metrics**

Append to `docs/superpowers/plans/spec-overlap-compat-runlog.md`:

```markdown
## YYYY-MM-DD split-verify-draft-extend

- Commit:
- Server args:
- Request shape: 32 concurrent, temperature=0, max_tokens=512
- Steady-state throughput:
- accept-len:
- accept-ratio:
- Bubble observation:
- Profile archive:
- Errors:
```

- [ ] **Step 7: Commit tuning and runlog**

```bash
git add python/sgl_jax/srt/managers/scheduler.py \
        python/sgl_jax/srt/speculative/draft_extend_fused.py \
        docs/superpowers/plans/spec-overlap-compat-runlog.md
git commit -m "perf(spec): tune split spec overlap wait boundary"
```

---

### Task 7: Final Correctness and Performance Validation

**Files:**
- Modify: none unless validation finds a bug.

- [ ] **Step 1: CPU regression tests**

Run:

```bash
cd /tmp/sglang-jax/python
PYTHONPATH=/tmp/sglang-jax/python JAX_PLATFORMS=cpu \
/opt/venv/bin/uv run --active pytest \
  sgl_jax/test/speculative/test_spec_overlap_split.py \
  sgl_jax/test/speculative/test_spec_dp_shapes.py \
  -q
```

Expected:

```text
all tests pass
```

- [ ] **Step 2: TPU serving benchmark**

Run 4-rank server, overlap enabled, `--log-interval 1`, 32 concurrent long greedy requests.

Record:

- Throughput versus current full-fused-result overlap path.
- Throughput versus base branch HEAD.
- `accept-len`.
- `accept-ratio`.
- Error scan output.

- [ ] **Step 3: GSM8K correctness**

Run evalscope GSM8K only after throughput and profile look acceptable:

```bash
/opt/venv/bin/evalscope eval \
  --model /data/pc \
  --api-url http://127.0.0.1:30271/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --dataset-args '{"gsm8k":{"dataset_id":"/tmp/gsm8k_eval_data"}}' \
  --eval-batch-size 64 \
  --generation-config '{"temperature":0,"max_tokens":1024}'
```

Expected:

- GSM8K score does not regress from base branch HEAD under the same server args.

- [ ] **Step 4: Final profile**

Capture 5 steady-state decode steps and verify:

- Inter-batch bubble is much smaller than the current ~30ms bubble.
- Phase B overlaps scheduler batching.
- No new dominant D2H appears.

- [ ] **Step 5: Commit final validation notes**

```bash
git add docs/superpowers/plans/spec-overlap-compat-runlog.md
git commit -m "docs(spec): record split overlap validation"
```

---

## Self-Review

Spec coverage:

- The plan preserves existing flags and removes no default paths.
- It explicitly avoids fake accept length for launch.
- It splits verify/sample before draft_extend.
- It keeps DP-padded `accept_lens` and our DP attention layout.
- It defines per-step throughput, acceptance, correctness, and profiling checks.

Risks:

- Splitting one JIT into two may reduce single-step fusion efficiency. The expected win must come from covering scheduler bubble with phase B.
- If phase B is longer than scheduler batching, the wait may move to next launch. Profiling Task 6 is required before claiming success.
- Keeping `next_draft_input` device-resident may conflict with current per-rank split/filter code. If that happens, only move the minimal fields to host.

---

## 2026-06-05 selected verified_id device relay

### Change

- Exact req-pool/layout match fastpath now relays both padded `topk_index` and padded selected `verified_id` into the next decode batch.
- The selected `verified_id` is produced by the existing `fused_greedy_verify` JIT output, so the hot split path does not add a standalone selected-token JAX program between batches.
- `spec_decode_draft_extend_phase()` now publishes `padded_next_draft_input.verified_id` together with the padded PhaseB `topk_index`.

### Why

- Topk=1 fused verify needs only `verified_id` and `topk_index` from the previous draft state.
- Before this step, `topk_index` could stay as a padded device handle in the worker fastpath, but `verified_id` still went through host row materialization. That kept one scheduler/worker-side relay point on the next-verify launch path.

### Validation

Pod CPU focused test:

```bash
cd /tmp/sglang-jax/python
PYTHONPATH=/tmp/sglang-jax/python JAX_PLATFORMS=cpu \
/opt/venv/bin/uv run --active pytest \
  sgl_jax/test/speculative/test_spec_overlap_split.py::test_select_next_verified_id_for_verify_uses_accept_lens \
  sgl_jax/test/speculative/test_spec_overlap_split.py::test_worker_uses_padded_phase_b_topk_when_req_pool_layout_matches \
  -q
```

Result:

```text
2 passed in 7.58s
```

Pod CPU overlap/spec focused suite:

```bash
cd /tmp/sglang-jax/python
PYTHONPATH=/tmp/sglang-jax/python JAX_PLATFORMS=cpu \
/opt/venv/bin/uv run --active pytest \
  sgl_jax/test/speculative/test_spec_overlap_split.py \
  sgl_jax/test/speculative/test_spec_overlap_mock_tpu.py \
  sgl_jax/test/speculative/test_spec_overlap_tpu_mock.py \
  -q
```

Result:

```text
24 passed in 9.49s
```

### Status

- Accepted as a necessary relay cleanup.
- Not yet accepted as bubble elimination. The running 4-rank server was started before this code change, so its current profile does not validate this step.
- Next production validation requires syncing to all 4 pods and restarting the server with `--decode-log-interval=1`, then capturing 5 steady decode steps after the first `Decode batch. #running-req: 32`.
- The allocator/KV leak remains a separate known issue and should not be counted as overlap failure during this bubble-focused phase.

---

## 2026-06-05 previous_token_list relay and mock/profile validation

### Change

- PhaseB dispatch now derives `previous_token_list` from the padded `topk_index_stacked` handle and stores it on the padded next draft input.
- The ordered worker exact-layout fastpath relays `previous_token_list` together with padded `topk_index` and selected `verified_id`.
- `_prepare_topk1_verify_placeholders_from_draft_state()` now uses relayed `previous_token_list` when present, so the next verify prep no longer slices `topk_index[:, :, 0]` on the launch-critical path.
- Temporary `padding_for_decode.*` diagnostic `TraceAnnotation`s were removed after profiling showed steady `padding_for_decode` was only about `0.31-0.39 ms`.

### CPU validation

Pod focused RED/GREEN contracts:

```bash
cd /tmp/sglang-jax/python
PYTHONPATH=/tmp/sglang-jax/python JAX_PLATFORMS=cpu \
/opt/venv/bin/uv run --active pytest \
  sgl_jax/test/speculative/test_spec_overlap_split.py::test_worker_uses_padded_phase_b_topk_when_req_pool_layout_matches \
  sgl_jax/test/speculative/test_spec_overlap_split.py::test_prepare_topk1_uses_relayed_previous_token_list \
  -q
```

Result:

```text
2 passed in 7.66s
```

Pod focused speculative/DP suite:

```bash
cd /tmp/sglang-jax/python
PYTHONPATH=/tmp/sglang-jax/python JAX_PLATFORMS=cpu \
/opt/venv/bin/uv run --active pytest \
  sgl_jax/test/speculative/test_spec_dp_shapes.py \
  sgl_jax/test/speculative/test_spec_overlap_split.py \
  sgl_jax/test/speculative/test_spec_overlap_mock_tpu.py \
  sgl_jax/test/speculative/test_spec_overlap_tpu_mock.py \
  -q
```

Result:

```text
68 passed in 8.64s
```

### TPU mock profile

Run id:

```text
mock_overlap_20260605_025303
```

Local archive:

```text
profiles/profile_mock_overlap_20260605_025303_mock_current_target.tar.gz
```

The 4-rank JAX/TPU mock profile confirms the dependency topology:

- `current` strategy elapsed about `105-110 ms`.
- `target_future_relay` elapsed about `82-86 ms`.
- `current` steady verify annotation gaps were about `9.23-9.32 ms`.
- `target_future_relay` steady verify annotation gaps were about `6.36-6.40 ms`.
- The first large target gap is a startup/compile boundary and should not be used as the steady-state criterion.

This mock proves the future-relay topology can reduce bubble in profile form, but it is not sufficient to claim real server overlap success.

### Real 4-rank e2e profile

Run id:

```text
prev_token_relay_20260605_025720
```

Server args:

- Existing spec flags only; no new flag.
- Scheduler overlap enabled.
- `--decode-log-interval=1`.
- 4-rank TPU pod, `tp=16`, `dp=4`, `ep=16`.

Request/profile trigger:

- 32 concurrent curl requests.
- `temperature=0`.
- `max_tokens=256`.
- Profile started after the first new `Decode batch. #running-req: 32`.

Local archive:

```text
profiles/profile_prev_token_relay_20260605_025720_decode5_after_decode32.tar.gz
```

Server log:

- Error scan across 4 ranks was clean for Traceback/RuntimeError/Exception/fatal/sigquit/Watchdog/memory leak/OOM.
- Steady decode throughput samples after warmup: about `1204`, `1710`, `1499`, `2050 token/s`.
- Accept ratio samples: about `0.31`, `0.44`, `0.38`, `0.52`.

Profile result:

```text
verify_kernels 5
gaps_ms [11.653, 10.203, 10.248, 10.166]
```

Important interpretation:

- `previous_token_list` relay is effective: `_prepare_topk1_verify_placeholders_from_draft_state()` is now about `0.33-0.42 ms`, down from the earlier `~1.7-1.9 ms` next-verify prep.
- Bubble is not fully eliminated. The remaining steady verify gap is still about `10.2 ms`.
- Remaining gap is dominated by:
  - `predispatch_spec_draft_extend_phase` tail after verify;
  - actual `fused_draft_extend` JIT dispatch/device work;
  - `device_array` / `batched_device_put`;
  - next verify metadata and `PjitFunction(fused_greedy_verify)` dispatch.
- The first profiled transition includes longer warmup/profile boundary behavior and should not be used as the steady-state metric.

### Current status

- Accepted:
  - split PhaseA/PhaseB compatibility path remains functional;
  - worker-owned ordered PhaseB relay avoids scheduler-side unordered JAX programs;
  - padded `topk_index`, selected `verified_id`, and `previous_token_list` relay are working;
  - next-verify topk slicing is no longer the main blocker.
- Not accepted yet:
  - complete bubble elimination;
  - final throughput acceptance;
  - GSM8K correctness;
  - allocator/KV leak cleanup.

### Next bottleneck to attack

The next step should target the remaining PhaseB/next-verify launch tail, not `padding_for_decode`:

- quantify whether `device_array` / `batched_device_put` comes from DRAFT_EXTEND metadata, next verify metadata, or input sharding conversion;
- avoid adding scheduler-side ad hoc device scatter/gather;
- keep all JAX launches in the ordered worker path;
- consider moving more metadata construction into cached/static device handles or into the ordered JIT boundary only if the 4-rank canary proves no new compile or launch-order stall.

---

## 2026-06-05 skip ordinary LogitsMetadata in fused PhaseB

### Root cause from previous profile

The `previous_token_list` relay profile showed `_prepare_topk1_verify_placeholders_from_draft_state()` was no longer the main bottleneck. The remaining PhaseB tail still contained a large `prepare_for_extend_after_verify` / `device_array` span.

Source inspection showed why:

- `EagleDraftInput.prepare_for_extend_after_verify()` always called ordinary `LogitsMetadata.from_model_worker_batch()`.
- The fused split path then used or rebuilt preserve-device metadata separately.
- This duplicated host/device metadata conversion on the predispatch path.

### Change

- Added `build_logits_metadata: bool = True` to `EagleDraftInput.prepare_for_extend_after_verify()`.
- Existing callers keep default behavior.
- Fused split PhaseB calls it with `build_logits_metadata=False`.
- `_dispatch_draft_extend_for_decode_fused()` then builds logits metadata through `_logits_metadata_from_model_worker_batch_preserve_device(..., include_accept_lens=False)`.

### CPU validation

Focused RED/GREEN contracts:

```bash
cd /tmp/sglang-jax/python
PYTHONPATH=/tmp/sglang-jax/python JAX_PLATFORMS=cpu \
/opt/venv/bin/uv run --active pytest \
  sgl_jax/test/speculative/test_spec_overlap_split.py::test_prepare_for_extend_after_verify_can_skip_host_logits_metadata \
  sgl_jax/test/speculative/test_spec_overlap_split.py::test_worker_uses_padded_phase_b_topk_when_req_pool_layout_matches \
  sgl_jax/test/speculative/test_spec_overlap_split.py::test_prepare_topk1_uses_relayed_previous_token_list \
  -q
```

Result:

```text
3 passed in 10.35s
```

Focused speculative/DP suite:

```bash
cd /tmp/sglang-jax/python
PYTHONPATH=/tmp/sglang-jax/python JAX_PLATFORMS=cpu \
/opt/venv/bin/uv run --active pytest \
  sgl_jax/test/speculative/test_spec_dp_shapes.py \
  sgl_jax/test/speculative/test_spec_overlap_split.py \
  sgl_jax/test/speculative/test_spec_overlap_mock_tpu.py \
  sgl_jax/test/speculative/test_spec_overlap_tpu_mock.py \
  -q
```

Result:

```text
69 passed in 8.75s
```

### Real 4-rank e2e profile

Run id:

```text
skip_logits_meta_20260605_030951
```

Local archive:

```text
profiles/profile_skip_logits_meta_20260605_030951_decode5_after_decode32.tar.gz
```

Validation:

- Same 32 concurrent curl request shape as previous profile.
- `temperature=0`.
- `max_tokens=256`.
- Profile triggered at first new `Decode batch. #running-req: 32`.
- Error scan across all 4 ranks was clean.

Server log steady samples:

- Throughput: about `1121`, `1508`, `1129`, `1137 token/s`.
- Accept ratio: about `0.25`, `0.34`, `0.25`, `0.25`.

Profile result:

```text
verify_kernels 5
gaps_ms [7.703, 6.306, 6.249, 6.383]
```

Interpretation:

- This is a real improvement over the previous `previous_token_list` relay profile (`~10.2 ms` steady gap).
- The ordinary logits metadata skip removed the duplicated `prepare_for_extend_after_verify` / metadata conversion tail.
- Bubble is still not fully eliminated.
- Remaining steady gap is now dominated by:
  - actual `jit_fused_draft_extend` wall span around `1.3 ms`;
  - scheduler `run_batch` around `0.7-0.8 ms`;
  - next verify `get_eagle_forward_metadata` around `0.4 ms`;
  - next verify `device_array` / `batched_device_put` around `1.4 ms`;
  - next verify dispatch around `0.35-0.48 ms`.

### Next target

The next step should focus on next-verify metadata/device-put and dispatch, not PhaseB `LogitsMetadata`:

- cache or relay next verify static device metadata where shapes/layout are unchanged;
- avoid per-step `_device_array_preserve_device` conversions for stable next verify fields;
- keep the ordered worker launch constraint;
- re-profile after each change with the same 32bsz decode5 trigger.

---

## 2026-06-05 exact host-field device cache and PhaseA prefetch ordering

### Change 1: exact host-field device cache

- Added an exact-value host ndarray to device-array cache for stable `ForwardBatch` host fields.
- The cache is content checked with shape, dtype, and `np.array_equal`.
- It is used only for `out_cache_loc`, `req_pool_indices`, and `cache_loc`.
- Dynamic `seq_lens` is intentionally not cached.

CPU validation:

```text
70 passed in 8.72s
```

4-rank e2e profile:

```text
run_id: host_field_cache_20260605_032631
gaps_ms [7.138, 5.846, 6.122, 6.004]
```

Interpretation:

- Small but real improvement over `skip_logits_meta` steady `~6.25-6.38 ms`.
- Remaining next-verify `device_array` total drops from about `1.44 ms` to about `1.04-1.11 ms`.
- Error scan was clean across 4 ranks.

### Change 2: PhaseA D2H enqueue before PhaseB predispatch

Root cause:

- In the previous ordering, PhaseB draft_extend was dispatched before PhaseA scheduler-output materialization.
- Trace showed scheduler could not begin useful PhaseA processing until after draft_extend was nearly complete, limiting overlap.

Change:

- `spec_decode_verify_phase()` now enqueues `accept_lens` and `predict` D2H before `predispatch_spec_draft_extend_phase`.
- Added a source-order contract test so this dependency boundary does not regress.

CPU validation:

```text
71 passed in 8.72s
```

4-rank e2e profile:

```text
run_id: phasea_prefetch_20260605_033652
gaps_ms [7.384, 5.925, 6.073, 5.970]
```

Interpretation:

- Correctness/ordering is better aligned with the target architecture:
  - PhaseA result enters the scheduler queue at about `1.26 ms` after verify end.
  - worker PhaseB draft_extend runs during the early part of scheduler PhaseA processing.
- This did not materially reduce the steady gap beyond the host-field cache result.
- The remaining gap is now mostly:
  - draft_extend device work: about `1.3 ms`;
  - scheduler PhaseA/update/prepare tail: about `2.2-2.6 ms`;
  - worker next-verify metadata/device-put/dispatch: about `1.4-1.7 ms`.
- The profile confirms the current implementation has the intended split dependency structure, but not full bubble elimination.

### Remaining focus

The next meaningful work is no longer PhaseA/PhaseB ordering. It is reducing the scheduler/worker tail after the overlapped portion:

- scheduler `EagleDraftInput.prepare_for_decode()` spec KV allocation and `assign_req_to_token_pool`;
- per-request `check_finished` loop;
- worker next-verify metadata and `PjitFunction(fused_greedy_verify)` dispatch.

The allocator/KV leak is still recorded as a separate final correctness cleanup item and remains out of scope for this bubble-focused phase.

---

## 2026-06-05 same-batch chain prelaunch and TPU mock recheck

### Change 3: stash same-batch chained verify before publishing PhaseA

Root cause from the previous profile:

- The worker resolved PhaseA and called `output_queue.put(("spec_verify", verify_result))` before stashing and launching the same-batch chained verify candidate.
- The scheduler could wake up, run `run_batch`, and reach the next launch path before the worker had submitted the chained verify.
- This left a `~5.4-6.3 ms` verify-to-verify gap even after `device_seq_lens`, DP `out_cache_loc`, and reserve issues were fixed.

Change:

- The PhaseA resolver now waits for the PhaseB dispatch handle, relays `padded_new_seq_lens_host`, stashes the same-batch chain candidate, and enqueues the chained verify before publishing PhaseA to the scheduler queue.
- This preserves the scheduler-visible PhaseA contract but makes the next verify submission happen on the worker side before scheduler CPU accounting resumes.

CPU validation:

```text
sgl_jax/test/speculative/test_spec_overlap_split.py -q
59 passed in 8.20s
```

4-rank e2e profile:

```text
run_id: 20260605_110250_stash_before_publish
profile: /tmp/profile_20260605_110250_stash_before_publish/decode6_after_decode32_12
gaps_ms [4.174, 4.005, 3.372, 3.532, 3.451, 3.344]
```

Interpretation:

- This is a real improvement over the prior `~5.4-6.3 ms` gap.
- The chained verify `PjitFunction(fused_greedy_verify)` is now submitted before scheduler `run_batch`; the previous CPU wakeup ordering problem is fixed.
- The remaining visible gap is no longer scheduler queue blocking.
- The gap contains required PhaseB device work (`jit_fused_draft_extend`, about `1.3 ms`) plus device scheduling / metadata / launch boundary before the next `jit_fused_greedy_verify`.
- Temporary skip diagnostics still showed a few reserve skips outside the captured steady window; remove the diagnostics only after the final chain design is accepted.

### TPU mock recheck

Added a lightweight TPU mock script:

```text
scripts/mock_spec_overlap_profile.py
```

The script supports 4-host JAX distributed execution and two modes:

- `split`: synthetic `verify_i -> draft_extend_i -> verify_{i+1}` dependency chain.
- `fused-chain`: synthetic `draft_extend + verify` inside one JIT module.

Validation:

```text
split run_id: 20260605_112500_mock_spec_overlap
split trace: /tmp/profile_20260605_112500_mock_spec_overlap_split
fused run_id: 20260605_113000_mock_spec_overlap
fused trace: /tmp/profile_20260605_113000_mock_spec_overlap_fused_chain
```

Result:

- Single-pod TPU mock is not viable on this pod setup; JAX waits for all TPU hosts.
- 4-host distributed mock runs successfully.
- Split mode reproduces the structural shape: verify kernels are separated by a draft kernel and a small dependency/launch interval.
- Fused-chain mode removes the separate draft kernel boundary from the inter-step view and leaves only small JIT-to-JIT gaps.
- The mock workload is intentionally tiny, so its absolute milliseconds are not comparable to the model. It is useful only for topology validation.

Updated conclusion:

- The current production implementation has completed the original scheduler-overlap split goal: scheduler no longer waits for full fused spec decode, PhaseA is published early, PhaseB no longer blocks CPU batching, and same-batch chained verify is submitted before scheduler `run_batch`.
- Complete xprof-level bubble elimination is not achieved by further PhaseA/PhaseB queue ordering changes.
- The remaining work is a topology change at the device launch boundary:
  - either fuse `draft_extend -> next verify` into a coarser device/JIT chain;
  - or prebuild/cache enough next-verify device metadata so the queued verify can start immediately after `jit_fused_draft_extend`, with no Python-side or PJRT launch tail on the critical path.
- The true `jit_fused_draft_extend` kernel should be counted as useful spec work, not idle bubble. The remaining idle target is the post-draft scheduling/launch gap before next verify.

### Remaining work after this checkpoint

1. Remove temporary diagnostic logs/TraceAnnotations only after the final topology is chosen:
   - `same_batch_chain_peek_skip:*`
   - `same_batch_chain_build_skip:*`
   - related `logger.info` skip logs.
2. Decide between two implementation paths:
   - a coarser fused chain JIT for `draft_extend + next verify`;
   - or a stronger same-batch future relay that precomputes/caches next-verify metadata and submits the verify as a device dependency with minimal post-draft host work.
3. Validate the chosen path with:
   - CPU focused tests;
   - 4-rank decode32 steady profile after the first `Decode batch. #running-req: 32` window;
   - throughput clearly improved versus the current commit;
   - accept ratio not lower for identical deterministic prompts;
   - later GSM8K score non-regression.
4. Fix the allocator/KV leak after overlap topology is accepted. The leak remains tracked separately and is not used to judge the current bubble-focused profiles.

---

## 2026-06-05 logits metadata exact-value cache

Change:

- Extended `_logits_metadata_from_model_worker_batch_preserve_device()` with an optional `cache_owner`.
- Hot fused draft_extend and target verify paths pass the model runner as cache owner.
- Stable host ndarray fields in `LogitsMetadata` now reuse exact-value device arrays:
  - `extend_seq_lens`
  - `logits_indices`
  - `extend_input_logprob_token_ids`
- The cache is shape/dtype/content checked through the existing exact host-device cache helper, so dynamic values still get refreshed when content changes.

CPU validation:

```text
sgl_jax/test/speculative/test_spec_overlap_split.py -q
59 passed in 8.38s
```

4-rank e2e profile:

```text
run_id: 20260605_114500_logits_meta_cache
profile: /tmp/profile_20260605_114500_logits_meta_cache/decode6_after_decode32_12
gaps_ms [4.032, 3.855, 3.083, 3.931, 3.698, 3.069]
```

Interpretation:

- This is a small improvement over `20260605_110250_stash_before_publish` (`[4.174, 4.005, 3.372, 3.532, 3.451, 3.344]`), but it is not enough to claim bubble elimination.
- Trace shows the cache changes convert some logits metadata conversions into short cache checks.
- The remaining tail is still dominated by:
  - `get_eagle_forward_metadata` around `0.46-0.66 ms`;
  - `device_array` / `batched_device_put` around `1.0-1.3 ms`;
  - `PjitFunction(fused_greedy_verify)` launch around `0.25-1.9 ms`;
  - the true `jit_fused_draft_extend` dependency, which is useful work and should not be counted as idle.
- The run later hit the known `token_to_kv_pool_allocator memory leak detected` after the profile was already captured. This remains a separate leak/KV accounting issue, not evidence that the overlap path failed.

Updated next step:

- More exact-value metadata caching may shave tenths of milliseconds, but it will not completely remove the remaining gap.
- To finish the objective, the next implementation should move to the larger topology item already identified:
  - build a coarser `draft_extend + next verify` chain, or
  - precompute/submit next-verify metadata and JIT launch as a true device dependency so no Python/PJRT launch tail remains after `jit_fused_draft_extend`.
