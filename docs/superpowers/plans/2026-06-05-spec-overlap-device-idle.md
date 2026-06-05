# Spec Overlap Device Idle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce speculative decode scheduler overlap device idle tail to tens of microseconds while preserving deterministic greedy accept behavior and final GSM8K correctness.

**Architecture:** Keep the existing split verify/Phase-B overlap path. Move same-batch chained verify host construction, metadata, and PJRT submit preparation ahead of the Phase-B device dependency, then relay Phase-B device handles directly into the next verify input so the device queue runs `verify -> draft_extend/gather -> next verify` with no millisecond-scale host idle tail.

**Tech Stack:** Python, JAX/PJRT, TPU xprof traces, existing SGLang-JAX scheduler/speculative decode code, `pytest`.

---

## Baseline And Scope

The authoritative starting point is:

- Worktree: `/Users/niu/code/sglang-jax/.worktrees/spec-overlap-bubble-followup-codex`
- Base branch: `origin/dev/fused-greedy-device-chain-verify-inputs-codex`
- Current base commit: `aafd1eb8855d893628eb462b1041d1804a6c75dc`
- Handoff: `docs/handoff/2026-06-05-spec-overlap-bubble-handoff-zh.md`

Primary metric:

- Device idle tail between useful TPU work should be tens of microseconds.

Secondary metric:

- Verify-to-verify wall gap is expected to remain around the real device work floor, currently about `1.8-2.0 ms`, because it includes `jit_fused_draft_extend` and gather/broadcast work.

Out of scope for the first optimization loop:

- Do not redesign speculative decoding.
- Do not add a new user-facing spec-overlap flag.
- Do not treat KV allocator leak/OOM as an overlap failure during bubble profiling; fix it after the device-idle path is validated.

## File Structure

- Modify: `python/sgl_jax/srt/managers/tp_worker_overlap_thread.py`
  - Owns ordered worker-thread overlap orchestration, pending Phase-B result handling, same-batch chained candidate prebuild, and chained verify enqueue.

- Modify: `python/sgl_jax/srt/speculative/draft_extend_fused.py`
  - Owns fused greedy verify enqueue/materialization, fused draft_extend dispatch/materialization, and any reusable prepared verify launch payload.

- Modify: `python/sgl_jax/srt/managers/scheduler.py`
  - Owns scheduler-side same-batch chain eligibility, reservation preview, pending Phase-B writeback, and run_batch integration.

- Modify: `python/sgl_jax/srt/layers/attention/flashattention_backend.py`
  - Only if metadata reuse/preparation requires an explicit helper or cache boundary around `get_eagle_forward_metadata`.

- Modify: `python/sgl_jax/test/speculative/test_spec_overlap_split.py`
  - Add focused CPU tests for prebuild ordering, prepared verify payload reuse, and relay guard behavior.

- Modify as needed: `python/sgl_jax/test/speculative/test_eagle_utils.py`
  - Keep fused greedy correctness coverage aligned with any `EagleDraftInput` or materialization changes.

---

## Task 1: Confirm Clean Baseline

**Files:**
- Read: `docs/handoff/2026-06-05-spec-overlap-bubble-handoff-zh.md`
- Test: `python/sgl_jax/test/speculative/test_spec_overlap_split.py`
- Test: `python/sgl_jax/test/speculative/test_eagle_utils.py`

- [ ] **Step 1: Confirm branch and dirty state**

Run:

```bash
cd /Users/niu/code/sglang-jax/.worktrees/spec-overlap-bubble-followup-codex
git status --short --branch
git rev-parse HEAD
git rev-parse origin/dev/fused-greedy-device-chain-verify-inputs-codex
```

Expected:

```text
## dev/spec-overlap-bubble-followup-codex...origin/dev/fused-greedy-device-chain-verify-inputs-codex
?? docs/handoff/
?? docs/superpowers/
aafd1eb8855d893628eb462b1041d1804a6c75dc
aafd1eb8855d893628eb462b1041d1804a6c75dc
```

- [ ] **Step 2: Run CPU split-overlap tests locally or on rank0 CPU**

Preferred rank0 pod command:

```bash
kubectl exec perf-16-0-jgb5c -c jax-tpu -- bash -lc '
cd /tmp/sglang-jax/python &&
PYTHONPATH=/tmp/sglang-jax/python JAX_PLATFORMS=cpu \
/opt/venv/bin/uv run --active pytest \
  sgl_jax/test/speculative/test_spec_overlap_split.py -q
'
```

Expected: all tests pass.

- [ ] **Step 3: Run fused greedy utility tests**

```bash
kubectl exec perf-16-0-jgb5c -c jax-tpu -- bash -lc '
cd /tmp/sglang-jax/python &&
PYTHONPATH=/tmp/sglang-jax/python JAX_PLATFORMS=cpu \
/opt/venv/bin/uv run --active pytest \
  sgl_jax/test/speculative/test_eagle_utils.py -q
'
```

Expected: all tests pass. If `test_eagle_utils.py` is absent on the pod for this branch, sync the file first or run the local equivalent with the same `PYTHONPATH` and `JAX_PLATFORMS=cpu`.

- [ ] **Step 4: Commit the plan**

```bash
git add docs/handoff/2026-06-05-spec-overlap-bubble-handoff-zh.md \
  docs/superpowers/plans/2026-06-05-spec-overlap-device-idle.md
git commit -m "docs(spec): plan device-idle overlap followup"
```

Expected: commit contains only the handoff and this plan.

---

## Task 2: Add CPU Coverage For Phase-A Prebuild Ordering

**Files:**
- Modify: `python/sgl_jax/test/speculative/test_spec_overlap_split.py`
- Modify: `python/sgl_jax/srt/managers/tp_worker_overlap_thread.py`

- [ ] **Step 1: Add a test proving Phase-A prebuild can build a same-layout candidate before Phase-B materialization**

Append this test to `python/sgl_jax/test/speculative/test_spec_overlap_split.py`:

```python
def test_phase_a_prebuild_builds_same_batch_candidate_from_padded_new_seq_lens():
    from sgl_jax.srt.managers.tp_worker_overlap_thread import TpModelWorkerClient

    worker = TpModelWorkerClient.__new__(TpModelWorkerClient)
    req_pool = np.array([10, 11, 12, 13], dtype=np.int32)
    spec_info = EagleDraftInput(
        allocate_lens=np.full((4,), 64, dtype=np.int32),
        verify_write_lens=np.array([20, 21, 22, 23], dtype=np.int32),
    )
    model_worker_batch = SimpleNamespace(
        allow_same_batch_spec_chain=True,
        req_pool_indices=req_pool,
        same_batch_chain_req_pool_indices=req_pool.copy(),
        same_batch_chain_out_cache_loc_chunks=[
            np.arange(4, dtype=np.int32),
            np.arange(4, 8, dtype=np.int32),
        ],
        same_batch_chain_verify_write_lens=np.array([24, 25, 26, 27], dtype=np.int32),
        same_batch_chain_allocate_lens=np.full((4,), 64, dtype=np.int32),
        spec_info_padded=spec_info,
        seq_lens=np.array([16, 17, 18, 19], dtype=np.int32),
        seq_lens_sum=70,
        out_cache_loc=np.arange(8, dtype=np.int32),
        bid=9,
    )
    verify_result = SimpleNamespace(
        padded_new_seq_lens_host=np.array([21, 22, 23, 24], dtype=np.int32)
    )

    candidate = worker._prebuild_same_batch_spec_chain_candidate_after_phase_a(
        model_worker_batch,
        verify_result,
    )

    assert candidate is not None
    np.testing.assert_array_equal(candidate.req_pool_indices, req_pool)
    np.testing.assert_array_equal(candidate.seq_lens, verify_result.padded_new_seq_lens_host)
    assert candidate.seq_lens_sum == int(verify_result.padded_new_seq_lens_host.sum())
    assert candidate.skip_fused_verify_padding_for_decode is True
```

- [ ] **Step 2: Run the new test and confirm current behavior**

```bash
PYTHONPATH=python JAX_PLATFORMS=cpu pytest \
  python/sgl_jax/test/speculative/test_spec_overlap_split.py::test_phase_a_prebuild_builds_same_batch_candidate_from_padded_new_seq_lens -q
```

Expected: PASS. If it fails because the synthetic batch lacks an attribute, add only that explicit attribute to the test fixture.

- [ ] **Step 3: Add trace annotations to distinguish prebuild timing**

In `python/sgl_jax/srt/managers/tp_worker_overlap_thread.py`, wrap `_prebuild_same_batch_spec_chain_candidate_after_phase_a` body:

```python
with jax.profiler.TraceAnnotation("prebuild_same_batch_spec_chain_candidate_after_phase_a"):
    ...
```

Expected: xprof can distinguish Phase-A prebuild from Phase-B pending dispatch and chained verify enqueue.

- [ ] **Step 4: Run focused tests**

```bash
PYTHONPATH=python JAX_PLATFORMS=cpu pytest \
  python/sgl_jax/test/speculative/test_spec_overlap_split.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/sgl_jax/test/speculative/test_spec_overlap_split.py \
  python/sgl_jax/srt/managers/tp_worker_overlap_thread.py
git commit -m "test(spec): cover same-batch chain phase-a prebuild"
```

---

## Task 3: Prepare Verify Launch Metadata Before Phase-B Completion

**Files:**
- Modify: `python/sgl_jax/srt/speculative/draft_extend_fused.py`
- Modify: `python/sgl_jax/srt/managers/tp_worker_overlap_thread.py`
- Modify: `python/sgl_jax/test/speculative/test_spec_overlap_split.py`

- [ ] **Step 1: Introduce a small prepared verify payload type**

In `python/sgl_jax/srt/speculative/draft_extend_fused.py`, add near `FusedGreedyVerifyPhaseAsync`:

```python
@dataclasses.dataclass(frozen=True)
class PreparedFusedGreedyVerifyLaunch:
    model_worker_batch: object
    padded_allocate_lens: object
    compact_allocate_lens: object
    target_forward_batch: object
    target_logits_metadata: object
    previous_verified_id: object
    previous_token_list: object
    return_target_logits: bool
    return_target_hidden: bool
```

Use `object` annotations to avoid importing runtime-only types into tests and to match the existing loose worker-batch style in this file.

- [ ] **Step 2: Extract host/metadata preparation from `spec_decode_verify_phase_enqueue`**

Add a helper in `python/sgl_jax/srt/speculative/draft_extend_fused.py`:

```python
def prepare_fused_greedy_verify_launch(
    spec_worker,
    model_worker_batch,
    padded_allocate_lens,
    compact_allocate_lens,
):
    draft_worker = spec_worker.draft_worker
    target_worker = spec_worker.target_worker
    target_mr = target_worker.model_runner
    draft_verify_write_lens = getattr(
        model_worker_batch.spec_info_padded,
        "verify_write_lens",
        None,
    )
    previous_verified_id, previous_token_list = _prepare_topk1_verify_placeholders_from_draft_state(
        draft_worker, model_worker_batch
    )
    spec_info = model_worker_batch.spec_info_padded
    spec_info.allocate_lens = padded_allocate_lens
    spec_info.prepare_for_verify(model_worker_batch, spec_worker.page_size, target_worker)
    target_mr.attn_backend.forward_metadata = target_mr.attn_backend.get_eagle_forward_metadata(
        model_worker_batch
    )
    target_forward_batch = _forward_batch_init_new_preserve_device(model_worker_batch, target_mr)
    target_forward_batch.bid = model_worker_batch.bid
    target_logits_metadata = _logits_metadata_from_model_worker_batch_preserve_device(
        model_worker_batch,
        spec_worker.mesh,
        cache_owner=target_mr,
    )
    model_worker_batch.spec_info_padded.verify_write_lens = draft_verify_write_lens
    return PreparedFusedGreedyVerifyLaunch(
        model_worker_batch=model_worker_batch,
        padded_allocate_lens=padded_allocate_lens,
        compact_allocate_lens=compact_allocate_lens,
        target_forward_batch=target_forward_batch,
        target_logits_metadata=target_logits_metadata,
        previous_verified_id=previous_verified_id,
        previous_token_list=previous_token_list,
        return_target_logits=bool(
            getattr(model_worker_batch, "return_logprob", False)
            or getattr(model_worker_batch, "return_output_logprob_only", False)
        ),
        return_target_hidden=bool(getattr(model_worker_batch, "return_hidden_states", False)),
    )
```

- [ ] **Step 3: Make enqueue accept a prepared payload**

Change `spec_decode_verify_phase_enqueue` signature to accept an optional keyword:

```python
def spec_decode_verify_phase_enqueue(
    spec_worker,
    model_worker_batch,
    padded_allocate_lens,
    compact_allocate_lens,
    *,
    prepared_launch=None,
):
```

Inside it, if `prepared_launch is None`, call `prepare_fused_greedy_verify_launch(...)`. Otherwise use the payload's `target_forward_batch`, `target_logits_metadata`, `previous_verified_id`, `previous_token_list`, `return_target_logits`, and `return_target_hidden`.

Preserve these local values from the current function before calling the JIT:

```python
selector = np.asarray(model_worker_batch.logits_indices_selector)
seq_lens_host = np.asarray(model_worker_batch.seq_lens)
draft_verify_write_lens = getattr(model_worker_batch.spec_info_padded, "verify_write_lens", None)
```

Expected: behavior is unchanged when no prepared payload is passed.

- [ ] **Step 4: Attach prepared launch to prebuilt same-batch candidate**

In `tp_worker_overlap_thread.py`, after `_prebuild_same_batch_spec_chain_candidate_after_phase_a` gets `candidate_batch`, call:

```python
from sgl_jax.srt.speculative.draft_extend_fused import prepare_fused_greedy_verify_launch

prepared_launch = prepare_fused_greedy_verify_launch(
    self.spec_worker,
    candidate_batch,
    getattr(candidate_batch.spec_info_padded, "allocate_lens"),
    getattr(candidate_batch.spec_info_padded, "allocate_lens"),
)
candidate_batch.prepared_fused_greedy_verify_launch = prepared_launch
```

Do this inside a trace annotation:

```python
with jax.profiler.TraceAnnotation("prepare_chained_verify_launch_after_phase_a"):
    ...
```

If preparation raises because Phase-B device fields are still missing, return the candidate without a prepared payload and let Task 4 split stable metadata from Phase-B handles.

- [ ] **Step 5: Use prepared launch when enqueueing prebuilt candidate**

In `_stash_prebuilt_same_batch_spec_chain_candidate`, call:

```python
verify_async_result = self.spec_worker.forward_batch_speculative_verify_phase_enqueue(
    candidate_batch,
    prepared_launch=getattr(candidate_batch, "prepared_fused_greedy_verify_launch", None),
)
```

If `BaseSpecWorker.forward_batch_speculative_verify_phase_enqueue` does not yet forward `prepared_launch`, update that method to accept the keyword and pass it through to `spec_decode_verify_phase_enqueue`.

- [ ] **Step 6: Add a CPU test for the keyword passthrough contract**

In `test_spec_overlap_split.py`, add:

```python
def test_prebuilt_candidate_keeps_prepared_verify_launch_payload():
    candidate = SimpleNamespace(prepared_fused_greedy_verify_launch=object())
    assert getattr(candidate, "prepared_fused_greedy_verify_launch", None) is not None
```

This is intentionally shallow; the expensive behavior is verified by xprof annotations in Task 6.

- [ ] **Step 7: Run tests**

```bash
PYTHONPATH=python JAX_PLATFORMS=cpu pytest \
  python/sgl_jax/test/speculative/test_spec_overlap_split.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add python/sgl_jax/srt/speculative/draft_extend_fused.py \
  python/sgl_jax/srt/managers/tp_worker_overlap_thread.py \
  python/sgl_jax/srt/speculative/base_worker.py \
  python/sgl_jax/test/speculative/test_spec_overlap_split.py
git commit -m "perf(spec): preprepare chained verify launch metadata"
```

---

## Task 4: Relay Phase-B Device Handles Into The Prepared Candidate

**Files:**
- Modify: `python/sgl_jax/srt/managers/tp_worker_overlap_thread.py`
- Modify: `python/sgl_jax/srt/speculative/draft_extend_fused.py`
- Modify: `python/sgl_jax/test/speculative/test_spec_overlap_split.py`

- [ ] **Step 1: Define the exact Phase-B fields that may be patched after prebuild**

Use only these fields when patching the candidate's `spec_info_padded` in `_stash_prebuilt_same_batch_spec_chain_candidate`:

```python
PHASE_B_DEVICE_RELAY_FIELDS = (
    "topk_index",
    "topk_p",
    "verified_id",
    "hidden_states",
    "previous_token_list",
)
```

Keep `new_seq_lens`, `allocate_lens`, and `verify_write_lens` on the prebuilt candidate from Phase-A/pre-reservation data.

- [ ] **Step 2: Add a focused test for field patching**

Append:

```python
def test_phase_b_relay_patches_only_device_dependent_fields():
    relayed = EagleDraftInput(
        topk_index=np.array([[[1]], [[2]]], dtype=np.int32),
        topk_p=np.ones((2, 1, 1), dtype=np.float32),
        hidden_states=np.ones((2, 4), dtype=np.float32),
        verified_id=np.array([7, 8], dtype=np.int32),
        new_seq_lens=np.array([100, 200], dtype=np.int32),
        allocate_lens=np.array([300, 400], dtype=np.int32),
    )
    relayed.previous_token_list = np.array([[1], [2]], dtype=np.int32)
    candidate_spec = EagleDraftInput(
        new_seq_lens=np.array([10, 20], dtype=np.int32),
        allocate_lens=np.array([30, 40], dtype=np.int32),
    )

    for field in ("topk_index", "topk_p", "verified_id", "hidden_states", "previous_token_list"):
        setattr(candidate_spec, field, getattr(relayed, field))

    np.testing.assert_array_equal(candidate_spec.new_seq_lens, np.array([10, 20], dtype=np.int32))
    np.testing.assert_array_equal(candidate_spec.allocate_lens, np.array([30, 40], dtype=np.int32))
    np.testing.assert_array_equal(candidate_spec.verified_id, np.array([7, 8], dtype=np.int32))
```

- [ ] **Step 3: Patch `_stash_prebuilt_same_batch_spec_chain_candidate`**

Replace its broad field copy with:

```python
for field in PHASE_B_DEVICE_RELAY_FIELDS:
    value = getattr(padded_next_draft_input, field, None)
    if value is not None:
        setattr(candidate_spec_info, field, value)
```

Do not overwrite prebuilt `new_seq_lens`, `allocate_lens`, or `verify_write_lens` except for an explicit assertion that shapes match.

- [ ] **Step 4: Preserve device handles**

Do not call `np.asarray`, `jax.device_get`, or `copy_to_host_async` on relayed Phase-B fields in `_stash_prebuilt_same_batch_spec_chain_candidate`. The relayed values should remain JAX arrays/futures so PJRT orders the dependency.

- [ ] **Step 5: Run tests**

```bash
PYTHONPATH=python JAX_PLATFORMS=cpu pytest \
  python/sgl_jax/test/speculative/test_spec_overlap_split.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add python/sgl_jax/srt/managers/tp_worker_overlap_thread.py \
  python/sgl_jax/test/speculative/test_spec_overlap_split.py
git commit -m "perf(spec): relay phase-b device handles into chained verify"
```

---

## Task 5: Ensure Chained Verify Submit Is A Hot Path

**Files:**
- Modify: `python/sgl_jax/srt/speculative/draft_extend_fused.py`
- Modify: `python/sgl_jax/srt/layers/attention/flashattention_backend.py` only if required by profiling
- Modify: `python/sgl_jax/test/speculative/test_spec_overlap_split.py`

- [ ] **Step 1: Add trace annotations around hot-path boundaries**

In `spec_decode_verify_phase_enqueue`, annotate:

```python
with jax.profiler.TraceAnnotation("prepare_fused_greedy_verify_launch"):
    ...
with jax.profiler.TraceAnnotation("submit_fused_greedy_verify_jit"):
    ...
```

Expected: xprof can separate metadata prep from actual PJRT submit.

- [ ] **Step 2: Confirm metadata prep does not run after Phase-B relay for prebuilt candidates**

Run one 4-rank profile after Tasks 3-4. In xprof, inspect the interval from `jit_fused_draft_extend` end to next `jit_fused_greedy_verify` start.

Expected:

```text
prepare_chained_verify_launch_after_phase_a occurs before or during jit_fused_draft_extend.
prepare_fused_greedy_verify_launch is absent or tiny after Phase-B relay.
submit_fused_greedy_verify_jit is the only host segment immediately before next verify.
```

- [ ] **Step 3: If `get_eagle_forward_metadata` remains in the tail, cache the safe same-batch metadata**

Only if profiling proves it remains after Phase-B, add a same-batch cache key to `model_worker_batch`:

```python
candidate.same_batch_chain_forward_metadata = target_mr.attn_backend.forward_metadata
```

Then in prepared-launch enqueue, reuse it instead of calling `get_eagle_forward_metadata` again.

Expected: `get_eagle_forward_metadata` no longer appears in the Phase-B-to-next-verify idle tail.

- [ ] **Step 4: If `_forward_batch_init_new_preserve_device` remains in the tail, reuse prepared `target_forward_batch`**

Ensure `PreparedFusedGreedyVerifyLaunch.target_forward_batch` is used directly and not rebuilt.

Expected: `_forward_batch_init_new_preserve_device` no longer appears in the Phase-B-to-next-verify idle tail.

- [ ] **Step 5: Run CPU tests**

```bash
PYTHONPATH=python JAX_PLATFORMS=cpu pytest \
  python/sgl_jax/test/speculative/test_spec_overlap_split.py \
  python/sgl_jax/test/speculative/test_eagle_utils.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add python/sgl_jax/srt/speculative/draft_extend_fused.py \
  python/sgl_jax/srt/layers/attention/flashattention_backend.py \
  python/sgl_jax/test/speculative/test_spec_overlap_split.py
git commit -m "perf(spec): keep chained verify submit on hot path"
```

---

## Task 6: TPU Profile Device Idle And Iterate

**Files:**
- Modify: profiling scripts only if a local helper is added
- Read: `docs/handoff/2026-06-05-spec-overlap-bubble-handoff-zh.md`

- [ ] **Step 1: Sync changed files to all pods**

```bash
cd /Users/niu/code/sglang-jax/.worktrees/spec-overlap-bubble-followup-codex
bash -lc '
PODS=(perf-16-0-jgb5c perf-16-1-zhn6d perf-16-2-hs7kc perf-16-3-vqw2x)
FILES=(
  python/sgl_jax/srt/layers/attention/flashattention_backend.py
  python/sgl_jax/srt/managers/scheduler.py
  python/sgl_jax/srt/managers/tp_worker_overlap_thread.py
  python/sgl_jax/srt/speculative/base_worker.py
  python/sgl_jax/srt/speculative/draft_extend_fused.py
  python/sgl_jax/test/speculative/test_spec_overlap_split.py
  python/sgl_jax/test/speculative/test_eagle_utils.py
)
for pod in "${PODS[@]}"; do
  echo "sync:$pod"
  COPYFILE_DISABLE=1 tar cf - "${FILES[@]}" |
    kubectl exec -i "$pod" -c jax-tpu -- tar xf - -C /tmp/sglang-jax/
done
'
```

- [ ] **Step 2: Run rank0 CPU tests on pod**

```bash
kubectl exec perf-16-0-jgb5c -c jax-tpu -- bash -lc '
cd /tmp/sglang-jax/python &&
PYTHONPATH=/tmp/sglang-jax/python JAX_PLATFORMS=cpu \
/opt/venv/bin/uv run --active pytest \
  sgl_jax/test/speculative/test_spec_overlap_split.py \
  sgl_jax/test/speculative/test_eagle_utils.py -q
'
```

Expected: PASS.

- [ ] **Step 3: Restart 4-rank server without `--disable-overlap-schedule`**

Use the exact server command from the handoff. Keep:

```text
--decode-log-interval=1
--speculative-algorithm NEXTN
--speculative-eagle-topk 1
--speculative-num-steps 3
--speculative-num-draft-tokens 4
```

Expected: rank0 log reaches ready state and no fatal errors appear in any rank log.

- [ ] **Step 4: Run 32 concurrent deterministic greedy curl test**

Use the handoff curl block with:

```json
"temperature": 0,
"max_tokens": 512
```

Expected: rank0 logs steady decode batches with `#running-req: 32`.

- [ ] **Step 5: Capture a 5-6 step steady decode profile**

Use `/start_profile` or `python -m sgl_jax.profiler` from the handoff after the first new `Decode batch. #running-req: 32`.

Expected: downloaded profile contains `perf-16-0.trace.json.gz`.

- [ ] **Step 6: Analyze device idle**

Use a local trace script that reports:

```text
verify_end_to_draft_start_idle_us
draft_or_gather_end_to_next_verify_start_idle_us
verify_to_verify_gap_ms
device_busy_ms
idle_tail_ms
```

Expected:

```text
draft_or_gather_end_to_next_verify_start_idle_us p50 <= 100 us
draft_or_gather_end_to_next_verify_start_idle_us p90 <= 300 us
verify_to_verify_gap_ms remains near  useful work, around 1.8-2.5 ms
No cache-miss-scale stall
```

- [ ] **Step 7: Commit profile notes**

Add a short markdown note under `docs/handoff/` with run id, commit, throughput, accept-len, accept-ratio, and idle metrics.

```bash
git add docs/handoff/
git commit -m "docs(spec): record device-idle overlap profile"
```

---

## Task 7: Fix KV Allocator Leak After Device Idle Target Is Met

**Files:**
- Modify likely: `python/sgl_jax/srt/managers/schedule_batch.py`
- Modify likely: `python/sgl_jax/srt/managers/scheduler.py`
- Modify likely: memory-pool/allocator files found by searching `token_to_kv_pool_allocator memory leak detected`
- Test likely: `python/sgl_jax/test/mem_cache/test_swa_allocator.py`

- [ ] **Step 1: Locate leak assertion and accounting paths**

```bash
rg -n "token_to_kv_pool_allocator memory leak detected|memory leak detected|verify_write_lens|allocate_lens|free|evict|protected" python/sgl_jax/srt python/sgl_jax/test
```

Expected: identify the allocator assertion and all spec decode paths that allocate or release KV slots.

- [ ] **Step 2: Reproduce leak independently from overlap profile**

Run 32 concurrent greedy requests to completion with `max_tokens=512`.

Expected: either reproduce `token_to_kv_pool_allocator memory leak detected` or record that it no longer reproduces at the current commit.

- [ ] **Step 3: Write a focused accounting test before changing allocator logic**

Add or extend a test in `python/sgl_jax/test/mem_cache/test_swa_allocator.py` that constructs the exact allocate/free/reclaim sequence found in Step 1.

Expected before fix: FAIL due to leaked full/SWA slot accounting or mismatched expected/available/protected counts.

- [ ] **Step 4: Implement minimal accounting fix**

Change only the allocator or scheduler accounting path identified by the failing test. Do not change overlap timing code unless the leak root cause is directly caused by same-batch chained verify reservation.

- [ ] **Step 5: Run allocator tests**

```bash
PYTHONPATH=python JAX_PLATFORMS=cpu pytest \
  python/sgl_jax/test/mem_cache/test_swa_allocator.py -q
```

Expected: PASS.

- [ ] **Step 6: Re-run 32 concurrent greedy completion test**

Expected:

```text
No token_to_kv_pool_allocator memory leak detected
No allocator-accounting OOM near request tail
```

- [ ] **Step 7: Commit**

```bash
git add python/sgl_jax/test/mem_cache/test_swa_allocator.py \
  python/sgl_jax/srt
git commit -m "fix(spec): repair kv allocator accounting for overlapped decode"
```

---

## Task 8: Final Correctness And Performance Acceptance

**Files:**
- Modify: `docs/handoff/` final result note

- [ ] **Step 1: Run final 32 concurrent deterministic greedy throughput test**

Expected steady-state fields to record:

```text
commit
run_id
#running-req: 32 plateau throughput
accept-len
accept-ratio
device idle p50/p90
verify-to-verify gap p50/p90
```

- [ ] **Step 2: Run final steady decode profile**

Expected:

```text
No 25s cache miss
No millisecond-scale host idle tail between draft/gather and next verify
No allocator leak/OOM tail crash
```

- [ ] **Step 3: Run GSM8K evalscope**

```bash
RUN_ID=$(cat /tmp/current_specdecode_route_run_id)
kubectl exec perf-16-0-jgb5c -c jax-tpu -- bash -lc "
cd /tmp/sglang-jax/python &&
/opt/venv/bin/evalscope eval \
  --model /data/pc \
  --api-url http://127.0.0.1:30271/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --dataset-args '{\"gsm8k\":{\"dataset_id\":\"/tmp/gsm8k_eval_data\"}}' \
  --eval-batch-size 64 \
  --generation-config '{\"temperature\":0,\"max_tokens\":1024}' \
  2>&1 | tee /tmp/evalscope_gsm8k_${RUN_ID}.log
"
```

Expected: GSM8K does not regress versus the accepted baseline for the same model/config.

- [ ] **Step 4: Write final result note**

Create `docs/handoff/YYYY-MM-DD-spec-overlap-device-idle-result.md` with:

```markdown
# Spec Overlap Device Idle Result

- Commit:
- Run ID:
- Server flags:
- 32-concurrency steady throughput:
- accept-len:
- accept-ratio:
- verify-to-verify gap p50/p90:
- device idle tail p50/p90:
- Profile path:
- GSM8K result:
- KV allocator status:
- Known residual risk:
```

- [ ] **Step 5: Commit final results**

```bash
git add docs/handoff/
git commit -m "docs(spec): record final device-idle overlap validation"
```

---

## Self-Review

- Spec coverage: The plan covers the handoff's remaining `3-4 ms` gap, the real goal of tens-of-microseconds device idle, metadata/ForwardBatch prebuild, Phase-B device relay, hot submit path verification, KV allocator leak, and GSM8K final acceptance.
- Placeholder scan: No unresolved placeholder markers are used. Conditional profiling branches are explicit and tied to observed trace evidence.
- Type consistency: New prepared-launch payload is named `PreparedFusedGreedyVerifyLaunch`; all later references use the same name. Existing functions and fields match the current branch names found in source.
