# Fused Greedy Spec Decode Handoff - 2026-06-02 Context Switch

This handoff is for continuing the active goal in a new larger-context Codex
thread:

> Execute the plan until the whole greedy fixed-shape decode path is fused into
> one JIT.

The current work is not complete. The active branch has committed Step1/Step2
and initial Step3 work, plus uncommitted changes that move Step3 closer to one
JIT by fusing greedy verify postprocess into the MTP draft-extend JIT.

## Current Worktree

- Repo root: `/Users/niu/code/sglang-jax`
- Active worktree:
  `/Users/niu/code/sglang-jax/.worktrees/fused-greedy-spec-decode-step3`
- Branch: `dev/fused-greedy-spec-decode-step3`
- Base branch: `origin/dev/spec-decode-performance-optim`
- Branch status at handoff: ahead by 6 committed changes, with 5 modified files
  not committed.
- Do not use the original repo root for edits unless you intentionally switch
  worktrees.

Useful startup command for the new thread:

```bash
cd /Users/niu/code/sglang-jax/.worktrees/fused-greedy-spec-decode-step3
git status --short --branch
```

Expected status:

```text
## dev/fused-greedy-spec-decode-step3...origin/dev/spec-decode-performance-optim [ahead 6]
 M python/sgl_jax/srt/speculative/base_worker.py
 M python/sgl_jax/srt/speculative/draft_extend_fused.py
 M python/sgl_jax/srt/speculative/eagle_util.py
 M python/sgl_jax/test/speculative/test_eagle_utils.py
 M python/sgl_jax/test/speculative/test_fused_greedy_decode_step.py
```

Recent commits already on the branch:

```text
29fa0017 perf(spec): move greedy safe-index postprocess to device
85b773d7 perf(spec): add fused greedy decode step3 route
1d9b8440 perf(spec): gate fused greedy decode step3 path
16c445e1 perf(spec): jit greedy verify safe-index postprocess
f0195186 perf(spec): expose greedy sample device outputs
88d5b005 docs(spec): add greedy decode fusion plan and handoff
d4a580b2 origin/dev/spec-decode-performance-optim
```

Existing docs to read first:

- `docs/handoff_spec_decode_optim_2026_06_02.md`
- `docs/pitfalls_spec_decode_optim_2026_06_02.md`
- `docs/superpowers/plans/2026-06-02-fused-greedy-spec-decode-step3.md`

## User Constraints

- Scope is only the greedy fixed-shape path:
  `temperature=0`, `topk=1`, `spec_steps=3`, `speculative_num_draft_tokens=4`,
  padded decode batch `bsz=32`.
- Do not broaden to non-greedy, topk>1, variable shapes, or non-Step3 paths.
- Do not run local tests. The user explicitly said to test in pods with `uv`.
- If TPU test needs four hosts and is expensive, CPU verification in the pod is
  acceptable for focused unit tests.
- Use `/opt/venv/bin/uv` for long-running pod startup commands; plain `uv` is
  fine under `kubectl exec ... bash -c 'cd ... && uv run ...'`, but `bash -lc`
  or `nohup` may lose PATH and fail with `uv: command not found`.

## Current Pod State

Pods:

```text
perf-16-0-jgb5c
perf-16-1-zhn6d
perf-16-2-hs7kc
perf-16-3-vqw2x
```

Container name: `jax-tpu`.

At handoff, I killed all `sgl_jax.launch_server` processes and confirmed no
process lines remained:

```bash
for pod in perf-16-0-jgb5c perf-16-1-zhn6d perf-16-2-hs7kc perf-16-3-vqw2x; do
  echo $pod
  kubectl exec $pod -c jax-tpu -- bash -c \
    "ps -ef | grep 'sgl_jax.launch_server' | grep -v grep || true"
done
```

Expected output is just the pod names.

## What The Committed Work Already Did

The 6 committed changes before the current uncommitted patch:

1. Added the new plan and copied the earlier handoff/pitfall docs.
2. Added `GreedySampleDeviceOutputs` and
   `greedy_sample_device_outputs(...)` in `eagle_util.py`, mirroring greedy
   `EagleVerifyInput.sample()` without host `np.asarray`.
3. Added `_greedy_verify_postprocess_jit(...)` in
   `draft_extend_fused.py` for safe-index gather and next-draft fields.
4. Added `_can_use_fused_greedy_decode_step3(...)` in `base_worker.py`,
   gating only greedy topk1/spec3/draft_tokens4/bs32.
5. Routed `MultiLayerDraftWorker.draft_extend_for_decode(...)` to
   `draft_extend_for_decode_fused_step3(...)` under the narrow predicate.
6. Moved greedy sample and safe-index outputs onto device for Step3 path.

Those commits had pod CPU focused tests passing before this handoff work.

## Current Uncommitted Changes

### 1. `eagle_util.py`

`greedy_sample_device_outputs(...)` now accepts optional `mesh: Mesh | None`.

Root cause for this change:

- TPU `SPEC_DECODE` precompile failed at `bs=32` with:

```text
ValueError: The context mesh cannot be empty. Use `jax.set_mesh(mesh)` to enter into a mesh context
```

- Stack:
  `BaseSpecWorker.verify() -> greedy_sample_device_outputs() -> verify_tree_greedy() -> jax.shard_map(...)`
- Old `EagleVerifyInput.sample()` enters `jax.sharding.use_mesh(mesh)` or
  `jax.set_mesh(mesh)` before calling `verify_tree_greedy`.
- The new helper had been split out but lacked that mesh context.

Fix implemented:

- Add optional `mesh` parameter.
- If `mesh is not None`, call `verify_tree_greedy` inside the same
  `use_mesh` / `set_mesh` fallback pattern as old `sample()`.

### 2. `base_worker.py`

The Step3 greedy path now passes `mesh=self.mesh` into
`greedy_sample_device_outputs(...)`.

This is the source-level fix for the TPU precompile failure above.

### 3. `draft_extend_fused.py`

This is the largest uncommitted change.

Current goal of this patch:

- Stop doing Step3 as:
  `_greedy_verify_postprocess_jit(...)` then
  `draft_extend_for_decode_fused(...)`.
- Instead call a Step3-specific fused implementation where the greedy safe-index
  gather and N-layer MTP draft extend are in one JIT.

New pieces:

- `GreedyStep3DraftInputs(NamedTuple)`
- `_greedy_step3_prepare_draft_inputs(...)`
  - Computes safe index.
  - Gathers target hidden and positions.
  - Computes `new_seq_lens`, `select_index`, `sel_pos`.
- `_build_fused_greedy_step3_draft_extend_jit(...)`
  - Builds a Step3-specific JIT.
  - Takes target hidden, accept index/lens, verified id.
  - Runs `_greedy_step3_prepare_draft_inputs(...)` inside the JIT.
  - Runs all MTP layers in one loop.
  - Returns layer0 hidden, stacked topk p/index, pool updates, select index,
    verified id, accept lens, and new seq lens.
- `_device_array_preserve_device(...)`
  - Avoids `device_array(np.asarray(...))` pulling JAX arrays back to host.
- `_logits_metadata_from_model_worker_batch_preserve_device(...)`
  - Builds `LogitsMetadata` while preserving device arrays.
- `_forward_batch_init_new_preserve_device(...)`
  - Mirrors `ForwardBatch.init_new(...)` but preserves device arrays instead of
    forcing `np.asarray`.
- `_prepare_step3_model_worker_batch_for_draft_extend(...)`
  - Does the remaining host-side `ModelWorkerBatch` mutation needed to build
    draft attention metadata.
  - Sets DRAFT_EXTEND mode and fixed extend lengths.
- `_draft_extend_for_decode_fused_step3_impl(...)`
  - Step3 host wrapper that prepares metadata, calls the new fused Step3 JIT,
    replaces memory pools, and only then materializes final small scheduler
    outputs.
- `draft_extend_for_decode_fused_step3(...)`
  - Now delegates directly to `_draft_extend_for_decode_fused_step3_impl(...)`.
  - No longer calls `_greedy_verify_postprocess_jit(...)` first.

Important caveat:

- This is still not the final "whole decode one JIT". Target verify forward
  still runs before this function, and host still constructs attention metadata.
- This patch fuses the Step3 verify-postprocess plus MTP draft-extend boundary.

### 4. Tests

`test_eagle_utils.py`:

- Adds `test_greedy_sample_device_outputs_enters_mesh_context`.
- This test monkeypatches `jax.set_mesh` and `verify_tree_greedy` to confirm
  the helper enters a mesh context before calling the greedy verify kernel.

`test_fused_greedy_decode_step.py`:

- Adds `test_greedy_step3_prepare_draft_inputs_matches_safe_index_logic`.
- Replaces the old route test with
  `test_step3_entrypoint_does_not_split_postprocess_before_fused_extend`.
- The route test now fails if `draft_extend_for_decode_fused_step3(...)` calls
  `_greedy_verify_postprocess_jit(...)` before draft extend.

## Verified Tests

Run from local worktree, but executing inside pod:

```bash
tar cf - -C . \
  python/sgl_jax/srt/speculative/eagle_util.py \
  python/sgl_jax/srt/speculative/base_worker.py \
  python/sgl_jax/srt/speculative/draft_extend_fused.py \
  python/sgl_jax/test/speculative/test_eagle_utils.py \
  python/sgl_jax/test/speculative/test_fused_greedy_decode_step.py | \
  kubectl exec -i perf-16-0-jgb5c -c jax-tpu -- tar xf - -C /tmp/sglang-jax/

kubectl exec perf-16-0-jgb5c -c jax-tpu -- bash -c '
cd /tmp/sglang-jax/python &&
PYTHONPATH=/tmp/sglang-jax/python JAX_PLATFORMS=cpu \
uv run --active pytest \
  sgl_jax/test/speculative/test_eagle_utils.py::TestVerifyTree::test_greedy_sample_device_outputs_enters_mesh_context \
  sgl_jax/test/speculative/test_eagle_utils.py::TestVerifyTree::test_verify_tree_greedy_device_outputs_match_host_postprocess \
  sgl_jax/test/speculative/test_fused_greedy_decode_step.py \
  -q'
```

Observed result:

```text
7 passed in 8.60s
```

Do not run full `test_eagle_utils.py` on CPU as the existing
`test_verify_tree_greedy` uses Pallas/TPU assumptions and can fail on CPU for
pre-existing reasons.

## TPU Attempts And Failures

### Attempt 1: full precompile, before mesh fix

Started 4-host server with normal default precompile:

- `SPEC_EXTEND`: all token buckets
- `SPEC_DECODE`: bs buckets `[4, 8, 16, 32]`

It reached `SPEC_DECODE` bs=32 and failed:

```text
ValueError: The context mesh cannot be empty. Use `jax.set_mesh(mesh)` to enter into a mesh context
```

Failure stack:

```text
scheduler.py run_scheduler_process
  -> Scheduler.__init__
  -> draft_worker.run_spec_decode_precompile()
  -> precompile_spec_decode()
  -> forward_batch_speculative_generation()
  -> BaseSpecWorker.verify()
  -> greedy_sample_device_outputs()
  -> verify_tree_greedy()
  -> jax.shard_map(...)
```

That caused the mesh-context fix described above.

### Attempt 2: reduced precompile, after mesh fix

To reduce startup cost for the curl smoke test, use:

```text
--precompile-bs-paddings 32 --precompile-token-paddings 256
```

This was started after syncing the mesh fix and Step3 fused changes. It had not
finished before the user asked to stop and switch context. I killed all server
processes afterwards.

Important: the mesh fix has not yet been TPU-verified after the reduced-bucket
restart. Only the pod CPU focused tests are green after the fix.

## Recommended Next Steps

1. Start in the worktree and inspect:

```bash
cd /Users/niu/code/sglang-jax/.worktrees/fused-greedy-spec-decode-step3
git status --short --branch
git diff --stat
```

2. Read these docs:

```bash
sed -n '1,260p' docs/superpowers/plans/2026-06-02-fused-greedy-spec-decode-step3.md
sed -n '1,260p' docs/handoff_fused_greedy_decode_context_switch_2026_06_02.md
```

3. Run the focused pod CPU tests again:

```bash
tar cf - -C . \
  python/sgl_jax/srt/speculative/eagle_util.py \
  python/sgl_jax/srt/speculative/base_worker.py \
  python/sgl_jax/srt/speculative/draft_extend_fused.py \
  python/sgl_jax/test/speculative/test_eagle_utils.py \
  python/sgl_jax/test/speculative/test_fused_greedy_decode_step.py | \
  kubectl exec -i perf-16-0-jgb5c -c jax-tpu -- tar xf - -C /tmp/sglang-jax/

kubectl exec perf-16-0-jgb5c -c jax-tpu -- bash -c '
cd /tmp/sglang-jax/python &&
PYTHONPATH=/tmp/sglang-jax/python JAX_PLATFORMS=cpu \
uv run --active pytest \
  sgl_jax/test/speculative/test_eagle_utils.py::TestVerifyTree::test_greedy_sample_device_outputs_enters_mesh_context \
  sgl_jax/test/speculative/test_eagle_utils.py::TestVerifyTree::test_verify_tree_greedy_device_outputs_match_host_postprocess \
  sgl_jax/test/speculative/test_fused_greedy_decode_step.py \
  -q'
```

4. Sync speculative code to all pods:

```bash
for pod in perf-16-0-jgb5c perf-16-1-zhn6d perf-16-2-hs7kc perf-16-3-vqw2x; do
  tar cf - -C . python/sgl_jax/srt/speculative/ python/sgl_jax/test/speculative/ | \
    kubectl exec -i $pod -c jax-tpu -- tar xf - -C /tmp/sglang-jax/
done
```

5. Start 4-host TPU server with reduced buckets:

Use timestamp:

```bash
TS=$(date +%Y%m%d_%H%M%S)_bs32tok256
```

Ranks 1-3:

```bash
for pod in perf-16-1-zhn6d perf-16-2-hs7kc perf-16-3-vqw2x; do
  kubectl exec $pod -c jax-tpu -- bash -c "
cd /tmp/sglang-jax/python &&
setsid /opt/venv/bin/uv run python -m sgl_jax.launch_server \
  --model-path /data/pc --trust-remote-code \
  --speculative-algorithm NEXTN --speculative-eagle-topk 1 \
  --speculative-num-steps 3 --speculative-num-draft-tokens 4 \
  --tp-size 16 --dp-size 4 --ep-size 16 --moe-backend epmoe \
  --host 0.0.0.0 --port 30271 --page-size 64 \
  --context-length 4096 --max-prefill-tokens 4096 \
  --dtype bfloat16 --mem-fraction-static 0.85 \
  --swa-full-tokens-ratio 0.5 --max-running-requests 32 \
  --attention-backend fa --disable-overlap-schedule \
  --precompile-bs-paddings 32 --precompile-token-paddings 256 \
  --nnodes 4 --node-rank \$TPU_WORKER_ID \
  --dist-init-addr perf-16-0.perf-16-headless-svc:5000 \
  </dev/null > /tmp/server_${TS}.log 2>&1 & echo launched"
done
```

Rank 0 foreground:

```bash
kubectl exec perf-16-0-jgb5c -c jax-tpu -- bash -c "
cd /tmp/sglang-jax/python &&
/opt/venv/bin/uv run python -m sgl_jax.launch_server \
  --model-path /data/pc --trust-remote-code \
  --speculative-algorithm NEXTN --speculative-eagle-topk 1 \
  --speculative-num-steps 3 --speculative-num-draft-tokens 4 \
  --tp-size 16 --dp-size 4 --ep-size 16 --moe-backend epmoe \
  --host 0.0.0.0 --port 30271 --page-size 64 \
  --context-length 4096 --max-prefill-tokens 4096 \
  --dtype bfloat16 --mem-fraction-static 0.85 \
  --swa-full-tokens-ratio 0.5 --max-running-requests 32 \
  --attention-backend fa --disable-overlap-schedule \
  --precompile-bs-paddings 32 --precompile-token-paddings 256 \
  --nnodes 4 --node-rank \$TPU_WORKER_ID \
  --dist-init-addr perf-16-0.perf-16-headless-svc:5000 \
  2>&1 | tee /tmp/server_${TS}.log"
```

Expected reduced precompile behavior:

- `SPEC_EXTEND` should precompile only `bs=32, tokens=256`.
- `SPEC_DECODE` should precompile only `bs=32`.
- This should avoid the earlier 5+ minute `SPEC_EXTEND` sweep.

6. Once server is ready, trigger the fixed greedy path:

```bash
kubectl exec perf-16-0-jgb5c -c jax-tpu -- bash -c '
for i in $(seq 1 32); do
curl -s http://localhost:30271/generate -H "Content-Type: application/json" \
  -d "{\"text\":\"What is the meaning of life? Answer in detail.\",\"sampling_params\":{\"temperature\":0,\"max_new_tokens\":256}}" &
done; wait'
```

7. If TPU passes the smoke, commit the current uncommitted patch:

Suggested message:

```bash
git add \
  python/sgl_jax/srt/speculative/base_worker.py \
  python/sgl_jax/srt/speculative/draft_extend_fused.py \
  python/sgl_jax/srt/speculative/eagle_util.py \
  python/sgl_jax/test/speculative/test_eagle_utils.py \
  python/sgl_jax/test/speculative/test_fused_greedy_decode_step.py

git commit -m "perf(spec): fuse greedy step3 postprocess into draft extend"
```

## Likely Next TPU Failure Points

The next TPU run may still fail after the mesh fix. Watch for these:

1. `ForwardBatch` mutation inside a jitted function.
   - The new Step3 fused JIT mutates `forward_batch.positions`,
     `forward_batch.spec_info.hidden_states`, and `logits_metadata.accept_lens`
     inside the JIT body.
   - Similar mutation exists in the older fused draft extend JIT, so it may be
     acceptable, but TPU compile is the real proof.

2. `with_sharding_constraint(..., P())` on `prepared.select_index`,
   `prepared.verified_id`, etc.
   - These small outputs are intended to be final host boundary outputs.
   - If the sharding object is incompatible, simplify by returning them without
     forcing `P()` first, then re-add constraints carefully.

3. Host metadata still depends on mutated `ModelWorkerBatch`.
   - `_prepare_step3_model_worker_batch_for_draft_extend(...)` still updates
     `seq_lens`, `extend_seq_lens`, `logits_indices`, etc. on host.
   - This is acceptable for current Step3, but not final full-decode fusion.

4. `accept_lens` shape/order.
   - The Step3 path assumes padded bs32 and `accept_lens.shape[0] == bs`.
   - `select_index = arange(bs) * 4 + accept_lens - 1`.

5. Donation/replace_all.
   - The JIT donates `all_memory_pools` and returns `all_pool_updates`.
   - Do not delete donated JAX arrays before `replace_all`.

## Current Fusion Status

Achieved:

- Greedy sample output can stay on device.
- Safe-index gather can be computed on device.
- Step3 route exists and is gated to fixed greedy bs32 path.
- Current uncommitted patch moves verify postprocess into the MTP draft-extend
  fused JIT.
- Focused pod CPU tests pass.

Not yet achieved:

- Whole decode is not one JIT.
- Target verify forward is still separate.
- `replicate_to_mesh(logits, hidden)` after target verify still exists in
  `base_worker.py`.
- Full-vocab logits all-gather is not yet replaced by sharded greedy top1.
- Draft tree construction (`draft()`) still has host/device boundaries.
- Attention metadata remains host-built.
- Final scheduler boundary still materializes host arrays, which is expected for
  this phase but not ideal for the final design.

## Important Context From Communication Review

Highest-value remaining communications:

1. `base_worker.py` target verify still calls `replicate_to_mesh(...)` for full
   logits and hidden. For greedy top1, full logits replication should eventually
   be replaced by sharded argmax/logsumexp.
2. Existing fused draft extend still reshards per-layer logits/hidden to `P()`.
   Current Step3 fused JIT keeps this behavior. Future step should implement a
   sharded top1 helper.
3. Host `draft()` tree assembly and `device_put` of spec tree remain outside
   the JIT. Future full-decode fusion needs device-side tree state or a
   scheduler/spec-state redesign.

## Cleanup Commands

If server is left running:

```bash
for pod in perf-16-0-jgb5c perf-16-1-zhn6d perf-16-2-hs7kc perf-16-3-vqw2x; do
  kubectl exec $pod -c jax-tpu -- bash -c \
    "pkill -f 'sgl_jax.launch_server' || true"
done
```

Confirm:

```bash
for pod in perf-16-0-jgb5c perf-16-1-zhn6d perf-16-2-hs7kc perf-16-3-vqw2x; do
  echo $pod
  kubectl exec $pod -c jax-tpu -- bash -c \
    "ps -ef | grep 'sgl_jax.launch_server' | grep -v grep || true"
done
```

## Context Window Note

The user is switching to a new 1M context window because this thread remained
at 258K despite `~/.codex/config.toml` containing:

```toml
model = "gpt-5.5"
model_context_window = 1000000
model_auto_compact_token_limit = 900000
```

Do not assume the new thread has this entire conversation. Start from this file,
the three earlier docs, `git status`, and the current diff.
