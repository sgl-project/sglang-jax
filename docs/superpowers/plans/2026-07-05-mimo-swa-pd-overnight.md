# MiMo SWA PD Overnight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make MiMo-V2-Flash PD disaggregation functionally correct first, then iterate on TTFT, IOT, and throughput while preserving correctness.

**Architecture:** Keep the current epic branch and GKE job shape. Fix the immediate DP2 Raiden page-id correctness issue with CPU regression coverage, validate on MiMo SWA PD GKE, then run a benchmark ladder before performance work.

**Tech Stack:** Python, pytest, JAX/TPU, GKE indexed jobs, Raiden KV transfer, MiMo-V2-Flash SWA KV pools.

---

## Scope And Stop Rules

- The branch is `epic/mimo-pd-disggragation`; preserving unrelated mainline behavior is not a goal for this epic.
- Touch only PD/Raiden/SWA/GKE validation files needed for MiMo-V2-Flash PD.
- Commit and push at stable nodes: after a tested correctness fix, after each validated GKE milestone, and after any performance iteration with evidence.
- Stop instead of guessing if two GKE reruns expose different root causes, or if a fix requires changing Raiden manager architecture rather than local block-id semantics.
- Do not reset the dirty worktree. Stage only files touched for this work.

## Current Evidence

- Job `mimo-swa-pd-dp2-fix` reached healthy bootstrap, prefill, decode, and router.
- GSM8K q32/p32 started but stayed at `0/32`.
- Prefill crashed its scheduler child with:

```text
Failed to issue D2H in StartPushInternal layer loop: Copy range exceeds source device buffer size
```

- The first register/read used blocks `[6703, 6704, 6705, 6706]` while the full Raiden manager shape showed `bf16[6702,...]`. This points to globalized DP page ids being passed into a per-local-manager buffer index.

## Task 1: Correct DP2 Raiden Block ID Semantics

**Files:**
- Modify: `python/sgl_jax/srt/disaggregation/prefill.py`
- Modify: `python/sgl_jax/srt/disaggregation/decode.py`
- Modify: `python/sgl_jax/srt/disaggregation/jax_transfer/conn.py`
- Test: `python/sgl_jax/test/test_pd_swa_basic.py`

- [ ] **Step 1: Write a failing CPU regression test**

Add or extend a test proving that DP rank is used for allocator/mapping lookup, but Raiden block ids passed to the local manager remain within the local per-rank page range.

- [ ] **Step 2: Run the regression test and verify it fails**

Run:

```bash
USE_DEVICE_TYPE=cpu .venv/bin/python -m pytest python/sgl_jax/test/test_pd_swa_basic.py -q
```

Expected: one new test fails because the current code emits globalized block ids for local Raiden manager access.

- [ ] **Step 3: Implement the minimal correctness fix**

Keep DP-aware routing and DP-aware SWA mapping selection. Stop passing cross-rank global page ids to Raiden manager `register_read` and `start_read` when the manager is constructed over local buffers.

- [ ] **Step 4: Run targeted CPU tests**

Run:

```bash
USE_DEVICE_TYPE=cpu .venv/bin/python -m pytest python/sgl_jax/test/test_pd_swa_basic.py -q
USE_DEVICE_TYPE=cpu .venv/bin/python -m pytest python/sgl_jax/test/mem_cache/test_swa_allocator.py -q
USE_DEVICE_TYPE=cpu .venv/bin/python -m pytest test/srt/disaggregation/test_pd_router.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit and push**

Commit only touched correctness files and push to `epic/mimo-pd-disggragation`.

## Task 2: Re-run MiMo SWA PD Correctness

**Files:**
- Use: `scripts/disaggregation/gke/deploy_mimo_swa_pd_e2e.sh`
- Use: `scripts/disaggregation/gke/mimo_swa_pd_e2e_job.yaml`

- [ ] **Step 1: Relaunch only the current E2E job**

Run with the same job name and correctness-only settings:

```bash
export PATH="/opt/homebrew/share/google-cloud-sdk/bin:$PATH"
JOB_NAME=mimo-swa-pd-dp2-fix \
BRANCH=epic/mimo-pd-disggragation \
EPHEMERAL_STORAGE=38Gi \
DP_SIZE=2 \
SKIP_GCSFUSE_WARMUP=1 \
RUN_GSM8K=1 \
GSM_Q=32 \
GSM_PAR=32 \
GSM_MAXTOK=2048 \
GSM_MIN_ACC=0.80 \
RUN_LONG_BENCH=0 \
RUN_MMLU_PRO=0 \
scripts/disaggregation/gke/deploy_mimo_swa_pd_e2e.sh
```

- [ ] **Step 2: Monitor through q32 GSM8K**

Watch decode health, router health, `gsm8k_stdout.log`, `gsm8k_result.jsonl`, and Raiden error markers.

- [ ] **Step 3: Commit/push a handoff update if q32 passes**

Record accuracy, failure count, and key log markers in the handoff document or a new validation note.

## Task 3: Correctness Ladder

**Files:**
- Use pod-local benchmark scripts under `/tmp/sglang-jax/benchmark`

- [ ] **Step 1: Run GSM8K q128/p128**
- [ ] **Step 2: Run random 16k input / 4k output benchmark at concurrency 32**
- [ ] **Step 3: If c32 is stable, run concurrency 64 and 128**
- [ ] **Step 4: Run MMLU-Pro q200/p128**
- [ ] **Step 5: Save results and commit/push validation notes**

## Task 4: First Performance Iteration

**Files:**
- To be selected after correctness data, likely scheduler, prefill/decode Raiden transfer paths, or overlap gating.

- [ ] **Step 1: Pick one measured bottleneck**
- [ ] **Step 2: Add a failing or baseline-locking test/benchmark harness**
- [ ] **Step 3: Implement one optimization**
- [ ] **Step 4: Re-run correctness and the relevant benchmark**
- [ ] **Step 5: Commit/push only if correctness remains stable**
