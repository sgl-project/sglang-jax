# Agent Guide: sgl-jax Data Parallelism

This document guides code agents working on the data-parallelism feature for the sgl-jax project. All agents work in **isolation** on separate git worktrees and submit individual PRs to a shared integration branch.

---

## Project Background

### Goal

Rebase and fix compatibility issues in the sgl-jax data-parallelism (DP) implementation so it can be merged into `main`.

The DP feature has been developed on a long-lived branch and has accumulated significant drift from `main` — including rebase conflicts, API incompatibilities, and test failures. The objective of this effort is to **fix these issues one by one** so all changes can be merged cleanly.

### Repository

- **Project**: sgl-jax — JAX backend for SGLang
- **Main source**: `python/sgl_jax/`, core serving runtime under `python/sgl_jax/srt/`
- **Tests**: `test/` directory; test suite entry point: `test/srt/run_suite.py`
- **Integration branch**: `feat/data-parallelism` — all agent PRs target this branch
- **End goal**: `feat/data-parallelism` passes CI and merges into `main`

### Key Constraints

- **TPU required**: All JAX tests must run on a remote TPU cluster. Never run JAX/TPU tests locally.
- **Remote execution**: Use the `sglang-jax-skypilot-dev` skill for all remote test and debug sessions.
- **Sky commands**: Do not prepend proxy environment variables to sky commands. Use `sky exec`, `sky status`, etc. directly.

### DP Architecture Overview

Core components of data-parallelism in sgl-jax:

| Component | Path | Description |
|-----------|------|-------------|
| Scheduler | `python/sgl_jax/srt/managers/` | DP-aware request scheduling |
| Allocator | `python/sgl_jax/srt/mem_cache/` | Memory allocation across DP ranks |
| Radix Cache | `python/sgl_jax/srt/mem_cache/` | DP-safe KV cache management |
| Control Plane | `python/sgl_jax/srt/` | Communication and coordination between DP ranks |

---

## SOP

Follow these phases **in order**. Do not skip ahead.

### Phase 0 — Receive Task

Before touching any code:

- [ ] Confirm from your task description: **feature name**, **integration branch name**, **task type** (bugfix / feature), **test mode** (test file / service debug)
- [ ] If any of the above is missing or ambiguous → **stop and report immediately**. Do not guess.

---

### Phase 1 — Set Up Worktree

Create an isolated worktree based on the integration branch. **All subsequent work happens inside this worktree only.**

```bash
# Create worktree and working branch
git worktree add .worktrees/<feature-name> <integration-branch>
cd .worktrees/<feature-name>
git checkout -b <feature-name>
```

Rules:
- Never modify the main working directory or any other worktree.
- Never push directly to the integration branch or main.

---

### Phase 2 — Analyze the Problem

Before writing any code, produce a short written analysis (this will become the PR description draft):

**For bugfix:**
- Root cause of the bug
- Affected modules / files
- Expected behavior after the fix

**For feature development:**
- Functional boundary: what is in scope, what is out of scope
- Affected modules / files
- Expected behavior / interface

If during analysis you find you need to modify files that belong to another agent's functional area → **stop and report**. Do not modify those files.

---

### Phase 3 — Write Tests First

Before implementing, write the tests that define success. Tests must **fail** (red) at this point.

**Test File mode:**
- Write or update test files under `test/`
- Run the tests on the remote TPU cluster and confirm they fail
- Commit the failing tests

**Service Debug mode:**
- Write the client script (benchmark / debug / accuracy test) that will be run against the live server
- Document the expected output or pass criteria
- Commit the client script

Use the `sglang-jax-skypilot-dev` skill for all remote execution.

If the test infrastructure itself is broken or unclear → **stop and report**.

---

### Phase 4 — Implement

Write the implementation to make the tests pass.

Rules:
- Only modify files within your functional area (identified in Phase 2)
- Follow project code style: lazy log formatting (`logger.info("msg %s", var)`), Ruff-compliant code
- If you discover a necessary change is outside your functional area → **stop and report**

Commit incrementally with clear messages.

---

### Phase 5 — Verify

Run your tests on the remote TPU cluster and confirm they pass (green).

**Test File mode:**
```bash
# Via sglang-jax-skypilot-dev skill — SSH into cluster, then:
uv run --extra tpu python -m pytest test/srt/<your_test_file.py> -v
```

**Service Debug mode:**
```bash
# Via sglang-jax-skypilot-dev skill — SSH into cluster, then:
# tmux session "server": start the service
# tmux session "client": run the client/benchmark/accuracy test after service is ready
```

Do not proceed to Phase 6 until all your tests are green.

---

### Phase 6 — Submit PR

Open a PR from your working branch targeting the **integration branch**.

**PR title format:** `[DP] <feature-name>: <one-line description>`

**PR description must include:**

```
## Problem Analysis
<Root cause (bugfix) or functional boundary (feature)>

## Changes
<List of modified files and what changed in each>

## Test Results
Test mode: [test file | service debug]
Command: <exact command used>
Result: <pass/fail counts or benchmark output summary>
```

- If all tests pass → open as a **ready-for-review** PR
- If there are unresolved blockers → open as a **Draft PR** and add a comment explaining the blocker (what the problem is, which phase you are stuck in)

---

### Blocker Protocol

At **any** phase, if you encounter a blocker:

1. **Stop immediately** — do not attempt workarounds
2. Commit your current state with message: `WIP: blocked on <description>`
3. Open a Draft PR (or add a comment to an existing one) with:
   - Phase you are stuck in
   - Exact error or conflict
   - What you have already tried
4. Wait for human intervention

---

## Acceptance Criteria

A PR is ready to merge into the integration branch when **all** of the following are true:

| Criterion | Requirement |
|-----------|-------------|
| Linter | `ruff check` passes with no errors |
| Code hygiene | No unresolved TODO/FIXME in newly added code |
| Tests (test file mode) | All specified test files pass; no new failures introduced |
| Tests (service debug mode) | Service starts successfully; client/benchmark/accuracy test completes with results meeting expected criteria |
| PR description | Problem analysis, file list, and test output summary all present |
| Branch state | Based on integration branch, no unresolved merge conflicts |
| Commit history | No stray debug commits; history is clean or squashed |
| Blockers | If any blocker exists, PR must be Draft with blocker described in a comment |
