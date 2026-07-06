# SGLang-JAX CI/CD Architecture Design

## Table of Contents
- [Overview](#overview)
- [Architecture Principles](#architecture-principles)
- [CI/CD Pipeline Workflows](#cicd-pipeline-workflows)
  - [Slash Command Handler](#slash-command-handler-slash-commandyml)
  - [PR Review](#pr-review-pr-reviewyml)
  - [CI Auto Bisect](#ci-auto-bisect-ci-auto-bisectyml)
  - [Pull Request Testing](#pull-request-testing-pr-testyml)
  - [Nightly Testing](#nightly-testing)
  - [Release Automation](#release-automation)

## Overview

SGLang-JAX maintains a comprehensive CI/CD infrastructure consisting of several workflows across multiple testing scenarios. The CI system is designed to ensure code quality, inference accuracy and performance.

### Supported Platforms
- *TPU v6e*

## Architecture Principles

### 1. Comprehensive Test Coverage
Testing is organized into multiple layers:
- *Lint Check:* pylint
- *Unit Tests:* core functionality (e.g. attention, eagle, rotary embedding)
- *E2E Tests:* Multi-TPU configurations (1, 4 TPUs)
- *Performance Tests:* Latency (e.g. TTFT, ITL), Throughput
- *Accuracy Tests:* Model evaluation (GSM8K, MMMU-100, Math)

### 2. Gated CI Execution
To optimize resource usage and prevent wasted compute:
- PRs require `run-ci` label to trigger CI, reviewer add label manually
- Draft PRs are automatically rejected
- Change detection filters determine which workflows run
- Concurrency controls cancel redundant runs

### 3. Artifact Reuse
Built artifacts are shared across jobs to reduce redundancy:
- Docker images cached locally for faster subsequent runs
- Some dependency (e.g. Rust) compilation cache shared via sccache

### 4. Progressive Testing
Testing follows a progressive strategy:
- Quick checks first (e.g. lint) when pull request or any other change happen
- Unit tests before integration tests
- Performance benchmarks run in parallel partitions
- Expensive tests (4-TPUs, accuracy) run last

## CI/CD Pipeline Workflows

### Slash Command Handler (slash-command.yml)

Allows maintainers to trigger targeted CI operations from PR comments instead of
re-running the entire workflow.

*Trigger:* `issue_comment` (created) — only processed when the comment starts with `/`.

*Permission Model:*
- `OWNER`, `MEMBER`, `COLLABORATOR` — allowed
- All other associations — rejected with a reply comment

> **Fork-PR safety:** these commands run the PR HEAD code on a runner with elevated
> permissions (`pull-requests: write`, GKE/GCP OIDC, inherited secrets). The only
> gate is the permission model above — same trust model as `/rerun-group` /
> `/rerun-stage`. **Review a fork PR's diff before commenting `/run-nightly` (or any
> rerun command) on it**, since you are authorizing untrusted code to run with those
> permissions.

*Supported Commands:*

| Command | Description |
|---------|-------------|
| `/rerun-failed-ci` | Re-run only the failed jobs from the most recent `pr-test` run on the PR branch |
| `/test perf` | Add the `test:perf` label — the label event triggers CI with perf tests enabled |
| `/rerun-group <suite>` | Run a specific test suite on the matching runner via `rerun-test.yml`. Suite names from `test/srt/run_suite.py`; runner derived from suffix (`-v6e-1`, `-v6e-4`, `-cpu`) |
| `/rerun-stage <stage>` | Dispatch all suites in a stage independently via `rerun-test.yml`. Stage: `1`/`fast` (unit), `2`/`medium` (e2e+accuracy), `3`/`heavy` (performance) |
| `/run-nightly <case_key>` | Run a single nightly case against the PR HEAD via `nightly-test-daily.yml` (`job_filter` + `--cases`); the result is posted back on the PR. `<case_key>` is a case name (e.g. `qwen3-8b-fa`). A 🚀 reaction marks the nightly as started (it comments PASS/FAIL back); an invalid key gets 😕 plus the valid list; a failed case also gets an AI bisect comment |
| `/run-nightly` or `/run-nightly ?` | List the available `case_key`s (grouped by job). Use this to discover what you can run |

*Where the case_keys come from:* not a hand-maintained list — the slash parser enumerates the single-host suite runner's catalog by invoking its `--caselist` (`test/srt/nightly/single_host/suite_runner.py`, a subprocess; the case list is dependency-light and needs no jax, so it runs on the CPU runner), so **adding a case to a suite makes it runnable via `/run-nightly` with nothing else to update**. It exposes all accuracy cases plus each perf model's representative point (`PerfCase.capture_trace`), since the sweep's other points only make sense as a set. Multi-host is not wired into `/run-nightly` — its nightly job stays `if: false` until CI has 4-node v6e capacity. The parser holds only a single-host suite→job map; any suite the runner exposes but the map doesn't name is silently skipped.

*Using `/run-nightly` (maintainers):*
- **Discover:** comment `/run-nightly ?` (or bare `/run-nightly`) → it replies with the runnable `case_key`s, grouped by job.
- **Run one case:** comment `/run-nightly <case_key>` — e.g. `/run-nightly qwen3-8b-fa` (an accuracy case) or `/run-nightly qwen3-32b-c32-i4096-o1024` (a perf representative point). It runs that one case against the PR HEAD.
- **Who can trigger:** `OWNER` / `MEMBER` / `COLLABORATOR` only (the shared slash-command permission gate); anyone else is rejected. Review a fork PR's diff first — this runs PR-HEAD code with elevated permissions.
- **What you get:** a 🚀 reaction when the run starts, then a `PASS`/`FAIL` comment (with a run link) when it finishes. An invalid `case_key` gets a 😕 reaction plus the list of valid keys; a failed case also gets an AI root-cause (`ci-auto-bisect`) comment.

*Architecture:*
- `scripts/ci/slash_command_parse.py` — pure-Python parser (stdlib only), reads
  `COMMENT_BODY` / `ACTOR` / `ACTOR_ASSOCIATION` env vars, writes results to `$GITHUB_OUTPUT`
- `scripts/ci/slash_command_dispatch.py` — dispatch logic for `/rerun-failed-ci`
- `scripts/ci/slash_command_stage_dispatch.py` — dispatch logic for `/rerun-stage`,
  loops through each suite in the stage and dispatches `rerun-test.yml` independently
- `rerun-test.yml` — `workflow_dispatch` workflow for `/rerun-group` and `/rerun-stage`,
  accepts suite name, runner label, and git ref as inputs
- `nightly-test-daily.yml` — exposes `workflow_call`; `/run-nightly` invokes it as a
  reusable workflow (`uses:` from slash-command.yml) with `ref=refs/pull/<N>/head`
  (resolves for fork PRs too), `job_filter`, `cases`, and `pr_number`. Running it via
  `uses:` keeps the nightly part of the Slash Command Handler run, so it never appears
  as a standalone "Nightly Test Daily" run for the notifier / auto-bisect to mistake
  for the scheduled nightly. The run job (read-only, no write token) executes the case; a separate clean job posts PASS/FAIL back to the PR
- Most commands react 👍 and post result comments; `/run-nightly` instead reacts
  🚀 when the nightly starts / 😕 on an invalid key, to keep the PR thread quiet — only
  the final PASS/FAIL (and the on-demand case list) are comments

### PR Review (pr-review.yml)

AI-powered code review using Claude Code Action. Auto-triggers on PR open/ready_for_review;
also available on-demand via `/review` comment (`OWNER`/`COLLABORATOR` only).

### CI Auto Bisect (ci-auto-bisect.yml)

AI-powered CI failure analysis. Auto-triggers on failed workflow runs (PR Test, Nightly Test, etc.);
also available on-demand via `/auto-bisect [run_id]` comment (`OWNER`/`COLLABORATOR` only).
Classifies failures as `code_regression`, `flaky_test`, `infrastructure`, or `environment`
and posts analysis results on the associated PR.

### Pull Request Testing (pr-test.yml)
*Purpose:* Comprehensive testing for JAX

*Trigger Conditions:*
- Push to `main` branch
- Pull request to `main` (synchronize, labeled)
- Requires `run-ci` label for PRs
- Rejects draft PRs

*Test Scenario:*
- models: Qwen3-8B(enable TP), Qwen3Qwen3-30B-A3B (enable TP, EP)
- 200 MMLU test cases
- ISL/OSL: 1k/1k

*Jobs:*

| Test Category | TPU Config | Partitions | Key Features |
|--------------|------------|------------|--------------|
| unit-test-1-tpu | 1x TPU | - | All unit test pass|
| unit-test-4-tpu | 4x TPU | - | All unit test pass|
| performance-test-1-tpu (including cache miss check) | 1x TPU | - | Latency, Throughput|
| performance-test-4-tpu (including cache miss check) | 4x TPU | - | Latency, Throughput|
| accuracy-test-1-tpu | 1x TPU | - | Model Evaluation  |
| accuracy-test-4-tpu | 4x TPU | - | Model Evaluation  |
| pallas-kernel-benchmark | 1x TPU | - | Speed |
| e2e-test-1-tpu | 1x TPU | - | Accuracy |
| e2e-test-4-tpu | 4x TPU | - | Accuracy |

### Nightly Testing
Coming soon
<!--
#### Comprehensive Weekly Tests (`weekly-test.yml`)
*Schedule:* Weekly, Sunday 00:00 UTC+8

*Trigger Conditions:*
- Cron schedule: `0 16 * * 0`
- Push to `main` when `version.py` changes
- Manual dispatch

*Test Scenario:*
- models: Qwen3-8B(enable TP), Qwen3Qwen3-30B-A3B (enable TP, EP), bailing, gemma2
- MMLU, MMUL-PRO, AIME-24/25, Math-500
- ISL/OSL: 1k/1k, 4k/1k, 8k/1k, 1k/1, 4k/1, 8k/1

*Jobs:*

| unit-test-1-tpu | 1x TPU | - | All unit test pass|
| unit-test-4-tpu | 4x TPU | - | All unit test pass|
| performance-test-1-tpu (including cache miss check) | 1x TPU | - | Latency, Throughput|
| performance-test-4-tpu (including cache miss check) | 4x TPU | - | Latency, Throughput|
| accuracy-test-1-tpu | 1x TPU | - | Model Evaluation  |
| accuracy-test-4-tpu | 4x TPU | - | Model Evaluation  |
| pallas-kernel-benchmark | 1x TPU | - | Speed |
| e2e-test-1-tpu | 1x TPU | - | Accuracy |
| e2e-test-4-tpu | 4x TPU | - | Accuracy |

**[TODO]** *Performance Monitoring:*
- Traces published to `sgl-jax-ci-data` repository
- Perfetto UI visualization
- Historical performance tracking

*Repository Restriction:*
- Only runs on official ```sgl-project/sglang-jax``` repository
-->
### Release Automation
#### Package Releases (`release-pypi.yml`)
##### Main Package Release
*Trigger:*
- Push to `main` when `version.py` changes
- Manual dispatch

*Process:*
- Build Python package
- Upload to PyPI
- Only runs on official repository

#### Docker Releases (`release-docker.yml`)
##### Docker Release
*Trigger:*
- Push to `main` when `version.py` changes
- Manual dispatch

*Tag Strategy:*
- Version-specific tags: `v0.5.3-amd64`
- Multi-arch manifest: `v0.5.3`, `latest`

*Environment:*
- Requires `prod` environment approval

##### Nightly Docker Releases
*Trigger:*
- Cron schedule: `0 0 * * *`
- Manual dispatch

*Tag Strategy:*
- Version-specific tags: `v0.5.3-nightly-amd64`
- Multi-arch manifest: `v0.5.3`, `latest`

## Detailed Test Matrix
### Unit Test
Test files are located in the folder ```python/srt/test/*``` and start with prefix ```test_```
### E2E Test
Test files are located in the folder ```test/srt/*``` and start with prefix ```test_```
### Accuracy Test (1x TPU, 4x TPUs)
#### PR Test Workflow
| Model | Dataset | Floating Threshold |
| -------------- | -------------- | -------------- |
| Qwen3-8B (TP only) | 200 mmlu cases | 2% |
| Qwen3-30B-A3B (enable TP and EP) | 200 mmlu cases | 2% |

#### Nightly Test Workflow
| Model | Dataset | Floating Threshold |
| --- | --- | --- |
| Qwen-7B (TP only)<br>Qwen3-8B (TP only)<br>Qwen3-30B-A3B (enable TP and EP)<br>gemma2<br>bailing_moe(enable TP and EP) | • mmlu<br>• mmlu-pro<br>• aime-24/25<br>• gsm8k | 2% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | • mmlu<br>• mmlu-pro<br>• aime-24/25<br>• gsm8k<br>• math | 2% |

### Performance Test (1x TPU, 4x TPUs)
#### PR Test Workflow
| Model | Latency | Throughput | Floating<br>Threshold |
| --- | --- | --- | --- |
| Qwen3-8B (TP only)<br>Qwen3-30B-A3B (TP and EP) | **Max Concurrency:** 256<br>**Seq Num:** 1<br>**ISL/OSL:** 1k/1k<br><br>**Main Metrics:**<br>1. TTFT<br>(e.g. < 300ms + 8%)<br><br>2. ITL<br>(e.g. < 8ms + 8%) | **Max Concurrency:** 256<br>**Seq Num:** 500<br><br>**Main Metrics:**<br>1. Input Throughput<br>ISL/OSL: 1k/1<br><br>2. Output Throughput<br>ISL/OSL: 1/1k | 5-10% |

#### Nightly Test Workflow
| Model | Latency | Throughput | Floating<br>Threshold |
| --- | --- | --- | --- |
| Qwen-7B (TP only)<br>Qwen3-8B (TP only)<br>Qwen3-30B-A3B (enable TP and EP)<br>gemma2<br>bailing_moe(enable TP and EP)<br>qwen2 | **Max Concurrency:**<br>[8, 16, 32, 64, 128, 256]<br><br>**Seq Num:**<br>3x Max Concurrency<br><br>**Main Metrics:**<br>1. TTFT<br>ISL/OSL:<br> • 1k/1<br> • 4k/1<br> • 8k/1<br><br>2. ITL<br>ISL/OSL:<br> • 1k/1k<br> • 4k/1k<br> • 8k/1k | **Max Concurrency:**<br>[8, 16, 32, 64, 128, 256]<br><br>**Seq Num:**<br>3x Max Concurrency<br><br>**Main Metrics:**<br>1. Input Throughput<br>ISL/OSL:<br> • 1k/1<br> • 4k/1<br> • 8k/1<br><br>2. Output Throughput<br>ISL/OSL:<br> • 1k/1k<br> • 4k/1k<br> • 8k/1k | 5-10% |
