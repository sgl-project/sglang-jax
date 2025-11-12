# SGLang-JAX CI/CD Architecture Design

## Table of Contents
- [Overview](#overview)
- [Architecture Principles](#architecture-principles)
- [CI/CD Pipeline Categories](#cicd-pipeline-categories)
  - [Pull Request Testing](#pull-request-testing)
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
| accurancy-test-1-tpu | 1x TPU | - | Model Evaluation  |
| accurancy-test-4-tpu | 4x TPU | - | Model Evaluation  |
| pallas-kernel-benchmark | 1x TPU | - | Speed |
| e2e-test-1-tpu | 1x TPU | - | Accuracy |
| e2e-test-4-tpu | 4x TPU | - | Accuracy |

### Nightly Testing
Coming soon
<!--
#### Comprehensive Nightly Tests (`nightly-test.yml`)
*Schedule:* Daily at midnight UTC

*Trigger Conditions:*
- Cron schedule: `0 0 * * *`
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
| accurancy-test-1-tpu | 1x TPU | - | Model Evaluation  |
| accurancy-test-4-tpu | 4x TPU | - | Model Evaluation  |
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
