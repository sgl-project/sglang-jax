# Contribution Guide

Welcome to **SGLang-Jax**! We appreciate your interest in contributing. This guide provides a concise overview of how to set up your environment, run tests, build documentation, and open a Pull Request (PR).


## How to contribute your first PR?

We divide the issues into three types:
- Features, such as support [structured output](https://github.com/sgl-project/sglang-jax/issues/314)
  - Note: Sometimes you just want to add a small feature like an api, please still link a google document.
    - This is encouraged to discuss. We want to make a tradeoff between convenience and feature iterations.
- Bugs
- Add unit tests, E2E tests, accuracy tests, performance tests, etc.

Note:
- You can select the existing or create a new issue for yourself. Please fill the template according to issue type.
- Before dive deep into different issues contribution requirements, please let me introduce you about CI which is necessary before merging into main. CI consists of pull requests tests and nightly tests. The former is design to ensure fundamental features and bugfixes work and pass few performance and accuracy tests. It includes unit tests, E2E tests, accuracy and performance tests for Qwen3-8B and Qwen3-30B-A3B. The latter is design to check tests as much as possible, such as more models, more datasets, more performance scenarios and so on. See more details in [CI Architecture](./ci_architecture.md).

**Features**

Here we give the feature which is to support structure output as an example. Note: If this is a small feature, only one issue is required.

1. Create a root issue to trace your job in this feature, like [314](https://github.com/sgl-project/sglang-jax/issues/314)

2. Split your job to at least two subissues, like:
  - Design document subissue, such as [315](https://github.com/sgl-project/sglang-jax/issues/315)
    - Note: A google document is required to elaborate your design, such as [structure output](https://docs.google.com/document/d/1lZ09hEB00KZjJW_W1Bht1euGwgl3jxu8bVnStJihHXg/edit?tab=t.0). This document is required to be reviewed. You have to include motivation, goals, design and tests.
  - Code development subissue, such as [331](https://github.com/sgl-project/sglang-jax/issues/331)

3. Codes development:
  - TPU resources: Please refer to [TPU Resources Guide](./tpu_resources_guide.md).
  - Setup environment: Please refer to [install SGLang-Jax from source](#install-sglang-jax-from-source).
  - Code style: Please refer to [format codes](#format-code-with-pre-commit) before you push.
  - Testing:
    - Please add unit tests under `python/srt/test/*` and E2E tests under `test/srt/*` to ensure the feature works. File names are required to start with `test_`.
    - If the feature were to add new accuracy or performance baselines or influence the existing accuracy or performance, please add them in nightly tests. Nightly tests are under construction.
      - Example: Add a new model implementation.
  - Description in PR: Please add accuracy or benchmark baselines in pull requests if the feature meeted the above scenarios.
  - Review: Assign at least one reviewer.

4. If you resolved all comments from code reviews, the pull request would be merged into main. Congrantulations to you!


**Bugs && Add tests**

1. Create an issue and fill the content according to template.

2. You can fix it by your self or assign it to others. If you select the former, please continue.

3. Code development: Keep the same to **Features**.

4. If you resolved all comments from code reviews, the pull request would be merged into main. Congrantulations to you!


## Install SGLang-Jax from Source

### Fork and clone the repository

**Note**: New contributors do **not** have the write permission to push to the official SGLang-Jax repo. Please fork the repository under your GitHub account, then clone your fork locally.

```bash
git clone https://github.com/<your_user_name>/sglang-jax.git
```

### Build from source

Refer to [Install SGLang-Jax from Source](../get_started/install.md#method-2-from-source).

## Format code with pre-commit

We use [pre-commit](https://pre-commit.com/) to maintain consistent code style checks. Before pushing your changes, please run:

```bash
pip3 install pre-commit
pre-commit install
pre-commit run --all-files
```

- **`pre-commit run --all-files`** manually runs all configured checks, applying fixes if possible. If it fails the first time, re-run it to ensure lint errors are fully resolved. Make sure your code passes all checks **before** creating a Pull Request.
- **Do not commit** directly to the `main` branch. Always create a new branch (e.g., `feature/my-new-feature`), push your changes, and open a PR from that branch.


## General code style
- Avoid code duplication. If the same code snippet (more than five lines) appears multiple times, extract it into a shared function.
- Keep files concise. If a file exceeds 2,000 lines of code, split it into multiple smaller files.
- Strive to make functions as pure as possible. Avoid in-place modification of arguments.


## Q & A

### How to do benchmark?

Please refer to [Benchmark and Profiling](./benchmark_and_profiling.md).

### How to evaluate the accuracy?

We recommend to refer to public configurations like sampling parameters and accuracy results on HuggingFace.

```bash
# Launch a server
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache \
python3 -m sgl_jax.launch_server \
--model-path Qwen/Qwen-7B-Chat \
--trust-remote-code  \
--dist-init-addr=0.0.0.0:10011 \
--nnodes=1  \
--tp-size=4 \
--device=tpu \
--random-seed=3 \
--node-rank=0 \
--mem-fraction-static=0.8 \
--max-prefill-tokens=8192 \
--download-dir=/tmp \
--dtype=bfloat16  \
--skip-server-warmup \
--port 30000 \
--max-running-requests 256 \
--page-size 128

# Evaluate By EvalScope
## Note: evalscope==0.17.1 is recommended.
evalscope eval  \
--model Qwen/Qwen3-8B \
--api-url http://127.0.0.1:30000/v1/chat/completions \
--api-key EMPTY \
--eval-type service \
--datasets gsm8k \
--eval-batch-size 64 \
--generation-config '{"temperature": 0.7,"top_p":0.8,"top_k":20,"min_p":0.0,"presence_penalty":0.5}'
```

Some details about evaluations:
- Evalscope Usage: You can set more arguments for evaluation, please refer to [official documents](https://evalscope.readthedocs.io/en/latest/get_started/parameters.html).
- Accuracy Deviation: This test can have significant variance (1%–5%) in accuracy due to batching and the non-deterministic nature of the inference engine. Please run multi times to get the average result.
- Dataset Selection: GSM8K is too easy for state-of-the-art models nowadays. Please try your own more challenging accuracy tests. You can find additional accuracy eval examples in [test_eval_accuracy_large.py](https://github.com/sgl-project/sglang-jax/blob/main/test/srt/test_eval_accuracy_large.py).
- Sampling Parameters: Please set proper sampling parameters for your model. We recommend to use configurations on Hugging Face.
