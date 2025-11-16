# [Draft]Contribution Guide

Welcome to **SGLang-Jax**! We appreciate your interest in contributing. This guide provides a concise overview of how to set up your environment, run tests, build documentation, and open a Pull Request (PR).


## How to contribute your first PR?

We divide the issues into three types:
- Features, such as support structured output
- Bugs
- Add unit tests or CI tests

**Features**

Here we give the feature which is to support structure output as an example.

1. Create a root issue to trace your job in this feature, like [314](https://github.com/sgl-project/sglang-jax/issues/314)

2. Split your job to at least two subissues, like:
   - Design document subissue, such as [315](https://github.com/sgl-project/sglang-jax/issues/315)
     - Note: A google document is required to elaborate your design, such as [structure output](https://docs.google.com/document/d/1lZ09hEB00KZjJW_W1Bht1euGwgl3jxu8bVnStJihHXg/edit?tab=t.0). This document is required to be reviewed.
   - Code development subissue, such as [331](https://github.com/sgl-project/sglang-jax/issues/331)

3. Codes development:
   - TPU resources: Please refer to [how to get TPU resources to develop](#how-to-get-tpu-resources-to-develop).
   - Please refer to [install SGLang-Jax from source](#install-sglang-jax-from-source) to setup the development environment.
   - Please refer to [format codes](#format-code-with-pre-commit) before you push.
   - Please add unit tests and e2e tests in CI to ensure the feature works. You can refer to [how to add tests in CI](#how-to-add-tests-in-ci).
   - Codes are required to review.
   - Please add accuracy and benchmark results when meeting the following cases:
     - Add a new model: please give both of them in **PR** and **CI tests**.
     - Feature may require new dataset accuracy: please add accuracy tests in **PR** and **CI tests**.
     - Feature may require better performance baseline, such as overlapping scheduler: please add or update benchmark result in **PR** and **CI tests**.
     - **Note**:
       - Accuracy: please refer to [how to evaluate the accuracy](#how-to-evaluate-the-accuracy).
       - Benchmark: please refer to [how to do benchmark](#how-to-do-benchmark).
       - Add accuracy check in CI tests: please refer to the corresponding section in [how to add tests in CI](#how-to-add-tests-in-ci).
       - Add benchmark check in CI tests: please refer to the corresponding section in [how to add tests in CI](#how-to-add-tests-in-ci)

4. If you resolved all comments from code reviews, the pull request would be merged into main. Congrantulations to you!


**Bugs**

1. Create an bug issue and fill the content, such as reproduce steps, environment, solution and so on.
2. You can fix it by your self or assign it to others. If you select the former, please continue.
3. Code reviews are required.
4. Fix bugs and add unit tests or e2e tests in CI to ensure the bug has been fixed. You can refer to [how to get TPU resources to develop](#how-to-get-tpu-resources-to-develop) and [how to add tests in CI](#how-to-add-tests-in-ci).
5. Optional: Check the accuracy and benchmark if you think the modification will influence them.
6. If you resolved all comments from code reviews, the pull request would be merged into main. Congrantulations to you!

**Add tests**

1. Create the issue or find an existing test issue to solve.
2. Add unit test or [CI tests](#how-to-add-tests-in-ci). Run tests with [resource](#how-to-get-tpu-resources-to-develop).
3. Code reviews are required.
4. Merge branch.

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

### Q1: When I add a new model for SGLang-Jax, what do I need to do before merge into main?


1. Please add accuracy and benchmark baselines for new model in PR.
2. Please add accuracy baseline in CI and refer to `Adding baselines in CI` in [test/README.md](https://github.com/sgl-project/sglang-jax/tree/main/test/README.md).

Thank you for your interest in SGLang-Jax. Happy coding!

## Q & A

### How to add tests in CI?

TODO.
We will provide a document to introduce the following parts about CI:
1. CI workflow.
2. Principles and steps.
3. How to add accuracy tests?
4. How to add benchmark tests?

### How to get TPU resources to develop?

TODO.

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

### How to do benchmark?

Please refer to [Benchmark and Profiling](./benchmark_and_profiling.md).


## Contact us

TODO.
