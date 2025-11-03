# Contribution Guide

Welcome to **SGLang-Jax**! We appreciate your interest in contributing. This guide provides a concise overview of how to set up your environment, run tests, build documentation, and open a Pull Request (PR). Whether you’re fixing a small bug or developing a major feature, we encourage following these steps for a smooth contribution process.

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

## Run and add unit tests

If you add a new feature or fix a bug, please add corresponding unit tests to ensure coverage and prevent regression.
SGLang-Jax uses Python's built-in [unittest](https://docs.python.org/3/library/unittest.html) framework.
For detailed instructions on running tests and integrating them into CI, refer to [test/README.md](https://github.com/sgl-project/sglang-jax/tree/main/test/README.md).


## Write documentations

We recommend new contributors start from writing documentation, which helps you quickly understand SGLang-Jax codebase.

## Test the accuracy
If your code changes the model output, please run the accuracy tests. A quick sanity check is the few-shot GSM8K.

```
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
## Note:
## 1. evalscope==0.17.1 is recommended.
## 3. Please set proper sampling parameters for your model. We recommend to use configurations on Hugging Face.
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

## Benchmark the speed
Refer to [Benchmark and Profiling](./benchmark_and_profiling.md)


## Request a review
Waiting for completion.

## General code style
- Avoid code duplication. If the same code snippet (more than five lines) appears multiple times, extract it into a shared function.
- Keep files concise. If a file exceeds 2,000 lines of code, split it into multiple smaller files.
- Strive to make functions as pure as possible. Avoid in-place modification of arguments.


## Tips for newcomers

Waiting for completion.

## Q & A

### Q1: When I add a new model for SGLang-Jax, what do I need to do before merge into main?


1. Please add accuracy and benchmark baselines for new model in PR.
2. Please add accuracy baseline in CI and refer to `Adding baselines in CI` in [test/README.md](https://github.com/sgl-project/sglang-jax/tree/main/test/README.md).

Thank you for your interest in SGLang-Jax. Happy coding!
