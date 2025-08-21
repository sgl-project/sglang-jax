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
For detailed instructions on running tests and integrating them into CI, refer to #TODO.

## Write documentations

We recommend new contributors start from writing documentation, which helps you quickly understand SGLang-Jax codebase.

## Test the accuracy
If your code changes the model output, please run the accuracy tests. A quick sanity check is the few-shot GSM8K.

```
# Launch a server
python3 -m sgl-jax.launch_server --model Qwen-7B-Chat

# Evaluate By EvolScope
evalscope eval  --model Qwen-7B-Chat --api-url http://127.0.0.1:30000/v1/chat/completions --api-key EMPTY --eval-type service --datasets gsm8k --eval-batch-size 8 --limit 500
```

Please note that the above script is primarily a sanity check, not a rigorous accuracy or speed test.
This test can have significant variance (1%–5%) in accuracy due to batching and the non-deterministic nature of the inference engine.
Also, do not rely on the "Latency/Output throughput" from this script, as it is not a proper speed test.

GSM8K is too easy for state-of-the-art models nowadays. Please try your own more challenging accuracy tests.
You can find additional accuracy eval examples in:
- [test_eval_accuracy_large.py](https://github.com/sgl-project/sglang/blob/main/test/srt/test_eval_accuracy_large.py) ##TODO

## Benchmark the speed
Waiting for completion.

## Request a review
Waiting for completion

## General code style
- Avoid code duplication. If the same code snippet (more than five lines) appears multiple times, extract it into a shared function.
- Keep files concise. If a file exceeds 2,000 lines of code, split it into multiple smaller files.
- Strive to make functions as pure as possible. Avoid in-place modification of arguments.

## Tips for newcomers

TODO

Thank you for your interest in SGLang-Jax. Happy coding!
