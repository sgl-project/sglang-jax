# Run Unit Tests

SGLang-Jax uses the built-in library [unittest](https://docs.python.org/3/library/unittest.html) as the testing framework.

## Test Backend Runtime
```bash
cd sglang-jax/test/srt

# Run single file
python3 test_abort.py

# Run a suite with multiple files on tpu
python3 run_suite.py --suite per-commit-tpu-v6e-1
```

## Adding or Updating Tests in CI

- Create new test files under `test/srt`.
- Ensure they are referenced in the respective `run_suite.py` (e.g., `test/srt/run_suite.py`) so they’re picked up in CI. For most small test cases, they can be added to the `per-commit-tpu-v6e-1` suite. Sort the test cases alphabetically.
- The CI will run the `per-commit-tpu-v6e-1` automatically. If you need special setup or custom test groups, you may modify the workflows in [`.github/workflows/`](https://github.com/sgl-project/sglang-jax/tree/main/.github/workflows).


## Writing Elegant Test Cases

- Examine existing tests in [sglang-jax/test](https://github.com/sgl-project/sglang-jax/tree/main/test/srt) for practical examples.
- Keep each test function focused on a single scenario or piece of functionality.
- Give tests descriptive names reflecting their purpose.
- Use robust assertions (e.g., assert, unittest methods) to validate outcomes.
- Clean up resources to avoid side effects and preserve test independence.
- Reduce the test time by using smaller models and reusing the server for multiple test cases.
