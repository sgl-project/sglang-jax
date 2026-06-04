# Multi-host tests

Accuracy and performance tests for large models (DeepSeek, MiMo, and others that require multi-host deployment).

Model owners just declare a suite; CI takes care of provisioning the runtime environment, coordinating rank lifecycles, executing the tests, and cleaning up resources. **No need to edit workflows or infrastructure configs.**

## Adding a new test takes two steps

### Step 1: Write a launch profile

Create a YAML under `launch_profiles/` describing how to start the server.

```yaml
# launch_profiles/<your-model>-<target>.yaml
name: <run name>                    # label for this launch
target: <target-name>               # runtime environment identifier (provided by CI, see below)
model_path: <your-model-path>       # model path
tp_size: 16
dp_size: 1
ep_size: null                       # optional; only for MoE
port: 30000
server_args:                        # extra flags forwarded to launch_server
  - --trust-remote-code
  - --dtype
  - bfloat16
  - --context-length
  - "8192"
```

#### Field reference

| Field                             | Required           | Description                                                                                                                                                                                                                                                              |
|-----------------------------------|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `name`                            | Yes                | Label for this launch (appears in logs and artifact paths)                                                                                                                                                                                                               |
| `target`                          | Yes                | Runtime environment identifier. Allowed values are maintained by CI; pick one that CI currently supports                                                                                                                                                                 |
| `model_path`                      | Yes                | Path the model is reachable at inside the runtime environment, per the CI-side storage convention                                                                                                                                                                        |
| `tp_size` / `dp_size` / `ep_size` | tp/dp required     | Parallelism config. `tp_size * dp_size * (ep_size or 1)` must equal the total accelerator count provided by the target                                                                                                                                                   |
| `port`                            | No (default 30000) | HTTP server port                                                                                                                                                                                                                                                         |
| `server_args`                     | No                 | Any flag you want forwarded to `sgl_jax.launch_server`. `--model-path / --host / --port / --tp-size / --dp-size / --ep-size / --nnodes / --node-rank / --dist-init-addr` are managed by the framework and **cannot** be overridden here (the validator will reject them) |

### Step 2: Register the suite in `SUITES`

Open `test/srt/multi_host/suite_runner.py` and add an entry to the `SUITES` dict at the top:

```python
SUITES: dict[str, MultiHostSuite] = {
    # ... existing entries ...
    "<unique-suite-name>": MultiHostSuite(
        name="<unique-suite-name>",
        runs=[
            ModelRun(
                launch_profile="launch_profiles/<your-model>-<target>.yaml",
                cases=[
                    PerfCase(
                        name="random-1024-128",
                        input_len=1024,
                        output_len=128,
                        num_prompts=24,
                        max_concurrency=8,
                    ),
                    AccuracyCase(
                        name="gsm8k-smoke",
                        dataset="gsm8k",
                        model_id="<hub-id-for-tokenizer>",
                    ),
                ],
            ),
        ],
    ),
}
```

Same explicit-dict style as the existing `test/srt/run_suite.py` — adding a new suite means adding one dict entry.

## Triggering CI

The multi-host suite is run by `.github/workflows/multi-host-test.yml`, currently triggered manually via `workflow_dispatch` (Actions tab → "Run workflow"). It does not auto-run on push or PR.

Once dispatched, CI brings up the runtime environment for the supported target and tells `suite_runner` via the `TARGET` environment variable, which then selects suites whose `target` field matches. As long as your suite uses a target CI currently supports, it gets picked up automatically — you don't need to worry about how the environment is provisioned.
