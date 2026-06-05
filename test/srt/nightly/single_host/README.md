# Single-host tests

Accuracy tests for models that fit on a single host (e.g. v6e-1 / v6e-4, issue #1200).

This is the single-host analogue of `../multi_host/`. The two share the same
host-neutral contract (`../cases.py`, `../profiles.py`, `../results.py`) and the
same `../launch_profiles/` directory, so adding a case looks the same on both
sides. **The only difference:** the multi-host runner coordinates several pod
ranks around one logical server, while the single-host runner launches one
server locally (`nnodes=1`) and evaluates against it directly.

## Adding a new test takes two steps

### Step 1: Write a launch profile

Create a YAML under `test/srt/nightly/launch_profiles/` describing how to start
the server. This is the **same directory and same schema** the multi-host suites
use (see `../multi_host/README.md` for the full field reference).

```yaml
# test/srt/nightly/launch_profiles/<your-model>-<target>.yaml
name: <run name>                    # label for this launch
target: <target-name>               # hardware target tag, e.g. v6e-4
model_path: <your-model-path>       # HF repo path or local path
tp_size: 4
dp_size: 1
ep_size: null                       # optional; only for MoE
port: 30010
server_args:                        # extra flags forwarded to launch_server
  - --attention-backend
  - fa
  - --dtype
  - bfloat16
  - --page-size
  - "256"
```

`--model-path / --host / --port / --tp-size / --dp-size / --ep-size / --nnodes /
--node-rank / --dist-init-addr` are managed by the launch runtime and **cannot**
be set in `server_args` (the validator rejects them). One profile == one full
server config — there is no per-run override layer.

### Step 2: Register the run in `SUITES`

Open `test/srt/nightly/single_host/suite_runner.py` and add a `SingleHostRun` to
the suite's `runs` list. A `SingleHostRun` mirrors the multi-host `ModelRun`
(`launch_profile` + `cases`):

```python
SUITES: dict[str, SingleHostSuite] = {
    "accuracy-text-models-v6e-4": SingleHostSuite(
        name="accuracy-text-models-v6e-4",
        runs=[
            # ... existing entries ...
            SingleHostRun(
                launch_profile="<your-model>-<target>.yaml",
                cases=[_gsm8k_case("<case-name>", "<hub-id-for-tokenizer>", 0.90, 64)],
            ),
        ],
    ),
}
```

`launch_profile` is the **bare filename** under `launch_profiles/` — identical to
the multi-host convention. `_gsm8k_case(name, model_id, threshold, eval_batch_size)`
is the in-file helper for the standard gsm8k gate; build an `AccuracyCase`
directly if you need a different dataset or sampling config. Set
`score_threshold=None` for a warn-only case (runs and records a score, but does
not gate / fail the suite).

## Running it

```bash
cd test/srt
python3 nightly/single_host/suite_runner.py --suite accuracy-text-models-v6e-4
```

- `--cases qwen3-8b-fa,ds-v2-lite-mla` — run only the named cases (comma-separated).
- `--dry-run` — print the resolved suite as JSON without launching anything.

Pass/fail is conveyed through the process **exit code** (same scheme as the
multi-host runner) so CI can classify the result:

| Exit code | Meaning                                              |
|-----------|-----------------------------------------------------|
| `0`       | all cases passed                                    |
| `10`      | infra / server launch failure — caller may retry    |
| `20`      | a case finished but scored below threshold          |
| `30`      | a case crashed unexpectedly (bug) — do not retry    |

## Triggering CI

The v6e-4 suite is wired into `.github/workflows/nightly-test-daily.yml` (job
`nightly-test-accuracy-text-models-4-tpu-daily`), which runs
`--suite accuracy-text-models-v6e-4` on the `arc-runner-v6e-4` runner.
