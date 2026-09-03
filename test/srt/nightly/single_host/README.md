# Single-host tests

Accuracy and performance tests for models that fit on a single host (v6e-1 / v6e-4).

The single-host analogue of `../multi_host/`. Both share the same host-neutral
contract — `../cases.py` (case dataclasses), `../profiles.py` (launch),
`../results.py` (emitters + gate), `../drivers.py` (case drivers) — and the same
`../launch_profiles/` directory, so adding a case looks the same on both sides.
The difference: multi-host coordinates several pod ranks around one logical
server; single-host launches one local server (`nnodes=1`) and evaluates against
it directly.

`AccuracyCase` and `PerfCase` are dispatched by type to their case runner —
accuracy gates on score, perf on a dual threshold (completion + absolute floor +
trailing baseline) — sharing the same profile contract, result emitter, and
exit-code scheme.

## Adding an accuracy gate

### Step 1: Write a launch profile

A YAML under `../launch_profiles/` — the same directory and schema the multi-host
suites use (see `../multi_host/README.md` for the full field reference):

```yaml
# ../launch_profiles/<your-model>-<target>.yaml
name: <run name>
target: v6e-4                        # hardware target tag
model_path: <hf-repo-or-local-path>
tp_size: 4
dp_size: 1
ep_size: null                        # optional; only for MoE
port: 30010
server_args:                         # extra flags forwarded to launch_server
  - --attention-backend
  - fa
  - --dtype
  - bfloat16
  - --page-size
  - "256"
```

`--model-path / --host / --port / --tp-size / --dp-size / --ep-size / --nnodes /
--node-rank / --dist-init-addr` are managed by the runtime and rejected in
`server_args`. One profile == one full server config; no per-run override layer.

### Step 2: Register the run in `SUITES`

Add a `SingleHostRun` (mirrors the multi-host `ModelRun`: `launch_profile` +
`cases`) to the suite's `runs` list in `suite_runner.py`:

```python
SingleHostRun(
    launch_profile="<your-model>-<target>.yaml",
    cases=[_gsm8k_case("<case-name>", "<hub-id-for-tokenizer>", 0.90, 64)],
),
```

`_gsm8k_case(name, model_id, threshold, eval_batch_size)` is the in-file helper
for the standard gsm8k gate; build an `AccuracyCase` directly for a different
dataset. `score_threshold=None` makes a warn-only case (records a score, does not
gate).

## Adding a perf sweep

A perf case is a sweep split by intent — prefill (`c{8,32} x i{4k,8k} x o1`) and
decode (`c{…} x i4k x o1024`). Decode concurrency is KV-bound, so points are
filtered to what each server can actually run concurrently (dense ceiling ~41 →
`c{16,32}`; MoE ~118 → `c{32,64,96}`). `perf_sweep_cases(prefix, PerfParams(...))`
(in `../cases.py`) builds the points; `PerfParams` defaults to a KV-safe grid
(decode `c{32,64}`) — override grid / floors per suite.

### Step 1: Write a launch profile

Same as accuracy, with two perf conventions: one profile per variant (epmoe vs
fused are separate files) and `--disable-radix-cache` so cache hit-rate can't
skew the baseline.

### Step 2: Register the run in the perf suite

```python
SingleHostRun(
    launch_profile="<model>-<variant>-perf-v6e-4.yaml",
    cases=perf_sweep_cases("<prefix>", PerfParams(
        decode_concurrencies=(16, 32),
        profile_point=(32, 4096, 1024),
        floors={"out_tps": 865.6},          # decode repr: out_tps floor + xprof trace
        prefill_floor_point=(32, 4096, 1),
        prefill_floors={"in_tps": 15629.7}, # prefill repr: in_tps floor, no trace
    )),
),
```

`PerfParams` carries two absolute floors, each on a representative point that
must exist in the grid: `floors` (`out_tps`) on `profile_point` — the decode repr,
which also captures the xprof trace — and `prefill_floors` (`in_tps`) on
`prefill_floor_point` — the prefill repr (no trace). Floors travel next to the
case so they can't be forgotten. Every point appends a row to the per-model CSV
(`daily_performance_results_<model>_tp_<n>.csv`, the layout `plot_perf.py` and the
trailing baseline read); on main nightly the CSV publishes to ci-data and the
per-case JSON to the observability dashboard. See the `PerfParams` docstring in
`../cases.py` for more.

> `perf_sweep_cases` is single-host only. Multi-host perf is a separate, lighter
> path (completion check + dashboard JSON only — no CSV / floor / trailing gate).

## Running it

```bash
cd test/srt
python3 nightly/single_host/suite_runner.py --suite accuracy-text-models-v6e-4
python3 nightly/single_host/suite_runner.py --suite perf-text-models-v6e-4
```

- `--cases qwen3-8b-fa,qwen3-32b-c32-i4096-o1024` — run only the named cases
  (for perf, individual sweep-point names — handy for a single-point smoke run).
- `--dry-run` — print the resolved suite as JSON without launching.

Pass/fail is conveyed through the process **exit code**:

| Exit code | Meaning |
|-----------|---------|
| `0`  | all cases passed |
| `10` | infra / server launch failure — caller may retry |
| `20` | a threshold miss — accuracy below score, or a perf metric below its floor / trailing baseline |
| `30` | a case crashed, or a perf point did not complete — do not retry |

## Triggering CI

The v6e-4 suites run on the `arc-runner-v6e-4` runner via
`.github/workflows/nightly-test-daily.yml`:

- `nightly-test-accuracy-text-models-4-tpu-daily` → `--suite accuracy-text-models-v6e-4`
- `nightly-test-perf-text-models-4-tpu-daily` → `--suite perf-text-models-v6e-4`
