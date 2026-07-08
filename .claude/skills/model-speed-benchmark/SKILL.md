---
name: model-speed-benchmark
description: Use when running throughput/latency (serving speed) benchmarks against an already-running model server. Uses sgl_jax.bench_serving to sweep batch-size/concurrency and measure throughput and latency, producing summary.csv and per-point bs_<N>/result.jsonl. Speed only — accuracy/eval belongs to model-accuracy-benchmark. Triggers on throughput, latency, bench_serving, batch-size sweep, tok/s, TTFT, TPOT — not accuracy/eval.
---

# Model Speed Benchmark

Run a **throughput/latency** benchmark on a model that has passed the serving smoke test. By
default run on the user-specified TPU machine or remote environment, not on the local
development checkout, unless the user explicitly asks for a local-only dry run.

This skill covers speed only. Accuracy (eval) belongs to `model-accuracy-benchmark`. The two
are independent contracts: different triggers, required inputs, validation checks, and output
fields. If the user asks to "run both", invoke the two skills explicitly and produce one
report each.

## Workflow

1. Confirm the target resource, repo path/source, branch/commit, model path, host, and port.
2. Confirm the model server is already running on the target device and verify it is live. `/health` returns HTTP **200 with an empty body**, so check the status code — not the body — (e.g. `curl -s -o /dev/null -w '%{http_code}' http://HOST:PORT/health`), or treat a populated `/v1/models` response as the liveness signal.
3. Record the first served model id from `/v1/models` and check whether another throughput benchmark is already running against the same host+port.
4. Collect the serving process id (PID) that owns the endpoint port, so `Server id` in the report is never left as `unknown` when the process is reachable. See "Collecting the server id".
5. Start a fresh, unique output directory for the current request.
6. Run the batch-size sweep with `scripts/speed_benchmark.sh`.
7. If the target machine does not have the wrapper scripts, copy `scripts/` to the target checkout first.
8. Validate the produced artifacts before reporting them.
9. Write the final markdown report by filling `templates/report-template.md` (see "Output format").

### Collecting the server id

`Server id` is a required report field. Do not write `unknown` unless every probe below
fails. `/health` and `/v1/models` do not expose a PID, so query the process that owns the
serving port on the target host:

```bash
ssh <host> 'bash -lc "ss -ltnp 2>/dev/null | grep :<port> || sudo ss -ltnp 2>/dev/null | grep :<port>; pgrep -af launch_server | head"'
```

Report the PID of the process listening on the endpoint port (for the sgl-jax launcher this
is the `sgl_jax.launch_server` process). Use `exp_id` instead of PID only when the run is
managed by an experiment runner that exposes one. Write `unknown` only when the process is
genuinely unreachable, and say so in `## Notes`.

## Speed Benchmark

`scripts/speed_benchmark.sh` wraps `python -m sgl_jax.bench_serving`, sweeps batch sizes, and
writes a `summary.csv`. Pass the server's Python (`.venv/bin/python`) via `--python-bin` so
`sgl_jax` imports resolve.

The command block in the final report must show the **raw benchmark loop or one-off
invocation that actually produced the artifacts**, not just the wrapper invocation. If the
run was a batch sweep, expand the loop with the concrete `bs` values.

Default command (random workload; pass the shape explicitly via `--` — do not rely on
`bench_serving`'s implicit random defaults):

```bash
scripts/speed_benchmark.sh --dataset-name random --python-bin <repo>/.venv/bin/python \
  -- --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1
```

Defaults:

| Parameter | Default |
|---|---|
| `--host` | `127.0.0.1` |
| `--port` | `30000` |
| `--model` | unset; `bench_serving` discovers from `/v1/models` |
| `--dataset-name` | `random` |
| `--dataset-path` | unset; required by `sharegpt` / `custom` / `image` / `mmmu` / `mooncake` |
| `--out` | `/tmp/sgl_jax_speed_benchmark` |
| `--batch-sizes` | `2 4 8 16` |
| `--num-prompts-multiplier` | `3` (num_prompts = bs × multiplier) |
| `--backend` | `sgl-jax` |

Examples (wrapper flags shown; the random shape still goes after `--`):

```bash
scripts/speed_benchmark.sh --batch-sizes "32 64 128" -- --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1
scripts/speed_benchmark.sh --model mistralai/Mistral-7B-Instruct-v0.3 --out /tmp/mistral_speed -- --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1
```

### Dataset-specific flags

The wrapper is thin: it owns only dataset-agnostic concerns (the bs sweep,
`num_prompts = bs × multiplier`, the common flags, and `summary.csv` extraction). It does
**not** translate any dataset-specific flag. Pass everything dataset-specific verbatim after
`--`; it is forwarded to `bench_serving` unchanged. This is deliberate — a translation layer
is where flags get silently dropped, and it cannot scale to every dataset. `--dataset-path`
is a first-class wrapper flag because several datasets require it.

Do **not** pass `--max-concurrency` / `--num-prompts` after `--`; the wrapper sets them per
batch point.

### Per-dataset recipes

```bash
# random / random-ids — fixed shapes, the canonical speed workload
scripts/speed_benchmark.sh --dataset-name random --python-bin <repo>/.venv/bin/python \
  -- --random-input-len 16384 --random-output-len 1024 --random-range-ratio 1

# sharegpt — needs --dataset-path; --sharegpt-output-len caps the output length
scripts/speed_benchmark.sh --dataset-name sharegpt --dataset-path /data/sharegpt.json \
  --python-bin <repo>/.venv/bin/python \
  -- --sharegpt-output-len 1024

# generated-shared-prefix — shared-prefix workload
scripts/speed_benchmark.sh --dataset-name generated-shared-prefix \
  --python-bin <repo>/.venv/bin/python \
  -- --gsp-num-groups 64 --gsp-prompts-per-group 16 --gsp-system-prompt-len 2048 \
     --gsp-question-len 128 --gsp-output-len 256

# mooncake — trace-driven; needs --dataset-path
scripts/speed_benchmark.sh --dataset-name mooncake --dataset-path /data/mooncake_trace.jsonl \
  --python-bin <repo>/.venv/bin/python \
  -- --mooncake-workload conversation --mooncake-num-rounds 4

# image / mmmu / custom — mostly driven by --dataset-path
scripts/speed_benchmark.sh --dataset-name mmmu --dataset-path /data/mmmu \
  --python-bin <repo>/.venv/bin/python
```

Confirm the exact dataset-specific flags for the deployed version with
`<repo>/.venv/bin/python -m sgl_jax.bench_serving --help` before a long run — flag names are
version-sensitive.

`summary.csv` has one row per batch point with columns: `Max Concurrency`, `Input tok/s`,
`Output tok/s`, `Peak output tok/s`, `Total tok/s`, `Mean E2E ms`, `Mean TTFT ms`,
`Mean TPOT ms`, `Mean ITL ms`, `P99 ITL ms`. Report it plus every `bs_<N>/result.jsonl` path.
Pass `--model` only when auto-discovery fails or the tokenizer id must differ from the served
model id.

### Radix cache and cache hits

Radix (prefix) cache is a **server-launch** setting, enabled by default (`disable_radix_cache
= False`). When measuring input/prefill throughput you generally want it **off**, launched
with `--disable-radix-cache`; otherwise repeated or shared prefixes are served from cache and
`Input tok/s` / `TTFT` are inflated (decode-side `Output tok/s` / `TPOT` / `ITL` stay valid).

Detect the state from the captured `Server launch args`: radix cache is `disabled` iff
`--disable-radix-cache` is present, otherwise `enabled`. Confirm the effect with the
`cache_hit_rate` field in `result.jsonl` — a value near `1.0` on a fresh workload means
prefill was cache-served. Record both `Radix cache` (Status) and `Cache hit rate` (Speed
Benchmark), and if the cache was on while measuring input throughput, call it out in
`## Notes`.

Do not restart or reconfigure the server yourself. If the cache state is wrong for the
requested measurement, report it and recommend relaunching with `--disable-radix-cache`
rather than changing it unprompted.

### Concurrency conflicts and existing artifacts

- Before starting a new speed run, check whether another throughput benchmark is already running against the same host+port. If it is still active, wait or choose a fresh time window; do not intentionally overlap remote speed runs unless the user explicitly asks (it pollutes latency/throughput results).
- Do not kill an existing benchmark process unless the user explicitly asks. If the conflict does not clear in a reasonable window, stop before launching, and report the run as blocked or partial, including the conflicting PID and command when available.
- When `result.jsonl` already exists, summarize only the current run or start from a fresh output directory to avoid mixing historical results.
- Do not reuse old artifacts as the result for a new request unless the user explicitly asks for historical comparison. For normal requests: always run a fresh benchmark, always use a new unique output directory.

## Artifact validation

Before reporting success, validate the outputs that were actually produced:

- The output directory exists.
- `summary.csv` exists and has the expected header row.
- Every requested `bs_<N>/result.jsonl` exists and is non-empty.
- The number of metric rows in `summary.csv` matches the number of completed batch points.

If validation fails after a run completed, report `partial` or `failed` instead of claiming
success. If the sweep fails partway, keep the completed `bs_<N>` artifacts, report `partial`,
and do not silently discard completed points; prefer a fresh output directory for retries.
The wrapper enforces this: a failing batch point is skipped (no summary row is written for
it), the sweep continues with the remaining points, and the wrapper prints
`PARTIAL: failed batch points: ...` and exits `3`. Treat a non-zero wrapper exit with some
rows present as `partial`, and an empty `summary.csv` as `failed`.

## Output format

The report shape is defined by `templates/report-template.md`. Fill every placeholder and
keep the section order, headings, and field labels exactly as in the template — it is the
single source of truth for this skill's output. Write the report in English (headings, field
labels, notes); any brief blocking/failure note emitted mid-run should also be in English.

By default, do not show incremental progress: run the full benchmark first, then present one
consolidated report. Only interrupt this default when the run is blocked and needs user
input, the run failed and the failure itself is the result, or the user explicitly asks for
live progress.

Requirements for filling the template:

- In the command block, include the concrete command that was actually executed. If it was a batch sweep, expand the raw `bench_serving` loop with the concrete `bs` values instead of just the wrapper invocation.
- Do not omit fields within a section. If a field is unavailable, write `unknown`.
- In `## Status`, always include target resource, endpoint, model path, server health, the known PID/exp_id, `Radix cache`, the `Hardware` group, the `Versions` group, and the `Server launch args`. Speed benchmarks do not use evalscope, so the Status does **not** include an `evalscope` version.
- `Hardware`: fill TPU type (e.g. `tpu-v6e`), topology (e.g. `4x4`), and chip count (e.g. `16`). Take these from the target resource spec the user provided, or from the host's TPU metadata; write `unknown` for any that cannot be determined.
- `Backend` is the `--backend` value passed to `bench_serving` (choices: `sglang`, `sgl-jax`, `sglang-native`, `sglang-oai`, `sglang-oai-chat`, `vllm`, `vllm-chat`, `lmdeploy`, `lmdeploy-chat`, `trt`, `gserver`, `truss`). The wrapper defaults to `sgl-jax`; `sglang` and `sgl-jax` share the same `/generate` client protocol. Record exactly what was used — it determines the engine under test.
- `Versions`: `engine` names the serving engine under test (determined by `--backend`) plus its version. For the default `sgl-jax` / `sglang(-*)` backends the engine is this sglang-jax checkout — get the commit with `git -C <repo> rev-parse --short HEAD`, the branch with `git -C <repo> rev-parse --abbrev-ref HEAD`, and the package version via `<repo>/.venv/bin/python -c "import sgl_jax; print(sgl_jax.__version__)"` when available; format as `sglang-jax <version> @ <commit> (<branch>)`. For a non-sglang-jax backend (e.g. `vllm`, or upstream `sglang` on GPU) record that engine's own version instead. Collect `jax` and `libtpu` from the remote project runtime (typically `<repo>/.venv/bin/python`); they are TPU-specific, so write `n/a` when the engine under test is not on TPU.
- `Server launch args`: the full command line that started the server. Reuse the `pgrep -af launch_server` / `ps -o args= -p <PID>` output from "Collecting the server id" — the `sgl_jax.launch_server` process command line is exactly the launch args. Write `unknown` only if the process is unreachable.
- `Radix cache`: `disabled` iff `Server launch args` contains `--disable-radix-cache`, otherwise `enabled` (server default). See "Radix cache and cache hits".
- `Cache hit rate`: read `cache_hit_rate` from `result.jsonl`. If it is near `1.0` while `Radix cache` is `enabled` and you are measuring input throughput, note in `## Notes` that `Input tok/s` / `TTFT` are cache-inflated.
- `Dataset` and `Workload shape` come from the actual `bench_serving` invocation: `--dataset-name` for the dataset, and the dataset-specific shape flags passed after `--` for the workload shape (e.g. `input_len=16384, output_len=1024` for random; `sharegpt_output_len=1024` for sharegpt; `system_prompt_len/question_len/output_len` for generated-shared-prefix). Record exactly what was passed.
- Render the `summary.csv` metrics as the Markdown table in the template (use the CSV header names as columns), one row per batch point. A one-row table is still preferred over prose.
- In `## Speed Benchmark`, always include status; if completed, also include output dir, the `summary.csv` path, backend, dataset, workload shape, cache hit rate, and the metrics table.
- Prefer one concise `Artifacts` list with the primary result paths (the original paths on the target machine by default). Do not mention local copies unless the user explicitly asks.
- In `## Notes`, explain any deviation from the full workflow, for example using a wrapper but showing the expanded raw command, a conflict causing blocked/partial, or a non-value-changing header cleanup.
- Unless the user explicitly asks for live progress, do not emit in-progress sections with `unknown`/`pending`/`collecting` placeholders before execution actually completes.

State selection rules:

- `success`: every requested batch point finished and its artifacts passed validation.
- `partial`: at least one batch point produced usable artifacts but another failed, was interrupted, or is missing validated artifacts.
- `failed`: no batch point produced usable, validated artifacts.

## Remote setup

Sync the wrapper when it is missing on the target host (run from the repo root):

```bash
scp .claude/skills/model-speed-benchmark/scripts/speed_benchmark.sh <target_host>:<sglang-jax root>/scripts/
```
