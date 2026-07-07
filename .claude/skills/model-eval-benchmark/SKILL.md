---
name: model-eval-benchmark
description: Use when evaluating or benchmarking a newly adapted model, including evalscope accuracy passes and serving throughput benchmarks across a batch-size sweep with a summary.csv, output directory inspection, and upstream baseline comparison.
---

# Model Eval Benchmark

Use this skill to evaluate a newly adapted model after the serving smoke test has passed.
Run benchmarks on the user-specified TPU machine or remote environment, not on the local
development checkout, unless the user explicitly asks for local-only dry runs.

## Workflow

1. Confirm the target resource, repo path/source, branch/commit, model path, host, and port.
2. Confirm the model server is already running on the target device and verify `/health`.
3. Record the first served model id from `/v1/models` and check whether another benchmark is already running against the same endpoint.
4. Collect the serving process id (PID) that owns the endpoint port, so `Server id` in the report is never left as `unknown` when the process is reachable. See "Collecting the server id" below.
5. Start a fresh run for the current request. Use a new unique output directory for every benchmark run, following the "Output directory layout" convention below.
6. Run an accuracy benchmark with `evalscope eval`.
7. If `evalscope` cannot be installed or invoked from the target environment, fall back to `scripts/accuracy_benchmark.sh` (wraps `test/srt/run_eval.py`).
8. Run a speed benchmark with `scripts/speed_benchmark.sh`.
9. If the target machine does not have the wrapper scripts, copy `model-eval-benchmark/scripts/` to the target checkout first.
10. Validate the produced artifacts before reporting them.
11. Render the final markdown benchmark document with `scripts/render_benchmark_report.py` when the helper still matches the commands and artifacts that were actually produced.

### Collecting the server id

`Server id` in the report is a required field. Do not write `unknown` unless every probe
below fails. `/health` and `/v1/models` do not expose a PID, so query the process that owns
the serving port on the target host:

```bash
ssh <host> 'bash -lc "ss -ltnp 2>/dev/null | grep :<port> || sudo ss -ltnp 2>/dev/null | grep :<port>; pgrep -af launch_server | head"'
```

Report the PID of the process listening on the endpoint port (for the sgl-jax launcher this
is the `sgl_jax.launch_server` process). Use `exp_id` instead of PID only when the run is
managed by an experiment runner that exposes one. Write `unknown` only when the process is
genuinely unreachable, and say so in `## Notes`.

### Output directory layout

Use one timestamped run root per request and place each benchmark in a fixed subdirectory,
so accuracy and speed artifacts for the same request stay together:

```
/tmp/model_eval_<dataset>_<limit-or-full>_bs<bs-list>_<YYYYMMDD_HHMMSS>/
  accuracy/    # evalscope --work-dir
  speed/       # speed_benchmark.sh --out
```

For example, a gsm8k `--limit 20` run with a `1 2 4` batch sweep uses run root
`/tmp/model_eval_gsm8k_20_bs1-2-4_<YYYYMMDD_HHMMSS>`, with `--work-dir <root>/accuracy` and
`--out <root>/speed`. Generate the timestamp on the target host (for example with
`date +%Y%m%d_%H%M%S`). For accuracy-only runs a flat `/tmp/evalscope_<dataset>_<limit-or-full>_<YYYYMMDD_HHMMSS>`
directory is acceptable.

If the user says "two benchmarks" without more detail, run one accuracy benchmark and one
speed benchmark. If they say "two speed benchmarks", run the first two batch-size points
from the requested/default sweep.

## Accuracy Benchmark

Preferred path: run `evalscope eval` directly on the target host. Use this path whenever
`evalscope` is available via `uv tool install evalscope` or is already installed. This is
the default for all benchmarks, including ones that are not wired into the repo-local
`run_eval.py` wrapper.

Default command:

```bash
evalscope eval --model <served model id or path> --api-url http://HOST:PORT/v1/chat/completions --eval-type openai_api --datasets <dataset> --eval-batch-size 16 --generation-config '{"max_tokens":4096,"temperature":0.0}' --work-dir <unique work dir> --no-timestamp
```

Defaults:

| Parameter | Default |
|---|---|
| `--model` | unset; if omitted, prefer the first model `id` from `/v1/models` |
| `--api-url` | `http://127.0.0.1:30000/v1/chat/completions` |
| `--eval-type` | `openai_api` |
| `--datasets` | `gsm8k` |
| `--eval-batch-size` | `16` |
| `--limit` | unset; use dataset default unless the user specifies a sample count |
| `--generation-config.max_tokens` | `4096` |
| `--generation-config.temperature` | `0.0` |
| `--work-dir` | `/tmp/evalscope_results` |

Examples:

```bash
evalscope eval --model /models/mistralai/Mistral-7B-Instruct-v0.3 --api-url http://127.0.0.1:30000/v1/chat/completions --eval-type openai_api --datasets aime24 --limit 30 --eval-batch-size 16 --generation-config '{"max_tokens":4096,"temperature":0.0}' --work-dir /tmp/evalscope_aime24 --no-timestamp
evalscope eval --datasets gsm8k --limit 10 --work-dir /tmp/evalscope_gsm8k_10 --no-timestamp
```

Important compatibility and quoting notes:

- The raw JSON string form `--generation-config '{"max_tokens":4096,"temperature":0.0}'` is confirmed working on `evalscope` 1.8.1. Prefer it.
- Some older `evalscope` versions expect the `key=value,key=value` form (`--generation-config max_tokens=4096,temperature=0.0`) instead. If the JSON form is rejected, probe with `evalscope eval --help` or a one-sample run, then switch formats explicitly.
- Quoting hazard over SSH: nesting `ssh <host> 'bash -lc "... --generation-config {...} ..."'` strips the inner JSON quotes and breaks the argument (`ValueError: not enough values to unpack`). Write the command to a small remote script and run that script, or otherwise guarantee the JSON quotes survive both shell layers.
- Treat `evalscope` CLI incompatibility as an invocation failure. If it cannot be resolved quickly, use `scripts/accuracy_benchmark.sh` and note the deviation in `## Notes`.

Keep `--work-dir` unique per run or per report to avoid mixing artifacts across runs.
Prefer a descriptive timestamped directory such as
`/tmp/model_eval_<dataset>_<limit-or-full>_<speed-bs-list>_<YYYYMMDD_HHMMSS>` for combined
runs, or `/tmp/evalscope_<dataset>_<limit-or-full>_<YYYYMMDD_HHMMSS>` for accuracy-only
runs.
Use concrete dataset names in `--datasets`, for example `gsm8k`, `aime24`, or `aime25`.
Prefer `--api-url` for normal use with OpenAI-compatible endpoints. Pass `--model` when
the served model id should be explicit in the report or when auto-discovery is unreliable.

Fallback path: use `scripts/accuracy_benchmark.sh` only if `evalscope` cannot be installed
or invoked on the target host. This wrapper calls `test/srt/run_eval.py`, not `evalscope
eval`, and supports only the benchmark names exposed by that repo-local runner.

Fallback command:

```bash
scripts/accuracy_benchmark.sh --repo-dir <sglang-jax root> --python-bin <repo>/.venv/bin/python --base-url http://127.0.0.1:30000 --eval-name gsm8k --num-examples 30 --max-tokens 4096 --temperature 0.0
```


## Speed Benchmark

A wrapper around `python -m sgl_jax.bench_serving` that sweeps batch sizes and writes a
`summary.csv`. Pass the server's Python (`.venv/bin/python`) via `--python-bin` so
`sgl_jax` imports resolve.

When you need the underlying runnable benchmark command instead of the wrapper, use
`python -m sgl_jax.bench_serving` directly and loop over the concurrency points you want
to measure.
Wrapper execution is acceptable, but the final answer should still show the concrete raw
benchmark loop or one-off invocation that actually produced the artifacts.
Do not present only the wrapper invocation as the command block when the actual run was a
batch-sweep benchmark.

Default command:

```bash
scripts/speed_benchmark.sh --python-bin <repo>/.venv/bin/python
```

Defaults:

| Parameter | Default |
|---|---|
| `--host` | `127.0.0.1` |
| `--port` | `30000` |
| `--model` | unset; `bench_serving` discovers from `/v1/models` |
| `--dataset-name` | `random` |
| `--out` | `/tmp/sgl_jax_speed_benchmark` |
| `--batch-sizes` | `2 4 8 16` |
| `--num-prompts-multiplier` | `3` |
| `--input-len` | `1024` |
| `--output-len` | `4096` |
| `--backend` | `sgl-jax` |

Examples:

```bash
scripts/speed_benchmark.sh --batch-sizes "32 64 128"
scripts/speed_benchmark.sh --model mistralai/Mistral-7B-Instruct-v0.3 --out /tmp/mistral_speed
scripts/speed_benchmark.sh --dataset-name sharegpt --output-len 1024
```

Raw benchmark form for a ShareGPT sweep:

```bash
out=/tmp/sgl_jax_speed_benchmark_20260707_0001

for bs in 1 2; do
  num_prompts=$((bs * 3))
  mkdir -p "$out/bs_${bs}"

  /home/gcpuser/sky_workdir/sglang-jax/.venv/bin/python -m sgl_jax.bench_serving \
    --backend sgl-jax \
    --host 127.0.0.1 \
    --port 30000 \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 4096 \
    --random-range-ratio 1 \
    --max-concurrency "$bs" \
    --num-prompts "$num_prompts" \
    --request-rate inf \
    --output-file "$out/bs_${bs}/result.jsonl"
done
```

The `summary.csv` has one row per batch point with columns: `Max Concurrency`,
`Input tok/s`, `Output tok/s`, `Peak output tok/s`, `Total tok/s`, `Mean E2E ms`,
`Mean TTFT ms`, `Mean TPOT ms`, `Mean ITL ms`, `P99 ITL ms`. Report it plus every
`bs_<N>/result.jsonl` path.
When `result.jsonl` already exists, summarize only the current run or start from a fresh
output directory to avoid mixing historical results.
Pass `--model` only when auto-discovery fails or the tokenizer id must differ from the
served model id.
For non-`random` datasets, the script omits the random-length flags and forwards the
dataset-specific `bench_serving` options unchanged.

For remote serving, run the accuracy and speed benchmarks serially unless the user asks
for overlap. This avoids cross-benchmark interference in latency and throughput results.

Before starting a new speed run, check whether another throughput benchmark is already
running against the same host and port. If it is still active, either wait for it to
finish or choose a fresh time window. Do not intentionally overlap remote speed runs
unless the user explicitly asks for it.

Do not kill an existing benchmark process unless the user explicitly asks you to do so.
If another benchmark is already running on the same endpoint, wait and re-check until the
conflict clears. If it does not clear in a reasonable time window, stop before launching
the new speed run and report the run as blocked or partial, including the conflicting PID
and command when available.

## Existing Artifacts

Do not reuse old benchmark artifacts as the result for a new request unless the user
explicitly asks for historical comparison or artifact inspection.

For normal benchmark requests:

- Always run a fresh benchmark.
- Always use a new unique output directory.
- Do not report an old artifact as if it were produced by the current run.

Old artifacts may still be inspected for debugging, validation, or comparison, but they do
not replace the current run.

## Artifact Validation

Before reporting success, validate the outputs that were actually produced.

Accuracy validation checklist:

- The work directory exists.
- `reports/<model_id>/<dataset>.json` exists and is readable.
- The reported sample count matches the requested `--limit` when a limit was specified.
- The score can be read from the report artifact.

Speed validation checklist:

- The output directory exists.
- `summary.csv` exists and has the expected header row.
- Every requested `bs_<N>/result.jsonl` exists and is non-empty.
- The number of metric rows in `summary.csv` matches the number of completed batch points.

If validation fails after a run completed, report `partial` or `failed` instead of
claiming success.

## Required Output Format

All user-visible responses produced while using this skill should follow one fixed markdown
shape. Do not switch between prose-only answers, ad hoc bullet lists, and free-form report
layouts.

Always write the final benchmark report in English, including every section heading, field
label, table header, and note. This is a fixed requirement even when the surrounding
conversation is in another language. Any brief blocking or failure note emitted mid-run
should also be in English.

By default, do not show incremental benchmark progress to the user. Execute the full
benchmark workflow first, then present one consolidated report after all requested
benchmarks have finished. Only interrupt this default behavior when:

- the run is blocked and needs user input,
- the run has failed and the failure itself is the result to report, or
- the user explicitly asks for live progress updates.

For long-running benchmarks, prefer silence over partial status reports. If an intermediate
update is unavoidable, keep it to a single concise blocking/failure note and still emit the
final report in the required format after execution ends.

The final answer must use these top-level sections in this order when they are applicable:

```md
# Benchmark

## Status

- State: success | partial | failed
- Target resource: ...
- Endpoint: ...
- Model path: ...
- Server: healthy | unhealthy | unknown
- Server id: PID or exp_id, or `unknown`
- Versions:
  - `evalscope`: ...
  - `jax`: ...
  - `libtpu`: ...

## Accuracy Benchmark

### {{dataset}}

```bash
actual command
```

- Sample count: ...
- Sampling params: `temperature=...`, `top_p=...`
- Score: ...
- Summary: ...
- Artifacts: ...

## Speed Benchmark

```bash
actual command
```

- Status: completed | failed
- Output dir: ...
- Summary CSV: ...
- Metrics: ...
- Artifacts: ...

## Notes

- ...
```

Requirements for filling the template:

- Replace `{{dataset}}` with the actual accuracy dataset name, for example `gsm8k` or `aime25`.
- If multiple accuracy datasets are run, repeat one `### {{dataset}}` block per dataset under `## Accuracy Benchmark`.
- Include `## Accuracy Benchmark` only when at least one accuracy benchmark was actually run.
- If no accuracy benchmark was run, omit the entire `## Accuracy Benchmark` section.
- Include `## Speed Benchmark` only when a speed benchmark was actually run.
- If no speed benchmark was run, omit the entire `## Speed Benchmark` section.
- In each command block, include the concrete command that was actually executed.
- For sections that are included, do not omit required fields within that section. If a field is unavailable, write `unknown`.
- In `## Status`, always include the target resource, endpoint, model path, server health, PID/`exp_id` when known, and the versions of `evalscope`, `jax`, and `libtpu`.
- For version collection, prefer concrete command results from the target environment. Do not assume all versions come from the same environment.
- Collect `evalscope` as a `uv tool` installation first, not from the project `.venv`. On remote hosts, run version checks through a login shell such as `ssh <host> 'bash -lc "evalscope --version || uv tool list"'` so `~/.local/bin` is present in `PATH`. If the login shell still cannot find it, explicitly try `~/.local/bin/evalscope --version` and `~/.local/bin/uv tool list` before falling back. Only write `unknown` if all of these checks fail.
- Collect `jax` and `libtpu` from the remote project runtime, typically via `<repo>/.venv/bin/python`, because those versions should reflect the actual benchmark execution environment.
- Do not use `.venv/bin/python -c 'import importlib.metadata ...'` to infer the `evalscope` version when `evalscope` is installed via `uv tool install`; that lookup can be wrong because the tool environment is separate from the project virtualenv.
- In `## Accuracy Benchmark`, always include at least sample count, sampling params, score, and artifact path.
- In `## Speed Benchmark`, always include status. If completed, also include output directory, `summary.csv` path, and a concise throughput/latency summary.
- When the speed benchmark includes one or more batch/concurrency points, render the concrete metrics from `summary.csv` as a Markdown table in the final answer instead of a long inline sentence. Prefer the CSV header names (for example `Max Concurrency`, `Input tok/s`, `Output tok/s`, `Total tok/s`, `Mean E2E ms`, `Mean TTFT ms`, `Mean TPOT ms`, `Mean ITL ms`, `P99 ITL ms`) as table columns. A one-row table is still preferred over prose when `summary.csv` is available.
- For speed benchmark command blocks, show the raw benchmark loop or the exact one-off `bench_serving` invocation that produced the artifacts. If the benchmark was a batch sweep, expand the loop with the concrete `bs` values that were run.
- In artifact reporting, prefer one concise `Artifacts` line or list containing the primary result paths the user can use directly. Do not add separate `local` versus `remote` labels unless the distinction matters for the task.
- By default, prefer the original benchmark output paths on the target machine in the final answer. Do not mention copied local artifact paths unless the user explicitly asks for local copies, the workflow depends on local post-processing, or the remote paths are no longer sufficient for follow-up work.
- In `## Notes`, explain any deviation from the full workflow, for example "user requested accuracy only, so speed benchmark was not run."
- In `## Notes`, also explain when a wrapper command was used but the report shows the expanded raw command, when an old artifact was inspected only for debugging or comparison, or when a validation fix such as header cleanup was needed without changing benchmark values.
- Keep wording stable across runs. Prefer the same field names every time.
- Unless the user explicitly asks for live progress, do not emit in-progress benchmark sections with placeholder values such as `unknown`, `pending`, `collecting`, or `not run` before execution is actually complete. Wait until all requested benchmarks finish, then fill the template once with concrete results.

State selection rules:

- Use `success` only when every requested benchmark finished and the artifacts passed validation.
- Use `partial` when at least one requested benchmark finished with usable artifacts but another requested benchmark failed, was interrupted, or is missing validated artifacts.
- Use `failed` when no requested benchmark produced usable validated artifacts.

Retry guidance:

- If accuracy fails before producing a valid report, retry once with corrected invocation when the failure is clearly due to command construction or CLI argument format.
- If speed fails partway through a sweep, keep the completed `bs_<N>` artifacts, report `partial`, and do not silently discard completed points.
- Prefer a fresh output directory for retries to avoid mixing partial and retried artifacts.

Preferred rendering flow:

1. Run the benchmarks and collect the concrete commands that were executed.
2. Save raw artifacts such as the evalscope result json/work directory, benchmark log, `summary.csv`, and `bs_<N>/result.jsonl`.
3. Generate the final markdown document directly, or with `scripts/render_benchmark_report.py` if the helper still matches the executed commands and produced artifacts.

Remote sync step when wrappers are missing:

```bash
scp model-eval-benchmark/scripts/accuracy_benchmark.sh model-eval-benchmark/scripts/speed_benchmark.sh <target_host>:<sglang-jax root>/scripts/
```

Install step when `evalscope` is missing on the target environment:

```bash
uv tool install evalscope
```

Example:

```bash
python3 model-eval-benchmark/scripts/render_benchmark_report.py \
  --out model-eval-benchmark/benchmark.md \
  --dataset aime25 \
  --accuracy-command 'evalscope eval --model /models/mistralai/Mistral-7B-Instruct-v0.3 --api-url http://127.0.0.1:30000/v1/chat/completions --eval-type openai_api --datasets aime25 --limit 30 --eval-batch-size 16 --generation-config '\''{"max_tokens":4096,"temperature":0.0}'\'' --work-dir /tmp/evalscope_aime25 --no-timestamp' \
  --speed-command 'cd /remote/sglang-jax && .venv/bin/python -m sgl_jax.bench_serving --backend sgl-jax --host 127.0.0.1 --port 30000 --dataset-name sharegpt --sharegpt-output-len 1024 --max-concurrency 1 --num-prompts 3 --request-rate inf --output-file /tmp/sharegpt/bs_1/result.jsonl' \
  --target-resource tpu-v6e \
  --endpoint http://127.0.0.1:30000 \
  --model-path /models/mistralai/Mistral-7B-Instruct-v0.3 \
  --sample-count 10 \
  --server-id 343391 \
  --accuracy-json /tmp/evalscope_aime25/reports/report.json \
  --accuracy-artifact /tmp/evalscope_aime25 \
  --speed-output-dir /tmp/sharegpt_bench_20260706_081452 \
  --speed-summary-csv /tmp/sharegpt_bench_20260706_081452/summary.csv \
  --speed-kv dataset=sharegpt \
  --speed-kv output_length=1024 \
  --speed-artifact /tmp/sharegpt_bench_20260706_081452/bs_1/result.jsonl \
  --speed-artifact /tmp/sharegpt_bench_20260706_081452/bs_2/result.jsonl
```
