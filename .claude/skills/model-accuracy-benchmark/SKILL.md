---
name: model-accuracy-benchmark
description: Use when evaluating a newly adapted model's accuracy/eval score. Runs evalscope eval (or the test/srt/run_eval.py fallback) against an OpenAI-compatible endpoint on datasets like gsm8k / aime, producing per-dataset reports and a score. Accuracy only — throughput/latency benchmarks belong to model-speed-benchmark. Triggers on accuracy, eval, evalscope, gsm8k, aime, run_eval — not throughput.
---

# Model Accuracy Benchmark

Run an **accuracy (eval)** pass on a newly adapted model after the serving smoke test has
passed. By default run on the user-specified TPU machine or remote environment, not on the
local development checkout, unless the user explicitly asks for a local-only dry run.

This skill covers accuracy only. Throughput/latency benchmarks belong to
`model-speed-benchmark`. The two are independent contracts: different triggers, required
inputs, validation checks, and output fields. If the user asks to "run both", invoke the
two skills explicitly and produce one report each.

## Workflow

1. Confirm the target resource, repo path/source, branch/commit, model path, host, and port.
2. Confirm the model server is already running on the target device and verify it is live. `/health` returns HTTP **200 with an empty body**, so check the status code — not the body — (e.g. `curl -s -o /dev/null -w '%{http_code}' http://HOST:PORT/health`), or treat a populated `/v1/models` response as the liveness signal.
3. Record the first served model id from `/v1/models` and check whether another eval is already running against the same endpoint.
4. Collect the serving process id (PID) that owns the endpoint port, so `Server id` in the report is never left as `unknown` when the process is reachable. See "Collecting the server id".
5. Start a fresh, unique work directory for the current request, following "Output directory layout".
6. Run the accuracy benchmark with `evalscope eval` (preferred path).
7. If `evalscope` cannot be installed or invoked on the target environment, fall back to `scripts/accuracy_benchmark.sh` (wraps `test/srt/run_eval.py`).
8. If the target machine does not have the wrapper scripts, copy `scripts/` to the target checkout first.
9. Validate the produced artifacts before reporting them.
10. Write the final markdown report by filling `templates/report-template.md` (see "Output format").

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

### Output directory layout

Use one timestamped, unique directory per request. For a standalone accuracy run a flat
directory is fine:

```
/tmp/evalscope_<dataset>_<limit-or-full>_<YYYYMMDD_HHMMSS>/
```

Generate the timestamp on the target host (for example `date +%Y%m%d_%H%M%S`). Keep
`--work-dir` unique per run/per report to avoid mixing artifacts across runs.

## Accuracy Benchmark

**Preferred path**: run `evalscope eval` directly on the target host. Use this path whenever
`evalscope` is available via `uv tool install evalscope` or is already installed. This is
the default for all accuracy benchmarks, including datasets not wired into the repo-local
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

Compatibility and quoting notes:

- The raw JSON string form `--generation-config '{"max_tokens":4096,"temperature":0.0}'` is confirmed working on `evalscope` 1.8.1. Prefer it.
- Some older `evalscope` versions expect the `key=value,key=value` form (`--generation-config max_tokens=4096,temperature=0.0`). If the JSON form is rejected, probe with `evalscope eval --help` or a one-sample run, then switch formats explicitly.
- **SSH quoting hazard**: nesting `ssh <host> 'bash -lc "... --generation-config {...} ..."'` strips the inner JSON quotes and breaks the argument (`ValueError: not enough values to unpack`). Write the command to a small remote script and run that, or otherwise guarantee the JSON quotes survive both shell layers.
- Treat `evalscope` CLI incompatibility as an invocation failure. If it cannot be resolved quickly, use `scripts/accuracy_benchmark.sh` and note the deviation in `## Notes`.

Use concrete dataset names in `--datasets`, for example `gsm8k`, `aime24`, or `aime25`.
Prefer `--api-url` for normal OpenAI-compatible endpoints. Pass `--model` when the served
model id should be explicit in the report or when auto-discovery is unreliable.

**Fallback path**: use `scripts/accuracy_benchmark.sh` only if `evalscope` cannot be
installed or invoked on the target host. This wrapper calls `test/srt/run_eval.py` (not
`evalscope eval`) and supports only the benchmark names exposed by that repo-local runner.
Its artifact layout differs from `evalscope`: `run_eval.py` writes
`/tmp/<eval_name>_<model>.json` and `/tmp/<eval_name>_<model>.html`. When using the fallback,
validate and report those `/tmp` files instead of the `evalscope`
`reports/<model_id>/<dataset>.json` path.

Fallback command:

```bash
scripts/accuracy_benchmark.sh --repo-dir <sglang-jax root> --python-bin <repo>/.venv/bin/python --base-url http://127.0.0.1:30000 --eval-name gsm8k --num-examples 30 --max-tokens 4096 --temperature 0.0
```

## Existing artifacts

Do not reuse old eval artifacts as the result for a new request unless the user explicitly
asks for historical comparison or artifact inspection. For normal requests: always run a
fresh eval; always use a new unique work directory; do not report an old artifact as if it
were produced by the current run. Old artifacts may still be inspected for debugging,
validation, or comparison, but they do not replace the current run.

## Artifact validation

Before reporting success, validate the outputs that were actually produced:

- The work directory exists.
- For `evalscope`, `reports/<model_id>/<dataset>.json` exists and is readable.
- For the fallback wrapper, `/tmp/<eval_name>_<model>.json` and its matching `.html` report
  exist and are readable.
- The reported sample count matches the requested `--limit` when a limit was specified.
- The score can be read from the report artifact.

If validation fails after a run completed, report `partial` or `failed` instead of claiming
success.

## Output format

The report shape is defined by `templates/report-template.md`. Fill every placeholder and
keep the section order, headings, and field labels exactly as in the template — it is the
single source of truth for this skill's output. Write the report in English (headings, field
labels, notes); any brief blocking/failure note emitted mid-run should also be in English.

By default, do not show incremental progress: run the full eval first, then present one
consolidated report. Only interrupt this default when the run is blocked and needs user
input, the run failed and the failure itself is the result, or the user explicitly asks for
live progress.

Requirements for filling the template:

- Repeat one `### {dataset}` block per dataset that was actually run.
- In the command block, include the concrete command that was actually executed.
- Do not omit fields within a section. If a field is unavailable, write `unknown`.
- In `## Status`, always include target resource, endpoint, model path, server health, the known PID/exp_id, the `Hardware` group, the `Versions` group (`engine (sglang-jax)`, `evalscope`, `jax`, `libtpu`), and the `Server launch args`.
- `Hardware`: fill TPU type (e.g. `tpu-v6e`), topology (e.g. `4x4`), and chip count (e.g. `16`). Take these from the target resource spec the user provided, or from the host's TPU metadata; write `unknown` for any that cannot be determined.
- `engine (sglang-jax)` is the served engine's version, git commit, and branch — get the commit with `git -C <repo> rev-parse --short HEAD`, the branch with `git -C <repo> rev-parse --abbrev-ref HEAD`, and the package version via `<repo>/.venv/bin/python -c "import sgl_jax; print(sgl_jax.__version__)"` when available; format as `<version> @ <commit> (<branch>)`. This ties the score to a specific build for regression tracking.
- `Server launch args`: the full command line that started the server. Reuse the `pgrep -af launch_server` / `ps -o args= -p <PID>` output from "Collecting the server id" — the `sgl_jax.launch_server` command line is exactly the launch args, and its flags (`--tp-size`, `--attention-backend`, `--dtype`, `--page-size`, ...) can affect scores. Write `unknown` only if the process is unreachable.
- Prefer concrete version command results from the target environment; do not assume all versions come from the same environment.
- Collect `evalscope` as a `uv tool` install first, not from the project `.venv`. On remote hosts run version checks through a login shell (`ssh <host> 'bash -lc "evalscope --version || uv tool list"'`) so `~/.local/bin` is present; if still not found, explicitly try `~/.local/bin/evalscope --version`. Only write `unknown` if all checks fail.
- Collect `jax` and `libtpu` from the remote project runtime (typically `<repo>/.venv/bin/python`) so they reflect the actual eval execution environment.
- Do not use `.venv/bin/python -c 'import importlib.metadata ...'` to infer the `evalscope` version — the `uv tool install` environment is separate from the project virtualenv and this lookup can be wrong.
- Prefer one concise `Artifacts` list with the primary result paths (the original paths on the target machine by default). Do not mention local copies unless the user explicitly asks.
- In `## Notes`, explain any deviation from the full workflow, for example using a wrapper, inspecting an old artifact for debugging only, or a non-value-changing header cleanup.
- Unless the user explicitly asks for live progress, do not emit in-progress sections with `unknown`/`pending`/`collecting` placeholders before execution actually completes.

State selection rules:

- `success`: every requested dataset finished and its artifacts passed validation.
- `partial`: at least one dataset produced usable artifacts but another failed, was interrupted, or is missing validated artifacts.
- `failed`: no dataset produced usable, validated artifacts.

Retry: if accuracy fails before producing a valid report and the failure is clearly due to
command construction or CLI argument format, retry once with a corrected invocation; prefer
a fresh work directory for retries.

## Remote setup

Sync the wrapper when it is missing on the target host (run from the repo root):

```bash
scp .claude/skills/model-accuracy-benchmark/scripts/accuracy_benchmark.sh <target_host>:<sglang-jax root>/scripts/
```

Install `evalscope` when it is missing on the target environment:

```bash
uv tool install evalscope
```
