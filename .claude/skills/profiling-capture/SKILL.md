---
name: profiling-capture
description: "Use when you need to CAPTURE / PRODUCE XProf (jax.profiler) profiling artifacts from sgl-jax (SGLang on JAX/TPU) for later performance analysis: driving a live server (/start_profile, stage-separated prefill/decode), offline bench_one_batch --profile, or kernel microbench. This skill produces the trace files; it does not analyze them."
---

# XProf Profiling Capture

Produce XProf profiling files from sgl-jax for later analysis. sgl-jax is **JAX/TPU**, so the
profiler is `jax.profiler` and it emits **XProf/TensorBoard** artifacts (`*.xplane.pb` +
`trace.json.gz` under `plugins/profile/`), not `torch.profiler`. Scope is **capture only** —
producing the trace, not reading it. Analysis belongs to a separate skill.

## Workflow

1. Pick a capture path (A/B/C) from the table below by your situation.
2. Warm up before tracing so JIT/autotune is excluded — **never trace a cold run**. Path A
   uses `--warmup-requests`; Path B auto-warms.
3. Default to **one non-stage trace**; prefill/decode are separated at analysis time (out of
   scope here). Add `--profile-by-stage` only when you need pre-split dirs.
4. For prefill, force cache-missing prompts (`--dataset-name random` / `--disable-radix-cache`)
   plus `--flush-cache` — unless the change under test is cache-related.
5. Run the capture. The trace dir is **server-side** — confirm it is writable on the host.
6. Wait for flush: confirm `*.xplane.pb` exists and `profile_status == idle`. Whether the
   needed device compute actually landed is confirmed at analysis time, not here.
7. Prefer a **single-rank** trace when TP emits one per rank.
8. Hand off (see Handoff) — do **not** analyze; analysis is a separate skill.

## Pick a capture path

| Situation | Path |
| --- | --- |
| running HTTP server | **A** — live capture (non-stage default) |
| one batch shape, no server | **B** — `bench_one_batch --profile` |
| one kernel | **C** — kernel microbench |

## A — live server

`bench_serving --profile` is the recommended default: it warms up, arms the profiler, drives
a real dataset workload, then stops — **one trace over the whole run**. A single window
reliably captures device ops for **both** stages; separating prefill from decode is an
analysis-time step (out of scope here).

Default command:

```bash
python -m sgl_jax.bench_serving \
  --backend sgl-jax --base-url http://127.0.0.1:30000 \
  --dataset-name random --num-prompts 8 \
  --random-input-len 4096 --random-output-len 128 --warmup-requests 10 \
  --flush-cache --profile --profile-num-steps 5
```

Key flags:

| Flag | Effect |
| --- | --- |
| `--profile` | arm the profiler for this run |
| `--profile-num-steps` | number of steps to trace after warmup |
| `--warmup-requests` | untraced requests first, so JIT/autotune is excluded |
| `--flush-cache` | clear the radix cache once before the run (see Capture quality) |
| `--profile-by-stage` | write pre-split `prefill/` + `decode/` dirs (prefill unreliable) |
| `--profile-stages prefill decode` | with `--profile-by-stage`, which stages to profile |

**Need pre-split `prefill/` + `decode/` dirs** to compare the stages directly? Add
`--profile-by-stage`, optionally with `--profile-stages prefill decode` — pass one stage to
profile only that stage, omit for both.

**Manual path:** `POST /start_profile` (body = `ProfileReqInput`, may include `output_dir`) →
send requests → `POST /stop_profile` → `GET /profile_status` until `status == "idle"` (else
`"in_progress"`):

```bash
# 1. arm the profiler (body = ProfileReqInput; all fields optional).
#    Omit num_steps so the window runs until you call stop_profile below.
curl -s -X POST http://127.0.0.1:30000/start_profile \
  -H 'Content-Type: application/json' \
  -d '{"output_dir": "/tmp/myprofile"}'

# 2. send your requests here (bench_serving, curl /generate, real traffic, …)

# 3. stop, then poll status until it flushes to idle
curl -s -X POST http://127.0.0.1:30000/stop_profile
until [ "$(curl -s http://127.0.0.1:30000/profile_status | jq -r .status)" = idle ]; do sleep 1; done
```

`start_profile` / `stop_profile` also accept `GET`; `start_profile` with no body uses server
defaults. Key `ProfileReqInput` fields: `output_dir`, `num_steps`, `start_step`,
`profile_by_stage`, `profile_stages` (e.g. `["prefill","decode"]`).

**Traces are written server-side**, into `SGLANG_JAX_PROFILER_DIR` (default `/tmp`), read from
the server process's environment. To steer `bench_serving` output, export it **at server
launch** — a running process cannot be changed. The manual `/start_profile` path can pass
`output_dir` per request instead. Either way the dir must be writable on the host.

## B — offline batch

`bench_one_batch` takes the **same server args as `launch_server`**, so fill the model and
parallelism from your own deployment context, then add the batch shape and `--profile`:

Replace `...` with your launch_server config; writes to `*.tb/`.

```bash
python -m sgl_jax.bench_one_batch \
  --model-path <model> --tp-size <n> --device tpu ... \
  --batch-size 1 --input-len 4096 --output-len 128 --profile
```

- **Match your server config.** Pass the same parallelism/model flags you launch with
  (`--tp-size`, `--dp-size`, `--trust-remote-code`, `--dtype`, …); mismatched config profiles a
  different program than you run in production.
- **Auto-warms up** first (a full untraced run), so JIT/autotune is already excluded — do not
  add warmup here.
- The trace covers prefill **plus all decode steps mixed in one `*.tb/` dir**, with no stage
  separation.
- Only **tp_rank 0** writes the trace.
- Dir name pattern: `<prefix>_batch<bs>_input<il>_output<ol>.tb` (`--profile-filename-prefix`).

## C — kernel microbench

`jax.profiler.trace(dir)` around a warmed-up `run_kernel()`. Produces Pallas/Mosaic LLO
detail. For LLO / custom-call / perf-counter detail, set the dump flags **before `import jax`**
and pin libtpu ≥ 0.0.39. Full recipe — region vs counter mode, flags, verification, delivery,
existing harnesses — in [references/pallas-kernel-profiling.md](references/pallas-kernel-profiling.md).

## Capture quality

Details in [references/capture-recipes.md](references/capture-recipes.md). Musts:

- **Warm up** (`--warmup-requests 5`) so JIT/autotune is excluded (a *step* = one forward over
  the batch; decode step = one token, prefill step ≈ one prompt batch).
- **Capture both stages in one non-stage trace**; splitting prefill from decode is done at
  analysis time. Reach for `--profile-by-stage` only when you need pre-split dirs, knowing its
  prefill trace is unreliable.
- **Avoid the prefix-cache shortcut for prefill.** `--dataset-name random` gives unique
  (cache-missing) prompts, or run with `--disable-radix-cache`. Add `--flush-cache` too —
  warmup populates the radix cache, so `random` alone still traces *hits* where run prompts
  overlap warmup; flush clears that once before the run. Decode is unaffected. Exception: when
  profiling a KV/radix/prefix/HiCache change, do **not** flush — you want controlled cache
  state, and flushing turns a reuse trace into miss-then-hit.
- **Wait for flush, then confirm the capture.** `*.xplane.pb` exists and
  `profile_status == idle` are necessary but not sufficient — a truncated window still writes a
  valid-but-empty file. Whether the device compute you need actually landed is confirmed when
  the trace is opened for analysis (the device/TensorCore track is non-empty over the stage of
  interest), not from the file itself. A stage file orders of magnitude smaller than its sibling
  is under-capture, not success.
- **Prefer a single-rank trace** when TP emits one per rank.

## Output layout

```
$SGLANG_JAX_PROFILER_DIR/            # server-side; default /tmp
    plugins/profile/<run>/<host>.xplane.pb           # default: single tree (+ .trace.json.gz)
# with --profile-by-stage: split into prefill/ and decode/ subtrees instead
    prefill/plugins/profile/<run>/<host>.xplane.pb
    decode/plugins/profile/<run>/<host>.xplane.pb
```

## Handoff

Do not analyze here — this skill is capture-only. Hand the trace to a companion
analysis skill if one is installed in your environment, else open it manually:

- Point TensorBoard's profile plugin at the dir that *contains* `plugins/profile` (not the
  `.xplane.pb` file itself):

  ```bash
  pip install tensorboard tensorboard-plugin-profile
  tensorboard --logdir /tmp/myprofile   # open http://localhost:6006 → PROFILE tab
  ```

- Or load the `*.xplane.pb` directly in the standalone [XProf](https://github.com/openxla/xprof) viewer.

Either way, the dir containing `plugins/profile` is what the tool consumes. Always report:

- **Trace dir(s)** and the stage(s) captured.
- **Workload shape** — dataset, input/output len, warmup / `--profile-num-steps`.
- **Server config** — `GET /get_server_info`.
- **Verification** — `*.xplane.pb` present and `profile_status == idle`.
