# FULL+SWA L2 HiCache

## Scope and completion rules

Enable with `--enable-unified-radix-tree --hicache-storage none`. The supported
route is single-host UnifiedRadixCache with exactly FULL and SWA components,
using `jax` or `raiden` transfers and `write_through` or `write_back`. Existing
FULL-only HiCache and disabled-HiCache paths keep their existing behavior.
Hybrid recurrent, speculative decoding, PD disaggregation, legacy SWA cache,
and L3 storage are rejected for this route.

FULL and SWA use separate host pools, controller instances, device reservations,
and per-DP-rank budgets. A host hit is usable only when FULL ancestry and the
required SWA window are continuous across device and completed host copies.
Pending backup reservations are private. SWA restore accounts for entire
covering nodes, including page-aligned boundary overhang. Independent SWA
write-back covers internal-node eviction as well as FULL-leaf eviction.

Restore reserves missing FULL and SWA pages independently, preserves resident
FULL addresses during SWA-only healing, and publishes tree values and mappings
only after both transfers complete. JAX hybrid restore is synchronous at
admission: it stages host data, waits for the donation barrier, then scatters
before `prepare_for_extend` can consume mappings. This does not establish
transfer/compute overlap. Native completion failures quarantine uncertain
resources; reset or flush must not recycle them. Restart a failed dedicated
process. CPU tests use a fake native engine and do not validate Raiden transport.

## Local CPU regression

The implementation checkpoint on 2026-09-24 passed 24 targeted suites (500
JUnit items including subtests; no failures, errors or skips), covering existing
FULL-only/recurrent/device-only SWA behavior and the new hybrid tests. This is
CPU evidence only. New cases are registered in the existing `test/srt/run_suite.py`
CPU suite; DP tests select four simulated devices in their own process.

For a focused component run:

```bash
USE_DEVICE_TYPE=cpu JAX_PLATFORMS=cpu \
  XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  .venv/bin/python -m pytest -q \
  python/sgl_jax/test/mem_cache/test_hybrid_hicache.py \
  python/sgl_jax/test/mem_cache/test_hybrid_hicache_core.py \
  python/sgl_jax/test/mem_cache/test_hybrid_hicache_core_dp.py \
  python/sgl_jax/test/mem_cache/test_hybrid_hicache_scheduler.py
```

The test-only scatter preserves rank-local indexing on CPU. Fake native futures
exercise completion and failure ownership but cannot validate TPU DMA, buffer
registration, model outputs or performance.

## Manual acceptance

This is a **manual, text-only TPU acceptance protocol**, not a claim that the
models or transfer backends have passed. Run it after the FULL+SWA L2 route is
implemented and unit tests pass. Do not substitute CPU simulation, launch flags,
or `cached_tokens` for physical D2H/H2D evidence.

## Freeze the run before starting

Use one single-host v7x `2x2x1` slice (4 chips, 8 JAX devices) per run. Pin the
code HEAD, dependencies, model and tokenizer revisions, sampling seed 3, greedy decoding,
request token IDs, RID, request order, concurrency, output limit, and device KV
budget. Use separate fresh servers for each of `off`, `off_repeat`, `on`, and
`on_repeat`; all four use UnifiedRadixCache. The off runs omit HiCache backend
and write-policy options. A single off pair may be shared across on combinations
only if code, dependency, hardware, revisions, KV budget, parallelism, overlap,
requests, and generation settings are identical; record that mapping.

| Model | Fixed target configuration | Main cases |
| --- | --- | --- |
| `google/gemma-4-31B-it` | TP8, DP1, attention TP8, EP1, page 128, actual SWA window 1024 | FULL+SWA and SWA-only restore, page/window boundaries, chunked prefill, finish/retract/abort/reset |
| `XiaomiMiMo/MiMo-V2.5` | TP8, DP2, attention TP4, EP8, page 256, actual SWA window 128, `--swa-full-tokens-ratio 0.25` | rank 0/1 isolation, independent pressure, mixed device/host window and SWA-only restore |

For each model, cover JAX and Raiden with both `write_through` and
`write_back`. Repeat Gemma's core restore/lifecycle and MiMo's DP isolation/
pressure with overlap enabled and disabled. Record the actual topology, cache
class/components, transfer backend, page/window, per-rank FULL/SWA device
capacities, and per-component host capacities after load. Stop if the selected
route or these model properties differ from the target.

The CLI has `--revision`, which is passed to both model and tokenizer loading
when `--tokenizer-path` is the same repository. It has no independent
`--tokenizer-revision` flag. Resolve and record each exact model/tokenizer
commit SHA before running; use that SHA as `MODEL_REVISION` below. These are
on-group examples. Calibrate the device KV budget and host ratio from actual
loaded capacity before freezing the workload.

```bash
# Gemma on-group, with MODEL_REVISION set to its pinned commit SHA.
python -m sgl_jax.launch_server \
  --model-path google/gemma-4-31B-it \
  --tokenizer-path google/gemma-4-31B-it --revision "$MODEL_REVISION" \
  --device tpu --tp-size 8 --dp-size 1 --ep-size 1 --page-size 128 \
  --enable-unified-radix-tree --hicache-storage none \
  --hicache-transfer-backend jax --hicache-write-policy write_through

# MiMo-V2.5 on-group (not MiMo-V2.5-Pro).
python -m sgl_jax.launch_server \
  --model-path XiaomiMiMo/MiMo-V2.5 \
  --tokenizer-path XiaomiMiMo/MiMo-V2.5 --revision "$MODEL_REVISION" \
  --device tpu --tp-size 8 --dp-size 2 --ep-size 8 --page-size 256 \
  --swa-full-tokens-ratio 0.25 \
  --enable-unified-radix-tree --hicache-storage none \
  --hicache-transfer-backend jax --hicache-write-policy write_through
```

For on-group combinations, replace `jax` with `raiden` and/or
`write_through` with `write_back`. The off group keeps model, revision,
topology, page size, Unified route, device KV budget and overlap setting, but
uses `--hicache-storage disable` and omits backend/write-policy options.
Overlap is enabled by default; append `--disable-overlap-schedule` for the
disabled half of required coverage. The model runner derives attention TP as
`tp_size / dp_size`, giving 8 and 4 respectively. Confirm both models load and
inspect actual pool capacities before pressure calibration.

The repository's `/generate` request accepts `rid`, `input_ids`, `dp_rank`,
`stream: false`, and `sampling_params`; its response has full `output_ids` and
`meta_info.id`, `dp_rank`, `completion_tokens`, and `finish_reason`. For example,
submit one frozen request with the existing HTTP client or curl:

```json
{"rid":"reuse-r0","input_ids":[101,102],"dp_rank":0,"stream":false,"sampling_params":{"temperature":0,"max_new_tokens":32,"sampling_seed":3}}
```

Replace the example IDs with token IDs produced by the **pinned tokenizer**;
never send those two example IDs as an acceptance prompt. Build boundaries from
tokenized lengths, including both sides of 1024/128 for Gemma and 128/256 for
MiMo. Keep the raw request and unmodified response. Check `/get_model_info` and
startup diagnostics, then run a single-request prefill/decode smoke before
pressure. The server may reject a target configuration while implementation is
in progress; record that as blocked, not passed.

## Capacity-calibrated cases and evidence

Before collecting results, measure capacity and calibrate pressure so a
controlled prefix is backed up, evicted from device, its old slot overwritten,
then restored from host. Freeze a JSON manifest with every RID, `kind`
(`normal`, `retract`, `abort`), target `dp_rank`, `max_new_tokens`, pinned
`input_ids`, `l2_components` (`FULL`, `SWA`, or both), and for normal/retract
an `expected_finish_reason` (`stop` or `length`). Record the expected
page-aligned reusable prefix, missing FULL/SWA pages, expected restore pages,
pressure request sequence, and source diagnostics alongside each case. For
SWA, count actual nodes covering the window; this may exceed one window and
must not be assumed equal to the FULL prefix length. A run that does not
trigger the specified path is inconclusive.

For each controlled L2 case, capture source-referenced events for **each**
restored component/rank: `d2h`, `device_evict`, `slot_overwrite`, and `h2d`.
The source should identify a trace/log record or diagnostic snapshot and its
page/node IDs. Verify layer/page contents and FULL→SWA mapping in the separate
real-transfer probe: known KV → D2H → device eviction and overwrite → H2D →
content/mapping comparison. Do this for both models and both backends, covering
double-component and SWA-only restore; MiMo uses distinct rank 0/1 data.

The transfer entrypoint is `test/srt/hybrid_hicache_transfer_probe.py`:
call `run_transfer_probe(scheduler, destructive_opt_in=True,
report_path="transfer.json", page_count=1)` synchronously on the scheduler
thread of a dedicated freshly loaded TPU process, before serving requests.
Put `test/srt` on `PYTHONPATH` for the test hook. It checks the actual loaded
layer map and pool shapes, requires an idle empty cache and allocators, and
refuses CPU execution. It writes known values, proves eviction and overwrite,
compares every local layer/page after dual and SWA-only restoration on every
rank, and checks mapping, rank isolation, native buffer addresses, and pristine
coordinated flush. The probe deliberately mutates and flushes cache state;
it must not be injected into a serving instance. Run fresh processes for each
model, backend, and write policy. Its JSON report is transfer evidence only;
serving, lifecycle, and performance checks below are separate.
 A
`cached_tokens` field alone proves neither L2 transfer nor restored content.

Run the fixed stage sequence “device hit → backup → device eviction → L2
restore → host eviction and recompute”, plus SWA-only healing, on both first
and repeat passes. For host-capacity or window-gap cases, assert the documented
shorter prefix/recompute outcome separately. At idle and after each stage,
record per-component/per-rank free, tree-owned, request-owned, in-flight, and
quarantined pages, host handles, locks, mappings, eviction and transfer counts.
Under DP2, explicitly target both ranks, confirm returned `meta_info.dp_rank`,
and show the other rank's first request has zero cache hit before it builds its
own prefix. Pressure one rank while the other remains idle; the idle rank's
FULL/SWA/host capacity and data must remain stable.

Capture actual retract → reschedule → completion evidence and compare its
final full output like a normal request. Capture abort terminal cancellation
and cleanup separately. Distinguish tree `reset()` from coordinated service
`flush_cache()`: only a successful idle, healthy flush after transfers finish
can establish restoration of initial FULL/SWA/host free capacity and cleared
mapping/cache counters. Quarantined resources cannot be declared free.

## Offline capture and check

Each run file is JSON with `manifest_sha256` (SHA-256 of the exact frozen
manifest bytes) and `results`, one result per RID. A result is the unmodified
`/generate` response plus an `events` array gathered from the source evidence:

```json
{"manifest_sha256":"<sha256>","results":[{"output_ids":[11,12],"meta_info":{"id":"reuse-r0","dp_rank":0,"completion_tokens":2,"finish_reason":{"type":"length","length":2}},"events":[{"type":"d2h","component":"FULL","dp_rank":0,"source":"trace:line-123 node=7 page=9"}]}]}
```

The example event is illustrative; capture all required events with real
sources. Preserve HTTP errors under `error` rather than silently dropping them.
The checker requires every manifest RID exactly once, verifies complete IDs
against `completion_tokens`, normal terminal reason/length, actual rank, zero
unexpected errors, retract resume, abort cancellation, and L2 event evidence
in both on runs. It compares off/off-repeat, on/on-repeat, and off/on by full
token-ID sequence and terminal reason; abort output is deliberately excluded.
`stop` may be shorter than `max_new_tokens`; `length` must equal it. Normal and
retract output must be nonempty and match the frozen terminal type. Abort
requires both cancellation and cleanup evidence.

For normal cases, the narrow capture tool sends frozen `input_ids`/RID/rank
in manifest order to `/generate` without streaming. It neither forces device
pressure nor sends retract or abort commands. Keep the server and cache state
at the manually specified stage. It writes raw responses after every request,
so they survive if evidence assembly is not yet ready:

```bash
.venv/bin/python test/srt/hybrid_hicache_capture.py \
  --manifest manifest.json --base-url http://127.0.0.1:30000 \
  --raw-output on-raw.json --evidence on-evidence.json --output on.json
```

The operator-supplied evidence JSON maps each captured RID to an event list,
for example `{"reuse-r0":[{"type":"h2d","component":"SWA",
"dp_rank":0,"source":"trace:..."}]}`. Every source must identify this same
run. If the evidence file is not ready when requests end, `on-raw.json`
remains; assemble it offline after gathering real diagnostics:

```bash
.venv/bin/python test/srt/hybrid_hicache_capture.py \
  --manifest manifest.json --from-raw on-raw.json \
  --evidence on-evidence.json --output on.json
```

Manually append the retract and abort responses with source-referenced
events, preserving `manifest_sha256`. The checker rejects missing RIDs. Do not
copy evidence from another run.

```bash
.venv/bin/python test/srt/hybrid_hicache_acceptance.py \
  --manifest manifest.json --off off.json --off-repeat off-repeat.json \
  --on on.json --on-repeat on-repeat.json
```

The checker confirms **record consistency**, not authenticity of attached
sources or resource ownership. A reviewer must inspect referenced traces,
physical KV comparisons, and capacity snapshots. If the current build lacks
diagnostics for an event or cannot force the target pressure state safely,
leave that case blocked and add the narrow observability point in the core
implementation; do not invent an event to make the checker pass.

## Performance protocol

Use the existing serving benchmark after correctness acceptance. For each
model's representative configuration and each of Random (no shared prefix),
GSP resident in L1, and GSP beyond L1 but fitting L2, run **three independent
full runs** for off and three for on. Keep every run in statistics. Start a new
server each time, compile and warm the same shape set, exclude load/compile/
warmup, then rebuild the workload's specified cache initial state before
measurement. Keep hardware, revisions, device KV budget, concurrency, inputs,
outputs, and sequence fixed. Share an off baseline only under the same exact
equivalence conditions above.

Report all three raw values and arithmetic mean ± sample standard deviation
for output throughput and total throughput, plus TTFT p50/p95/p99. Compute
throughput ratio as `mean(on throughput) / mean(off throughput)` and relative
change as `100 × (ratio − 1)%`; compute TTFT ratio as
`on percentile / off percentile` for each percentile. Include L1 hits, actual
L2 restored pages/tokens, recomputed tokens, and host occupancy separately.
The Random throughput soft target is ratio ≥ 0.95; missing it calls for
analysis, not a correctness failure. Do not require a fixed L2 speedup.

Record CPU, real TPU transfer, serving, CI, and performance outcomes as
distinct evidence. No TPU runs or model downloads are implied by this page.
