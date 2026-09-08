# Single-pod Raiden PD validation

This development harness runs Qwen3-30B-A3B on one Falcon v7x-8 worker,
with chips 0,1 for P and 2,3 for D (TP4/DP1/EP4 per role). It mounts an
existing model and a SHA256-pinned Raiden wheel read-only. The wheel uses
JAX/JAXLIB 0.11.1 and libtpu 0.0.46.1; do not substitute an arbitrary wheel.
Cluster, image and mounts are explicit in `render_single_pod.py`.

Render and submit from the repository root after staging new source/test
files so that `git diff BASE` includes them:

```sh
python scripts/disaggregation/falcon/render_single_pod.py \
  --name UNIQUE_EXPERIMENT_NAME --output /tmp/pd-validation.yaml
falcon workflow exp submit -f /tmp/pd-validation.yaml --output json
```

The manifest contains the exact base revision, source patch and runner hashes.
The session is bounded to six hours and starts `correctness-01` automatically.
A running Falcon session is not evidence that its child checks passed.

Use `falcon exp cp` to inspect `/tmp/tpu_logs/pd-session/status.json` and
`/tmp/tpu_logs/pd-results/<id>/summary.json`. Use `falcon exp logs` for progress.
Each child records `source.patch`, `runner-sha256.json`, `command.json` and
`session-result.json`; results are copied into the Falcon artifact after each
child. After a runtime change, rerun full correctness rather than inheriting
results from the previous source patch.

Queue subsequent checks serially as JSON files under
`/tmp/tpu_logs/pd-session/commands/`. Upload outside that directory first,
then atomically move the complete file into place using `falcon exp exec`.
For example, after `correctness-01`:

```json
{
  "id": "stream-01",
  "command": ["python3", "/workspace/sglang-jax/scripts/disaggregation/falcon/stream_regression.py"],
  "env": {"PD_REQUIRE_SUMMARY": "/tmp/tpu_logs/pd-results/correctness-01/summary.json"},
  "timeout_s": 1800
}
```

Run `fault_suite.py` with a distinct id and the same correctness prerequisite.
Run `performance_suite.py` last, also setting `PD_REQUIRE_FAULT_SUMMARY` to the
passed `fault-summary.json` and `PD_REQUIRE_STREAM_SUMMARY` to the passed
`stream-summary.json`. Run `eos_checks.py` as well and supply its passed
`eos-summary.json` through `PD_REQUIRE_EOS_SUMMARY`. Suggested child bounds are 1800 seconds for fault
checks and 7200 seconds for performance; the session deadline remains binding.

Checks:

- Native BF16 page integrity and reuse, non-PD token reference, B0/B1/P/D/PD
  and PD without chunk transfer; page-boundary prompts, long output, SSE,
  concurrency 4/16, exact token/logprob counts and original allocator capacity.
- Mixed distinct-output prompts through B1 and PD, two rounds of concurrency 16
  with original-capacity recovery (`mixed_requests.py`).
- Pause, retract, real D queue cancellation and sentinel request reuse.
- Long 4K/1024-token logprob SSE within 90 seconds; natural EOS on short
  and multi-chunk prompts, compared with non-PD tokens and finish reason.
- Window-one backpressure, cancellation while a completed native transfer's
  notification is held, D process loss, P resource recovery and restart.
  The hook delays completion visibility by two seconds, not physical DMA.
- Performance without logprobs: three alternating B1/PD repetitions, P-only
  and D-only ablations, raw TTFT/normalized inter-token latency/throughput,
  and separate P/D profiles. Warmups and profile traffic are excluded from
  measured request samples.

The fault hook is injected only into fault-check servers. Normal correctness
and performance runs must not inherit its `PYTHONPATH` or `PD_TEST_*` settings.
Native multi-NUMA failure recovery and cross-host networking are outside scope.

When validation is complete, atomically write `{"exit_code": 0}` to
`/tmp/tpu_logs/pd-session/finish.json` (nonzero for unresolved failures), verify
Falcon's terminal state, then analyze each profile through Falcon's
`xprof-summary` plugin with a separate `params.subpath`. Profile capture alone
does not establish physical Raiden compute/transfer overlap. Stop any temporary
monitor after collecting and reporting the results.

## Continuous steady-state PR gates

Pass `--pr-validation` to the renderer to run correctness followed by stream,
fault, EOS, mixed-output, continuous performance, soak and profiling checks.
The supervisor fails fast on any failed gate and exits automatically after all
stages finish; the six-hour deadline still applies. It does not create a PR.

`performance_suite.py` is the original warmed **finite-wave** benchmark. Use
`steady_suite.py` for steady-state claims: independent client slots replenish
immediately in one persistent session, with 60 seconds of uninterrupted warm
traffic and a fixed 180-second measurement window. Count observed token deltas
inside the window, including partial requests; latency includes the full drain
of the start-in-window cohort. Workloads and per-slot payload sequences are
identical across configurations, although completion timing naturally differs.

Three paired B1/PD repetitions cover 4K/16K input, 256 output, C16/C32 and mixed
4K/16K plus 256/1024 at C16. P-only/D-only use the mixed case. A separate C32
20-minute soak checks every output against non-PD references and requires full
KV/queue recovery. Sample server states at one-second intervals to distinguish
active decode with incoming PD work from client concurrency alone. Profiles are
captured separately for five seconds during continuous mixed traffic; profile
and export delays are never included in reported performance measurements.

The PR sequence also runs a final 4K/256 C32 comparison of B1, D-only,
P-only and PD (`steady_ablation.py`). These contemporaneous controls help
separate queueing changes from a role-specific regression when throughput
improves but steady TTFT increases. They use the same measurement windows and
require the main steady/soak suite to have passed.

The follow-up also compares B1 and PD at matched deterministic arrival rates:
5 requests/s for 4K/256 and 1.2 requests/s for 16K/256, below the observed B1
capacity. `steady_rate_client.py` schedules arrivals independently of responses,
checks actual cohort RPS and arrival lateness, and fails if its in-flight cap
would silently throttle the load. It reports latency for the start-in-window
cohort, including drain; it does not present completion-count throughput as
fixed-window token throughput. The already tested SSE request parser is reused.
