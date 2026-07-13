---
title: "Serve Tuning Playbook"
---

# Serve Tuning Playbook

This playbook turns a model, TPU topology, and serving objective into a reviewable SGL-JAX launch and benchmark plan. It is written for both humans and coding agents: every stage has required inputs, commands, exit criteria, and artifacts.

Use this page for the workflow. Use [Launch Flag Reference](/base/launch-flags-reference) for flag definitions, [TPU Topology Reference](/base/tpu-topology-reference) for device math, [Basic API Usage](/base/basic-api-usage) for request examples, and the target model recipe for validated model-specific values.

## 0. Operating contract

An agent following this playbook must:

1. Pin the SGL-JAX revision and record it before generating commands.
2. Prefer the target model recipe over generic values in this page.
3. Verify flags against the current checkout's `--help`; never rely on a remembered default.
4. Resolve every `{{PLACEHOLDER}}` before executing a command.
5. Change one tuning dimension at a time during a comparison.
6. Keep cold-start/JIT time separate from steady-state serving metrics.
7. Preserve commands, resolved server arguments, logs, raw benchmark output, and failures.
8. Stop when a required check fails. Do not silently skip it and report success.
9. Obtain explicit authorization before creating or deleting cloud resources, exposing a public port, or terminating another user's job.

The source-of-truth order is:

```text
current source and --help
  > target model recipe
  > launch flag and topology references
  > this generic playbook
  > old experiment notes or remembered defaults
```

## 1. Input contract

Create a plan with this schema before launching anything:

```yaml
revision: "{{GIT_COMMIT}}"
model:
  path: "{{MODEL_PATH}}"
  revision: "{{MODEL_REVISION_OR_NULL}}"
  kind: "{{dense|moe|multimodal}}"
  trust_remote_code: "{{true|false}}"
hardware:
  accelerator: "{{TPU_SLICE}}"
  nodes: "{{NODE_COUNT}}"
  jax_devices_per_node: "{{DEVICES_PER_NODE}}"
  total_jax_devices: "{{TOTAL_DEVICES}}"
parallelism:
  tp_size: "{{TP_SIZE}}"
  dp_size: "{{DP_SIZE}}"
  ep_size: "{{EP_SIZE}}"
  sequence_parallel: "{{true|false}}"
workload:
  objective: "{{latency|throughput|balanced}}"
  context_length: "{{CONTEXT_LENGTH}}"
  input_length: "{{ISL}}"
  output_length: "{{OSL}}"
  concurrency: ["{{C1}}", "{{C2}}"]
runtime:
  dtype: "{{DTYPE}}"
  mem_fraction_static: "{{MEM_FRACTION}}"
  max_running_requests: "{{MAX_RUNNING_REQUESTS}}"
  max_prefill_tokens: "{{MAX_PREFILL_TOKENS}}"
  chunked_prefill_size: "{{CHUNKED_PREFILL_SIZE}}"
  attention_backend: "{{ATTENTION_BACKEND}}"
output_dir: "{{OUTPUT_DIR}}"
```

Do not infer missing model-specific values for a MoE, hybrid-attention, quantized, speculative, or multimodal model. Find a matching recipe or stop and request the missing choice.

## 2. Execution state machine

| Stage | Required action | Exit criterion | Required artifact |
|---|---|---|---|
| `PLAN` | Fill the input contract and choose the target recipe. | No unresolved required field. | `serve-plan.yaml` |
| `PREFLIGHT` | Check revision, CLI, devices, model config, ports, and existing processes. | Device and model constraints pass. | `preflight.txt` |
| `LAUNCH` | Render and run one launch command. | Server reports ready without fatal errors. | `launch-command.sh`, `server.log` |
| `SMOKE` | Query health/model endpoints and generate a short response. | All requests return a valid response. | `smoke.json` |
| `BENCHMARK` | Warm up and run a fixed workload. | All requests succeed and no measured request triggers an unexpected compile. | one JSONL file per run |
| `ANALYZE` | Compare only compatible runs and explain the bottleneck. | Metrics, assumptions, and failures are recorded. | `summary.md` |
| `DONE` | Audit artifacts and active resources. | Artifacts exist; resource state is explicitly reported. | final status block |

Do not jump directly from `PLAN` to `BENCHMARK`. A benchmark from an unverified server is invalid evidence.

## 3. Preflight: collect facts before choosing flags

### 3.1 Pin the checkout and inspect the CLI

```bash
git rev-parse HEAD
python -m sgl_jax.launch_server --help
python -m sgl_jax.bench_serving --help
```

Record the output commit. If a documented flag is absent from `--help`, stop using it for that checkout.

### 3.2 Inspect the visible JAX topology

Run this on every serving host:

```bash
python -c "import jax; print('process_count=', jax.process_count()); print('local_device_count=', jax.local_device_count()); print('device_count=', jax.device_count()); print('devices=', jax.devices())"
```

Checks:

- Single host: `local_device_count` must cover the selected devices.
- Multi-host: all hosts must agree on node count and expose the expected local devices.
- `total_jax_devices` in the plan must equal the devices intended for the service, not the marketing chip count. See [TPU Topology Reference](/base/tpu-topology-reference), especially the v7x two-devices-per-chip rule.

### 3.3 Inspect the model configuration

Read the matching recipe and the model's `config.json`. At minimum record:

- architecture and whether the model is dense or MoE;
- `num_attention_heads` and `num_key_value_heads` when present;
- expert count and MoE intermediate size for MoE models;
- native context length;
- quantization config;
- whether custom remote code is required.

Do not use a dense model to evaluate Expert Parallelism. Do not enable Sequence Parallelism merely because the flag exists; require a model path and workload for which the recipe or implementation supports it.

### 3.4 Check host state

Before launch, confirm:

- the target port is free;
- no old server owns the selected TPU devices;
- the JIT cache directory is writable;
- the model download/cache directory has enough space;
- multi-host rank 0 is reachable from every worker.

## 4. Build and validate the parallelism plan

For the standard scheduler path in the current implementation:

```text
tensor_axis = tp_size / dp_size
mesh        = [dp_size, tensor_axis]
```

Validate all applicable constraints:

```text
tp_size % nnodes == 0
tp_size % dp_size == 0
num_attention_heads % tensor_axis == 0
max_running_requests >= dp_size
```

The final `max_running_requests` is bounded by the configured limit, request/token pool, and attention backend, then rounded down to a multiple of `dp_size`. Always read the resolved startup log; the CLI value is not necessarily the final value.

Model-specific constraints from the recipe still apply. Examples include KV-head sharding, expert divisibility, fused-MoE tile constraints, hybrid-attention cache requirements, and pre-sharded checkpoints.

### 4.1 Starting rules

| Situation | Starting plan | Reason |
|---|---|---|
| Dense model, latency-first | `dp=1`, smallest TP that fits | Avoid unnecessary collectives. |
| Dense model, throughput-first | Start with full-model TP baseline, then test DP only at useful concurrency. | DP needs enough requests to keep replicas busy. |
| MoE model | Use the recipe's TP/DP/EP/backend combination. | Expert layout and all-to-all constraints are model-specific. |
| Long-context experiment | Keep model/topology fixed; vary context/chunk/SP separately. | Avoid confounding memory, compilation, and sharding changes. |
| Multi-host | Use all hosts' routable rank-0 address and unique ranks. | `0.0.0.0` is a bind address, not a peer rendezvous address. |

On multi-host runs, do not set `--device-indexes`; the current server normalizes it away when `nnodes > 1`.

## 5. Render the launch command

Before execution, assert that the rendered command contains no `{{` or `}}` tokens.

### 5.1 Single-host template

```bash
JAX_COMPILATION_CACHE_DIR="{{JIT_CACHE_DIR}}" \
python -u -m sgl_jax.launch_server \
  --model-path "{{MODEL_PATH}}" \
  --tp-size "{{TP_SIZE}}" \
  --data-parallel-size "{{DP_SIZE}}" \
  --ep-size "{{EP_SIZE}}" \
  --device tpu \
  --dtype "{{DTYPE}}" \
  --context-length "{{CONTEXT_LENGTH}}" \
  --mem-fraction-static "{{MEM_FRACTION}}" \
  --max-running-requests "{{MAX_RUNNING_REQUESTS}}" \
  --max-prefill-tokens "{{MAX_PREFILL_TOKENS}}" \
  --chunked-prefill-size "{{CHUNKED_PREFILL_SIZE}}" \
  --attention-backend "{{ATTENTION_BACKEND}}" \
  --host 127.0.0.1 \
  --port "{{PORT}}"
```

Add only the options justified by the plan or recipe, for example:

- `--trust-remote-code` for a trusted model that requires it;
- `--revision` to pin model weights;
- `--device-indexes ...` to select a single-host subset;
- `--enable-sequence-parallel` for an explicit SP experiment;
- the recipe's MoE, quantization, parser, or cache flags.

Binding to `127.0.0.1` is the safe default and works with an SSH tunnel. Use `0.0.0.0` only when a non-local client must connect and the deployment has the intended authentication and network policy.

### 5.2 Multi-host additions

Every host uses the same model and global parallelism values, but a unique node rank:

```bash
  --nnodes "{{NODE_COUNT}}" \
  --node-rank "{{THIS_NODE_RANK}}" \
  --dist-init-addr "{{RANK0_ROUTABLE_HOST}}:{{DIST_PORT}}" \
  --watchdog-timeout "{{WATCHDOG_SECONDS}}"
```

Current implementation note: `--dist-timeout` is accepted by the CLI but is not passed to `jax.distributed.initialize()` in the standard Scheduler path. Do not rely on it to change rendezvous behavior unless the current source shows that wiring.

### 5.3 Compilation controls

These controls are different:

```text
--skip-server-warmup       skips the HTTP dummy warmup
--disable-precompile       skips proactive model bucket compilation
JAX_COMPILATION_CACHE_DIR  selects the persistent JAX compilation cache
```

Do not claim a cache hit because the directory exists. Confirm it from startup timing and logs. Use separate cache directories when model, JAX/libtpu version, mesh, context, page size, or compilation buckets change.

## 6. Launch checkpoint

Capture the exact rendered command and the full server log. The log must show:

- the resolved `ServerArgs`;
- expected devices and Mesh shape;
- expected TP/DP/EP settings;
- final `max_running_requests` and token-pool capacity;
- compilation completion or an explicit decision to defer JIT;
- the HTTP server listening on the planned host and port.

Stop on any of these conditions:

- selected device count differs from the plan;
- a parallelism assertion fails;
- OOM occurs during load, precompile, or first prefill;
- the server silently changes a value that invalidates the experiment;
- another process owns the port or device;
- any multi-host worker exits or fails to join.

## 7. Smoke-test checkpoint

Use the endpoint paths from [Basic API Usage](/base/basic-api-usage). At minimum:

```bash
curl --fail --show-error "http://127.0.0.1:{{PORT}}/health"
curl --fail --show-error "http://127.0.0.1:{{PORT}}/v1/models"
curl --fail --show-error \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":8}}' \
  "http://127.0.0.1:{{PORT}}/generate"
```

Save the responses. Do not start performance testing until the model name and generated response are valid.

## 8. Benchmark protocol

### 8.1 Fixed-workload command

Render one output file per run; `bench_serving` writes JSONL and may append when a path is reused.

```bash
python -m sgl_jax.bench_serving \
  --backend sgl-jax \
  --base-url "http://127.0.0.1:{{PORT}}" \
  --dataset-name random \
  --model "{{MODEL_PATH}}" \
  --tokenizer "{{MODEL_PATH}}" \
  --num-prompts "{{NUM_PROMPTS}}" \
  --random-input-len "{{ISL}}" \
  --random-output-len "{{OSL}}" \
  --random-range-ratio 0 \
  --max-concurrency "{{CONCURRENCY}}" \
  --warmup-requests "{{WARMUP_REQUESTS}}" \
  --flush-cache \
  --tokenize-prompt \
  --seed "{{DATASET_SEED}}" \
  --output-details \
  --output-file "{{OUTPUT_DIR}}/{{RUN_ID}}.jsonl"
```

EOS policy must be explicit:

- omit `--disable-ignore-eos` for a fixed generated-token workload that ignores EOS;
- add `--disable-ignore-eos` for realistic generation that may stop early;
- never compare the two policies as if their output-token counts were equivalent.

### 8.2 Sweep rules

For a throughput-latency curve:

1. Fix revision, model revision, dtype, topology, TP/DP/EP/SP, context, ISL, OSL, scheduler, cache policy, and dataset seed.
2. Sweep concurrency from low to high.
3. Run at least three measured repetitions per point.
4. Use a unique output file per repetition.
5. Record compilation or restart events separately.
6. Mark failed/OOM points instead of dropping them.
7. Compare request throughput, output-token throughput, E2E latency, TTFT, TPOT/ITL, P99, success rate, and peak concurrency.

For a sharding comparison, keep the workload fixed and change only the TP/DP/EP/SP plan. Report both total throughput and per-device efficiency.

## 9. Deterministic diagnosis table

| Signal | Inspect first | Next controlled action |
|---|---|---|
| Load/precompile OOM | resolved HBM fraction, token pool, old processes | Stop old processes; then lower `max-running-requests` or chunk size. Change one value. |
| Prefill OOM | ISL, max/chunked prefill, context, temporary buffers | Lower `chunked-prefill-size`, then retry the same request. |
| High TTFT, normal TPOT | prefill size, queueing, chunking, cache hit | Compare one chunk size or concurrency change. |
| High TPOT | tensor-axis collectives, decode batch, attention backend | Compare TP shapes at the same workload; inspect profiler evidence before changing backend. |
| Throughput plateaus | achieved concurrency, final request limit, DP utilization | Raise concurrency only if the server limit and KV pool allow it. |
| DP is slower | requests per replica and tensor-axis change | Increase useful concurrency or reduce DP; report replica underfill. |
| First measured request compiles | precompile buckets and cache logs | Discard the run, add/adjust buckets or warmup, then rerun. |
| Multi-host initialization hangs | rank-0 route, rank uniqueness, node count, worker logs | Fix rendezvous inputs; do not mask the issue with an assumed `dist-timeout`. |
| Repeated workload gets faster | radix/prefix cache and `--flush-cache` result | Choose cold-cache or warm-cache semantics and keep it fixed. |
| Output tokens differ across runs | EOS policy, tokenizer, sampling, seed | Fix the policy and report server vs retokenized counts. |

Use a TPU profiler before labeling a run compute-bound, HBM-bound, or communication-bound. A throughput number alone is not Roofline evidence.

## 10. Artifact and report contract

Recommended layout:

```text
{{OUTPUT_DIR}}/
  serve-plan.yaml
  preflight.txt
  launch-command.sh
  server.log
  smoke.json
  benchmark/
    {{RUN_ID}}.jsonl
  summary.md
```

`summary.md` must include:

```markdown
# Serve experiment summary

## Status
PASS, FAIL, or PARTIAL — with the failed checkpoint if applicable.

## Revisions
SGL-JAX commit, model revision, JAX/JAXLIB/libtpu version.

## Hardware and topology
Accelerator, nodes, JAX devices, Mesh, TP/DP/EP/SP.

## Resolved server configuration
Values from startup logs, not only the requested CLI values.

## Workload
Dataset, request count, ISL, OSL, concurrency, seed, EOS/cache policy, repetitions.

## Results
Throughput, TTFT, TPOT/ITL, E2E/P99, failures, and per-device efficiency.

## Interpretation
Observed bottleneck, supporting evidence, and alternative explanations.

## Known limits
Untested model paths, shapes, backends, topologies, or profiler gaps.

## Artifacts
Paths to plan, command, logs, raw JSONL, plots, and profiler captures.
```

An agent's final response must name the checkpoint reached, list skipped checks, link every artifact, report active resource state, and avoid saying “complete” if a required artifact or validation is missing.

## 11. Worked plan: Qwen-7B-Chat on one v6e-4 host

The validated values and launch command live in the [Qwen-7B-Chat recipe](/autoregressive/Qwen/Qwen). A minimal plan derived from it is:

```yaml
revision: "{{PIN_CURRENT_COMMIT}}"
model:
  path: "Qwen/Qwen-7B-Chat"
  kind: dense
  trust_remote_code: true
hardware:
  accelerator: tpu-v6e-4
  nodes: 1
  jax_devices_per_node: 4
  total_jax_devices: 4
parallelism:
  tp_size: 4
  dp_size: 1
  ep_size: 1
  sequence_parallel: false
workload:
  objective: balanced
  input_length: 512
  output_length: 128
  concurrency: [1, 2, 4, 8]
runtime:
  dtype: bfloat16
  mem_fraction_static: 0.8
  max_prefill_tokens: 8192
  attention_backend: fa
```

This is a starting recipe, not proof that TP4 is optimal. To claim an optimum, run the same benchmark protocol for every legal comparison and preserve the raw evidence.

## 12. Scope boundaries

This playbook covers the standard autoregressive serving workflow. It does not invent generic configurations for LoRA, speculative decoding, PD disaggregation, HiCache, multimodal stages, quantization, or MoE kernels. Use the matching feature documentation and model recipe for those paths, then apply the same checkpoints and artifact contract.
