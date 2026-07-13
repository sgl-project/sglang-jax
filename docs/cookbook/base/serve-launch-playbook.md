---
title: "Serve Launch Playbook"
---

# Serve Launch Playbook

Use this playbook to turn a model and TPU topology into a smoke-tested SGL-JAX server. It is written for first-time operators and coding agents: each stage names the facts to collect, the command to run, the checks that must pass, and the conditions that should stop the launch.

This playbook ends when the server passes its smoke test. Throughput/latency benchmarking and accuracy evaluation are separate workflows maintained by the existing `model-speed-benchmark` and `model-accuracy-benchmark` skills.

Use [Launch Flag Reference](/base/launch-flags-reference) for flag definitions, [TPU Topology Reference](/base/tpu-topology-reference) for device math, [Basic API Usage](/base/basic-api-usage) for request examples, and the target model recipe for validated model-specific values.

## 1. Before you start

Use this source-of-truth order when values disagree:

```text
current source and launch_server --help
  > target model recipe
  > launch flag and topology references
  > this generic playbook
  > old experiment notes or remembered defaults
```

Before creating a process or cloud resource:

1. Pin and record the SGL-JAX revision.
2. Prefer the target model recipe over generic values on this page.
3. Verify every flag against the current checkout's `launch_server --help`.
4. Resolve every `{{PLACEHOLDER}}` in a command before executing it.
5. Obtain authorization before creating or deleting cloud resources, exposing a public port, or terminating another user's process.
6. Stop on a failed required check; do not silently skip it and report a successful launch.

The launch has four stages:

| Stage | Required action | Exit criterion |
|---|---|---|
| `PLAN` | Record the model, hardware, parallelism, runtime, and network values. | No unresolved required value. |
| `PREFLIGHT` | Check the revision, CLI, devices, model config, ports, and old processes. | Device and model constraints pass. |
| `LAUNCH` | Render and run one launch command. | Every worker stays up and the server reports ready. |
| `SMOKE` | Check liveness, model discovery, and one short generation. | All three checks return valid responses. |

## 2. Record the launch inputs

Fill this launch plan before generating a command:

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
runtime:
  dtype: "{{DTYPE}}"
  context_length: "{{CONTEXT_LENGTH}}"
  mem_fraction_static: "{{MEM_FRACTION}}"
  max_running_requests: "{{MAX_RUNNING_REQUESTS}}"
  max_prefill_tokens: "{{MAX_PREFILL_TOKENS}}"
  chunked_prefill_size: "{{CHUNKED_PREFILL_SIZE}}"
  attention_backend: "{{ATTENTION_BACKEND}}"
network:
  host: "{{BIND_HOST}}"
  port: "{{HTTP_PORT}}"
  dist_port: "{{DIST_PORT_OR_NULL}}"
```

Do not invent generic values for a MoE, hybrid-attention, quantized, speculative, or multimodal model. Find a matching recipe or ask for the missing model-specific choice.

## 3. Run preflight checks

### 3.1 Pin the checkout and inspect the CLI

Run these commands in the environment that will host the server:

```bash
git rev-parse HEAD
python -m sgl_jax.launch_server --help
```

Record the commit. If a documented flag is absent from `--help`, do not use it for that checkout.

### 3.2 Inspect the visible JAX topology

Run this on every serving host:

```bash
python -c "import jax; print('process_index=', jax.process_index()); print('process_count=', jax.process_count()); print('local_device_count=', jax.local_device_count()); print('device_count=', jax.device_count()); print('devices=', jax.devices())"
```

Check that:

- a single host exposes every device selected by `--device-indexes`;
- every multi-host worker sees the expected local device count;
- the expected ranks are unique and cover `0..nnodes-1`;
- `total_jax_devices` is the number of JAX devices intended for the service, not a marketing chip count.

See [TPU Topology Reference](/base/tpu-topology-reference), especially the v7x two-devices-per-chip rule.

### 3.3 Inspect the model configuration

Read the matching recipe and the model's `config.json`. Record at least:

- architecture and whether the model is dense or MoE;
- `num_attention_heads` and `num_key_value_heads`, when present;
- expert count and MoE intermediate size for a MoE model;
- native context length;
- quantization config;
- whether trusted custom model code is required.

Do not use a dense model to validate Expert Parallelism. Do not enable Sequence Parallelism only because the flag exists; require a supported model path and workload from the recipe or implementation.

### 3.4 Check host state

Before launch, confirm:

- the HTTP and distributed rendezvous ports are free;
- no old server owns the selected TPU devices;
- the JIT cache directory is writable;
- the model cache has enough disk space;
- every worker can reach the rank-0 address on the distributed port.

## 4. Validate the parallelism plan

For the standard Scheduler path:

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

The final `max_running_requests` is bounded by the configured limit, request/token pool, and attention backend, then rounded down to a multiple of `dp_size`. The resolved startup log, not only the CLI input, determines the value that the server actually uses.

Model recipes may add KV-head sharding, expert divisibility, fused-MoE tile, hybrid-attention cache, or pre-sharded checkpoint constraints.

### 4.1 Safe starting points

| Situation | Starting plan | Reason |
|---|---|---|
| Dense model, latency-first | `dp=1`, smallest TP that fits | Avoid unnecessary collectives. |
| Dense model, throughput-first | Start with a full-model TP baseline. | DP needs enough requests to keep replicas busy. |
| MoE model | Use the recipe's TP/DP/EP/backend combination. | Expert layout and all-to-all constraints are model-specific. |
| Long context | Keep the topology fixed while selecting context/chunk/SP values. | Avoid mixing memory and sharding changes during bring-up. |
| Multi-host | Use a routable rank-0 address and unique ranks. | `0.0.0.0` is a bind address, not a peer rendezvous address. |

On multi-host runs, do not set `--device-indexes`; the current server clears it when `nnodes > 1` and uses the distributed mesh.

## 5. Render the launch command

Assert that the rendered command contains no `{{` or `}}` tokens before running it.

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
  --host "{{BIND_HOST}}" \
  --port "{{HTTP_PORT}}"
```

Add only options justified by the plan or recipe, for example:

- `--trust-remote-code` for a trusted model that requires it;
- `--revision` to pin model weights;
- `--device-indexes ...` to select a single-host subset;
- `--enable-sequence-parallel` for an explicit supported SP launch;
- the recipe's MoE, quantization, parser, or cache flags.

Binding to `127.0.0.1` is the safe default and works with an SSH tunnel. Use `0.0.0.0` only when a non-local client must connect and the deployment has the intended authentication and network policy.

### 5.2 Multi-host additions

Every worker uses the same model and global parallelism values, but a unique node rank:

```bash
  --nnodes "{{NODE_COUNT}}" \
  --node-rank "{{THIS_NODE_RANK}}" \
  --dist-init-addr "{{RANK0_ROUTABLE_HOST}}:{{DIST_PORT}}" \
  --watchdog-timeout "{{WATCHDOG_SECONDS}}"
```

The rank-0 host must be routable from every worker. Do not use `0.0.0.0` or an external address that workers cannot reach.

Current implementation note: `--dist-timeout` is accepted by the CLI but is not passed to `jax.distributed.initialize()` in the standard Scheduler path. Do not rely on it to change rendezvous behavior unless the current source shows that wiring.

### 5.3 Compilation controls

These controls are different:

```text
--skip-server-warmup       skips the HTTP dummy warmup
--disable-precompile       skips proactive model bucket compilation
JAX_COMPILATION_CACHE_DIR  selects the persistent JAX compilation cache
```

Do not claim a cache hit only because the directory exists. Confirm it from startup timing and logs. Use separate cache directories when the model, JAX/libtpu version, mesh, context, page size, or compilation buckets change.

## 6. Verify the launch

Capture the exact command and full server log. Confirm that the log shows:

- the resolved `ServerArgs`;
- the expected devices and Mesh shape;
- the expected TP/DP/EP values;
- the final `max_running_requests` and token-pool capacity;
- compilation completion or an explicit decision to defer JIT;
- the HTTP server listening on the planned host and port.

Stop the launch if:

- the selected device count differs from the plan;
- a parallelism assertion fails;
- an OOM occurs during load, precompile, or the first prefill;
- the server changes a value that invalidates the plan;
- another process owns the port or TPU device;
- any multi-host worker exits or fails to join.

### 6.1 Launch diagnostics

| Symptom | Inspect first | Next action |
|---|---|---|
| Port already in use | Listener PID and old `launch_server` processes | Stop only the process you are authorized to terminate, or choose another port. |
| Model load fails | Model path/revision, auth, disk, remote-code requirement | Correct the input; do not hide it with unrelated runtime flags. |
| Load/precompile OOM | Resolved HBM fraction, token pool, old processes | Remove stale processes, then change one memory-related value. |
| Mesh assertion fails | Visible devices, TP/DP/EP, attention/expert divisibility | Fix the plan before relaunching. |
| Multi-host initialization hangs | Rank-0 route, unique ranks, node count, worker logs | Fix the rendezvous inputs; do not mask the issue with `--dist-timeout`. |
| One worker exits | The first failing worker log | Treat the whole launch as failed; all workers are required. |
| First request still compiles | Precompile buckets, warmup choice, JIT cache logs | Finish or adjust compilation before handing off to a downstream test. |

## 7. Run the smoke test

Use the endpoint paths from [Basic API Usage](/base/basic-api-usage).

First check liveness by status code. `/health` may return an empty body:

```bash
curl -sS -o /dev/null -w '%{http_code}\n' \
  "http://127.0.0.1:{{HTTP_PORT}}/health"
```

The expected status is `200`. Next discover the served model:

```bash
curl --fail --show-error \
  "http://127.0.0.1:{{HTTP_PORT}}/v1/models"
```

Finally run a short generation:

```bash
curl --fail --show-error \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":8}}' \
  "http://127.0.0.1:{{HTTP_PORT}}/generate"
```

The smoke test passes only if:

1. `/health` returns HTTP 200;
2. `/v1/models` returns the expected model id;
3. `/generate` returns a valid generated response;
4. no worker exits or reports a new fatal error during the requests.

Do not start a downstream test from an unverified server.

## 8. Hand off after smoke test

This playbook intentionally does not define benchmark commands, concurrency sweeps, accuracy datasets, result files, retry rules, or report formats.

After the smoke test passes:

- for throughput or latency, use [`model-speed-benchmark`](https://github.com/sgl-project/sglang-jax/tree/main/.claude/skills/model-speed-benchmark);
- for model quality or dataset scores, use [`model-accuracy-benchmark`](https://github.com/sgl-project/sglang-jax/tree/main/.claude/skills/model-accuracy-benchmark).

Pass the downstream workflow these launch facts:

```text
target resource and endpoint
served model id
server PID or experiment id
SGL-JAX commit and runtime versions
TPU type, topology, and device count
full resolved server launch command
radix-cache state
```

The downstream skill owns its benchmark/evaluation procedure and report contract. Add new performance or evaluation rules there rather than creating a second standard in this playbook.

## 9. Scope boundaries

This playbook covers the standard autoregressive serving launch path. It does not invent generic configurations for LoRA, speculative decoding, PD disaggregation, HiCache, multimodal stages, quantization, or model-specific MoE kernels. Use the matching feature documentation and model recipe for those paths, then return to the launch and smoke-test checks that apply.
