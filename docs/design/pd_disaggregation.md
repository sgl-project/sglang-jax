# Prefill-Decode Disaggregation Design Document for SGLang-JAX

## 1. Motivation

Prefill and decode have opposite resource profiles. Prefill is compute-bound
and bursty (it builds the full KV cache for a prompt in one forward), while
decode is memory-bandwidth-bound and steady (one token at a time). Co-locating
both on one engine forces a single batching/scheduling policy to serve two
opposite workloads, and a long prefill stalls the decode of every other request
sharing the engine.

PD (Prefill-Decode) disaggregation splits the two phases across separate
engines: a prefill engine produces the KV cache for a request, transfers it to
a decode engine, and the decode engine generates the output tokens. Each side
can then be sized and scheduled independently, matching the multi-host TP layout
used in production.

## 2. Architecture

The data flow for a single request:

```
client → router (mini_lb) → prefill engine → KV transfer → decode engine → client
```

Components:

- **Bootstrap server** (`bootstrap.py`): an HTTP registry. Prefill processes
  register their `host:transfer_port`; the decode side resolves a prefill peer
  per room from a locally cached layout. A heartbeat daemon keeps the registry
  live and evicts dead prefills on a bounded refresh interval.
- **JAX transfer wrapper + connection** (`jax_transfer/wrapper.py`,
  `jax_transfer/conn.py`): a process-level singleton over
  `jax.experimental.transfer`. Prefill registers KV buffers for a remote pull
  keyed by a transfer id; decode pulls them. A pool of background pull workers
  runs the blocking native pull off the decode event loop.
- **Host KV pool** (`mem_cache/host_kv_pool.py`): an optional bounded
  pinned-host staging pool (`QueueHostKVPool`) used by transfer path A.
- **ZMQ side channel** (`common/zmq_notifier.py`): an out-of-band pull-done
  signal from decode back to prefill, so prefill can release its buffers.
- **Scheduler mixins** (`prefill.py`, `decode.py`): wire the produce/consume
  steps into the single-controller scheduler.
- **Router** (`mini_lb.py`): a single-entry load balancer that fans a request
  out to a prefill and a decode endpoint and assigns the shared transfer id.

### Transfer paths

Two transfer paths differ only on the prefill side; both are transparent to
decode.

- **Path A — staged D2H** (`--disaggregation-enable-d2h`): prefill HBM → prefill
  pinned host pool → cross-pod pull → decode HBM. The device KV slot is freed as
  soon as the host copy is registered, relieving prefill HBM pressure; the
  bounded host pool also provides admission backpressure. Uses
  `memory_kind="pinned_host"` (unpinned host triggers a bf16 stride bug in
  `jax.experimental.transfer` that corrupts the KV).
- **Path B — direct HBM** (default): prefill registers the gathered device KV
  arrays directly for pull (device → device); they stay alive until the decode
  ack. Slightly faster than path A (no host hop).

### Single-host vs multi-host

Single-host runs one prefill and one decode process, each owning the full KV
cache. Multi-host is TP-style and SPMD-symmetric: every prefill process runs the
same Python on the same request set, pairing is by `jax_process_index`, and each
decode process pulls its `1/nproc` head shard. Multi-host + path A is rejected at
startup because the host pool is built on the global KV mesh while multi-host
prefill extracts a local-mesh shard.

## 3. Testing

Verified on GKE (cluster `ainfer-tpu-bench`, node pool `pd-v6e-1`),
DeepSeek-R1-Distill-Qwen-1.5B, TP=1, single-host:

- **GSM8K** (200 questions, 5-shot, parallel 16): path A **0.655**, path B
  **0.675**.
- **Smoke** (input 4096 / output 1024, concurrency 64): both **64/64**
  completed, **0 OOM**.
- **Baseline** (admission cap 8, concurrency 48, input 4096 / output 32, 192
  prompts): **192/192**, **0 OOM**.

Multi-host + path B was validated separately (#1366).

The single-host eval/benchmark job is reproducible via
`scripts/disaggregation/gke/deploy.sh`.

## 4. Feature support

| Feature | Supported |
|---|---|
| Single-host, path A (staged D2H) | ✅ |
| Single-host, path B (direct HBM) | ✅ |
| Multi-host, path B | ✅ |
| Multi-host, path A | ❌ (fail-fast at startup) |
| Single-entry router (mini_lb) | ✅ |
| Bootstrap registry + heartbeat | ✅ |
| Capacity-gated decode admission | ✅ |
| Prometheus metrics | ✅ |
| Shared-secret auth (bootstrap / side channel) | ✅ |
| Chunked prefill under PD | ✅ (requires ChunkCache, i.e. `--disable-radix-cache`) |
| `dp_size > 1` | ❌ |
| Prefill radix / prefix cache reuse | ❌ (chunked prefill needs ChunkCache; RadixCache continuation is unimplemented) |
| DP attention | ❌ |
| Overlap / chunked transfer | ❌ |
| SWA / MTP / EPD | ❌ |
