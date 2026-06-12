---
title: "Deployment Templates"
---

# Deployment Templates

Launcher / orchestrator templates referenced by cookbook recipes. Model cookbooks use GKE as the primary path for multi-host serving; SkyPilot remains an advanced v6e experiment path for users who already operate SkyPilot.

| Template | When to use |
|---|---|
| [`single-host-docker.md`](single-host-docker.md) | One TPU host (v6e-4, v6e-8, v7x-8). Lowest-friction starting point. |
| [`gke-indexed-job.md`](gke-indexed-job.md) | Multi-host TPU slice on GKE. Stable pod DNS via Indexed Job + headless Service. |
| [`skypilot.md`](skypilot.md) | Advanced multi-host v6e experiments via SkyPilot. v6e-only today (template hardcodes `v2-alpha-tpuv6e`). |

## Common conventions across all three

- **Image**: `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` (matches the SGL-JAX `[tpu]` extras pin).
- **Install**: `pip install -e "python[tpu]"` after `git clone https://github.com/sgl-project/sglang-jax.git`.
- **Compilation cache**: every launch sets `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` (or any persistent path). Without this, the first request blocks ~4 min while XLA/Pallas re-compiles.
- **Entrypoint**: `python -m sgl_jax.launch_server` (not `sglang serve` — that is the upstream SGLang command).
- **Multi-node coordination**: all three rely on `--nnodes`, `--node-rank`, `--dist-init-addr=<rank0>:<port>` plus the `TPU_PROCESS_ADDRESSES` / `TPU_WORKER_HOSTNAMES` env vars (set automatically by GKE/SkyPilot; manual for raw Docker).

## Picking flags

For TP/EP/page-size/SWA-ratio choices, see the per-recipe Key Flags table. For the full flag inventory: [Launch flags reference](../base/launch-flags-reference.md). For TPU generation specs: [TPU topology reference](../base/tpu-topology-reference.md).
