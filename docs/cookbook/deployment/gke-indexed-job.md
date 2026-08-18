---
title: "GKE Indexed Job"
---

# GKE Indexed Job

Use this deployment pattern for multi-host TPU slices on Google Kubernetes Engine. Recipes provide the tested node count, topology, tensor parallelism, data parallelism, expert parallelism, and model-specific launch flags.

Common conventions:

- Use an Indexed Job plus a headless Service so every rank has a stable DNS name.
- Set `--nnodes`, `--node-rank`, and `--dist-init-addr` consistently across pods.
- Let GKE provide TPU worker environment variables such as `TPU_PROCESS_ADDRESSES` and `TPU_WORKER_HOSTNAMES`.
- Keep `JAX_COMPILATION_CACHE_DIR` set for every rank.

For topology sizing, see [TPU topology reference](/base/tpu-topology-reference). For multi-node startup failures, see [Troubleshooting](/deployment/troubleshooting).

The full canonical template lives in the main Sphinx docs at `docs/deployment/gke-indexed-job.md`.
