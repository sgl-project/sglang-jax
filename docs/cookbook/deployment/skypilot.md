---
title: "SkyPilot"
---

# SkyPilot

Use this deployment pattern for advanced multi-host v6e experiments via SkyPilot. Recipes that mention SkyPilot still keep the model-specific command and benchmark evidence in the recipe itself.

Common conventions:

- SkyPilot TPU resources use `tpu-v6e-N` names for v6e slices.
- The current template targets the TPU v6e runtime family.
- Keep distributed flags aligned with the recipe: `--nnodes`, `--node-rank`, `--dist-init-addr`, `--tp-size`, `--dp-size`, and `--ep-size`.
- Use a persistent compilation cache when running repeated experiments.

For TPU sizing, see [TPU topology reference](/base/tpu-topology-reference). For provisioning and quota notes, see [TPU resources guide](/developer_guide/tpu_resources_guide).

The full canonical template lives in the main Sphinx docs at `docs/deployment/skypilot.md`.
