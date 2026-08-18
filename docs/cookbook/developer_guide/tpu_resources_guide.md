---
title: "TPU Resources Guide"
---

# TPU Resources Guide

Use this page as the cookbook-local pointer for TPU provisioning notes referenced by topology and SkyPilot guidance.

For cookbook sizing math, start with [TPU topology reference](/base/tpu-topology-reference). Recipes list only topologies that have been validated or partially validated.

When provisioning TPU resources:

- Confirm quota and region availability before choosing a topology.
- Keep v6e and v7x device-count differences in mind: v7x exposes two JAX devices per chip.
- Match `--tp-size`, `--dp-size`, `--ep-size`, `--nnodes`, and `--node-rank` to the selected slice.
- Use the recipe's tested deployment path when available instead of deriving a topology from scratch.

The full canonical resource guide lives in the main Sphinx docs at `docs/developer_guide/tpu_resources_guide.md`.
