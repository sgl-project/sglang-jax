---
title: "Deployment References"
---

# Deployment References

Cookbook recipes keep their model-specific launch commands inline. This page is the cookbook-local entry point for cross-cutting deployment templates so Mintlify links stay inside the `docs/cookbook` root.

| Topic | Use it for |
|---|---|
| [Single-host Docker](../deployment/single-host-docker.md) | One TPU host such as v6e-4, v6e-8, or v7x-8. |
| [GKE Indexed Job](../deployment/gke-indexed-job.md) | Multi-host TPU slices on GKE. |
| [SkyPilot](../deployment/skypilot.md) | Advanced multi-host v6e experiments. |
| [Troubleshooting](../deployment/troubleshooting.md) | Startup, multi-node, runtime, tokenizer, and cache failures shared across recipes. |

The canonical Sphinx pages live under `docs/deployment/`; these cookbook pages are thin Mintlify-safe bridges for recipe links.
