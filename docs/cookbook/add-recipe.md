---
title: "Add a Cookbook Recipe"
---

# Add a Cookbook Recipe

Cookbook pages answer one question: how to run a specific model on specific hardware with SGL-JAX.

## Placement

| Model type | Path |
|---|---|
| Autoregressive LLM or VLM | `docs/cookbook/autoregressive/<Vendor>/<Model>.md` |
| Diffusion or media generation model | `docs/cookbook/diffusion/<Vendor>/<Model>.md` |
| Shared launch or benchmark reference | `docs/cookbook/base/<topic>.md` |
| Cookbook-local operational bridge | `docs/cookbook/deployment/<topic>.md`, `docs/cookbook/get_started/<topic>.md`, or `docs/cookbook/developer_guide/<topic>.md` |

Use vendor directories with upstream-style names, for example `Qwen/`, `DeepSeek/`, `Xiaomi/`, or `Wan/`.

## Required sections

Every model recipe should keep the same structure:

1. Model Introduction
2. Deployment
3. Invocation
4. Benchmark

For a validated recipe, include:

| Evidence | Requirement |
|---|---|
| Hardware | TPU/GPU type, topology, node count, chip count, and `--tp-size` / `--dp-size` / `--ep-size`. |
| Environment | SGL-JAX commit or version, JAX/JAXLIB/libtpu versions, container image when relevant. |
| Launch | Copy-pasteable server command for the claimed primary path. |
| Invocation | At least one `curl` or OpenAI-compatible request example. |
| Accuracy | Dataset, tool version, sampling config, and result. |
| Performance | `bench_serving` command, concurrency/batch settings, ISL/OSL, and representative throughput/latency. |
| Limits | Known unsupported variants, memory limits, or unvalidated topologies. |

Keep a recipe marked as Starter until at least one primary path has real accuracy and performance evidence.

## Navigation checklist

When adding a page:

1. Add it to `docs/cookbook/docs.json`.
2. Add it to the relevant model index, such as `autoregressive/index.md` or `diffusion/index.md`.
3. Add or update the hardware coverage table in `docs/cookbook/index.md`.
4. Keep benchmark matrices in the recipe benchmark section or summarize them there; do not create a standalone benchmark-report section.
5. If the recipe introduces a non-obvious runtime flag, link to `base/launch-flags-reference.md` using a file-relative path from the current page (for example, `../../base/launch-flags-reference.md` from a vendor recipe).
6. Keep internal cookbook links inside the `docs/cookbook` root. Use file-relative paths with the `.md` extension, and point directory landing pages at their explicit `index.md` file; do not use root-relative site routes or paths outside the cookbook root.
7. For cross-page section links, add a stable explicit anchor such as `<a id="configuration-tips"></a>` immediately before the target heading and link to that ID. Do not rely on an automatically generated heading slug, because GitHub and Mintlify generate different IDs for numbered headings.
