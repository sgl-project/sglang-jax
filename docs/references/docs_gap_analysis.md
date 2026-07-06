# Documentation Gap Analysis

This page tracks the result of the first documentation reorganization pass.

## Phase 1 structure

```text
docs/
  conf.py, index.rst, Makefile, requirements.txt
  get_started/          # Sphinx: installation and environment setup
  basic_usage/          # Sphinx: general and legacy usage notes
  features/             # Sphinx: runtime features
    design/             # Sphinx: feature-owned design/RFC docs
  architecture/         # Sphinx: architecture walkthrough
  multimodal/           # Sphinx: multimodal usage and architecture
  developer_guide/      # Sphinx: contributor and maintainer docs
  references/           # Sphinx: meta docs and gap analysis
  cookbook/             # Mintlify: model recipes and validation notes
```

## Content assessment

| Area | Current state | Follow-up |
|---|---|---|
| `architecture/` | Strong numbered walkthrough exists. | Needs technical review against the current runtime and diagrams. |
| `basic_usage/` | Model-specific pages now point to cookbook recipes instead of duplicating launch commands. | Keep this section limited to stable entry points and general usage notes. |
| `features/` | Good feature-level material, now merged with tracked design docs. | Normalize page structure and add owner/status metadata. |
| `developer_guide/` | Contribution, CI, TPU, benchmarking, release, and community docs exist. | Add a focused "add a new model" engineering guide that complements cookbook recipes. |
| `evaluations/` | Former single evaluation note moved to cookbook benchmark references. | Define a repeatable accuracy evaluation methodology and accepted datasets. |
| `performance/` | Former Qwen3 report moved to cookbook benchmark references. | Add benchmark methodology, hardware/version metadata, and quality bars for posted numbers. |
| `cookbook/` | Strong Mintlify-style model recipe tree with a working `docs.json`. | Add missing validation metadata, benchmark references, and template/checklist consistency. |
| `multimodal/` | Usage and architecture RFC exist in corrected path. | Fill offline inference and validate online API examples. |

## Missing docs

| Priority | Gap | Suggested owner area |
|---|---|---|
| P0 | Cookbook recipe quality bar: required hardware, command, accuracy, throughput, version, and known limits. | Cookbook |
| P0 | New model onboarding guide from code changes to cookbook recipe and validation. | Developer guide + cookbook |
| P1 | Architecture review pass for scheduler, executor, cache, DP, PD, and multimodal docs. | Architecture |
| P1 | Performance methodology with reproducible commands, acceptable model/hardware matrices, and "do not publish if too slow" criteria. | References + cookbook benchmarks |
| P1 | Accuracy evaluation methodology for EvalScope datasets and sampling configs. | References + cookbook benchmarks |
| P2 | CLI/server API reference generated or synced from launch arguments. | Features + references |
| P2 | Troubleshooting split between general runtime failures and model-specific recipe failures. | Developer guide + cookbook |

## Work estimate

| Workstream | Estimate | Notes |
|---|---:|---|
| Phase 1 structure and navigation | 0.5-1 day | Covered by this reorganization pass. |
| Link cleanup and duplicate usage consolidation | 1-2 days | Mainly `basic_usage/` to cookbook redirects or merges. |
| Architecture technical review | 2-4 days | Requires runtime owners to verify current behavior. |
| New model onboarding guide | 1-2 days | Should cover model code, config, tests, recipe, and benchmark evidence. |
| Performance/evaluation methodology | 2-3 days | Needs agreed benchmark matrix and quality bar. |
| Full Mintlify conversion for all docs | 3-5 days for config, 1-2 weeks for content polish | Not recommended yet; current content better fits a Sphinx main tree plus Mintlify cookbook. |
