# sglang-jax-doc-maintainer iteration 1 benchmark

## Summary

| Eval | Baseline result | With-skill result | Improvement |
|---|---|---|---|
| config-field | Updated only `13-configuration-reference.md`; mentioned `07-kv-cache.md` only as unchanged; no separate pre-edit impact report/update plan; changed-files list included code/doc line-number traces. | Created a pre-edit impact report and update plan; marked both `13-configuration-reference.md` and `07-kv-cache.md` as needing text updates; simulated docs avoid PR/commit/line-number traces and unsupported performance claims; validation cites `enable_hybrid_memory_pool`, `ReqToTokenPool`, and `HybridReqToTokenPool`. | Better targeting, confirmation-gated planning, trace hygiene, and validation evidence. |
| scheduler-flow | Identified `03-scheduler.md` and left `04-model-executor.md` unchanged, but appended a note instead of restructuring stale responsibility prose; did not explain overview-doc impact. | Marked `03-scheduler.md` as primary, chose `章节重构`, explained why `04-model-executor.md` and `01-architecture-overview.md` stay unchanged, and validated references to `ScheduleBatch`, `compute_prefill_token_budget`, `PrefillBudget`, and `prepare_for_prefill`. | Better structure-aware update decision and candidate-doc reasoning. |
| attention-backend | Updated `06-layers-and-attention.md` and also `04-model-executor.md`; mixed general block-sparse motivation with project-specific claims; asserted faster long-context behavior/reduced cost without project evidence. | Updated only `06-layers-and-attention.md`; separated `通用背景：Block Sparse Attention` from `sglang-jax 实现边界`; used code-derived facts for `backend_selector.py` and `BlockSparseAttentionBackend`; explicitly avoided throughput, speedup, memory-saving, hardware-support, and default-enable claims. | Better factual discipline, external-research boundaries, and doc targeting. |
| no-doc-update | Correctly reported no architecture docs need updates, explained CI/test-only scope, left wiki docs unchanged, and included validation. | Also reported no docs update needed; explicitly listed `03-scheduler.md` and `development.md` as no-update candidates; update plan says no text or diagram edits; validation states empty doc diff, no trace introduction, reference checks not applicable, and docs build not needed. | Similar correct no-op behavior with more structured evidence. |

## Hard-rule evaluation

| Hard rule | With-skill result | Evidence |
|---|---|---|
| Impact report before edits | Passed | Each eval has `with_skill/outputs/impact_report.md` and `update_plan.md` before simulated docs under `outputs/docs/` or before the no-edit decision. |
| Structure-aware update plan | Passed | `config-field` uses paragraph integration for config/KV cache docs; `scheduler-flow` chooses section restructuring; `attention-backend` integrates into the Attention Backend section; `no-doc-update` explicitly chooses no update. |
| No PR/commit/line-number traces in final docs | Passed | Simulated final docs avoid PR numbers, commit hashes, code line references, author/reviewer names, and release-note wording; each validation report records this check. |
| No unsupported project-specific performance claims | Passed | `attention-backend` final docs separate general background from project facts and explicitly avoid speedup/throughput/memory/hardware/default-enable claims; `config-field` also avoids hybrid-pool performance claims. |
| External research only for general mature-technology background | Passed | `attention-backend` uses general block sparse attention background only in a labeled background section; project facts come from eval snippets. |
| No doc update for CI/test-only change | Passed | `no-doc-update` produced no changed-doc artifacts and states CI cache keys/test formatting do not affect documented architecture. |
| Validation evidence included | Passed | All four with-skill runs include `validation_report.md` with diff scope, trace check, reference checks or non-applicability, and docs build status/reason not run. |

## Patterns

- The skill addressed baseline misses around missing impact reports, weak update decisions, PR/line trace risk, append-only documentation, unsupported project-specific claims, and missing validation evidence.
- No hard-rule failures were observed in iteration 1 with-skill outputs, so Task 4 refinements are not required from this benchmark.

## Decision

Proceed to final verification/handoff. With-skill outputs clearly improve impact analysis, update planning, writing constraints, and validation over baseline for `config-field`, `scheduler-flow`, and `attention-backend`; `no-doc-update` remains correct and more structured than baseline.
