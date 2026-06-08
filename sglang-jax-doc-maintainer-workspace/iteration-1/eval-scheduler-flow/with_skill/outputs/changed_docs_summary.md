# changed docs summary

- `docs/03-scheduler.md`：重构 prefill 批次构造说明，将 prefill token budget 的直接归属从 scheduler-level prose 移到 `ScheduleBatch.compute_prefill_token_budget`。
- 未修改 `04-model-executor.md`，因为 eval 输入未显示 executor 接收 batch 的边界发生变化。
- 未修改 overview 或图表。
