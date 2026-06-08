# final impact report

## 文档影响定位

### 需要更新
- `wiki/docs/projects/sglang-jax/03-scheduler.md`（eval 输入中对应 `scheduler.md` 片段）：文档仍说 scheduler 直接计算 prefill token budget；输入代码显示 `scheduler.py` 现在委托 `ScheduleBatch.compute_prefill_token_budget` 计算预算，并由 batch 准备 prefill。

### 可能需要单独确认
- 无图表修改。输入未证明现有图表展示了过期的 batch-to-worker 边界，也未获得图表确认。

### 不需要更新
- `wiki/docs/projects/sglang-jax/04-model-executor.md`（eval 输入中对应 `model-executor.md` 片段）：executor 仍接收准备好的 batch 并执行 prefill/decode；输入未显示 batch-to-worker 接口或 executor 责任变化。
- `wiki/docs/projects/sglang-jax/01-architecture-overview.md`：变化发生在 scheduler 内部和 `ScheduleBatch` 的职责划分，不改变 overview 级别的组件关系。
