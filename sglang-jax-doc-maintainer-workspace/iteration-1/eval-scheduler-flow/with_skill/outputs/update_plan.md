# update plan

## 更新计划

### `wiki/docs/projects/sglang-jax/03-scheduler.md`
决策：章节重构
原因：原文把 prefill token budget、chunked prefill 判定和 batch 构造都归到 scheduler 直接职责；简单追加说明会形成补丁式矛盾。需要把“scheduler 选择请求”和“ScheduleBatch 承载预算计算/准备”的职责重新组织到同一流程段落。
计划：
- 改写 scheduler prefill 流程段落：scheduler 发起 prefill 调度并创建 `ScheduleBatch`。
- 将 token budget 计算描述移动到 `ScheduleBatch.compute_prefill_token_budget`。
- 在流程列表中说明 `ScheduleBatch.prepare_for_prefill` 使用预算结果准备 prefill batch。
- 保留 `model-executor.md` 不变，并在完成报告中说明 executor 边界未变。

确认状态：eval 提示假定用户已确认文本编辑；未确认图表编辑，因此不修改图表。
