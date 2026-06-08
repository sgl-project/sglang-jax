# Scheduler（模拟变更片段）

## Prefill 批次构造

Scheduler 负责选择进入 prefill 的请求，并以运行队列和 request-to-token pool 初始化 `ScheduleBatch`。prefill token budget 的计算由 `ScheduleBatch.compute_prefill_token_budget` 承担：它根据 `max_prefill_tokens`、已缓存前缀 token 数和是否存在 chunked request 生成 `PrefillBudget`。

调整后的流程如下：

1. `scheduler.py` 从运行队列构造新的 `ScheduleBatch`。
2. `ScheduleBatch.compute_prefill_token_budget` 计算本轮 prefill 可使用的 token 预算，并标记是否需要 chunked prefill。
3. `scheduler.py` 将预算交给 `ScheduleBatch.prepare_for_prefill`，由 batch 对象准备后续执行所需的 prefill 状态。
4. 准备好的 batch 再沿既有执行边界交给模型执行组件。

这种职责划分让 scheduler 保持调度入口职责，批次相关的 token 预算和 chunked 状态则集中在 `ScheduleBatch` 内部表达。
