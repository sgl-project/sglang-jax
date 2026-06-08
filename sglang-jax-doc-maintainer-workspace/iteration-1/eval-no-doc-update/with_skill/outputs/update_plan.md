# update plan

## 更新计划

### `wiki/docs/projects/sglang-jax/03-scheduler.md`
决策：不更新
原因：测试文件只是格式化断言，未改变 scheduler 行为或文档化语义。
计划：
- 不做文本编辑。

### `wiki/docs/projects/sglang-jax/development.md`
决策：不更新
原因：CI cache key 只影响依赖缓存命中策略，不属于架构文档粒度；输入未显示 development 文档中有需要同步的 cache-key 说明。
计划：
- 不做文本编辑。

### 图表
决策：不更新
原因：没有模块关系、数据流或控制流变化。
计划：
- 不做图表编辑。

确认状态：eval 提示假定用户确认 no-edit 决策；因此保持文档不变。
