# update plan

## 更新计划

### `wiki/docs/projects/sglang-jax/13-configuration-reference.md`
决策：段落融入
原因：新增字段是已有 ServerArgs / Memory 配置表的一行，不需要新章节。
计划：
- 在内存相关配置表中加入 `enable_hybrid_memory_pool`。
- 说明默认值为 `False`，用途是选择是否启用 hybrid request-to-token memory pool。
- 避免写入性能提升、默认启用、模型/设备支持等未由输入代码证明的结论。

### `wiki/docs/projects/sglang-jax/07-kv-cache.md`
决策：段落融入
原因：当前文档说初始化时构建 request-to-token pool，但未体现该池现在可以由配置切换为 hybrid 实现。
计划：
- 改写初始化内存池说明，描述 `create_memory_pool` 根据 `ServerArgs.enable_hybrid_memory_pool` 选择 `ReqToTokenPool` 或 `HybridReqToTokenPool`。
- 保留架构粒度，说明选择点和责任边界，不展开未提供的内部实现细节。

确认状态：eval 提示假定用户已确认文本编辑；未确认图表编辑，因此不修改图表。
