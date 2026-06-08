# update plan

## 更新计划

### `wiki/docs/projects/sglang-jax/06-layers-and-attention.md`
决策：段落融入
原因：新增 backend 扩展现有 Attention Backend interface 和 selector 表述，不需要独立文档。若当前页面将 layers 与 attention 分开，可分别在 backend 列表和接口段落内融入，而不是追加 release-note。
计划：
- 在 Attention Backend 选择说明中加入 `attention_backend == "block_sparse"` 对应 `BlockSparseAttentionBackend`。
- 在 backend 表中加入 `block_sparse`，只写代码输入可证明的事实：类名、selector 条件、`forward_extend` 接收 `block_mask` 并调用 `block_sparse_attention`。
- 将 block sparse attention 的一般背景写成“通用背景”，与 sglang-jax 具体实现事实分开。
- 不写吞吐提升、显存节省、硬件支持、默认启用或完整支持等无证据项目结论。

确认状态：eval 提示假定用户已确认文本编辑；未确认图表编辑，因此不修改图表。
