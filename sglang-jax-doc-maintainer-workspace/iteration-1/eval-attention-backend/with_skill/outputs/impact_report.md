# final impact report

## 文档影响定位

### 需要更新
- `wiki/docs/projects/sglang-jax/06-layers-and-attention.md`（eval 输入中对应 `layers.md` / `attention.md` 片段）：新增 `BlockSparseAttentionBackend` 与 selector entry，Attention Backend 列表和接口说明需要包含 `block_sparse` 路径。

### 可能需要单独确认
- 无图表修改。输入未证明已有图表已覆盖 Attention Backend selector 的具体分支，也未获得单独图表确认。

### 不需要更新
- `wiki/docs/projects/sglang-jax/07-kv-cache.md`：eval 输入只显示 backend 接收 `kv_cache`，没有证明 KV Cache 数据结构、分页策略或内存池行为变化。
- `wiki/docs/projects/sglang-jax/08-pallas-kernels.md`：输入未证明新增 backend 使用 Pallas kernel 或更改 kernel 层。
- `wiki/docs/projects/sglang-jax/04-model-executor.md`：selector entry 位于 attention backend 选择路径；输入未显示 model executor 生命周期或 worker 边界变化。
