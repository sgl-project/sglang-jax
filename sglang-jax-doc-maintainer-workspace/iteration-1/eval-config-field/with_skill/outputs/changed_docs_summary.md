# changed docs summary

- `docs/13-configuration-reference.md`：在 Memory 配置片段中加入 `enable_hybrid_memory_pool`，说明默认值和 `HybridReqToTokenPool` 选择语义。
- `docs/07-kv-cache.md`：在 KV Cache 内存池初始化片段中说明 `create_memory_pool` 根据 `ServerArgs.enable_hybrid_memory_pool` 选择 request-to-token pool 实现。
- 未修改图表；没有把 PR 号、提交哈希、代码行号、作者或 release-note 语气写入模拟文档。
