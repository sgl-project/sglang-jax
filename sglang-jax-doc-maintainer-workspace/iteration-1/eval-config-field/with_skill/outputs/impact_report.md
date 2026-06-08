# final impact report

## 文档影响定位

### 需要更新
- `wiki/docs/projects/sglang-jax/13-configuration-reference.md`（eval 输入中对应 `configuration.md` 片段）：`server_args.py` 新增公开 `ServerArgs.enable_hybrid_memory_pool`，属于用户可见配置，需要进入配置参考。
- `wiki/docs/projects/sglang-jax/07-kv-cache.md`（eval 输入中对应 `kv-cache.md` 片段）：`memory_pool.py` 根据 `enable_hybrid_memory_pool` 在 `ReqToTokenPool` 与 `HybridReqToTokenPool` 之间选择，请求到 token 的池选择逻辑会影响 KV Cache 章节对初始化阶段内存池的说明。

### 可能需要单独确认
- 无图表修改。当前输入只证明文本说明需要更新；没有确认任何 SVG 或 Excalidraw 关系图已过期。

### 不需要更新
- `wiki/docs/projects/sglang-jax/01-architecture-overview.md`：新增配置只改变内存池选择入口，不改变总体运行时拓扑。
- `wiki/docs/projects/sglang-jax/03-scheduler.md`：调度行为未变，`memory_pool.py` 的池选择不改变 scheduler 责任边界。
