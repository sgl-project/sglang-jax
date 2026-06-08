# Layers and Attention（模拟变更片段）

## Attention Backend 选择

Attention 层通过 backend selector 根据运行参数选择具体 backend。除已有 dense attention 路径外，selector 现在包含 `attention_backend == "block_sparse"` 分支，并返回 `BlockSparseAttentionBackend(model_runner, kv_cache)`。

| backend 名称 | 实现 | 代码事实 |
|---|---|---|
| `block_sparse` | `BlockSparseAttentionBackend` | `forward_extend` 接收 `q`、`k`、`v` 和 `block_mask`，并调用 `block_sparse_attention(q, k, v, block_mask)`。 |

### 通用背景：Block Sparse Attention

Block sparse attention 通常通过 block mask 限制参与 attention 的位置集合，用于表达稀疏注意力模式。这是通用技术背景，不等同于 sglang-jax 对吞吐、显存、硬件或默认策略的承诺。

### sglang-jax 实现边界

当前输入能够确认的项目事实是：`backend_selector.py` 增加了 `block_sparse` 选择分支，`blocksparse_backend.py` 提供 `BlockSparseAttentionBackend`，并在 extend 路径中使用 `block_mask` 调用 `block_sparse_attention`。decode 路径、默认启用条件、性能收益和设备覆盖范围需要以更多代码或基准结果为依据，不能由本次输入推断。
