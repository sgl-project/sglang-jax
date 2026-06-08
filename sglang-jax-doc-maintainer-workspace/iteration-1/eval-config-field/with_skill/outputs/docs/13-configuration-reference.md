# 配置参考（模拟变更片段）

## Memory

`mem_fraction_static` 控制为 KV Cache 预留的静态比例。请求到 token 的映射池由内存池初始化流程创建，并可通过 `enable_hybrid_memory_pool` 选择是否使用 hybrid request-to-token pool。

| 字段 | 默认值 | 说明 |
|---|---:|---|
| `mem_fraction_static` | `0.88` | 控制 KV Cache 等静态内存区域的预留比例。 |
| `enable_hybrid_memory_pool` | `False` | 控制 `memory_pool.py` 在初始化请求到 token 的映射池时是否选择 `HybridReqToTokenPool`。关闭时继续使用默认 `ReqToTokenPool`。 |
