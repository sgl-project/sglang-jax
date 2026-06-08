# KV Cache（模拟变更片段）

## 内存池初始化

运行时初始化 KV Cache 相关结构时，会创建请求到 token 的映射池。`create_memory_pool` 读取 `ServerArgs.enable_hybrid_memory_pool`：未启用时使用默认 `ReqToTokenPool`，启用时选择 `HybridReqToTokenPool`。该配置只描述池实现的选择入口，具体容量比例仍由 `mem_fraction_static` 等内存参数控制。
