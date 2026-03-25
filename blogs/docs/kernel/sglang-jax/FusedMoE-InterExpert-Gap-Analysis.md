# Fused MoE Expert 间隙分析：FFN 结束到下一 FFN 开始之间发生了什么

> **内核源码**：`python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py`
> **硬件平台**：TPU v7 (Ironwood)，16 设备 (32 cores)，EP=32
> **模型配置**：Ring-1T FP8（hidden=8192, intermediate=2048, 256 experts, top_k=8, 8 experts/core）
> **分析依据**：TPU profiling trace + 内核源码走读

---

## 1. 问题定义

从 profiling trace 中可以观察到：在 fused-moe kernel 执行期间，Tensor Core / MXU 活动呈现**间歇性的 burst 模式**——每段 MXU 密集计算（即一个 expert 的 FFN）之间存在明显的**空隙（gap）**。这些空隙中 MXU 大部分空闲，但 Vector ALU、Vector Load/Store 仍有活动。

本文档逐步拆解每个 expert FFN 结束到下一个 expert FFN 开始之间发生的所有操作，量化每个操作的开销来源，判断访存瓶颈的根因。

---

## 2. Expert 循环结构

每个 expert 的处理在 `run_per_expert_pipelined()` (L2247) 中执行，循环体的完整操作序列如下：

```
Expert E_i 的完整执行流（一次循环迭代）:

  ① [仅 E_0] _first_load()             ← 预取 E_0 的 W1/W3
  ② [仅 SE_block=0] run_shared_expert_slice(0)  ← 第一个 SE block
  ③ start_a2a_scatter(E_{i+1})          ← 启动下一 expert 的 scatter DMA
  ④ run_shared_expert_slice(block_N)    ← SE block 穿插
  ⑤ wait_a2a_scatter_recv(E_i)          ← 等待当前 expert 的 scatter 接收完成
  ⑥ expert_ffn(E_i)                     ← ★ FFN 计算（MXU 密集）
  ⑦ start_a2a_gather(E_i)              ← 启动当前 expert 的 gather DMA
  ⑧ run_shared_expert_slice(block_N+1)  ← SE block 穿插
  ⑨ wait_a2a_scatter_send(E_i)          ← 等待当前 expert 的 scatter 发送完成
  ⑩ sync_barrier()                      ← 全局 barrier 同步
```

因此，**E_i 的 FFN 结束到 E_{i+1} 的 FFN 开始之间的完整间隙**为：

```
expert_ffn(E_i) 结束
  ↓
  ⑦ start_a2a_gather(E_i)              [DMA 启动，非阻塞]
  ⑧ run_shared_expert_slice(block)      [MXU 计算，SE 穿插]
  ⑨ wait_a2a_scatter_send(E_i)          [可能阻塞等待 DMA]
  ⑩ sync_barrier()                      [全局同步]
  ── 循环边界 ──
  ③ start_a2a_scatter(E_{i+2})          [DMA 启动，非阻塞]
  ④ run_shared_expert_slice(block)      [MXU 计算，SE 穿插]
  ⑤ wait_a2a_scatter_recv(E_{i+1})      [可能阻塞等待 DMA]
  ↓
expert_ffn(E_{i+1}) 开始
  ↓ (内部第一步)
  wait_fetch_bw1/bw3                    [等待权重预取完成]
```

---

## 3. 逐操作详细分析

### 3.1 start_a2a_gather(E_i) — Gather DMA 启动

**代码**：L879-918

**操作**：遍历所有 `num_devices` 个设备，为每个有 token 的设备发起 HBM→HBM 的异步 DMA 拷贝（local copy 或 remote copy via ICI）。

**耗时构成**：
- Python unrolled loop over `num_devices`（EP=32 时展开 32 次）
- 每次迭代：1 次 SMEM 读取（`d2e_count`）+ 条件 DMA 启动
- DMA 是异步的（`.start()`），本身不等待完成

**估算**：
- 32 个 device 的 DMA 启动，每个 DMA 启动约 0.1-0.3 us
- 总计约 **3-10 us**（纯 DMA 命令发射开销）
- 这里 **不是阻塞操作**，但 DMA 命令发射有 Scalar ALU 开销

**profiling 特征**：Scalar ALU 活跃，Vector 单元低活跃

### 3.2 run_shared_expert_slice(block) — SE 穿插计算

**代码**：L2071-2204

**操作**：执行一个 shared expert 的 FFN 计算块（bse 大小的 intermediate 维度切片）。

**耗时构成**：
- MXU matmul：`[local_tokens, hidden] × [hidden, bse]`（SE FFN1）或 `[local_tokens, bse] × [bse, hidden]`（SE FFN2）
- 权重双缓冲预取（与计算重叠）
- Token VMEM 缓冲管理

**估算（BS=512, DP=4, local_tokens≈128, bse=256）**：
- SE FFN1 matmul: `128 × 8192 × 256 × 2 = 537M FLOPs` → TPU v7 FP8 4614 TFLOPS → 约 **0.12 us**
- 但受限于权重加载：`8192 × 256 × 1B (FP8) = 2 MB` → HBM 7380 GB/s → 约 **0.27 us**
- 实际约 **0.3-1 us**（含 token staging 等 overhead）

**profiling 特征**：间隙中可见的 MXU 短脉冲 + Vector Load 活跃（权重加载）

### 3.3 wait_a2a_scatter_send(E_i) — 等待 Scatter 发送完成

**代码**：L865-877

**操作**：等待当前 expert 的所有远程 scatter DMA 发送完成。scatter 在**上一次循环迭代**的步骤③中启动。

**耗时构成**：
- 如果 scatter 在 FFN 计算期间已经完成 → **0 us**（信号量立即返回）
- 如果 scatter 尚未完成 → **等待 ICI 传输完成**

**关键因素**：
- Scatter 数据量：每个 expert 从 bt 个 token 中抽取路由到该 expert 的 token，发送到各设备
- BS=512, top_k=8, EP=32 → 每 core 每 expert 平均 16 个 token
- 每个 token：`8192 × 2B (bf16) = 16 KB`
- 总 scatter 数据量 per expert：`16 × 16 KB = 256 KB`
- ICI 带宽约 40 GB/s → 传输时间 `256 KB / 40 GB/s ≈ 6 us`

**实际情况**：scatter(E_i) 在 E_{i-1} 的循环中启动（步骤③启动 E_{i+1} 的 scatter，即当前 expert E_i 的 scatter 是在上一轮启动的），因此 scatter 有整个 FFN(E_{i-1}) 的时间来完成。如果 FFN 耗时 > 6 us，则此处不阻塞。

**profiling 特征**：如果阻塞，表现为 P state stall（所有单元空闲）

### 3.4 sync_barrier() — 全局 Barrier 同步

**代码**：L2296

**操作**：所有 32 个 core 之间的全局同步点。

**耗时构成**：
- 等待最慢的 core 到达 barrier
- Expert 负载不均衡时，快 core 等待慢 core
- 固定 barrier 延迟 + 负载不均衡的等待时间

**估算**：
- 固定 barrier 延迟：约 **1-3 us**
- 负载不均衡附加：取决于各 core 的 expert token 分布差异
- 如果某个 core 的 expert 有 40 个 token 而其他 core 平均 16 个 → 该 core FFN 耗时 2.5x → 其他 core 等待额外 **10-20 us**

**profiling 特征**：P state stall（完全空闲），持续时间因 core 而异

**这是间隙中最不可预测、也可能最大的开销来源。**

### 3.5 start_a2a_scatter(E_{i+2}) — 下一 Expert 的 Scatter 启动

**代码**：L791-849

**操作**：遍历 bt 个 token，根据路由表为下下个 expert 发起 scatter DMA。使用 `lax.fori_loop` 避免代码膨胀。

**耗时构成**：
- `fori_loop` 遍历 bt 个 token（bt=64 或 128）
- 每个 token：top_k 次 SMEM 读取（路由表）+ 条件 DMA 启动
- DMA 是异步的

**估算（bt=64, top_k=8）**：
- 64 × 8 = 512 次路由查表 + 条件判断
- 每次 ~0.05-0.1 us（Scalar ALU + SMEM 访问）
- 总计约 **25-50 us**

**这是间隙中开销最大的操作之一！** `fori_loop` 的循环开销在 Pallas 中不可忽视。

**profiling 特征**：Scalar ALU 高活跃，Vector Store 间歇活跃（DMA 命令发射）

### 3.6 run_shared_expert_slice(block) — 第二个 SE 穿插

与 3.2 相同，约 **0.3-1 us**。

### 3.7 wait_a2a_scatter_recv(E_{i+1}) — 等待 Scatter 接收完成

**代码**：L851-863

**操作**：等待 E_{i+1} 的所有 scatter 数据到达本地 HBM。

**耗时构成**：
- scatter(E_{i+1}) 在步骤③中刚启动
- 从启动到等待之间只隔了一个 SE block（0.3-1 us）
- scatter 传输需要约 6 us
- **几乎一定会阻塞**，等待约 **5-6 us**

**profiling 特征**：P state stall

### 3.8 expert_ffn 内部的首次 wait — 权重预取等待

**代码**：L1636-1638

**操作**：`wait_fetch_bw1` / `wait_fetch_bw3` — 等待下一 expert 的 W1/W3 权重从 HBM 加载到 VMEM。

**权重预取时序**：
- 预取在 **E_i 的 FFN2 最后一个 tile** 中启动（L1757-1769）
- 从预取启动到此处等待，间隔了：gather 启动 + SE block + scatter send wait + barrier + scatter 启动 + SE block + scatter recv wait
- 预取的数据量：`W1 [hidden, bf] × FP8 = 8192 × 512 × 1B = 4 MB`（W1 + W3 共 8 MB）
- HBM 带宽 7380 GB/s → 预取时间 `8 MB / 7380 GB/s ≈ 1.1 us`

**实际情况**：间隙已经有 30-80 us，远超 1.1 us 的预取时间 → **权重预取一般不会阻塞**。

但如果间隙中 HBM 带宽被其他操作（scatter/gather DMA、SE 权重加载）竞争 → 预取可能延迟。

---

## 4. 间隙时序总结

### 4.1 理想情况（负载均衡、无竞争）

| 步骤 | 操作 | 估算耗时 | 是否阻塞 |
|------|------|----------|----------|
| ⑦ | start_a2a_gather(E_i) | 3-10 us | 否 |
| ⑧ | run_shared_expert_slice | 0.3-1 us | 否（MXU 计算） |
| ⑨ | wait_a2a_scatter_send(E_i) | 0 us* | 视情况 |
| ⑩ | sync_barrier() | 1-3 us | 是（等待同步） |
| ③ | start_a2a_scatter(E_{i+2}) | **25-50 us** | 否（但 CPU 密集） |
| ④ | run_shared_expert_slice | 0.3-1 us | 否（MXU 计算） |
| ⑤ | wait_a2a_scatter_recv(E_{i+1}) | **5-6 us** | 是（等待 DMA） |
| | wait_fetch_bw1/bw3（FFN 内部） | 0 us* | 一般不阻塞 |
| | **总间隙** | **~35-70 us** | |

*: 已被前序操作完全重叠

### 4.2 实际瓶颈情况（负载不均衡）

| 步骤 | 额外开销 |
|------|----------|
| sync_barrier | +10-30 us（等待慢 core） |
| wait_a2a_scatter_send | +0-5 us（scatter 与 FFN 未完全重叠） |
| wait_fetch_bw1/bw3 | +0-2 us（HBM 带宽竞争） |
| **实际总间隙** | **~45-100+ us** |

### 4.3 与 FFN 计算时间对比

| 项目 | 耗时 |
|------|------|
| 单 expert FFN（T_avg=16, FP8） | ~6-10 us |
| Expert 间隙 | ~35-100 us |
| **间隙/FFN 比率** | **3.5x - 15x** |

**间隙时间远大于 FFN 计算时间**，这解释了 profiling 中 MXU 利用率低的现象。

---

## 5. 瓶颈根因分析

### 5.1 第一大瓶颈：start_a2a_scatter 的 fori_loop 开销

`start_a2a_scatter` 中的 `fori_loop` 遍历 bt 个 token × top_k 次路由查表，是间隙中**耗时最长的单一操作**（25-50 us）。

**根因**：
- 每个 token 需要 top_k=8 次 SMEM 查表确定路由目标
- 每次路由可能触发一个条件 DMA 启动
- `fori_loop` 在 Pallas 中编译为顺序执行的标量循环
- bt=64/128 × top_k=8 = 512/1024 次循环迭代

**这不是传统意义上的"访存瓶颈"，而是标量控制流瓶颈**。Scatter 的 DMA 数据传输本身是异步的，但 DMA 命令的发射是串行的标量操作。

### 5.2 第二大瓶颈：sync_barrier 的负载不均衡等待

`sync_barrier()` 要求所有 32 个 core 同步。当 expert 负载不均衡时：

- 部分 core 的 expert 收到远超平均数的 token → FFN 耗时长
- 其他 core 在 barrier 处空等
- 使用 EPLB 后有所缓解，但无法完全消除

**这是通信+同步瓶颈，不是纯访存瓶颈。**

### 5.3 第三大瓶颈：wait_a2a_scatter_recv 的 ICI 延迟

Scatter(E_{i+1}) 在步骤③启动后，仅隔一个 SE block（~1 us）就需要在步骤⑤等待接收完成。ICI 传输需要约 6 us。

**这是 ICI 通信带宽瓶颈**。改进方式是增大 scatter 启动到等待之间的距离（即在间隙中插入更多非阻塞工作）。

### 5.4 第四大瓶颈：start_a2a_gather 的命令发射开销

`start_a2a_gather` 中 Python unrolled loop over `num_devices=32`，每个设备发起一次条件 DMA。32 次 DMA 启动约 3-10 us。

**这也是标量控制流瓶颈。**

---

## 6. 瓶颈本质总结

| 瓶颈类型 | 操作 | 耗时占比 | 本质 |
|----------|------|----------|------|
| **标量控制流** | start_a2a_scatter (fori_loop) | ~50-60% | Scalar ALU 串行循环 |
| **ICI 通信** | wait_a2a_scatter_recv | ~10-15% | ICI 带宽 + 延迟 |
| **负载均衡** | sync_barrier | ~15-25% | Core 间不均衡等待 |
| **DMA 命令发射** | start_a2a_gather | ~5-10% | 32 设备串行发射 |
| **SE 计算** | run_shared_expert_slice ×2 | ~2-5% | 有效利用（非浪费） |

**核心结论：间隙的主要瓶颈不是 HBM 访存，而是标量控制流（scatter 路由循环）和通信同步（barrier + ICI 延迟）。**

---

## 7. Profiling Trace 对应解读

从 profiling 截图中可以观察到的信号与上述分析的对应关系：

| Profiling 行 | 间隙中表现 | 对应操作 |
|-------------|-----------|----------|
| **MXU** | 间隙中几乎无活动，仅有极短脉冲 | SE block 的小 matmul |
| **Scalar ALU** | 间隙中**持续高活跃** | scatter/gather 的路由循环、条件判断 |
| **Vector ALU** | 间隙中低活跃 | SE 的激活函数（SiLU） |
| **Vector Load** | 间隙中间歇活跃 | SE 权重预取、token staging |
| **Vector Store** | 间隙中间歇活跃 | DMA 结果写回 |
| **Vector Fills** | 间隙中偶尔出现 | 累加器初始化 |
| **Tensor Core** | 间隙中无活动 | 无大型 matmul |
| **P state** | 间隙中可能出现 stall | barrier 等待、DMA 等待 |

---

## 8. 与 Roofline 分析的关系

Roofline 分析（见 `FusedMoE-Roofline-Analysis.md`）得出的结论是 FFN 在 decode 场景下是 **HBM 带宽瓶颈**（AI=33.6 vs 临界值 625.2），即权重加载是限制因素。

但 Roofline 分析假设的是**稳态流水线**——连续不断地执行 FFN matmul，权重加载与计算重叠。

实际 profiling 显示的间隙暴露了 Roofline 模型未捕获的**流水线气泡**：

```
Roofline 假设:
  weight_load(E0) → FFN(E0) → weight_load(E1) → FFN(E1) → ...
  ↑ 重叠 ↑                  ↑ 重叠 ↑

实际执行:
  FFN(E0) → [gap: scatter/gather/barrier ~50us] → FFN(E1) → [gap ~50us] → ...
            ↑ MXU 空闲 ↑                                   ↑ MXU 空闲 ↑
```

**MXU 有效利用率** ≈ `FFN_time / (FFN_time + gap_time)` ≈ `8 / (8 + 50)` ≈ **14%**

这意味着即使权重预取完全隐藏了加载延迟，MXU 利用率仍然很低，因为间隙中的标量控制流和通信同步开销占据了大部分时间。

---

## 9. 潜在优化方向

### 9.1 减少 Scatter 路由循环开销

- **向量化路由查表**：将 per-token 的标量路由查表改为向量化操作（Vector ALU 批量处理）
- **预计算 scatter 计划**：在内核外或 routing 阶段预计算每个 expert 的 scatter DMA 列表，内核中直接执行预计算的 DMA 列表
- **减少 bt**：更小的 bt 减少 scatter 循环次数，但可能影响其他性能

### 9.2 增加 Scatter 启动到等待的距离

当前 scatter(E_{i+1}) 在步骤③启动，步骤⑤等待，中间仅隔一个 SE block。可以考虑：
- **更早启动 scatter**：在 FFN 计算期间就启动下一 expert 的 scatter
- **更晚等待 scatter**：将等待推迟到 FFN 内部的第一次 token 使用时

### 9.3 消除或弱化 sync_barrier

- 评估 `sync_barrier()` 的必要性——是否可以用更细粒度的信号量同步替代全局 barrier
- 如果 barrier 是为了保护 scatter/gather 缓冲区的双缓冲一致性，可以考虑更细粒度的 per-buffer 同步

### 9.4 Gather DMA 命令发射优化

- 将 Python unrolled 32-device 循环改为 `fori_loop`（减少代码体积）
- 或反之，如果能利用 Pallas 的并行 DMA 启动

---

## 10. 附录：Block 配置参数对间隙的影响

| 参数 | 对间隙的影响 |
|------|------------|
| `bt` | bt 增大 → scatter fori_loop 迭代数增多 → 间隙增大 |
| `bts` | 不直接影响间隙（影响 FFN 内部） |
| `bf` | 不直接影响间隙（影响 FFN 内部 bf loop 数） |
| `bse` | bse 增大 → SE block 变少 → 间隙中 SE 耗时增加但次数减少 |
| `bd1/bd2` | 不直接影响间隙（影响 FFN 内部 tile 数） |

当前 profiling 的 block 配置（从 kernel 名称解析）：
```
bt=64, bf=2048, bd1=2048, bd2=2048
btc=128 (inferred), bfc=256, bd1c=512, bd2c=2048
```
