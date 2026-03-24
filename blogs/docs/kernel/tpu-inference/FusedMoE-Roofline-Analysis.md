# Fused MoE 内核 Roofline 分析（TPU v7，基于 tpu-inference v1 kernel）

> **分析对象**：`blogs/tpu-inference/fused_moe/v1/kernel.py`（Fused EP MoE Pallas 内核）
> **硬件平台**：TPU v7 (Ironwood)

## 1. 什么是 Roofline 分析

Roofline 模型用于判断一个计算任务的性能瓶颈在哪里：**算力不够**（compute-bound）还是**数据搬运不够快**（memory/bandwidth-bound）。

核心概念：

```
算术强度 (AI) = 计算量 (FLOPs) / 数据搬运量 (Bytes)
临界算术强度 = 硬件峰值算力 / 硬件带宽
```

- 当 AI < 临界值 → **带宽瓶颈**：MXU 在等数据，算力空转
- 当 AI > 临界值 → **算力瓶颈**：数据供给够了，但算不过来

对于 MoE 内核来说，这帮助我们判断：每个阶段（FFN、通信、路由）到底被什么资源卡住了，从而指导优化方向。

## 2. TPU v7 (Ironwood) 硬件规格

来源：[Google Cloud TPU v7 官方规格](https://cloud.google.com/tpu/docs/v7)

| 资源 | 数值 | 备注 |
|------|------|------|
| BF16 峰值算力 | **2307 TFLOPS** | 单芯片（2 个 TensorCore） |
| FP8 峰值算力 | **4614 TFLOPS** | 单芯片 |
| HBM 带宽 | **7380 GB/s (7.38 TB/s)** | 单芯片 |
| HBM 容量 | 192 GiB | 单芯片 |
| ICI 带宽（双向总计） | 1200 GBps | 单芯片双向总计 |
| TensorCore 数 | 2 | 单芯片 |
| SparseCore 数 | 4 | 单芯片 |

ICI 实测带宽（来自 `Collectives_Microbenchmark_Summary.md`）：

| 轴 | 理论带宽 (GB/s) | 实测 allgather (GB/s) | 利用率 |
|----|-----------------|----------------------|--------|
| x | 100 | 86.1 | 86% |
| y | 100 | 91.8 | 92% |
| z | 100 | 90.4 | 90% |
| core_on_chip | 600 | 311.4 | 52% |

**临界算术强度**：

| 场景 | 公式 | 临界 AI (FLOP/Byte) |
|------|------|---------------------|
| BF16 计算 vs HBM | 2307T / 7.38T | **312.6** |
| FP8 计算 vs HBM | 4614T / 7.38T | **625.2** |
| BF16 计算 vs ICI (x/y/z 轴) | 2307T / 90G | **25,633** |

> 对比前代 v6e (Trillium)：BF16 峰值 918 TFLOPS，HBM 带宽 1638 GB/s，临界 AI = 560.7 FLOP/Byte。
> TPU v7 的 HBM 带宽提升 4.5x，算力提升 2.5x，所以**临界 AI 从 560 降到 313**，更容易达到 compute-bound。

## 3. 模型配置（BailingMoeV2 / Ring-1T）

来自实际模型 `config.json`：

| 参数 | 值 | 说明 |
|------|-----|------|
| `hidden_size` | 8192 | |
| `moe_intermediate_size` | 2048 | 每个 routed expert 的中间维度 |
| `intermediate_size` | 18432 | dense 层（前 4 层）的中间维度 |
| `num_experts` | 256 | |
| `num_experts_per_tok` (top_k) | 8 | |
| `num_shared_experts` | 1 | 所有 token 都经过的 shared expert |
| `first_k_dense_replace` | 4 | 前 4 层是 dense FFN，不走 MoE |
| `num_hidden_layers` | 80 | 共 80 层，其中 76 层是 MoE |
| `score_function` | sigmoid | 路由评分函数 |
| `n_group` | 8, `topk_group` = 4 | grouped top-k 路由 |
| `routed_scaling_factor` | 2.5 | |
| 量化 | FP8 W8A8 | 权重 FP8 静态 per-channel，激活 FP8 动态 per-token |
| `dtype` | bf16 | |
| **FP8 部署** | **16 设备 (chips)** | **每设备 2 个 TensorCore → 32 cores → EP size 32** |

## 4. 单 Expert Roofline 基础计算

### 4.1 权重大小

| | BF16 (2B/元素) | FP8 (1B/元素) |
|---|----------------|---------------|
| w1 (gate+up): 2 × 8192 × 2048 | 64 MB | **32 MB** |
| w2 (down): 2048 × 8192 | 32 MB | **16 MB** |
| **单 expert 总计** | **96 MB** | **48 MB** |

### 4.2 计算量

```
FFN1 (gate + up): 2 × 2 × T × 8192 × 2048 = 67,108,864 × T
FFN2 (down):      2 × T × 2048 × 8192     = 33,554,432 × T
单 expert 总计:  100,663,296 × T ≈ 100.7M × T FLOPs
```

### 4.3 不同 T_expert 下的算术强度

> **T_avg = BS × top_k / num_experts = BS × 8 / 256 = BS / 32**

| T_expert | 对应 BS | FLOPs (GFLOP) | FP8 权重 (MB) | AI (FP8) | BF16 权重 (MB) | AI (BF16) | 瓶颈 (BF16 计算) |
|----------|---------|---------------|---------------|----------|----------------|-----------|-------------------|
| 1 | 32 | 0.10 | 48.0 | 2.1 | 96.0 | 1.0 | **带宽** |
| 4 | 128 | 0.40 | 48.0 | 8.4 | 96.1 | 4.2 | **带宽** |
| **16** | **512** | **1.61** | **48.0** | **33.6** | **96.2** | **16.8** | **带宽** |
| **32** | **1024** | **3.22** | **48.1** | **67.1** | **96.3** | **33.6** | **带宽** |
| 64 | 2048 | 6.44 | 48.1 | 134 | 96.5 | 67 | **带宽** |
| 128 | 4096 | 12.89 | 48.2 | 267 | 96.8 | 134 | **带宽** |
| **~149** | **~4768** | — | — | **~313** | — | — | **BF16 平衡点 (FP8 权重)** |
| 256 | 8192 | 25.77 | 48.4 | 532 | 97.2 | 266 | **带宽** |
| **~298** | **~9536** | — | — | **~625** | — | — | **FP8 平衡点 (FP8 权重+计算)** |
| **~301** | **~9632** | — | — | — | — | **~313** | **BF16 平衡点 (BF16 权重)** |
| 512 | 16384 | 51.54 | 48.8 | 1056 | 97.6 | 528 | 算力 |

**关键结论**：
- **BS=512**（T_avg=16）：算术强度 33.6 (FP8权重)，远低于临界值 313 → **深度带宽瓶颈**
- **BS=1024**（T_avg=32）：算术强度 67.1 → **仍然深度带宽瓶颈**
- BF16 计算 + FP8 权重：需要 T_expert ≈ 149（对应 **BS ≈ 4768**）达到 compute-bound
- FP8 计算 + FP8 权重：需要 T_expert ≈ 298（对应 **BS ≈ 9536**）达到 compute-bound
- BF16 计算 + BF16 权重：需要 T_expert ≈ 301（对应 **BS ≈ 9632**）达到 compute-bound

> 对比 v6e：临界 AI=561，同样条件需要 T_expert ≈ 267。**v7 由于 HBM 带宽提升更大，更容易 compute-bound**（临界 AI 从 561 降到 313），这意味着权重加载瓶颈相对减轻了。

**直觉理解**：每个 expert 的 FP8 权重 48 MB，以 7.38 TB/s 只需 **6.5 µs** 就能加载完。但 BS=512 时每个 expert 只算 16 个 token 的矩阵乘法（1.61 GFLOP），以 2307 TFLOPS 也只需 **0.7 µs**。两者比例 = 6.5/0.7 ≈ 9.3x，权重搬运仍然是主要瓶颈，但差距比 v6e 上小很多。

## 5. 峰值吞吐场景分析（BS=512/1024）

### 5.1 不同 EP size 下的整层 MoE 耗时

> **EP size 与设备数的关系**：TPU v7 每个设备（chip）有 2 个 TensorCore（core），每个 core 作为一个独立的 EP 计算单元。因此 EP size = 设备数 × 2。

| 设备数 (chips) | Core 数 | EP size | 每 core expert 数 |
|---------------|---------|---------|-------------------|
| 4 | 8 | 8 | 32 |
| 8 | 16 | 16 | 16 |
| **16** | **32** | **32** | **8** |
| 32 | 64 | 64 | 4 |

#### FP8 权重（1B/元素），BF16 计算

| 设备数 | EP size | 每 core expert 数 | 每 core 权重量 | HBM 耗时/core | 计算耗时/core (BS=512) | 计算耗时/core (BS=1024) |
|--------|---------|-------------------|---------------|--------------|----------------------|------------------------|
| 4 | 8 | 32 | 1.54 GB | **0.208 ms** | 0.022 ms | 0.044 ms |
| 8 | 16 | 16 | 768 MB | **0.104 ms** | 0.011 ms | 0.022 ms |
| **16** | **32** | **8** | **384 MB** | **0.052 ms** | **0.006 ms** | **0.011 ms** |
| 32 | 64 | 4 | 192 MB | **0.026 ms** | 0.003 ms | 0.006 ms |

#### BF16 权重（2B/元素），BF16 计算

| 设备数 | EP size | 每 core expert 数 | 每 core 权重量 | HBM 耗时/core | 计算耗时/core (BS=512) | 计算耗时/core (BS=1024) |
|--------|---------|-------------------|---------------|--------------|----------------------|------------------------|
| 4 | 8 | 32 | 3.07 GB | **0.416 ms** | 0.022 ms | 0.044 ms |
| 8 | 16 | 16 | 1.54 GB | **0.208 ms** | 0.011 ms | 0.022 ms |
| **16** | **32** | **8** | **768 MB** | **0.104 ms** | **0.006 ms** | **0.011 ms** |
| 32 | 64 | 4 | 384 MB | **0.052 ms** | 0.003 ms | 0.006 ms |

> 注：HBM 耗时 = 每 core 权重量 / (7.38 TB/s ÷ 2)（单 core 占单设备一半 HBM 带宽，即 3.69 TB/s）。计算耗时 = 总 FLOPs / (2307 TFLOPS ÷ 2)（单 core 算力 1153.5 TFLOPS）。实际耗时取两者较大值（当前均为 HBM 主导）。临界 AI 不变：312.6 FLOP/Byte（算力和带宽同比缩放）。

### 5.2 关键发现

#### v7 的高 HBM 带宽显著降低了权重加载耗时

对比 v6e（HBM 1.64 TB/s）：

| 配置 | 设备数 | EP size | v6e HBM 耗时/core | v7 HBM 耗时/core | 加速比 |
|------|--------|---------|-------------------|-------------------|--------|
| FP8, 8 experts/core | 16 | 32 | 0.234 ms | **0.052 ms** | **4.5x** |
| FP8, 16 experts/core | 8 | 16 | 0.468 ms | **0.104 ms** | **4.5x** |

v7 的 HBM 带宽提升 4.5x 直接将权重加载耗时降到 v6e 的 1/4.5。

#### FP8 的价值仍然在带宽

在 decode 场景下仍是带宽瓶颈，FP8 权重减半搬运量：

```
同一 EP size 下：FP8 vs BF16 加速比 ≈ 2x（纯带宽收益）
```

#### BS 翻倍几乎不增加耗时

从 BS=512 到 BS=1024：
- 权重加载量不变（每个 expert 的权重总要加载一遍）
- 计算量翻倍，但计算不是瓶颈

```
BS=512 → BS=1024: MoE 耗时几乎不变，吞吐接近翻倍
```

这是 **带宽瓶颈的红利** — 增加 batch size 近似"免费"。

#### EP size 越大越快（但通信开销增加）

EP 增大 → 每设备加载更少 expert 权重 → HBM 耗时线性降低。但 EP 增大也增加 all-to-all 通信开销（更多跨设备传输、更多 ICI 跳数）。

以 BS=512, FP8 权重, DP=4 为例：

| 设备数 | EP size | DP | 权重耗时/core | A2A 估算（*） | 总计估算 |
|--------|---------|-----|--------------|-------------|----------|
| 4 | 8 | 4 | 0.208 ms | ~0.05 ms | ~0.26 ms |
| 8 | 16 | 4 | 0.104 ms | ~0.08 ms | ~0.18 ms |
| **16** | **32** | **4** | **0.052 ms** | **~0.15 ms** | **~0.20 ms** |
| 32 | 64 | 4 | 0.026 ms | ~0.25 ms | ~0.28 ms |

> （*）A2A 估算考虑了 DP=4 后每 core local tokens 减少（BS/32=16 → 实际发送量约 512 KB/core），基于纯 A2A 微基准（40 GB/s @4096 tokens, EP=32）的吞吐按比例缩放。实际耗时受消息大小、ICI 拓扑等因素影响，仍需逐配置实测校准。

在 v7 上，由于 HBM 带宽极高，**权重加载耗时已经与 A2A 通信耗时在同一量级**。DP 的引入进一步降低了每设备的 A2A 通信量，使得权重加载重新成为更主要的瓶颈。但随着设备数增加（EP 增大），A2A 跳数和延迟增长仍是制约因素。

### 5.3 打满算力需要多大 BS？

以 BF16 计算 + FP8 权重为例（临界 AI = 312.6 FLOP/Byte）：

```
所需 T_expert = 312.6 × 48 MB / 100.7M ≈ 149 tokens/expert
所需 BS = 149 × 256 / 8 = 4,768
```

以 FP8 计算 + FP8 权重为例（临界 AI = 625.2 FLOP/Byte）：

```
所需 T_expert = 625.2 × 48 MB / 100.7M ≈ 298 tokens/expert
所需 BS = 298 × 256 / 8 = 9,536
```

实际场景：
- **Decode**: BS=512–1024，带宽瓶颈（T_avg=16–32）
- **Prefill**: chunked_prefill_size=8192–16384 时，T_avg=256–512，可以接近甚至超过 BF16 计算的 compute-bound 点

## 6. Shared Expert 分析

模型有 1 个 shared expert，所有 token 都经过（不走路由），相当于一个 dense FFN：

```
Shape: [BS, 8192] @ [8192, 2048] (gate+up) + [BS, 2048] @ [2048, 8192] (down)
FLOPs = 100.7M × BS（与单个 routed expert 相同，但 BS 个 token 全部参与）
权重量 = 48 MB (FP8) / 96 MB (BF16)
```

| BS | FLOPs (GFLOP) | AI (FP8 权重) | AI (BF16 权重) | 瓶颈 (BF16 计算, 临界 AI=313) |
|----|---------------|---------------|----------------|-------------------------------|
| 512 | 51.5 | 1056 | 528 | **算力** (两种权重都 compute-bound) |
| 1024 | 103.1 | 2074 | 1037 | **算力** |

**Shared expert 在 BS=512 时已经是 compute-bound**（AI=1056 >> 临界值 313），与 routed expert 截然不同。这是因为它处理全部 BS 个 token，而非 T_avg=16 个。

在内核实现中，shared expert 计算被切分到 routed expert 的流水线间隙中执行（与 scatter 等待时间重叠），不额外增加延迟。

## 7. 逐阶段 Roofline 分析

### 7.1 FFN1 — Gate + Up 投影

**操作**：`token_block @ w1_block`（gate）+ `token_block @ w3_block`（up），逐 expert 计算

**单 expert matmul shape**：`[T_expert, 8192] @ [8192, 2048]` × 2

见第 4.3 节算术强度表。在 BS=512/1024 下，AI=33.6/67.1（FP8 权重），远低于 313 临界值。

### 7.2 FFN2 — Down 投影

**操作**：`activation(acc1, acc3) @ w2_block`

**单 expert matmul shape**：`[T_expert, 2048] @ [2048, 8192]`

FLOPs 为 FFN1 的一半，权重量也是一半，算术强度特征一致。

### 7.3 All-to-All Scatter/Gather

**操作**：通过 ICI DMA 将 token 从源设备发送到 expert 所在设备，计算后发回。

#### 通信量计算

每个 core 的 local tokens 需要发送到被选中 expert 所在的 core。每个 token 选 top_k=8 个 expert，其中大部分在远程 core 上：

```
每 token scatter 数据量 = top_k × hidden_size × 2B (bf16 激活)
                       = 8 × 8192 × 2 = 131,072 B ≈ 128 KB

每 core 的 scatter 总量 = local_tokens × 128 KB
```

其中 `local_tokens` 取决于 DP 配置（见下文 DP+EP 分析）。

#### A2A 估算数据来源

现有 A2A 耗时基于两个测量点：

1. **纯 A2A 微基准**（`FusedMoE_benchmark.md`）：独立的 `_remote_copy_kernel` 隔离通信耗时
   - 配置：tokens=4096, EP=32, experts=256, hidden=8192, top_k=8, bf16
   - 结果：**0.387 ms**，有效吞吐 **~40 GB/s**

2. **内核 Ablation**（`FusedMoE_benchmark.md`）：仅开启 `a2a` 阶段，与全关基线对比
   - 128 tokens, EP=32：A2A 贡献 **0.25–0.30 ms**（随负载分布变化）
   - 4096 tokens, EP=32：A2A 贡献 **1.14–3.04 ms**

> 注：第 5 节中不同 EP size 的 A2A 估算为基于上述数据点的粗略插值，缺乏严格推导。实际 A2A 耗时受 ICI 拓扑、消息大小、路由模式等因素影响，需要逐 EP size 的实测数据来校准。

#### DP+EP 下的通信量变化

实际部署采用 **DP Attention + EP MoE** 策略：

- **Attention 阶段**：DP=N，N 个 DP 组各自独立做 TP Attention
- **MoE 阶段**：EP=32，全部 16 设备（32 cores）组成一个 EP 组

DP 对 A2A 通信的影响：每个 DP 组只处理 **BS/DP** 个 token，这些 token 从该 DP 组的设备发出、经 EP all-to-all 路由到全部 32 cores 上的 expert。因此**每个设备的 local tokens 减少，A2A scatter 通信量相应降低**。

以 16 设备（32 cores），EP=32，DP=4，TP=8 为例：

| 配置 | 每 DP 组 tokens | 每 core local tokens | 每 core scatter 量 |
|------|----------------|---------------------|-------------------|
| BS=512, DP=4 | 128 | ~4 | ~512 KB |
| BS=512, DP=1 | 512 | ~16 | ~2 MB |
| BS=1024, DP=4 | 256 | ~8 | ~1 MB |

> 注：每 core local tokens ≈ (BS/DP) / (TP 组内 core 数)。DP=4/TP=8 时，每 DP 组 8 cores，每 core 约 (BS/4)/8 = BS/32 个 token。

在内核中 scatter/gather 与 FFN 计算重叠执行，有效隐藏了大部分通信延迟。DP 进一步降低每设备的通信量，使得 overlap 更容易覆盖 A2A 延迟。

### 7.4 路由（Sigmoid + Top-K + Ring All-Reduce）

计算量相对 FFN 可忽略不计。

**实测**（来自 `FusedMoE_benchmark.md` ablation，128 tokens）：
- all_reduce_metadata ~53 µs
- top_k ~6 µs
- sync_barrier ~39 µs
- 总计 ~98 µs，为固定开销

### 7.5 累加（Weighted Sum）

```
FLOPs = bt × top_k × hidden_size × 2
      = bt × 8 × 8192 × 2 = bt × 131,072
```

纯带宽瓶颈（AI ≈ 2 FLOP/Byte），但数据量小，耗时占比低。

## 8. 优化效果分析

### Benchmark Ablation 数据（128 tokens, sparse_hotspot）

来自 `FusedMoE_benchmark.md`（注：该 benchmark 在旧模型配置 hidden=8192, intermediate=2048, experts=256 上测试，EP=32）：

| 优化项 | 机制 | 影响 (ms) | Roofline 关联 |
|--------|------|-----------|---------------|
| All-to-all | 异步 DMA scatter/gather | 0.256–0.297 | 通信与计算重叠；减轻 ICI 瓶颈 |
| Dynamic FFN1 | 跳过空 expert 的计算 | 0.320–1.689 | 减少浪费的 FLOPs；不均衡越大影响越大 |
| Dynamic FFN2 | 跳过空 expert 的计算 | 0.273–1.004 | 同 FFN1 |
| 权重预取 | 权重 DMA 与前一轮 FFN 重叠 | 0.465–0.489 | 直接缓解主要瓶颈（HBM 带宽） |
| A2A scatter tile 读取 | scatter 时分块读取 token | 0.249–0.479 | 提高 DMA 效率 |
| A2A scatter acc tile 写入 | gather 时分块写入结果 | 0.242–0.353 | 提高 DMA 效率 |
| Shared expert 切分 | 按 hidden_size 维度切分 shared expert | 0.271–0.288 | 更好地利用 VMEM |

### 优化优先级（Roofline 视角）

1. **权重预取重叠** — 直接缓解主要的 HBM 带宽瓶颈。当前权重获取在每个 expert 开始时才启动（`run_per_expert` 中的 `start_fetch_bw1`）。三缓冲可以在当前 expert 的 FFN2 期间预取下一个 expert 的权重。

2. **FP8 量化** — 权重从 2B 降到 1B，HBM 搬运量减半。在带宽瓶颈下等效于 2x 加速。

3. **Shared expert 按 hidden_size 切分** — 将 shared expert 计算拆分到其他操作的间隙中运行，提高流水线利用率。

4. **FFN2 到 FFN1 权重预取** — 在 FFN2 计算期间预取下一个 bf_id 的 FFN1 权重。减少权重加载阻塞。

5. **Block size 调优** — 更大的 `bd1/bd2/bf` 减少迭代次数、提高权重复用，但受 VMEM 容量限制（100 MB 上限）。

## 9. Roofline 差距分析

### 理论 vs 实测（来自 `FusedMoE_benchmark.md`，128 tokens）

**仅 FFN**（隔离 dynamic_ffn1 单独开启的测量）:
- 实测：0.320 ms - 0.223 ms（全关）= **0.097 ms**
- 理论 FLOPs：2.147 GFLOPs
- 实测吞吐：**22.1 TFLOPS**
- 独立 matmul benchmark：**25.5 TFLOPS**
- 利用率：22.1 / 25.5 = **86.7%**（相对独立 matmul），**0.96%**（相对 v7 BF16 峰值 2307 TFLOPS）

独立 matmul 与峰值的差距是预期内的 — 这些是无法打满 MXU 的微小矩阵乘法（[4, 2048] × [2048, 1024]）。内核 matmul 与独立 matmul 的差距（~13%）来自内核开销（循环控制、DMA 同步、打包/解包）。

> 注：上述 benchmark 数据来自 v6e 或早期 v7 测试环境，峰值参考值可能不同。v7 正式环境的实测数据待补充。

### 缩小差距的方向

1. **增加 T_expert** — 更大的 batch size 推高算术强度（但 decode 受限）
2. **更大的计算 tile**（btc, bfc, bd1c, bd2c） — 更大的 matmul shape 提高 MXU 利用率
3. **减少权重重复加载** — 跨 bt 块缓存活跃 expert 权重（受 VMEM 限制）

## 10. 总结

### 逐阶段瓶颈

| 阶段 | 瓶颈 | 算术强度（BS=512, FP8 权重） | 临界 AI (BF16 计算) | 优化手段 |
|------|------|------------------------------|---------------------|----------|
| 路由 | Barrier 延迟 | N/A | — | ~98 µs 固定开销 |
| A2A Scatter | ICI 带宽 | N/A（纯通信） | — | 与 FFN 重叠；DP 降低每设备通信量 |
| FFN1（gate+up） | HBM 带宽（权重） | ~33.6 FLOP/B | 312.6 | 权重预取、FP8、更大 block |
| FFN2（down） | HBM 带宽（权重） | ~33.6 FLOP/B | 312.6 | 权重预取、FP8、更大 block |
| A2A Gather | ICI 带宽 | N/A（纯通信） | — | 与 FFN 重叠；DP 降低每设备通信量 |
| Shared Expert | **算力瓶颈** | ~1056 FLOP/B | 312.6 | 切分到流水线间隙 |
| 累加 | HBM 带宽 | ~2 FLOP/B | 312.6 | 相对耗时小 |
| 输出写回 | HBM 带宽 | N/A（纯写入） | — | 双缓冲 |

### 核心结论

1. **Fused MoE 内核在 BS=512/1024 的 decode 场景下仍受限于 HBM 带宽**，但 v7 的 7.38 TB/s HBM 带宽大幅缓解了这一瓶颈（相比 v6e 加速 4.5x）。

2. **FP8 的主要价值在于减半权重搬运量**（带宽收益），在当前瓶颈下可带来接近 2x 的 MoE 层加速。

3. **增加 BS 近似"免费"**：BS=512 → 1024 时 MoE 耗时几乎不变，吞吐接近翻倍。

4. **BF16 计算 + FP8 权重 → BS ≈ 4768 达到 compute-bound**；FP8 计算 + FP8 权重 → BS ≈ 9536。v7 由于高 HBM 带宽，比 v6e 更容易触达 compute-bound。

5. **DP+EP 协同降低 A2A 开销**：实际部署采用 DP Attention + EP MoE（EP=32 全局，DP=4 拆分 attention batch）。DP 使每设备的 local tokens 减少，降低 A2A scatter/gather 通信量，有利于通信与计算的 overlap 覆盖 A2A 延迟。

6. **v7 上权重加载耗时已与 A2A 通信在同一量级**（16 设备/EP=32/DP=4 时，权重 0.052 ms/core vs A2A ~0.15 ms）。DP 的引入缓解了 A2A 压力，使权重加载重新成为更主要的瓶颈。
