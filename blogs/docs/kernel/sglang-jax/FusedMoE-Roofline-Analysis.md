# Fused MoE 内核 Roofline 分析（TPU v7，基于 sglang-jax v1 kernel）

> **分析对象**：`python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py`（Fused EP MoE Pallas 内核）
> **硬件平台**：TPU v7 (Ironwood)
> **基础 Roofline 参考**：`blogs/docs/kernel/tpu-inference/FusedMoE-Roofline-Analysis.md` §1-§5

## 1. 概述

本文档分析 sglang-jax 版本的 Fused MoE 内核在 TPU v7 上的 Roofline 特征。由于硬件规格和模型配置与 tpu-inference 版本完全相同，理论 Roofline 上界不变。本文聚焦于 sglang-jax 内核的优化如何缩小实际性能与理论上界之间的差距。

完整的 Roofline 理论推导（硬件规格、临界 AI、单 expert 计算量、不同 BS 下的算术强度等）请参考 tpu-inference 版本的 Roofline 分析文档。

## 2. Roofline 关键数字回顾

来自 tpu-inference Roofline 分析（不重复推导）：

| 项目 | 数值 |
|------|------|
| TPU v7 BF16 峰值算力 | 2307 TFLOPS |
| TPU v7 FP8 峰值算力 | 4614 TFLOPS |
| TPU v7 HBM 带宽 | 7380 GB/s (7.38 TB/s) |
| 临界 AI（BF16 计算 vs HBM） | 312.6 FLOP/Byte |
| 临界 AI（FP8 计算 vs HBM） | 625.2 FLOP/Byte |
| BS=512 时 T_avg=16, AI=33.6 (FP8 权重) | **深度带宽瓶颈** |
| BS=1024 时 T_avg=32, AI=67.1 | **仍然带宽瓶颈** |
| Shared expert BS=512, AI=1056 | **算力瓶颈** |
| BF16 计算 + FP8 权重 compute-bound 点 | BS ≈ 4768 |
| 单 expert FP8 权重量 | 48 MB |
| 单 expert 计算量 | 100.7M × T FLOPs |

**部署配置**：16 设备 (chips)，EP=32，DP=4，TP=8，每 core 8 个 expert

> 以上数字对 tpu-inference 和 sglang-jax 内核完全相同。差异在于内核实现的优化，影响实际能达到理论值的百分比。

## 3. HBM Staging 开销分析

sglang-jax 引入了 bts token staging（HBM→VMEM 双缓冲），这是与 tpu-inference 最大的架构差异。tpu-inference 将所有 expert token 保持在 VMEM 中（无 staging 开销），而 sglang-jax 将 token 存放在 HBM，按 bts 大小分块加载。

### 3.1 额外 HBM 带宽消耗

以 Ring-1T FP8 (EP=32, hidden=8192, intermediate=2048, 8 experts/core) 为例：

**FFN1 Token 读取**：每个 expert 的每个 bd1 slice，需要从 HBM 加载 bts 个 token 到 VMEM
```
每次加载: bts × hidden_size × 2B (bf16 激活)
每 expert 总加载: T_expert × hidden_size × 2B × num_bd1 (= hidden/bd1 次)

BS=512, T_avg=16:
  num_bd1 = 8192/bd1 (假设 bd1=2048, 则 num_bd1=4)
  每 expert FFN1 staging: 16 × 8192 × 2 × 4 = 1.0 MB
```

**FFN2 结果读写**：三缓冲中每个 tile 需要 load + store
```
每次读写: bts × hidden_size × 2B × 2 (读+写)
每 expert 总读写: T_expert × hidden_size × 2B × 2 × num_bd2 × num_bf

BS=512, T_avg=16:
  num_bd2 = 8192/bd2 (假设 bd2=2048, 则 num_bd2=4)
  num_bf = 2048/bf (假设 bf=512, 则 num_bf=4)
  每 expert FFN2 staging: 16 × 8192 × 2 × 2 × 4 × 4 = 16 MB (理论最大)
  实际当 should_init_ffn2=True（第一个 bf slice）跳过 load，减少 ~25%
```

### 3.2 与权重加载的对比

| 项目 | 每 expert 数据量 | 占 HBM 带宽比例 |
|------|-----------------|----------------|
| 权重加载（FP8） | 48 MB | **基准** |
| FFN1 token staging | ~1 MB (BS=512) | ~2% |
| FFN2 result staging | ~12 MB (BS=512) | ~25% |
| **Staging 总开销** | **~13 MB** | **~27%** |

### 3.3 Trade-off 分析

**收益**：
- VMEM 限制从 100 MB 降到 64 MB，减少 VMEM 压力
- bts 可以 > bt，支持更大的 GEMM M-tile，提高 MXU 利用率
- 更大 token count 不受 VMEM 容量限制

**代价**：
- 额外 ~27% 的 HBM 带宽开销
- 在已经带宽瓶颈的 decode 场景下，staging 增加了 HBM 竞争

**结论**：staging 开销在 decode 场景下是可接受的，因为权重加载（48 MB/expert）仍然是 HBM 带宽的主要消耗者。staging 的 13 MB 增量不改变瓶颈本质。在 prefill 场景下（token 数多），staging 可以实现更大的 GEMM tile，收益更明显。

## 4. 逐优化 Roofline 影响分析

### 4.1 Recursive-Doubling Allgather

| | tpu-inference | sglang-jax |
|---|---|---|
| 算法 | Ring O(N) | Recursive-doubling O(log2(N)) |
| EP=32 轮数 | 31 轮 | 5 轮 |
| EP=64 轮数 | 63 轮 | 6 轮 |

Metadata allgather 在 tpu-inference 中约 53 us（128 tokens, EP=32）。降低到 5 轮可节省大部分 barrier 同步开销，但绝对值较小（整个路由阶段 ~98 us），相对 FFN 耗时（~0.05-0.2 ms/expert）是次要的。

对于更大的 EP（64/128），收益更显著：63→6 轮 或 127→7 轮。

### 4.2 Skip-Expert 优化

- tpu-inference：dynamic loop bounds (cdiv(dyn_sz, btc))，空 expert 循环 0 次但仍进入 FFN 函数体
- sglang-jax：lax.cond(has_tokens) 跳过整个 FFN 函数体，包括 weight fetch 初始化、loop setup 等

**影响**：
- BS=512, 256 experts, top_k=8, EP=32 → 每 core 8 experts, T_avg=16
- 实际分布不均匀：部分 expert 得到 0 token
- 跳过空 expert 避免了权重预取启动、循环初始化等固定开销
- 在 hotspot 分布下（少数 expert 集中大量 token），多数 expert 为空，收益更大

### 4.3 Next-Expert 权重预取

**机制**：FFN2 最后一个 tile 计算期间，预取下一 expert 的 W1/W3 (L1886-1897)

**Roofline 影响**：
```
不预取: weight_load(E0) → FFN(E0) → weight_load(E1) → FFN(E1) → ...
                                   ↑ 等待权重加载 ↑
预取:   weight_load(E0) → FFN(E0) → FFN(E1) → FFN(E2) → ...
                           ↑ E1 权重预取 ↑ E2 权重预取 ↑
```

在带宽瓶颈下（weight load 6.5 us vs compute 0.7 us @BS=512/FP8），预取不能完全隐藏权重加载（因为 FFN 计算太短），但可以部分重叠，减少等待时间。

理论最优：8 个 expert 的总耗时从 8 × (6.5+0.7) = 57.6 us 降到 ~6.5 + 8 × 0.7 + 7 × max(0, 6.5-0.7) = ~52 us（仍然受限于权重加载）。

### 4.4 W2 Early Prefetch

FFN1 最后一个 bd1 slice 完成时开始预取 W2 (L1646-1648)，而非等到 FFN2 开始才取。

**影响**：隐藏 W2 首个 tile 的加载延迟。W2 (down) 权重 FP8 = 16 MB/expert，预取一个 bd2 slice ≈ 16/num_bd2 MB，在 FFN1 最后一个 token tile 计算期间完成加载。

### 4.5 Token HBM Staging (bts)

bts 参数允许每次加载更多 token 到 VMEM 中的计算缓冲：

- tpu-inference: GEMM M-tile = btc（受限于 VMEM 中的全部 expert token）
- sglang-jax: bts 可以 > bt，最大到 bt × ep_size

更大的 GEMM M-tile 提高 MXU 利用率。对于极小的 matmul（[4, 2048] × [2048, 1024]），MXU 利用率很低。增大 M-tile 可以部分缓解这一问题。

但在 decode 场景（T_avg=16），即使 bts = bt × ep_size，M-tile 仍然很小，MXU 利用率提升有限。Prefill 场景收益更大。

### 4.6 FFN2 三缓冲

**三缓冲 vs 双缓冲**：
```
双缓冲: load(N) → compute(N) → store(N) → load(N+1) → compute(N+1) → store(N+1)
                               ↑ 等待 store ↑

三缓冲: load(N+1) / compute(N) / store(N-1)  ← 三者同时进行
```

消除了 store-wait stall，每个 FFN2 tile 的有效计算时间 = max(load, compute, store)，而非 load + compute + store。

在带宽瓶颈下，load/store 是主要耗时，但三缓冲允许 compute 与 load/store 完全重叠。

### 4.7 Shared Expert 穿插

- tpu-inference: shared expert 在 roofline 分析中提到但未在内核中融合
- sglang-jax: SE 计算穿插在 routed expert 的通信等待间隙中

SE 是 compute-bound（AI=1056 @BS=512，远超临界值 312.6）。将 compute-bound 的 SE 计算插入到 bandwidth-bound 的 A2A 等待间隙中，有效利用了原本空闲的 MXU 算力。

**每个 expert 的穿插**: 2 个 SE block（scatter 后 + gather 后），加上 expert 0 前的 1 个 block
**总 SE blocks**: cdiv(se_inter_size, bse) = cdiv(2048, bse)
- bse=256: 8 blocks，可以在 8 experts × 2 + 1 = 17 个间隙中执行完
- bse=512: 4 blocks，只需 4 个间隙

**理论影响**：如果 SE 计算完全隐藏在通信间隙中，MoE 层总耗时 = routed expert 耗时（无额外 SE 开销）。

### 4.8 fori_loop Scatter + pl.when 条件 DMA

- fori_loop 代替 Python 展开：减少编译后的 MLIR/HLO 代码体积，降低 instruction cache 压力
- @pl.when 条件 DMA：避免发起零大小的 DMA 操作
- 对运行时性能影响较小，主要改善编译时间和大 bt 配置下的可行性

## 5. 优化优先级（Roofline 视角）

基于对 roofline 差距的影响大小排序：

1. **权重预取（next-expert + W2 early）** -- 直接缩小 HBM 带宽瓶颈的流水线气泡 -- 已实现
2. **FP8 量化** -- 权重减半，带宽瓶颈下近似 2x 加速 -- 已实现
3. **Shared expert 穿插** -- 将 compute-bound 计算填入通信间隙 -- 已实现
4. **HBM staging (bts)** -- 降低 VMEM 压力，支持更大 GEMM tile -- 已实现
5. **FFN2 三缓冲** -- 消除 store-wait stall -- 已实现
6. **Skip-expert** -- 减少空 expert 的固定开销 -- 已实现
7. **Block size 调优** -- 10 个参数（含 bts, bse）精细调整 -- 已实现

## 6. A2A 通信分析

A2A 通信特征与 tpu-inference 版本相同（参考 tpu-inference Roofline §7.3）：

- 实测 ICI 吞吐：~40 GB/s（4096 tokens, EP=32）
- 内核中 scatter/gather 与 FFN 计算重叠执行
- DP+EP 拓扑：EP=32 全局，DP=4 拆分 attention batch → 每设备 local tokens 减少 → A2A 通信量下降

sglang-jax 的 recursive-doubling 优化影响的是 **metadata allgather**（expert sizes/starts），而非数据 A2A。数据 A2A 的 scatter/gather pattern 两版本一致。

## 7. 总结

### 逐阶段瓶颈与优化状态

| 阶段 | 瓶颈 | AI (BS=512, FP8) | sglang-jax 优化 |
|------|------|-------------------|-----------------|
| 路由 | Barrier 延迟 | N/A | Recursive-doubling (31→5 轮) |
| A2A Scatter | ICI 带宽 | N/A | fori_loop + pl.when 条件 DMA |
| FFN1 (gate+up) | HBM 带宽 | ~33.6 | bts staging + W2 early prefetch |
| FFN2 (down) | HBM 带宽 | ~33.6 | 三缓冲 + next-expert prefetch |
| A2A Gather | ICI 带宽 | N/A | 动态 per-expert 等待 |
| Shared Expert | 算力 | ~1056 | 穿插在通信间隙 |
| 累加 | HBM 带宽 | ~2 | 双缓冲 acc pipeline |
| 输出写回 | HBM 带宽 | N/A | 双缓冲 + valid guard |

### 核心结论

1. **理论 Roofline 不变**：sglang-jax 和 tpu-inference 面对相同的硬件和模型，理论性能上界完全一致。FFN 在 BS=512/1024 下仍然是 HBM 带宽瓶颈（AI=33.6/67.1 vs 临界值 312.6）。

2. **sglang-jax 缩小了 Roofline 差距**：通过 7 项主要优化，减少了理论上界与实际性能之间的 overhead：
   - 权重预取：减少流水线气泡
   - 三缓冲：消除 store stall
   - Skip-expert：减少空 expert 开销
   - Shared expert 穿插：利用通信间隙
   - Recursive-doubling：减少 metadata 轮数

3. **HBM staging 是架构权衡**：增加 ~27% 的 HBM staging 开销，换取 VMEM 压力降低（100MB→64MB）和更大 GEMM tile 支持。在 decode 场景下开销可接受；prefill 场景收益更大。

4. **基本瓶颈未改变**：权重加载仍是 decode 场景的主要瓶颈。所有优化都在"围绕瓶颈减少 overhead"，而非消除瓶颈本身。真正消除带宽瓶颈需要 BS ≈ 4768（BF16 计算 + FP8 权重）。
