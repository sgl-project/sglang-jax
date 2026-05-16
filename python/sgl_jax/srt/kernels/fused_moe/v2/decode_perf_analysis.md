# V2 Decode Performance Analysis — Ablation + LLO Dump

**日期：** 2026-05-17
**配置：** MiMo V2 Pro, E=384, H=6144, I=2048, top_k=8, fp8, ep=32
**Token：** 512 (decode), bt=16, bf=512, btc=16, bts=16
**Local state：** 16 tokens/device, 12 experts/device, 4 bf tiles

---

## 1. Wall vs Device Time

| 指标 | 时间 (ms) | 占比 |
|------|-----------|------|
| Wall time (端到端) | 0.921 | 100% |
| Device time (trace) | 0.362 | 39% |
| **Host dispatch OH** | **0.559** | **61%** |

> Host dispatch 是最大单项开销。JAX JIT dispatch + shard_map wrapper + XLA command submission = 0.559ms。
> V1 wall time = 0.819ms，假设 host OH 相近 (~0.55ms)，则 V1 device time ≈ 0.269ms vs V2 = 0.362ms。

---

## 2. Ablation Sweep Results

### 2.1 Wall Timing

| 实验 | Wall (ms) | Δ (ms) | 节省 % |
|------|-----------|--------|--------|
| BASELINE | 0.921 | — | — |
| NO_A2A + NO_SYNC | 0.854 | -0.067 | 7.3% |
| NO_WEIGHT_LOAD | 0.870 | -0.051 | 5.5% |
| NO_FFN1 | 0.845 | -0.076 | 8.3% |
| NO_FFN2 | 0.843 | -0.078 | 8.5% |
| ALL_DISABLE | 0.580 | -0.341 | 37.0% |

### 2.2 Trace Timing (device_duration_ps)

| 实验 | Device (ms) | Δ (µs) | 占 device % |
|------|-------------|--------|-------------|
| BASELINE | 0.362 | — | 100% |
| NO_A2A + NO_SYNC | 0.270 | -92 | 25.4% |
| NO_FFN1 | 0.278 | -84 | 23.2% |
| NO_FFN2 | 0.315 | -47 | 13.0% |
| ALL_DISABLE | 0.031 | -331 | 91.4% |

### 2.3 Derived Breakdown

| Stage | Device Time (µs) | 占 device % | 说明 |
|-------|------------------|-------------|------|
| A2A + sync_barrier | 92 | 25% | scatter/gather DMA + ICI + barrier |
| FFN1 (gate_up) | 84 | 23% | 3 dot: x×w1, x×w3, + dequant |
| FFN2 (act_down) | 47 | 13% | 1 dot: act×w2, + dequant |
| Weight load + pipeline | ~108 | 30% | 12 experts × 3 weights DMA |
| Fixed (metadata+scatter+output) | 31 | 9% | in-kernel metadata, token routing, output acc |
| **Total device** | **362** | **100%** | |

---

## 3. LLO Dump Analysis

### 3.1 Instruction Statistics

| 指标 | 值 |
|------|----|
| Total bundles | 31,902 |
| MXU active bundles | 2,574 (8.1%) |
| **MXU utilization** | **7.2%** (4,600 / 63,804 slots) |
| VALU ops | 112,325 (3.52/bundle avg) |
| VLOAD (regular) | 15,732 |
| VLOAD:FILL (spill) | 4,446 in 2,804 bundles (8.8%) |
| VSTORE:SPILL | 4,194 in 2,561 bundles (8.0%) |
| SALU ops | 4,894 |

> **核心发现：Kernel 是 VALU-dominated，MXU 几乎闲置。**
> 7.2% MXU 利用率意味着 decode 的瓶颈不是矩阵乘法，而是地址计算、fp8 dequant、DMA 调度、控制流。

### 3.2 VREG Pressure

| 指标 | 值 |
|------|----|
| VREG pressure | 434 |
| VREG capacity (v7x) | ~256 |
| Estimated spill VREGs | ~178 |
| Spill store bundles | 2,561 (8.0%) |
| Fill load bundles | 2,804 (8.8%) |

> VREG 严重溢出。原因：double-buffered weights (2 slots × 3 weights × 2 t_packing) + accumulators + tokens + dequant buffers 同时 live。
> 对 device time 的影响：~8% × 362µs ≈ 29µs。

### 3.3 VMEM Allocation

| Buffer | Shape | Size (MB) | 用途 |
|--------|-------|-----------|------|
| alloc8 | bf16[2,8,16,2,3072] | 3.0 | x token buffer (bts × double-buf) |
| alloc12 | f8e4[2,2,3072,512] | 6.0 | w1 fp8 (×2 slot, ×2 packing) |
| alloc13 | f8e4[2,2,3072,512] | 6.0 | w3 fp8 |
| alloc14 | f8e4[2,2,512,3072] | 6.0 | w2 fp8 |
| alloc18 | bf16[2,3072,512] | 6.0 | w1 dequant staging |
| alloc19 | bf16[2,3072,512] | 6.0 | w3 dequant staging |
| alloc20 | bf16[2,512,3072] | 6.0 | w2 dequant staging |
| alloc15-17 | f32 scales | 0.6 | fp8 scale buffers |
| alloc9-11,21-25 | misc | 1.3 | accumulators, output, topk |
| alloc1 | u32[144,128] | 0.07 | internal scratch |
| **Total VMEM** | | **~41** | **64% of 64MB** |

> fp8 dequant staging (alloc18-20) 占 18MB。如果 MXU 直接支持 fp8 accumulation，这部分可以省掉。

---

## 4. Root Cause 诊断

### 4.1 Host Dispatch Overhead (0.559ms, 61%)

最大单项。包含：
- JAX JIT dispatch (compiled function lookup, argument binding)
- shard_map wrapper (per-device argument slicing)
- XLA runtime command submission
- pjrt buffer management

**不可从 kernel 内部优化。** 需要 JAX/XLA runtime 层面的改进：
- Command buffer / executable cache reuse
- 减少 kernel 参数数量 (当前 15+ HBM operands)
- Fuse pre-processing (padding, gating) into kernel

### 4.2 VALU-Dominated Instruction Mix (device time)

MXU 利用率仅 7.2%。31,902 bundles 中 29,328 不使用 MXU。

主要 VALU 来源：
1. **fp8 dequant-in-VMEM**: 每个 weight tile (3072×512) 需要 vunpack + vscale → bf16。12 experts × 3 weights × 4 bf tiles = 144 次 dequant，每次 ~1.5M 元素 × 2+ VALU ops
2. **Address computation**: 动态 expert ID → HBM offset 计算，per-expert per-token
3. **Token routing (scatter)**: 遍历 16 tokens × 8 top_k = 128 assignments
4. **Sync barrier**: 32 devices unrolled → ~400 bundles 纯 SALU
5. **fori_loop control**: 12 experts × 4 bf = 48 iterations of loop control overhead

### 4.3 VREG Spill (~29µs, 8%)

根因：Weight double-buffer 占用 6 个 VMEM buffer slots (w1/w3/w2 × 2 slots)，每个含 2 t_packing sublanes。加上 dequant staging buffers、accumulators、tokens，总 live VREG 需求远超 256。

### 4.4 A2A + Sync Barrier (92µs, 25%)

Scatter/gather 涉及 32 devices × 12 local experts 的 ICI DMA。Sync barrier 在 32 devices 间做全局同步。对 decode 而言，每次同步的数据量极小 (16 tokens × 6144 × 2B = 192KB per scatter)，ICI startup latency 主导。

---

## 5. V1 vs V2 Device Time 对比

| | V1 (推算) | V2 (实测) | 差距 |
|---|-----------|-----------|------|
| Wall time | 0.819ms | 0.921ms | +12.5% |
| Host OH | ~0.55ms | 0.559ms | ~相同 |
| Device time | ~0.269ms | 0.362ms | **+34.6%** |

V2 device time 比 V1 多 ~93µs。可能原因：
1. V2 每个 bf tile 更大 (512 vs V1 可能更小)，dequant VALU 更重
2. V2 不做 hidden-dim tiling → 每个 GEMM 输入更大 → VMEM 占用更大 → 更多 spill
3. V2 的 double-buffer 结构对 decode (仅 12 experts) 的 amortization 不如 V1

---

## 6. 优化方向

### 6.1 Decode 专用路径 (最大 ROI)

为 decode (bt ≤ 32) 做专用 kernel 变体：
- **去掉 double-buffering**: 12 experts 太少，pipeline overlap 收益不大
- **去掉 bf tiling**: f=2048, 直接全量加载不做 bf 分块 → 消除 bf loop
- **简化 DMA**: 单次 weight load，不用 start/wait 交错
- **目标**: bundle count < 15K, device time < 200µs

### 6.2 fp8 Dequant 优化

当前：HBM → fp8 VMEM → VALU dequant → bf16 VMEM → MXU dot
可能改进：
- 如果 v7x MXU 支持 fp8 input（待确认），直接 fp8 进 MXU 省掉 dequant
- 或者 dequant 与 GEMM fuse，一边 dequant 一边 dot，减少 live VREG

### 6.3 VREG Pressure 缓解

- Decode 路径去掉 double-buffer → weight slots 从 2 降到 1 → VREG 需求减半
- dequant staging buffers 可以和 weight buffer 共享 (时间互斥)

### 6.4 Sync Barrier 简化

- 32-device barrier unrolling → 400+ SALU bundles
- 考虑用 hardware barrier 替代 software vsyncadd pattern
- 或者对 decode 路径跳过 sync barrier (如果能证明 A2A 时序安全)

### 6.5 Host Dispatch 研究 (长期)

- Profile JAX dispatch path (jax.profiler + py-spy)
- 评估 XLA command buffer reuse 是否能减少 dispatch overhead
- 考虑 batched kernel launch (多个 layer 的 fused_moe 合并 dispatch)

---

## 7. 数据收集命令参考

```bash
# Wall ablation (on ablation-16 pods)
BENCH_FP8=1 BENCH_TOKENS=512 BENCH_BT=16 BENCH_BF=512 BENCH_WALL=1 BENCH_WARMUP=3 BENCH_ITERS=10

# Trace ablation
BENCH_WALL=0  # uses trace_timeit with device_duration_ps

# LLO dump
export LIBTPU_INIT_ARGS="--xla_xprof_register_llo_debug_info=true --xla_jf_dump_to=/tmp/tpu_logs/llo2/llo --xla_jf_dump_hlo_text=true --xla_jf_dump_llo_text=true --xla_jf_emit_annotations=true --xla_mosaic_dump_to=/tmp/tpu_logs/llo2/mosaic --xla_mosaic_enable_llo_source_annotations=true"
```
