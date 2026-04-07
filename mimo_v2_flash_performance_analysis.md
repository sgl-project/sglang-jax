# MiMo-V2-Flash TPU v6e-16 性能分析与优化路线图

**更新**: 2026-03-29 (零拷贝 KV Cache 优化后)

## 1. 当前部署配置

| 项目 | 值 |
|------|---|
| 模型 | MiMo-V2-Flash (256 experts, FP8, block_size=128) |
| 硬件 | TPU v6e-16 (4 nodes × 4 chips = 16 chips) |
| 并行策略 | TP=16, EP=16, MoE backend: epmoe |
| 架构特殊点 | head_dim=192 (Q/K), v_head_dim=128 (V), 每层 KV head 数量不同 |
| 量化 | FP8 weight + FP8 activation, blockwise quantization |
| 价格 | Spot ~$9.14/h (us-central1-b) |

## 2. 性能数据

### 2.1 优化前 Decode 基线 (Precompile 开启, context-length=32768)

| input_len \ bs | 1 (tok/s) | 4 (tok/s) | 16 (tok/s) | 1 (ITL ms) | 4 (ITL ms) |
|---------------|---:|---:|---:|---:|---:|
| 128 | 16.9 | 58.0 | 231.3 | 59 | 69 |
| 512 | 15.3 | 43.9 | 175.2 | 65 | 86 |
| 1024 | 14.1 | 35.2 | 140.9 | 71 | 108 |
| 2048 | 12.2 | 25.3 | 101.0 | 82 | 152 |
| 4096 | 9.6 | - | - | 104 | - |
| 8192 | 6.7 | - | - | 149 | - |
| 16384 | 4.2 | - | - | 237 | - |
| 30720 | 2.6 | - | - | 391 | - |

### 2.2 零拷贝优化后 Decode 性能

| 配置 | 优化前 ITL | 优化后 ITL | 优化前 tok/s | 优化后 tok/s | 提升 |
|------|----------|----------|------------|------------|------|
| bs=1, in=128 | 59ms | **14.7ms** | 16.9 | **68.1** | **4.0x** |
| bs=1, in=512 | 65ms | **18.9ms** | 15.3 | **52.8** | **3.4x** |
| bs=4, in=128 | 69ms | **23.1ms** | 58.0 | **173.4** | **3.0x** |
| bs=4, in=512 | 86ms | **18.9ms** | 43.9 | **211.7** | **4.8x** |

### 2.3 TTFT (Prefill 延迟, 秒)

| input_len \ bs | 1 | 4 |
|---------------|---:|---:|
| 128 | 0.09 | 0.17 |
| 512 | 0.17 | 0.31 |
| 1024 | 0.28 | 0.56 |
| 2048 | 0.33 | 1.19 |
| 4096 | 0.67 | - |
| 8192 | 1.52 | - |
| 16384 | 3.69 | - |
| 30720 | 9.28 | - |

### 2.4 GPU 对比 (优化后)

| 指标 | 优化后 TPU v6e-16 | H100×8 (估算) | 差距 |
|------|------------------|---------------|------|
| bs=1 tok/s | 68.1 | ~60-80 | **接近持平** |
| bs=4 tok/s | 173.4 | ~250-350 | ~2x |
| Prefill TTFT (128 tok) | 0.09s | ~0.02-0.03s | ~3-4x |

| 平台 | 价格/h | 优化后 tok/s (bs=4) | tok/s/$ |
|------|--------|-------------------|---------|
| TPU v6e-16 Spot | $9.14 | 173.4 | **19.0** |
| H100×8 Spot | ~$15 | ~350 | ~23 |

优化后 TPU v6e-16 价格性能比已接近 H100 Spot 水平。

## 3. XProf Profiling 分析

### 3.1 优化前 (broadcast 主导)

**环境**: v6e-16, precompile 开启, decode-only

```
优化前 Decode 时间分布:
  broadcast (KV tile)   ████████████████████████████  49.9%
  data formatting       ████████                      15.2%
  slice (KV head)       ███████                       14.8%
  custom-call (MXU)     █████                          9.7%
  all-reduce            ███                            6.1%
  loop fusion           ██                             3.0%
  other                 █                              2.9%
```

**核心问题**: broadcast + slice + data formatting = **79.9%** 时间花在 KV cache 数据搬运，MXU 计算仅 9.7%。

### 3.2 优化后 (计算主导)

**环境**: v6e-16, precompile 开启, decode-only, 零拷贝 KV Cache 启用

```
优化后 Decode 时间分布:
  custom-call (MXU)     ████████████████████           35.91%
  all-reduce (通信)     ███████████████                26.74%
  loop fusion           ███████                        12.71%
  data formatting       ██████                         12.13%
  pad                   ██                              4.13%
  slice                 ▏                               0.44%
  broadcast             ▏                               0.01%
```

| Op 类别 | 优化前 | 优化后 | 变化 |
|---------|--------|--------|------|
| broadcast (KV cache tile) | 49.9% | 0.01% | **-49.9pp** |
| slice (head slicing) | 14.8% | 0.44% | **-14.4pp** |
| data formatting (copy) | 15.2% | 12.13% | -3.1pp |
| custom-call (MXU 计算) | 9.7% | 35.91% | **+26.2pp** |
| all-reduce | 6.1% | 26.74% | +20.6pp |
| loop fusion | 3.0% | 12.71% | +9.7pp |
| pad | — | 4.13% | — |

**关键变化**: 数据搬运从 79.9% 降至 12.6%，MXU 计算从 9.7% 升至 35.9%，通信从 6.1% 升至 26.7%（绝对时间不变，占比因总时间缩短而上升）。

## 4. 当前瓶颈拆解 (优化后)

### 4.1 MXU 计算 — 35.91% (正常不可压缩)

每层 decoder (~56 层) 的计算链路:

| 计算 | 操作 | 说明 |
|------|------|------|
| Q/K/V Projection | 3 个 FP8 matmul | q_proj: blockwise kernel; **k/v_proj: dequant fallback** |
| FlashAttention | Pallas kernel | `ragged_paged_attention_split` |
| O Projection | 1 个 FP8 matmul | |
| MoE Gate | 1 个 matmul | Router logits |
| MoE Up+Gate | GMM (Megablox) | 256 experts, top_k 选择后 |
| MoE Down | GMM (Megablox) | |
| Shared Expert | 3 个 matmul | gate+up+down |

**k/v_proj 退化**: TP=16 后 per-device n_out=192/128 < MXU tile 256，走 dequant fallback (scale expand + bf16 matmul)。

### 4.2 All-Reduce 通信 — 26.74% (第一优化目标)

每层 decoder 至少 3-4 次 all-reduce，~56 层共 **168-224 次**:

| 位置 | 操作 | 通信量 |
|------|------|--------|
| o_proj 输出 | TP reduction (implicit via out_sharding) | `[T, hidden_size]` |
| MoE TP | `lax.psum(output, "tensor")` | `[T, hidden_size]` |
| MoE EP | `lax.psum(data, "expert")` | `[T, hidden_size]` |
| Shared Expert down_proj | TP reduction | `[T, hidden_size]` |

跨 4 个物理节点 (ICI 带宽有限)，每次 all-reduce 需要多 hop 通信。

### 4.3 Data Formatting (Copy) — 12.13%

| 来源 | 说明 |
|------|------|
| MoE permute/unpermute | 每层 `argsort` + `jnp.take` 重排 tokens 到 expert |
| MoE mesh 切换 | 每层 4 次 `reshard` (3 输入 + 1 输出) |
| KV cache 3D→4D reshape | 每层 `jax.lax.reshape(cache, (pages, page_size, heads, dim))` |
| Pallas kernel 准备 | `_prepare_single_kv_cache()` 的 reshape + padding |

### 4.4 Loop Fusion — 12.71%

XLA 编译器自动融合的小算子 (activation, RMSNorm, gate 等)，属正常计算开销。

### 4.5 Pad — 4.13%

- Pallas kernel 内部 head/dim 对齐 padding (double padding: `_call_split` + `_prepare_single_kv_cache`)
- Q tensor packing 对齐

### 4.6 端到端 Decode 每步时间线

以 bs=1, input_len=128 为例 (ITL=14.7ms 优化后):

```
一步 Decode (~14.7ms, ~56 layers × ~0.24ms/layer + overhead):
  ┌─ Per-Layer ──────────────────────────────────────────────┐
  │ [Q/K/V proj 0.03ms] → [FlashAttn 0.05ms] → [O proj]    │
  │ [All-Reduce 0.07ms]                                      │
  │ [MoE: Route→Permute→GMM×3→Unpermute 0.06ms]             │
  │ [MoE All-Reduce×2 0.03ms]                                │
  └──────────────────────────────────────────────────────────┘
  [Sampler + KV Update + Scheduling]
```

## 5. 优化路线图 (按 ROI 排序)

### 5.1 P0: MTP Speculative Decoding — 预期 2-2.6x 额外提升

| 项目 | 说明 |
|------|------|
| **原理** | MiMo-V2-Flash 原生支持 Multi-Token Prediction (MTP head) |
| **收益** | 假设 MTP 接受率 60-70%: 2 draft → ~1.6-1.7x, 3 draft → ~2.0-2.3x |
| **优点** | 不改变单步延迟，通过减少步数提升吞吐; all-reduce 开销被多 token 摊薄 |
| **挑战** | 实现 MTP draft/verify 逻辑; KV cache 需支持 speculative rollback |
| **影响** | bs=1 ITL 14.7ms 不变，有效 tok/s 68 → ~110-170 |

### 5.2 P1: All-Reduce 通信优化 — 预期 10-15% 总时间缩减

当前 all-reduce 占 26.74%，方案:

| 方案 | 预期收益 | 难度 | 说明 |
|------|---------|------|------|
| 通信-计算重叠 (async psum) | ~50% 通信等待 | 中 | 计算和通信可并行 |
| MoE TP+EP 合并通信 | 减少 1 次 psum/层 | 中 | 两次 psum 合并为 1 次 |
| DP=2 × TP=8 | 减少跨节点通信 | 高 | 同时修复 k/v_proj 退化 |

### 5.3 P2: k/v_proj Blockwise Kernel — 预期 3-5% 总时间缩减

| 方案 | 说明 |
|------|------|
| **方案 A** (简单) | 启动时检测 fallback，一次性 dequant 为 bf16，省掉每步 scale expand |
| **方案 B** (彻底) | DP+TP hybrid: Attention 用 DP=4×TP=4, MoE 保持 EP=16 |

### 5.4 P3: MoE Dispatch 优化 — 预期 5-8% 总时间缩减

| 方向 | 说明 |
|------|------|
| 缓存 permutation indices | 相邻层 routing 可能复用 argsort 结果 |
| 减少 reshard 次数 | 每层 4 次 reshard → 可能合并 |
| EP mesh 常驻 | 避免每层 `use_abstract_mesh` 切换 |

### 5.5 P4: FlashDecoding for Long Context — 预期长 context 2-3x 提升

| 项目 | 说明 |
|------|------|
| **问题** | Decode attention 逐 token 扫描 KV cache，ITL 随 context 线性增长 |
| **现象** | 128 tok → 14.7ms, 30k tok → ~100ms (7x) |
| **方案** | Sequence 维度并行拆分 KV cache，多 core 同时计算 partial attention |
| **预期** | 30k context ITL: ~100ms → ~40ms |

### 5.6 中长期: Prefill-Decode 分离 + 编译优化

- **PD Disaggregation**: Prefill 和 Decode 分配到不同 TPU 节点，提升并发
- **XLA 编译优化**: 减少 kernel launch 次数，算子融合

## 6. 组合提升预期

| 优化项 | 单步 ITL (bs=1,128) | 有效 tok/s (bs=1) | 累计提升 |
|--------|-------------------|--------------------|---------|
| **当前 (零拷贝)** | 14.7ms | 68 | 1x (基线) |
| + MTP (2.0x) | 14.7ms | **~136** | 2.0x |
| + All-Reduce 优化 | ~12.5ms | **~160** | 2.4x |
| + MTP (2.5x) | ~12.5ms | **~200** | 2.9x |
| + FlashDecoding (长context) | ~5-8ms (30k) | ~125-200 (30k) | — |
| 理论上限 (仅 MXU) | ~5.3ms | ~470 | 6.9x |

**建议执行顺序**: MTP → All-Reduce 优化 → FlashDecoding → k/v_proj 修复

MTP 是 ROI 最高的单项优化: 不改变底层 kernel，直接倍增吞吐。

## 7. 精度验证摘要

| 方法 | MMLU 5-shot | 说明 |
|------|-----------|------|
| Chat API (FP8 TPU) | 67.75% | Chat template 降低 ~14pp |
| **Completion API (FP8 TPU)** | **81.56%** (提取成功) | 接近官方，FP8 仅降 ~5pp |
| 官方 (BF16) | 86.70% | base model reference |

差距分解: ~14pp Chat 模板 + ~5pp FP8 量化。零拷贝优化无精度回归。

---

*文档更新于 2026-03-29，基于 TPU v6e-16 实测数据*
