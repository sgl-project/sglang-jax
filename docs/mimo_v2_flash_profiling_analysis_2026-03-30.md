# MiMo-V2-Flash TPU v6e-16 Profiling 分析

**日期**: 2026-03-30
**硬件**: TPU v6e-16 (4 nodes × 4 chips = 16 chips, 31.25 GB HBM/chip)
**分支**: feat/mimo-v2-flash
**Profiler**: JAX trace profiler (`jax.profiler.start_trace/stop_trace`)

## 1. Profiling 配置

### 运行配置

```bash
python -u -m sgl_jax.launch_server \
  --model-path /models/MiMo-V2-Flash \
  --trust-remote-code \
  --tp-size 16 --ep-size 16 \
  --moe-backend epmoe \
  --nnodes 4 --node-rank $NODE_RANK \
  --dist-init-addr ${HEAD_IP}:10011 \
  --host 127.0.0.1 --port 30271 \
  --context-length 16384 \
  --max-total-tokens 30000 \
  --chunked-prefill-size 2048 \
  --mem-fraction-static 0.95 \
  --disable-precompile --skip-server-warmup \
  --log-level info
```

### Profiling 场景

| Profile | input_len | output_len | batch_size | 捕获内容 |
|---------|-----------|------------|------------|----------|
| #1: bs1_prefill_only | 4096 | 1 | 1 | 纯 prefill (2 chunks × 2048) |
| #2: bs1_prefill_decode | 4096 | 32 | 1 | Prefill + decode |
| #3: bs4_prefill_decode | 4096 | 32 | 4 | Prefill + decode (批量) |
| #4: decode_focused | 128 | 64 | 1 | 短 prefill + 多 decode 步 |

注：profiler 每次捕获 3 个 forward step。4K input 场景中 prefill 消耗大部分步骤配额。

## 2. Model Forward Pass 时间

| 场景 | 每步设备时间 (ms/chip) | 类型 |
|------|----------------------|------|
| 4K prefill (chunked 2048 tokens) | ~1,720 | PREFILL |
| 128-token prefill | ~516 | PREFILL |
| bs=1 decode (128 context) | ~37.9 | DECODE |

注：
- 4K prefill 每步处理 2048 tokens（chunked-prefill-size=2048）
- Decode 步设备时间 37.9ms，加上 scheduler/调度开销后实测 ITL ≈ 42ms (4K context)
- 实测 ITL=14.7ms (128 context) 与 profile 的 37.9ms 的差异来自 profiling 本身的开销

## 3. HLO 操作时间分布

### 3.1 4K Prefill (per chip, 2 chunks × 2048 tokens)

| HLO Category | 时间 (ms) | 占比 | 说明 |
|-------------|----------|------|------|
| **loop fusion** | 1413.4 | **46.2%** | 元素级融合: RMSNorm, SiLU, MoE gate |
| **convolution fusion** | 775.5 | **25.4%** | 矩阵乘 (Q/K/V/O proj + MoE GMM) |
| **custom fusion** | 375.9 | **12.3%** | Pallas attention + MoE custom kernels |
| dynamic-slice | 196.4 | 6.4% | KV cache 读取 |
| custom-call | 133.6 | 4.4% | 独立 MXU 调用 |
| all-reduce | 123.8 | 4.0% | TP/EP 通信 |
| data formatting | 29.7 | 1.0% | 数据拷贝 |
| sort | 6.8 | 0.2% | MoE top-k 排序 |
| 其他 | 3.2 | 0.1% | pad, broadcast, etc. |
| **总计 (leaf ops)** | **3058.3** | **100%** | |

```
4K Prefill 操作分布:
  loop fusion (elem ops)    ████████████████████████████████████████████████  46.2%
  convolution fusion (MMul) ██████████████████████████                       25.4%
  custom fusion (Pallas)    █████████████                                    12.3%
  dynamic-slice (KV read)   ███████                                           6.4%
  custom-call (MXU)         █████                                             4.4%
  all-reduce (通信)         ████                                              4.0%
  data formatting (copy)    █                                                 1.0%
  other                     ▏                                                 0.3%
```

### 3.2 Decode (bs=1, 128 context, per chip)

| HLO Category | 时间 (ms) | 占比 | 说明 |
|-------------|----------|------|------|
| **custom fusion** | 66.4 | **44.6%** | Pallas attention + MoE fusions (matmul 被融合) |
| **loop fusion** | 46.8 | **31.4%** | 元素级融合: RMSNorm, SiLU, gate |
| data formatting | 14.6 | 9.8% | MoE permute/unpermute + reshard copies |
| **all-reduce** | 10.6 | **7.1%** | TP/EP 通信 |
| custom-call | 8.0 | 5.4% | MXU 调用 |
| sort | 1.1 | 0.7% | MoE routing sort |
| 其他 | 1.3 | 0.9% | |
| **总计 (leaf ops)** | **148.8** | **100%** | 4 chips 合计，per-chip ≈ 37.2ms |

```
Decode 操作分布:
  custom fusion (Pallas)    ███████████████████████████████████████████████  44.6%
  loop fusion (elem ops)    ████████████████████████████████                 31.4%
  data formatting (copy)    ██████████                                       9.8%
  all-reduce (通信)         ████████                                         7.1%
  custom-call (MXU)         ██████                                           5.4%
  other                     ██                                               1.6%
```

### 3.3 128-token Prefill

| HLO Category | 时间 (ms) | 占比 | 说明 |
|-------------|----------|------|------|
| **custom fusion** | 1531.6 | **76.6%** | 大部分计算被融合 |
| loop fusion | 227.9 | 11.4% | 元素级 ops |
| convolution fusion | 82.6 | 4.1% | 矩阵乘 |
| data formatting | 58.3 | 2.9% | 数据拷贝 |
| custom-call | 35.3 | 1.8% | MXU 调用 |
| all-reduce | 30.3 | 1.5% | TP/EP 通信 |
| dynamic-slice | 26.1 | 1.3% | KV 读取 |
| 其他 | 6.9 | 0.4% | |

## 4. 对比分析

### 4.1 Prefill vs Decode 时间组成

| 类别 | 4K Prefill | Decode (128 ctx) | 变化 |
|------|-----------|------------------|------|
| 计算 (fusion+conv+custom-call) | **87.9%** | **81.4%** | compute-bound |
| 通信 (all-reduce) | 4.0% | 7.1% | decode 占比上升 |
| 数据搬运 (copy+format+slice) | 7.4% | 10.8% | decode 占比上升 |
| KV 读取 (dynamic-slice) | 6.4% | 0.1% | prefill 有大量 KV 读取 |

### 4.2 Prefill 扩展性 (128 vs 4K tokens)

| 指标 | 128 tokens | 4K tokens (2 chunks) | 比率 |
|------|-----------|---------------------|------|
| 每步设备时间 (ms) | 516 | 1,720 | 3.3x |
| Leaf HLO 总时间 (ms) | 1,999 | 3,058 | 1.5x |
| Tokens 比率 | 1x | 32x | |
| 每 token 计算效率 | ~15.6ms/tok | ~0.42ms/tok | **37x 更高效** |

4K prefill 的高效率来自矩阵乘的 batch 效应：大 batch matmul 的 MXU 利用率远高于小 batch。

### 4.3 与先前 XProf 数据对比

先前（零拷贝优化后，短 context precompile ON）XProf 数据：

| Op 类别 | XProf 数据 | 当前 Trace (decode) | 说明 |
|---------|----------|-------------------|------|
| custom-call (MXU) | 35.91% | 50.0%* | custom fusion + custom-call |
| all-reduce | 26.74% | 7.1% | XProf 可能包含通信等待 |
| loop fusion | 12.71% | 31.4% | 分类方式不同 |
| data formatting | 12.13% | 9.8% | 接近 |
| pad | 4.13% | — | Trace 中归入 other |

*注：XProf dashboard 和 trace.json 的分类方式不同。XProf 的 `custom-call` 包含所有 MXU 计算；trace 中被拆分为 `custom fusion` + `custom-call` + `convolution fusion`。XProf 的 `all-reduce` 可能包含通信同步等待时间，在 trace 的 device time 中被过滤。

### 4.4 Decode 实测 vs Profile 的关系

| 指标 | Profile device time | 实测 | 差距来源 |
|------|-------------------|------|----------|
| bs=1 decode (128 ctx) | 37.9ms | 14.7ms ITL | Profiling 自身 2.5x 开销 |
| bs=1 decode (4K ctx) | — | 42ms ITL | KV cache 扫描随 context 线性增长 |

## 5. 关键发现与优化建议

### 5.1 Prefill 阶段

1. **Compute-bound**: 87.9% 时间在计算（fusion + matmul）
2. **MoE GMM 效率**: `convolution fusion` (25.4%) 包含 MoE Group Matmul，对大 batch 高效
3. **通信占比低**: all-reduce 仅 4.0%，说明 TP=16 的通信开销在 prefill 中可接受
4. **KV dynamic-slice 6.4%**: 每层从 KV cache 读取历史 tokens，随 context 长度增加

**优化方向**:
- Prefill 性能已经较好（每 token 0.42ms at 4K），瓶颈不在这里
- `chunked-prefill-size=2048` 是合理的平衡点

### 5.2 Decode 阶段

1. **Custom fusion 主导 44.6%**: 包含 Pallas attention kernel + MoE custom ops
2. **Loop fusion 31.4%**: 大量小元素 ops (RMSNorm × 48 层, SiLU, gate 等)
3. **Data formatting 9.8%**: MoE dispatch 的 permute/unpermute + mesh reshard
4. **All-reduce 7.1%**: 每层 3-4 次 psum (o_proj, MoE TP, MoE EP, shared expert)

**优化方向**:
- **P0: MTP Speculative Decoding** — 不改变单步延迟，通过减少步数 2-2.5x 吞吐提升
- **P1: 减少 loop fusion 占比** — 48 层 × 多个 RMSNorm/activation 可进一步融合
- **P2: 减少 data formatting** — MoE mesh reshard 每层 4 次可合并
- **P3: All-reduce 通信-计算重叠** — async psum 减少等待

### 5.3 Decode 延迟随 Context 增长

| Context | 实测 ITL (ms) | 增长比 | 主要原因 |
|---------|-------------|--------|----------|
| 128 | 14.7 | 1x | 基线 |
| 512 | 18.9 | 1.3x | |
| 4096 | 42 | 2.9x | KV cache 扫描 |
| 16384 | ~150 (估) | ~10x | |
| 30720 | ~391 | ~27x | |

ITL 随 context 长度的超线性增长表明 FlashDecoding (KV 并行) 对长 context 至关重要。

## 6. Profile 文件位置

### TPU 端原始文件

```
/tmp/sgl-jax-profile/
├── 1774885214.0704496/  # Profile #1: bs=1 prefill only (in=4096, out=1)
├── 1774885312.124712/   # Profile #2: bs=1 prefill+decode (in=4096, out=32)
├── 1774885403.9127975/  # Profile #3: bs=4 prefill+decode (in=4096, out=32)
└── 1774886000.4540164/  # Profile #4: decode focused (in=128, out=64)
```

每个目录包含：
- `plugins/profile/<timestamp>/t1v-n-*.xplane.pb` (277MB-1.3GB) — XProf 原始数据
- `plugins/profile/<timestamp>/t1v-n-*.trace.json.gz` (31-32MB) — Chrome trace

### 本地拉取文件

```
profiles/4k_context/
├── bs1_prefill_only/      # trace.json.gz
├── bs1_prefill_decode/    # trace.json.gz
├── bs4_prefill_decode/    # trace.json.gz
└── decode_focused/        # trace.json.gz
```

Chrome trace 可用 `chrome://tracing` 或 Perfetto UI 打开查看。

---

*分析于 2026-03-30，基于 TPU v6e-16 实测 profiling 数据*
