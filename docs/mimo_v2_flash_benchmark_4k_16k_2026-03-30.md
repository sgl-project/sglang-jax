# MiMo-V2-Flash 4K/16K Benchmark 报告

**日期**: 2026-03-30
**硬件**: TPU v6e-16 (4 nodes × 4 chips = 16 chips, 31.25 GB HBM/chip)
**分支**: feat/mimo-v2-flash (commits: 9f6678ce weight refactoring, 8e008f10 KV head replication removal)

## 1. 运行配置

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

### 关键参数说明

| 参数 | 值 | 说明 |
|------|---|------|
| `--mem-fraction-static` | 0.95 | HBM 中可用于 KV cache 的比例上限。设高避免 "Not enough memory" 检查 |
| `--max-total-tokens` | 30000 | KV cache pool 的实际 token 上限。限制实际分配量，为 XLA 运行时留内存 |
| `--chunked-prefill-size` | 2048 | Prefill 每次处理的 token 数上限。降低以避免 MoE 前向传播 OOM |
| `--context-length` | 16384 | 单请求最大序列长度 |
| `--disable-precompile` | — | 跳过启动时 XLA 预编译（256-expert MoE 编译 OOM） |

### 实际 KV Pool 容量 (Hybrid Cache)

由 `set_num_token_hybrid()` 按 `swa_full_tokens_ratio=0.8` 分配：

| 池类型 | 层数 | per-layer token 数 | 说明 |
|--------|-----|-------------------|------|
| Full-Attention | 9 层 | 35,820 | head=16, head_dim=256, v_dim=128, bf16 |
| SWA | 39 层 | 28,656 | 同上（SWA pool = full pool × 0.8） |

有效容量受限于 `min(full_pool, swa_pool)` = **28,656 tokens**（SWA 池为瓶颈）。

注：SWA 层的 KV cache 只需保留 `sliding_window_size` 个 token，超出窗口的 token 可被 tombstone 释放。
因此长 context 场景下，SWA 池会通过 tombstone 回收空间，有效容量高于 28,656。

### Precompile OOM 问题

尝试了多种 `mem-fraction-static` 配置均无法完成 precompile：

| mem_fraction | KV pool tokens | precompile 结果 |
|-------------|---------------|----------------|
| 0.70 | 80,773 | EXTEND 4096 tokens 编译 OOM (34.68G/31.25G) |
| 0.65 | — | KV 分配失败 "Not enough memory" |
| 0.60 | — | KV 分配失败 |
| 0.95 | 298,112 | EXTEND 2048 tokens 编译 OOM (47.78G/31.25G) |
| 0.95 + max_tokens=80K | 80,773 | EXTEND 2048 tokens 编译 OOM (34.69G/31.25G) |

根因：256-expert MoE 模型编译大 batch extend 时需要巨量 HBM 临时内存，无法在 31.25GB/chip 的 v6e 上完成。

## 2. Benchmark 结果

### 2.1 Decode 测试 (input=4096, output=1024)

| batch_size | TTFT (s) | output throughput (tok/s) | ITL / TPOT (ms) | KV usage | 状态 |
|-----------|----------|--------------------------|----------------|----------|------|
| 1 | 3.48 | 23.85 | ~42 | ~18% | 完成 |
| 4 | 13.89 | 10.73 | ~373 | ~70% | 完成 |
| 8 | — | ~2.47 | ~3000+ | 88%+ retract | KV 不足 |

注：TTFT 偏高因为 disable-precompile，首次运行各 shape 需运行时 JIT 编译。第二次运行 bs=1 的 TTFT 从 71.8s 降至 3.48s。

### 2.2 对比项目预期

| 指标 | 项目预期 | 实测 (bs=1) | 实测 (bs=4) | 是否达标 |
|------|---------|------------|------------|---------|
| TPS ≥ 50 | ✅ | 23.85 | 10.73 | 不达标 |
| TPOT ≤ 20ms | ✅ | 42ms | 373ms | 不达标 |
| TTFT < 5s | ✅ | 3.48s | 13.89s | bs=1 达标 |
| bs ≥ 64 可运行 | ✅ | — | — | KV 容量不足 |

### 2.3 短 context 对比数据 (input=128/512, context-length=1024, mem=0.7, precompile ON)

此前同一分支在 context-length=1024 配置下的 benchmark：

| 配置 | 基线 tok/s (head_repl) | 当前 tok/s (no head_repl) | 变化 | ITL (ms) |
|------|----------------------|-------------------------|------|---------|
| bs=1, in=128 | 68.1 | 77.7 | **+14.1%** | 13.2 |
| bs=4, in=128 | 173.4 | 124.4 | **-28.3%** | 32.5 |
| bs=4, in=512 | 211.7 | 310.0 | **+46.4%** | 13.0 |
| bs=16, in=128 | 231.3 | 237.3 | **+2.6%** | — |

## 3. 瓶颈分析

### 3.1 HBM 内存分布 (每 chip)

| 用途 | 占用 (GB) | 占比 | 说明 |
|------|----------|------|------|
| 模型权重 (FP8) | ~20 | 64% | 256 experts × (wi_0 + wi_1 + wo) × FP8 + scales |
| KV Cache (30K tokens) | ~0.9 | 3% | FA 9层 + SWA 39层, 1 head × (256+128) dim × bf16, hybrid分池 |
| XLA 编译/运行时 | ~10 | 32% | MoE routing, activation, compilation temps |
| **总计** | ~31 | 100% | 刚好打满 31.25 GB |

### 3.2 KV Cache 容量限制

Per-token KV 内存（每 chip, TP=16 后每 chip 1 个 KV head）：
```
Full-Attention (9 层):  1 head × (256+128) dim × 9 layers  × 2 bytes =  6,912 bytes ≈ 7 KB/token
SWA (39 层):            1 head × (256+128) dim × 39 layers × 2 bytes = 30,576 bytes ≈ 30 KB/token
```

Hybrid Cache 分池后的有效容量为 min(full_pool, swa_pool) = 28,656 tokens。
但 SWA 层通过 sliding window tombstone 回收机制，实际 per-sequence 只需 min(context_len, sliding_window_size) 个 SWA slot。

各 batch 配置所需 full-pool tokens vs 当前容量 (35,820 tokens)：

| batch_size | input+output | 所需 tokens | vs full pool (35K) | 可否运行 |
|-----------|-------------|------------|-------------------|---------|
| 1 × 5K | 5,120 | 5,120 | 14% | 可以 |
| 4 × 5K | 20,480 | 20,480 | 57% | 可以 |
| 8 × 5K | 40,960 | 40,960 | 114% | 不可以 |
| 16 × 5K | 81,920 | 81,920 | 229% | 不可以 |
| 64 × 5K | 327,680 | 327,680 | 915% | 不可以 |
| 4 × 17K | 69,632 | 69,632 | 194% | 不可以 |

### 3.3 单步 Decode 性能

bs=1 时 ITL=42ms（TPOT 预期 ≤20ms 的 2.1×）。在短 context (in=128) 下 ITL 可达 13ms，说明 decode 延迟随 KV cache 长度线性增长：

| context (KV len) | ITL (ms) | 说明 |
|-----------------|----------|------|
| ~128 | 13 | 短 context 基线 |
| ~512 | 13 | 零拷贝优化后短 context 类似 |
| ~4096 | 42 | 4K context |
| ~16384 | ~150 (估算) | 线性外推 |

## 4. 要达到项目预期的路径

### 4.1 硬件层面

| 方案 | 说明 | 预期效果 |
|------|------|---------|
| 更大 TPU (v6e-64 / v5p) | 更多 chip，每 chip 分摊更少 expert 权重 | KV 容量 4-8×，bs=64 可行 |
| TP=8 + DP=2 | 每 chip 加载更多 expert 但 all-reduce 跨更少 chip | HBM 更均衡 |

### 4.2 软件层面

| 方案 | 预期效果 | 优先级 |
|------|---------|--------|
| MTP Speculative Decoding | 有效 TPS 2-2.5×（不改变单步延迟） | P0 |
| FlashDecoding (KV 并行) | 长 context ITL 降低 2-3× | P1 |
| All-Reduce 通信-计算重叠 | ITL 降低 10-15% | P2 |
| Precompile OOM 修复 | 减少运行时 JIT 开销，降低 TTFT | P2 |

### 4.3 配置优化

| 方案 | 说明 |
|------|------|
| 增大 `--max-total-tokens` + 降低 `--chunked-prefill-size` | 用 prefill 速度换 KV 容量 |
| 分离 prefill / decode 资源 | PD Disaggregation |

## 5. 结论

在当前 TPU v6e-16 (31.25 GB HBM/chip) 上运行 MiMo-V2-Flash (256-expert MoE, FP8)：

1. **模型权重占 HBM 的 ~64%**，留给 KV cache 和运行时的空间极其有限
2. **KV cache 最大约 30K tokens**（max-total-tokens=30000 时），仅支持 bs=4 × 4K input
3. **单步 decode 在 4K context 下 ITL=42ms**，不满足 TPOT≤20ms 的要求
4. **bs≥8 with 4K input 因 KV 不足无法正常运行**，更不用说 bs=64/128
5. **Precompile 无法完成**，256-expert MoE 的 XLA 编译需要大量临时 HBM

**核心结论：TPU v6e-16 的 HBM 容量不足以同时满足 MiMo-V2-Flash 的模型权重 + KV cache + 运行时内存需求。要达到项目预期 (TPS≥50, TPOT≤20ms, bs≥64)，需要更大规模的 TPU 硬件或架构级优化（MTP, FlashDecoding）。**
