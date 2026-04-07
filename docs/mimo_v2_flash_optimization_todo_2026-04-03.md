# MiMo-V2-Flash Optimization TODO (2026-04-03 Tonight)

## Current Baseline

| Metric | Value |
|--------|:-----:|
| Server gen throughput (decode only) | **570 tok/s** @ 18 running |
| bench_serving E2E (rate=3, 64 prompts) | **410 tok/s** |
| Prefill throughput | **8,100 tok/s** |
| ITL (median) | **31.6 ms** |
| 昨晚 baseline (custom bs=16) | **422.9 tok/s** |

**已确认的瓶颈**: Prefill-Decode 交替执行，decode 在 prefill step 空转 (~28% 时间浪费)

---

## TODO 1: `--enable-mixed-chunk` (高优先级)

**目标**: 消除 prefill 期间 decode 空窗，提升 E2E 吞吐

**原理**: 当前每个 forward step 要么是纯 prefill、要么是纯 decode。开启后 prefill+decode 合并到同一个 forward step，decode 不再停滞。

**操作**:
```bash
# 重启 server，加 --enable-mixed-chunk
python -m sgl_jax.launch_server \
    --model-path /models/MiMo-V2-Flash \
    --trust-remote-code \
    --tp-size 16 --ep-size 16 \
    --moe-backend epmoe \
    --nnodes 4 --node-rank $RANK \
    --dist-init-addr $HEAD_ADDR:10011 \
    --host 0.0.0.0 --port 30271 \
    --page-size 128 \
    --context-length 16384 \
    --disable-radix-cache \
    --chunked-prefill-size 16384 \
    --dtype bfloat16 \
    --mem-fraction-static 0.80 \
    --disable-precompile --skip-server-warmup \
    --enable-mixed-chunk \
    --log-level info
```

**验证**:
```bash
# Decode benchmark (rate=3)
python -m sgl_jax.bench_serving --backend sgl-jax \
    --host 127.0.0.1 --port 30271 \
    --dataset-name random --random-input-len 4096 --random-output-len 1024 \
    --random-range-ratio 1.0 --num-prompts 64 --request-rate 3 --flush-cache

# Prefill benchmark
python -m sgl_jax.bench_serving --backend sgl-jax \
    --host 127.0.0.1 --port 30271 \
    --dataset-name random --random-input-len 4096 --random-output-len 1 \
    --random-range-ratio 1.0 --num-prompts 64 --request-rate 3 --flush-cache
```

**预期**: E2E 吞吐从 410 → ~500-550 tok/s (接近 server gen 的 570)

**风险**: mixed batch 的 kernel 可能不兼容 (需要 attention backend 同时处理 prefill token 和 decode token)。如果 crash，回退到 TODO 2。

---

## TODO 2: 调整 `chunked_prefill_size` (中优先级)

**目标**: 在不开 mixed-chunk 的情况下，优化 prefill/decode 交替频率

当前 chunked_prefill_size=16384，每次 prefill 4 个请求 (4×4096)，耗时 ~2s。

| chunked_prefill_size | 每轮 prefill 请求数 | Prefill 耗时 (估) | Decode 空窗 | 总 prefill 轮数 (64 prompts) |
|:---:|:---:|:---:|:---:|:---:|
| 4096 | 1 | ~0.5s | 短但频繁 | 64 轮 |
| 8192 | 2 | ~1s | 中等 | 32 轮 |
| **16384** (当前) | **4** | **~2s** | **长** | **16 轮** |
| 32768 | 8 | ~4s | 很长 | 8 轮 |

**操作**: 分别试 `--chunked-prefill-size 8192` 和 `--chunked-prefill-size 4096`，对比吞吐。

更小的 chunk → 更短的 decode 中断 → 更好的 ITL → 可能更好的 E2E 吞吐

**注意**: 如果 TODO 1 成功 (mixed-chunk)，这个就不需要了。

---

## TODO 3: 提交 schedule_policy.py 修复 ~~(必做)~~ ✅ 已完成

已提交两个 commit:
- `5f15edae` fix: use full pool capacity for hybrid model decode admission budget
- `a85c1ba7` feat: add force_dequant for q_proj to bypass blockwise kernel

---

## TODO 4: 验证 SWA eviction 时机 (低优先级)

**现象**: Prefill 后 SWA token 不立即释放 (swa_usage 从 0.49→0.88)，限制了 ramp-up 速度。理论上 18 running 时只需 18×256=4608 SWA tokens (11%)，但 prefill 暂态会占到 88%。

**调查方向**: SWA eviction 在 `schedule_batch.py` 中的触发时机。如果能在 prefill→decode 转换时立即 evict，可以更快地 admit 新请求。

**优先级低**: 这主要影响 ramp-up 速度和 burst 场景，对稳态 decode 吞吐影响不大。

---

## TODO 5: 尝试更大 batch size (低优先级)

**背景**: 目前稳定在 18 running。昨晚测试显示 bs=16 是峰值 (422.9 tok/s)，bs=32 略降 (405.4)。但那是不同的 server 配置 (可能没有 schedule_policy 修复)。

**操作**: 用 `--max-running-requests 32` 强制允许更大并发，看吞吐是否进一步提升。

**预期**: 可能不会提升 (compute-bound)，但值得确认。

---

## TODO 6: 全面性能 Benchmark (高优先级)

**目标**: 在限定条件下建立完整的性能曲线，覆盖 prefill 和 decode，4K 和 16K 场景。

### 4 个场景

| # | Input | Output | 测量重点 | Batch Sizes |
|---|:-----:|:------:|:--------:|:-----------:|
| 1 | 4096 | 1 | Prefill 吞吐 | 1, 4, 8, 16, 24, 32, 48, 64, 96, 128 |
| 2 | 4096 | 1024 | Decode 吞吐 | 1, 4, 8, 16, 24, 32, 48, 64, 96, 128 |
| 3 | 16384 | 1 | 长 Prefill 吞吐 | 1, 4, 8, 16, 24, 32 |
| 4 | 16384 | 1024 | 长 Decode 吞吐 | 1, 4, 8, 16, 24, 32 |

大 batch size 可能 OOM，脚本会自动检测 server 是否存活，OOM 后标记为 SERVER DOWN。

### 执行步骤

**Phase 1: 4K Benchmark** (当前 server 配置即可)

```bash
cd /workspace/sgl-jax
nohup bash benchmark/mimo_bench_suite.sh 4k > /tmp/bench_4k.log 2>&1 &
# 用 nohup 防止 kubectl 超时断连
```

脚本会自动：
1. 先对所有 batch size (1~128) 做 JIT warmup（短请求触发编译，不计入结果）
2. 再逐个跑正式 benchmark

**Phase 2: 重启 server (context-length 改为 32768)**

场景 3/4 的 16K input + 1K output = 17408 tokens，超过当前 context_length=16384。

```bash
# 4 个 pod 都要重启，只改 --context-length
python -m sgl_jax.launch_server \
    ... (其他参数不变) \
    --context-length 32768       # 16384 → 32768
```

注意：context-length 增大后，max_running_requests 可能变小（KV cache 按更大 context 预留），
16K 场景的 batch size 测到 32，更大的可能 OOM。

**Phase 3: 16K Benchmark**

```bash
# 用 Phase 1 的同一个 RESULT_DIR，合并结果
RESULT_DIR=/tmp/mimo_bench_XXXXXXXX nohup bash benchmark/mimo_bench_suite.sh 16k > /tmp/bench_16k.log 2>&1 &
```

**Phase 4: 生成报告**

```bash
RESULT_DIR=/tmp/mimo_bench_XXXXXXXX bash benchmark/mimo_bench_suite.sh report
```

### JIT 编译 Warmup 机制

`--disable-precompile` 下每遇到新的 padded batch size 会触发 JIT 编译（30-60s）。
如果不 warmup，编译时间会计入 benchmark 结果导致不准。

脚本的解决方案：
- 正式 benchmark 前，对每个 batch size 发短请求 (input=128, output=1) 触发编译
- 编译结果缓存在 JAX 进程中，后续同 batch size 直接复用
- warmup 请求不计入结果

### 脚本

- `benchmark/mimo_bench_suite.sh` — 自动化 bench_serving 跑全部场景 + 生成汇总表
- 每个 batch size 用 `--request-rate 1000 --flush-cache` 全量瞬发
- 结果存到 `/tmp/mimo_bench_<timestamp>/`

### Decode-Only 吞吐 (补充)

除了上述 4 个场景，还可以用短 input 隔离测 decode 性能，评估 PD 分离收益：

```bash
# 短 input 近似 decode-only
python benchmark/decode_bench.py \
    --host 127.0.0.1 --port 30271 \
    --batch-sizes 1,4,8,16,24,32 \
    --input-len 128 --output-len 2048
```

---

## 实验优先级

1. ~~**TODO 3**~~ ✅ 已完成
2. **TODO 6** — 全面性能 Benchmark（4K + 16K，prefill + decode）
3. **TODO 1** — 试 `--enable-mixed-chunk` (最大潜在收益)
4. **TODO 2** — 如果 TODO 1 不行，试调 chunked_prefill_size
5. **TODO 5** — 如果 TODO 1 成功，试更大 batch
6. **TODO 4** — 有时间再调查
7. **TODO 7** — MMLU-Pro Eval 准确率验证

---

## TODO 7: MMLU-Pro Eval (最低优先级)

**目标**: 验证当前代码在 MMLU-Pro 上的准确率，确认 attention 实现正确性

**前提**: 需要 server 的 context-length 足够大，避免 thinking 截断导致准确率虚低。

### Server 配置

```bash
# 用尽可能大的 context-length，给 thinking 留足空间
python -m sgl_jax.launch_server \
    ... (其他参数不变) \
    --context-length 65536 \
    --mem-fraction-static 0.80
```

注意：context-length=65536 会大幅减少 KV cache 可用 token 数，max_running_requests 变小。
如果 OOM，降到 32768 (保证 max_new_tokens=32000 不截断)。

### 执行

Eval 脚本和数据在 pod 上已有（来自上次 overnight 实验）：

```bash
# MMLU-Pro (493 题，temp=0.6，max_tokens=32000)
# concurrency=2 避免 KV cache 不够（context-length 大了之后可用 token 更少）
python3 /tmp/eval_mmlu_pro_v4.py /tmp/mmlu_pro_500.json 0 32000 2 0.6 \
    2>&1 | tee /tmp/mmlu_pro_eval_$(date +%Y%m%d).txt
```

### 关键参数

| 参数 | 值 | 说明 |
|------|:--:|:-----|
| temperature | 0.6 | 比 chat 推荐的 0.8 低，减少推理循环 |
| max_new_tokens | 32000 | 给 thinking 留 32K 空间 |
| context-length | 65536 (或 32768) | 防止 server 端截断 |
| concurrency | 2 | 低并发避免 OOM |

### 预期

- 上次结果：非截断准确率 81-91%（对标官方 84.9%）
- 截断率是主要影响因素：temp=0.6 + max_tokens=32K 下约 16%
- 如果 context-length 设到 65536 且 max_tokens 足够大，截断率应进一步降低

### 注意事项

- 这个测试耗时长（493 题 × 平均 30s/题 ≈ 4-5 小时）
- 需要 server 一直保持稳定运行
- 放在所有 benchmark 之后跑，优先级最低
- 如果 eval 脚本不在 pod 上了（pod 可能重建过），需要重新上传

---

## 硬件约束提醒

- TPU v6e-16: 4 pods × 4 chips = 16 chips, 31.25 GB HBM/chip
- TP=16, EP=16 (模型必须用 16 chips)
- PD 分离需要 2x 硬件 (v6e-32)，当前不可行
- 模型权重 ~20 GB/chip (64% HBM)，剩余用于 KV cache
