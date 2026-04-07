# MiMo-V2-Flash 已知问题与测试记录

## Issue 1: 多节点 TPU 下 Radix Cache 命中 n-1 token 导致卡死

**状态**: 已测试通过（2026-03-27，v6e-16 4 nodes）

**现象**: 多节点 TPU (process_count > 1) 环境下，当 radix cache 命中了请求的前 n-1 个 token，剩余 extend_len=1 时，服务卡死无响应。

**触发条件**:
1. 多节点 TPU 运行环境（如 v6e-16, 4 nodes）
2. Radix cache 中已缓存某个 prefix
3. 新请求与缓存 prefix 仅差最后 1 个 token（`len(fill_ids) - len(prefix_indices) == 1`）
4. 此时 extend_len=1，被 decode fastpath 优化转换为 DECODE 模式

**根因分析**:
- `schedule_batch.py` 中 `use_decode_fastpath_for_extend` 将 extend_len=1 的 EXTEND 请求转为 DECODE 模式
- 但此时 prefix cache 已命中 n-1 个 token，请求状态处于 EXTEND 流程（有 prefix_indices）
- DECODE 模式下不会传递 `extend_prefix_lens` / `extend_seq_lens`，导致 KV cache 处理与实际状态不一致
- 多节点环境下此不一致可能导致跨节点同步卡死

**之前的 workaround**（已移除）:
```python
# schedule_batch.py — 检测到 radix cache 命中 n-1 token 时，强制清空 prefix cache
if (
    is_multinode_tpu_runtime()
    and DISABLE_MULTINODE_TPU_SINGLE_TOKEN_PREFIX_CACHE
    and len(self.fill_ids) - len(self.prefix_indices) == 1
    and len(self.prefix_indices) > 0
):
    self.prefix_indices = []
    root_node = getattr(tree_cache, "root_node", None)
    self.last_node = root_node
    self.last_host_node = root_node
    self.host_hit_length = 0
```

**测试结果** (2026-03-27, v6e-16 asia-northeast1-b, `--disable-precompile`):
- [x] 连续发送相同 prompt 3 次：全部正常，第 2/3 次 TTFT 0s（命中 radix cache）
- [x] 5 个并发相同 prompt 请求：全部正常，57s 内完成
- [x] n-1 token prefix cache 场景（先发长 prompt，再发同 prefix + 1 token）：正常，0s 完成
- **结论**: workaround 移除后问题未复现，可能已被其他修复解决。建议保持观察。

---

## Issue 2: 首次遇到新 token shape 时 TTFT 异常高

**状态**: 根因已确认——XLA JIT 编译延迟

**现象**: `--disable-precompile` 模式下，首次使用新的 input_len shape 时 TTFT 异常高（40-45s），第二次相同 shape 恢复正常。

**测试数据** (2026-03-27, v6e-16, `--disable-precompile`, batch_size=1):

| input_len | 首次 TTFT | 第二次 TTFT | 编译开销 |
|-----------|----------|------------|---------|
| 128       | 0.04s    | -          | 已有缓存 |
| 256       | 41.21s   | 0.06s      | ~41s    |
| 512       | 44.99s   | 0.11s      | ~45s    |

**根因**: `--disable-precompile` 跳过了启动时的 XLA kernel 预编译。运行时遇到新的 (batch_size, num_tokens) shape 组合需要即时 JIT 编译，每个新 shape 约 40-45s。

**precompile 配置**:
- 默认 token_paddings: `[64, 128, 256, 512, 1024, 2048, 4096, 8192]`
- 默认 bs_paddings: `[1, 2, 4, 8, 16, 32, 64, 128, 256]`
- 实际预编译范围受 `--max-prefill-tokens` 限制
- 实测预编译耗时: EXTEND 323s + DECODE 82s ≈ **7 分钟**

**解决方案**: 生产环境应开启 precompile（不加 `--disable-precompile`），启动时间增加约 7 分钟但运行时无编译延迟。

---

## Issue 3: MMLU 分数受 context-length 影响显著

**状态**: 已确认并验证修复方案

**现象**: MMLU 评测分数随 context-length 变化剧烈。

**测试数据** (2026-03-27, v6e-16, 20 examples, temperature=0.3, top_p=0.95):

| context-length | max_tokens | MMLU Score | Humanities | STEM | Social Sci | Other |
|---------------|------------|-----------|------------|------|------------|-------|
| 1024          | 256        | **0.250** | 0.00       | 0.25 | 0.60       | 0.14  |
| 4096          | 2048       | **0.650** | 0.75       | 0.50 | 1.00       | 0.43  |

**根因**: MiMo-V2-Flash 是推理模型，需要较长的 chain-of-thought 输出空间。context-length=1024 严重限制了 CoT 长度，导致模型无法充分推理。

**KV cache 内存影响**:
- context-length=1024: available_kv_cache=2.4GB, max_tokens=69487
- context-length=4096: available_kv_cache=2.9GB, max_tokens=85444
- context-length=32768: available_kv_cache=2.9GB, max_tokens=85444

**建议**: 生产环境 context-length 至少设为 4096，推荐 8192+ 以充分发挥推理能力。

---

## Benchmark 数据

**测试环境**: v6e-16 (asia-northeast1-b), TP=16, EP=16, precompile 开启, context-length=32768, max-prefill-tokens=8192, chunked-prefill-size=4096, mem-fraction-static=0.7

### batch_size=1

| input_len | output_len | TTFT (s) | Latency (s) | Input Tput (tok/s) | Output Tput (tok/s) | ITL (ms) |
|-----------|------------|----------|-------------|-------------------|--------------------|---------:|
| 128       | 64         | 0.12     | 3.91        | 1,091             | 16.87              | 59.3     |
| 128       | 256        | 0.09     | 15.77       | 1,483             | 16.33              | 61.3     |
| 128       | 512        | 0.08     | 32.16       | 1,512             | 15.96              | 62.7     |
| 512       | 64         | 0.24     | 4.31        | 2,122             | 15.74              | 63.5     |
| 512       | 256        | 0.17     | 16.89       | 3,072             | 15.30              | 65.3     |
| 512       | 512        | 0.17     | 34.42       | 3,069             | 14.95              | 66.9     |
| 1024      | 64         | 0.28     | 4.70        | 3,668             | 14.47              | 69.1     |
| 1024      | 256        | 0.27     | 18.43       | 3,790             | 14.09              | 71.0     |
| 1024      | 512        | 0.28     | 37.41       | 3,663             | 13.79              | 72.5     |
| 2048      | 64         | 0.34     | 5.48        | 6,107             | 12.44              | 80.4     |
| 2048      | 256        | 0.32     | 21.34       | 6,403             | 12.18              | 82.1     |
| 2048      | 512        | 0.33     | 43.19       | 6,143             | 11.95              | 83.7     |
| 4096      | 64         | 0.67     | 7.21        | 6,092             | 9.79               | 102.1    |
| 4096      | 256        | 0.67     | 27.40       | 6,085             | 9.58               | 104.4    |
| 8192      | 64         | 1.50     | 10.83       | 5,472             | 6.86               | 145.8    |
| 8192      | 256        | 1.52     | 39.54       | 5,378             | 6.73               | 148.5    |
| 16384     | 64         | 3.69     | 18.60       | 4,439             | 4.29               | 233.0    |
| 16384     | 256        | 3.69     | 64.32       | 4,439             | 4.22               | 236.8    |
| 20480     | 64         | 5.01     | 22.70       | 4,092             | 3.62               | -        |
| 20480     | 256        | 5.10     | 77.00       | 4,017             | 3.56               | -        |
| 30720     | 64         | 9.28     | 33.95       | 3,310             | 2.59               | 385.5    |
| 30720     | 256        | 9.20     | 109.32      | 3,341             | 2.56               | 391.1    |

### batch_size=4 (precompile 覆盖)

| input_len | output_len | TTFT (s) | Latency (s) | Input Tput (tok/s) | Output Tput (tok/s) | ITL (ms) |
|-----------|------------|----------|-------------|-------------------|--------------------|---------:|
| 128       | 64         | 0.17     | 4.59        | 3,009             | 57.96              | 69.0     |
| 128       | 256        | 0.17     | 19.18       | 3,006             | 53.88              | 74.2     |
| 128       | 512        | 0.17     | 41.11       | 3,065             | 50.02              | 80.0     |
| 512       | 64         | 0.31     | 5.81        | 6,696             | 46.55              | 85.9     |
| 512       | 256        | 0.31     | 23.65       | 6,683             | 43.87              | 91.2     |
| 512       | 512        | 0.31     | 49.92       | 6,707             | 41.28              | 96.9     |
| 1024      | 64         | 0.56     | 7.48        | 7,374             | 36.99              | 108.1    |
| 1024      | 256        | 0.56     | 29.65       | 7,377             | 35.20              | 113.6    |
| 1024      | 512        | 0.57     | 61.73       | 7,249             | 33.48              | 119.5    |
| 2048      | 64         | 1.19     | 10.94       | 6,905             | 26.24              | 152.5    |
| 2048      | 256        | 1.19     | 41.74       | 6,902             | 25.25              | 158.4    |
| 2048      | 512        | 1.18     | 85.35       | 6,934             | 24.33              | 164.4    |

### batch_size=16

注意: bs=16 超出了 precompile 的 bs_paddings=[1,2,4] 范围，首组 TTFT 含运行时编译开销。

| input_len | output_len | TTFT (s) | Latency (s) | Input Tput (tok/s) | Output Tput (tok/s) | ITL (ms) |
|-----------|------------|----------|-------------|-------------------|--------------------|---------:|
| 128       | 64         | 13.96    | 18.38       | 147               | 231.31             | 69.2     |
| 128       | 256        | 57.66    | 76.67       | 36                | 215.47             | 74.3     |
| 128       | 512        | 123.55   | 164.55      | 17                | 199.79             | 80.1     |
| 512       | 64         | 17.72    | 23.22       | 462               | 186.22             | 85.9     |
| 512       | 256        | 71.30    | 94.67       | 115               | 175.23             | 91.3     |
| 512       | 512        | 150.23   | 199.95      | 55                | 164.78             | 97.1     |
| 1024      | 64         | 22.97    | 29.89       | 713               | 147.92             | 108.2    |
| 1024      | 256        | 89.47    | 118.55      | 183               | 140.87             | 113.6    |
| 1024      | 512        | 185.77   | 246.86      | 88                | 134.09             | 119.3    |
| 2048      | 64         | 33.99    | 43.75       | 964               | 105.01             | 152.4    |
| 2048      | 256        | 126.39   | 166.95      | 259               | 100.99             | 158.4    |
| 2048      | 512        | 256.95   | 341.03      | 128               | 97.43              | 164.2    |

### 关键观察

1. **TTFT 随 input_len 线性增长**: bs=1 时 128→0.12s, 2048→0.34s, 16384→3.69s, 30720→9.28s
2. **Decode 吞吐 (ITL) 随 KV cache 长度增长而降低**: bs=1 时 128 token→59ms/tok, 30720 token→391ms/tok
3. **Batch 吞吐量线性扩展**: bs=1→16 tok/s, bs=4→58 tok/s, bs=16→231 tok/s (input_len=128, output_len=64)
4. **Precompile 对 TTFT 影响显著**: bs=4 (precompile 覆盖) TTFT=0.17s vs bs=16 (未覆盖) TTFT=13.96s
5. **长 context (30k) 可用但 decode 较慢**: 30k input 的 decode 速度约 2.6 tok/s (bs=1)
