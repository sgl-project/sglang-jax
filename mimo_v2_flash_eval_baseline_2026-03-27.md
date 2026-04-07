# MiMo-V2-Flash Eval Baseline (2026-03-27)

## Scope

This document records the end-to-end quality and performance baseline for `MiMo-V2-Flash` after block-quant scale fixes (q_proj 3D scale, kv_head_padding fixes, kernel.py cleanup).

## Code Baseline

- Branch: `feat/mimo-v2-flash`
- Key changes since golden baseline (2026-03-25):
  - `weight_utils.py`: block-quant scale_inv forced through SLOW path for 2D→3D expansion; `_apply_kv_head_padding` 3 bug fixes; `_maybe_expand_linear_block_scale` per-channel handling
  - `kernel.py`: removed dead 2D branch, only 3D (block-quant) + 1D (per-channel) paths remain

## Runtime

- Cluster: `sky-d85e-jiongxuan`
- Device: TPU v6e-16 (4 nodes × 4 chips)
- TP=16, EP=16, MoE backend: `epmoe`
- Model path: `/mnt/model_ram/MiMo-V2-Flash`
- Server: `context-length=1024, max-total-tokens=2048, mem-fraction-static=0.7`
- Reasoning parser: `qwen3`

## Prompting / Sampling

English system prompt:
```
You are MiMo, an AI assistant developed by Xiaomi.
Today's date: 2026-03-25 Wednesday. Your knowledge cutoff date is December 2024.
```

Chinese system prompt:
```
你是MiMo（中文名称也是MiMo），是小米公司研发的AI智能助手。
今天的日期：2026-03-25 星期三，你的知识截止日期是2024年12月。
```

Sampling: `top_p=0.95`, `temperature=0.8` (math), `temperature=0.3` (QA/MC)

## Quality Results

### GSM8K

| Metric | Current (n=30, max_tokens=768) | Golden (n=10, max_tokens=256) |
|--------|-------------------------------|-------------------------------|
| Score  | **0.9000**                    | 0.7000                        |

### MGSM (Chinese)

| Metric | Current (n=30, max_tokens=768) | Golden (n=20, max_tokens=256) |
|--------|-------------------------------|-------------------------------|
| Score  | **0.9000**                    | 0.5000                        |

### Chinese QA Suite (6 cases)

| Case | Prompt | Result |
|------|--------|--------|
| capital_cn | 中国的首都是哪里？ | PASS (北京) |
| sky_blue_cn | 用一句中文解释天空为什么是蓝色的 | FAIL* (关键词匹配偏严) |
| tcp_udp_cn | TCP 和 UDP 的主要区别 | PASS |
| python_max_fn_cn | Python 最大值函数 | PASS |
| france_capital_cn | 法国的首都是哪里？ | PASS (巴黎) |
| binary_search_cn | 二分查找为什么要求有序 | PASS |

Score: **5/6** (human judgment: 6/6)，与 golden baseline 一致。

\* `sky_blue_cn` 回答正确但 `any_of` 关键词匹配未命中（回答用 "蓝光波长较短" 而非 "瑞利" 或 "短波+蓝光"）。

### MMLU

| Metric | Current (n=30, max_tokens=256) | Golden (n=10, max_tokens=256) |
|--------|-------------------------------|-------------------------------|
| Score  | **0.2667**                    | 0.0000                        |
| STEM   | 0.5000                        | -                             |
| Humanities | 0.1250                   | -                             |
| Social Sciences | 0.3750              | -                             |
| Other  | 0.1250                        | -                             |

MMLU 仍偏低，主要因 `context-length=1024` 限制了推理模型 chain-of-thought 输出。STEM 表现最好(0.5)。

## Performance Results (bench_one_batch_server)

Server config: precompile enabled, warmup completed.

| Batch Size | Input Len | Output Len | TTFT (s) | Latency (s) | Input Tput (tok/s) | Output Tput (tok/s) |
|-----------|-----------|-----------|----------|-------------|--------------------|--------------------|
| 1         | 128       | 64        | 0.04     | 1.02        | 3,111              | 65.7               |
| 1         | 512       | 64        | 46.53    | 47.77       | 11.0               | 51.6               |
| 16        | 128       | 64        | 2.25     | 3.62        | 910                | 749                |
| 16        | 512       | 64        | 58.08    | 59.32       | 141                | 827                |
| 64        | 128       | 64        | 10.56    | 12.55       | 775                | 2,063              |
| 64        | 512       | 64        | 145.33   | 146.57      | 225                | 3,297              |

**Note**: `input_len=512` TTFT 异常高 (46-145s)，疑为 precompile 缺少对应 shape 的编译缓存导致运行时重编译。`input_len=128` 的 TTFT 正常 (0.04-10.56s)。

## Key Observations

1. **Quality improvement**: GSM8K 和 MGSM_ZH 从 0.5-0.7 提升至 0.9，主要因 `max_tokens=768` 给了推理模型足够的 chain-of-thought 空间
2. **Block-quant fix verified**: q_proj scale 现在正确存为 3D `(32, 1, 12288)`，走 blockwise kernel 而非 dequant fallback
3. **KV head padding fixes verified**: k_proj/v_proj scale 正确展开到 `(3072, 32)` 后做 head replication
4. **Performance**: bs=64/input=128 下 decode throughput 2063 tok/s，是合理的 MoE 推理性能

## Files Modified (vs golden baseline)

| File | Change |
|------|--------|
| `python/sgl_jax/srt/utils/weight_utils.py` | block-quant scale_inv 强制 SLOW path; `_apply_kv_head_padding` 3 bug fixes; `_maybe_expand_linear_block_scale` per-channel handling |
| `python/sgl_jax/srt/kernels/quantized_matmul/kernel.py` | 删除 2D 分支，只保留 3D + 1D 路径；移除 `convert_block_scale_to_kernel_layout` import |
