# v3 Baseline Cache Miss Report

**Date**: 2026-05-21
**Branch**: `dev/fix-cache-miss-v3` @ `ed97c63f` (基于 `origin/dev/spec-info-option-c`)
**Pod**: `niu-v6e4-sleep` (v6e 4 chip)
**Model**: Llama-3.1-8B-Instruct + EAGLE3-LLaMA3.1-Instruct-8B
**Config**: `--precompile-bs-paddings 1 4 8 16 --precompile-token-paddings 256 1024 2048 4096 --max-prefill-tokens 4096 --max-running-requests 16 --page-size 64 --attention-backend fa --dtype bfloat16`
**Method**: 文本 prompt + `max_new_tokens=1`（纯 prefill）+ cold start（清 `/tmp/jax_cache/*`）

## 测试矩阵覆盖

| dp | cells | bsz × seqlen 约束 |
|---|---|---|
| 1 | 50 | bsz ∈ 1-16，seqlen ∈ {128, 256, 500, 1024, 1500, 2048, 3000, 4096}，bsz×seqlen ≤ 4096 |
| 2 | 24 | bsz ∈ {2,4,6,8,10,12,14,16} （偶数），同 seqlen，同约束 |
| **合计** | **74** | |

## 结果速览

| 指标 | dp=1 | dp=2 |
|---|---|---|
| Cell 数 | 50 | 24 |
| Cell 全部跑通 | ✅ 50/50 | ✅ 24/24 |
| 0 crash | ✅ | ✅ |
| 多 req first_token 一致性 | ✅ 50/50 | ✅ 24/24 |
| dp=1 vs dp=2 same (seqlen,bsz) first_token match | — | ✅ 24/24 |
| **TRACING CACHE MISS** | **64** | **42** |
| PERSISTENT MISS（cold start 落盘） | 228 | (未单独统计) |

**正确性结论**：option C 重构完整解决了 dp>1 多 req 的 OOO + IndexError + filter_batch crash 问题。

## TRACING CACHE MISS 分布

### dp=1（64 trace）

| trace 位置 | 计数 | jit 函数 |
|---|---|---|
| `model_runner.py:252` | 40 | `jitted_run_model` |
| `eagle_draft_worker.py:285` | 24 | `topk_probs_from_logits` (`eagle_draft_worker.py:526`) |

### dp=2（42 trace）

| trace 位置 | 计数 | jit 函数 |
|---|---|---|
| `model_runner.py:252` | 26 | `jitted_run_model` |
| `eagle_draft_worker.py:285` | 16 | `topk_probs_from_logits` (`eagle_draft_worker.py:526`) |

**两种 dp 下 trace 位置完全一致**，dp_size 不影响 trace 类型。

## Root Cause 分析

### Trace 1：`jitted_run_model` (40 + 26 = 66 个)

closest-key 报告：
```
at forward_batch[12][3], now i32[1] and before i32[2]
at forward_batch[12][3], now i32[3] and before i32[1]
at forward_batch[12][3], now i32[3] and before i32[2]
at forward_batch[12][3], now i32[4] and before i32[1]
...
```

`forward_batch[12]` 是 spec_info（pytree leaf index 12 是 spec_info_padded），`[3]` 是 spec_info 的第 4 个字段（很可能是 `verified_id` 或 `allocate_lens`）。

**问题**：spec_info 字段在 forward 时长度 = `real_bs`（不是 padded `bucket_size`），每个独立 `real_bs` 触发一次 trace。

### Trace 2：`topk_probs_from_logits` (24 + 16 = 40 个)

closest-key 报告：
```
at logits, now bf16[1,32000] and before bf16[16,32000]
at logits, now bf16[2,32000] and before bf16[4,32000]
at logits, now bf16[3,32000] and before bf16[8,32000]
...
```

`capture_for_decode` 调用 `topk_probs_from_logits(logits)`，logits shape 是 `(real_bs, vocab)` 而不是 padded。每个独立 `real_bs` 一次 trace。

### 总结

**两个 trace 都是同类问题**：spec 路径上的 array shape 是 `real_bs` 而不是 `padded_bs`（bucket-sized），导致每个独立 `real_bs` 触发一次 jit trace。

Option C 把 spec_info 改成 per-rank 持久化，但 forward 进入路径（concat + scatter to dp-padded）后仍然存在 `real_bs` shape 的字段（如 `verified_id`），没有 pad 到 bucket。

## Step 3 优先级（数据驱动）

按 trace 数从高到低排：

1. **#1 修 `forward_batch[12][3]` (spec_info 字段) shape 跟随 real_bs 切换** — 影响 66 trace
   - 定位 `forward_batch[12][3]` 是 spec_info 的哪个字段
   - 在 forward 前 pad 到 `padded_bs`（与 seq_lens 同 bucket）
   - forward 后视情况 slice 回 real_bs（避免污染 per-rank spec_info 持久化）

2. **#2 修 `topk_probs_from_logits` 的 logits 输入 shape 切换** — 影响 40 trace
   - `capture_for_decode` 调用前把 logits pad 到 padded_bs
   - 调用后 slice 回 real_bs（仅取 sel-indexed entries 作为 spec_info）

两个 fix 可能耦合（spec_info.verified_id 来自 logits 的 argmax），改 #2 时可能顺便 cover #1 的一部分。先 #1 再 #2，每个独立 commit + 全 sweep 双 dp 验证。

## 借鉴池（旧 dev/fix-spec-precompile-dp）

旧分支 Phase 4.10 / 4.11 fix 思路适用：
- 4.10：spec_info host-manage + numpy（option C 后**不再需要**，per-rank 持久化已避免 device-side mutation）
- 4.11：`draft_extend_for_prefill` pad `verified_id` 到 bucket，forward 后 restore real_bs —— **思路适用 Trace #1**，但实现细节不同（option C 后 spec_info per-rank，pad 操作在 forward 进入边界 concat 之后）

## 测试数据存档

- `baseline_data/baseline_dp1.jsonl` — 50 cell 详细结果
- `baseline_data/baseline_dp2.jsonl` — 24 cell 详细结果
- pod `/tmp/trace_dp{1,2}_sample.txt` — trace 详情样本

## 下一步

进入 **Step 3 Prefill cache miss 数据驱动逐个修**：
1. 先 fix #1（spec_info 字段 pad）
2. 全 sweep 双 dp 验证 trace 数下降到目标值（≥66 减少）
3. fix #2（logits pad）
4. 同样验证
5. Step 4 最终验收：0 TRACING WARN + 0 crash + cold start 0 PERSISTENT MISS
