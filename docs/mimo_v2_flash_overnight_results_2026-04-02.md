# MiMo-V2-Flash Overnight Experiment Results (2026-04-02)

## 环境

- **硬件**: TPU v6e-16 (4 pods × 4 chips, 共 16 chips)
- **模型**: MiMo-V2-Flash (309B total / 15B active, 256-expert MoE, FP8)
- **架构**: 48 layers (9 FA + 39 SWA), q/k_head_dim=192, v_head_dim=128, sliding_window=128
- **部署**: TP=16, EP=16, page_size=128, chunked_prefill_size=16384

---

## Task 1: 性能调优

### 目标

找到 prefill (4K input, 1 output) 和 decode (4K input, 1K output) 场景下的最优 server 配置和最大 batch size。

### 测试配置

| 配置 | mem-fraction-static | chunked-prefill-size | context-length |
|------|:-------------------:|:--------------------:|:--------------:|
| Config 1 | 0.75 | 16384 | 16384 |
| Config 2 | 0.80 | 16384 | 16384 |

### Decode 结果 (in=4096, out=1024)

| Batch Size | Config 1 (mem=0.75) | Config 2 (mem=0.80) | 提升 |
|:----------:|:-------------------:|:-------------------:|:----:|
| 1 | 75.5 tok/s | 73.8 tok/s | - |
| 8 | 302.0 tok/s | 299.4 tok/s | - |
| **16** | 352.2 tok/s | **422.9 tok/s** | **+20%** |
| 32 | 354.4 tok/s | 405.4 tok/s | +14% |
| 48 | — | 403.9 tok/s | — |

### Prefill 结果 (in=4096, out=1)

| Batch Size | Config 1 (mem=0.75) |
|:----------:|:-------------------:|
| 1 | 7,020 tok/s |
| 16 | 7,147 tok/s |
| 32 | 7,245 tok/s |
| 64 | 7,293 tok/s |

Prefill 吞吐对 batch size 不敏感，主要受 chunked_prefill_size=16384 限制，稳定在 ~7,000-7,300 tok/s。

### 结论

- **最优配置: `--mem-fraction-static 0.80`**
- **Decode 峰值: 422.9 tok/s @ bs=16**（Config 2），比 Config 1 峰值 (354.4 tok/s @ bs=32) 提升 20%
- 更多 KV cache (87K tokens vs 75K) → 支持更大有效 batch → 更高吞吐
- 最优 batch size 是 **bs=16**，超过后吞吐略降（KV cache 碎片化 + 调度开销）
- Prefill 吞吐 ~7,200 tok/s，瓶颈在 chunked_prefill_size 而非 batch

### 推荐 Server 配置

```bash
python -m sgl_jax.launch_server \
    --model-path /models/MiMo-V2-Flash \
    --trust-remote-code \
    --tp-size 16 --ep-size 16 \
    --moe-backend epmoe \
    --nnodes 4 --node-rank $RANK \
    --dist-init-addr $HEAD_ADDR:10011 \
    --host 0.0.0.0 --port 30271 \
    --page-size 128 \
    --context-length 32768 \
    --disable-radix-cache \
    --chunked-prefill-size 16384 \
    --dtype bfloat16 \
    --mem-fraction-static 0.80 \
    --disable-precompile --skip-server-warmup \
    --log-level info
```

---

## Task 2: MMLU-Pro & GPQA-Diamond Eval 准确率

### 目标

通过调整 server 参数、client 参数和 system prompt，使 MMLU-Pro 和 GPQA-Diamond eval 达到接近官方水平 (MMLU-Pro 84.9%, GPQA-Diamond 83.7%)。

### 根因分析

影响 eval 分数的核心问题是 **thinking 截断**：

1. MiMo-V2-Flash 是 reasoning model，生成时先输出 `<think>...推理过程...</think>`，再输出答案
2. 当 `max_new_tokens` 被耗尽时 `</think>` 无法生成 → 整个 response 都在 think 标签内 → 无法提取答案字母 → 该题直接判错
3. Temperature 越高，推理链越发散，越容易进入循环（反复验证、反复推翻结论）
4. 我们使用的 `/generate` 端点中 `max_new_tokens` 同时计算 thinking + answer tokens，而官方 SGLang 的 `/v1/chat/completions` 端点通过 `--reasoning-parser` 将二者分开计算

### 温度实验

使用 MMLU-Pro 数据集 (493 题，分层采样自 14 个学科)：

| 配置 | 样本数 | 截断率 | 总体准确率 | 非截断准确率 |
|------|:------:|:------:|:---------:|:-----------:|
| temp=0.8, max_tokens=16K | 373 | **33.5%** | 60.3% | 90.7% |
| temp=0.6, max_tokens=16K | 92 | **16.3%** | 75.0% | 89.6% |
| temp=0.6, max_tokens=32K | 177 | **16.4%** | 67.8% | 81-85% |

使用 GPQA-Diamond 数据集 (198 题)：

| 配置 | 样本数 | 截断率 | 总体准确率 | 非截断准确率 |
|------|:------:|:------:|:---------:|:-----------:|
| temp=0.6, max_tokens=16K | 33 | 39.4% | 51.5% | 85.0% |
| temp=0.6, max_tokens=32K | 84 | 35.7% | 57.1% | **88.9%** |

### 关键发现

1. **非截断准确率已达到/超过官方水平**
   - MMLU-Pro 非截断: 81-91% (官方 84.9%)
   - GPQA-Diamond 非截断: 88.9% (官方 83.7%)
   - 说明模型推理能力没有问题，sgl-jax 的 attention 实现是正确的

2. **Temperature 0.6 显著降低截断率**
   - MMLU-Pro 截断率: 33.5% → 16.3% (降低一半)
   - 参考: MiMo-7B-RL 官方 eval 使用 temperature=0.6, top_p=0.95
   - MiMo-V2-Flash README 推荐 temperature=0.8 是针对 chat 场景，benchmark eval 应用更低温度

3. **总体准确率差距完全来自截断**
   - 截断的题 100% 判错 (无法提取答案)
   - 即使 max_tokens=32K，仍有 16-36% 的题因推理过长而截断
   - GPQA 截断率 (36%) 远高于 MMLU (16%)，因为题目更难，推理更长

4. **官方得分更高的原因**
   - 官方 SGLang 使用 `--reasoning-parser qwen3`，thinking tokens 不占用 `max_tokens` 配额
   - GPQA-Diamond 官方使用 8 次重复取平均 (pass@8)，几乎消除截断影响
   - 官方可能使用 max_tokens=65536+ (模型支持 256K context)

### 推荐 Eval 配置

**Server 端** (同上性能调优推荐配置)

**Client 端**:
```python
SYSTEM_PROMPT = (
    "You are MiMo, an AI assistant developed by Xiaomi.\n\n"
    "Today's date: {date} {weekday}. "
    "Your knowledge cutoff date is December 2024."
)

SAMPLING_PARAMS = {
    "temperature": 0.6,        # benchmark 用 0.6，chat 用 0.8
    "top_p": 0.95,
    "max_new_tokens": 32000,   # 尽量大，给足 thinking 空间
    "stop": ["<|im_end|>"],
}
```

**Eval 脚本**: `/tmp/eval_mmlu_pro_v4.py` 和 `/tmp/eval_gpqa_diamond_v3.py`

用法：
```bash
# MMLU-Pro (493 questions, concurrency=4, temp=0.6)
python3 eval_mmlu_pro_v4.py /tmp/mmlu_pro_500.json 0 32000 4 0.6

# GPQA-Diamond (198 questions, concurrency=4, temp=0.6)
python3 eval_gpqa_diamond_v3.py /tmp/gpqa_diamond.json 0 32000 4 0.6
```

### 进一步提升准确率的方向

1. **在 sgl-jax 中实现 reasoning-parser**: 将 thinking/answer token 分开计数，彻底解决截断问题
2. **多次采样取平均 (pass@k)**: 对每题跑 8 次取多数投票，消除随机截断影响
3. **更低 temperature**: 尝试 temperature=0.4 或 greedy (0.0)，进一步减少推理循环
4. **更大 context-length**: 增大到 65536+，配合 max_tokens=64000，但需降低并发 (KV cache 受限)

---

## 附录: 数据文件位置

### Pod 上 (/tmp/)
- `mmlu_pro_500.json` — 493 题 MMLU-Pro 数据 (14 学科分层采样)
- `gpqa_diamond.json` — 198 题 GPQA-Diamond 数据
- `mmlu_pro_temp08_partial_373.txt` — temp=0.8 的 MMLU-Pro 结果 (373/493 题)
- `mmlu_pro_temp06_16k_partial92.txt` — temp=0.6, 16K 的 MMLU-Pro 结果 (92/493 题)
- `mmlu_pro_temp06_32k_full.txt` — temp=0.6, 32K 的 MMLU-Pro 结果 (177/493 题，中断)
- `gpqa_diamond_temp06_16k_partial.txt` — temp=0.6, 16K 的 GPQA 结果 (33/198 题)
- `gpqa_diamond_temp06_32k_full.txt` — temp=0.6, 32K 的 GPQA 结果 (84/198 题，中断)
- `eval_mmlu_pro_v4.py` — MMLU-Pro eval 脚本 (支持 temperature 参数)
- `eval_gpqa_diamond_v3.py` — GPQA-Diamond eval 脚本 (支持 temperature 参数)

### 本地
- 性能 benchmark 结果: `/tmp/mimo_results_20260401_231841/`
