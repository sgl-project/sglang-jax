# Step-3.5 GSM8K Completion 评测设计

## 问题

`test_step3p5_mtp_e2e.py` 将其 GSM8K 检查描述为上游 SGLang 测试的移植，
但两边当前测量的并不是同一个任务。上游使用 5-shot completion 评测，分数门槛为
0.83；本地测试却使用通用的 zero-shot chat-completion evaluator，同时期待
5-shot completion benchmark 的分数。

该差异与 speculative decoding 无关：本地 chat 评测在 spec 和 no-spec server
上都只有约 0.55，而现有 5-shot `/generate` benchmark 约为 0.9。因此，本地
accuracy failure 并不能证明 serving、speculative decoding、chunked prefill 或
并发路径存在回归。

## 目标

- 让 Step-3.5 MTP E2E GSM8K 测试与上游 SGLang 一样，测量 5-shot completion
  任务。
- 保持通用 zero-shot chat GSM8K 评测及其现有调用者的行为不变。
- 保留与 no-spec baseline 严格对比的可选检查。
- 用 CPU 测试覆盖 prompt 构造、请求路由和答案提取。

## 非目标

- 修改模型 chat template。
- 修改 speculative decoding 或 chunked prefill serving 代码。
- 替换 benchmark CLI 或修改其输出格式。
- 将所有 `run_eval` 评测切换为 completion 请求。

## 设计

为本地评测框架增加显式 completion 模式。`run_eval` 默认仍使用
`ChatCompletionSampler`；只有调用者传入 `api="completion"` 时，才构造
completion sampler，并使用调用者指定的 few-shot 数量配置 GSM8K evaluator。

Completion GSM8K prompt 采用仓库现有 benchmark 已使用的标准结构：

```text
Question: <示例问题>
Answer: <示例完整解答>

...

Question: <待测问题>
Answer:
```

Sampler 通过 OpenAI-compatible completion endpoint 发送该 prompt。Evaluator 从
模型响应中提取最后一个数字，与现有 benchmark 行为一致。原有 chat 模式继续使用
当前 instruction、`Answer:` parser 和 conversation report，不发生行为变化。

`test_step3p5_mtp_e2e.py` 显式设置 `api="completion"`、`num_shots=5` 和
`max_tokens=512`。没有 baseline 时，fallback 分数门槛改为 0.83，与上游
Step-3.5 测试一致；设置了 `SGLANG_GSM8K_BASELINE` 时，仍以更严格的
spec-versus-greedy 对比为准。

## 数据流

1. Step-3.5 测试构造带有 completion 模式和 5-shot 参数的 `run_eval` 参数。
2. `run_eval` 选择 completion sampler，并配置 `GSM8KEval`。
3. `GSM8KEval` 为每个 held-out 测试样本构造一个 5-shot prompt。
4. Completion sampler 将 prompt 发送至 server 的 completion API。
5. Evaluator 提取最终数字，与 GSM8K target 比较，并返回与当前相同结构的聚合
   metrics 和 report。

## 错误处理

网络重试仍由 sampler 层负责。单个样本评测异常时，继续产生空 prediction，使传输
失败降低分数，而不会被误判为正确答案。不支持的 `api` 值立即抛出带有清晰信息的
`ValueError`。

## 测试

CPU 测试需要证明：

- completion prompt 包含准确数量的 demonstrations，末尾紧跟 held-out 问题；
- completion 模式选择 completion endpoint，并转发确定性 generation 参数；
- completion 答案解析使用模型响应中的最后一个数字；
- 默认 GSM8K chat 模式行为保持不变；
- Step-3.5 E2E 测试使用 5-shot completion 模式及上游 fallback 门槛。

Focused tests 通过后，运行已注册的完整 CPU unit suite 和 pre-commit。TPU 只做
一次组合验收：同一 commit 上启动 Step-3.5 spec 和 greedy server，以并发 32
运行 200 道 5-shot completion GSM8K；同时复核 Problem 2 已有的 pool-idle 和
Weng 检查。

## 验收标准

- 未设置 `api` 的现有调用者继续使用 chat completion。
- Step-3.5 E2E 使用 5-shot completion，不再把已知的 zero-shot chat 低分误报为
  serving 回归。
- Spec 分数与传入的 greedy baseline 之差不超过配置 tolerance；未传入 baseline
  时，分数不低于 0.83。
- Problem 2 的 slot ownership 行为和测试不发生变化。
