# MiMo V2 Flash Eval Golden Baseline (2026-03-25)

## Scope

This document records the current end-to-end quality baseline for `mimo-v2-flash` so future implementation changes can be compared against a fixed reference.

## Code Baseline

- Branch: `feat/mimo-v2-flash`
- Commit: `d62c6ace922b2501e1bc38ee1e513478b0e6c666`
- Note: local temp eval scripts used for this run were untracked helper files and did not change repository code.

## Remote Runtime

- Cluster: `mimo-v6e16-min-0325b`
- Device: spot `tpu-v6e-16`
- Zone: `us-central1-b`
- Observed spot price during provisioning: about `$9.14/h`
- Model path: `/models/MiMo-V2-Flash`
- Service endpoint: `http://127.0.0.1:30271/v1`

## Prompting / Sampling

Official MiMo system prompts were used.

English:

```text
You are MiMo, an AI assistant developed by Xiaomi.

Today's date: 2026-03-25 Wednesday. Your knowledge cutoff date is December 2024.
```

Chinese:

```text
你是MiMo（中文名称也是MiMo），是小米公司研发的AI智能助手。

今天的日期：2026-03-25 星期三，你的知识截止日期是2024年12月。
```

Sampling settings:

- `top_p=0.95`
- `temperature=0.8` for math
- `temperature=0.3` for general QA / multiple choice smoke

## Service Health

The service remained healthy before and after the evals.

- `/get_model_info` -> `200`
- `/v1/models` -> `200`
- `/v1/completions` -> `200`
- `/v1/chat/completions` -> `200`

## Golden Results

### GSM8K

- Dataset: `gsm8k`
- Sample count: `10`
- Threads: `1`
- `max_tokens=256`
- Score: `0.7000`

Interpretation:

- This is a strong enough signal that the main reasoning/generation path is no longer in the earlier broken/garbled state.

### MGSM (Chinese)

- Dataset: `mgsm_zh`
- Sample count: `20`
- Threads: `1`
- `max_tokens=256`
- Score: `0.5000`

Interpretation:

- Chinese math capability is present and materially better than the earlier broken state.
- This is weaker than the `gsm8k` smoke, so Chinese math should still be treated as an area worth tracking.

### Chinese QA Suite

A small manual Chinese QA suite was run against the live service.

- Case count: `6`
- Rule-based score: `5/6` = `0.8333`
- Human judgment: effectively `6/6`

Case summary:

1. `中国的首都是哪里？只回答城市名。`
   - Output: `北京`
   - Result: pass
2. `用一句中文解释天空为什么是蓝色的。`
   - Output explained that shorter-wavelength blue light is more easily scattered by air molecules.
   - Rule-based script marked this as fail because it required a narrower keyword match.
   - Human judgment: pass
3. `请用中文简要说明 TCP 和 UDP 的一个主要区别。`
   - Output correctly described reliability / connection differences.
   - Result: pass
4. `写一个 Python 函数，输入一个整数列表，返回其中最大值。`
   - Output gave a valid Python function using `max()`.
   - Result: pass
5. `法国的首都是哪里？只回答城市名。`
   - Output: `巴黎`
   - Result: pass
6. `请用中文一句话说明二分查找为什么要求输入序列有序。`
   - Output correctly explained ordered data enables halving the search space.
   - Result: pass

Interpretation:

- General Chinese QA quality looks broadly normal.
- No garbled-token behavior was observed in this suite.

## MMLU Caveat

`mmlu` remained an outlier in this run.

- `mmlu(10)` with strict final-line answer formatting still scored `0.0000`

This should not be treated as proof that the whole model is still broadly broken. Raw inspection showed mixed failure modes:

- some samples were truncated before the final answer line
- some samples produced clearly wrong options
- at least one inspected sample produced a semantically plausible answer that did not match the dataset label / expected option for that item

Current interpretation:

- the main service path is substantially recovered
- `mmlu` is still not a trustworthy headline metric for this branch without further investigation

## Practical Baseline Conclusion

For future A/B comparisons, this run should be treated as the current golden baseline:

- service availability: healthy
- English math smoke: good (`gsm8k(10)=0.7`)
- Chinese math smoke: usable but weaker (`mgsm_zh(20)=0.5`)
- Chinese general QA: broadly normal (`5/6` rule-based, `6/6` by human reading)
- MMLU: still anomalous and not yet suitable as the sole quality gate

## Suggested Comparison Targets For Future Changes

If a later implementation changes attention, quantized linear, KV storage, or block-scale loading, compare against this baseline on at least:

1. service health (`/get_model_info`, `/v1/models`, `/v1/completions`)
2. `gsm8k(10)`
3. `mgsm_zh(20)`
4. the 6-case Chinese QA suite
5. optional `mmlu` smoke, but only as a secondary signal
