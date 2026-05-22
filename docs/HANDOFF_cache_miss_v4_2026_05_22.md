# HANDOFF — Cache Miss v4 修复 (2026-05-22 22:00 CST)

## 一句话状态

`dev/fix-cache-miss-v3` 10 commits on `sgl-project/epic/mtp-refactor-phase1`。TRACING CACHE MISS 已全部清零（dp=1 和 dp=2）。剩余问题：PERSISTENT COMPILATION CACHE MISS — precompile 编译的 XLA artifact 与 runtime 的 persistent cache key 不匹配，导致冷启动首次请求需重新 XLA 编译。

---

## 分支 / Worktree

- **本地 worktree**：`/Users/niu/code/sglang-jax/.worktrees/dev-fix-cache-miss-v3`
- **分支**：`dev/fix-cache-miss-v3`
- **远端**：`origin/dev/fix-cache-miss-v3` @ `29602e58`
- **PR**: https://github.com/sgl-project/sglang-jax/pull/1188

## 完成的 commits（10 个）

```
29602e58 fix(spec): pin custom_mask per-rank size in dp>1 DP repacking
6c870f08 fix(spec): pin per_req_tokens to draft_token_num in _get_spec_decode_mwb_dp
86535c4e fix(spec): pin total_bs to max precompile bucket in _get_spec_decode_mwb_dp
a51d0627 fix(spec): pin build_tree_kernel max_context_len to max precompile bucket
49cb2c82 fix(spec): pad out_cache_loc to bucket in _get_spec_decode_mwb_dp + precompile
26d28d53 refactor(spec): unify dp=1 spec decode path with _get_spec_decode_mwb_dp
89a73653 fix(spec): pad logits in draft_extend_for_decode topk path
bc38af72 fix(spec): propagate spec_algorithm in get_model_worker_batch
3bfd7057 fix(spec): pad logits in capture_for_decode, gather to real_bs after jit
7e2d74cf fix(spec): pad spec_info_padded.verified_id in draft_extend_for_prefill
```

## 已修 vs 待修

### 已修 ✅

| 类型 | 验证 |
|------|------|
| TRACING CACHE MISS — spec decode 路径 shape 跳变（9 commits） | dp=1: 0 TRACING, dp=2: 0 TRACING |
| TRACING CACHE MISS — custom_mask dp>1 DP repacking shape 跳变 | dp=2: 316→0 TRACING |

### 待修 ❌

| 类型 | 数量 | 影响 |
|------|------|------|
| PERSISTENT COMPILATION CACHE MISS | dp=1: 234 after "fired up"，20 unique `jit_jitted_run_model` keys | 冷启动首次请求延迟 ~5-10s |

---

## PERSISTENT CACHE MISS 深度分析

### 已排除的假设

1. ❌ "ForwardBatch.init_new 路径未被 precompile 覆盖" — 实际 precompile 的 `forward_batch_speculative_generation` 确实走了 `ForwardBatch.init_new`
2. ❌ "compile_options 不同" — precompile 和 runtime 的 `num_replicas=1 num_partitions=4 device_assignment` 完全一致
3. ❌ "shape 没 pad 到 bucket" — TRACING CACHE MISS 为 0 证明 JIT trace 层面 shape 匹配

### 确认的事实

1. **20 个不同的 `jit_jitted_run_model` persistent cache key** 出现在 runtime（在 "fired up" 之后），全都不在 precompile 期间的 key 列表中
2. **0 TRACING CACHE MISS** — JIT trace cache 命中，说明 abstract shape 和 pytree structure 一致
3. **persistent cache key = hash(XLA HLO + compilation_options)**。JIT trace 相同 → HLO 生成的源码相同，但 persistent cache key 不同 → **XLA HLO 的具体内容不同**
4. 这可能因为 JIT 函数内部有 **data-dependent constant folding** 或 **autotuning hints** 被编入了 HLO

### 实验结果

```
# 不重启 server，连续发请求：
/generate 请求：794 → 794（0 新 miss）
/generate 请求（不同长度）：794 → 794（0 新 miss）
/v1/chat/completions 请求：794 → 822（28 新 miss）
```

**chat endpoint 产生新 miss**，因为 chat template 改变了 token 结构。不同 endpoint 走的代码路径在 tokenizer/chat template 层面不同，导致 ForwardBatch 的某些字段内容不同 → XLA HLO 中的常量不同 → persistent cache key 不同。

### 下一步方向

1. **对比 precompile 和 runtime 的 HLO**：在 `jit_jitted_run_model` 编译前 dump HLO，对比 precompile 版本和 runtime 版本的差异
2. **检查 `ForwardBatch` 哪些字段影响 HLO**：`spec_info`、`capture_hidden_mode`、`forward_mode` 等 non-shape 属性可能作为常量被编入 HLO
3. **考虑 XLA persistent cache 的 key 计算逻辑**：可能包含了 autotuning 结果或 Pallas kernel 版本信息
4. **评估是否值得修**：这些 persistent miss 每个只需 0.02-0.19s（基础算子）或 25s（`jit_jitted_run_model`，但有 disk cache fallback），且第二次相同请求就 hit

---

## Pod 状态

- `niu-v6e4-sleep` — 4 chip v6e，dp=1 server 正在运行，evalscope 已跑完
- `niu-mimo-v16` (tpu-v6e-16-xc) — 16 chip v6e-16 job，可用（MiMo spec decode 因 HBM 不足无法启动）

## 关键文件

| 文件 | 用途 |
|------|------|
| `schedule_batch.py` | `get_model_worker_batch` / `_get_spec_decode_mwb_dp` — batch 构造和 padding |
| `flashattention_backend.py` | `get_eagle_forward_metadata` — custom_mask DP repacking (已修) |
| `eagle_worker.py` | `precompile_spec_extend` / `precompile_spec_decode` — precompile 入口 |
| `forward_batch_info.py` | `ForwardBatch.init_new` — numpy→device array 转换 |
| `model_runner.py:205-252` | `jitted_run_model` / `run_model_wrapper` — JIT 模型 forward |
| `jax_utils.py:199` | `device_array` — `jax.make_array_from_callback` |
| `compilation_manager.py` | `_make_dummy_batch` — precompile 用的 dummy batch |

## 验证配置

```bash
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
export JAX_DEBUG_LOG_MODULES=jax._src.compiler,jax._src.interpreters.pjit
export JAX_EXPLAIN_CACHE_MISSES=1
```

清 cache：`rm -rf /tmp/jax_cache /tmp/xla_dump /tmp/jit_cache ~/.cache/jax && mkdir -p /tmp/jax_cache`

通过标准：**"fired up" 之后 0 PERSISTENT COMPILATION CACHE MISS + 0 TRACING CACHE MISS**

## 计划文档

- `docs/plan_cache_miss_v4_2026_05_22.html` — 诊断测试矩阵 + 修复方案
- 测试脚本 `/tmp/diag_cache_miss.py` 在 pod 上

## 新线程启动 prompt

> 继续 cache miss v4 修复。读 `.worktrees/dev-fix-cache-miss-v3/docs/HANDOFF_cache_miss_v4_2026_05_22.md`。
>
> 当前进度：10 commits 已 push，TRACING CACHE MISS 清零。剩余：PERSISTENT COMPILATION CACHE MISS — precompile 编译了 530 个 XLA artifact，但 runtime 产生了 234 个新 miss（key 不同）。20 个 `jit_jitted_run_model` miss 意味着模型 forward 被重新 XLA 编译。
>
> 已确认：compile_options 一致、shape pad 到 bucket（0 TRACING）、precompile 走了 ForwardBatch.init_new。问题在于 precompile 的 HLO 和 runtime 的 HLO 不同（可能因为 ForwardBatch 的某些 non-shape 属性作为常量编入了 HLO）。
>
> 下一步：dump 对比 precompile vs runtime 的 XLA HLO，找出具体哪个字段/常量导致 key 不同。Pod `niu-v6e4-sleep` 可用。
