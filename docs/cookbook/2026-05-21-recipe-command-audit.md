# Cookbook Recipe Command Audit (2026-05-21)

> **文档性质**: 长期跟踪文档,记录"对每个 recipe 的 launch / benchmark 命令做源码级 review → fix → TPU 测试"全流程。和 [`mintlify-migration.md`](mintlify-migration.md)、[`cookbook-recipe-design.md`](cookbook-recipe-design.md)、[`2026-05-18-cookbook-research.md`](2026-05-18-cookbook-research.md) 一样属于 author tooling,**不被 cookbook 用户阅读**。
>
> **来源 goal**: "仔细 review 每个模型的 cookbook 文档,对比 sglang-jax 源码仔细核对,所有模型起 server 的命令、跑 benchmark 的命令是对的吗,如果有不对,请你指出来,后面我会要求你按照这个命令在 tpu 上跑一下并把文档中缺失的跑完的结果贴上去。"
>
> **核对基准**:
> - `python/sgl_jax/srt/server_args.py`(1367 行)— launch 真值
> - `python/sgl_jax/bench_serving.py`(3166 行)— benchmark 真值
> - `python/sgl_jax/srt/reasoning_parser.py` / `function_call_parser/function_call_parser.py` — parser key 真值
> - `python/sgl_jax/srt/multimodal/common/ServerArgs.py` — 多模态 flag
> - `python/sgl_jax/launch_server.py`、`PortArgs.init_new`、`scheduler.py` — 入口和 distributed init 行为

## 1. 总体结论

23 个 recipe 的 launch / benchmark 命令在严格 argparse 层面**全部能跑**——所有 flag 名、choices、tp/nnodes 整除、chunked_prefill/page_size 整除、parser key 注册等约束都通过。**没有会让 launcher 直接报错退出的命令**。但有若干**文档准确性 / 一致性 bug** 和**潜在跑不出预期结果的隐患**。

## 2. 双重确认通过的全局约束

- 所有 `--tp-size % --nnodes == 0` ✅(23 recipe 全部验证)
- 所有 `--chunked-prefill-size % --page-size == 0` ✅
- 所有 `--reasoning-parser` key 真实注册: `deepseek-r1` / `qwen3` / `mimo` / `kimi` / `glm45` ✅
- 所有 `--tool-call-parser` key 真实注册: `qwen25` / `qwen3_coder` / `mimo` / `glm47` / `glm45` ✅
- 所有 `--moe-backend` ∈ {`epmoe`, `fused`, `auto`} ✅
- 所有 `--attention-backend` ∈ {`native`, `fa`, `fa_mha`} ✅
- 所有 `--dtype bfloat16` ✅
- 所有 `--speculative-algorithm NEXTN` ∈ {EAGLE/EAGLE3/NEXTN/STANDALONE} ✅
- bench_serving `--random-input` / `--random-output` 短形 ✅(argparse `allow_abbrev=True` + 源码 docstring 用同名)
- bench_serving `--backend sgl-jax` 在 `ASYNC_REQUEST_FUNCS` 中 ✅
- `--multimodal` 正确触发 multimodal entrypoint ✅
- Wan `--precompile-width-heights` 格式 `WIDTH*HEIGHT` ✅(`__post_init__` 校验)
- MultimodalServerArgs `disable_radix_cache` 自动 True ✅

## 3. 发现的问题清单

### 🔴 P0 — 实际命令错误,会影响运行行为

| ID | 文件 | 问题 | 修复方案 | 状态 |
|---|---|---|---|---|
| B2 | `Grok/Grok2.md` L192, 204 | `--eval-type openai_api`(其他 recipe 全用 `service`) | 改为 `--eval-type service` | ✅ fixed 2026-05-21 |
| B4 | `Qwen/Qwen.md` §4.2 / `Qwen/Qwen3.md` §4.2 driver | bench_serving 命令漏 `--tokenizer` | 加 `--tokenizer <repo>`,Qwen3 driver 用 `${MODEL_NAME}` | ✅ fixed 2026-05-21 |
| C1 | `Qwen/Qwen.md` §3.1 curl / §3.1 python / §4.1 evalscope | `model=Qwen-7B-Chat` 缺 vendor 前缀,与 `--model-path Qwen/Qwen-7B-Chat` 不匹配 | 统一改为 `Qwen/Qwen-7B-Chat` | ✅ fixed 2026-05-21 |

### 🟡 P1 — 跑得起来但有 review 价值

| ID | 文件 | 问题 | 修复方案 | 状态 |
|---|---|---|---|---|
| B1 | `Qwen/Qwen.md` L62-68 | 单节点 launch 加 `--dist-init-addr 0.0.0.0:10011 --nnodes 1 --node-rank 0`,note 写"required for single-host"但源码确认 `nnodes=1` 时 jax.distributed 不初始化 | 删除 trio + note | ✅ fixed 2026-05-21 |
| B3 | `Wan/Wan-2.x.md` L94, 117 | `--vae-tiling` flag 是 store_true 但默认 True → 命令行 no-op | 从 launch 命令删除该 flag,Configuration Tips 改写说明 "tiling is on by default; no `--no-vae-tiling` to disable" | ✅ fixed 2026-05-21 |
| C2 | `Xiaomi/MiMo-V2.5-Pro.md` L91 | v6e-64 launch 有 `--max-prefill-tokens 16384 --max-seq-len 4096`,两者都是默认值 | 删除冗余 flag | ✅ fixed 2026-05-21 |

### 🟢 P2 — 可选优化

| ID | 文件 | 问题 | 修复方案 | 状态 |
|---|---|---|---|---|
| C3 | `Xiaomi/MiMo-V2.5-Pro.md` L247 | Speculative §2.4 写 `<draft-checkpoint>` 占位,用户无法直接抄 | 实测后填入 MiMo 模型卡里 NEXTN 的真实路径 | ⏳ pending(可结合 TPU 测试做) |
| C4 | `Wan/Wan-2.x.md` §2.1 / §2.4 | Wan 2.2 text encoder 调度在 CPU(`wan2_2_stage_config.yaml` `device_kind: cpu`),recipe 未点明 | Hardware Matrix 加备注:Wan 2.2 UMT5 跑 CPU,`--tp-size` 只覆盖 DiT + VAE | ⏳ pending |
| C5 | `Wan/Wan-2.x.md` L120 | `--text-encoder-precisions bf16` HBM 收益叙述对 Wan 2.2 (CPU encoder) 不适用 | 按 variant 拆分(Wan 2.1 TPU encoder / Wan 2.2 CPU encoder) | ⏳ pending |
| C6 | `cookbook-recipe-design.md` L349 | 设计 doc 列了不存在的 `--mm-attention-backend`(只在 upstream sglang 有) | 从设计 doc 删除或换为真实 flag | ⏳ pending |

## 4. 修复日志

### 2026-05-21 — 批次 1: P0 + P1 fix

- **B2 — Grok2 `--eval-type openai_api` → `service`**(2 处:§4.1 GSM8K + §4.1 GPQA Diamond)
  - Files: `autoregressive/Grok/Grok2.md`
- **B4 — bench_serving 加 `--tokenizer`**
  - `autoregressive/Qwen/Qwen.md` §4.2: `--tokenizer Qwen/Qwen-7B-Chat`
  - `autoregressive/Qwen/Qwen3.md` §4.2 driver script: `--tokenizer "${MODEL_NAME}"`(所有"adapt Qwen3.md driver"的下游 recipe 跟着受益)
- **C1 — Qwen.md `model` 字段补全 vendor**
  - §3.1 curl: `Qwen-7B-Chat` → `Qwen/Qwen-7B-Chat`
  - §3.1 python: `model="Qwen-7B-Chat"` → `model="Qwen/Qwen-7B-Chat"`(两处)
  - §4.1 evalscope: `--model Qwen-7B-Chat` → `--model Qwen/Qwen-7B-Chat`
- **B1 — Qwen.md 删 `--dist-init-addr` trio + note**
  - 删除第 62 行的 `--dist-init-addr 0.0.0.0:10011 --nnodes 1 --node-rank 0 \`
  - 删除第 68 行的误导 note(关于"always initializes JAX distributed")
- **B3 — Wan `--vae-tiling` 描述更正**
  - 从 Wan 2.2 launch 命令删除 `--vae-tiling` 行(因为是 no-op)
  - Configuration Tips bullet 改写: "VAE tiling 默认 ON,源码当前没有 `--no-vae-tiling` 反向 flag 可关"
  - Troubleshooting "OOM during VAE decode" 一行的 fix 从 "Enable `--vae-tiling`" 改为 "tiling is already on by default; lower request size"
- **C2 — MiMo-V2.5-Pro v6e-64 删冗余 default**
  - 第 91 行 `--chunked-prefill-size 4096 --max-prefill-tokens 16384 --max-seq-len 4096 \` → `--chunked-prefill-size 4096 \`

## 5. 测试记录

> **调试约定 — 参数怀疑时的查证顺序**: 跑模型时如果对任何 launch flag / benchmark 命令的取值/行为/默认有怀疑(例:某个 flag 不生效、某 parser key 对不上、某默认值不对、某 multimodal 行为意外),**优先查 HuggingFace 模型卡 + sglang-jax 源码确认**,而不是按 cookbook 文字推测。具体顺序:
>
> 1. **HuggingFace model card** — 模型的 chat template、tokenizer、推荐 generation 参数、量化格式、tool-call schema 等以 HF 卡为准
> 2. **sglang-jax 源码**(优先级如下) — 命令行 flag 真值在源码,不在 cookbook:
>    - `python/sgl_jax/srt/server_args.py`(`ServerArgs.add_cli_args`)— launch flag 名/默认/choices
>    - `python/sgl_jax/srt/multimodal/common/ServerArgs.py`(`MultimodalServerArgs.add_cli_args`)— 多模态扩展 flag
>    - `python/sgl_jax/srt/reasoning_parser.py`(`ReasoningParser.DetectorMap`)— reasoning parser key
>    - `python/sgl_jax/srt/function_call/function_call_parser.py`(`FunctionCallParser.ToolCallParserEnum`)— tool-call parser key
>    - `python/sgl_jax/srt/models/<model>.py` 的 `EntryClass = ...` 行 — 模型在仓内的注册名
>    - `python/sgl_jax/srt/managers/scheduler.py`、`PortArgs.init_new` — multi-node / dist init 行为
>    - `python/sgl_jax/bench_serving.py`(底部 `argparse` 块)— benchmark flag 名/默认/choices
> 3. **cookbook 文档(本仓 `docs/cookbook/`)** — 排在最后,因为 cookbook 落后于源码是常态;cookbook 与源码冲突时以源码为准(对齐 [`2026-05-18-cookbook-research.md` §2.5 以代码为准](2026-05-18-cookbook-research.md))
>
> **发现 cookbook 与源码/HF 不一致** → 加入 §3 问题清单(标 P0/P1/P2),fix 后写入 §4 日志,**禁止**在 cookbook 里写"猜测"的命令行。

> 后续在 TPU 上按各 recipe 的 launch / benchmark 命令实测时,在此追加记录。

### 模板(每次跑后填):

```
#### <Recipe 名> — <run-date>
- 命令: <copy-paste 实际命令>
- TPU: <e.g. v6e-4>
- Build: sglang-jax <commit-hash>
- 结果: <evalscope / bench_serving 完整输出片段>
- 异常: <若有>
- 文档 PR 链接: <把跑出的数据回填 recipe 的 PR>
```

### 测试 backlog(按优先级)

(按现有 ✅ Validated / 🚧 Starter 升级路径选)

- 🟡 **Starter 升级最有价值**(社区已经在用): Qwen3-MoE, DeepSeek-R1, GLM-4.5
- 🟡 **Multimodal 首次实测**: Qwen2.5-VL 7B, Wan 2.1 1.3B
- 🟢 验证 P0 fix 不破坏现有 ✅ Validated 跑通: MiMo-V2-Flash, MiMo-V2.5-Pro, Qwen3, Grok-2

## 6. 跨 PR 协调约定

- 任何动 cookbook recipe 的 PR(尤其 launch / benchmark 命令)**先 grep 本文**,看是否相关项已 fix
- 本文是 author tooling,**禁止在 user-facing recipe 里反向链接此文档**
- 后续发现新的命令 bug → 加到 §3 表里 → fix 后写 §4 日志
- TPU 实测后填 §5 — 给后续 reviewer 留下"在 X 配置上跑通过"的可信痕迹

## 7. 参考链接

- [`cookbook-recipe-design.md`](cookbook-recipe-design.md) — recipe 章节 schema
- [`2026-05-18-cookbook-research.md`](2026-05-18-cookbook-research.md) — 整体设计原则
- [`mintlify-migration.md`](mintlify-migration.md) — 站点迁移跟踪
- [`base/launch-flags-reference.md`](base/launch-flags-reference.md) — flag 速查(本审计后可能需同步更新)
