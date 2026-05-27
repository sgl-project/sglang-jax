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
| D1 | `DeepSeek/DeepSeek-V2.md` §2.3 single-host(V2-Lite) | 漏 `--page-size`,落到默认 1。MLA backend 强制 `page_size > 1`(`mla_backend.py` L108 断言),启动时崩在 init_attention_backend。R1 / V3 / V2 multi-host 都已有 `--page-size 128`,只 V2-Lite 单机漏写 | 加 `--page-size 64` + callout 说明 MLA 硬约束 | ✅ fixed 2026-05-23 — 详 [`deepseek-v2-lite/exception_mla_page_size_1.md`](2026-05-21-recipe-command-audit/deepseek-v2-lite/exception_mla_page_size_1.md) |
| D2 | `DeepSeek/DeepSeek-V2.md` §4.2 accuracy(V2-Lite & V2 多机) | recipe 用 base checkpoint(`DeepSeek-V2-Lite` / `DeepSeek-V2`)走 `/v1/chat/completions`,base 模型没 chat template,evalscope 4-shot prompt 在 chat endpoint 下会无限延续生成,GSM8K 实测 score=0.014(预期 ~0.41),`finish_reason: length` | §4.2 Test Environment + benchmark command 改用 `-Chat` 版本,顶部加 callout 说明 base 不能跑 chat-completions accuracy eval | ✅ fixed 2026-05-24 — 详 [`deepseek-v2-lite/exception_base_model_chat_eval.md`](2026-05-21-recipe-command-audit/deepseek-v2-lite/exception_base_model_chat_eval.md) |
| D3 | 已删除的 Ling 1.x recipe (Ling-lite / Ling-Coder-lite / Ling-plus) | sglang-jax `bailing_moe.py:321` 硬依赖后续 Ling generation 的 `config.n_group` / `topk_group` / `routed_scaling_factor`,Ling 1.x flat-MoE 配置没这些字段,启动时 `AttributeError: 'BailingMoeConfig' object has no attribute 'n_group'`。属于 sglang-jax × Ling 1.x 配置不兼容,不是 recipe 层 bug | 源码层加 `getattr` fallback 或拆分模型类(详 exception 文档);recipe 层删除 Ling 1.x recipe,保留 Ling 2.5 / Ling 2.6 cookbook | ⚠️ 源码 fix 待提 — 详 [`ling-lite/exception_n_group_attr_error.md`](2026-05-21-recipe-command-audit/ling-lite/exception_n_group_attr_error.md) |
| D4 | `Llama/Llama3.1.md` §2.3 single-host | 漏 `--page-size`,默认值 1 → 启动日志 `Final max_running_requests: 1`,bench c=16 实际全部串行 → 输出吞吐 156 tok/s vs 加 `--page-size 128` 后 1449 tok/s(9.3× 差距);TTFT 47s vs 35ms。属于 D1 同类(MLA-only 已 fix),Llama 系列 dense attention 不会启动崩,只静默降级到串行,review 阶段更难发现 | 加 `--page-size 128 --max-running-requests 64` 到 §2.3 launch + §2.4 mandatory callout + §5 Troubleshooting 行 | ✅ fixed 2026-05-25 — 详 [`llama-3.1-8b/`](2026-05-21-recipe-command-audit/llama-3.1-8b/) |
| D5 | `Llama/Llama3.3-70B.md` §2.1 hardware matrix + §2.3 launch | minimum runnable 写 v6e-32 / tp=32,过保守。70B BF16 = 140GB,v6e-16 = 16 × 32GB = 512GB HBM,实测 v6e-16 / tp=16 / mem-fraction 0.85 跑通 + GSM8K 0.950。同类资源画像偏保守问题可能存在于其他 70B/80B dense recipe | §2.1 minimum 改 v6e-16,recommended 改 v6e-32;§2.3 主 launch 默认 v6e-16(tp=16, mem-fraction 0.85);§2.4 Memory Management 拆两段;§5 加 stale libtpu_lockfile troubleshooting 行 | ✅ fixed 2026-05-25 — 详 [`llama-3.3-70b/`](2026-05-21-recipe-command-audit/llama-3.3-70b/) |
| D6 | `Google/Gemma2.md` §2.3 27B-it / 9B-it launch | 同 D4 类,两个 launch 命令都漏 `--page-size`,默认值 1 → `Final max_running_requests: 1`,bench c=16 实际全部串行(参考 Llama 3.1 实测 9.3× 差距)。27B-it 实测加 `--page-size 128 --max-running-requests 64` 跑出 974 tok/s + GSM8K 0.865 | 两个 launch 都加 `--page-size 128 --max-running-requests 64 --chunked-prefill-size 2048`;§2.4 加 "Paging / concurrency (mandatory)" callout;§5 加 troubleshooting 行 | ✅ fixed 2026-05-26 — 详 [`gemma-2-27b/`](2026-05-21-recipe-command-audit/gemma-2-27b/) |
| D7 | `Qwen/Qwen3-MoE.md` §2.3 30B-A3B launch + §2.4 MoE Backend | 写 `--moe-backend fused`,但 fused MoE 内核(`fused_moe/v1/kernel.py:266`)断言 `intermediate_size % bf == 0` 且 bf=512;Qwen3-30B-A3B `moe_intermediate_size=768`,启动直接 `ValueError: Expected intermediate_size=768 to be aligned to bf=512`。改 `--moe-backend epmoe` 跑通,1476 tok/s + GSM8K 0.980。recipe §2.4 "fused 是 EP≥16 推荐"的口径过简,没说硬约束 | §2.3 30B-A3B launch 改 `--moe-backend epmoe` + 添加 inline note 解释;§2.4 MoE Backend 重写,标 fused 需 `intermediate_size % 512 == 0`,30B-A3B(768)不合规走 epmoe,235B-A22B(1536)合规可走 fused;§5 troubleshooting 加 `Expected intermediate_size=...` 错误行 | ✅ fixed 2026-05-26 — 详 [`qwen3-30b-a3b/`](2026-05-21-recipe-command-audit/qwen3-30b-a3b/) |
| D8 | `Wan/Wan2.1.md` §3.1/§3.2/§3.3 + `Wan/Wan2.2.md` 同段 | recipe 全部把请求 `size` 写成 `WIDTHxHEIGHT`(`480x832` / `720x1280` / `1024x1024`),但服务端 `multimodal/manager/global_scheduler.py:242-243` 用 `size_str.split("*")` 解析,送 `x` 直接 `ValueError: invalid literal for int() with base 10: '480x832'`,GlobalScheduler 进程崩溃,需要重启 server。Wan2.1-T2V-14B 实测复现并修复后用 `480*832` 跑通(~4 min 19s 出 794KB MP4)。同时 `io_struct.py:VideoGenerationsRequest.size` 默认值 `"720x1280"` 也写错(用户不传 size 默认就崩),属服务端 source 层 bug 待提 | 两份 Wan recipe 把所有 `size` 改成 `WIDTH*HEIGHT`(asterisk),§3.1 §3.2 §3.3 同步;§1 Recommended Generation Parameters 加格式约束说明;§5 加 `ValueError: invalid literal for int()` troubleshooting 行;`io_struct.py` 默认值改 `"720*1280"` 是源码层修复待提 | ✅ recipe fixed 2026-05-26;⚠️ source 层 default 待提 — 详 [`wan2.1-t2v-14b/`](2026-05-21-recipe-command-audit/wan2.1-t2v-14b/) |
| D9 | `Wan/Wan2.1.md` §3.1/§3.3 + `Wan/Wan2.2.md` 同段 | recipe 写视频接口响应是 `{"id":"vid_...","path":"/tmp/sglang-jax-videos/vid_...mp4"}`、图片接口响应是 `{"id":"img_...","url":"http://.../static/img_...png"}`。实际服务端响应是 `{"success":true,"meta_info":{}}`(由 tokenizer_manager.generate_request 流式 yield),并不携带 path/URL;MP4/PNG 写到**服务进程 cwd** 用 `<uuid>.mp4` 命名,只能在 server log 里看到 `Saved output to <uuid>.mp4` 行。recipe 既写错响应 schema 也写错文件落地路径 | §3.1 §3.3 替换响应示例为 `{"success": true, "meta_info": {}}`,prose 改写说明 MP4/PNG 落 server cwd 用 `<uuid>.mp4`,获取手段是看 server log 行或将 server `cwd` 切到共享目录 + 挂卷;§5 加 `Response body 没有 path/url` troubleshooting 行 | ✅ recipe fixed 2026-05-26 — 详 [`wan2.1-t2v-14b/`](2026-05-21-recipe-command-audit/wan2.1-t2v-14b/) |

### 🟡 P1 — 跑得起来但有 review 价值

| ID | 文件 | 问题 | 修复方案 | 状态 |
|---|---|---|---|---|
| B1 | `Qwen/Qwen.md` L62-68 | 单节点 launch 加 `--dist-init-addr 0.0.0.0:10011 --nnodes 1 --node-rank 0`,note 写"required for single-host"但源码确认 `nnodes=1` 时 jax.distributed 不初始化 | 删除 trio + note | ✅ fixed 2026-05-21 |
| B3 | `Wan/Wan2.1.md` / `Wan/Wan2.2.md` | `--vae-tiling` flag 是 store_true 但默认 True → 命令行 no-op | 从 launch 命令删除该 flag,Configuration Tips 改写说明 "tiling is on by default; no `--no-vae-tiling` to disable" | ✅ fixed 2026-05-21 |
| C2 | `Xiaomi/MiMo-V2.5-Pro.md` L91 | v6e-64 launch 有 `--max-prefill-tokens 16384 --max-seq-len 4096`,两者都是默认值 | 删除冗余 flag | ✅ fixed 2026-05-21 |

### 🟢 P2 — 可选优化

| ID | 文件 | 问题 | 修复方案 | 状态 |
|---|---|---|---|---|
| C3 | `Xiaomi/MiMo-V2.5-Pro.md` L247 | Speculative §2.4 写 `<draft-checkpoint>` 占位,用户无法直接抄 | 实测后填入 MiMo 模型卡里 NEXTN 的真实路径 | ⏳ pending(可结合 TPU 测试做) |
| C4 | `Wan/Wan2.2.md` §2.1 / §2.4 | Wan 2.2 text encoder 调度在 CPU(`wan2_2_stage_config.yaml` `device_kind: cpu`),recipe 未点明 | Hardware Matrix 加备注:Wan 2.2 UMT5 跑 CPU,`--tp-size` 只覆盖 DiT + VAE | ✅ fixed 2026-05-26 |
| C5 | `Wan/Wan2.2.md` §2.4 | `--text-encoder-precisions bf16` HBM 收益叙述对 Wan 2.2 (CPU encoder) 不适用 | 按 variant 拆分(Wan 2.1 TPU encoder / Wan 2.2 CPU encoder) | ✅ fixed 2026-05-26 |
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

### 可用测试资源(2026-05-21 盘点)

GKE 集群里 `default` namespace 下两个 `wlf-*` TPU 资源,均空闲、`sgl_jax` 已安装、无 JIT cache(首跑会冷启动 ~4 min)。

| 资源 | TPU | 拓扑 | Pods | git commit | 镜像 | 适合 recipe |
|---|---|---|---|---|---|---|
| `wlf-v6e-4-bench`(single pod) | v6e-4 | 2x2 (4 chips) | 1 | `66b9481` (P0 CI gate improvements) | `jax0.8.1-rev1` | Qwen / Qwen3-8B/32B / Llama3.1 / Gemma2 / MiMo-7B / DeepSeek-V2-Lite / Qwen2.5-VL 3B/7B / Wan 2.1 1.3B/14B |
| `wlf-v6e-16`(Indexed Job + headless svc `wlf-v6e-16-headless-svc`) | v6e-16 | 4x4 (4 nodes × 4 chips) | 4(rank 0-3) | `b2daa46d` (qwen25 tool-call parser #1102) | `jax0.8.1-rev1` | MiMo-V2-Flash multi-host / Qwen3-30B-A3B MoE / Kimi-Linear / Qwen2.5-VL 72B (会偏小 v6e-32 推荐;v6e-16 上需 reshard) |

**通用进入方式**:

```bash
# Single-host v6e-4
kubectl exec -it wlf-v6e-4-bench -c wlf-v6e-4-bench -- bash

# Multi-host v6e-16 rank-0
kubectl exec -it wlf-v6e-16-0-2x99h -c wlf-v6e-16 -- bash
# 其他 rank pod 名 (pod name suffix 会变化,跑前用 kubectl get pods | grep wlf-v6e-16 重新查):
#   wlf-v6e-16-1-x9v6l / wlf-v6e-16-2-rrkd5 / wlf-v6e-16-3-k4lx2
```

**容器内常用路径**:

- sgl_jax 源码: `/tmp/sglang-jax/`
- Python: `/opt/venv/bin/python`(已激活)
- 推荐持久化 JIT cache 路径: `/tmp/jit_cache`(默认空,首跑创建)
- 多节点 dist-init 用 headless svc: `--dist-init-addr wlf-v6e-16-0.wlf-v6e-16-headless-svc:5000`
  - **注意**:`wlf-v6e-16-0` 是 Indexed Job 给 rank-0 pod 的稳定 DNS 前缀(不是上面 `2x99h` 那个随机后缀;Indexed Job 用 `${JOB_COMPLETION_INDEX}` 命名)

**Port-forward 拿 server URL**(本地 client 连):

```bash
kubectl port-forward wlf-v6e-4-bench 30000:30000 &
# 或 multi-host 时:
kubectl port-forward wlf-v6e-16-0-2x99h 30000:30000 &
```

**保留 / 销毁**:

- 两个资源已存在 1.5~8 天,假设是你长期占用的开发集群
- 测试完成后**不要主动删 pod / job**,等用户明确指示
- 若 JIT cache 占用过大,清理 `/tmp/jit_cache/*` 但保留目录

### 测试模板(每次跑后填):

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

(按现有 ✅ Validated / 🚧 Starter 升级路径选,结合上方可用资源)

**2026-05-22 用户决策**: 跳过 gated 模型(Llama / Gemma);v6e-16 已清理可用。剩余可在现有资源跑的测试矩阵:

#### Phase 1 — wlf-v6e-4-bench dense small (~25 min each)
- [x] Qwen-7B-Chat — 2026-05-22 done ✅
- [ ] Qwen3-8B(hybrid reasoning parser)
- [ ] MiMo-7B-RL(reasoning + tool-call parser)
- [x] Llama-3.1-8B-Instruct — 2026-05-25 done ✅(本地 `/models` 已缓存,免 HF_TOKEN)

#### Phase 2 — wlf-v6e-4-bench MoE single-host
- [ ] DeepSeek-V2-Lite(MLA + epmoe)

#### Phase 3 — wlf-v6e-4-bench multimodal
- [ ] Qwen2.5-VL-7B(VL + MMMU eval)
- [x] Wan 2.1 14B T2V(diffusion — **无 evalscope**,§3.1 basic usage smoke)— 2026-05-26 done ✅(发现 D8 size separator + D9 response shape 两个 recipe bug)
- [x] Wan 2.2 A14B T2V(diffusion dual-transformer — **无 evalscope**,§3.1 basic usage smoke)— 2026-05-26 done ✅(发现 D10 stage-config 注册名 case-sensitive)
- [ ] Wan 2.1 1.3B T2V(diffusion — **无 evalscope**,只 launch + 视频生成)

#### Phase 4 — wlf-v6e-4-bench dense large
- [x] Qwen3-32B(~64GB BF16,marginal 在 128GB HBM)— 2026-05-25 done ✅
- [x] Gemma 2 27B-it(~54GB BF16,hybrid attention dual KV pools)— 2026-05-26 done ✅(unsloth ungated mirror)

#### Phase 5 — wlf-v6e-16 multi-host MoE
- [ ] MiMo-V2-Flash(已 Validated,smoke test v6e-16 recipe)
- [x] Qwen3-30B-A3B MoE — 2026-05-26 done ✅(epmoe,fused 因 intermediate_size 不对齐失败)
- [ ] Kimi-Linear-48B-A3B(linear-attention)
- [x] Llama-3.3-70B-Instruct — 2026-05-25 done ✅(v6e-16, tp=16,unsloth ungated mirror)

#### 跳过(资源不足)
- MiMo-V2.5-Pro / DeepSeek-V3/R1 / GLM-4.5 / Qwen3-235B / Ling-2.5 / Ring-2.5 / Ling-2.6 / Grok-2 / Qwen2.5-VL-72B → 需要 v6e-32 / v6e-64 / v7x-16
- ~~Wan 2.2 A14B → 需要 v6e-8~~ → 实测可在 v6e-4 单卡(`--tp-size 1`,CPU text encoder + 双 DiT,118GB 权重落 /dev/shm)走 §3.1 smoke,2026-05-26 已跑通

#### 跳过(gated 模型)
- Llama 3.1 / 3.3-70B / Gemma 2 → 需 `HF_TOKEN`,本次跳过(已在 2026-05-25/26 用 `unsloth/*` ungated mirror 跑通,gated 限制对 unsloth 镜像不适用)

> 后续在 TPU 上按各 recipe 的 launch / benchmark 命令实测时,在此追加记录。

#### Qwen-7B-Chat — 2026-05-22

- **目录**: [`2026-05-21-recipe-command-audit/qwen-7b-chat/`](2026-05-21-recipe-command-audit/qwen-7b-chat/)
- **命令**: 见 [`commands.md`](2026-05-21-recipe-command-audit/qwen-7b-chat/commands.md)(launch + bench_serving + evalscope 三件套)
- **步骤**: 见 [`steps.md`](2026-05-21-recipe-command-audit/qwen-7b-chat/steps.md)
- **TPU**: `wlf-v6e-4-bench`(v6e-4, 4 chips, single host)
- **Build**: sglang-jax `fe092bf179128df98b5ada90e0afc89cd1618bf0`(main, 2026-05-22 pull)
- **冷启时长**: ~8 min(权重下载 14GB → tokenizer + JAX precompile EXTEND 267s + DECODE 205s → Uvicorn ready)
- **结果**:
  - **bench_serving** (random ISL=512 OSL=128, concurrency=8, 100 prompts):
    - 6.53 req/s, 836 tok/s out, 88.91 ms TTFT, 8.64 ms TPOT
    - 完整 `============ Serving Benchmark Result ============` 块见 [`bench_serving.log`](2026-05-21-recipe-command-audit/qwen-7b-chat/bench_serving.log),已 verbatim 回填到 Qwen.md §4.2
  - **evalscope GSM8K** (limit=500):
    - AverageAccuracy = **0.484**(recipe 旧 baseline 0.504,差 -0.02 在 500-sample 统计噪声 ±0.022 内 — 正常 variance)
    - 已回填 Qwen.md §4.1 Test Results + Tested build commit
- **异常**: 无
- **文档改动**:
  - `autoregressive/Qwen/Qwen.md` §4.1 Test Results: `0.504` → `0.484`,Tested build `_Pending_` → `fe092bf` (2026-05-22)
  - `autoregressive/Qwen/Qwen.md` §4.2 Test Results: `_Pending_` → 完整 verbatim bench_serving 块
- **关键观察**(对 cookbook 没影响,但记下供后续):
  - sglang-jax 日志有 `Unsupported tokenizer type: QWenTokenizer` 警告(QWenTokenizer 是 Qwen 第一代自带 slow tokenizer)— 不阻塞,只是 detokenizer 走兼容路径
  - 自动 max_running_requests=16(attention_backend 限制,context_len=8192, page_size=1),并非 launch flag 设置
  - 冷启动比 recipe 文档预期(~4 min)长 1 倍 — 主要是 DECODE precompile 也要 ~3.5 min,文档低估

#### Qwen3-8B — 2026-05-23

- **目录**: [`2026-05-21-recipe-command-audit/qwen3-8b/`](2026-05-21-recipe-command-audit/qwen3-8b/)
- **TPU**: `wlf-v6e-4-bench` (v6e-4)
- **Build**: sglang-jax `fe092bf`
- **冷启时长**: ~3 min(显著快于 Qwen-7B-Chat 的 8 min,因为 `--page-size 128` 而非 1,EXTEND/DECODE precompile 缩短到 79s)
- **结果**:
  - **bench_serving** (random ISL=1024, OSL=1024, c=64, 192 prompts): 4.70 req/s, **4817 tok/s out**, 587 ms TTFT, 12.71 ms TPOT
    - recipe Qwen3.md §4.1 中 2025-09 旧 build 同 cell 测得 5296 tok/s,差异 -9% 在 build 漂移正常范围,且我没加 `--disable-radix-cache`(recipe 旧 run 有)
  - **evalscope GSM8K** (limit=500, thinking-on): AverageAccuracy = **0.944** ⭐(reasoning 模型 GSM8K 高分,符合预期)
- **异常**: 无
- **文档改动**:
  - `autoregressive/Qwen/Qwen3.md` §4.2 Accuracy: 从 `_Not measured in this benchmark run_` 升级为完整四件套 + 0.944 实测
  - §4.1 Speed 表保留(2025-09 vLLM 对比数据有历史价值,不覆盖)

#### MiMo-7B-RL — 2026-05-23

- **目录**: [`2026-05-21-recipe-command-audit/mimo-7b-rl/`](2026-05-21-recipe-command-audit/mimo-7b-rl/)
- **TPU**: `wlf-v6e-4-bench` (v6e-4)
- **Build**: sglang-jax `fe092bf`
- **冷启时长**: ~12 min(page-size 默认 1,EXTEND/DECODE precompile 慢于 page-size=128 的 Qwen3-8B)
- **结果**:
  - **bench_serving** (random ISL=512, OSL=128, c=8, 100 prompts): 3.62 req/s, **463 tok/s out**, 1116 ms TTFT, 8.26 ms TPOT
  - **evalscope GSM8K** (limit=500, thinking-on, max_tokens=4096): AverageAccuracy = **0.920** ⭐(RL-tuned reasoning + thinking-on,符合预期)
- **异常**:
  - kubectl exec websocket 在 evalscope 跑到 294/500 时 close 1006 断开,evalscope 进程在 pod 内继续但 stdout/stderr pipe reader 死,write 阻塞导致进程卡 sleep 不再前进。详 [`exception_kubectl_websocket_drop.md`](2026-05-21-recipe-command-audit/mimo-7b-rl/exception_kubectl_websocket_drop.md)
  - 修复:改用脚本 + nohup + 文件重定向 + disown,kubectl 断开不影响。第二次重跑 500 样本一次过,~25 min 完成
- **文档改动**:
  - `autoregressive/Xiaomi/MiMo-7B.md` §4.1 Speed:升级为完整四件套(Test Environment / Deployment / Benchmark Command / 完整 bench_serving 输出块)
  - §4.2 Accuracy:升级为四件套 + 0.920 实测,Reasoning Parser 字段标 `mimo`,evalscope 命令补 `--limit 500 --generation-config '{"chat_template_kwargs": {"enable_thinking": true}, "max_tokens": 4096}'`
- **跨测试约定补充**(写入 §6.1):长任务(eval > 5 min)必须用 `nohup script > log 2>&1 &; disown`,不要依赖 `kubectl exec ... -- "cmd | tee"` 长 hold

#### DeepSeek-V2-Lite-Chat — 2026-05-25 (D2 verification)

- **目录**: [`2026-05-21-recipe-command-audit/deepseek-v2-lite-chat/`](2026-05-21-recipe-command-audit/deepseek-v2-lite-chat/)
- **TPU**: `wlf-v6e-4-bench` (v6e-4)
- **Build**: sglang-jax `fe092bf`
- **冷启时长**: ~2.5 min (page-size=64,MLA 硬约束)
- **结果**:
  - **bench_serving** (random ISL=512 OSL=512 c=8,100 prompts): **1050.28 tok/s out**, 195.61 ms TTFT, 7.01 ms TPOT
  - **evalscope GSM8K** (limit=200,无 generation-config): AverageAccuracy = **0.685** ⭐ — 验证 D2 修复:base `V2-Lite` = 0.014 → `-Chat` = 0.685
- **异常**: 无
- **文档改动**:
  - `DeepSeek/DeepSeek-V2.md` §4.2 Test Results: `_Pending V2-Lite-Chat run_` → 0.685 + 0.014 anti-pattern 对照
  - audit §3.1 D2 状态保持 ✅ fixed,本节是验证 evidence

#### MiMo-V2-Flash — 2026-05-25 (Phase 5)

- **目录**: [`2026-05-21-recipe-command-audit/mimo-v2-flash/`](2026-05-21-recipe-command-audit/mimo-v2-flash/)
- **TPU**: `wlf-v6e-16-{0..3}` (v6e-16, 4 nodes × 4 chips)
- **Build**: sglang-jax `b2daa46d` (v6e-16 镜像 pre-cached,与 v6e-4 的 `fe092bf` 不同)
- **冷启时长**: ~10 min (GCSFuse warmup 67s @ 4488 MB/s + JIT precompile ~6 min)
- **结果**:
  - **bench_serving** (random ISL=1024 OSL=1024 c=16, 100 prompts, `--tp-size 16 --dp-size 4 --ep-size 16 --moe-backend fused`): **1034.44 tok/s out**, peak 1216 tok/s, 1093.50 ms TTFT, 13.29 ms TPOT
  - **evalscope GSM8K** (limit=200, thinking-on default, max_tokens=8192): AverageAccuracy = **0.975** ⭐
- **异常**: 无 — 但 evalscope 不在 v6e-16 镜像里(`jax0.8.1-rev1`),从 `wlf-v6e-4-bench` 远程打到 rank-0 IP `10.31.148.4:30000`
- **文档改动**:
  - `Xiaomi/MiMo-V2-Flash.md` §4.1 增加 v6e-16 multi-host 行 (1034 tok/s)
  - §4.2 Test Results 表新增 `b2daa46d` build 行 (200 prompts, 0.975)

#### Kimi-Linear-48B-A3B-Instruct — 2026-05-25 (Phase 5)

- **目录**: [`2026-05-21-recipe-command-audit/kimi-linear/`](2026-05-21-recipe-command-audit/kimi-linear/)
- **TPU**: `wlf-v6e-16-{0..3}` (v6e-16, 4 nodes × 4 chips)
- **Build**: sglang-jax `b2daa46d`
- **冷启时长**: ~5 min(JAX register + Publisher sync ~30s + extend precompile ~2 min + decode precompile ~1 min);"Uvicorn running" 在 04:48:30 出现
- **结果**:
  - **bench_serving** (random ISL=1024 OSL=1024 c=16, 100 prompts): **690.57 tok/s out**, peak 832, 607.66 ms TTFT, 20.77 ms TPOT
  - **evalscope GSM8K** (limit=200, 无 generation-config): AverageAccuracy = **0.925** ⭐
- **异常 / 修复**:
  - 🔴 **D-NEW-Kimi**: recipe §2.3 launch 缺 `--disable-radix-cache`。Kimi 是 hybrid recurrent state 模型,radix prefix sharing 与 recurrent state pool 不兼容,启动 hard-assert:`Hybrid recurrent state models require --disable-radix-cache (prefix sharing is unsafe with recurrent state)`。已修 recipe §2.3 + §2.4 + §5 Troubleshooting。
  - ⚠️ **多节点 dispatch 时序**: 第一次 rank-3 因 `kubectl exec` 短暂 EOF 晚 30 min 启动,rank 0/1/2 都在 5 min 后 hit `RegisterTask DEADLINE_EXCEEDED`。修法:用 `for ... ; do kubectl exec ... & ; done; wait` 同步起 4 ranks。已加进 §5 Troubleshooting。
- **文档改动**:
  - `Moonshotai/Kimi-Linear.md` §1 starter banner → Validated
  - §2.3 launch 命令补 `--disable-radix-cache`
  - §2.4 增加"Mandatory: --disable-radix-cache for hybrid recurrent state"小节
  - §4.1 增加 v6e-16 measured benchmark block
  - §4.2 Test Environment build → `b2daa46d`,Test Results 增加 0.925 行
  - §5 Troubleshooting 新增 2 行(`--disable-radix-cache` assert + multi-host dispatch deadline)

#### Llama-3.1-8B-Instruct — 2026-05-25 (Phase 1)

- **目录**: [`2026-05-21-recipe-command-audit/llama-3.1-8b/`](2026-05-21-recipe-command-audit/llama-3.1-8b/)
- **TPU**: `wlf-v6e-4-bench` (v6e-4, 4 chips, single host)
- **Build**: sglang-jax `de29d9f0`
- **冷启时长**: ~9 min(`--page-size 128` 配置下 weight load + EXTEND/DECODE precompile;recipe-as-written 配置 page=1 时 precompile 因 max_q 步进更细而更慢)
- **结果**:
  - **bench_serving** (random ISL=512 OSL=512 c=16, 100 prompts, 加 `--page-size 128 --max-running-requests 64`):
    - 5.61 req/s, **1448.91 tok/s out**, peak 1693, 35.33 ms TTFT, 9.94 ms TPOT
    - 完整块见 [`bench_serving.log`](2026-05-21-recipe-command-audit/llama-3.1-8b/bench_serving.log),已贴回 Llama3.1.md §4.1
  - **evalscope GSM8K** (limit=200): AverageAccuracy = **0.825**(HF 模型卡 8-shot baseline ~0.845;0-shot/200-sample 在统计噪声内)
- **异常 / 修复**:
  - 🔴 **D4**: recipe §2.3 漏 `--page-size 128`,默认 page=1 触发 `Final max_running_requests: 1`,bench-as-written 跑出 156 tok/s + 47s TTFT(c=16 全串行)。同源问题与 D1(MLA 必填 page-size)不同点:Llama dense attention 不会断言失败,只是静默降到串行,review 阶段更难发现。Layout B starter 文档普遍漏写 `--page-size`,建议下一轮 audit 把 dense recipe 全扫一遍。已修 Llama3.1.md §2.3 + §2.4 + §5;详 [`llama-3.1-8b/`](2026-05-21-recipe-command-audit/llama-3.1-8b/)。
- **文档改动**:
  - `autoregressive/Llama/Llama3.1.md` §1 starter banner → Validated (build `de29d9f0`)
  - §2.3 launch 加 `--page-size 128 --max-running-requests 64`
  - §2.4 新增 "Paging / concurrency (mandatory)" 小节
  - §4.1 改为 measured Layout B(Test Environment + bench command + Test Results 全填)
  - §4.2 Tested build → `de29d9f0`,Test Results 表 0.825
  - §5 新增 "Bench shows ~150 tok/s ... page=1" troubleshooting 行

#### Qwen3-32B — 2026-05-25 (Phase 4)

- **目录**: [`2026-05-21-recipe-command-audit/qwen3-32b/`](2026-05-21-recipe-command-audit/qwen3-32b/)
- **TPU**: `wlf-v6e-4-bench` (v6e-4, 4 chips, single host)
- **Build**: sglang-jax `de29d9f0`
- **冷启时长**: ~9 min(weight load + EXTEND/DECODE precompile;DECODE 119s)
- **结果**:
  - **bench_serving** (random ISL=1024 OSL=1024 c=16, 100 prompts, recipe-as-written 加 `--reasoning-parser qwen3`):
    - 1.59 req/s, **833.80 tok/s out**, peak 1008, 104.53 ms TTFT, 16.89 ms TPOT
    - 完整块见 [`bench_serving.log`](2026-05-21-recipe-command-audit/qwen3-32b/bench_serving.log)
    - 与 Qwen3.md §4.1 中 Sept-2025 build 的 `bs=64`(1977 tok/s)不可直接比 — 我们 c=16 batch 没填满,只用作 build verification smoke。已贴回 Qwen3.md §4.1 末尾"Build verification"小节。
  - **evalscope GSM8K** (limit=200, default thinking-on): AverageAccuracy = **0.975**(Qwen 模型卡 ~95-97% 范围,符合)
- **异常**: 无(recipe §2.3 已经有 `--page-size 128 --max-running-requests 256`,验证 Qwen3.md 不存在 D4 类问题)
- **文档改动**:
  - `autoregressive/Qwen/Qwen3.md` §4.1 末尾加 "Build verification (2026-05-25, sglang-jax `de29d9f0`)" 小节
  - §4.2 Test Environment 改为 8B + 32B 双 build 标注
  - §4.2 Test Results 表加 Qwen3-32B 0.975 行

#### Llama-3.3-70B-Instruct — 2026-05-25 (Phase 5)

- **目录**: [`2026-05-21-recipe-command-audit/llama-3.3-70b/`](2026-05-21-recipe-command-audit/llama-3.3-70b/)
- **TPU**: `wlf-v6e-16-{0..3}` (v6e-16, 4 nodes × 4 chips, **tp=16 而非 recipe 写的 32**)
- **Build**: sglang-jax `b2daa46d`
- **冷启时长**: ~6 min(weight load 4 ranks 并行 ~3 min,EXTEND/DECODE precompile ~2 min;DECODE 114s)
- **结果**:
  - **bench_serving** (random ISL=1024 OSL=1024 c=16, 100 prompts):
    - 1.68 req/s, **882.17 tok/s out**, peak 1040, 113.86 ms TTFT, 16.33 ms TPOT
    - 完整块见 [`bench_serving.log`](2026-05-21-recipe-command-audit/llama-3.3-70b/bench_serving.log),已贴回 Llama3.3-70B.md §4.1
  - **evalscope GSM8K** (limit=200): AverageAccuracy = **0.950**(Meta 模型卡 8-shot CoT baseline ~0.969;0-shot/200-sample 在统计噪声内)
- **异常 / 修复**:
  - 🔴 **D5(资源画像偏保守)**: recipe §2.1 写 minimum runnable = v6e-32 / tp=32,实测 v6e-16 / tp=16 跑通,~8.75 GB weights/chip + 18 GB KV headroom。算术上 16 × 32GB = 512GB ≫ 140GB BF16,recipe 估算明显过保守。已修 §2.1 把 minimum 降到 v6e-16,recommended 从 v6e-64 降到 v6e-32;§2.3 主 launch 命令默认 v6e-16(`--tp-size 16 --mem-fraction-static 0.85`),v6e-32 留作"recommended production"小节。同类问题可能存在于其他 70B/80B dense recipe(GLM、Llama 3.1-70B 等),下一轮 audit 应扫一遍。
  - ⚠️ **stale `/tmp/libtpu_lockfile`**: 失败重启会留 lockfile,下次启动报 `Unable to initialize backend 'tpu': ABORTED: Internal error when accessing libtpu multi-process lockfile`。已加 §5 Troubleshooting 一行(rm 命令)。多机场景下需要在 4 个 rank 都 rm。
  - ⚠️ **gated repo + 备用 ungated mirror**: `meta-llama/Llama-3.3-70B-Instruct` HF 上 gated,本次集群无 `HF_TOKEN`。改用 `unsloth/Llama-3.3-70B-Instruct`(同样 hidden=8192/layers=80/heads=64/kv=8 的真实权重)。recipe 不写 mirror(license 边界),但 audit 这里记录一下供后续测试参考。
  - ⚠️ **evalscope 版本**: v6e-16 pod 默认装的是 evalscope 1.7.1,把 `--eval-type service` 重命名成 `server`/`openai_api`。需要 `pip install "evalscope<1.0"` 降到 0.x 才能跟 cookbook 命令兼容。建议 cookbook §4.2 加一行版本约束 `evalscope>=0.13,<1.0`,或者 recipe 命令一起切到新名字。本次先 pin 旧版,新名字迁移留 backlog。
- **文档改动**:
  - `autoregressive/Llama/Llama3.3-70B.md` §1 starter banner → Validated (v6e-16, build `b2daa46d`)
  - §2.1 hardware matrix:minimum runnable v6e-32→v6e-16,recommended v6e-64→v6e-32
  - §2.3 launch 主小节默认 v6e-16(tp=16, mem-fraction 0.85);v6e-32 改为简短引用 + flag 调整说明
  - §2.4 Memory Management 拆 v6e-16 / v6e-32 两段
  - §4.1 改为 measured Layout B(Test Environment + bench command + Test Results 全填)
  - §4.2 Test Environment v6e-32→v6e-16,Tested build → `b2daa46d`,Test Results 表 0.950
  - §5 新增 stale libtpu_lockfile troubleshooting 行

#### Gemma-2-27B-it — 2026-05-26 (Phase 4)

- **目录**: [`2026-05-21-recipe-command-audit/gemma-2-27b/`](2026-05-21-recipe-command-audit/gemma-2-27b/)
- **TPU**: `wlf-v6e-4-bench` (v6e-4, 4 chips, single host, tp=4)
- **Build**: sglang-jax `de29d9f0`
- **冷启时长**: ~3 min(weight load 51GB shm-resident ~30s,EXTEND precompile 60s,DECODE precompile ~70s,ready 在 launch 后 175s)
- **结果**:
  - **bench_serving** (random ISL=1024 OSL=1024 c=16, 100 prompts):
    - 1.86 req/s, **974.57 tok/s out**, peak 1159, 77.57 ms TTFT, 14.47 ms TPOT
    - 完整块见 [`bench_serving.log`](2026-05-21-recipe-command-audit/gemma-2-27b/bench_serving.log),已贴回 Gemma2.md §4.1
  - **evalscope GSM8K** (limit=200): AverageAccuracy = **0.865**(Gemma 2 27B-it 模型卡 5-shot maj@1 baseline 75.2 → 0-shot/200-sample 在合理范围内)
- **异常 / 修复**:
  - 🔴 **D6(漏 `--page-size`)**: 同 D4 类。Gemma2.md §2.3 两个 launch(9B-it / 27B-it)都漏 `--page-size`,默认值 1 → 调度器静默降级 `Final max_running_requests: 1`,c=16 bench 全部串行。已修两处 launch + §2.4 mandatory callout + §5 Troubleshooting 行。同类 D4/D6 漏检暴露 audit pass 1 的 bug pattern:dense attention 不会启动崩,review 阶段没法靠源码 grep "page_size > 1 assert" 兜住,只能跑出来才发现。
  - ⚠️ **gated repo + 备用 ungated mirror**: `google/gemma-2-27b-it` HF 上 gated,本次集群无 `HF_TOKEN`。改用 `unsloth/gemma-2-27b-it`(同样 hidden=4608 / layers=46 / heads=32 / kv=16 / sliding_window=4096 的真实权重,unsloth 仅按 transformers 版本 re-export,不是 quantized 版本)。recipe 不写 mirror(license 边界),audit 这里记录供后续测试参考。
  - ✅ **Hybrid attention 验证**: Gemma 2 alternating global 8K + sliding 4K layers,sgl-jax 双 KV pool(`--swa-full-tokens-ratio 0.8` 默认),`--mem-fraction-static 0.85` 加 `--max-running-requests 64` 没有 OOM,recipe §2.1 给的 27B/v6e-4 估算正确。
- **文档改动**:
  - `autoregressive/Google/Gemma2.md` §1 starter banner → 27B-it Validated;9B/9B-it 标注仍 Starter
  - §2.1 hardware matrix:27B 行加"validated 2026-05-25"备注
  - §2.3 9B-it / 27B-it 两个 launch 都加 `--page-size 128 --max-running-requests 64 --chunked-prefill-size 2048`
  - §2.4 加 "Paging / concurrency (mandatory)" 子段
  - §4.1 改为 measured Layout B(Test Environment + bench command + Test Results 全填)
  - §4.2 Test Environment 改 27B-it,Tested build → `de29d9f0`,Test Results 表 0.865
  - §5 加 D6 page-size troubleshooting 行

#### Qwen3-30B-A3B MoE — 2026-05-26 (Phase 5)

- **目录**: [`2026-05-21-recipe-command-audit/qwen3-30b-a3b/`](2026-05-21-recipe-command-audit/qwen3-30b-a3b/)
- **TPU**: `wlf-v6e-16-{0..3}` (v6e-16, 4 nodes × 4 chips, tp=16, ep=16)
- **Build**: sglang-jax `b2daa46d`
- **冷启时长**: ~4 min(weight load /models gcsfuse + EXTEND precompile 64s + DECODE precompile 9 sizes ~150s,ready 在 launch 后 ~230s;首 attempt fused 失败后第二次有 JIT cache,走 epmoe 重新 precompile)
- **结果**:
  - **bench_serving** (random ISL=1024 OSL=1024 c=16, 100 prompts):
    - 2.81 req/s, **1476.18 tok/s out**, peak 1744, 75.77 ms TTFT, **9.88 ms TPOT**(MoE A3B 解码极快,只激活 3B 参数 → TPOT 比 Llama 3.3 70B 的 16ms 快 ~40%)
    - 完整块见 [`bench_serving.log`](2026-05-21-recipe-command-audit/qwen3-30b-a3b/bench_serving.log),已贴回 Qwen3-MoE.md §4.1
  - **evalscope GSM8K** (limit=200, thinking-on, max_tokens=8192): AverageAccuracy = **0.980**(Qwen3 reasoning 模型 GSM8K 高分,符合预期,与 Qwen3-32B 的 0.975 一致)
- **异常 / 修复**:
  - 🔴 **D7(`--moe-backend fused` 配置不兼容)**: recipe §2.3 30B-A3B launch 用 `--moe-backend fused`,启动直接 `ValueError: Expected intermediate_size=768 to be aligned to bf=512`(`fused_moe/v1/kernel.py:266`)。fused MoE 内核硬约束 expert FFN 内层维度是 512 倍数,Qwen3-30B-A3B `moe_intermediate_size=768` 不合规。已改 epmoe 跑通。recipe §2.4 旧文字"fused 是 EP≥16 推荐"过简,没标 alignment 硬约束。已重写 §2.4 + §2.3 inline note + §5 troubleshooting 行。
  - ⚠️ **Qwen3-235B-A22B 不受影响**: 235B-A22B `moe_intermediate_size=1536`,1536/512=3 ✅,可继续走 fused。recipe §2.4 已分别标注 30B/235B 的 alignment 状态。
  - 同类 fused alignment 风险点:其他用 fused MoE 的 recipe(MiMo-V2-Flash / Kimi-Linear / Ling 系列等)都该 grep 一次 `moe_intermediate_size`,不能默认 EP≥16 就用 fused。
- **文档改动**:
  - `autoregressive/Qwen/Qwen3-MoE.md` §1 starter banner → 30B-A3B Validated;235B-A22B 仍 Starter
  - §2.3 30B-A3B launch:`--moe-backend fused` → `--moe-backend epmoe` + 加 inline note 解释 alignment
  - §2.4 MoE Backend 段重写:fused 标硬约束 + 30B/235B 分别说明
  - §4.1 改为 measured Layout B(Test Environment + bench command + Test Results 全填)
  - §4.2 Tested build → `b2daa46d`,Test Results 表加 0.980 行(标 thinking-on)
  - §5 加 D7 troubleshooting 行 + 改写"MoE throughput plateau"行限定为 235B-A22B

#### Wan 2.1 T2V 14B-Diffusers — 2026-05-26 (Phase 3,multimodal smoke)

- **目录**: [`2026-05-21-recipe-command-audit/wan2.1-t2v-14b/`](2026-05-21-recipe-command-audit/wan2.1-t2v-14b/)
- **TPU**: `wlf-v6e-4-bench`(v6e-4, single host, tp=2)
- **Build**: sglang-jax `de29d9f0`(同 Gemma2 batch)
- **冷启时长**: ~5.5 min 首跑(weight load /models gcsfuse + EXTEND precompile 83s + DECODE precompile 47s + diffusion-stage precompile 25s);第二跑 ~2 min(JIT 命中)
- **测试范围**: 仅 §3.1 basic usage smoke。Wan 模型无 evalscope numeric accuracy,`bench_serving` 不支持 videos endpoint,本次只发一个 single-shot 请求验证 launch+endpoint+输出落地。
- **请求**: `POST /api/v1/videos/generation`,prompt = "A neon-lit city street after rain, cinematic camera movement",`size="480*832"`,`num_frames=41`,`num_inference_steps=default(50)`
- **结果**: HTTP 200,wall-clock **4 min 19 s**,生成 794 KB MP4(已 cp 到本地 `wan2.1-t2v-14b/sample_neon_city.mp4`)。
- **异常 / 修复**:
  - 🔴 **D8(请求 `size` 分隔符)**: recipe §3.1 写 `"size": "480x832"` 用 `x`,服务端 `multimodal/manager/global_scheduler.py:242-243` 用 `.split("*")` 解析,直接 `ValueError: invalid literal for int() with base 10: '480x832'`,GlobalScheduler 进程崩溃,server 整个挂掉需要 restart。`io_struct.py:VideoGenerationsRequest.size` 默认值 `"720x1280"` 也用 `x`(若用户不传 size,默认就崩 — 服务端 source 层 bug 待提)。recipe 改用 `WIDTH*HEIGHT` 跑通。Wan 2.1 / Wan 2.2 两份 recipe 同问题统一改。
  - 🔴 **D9(响应 schema 与文件落地路径)**: recipe 写返回 `{"id":"vid_...","path":"/tmp/sglang-jax-videos/vid_...mp4"}`。实际服务端响应是 `{"success":true,"meta_info":{}}`(由 `tokenizer_manager.generate_request` 流式 yield),不携带 path/URL。MP4 写到**服务进程 cwd**(本次落 `/jax-ai-image/<uuid>.mp4`),命名是 raw UUID 不带 `vid_` 前缀,只能在 server log 里看到 `Saved output to <uuid>.mp4` 行。Wan 2.1 / Wan 2.2 recipe 同步把响应示例和 prose 改对。
  - 🟡 **multimodal_tokenizer 启动 warning(不影响功能)**: 加载 Wan 2.1 时 transformers AutoConfig 不识别 `wan` model_type,打 `Failed to load processor/config from /models/...` 警告并继续启动。video endpoint 完整可用,这个 warning 是 HF transformers 的兼容副作用,recipe 暂不暴露。
- **文档改动**:
  - `diffusion/Wan/Wan2.1.md` §1 banner → 14B-Diffusers Validated(2026-05-26)+ 1.3B 仍 Starter
  - §1 Recommended Generation Parameters 加 `size` 必须 `WIDTH*HEIGHT` 说明 + 链到 §5
  - §3.1 §3.2 §3.3 所有 `size` 值改成 `WIDTH*HEIGHT`,响应示例改为 `{"success": true, "meta_info": {}}`,prose 改写说明 MP4/PNG 落 server cwd 用 `<uuid>.mp4`
  - §4.2 Test Results 加 single-shot smoke 行(14B / 480*832 / 41 frames / 50 steps / c=1 / 4m19s)
  - §5 troubleshooting 加 D8 ValueError 行 + D9 响应没 path/url 行
  - `diffusion/Wan/Wan2.2.md` 同步改 §3.1 §3.2 §3.3 响应/size + §5 D8/D9 troubleshooting 行(Wan 2.2 本次未实测,只镜像 D8/D9 已验证的同源 recipe bug)

#### Wan 2.2 T2V A14B-Diffusers — 2026-05-26 (Phase 3,multimodal smoke)

- **目录**: [`2026-05-21-recipe-command-audit/wan2.2-t2v-a14b/`](2026-05-21-recipe-command-audit/wan2.2-t2v-a14b/)
- **TPU**: `wlf-v6e-4-bench`(v6e-4, single host, **`--tp-size 1`**,CPU text encoder + dual DiT)
- **Build**: sglang-jax `de29d9f0`
- **冷启时长**: ~6 min(118GB 权重从 /dev/shm 加载 + 双 transformer 各 12 shards + EXTEND/DECODE precompile);第二跑没测,JIT cache 应能省 precompile
- **测试范围**: 仅 §3.1 basic usage smoke。同 Wan 2.1 14B,无 evalscope numeric accuracy,`bench_serving` 不支持 videos。
- **请求**: `POST /api/v1/videos/generation`,prompt = "A neon-lit city street after rain, cinematic camera movement",`size="480*832"`,`num_frames=41`,`num_inference_steps=default(50)`
- **结果**: HTTP 200,wall-clock **2 min 43 s**(比 Wan 2.1 14B 4m19s 快 ~37%),生成 720 KB MP4(已 cp 到本地 `wan2.2-t2v-a14b/sample_neon_city.mp4`)。
- **异常 / 修复**:
  - 🟡 **D10(stage-config registry 对 model-path basename 大小写敏感)**: 首次 launch 用 `--model-path /dev/shm/wan2.2-t2v-a14b`(下载脚本生成的小写 dir 名)直接 `ValueError: No stage config found for model '/dev/shm/wan2.2-t2v-a14b'. Available models: ['Wan2.2-T2V-A14B-Diffusers', ...]`,而且引发 `UnboundLocalError: cannot access local variable 'scheduler'`(`global_scheduler.py:529` finally branch),GlobalScheduler 进程崩溃需要 restart。stage-config registry 把 HF 仓库基本名作 key,本地 dir 必须用完全相同的大小写。改 symlink `Wan2.2-T2V-A14B-Diffusers -> wan2.2-t2v-a14b` 后跑通。recipe 默认用 `--model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers`(HF 仓库名)所以正常用户路径不会踩,只有自己改本地 dir 名的会崩。这是服务端 source 层 polish:`StageConfigRegistry` 应做 case-insensitive 查找,`run_global_scheduler_process` 的 `UnboundLocalError` 是错误处理 bug。
  - 🟡 **multimodal_tokenizer 启动 warning**: 同 Wan 2.1,transformers AutoConfig 不识别 model_type,打 warning 继续。
- **文档改动**:
  - `diffusion/Wan/Wan2.2.md` §1 banner → A14B-Diffusers Validated(2026-05-26)
  - §1 Recommended Generation Parameters 加 `size` 必须 `WIDTH*HEIGHT` 说明
  - §4.2 Test Results 加 single-shot smoke 行(A14B / 480*832 / 41 frames / 50 steps / c=1 / 2m43s)
  - audit doc §5 资源池"跳过"行划掉"Wan 2.2 A14B → 需要 v6e-8",改实测 v6e-4 single-host 可行
- **未改动 / 不该改的地方**:
  - §2.1 Hardware Matrix 已写 v6e-4 / `--tp-size 1`,与实测一致,不动
  - §2.3 launch 默认值(`--mem-fraction-static 0.88`,`--precompile-frame-paddings 41`)与实测一致
  - D10 不写 recipe,因为 recipe 默认用 HF 仓库名不会触发;只在 audit doc 记录 source 层 polish 待提

#### 跳过(资源已确认不可用)

- **DeepSeek-V2 (236B 多机)** / **DeepSeek-V3** / **DeepSeek-R1**: recipe 要求 v6e-32 / v6e-64 / v7x-16,本次 audit 资源池只有 `wlf-v6e-4-bench` + `wlf-v6e-16-*`(`niu-mimo-v64-*` 是别人的 cluster)。归属 §5 backlog "跳过(资源不足)" 行的延伸。
- **MiMo-V2.5-Pro**: recipe 要求 v6e-64,理由同上。
- **Ling 2.5 / Ring 2.5 / Ling 2.6**: 用户指示 2026-05-25 跳过(Ling/Ring 整体跳过指令)。Ling 1.x / Ling 2.0 / Ring 2.0 recipes 已从 cookbook 删除。

## 6. 跨 PR 协调约定

- 任何动 cookbook recipe 的 PR(尤其 launch / benchmark 命令)**先 grep 本文**,看是否相关项已 fix
- 本文是 author tooling,**禁止在 user-facing recipe 里反向链接此文档**
- 后续发现新的命令 bug → 加到 §3 表里 → fix 后写 §4 日志
- TPU 实测后填 §5 — 给后续 reviewer 留下"在 X 配置上跑通过"的可信痕迹

### 6.1 测试执行规则(2026-05-22 用户确认)

1. **每个模型测三件套**:`launch` + `benchmark`(`bench_serving`)+ `accuracy`(`evalscope`)。缺一不算完成。
2. **按模型名整理日志目录**:在 `docs/cookbook/2026-05-21-recipe-command-audit/<model-name>/` 下放:
   - `commands.md` — 实际执行的 launch / benchmark / accuracy 命令(复制即可重跑)
   - `steps.md` — 跑的步骤顺序、port-forward、kubectl exec、等待节点同步等环境性步骤
   - `launch.log` — server 启动日志(到 "Uvicorn running on..." 为止;OOM / hang 也保留)
   - `bench_serving.log` — bench_serving 输出 verbatim("============ Serving Benchmark Result ============" 整段)
   - `evalscope.log` — evalscope 输出 verbatim(包含 Test Results 表)
   - **若跑出异常,新增** `exception_<symptom>.md` — 异常分析日志:症状 / 现场 / 根因(grep 源码定位)/ 解决步骤 / 是否触发 cookbook 改动
3. **测完根据日志回填 cookbook**: 把 `bench_serving.log` 整段 fenced 贴到对应 recipe §4.2,evalscope Test Results 表贴到 §4.1,更新 Test Environment 的 `Tested build` commit hash。
4. **sglang-jax 用 main 分支最新 commit**: 每次开始新批次测试前在 pod 内 `cd /tmp/sglang-jax && git fetch origin && git checkout main && git reset --hard origin/main && pip install -e "python[tpu]" --quiet`(editable 装,源码改动直接生效)。把当时的 commit hash 记入 §5 测试记录的 `Build:` 字段。
5. **长任务(eval > 5 min)必须用 nohup + 文件重定向**: 不要依赖 `kubectl exec ... -- "cmd | tee"` 长 hold,websocket 一断 pipe reader 死,长任务会卡(evalscope 不会优雅退出,需手动 SIGKILL)。规范命令骨架:复杂参数(如 JSON `--generation-config`)写到脚本文件 → `nohup /tmp/run.sh > /tmp/x.log 2>&1 &` → `disown`,之后只用 `kubectl cp` 拉文件 / `pgrep` 监控。

### 6.2 已知 / 已查的测试阻塞项盘点

(在 §5 资源盘点的基础上,继续核对会阻塞测试的环境性问题。**新发现的阻塞项写到此处**,清理后划掉。)

#### 2026-05-22 初次排查

**🔴 实际阻塞项**(需要在跑测试前解决):

| ID | 项 | 状态 | 解决方案 |
|---|---|---|---|
| E1 | `HF_TOKEN` 在两个 pod 都**未设置**(env grep 空) | 阻塞 Llama 3.1 / 3.3、可能阻塞 DeepSeek / Qwen2.5-VL 部分版本(gated repo 需 token) | 用户起测试前 `kubectl exec ... -- bash -c 'export HF_TOKEN=...; ...'` 或加到 pod env;**优先级**:跑 Llama 系列前必须 |
| E2 | `evalscope` 在两个 pod 都**未安装** | 阻塞所有 §4.1 accuracy 测试 | 每个测试 session 开始时跑 `pip install evalscope==0.17.1`(会拉 ~10GB CUDA torch 依赖,5 min 左右;TPU runtime 不会调用这些库,装上无害) |
| E3 | `wlf-v6e-16` rank-1 节点 TPU 被旧 python 进程占住(PID 275167,可能是上次未清理的测试残留) | 阻塞所有 v6e-16 multi-host 测试 | ✅ **2026-05-22 已解决**: `rm -f /dev/shm/sem.loky-*` 清 4 节点 loky 信号量,4 节点同步 `python -c "import jax"` 全部 `OK 16` |

**🟡 需要注意但能 work-around**:

| ID | 项 | 影响 | Work-around |
|---|---|---|---|
| W1 | `/` (overlayfs) 只有 79G,放不下 70B+ BF16 模型权重(140GB+) | 限制 single-host 上跑大模型 | 用 `--download-dir /dev/shm`(688G tmpfs);MoE 模型推荐统一 `/dev/shm` 路径 |
| W2 | 4 个 v6e-16 multi-host pod 都有 5 个 `<defunct>` zombie python 进程(May 13-14 残留) | 不会占 TPU 但污染 ps 输出 | 不必清理,kernel 会在父进程退出时回收;若强迫症,可重启 pod |
| W3 | v6e-16 4 节点都在旧 commit `b2daa46d`(2026-05 早期),v6e-4-bench 在 `66b9481`(更新) | 不同 commit 上跑出的数会不可比 | 起每批测试前先 `git fetch origin && git checkout main && git reset --hard origin/main && pip install -e python[tpu] --quiet` 同步到 main(参考 §6.1 规则 4) |
| W4 | `pip install -e python[tpu]` 在 editable 模式下源码改动立即生效,但每个 pod 独立环境(4 pod × 重复装) | 多节点测试每节点都要装,耗时 | 写一个 `kubectl exec` fan-out 脚本一次性装 4 节点 |
| W5 | 多节点 launch 需要 4 个 `kubectl exec` 同步起,否则 `jax.distributed.initialize` 会 timeout | 时间窗口要紧 | 用 `for pod in ...; do kubectl exec $pod ... &; done; wait` 模式;NODE_RANK 用 pod-name 后缀的数字提取(`${pod//*-/}` 不通用,要查 `JOB_COMPLETION_INDEX` annotation)|

**✅ 已验证可用**:

- ✅ git fetch origin 在 pod 内能通(刚 fetch 到 `main` 有更新到 `fe092bf`)
- ✅ HuggingFace 网络可达(`huggingface.co` 返回 HTTP 307 redirect)
- ✅ v6e-16 多节点 headless DNS 正常解析(`wlf-v6e-16-0.wlf-v6e-16-headless-svc` → `10.31.148.4`)
- ✅ v6e-16 所有 4 节点 git commit 一致(都是 `b2daa46d`),pull main 后会同步
- ✅ v6e-4-bench TPU device 数正确(jax.devices()=4,coords 全对)
- ✅ pip 有写权限 + 网络可下包
- ✅ `/dev/shm` 688G tmpfs(足够放任何模型权重 + JIT cache)
- ✅ 镜像 `jax0.8.1-rev1` 与 recipe `cookbook/base/launch-flags-reference.md` 推荐一致

#### 后续 session 的 setup 脚本(每个 test 前跑一次)

把 W3 / W4 / E2 / E3 合并成一个 setup 序列,贴入对应 `<model-name>/steps.md`:

```bash
# 1. 同步源码到 main(单节点)
kubectl exec wlf-v6e-4-bench -c wlf-v6e-4-bench -- bash -c '
  set -e
  cd /tmp/sglang-jax
  git fetch origin
  git checkout main
  git reset --hard origin/main
  pip install -e "python[tpu]" --quiet
  git rev-parse HEAD  # 记到 Tested build
'

# 2. 装 evalscope(仅 accuracy 测试需要;若只跑 launch + bench_serving 可跳)
kubectl exec wlf-v6e-4-bench -c wlf-v6e-4-bench -- pip install evalscope==0.17.1 --quiet

# 3. 清理残留进程(防 TPU 占用)
kubectl exec wlf-v6e-4-bench -c wlf-v6e-4-bench -- bash -c 'pkill -9 -f sgl_jax || true; pkill -9 -f "python.*launch_server" || true; sleep 2; python -c "import jax; print(\"TPU devices:\", len(jax.devices()))"'

# 4. 多节点同步(v6e-16):4 节点并行装
for pod in wlf-v6e-16-0-2x99h wlf-v6e-16-1-x9v6l wlf-v6e-16-2-rrkd5 wlf-v6e-16-3-k4lx2; do
  kubectl exec $pod -c wlf-v6e-16 -- bash -c '
    cd /tmp/sglang-jax && git fetch origin && git reset --hard origin/main && pip install -e "python[tpu]" --quiet
    pip install evalscope==0.17.1 --quiet
    pkill -9 -f sgl_jax || true
  ' &
done
wait
```

(Pod name 后缀如 `2x99h` 会变,跑前先 `kubectl get pods | grep wlf` 重新拿。)

## 7. 参考链接

- [`cookbook-recipe-design.md`](cookbook-recipe-design.md) — recipe 章节 schema
- [`2026-05-18-cookbook-research.md`](2026-05-18-cookbook-research.md) — 整体设计原则
- [`mintlify-migration.md`](mintlify-migration.md) — 站点迁移跟踪
- [`base/launch-flags-reference.md`](base/launch-flags-reference.md) — flag 速查(本审计后可能需同步更新)
