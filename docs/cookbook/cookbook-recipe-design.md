# Cookbook Recipe 设计

> 单篇 recipe 应当遵循的章节结构和内容范围。本文档只陈述设计事实——"recipe 长什么样、每节写什么"；设计原则、写作约束、checklist 等辅助内容见 [`2026-05-18-cookbook-research.md`](2026-05-18-cookbook-research.md)。

---

## 1. Recipe 5 节固定结构

每篇 recipe 是一个 markdown 文件,位于 `docs/cookbook/<task-type>/<model-name>.md`:

- `<task-type>` ∈ `autoregressive` / `multimodal`
- `<model-name>` 是扁平模型名,如 `mimo-v2-flash` / `qwen3-moe`

文件主体 5 节固定结构:

```markdown
# <Model Name> on SGL-JAX

## 1. Model Introduction

## 2. Deployment
   ### 2.1 Hardware Matrix
   ### 2.2 Environment
   ### 2.3 Launch
       #### Single-host (Docker)              (如适用)
       #### Multi-host (GKE 或 SkyPilot)       (如适用)
   ### 2.4 Configuration Tips

## 3. Invocation
   ### 3.1 Basic Chat Completion
   ### 3.2 Reasoning                (如模型支持 reasoning)
   ### 3.3 Tool Calling             (如模型支持 tool-calling)
   ### 3.x Multimodal Input         (VL / Audio 模型;首位置取代 §3.2)

## 4. Benchmark
   ### 4.1 Accuracy
   ### 4.2 Speed                    (从 6 种 layout 菜单选一种)

## 5. Troubleshooting                (可选)
```

---

## 2. 各节内容概要

### 2.1 §1 Model Introduction

| 项 | 必填 / 推荐 | 内容 |
|---|---|---|
| 模型简介 | 必填 | 一句开头 `[<Model>](<HF-url>) is/are ...`,2-3 句描述模型定位 |
| HF checkpoint 链接 | 必填 | 至少一个;多 variant 时每个列加粗段 |
| Key Features | 推荐 | 4-7 条架构特性 bullets,从**用户视角**取材(模型擅长什么 / 关键参数规模 / 加速能力等);避免堆底层实现细节 |
| Recommended Generation Parameters | 推荐 | temperature / top_p / max_tokens;reasoning 模型给 thinking-on/off 双值 |
| License | 推荐 | 引导读者去 HF model card 看授权;不在 recipe 内复述具体 license 文本 |
| 官方 blog / 论文 | 可选 | 一行 link |

> **写作建议(对应辅助文档 §2.8)**:写新 recipe 前先检查 `sgl-project/sgl-cookbook` 是否有同名 recipe;若有,§1 措辞 / Key Features 取舍角度可借鉴(冲突以 sglang-jax 代码为准)。

**single-variant 规则**:只有 1 个 HF checkpoint 时,**不要**列 "Available Models",直接 inline 到第一句:`[Model](url) is ...`

**multi-variant 规则**:每个 variant 一段加粗段 `**[Variant Name](HF-url)** is <one-line description + 推荐用途>`。

**好模式**(从 sgl-cookbook 对比中固化):

- **关键参数规模优先**——开头 2 句话内出现总/活跃参数(如 `309B total / 15B activated`)、context window 长度。用户最先想知道模型量级。
- **设计意图 / 定位句**——讲 why,不只讲 what。例:`inference-centric model designed to maximize decoding efficiency, enabling flexible tradeoffs between throughput and latency`。
- **架构关键比例 + 实际收益**——不只数字,还要给用户视角的收益。例:`5:1 SWA:GA ratio with 128-token window — reduces KV cache 6x while maintaining long-context quality`。
- **自带加速能力放 Key Features**——MTP / Speculative 这种"开箱即用的 throughput 收益"用户关心,放前面。
- **能力定位标签**——一行 caps:`reasoning / agentic / vision / code` 让用户秒判模型适用场景。
- **训练 / context scale**——`27T tokens, native 32k seq, supports 256k context` 这种规模数据增强信任。
- **避免**:堆 head_dim / phantom token softmax 这类 kernel-level 实现细节(用户不关心,放 §2.4 调参里);避免内部 class 名 / 源码路径。

**参考样本**(写作时对照):`sgl-cookbook/docs/autoregressive/Xiaomi/MiMo-V2-Flash.md`、`Moonshotai/Kimi-K2.6.md`、`DeepSeek/DeepSeek-V3_2.md`。

### 2.2 §2 Deployment

#### 2.2.1 §2.1 Hardware Matrix

一张表,列字段:

```
| TPU | Topology | Chips/node | Nodes | Total chips | --tp-size | --dp-size | --ep-size | Notes |
```

至少包含一行"最小可跑配置"。多 TPU 平台支持时各一行。

**好模式**(从 sgl-cookbook 对比中固化):

- **多档配置并列**——"minimum runnable" + "recommended production" 两行,让读者看清取舍。参考 sgl-cookbook Llama3.1 §3.2 列出 MI300X/MI325X/MI355X 各自 verified TP。
- **Notes 列说"为什么"**——不只列数字,标注关键 tradeoff(例:`v6e HBM 较紧,推荐降低 mem-fraction-static 到 0.92`)。
- **TPU-specific 提醒明文标注**——v7x "2 JAX devices/chip" 不写出来普通用户会算错 `--tp-size`,必须 cross-link `base/tpu-topology-reference.md`。
- **不要**散文描述硬件 —— TPU 多维信息(generation × chips × nodes × tp × dp × ep)只有表格能扫读清晰。

#### 2.2.2 §2.2 Environment

三块内容:

1. 一句话引用安装入口(`get_started/install.md` + `cookbook/deployment/<launcher>.md`),不重复完整步骤
2. **JAX TPU Image by Hardware 表**(必填):

```markdown
| Hardware Platform                  | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)     | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)                 | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
```

3. 本 recipe 特有的版本固定(如额外 pip 包)

**好模式**(从 sgl-cookbook 对比中固化):

- **Docker Images 表跨多 image tag 时分列**——参考 sgl-cookbook Gemma4 §2 列 CUDA 12.9 / 13 / AMD ROCm 各一行。TPU 类似:v5/v6e/v7x 若用同一 image 可合并,若 docker tag 不同则分行。
- **明确 image tag pin 到版本**(不要用 `latest`)——如 `jax0.8.1-rev1` 锁定。
- **本 recipe 特有 pip 包独立列**(如 `evalscope==0.17.1`)—— 通用 install 步骤交给 deployment 模板,这里只列差异。

#### 2.2.3 §2.3 Launch

用四级标题区分部署规模。每个四级标题**自包含完整链路**——读者按部署规模直接跳到对应子节,不需要跨节拼参数。

**`#### Single-host (Docker)`(如适用)**:
- 完整 launch_server 命令
- 必带:`JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` / `--trust-remote-code` / `--device tpu` / `--dtype bfloat16` / `--skip-server-warmup` + 模型特有 flag
- 推荐在 launch 命令前贴单行 `docker run`(含 TPU 必备 flag)

**`#### Multi-host (GKE 或 SkyPilot)`(如适用)**:
- 多节点 launch 命令,带 `${NODE_RANK}` / `${MASTER_ADDR}` 占位 + rank 范围说明
- 必须显式 launcher 选择:GKE Indexed Job + headless Service 或 SkyPilot;给出**差异化字段**(model path / TPU label / TP-DP-EP 等),引用 `cookbook/deployment/<launcher>.md` 通用模板,不复制整段未差异化的 manifest

**好模式**(从 sgl-cookbook 对比中固化):

- **命令缩进 2 空格**——sgl-cookbook 12/12 一致;flag 一行一个,便于读者复制 / 注释 / 删除。
- **关键 env 紧贴命令头**——`JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m sgl_jax.launch_server \\` 一起写在第一行,提醒读者必带。
- **占位用 `${VAR}` 形式**——`${NODE_RANK}` / `${MASTER_ADDR}` 而不是 `<NODE_RANK>` / `<MASTER_ADDR>`(读者直接在 shell 用 `export NODE_RANK=...` 即可生效)。
- **rank 范围说明跟在命令后一行**——"${NODE_RANK} ranges from 0 to 3"(DeepSeek / GLM / Qwen3 共同模式)。
- **flag 分组用空格隔开**——`--tp-size 32 --dp-size 4 --ep-size 32 \\` 一行,`--mem-fraction-static 0.95 \\` 一行,按"逻辑组"折行(并行参数一组、内存参数一组、parser 一组)。
- **避免 hardcode 端口**(如 8000)—— 统一 30000 与 §3 invocation 一致;sgl-cookbook 自家有 5/12 违反这条,被 SKILL.md review checklist 明令禁止。
- **避免 `python -m sglang.launch_server`** —— 那是 SGLang 主仓的 deprecated 命令;sgl-cookbook 自家 6/12 还在违反,我们不要踩。

#### 2.2.4 §2.4 Configuration Tips

按主题驱动 bullet 组织,每个主题独立可执行。主题来源见 §4 主题菜单。

格式(主题驱动 bullet):

```markdown
**主题名**
- 本 recipe 的取值 + 理由
- 调参方向(raise/lower 各自影响)
- 配套 observation(server logs / metric / flag 组合)
```

**好模式**(从 sgl-cookbook 对比中固化):

- **主题用 `**Subject:**` 起头,不用 `### Subject`**——参考 sgl-cookbook Llama3.1 / Kimi-K2.6 / Mistral-Small-4。视觉紧凑,不打断 §2 整体节奏。
- **给具体可粘贴的配置表**——Llama3.1 §3.2 "Verified TP Configurations" 给 `MI300X/MI325X: 405B BF16 (TP=8), 405B FP8 (TP=4)`,读者可以直接抄。
- **给具体 HF 路径,不只给类型**——Llama3.1 §3.2 "FP8 Model Variants" 给 `meta-llama/Llama-3.1-405B-Instruct-FP8` / `amd/Llama-3.1-70B-Instruct-FP8-KV`,而不只说"用 FP8 版本"。
- **platform-required flag 明文标"必带"**——AMD `--attention-backend triton` + `SGLANG_USE_AITER=1` 这种必须无条件加,不要藏在"推荐"语气下让用户漏看。
- **`fused_moe_triton` 等 tuning 入口给 cross-link**(Qwen3 §3.2 风格)—— recipe 不深入 kernel tuning 细节,指向 base/ 参考文档。
- **观测点配 server logs 字段**——"watch `swa token usage` / `full token usage` in logs"(MiMo SWA 主题模式)。让调参可验证,不是黑盒猜。
- **本 recipe 选这个值的"为什么"必备**——不能只列"我们用 `--mem-fraction-static 0.95`",必须给理由("weights ~20 GB/chip in FP8 — fits with headroom")。

**参考样本**:`sgl-cookbook/docs/autoregressive/Llama/Llama3.1.md` §3.2(主题分类最规整);`Moonshotai/Kimi-K2.6.md` §3.2(覆盖 AMD-specific 主题最全)。

### 2.3 §3 Invocation

#### 2.3.1 §3.1 Basic Chat Completion

- 1 个 curl 例子(OpenAI-compatible `/v1/chat/completions`)
- 推荐再加一个简短 Python OpenAI client 例子(非 streaming)

**好模式**:
- **不 hardcode 采样参数**(`temperature` / `top_p`)—— 让 server 用 `generation_config.json` 默认;recommended params 已经在 §1 informational 列出。这对应 SKILL.md 硬规则。
- **端口与 §2.3 launch 一致**(30000)——sgl-cookbook 自家 GLM-4.5/4.6 同 page 混用 8000/30000 是 review 高频问题,我们不踩。

#### 2.3.2 §3.2 Reasoning(如模型支持 reasoning)

- 完整 launch 命令(带 `--reasoning-parser <key>`)
- 完整 Python streaming 客户端
- Output Example(用 ```text fenced block)
- **Hybrid 模型必须 thinking-on(默认)+ thinking-off 两个例子**

**好模式**(从 sgl-cookbook DeepSeek-V3_2 / GLM-4.6 / Qwen3 几乎逐字共用的 streaming 模板):

- **状态变量管理**:`thinking_started` / `has_thinking` / `has_answer`(或简化为 `thinking_started` / `content_started`)管理 streaming 输出的 section 切换
- **delta 双重检查**:`if hasattr(delta, 'reasoning_content') and delta.reasoning_content:`(防 None / 缺字段)
- **打印分隔块**:固定用 `=============== Thinking =================` / `=============== Content =================` 作为人眼可辨的 section 边界
- **`flush=True` 必带**:streaming 体验不 flush 用户看不到 token-by-token 输出
- **Output Example 真实截断,不编**:thinking 段保留前 3-5 行真实推理 + ellipsis + Content 段;不要编"理想化"输出
- **Inline tags parser**(如未来支持 `minimax-append-think`)用另一套客户端代码:walk a buffer looking for `<think>...</think>`,split as 你 print

**参考样本**:`sgl-cookbook/docs/autoregressive/DeepSeek/DeepSeek-V3_2.md` §4.2.1 是这套模板的最完整版本(60+ 行 Python + Output Example)。

#### 2.3.3 §3.3 Tool Calling(如模型支持 tool-calling)

- 完整 launch 命令(带 `--tool-call-parser <key>`,如有 reasoning 也带 `--reasoning-parser`)
- 完整 Python tools=[...] 例子(含 function definition + `tool_choice="auto"`)
- Output Example
- **必备 Handling Tool Call Results 段**(多轮对话例子,从 tool result 回到 assistant final response)
- Thinking 模型 + tool-call 时,final response 必须**同时打印 reasoning_content + content**

**好模式**(从 sgl-cookbook DeepSeek-V3_2 / Qwen3 共用):

- **`tool_calls_accumulator = {}` 累加 streaming tool calls**——streaming 时一个 tool call 的 name / arguments 跨多个 delta,需要按 `tool_call.index` 累加
- **打印格式 `🔧 Tool Call: <name>` + `   Arguments: <args>`** —— 缩进 + emoji 让 tool call 在 thinking + content 输出中视觉突出
- **function definition 用 `get_weather(location, unit)` 这种通用示例**——`tools` 定义 schema 时用 OpenAI 标准 JSON Schema 而不是简化版
- **Handling Tool Call Results 段 3 消息**:user prompt / assistant (`content=None, tool_calls=[...]`) / tool (`tool_call_id + content=<function result>`)
- **thinking + tool-call final response 双打印**——Validated recipe 必带:`print("Reasoning:", final.choices[0].message.reasoning_content)` + `print("Content:  ", final.choices[0].message.content)`。Hybrid 模型 thinking 模式下 content 可能为 None,只打 content 会误导。

**参考样本**:`sgl-cookbook/docs/autoregressive/Qwen/Qwen3.md` §4.2.3(streaming + thinking + tool-call 三合一最完整);`DeepSeek-V3_2.md` §4.2.2 Handling Tool Call Results 段范本。

#### 2.3.4 §3.x Multimodal Input(VL / Audio / Diffusion 模型)

仅 VL/Audio 模型加。**取代 §3.2 Reasoning 作为首位 advanced 子节**。包含:

- Image Input
- Multi-Image Input
- Video Input(或 Audio Input,按 modality 调整)

每个 block 含完整 Python + Output Example。

**好模式**(从 sgl-cookbook Qwen3-VL 学):
- **三个 block 各完整**:Single Image / Multi-Image / Video 三个 Python 块独立可执行,不省略
- **`image_url` / `video_url` 标准字段**——OpenAI Vision API 兼容字段
- **§3.1 Basic Usage link 多一条**——`SGLang OpenAI Vision API Guide`(我们对应 sglang-jax 的 vision API doc)

**参考样本**:`sgl-cookbook/docs/autoregressive/Qwen/Qwen3-VL.md`。

### 2.4 §4 Benchmark

#### 2.4.1 §4.1 Accuracy

四件套结构:

```
**Test Environment** 表(字段全集见下)
**Deployment Command** 引用 §2.3 命令(不重复贴)
**Benchmark Command** 完整 evalscope 命令
**Test Results** 表(Model / Dataset / Metric / Subset / Num / Score)
```

**Test Environment 字段**(对齐 sgl-cookbook 主流的 4 核心字段 + 5 个按需 optional):

| 字段 | 必填? | 示例 |
|---|---|---|
| Hardware | ✓ | `TPU v7x (16 chips, 4 nodes, 2x2x4)` |
| Model | ✓ | `XiaomiMiMo/MiMo-V2.5-Pro (FP8)` —— quantization 后缀必须匹配 §1 列的 variant;dtype 信息也通过此后缀承载 |
| Tensor Parallelism | ✓ | `32` |
| Tested build | ✓ | `sglang-jax <hash>` 或 `sglang-jax v0.x.y`;可附 `(YYYY-MM-DD)` 测试日期 |
| Data Parallelism | optional | `4` —— DP 启用时填 |
| Expert Parallelism | optional | `32` —— MoE 模型填 |
| Reasoning Parser | optional | `mimo` —— reasoning 启用时填 |
| Tool Call Parser | optional | `mimo` —— tool-call 启用时填 |
| Docker Image | optional | `us-docker.pkg.dev/.../tpu:jax0.8.1-rev1` —— 跨多 docker 时填 |

> **Speed-only 字段(不放表里)**:Workload 描述(如 `256 prompts, ISL=16384, OSL=1024, concurrency=64`)写在 Test Environment 段之后的散文里,不作为表格字段。

**Accuracy dataset 推荐(按模型类型)**:

| 模型类型 | 必跑 | 推荐补 |
|---|---|---|
| Dense LLM (general) | GSM8K | + MMLU |
| Reasoning LLM | AIME 2025 | + GPQA Diamond + MATH |
| MoE LLM | GSM8K | + MMLU |
| VL 模型 | MMMU | + MMMU Pro Vision |
| Code 模型 | HumanEval | + MBPP |

#### 2.4.2 §4.2 Speed

四件套 + 从 6 种 layout 菜单(见 §5)选一种。Layout 选择本身要在 §4.2 顶部一行注明:`本 recipe 使用 Layout <X>`。

**好模式**(从 sgl-cookbook 对比中固化):

- **Accuracy 结果用 markdown 表**(Kimi-K2.6 / Gemma4 风格)而不是 fenced 包文本——便于跨 recipe 横向对比 + GitHub 渲染对齐
- **Reasoning 模型 accuracy 加 `pass@1 (avg-of-N)` / `majority@N` / `pass@N` 多列 + No Answer 列**——SKILL.md §115-122 建议
- **Speed 结果(bench_serving 输出)用 ```text fenced 包**——50+ 行 `============ Serving Benchmark Result ============` 块保留 verbatim,**不要 paraphrase**
- **多 config sweep 用表对比**——参考 mimo-v2-flash 的 MoE backend × prefill × SWA × mem 4 行表;一目了然
- **跨硬件用 top-level 多 §4.x**(Kimi-K2.6 风格)——而不是把所有硬件挤在一个表里;每个硬件一个完整 Test Environment + 数据
- **多 variant 用 ##### sub 命名**(Gemma4 风格 `##### gemma-4-31B-it (1x H200, TP=2)`)——variant + hardware + TP 在标题里一眼看清
- **Workload 描述放 Test Environment 段之后的散文里**(已固化)
- **加 vLLM 等竞品对比是加分项**(参考 qwen3 现有 recipe)——但不强制

**参考样本**:`sgl-cookbook/docs/autoregressive/Moonshotai/Kimi-K2.6.md` §5(多硬件分块);`Google/Gemma4.md` §5(变体 × 硬件矩阵)。

### 2.5 §5 Troubleshooting

可选。仅放本模型特有 3-5 条问题。

格式:三列表 `Symptom | Likely cause | Fix`。

通用问题指向 `cookbook/troubleshooting.md`,不在每个 recipe 重复。

**好模式**:

- **Symptom 用具体可观测信号**——"OOM at startup" / "SWA pool exhaustion at runtime" 比 "memory error" 更可定位
- **Likely cause 给一句技术解释**——不只重复 symptom,要说"为什么"(如 "weights don't fit" / "too much concurrent decode demand")
- **Fix 给具体 flag 调整数值**——"Lower `--mem-fraction-static` to 0.90" 比 "reduce memory" 可执行
- **本模型特有**——通用 OOM / cache / 多节点连接问题进 `cookbook/troubleshooting.md`,这里只放因模型架构特性(SWA pool / linear attention recurrent state 等)引发的问题
- **每条 Fix 与 §2.4 Configuration Tips 主题呼应**——troubleshooting 是 Configuration Tips 在"出问题时"的视角,两者交叉参照

---

## 3. Starter vs Validated 完成度矩阵

每篇 recipe 有两档状态:
- **🚧 Starter**:基础骨架,允许 Pending / TODO
- **✅ Validated**:有真实实测数据,移除 Starter banner

| 节 | Starter 最低 | Validated 增量 |
|---|---|---|
| §1 Introduction | 简介 + variant + HF link | + Key Features bullets + Recommended Generation Parameters |
| §2.1 Hardware Matrix | 至少一行 starter target(数值可 TODO) | 全部数值实测 |
| §2.2 Environment | 引用 deployment + install;JAX TPU Image by Hardware 表必填 | 表覆盖所有 §2.1 列出的 TPU 平台 |
| §2.3 Launch | 至少 single-host 或 multi-host 一个完整命令(含 launcher manifest 如多 host) | 全部支持的 launcher × topology 覆盖 |
| §2.4 Config Tips | 可选 | 必填 4-6 个主题(含 flag 取值理由) |
| §3.1 Basic | curl 一个例子 | + 简短 Python client |
| §3.2 Reasoning | 一句话引用其他 recipe | 完整 Python streaming + Output Example;hybrid 模型必须 thinking-on + off 双例 |
| §3.3 Tool Calling | §2.4 主题里提一行 parser key | 完整 Python tools + Output + 必备 Handling Tool Call Results 段 |
| §3.x Multimodal | 仅 VL/Audio 模型 | Single/Multi-Image/Video 三个 block |
| §4.1 Accuracy | Test Env + 命令模板 + Test Results 标 Pending | 至少 1 个 dataset 真实输出 |
| §4.2 Speed | Test Env + 命令模板 + 选定 layout 占位结构 | 至少 1 格真实 `============ Serving Benchmark Result ============` 块 |
| §5 Troubleshooting | 可选 | 列出本模型特有 3-5 条 |

**Banner**:
- Starter 顶部: `> **Starter recipe** — Not yet empirically validated on TPU. Tune values for your hardware and PR-back tested numbers.`
- Validated:移除该 banner;在 `autoregressive/index.md` 中 status 从 🚧 升级为 ✅

---

## 4. Configuration Tips 主题菜单

按模型类型组合挑 4-6 个主题:

| 类别 | 主题 | 适用模型 |
|---|---|---|
| 通用 | Memory Management (`--context-length` / `--mem-fraction-static`) | 所有 |
| 通用 | Compilation Cache Hygiene (`JAX_COMPILATION_CACHE_DIR`) | 所有 |
| 通用 | Chunked Prefill Tuning (`--chunked-prefill-size`) | 所有 |
| 并行 | Tensor Parallelism (`--tp-size` + v7x 2 device/chip 提醒) | 所有 multi-device |
| 并行 | DP Attention (`--dp-size` + attention TP 关系) | MoE / 多节点 dense |
| 并行 | Expert Parallelism (`--ep-size = --tp-size` 模式 + 例外) | MoE |
| MoE | MoE Backend Selection (`--moe-backend fused` vs `epmoe`) | MoE |
| MoE | Kernel Tuning(tuned block configs / fused 的 EP shape 覆盖) | MoE |
| Attention | SWA Pool Sizing (`--swa-full-tokens-ratio` per-layer 配比) | hybrid attention(MiMo / Gemma 2) |
| Attention | Attention Backend Selection (`--attention-backend fa` / `fa_mha`) | MLA / 特殊 attention |
| Linear | Recurrent State Memory Ratio (`--recurrent-state-memory-ratio`) | linear attention(Kimi-Linear / Bailing Linear) |
| 加速 | Speculative Decoding (`--speculative-algorithm NEXTN/EAGLE` + draft model) | 支持的模型 |
| Reasoning | Reasoning Parser 选择 (`--reasoning-parser` key) | reasoning 模型 |
| Tool | Tool Call Parser 选择 (`--tool-call-parser` key) | 支持 tool-call 的模型 |
| 多模态 | Multimodal Attention Backend (`--mm-attention-backend`) | VL 模型 |
| 多模态 | TTFT Optimization for Vision | VL 模型 |

---

## 5. Benchmark Layout 菜单

不强制 3×3 scenario × concurrency 矩阵。从下列 6 种 layout 按 recipe 性质挑一种:

| Layout | 子节结构 | 适用场景 | 工作量 |
|---|---|---|---|
| **A. 完整 3×3 scenario × concurrency 矩阵** | §4.2.1/2/3 = Standard/Reasoning/Summarization;每节下 ##### Low/Medium/High Concurrency | 标杆 / Reference recipe(社区横向对比金标准) | 高(9 次 bench_serving run) |
| **B. 方法论 + 命令模板**(无真实数据) | §4.2.1 Standard Test Scenarios / §4.2.2 Concurrency Levels / §4.2.3 Number of Prompts / §4.2.4 Benchmark Commands / §4.2.5 Understanding the Results | 早期 recipe / Starter 起步 | 低(无数据,仅命令模板) |
| **C. Latency vs Throughput 二分** | §4.2.1 Latency-Sensitive / §4.2.2 Throughput-Sensitive | batch 与 realtime 区分明显的模型 | 中(2 次 run) |
| **D. 多硬件分块** | §4.2.1 / §4.2.2 ... 每个 top-level 子节是不同 TPU 代次 | 跨多个 TPU 代次都跑过的 recipe | 中-高(N 硬件 × M run) |
| **E. 变体 × 硬件矩阵** | §4.2.1 / §4.2.2 ... 按 `<variant> on <hardware>` 命名 | 多 variant 的模型家族 | 中(N 变体 × M 硬件) |
| **F. 单 workload sweep**(配置 sweep) | 单一 ISL/OSL/concurrency,变化 flag 维度(`--moe-backend` / `--chunked-prefill-size` / `--swa-full-tokens-ratio` 等) | Starter recipe / 配置选型 sweep | 极低(1-N 次 run,共享 workload) |

**Scenarios 定义**(layout A 用):

| Scenario | Random ISL | Random OSL | 模拟 |
|---|---|---|---|
| Standard chat | 1000 | 1000 | prefill/decode 平衡 |
| Reasoning long output | 1000 | 8000 | 压 decode + SWA pool |
| Summarization long input | 8000 | 1000 | 压 `--chunked-prefill-size` |

**Concurrency 档位**(layout A 用):

| Level | `--max-concurrency` | `--num-prompts` | 测什么 |
|---|---|---|---|
| Low | 1 | 10 | min TTFT / ITL 基线 |
| Medium | 16 | 80 | 生产典型 |
| High | 64(或 100) | 320(或 500) | 吞吐天花板 |

---

## 6. Special-case 章节裁剪规则

5 类 special-case 的章节调整:

### A. Reasoning-only model

- §2.4 Configuration Tips 必须包含 Reasoning Parser 主题
- §3 可省略 §3.3 Tool Calling(如模型不支持);§3.2 Reasoning 必填
- §4.1 Accuracy 用 reasoning dataset(AIME / GPQA / MATH)+ `--reasoning-parser` 启用

### B. VL / Multimodal model

- §3.2 Multimodal Input **取代 Reasoning 作为首位 advanced 子节**
- §3.2.x 拆为 Image / Multi-Image / Video 三个 Python block
- §4.1 Accuracy 用 vision dataset(MMMU / MMMU Pro Vision)
- §2.4 必含 Multimodal Attention Backend 主题

### C. Audio model

- §3.2 Audio Input 首位
- §4.1 用 ASR dataset(CommonVoice / LibriSpeech / FLEUR)
- §2.4 含 Audio-specific 主题

### D. Linear-attention model

- §2.4 **必须**含 Recurrent State Memory Ratio 主题
- §2.3 Launch 命令必带 `--recurrent-state-memory-ratio`(即便用默认 0.9)

### E. 极简 / dev-pinned recipe

- §1 可不分 numbered section
- §2.2 Environment 内嵌完整 docker pull + pip install(pin 到 PR build 时)
- §4 可省略(无 benchmark 数据时)
- §5 Troubleshooting 上升为必填(dev build 风险大)
- 顶部 banner 改用 dev-pinned 标识

---

## 附录:Cookbook 整体目录结构

```
docs/cookbook/
├── index.md                        # 入口:定位 + 已支持模型清单 + 硬件覆盖矩阵
├── autoregressive/
│   ├── index.md
│   └── <model>.md                  # 单篇 recipe(本文档定义其结构)
├── multimodal/
│   ├── index.md
│   └── <model>.md
├── base/
│   ├── tpu-topology-reference.md   # TPU 代次 / 拓扑 / chips per host / HBM 速查
│   └── launch-flags-reference.md   # cookbook 常用 launcher flag + parser→recipe 映射
├── deployment/
│   ├── index.md
│   ├── single-host-docker.md
│   ├── gke-indexed-job.md
│   └── skypilot.md
└── troubleshooting.md              # 跨 recipe 通用问题
```

一级分类按 task type;第二层用扁平模型名;deployment / base / troubleshooting 为横切共用,所有 recipe 引用之。
