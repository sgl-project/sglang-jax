# Cookbook 设计辅助文档

> 配套规范:[`cookbook-recipe-design.md`](cookbook-recipe-design.md)。
>
> 站点改造跟踪:[`mintlify-migration.md`](mintlify-migration.md) —— Mintlify 渲染迁移计划,跨多个 PR 推进。
>
> 本文档面向 recipe 作者 / 方案维护者,提供写 recipe 时的设计原则、写作约束、review checklist、设计决策依据,以及方案演进背景。Recipe 作者写新 recipe 时主要看 `cookbook-recipe-design.md`,本文档作为辅助查阅。

---

## 1. 设计目标

为 `sgl-project/sglang-jax` 设计并维护 Cookbook 体系,面向"如何在 TPU 拓扑 Y 上部署模型 X"这一类问题。

**与 sgl-cookbook 主仓的差异化范围**:
- sglang-jax cookbook 的核心维度是 **model × TPU generation × topology**(v5e / v5p / v6e / v7x × 2x2 / 2x4 / 4x4 / 2x2x4 / 4x4x4)
- 部署形态是 **GKE / SkyPilot / 单节点 docker** 多 launcher 并存,不像主仓 GPU 主要单机多卡
- TPU 编译缓存(`JAX_COMPILATION_CACHE_DIR`)是 launch 必备(~4 min 冷启动),sgl-cookbook 没有这维度
- 没有 ConfigGenerator React 组件(理由见 §6.1),所有结构化信息用 markdown 表 + 主题驱动 bullet 表达

---

## 2. 设计原则

### 2.1 Lean(§1 极简)

§1 Model Introduction 只承载部署相关的信息——读者来 cookbook 是为了**怎么跑起来**,不是为了读模型 capability 宣传。

- 不复制 HF model card 整段简介
- 不复制 HF Architecture 表(除非有增量信息)
- 不列举与部署无关的能力宣传
- 单 variant 时 inline 不列表

### 2.2 横向 taxonomy

把跨多 recipe 的公共概念抽到顶层 `base/` 或 `deployment/`,recipe 内部引用而不重复定义。具体:

- **硬件矩阵**(`base/tpu-topology-reference.md`):TPU 代次 / chips per host / HBM / 拓扑速查
- **launcher 模板**(`deployment/{single-host-docker, gke-indexed-job, skypilot}.md`):每个 launcher 一份通用模板;recipe 只贴差异化字段
- **常见 launch flag**(`base/launch-flags-reference.md`):flag 解释 + parser key → recipe 映射
- **通用 troubleshooting**(`cookbook/troubleshooting.md`):跨 recipe 共性问题(OOM / compile timeout / 多节点连接失败)

### 2.3 Peer hierarchy

- `docs/get_started/install.md` 是 installation 的权威入口;cookbook 引用但不内嵌
- `docs/architecture/` / `docs/design/` 讲框架内部,与 cookbook 正交
- `docs/developer_guide/` 是开发视角(profiling / CI / 贡献流程);cookbook 是**用户视角**(怎么部署 / 怎么压测)

### 2.4 用户视角(不暴露内部代码细节)

cookbook 是给**普通用户**看的,**不要**在 recipe 中出现:
- sglang-jax 仓内部源码路径(如 `python/sgl_jax/srt/models/...`)
- 内部 class / enum 名(如 `MiMoV2FlashForCausalLM`、`ReasoningParser.DetectorMap`、`FunctionCallParser.ToolCallParserEnum`)
- PR / Issue 号链接
- "见 git blame"等开发者动作
- **工具辅助文档的引用**(`cookbook-recipe-design.md` / `2026-05-18-cookbook-research.md`)—— 这两份是给作者 / add-model skill 用的写作规范和设计辅助,不属于用户视角的 cookbook 内容
- **作者侧术语**(如 "Layout F variant" / "Pattern 3" 这种内部分类标签)—— recipe 标题和散文用直白描述(如 "single-workload configuration sweep"),不暴露分类编号

**例外**:`Tested build` 的 commit hash 作为 reproduce anchor,用户视角是合理的"版本标识"(可 `git checkout` 复现)。措辞用 `sglang-jax build <hash>` 这种用户友好的描述,而不是 "commit `<hash>`"。

详细写作约束见 §3。

### 2.5 以代码为准(作者侧 review 用,不暴露给读者)

Recipe 中提到的 launch flag、parser key、模型支持状况等,**作者写作时**必须 grep 源码核对(避免编造或滞后)。但**读者看 recipe 时**不应看到任何源码引用——验证责任在作者侧,通过 §4 Review Checklist 把关。

### 2.6 自包含(每个 launcher 形态自包含完整链路)

§2.3 Launch 内 `#### Single-host (Docker)` 和 `#### Multi-host (GKE Indexed Job)` 各自自包含完整链路——读者按部署规模直接跳到对应子节,不需要跨节拼参数。Multi-host 的 launch 命令和 GKE launcher manifest 差异字段在同一节,不分裂；SkyPilot 只作为高级 v6e 实验备选链接。

### 2.7 Staleness tolerant

Benchmark 数据是**测试时快照**,不随 release 实时刷新。

- 每个 §4.x Test Environment 显式 pin `Tested build`(commit hash 或 release tag,可附 `(YYYY-MM-DD)` 测试日期)
- 新数据通过新 PR 添加;旧数据保留作为该版本的历史记录,不强制定期回填刷新
- 当前没有真实数据时,标 `Pending` / `TODO` 占位
- Test Environment 元数据**必须真实**(不能编 quantization / TP / version;`(BF16)` benchmark metadata on INT4-only model 是 factual bug),但 Test Results 允许 Pending
- PR review 阶段允许带 Pending 进仓库,只要作者声明
- 历史数据未记录 version 时,允许标 `Pending (run pre-dates pin convention)`

### 2.8 参考 sgl-cookbook 主仓同名 recipe

写每一个 sglang-jax recipe 前,**先检查** `sgl-project/sgl-cookbook` 仓中是否有对应的同名模型 recipe(如 `docs/autoregressive/<Vendor>/<Model>.md`)。如果有,作为**措辞和内容的参考**——sgl-cookbook 经过更长时间打磨,§1 介绍、Configuration Tips 主题分类、Benchmark workload 描述等通常更面向用户。

**借鉴范围**(从 sgl-cookbook 学):
- §1 Model Introduction 的措辞、参数规模标注(`309B total / 15B activated` 这种)、模型定位句、Key Features 的取舍角度
- §2.4 Configuration Tips 的主题分类与命名
- §4 Benchmark workload 的描述方式(dataset 选择、ISL/OSL 取值理由)
- 用户视角的能力概览(agentic / reasoning / vision 等)

**不照搬**(因 sglang-jax 与主仓有本质差异):
- 硬件相关内容(主仓 GPU vs 我们 TPU)
- 命令格式(`sglang serve` vs `python -m sgl_jax.launch_server`)
- ConfigGenerator 相关(主仓有,我们没)
- parser key 集合(以 sglang-jax 实际注册为准)
- `--<flag>` 集合(以 sglang-jax `server_args.py` 实际定义为准)
- 端口偏好(我们统一 30000)

**冲突解决**:任何"sgl-cookbook 写法 vs sglang-jax 代码实际"的冲突,**以 sglang-jax 代码为准**(对齐 §2.5 以代码为准原则)。例如:
- sgl-cookbook 某 recipe 用 `--reasoning-parser deepseek-v3`,但 sglang-jax `ReasoningParser.DetectorMap` 实际注册的是 `deepseek-r1` → 我们写 `deepseek-r1`
- sgl-cookbook 某 recipe 的 launch 命令缩进 4 空格,我们 cookbook 全仓 2 空格 → 我们写 2 空格
- sgl-cookbook 某 recipe 端口 8000,我们统一 30000 → 我们写 30000

**审阅角度**:写完 sglang-jax recipe 后,把 sgl-cookbook 同名 recipe 的 §1 拉出来对照——如果对方写得更面向用户、更精炼、关键数字更突出,**应当借鉴重写**,而不是停留在我们较低的水准。

### 2.9 好模式累加机制

每次写新 sglang-jax recipe(无论是否有 sgl-cookbook 同名)都是固化"好模式"的机会:

**累加流程**:
1. **对比**——写完新 recipe 后,把同型号或同类(reasoning/MoE/VL/linear-attention)sgl-cookbook recipe 各拉一篇做横向对照
2. **提炼**——找出对方写得比我们好的**普适模式**(不依赖具体模型),如"Key Features 用什么角度取材"、"Configuration Tips 主题分类法"、"streaming 客户端的状态管理结构"
3. **固化**——把新发现的模式回写到 [`cookbook-recipe-design.md`](cookbook-recipe-design.md) 对应章节的"**好模式**"小节
4. **应用范围**——回写后,该模式适用于**所有**未来 recipe(包括没有 sgl-cookbook 同名样本的);避免相同的 pattern 在多个 recipe 中重新发明

**适用于没有同名 recipe 的情况**:
- sglang-jax 独有的模型(如 MiMo-V2.5-Pro 在 sgl-cookbook 没有同名)仍然要遵守已固化的好模式
- 例:写 sglang-jax 独有的 starter recipe 时,§1 Introduction 仍按"关键参数规模优先 / 设计意图 / 能力定位"的模式落地

**新模式何时回写**(判据):
- 该模式在 ≥ 2 个 sgl-cookbook 样本中可观察到(单一样本可能是特例,不应固化)
- 该模式不依赖具体模型类型(可跨 dense / MoE / reasoning / VL 通用),或明确标"仅适用 X 类模型"
- 该模式与 sglang-jax 代码、TPU 部署形态不冲突(冲突按 §2.5 以代码为准)

**反例**(不固化):
- sgl-cookbook 个别 recipe 的特殊处理(如某 recipe 用了 `:::admonition` Docusaurus 语法)——单点,不固化
- 与 sglang-jax 本质差异的模式(如 ConfigGenerator React 组件、`sglang serve` 命令)——属于 §2.8 "不照搬"范围,不固化

---

## 3. 写作约束 (Style Guide)

### 3.1 命令规范

| 项 | 规则 |
|---|---|
| entrypoint | **必须** `python -m sgl_jax.launch_server`(不是 `sglang serve`,那是 SGLang 主仓的命令) |
| bench entrypoint | **必须** `python -m sgl_jax.bench_serving` |
| 端口 | **统一 30000**;launch / curl / Python client / bench_serving 全部一致 |
| 缩进 | 2 空格(命令的 `\` 续行 flag) |
| Shell env | 用 `export VAR=<your-value>` placeholder,**不能** `export VAR=${VAR}`(bash no-op) |
| `--device tpu` | TPU 部署必带 |
| `--dtype bfloat16` | TPU 原生 dtype;**不要** float16 在 TPU |
| `--skip-server-warmup` | 必带(JIT cache 覆盖 warmup) |
| `JAX_COMPILATION_CACHE_DIR` | 必带(否则首次请求阻塞 ~4 min) |

### 3.2 Markdown 规范

| 项 | 规则 |
|---|---|
| 嵌套 code block | 外层包含 ``` 时用 **4 个 backtick** ```` ``` ```` |
| 链接风格 | 相对路径(便于 GitHub 渲染 + 重构)。例:`[base/launch-flags-reference.md](base/launch-flags-reference.md)` |
| 重要 note | 用 `> blockquote` 或 `**Note:**` 内联(避免 `:::admonition` —— sglang-jax 无 Docusaurus) |
| Output Example | 用 ```text fenced block,内容 **verbatim from server**(不 paraphrase) |
| Emoji | **不用于章节标题**;可用于 cookbook/autoregressive/index.md 的 status 列(✅ / 🚧 / 📝)和 output sample 中的装饰 |
| 表格 vs bullet | TPU 多维信息(generation × tp × dp × ep)用表;调参指南用 bullet |

### 3.3 用户视角约束(关键)

| 不要出现 | 替代方案 |
|---|---|
| 内部源码路径(`python/sgl_jax/srt/...`) | 直接给出可执行命令;parser key 列出可选值,告诉读者运行 `python -m sgl_jax.launch_server --help` 查完整列表 |
| 内部 class 名(`MiMoV2FlashForCausalLM` / `ReasoningParser.DetectorMap`) | 在 recipe 中不出现;架构 class 名是 add-model skill 内部用来填模板的,不应该 surfaces 给读者 |
| PR / Issue 号链接 | 结论留下(如 "At EP ≤ 8 epmoe wins"),开发讨论链接删除 |
| commit hash 措辞 | 用 `Tested build: <hash>` 或 `sglang-jax build <hash>`,不写"commit"二字 |
| "见 git blame / git log" | 改为 "按 cookbook 维护流程,新数据由新 PR 添加" |
| **工具辅助文档引用**(`cookbook-recipe-design.md` / `2026-05-18-cookbook-research.md`) | recipe 自包含 — staleness / layout 选择等概念用散文直接表述,不 link 设计文档;那两份是给作者写作时辅助用的,不属于 cookbook 用户内容 |
| **作者侧分类术语**(如 "Layout F variant" / "Pattern 3") | 用直白描述命名(如 "single-workload configuration sweep");不要把作者侧的 layout 编号 / pattern 编号暴露给读者 |

### 3.4 不要 hardcode 的项

| 项 | 原因 / 替代 |
|---|---|
| 采样参数(`temperature` / `top_p`)在 sample code 里 | 用 `generation_config.json` 默认。Recommended params 留在 §1 informational 列出 |
| 模型名缩写 | 全名优先(`XiaomiMiMo/MiMo-V2-Flash`,不只是 `MiMo-V2-Flash`) |
| TPU 拓扑数字 | 拓扑表里清晰列出 chips/node × nodes,不要在散文里散落 `v6e-16 = 16 chips` |

### 3.5 Reasoning / Tool-call 特殊规则

| 场景 | 规则 |
|---|---|
| Hybrid reasoning 模型(支持 thinking-on/off 切换) | §3.2 必须**双例** thinking-on(默认)+ thinking-off |
| Tool-call thinking 模型 | final response 必须**同时打印** `reasoning_content` + `content`(thinking 模式下 content 可能为 None,只打 content 会误导) |
| Reasoning parser 两种客户端模式 | 必须配对样本代码:<br>(a) **Separate field** parser(mimo / qwen3 / glm45 / kimi / deepseek-r1):thinking 在 `delta.reasoning_content`,answer 在 `delta.content`,双 print<br>(b) **Inline tags** parser:thinking 在 `<think>...</think>` 内嵌于 `content`,客户端要 parse tags |
| Raw `ChatCompletionMessage(...)` Python repr | 必须 format 成 readable structured output(Reasoning/Content/Tool Calls 各一段),不要原样 print object |
| Tool Calling §3.3 | 必备 `Handling Tool Call Results` 多轮对话段(从 tool result 回到 assistant final response) |

### 3.6 历史数据处理

| 场景 | 规则 |
|---|---|
| 历史 recipe 已有但未 pin commit/date | 标 `Pending (run pre-dates pin convention)` + 估算大致时间;不要求重新跑 |
| Validated 升级后旧数据是否回填 | 不强制;旧数据保留作为版本快照,新数据通过新 PR 添加 |
| Test Environment quantization 与 §1 variant 不一致 | factual bug,必须修;不能保留 |

---

## 4. Final Review Checklist (PR 合入前)

32 项强制 review,按类别整理:

### 命令格式
1. ✅ entrypoint 统一 `python -m sgl_jax.launch_server`
2. ✅ 端口 30000 全程一致(launch / curl / Python client / bench_serving)
3. ✅ deploy 命令与 bench 命令分清:deploy 用 `python -m sgl_jax.launch_server`,bench 用 `python -m sgl_jax.bench_serving`
4. ✅ 命令缩进 2 空格
5. ✅ Shell env 用 `<your-value>` placeholder

### 章节结构
6. ✅ §1 lean:单 variant inline 不列表 / Key Features bullets / 不暴露内部 class 名
7. ✅ §2.2 Environment 含 JAX TPU Image by Hardware 表(覆盖所有 §2.1 列出的 TPU)
8. ✅ §2.3 Launch 命令带必填 flag 集
9. ✅ §2.4 Configuration Tips ≥ 4 个主题(Validated)
10. ✅ §3.x 每个 code block 后立刻接 `**Output Example:**` + ```text fenced block(Starter 允许 `Pending update...`)
11. ✅ Hybrid reasoning 模型 §3.2 双例 thinking-on + off
12. ✅ Tool-call thinking 模型 §3.3 final response 双打印 reasoning_content + content
13. ✅ Tool Calling §3.3 必须有 Handling Tool Call Results 多轮段
14. ✅ §4.1 / §4.2 Test Environment 含 `Tested build`(可附测试日期)

### 代码与数据真实性(作者侧 grep,不暴露给读者)
15. ✅ 所有 `--<flag>` 在 `python -m sgl_jax.launch_server --help` 输出中存在
16. ✅ `--reasoning-parser <key>` / `--tool-call-parser <key>` 是真实可用的 key
17. ✅ §4 Test Environment 的 quantization 标注**必须**匹配 §1 列的 variant
18. ✅ Output Example 是 verbatim from server(不 paraphrase)
19. ✅ 不 hardcode 采样参数在 sample code

### Markdown 格式
20. ✅ 嵌套 code block 用 4-backtick(外层包含 ``` 时)
21. ✅ 所有 cross-link `.md` 文件存在
22. ✅ 相对路径 cross-link(不用绝对 URL)
23. ✅ 重要 note 用 `> blockquote` 或 `**Note:**` 内联

### 用户视角(从 §3.3 抽出)
24. ✅ recipe 中**不出现** sglang-jax 仓内部源码路径(`python/sgl_jax/srt/...`)
25. ✅ recipe 中**不出现** 内部 class / enum 名(`*ForCausalLM` / `*ParserEnum` / `*DetectorMap`)
26. ✅ recipe 中**不出现** PR / Issue 号链接
27. ✅ recipe 中**不引用工具辅助文档**(`cookbook-recipe-design.md` / `2026-05-18-cookbook-research.md`);staleness / layout 选择等概念用散文自包含表述
28. ✅ recipe 中**不暴露作者侧分类术语**(如 "Layout F variant" / "Pattern 3");章节标题和散文用直白描述

### sgl-cookbook 参考(对应 §2.8)
29. ✅ 已检查 sgl-cookbook 是否有同名 recipe;若有,§1 措辞 / Key Features 取舍角度 / Configuration Tips 主题分类已对照借鉴(冲突以 sglang-jax 代码为准)

### License 与版权
30. ✅ §1 License 段引导读者 verify HF model card,**不复述具体 license 文本**

### 状态一致性
31. ✅ Starter recipe 顶部 banner 存在;Validated 移除该 banner
32. ✅ `autoregressive/index.md` 中 status 列(✅/🚧/📝)与 recipe 实际状态匹配

---

## 5. 调研背景

### 5.1 sgl-cookbook 体系全景

SGLang 主仓的 cookbook 有两个层次:

| 层 | 仓库 / 站点 | 定位 | 形态 |
|---|---|---|---|
| 独立 Cookbook 站 | `sgl-project/sgl-cookbook` → `cookbook.sglang.io` | model × hardware × task 端到端 recipe | Docusaurus,含 React ConfigGenerator |
| 主 docs 内的 cookbook 章节 | `docs.sglang.io/cookbook/` | 同上,已合并到官方 docs 域名下 | Docusaurus |
| 主 docs 框架文档 | `docs.sglang.io/docs/` | SGLang 框架:安装 / server 参数 / 基础 API | Markdown |

sgl-cookbook 仍是结构最完整、写法最规范的参考样本。

### 5.2 sgl-cookbook 5 节模板

样本:`docs/autoregressive/DeepSeek/DeepSeek-V3_2.md`、`GLM/GLM-4.6.md`、`Qwen/Qwen3.md`。固定 5 节:

```
## 1. Model Introduction
## 2. SGLang Installation              ← 我们的 §2.2 Environment 对应
## 3. Model Deployment
   ### 3.1 Basic Configuration         ← ConfigGenerator
   ### 3.2 Configuration Tips
## 4. Model Invocation
   ### 4.1 Basic Usage
   ### 4.2 Advanced Usage              ← Reasoning / Tool Calling
## 5. Benchmark
```

我们调整为 5 节(§1 / §2 Deployment / §3 Invocation / §4 Benchmark / §5 Troubleshooting)的理由:
- 把 sgl-cookbook 的 §2 SGLang Installation 合并入 §2.2 Environment(只是一句话指向 install guide,不值得独立成节)
- 把 sgl-cookbook 的 §3 Model Deployment 拆成 §2.1-§2.4(因为我们没有 ConfigGenerator,需要分子节呈现 Hardware Matrix + Environment + Launch + Configuration Tips)
- 新增 §5 Troubleshooting(TPU 部署的特殊问题多,值得独立成节)

### 5.3 sgl-cookbook 12 recipe 实际分布

调研 12 个 recipe(DeepSeek-V3.2 / DeepSeek-R1 / GLM-4.5 / GLM-4.6 / Qwen3 / Qwen3-VL / Llama3.1 / Kimi-Linear / Kimi-K2.6 / MiMo-V2-Flash / Gemma4 / Mistral-Small-4),发现**实际分布远比 SKILL.md 规定的松散**:

| SKILL.md 硬规则 | 12 recipe 中违反数 |
|---|---|
| §1 必须有 Benchmarks **表**(不是 bullets) | 11/12 没做 |
| §1 必须有 License 段 | 8/12 没做 |
| §1 必须有 Recommended Generation Parameters | 11/12 没做 |
| §2 必须有 **Docker Images by Hardware Platform 表** | **12/12 都没做** |
| 必须 `sglang serve`(不能 `python -m sglang.launch_server`) | 6/12 违反 |
| 端口必须 30000 | 5/12 违反(Qwen3 用 8000,GLM 同页混用) |

**§4 Benchmark 6 种 layout 实际分布**(支撑我们 layout 菜单的依据):

| Layout | sgl-cookbook 实际使用 |
|---|---|
| A. 完整 3×3 scenario × concurrency 矩阵 | **唯 Qwen3** 一个 recipe |
| B. 方法论 + 命令模板(无真实数据) | GLM-4.5 / GLM-4.6 / DeepSeek-R1 |
| C. Latency vs Throughput 二分 | DeepSeek-V3.2 / Qwen3-VL |
| D. 多硬件分块 | Kimi-K2.6 |
| E. 变体 × 硬件矩阵 | Gemma4 |
| F. 极简 / 单 workload sweep | Kimi-Linear / Mistral-Small-4 |

**启示**:SKILL.md 是"作者希望的样子",实际 cookbook 是"写到哪算哪"。我们方案**应吸收 SKILL.md 的硬规则,但严格执行**,而不是盲目以现存 recipe 为榜样。

### 5.4 SKILL.md 硬规则速查

从 sgl-cookbook `.claude/skills/add-model/SKILL.md`(302 行)提取的硬性规则。这是设计我们 recipe 规范的直接参考依据。

**A. 命令格式硬规则**:
- ✅ 必须 `sglang serve`(主仓) / `python -m sgl_jax.launch_server`(sglang-jax)
- ❌ 禁止 `python -m sglang.launch_server`(主仓 deprecated)
- ✅ 端口必须 30000
- ✅ Launch 端口必须与 client/curl `base_url` 端口相同
- ✅ Benchmark 分两条命令:deploy + bench 各自独立
- ✅ Shell env 块用 `<your-value>` placeholder

**B. 章节结构硬规则**:
- ✅ §1 lean:Key Features + Benchmarks as **table** + Recommended Params + License + HF/blog links
- ✅ §1 single variant inline,不列表;不复制 HF Architecture 表
- ✅ §2 必须 Docker Images by Hardware Platform 表
- ✅ §3 embed ConfigGenerator + Configuration Tips
- ✅ §4 顶部一个 deploy command,下面各 test script + Output Example + ```text fenced
- ✅ §4 未部署时用 `Pending update...` 占位
- ✅ §5 accuracy 在前 speed 在后;不要默认加 multiple scenarios / concurrency levels
- ✅ §5 Test Environment 的 quantization 必须匹配 §1 列的 variant

**C. Markdown / 代码块硬规则**:
- ✅ 嵌套代码块用 4-backtick
- ✅ 每个 invocation code block 之后 `**Output Example:**` + ```text fenced
- ✅ Output text **verbatim from server**
- ❌ 不要 hardcode `temperature` / `top_p` 在 sample code

**D. Reasoning / Tool-call 硬规则**:
- ✅ Hybrid reasoning 模型必须 thinking-on + thinking-off 双例
- ✅ Raw `ChatCompletionMessage(...)` 必须 format 成 readable structured output
- ✅ Tool-call thinking 模型 final response 必须双打印 `reasoning_content` + `content`
- ✅ Reasoning parser 两种客户端模式必须配对:Separate field vs Inline tags

---

## 6. 关键设计决策与依据

### 6.1 不做 ConfigGenerator(第一阶段)

**决策**:不引入 React ConfigGenerator。

**依据**:
- sglang-jax 当前 docs 未集成任何静态站生成器(无 `conf.py` / `mkdocs.yml` / `docusaurus.config.js`),全是 GitHub 直接渲染的 Markdown;引入 React 组件需要先选并搭建静态站,成本高
- TPU 配置组合相对简单(拓扑 + dp/tp/ep + dtype 三轴),用 markdown 表格 + Configuration Tips 已能覆盖
- 模型数量还小(~15),手动维护成本可控

**第二阶段触发条件**(任一满足):
- 模型数 > 25
- 文档站迁到 Docusaurus / 其他静态生成器
- 同一模型在 ≥ 3 种 TPU 代次上都有 recipe(组合爆炸)

### 6.2 Staleness Policy

**决策**:Benchmark 数据是测试时快照,不实时刷新;允许 `Pending` 占位。

**依据**:
- sgl-cookbook 12/12 recipe 都 pin `sglang version`(各种 0.5.6 / 0.5.7 / 0.5.9 / `gemma4 branch`),从来不刷新旧数据
- sgl-cookbook SKILL.md 明文允许 `Pending update...` placeholder
- 强制定期刷新成本极高(需要 TPU slice 时间),性价比不值
- 用户判断数据时效靠 Test Environment 的 version + date pin,而不是依赖 cookbook 维护方实时性

### 6.3 §2 Deployment 4 子节合并依据

**决策**:从原 6 节(§1 Intro / §2 Hardware / §3 Environment / §4 Launch / §5 Invocation / §6 Benchmark / §7 Troubleshooting)合并到 5 节(§1 / §2 Deployment / §3 Invocation / §4 Benchmark / §5 Troubleshooting)。

**依据**:
- 原 §3 Environment 内容薄(80% 已 link 到 deployment 模板),独立成节制造形式化空节
- 原 §6 Key Flags 表与 §7 Configuration Tips 职责重叠(都是"为什么这个 flag 取这个值"),sgl-cookbook 完全不做 Key Flags 独立节,把 flag 取值理由放进 Configuration Tips 的主题 bullet 里
- §2.3 Launch 内嵌 `#### Single-host` / `#### Multi-host` 四级标题,比原 §4 Launch + §5 GKE/SkyPilot 分两节更紧凑——multi-host 的命令 + manifest 在同一节,不切两半
- §2 Hardware Matrix 独立保留:TPU 拓扑信息量大且 sglang-jax 没 ConfigGenerator,独立成节方便快速扫读

### 6.4 不强制 3×3 Benchmark 矩阵

**决策**:§4.2 Speed 提供 6 种 layout 菜单,不强制 3×3 scenario × concurrency。

**依据**:
- sgl-cookbook 12/12 recipe 中**只有 Qwen3 一个**用了完整 3×3 矩阵
- 11/12 用其他 layout(B 方法论模板 3 个 / C Latency-Throughput 二分 2 个 / D 多硬件分块 1 个 / E 变体矩阵 1 个 / F 极简 3 个 / 完全没 §5 一个)
- 强制 3×3 会让 Validated 升级门槛过高(9 次 bench_serving run 需要 TPU slice 时间);sglang-jax 自家 cookbook 应该比 sgl-cookbook 主仓更宽松,不更严苛

### 6.5 自动化贡献 skill 设计

设计一个 `add-model` Claude skill,分 6 阶段执行:

| Phase | 工作 |
|---|---|
| 1. 收集输入 | Model Card / Variants / TPU Platforms / sglang-jax 版本 / 部署形态 |
| 2. 脚手架生成 | recipe 骨架(按 [`cookbook-recipe-design.md`](cookbook-recipe-design.md) 5 节结构),starter banner 默认开;同时在 `autoregressive/index.md` 加一行 status=🚧 |
| 3. 自动校验 | 跑 `python -m sgl_jax.launch_server --help` 抓最新 flag;校验所有 `--<flag>` 真实存在;校验 parser key 真实存在;校验 cross-link `.md` 文件存在;校验端口在所有命令中一致 |
| 4. 用户实测填回 | 引导用户跑 curl / Reasoning / Tool Calling / evalscope / bench_serving |
| 5. Configuration Tips 完善 | 让用户回答 mem-fraction 取值理由 / SWA pool 观测点 / Speculative 启用情况等 |
| 6. Final Review | 跑 §4 Final Review Checklist 29 项 |

**实施策略**:
- 第一版本地手动跑,不挂 GitHub Action
- 第二版(模型数 > 25 时)考虑挂 pre-commit hook 自动跑 Phase 3 校验项

---

## 7. 现状与缺口(cookbook 引入前)

`docs/` 已存在的"模型类"内容散落多处:

| 现状路径 | 内容 | 风格 |
|---|---|---|
| `docs/basic_usage/qwen.md` | Qwen 7B-Chat TPU 部署 | Quick Start / Configuration Tips / Benchmarking |
| `docs/basic_usage/mimo_v2_flash.md` | MiMo V2 Flash | 同上风格 |
| `docs/basic_usage/mimo_v2.5_pro.md` | MiMo V2.5 Pro(v7x-16 + v6e-64 + GKE manifest) | Hardware / Environment / Launching / GKE / Accuracy |
| `docs/basic_usage/grok2-skypilot-serving.md` | Grok-2 + SkyPilot v6e-32 | 偏 SkyPilot 部署流程 |
| `docs/performance/qwen3_benchmark.md` | Qwen3-8B/32B vs vLLM benchmark | 仅 benchmark,部署命令重复 |
| `docs/mutlimodal/multimodal_usage.md` | 多模态用法 | 通用说明(拼写 `mutlimodal` 是 typo)|

**主要缺口**:
- 没有 "Cookbook" 这个一级入口,新人不知道去哪找模型部署
- `basic_usage/` 名字误导——既放 framework 基础用法,又放每模型 recipe(peer hierarchy 被污染)
- 缺统一 recipe 模板(章节名各不相同,有的没 benchmark,有的没 troubleshooting)
- benchmark 与部署 recipe 分裂在两个目录,启动命令出现两次容易漂移
- 缺横向"硬件矩阵 / TPU 选型 / 启动命令通用片段"等公共内容的集中页
- 没有"已支持模型 vs 推荐配方"的全景索引页

---

## 8. 迁移路径与状态

**P0(已完成)**:
1. ✅ 新建 `docs/cookbook/index.md`、`cookbook/autoregressive/index.md`、`cookbook/multimodal/index.md`
2. ✅ 把 `docs/basic_usage/{qwen, mimo_v2_flash, mimo_v2.5_pro}.md` 迁入 `cookbook/autoregressive/`,原路径保留 stub 重定向
3. ✅ `docs/basic_usage/grok2-skypilot-serving.md` 拆为 `cookbook/autoregressive/grok2.md` + `cookbook/deployment/skypilot.md`
4. ✅ 抽出 `cookbook/base/tpu-topology-reference.md` 与 `cookbook/base/launch-flags-reference.md`
5. ✅ `docs/performance/qwen3_benchmark.md` 部署命令合并回 `cookbook/autoregressive/qwen3.md` §4.2,原文件改为 Benchmark Report

**P1(已完成)**:
6. ✅ 补齐 11 个 starter recipe:llama / gemma2 / mimo-7b / qwen3-moe / deepseek-v3 / glm4-moe / glm5-moe / bailing-moe / bailing-moe-linear / ling-2.6 / kimi-linear
7. ✅ 落地 `cookbook/deployment/{single-host-docker, gke-indexed-job, skypilot}.md` 通用模板 + `deployment/index.md`
8. ✅ 写 `cookbook/troubleshooting.md` 汇总各 recipe 散落踩坑提示

**P1.5(进行中 — 按新方案升级 recipe 内容深度 + 模板结构调整)**:
9. ⏳ 模板从原 6 节合并为新 5 节(§2 Deployment 4 子节);依据 [`cookbook-recipe-design.md`](cookbook-recipe-design.md)
10. ⏳ mimo-v2-flash.md 作为新方案首个试改样本,验证模板可行性
11. ⏳ 把新模板推广到其他 4 个 P0 validated recipe(qwen / qwen3 / mimo-v2.5-pro / grok2)
12. ⏳ §3.2 Reasoning / §3.3 Tool Calling 完整 Python streaming 例子推广到所有 Validated recipe;Hybrid 模型加 thinking-on + off 双例;Tool-call 必备 Handling Tool Call Results 段
13. ⏳ §2.2 Environment 加 JAX TPU Image by Hardware 表(所有 recipe)
14. ⏳ 清理所有 recipe 中的内部源码路径 / class 名 / PR 号(按 §3.3 用户视角约束)

**P2(可选)**:
15. 评估文档站迁 Docusaurus / 其他静态生成器(前提:要做交互式 ConfigGenerator 才有性价比)
16. 实现 `/add-model` skill(按 §6.5 设计)
17. 补 multimodal recipe(Qwen2.5-VL / Wan 2.1/2.2 T2V / Qwen3-Omni MoE / MiMo Audio)

---

## 9. 与现有 docs 的关系

- `docs/architecture/`、`docs/design/`、`docs/features/` 保持不变——这些讲框架内部,与 cookbook 正交
- `docs/basic_usage/features/` 等"框架基础用法"的内容留在原位;只把"特定模型部署"类内容迁走
- `docs/get_started/install.md` 与 `docs/get_started/sglang-jax-gpu-install.md` 是 cookbook 的依赖项,cookbook 里全部用引用而不重复内容(**peer hierarchy 原则**)
- `docs/developer_guide/` 中的 `tpu_resources_guide.md`、`benchmark_and_profiling.md` 与 cookbook 的 `base/` 有重叠,分工:开发视角(profiling、CI、贡献流程)留在 developer_guide;用户视角(怎么选 TPU、怎么跑标准 benchmark)抽到 cookbook/base

---

## 10. 参考链接

| 用途 | 链接 |
|---|---|
| sgl-cookbook 仓(主仓 cookbook,作为方案参考) | https://github.com/sgl-project/sgl-cookbook |
| sgl-cookbook 站点 | https://cookbook.sglang.io / https://docs.sglang.io/cookbook/intro |
| 标杆 recipe(主仓最完整样本) | `sgl-cookbook/docs/autoregressive/DeepSeek/DeepSeek-V3_2.md` / `Qwen/Qwen3.md` |
| 主仓 add-model skill(我们 §6.5 设计的参考) | `sgl-cookbook/.claude/skills/add-model/SKILL.md` |
| 主仓 ConfigGenerator 实现(若第二阶段引入,参考) | `sgl-cookbook/src/components/base/ConfigGenerator/` |
| sglang-jax 主仓 | https://github.com/sgl-project/sglang-jax |
