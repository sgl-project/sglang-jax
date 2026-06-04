# MiMo-V2.5 接入 · 第一步：三项必做前置改造

> **范围**：接入 MiMo-V2.5 前必须先完成的三项前置改造，给出问题背景与具体整改方案。
> **依据**：以源码为准，分析对象为 sglang-jax 分支 `fix/dp-multimodal-input-embedding`（`/Users/lianfang/primatrix/sglang-jax-fix-dp-mm`）。证据以 `文件:行号` 标注（行号以审读时为准，实施前复核）。
> **结构**：§0 全流程概览 → 问题 1（入口与多模态标识）→ 问题 2（编码 stage 编译加速）→ 问题 3（编码 stage 模型无关契约）→ 限制/后续（DP>1、batch>1）→ 复盘备查。

---

## 0. 请求处理全流程概览

一条包含图像/音频的对话请求，自进入至返回需经过五个阶段；后续三项改造分别针对其中的特定阶段。

```
① HTTP 入口          用户 POST /v1/chat/completions（消息含 text + image/video/audio）
        │
② 预处理(Tokenizer)   文字 → token id；图/音 → HF 处理器抽取特征；打包为 mm_inputs
        │             并在 token 序列中为每段媒体保留"占位符 token"（如 image_token）
        │
③ 调度(GlobalScheduler) 将请求转为内部 Req 对象，按 stage 配置顺序逐级下发
        │
④ 各 stage（每站由 scheduler 调度 + worker 包装 + runner 执行 + model 模型 四层构成）
        │   ④a 编码 stage：将媒体特征编码为 embedding，并注入 ② 预留的占位符位置
        │   ④b 生成 stage：大语言模型基于合并后的序列逐 token 生成
        │
⑤ 回包(Detokenizer)   将生成的 token 解码为文字，返回用户
```

**三项前置改造对应的阶段**：
- **问题 1** → 阶段 **①**（请求入口归属）+ 多模态标识开关。
- **问题 2** → 阶段 **④a** 编码 stage 的**执行效率**（当前未启用编译、且输出契约脆弱）。
- **问题 3** → 阶段 **④a** 编码 stage 的**模型适配能力**（装载/输入/注册三处均硬编码为 Qwen 形状，换 MiMo 即失败）。

> **术语**
> - **token / token id**：模型将文字切分为的最小单位，每个对应一个整数 id。
> - **embedding**：将 token id 或媒体特征映射为浮点向量；模型实际处理的是向量。
> - **占位符 token**：图像/音频本身非文字，需在 token 序列中占位——以特殊 token（如 `image_token`）重复 N 次占据 N 个位置，后续将媒体特征注入这 N 个位置。
> - **JIT（即时编译）**：将逐算子解释执行的代码编译为优化后的计算图，在 TPU/GPU 上显著提速；张量的**分片注解**亦仅在编译后生效。
> - **mesh / sharding（网格 / 分片）**：将大张量切分到多芯片并行；"网格"描述芯片排布，"分片注解"指定张量沿哪根轴切分。

---

## 问题 1 · 请求入口归属错误，且系统缺少可靠的多模态标识（阶段 ① + 总开关）

包含两个相互独立的基线缺陷。

### 1.1 入口路由重复注册，多模态处理函数为不可达死代码

**现状**
服务启动时将"URL → 处理函数"登记入路由表，而 `/v1/chat/completions` 被登记了**两次**：

1. 核心模块先登记 `/v1/chat/completions`（`entrypoints/http_server.py:697`）。
2. 多模态模块为加入自有逻辑，又在**同一 app 对象**上登记了**同名**路由（`multimodal/entrypoint/http_server.py:300`，本意走多模态专用构造 `_extract_openai_prompt → GenerateOmniReqInput`）。
3. Starlette 的匹配规则为"**先注册者优先、不去重**"。核心路由先注册 → **核心处理函数生效**，多模态处理函数**永不被调用**（已实测确认为死代码）。`/abort_request` 存在同样的重复（核心 `:604` vs 多模态 `:350`）。

**为何当前仍可运行（易误解处）**
模型执行链路（②③④⑤）由进程启动时的 `launch()` + `set_global_state(...)` 独立拉起，**与 HTTP 路由归属无关**；且 `set_global_state` 将 `tokenizer_manager` 置为多模态版 `MultimodalTokenizer`，故即便请求由核心处理函数接收，仍会转交多模态 tokenizer 处理。**结论：服务可运行，但实际走的是核心（且不完整的）请求构造逻辑，多模态专用构造被跳过；且"谁生效"取决于 import 顺序，稳定性差。**

**整改方案**
- 仅保留**一条** `/v1/chat/completions`，在其内部按"是否多模态模型"分流至多模态 / 纯文本两套构造；删除多模态侧的重复登记。整改后请求必走正确构造路径，且与 import 顺序无关。
- 过渡方案（若暂无法合并）：多模态入口在登记自有路由前，**先移除核心的同名路由**，确保自身唯一生效。

### 1.2 多模态标识 `is_multimodal` 恒为 False

**现状**
核心配置中本应表征"是否多模态"的字段 `is_multimodal` 被**无条件置为 False**（`configs/model_config.py:162`）——无论加载何种模型均返回 False。实际的多模态判定依赖两处带外信号：命令行开关 `server_args.multimodal` 与"启动时 import 了哪个模块"；而下游消费方（`serving_chat.py:66-70`）读取的恰是该恒 False 字段。另有 `multimodal_model_archs` 列表（`model_config.py:791`）已定义但无消费方。

**问题分析**
核心调度/预处理需要一个可靠字段来判定"是否执行多模态逻辑（如 embedding 注入）"。该字段恒 False，使判定只能依赖命令行参数与 import 副作用隐式进行，难以推理、易出错。

**整改方案**
- 由模型自身的 `hf_config.model_type / architectures` 推导 `is_multimodal`（MiMo-V2.5 → True，纯文本模型 → False）；核心 serving/scheduler 统一读取该字段。

### 1.3 意义 / 验收 / 风险
- **意义**：image/video/audio 对话的主入口必须命中正确处理函数；核心需准确识别多模态以执行后续 embedding 注入。二者为后续工作的基础。
- **验收**：多模态模式下 `/v1/chat/completions` 命中多模态分支；`is_multimodal` 对 MiMo-V2.5 返回 True、对纯文本模型返回 False。
- **风险**：低，且两缺陷相互独立。注意点：合并为单一处理函数时需确认 `OpenAIServingChat` 可承接多模态请求构造，否则采用上述过渡方案。

---

## 问题 2 · 编码 stage 未启用编译加速、且输出契约脆弱（热路径性能最高优先项）

该项位于**算力消耗最高的热路径**上，整改方案明确、收益直接，且所有多模态模型均受益。（其中"输出写死 3-tuple"亦属问题 3"编码 stage 缺模型无关契约"的一个侧面。）

### 2.0 背景与关键概念

**编码 stage（④a）职责**：将图像/音频经各自 encoder（视觉 ViT、音频 encoder）编码为 embedding，注入 ② 预留的占位符位置，形成"文字 + 媒体"完整序列后交由 ④b 的大语言模型。

**JIT（即时编译）**：未启用 JIT 时代码逐算子解释执行，性能较低；启用后整段计算被编译为优化计算图，TPU 上显著提速，且张量的**分片注解仅在编译后生效**（即大矩阵切分到多芯片）。

**形状分桶（shape bucketing）**：JIT 缓存**以输入张量形状为键**——形状变化即触发一次重编译。多模态输入形状高度可变（图像尺寸→图块数、音频时长→帧数、序列长度均不同），若不处理将导致**频繁重编译**。分桶即将可变尺寸**以补零方式 pad 至少量固定档位**（如 512 / 1024 / 2048…），使 JIT 仅需编译"档位数"次，其余命中缓存。

### 2.1 现状

编码 stage 的执行器 `EmbedModelRunner` 存在三处缺陷：

1. **未启用 JIT**：`embed_model_runner.py:48-102` 中启用 JIT 的代码被**整段注释**（且无说明），`self.jitted_embedding` 实为普通 Python 函数，直接 eager 调用 `self.model(...)`。后果：整段 ViT / 音频 encoder **逐算子解释执行、分片注解失效**——而其余 6 个 stage 执行器均已启用 JIT。
2. **输出契约写死**：`forward`（`:181`）以 `a, b, c = self.jitted_embedding(...)` **写死"模型须返回 3 个值"**，返回结构变化即失败。
3. **设备固定为 CPU**：该执行器目前仅服务 Qwen3-Omni，其流水线配置将编码 stage 置为 `device_kind: cpu`（`qwen3_omni_stage_config.yaml:7`）。

**关闭编译的原因分析（避免"仅取消注释"的误判）**
并非技术上必须关闭——JIT 在 CPU 上同样可编译，同仓 `qwen2_5_vl_stage_config_tp4.yaml` 的视觉 stage 亦位于 CPU 却**启用了 JIT**。更可能的原因为：编码 stage 的输入**形状高度可变**，未分桶时启用 JIT 会**频繁重编译**；叠加"位于 CPU、分片无意义、编译耗时"，遂改为 eager。**根因是缺少形状分桶，CPU 部署只是降低了启用 JIT 的动机。**

### 2.2 整改方案（按改动位置分组）

整改不止"取消注释启用 JIT"，须一并解决"当初关闭的原因"。共五处，标明各自所在文件/函数：

**(A) 启用 JIT + 结构化输出 —— `EmbedModelRunner.initialize_jit`**
- 沿用同仓已验证的视觉执行器 `vit_model_runner.py:44-102` 的骨架：在外层以 `nnx.split` / `tree_flatten` 拆分"定义 + 权重"，在被 JIT 的函数内以 `tree_unflatten` / `nnx.merge` 合回，外层套 `jax.jit(..., static_argnames=["model_state_def"])`。
- **结构化输出**：编码模型 `__call__` 返回**带字段名的结构**（如 `EmbedOutput(input_embeds, deepstack_embeds, deepstack_pos_mask)`），执行器**按字段名取用**，不再写死 3-tuple；不同模型可仅填充自身字段（MiMo-V2.5 无 deepstack，仅填 `input_embeds`）。

**(B) 形状分桶 + 补零 —— `EmbedModelRunner._prepare_input`（分桶落点）**
分桶**全部在 host 侧、JIT 调用前**完成：对每条可变轴定义一组固定档位，将实际尺寸 pad 至"不小于其的最近档位"，同时记录真实长度：

| 输入 | 形状 | 可变轴 | 档位（示例，按真实分布调整） |
|---|---|---|---|
| `input_ids` | `[seq]` | 序列长度 seq | 512 / 1024 / 2048 / 4096 |
| `pixel_values`（图） | `[图块数, patch维]` | 图块数 | 256 / 512 / 1024 / 2048 / 4096 |
| `input_features`（音频） | `[特征维, 帧数]` | 帧数 | 按音频分段大小的倍数 |
| `image_grid_thw` | `[图数, 3]` | 图数 | 1 / 2 / 4 / 8（或设为静态） |

`_prepare_input` 改为：计算实际尺寸 → 选定档位 → 补零 pad 至档位尺寸 → 产出 `valid_len / valid_mask`（标识有效数据与补零）→ 连同元信息（真实序列长度等）一并返回。
> 原理：JIT 以形状为缓存键。补零后所有请求形状仅落在少量档位内，JIT 仅编译档位数次，其余命中缓存，频繁重编译被消除。
> 网格/位置等辅助量（如视觉 `grid_thw`）仍按视觉执行器做法转为**静态值**，在 host 侧算好辅助数组后再喂入 JIT。

**(C) 屏蔽补零 —— 编码模型内部（正确性，不可省）**
补零**不得污染计算结果**：
- **encoder**：视觉/音频 encoder 须忽略补零部分（通过有效段边界 / 注意力 mask / `valid_len`），否则补零将作为有效数据参与注意力。
- **合并（scatter）**：仅将**有效**特征行注入占位符（按真实计数，而非 pad 后计数），并加入"占位符数量 == 有效特征行数"断言，越界即报错。

**(D) 输出去 pad —— `EmbedModelRunner.forward`**
JIT 输出的 `input_embeds` 长度为档位长度，须**切回真实序列长度** `input_embeds[:真实seq长度]` 后再写入 `mm_inputs["multimodal_embedding"]`；下游 ④b 按真实序列长度取用该 embedding，长度不一致将导致错位。

**(E) 编码 stage 部署至 TPU —— 流水线配置 yaml**
MiMo-V2.5 的编码 stage 设 `device_kind: tpu`（**不沿用** Qwen3-Omni 的 `cpu`）。该 stage 承载 729M 视觉塔 + 音频编码，CPU eager 执行性能不可接受；部署至 TPU 后 (A) 的 JIT 与分片方有意义。

### 2.3 整改后流程

```
EmbedModelRunner.forward(请求):
  ┌ _prepare_input(请求):                         # host 侧，JIT 之前
  │    计算实际尺寸 → 选定档位 → 补零 pad(input_ids / pixel_values / 音频)
  │    产出 valid_len / valid_mask + 元信息(真实seq长度…)
  │    grid_thw → 静态值 → host 侧算辅助数组(段边界等)
  ├ out = jitted_embedding(已 pad 的输入, 辅助数组, valid_len)
  │        # 形状仅落在档位集合内 → 每个档位组合编译一次，其余命中缓存
  │        └ 模型内：encoder（按 valid_len 屏蔽补零）
  │                  → 合并（按真实计数将特征 scatter 至占位符）→ 返回 EmbedOutput(带字段名)
  ├ input_embeds = out.input_embeds[:真实seq长度]            # 去 pad
  └ 写 mm_inputs["multimodal_embedding"] = input_embeds（deepstack 等按字段名写）
```

### 2.4 验收 / 风险与注意事项
- **验收**：编码 stage 于 TPU 成功编译、分片注解生效；档位集合内**不再频繁重编译**（编译次数 ≈ 档位组合数）；补零位置不影响输出（数值与 eager 版一致）；输出长度等于真实序列长度。
- **风险与注意事项**：
  1. **档位需覆盖真实输入分布**：过粗导致补零浪费算力，过细导致编译变体过多；建议先粗后调。
  2. **补零必须被屏蔽**：否则补零进入 encoder/合并将污染结果——属正确性问题而非性能问题。
  3. **图像布局亦影响形状**：图块数相同但图数/长宽不同时，辅助数组（段边界）长度变化，可能产生额外编译变体；首轮可对"图数"一并分桶或容忍少量变体。
  4. **去 pad 长度须对齐生成 stage**：编码输出长度须等于真实序列长度，否则下游取用错位。

---

## 问题 3 · 编码 stage 对模型形状硬编码，非 Qwen 模型无法装载（结构性关键项）

该项是 MiMo-V2.5 能否被编码 stage 装载运行的**硬前提**——不整改则模型在启动阶段即抛异常。它与问题 2 中"输出 3-tuple 写死"为同源问题。

### 3.0 概述

编码 stage 执行器（`EmbedModelRunner`）自始仅服务 Qwen-VL / Qwen3-Omni 类模型，从未定义"新 omni 模型接入需实现哪些约定"。因此其在**三个层面**均硬编码了 Qwen 形状：如何从 HF 配置取子配置、forward 接收哪些输入、模型返回哪些输出。MiMo-V2.5 是**首个走此路径的非 Qwen omni 模型**，故三处均受阻。

### 3.1 现状（三处硬编码，均仅适配 Qwen）

1. **config 装载写死 `.thinker_config`（启动即失败处）**：执行器装载模型时直接取 `model_config.thinker_config`（`embed_model_runner.py:39`）。`thinker_config` 为 Qwen3-Omni 特有子配置；**MiMo-V2.5 配置为扁平结构、无 `thinker_config` 属性** → 抛 `AttributeError`、**启动即失败**。
2. **forward 输入签名写死 Qwen 形状**：被调用的 `forward_model(...)` 入参写死为 `input_features / pixel_values_videos / image_grid_thw …`（`embed_model_runner.py:60-67`），系 Qwen 视觉+音频的输入种类。**MiMo-V2.5 音频走 `audio_codes`**（RVQ 码），无法匹配该签名。
3. **配置注册表仅含 Qwen**：`config_registry` 按 `model_path` 字符串匹配，**默认维度为 Qwen2.5-VL**（depth32 / hidden3584 / patch14），缺少 `mimovl`（depth28 / hidden1280 / patch16）与 V2.5 音频子配置入口 → 多模态侧配置对象（`mm_config`）无法获取 MiMo 的占位符 token id 与各塔维度。

> 上述三处 + 问题 2.2(A) 的"输出 3-tuple 写死"，恰构成 embed-stage 缺契约的四个层面：**config 提取、输入、输出、注册**。

### 3.2 问题分析与关键澄清

- **本质是缺接口，而非"MiMo 特殊"**：任何非 Qwen 的 omni 模型至此均会同样失败。整改收益是**一次性为后续所有 omni 模型铺路**，而非仅为 MiMo 打补丁。
- **关键澄清：占位符 token id 本身不缺，不应在核心配置中"补解析"**。易误判为"框架未解析 video/audio token id，需在核心配置补上"——**方向错误**。依据：
  - 编码 stage 模型读取的是**原始 HF 配置对象**（`embed_model_runner.py:35-39`），而非核心引擎的 `ModelConfig`；编码模型直接使用 `self.config.{image,video,audio}_token_id`。
  - MiMo-V2.5 的 HF 配置类**已将三个 token id 暴露为属性**：image/video 位于顶层，audio 在构造时由 `processor_config` 复制至 `self`，故 `config.audio_token_id` 可取得。
  - 核心 `ModelConfig` 中"仅解析 image_token_id"的字段，**唯一消费方是非多模态单栈入口（`tokenizer_manager`）**，omni 链路不经过它。
  - **结论**：需补的并非"解析 token id"，而是"**令编码 stage 识别 MiMo 的配置结构 / 输入种类 / 在注册表中具备维度**"。

### 3.3 整改方案：将编码 stage 收敛为薄契约

不应以 `if 是MiMo: … else: …` 打补丁——那只会使 MiMo 成为**第二个**硬编码模型，后续 omni 模型继续累积债务。正确做法是抽出一个**模型无关的薄契约**，由模型自行声明，执行器不再识别具体模型：

- **(A) config 提取按模型声明**：移除 `embed_model_runner.py:39` 写死的 `.thinker_config`；由模型声明"如何从 HF 配置获取自身子配置"——MiMo 直接使用配置本体（token id 与 vision/audio 子配置均挂其上），Qwen-Omni 仍取 `.thinker_config`。
- **(B) 输入按模型声明的输入种类组织**：`forward_model` 不再写死字段，改为按模型声明的输入集合传入（支持 `audio_codes` / `pixel_values` / `image_grid_thw` / … 的并集），模型仅取自身所需。
- **(C) 注册表新增 MiMo 入口**：在 `config_registry` 注册 `mimovl` 视觉子配置（depth28/hidden1280/patch16…）与 V2.5 音频子配置；其中 audio_tokenizer 为**独立子目录、独立 model_type**，走单独注册入口（不能从主配置的 audio_config 获取）。
- **(D) 输出结构化（与问题 2 合流）**：模型返回**带字段名的结构**而非 3-tuple，执行器按字段名取用。
- 综上，编码 stage 契约 = **(config 提取, 输入 spec, 输出字段, 注册维度)** 四项由模型声明，runner 保持通用。
- **配置类装载**：MiMoV2 配置类经 HF 远程代码加载（auto_map + `trust_remote_code`，默认已开），**不应**纳入核心本地 typed-config 注册表（后者要求本地可导入该类）。

### 3.4 意义 / 验收 / 风险
- **意义**：此为 MiMo 能被编码 stage 装载运行的硬前提（否则启动 `AttributeError`）；同时将 embed-stage 由"仅适配 Qwen"升级为"面向接口"，后续 omni 模型可直接复用。
- **验收**：MiMo 配置经 `EmbedModelRunner` 正常装载（无 `AttributeError`）；`audio_codes` 可传入 forward；`config_registry` 返回 `mimovl` / V2.5 音频的正确维度；编码模型读 `self.config.*_token_id` 三者均可取得。
- **风险**：中。抽取薄契约较加 `if/else` 工作量略大，但可避免"第二个硬编码模型"的债务累积；改动集中于 `embed_model_runner` 与 `config_registry`，不涉及调度/生成 stage。

---

## 限制 / 后续任务项（三项前置之外，各自独立立项）

> 在 **DP=1（不启用数据并行）且单 batch（每批不与纯文本请求混跑）** 下，完成上述三项前置即可使 MiMo-V2.5 完整运行。以下两条为**放开并发规模时触发的正确性硬约束**——不在三项前置范围内，但须明确，否则放开后将静默出错。

### A. DP > 1（数据并行）

> 若首轮需启用 **DP>1（将请求分配至多份模型副本以提升吞吐）**，存在一项必须单独整改的正确性硬约束。

**问题**
数据并行需要一张将芯片划分为 `[data, tensor]` 两轴的网格：`data` 轴为模型副本数，`tensor` 轴为每副本内部切分数。但当前各 stage 自建网格时写死 `ici_parallelism=[-1, 芯片数]`（`stage.py:98-109`），解析后 **`data` 轴恒为 1**（即仅一份副本）；而核心调度器又按命令行 `dp_size`（可能 >1）交织排布各副本数据（`schedule_batch.py` 的 `_merge_multimodal`，按 `range(dp_size)` 排布）。

**后果**：当 `dp_size>1` 时，host 侧按"多副本"排布数据，而网格仅"一份副本"——二者不一致，张量分片（`input_ids` 标注按 `data` 轴切分，但 `data` 轴大小为 1）与多副本布局冲突，导致形状/分片错误。**DP=1 时该约束休眠**（`data=1` 恰等于 `dp_size=1`，自洽）。

**最小整改（不涉及大重构）**
- 编码/生成 stage 自建网格时**令 `data` 轴取真实 `dp_size`**：`ici_parallelism=[dp_size, 芯片数 // dp_size]`，来源取命令行 `dp_size`；并在调度器接收外部网格处加入断言 `mesh.shape["data"] == dp_size`，固化该隐式约束。
- 备选：生成 stage **不自建网格，交由核心调度器按 `[dp_size, tensor]` 构建**。
- 附带：若同时启用 MoE 专家并行（MiMo-V2.5 为 256 专家），`ep_size` 须同时整除 256 与芯片总数（data×tensor）。

**定位**：此为本分支 `fix/dp-multimodal-input-embedding` 的命题，属**后续任务**。首轮若 **DP=1** 可不动；若需 **DP>1**，则在三项前置之外额外执行此项最小整改。

### B. batch > 1（同批混跑多请求 / 含纯文本请求）

**问题**
首轮按**单 batch（一次处理一条请求，或同批不混入纯文本）** 运行。编码 stage 合并媒体 embedding 时采用**整段替换**：将该请求完整的"文字 + 媒体" embedding 整段传给生成 stage。单请求时无误——该整段即其自身数据。

**放开 batch>1 时的错误**
当同批**同时包含"带媒体请求"与"纯文本请求"**时，纯文本请求**不含**媒体 embedding 数据；而整段替换的逻辑为"只要批内有请求带媒体即整批走替换分支"——纯文本请求对应的行将被填为**全零向量**（producer 仅为带媒体请求写入 embedding，纯文本位置为零初始化）。**后果**：同批纯文本请求获得全零 embedding，输出异常。**单 batch / 不混纯文本时该约束休眠**。

**最小整改（放开 batch>1 时执行）**
- 将"整段替换"改为**按 token 位置的 scatter-merge**：生成 stage 先经文字 embedding 查表得到全部文本向量，再**仅替换媒体占位符所在行**为媒体 embedding；producer 侧同时携带"每个 token 是否为媒体占位符"的位置掩码。
- 如此，混批中的纯文本请求保留自身文字 embedding，不再被整段覆盖为零。

**定位**：属**后续任务**，与 DP>1 独立。首轮**单 batch** 可不动；需支持**批内混跑/含纯文本**时，额外执行此项 scatter-merge 改造。

---

## 复盘备查：易误判点与潜在缺陷

> 以下为分析过程中**已修正或差点误判**的要点：部分为"表象类似缺陷但实为设计如此"，部分为"表象正常但实为潜在缺陷"。记录以备后续复核。各条格式：现象 → 源码证据 → 结论/建议。

**① pad_value 仅为 radix-cache key，不参与 embedding（最易误判）**
- 现象：`pad_input_tokens` 会将占位符 token 替换为 `pad_value`，易据此推断"合并时也按 pad_value 定位、故与按 token_id 的合并代码口径不一致、且需 clamp"。
- 源码：`set_pad_value` 中 `pad_value = hash % (1<<24)`，注释明确 **"used for radix cache differentiation, NOT for embedding lookup"**（`modality_enum.py:226-228`）；`pad_input_tokens` 的返回值赋给**独立字段** `req.cache_input_ids`（`global_scheduler.py:369`），**非**被 embed 的 `input_ids`；合并路径（如 `vit_model_runner._merge_multimodal_embeddings`）使用的是**原始 token_id**。
- 结论：embed 路径全程使用原始 token_id（位于词表内）、自洽，**合并按 token_id、无需 clamp**。**不应照搬 sglang 上游**——上游为"pad_value 写入被 embed 的 input_ids + clamp"，sglang-jax 为"两个字段、两种用途"，结论相反。

**② 多图共享同一 pad_value，导致 radix cache 区分失效（潜在缺陷）**
- 现象：多图场景下前缀缓存本应区分不同图，实际无法区分。
- 源码：`MultimodalDataItem` 为"一 item = 一个模态的全部输入"（`modality_enum.py:174`）；`pad_input_tokens` 中 `image_idx` **从不自增**（`:122-129`，注释"same pad_value for all tokens of an image"）→ 多图落同一 pad_value。
- 结论：影响 **radix cache 命中率（非正确性）**。恢复 per-image 区分需细化 item 粒度 / 令 idx 随图自增。

**③ VL 合并缺数量校验，存在静默越界风险（潜在缺陷）**
- 现象：若 ViT 输出的视觉行数 ≠ 占位符数（grid/merge 计算有误等），不报错而静默错位。
- 源码：`vit_model_runner._merge_multimodal_embeddings`（`:105-139`）以 `cumsum(mask)` 为索引 gather 视觉行，**无**"占位符数 == 特征行数"断言；而 Qwen3-Omni 的合并**有**该校验。
- 建议：为 VL（及任何新模型的合并）补充数量断言，越界即报错。

**④ embed stage 的 JIT 被注释且无说明，易误判为"取消注释即可"**
- 现象：`embed_model_runner.py:48-102` 整段 JIT 被注释、`jitted_embedding` 实为 eager，且无任何说明。
- 反例：同仓 `qwen2_5_vl_stage_config_tp4.yaml` 的视觉 stage 同为 `device_kind: cpu`，却**启用了** JIT（`vit_model_runner.py:64`）。可见 CPU 并不要求 eager。
- 结论：真因更可能为**可变输入形状导致频繁重编译**（叠加"位于 CPU、分片无意义"），遂改为 eager。**整改须配合形状分桶**，非仅取消注释（见问题 2）。

**⑤ 多模态入口路由为死代码但模型仍可运行，易误判为"路由生效"**
- 现象：多模态自注册的 `/v1/chat/completions`（`http_server.py:300`）永不执行，但服务仍能处理多模态请求。
- 源码：核心同名路由先注册（`entrypoints/http_server.py:697`）、Starlette 首个匹配优先；而 `set_global_state` 将 `tokenizer_manager` 置为多模态版，核心处理函数转交其处理。
- 结论：可运行 ≠ 走正确路径；实际走核心（不完整的）构造，且依赖 import 顺序（见问题 1）。

**⑥ `is_multimodal` 恒为 False，易误判为"系统已识别多模态"**
- 源码：`configs/model_config.py:162` 无条件 False；`multimodal_model_archs`（`:791`）已定义但无消费方。多模态判定实际依赖 `server_args.multimodal` + import 副作用（见问题 1）。

---

### 附录
- 本文覆盖三项必做前置（问题 1/2/3）+ 两条后续限制（DP>1、batch>1）+ 复盘备查；问题编号为本文内部编号，独立成文。
- 全程以源码为准（branch `fix/dp-multimodal-input-embedding`）；`文件:行号` 为审读时所见，实施前请复核（行号可能随提交漂移）。
