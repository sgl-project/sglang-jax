# Text-out 多模态模型架构：现状分析、分 stage 接入问题与长期重构方向

> **目的**：本文分三步推进——
> 1. **现状（第 1 章）**：逐文件分析 sglang-jax 多模态子系统的当前（as-is）架构，提炼出"一条 multi-stage 流水线"这一承重设计；
> 2. **问题（第 2 章）**：用 MiMo-V2.5 在 v6e-16 上的真机数据，暴露"理解类、纯文本输出"模型走当前**分 stage 架构**接入时的代价；
> 3. **重构方向（第 3–6 章）**：据此引出 text-out 多模态模型的长期架构方向与 MiMo-V2.5 的具体落地。
>
> **分析对象**：当前仓库 `python/sgl_jax/srt/multimodal/` 源码（第 1 章）+ MiMo-V2.5（text+audio→text）在 `lianfang-v6e-mimo`（v6e-16）的真机复盘（第 2 章）。
>
> **核心结论（先行）**：**only text-out 的理解类多模态模型应走标准 LLM generation runtime，把多模态 encode/splice 作为模型内的 prefill adapter，而非外部 runtime stage。** 本文的论证完全基于第 1 章的 as-is 架构与第 2 章的真机问题自洽推导；该模式在 sglang upstream（如 Qwen3-VL 的 text-out 接入）已有先例，文中仅作印证性引用，不展开分析。

---

## 1. sglang-jax 多模态子系统架构全景（as-is）：一条 multi-stage 流水线

> **本章性质**：对 `python/sgl_jax/srt/multimodal/`（95 个文件、约 2.7 万行）逐文件代码分析后建立的独立理解，描述**当前仓库实际实现**（as-is），不是目标架构。后续第 2~9 章讨论的"问题"与"重构方向"都以本章为事实基线；所有论断给出 `文件:行` 出处。
>
> **本章主线（multi-stage）**：多模态子系统不是一个模型，而是一套**把模型沿模态/计算边界切成若干 stage、串成一条线性流水线的 runtime**。`GlobalScheduler` 是总控；每个 stage = 一个 scheduler + 一块独立 device mesh + 一个线程；stage 间用进程内 `queue.Queue` 传递统一的 `Req`；**哪些 stage、用什么模型、占几块卡，由"每模型一份"的 YAML 决定**。其中 **text-out 理解链路的最后一段（自回归 LLM）直接复用标准文本 runtime 的 `Scheduler`**，多模态特有逻辑只发生在第 0 个 stage（编码 + 把视觉/音频 embedding 拼进文本序列）。
>
> **怎么读**：本章只为"staging 是承重设计轴、其代价/收益按模型族分化"这一结论服务，不是各组件的详设。1.1 给整体形态（拓扑 + 两条 runtime 如何交织），1.2 给"模型如何被装配成流水线"（每模型一份 YAML），1.3 走通本文重点的 text-out 理解全流程，1.4 速览其余支撑机制，**1.5 提炼结论并承接第 2 章**。需要 `文件:行` 级细节请直接看源码，本章只在关键处标注。

### 1.1 整体形态：两条 runtime，staged 嵌套 standard

仓库里其实并存两条 runtime，多模态子系统的核心设计就是**让 staged runtime 去"嵌套"标准文本 runtime**：

| | 标准文本 runtime | 多模态 staged runtime |
|---|---|---|
| 总控 | `srt/managers/scheduler.py: Scheduler` | `multimodal/manager/global_scheduler.py: GlobalScheduler` |
| 调度单位 | continuous batching 的 `Req` | 线性流水线的 stage（每 stage 一个 scheduler + 一块 mesh + 一个线程）|
| 入口 manager | `TokenizerManager` | `MultimodalTokenizer`（前者子类）|
| 跨组件通信 | ZMQ | ZMQ（对外）+ `queue.Queue`（stage 间）|

进程/线程拓扑（`http_server.py:launch()`）：`MultimodalTokenizer` 在主进程；`GlobalScheduler + 所有 Stage` 在一个子进程内（**stage 是线程不是进程**，各占一块独立 device mesh，用进程内 `queue.Queue` 串成线性流水线）；`MultimodalDetokenizer` 是独立子进程。三个 manager 都是标准文本 manager 的子类——控制面骨架是复用的。

```
HTTP ─▶ MultimodalTokenizer ──ZMQ──▶ GlobalScheduler ┌─ Stage0 (mesh0) ─queue─┐
        (TokenizerManager 子类)        (子进程)        ├─ Stage1 (mesh1) ◀──────┘
              ▲                            │           └─ StageN (meshN) …
              └────── ZMQ ── MultimodalDetokenizer ◀── ZMQ(final BatchTokenIDOut)
```

**交织点（整个 text-out 设计的精髓）**：stage YAML 里自回归那一段 `scheduler: auto_regressive` 被解析为**标准文本 `Scheduler` 本体**（`stage.py` `import Scheduler as AutoRegressiveScheduler`）。即：

> **only-text-out 多模态模型的"生成"完全跑在标准 LLM 控制面上**（radix cache / paged attention / sampling / KV pool / 多机 SPMD 全部复用）；多模态特有逻辑被压缩到流水线的第 0 个 stage——编码模态特征 + 把它拼进文本 embedding 序列。这正是第 3 章"prefill adapter"思路在 as-is 代码里的体现，只不过当前它以**外部 stage**（而非 LLM 内部 adapter）的形态存在。

### 1.2 模型如何被装配成流水线：每模型一份 stage YAML

**"哪些 stage、用什么模型、占几块卡"完全由一份 per-model YAML 决定**（`static_configs/*.yaml`，`StageConfigRegistry` 按 model_path 解析；scheduler 名→类、model_class 名→类都是 `stage.py` 里的硬编码字典）。本仓库已接入的全部流水线：

| 模型 | stage 链（scheduler / model_class）| 任务类型 |
|---|---|---|
| **MiMo-V2.5** | `embedding`(MiMoV2_5Embedding, CPU) → `auto_regressive`(MiMoV2ForCausalLM, 16 TPU) | 图/视频/音频 → 文本 |
| **Qwen3-Omni** | `embedding`(Qwen3OmniMoeThinkerEmbedding, CPU) → `auto_regressive`(…ThinkerText…, 4 TPU) | 图/视频/音频 → 文本 |
| **Qwen2.5-VL** | `vit`(Qwen2_5_VL_VisionModel) → `auto_regressive`(Qwen2_5_VL_Generation) | 图/视频 → 文本 |
| **MiMo-Audio** | `audio_encoder`(MiMoAudioTokenizer) → `audio_backbone`(MiMoAudioForCausalLM) | 语音 ASR/TTS/理解 |
| **FLUX.1-dev** | `text_encoder`(CLIP,T5) → `diffusion`(FluxTransformer2DModel) → `vae`(AutoencoderKL) | 文 → 图 |
| **Wan2.1 / 2.2** | `auto_regressive`(UMT5, capture_hidden) → `diffusion`(Wan[Dual]Transformer3DModel) → `vae`(AutoencoderKLWan) | 文 → 视频 |

两点关键观察：(1) text-out 三个模型（前 3 行）都是 **"stage0 编码 + stage1 标准 AR"** 的两段式，stage0 可被放到 CPU 以腾出全部 TPU 给 AR；(2) **AR/文本 runtime 被反复复用**——不只 text-out 的生成段，连 Wan 的 UMT5 文本编码器也跑在 `auto_regressive` scheduler 上（靠 `capture_hidden_mode` 取 last_hidden_state）。

### 1.3 Text-out 理解链路全流程（本文重点：图/视频/音频 → 文本）

以 **MiMo-V2.5 / Qwen3-Omni**（embedding-stage 形态）为主线，端到端只有两个多模态触点（②④），其余全是标准文本控制面：

```
① HTTP /v1/chat/completions ──(is_multimodal)──▶ 构造 GenerateOmniReqInput
② MultimodalTokenizer：mm_processor 编码 media → input_ids + 每模态一个 MultimodalDataItem
       打包成 mm_inputs dict（mm_items + grid_thw + token_ids + mrope）→ ZMQ
③ GlobalScheduler：建 Req（omni_inputs=mm_inputs），to_stage_reqs("embedding") → Stage0
④ Stage0 = EmbedScheduler（CPU mesh）：模型内部
       input_embeds = text_embed(input_ids) + 各模态 tower 编码 → scatter 进占位 token 行
       ⇒ 把结果写回 mm_inputs["multimodal_embedding"]            # ← 合并发生在这里
⑤ to_stage_reqs("auto_regressive")：构造标准 TokenizedGenerateReqInput，.mm_inputs 带上述 dict → Stage1
⑥ Stage1 = 标准文本 Scheduler（TPU mesh）：
       req.multimodal_embedding ← mm_inputs["multimodal_embedding"] → ForwardBatch.input_embedding
       LLM 读 input_embedding（而非 embed(input_ids)）→ 标准 RadixAttention + KV pool → 逐 token 出文本
⑦ MultimodalDetokenizer → 标准 detokenize → HTTP 流式文本
```

整条链路的多模态交接被压缩成 **`mm_inputs["multimodal_embedding"]` 这一个 dict 键**：Stage0 写、Stage1 读。**Qwen2.5-VL 是变体**：stage0 用 `vit` 而非 `embedding`，合并放在 runner（`cumsum`-gather）而非模型内，但对下游 AR stage 的契约完全一致。

### 1.4 其余支撑机制速览（皆为流水线的脚手架）

以下机制都只是为"让 `Req` 在 stage 间流动 + 各 stage 跑模型"服务，了解其存在即可，无需逐一深究：

- **核心数据结构**：`MultimodalDataItem`（一种模态一个 item，`feature`/`precomputed_embeddings` 二选一，model-specific 字段进 dict）是**多模态特征的唯一真相源（`mm_items`）**；`assemble_mm_inputs()`（纯 numpy、模型无关）把 `mm_items` 翻成各模型 forward 的 per-modality kwargs；`Req` 是 stage 间传递的**超大并集 dataclass**（扩散/Omni/音频字段全堆一起），靠 `to_stage_reqs()` / `from_stage()` 做 stage 适配。
- **stage 执行三层**：`Stage`(持 mesh+queue) → `XxxScheduler`(薄事件循环) → `XxxModelWorker`(极薄壳) → `XxxModelRunner`(load_model + `nnx.split/merge` + `jax.jit` + forward)，embed/vit/encoder/diffusion/vae/audio 全是这个统一套路。
- **embedding 合并两种实现**：模型内 scatter（MiMo/Qwen3-Omni 的 `embedding` stage）vs runner 内 gather（Qwen2.5-VL 的 `vit` stage）——契约统一在 `multimodal_embedding`，但实现未收敛。
- **生成/音频链路**：FLUX/Wan 是 `text_encoder → diffusion(去噪循环) → vae` 三段；MiMo-Audio 是 `audio_encoder(mel→RVQ码) → audio_backbone(stage 内跑完整自回归循环)`。这些是与 text-out 对照的另一类（详见 1.5 的模型族划分）。
- **dispatch 现状**：stage 拓扑 / scheduler 类 / model_class / 模型 config / host processor / 是否走 omni 路径——**全部基于 model 名或路径子串匹配**（如 `"mimo-audio" in model_path.lower()`），这是第 3 章 §3.3 建议用 capability/registry 取代的根因。
- **依赖方向**：多模态层**单向依赖**标准 `srt/`（`Scheduler`/`TokenizerManager`/`ForwardBatch`/`RadixAttention`/`KVCache`/`BaseModelRunner`…），text-out 的 AR 本体甚至直接就是 `srt/models/{mimo_v2_pro, qwen2, …}`。

### 1.5 提炼：staging 是承重设计轴，代价/收益按模型族分化（承接第 2 章）

把 1.1–1.4 收拢成一个判断：**整个子系统的承重设计决策只有一个——"沿模态/计算边界把模型切成多个 stage、串成一条线性流水线"。** 其余机制都是为它服务的脚手架。三个不变量：

- stage 的最小定义 = 一个 scheduler + 一块独立 mesh + 一个线程，固定芯片池被**不重叠地**切给各 stage；
- 流水线严格线性（0→1→…→final），stage 间唯一耦合 = `Req` + `mm_inputs["multimodal_embedding"]` 这一个交接键；
- 唯一被"白嫖"的重型组件 = AR scheduler（就是标准文本 `Scheduler` 本体）。

**关键提炼：同一套 stage 机制，价值随模型族翻转。** "该不该分 stage"不是全局命题——这正是第 2 章真机复盘要验证的预判：

| 模型族 | stage 是否真异构 | 流水线并行收益 | stage 边界成本 | 净评价 |
|---|---|---|---|---|
| 生成类（FLUX 图 / Wan 视频）| 是：text-encoder / DiT / VAE 计算量与并发度迥异 | **有**：各 stage 可独立 batch / scale | 可被吞吐摊薄 | **收益 > 成本，stage 合理** |
| 音频生成（MiMo-Audio TTS）| 是：codec / backbone / vocoder | 有 | 可摊薄 | **收益 > 成本** |
| 理解-纯文本输出（MiMo-V2.5 / Qwen3-Omni / Qwen2.5-VL）| **否**：stage0 只是"算个 embedding"（ms 级），AR 是几百 GB 主体 | **≈ 0**：stage0 永远紧接喂 stage1 | **全额付**：①固定卡池被静态切分（embed 1 + AR 16 > 16 卡）②跨 mesh host-roundtrip 搬 `[seq,hidden]` ③embed 独立 config/mesh 与主引擎分叉 ④多 stage 各自跨 host collective 需额外协调 | **成本 > 收益，stage 是纯负担** |

一句话：**`mm_inputs["multimodal_embedding"]` 这个交接键，对生成类是"流水线的自然边界"，对 text-out 理解类却是"一次毫无必要的跨进程/跨 mesh 往返"**。同时也有正向价值：text-out 的"生成"段零成本复用标准 LLM 控制面，且 `mm_items` / `assemble_mm_inputs` 已体现"模态中立、模型自治"的雏形——是向"prefill adapter"收敛的良好起点。

**承接第 2 章**：上表"text-out 模型 stage 边界成本 > 收益"目前还只是从架构形态推出的*预判*。**第 2 章把它落到真机**——MiMo-V2.5 在 v6e-16（4 host × 4 chip）跑通过程暴露的 12 个 bug 里，有 7 个正是上表"stage 边界成本"四项的直接产物。第 3 章再据此给出"不分 stage、把多模态 encode/splice 收进单一模型 forward"的长期架构方向。

---

## 2. MiMo-V2.5 真机测试复盘：分 stage 导致的问题与设计难点

> **承接 1.5**：上一章末尾从架构形态推断出"对 text-out 理解类模型，stage 边界成本 > 收益"。本章把这个预判落到真机——用 MiMo-V2.5 在 v6e-16 上的一手 bug 数据检验它，并量化 stage 架构到底贡献了多少复杂度。
>
> **背景**：MiMo-V2.5（text+audio → text，audio 塔极小）在 `lianfang-v6e-mimo`（v6e-16，4 host × 4 chip）上完成 multi-host 改造 + text-only / text+audio / 真实语音端到端跑通。过程中真机暴露 12 个 bug。本节梳理其中**由"分 stage"架构直接导致或显著放大**的问题，作为"对 only-text-out 模型是否取消 stage"的一手论据。
>
> **复盘对象 = 两-stage 接入方案**（完整设计见 [`mimo_v25_step2_model_integration.md`](mimo_v25_step2_model_integration.md)）：沿用 `embedding → auto_regressive` 线性两段——**stage0 `embedding`** 用新建的 `MiMoV2_5Embedding`，在单个模型内把 MiMoVL ViT（图/视频）、audio understanding tower（host RVQ tokenizer 产出 `audio_codes` → speech_embeddings→input_local→projection）、text embedding 三塔按 token_id scatter 合并成 `input_embeds`，写回 `mm_inputs["multimodal_embedding"]`；**stage1 `auto_regressive`** 直接复用文本 backbone `MiMoV2ForCausalLM`，仅经 input_embedding hook 读 `forward_batch.input_embedding`（1-D RoPE、无 mrope/deepstack）。这正是第 1 章 1.2 / 1.3 描述的 as-is 两段式形态，当前以 **DP=1 / batch=1** 落地（DP>1、batch>1、ViT 模块、视频交错音轨均列为后续项）。**下面 12 个真机 bug 正是跑通这套方案时暴露的。**

### 2.1 量化结论：12 个真机 bug 中 7 个由 stage 架构引起/放大

| bug | stage 导致？ | 说明 |
|---|---|---|
| 1 标准 scheduler `num_subscribers` 缺失 | ✅ stage | comm_backend 只在 rank0 注入；stage 旁路了标准 scheduler 原生 pub/sub |
| 2 embed 量化 config 为 dict 崩 loader | ✅ stage | embed 走独立裸 `AutoConfig` 路径，与 AR 的 `ModelConfig` 分叉 |
| 3 fused MoE `num_tokens % ep_size` | ❌ 非 stage | fused MoE 固有约束，统一架构也有 |
| 4 fused MoE `local % t_packing` | ❌ 非 stage | 同上 |
| 5 embed 缺 `get_total_num_kv_heads` | ✅ stage | embed 裸 config 缺 ModelConfig 方法 |
| 6 多 stage 并行跨 host collective 死锁 | ✅ stage | **最典型**：两 stage 各发 `broadcast_one_to_all`，4 host 上顺序不一致 → 死锁 |
| 7 非 addressable shard 缺 dtype | ✅ stage | embed 单 CPU-device mesh 在非 rank0 无 addressable shard |
| 8 缺 torchvision | ❌ 非 stage | processor 依赖 |
| 10 codec relative-import / soundfile / audio_tokenizer | ❌ 非 stage | codec / host 依赖与打包 |
| 11 audio encoder reshape `ShardingTypeError` | ◑ stage 放大 | embed 独立 explicit-sharding mesh 使其单独触发 |
| 12 scatter `ShardingTypeError` | ◑ stage 放大 | 同上 |

**超过一半真机 bug 来自 stage 架构本身，而非模型或 MoE。** 这是"重新考虑架构"的最硬证据。

### 2.2 分 stage 引起的设计难点（按层面）

**(1) 设备预算 / cpu-tpu 拓扑**
- `DeviceManager.allocate` 把固定芯片池**不重叠地切给各 stage**：embed=1 + AR=16 = 17 > v6e-16 的 16 → 起不来。
- embed 实际只用 ~2GB（text-embed + 小 audio 塔），却被迫独占一整卡或挪到 CPU。
- static yaml 里"每个 stage 选 `device_kind: cpu/tpu`"这个配置负担**只因 stage 分开放置才存在**，且派生出 bug 7（CPU 单 device mesh 在多 host 上的 non-addressable shard）。统一架构只有一个 mesh，不存在"切给谁 / 放哪"。

**(2) 多 host 分布式协调**
- **bug 6 死锁**：两 stage 在不同线程各发 `broadcast_one_to_all`，host 间 embed-broadcast vs AR-broadcast 先后不确定 → collective 错配死锁。本质是"多个独立计算单元在同进程内跨 host 发 collective"；统一模型只发一条有序 collective 序列，根本不会有。需加"串行启动 stage"才绕过。
- **`MultiHostQueueBackend` 整层**：AR stage 用标准 Scheduler，但被 `QueueBackend` 喂数据 → 标准 scheduler 自带的多 host pub/sub 被旁路，只有 rank0 收到请求 → 必须**重造 pub/sub 广播层**让 AR batch 同步到所有 rank。统一架构里 scheduler 直接拥有 recv+broadcast，无此二次协调。
- **metadata broadcast 跑两遍**：embed 与 AR 各自扫权重元数据 + 各自 broadcast，两条独立跨 host 同步；统一模型一次加载。

**(3) 跨 stage 数据传输（host roundtrip）**
- stage0 的 `multimodal_embedding`（embed mesh 上的 jax.Array）要 device→host(`np.asarray`/pickle)→device 搬到 stage1 的另一 mesh，每请求拷一份完整 `[seq, 4096]`。
- 统一模型：embedding 留 device，直接进 LM forward，无序列化、无 host 往返。only-text-out 模型这里没有流水线收益（见 (6)）。

**(4) 配置 / 模型构建分叉（bug 2、5 之根）**
- embed：`AutoConfig.from_pretrained` → 裸 `PretrainedConfig`；AR：`ModelConfig.from_server_args` → 完整 `ModelConfig`。两条 config 构建路径。
- 后果：embed 裸 config 带 checkpoint 的 fp8 `quantization_config` dict（崩 loader，bug 2）、又缺 `get_total_num_kv_heads`（崩 WeightLoader，bug 5）。统一架构只有一个 ModelConfig，两 bug 都不存在。

**(5) 独立 mesh 的分片阻抗失配（bug 11、12）**
- embed 是自己的 jitted 模型 + 自己的 explicit-sharding mesh。小 audio 塔的 reshape `[N,4,1024]→[1,N,4096]` 和 scatter 在该 mesh 上单独触发 `ShardingTypeError`，需到处补 `with_sharding_constraint(P())`。
- 统一架构里 embedding merge 发生在主模型 forward 内、主 mesh 上，与 LM 分片本就一致，无"两 mesh 对接"的阻抗。

**(6) 控制平面复杂度 vs 收益为零**
- 为在 stage 间搬请求，有一整套：`GlobalScheduler` + `Stage` + 每 stage scheduler + `QueueBackend` + `Req.to_stage_reqs` + `req_store` 追踪 `current_stage`。
- 而 only-text-out 模型，stage0 永远紧接喂 stage1，embed 计算 ms 级、AR 是整个 293GB MoE —— **流水线并行吞吐收益 ≈ 0，却付全部协调成本**。
- stage 架构初衷是**异构计算流水线**（大 ViT 一组卡、LLM 另一组，各自 scale）。对"理解类、纯文本输出"模型，多模态部分本质只是"算点 embedding 塞进 LM 输入"，正是 sglang 在**单模型 forward 内**用 `get_input_embeddings` + multimodal merge 做的事，无需独立 stage。

### 2.3 诚实平衡：stage 架构确实给了什么

- **异构设备放置**：真有大 vision 塔（MiMo-V2.5 的 729M ViT）时，"ViT 放 CPU/小 TPU、LLM 放大 TPU"有意义；encoder/decoder 独立 scale 有价值。
- **复用现成 per-modality stage**（vit / audio_backbone / vae / diffusion scheduler 已存在）——对**有生成塔的模型**（TTS、diffusion）这套流水线是对的。
- 故问题不是"stage 架构错"，而是**"对 only-understanding-text-out 这一类模型，stage 拆分的成本远大于收益"**。

### 2.4 小结（承接第 3 章长期架构方向）

分 stage 在本次集中造成三件事：**(a) 固定芯片池被迫静态切分 + cpu/tpu 拓扑选择**、**(b) 多 stage 各自跨 host collective 的分布式协调地狱**、**(c) embed 独立 config/mesh 与主引擎的分叉**。这三件对 MiMo-V2.5（text+audio→text，audio 塔极小）**未换来任何流水线收益**。

因此方向——**对 only-text-output 的理解类模型，不分 stage、把多模态 embedding 合进单一模型 forward**——从本次 bug 分布与难点看站得住：可一次性消掉上述 7 个 stage-induced bug 的根因，且不牺牲这类模型的任何能力（该模式在 sglang upstream / Qwen3-VL 的 text-out 接入中已有先例）。下文第 3 章给出这一方向的长期架构设计，第 4 章给出 MiMo-V2.5 的具体落地。

## 3. 对长期架构的建议

> 本章不引入新依据，只把第 1 章（as-is 架构）与第 2 章（真机问题）的结论收敛成长期方向：**按"输出形态"而非"是否多模态"划分 runtime 主干**，并把 text-out 理解类模型的多模态 encode/splice 从外部 stage 收回为模型内的 prefill adapter。

### 3.1 两条 runtime 主干

第 1 章 1.1 / 1.5 已观察到：仓库里其实并存两条 runtime 主干，且 stage 切分的价值随模型族翻转。长期建议把这件事**显式化**：

```text
A. text-out multimodal models
   -> standard generation runtime
   -> multimodal prefill adapter

B. diffusion / image-video generation / audio generation / vocoder
   -> staged multimodal runtime
   -> stage execution supports rank_scope / distributed / output_transport
```

不要再用“是否多模态”决定是否进入 staged runtime，而应按输出形态和 runtime 主干决定。

### 3.2 text-out multimodal 主路径

这条主路径与第 1 章 staged 链路（`GlobalScheduler → stage0 embedding → stage1 AR`）的本质区别只有一句：**把多模态 encode/splice 收回到同一进程、同一模型 forward 内完成，不再有外部 stage、跨 mesh 往返或第二套 scheduler。** 整条链路绝大部分复用标准文本 generation 控制面，只在 **processor（host 侧）** 与 **ModelRunner prefill（device 侧）** 两处插入多模态逻辑。

建议目标链路：

```text
OpenAI / GenerateReqInput
  -> TokenizerManager
       MultimodalProcessorRegistry.resolve(...)
       processor returns input_ids + mm_inputs
  -> TokenizedGenerateReqInput(mm_inputs=...)
  -> standard Scheduler
       Req.mm_inputs = recv_req.mm_inputs
  -> ScheduleBatch
       collect modality tensors
       compute positions / mrope / cache ids
  -> ForwardBatch
       carry modality tensors
  -> ModelRunner prefill adapter
       encode_multimodal JIT(s)
       splice_multimodal JIT
       clear raw mm fields
       run LLM JIT
  -> decode standard AR
```

逐节点说明（①–④⑥⑧ 是原样复用的标准生成控制面，②⑤⑦ 才是多模态插入点）：

1. **OpenAI / `GenerateReqInput`**：text-out 多模态请求复用标准生成请求结构，而非 staged path 的 `GenerateOmniReqInput`，从而天然继承 `n` / `rid` / `extra_key` / logprobs / streaming / abort / DP 等字段（staged path 的 `GenerateOmniReqInput` 则容易丢失这些字段，见 §4.1）。
2. **TokenizerManager + Processor**（多模态插入点①，host 侧）：由 `MultimodalProcessorRegistry.resolve(model_config)`（见 §3.3）按能力选 processor，在 CPU 侧把 raw media 编码成特征、生成把占位符展开后的 `input_ids`，并产出 `mm_inputs`（以第 1 章 1.4 的 `mm_items` 为载体）。
3. **`TokenizedGenerateReqInput(mm_inputs=...)`**：`mm_inputs` 作为标准请求的 side-channel 透传——这条通道在当前仓库已经存在（第 1 章 1.3 的 AR stage 正是这样接收 `mm_inputs`），无需新建请求结构。
4. **standard Scheduler**：仅 `Req.mm_inputs = recv_req.mm_inputs`，其余（continuous batching / radix cache / 多机 broadcast）完全走标准路径。多机下 `mm_inputs` 由标准 scheduler 自带的 pub/sub 广播，**不再需要 staged path 的 `MultiHostQueueBackend` 二次协调**（消除第 2 章 bug 6 跨 host collective 死锁与 broadcast 重建）。
5. **ScheduleBatch**（多模态插入点②）：把各 `mm_items` 收集成 per-token 张量，并计算 positions / mrope / `cache_input_ids`（用 content-hash 区分不同 media，复用第 1 章 1.4 的 pad_value 机制）。
6. **ForwardBatch**：携带这些 modality 张量进入模型 forward——与第 1 章 1.3⑥ 的 `input_embedding` 通道一致，差别只是 encode 改到同一 forward 内做。
7. **ModelRunner prefill adapter**（多模态插入点③，device 侧、唯一在 device 上的多模态计算）：按 §3.6 切成三段 JIT——`encode_multimodal`（模态塔编码）→ `splice_multimodal`（按占位符把 modality embedding 拼进文本 embedding）→ 清除 raw mm 字段 → `run LLM`。因为编码/拼接与 LLM 同 mesh、同进程，**消除了第 2 章 §2.2(3)(5) 的跨 mesh host-roundtrip 与分片阻抗失配**。
8. **decode 标准 AR**：decode 阶段只看 `input_embedding`，**不触发** multimodal encoder——多模态成本只在 prefill 付一次。

**关键不变量（保证正确性与效率）**：

- **side-channel 复用**：多模态只经 `mm_inputs` 进出，不新增请求/调度结构，从而免费继承标准生成控制面；
- **两处插入点**：多模态逻辑只存在于 processor（host）与 prefill adapter（device），其余全是标准 path；
- **JIT 隔离**：raw modality 张量只进 encode/splice JIT，不进 LLM JIT 的 cache key（§3.6），decode 不触发 encode；
- **fail loud**：需要 processor 的模型，若 processor 初始化失败或遇到不支持的模态，必须显式失败，不静默退化为 text-only（§3.3）。

> 注：链路中的 `MultimodalProcessorRegistry`（§3.3）与通用 prefill adapter（§3.5）是长期目标；当前仓库已具备其骨架（标准 Scheduler/ScheduleBatch/ForwardBatch 已能透传 `mm_inputs`、模型内已能 encode+splice），第 4 章给出 MiMo-V2.5 先以 model-specific 分支落地、第二阶段再抽象的过渡方案。

### 3.3 Processor registry

建议定义：

```python
class MultimodalProcessor:
    @staticmethod
    def matches(model_config) -> bool: ...
    def process(request_or_text, media_inputs) -> tuple[list[int], dict]: ...
```

由 registry 负责选择：

```python
processor = MultimodalProcessorRegistry.resolve(model_config, model_path)
```

原则：

- processor 初始化失败必须 fail loud；
- unsupported modality 必须 fail loud；
- 不允许静默 text-only fallback。

### 3.4 Modality-neutral `mm_inputs`

第 1 章 1.4 的 `mm_items`（`MultimodalDataItem` + `assemble_mm_inputs`）已是"模态中立"的雏形，但 token-id / grid / codebook 等元信息仍分散在 dict 顶层。长期建议进一步标准化：

```python
mm_inputs = {
  "items": [
    {
      "modality": "image" | "video" | "audio",
      "feature": ...,
      "token_id": ...,
      "positions": ...,
      "metadata": {...},
    }
  ],
  "position_encoding": {
    "type": "mrope" | "rope",
    ...
  },
}
```

或者保留 `mm_items`，但要求每个 item 标准化：

```text
modality
feature / codes / precomputed_embedding
token_id
token_lengths / offsets / grid_thw / codebook_sizes
pad_value / content_hash
```

### 3.5 Model prefill adapter protocol

第 1 章 1.4 暴露的两种 embedding 合并实现（模型内 scatter / runner 内 gather）应收敛成一个由模型统一暴露的 prefill adapter protocol：

```python
class MultimodalPrefillAdapter:
    def has_multimodal(forward_batch) -> bool: ...
    def encode(forward_batch) -> EncodedMultimodal: ...
    def splice(input_ids, encoded, metadata) -> SplicedEmbeddings: ...
    def clear_raw_fields(forward_batch) -> ForwardBatch: ...
```

模型可以通过 capability 暴露：

```python
supports_multimodal_prefill = True
```

或：

```python
def get_multimodal_prefill_adapter(self): ...
```

### 3.6 JIT split 原则

把第 1 章 1.4 的 runner JIT 切分模式固化为三段（也是缓存隔离的要求）：

```text
modality encoder JIT(s)
splice JIT
LLM JIT
```

原则：

- raw multimodal fields 只能进入 encoder/splice JIT；
- LLM JIT 只看 `input_embedding` / `deepstack` 等标准字段；
- 进入 LLM JIT 前清除 raw mm fields；
- 按 modality bucket 预编译；
- decode path 不触发 multimodal encoder。

---

## 4. 对 MiMo-V2.5 接入方案的重构方向

### 4.1 从 staged 过渡到标准 path：现状与第一阶段目标

当前 MiMo-V2.5 是 staged 过渡方案（`GenerateOmniReqInput → MultimodalTokenizer → GlobalScheduler → stage0 MiMoV2_5Embedding → stage1 MiMoV2ForCausalLM AR`）：改动集中、对核心 runtime 侵入小，可先验证 host codec / audio tower / scatter / AR input_embedding hook，但正好踩中第 2 章列出的全部 stage 代价——multi-host 需在 stage boundary 重建 broadcast、OpenAI 字段易丢、`cache_input_ids` 易断链、stage0 CPU/TPU 拓扑复杂、与 HF 单 checkpoint 结构不一致。

**第一阶段目标**：只做 **audio+text+image/video → text** 的功能闭环，按第 3 章主路径接入标准 generation path（不再扩展 staged runtime；通用 registry/adapter、hardcode 清理留到第二阶段）。目标链路：

```text
GenerateReqInput(audio_data, text)
  -> TokenizerManager / MiMoV25Processor: raw audio→mel→audio_tokenizer/RVQ→audio_codes;
                                          prompt→input_ids（<audio_pad> 展开到 token_lengths）; mm_inputs 带 audio_codes item
  -> TokenizedGenerateReqInput(mm_inputs)  -> standard Scheduler (Req.mm_inputs)
  -> ScheduleBatch: 收集 audio_codes + placeholder positions + content hash
  -> ForwardBatch: audio_codes / audio_placeholder_positions
  -> ModelRunner prefill: jitted_audio_encode → jitted_splice_embeds → 清 raw audio → jitted_run_model(LLM)
  -> decode standard AR
```

验收标准：text-only 仍走标准 path；raw audio 单请求能产出 `audio_codes` 并完成 `encode→splice→LLM`；decode 不触发 audio encode；multi-host 复用标准 scheduler broadcast，不引入 staged AR queue broadcast。

### 4.2 关键设计结论（第一阶段）

基于 HF/upstream 现有实现与第 1 章 as-is 分析，以下几条是确定的：

1. **Processor 边界 = `audio_codes`（不是 mel）**：`mel → audio_codes` 需冻结的 `MiMoAudioTokenizer`/RVQ quantizer 参与（非纯信号处理），故第一阶段把它留在 host-side processor（加载 checkpoint 的 `audio_tokenizer/` 子目录），**不移植进 JAX JIT**，规避 torch/JAX 混合与 RVQ 动态长度问题；模型内的 audio tower 只接收 `audio_codes`。
2. **复用标准控制面，不新增 staged 结构**：不再新增 `GenerateOmniReqInput`/`GlobalScheduler`/stage yaml/stage0 embed，直接用标准 `GenerateReqInput` 的 `mm_inputs` side-channel（第 1 章 1.3 已证明 AR stage 本就这样收 `mm_inputs`），天然继承 `n`/`rid`/logprobs/streaming/abort/DP/multi-host broadcast。
3. **audio tower 权重并入标准 MiMo AR model class**：一次加载 `model.* + lm_head.* + speech_embeddings.* + audio_encoder.*`，`audio_tokenizer/` 仍由 host codec 加载——**消除 staged path 的两套 config/loader 分叉**（stage0 裸 `AutoConfig` + stage1 `ModelConfig`，即第 2 章 bug 2/5 之根）。
4. **three-JIT 切分（落地 §3.6）**：`jitted_audio_encode → jitted_splice_embeds → jitted_run_model`；进 LLM JIT 前清空 raw audio 字段只留 `input_embedding`，使 LLM JIT cache key 不受 `audio_codes` 长度影响。
5. **cache 正确性**：`MultimodalDataItem.set_pad_value()` 基于 `audio_codes` hash，`cache_input_ids` 把 audio 占位 token 替换成该 item 的 pad value（复用第 1 章 1.4 机制），保证 same text + different audio 不串 prefix；成本高时可临时禁用 radix cache 但需标注。
6. **collector 契约（hard-fail）**：`sum(token_lengths) == count(input_ids==audio_token_id)`、`ceil(T/group_size) == sum(token_lengths)`、per-channel code range 按 `codebook_sizes` 校验；placeholder positions 在 padded layout 计算、padded 槽指向 sink 避免越界。
7. **multi-host**：rank0 processor 产出 `mm_inputs`，标准 Scheduler broadcast 给所有 rank，各 rank 同序执行 `audio_encode→splice→LLM`、仅 rank0 输出——比 staged 双 queue broadcast 简单。

第一阶段允许 **MiMo-specific hardcode**（`TokenizerManager` 按 model_type 初始化 `MiMoV25Processor`、`ScheduleBatch` 用 MiMo audio collector、`ForwardBatch` 显式加 audio 字段、model class 用 `encode_audio`/`splice_embeds` capability 检测），第二阶段再抽 §3.3 registry 与 §3.5 adapter；但所有路径必须 **fail-loud**，不静默退化为 text-only。`mm_inputs` 中 audio item 的建议 schema：

```python
MultimodalDataItem(
  modality=Modality.AUDIO,
  feature=audio_codes,                 # np.int32 [T, 20], time-major
  model_specific_data={"is_codes": True, "token_lengths": [ceil(T/4)],
                       "group_size": 4, "codebook_sizes": [1024, 1024, 256, 128, ...]},
)
```

### 4.3 实施顺序、本轮落地与验证清单

**实施顺序**：① `TokenizerManager` 接 `MiMoV25Processor` 产出 `TokenizedGenerateReqInput.mm_inputs` → ② `ScheduleBatch` 加 audio collector → ③ `ForwardBatch` 携带 audio 字段 → ④ audio tower 并入标准 MiMo model class（暴露 `encode_audio`/`splice_embeds`）→ ⑤ `ModelRunner` 加 audio three-JIT → ⑥ 接 cache hash / `cache_input_ids` → ⑦ 先 single-host 再 multi-host 跑 text/audio。

**本轮 PoC 已落地**（按上述顺序）：`TokenizerManager` 按 `model_type + audio_config` 初始化 `MiMoV25Processor` 直接产出标准 `mm_inputs`（OpenAI chat 不再走 `GenerateOmniReqInput`）；标准 `Scheduler` 透传 `mm_inputs` 并基于 `mm_items` 生成 `cache_input_ids`；`ScheduleBatch._collect_audio_tensors` + `ForwardBatch`（`audio_codes`/`audio_placeholder_positions`/`n_real_audio_tokens`）携带张量；`ModelRunner` 跑 `encode_audio → splice_embeds → LLM` 三段 JIT；MiMoV2 Flash/Pro 模型类挂 `MiMoV25AudioUnderstandingEncoder` 并暴露 capability；FP8 config 追加 `audio_encoder.*` ignore。语法 + 3 个 MiMo 目标测试（22 tests + 4 subtests）已过（`uv run python`，Py 3.12）。

**待验证清单**（真机）：

| # | 验证项 | 通过判据 |
|---|---|---|
| 1 | codec parity | 固定 wav，host `audio_tokenizer.encode` 的 shape/range/token_lengths 与 upstream/HF 一致 |
| 2 | single request | raw audio + text 走标准 path 生成 token |
| 3 | text-only 回归 | MiMo text-only 不触发 audio branch |
| 4 | mixed batch | text-only + audio 不互相污染；若不支持 batch>1 须显式限制 |
| 5 | cache 正确性 | same text + different audio 不共享 prefix（或明确禁用 radix cache） |
| 6 | multi-host | v6e-16 上 broadcast 后各 rank `audio_encode/splice/LLM` 同序 |

**已知遗留**：codec parity / multi-host 尚未真机验证（#1、#6）；batch>1 与多段 audio 仅 PoC 单段；cache 多段 per-span hash 待完善；audio JIT 暂随 shape 首编译（AOT precompile 第二阶段）；权重映射复用 staged `weights_mapping.py`，需真实 checkpoint 跑 `load_weights` 确认无漏 key。

### 4.4 第二阶段（不阻塞第一阶段）

通用 `MultimodalProcessorRegistry`（§3.3）与 `MultimodalPrefillAdapter`（§3.5）；modality-neutral `ForwardBatch` / generic collector；image/video 接入并与 audio 分支统一到同一 adapter；多段 audio 完整 API、chunked-prefill 跨 chunk / per-audio embedding cache；audio_encode/splice 全量 AOT precompile；staged path 清理/删除。

---

## 5. 需要继续细化的问题

后续讨论建议围绕以下问题展开：

1. `mm_inputs` 是否采用现有 `MultimodalDataItem`，还是定义新的 text-out multimodal schema？
2. `ScheduleBatch` 中 modality-specific collect 逻辑如何注册化，避免 Qwen/MiMo hardcode？
3. `ForwardBatch` 字段是继续显式展开 `pixel_values/audio_codes/...`，还是用 structured modality batch？
4. ModelRunner 的 adapter protocol（§3.5）如何设计，既支持 vision three-JIT，又支持 MiMo audio/vision？
5. `cache_input_ids` / multimodal content hash 应在 processor、ScheduleBatch 还是 Req 层生成？
6. multi-host 下 large `mm_inputs` broadcast 成本如何控制？是否需要 rank-local processor 或 sharded generation？
7. staged runtime 与 standard generation runtime 的边界如何清晰划分？
8. 已有 Qwen2.5-VL / Qwen3-Omni staged 代码是否迁移，还是仅新模型走新架构？

---

## 6. 当前推荐原则

1. **Text-out MLLM / omni understanding**：优先接入 standard generation runtime；多模态 tower 是 prefill adapter，不是外部 stage。
2. **Diffusion / image-video generation / audio generation / vocoder**：保留 staged multimodal runtime，并把 multi-host 做成 stage execution 的一等能力。
3. **Processor fail loud**：需要 processor 的模型，processor 初始化失败即失败；不允许静默 text-only fallback。
4. **LLM JIT cache 隔离**：raw multimodal tensors 不进入 LLM JIT cache key；encoder/splice 与 LLM 分 JIT。
5. **Capability-based dispatch**：控制面通过 registry/capability 找 processor 和 prefill adapter，不在 TokenizerManager/ModelRunner 堆模型名分支。
6. **MiMo-V2.5 staged path 是过渡方案**：长期应向第 3 章的 standard generation path（模型内 prefill adapter）靠拢；该模式在 sglang upstream 已有先例。
