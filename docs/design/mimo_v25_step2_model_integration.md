# MiMo-V2.5 接入 · 第二步：模型级接入方案

> **前提**：第一步 prework 的三个前置问题已解决——① 入口路由归一 + `is_multimodal` 由 model_type 推导；② 编码 stage 已开 JIT + 形状分桶 + 结构化输出 + 放 TPU；③ 编码 stage 执行器已是**模型无关契约**（config 装载不再写死 `thinker_config`、`_prepare_input` 支持 per-model、注册声明式）。本文在此基础上只讲"MiMo-V2.5 这个模型本身怎么接进来"。
>
> **架构前提**：沿用现有 **2-stage `embedding → auto_regressive` 流水线**（不动调度/DP 双栈大重构）。本轮范围 **DP=1、batch=1**；**DP>1 / batch>1 见 §6 限制**。
>
> **数据来源（重要）**：以源码为准——分析对象为 sglang-jax 分支 `fix/dp-multimodal-input-embedding`（`/Users/lianfang/primatrix/sglang-jax-fix-dp-mm`）；模型结构数值取自 **MiMo-V2.5 真实 `config.json`（经 ModelScope 镜像读取）** 与 sglang 上游实现。本文自洽、不援引其它设计文档；不确定项进 §6 开放问题。`文件:行号` 为审读时所见，实施前复核。

---

## 1. MiMo-V2.5 的 HF 模型架构梳理

### 1.1 总览
MiMo-V2.5 是一个**原生 omni 的稀疏 MoE 大模型**：`model_type="mimo_v2"`、`architectures=["MiMoV2ForCausalLM"]`，**单一 checkpoint** 内同时持有：文本 MoE backbone（`self.model`）+ 视觉塔（`self.visual`）+ 音频理解 encoder（`self.audio_encoder`）。三模态全部做**理解**（text/image/video/audio in → text out），**不含**任何生成/vocoder。`config.json` 是**扁平结构**：`vision_config` / `audio_config` 直接挂在顶层，**没有 `thinker_config`** 这层（与 Qwen3-Omni 不同）。

```
                ┌──────────── MiMoV2ForCausalLM (单 checkpoint) ───────────┐
 image/video ─► │  self.visual = MiMoVL ViT ─┐                            │
       audio ─► │  self.audio_encoder ───────┤► 各自出 [N,4096] 特征      │
                │  self.model = MiMo-V2-Flash MoE LM（文本 + 生成）        │
        text ─► │  self.model.embed_tokens ──┘  把三模态特征 scatter 进序列 │
                └──────────────────────────────────────────────────────────┘
```

### 1.2 LLM backbone = MiMo-V2-Flash（JAX 已有 text-only 实现，主要复用）
稀疏 MoE + hybrid 滑窗/全局注意力。真实 config 关键字段：

| 项 | 值 |
|---|---|
| hidden_size / 层数 | 4096 / 48 层（`moe_layer_freq` → 第 0 层 dense，1–47 MoE） |
| MoE | `n_routed_experts=256`、`num_experts_per_tok=8`、无 shared expert、gate `scoring_func=sigmoid` + `topk_method=noaux_tc`、`moe_intermediate_size=2048` |
| 混合注意力 | `hybrid_layer_pattern` 实测 **9 个 Full(GA) + 39 个 SWA**；`sliding_window=128` |
| heads | `num_attention_heads=64`；**KV：GA=4 / SWA=8**（`num_key_value_heads=4` / `swa_num_key_value_heads=8`）；`head_dim=192`、`v_head_dim=128` |
| RoPE | **1-D 标准 RoPE**（`rope_scaling.type=default`，**无 mrope**）；`rope_theta`=1e7(GA)/1e4(SWA)；`partial_rotary_factor=0.334` |
| 其它 | `attention_value_scale=0.707`、`attention_projection_layout=fused_qkv`；**sink bias 仅 SWA 层**（`add_swa_attention_sink_bias=true`、full 层 false） |
| 量化 | FP8 e4m3，`weight_block_size=[128,128]`；**全部 `o_proj` 排除量化** |
| MTP | 3 层（投机解码） |

> JAX 现有 `python/sgl_jax/srt/models/mimo_v2_{flash,pro,nextn}.py` 是这套 backbone 的 **text-only** 实现 → **stage1 的 LM 主要复用之**，按上表逐字段对齐即可（详见 §3.2）。

### 1.3 Vision = MiMoVL ViT（JAX 无对应，需新建）
`vision_model_type="mimovl"`，真实 `vision_config`：

| 项 | 值 |
|---|---|
| depth | 28；full-attn 层 `fullatt_block_indexes=[0,9,18,27]`（4 full + 24 窗口） |
| 窗口注意力 | `vit_window_attn_types`∈{-1,0,1} = full/**行 1-D 窗口**/**列 1-D 窗口**；`window_size=128`、`visual_token_window_size=64`；`use_sink=true` |
| 维度 | hidden 1280、`num_heads=32` / KV 8（GQA，`num_query_groups=4`）、**head_dim=1280/32=40** |
| patch | `patch_size=16`、`spatial_merge_size=2`、`temporal_patch_size=2` |
| 输出 | `out_hidden_size=4096`（== LM hidden；ViT 自带 merger projector，**不需额外 adapter**） |

预处理为 Qwen2.5-VL 风格（`processor_config`：`image_min/max_pixels=8192/8388608`、`merge_size=2`、`patch_size=16`）。

### 1.4 Audio 理解 = 冻结 RVQ tokenizer + 小型 local transformer + projection（JAX 可复用 building blocks）
**需特别说明**：MiMo-V2.5 的音频理解**并非**"连续 mel→conv→projector encoder"，而是先经一段**冻结的 RVQ 音频 tokenizer**得到**离散码**，再经小型 transformer 与投影。真实 `audio_config` + `processor_config`：

```
波形 → 24kHz mono → log-mel [1,T,128]                （MiMoAudioProcessor：n_mels128/nfft960/hop240/win960/sr24000）
 → 冻结 RVQ MiMoAudioTokenizer.encode               （权重在 checkpoint 的 audio_tokenizer/ 子目录）
       → 离散码 [audio_channels=20, T']             （为 20 通道，即使用全部 RVQ 级；不应沿用 V1 的 8）
 → per-channel speech_embeddings（20 通道，speech_vocab_size=1280，zeroemb_idx=1024）
 → input_local_transformer：6 层 FULL-attention Qwen2（dim 1024、heads 16、head_dim 64、intermediate 4096、
                                                    rope_theta 640000、partial_rotary 1.0、add_post_norm、group_size 4）
 → projection（projection_layers=2）→ (N, 4096)
 → scatter 到 audio_token_id=151669 位置
```
**无 RVQ 之外的生成路径**（不含 patch_decoder / vocoder）。`feature_attention_mask` 对 V2.5 **恒为 None**——音频 mel 是自描述变长 list，长度由 tokenizer 内部按 `audio_segment_size=6000` 分段，不靠 mask 推。

### 1.5 三模态 token 与位置
- token id（已核 `config.json`/`processor_config`）：`image=151655`、`video=151656`、`audio=151669`；vision_start/end=151652/151653、video_start/end=151670/151671、audio_start/end=151673/151674；`eos=151645`、`pad=151643`、`vocab_size=152576`、`tie_word_embeddings=false`。
- 合并方式：LM 先 embed 文本，再把 ViT / audio_encoder 的 4096 维输出 **scatter** 到各自占位符位置（三塔输出都是 4096，无需额外投影）。
- 位置编码：**1-D RoPE**（processor `use_video_timestamps=true` 强制 `rope_type=rope`），**不是 mrope**；视频按时间戳展开 1-D 位置。

### 1.6 权重布局（用于 weight-mapping）
单 checkpoint，按权重名**前缀路由**：`visual./vision_model.` → ViT；`audio_*` / `speech_embeddings` → audio_encoder；`model.*` → 文本 backbone。**RVQ tokenizer 权重单独放在 checkpoint 的 `audio_tokenizer/` 子目录**，需单独加载（且冻结）。

---

## 2. 2-stage 架构总览

沿用现有 `embedding → auto_regressive` 线性两段流水线。omni 的"多 encoder 汇入一个 LLM"靠**把三塔塞进 stage0 一个模型内部完成合并**实现（不需要跨 stage 的 DAG）：

```
请求(text+image+video+audio)  ──ZMQ──▶  GlobalScheduler.convert_omni_request
  （AutoProcessor 产出 MultimodalDataItem[]：pixel_values / pixel_values_videos / 音频 mel；1-D positions）
        │  Req(omni_inputs=mm_inputs(dict), pixel_values_images, pixel_values_videos, audio_features, grids…)
        ▼  current_stage 0 → 1（线性）
  ┌─ Stage0 = "embedding"（device_kind: tpu，已开 JIT+分桶）────────────────┐
  │  EmbedScheduler → EmbedModelWorker → EmbedModelRunner（prework 后模型无关）│
  │    _prepare_input(Req)（含形状分桶/补零）→ MiMoV2_5Embedding(**inputs)│
  │       visual(MiMoVL ViT) + audio_encoder(RVQ encode+local+proj) + text_embed │
  │       按 token_id scatter 三模态 → input_embeds[seq,4096]                 │
  │    反 pad → 写 omni_inputs["multimodal_embedding"]                        │
  └────────────────────────────────────────────────────────────────────────┘
        │  to_stage_reqs("auto_regressive")：TokenizedGenerateReqInput.mm_inputs = omni_inputs
        ▼
  ┌─ Stage1 = "auto_regressive"（核心 Scheduler，复用 mimo_v2_flash）─────────┐
  │  handle_generate_request: req.multimodal_embedding = mm_inputs[...]       │
  │  ScheduleBatch._merge_multimodal：按 dp_rank offset 排进 ForwardBatch     │
  │       ForwardBatch.input_embedding（MiMo-V2.5 无 mrope/无 deepstack）     │
  │  MiMoV2_5Generation（MoE LM）：input_embedding hook + 1-D RoPE → 逐 token  │
  └────────────────────────────────────────────────────────────────────────┘
        ▼ BatchTokenIDOut → MultimodalDetokenizer → HTTP
```
要点：**stage0 由单一模型完成三模态编码与合并、产出 `input_embeds`**；stage1 的 LLM 仅读取 `forward_batch.input_embedding`，不接触原始 pixel/mel。

---

## 3. 接线方案详解

### 3.1 Embed-stage 模型 `MiMoV2_5Embedding`（新建）
满足现有 `EmbedModelRunner` 的契约（prework 后已模型无关），即可无需改动接入 `EmbedScheduler/Worker/Runner`。参照 `qwen3_omni_thinker_embedding.py`：

- **构造** `__init__(self, config, *, mesh, dtype=jnp.bfloat16, rngs=None)`，持有三塔：
  - `self.visual` = MiMoVL ViT（§3 新建），调用 `visual(pixel_values.astype(dtype), grid_thw)`，返回 `{"pooler_output":[N,4096], "deepstack_features": None}`（V2.5 无 deepstack）。
  - `self.audio_encoder` = §1.4 的 RVQ-理解通路，调用 `audio_encoder(mels: list)` → `[N,4096]`（**不**用 `feature_attention_mask`）。
  - `self.text_embed_tokens` = `Embed(vocab=152576, hidden=4096, kernel_axes=("tensor",None), mesh=mesh)`。
  - 从 config 读 `image/video/audio_token_id`。
- **`__call__` 签名**（runner 按 keyword 调用，且 prework 已支持 per-model 输入准备）：embed 文本 → 各塔（present 才跑）→ 按 **token_id** 建 mask（含"占位符数==特征行数"断言）→ `at[mask].set(ravel(feats))` 三模态 scatter。
- **合并按 token_id、不 clamp**：被 embed 的 `input_ids` 是原始 token_id（在词表内）；`pad_value` 只活在 `cache_input_ids`（radix cache），不进 embed。各模型自管 merge，不抽共享引擎。
- **结构化输出**（prework 问题 2 已支持）：返回 `EmbedOutput(input_embeds, deepstack=None, pos_mask=None)`，runner 按字段取。

### 3.2 AR stage（`auto_regressive`）模型（复用 mimo_v2_flash）

> 命名说明：stage 名是 **`auto_regressive`（AR）**；模型**类名**用 `*Generation` 后缀是沿用本仓既有约定（`Qwen2_5_VL_Generation`、`Qwen3OmniMoeThinkerTextForConditionalGeneration`），即 HF 风格"用于生成的因果 LM"类名，**不是 stage 名**。
复用 `models/mimo_v2_flash.py` 的 MoE 解码器，按生成模型契约接三个 hook（参照 `qwen3_omni_thinker.py`，hook 本身无需改）：
- **input_embedding hook**：`hidden = forward_batch.input_embedding if not None else embed_tokens(input_ids)`——stage0 合并结果由此进入 LM。
- **位置**：**1-D RoPE**，走 `forward_batch.positions`（**不建 MRotaryEmbedding、不产 mrope_positions**）。
- **deepstack**：MiMo-V2.5 **无**；`apply_for_deepstack=False`，逐层注入分支 no-op。
- **MoE/量化**：按 §1.2 对齐——256 experts top-8、hybrid 9Full+39SWA（KV GA=4/SWA=8）、`attention_value_scale=0.707`、`fused_qkv`、sink-bias-仅-SWA、`partial_rotary=0.334`、FP8 `[128,128]`（o_proj 排除）、3 层 MTP。`ep_size` 须整除 256 与设备总数（见 §6）。

### 3.3 Stage 配置（新建 `models/static_configs/mimo_v2.5_stage_config.yaml`）
```yaml
model_arch: MiMo-V2.5
stage_args:
 - stage_id: 0
   runtime: { num_tpus: <N>, device_kind: tpu, max_batch_size: 1 }   # 必须 tpu（prework 问题2：重型 encoder + JIT + 分桶）
   scheduler: embedding
   model_class: MiMoV2_5Embedding
   final_output: false
 - stage_id: 1
   runtime: { num_tpus: <M>, max_batch_size: 1 }
   scheduler: auto_regressive
   model_class: MiMoV2_5Generation
   precompile_params: { input_embedding: True, deepstack_visual_embedding: False, mrope: False }  # 无 deepstack、1-D rope
   final_output: true
```
并让注册（prework 问题3 的声明式注册）能按 MiMo-V2.5 的 model_path 解析到此 yaml + 两个 model_class + config。

### 3.4 请求流与 Req（基本复用现有 omni 管线）
- 入口：`GenerateOmniReqInput`（已含 `audio_data`）→ `convert_omni_request`（已拆 image/video/audio mm_items、拼 `req.audio_features/pixel_values_*`）→ Req。**无改动**。
- 预处理：`multimodal_tokenizer._tokenize_one_request` 走 **`AutoProcessor` 分支**（**不要**落到 `is_mimo_audio` 的旧 `MiMoAudioProcessor` 分支——那是离散码生成路径）；audio key 为 `audio_features`/`input_features`，`feature_attention_mask` 恒 None。
- 翻译：`to_stage_reqs("auto_regressive")` 把 `omni_inputs`（含 stage0 写入的 `multimodal_embedding`）经 `mm_inputs` 透传给核心。**无改动**。
- config 装载：经 prework 问题3 的模型无关契约——AR stage 用顶层 config、embed stage 读 `config.vision_config`/`config.audio_config`（不再 `.thinker_config`）。

---

## 4. 三模态具体路径

### 4.1 Image
- 预处理 Qwen2.5-VL 风格（`Qwen2VLImageProcessor`，patch16、merge_size 2、min/max_pixels 8192/8388608）→ `pixel_values` + `image_grid_thw`。
- `self.visual`（新建 `mimo_v2_5/vision_mimovl.py`）：相对 Qwen2.5-VL ViT 的差异 = **行/列 1-D 窗口注意力**（按 `vit_window_attn_types` 逐 block 选 full/行/列）、`fullatt_block_indexes=[0,9,18,27]`、window RoPE、**attention sink**（`use_sink`）；head_dim 40、out_hidden 4096（自带 merger）。
- scatter 到 `image_token_id`；位置用 1-D（processor 按 grid 展开）。

### 4.2 Video
- 复用同一 MiMoVL ViT，输入 `pixel_values_videos` + `video_grid_thw`（`temporal_patch_size=2`、`video_tokens_per_second=2`、`fps=1.0`、`max_frames=3600`）。scatter 到 `video_token_id`；位置 1-D（`use_video_timestamps=true`，按时间戳展开）。
- 视频内**交错音轨**（`encode_video_audio`，`video_audio_interleave_length=0.0` 已配置）→ 第一轮**只支持静音/忽略音轨的 basic 视频**；交错音轨与"独立 audio + 带音轨视频同请求"延后（上游对后者本身 NotImplemented）。

### 4.3 Audio（展开）
**端到端（全在 stage0 内，详见 §1.4）**：mel → 冻结 RVQ `MiMoAudioTokenizer.encode` → 离散码 `[20, T']` → 20 通道 `speech_embeddings`（vocab 1280）→ 6 层 full-attn `input_local_transformer`（dim 1024/heads 16/head_dim 64）→ `projection`(2 层) → `[N,4096]` → scatter 到 `audio_token_id`。

**尽量复用现有 jax `mimo_audio` 组件**（`multimodal/models/mimo_audio/*`）：

| V2.5 omni audio 组件 | 复用来源 | 动作 |
|---|---|---|
| RVQ tokenizer（mel→codes，**冻结**） | `mimo_audio_tokenizer.py:AudioEncoder`(`:580-721`) + RVQ(`:106-167`) 的 **encode 半部** | 复用 encode、保留 RVQ；**不**用 decoder/vocoder；权重自 `audio_tokenizer/` 子目录加载 |
| per-channel `speech_embeddings`（**20 通道**，vocab 1280） | `mimo_audio_backbone.py` 的 speech_embeddings（V1 为 8，**需扩到 20**） | 复用结构，通道/词表按 V2.5 改 |
| `input_local_transformer`（6 层 full-attn，dim 1024、head_dim 64） | backbone 的 patch_encoder（bidirectional input_local） | 复用结构，去掉 group downcast / patch_decoder |
| `projection`(2 层) → 4096 | 新增 | 新增 |

- **mel 前端**：复用 `MiMoAudioProcessor`（`multimodal_tokenizer.py:140-250`），已确认与官方 torchaudio 管线逐参数等价，**无需改**（建议加注释 `fmax=12000==torchaudio f_max=None` + 数值回归测试锁定）。
- **接口**：`audio_encoder(mels: list)` → `[N,4096]`，**不**需 `feature_attention_mask`/audio grid；长度由 tokenizer 内部按 `audio_segment_size=6000` 分段。
- **权重映射**：复用 `mimo_audio_tokenizer_weights_mapping.py` 的 encoder/conv 部分（conv 轴 `(2,1,0)`）含 RVQ codebook；speech_embeddings/input_local 复用 backbone 映射；新增 projection 映射。
- **与生成子系统的关系**：本方案仅使用 RVQ tokenizer 的 **encode 半部**完成理解；**不使用** `audio_backbone_scheduler` / `MiMoAudioForCausalLM` / vocoder（该路径属独立 ASR/TTS checkpoint 的生成路径）。

---

## 5. 新增 / 改动文件清单

**新增（模型 / 配置）**
- `multimodal/models/mimo_v2_5/embedding.py` — `MiMoV2_5Embedding`（三塔 + 按 token_id 合并 + 结构化输出）。
- `multimodal/models/mimo_v2_5/vision_mimovl.py` — MiMoVL ViT（行/列 1-D 窗口 + sink + merger）。
- `multimodal/models/mimo_v2_5/audio_encoder.py` — 音频理解通路（RVQ encode + speech_embeddings(20) + input_local(6) + projection(2)）。
- `multimodal/models/mimo_v2_5/generation.py` — `MiMoV2_5Generation`（复用 mimo_v2_flash MoE + input_embedding/1-D rope hook）。
- `multimodal/models/mimo_v2_5/*_weights_mapping.py` — ViT / audio / embed 权重映射（含 `audio_tokenizer/` 子目录加载）。
- `multimodal/models/static_configs/mimo_v2.5_stage_config.yaml` — 2-stage 配置（embed stage `device_kind: tpu`）。
- `MiMoV2_5Config`（顶层 + `vision_config` + `audio_config`，承接 §1 数值）。

**改动（较少，多数已由 prework 完成）**
- 注册：在 prework 的声明式注册里加 MiMo-V2.5 一条（model_path → stage yaml + 两 model_class + config 工厂 + token-ids）。
- `multimodal_tokenizer.py`：仅当 MiMo-V2.5 的 HF processor 需 Qwen 式视频抽帧时，把其类名加入 `_is_qwen_video_processor`（否则无改动）。
- 复用项确认：`models/mimo_v2_flash.py` 按 §1.2 逐字段对齐 V2.5 config（hybrid 模式/KV 4-8/`attention_value_scale`/`fused_qkv` FP8/sink-SWA/`partial_rotary`/1-D rope/MTP）。

**明确无改动**（prework 已处理或本就支持）：入口路由 / `is_multimodal`（prework 问题1）；`EmbedModelRunner` 的 JIT/分桶/结构化输出/模型无关 config（prework 问题2/3）；`io_struct` / `convert_omni_request` / `to_stage_reqs` / 核心 `_merge_multimodal`（DP=1）。

---

## 6. 里程碑 / 测试 / 风险与开放问题

### 6.1 里程碑（A→B→C→{E,F}→G）
- **A. LLM backbone 对齐**：`mimo_v2_flash` 适配 V2.5 顶层 config（MoE/hybrid/KV4-8/sink-SWA/FP8/`partial_rotary`/1-D rope/MTP）。验证纯文本与 HF 数值对齐。
- **B. 配置 + 注册**：`MiMoV2_5Config`、声明式注册条目、AR 用顶层 config / embed 读 vision_config+audio_config。验证服务能识别为多模态并加载两 stage。
- **C. MiMoVL ViT**：`vision_mimovl.py` + 权重映射；验证单图特征 `[N,4096]` 与 HF `get_image_feature` 数值对齐 + 图文端到端（DP=1）。
- **E. Video**：复用 ViT + `video_grid_thw` + 1-D 时间戳位置；验证 basic（静音）视频理解端到端。
- **F. Audio**：`audio_encoder.py`（RVQ encode 冻结 + speech_embeddings(20) + input_local(6) + projection(2)）+ `audio_tokenizer/` 子目录加载 + mel parity；验证音频特征对齐 + 音频理解端到端。
- **G. 三模态合一**：图+视频+音频混合输入，经 stage0 合并 → stage1 生成端到端。
- 依赖：A→B→C→{E,F 可并行}→G。

### 6.2 测试
- **逐塔数值对齐**：text / MiMoVL ViT / audio（mel→codes→input_local→proj）各与 HF/sglang 参考输出对比（bf16/fp8 容差）。
- **合并正确性**：每模态"占位符数 == 特征行数"断言；多图/多段音频各自 `pad_value`（radix cache 可区分）。
- **mel parity**：jax `MiMoAudioProcessor` vs torchaudio 参考（容差 ~1e-4）。
- **端到端 golden**：图文 / 视频 / 音频理解各一组，纳入 `test/srt`。

### 6.3 风险
- **音频架构需特别注意**：为"冻结 RVQ encode → 离散码 → 小型 transformer"，而非连续 encoder；复用现有 mimo_audio 组件时须将 `speech_embeddings` 由 8 通道**扩至 20**、`input_local` head_dim 取 64、projection 取 2 层。
- **MiMoVL 窗口注意力**为 JAX 无先例的行/列 1-D 窗口 + sink，需新写并做数值对齐。
- **形状分桶覆盖度**（prework 问题2）：桶集需覆盖真实图/音长度分布，否则补零浪费或编译变体过多。
- **backbone 字段易错项**：KV heads GA=4/SWA=8（model card 标注相反）、视觉 head_dim 40（model card 记为 64）——以 config 为准。

### 6.4 开放问题（须以真实 checkpoint 复核）
- `audio_tokenizer/` 子目录的 RVQ **内部** config（`num_quantizers`（应=20）、`encoder_layers`/heads/`d_model`、`hybrid_attention`/`swa_per_block`、`codebook_size`）受 gating，未直读。
- MoE `ep_size` 最终取值（须整除 256 且整除设备总数 `data×tensor`）。
- MiMo-V2.5 的 HF processor 是否需 Qwen 式视频抽帧（决定 `_is_qwen_video_processor` 是否加条目）。

### 6.5 限制 / 后续任务项（超出本轮范围）
- **DP > 1**：现每 stage 自建网格 `data` 轴恒 1，与核心 `dp_size` 冲突；要开 DP>1 须做 prework 末尾所述的"网格 `data` 轴取 `dp_size` + 启动断言"最小修复。本轮 **DP=1**。
- **batch > 1**：stage0/stage1 `max_batch_size=1`，编码逐请求、无跨请求批；要提吞吐需引入真正的批处理契约。本轮 **batch=1**。
- 音频/语音**生成**（ASR/TTS 独立 checkpoint 的 RVQ 生成路径、vocoder）：不在 omni 理解范围，另行立项。

---

### 附录
- 本文自洽、以源码 + MiMo-V2.5 真实 config 为准，不援引其它设计文档编号；前置依赖见同目录 prework 文档。
- 结构数值取自真实 `config.json`（ModelScope 镜像）+ sglang 上游；`audio_tokenizer/` 内部细节仍 gated（§6.4）。
- `文件:行号` 为审读时所见，实施前请复核。
