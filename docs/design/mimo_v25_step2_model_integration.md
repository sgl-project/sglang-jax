# MiMo-V2.5 接入 · 第二步：模型级接入方案

> **前提**：第一步 prework 的三个前置问题已解决——① 入口路由归一 + `is_multimodal` 由 model_type 推导；② 编码 stage 已开 JIT + 形状分桶 + 结构化输出 + 放 TPU；③ 编码 stage 执行器已是**模型无关契约**（config 装载不再写死 `thinker_config`、`_prepare_input` 支持 per-model、注册声明式）。本文在此基础上只讲"MiMo-V2.5 这个模型本身怎么接进来"。
>
> **架构前提**：沿用现有 **2-stage `embedding → auto_regressive` 流水线**（不动调度/DP 双栈大重构）。本轮范围 **DP=1、batch=1**；**DP>1 / batch>1 见 §6 限制**。
>
> **落地范围（V1，重要）**：本轮落地 **audio 理解 + 文本 + AR 生成的完整非视觉链路**：host codec + tokenizer 接线、stage0 `MiMoV2_5Embedding`（文本 + audio 塔）、`audio_encoder.py` / `weights_mapping.py` 模块化落地；**AR stage 直接复用 `MiMoV2ForCausalLM`**（MiMo-V2.5 的文本部分与之相同，无需 omni 专属 AR 包装类）。config 沿用 remote-code `AutoConfig`（与其它多模态模型一致，inline 读取，不新增 config 类）。**仅 Vision/Video 塔未实现**——`embedding.py` 对 `pixel_values/pixel_values_videos` 直接抛 `NotImplementedError`，`self.visual` 未构造；§1.3/§4.1/§4.2 的 ViT 内容为**设计描述（未实现）**，对应 `vision_mimovl.py` 见 §5「后续计划」。
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
| MTP | **omni checkpoint 不加载 MTP**（HF `_keys_to_ignore_on_load_unexpected` 丢弃 `model.mtp.*`，且无 MTP 模块实例化）；3 层 MTP 投机解码属 MiMo-V2-Flash 文本变体（`mimo_v2_nextn.py`），不在本 omni 理解 checkpoint 范围 |

> JAX 现有 `python/sgl_jax/srt/models/mimo_v2_{flash,pro,nextn}.py` 是这套 backbone 的 **text-only** 实现 → **stage1 的 LM 主要复用之**，按上表逐字段对齐即可（详见 §3.2）。

### 1.3 Vision = MiMoVL ViT（JAX 无对应，需新建）
`vision_model_type="mimovl"`，真实 `vision_config`：

| 项 | 值 |
|---|---|
| depth | 28；full-attn 层 `fullatt_block_indexes=[0,9,18,27]`（4 full + 24 窗口） |
| 窗口注意力 | `vit_window_attn_types`∈{-1,0,1} = full/**行 1-D 窗口**/**列 1-D 窗口**；`window_size=128`、`visual_token_window_size=64`；`use_sink=true` |
| 维度 | hidden 1280、`num_heads=32` / KV `num_key_value_heads=8`（GQA，派生 groups=32/8=4；config 另有未被实现读取的 `num_query_groups=4`）、**head_dim=64**（来自 `qk_channels` 默认；**不是** hidden/heads=1280/32=40，QKV 与 hidden_size 解耦） |
| patch | `patch_size=16`、`spatial_merge_size=2`、`temporal_patch_size=2` |
| 输出 | `out_hidden_size=4096`（== LM hidden；ViT 自带 merger projector，**不需额外 adapter**） |

预处理为 Qwen2.5-VL 风格（`processor_config`：`image_min/max_pixels=8192/8388608`、`merge_size=2`、`patch_size=16`）。

### 1.4 Audio 理解 = host 侧 RVQ tokenizer + JAX audio understanding tower
**需特别说明**：MiMo-V2.5 的音频理解**并非**"连续 mel→conv→projector encoder"，而是先经一段**冻结的 RVQ 音频 tokenizer**得到**离散码**，再经小型 transformer 与投影。首轮接入将这条链路拆成两段：

1. **host/server 预处理侧**：raw audio → 24kHz mono/log-mel → MiMo-V2.5 HF/torch `audio_tokenizer.encode` → `audio_codes`。
2. **stage0 JAX embed 侧**：只消费 `audio_codes`，执行 speech embeddings + input local transformer + projection + scatter。

这样做的原因是 V2.5 `audio_tokenizer/` 是独立 codec 子目录，含动态长度、RVQ argmin 与 tokenizer 专属 attention 细节；首轮不把它移植进 JAX embed 热路径，避免 JIT/bucketing/数值对齐风险。真实 `audio_config` + `processor_config` 对应流程：

```
host/server:
  波形 → 24kHz mono → log-mel [1,T,128]
       → 冻结 RVQ MiMo-V2.5 audio_tokenizer.encode
       → audio_codes [T',20] 或 [20,T']              （20 通道，即使用全部 RVQ 级；不沿用 V1 的 8）

stage0/JAX:
  audio_codes
       → per-channel speech_embeddings（20 通道，speech_vocab_size=1280，zeroemb_idx=1024）
       → 按 group_size=4 分组
       → input_local_transformer：6 层 FULL-attention Qwen2（dim 1024、heads 16、head_dim 64、intermediate 4096、
                                                          rope_theta 640000、partial_rotary 1.0、add_post_norm）
       → projection（projection_layers=2）→ (N, 4096)
       → scatter 到 audio_token_id=151669 位置
```
**无 RVQ 之外的生成路径**（不含 patch_decoder / vocoder）。`feature_attention_mask` 对 V2.5 的 JAX embed stage **不参与**；长度与 placeholder 数量由 `audio_codes.shape` 和 `group_size` 推导，而不是沿用 Qwen3-Omni 的 `audio_features + feature_attention_mask` contract。

### 1.5 三模态 token 与位置
- token id（数值已核真实 `config.json`）：`image=151655`、`video=151656`（**顶层** config）；`audio=151669`、video_start/end=151670/151671、audio_start/end=151673/151674（**仅在 `processor_config` 子块**，顶层不可直读，须 fallback `processor_config`，见 `embedding.py::_get_config_value`）；vision_start/end=151652/151653（顶层与 processor_config 都有）；`eos=151645`、`pad=151643`、`vocab_size=152576`、`tie_word_embeddings=false`。
- 合并方式：LM 先 embed 文本，再把 ViT / audio_encoder 的 4096 维输出 **scatter** 到各自占位符位置（三塔输出都是 4096，无需额外投影）。
- 位置编码：**1-D RoPE**（processor `use_video_timestamps=true` 强制 `rope_type=rope`），**不是 mrope**；视频按时间戳展开 1-D 位置。

### 1.6 权重布局（用于 weight-mapping）
单 checkpoint，按权重名**前缀路由**：`visual./vision_model.` → ViT；`audio_*` / `speech_embeddings` → JAX audio understanding tower；`model.*` → 文本 backbone。**RVQ tokenizer 权重单独放在 checkpoint 的 `audio_tokenizer/` 子目录**，由 host/server 侧 tokenizer encode 路径加载（且冻结），不作为 stage0 JAX embed 模型权重加载的一部分。

---

## 2. 2-stage 架构总览

沿用现有 `embedding → auto_regressive` 线性两段流水线。omni 的"多 encoder 汇入一个 LLM"靠**把三塔塞进 stage0 一个模型内部完成合并**实现（不需要跨 stage 的 DAG）：

```
请求(text+image+video+audio)  ──ZMQ──▶  MultimodalTokenizer / GlobalScheduler.convert_omni_request
  （AutoProcessor 产出 pixel_values / pixel_values_videos；MiMo-V2.5 host audio tokenizer 产出 audio payload；1-D positions）
        │  Req(omni_inputs=mm_inputs(dict), pixel_values_images, pixel_values_videos, audio_payload, audio_codes(stage0兼容), grids…)
        ▼  current_stage 0 → 1（线性）
  ┌─ Stage0 = "embedding"（device_kind: tpu，已开 JIT+分桶）────────────────┐
  │  EmbedScheduler → EmbedModelWorker → EmbedModelRunner（prework 后模型无关）│
  │    _prepare_input(Req)（含形状分桶/补零）→ MiMoV2_5Embedding(**inputs)│
  │       visual(MiMoVL ViT) + audio_encoder(codes→local+proj) + text_embed      │
  │       按 token_id scatter 三模态 → input_embeds[seq,4096]                 │
  │    反 pad → 写 omni_inputs["multimodal_embedding"]                        │
  └────────────────────────────────────────────────────────────────────────┘
        │  to_stage_reqs("auto_regressive")：TokenizedGenerateReqInput.mm_inputs = omni_inputs
        ▼
  ┌─ Stage1 = "auto_regressive"（核心 Scheduler，复用 mimo_v2_flash）─────────┐
  │  handle_generate_request: req.multimodal_embedding = mm_inputs[...]       │
  │  ScheduleBatch._merge_multimodal：按 dp_rank offset 排进 ForwardBatch     │
  │       ForwardBatch.input_embedding（MiMo-V2.5 无 mrope/无 deepstack）     │
  │  MiMoV2ForCausalLM / MiMoV2Flash（MoE LM）：input_embedding hook + 1-D RoPE│
  └────────────────────────────────────────────────────────────────────────┘
        ▼ BatchTokenIDOut → MultimodalDetokenizer → HTTP
```
要点：**stage0 由单一模型完成 image/video 编码、audio_codes 理解与合并，产出 `input_embeds`**；stage1 的 LLM 仅读取 `forward_batch.input_embedding`，不接触原始 pixel/audio/mel/codes。

---

## 3. 接线方案详解

### 3.1 Embed-stage 模型 `MiMoV2_5Embedding`（新建）
满足现有 `EmbedModelRunner` 的契约（prework 后已模型无关），即可无需改动接入 `EmbedScheduler/Worker/Runner`。参照 `qwen3_omni_thinker_embedding.py`：

- **构造** `__init__(self, config, *, mesh, dtype=jnp.bfloat16, rngs=None)`，持有三塔：
  - `self.visual` = MiMoVL ViT（§3 新建），调用 `visual(pixel_values.astype(dtype), grid_thw)`，返回 `{"pooler_output":[N,4096], "deepstack_features": None}`（V2.5 无 deepstack）。
  - `self.audio_encoder` = §1.4 的 JAX audio understanding tower，调用 `audio_encoder(audio_codes)` → `[N,4096]`（**不**接收 raw audio/mel，**不**用 `feature_attention_mask`）。
  - `self.text_embed_tokens` = `Embed(vocab=152576, hidden=4096, kernel_axes=("tensor",None), mesh=mesh)`。
  - 从 config 读 `image/video/audio_token_id`。
- **`__call__` 签名**（runner 按 keyword 调用，且 prework 已支持 per-model 输入准备）：embed 文本 → 各塔（present 才跑）→ 按 **token_id** 建 mask → JIT-friendly scatter。`#placeholder == #feature rows`、`audio placeholder 必须有 payload/codes` 这类动态错误由 `MiMoV25AudioCodecProcessor` / `EmbedModelRunner._prepare_input` 在 JIT 前强校验；模型内 scatter 假设输入已满足合同。
- **合并按 token_id、不 clamp**：被 embed 的 `input_ids` 是原始 token_id（在词表内）；`pad_value` 只活在 `cache_input_ids`（radix cache），不进 embed。各模型自管 merge，不抽共享引擎。
- **结构化输出**（prework 问题 2 已支持）：返回 `EmbedOutput(input_embeds, deepstack=None, pos_mask=None)`，runner 按字段取。

### 3.2 AR stage（`auto_regressive`）模型（复用 mimo_v2_flash）

> 命名说明：stage 名是 **`auto_regressive`（AR）**；AR stage `model_class` 直接用 **`MiMoV2ForCausalLM`**——MiMo-V2.5 的文本/AR backbone 与之相同（input_embedding / 1-D rope / no-deepstack hooks 已在共享的 `MiMoV2Model` 内，omni 合并在 stage0 完成，AR 只读 `forward_batch.input_embedding`），**无需新增 `*Generation` 包装类**。"generation/生成" 指 causal LM 的用途，不代表 stage 名。
复用现有 MiMoV2 causal LM / flash MoE 解码器，按生成模型契约接三个 hook（参照 `qwen3_omni_thinker.py`，hook 本身无需改）：
- **input_embedding hook**：`hidden = forward_batch.input_embedding if not None else embed_tokens(input_ids)`——stage0 合并结果由此进入 LM。
- **位置**：**1-D RoPE**，走 `forward_batch.positions`（**不建 MRotaryEmbedding、不产 mrope_positions**）。
- **deepstack**：MiMo-V2.5 **无**；`apply_for_deepstack=False`，逐层注入分支 no-op。
- **MoE/量化**：按 §1.2 对齐——256 experts top-8、hybrid 9Full+39SWA（KV GA=4/SWA=8）、`attention_value_scale=0.707`、`fused_qkv`、sink-bias-仅-SWA、`partial_rotary=0.334`、FP8 `[128,128]`（o_proj 排除）。MTP 不适用本 omni checkpoint（见 §1.2）。`ep_size` 须整除 256 与设备总数（见 §6）。

### 3.3 Stage 配置（新建 `models/static_configs/mimo_v2_5_stage_config.yaml`）
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
   model_class: MiMoV2ForCausalLM   # MiMo-V2.5 文本部分即 MiMoV2ForCausalLM，无需 *Generation 包装
   precompile_params: { input_embedding: True, deepstack_visual_embedding: False, mrope: False }  # 无 deepstack、1-D rope
   final_output: true
```
并让注册（prework 问题3 的声明式注册）能按 MiMo-V2.5 的 model_path 解析到此 yaml + 两个 model_class + config。

### 3.4 请求流与 Req（payload-first）
- 入口：`GenerateOmniReqInput` 接收 raw `audio_data`。如果上游已预编码，也允许显式携带 `audio_codes`；但 raw audio 与 pre-tokenized codes 的优先级必须明确，首轮建议 raw audio 走 host tokenizer，pre-tokenized `audio_codes` 仅作为测试/内部入口。
- 预处理：`multimodal_tokenizer._tokenize_one_request` 仍走 MiMo-V2.5 的 HF `AutoProcessor` / chat template 分支生成 prompt 与 `<|audio_pad|>` 占位符；audio 部分在 host/server 侧调用 MiMo-V2.5 `audio_tokenizer.encode` 得到 `audio_codes`，再统一封装成 `MiMoV25AudioPayload`。若 `AutoProcessor` 已返回 `audio_codes`，也必须经过同一个 payload helper 规范化；若只返回 `audio_features/input_features`，必须继续调用 host-side tokenizer encode，不能把 mel 传入 JAX embed stage。
- Req：`TokenizedGenerateOmniReqInput.audio_payload` / `mm_inputs["mimo_v25_audio_payload"]` → `GlobalScheduler.convert_omni_request` → `Req.audio_payload`；scheduler 再从 payload 回填 `Req.audio_codes` 作为 stage0 兼容字段。payload 存在时 tokenized request 不再重复携带完整 `audio_codes`。MiMo-V2.5 不依赖 `req.audio_features`；`audio_feature_attention_mask` 只属于 Qwen3-Omni 等连续特征模型，不进入 MiMo-V2.5 分支。
- 翻译：`EmbedModelRunner.forward()` 消费 payload 后写入 `omni_inputs["multimodal_embedding"]` 并删除 `omni_inputs["mimo_v25_audio_payload"]`；`to_stage_reqs("auto_regressive")` 只把含 `multimodal_embedding` 的 `mm_inputs` 透传给核心 AR。
- config 装载：经 prework 问题3 的模型无关契约——AR stage 用顶层 config；embed stage 经 `MiMoV2_5Embedding.get_embed_model_config()` 拿到**顶层扁平 config**（V2.5 无 `thinker_config`），由模型 `__init__` 内部自行从 `config.audio_config`（及后续 `vision_config`）构造各塔配置。runner 不在自身层拆分 `vision_config/audio_config`。

---

## 4. 三模态具体路径

### 4.1 Image
- 预处理 Qwen2.5-VL 风格（`Qwen2VLImageProcessor`，patch16、merge_size 2、min/max_pixels 8192/8388608）→ `pixel_values` + `image_grid_thw`。
- `self.visual`（新建 `mimo_v2_5/vision_mimovl.py`）：相对 Qwen2.5-VL ViT 的差异 = **行/列 1-D 窗口注意力**（按 `vit_window_attn_types` 逐 block 选 full/行/列）、`fullatt_block_indexes=[0,9,18,27]`、window RoPE、**attention sink**（`use_sink`）；head_dim 64、out_hidden 4096（自带 merger）。
- scatter 到 `image_token_id`；位置用 1-D（processor 按 grid 展开）。

### 4.2 Video
- 复用同一 MiMoVL ViT，输入 `pixel_values_videos` + `video_grid_thw`（`temporal_patch_size=2`、`video_tokens_per_second=2`、`fps=1.0`、`max_frames=3600`）。scatter 到 `video_token_id`；位置 1-D（`use_video_timestamps=true`，按时间戳展开）。
- 视频内**交错音轨**（`encode_video_audio`，`video_audio_interleave_length=0.0` 已配置）→ 第一轮**只支持静音/忽略音轨的 basic 视频**；交错音轨与"独立 audio + 带音轨视频同请求"延后（上游对后者本身 NotImplemented）。

### 4.3 Audio（展开）
**端到端（host encode + stage0 understanding，详见 §1.4）**：

```
raw audio
  → host/server 侧 24kHz mono + mel 前端
  → MiMo-V2.5 HF/torch audio_tokenizer.encode
  → audio_codes [T',20] 或 [20,T']
  → stage0 JAX：20 通道 speech_embeddings（vocab 1280）
  → group_size=4
  → 6 层 full-attn input_local_transformer（dim 1024/heads 16/head_dim 64）
  → projection(2 层)
  → [N,4096]
  → scatter 到 audio_token_id
```

**首轮复用边界**：

| V2.5 omni audio 组件 | 复用来源 | 动作 |
|---|---|---|
| RVQ tokenizer（mel→codes，**冻结**） | MiMo-V2.5 HF/torch remote code + checkpoint `audio_tokenizer/` 子目录 | **host/server 侧调用**；不在 JAX embed stage 内复用旧 `mimo_audio` tokenizer；不加载 decoder/vocoder |
| per-channel `speech_embeddings`（**20 通道**，vocab 1280） | 可参考 `mimo_audio_backbone.py` 的 speech_embeddings 结构 | V0 已在 `mimo_v2_5/embedding.py::MiMoV25AudioUnderstandingEncoder` 内实现；后续可按文件边界拆出 `audio_encoder.py` |
| `input_local_transformer`（6 层 full-attn，dim 1024、head_dim 64） | 可复用底层 transformer/linear building blocks | 仅复用通用模块形态；不复用旧 `MiMoAudioForCausalLM` / patch decoder / audio generation 合同 |
| `projection`(2 层) → 4096 | 新增 | 新增 |

- **host-side tokenizer**：raw audio 到 `audio_codes` 必须由 MiMo-V2.5 自身 `audio_tokenizer/` 完成。若 HF `AutoProcessor` 不直接产出 `audio_codes`，需要在 `multimodal_tokenizer` 或独立 helper 中加载 tokenizer 子目录并调用 encode。
- **stage0 接口**：`audio_encoder(audio_codes)` → `[N,4096]`，**不**接收 mel/list，不需要 `feature_attention_mask`/audio grid；长度由 `audio_codes.shape` 与 `group_size=4` 推导。
- **shape 约定**：host 侧允许 `[T',20]` 或 `[20,T']`，进入模型前归一到 `[B,20,T']`；`#audio_pad == ceil(T' / group_size)`。非法 code id 需 host 侧 validate，不在模型内 clip。
- **权重映射**：stage0 只加载主 checkpoint 中 audio understanding tower 的 `speech_embeddings`、`input_local_transformer`、`projection` 权重；`audio_tokenizer/` 子目录由 host tokenizer encode 路径加载。
- **与生成子系统的关系**：本方案仅使用 MiMo-V2.5 RVQ tokenizer 的 **encode 输出**完成理解；**不使用** `audio_backbone_scheduler` / `MiMoAudioForCausalLM` / vocoder（该路径属独立 ASR/TTS checkpoint 的生成路径）。

#### 4.3.1 sglang-jax V0 模块映射

MiMo-V2.5 audio understanding 可以拆成这条链：

```text
audio waveform
  → mel
  → tokenizer hidden_states
  → audio_codes
  → grouped code embeddings
  → final audio_embeds
  → LLM input_embedding
```

V0 的设计原则是：**host 侧负责 codec encode 到 `audio_codes`，JAX stage0 负责 `audio_codes` 之后的 audio understanding tower 与 scatter**。

| 链路步骤 | 输入 → 输出 | sglang-jax 负责模块 | 设计要求 |
|---|---|---|---|
| 1. 音频解码/重采样 | file/base64/waveform → 24kHz mono waveform | `multimodal_tokenizer.py` 调用新增 `models/mimo_v2_5/audio_codec_processor.py` | host 侧执行；错误在请求预处理阶段返回，避免进入 stage0 后失败 |
| 2. mel 前端 | waveform → `mel [T_mel,128]` | 同上 | 参数对齐上游 MiMo-V2.5：`sample_rate=24000`、`n_fft=960`、`hop_length=240`、`win_length=960`、`n_mels=128`、`power=1.0`、`log(clamp(min=1e-7))` |
| 3. tokenizer encoder | `mel [T_mel,128]` → continuous hidden states `[T_code,D]` | host-side MiMo-V2.5 audio tokenizer helper | 加载 checkpoint 的 `audio_tokenizer/` 子目录；V0 不把 tokenizer encoder 放进 JAX stage0 |
| 4. RVQ 离散化 | hidden states → `audio_codes [T_code,20]` | host-side MiMo-V2.5 audio tokenizer helper | 调用 `audio_tokenizer.encoder.encode(..., return_codes_only=True)` 或等价实现；输出必须和上游 sglang/HF 对齐 |
| 5. payload 传递 | codes + span 信息 → `MiMoV25AudioPayload` | `io_struct.py` / `global_scheduler.py` / `schedule_batch.py` | 新增 V2.5 专用 payload；不要复用旧 MiMo-Audio 8-channel `Req.audio_codes` 语义 |
| 6. codes 分组 | `[T_code,20]` → `[N_group,4,20]` | `multimodal/models/mimo_v2_5/embedding.py::MiMoV25AudioUnderstandingEncoder._group_audio_codes` | `N_group=ceil(T_code/4)`；不足 4 的尾部重复最后一帧 code；不在模型内 clip 非法 code id |
| 7. code embedding | `[N_group,4,20]` → `[N_group,4,1024]` | `MiMoV25AudioUnderstandingEncoder.speech_embeddings` | 20 个 channel 分别查 `speech_embeddings[i]`，再逐 channel 相加 |
| 8. local transformer | `[N_group,4,1024]` → `[N_group,4,1024]` | `MiMoV25AudioUnderstandingEncoder.input_local_transformer` | 6 层 full-attn Qwen2-like local transformer；`hidden=1024`、`heads=16`、`head_dim=64`、`intermediate=4096`、非 causal |
| 9. projection | `[N_group,4,1024]` → `[N_group,4096]` | `MiMoV25AudioUnderstandingEncoder.proj_fc1/proj_fc2` | 先 reshape 成 `[N_group,4096]`，再走 2-layer projection，输出维度等于 LM hidden size |
| 10. scatter | `audio_embeds [N_group,4096]` → `input_embeds [seq,4096]` | `MiMoV2_5Embedding` | 按 `audio_token_id` / payload offsets 替换 `<audio_pad>` token；强校验 `sum(token_lengths)==audio_embeds.shape[0]` |
| 11. LLM 消费 | `input_embedding` → text generation | `MiMoV2ForCausalLM` / `mimo_v2_flash` | AR stage 只读 `forward_batch.input_embedding`，不直接接触 raw audio、mel 或 codes |

#### 4.3.2 `mimo_audio_backbone.MiMoAudioTransformer` 复用准入

本地 `mimo_v2_5/audio_encoder.py` 当前把 MiMo-V2.5 的 `input_local_transformer` 构造成：

```text
MiMoV25AudioUnderstandingEncoder
  └─ input_local_transformer = mimo_audio_backbone.MiMoAudioTransformer(...)
       use_qwen2_layers=False  # 当前默认
```

这个复用方向只能视为 **Qwen2-like local transformer 的实现基底**，不能视为已证明等价的直接复用。上游 HF / sglang 的 MiMo-V2.5 source of truth 是专门的 `MiMoAudioEncoder`：

```text
audio_codes
  → pad/group
  → speech_embeddings
  → Qwen2Model(input_local_config, inputs_embeds=..., is_causal=not input_full_attention)
  → projection
```

也就是说，上游并没有把独立 MiMo-Audio 纯音频 CausalLM / patch decoder 直接挂到 MiMo-V2.5 omni audio path；它只复用了 MiMo-Audio tokenizer / RVQ / utility 思路，并为 omni audio understanding 单独定义 `MiMoAudioEncoder`。

**等价性核对结论**

| 项 | HF / sglang MiMo-V2.5 `MiMoAudioEncoder` | sglang-jax 当前复用路径 | 判定 |
|---|---|---|---|
| transformer 类型 | `Qwen2Model(input_local_config)`，以 `inputs_embeds` 调用 | `MiMoAudioTransformer(use_qwen2_layers=False)`，stateless standard attention | 结构接近，但不是同一实现 |
| layer 结构 | Qwen2 decoder layer：RMSNorm → self-attn → RMSNorm → SwiGLU MLP → final norm | `MiMoAudioDecoderLayer` + `MiMoAudioMLP` + final RMSNorm | 基本匹配 |
| MLP | gate/up/down，无 bias，SiLU | gate/up/down，无 bias，SiLU | 匹配 |
| attention projection | q/k/v 有 bias，o 无 bias | q/k/v 由 `use_bias` 控制，o 无 bias | 当前应固定/assert 为匹配 |
| attention mask | `input_full_attention=true` → `is_causal=False` | 当前 `use_causal_mask=False` | 当前配置匹配 |
| RoPE | `rope_theta=640000`，`partial_rotary_factor=1.0` | 支持 `rope_theta`，未显式支持 partial factor | 仅在 partial=1.0 时匹配 |
| final norm | `add_post_norm=true` 时保留 final norm | 始终保留 final norm | 当前配置匹配；非当前配置不通用 |
| projection | `projection_layers=2` 时 Linear/GELU/Linear | 固定两层 `proj_fc1/GELU/proj_fc2` | 当前配置匹配；不支持 1 层 |
| 权重名 | `audio_encoder.input_local_transformer.layers.*` / `norm.weight` | 本地 target 路径同名 | 映射可落位，但仍需数值验证 |

**不建议使用的复用分支**：`MiMoAudioTransformer(use_qwen2_layers=True)` 不适合作为 MiMo-V2.5 stage0 local transformer 的默认路径。该分支接入本地 `Qwen2DecoderLayer` 的 serving/KV-cache 风格调用，更偏 AR decode；MiMo-V2.5 audio understanding 需要的是短序列、无 KV cache 的 `inputs_embeds` stateless full-attention 路径。若使用该分支，还需要额外改造 attention 调用，否则不能等价于 HF `Qwen2Model(inputs_embeds=...)`。

**允许复用的硬条件**

如果继续复用 `mimo_audio_backbone.MiMoAudioTransformer(use_qwen2_layers=False)`，必须把当前隐式假设改成显式准入条件：

1. `input_full_attention == true`，并固定 `use_causal_mask=False`；
2. `partial_rotary_factor == 1.0`，否则当前 RoPE 维度不等价；
3. `add_post_norm == true`，否则需要支持把 final norm 置为 Identity；
4. `projection_layers == 2`，否则当前两层 projection 结构不等价；
5. `input_local_hidden_dropout == 0.0`，否则本地无 dropout 语义；
6. q/k/v bias 语义必须与 checkpoint key 对齐；`o_proj` 仍为 no-bias；
7. `speech_vocab_size` / `speech_zeroemb_idx` 必须从 MiMo-V2.5 `audio_config` 读取并校验，不使用旧 MiMo-Audio 8-channel 默认。

**必须补的验证**

上线前必须增加真实 checkpoint parity test，至少覆盖：

```text
HF/sglang MiMoAudioEncoder(audio_codes)
  == JAX MiMoV25AudioUnderstandingEncoder(audio_codes)
```

测试应固定同一批随机/真实 `audio_codes [T,20]`，覆盖 `T % group_size == 0` 与 tail padding 两类长度，并分别检查：

- `speech_embeddings` 逐 channel sum 是否一致；
- `input_local_transformer` 输出是否一致；
- `projection` 输出是否一致；
- 最终 `audio_embeds [ceil(T/4),4096]` 是否在 bf16/fp32 容差内对齐。

只有该 parity 通过后，才能把当前复用表述为“等价实现”。在 parity 通过前，文档和代码注释应保持更保守的说法：**复用 `mimo_audio_backbone` 的 Qwen2-like building block，而不是复用纯 MiMo-Audio 模型本身**。

对应的数据结构边界：

```text
Host / processor side
  raw audio
    → mel
    → MiMo-V2.5 audio_tokenizer.encode
    → MiMoV25AudioPayload(
         raw code segments=[T_i,20]
         codes=concat(pad_to_multiple_of_4(raw segment i))  # stage0 input
         offsets=[audio_pad spans],
         token_lengths=[ceil(T_i/4) per audio segment],
         is_tokenized=True
      )

Stage0 embedding
  MiMoV2_5Embedding
    text_embed(input_ids)
    visual(...)
    audio_encoder(payload.codes) → audio_embeds [N_group,4096]
    scatter all modality embeds → multimodal_embedding

Stage1 AR
  forward_batch.input_embedding → MiMoV2 LM
```

把 audio 链路嵌进完整 server 调用后，V0 期望的端到端调用关系如下：

```text
OpenAI HTTP /v1/chat/completions
  → srt/entrypoints/openai/serving_chat.py
      OpenAIServingChat._preprocess_request(...)
        _process_messages(...)
          从 messages 中提取 text/image/video/audio_data
        if model_config.is_multimodal:
          GenerateOmniReqInput(
            prompt=...,
            image_data=...,
            video_data=...,
            audio_data=...,
            sampling_params=...
          )

  → multimodal request queue / MultimodalTokenizer
      multimodal_tokenizer.py::_tokenize_one_request(...)
        normalize image_data / video_data / audio_data / audio_codes
        load image/video/audio bytes or arrays
        call HF mm_processor(...)
          负责 chat template 后 input_ids、
          image/video pixel tensors、
          audio placeholder token 布局

        MiMo-V2.5 V0 extra branch:
          MiMoV25AudioCodecProcessor.encode(audio_data)
            raw audio → waveform → mel → tokenizer hidden_states → raw audio_codes [T_i,20]
          build MiMoV25AudioPayload(
            codes=concat(each segment padded to group_size=4),
            token_lengths=[ceil(T_i/4) per segment],
            audio_token_id=...,
            source="host_codec"
          )
          validate:
            code channel == 20
            code id in [0,1280)
            #audio_pad tokens == sum(token_lengths)

      multimodal_tokenizer.py::_make_omni_tokenized_request(...)
        TokenizedGenerateOmniReqInput(
          input_ids=...,
          mm_inputs={
            ...,
            "mimo_v25_audio_payload": payload.to_transport_dict()
          },
          audio_payload=payload,   # Python 内存态 source of truth
          audio_codes=None         # payload 存在时不重复携带完整 codes
        )

  → multimodal/manager/global_scheduler.py
      GlobalScheduler.convert_omni_request(...)
        Req.input_ids = input.input_ids
        Req.omni_inputs = input.mm_inputs
        Req.audio_payload = normalize(input.audio_payload)
        Req.audio_codes = Req.audio_payload.codes  # stage0 兼容字段
        Req.audio_features = concat(audio mm_items)  # only for continuous-audio models
        Req.cache_input_ids = pad_input_tokens(...)

  → Stage 0 embedding
      embed_model_runner.py::_prepare_input(...)
        if MiMo-V2.5:
          reject audio_features-only request
          require host-side audio_codes
          validate placeholder count / channel / id range
          pass audio_codes into model input

      mimo_v2_5/embedding.py::MiMoV2_5Embedding(...)
        text_embed(input_ids)
        visual(pixel_values, grids)
        audio_encoder(audio_codes)
          [T_code,20] → [N_group,4,20] → [N_group,4096]
        scatter visual/audio embeds into input_embeds by placeholder token positions
        write omni_inputs["multimodal_embedding"]
        drop omni_inputs["mimo_v25_audio_payload"] before AR

  → Stage 1 auto_regressive
      schedule_batch.py::Req.to_stage_reqs("auto_regressive")
        TokenizedGenerateReqInput.mm_inputs = omni_inputs

      core scheduler / model runner
        mm_inputs["multimodal_embedding"] → ForwardBatch.input_embedding
        MiMoV2ForCausalLM / MiMoV2 LM consumes input_embedding
```

从进程/队列角度看，audio 字段需要跟随现有 omni request 走，而不是新开一条 audio request pipeline：

```text
client
  │
  │ OpenAI-compatible chat request
  ▼
server process
  serving_chat.py / shared HTTP app
    - 解析 messages
    - 把 audio content part 抽成 audio_data
    - 构造 GenerateOmniReqInput
    - 不做 mel，不加载 codec，不产生 embedding
  │
  │ GenerateOmniReqInput(audio_data/audio_codes/audio_payload)
  ▼
multimodal_tokenizer process
  MultimodalTokenizer.generate_request(...)
    _tokenize_one_request(...)
      - HF processor / chat template 生成 input_ids 与 <audio_pad> 锚点
      - MiMoV25AudioCodecProcessor 生成或规范化 audio_codes
      - 根据 token_lengths 扩展 <audio_pad> span
      - 生成 MiMoV25AudioPayload 并写入 mm_inputs
    _make_omni_tokenized_request(...)
      - 输出 TokenizedGenerateOmniReqInput
  │
  │ TokenizedGenerateOmniReqInput(input_ids, mm_inputs, audio_payload)
  ▼
global_scheduler process
  GlobalScheduler.convert_omni_request(...)
    - Req.omni_inputs = mm_inputs
    - Req.audio_payload = normalize(input.audio_payload)
    - Req.audio_codes = Req.audio_payload.codes
    - 不重新 decode/resample，不重新跑 codec
  │
  │ Req(stage0)
  ▼
embed stage process
  EmbedScheduler → EmbedModelWorker → EmbedModelRunner.forward(...)
    _prepare_input(...)
      - 从 Req.audio_payload 或 mm_inputs["mimo_v25_audio_payload"] 取 payload
      - MiMo-V2.5 分支拒绝 audio_features-only
      - 校验 placeholder count / shape / range
    MiMoV2_5Embedding(...)
      - audio_codes → audio_embeds
      - scatter 到 input_embeds
    forward(...)
      - mm_inputs["multimodal_embedding"] = input_embeds
      - 删除 mm_inputs["mimo_v25_audio_payload"]
  │
  │ Req(stage1, only multimodal_embedding remains)
  ▼
auto_regressive stage process
  ordinary scheduler/model runner
    - mm_inputs["multimodal_embedding"] → ForwardBatch.input_embedding
    - MiMo-V2.5 LM 继续普通 prefill/decode
```

这个调用链里每一层的边界可以概括为：

| 层级 | 看到什么 audio 数据 | 允许做什么 | 禁止做什么 |
|---|---|---|---|
| HTTP / OpenAI server | request message 里的 audio content、可选显式 codes/payload | 解析并放入 `GenerateOmniReqInput` | 不做 codec；不做 span 扩展；不调用 JAX 模型 |
| `MultimodalTokenizer` | raw audio、processor 输出、placeholder、预编码 codes/payload | 唯一负责 raw audio 进入 codec、placeholder span 扩展、payload 构造 | 不把 MiMo-V2.5 请求转给 `audio_backbone_scheduler` |
| `MiMoV25AudioCodecProcessor` | raw audio / mel / codes segment | `audio -> mel -> hidden_states -> audio_codes`，并做 codes/payload 规范化 | 不生成 audio embedding；不做 vocoder/decode |
| `GlobalScheduler` | tokenized request 的 `audio_payload` / transport dict | 归一化并透传到 `Req`，回填 stage0 兼容的 `Req.audio_codes` | 不重新执行 codec；不把 mel 当 V2.5 stage0 输入 |
| `Req` / stage batch | `omni_inputs`、`audio_payload`、`audio_codes` | 在 stage0 前保存合同；stage0 后只保留 `multimodal_embedding` 给 AR | 不让 AR 继续携带 raw audio/mel/payload |
| `EmbedModelRunner` | `input_ids` + payload/codes | JIT 前校验合同；把 `audio_codes` 传给 `MiMoV2_5Embedding`；forward 后清理 payload | 不在 runner 里实现 codec 或 audio tower 细节 |
| `MiMoV2_5Embedding` | `audio_codes [T_pad,20]` | code embedding、local transformer、projection、scatter | 不解析 raw audio；不修复非法 code id/span |
| AR stage | `multimodal_embedding` | 普通 LM prefill/decode | 不感知 audio_codes、mel、codec |

按运行时模块展开后，audio 链路不是一条独立 scheduler pipeline，而是嵌在现有 omni request 的预处理、stage0 embedding、AR 三段之间：

| 调用层 | 现有/新增模块 | audio 相关输入 | audio 相关输出 | 需要承担的职责 |
|---|---|---|---|---|
| OpenAI server | `srt/entrypoints/openai/serving_chat.py` 等 server 入口 | OpenAI message 中的 audio url/base64/waveform，或扩展请求里的预编码 `audio_codes` | `GenerateOmniReqInput.audio_data` / `audio_codes` / `audio_payload` | 只负责把请求里的音频材料抽取成 omni request 字段；不在 server 入口做 mel、codec 或 JAX embedding |
| request schema | `GenerateOmniReqInput` / `TokenizedGenerateOmniReqInput` | `audio_data`、显式 `audio_codes`、可选 `MiMoV25AudioPayload` | tokenized omni request 上的 `audio_payload` 与 `mm_inputs["mimo_v25_audio_payload"]` | 给 MiMo-V2.5 建一个独立 payload 合同，避免把 20-channel V2.5 codes 和旧 MiMo-Audio 8-channel `Req.audio_codes` 混成同一语义 |
| multimodal tokenizer worker | `multimodal/manager/multimodal_tokenizer.py::_tokenize_one_request` | text/image/video/audio 原始输入，或预编码 codes/payload | `input_ids`、image/video tensors、audio placeholder、`MiMoV25AudioPayload` | 这里是 raw audio 进入 codec 的主入口：先让 HF processor / chat template 生成文本与 `<audio_pad>` 锚点，再根据 codec 得到的 `token_lengths` 扩展 audio pad span |
| MiMo-V2.5 codec helper | `multimodal/models/mimo_v2_5/audio_codec_processor.py`（新增） | raw audio / mel / 预编码 segment codes | stage0-ready `codes=concat(pad(segment_i,4))`、`token_lengths`、`offsets` | 集中封装 `audio -> mel -> hidden_states -> audio_codes` 和 codes 规范化；多段 audio 必须逐段 pad 后 concat，禁止跨段 grouping |
| tokenizer span 校验 | `multimodal_tokenizer.py` + codec helper | 扩展后的 `input_ids` 与 payload `token_lengths` | 带 `offsets` 的 payload | 校验 `<audio_pad>` 连续 span 数量、每段 span length、code channel、code range；错误在 host 侧失败 |
| global scheduler | `multimodal/manager/global_scheduler.py::convert_omni_request` | `TokenizedGenerateOmniReqInput.mm_inputs/audio_payload/audio_codes` | `Req.omni_inputs`、`Req.audio_payload`、`Req.audio_codes` | 只做结构化透传与 fallback 规范化；payload 存在时从 payload 回填 stage0 兼容的 `Req.audio_codes`，不重新跑 codec，也不把 V2.5 audio 当连续 `audio_features` |
| batch request | `multimodal/manager/schedule_batch.py::Req` | scheduler 写入的 audio payload/codes | stage0 batch 可见的 audio payload/codes；AR batch 只保留 `multimodal_embedding` | 保存 MiMo-V2.5 payload 到 stage0；`EmbedModelRunner.forward()` 消费后删除 `mm_inputs["mimo_v25_audio_payload"]`，`to_stage_reqs("auto_regressive")` 时只需要把 `multimodal_embedding` 放进 AR 请求 |
| stage0 runner | `multimodal/model_executor/embed/embed_model_runner.py::_prepare_input` | `Req.input_ids`、`Req.omni_inputs`、`Req.audio_payload/codes` | `MiMoV2_5Embedding` 的模型输入 | 对 MiMo-V2.5 强制要求 `audio_codes` / payload；拒绝 mel-only；再次做 shape/span 前置校验，避免 JIT 内部才暴露动态错误 |
| stage0 model | `multimodal/models/mimo_v2_5/embedding.py` + audio encoder | `input_ids`、visual tensors、`audio_codes [T_pad,20]` | `omni_inputs["multimodal_embedding"]` / `forward_batch.input_embedding` | 执行 `audio_codes -> speech_embeddings(20) -> input_local_transformer -> projection -> audio_embeds`，并 scatter 到 `<audio_pad>` span |
| AR stage | `Req.to_stage_reqs("auto_regressive")`、普通 model runner、`MiMoV2ForCausalLM` | stage0 已完成的 `input_embedding` | text generation | AR 只消费 embedding；不接触 raw audio、mel、audio_codes 或 codec helper |

从调用关系看，V0 需要新增的是 **模型专属转换 helper 和模型专属 stage0 tower**，不是新增一套 server 或调度系统：

| 判断 | 模块 | 结论 |
|---|---|---|
| 需要新增 | `audio_codec_processor.py` | 需要。它是 `audio -> mel -> hidden_states -> audio_codes` 的 host-side 适配层，也负责预编码 codes 的统一规范化、padding、span metadata。没有这个 helper，codec 逻辑会散落在 tokenizer、scheduler 和模型输入准备中。 |
| 需要新增 | `MiMoV25AudioPayload` | 需要。它是跨 `multimodal_tokenizer -> GlobalScheduler -> Req -> EmbedModelRunner` 的稳定合同，保存 `codes/token_lengths/offsets/audio_token_id/group_size`。 |
| 需要新增 | `multimodal/models/mimo_v2_5` 的 JAX audio understanding tower | 需要。MiMo-V2.5 的 20-channel code embedding、4-frame grouping、local transformer、projection 不等同于已有 `mimo_audio` 的 8-channel tokenizer/backbone。 |
| 需要改造 | OpenAI server request parsing | 小改造。只需要把 audio 原始输入或预编码字段放进 `GenerateOmniReqInput`，不要把 codec 放进 server 层。 |
| 需要改造 | `multimodal_tokenizer.py` | 需要。它是唯一知道 chat template、placeholder、raw audio 和 processor 输出的地方，适合完成 codec 调用和 span 扩展。 |
| 需要改造 | `global_scheduler.py` / `schedule_batch.py` / `embed_model_runner.py` | 小改造。新增 payload 字段透传和 MiMo-V2.5 分支校验，不新增 audio 专用 scheduler。 |
| 不需要新增 | `audio_backbone_scheduler` 或独立 audio scheduler | 不需要。那是 `mimo_audio` 纯音频生成/理解路径的两段式专用调度；MiMo-V2.5 audio 最终只是 stage0 multimodal embedding 的一部分。 |
| 不需要新增 | AR model runner 专用 audio 分支 | 不需要。audio embedding 在 stage0 已经 scatter 到普通 `input_embedding`，AR stage 继续走现有 text generation 路径。 |

这里要特别区分几类关键数据：

| 数据 | 生产位置 | 消费位置 | MiMo-V2.5 V0 约束 |
|---|---|---|---|
| `audio_data` | OpenAI server 从 request message 提取 | `MultimodalTokenizer` | 只在 host 侧存在，不进入 JAX stage0 |
| `audio_features` / mel | HF processor 或 codec helper 中间结果 | 非 V2.5 连续音频模型可消费 | MiMo-V2.5 stage0 不接受 mel-only；若只拿到 `audio_features` 必须报错 |
| `audio_codes [T_pad,20]` | MiMo-V2.5 host-side codec encode，或请求显式传入后由 payload helper 规范化 | `MiMoV25AudioPayload.codes` → `GlobalScheduler` → `Req.audio_payload` / stage0 兼容的 `Req.audio_codes` → `EmbedModelRunner` → `MiMoV2_5AudioEncoder` | V2.5 audio understanding 的唯一 stage0 输入；跨层 source of truth 是 payload，`Req.audio_codes` 只是 scheduler 到 stage0 的兼容字段；多段 audio 必须每段先独立 pad 到 `group_size=4` 的倍数，再 concat，禁止跨段 grouping |
| `multimodal_embedding` | Stage0 `MiMoV2_5Embedding` scatter 后产物 | Stage1 AR `ForwardBatch.input_embedding` | AR stage 不关心 raw audio/mel/codes |

V0 需要新增或改造的模块边界：

| 模块 | 新增/改造 | 责任 |
|---|---|---|
| `multimodal/models/mimo_v2_5/audio_codec_processor.py`（建议新增） | 新增 | 封装 MiMo-V2.5 `audio_tokenizer/` 的 host-side encode：load tokenizer 权重、音频解码/重采样、mel、RVQ encode，输出 `[T,20]` codes；避免把 torch/HF codec 逻辑散落在 `multimodal_tokenizer.py`，也避免和 JAX 模型权重模块混在一起 |
| `MiMoV25AudioPayload`（建议在 helper 中定义，`io_struct.py` 引用） | 新增 | 携带 stage0-ready `codes`、逐段 `token_lengths`、`offsets`、`audio_token_id`、`num_channels=20`、`codebook_size=1280`、`group_size=4`、`source`；多音频时 payload helper 先按段 pad 再 concat，保证 stage0 连续 grouping 不跨段 |
| `multimodal_tokenizer.py` | 改造 | 识别 MiMo-V2.5 model/config；在 HF processor 建好文本和 placeholder 后调用 codec processor；把 codes 和 span metadata 统一封装进 `TokenizedGenerateOmniReqInput.audio_payload`，并在 `mm_inputs["mimo_v25_audio_payload"]` 写 transport dict |
| `global_scheduler.py::convert_omni_request` | 小改造 | normalize `TokenizedGenerateOmniReqInput.audio_payload` 为 `Req.audio_payload`，从 payload 回填 stage0 兼容的 `Req.audio_codes`；若 `mm_inputs` 为空则创建 dict 并写入 transport payload；确保 V2.5 不把 mel `audio_features` 当 stage0 输入 |
| `schedule_batch.py::Req` / `to_stage_reqs` | 小改造 | 保存 `Req.audio_payload` 和兼容 `Req.audio_codes` 给 stage0；stage0 forward 消费 payload 后删除 `mimo_v25_audio_payload`，AR stage 只传 `omni_inputs["multimodal_embedding"]`，不再传 raw audio/codes |
| `embed_model_runner.py::_prepare_input` | 改造/已有部分 hook | 对 MiMo-V2.5 强制要求 host-side `audio_codes`；校验 channel、id range、placeholder count；拒绝 `audio_features`-only |
| `multimodal/models/mimo_v2_5/embedding.py` | 新增/改造 | V0 内聚实现 text embedding、MiMo-V2.5 audio understanding tower、scatter 到统一 `input_embeds`，并写入 `multimodal_embedding` 供 AR stage 使用；后续可把 audio tower 拆成 `audio_encoder.py` |
| `srt/managers/schedule_batch.py` / `forward_batch_info.py` | 改造 | AR 侧把 `Req.multimodal_embedding` 合并为 `ModelWorkerBatch.input_embedding`，再由 `ForwardBatch.init_new(...)` cast/传给 `forward_batch.input_embedding`；补长度前置校验 |

因此需要新增的核心模块只有两类：**host-side codec processor** 和 **JAX stage0 audio understanding tower**。围绕这两类还需要少量 schema/plumbing 字段，但它们属于已有模块扩展，不是新系统：

| 类别 | 模块 | 是否新增 | 原因 |
|---|---|---|---|
| 必需新增 | `audio_codec_processor.py` | 是 | 把 MiMo-V2.5 专属 `audio_tokenizer/` 的 host-side encode 与 payload 规范化集中起来。 |
| 必需新增 | `MiMoV25AudioPayload` | 是 | 给 `multimodal_tokenizer -> GlobalScheduler -> Req -> EmbedModelRunner` 一个稳定、可校验、可序列化的 20-channel audio 合同。 |
| 必需新增/整合 | `multimodal/models/mimo_v2_5/embedding.py` | 是 | V0 已内聚实现 text/audio embedding 和统一 scatter，并输出 `multimodal_embedding`；ViT 与 audio tower 后续可按边界拆分。 |
| 已有模块扩展 | OpenAI protocol / message parsing | 否，新字段即可 | 只需要支持 audio content part、预编码 codes/payload 进入 `GenerateOmniReqInput`。 |
| 已有模块扩展 | `MultimodalTokenizer` | 否，新增 MiMo-V2.5 分支即可 | 它已有 omni tokenizer worker、HF processor、mm_inputs 构造能力；只补 codec 调用和 span 校验。 |
| 已有模块扩展 | `GlobalScheduler` / `Req` / `ScheduleBatch` / `ForwardBatch` | 否，扩展现有字段与 merge | 它们已有 omni request 透传和 stage 切换能力；需补 payload normalize、AR 前清理、`multimodal_embedding -> input_embedding` 合并与长度校验。 |
| 已有模块扩展 | `EmbedModelRunner` | 否，新增模型分支校验即可 | 它已有 stage0 embedding runner 合同；只补 MiMo-V2.5 payload/codes 校验和 forward 后 payload drop。 |
| 明确不新增 | 独立 MiMo-V2.5 audio scheduler | 不需要 | V2.5 audio 是 omni embedding 的一部分，不是纯音频生成 pipeline。 |
| 明确不新增 | 复用/扩展 `audio_backbone_scheduler` | 不需要 | 该 scheduler 服务旧 `mimo_audio` 的 `[1,9,T]` backbone 输入，语义与 V2.5 的 `[T,20] -> input_embedding` 不同。 |
| 明确不新增 | detokenizer / vocoder | 不需要 | MiMo-V2.5 这里做 audio understanding，只生成文本，不做 audio waveform 输出。 |
| 明确不新增 | AR model runner audio 分支 | 不需要 | stage0 已经把 audio 融进 `input_embedding`，AR 继续普通 LM 路径。 |

最终设计判断：**需要新增模型专属 codec helper、payload 合同、stage0 audio tower；不需要新增 server、route、scheduler、detokenizer、vocoder 或 AR audio 分支**。MiMo-V2.5 audio 的集成点是现有 omni serving 链路里的 tokenizer worker 和 embed stage，不是独立的音频服务链路。

实现应分两个切片推进，避免把“结构化 codes 合同”和“真实 raw audio codec encode”混在一起：

| 切片 | 范围 | 输入支持 | 退出条件 |
|---|---|---|---|
| Slice 1：codes/payload plumbing | 新增 `MiMoV25AudioPayload` 与 `MiMoV25AudioCodecProcessor.build_payload_from_codes`；`multimodal_tokenizer` 把显式 `audio_codes` 或 processor 返回的 `audio_codes` 规范化为逐段 `[T_i,20]`，每段独立 pad 到 `group_size=4` 的倍数后 concat 为 stage0-ready `payload.codes`，并写入 `TokenizedGenerateOmniReqInput.audio_payload` / `mm_inputs["mimo_v25_audio_payload"]`；tokenizer 会把模板中每段单个 `<audio_pad>` 扩展为 `token_lengths[i]` 个 pad，并根据扩展后的连续 run 写入 `payload.offsets`；`GlobalScheduler` 兜底从 payload 回填 `Req.audio_codes` | 预 tokenized `audio_codes`，支持单段 2D `[T,20]` / `[20,T]`、多段 list、3D batch `[B,T,20]` / `[B,20,T]`；HF processor 若直接返回 `audio_codes` 也可用 | placeholder 数量、span 数、每段 span length、channel、code range 可在 host/tokenizer 或 runner 前置失败；多段 audio 不允许跨段 grouping；若模板完全没有 audio pad 或已经生成错误的多-pad run，则直接报错 |
| Slice 2：raw audio codec encode | `MiMoV25AudioCodecProcessor.encode(audio_data)` 加载 checkpoint `audio_tokenizer/`，完成 audio decode/resample/mel/RVQ encode，并产出同一个 payload | 普通 OpenAI raw audio | `audio_codes` 与 sglang/HF remote code 对齐；多段 audio span metadata 与 `<audio_pad>` 数逐段一致 |

也就是说，当前代码层面的首个落地点不是把 HF/torch codec 动态加载塞进共享 `multimodal_tokenizer.py`，而是先把 **MiMo-V2.5 专属 payload 合同**稳定下来。`MiMoV25AudioCodecProcessor.encode` 补齐后，下游 stage0/AR 合同不再变化；raw OpenAI audio 端到端还必须同时满足 **chat template / processor 至少为每段 audio 插入一个 `<audio_pad>` 锚点**。MiMo-V2.5 官方模板默认每段 audio 只放一个 `<audio_pad>`，sglang-jax tokenizer 在 codec 得到 `token_lengths` 后把这个锚点扩成实际 pad 数；若模板完全没有 audio pad，或已生成错误的多-pad run，则应在 tokenizer 的 span 校验处失败，而不是 silent trim。

如果调用方直接传 `MiMoV25AudioPayload`，`payload.codes` 也必须满足 stage0-ready 合同：单段未 pad codes 可以由 helper 自动 pad；多段 payload 必须已经是“每段独立 pad 后 concat”的结果。多段 raw-concat payload 无法从 `token_lengths` 反推出每段原始边界，必须改为传 list/3D `audio_codes`，由 helper 按段 padding。

注意不要直接套现有 `mimo_audio` 纯音频 pipeline：它的链路是 `mel → JAX MiMoAudioTokenizer → [8,T] codes → [1,9,T] 专用 backbone input`；MiMo-V2.5 需要的是 `mel → tokenizer → [T,20] codes → [N_group,4096] audio_embeds → scatter 到普通 LLM input_embedding`。因此可以复用底层写法（embedding list、local transformer、projection），不能复用 8-channel `Req.audio_codes` 合同、`[1,9,T]` backbone 输入格式和 `audio_backbone_scheduler`。

#### 4.3.2 sglang PR #23811 的实际 audio 数据流对照

上游 sglang 的 day-0 实现（PR #23811）没有在 processor 阶段强制把 raw audio 转成 `audio_codes` 后再传给模型；它把普通 raw audio 保留为 mel feature list，`audio_codes` 在 `MiMoAudioEncoder.get_audio_feature()` 内部由冻结 `audio_tokenizer.encoder` 现场生成。完整链路如下：

```
HTTP / OpenAI multimodal request
  → MiMoV2Processor.process_mm_data_async(...)
  → MiMoProcessor.process(contents)
  → _process_audio_content(...)
      raw audio(str/bytes/tuple/np.ndarray)
        → preprocess_audio(...)
        → torchcodec AudioDecoder / tuple waveform
        → resample to 24000 Hz
        → mono waveform
        → torchaudio MelSpectrogram(
              sample_rate=24000, n_fft=960, hop_length=240,
              win_length=960, n_mels=128, power=1.0, center=True,
              f_min=0, f_max=None
          )
        → log(clamp(spec, min=1e-7)).T
        → audio_spec [T_mel,128]
        → audio_token_len =
             ceil(((T_mel + 3 - kernel_size + 2 - kernel_size) // stride_size + 1)
                  / audio_avg_pooler / group_size)
      → input_ids = audio_start + audio_token_len * audio_token_id + audio_end
      → MultimodalDataItem(modality=AUDIO, feature=audio_inputs, offsets=audio_token spans)
  → general_mm_embed_routine(...)
  → MiMoV2ForCausalLM.get_audio_feature(items)
  → MiMoAudioEncoder.get_audio_feature(items)
      feature list/tuple 展平为 all_mels
      → tokenize_audio_batch(all_mels, audio_tokenizer.encoder, segment_size=6000)
      → 每段 mel 按 6000 帧切段，拼成 input_features；input_lens_flat 记录每段长度
      → encode_batch(...)
          → group_by_length(..., max_length=256000)
          → audio_tokenizer_encoder.encode(..., return_codes_only=True)
          → RVQ codes_packed [C,total_T_code]
      → codes_packed.transpose(0,1) = codes [total_T_code,C]
      → 按每条 mel 的 code_lengths split 回 list[Tensor[T_i,C]]
      → process_audio(codecs)
          → 取前 audio_channels=20
          → T_i pad 到 group_size=4 的倍数，pad 值重复最后一帧 code
          → reshape [ceil(T_i/4),4,20]
      → cat 得到 audio_codes [N_group,4,20]
      → apply_speech_embeddings(audio_codes)
          → 对 20 个 channel 分别查 `speech_embeddings[i](audio_codes[:,:,i])`
          → 逐 channel 相加，得到 [N_group,4,1024]
      → input_local_transformer(inputs_embeds=..., is_causal=False)
          → [N_group,4,1024]
      → reshape [N_group,4096]
      → projection(2 layers) → audio_embeds [N_group,4096]
  → embed_mm_inputs 用 audio pad_value mask 把 audio_embeds scatter 进 text embedding
  → LM prefill/decode
```

两点容易踩坑：

- **普通 raw audio 和预 tokenized audio 是两种入口**。上游 `AudioInput` 允许 `torch.Tensor [T,C]` 作为已经 tokenized 的输入，此时 processor 的 `process_audio()` 会直接按 `[T,C] → [ceil(T/4),4,20]` 分组并计算占位符长度；但 raw audio 路径产出的是 mel，不是 codes，codes 仍在模型侧生成。
- **上游 placeholder 数量和最终 embedding 行数绑定在 group 后长度上**。`preprocess_audio()` 先按 tokenizer conv/stride/avg_pooler/group_size 估算 `audio_token_len`；模型侧 `tokenize_audio_batch()` 再按实际 tokenizer output length 和 group padding 产出 `N_group` 行 embedding。两者必须相等，否则 `embed_mm_inputs()` 的 mask/token 数和 embedding 行数会不匹配。

#### 4.3.3 对 sglang-jax 的接入选择

当前文档采用 **host-side codec encode + payload-first transport** 是一个 **JAX 首轮工程化选择**，不是上游 sglang 的原样数据边界。它把上游 `MiMoAudioEncoder.get_audio_feature()` 里的 `tokenize_audio_batch()` 前半段提前到 host/server 侧，以避免首版在 JAX stage0 内移植 RVQ tokenizer 的动态切段、argmin codebook 搜索和 tokenizer 专属 attention；但跨 `multimodal_tokenizer -> GlobalScheduler -> EmbedModelRunner` 的主合同不是裸 `audio_codes`，而是结构化 `MiMoV25AudioPayload`。

建议把方案分成两个明确阶段：

- **V0 / 首轮落地：host-side codec + payload-first**。`MultimodalTokenizer` 或专用 helper 负责 raw audio → mel → `audio_tokenizer.encoder.encode(return_codes_only=True)` → `audio_codes`，并把 codes、`token_lengths`、`offsets` 统一写入 MiMo-V2.5 专用 `MiMoV25AudioPayload`。Stage0 JAX 只实现 `process_audio(codecs)` 之后的链路：`[T,20] → [N_group,4,20] → speech_embeddings sum → input_local_transformer → projection → [N_group,4096] → scatter`。
- **V1 / 上游 parity：model-side tokenizer**。保留 `MultimodalDataItem.feature` 为 mel list/tuple，Stage0 模型或 stage0 前置 worker 内部实现/调用 `tokenize_audio_batch()`，使边界与 sglang PR #23811 一致。只有当 JAX RVQ tokenizer 的数值、shape bucket、compile cache 和性能都稳定后再切换。

V0 必须补齐的契约：

- `audio_codes` 必须被封装进 MiMo-V2.5 专用 `MiMoV25AudioPayload(codes, offsets, token_lengths, is_tokenized=True)`；不要复用旧 `Req.audio_codes` 的 MiMo-Audio 8-channel 生成语义。`Req.audio_codes` 在 V0 中只作为 scheduler 回填给 stage0 的兼容字段，不是 tokenizer -> scheduler 的主传输合同。
- host 侧统一 shape 到 `[T,20]`，stage0 再 pad/reshape 到 `[N_group,4,20]`；`token_lengths` 必须等于每段 `ceil(T/4)`，并与 `input_ids` 中每个 audio span 的 `<|audio_pad|>` 数量一致。
- 预 tokenized tensor 入口只用于测试/内部调试时，也必须经过同一个 `process_audio(codecs)` 逻辑；不要让 raw mel、`[T,20]` codes、`[N_group,4,20]` grouped codes 混用同一个裸数组字段。
- `audio_tokenizer/` 子目录权重由 host-side tokenizer loader 单独加载并冻结；stage0 权重映射只覆盖主 checkpoint 里的 `speech_embeddings`、`audio_input_local_transformer`/`input_local_transformer`、`audio_projection`/`projection`。
- Stage0 scatter 前做强校验：`sum(audio_span_lengths) == audio_embeds.shape[0]`；失败时打印每个 span 的 `[start,end]`、`codes_T`、`ceil(T/4)`、embedding 行数，而不是在模型里 silent trim/clip。

保守建议：第一轮可以按 V0 合入，但文档和代码注释必须明示“这是为了避开 JAX codec 移植风险而前移 tokenizer 的实现选择”；数值验收仍应以 sglang PR #23811 的 `MiMoAudioEncoder.get_audio_feature()` 输出为参考。

#### 4.3.4 三条路径的数据流对照

先澄清本文里的 **codec**：它不是泛指“音频模型”，而是 MiMo audio tokenizer 这套**音频离散化/还原子系统**。在 audio understanding 的输入侧，codec 的 **encode** 半部把连续音频特征变成 RVQ 离散码：

```text
raw waveform / mel spectrogram
  → audio tokenizer encoder
  → RVQ codebook search
  → audio_codes
```

如果是音频生成/TTS 路径，codec 还可能有 **decode/vocoder** 半部，把离散码或隐藏状态还原成 waveform。但 MiMo-V2.5 omni understanding 只需要 encode 产出的 `audio_codes`，后续的 `speech_embeddings → input_local_transformer → projection` 是 **audio understanding tower**，不是 codec 本身。

下面三张图只看 audio 侧边界，目的是区分“codec 在哪里跑”“codes 的语义是什么”“最后是否 scatter 到普通 LLM embedding 序列”。

**A. sglang 上游 MiMo-V2.5：model-side tokenizer，单模型内完成三模态合并**

```text
OpenAI request(audio_data + prompt)
  │
  ▼
MiMoV2Processor / MiMoProcessor
  raw audio
    → 24kHz mono waveform
    → log-mel [T_mel,128]
  prompt
    → <audio_start> + N_group * <audio_pad> + <audio_end>
  │
  ▼
MultimodalDataItem(AUDIO)
  feature = list[mel Tensor[T_mel,128]]
  offsets = audio_pad spans in input_ids
  │
  ▼
MiMoV2ForCausalLM.forward()
  general_mm_embed_routine(...)
    → MiMoV2ForCausalLM.get_audio_feature(items)
      → MiMoAudioEncoder.get_audio_feature(items)
        → audio_tokenizer.encoder.encode(return_codes_only=True)
          mel → RVQ codes [T_code,20]
        → process_audio(codes)
          [T_code,20] → [N_group,4,20]
        → 20-channel speech_embeddings sum
          [N_group,4,20] → [N_group,4,1024]
        → input_local_transformer
          [N_group,4,1024] → [N_group,4,1024]
        → reshape + projection
          [N_group,4096]
    → scatter audio_embeds into text input_embeds by pad_value mask
  │
  ▼
MiMoV2 LM prefill/decode
```

这个路径的关键点：processor 传的是 mel list，`audio_codes` 在模型 forward 内部生成；audio embedding 与 image/video embedding 一样，被 scatter 到普通 LLM 的 `input_embeds`。

**B. sglang-jax 已有 MiMo-Audio 纯音频：专用 codec stage + 专用 9-row backbone 输入**

```text
MiMo-Audio request(audio/text)
  │
  ▼
host audio preprocess
  raw audio → mel_input [B,T_mel,128] + mel_input_lens
  │
  ▼
Stage 0: audio_encoder
  scheduler = audio_encoder
  model = MiMoAudioTokenizer
    → JAX/JIT tokenizer.encode(...)
      mel_input → RVQ codes [8,T_code]
    → valid_len 之后填 speech_empty_ids
  │
  ▼
Req.output / Req.audio_codes
  │
  ▼
Req.to_stage_reqs("audio_backbone")
  _build_backbone_input()
    text row:
      text tokens / <empty> / -100 padding
    8 audio rows:
      audio_codes or speech_empty_ids
    result: input_ids [1,9,T_grouped]
  │
  ▼
Stage 1: audio_backbone
  scheduler = audio_backbone
  model = MiMoAudioForCausalLM
    → _prepare_input_embeds(input_ids [B,9,T])
      split text row + 8 speech rows
      8-channel speech_embeddings sum
      patch_encoder
      speech_group_downcast
      text_embeds + speech_grouped_embeds
    → audio backbone LM / patch_decode
```

这个路径的关键点：它确实已经把 codec 移植到 JAX，但它是**纯音频专用 pipeline**。codes 是旧 MiMo-Audio 的 8-channel 生成/理解合同，进入的是 `[1,9,T]` 专用 backbone 输入，不是 MiMo-V2.5 的 20-channel omni audio embedding，也不 scatter 到 `audio_token_id` 对应的普通 LLM 序列。

**C. sglang-jax 接入 MiMo-V2.5 V0：host-side codes-first + stage0 只做 understanding tower**

```text
Omni request(text + image/video + audio)
  │
  ▼
MultimodalTokenizer / host helper
  text/image/video:
    → input_ids + pixel_values + grids
  raw audio:
    → 24kHz mono waveform
    → log-mel [T_mel,128]
    → host-side MiMo-V2.5 audio_tokenizer.encoder.encode(return_codes_only=True)
      → audio_codes [T_code,20]
    → token_lengths = ceil(T_code / 4)
    → input_ids 中放 token_lengths 个 <audio_pad>
  │
  ▼
MiMoV25AudioPayload
  codes = [T_code,20] per audio span
  offsets = audio_pad spans
  token_lengths = ceil(T_code/4) per span
  │
  ▼
Stage 0: embedding
  model = MiMoV2_5Embedding
    text_embed(input_ids)
    visual(pixel_values, grids) → image/video embeds [N,4096]
    audio_encoder(audio_codes)
      process_audio(codes)
        [T_code,20] → [N_group,4,20]
      20-channel speech_embeddings sum
        [N_group,4,20] → [N_group,4,1024]
      input_local_transformer
      reshape + projection
        [N_group,4096]
    assert sum(token_lengths) == audio_embeds.shape[0]
    scatter image/video/audio embeds into input_embeds by token_id/spans
  │
  ▼
Stage 1: auto_regressive
  MiMoV2ForCausalLM / MiMoV2 backbone
    forward_batch.input_embedding → LM prefill/decode
```

这个路径的关键点：V0 不复用 MiMo-Audio 的 `[8,T]` / `[1,9,T]` 合同，也不把 raw mel 直接塞进 JAX embed stage。host 侧先把 raw audio 变成 MiMo-V2.5 的 20-channel `audio_codes`，stage0 只实现 codes 之后的 audio understanding tower，并把结果 scatter 到 MiMo-V2.5 的普通 LLM `input_embedding`。

三者的核心差异可以压缩成这张表：

| 路径 | codec 运行位置 | 中间 codes | stage/model 边界 | 最终合并方式 |
|---|---|---|---|---|
| sglang MiMo-V2.5 | `MiMoAudioEncoder.get_audio_feature()` 内部，PyTorch/CUDA | `[T,20]`，模型 forward 内部临时值 | processor 传 mel list；模型自己生成 audio embedding | `general_mm_embed_routine` scatter 到 LLM `input_embeds` |
| sglang-jax MiMo-Audio | 独立 `audio_encoder` stage，JAX/JIT `MiMoAudioTokenizer` | `[8,T]`，写入 `Req.audio_codes` | `audio_encoder → audio_backbone` 专用两段 | 拼成 `[1,9,T]` 专用输入，backbone 内部 text/audio 相加 |
| sglang-jax MiMo-V2.5 V0 | host/server helper，HF/torch 或等价 tokenizer | `[T,20]`，显式 payload | `host tokenizer → embedding stage → AR stage` | stage0 产出 `[N,4096]` 并 scatter 到 `input_embedding` |

#### 4.3.5 HF / sglang processor 调研结论：sglang-jax 应采用自研 MiMoV25Processor

本节补充外部调研结论，用来解释为什么 §4.3.6 的 2b 方向不是"重写 HF processor"，而是把 MiMo-V2.5 的 serving processor 边界显式收回到 sglang-jax 自己的模型接入层。

**HF MiMo-V2.5：提供配置与模型合同，但不是完整三模态 serving processor**

HF repo 里的 `preprocessor_config.json` 仍声明为 Qwen2.5-VL 风格：

```json
{
  "image_processor_type": "Qwen2VLImageProcessor",
  "processor_class": "Qwen2_5_VLProcessor",
  "patch_size": 16,
  "temporal_patch_size": 2,
  "merge_size": 2
}
```

这说明 HF 侧 processor 的主体仍偏视觉/文本；它不是一个覆盖 raw image + video + audio 的 MiMo-V2.5 omni processor。`tokenizer_config.json` 注册了 `<|image_pad|>`、`<|video_pad|>`、`<|audio_pad|>`、`<|mimo_audio_start|>` / `<|mimo_audio_end|>` 等特殊 token，chat template 能插入占位符，但这仍只是**文本模板和特殊 token 层**，不会把 raw audio 编成 MiMo-V2.5 需要的 20-channel RVQ codes。

HF `config.json.processor_config` 保存了 MiMo-V2.5 专属运行参数与 token id，例如 `audio_sampling_rate=24000`、`audio_n_mels=128`、`audio_hop_length=240`、`audio_group_size=4`、`audio_channels=20`、`audio_token_id=151669`、`audio_start_token_id=151673`、`audio_end_token_id=151674`、`video_token_id=151656`、`fps=1.0`、`max_frames=3600`、`use_video_timestamps=true` 等；这些参数应被 sglang-jax processor 读取和复用，但它们本身不是 processor 执行逻辑。

HF `modeling_mimo_v2.py` 的 forward 合同也印证了这个边界：模型 forward 接收 `pixel_values` / `image_grid_thw`、`video_pixel_values` / `video_grid_thw`、`audio_codes` 或 `audio_embeds`，然后在模型内部调用 `self.visual(...)`、`self.audio_encoder(...)` 并把结果按 `image_token_id` / `video_token_id` / `audio_token_id` 替换进 `inputs_embeds`。也就是说，HF remote code 明确支持**model-side multimodal merge**，但它没有替 serving 系统完成 OpenAI multimodal request 到这些 kwargs 的完整转换。

因此，对 sglang-jax 来说，HF 可复用的内容是：

- `config.json` / `processor_config`：token id、音频采样与分组参数、视频帧率/时间戳参数、视觉 patch 参数；
- `preprocessor_config.json`：Qwen2.5-VL 风格图像/视频预处理参数；
- `tokenizer_config.json` / chat template：特殊 token 与基础 prompt 模板；
- `audio_tokenizer/` 子目录与 remote code：raw audio/mel 到 `audio_codes [T,20]` 的参考实现与权重；
- `modeling_mimo_v2.py`：stage0 audio/vision embedding 与 scatter 的数值参考合同。

不可直接依赖 HF `AutoProcessor` 完成的内容是：

- OpenAI-compatible `image_url` / `video_url` / `audio_url` / base64 content 的统一解析、下载、错误处理；
- raw audio → 24kHz mono/log-mel → RVQ tokenizer → `audio_codes [T,20]`；
- 多段 audio 的 `token_lengths`、`offsets`、placeholder 展开与校验；
- video + audio 混合输入的时间戳、音频切段和 interleave 策略；
- 面向 JAX stage0 的 `mm_items` / kwargs / shape bucket 传输合同。

**sglang 上游：确实自己实现 MiMo-V2.5 processor，而不是只靠 HF AutoProcessor**

sglang 的 MiMo-V2.5 serving 路径更接近一个完整 runtime processor：

```text
OpenAI multimodal request
  → MiMoV2Processor / MiMoProcessor
      image/video: load/decode/resize/patch flatten/grid
      audio: decode/resample/log-mel + audio placeholder length
      prompt: 插入 image/video/audio pad spans
  → MultimodalDataItem(feature + offsets)
  → MiMoV2ForCausalLM.forward()
      visual(...) / audio_encoder(...)
      general_mm_embed_routine scatter 到 input_embeds
  → LLM decode
```

这里 processor 负责 request 解析、媒体加载、placeholder 规划、视觉预处理、视频抽帧和 audio mel 前端；模型侧再运行 visual encoder、audio tokenizer/encoder 与最终 scatter。注意：sglang 上游的 audio raw 路径默认把 mel list 传到模型侧，`audio_codes` 是 `MiMoAudioEncoder.get_audio_feature()` 内部的临时值；这与 sglang-jax V0 的"host-side codes-first"边界不同，但两者都说明一个事实：**需要一个 MiMo-V2.5 专用 processor 层来补齐 HF processor 没有覆盖的 serving 逻辑**。

**对 sglang-jax 的建议**

sglang-jax 应采用自研 `MiMoV25Processor`，并把它设计成"组合 HF 能力 + 补齐 MiMo-V2.5 serving 合同"的薄模型专用 processor，而不是在通用 `MultimodalTokenizer` 里堆 `is_mimo_v25` 特判：

```text
MiMoV25Processor
  ├─ text/template/token ids: 复用 HF tokenizer/chat template
  ├─ image/video: 复用 Qwen2_5_VLProcessor 参数与可复用逻辑
  ├─ audio: 调 MiMo-V2.5 audio_tokenizer/ 或等价 helper 生成 audio_codes [T,20]
  ├─ layout: 展开 <audio_pad>/<image_pad>/<video_pad>，生成 offsets/token_lengths/grid
  └─ output: input_ids + mm_items(IMAGE/VIDEO/AUDIO) + model_specific_data
```

这个 processor 的目标输出应对齐 sglang-jax stage contract，而不是模仿 HF 的对象形态：

- `input_ids` 已包含正确数量的 `<image_pad>` / `<video_pad>` / `<audio_pad>`；
- image/video item 携带 `pixel_values`、`grid_thw` 和 offsets；
- audio item 携带 `audio_codes [T,20]`、`token_lengths=ceil(T/4)`、`group_size=4`、offsets；
- 所有 placeholder 数量在 host 侧强校验，stage0 只做 defense-in-depth 断言；
- `GlobalScheduler` 只透传 `mm_items` 并编排 stage，不解释 audio/mel/codes 语义；
- `EmbedModelRunner._prepare_input` 共享组装 `mm_items → kwargs`；
- `MiMoV2_5Embedding` 解释 `audio_codes`，执行 codes 之后的 audio understanding tower 与 scatter；
- `auto_regressive` stage 只读 `forward_batch.input_embedding`。

这也是 §4.3.6 的核心动机：**把 MiMo-V2.5 相对 HF 的缺口收敛到一个自研 processor，把 runtime transport 收敛到 `mm_items` 单一真相源，把 JAX stage0/stage1 的边界保持清楚**。这样既贴近 sglang 上游"processor + model-side merge"的 serving 思路，又保留 sglang-jax 的 staged TPU 执行优势。

#### 4.3.6 目标重构（方向 2b + P1）：audio 作为一等离散模态，mm_items 单一真相源

> **状态**：设计已定、**待落地**。本节把 MiMo-V2.5 audio 从「tokenizer 内 `if is_mimo_v25` 特判 / adapter 钩子 / `mimo_v25_audio_payload` 侧信道」重构为「与 image/video 同构的一等模态」，并一并理清 scheduler 边界——**`mm_items` 作为唯一特征契约，scheduler 只透传 + 编排，`_prepare_input` 共享组装**。已确认接受「会改到所有多模态模型(Qwen2.5-VL/Qwen3-Omni 等)的 stage0 输入契约 + mrope 判定，且本机无 torch/transformers/jax、只能 py_compile + stub，真实回归留到具备依赖的环境」。

**动机**：baseline(接 MiMo-V2.5 前) 的模型接入契约是「HF `AutoProcessor` 产出全部模态特征 + 展开占位符，tokenizer 只把 `processor_out` 泛型搬运成 `mm_items`」。MiMo-V2.5 唯一的偏差是它的 HF processor 是 `Qwen2_5_VLProcessor`（纯视觉、无 audio）。V0/V1 把这个偏差用内联特判 + adapter + payload 侧信道补在通用 tokenizer 里，导致 `_tokenize_one_request` 难读。2b 的思路是**把偏差收回到「一个 processor」里**，让 tokenizer 回到 baseline 纯泛型。

**统一契约（关键）**：audio 走和 image/video **完全相同**的 `mm_items` 通路，不要 payload 侧信道。复用现有 `MultimodalDataItem`（它已自带 `offsets` + `model_specific_data` dict + `feature`/`precomputed_embeddings`，**无需改类**）：

```
audio 模态 = MultimodalDataItem(
    modality = AUDIO,
    feature  = audio_codes,           # [T_pad, 20] 离散 RVQ 码（替代连续 mel）
    offsets  = [(start, end), ...],   # 复用现有字段：逐段 placeholder span
    model_specific_data = {           # 复用现有 dict，不改类
        "token_lengths": [...], "group_size": 4, "codebook_sizes": [...],
    },
)
```
谁解释这个 item：**embed 模型自己**（MiMo-V2.5 知道自己的 AUDIO=codes，Qwen3-Omni 知道自己的 AUDIO=mel）。transport 只搬运，语义归模型——回到 baseline 哲学。

**数据流（2b）**
```
__init__：按 model_type 选 processor      ← 唯一模型触点（与现有 mimo-audio 的 MiMoAudioProcessor 同模式）
          MiMo-V2.5 → MiMoV25Processor(组合 Qwen2_5_VLProcessor + RVQ codec)
  ▼
_tokenize_one_request                     ← 纯 baseline 泛型；无 is_mimo_v25 / 无 adapter / 无 payload 侧信道
   processor_out = self.mm_processor(images, videos, audio, text)
        # processor_out: input_ids(已展开<audio_pad>) + pixel_values/grids + audio_codes(+meta)
   通用规则：processor_out 有 audio_codes → mm_items.append(AUDIO item: codes+meta)
   mrope：仅当 config 有 rope_scaling.mrope_section 才算（MiMo-V2.5=default → 天然跳过）
  ▼
mm_inputs{mm_items:[IMAGE,VIDEO,AUDIO]}
  ▼
GlobalScheduler：把 mm_inputs(含 mm_items) 原样带进 Req + 编排；不 flatten、不建 req.pixel_values_*/audio_features
  ▼
EmbedModelRunner._prepare_input（共享一处，所有 embed 模型共用）：
   从 mm_items 按模态组装 kwargs(pixel_values / audio_codes / grids)，不做模型特定逻辑
  ▼
MiMoV2_5Embedding：读 audio kwargs(=codes) → speech_emb → local → proj → scatter
   （Qwen 等：读 image/video kwargs → ViT；每个模型只写自己的 forward）
```

**scheduler 功能边界（本轮一并理清）**
- **现状冗余（已核）**：`convert_omni_request` 把 `mm_items` flatten 成 `req.pixel_values_images/audio_features/*_grid_thw`，executor 只读这些 flatten 字段；而 `omni_inputs["mm_items"]` 被带着走但**下游零读者**（grep 确认）——特征存了两份。
- **目标边界**：scheduler 只管 ① 请求生命周期 ② transport 透传（原样带 `mm_items`）③ stage 编排 / 跨 stage 中继（stage0 把 `multimodal_embedding` 写回 `omni_inputs` 供 AR 读）；**不做按模态 concat/flatten**（那是 model-facing 的特征整形）。
- **组装归一（回应"会不会每个模型组装一遍"）**：`mm_items → kwargs` 的组装放在**共享的 `EmbedModelRunner._prepare_input`（及 `vit_model_runner`，或抽一个 `assemble_mm_inputs(mm_items)` util）一处**，所有模型共用；**不是每个模型各写一遍**。模型只写自己的 forward(ViT / audio tower)——本来就有，P1 不新增。
- **单一真相源**：`mm_items` 成为唯一特征契约，删除 `req.pixel_values_*/audio_features` 这些 flatten 冗余字段。

**分层落地计划**

| 动作 | 文件 / 位置 | 内容 |
|---|---|---|
| 新增 | `models/mimo_v2_5/processor.py::MiMoV25Processor` | HF-processor 同形 `__call__(images,videos,audio,text,...)`；内部组合 `Qwen2_5_VLProcessor`(视觉+文本) + 复用 `audio_codec_processor`(mel→RVQ codes)；产 `audio_codes`、**展开 `<audio_pad>`**、**host 侧校验 placeholder 数==codes 行数**（review D3-6 的校验搬进 processor）；任意模态子集分支（text-only / audio-only / 组合）在此处理 |
| 新增 | tokenizer `__init__` processor 选择 | 小注册表按 model_type 选 processor（与 `is_mimo_audio → MiMoAudioProcessor` 同模式）；这是**唯一**模型触点 |
| 改（泛型） | `_tokenize_one_request` | 回到 baseline 单一泛型流 + 一条通用规则「`processor_out` 有 `audio_codes` → AUDIO mm_item(codes→feature、token_lengths/codebook→`model_specific_data`、span→`offsets`)」；删除所有 `is_mimo_v25`/`consumes_audio`/`uses_mrope`/adapter 调用 |
| 改（泛型，惠及所有模型） | mrope 判定 | 由「有 `vision_config.spatial_merge_size` 就算」改为「有 `rope_scaling.mrope_section` 才算」；MiMo-V2.5(1-D rope) 天然跳过，无需模型特判 |
| 改（边界） | `global_scheduler.convert_omni_request` | 只透传 `mm_inputs`(含 `mm_items`) + 编排；**删掉 `mm_items → req.pixel_values_*/audio_features` 的 flatten/concat**；删 `mimo_v25_audio_payload` 透传 + `Req.audio_payload` |
| 改（共享组装） | `EmbedModelRunner._prepare_input`（+ `vit_model_runner`，或抽 `assemble_mm_inputs(mm_items)` util） | 从 `mm_items` 按模态组装 kwargs(pixel_values / audio_codes / grids)，**一处共享、非每模型**；删 `_validate_mimo_v25_*` 全家 + `uses_mimo_v25_audio_contract` flag + payload normalize；连续-audio 的 `feature_attention_mask` 变换按「该 meta 是否存在」判定或下沉 Qwen3-Omni 模型 |
| 删 | `Req.pixel_values_images/pixel_values_videos/audio_features` 等 flatten 字段 | 冗余；改由 `mm_items` 单一真相源（`audio_features` 命名/拼接路径随之消失，见风险 4） |
| 删 | `manager/audio_host_adapter.py` + `models/mimo_v2_5/audio_host_adapter.py` | 整个 adapter 机制 |
| 删 | `io_struct` `audio_payload` 字段、`Req.audio_payload`、payload 侧信道 | `MiMoV25AudioPayload` 降级为 codec 内部中间结构（或并入 mm_item 构造），不再是跨层 transport 合同 |
| 不变 | `MiMoV2_5Embedding` | 改为从组装好的 audio kwargs(=codes) 取，forward 本体不动 |
| 不变 | `MultimodalDataItem` | 已支持 `offsets`+`model_specific_data`，无需改类 |

**收益**：tokenizer 回到纯 baseline；scheduler 变薄（只透传+编排）；删掉整个 adapter + payload 侧信道 + runner 校验全家 + 能力 flag + flatten 冗余字段；特征**单一真相源**(`mm_items`)；audio 与 image/video 真正同构。

**风险项 / 验证缺口（落地时必须盯）**

1. **（高）`MiMoV25Processor` 组合体本机不可端到端验证**：依赖 transformers(Qwen 处理器)+torch(codec encode)，本机无。缓解：把组合体内「纯 codes/占位符」逻辑（展开、校验、meta 计算）拆成 numpy 可测小函数；Qwen 调用层薄到只剩转发；真实回归留到具备依赖的环境。
2. **（高）P1 改 executor 输入契约，涉及所有多模态模型**：image/video 也从「executor 读 `req.pixel_values_*`」改成「`_prepare_input` 读 `mm_items`」→ 动 Qwen2.5-VL / Qwen3-Omni 的 stage0 入参路径 + `vit_model_runner`；本机不可端到端验证 → **真实回归面 = 全部多模态模型**(图/视频/音 各跑通且不回归)。这是 P1 相对原 2b 扩大的部分。
3. **（中高）mrope 配置化碰 Qwen3-Omni**：mrope 由「有 vision_config」改成「有 `rope_scaling.mrope_section`」会影响 Qwen 系判定；需真实环境回归 Qwen mrope 位置正确。
4. **（已消解，原"scheduler audio 语义混淆"）**：scheduler 不再 flatten/concat → `req.audio_features` 命名碰撞 / 多段跨 grouping / 丢元数据 三个子问题随 flatten 路径**整条消失**（codes 只作为 `mm_items` 里的 AUDIO item，元数据在 `model_specific_data`/`offsets`，由 `_prepare_input` 共享组装、embed 模型解释）。
5. **（中）host 校验归属迁移**：review D3-6「placeholder 数==codes 行数」硬校验从 runner 搬进 `MiMoV25Processor`，必须保留；runner 侧 defense-in-depth 减弱，建议 embed 模型内保留一条断言。
6. **（中）改动面广**：tokenizer + io_struct + schedule_batch + global_scheduler + embed_runner + vit_runner + 多个模型；本机仅 py_compile + stub 测，真实回归覆盖（MiMo-V2.5 三模态端到端 + 所有现有多模态模型不回归）是验收前置。

**与 2a / 保守版的取舍记录**：2a（processor 包装 + 保留 payload 侧信道）不碰 Qwen3-Omni、可保留 numpy 实测，但不消除模型特判。保守版（scheduler 仍 flatten，但 codes 走独立 `req.audio_codes` 字段 + drop 死的 mm_items）回归面小、不碰 image/video 契约。**已确认走 2b + P1（mm_items 单一真相源 + scheduler 只透传/编排 + `_prepare_input` 共享组装）**：终局最干净（scheduler 变薄、去双份冗余、audio 风险整条消失、audio 真正一等），代价是回归面扩到全部多模态模型且本机不可端到端验证。

#### 4.3.7 V0 → V1：stage0 内 model-side tokenizer 的前提、流程变化与重构冲击

前面 §4.3.6 的 2b/P1 是 **V0 codes-first** 的清理版：audio 作为 `mm_items` 一等模态，但 `feature` 是已经编码好的 `audio_codes [T,20]`。如果最终目标不是最小实现，而是更接近 sglang 上游的 **V1 / model-side tokenizer**，则 audio item 可以改为承载 mel，`mel → audio_codes` 放回 stage0 内部完成：

```text
V0:
  MiMoV25Processor / host helper
    raw audio → mel → audio_tokenizer.encode → audio_codes [T,20]
    token_lengths = ceil(T_code / 4)
    AUDIO mm_item.feature = audio_codes
  Stage0:
    audio_codes → speech_embeddings → input_local_transformer → projection → scatter

V1:
  MiMoV25Processor / host helper
    raw audio → mel [T_mel,128]
    length oracle: mel_lengths → token_lengths
    AUDIO mm_item.feature = mel
    AUDIO mm_item.model_specific_data = {mel_lengths, token_lengths, offsets, segment_size, group_size}
  Stage0:
    mel → JAX MiMo-V2.5 audio tokenizer → audio_codes [T,20]
    audio_codes → speech_embeddings → input_local_transformer → projection → scatter
```

**先修正一个容易误判的点**：scheduler 边界是否干净不取决于 `AUDIO.feature` 是 mel 还是 codes，而取决于 scheduler 是否只透传 `mm_items`、不解释 audio 语义。因此 V1 并不会因为 scheduler 流经 mel 而天然更脏；只要 `GlobalScheduler` 继续只搬运 `mm_items` 并编排 stage，边界仍然干净。V0/V1 的真正差异在 **processor 与 stage0 模型合同**。

**V1 必须先解决的前提问题**

1. **JAX/TPU 版 MiMo-V2.5 audio tokenizer**
   需要把 checkpoint `audio_tokenizer/` 子目录里的 tokenizer encoder、动态切段、`get_output_length`、RVQ/codebook search、`return_codes_only=True` 语义移到 JAX stage0。当前 V0 的 `MiMoV25AudioUnderstandingEncoder` 只实现 codes 之后的 `speech_embeddings → input_local_transformer → projection`，并显式拒绝 mel-only 输入；V1 要在它前面增加 `MiMoV25AudioTokenizerEncoder`，或合并成 `MiMoV25AudioEncoder(mel_or_codes)`。

2. **host 侧精确 placeholder length oracle**
   即使 `mel → codes` 在 stage0 做，`input_ids` 仍必须在 tokenizer 阶段展开正确数量的 `<audio_pad>`。MiMo-V2.5 的 code 行数只由 mel 时间长度和 tokenizer 下采样结构决定，RVQ codebook search 只决定 code id，不改变长度。因此 host 不必跑完整 tokenizer，但必须使用与 stage0 tokenizer **同源配置 / 同源函数**计算 token length：

   ```text
   mel_len_i
     → 按 segment_size=6000 切成 mel_len_seg_j
     → conv_out_len_j =
          ((mel_len_seg_j + 3 - kernel_size + 2 - kernel_size) // stride_size) + 1
     → if avg_pooler != 1:
          code_len_j = ceil(conv_out_len_j / avg_pooler)
     → code_len_i = sum(code_len_j)
     → token_length_i = ceil(code_len_i / group_size)
   ```

   这个函数应抽成纯函数，例如 `compute_mimo_v25_audio_token_lengths(mel_lengths, tokenizer_length_config)`，由 `MiMoV25Processor` 与 stage0 JAX tokenizer 共同使用或共同复刻同一个配置源。禁止在 host 和 stage0 各写一份容易漂移的常量公式。

3. **真实 mel length 与 bucket padding 分离**
   V1 stage0 为 JIT 形状分桶会把 mel pad 到 bucket 长度，但 length oracle 只能使用真实 `mel_lengths`，不能使用 `T_bucket`：

   ```text
   AUDIO feature: mel_padded [B, T_bucket, 128]
   meta:
     mel_lengths=[T_real_i]
     token_lengths=[ceil(code_len_i / 4)]
     offsets=[(start_i,end_i)]
   ```

4. **多段 audio 逐段计算，禁止跨段 grouping**
   每段 audio 对应独立 `<audio_pad>` span。V1 仍必须逐段算 `mel_len_i → code_len_i → token_length_i`，stage0 tokenizer 也要逐段生成 codes、逐段 pad 到 `group_size=4` 后再 concat embeddings；不能把多段 mel concat 后统一分组，否则 `offsets` 与 placeholder span 会漂。

5. **stage0 运行时一致性断言**
   V1 不能因为 host 有 length oracle 就省掉模型侧校验。stage0 tokenizer 生成 codes 后必须检查：

   ```text
   computed_code_lengths == expected_code_lengths_from_meta
   sum(token_lengths) == audio_embeds.shape[0]
   sum(audio_pad span lengths) == audio_embeds.shape[0]
   ```

   失败时打印每段 `mel_len/code_len/token_length/span/audio_embeds_rows`，禁止 silent trim/pad。

6. **数值与长度双 parity**
   V1 上线前至少需要三层 golden：

   ```text
   mel → audio_codes          对齐 HF/sglang
   audio_codes → audio_embeds 对齐 HF/sglang
   mel → audio_embeds         对齐 HF/sglang
   ```

   其中第一层除了 code id 数值，还要验证 `get_output_length`、segment split、avg_pool ceil、group padding 与上游完全一致。

**从 V0 切到 V1 后的模块影响**

| 模块 | V0 状态 | V1 变化 |
|---|---|---|
| `MiMoV25Processor` | raw audio → mel → host `audio_tokenizer.encode` → `audio_codes`；输出 codes item | raw audio → mel；使用 length oracle 展开 `<audio_pad>`；输出 mel item + `mel_lengths/token_lengths/offsets` |
| `audio_codec_processor.py` | 主路径为 host-side codec encode + codes/payload 规范化 | 降级为 audio frontend + length oracle + parity/debug helper；预编码 `audio_codes` 可作为内部测试入口保留 |
| `MultimodalDataItem` | `AUDIO.feature = audio_codes`，meta 标记 `is_codes=True` | `AUDIO.feature = mel`，meta 标记 MiMo-V2.5 mel contract，例如 `kind=mimo_v25_mel`、`mel_lengths`、`token_lengths`、`segment_size`、`group_size` |
| `mm_assembly.py` / `_prepare_input` | 从 `mm_items` 组装 `audio_codes` 给 embed 模型 | 需要把 MiMo-V2.5 mel 与 Qwen3-Omni continuous audio 区分开；传 `input_features/mel_lengths/token_lengths/offsets` 给 `MiMoV2_5Embedding` |
| `MiMoV2_5Embedding` | `audio_codes → audio_embeds → scatter` | `mel → audio_tokenizer → audio_codes → audio_embeds → scatter`；同时保留 codes debug path 可降低迁移风险 |
| `audio_encoder.py` | 只含 codes 后的 understanding tower | 拆成 `audio_tokenizer_encoder.py` + `audio_encoder.py`，或合并为完整 `MiMoV25AudioEncoder` |
| `weights_mapping.py` | 只覆盖主 checkpoint 的 `speech_embeddings/input_local/projection` | 新增 `audio_tokenizer/` 子目录 tokenizer encoder 与 RVQ codebook 权重映射 |
| `GlobalScheduler` | 只透传 `mm_items` | 理想情况下不变；只要不 flatten、不解释 mel/codes，就没有 scheduler 大重构 |
| AR stage | 只读 `forward_batch.input_embedding` | 不变 |

**是否会带来大量重构**

如果已经采用 `mm_items` 单一真相源，V1 不应造成 server / scheduler / AR 的大规模重构；`GlobalScheduler` 继续只搬运 `mm_items` 即可。真正的大工作量集中在 **stage0 模型侧**：

- 大量新增：JAX audio tokenizer encoder、RVQ/codebook search、tokenizer 权重映射、shape bucket、parity tests；
- 中等修改：processor 从 codes-first 改为 mel-first，`mm_assembly` / `EmbedModelRunner` 增加 MiMo-V2.5 mel contract；
- 可逐步删除：V0 的 payload / host codes-first 主路径可以先降级为 debug path，而不是一次性删除。

建议的迁移顺序：

```text
Step 1: stage0 同时支持 audio_codes 与 mel，默认仍走 V0 codes path
Step 2: shadow mode：host codes 与 stage0 mel→codes 同时计算，比较 code_lengths/code ids/audio_embeds
Step 3: 默认切到 mel path，保留 audio_codes 作为测试/内部入口
Step 4: 删除或降级 V0 host codec encode 主路径，只保留 pre-tokenized codes debug contract
```

这样 V1 更接近 sglang 上游 model-side tokenizer 和单模型内三模态合并；同时把重构冲击限制在 processor + stage0 audio tokenizer，而不是扩散到 scheduler/AR。

---


## 5. 新增 / 改动文件清单

**当前 V0 已新增 / 已落地**
- `multimodal/models/mimo_v2_5/audio_codec_processor.py` — host-side codec/payload helper；封装 raw audio/mel/预编码 codes → `MiMoV25AudioPayload`，集中做 20-channel RVQ ids 归一化（支持 per-quantizer codebook 校验）、逐段 padding、placeholder span metadata、`codes_layout` 契约。
- `multimodal/models/mimo_v2_5/embedding.py` — `MiMoV2_5Embedding`（text embedding + 按 token_id scatter + 结构化输出 + capability flag）；`MiMoV25AudioUnderstandingEncoder` 在此 re-export。
- `multimodal/models/mimo_v2_5/audio_encoder.py` — 20-channel audio understanding tower（`audio_codes → speech_embeddings(20) → input_local(6) → projection(2)`），从 embedding 拆出。
- `multimodal/models/mimo_v2_5/weights_mapping.py` — text/speech/input_local/projection 权重映射构建函数（不含 `audio_tokenizer/` codec 子目录、不含 ViT）。
- `multimodal/models/static_configs/mimo_v2_5_stage_config.yaml` — 2-stage 配置（embed stage `device_kind: tpu`；AR stage `precompile_params.input_embedding=True`、`model_class=MiMoV2ForCausalLM`，直接复用文本 backbone）。
- `multimodal/models/mimo_v2_5/__init__.py` — lazy 导出 `MiMoV2_5Embedding`/`MiMoV25AudioUnderstandingEncoder`。

**AR stage 复用既有类（不新增）**
- AR 用 `models/mimo_v2_pro.py::MiMoV2ForCausalLM`（融合-QKV 的 MiMo-V2 MoE LM）+ `mimo_v2_flash.py::MiMoV2Model` 已有的 input_embedding/1-D rope/no-deepstack hooks。MiMo-V2.5 文本部分无 omni 专属行为，**不新增 `MiMoV2_5Generation` 包装类**。

**后续计划新增 / 拆分（仅剩 vision）**
- `multimodal/models/mimo_v2_5/vision_mimovl.py` — MiMoVL ViT（行/列 1-D 窗口 + sink + merger）。**本轮唯一未落地模块。**
- ViT / vision 权重映射：待 `vision_mimovl.py` 落地时加入 `weights_mapping.py`。
- raw audio→codes 的 host RVQ encode 数值对齐（§6.4）：代码已写，待真实 checkpoint 验证。

**当前 V0 已改动（较少，多数沿用现有系统）**
- 注册：在 prework 的声明式注册里加 MiMo-V2.5 一条（model_path → stage yaml + 两 model_class + config 工厂 + token-ids）。
- `multimodal_tokenizer.py`：仅当 MiMo-V2.5 的 HF processor 需 Qwen 式视频抽帧时，把其类名加入 `_is_qwen_video_processor`（否则无改动）。
- `multimodal_tokenizer.py` 或新增 host-side helper：加载 MiMo-V2.5 `audio_tokenizer/` 子目录，完成 raw audio/mel → `audio_codes`，并把 codes 规范化为 `MiMoV25AudioPayload`；payload 存在时 tokenized request 不再重复携带完整 `audio_codes`。
- `io_struct.py` / `global_scheduler.py` / `schedule_batch.py`：新增并透传 MiMo-V2.5 专用 `audio_payload`；scheduler 从 payload 回填 stage0 兼容的 `Req.audio_codes`。该兼容字段语义为 MiMo-V2.5 20-channel RVQ ids，不等同于旧 MiMo-Audio 8-channel 生成 codes。
- `srt/managers/schedule_batch.py`：核心 AR scheduler 侧把 `Req.multimodal_embedding` 合并为 `ModelWorkerBatch.input_embedding`，并对 embedding rank/长度做前置校验。
- `srt/model_executor/forward_batch_info.py`：复用既有 `ForwardBatch.init_new(...)` 的 `input_embedding` device-array/cast 路径，新增本地测试固化字段不丢失。
- 复用项确认：`models/mimo_v2_flash.py` **已有** §1.2 的 V2.5 字段实现（hybrid 模式/KV 4-8/`attention_value_scale`/`fused_qkv` FP8/sink-SWA/`partial_rotary`/1-D rope），属 text-only 既有能力；本轮**仅新增 input_embedding hook**（按 forward-mode gate 注入 stage0 合并结果），未做额外 V2.5 字段对齐。MTP 不适用本 omni checkpoint。

**明确无改动 / 不新增**
- 不新增独立 MiMo-V2.5 audio scheduler、`audio_backbone_scheduler` 分支、detokenizer/vocoder、AR model runner audio 分支。

**已改动入口/执行层（勿误读为「无改动」）**
- `configs/model_config.py`：`is_multimodal` 改由新增的 `is_multimodal_model()` 判定（`model_type=="mimo_v2"` 且 `vision_config`/`audio_config` 非空才进 omni 路径），不是「沿用旧推导」。
- `entrypoints/openai/serving_chat.py`：多模态分支新增 `GenerateOmniReqInput` 路由；本轮对 `logprobs/top_logprobs` 显式报错而非静默丢弃（见 §6.3 / review D5-5）。
- `multimodal/entrypoint/http_server.py`：删除原 `/v1/chat/completions` 路由与 OpenAI prompt 抽取，收敛到核心 `serving_chat`。
- `EmbedModelRunner`：复用 JIT/分桶/结构化输出框架，但**新增了 MiMo-V2.5 专属 host-side 校验分支**（payload/codes 规范化与校验、kwargs 按模型签名过滤、forward 后 payload 清理）。runner 通过模型类的 `requires_mimo_v25_audio_contract` capability flag 触发这些分支，保持对其它模型无侵入（不再按类名字符串判定）。

---

## 6. 里程碑 / 测试 / 风险与开放问题

### 6.1 里程碑（A→B→C→{E,F}→G）
- **A. LLM backbone 对齐**：`mimo_v2_flash` 已具备 V2.5 顶层 config 的字段实现（MoE/hybrid/KV4-8/sink-SWA/FP8/`partial_rotary`/1-D rope）；本轮工作是**新增 input_embedding hook** 让 stage0 合并结果进入 LM。MTP 不适用本 checkpoint（见 §1.2），不作为对齐项。验证纯文本与 HF 数值对齐。
- **B. 配置 + 注册**：声明式注册条目、AR 用顶层 config / embed 用顶层扁平 config（模型内部自取 `audio_config`/后续 `vision_config`）；config 经 remote-code `AutoConfig` 装载，不新增专用 config 类。验证服务能识别为多模态并加载两 stage。
- **C. MiMoVL ViT**：`vision_mimovl.py` + 权重映射；验证单图特征 `[N,4096]` 与 HF `get_image_feature` 数值对齐 + 图文端到端（DP=1）。
- **E. Video**：复用 ViT + `video_grid_thw` + 1-D 时间戳位置；验证 basic（静音）视频理解端到端。
- **F. Audio**：host-side MiMo-V2.5 `audio_tokenizer.encode` 产出 `audio_codes`；V0 的 `embedding.py::MiMoV25AudioUnderstandingEncoder` 实现 `audio_codes → speech_embeddings(20) → input_local(6) → projection(2)`；验证 codes、audio embedding 与端到端音频理解。
- **G. 三模态合一**：图+视频+音频混合输入，经 stage0 合并 → stage1 生成端到端。
- 依赖：A→B→C→{E,F 可并行}→G。

### 6.2 测试
- **逐塔数值对齐**：text / MiMoVL ViT / audio（host tokenizer codes、codes→input_local→proj）各与 HF/sglang 参考输出对比（bf16/fp8 容差）。
- **合并正确性**：每模态"占位符数 == 特征行数"断言；多图/多段音频各自 `pad_value`（radix cache 可区分）。
- **audio tokenizer parity**：host-side `audio_tokenizer.encode` 的 `audio_codes` 与 HF remote code 对齐；若保留本地 mel 前端，另做 mel parity（容差 ~1e-4）。
- **端到端 golden**：图文 / 视频 / 音频理解各一组，纳入 `test/srt`。

### 6.3 风险
- **音频架构需特别注意**：为"host-side 冻结 RVQ encode → 离散码 → JAX 小型 transformer"，而非连续 encoder；stage0 只复用/实现 `speech_embeddings`、`input_local`、projection，不能复用旧 MiMo-Audio 8-channel any-to-any 路径。
- **host tokenizer 是首轮关键依赖**：若 `audio_tokenizer.encode` 未接入，raw audio 请求无法工作；但这比在 JAX embed stage 移植 codec 更容易隔离、测试和回退。
- **MiMoVL 窗口注意力**为 JAX 无先例的行/列 1-D 窗口 + sink，需新写并做数值对齐。
- **形状分桶覆盖度**（prework 问题2）：桶集需覆盖真实图/音频 codes 长度分布，否则补零浪费或编译变体过多。
- **backbone 字段易错项**：KV heads GA=4/SWA=8（model card 标注相反，以 config 为准）；视觉 **head_dim=64**（来自 `qk_channels` 默认，**非** hidden/heads=1280/32=40）；MTP 不在 omni checkpoint。

### 6.4 开放问题（须以真实 checkpoint 复核）
- `audio_tokenizer/` 子目录的 RVQ **内部** config（`num_quantizers`（应=20）、`encoder_layers`/heads/`d_model`、`hybrid_attention`/`swa_per_block`、`codebook_size`）须由 host-side tokenizer encode 路径直接读取并对齐 HF remote code。
- raw audio、processor mel、pre-tokenized `audio_codes` 三种入口的优先级与错误信息需固化；首轮推荐 raw audio → host tokenizer，`audio_codes` 作为内部/测试入口。
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
