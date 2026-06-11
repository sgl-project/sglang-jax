# MiMo-V2.5 omni 接入 · 基于当前 multi-stage 架构的适配方案

> 本文是把 MiMo-V2.5 omni 模型接入 sglang-jax 的**适配方案**——它**基于当前的 multi-stage 架构**做接入，**不是目标/最终架构**：text-out 多模态的长期方向（不分 stage、把多模态 encode/splice 收进单一模型 forward 作为 prefill adapter）见 `text_out_multimodal_long_term_architecture.md`。本文由 step1/step2/实现笔记/review/vision 等工作文档整合而来，反映**当前代码**与**真机测试结果**；过程性材料（review 轮次、增量实现日志、里程碑叙事）已移除。
>
> 状态：**text / audio / image / video → text —— 全部模态均已实现，并在 v6e-16 真机上端到端验证通过。**
> 代码位置：`python/sgl_jax/srt/multimodal/models/mimo_v2_5/`。

---

## 1. 总览

MiMo-V2.5（`XiaomiMiMo/MiMo-V2.5`）是一个 omni 模型：**多模态输入（text/image/video/audio）、文本输出**。它只做理解——不含生成/vocoder。单一 checkpoint，扁平 config（`vision_config`/`audio_config` 直接挂在顶层，**没有 `thinker_config`**）。

本接入采用**两-stage（staged）** runtime：`embedding` stage 把各模态编码成 embedding 并注入文本序列；`auto_regressive` stage 用 MoE LM 逐 token 生成。两个 stage 在进程内通过 `queue.Queue` 串联，并通过 ZMQ 与 tokenizer/detokenizer 通信；AR stage 以 SPMD 方式跨 4 个 host 运行。

> 别混淆：`MiMo-V2.5-Pro` 是**纯文本** checkpoint；`MiMo-V2.5`（本文）是 omni 多模态版本。模型路由用精确别名匹配，且 `is_multimodal` 要求 `vision_config`/`audio_config` 非空。

---

## 2. 已核实的 HF checkpoint 事实

### 2.1 LLM backbone（AR stage 复用 `MiMoV2ForCausalLM` = mimo_v2_pro 的 fused-QKV MoE）

| 字段 | 值 |
|---|---|
| model_type / arch | `mimo_v2` / `MiMoV2ForCausalLM` |
| hidden / vocab / 层数 | 4096 / 152576 / 48（第 0 层 dense，1–47 MoE） |
| MoE | `n_routed_experts=256`、`num_experts_per_tok=8`、无 shared expert、gate sigmoid + noaux_tc、`moe_intermediate_size=2048` |
| 注意力 | `num_attention_heads=64`；KV：GA=4 / SWA=8；`head_dim=192`、`v_head_dim=128`；9 Full + 39 SWA（`sliding_window=128`）；sink bias 仅 SWA 层 |
| RoPE | **1-D 标准 RoPE**（`rope_scaling.type=default`，**无 mrope**）；`rope_theta`=1e7（GA）/ 1e4（SWA）；`partial_rotary_factor=0.334` |
| 量化 | FP8 e4m3，`weight_block_size=[128,128]`；全部 `o_proj` 排除量化；**omni checkpoint 不加载 MTP** |
| 其它 | `attention_value_scale=0.707`、`attention_projection_layout=fused_qkv`；总计约 293 GiB |

### 2.2 Vision = MiMoVL ViT（`MiMoVisionTransformer`）

| 字段 | 值 |
|---|---|
| depth | 28；full-attn 层 `fullatt_block_indexes=[0,9,18,27]`（4 full + 24 windowed） |
| 窗口注意力 | `vit_window_attn_types` ∈ {-1,0,1} = full / 行 1-D 窗口 / 列 1-D 窗口；`window_size=128`、`visual_token_window_size=64`；`use_sink=true`（attention sink） |
| 维度 | hidden 1280、`num_heads=32`、KV `num_key_value_heads=8`（GQA）；**head_dim=64**（来自 `qk_channels` 默认值，**不是** 1280/32=40） |
| patch | `patch_size=16`、`spatial_merge_size=2`、`temporal_patch_size=2`、`in_chans=3` |
| 输出 | `out_hidden_size=4096`（自带 merger projector，**无 bias**） |

> checkpoint 用的字段名是 `in_chans`（不是 `in_channels`），且**没有 `qk_channels`**（真实值为默认 64）；merger 的三层（`ln_q`/`mlp.0`/`mlp.2`）**只有 weight、无 bias**；patch_embed 的 Conv 是 `use_bias=False`，norm1/2 为 RMSNorm。这些在接入时被归一化/对齐（见 §7）。

### 2.3 Audio 理解 = host RVQ tokenizer + JAX audio tower

离散 RVQ 路径（不是连续 mel→projector）。拆成两部分：

- **Host 侧**：raw audio → 24kHz mono → log-mel → 冻结 RVQ `audio_tokenizer.encode` → `audio_codes [T',20]`（20 通道 RVQ）。`audio_tokenizer/` 是 checkpoint 的一个子目录（d_model 1024、24 层、16 heads、`num_quantizers=20`、`codebook_size=[1024,1024,256,128,...]`）；**缺失则 hard-fail，无 fallback**。mel 前端：`sr=24000`、`n_fft=960`、`hop=240`、`win=960`、`n_mels=128`。
- **JAX 侧（`MiMoV25AudioUnderstandingEncoder`）**：`audio_codes` → 20 通道 `speech_embeddings`（`speech_vocab_size=1280`、group_size=4）→ 6 层 full-attn input_local_transformer（dim 1024 / heads 16 / head_dim 64 / intermediate 4096，非 causal）→ 2 层 projection（`4096→16384→4096`，无 bias，**GELU**）→ `[N,4096]`。

> **复用准入（重要）**：`input_local_transformer` 复用了 `mimo_audio` 的 **`MiMoAudioTransformer` 作为 Qwen2-like building block**，而**不是**复用 mimo_audio 模型本身。复用只在以下硬条件成立时等价（`audio_encoder.py` 在 `__init__` 显式 guard，不匹配即报错）：`input_full_attention=true`（→ 非 causal）、`partial_rotary_factor=1.0`（当前仅支持 full rotary，`rotary_dim=head_dim`）、`add_post_norm=true`、`projection_layers=2`、`input_local_hidden_dropout=0`、`speech_vocab_size`/`speech_zeroemb_idx` 从 `audio_config` 读取（不沿用旧 8-channel 默认）。
>
> **别套用 mimo_audio 8-channel 链路**：旧的纯音频路径是 `mel → JAX MiMoAudioTokenizer → [8,T] codes → [1,9,T] 专用 backbone 输入`，由 `audio_backbone_scheduler` 调度；MiMo-V2.5 是 `audio_codes [T,20] → [N_group,4096] → scatter 到普通 input_embedding`。可复用底层写法（embedding list / local transformer / projection），**不能**复用 8-channel `Req.audio_codes` 合同、`[1,9,T]` 输入格式与 `audio_backbone_scheduler`。

### 2.4 各模态 token id（已从顶层 config 核实）

`image=151655`、`video=151656`、`vision_start=151652`、`audio=151669`（在 `processor_config` 子块里，需 fallback 查找）。三塔输出均为 `[N,4096]`（== LM hidden），按 token id scatter，无需 adapter。位置全程为 **1-D RoPE**。

### 2.5 权重前缀路由（用于 weight-mapping）

单 checkpoint，权重按名字**前缀路由**到三处目标：

- `model.*` → 文本 backbone（复用的 `MiMoV2Model`）；
- `visual.` / `vision_model.` → MiMoVL ViT（`self.visual`）；
- `audio_*` / `speech_embeddings.*` → JAX audio understanding tower（`self.audio_encoder`：`speech_embeddings(20)` / `input_local_transformer` / `projection`）；
- `audio_tokenizer/` 子目录 → **不进 stage0 JAX 权重**，由 host 侧 RVQ codec 加载并冻结（§2.3）。

`weights_mapping.py` 即按此前缀建 `{hf_key: target_path}`（如 `model.embed_tokens.weight`、`audio_encoder.input_local_transformer.layers.{i}.*`、`audio_encoder.projection.mlp.{0,2}.weight → proj_fc{1,2}`）；ViT 映射在 `self.visual` 构造后并入。

---

## 3. 上游 sglang 实现（parity 参考）

本接入的数值验收以上游 sglang 的 MiMo-V2.5 实现为 golden。本章梳理上游做法与"HF 提供什么 / serving 须自实现什么"的边界；§7（关键实现细节）与下文 host-side codec 的选择都以此为参照。

### 3.1 上游 serving 形态：自研 processor + model-side merge

上游不是只靠 HF `AutoProcessor`，而是有一个接近完整 runtime 的 `MiMoV2Processor / MiMoProcessor`，再由模型 forward 内部完成三模态合并：processor 负责 request 解析、媒体加载、placeholder 规划、视觉预处理、视频抽帧、audio mel 前端；模型侧再跑 `self.visual(...)` / `self.audio_encoder(...)` 并用 `general_mm_embed_routine` 把各模态 embed scatter 进 `input_embeds`。

### 3.2 audio 数据流（PR #23811）：mel 传到模型侧，codes 在模型内生成

关键差异：**上游 raw audio 路径把 mel-list 传给模型，`audio_codes` 是 `MiMoAudioEncoder.get_audio_feature()` 内部的临时值**，不在 processor 阶段固化（sglang-jax 则把这一步前移到 host 侧，见 §4 / §7 / §3.4）：

```text
MiMoProcessor._process_audio_content:
  raw audio → 24kHz mono → torchaudio MelSpectrogram(n_fft=960, hop=240, win=960, n_mels=128, power=1.0)
            → log(clamp(min=1e-7)).T → audio_spec [T_mel,128]
  audio_token_len = ceil( ((T_mel +3 -k +2 -k)//stride +1) / avg_pooler / group_size )
  input_ids = audio_start + audio_token_len*audio_token_id + audio_end
  → MultimodalDataItem(AUDIO, feature=mel-list)

MiMoV2ForCausalLM.get_audio_feature → MiMoAudioEncoder.get_audio_feature:
  mel-list → tokenize_audio_batch(..., audio_tokenizer.encoder, segment_size=6000, return_codes_only=True)
          → RVQ codes [T_code,20] → process_audio：取前 20 通道、按 group_size=4 pad(重复末帧)、reshape [N_group,4,20]
          → 20-channel speech_embeddings 逐通道相加 → input_local_transformer(is_causal=False)
          → reshape [N_group,4096] → projection(2) → audio_embeds [N_group,4096]
  → general_mm_embed_routine: 用 audio pad_value mask scatter 进 text input_embeds
```

两个易踩坑点：① **raw audio 与预 tokenized audio 是两种入口**——上游 `AudioInput` 允许 `Tensor[T,C]` 作为已 tokenized 输入，此时直接 `[T,C]→[ceil(T/4),4,20]` 分组；raw 路径产出 mel，codes 仍在模型侧生成。② **placeholder 数与 embedding 行数都绑定在 group 后长度**：processor 的 `audio_token_len` 估算必须等于模型侧 `N_group`，否则 mask 与 embedding 行数不匹配（这也是 §11.4 提到的 `code_lengths` 差一帧即错切的根因）。

### 3.3 HF 提供什么 / sglang 必须自实现什么

HF repo 的 `preprocessor_config.json` 仍是 Qwen2.5-VL 风格（`Qwen2VLImageProcessor` / `Qwen2_5_VLProcessor`），主体偏视觉/文本，**不是覆盖 raw image+video+audio 的 omni processor**；`config.json.processor_config` 存了 MiMo-V2.5 专属参数与 token id（`audio_sampling_rate=24000`、`audio_n_mels=128`、`audio_hop_length=240`、`audio_group_size=4`、`audio_channels=20`、`audio_token_id=151669`、`fps=1.0`、`use_video_timestamps=true` 等）；`modeling_mimo_v2.py` 的 forward 接收 `pixel_values`/`audio_codes` 并在模型内部 scatter（**支持 model-side merge**），但没有替 serving 完成 OpenAI request → 这些 kwargs 的转换。

| 可复用（HF / 上游） | 必须自实现（serving 缺口） |
|---|---|
| `config.json` / `processor_config`：token id、音频/视频/视觉参数 | OpenAI `image_url`/`video_url`/`audio_url`/base64 的统一解析、下载、错误处理 |
| `preprocessor_config.json`：Qwen2.5-VL 图像/视频预处理参数 | raw audio → 24kHz mono/log-mel → RVQ tokenizer → `audio_codes [T,20]` |
| `tokenizer_config.json` / chat template：特殊 token 与模板 | 多段 audio 的 `token_lengths`/`offsets`/placeholder 展开与校验 |
| `audio_tokenizer/` 子目录 + remote code：raw→codes 参考实现与权重 | video+audio 混合的时间戳/切段/interleave 策略 |
| `modeling_mimo_v2.py`：stage0 embedding/scatter 的**数值参考合同** | 面向 JAX stage0 的 `mm_items` / kwargs / shape bucket 传输合同 |

**结论**：需要一个 MiMo-V2.5 专用 processor 补齐 HF 没覆盖的 serving 逻辑——sglang-jax 因此自研 `MiMoV25Processor`（组合 Qwen2.5-VL 视觉/文本 + host RVQ codec），而非在通用 `MultimodalTokenizer` 里堆 `is_mimo_v25` 特判。

### 3.4 边界差异：model-side vs host-side codec

上游把 `mel→codes` 放在**模型 forward 内部**（model-side tokenizer）。sglang-jax 首轮把这一步**前移到 host 侧**（host-side codes-first，stage0 只做 codes 之后的 understanding tower），以规避在 JAX 内移植 RVQ tokenizer 的动态切段、argmin、专属 attention 与 JIT bucketing 风险。两者数值合同一致——因此 host 侧产出的 `audio_codes` 必须以上游 `MiMoAudioEncoder` 为 golden 对齐（§11.4）；向上游 model-side 形态收敛的路径见 §13。

---

## 4. 两-stage 架构

```
GlobalScheduler  (ZMQ ← tokenizer/detokenizer)
   │  omni request: text + image/video/audio placeholders
   ▼  queue.Queue
Stage-0  embedding        MiMoV2_5Embedding        device_kind=cpu  (num_tpus=1)
   │  text_embed_tokens(input_ids)
   │  _encode_audio   → self.audio_encoder (RVQ codes → [N,4096])
   │  _encode_visual  → self.visual (MiMoVL ViT; image + video share it)
   │  _scatter_modality: inject each modality's embeds at its placeholder token id
   ▼  input_embeds [seq,4096]  →  mm_inputs["multimodal_embedding"]
Stage-1  auto_regressive  MiMoV2ForCausalLM        16×v6e (tp=16 ep=16 fused MoE FP8)
   │  ForwardBatch.input_embedding hook (injected in extend/mixed; decode uses token embed)
   ▼  48-layer MoE backbone + 1-D RoPE
   text out
```

关键设计点：
- **设备切分**：embed stage 跑在 **CPU**（各塔都很小：audio + 一个 0.68B ViT，约 3 GB，放 host RAM），把全部 16 块 TPU 留给 293 GiB 的 AR backbone。AR mesh `[data=1, tensor=16]`，`ep_size=16` 整除 256 个 expert。
- **统一的模态接口**：`MiMoV2_5Embedding.__call__` 对 audio/image/video 跑同一套 `_encode_<modality>` → `_scatter_modality`；塔的命名对齐 HF checkpoint（`self.audio_encoder` / `self.visual`），且各 encoder 与 `embedding` 同目录（对齐 `qwen3_omni_moe` 的布局）。
- **AR 复用**：直接复用 `MiMoV2ForCausalLM`（已有的 text-only 实现），只加 input_embedding hook；无 `*Generation` 包装、无 mrope。
- **mm_items 作为单一真相源**：tokenizer 把 image/video/audio 统一构造成一组 `mm_items`（`MultimodalDataItem`），`assemble_mm_inputs` 再把它们拆成 per-modality kwargs，无任何 model-specific 逻辑。MiMo 用 1-D RoPE（无 mrope_section），所以 tokenizer 的 mrope 分支被正确跳过。

### Stage 配置（`models/static_configs/mimo_v2_5_stage_config.yaml`）

```yaml
model_arch: MiMo-V2.5
stage_args:
 - stage_id: 0
   runtime: { num_tpus: 1, device_kind: cpu, max_batch_size: 1 }
   scheduler: embedding
   model_class: MiMoV2_5Embedding
   final_output: false
 - stage_id: 1
   runtime: { num_tpus: 16, max_batch_size: 1 }
   scheduler: auto_regressive
   model_class: MiMoV2ForCausalLM
   precompile_params: { input_embedding: True, deepstack_visual_embedding: False, mrope: False }
   final_output: true
```

---

## 5. 各模态路径

- **Image**：Qwen2.5-VL 风格预处理 → `pixel_values` + `image_grid_thw` → `self.visual`（MiMoVL ViT）→ scatter 到 `image_token_id`。
- **Video**：同一 ViT，`pixel_values_videos` + `video_grid_thw`（`temporal_patch_size=2`）→ scatter 到 `video_token_id`，1-D 时间戳位置。预处理需要 `decord`。
- **Audio**：raw audio → host RVQ codec → `audio_codes [T',20]` → `self.audio_encoder`（speech_embeddings → group_size=4 → local transformer → projection）→ `[N,4096]` → scatter 到 `audio_token_id`。`#audio_pad == ceil(T'/4)`，并有 hard check `sum(token_lengths) == audio_embeds.shape[0]`；多段 audio 在 concat 前逐段 pad 到 4 的倍数（无跨段 grouping）；越界的 code id 在 host 侧校验。Audio 是可选的——任意子集请求会跳过它。
- **Text**：`text_embed_tokens(input_ids)`。

---

## 6. multi-host SPMD

**动机**：293 GiB 的 FP8 MoE 放不进单块 v6e 芯片（32 GiB），所以需要 16 块 = v6e-16 = 4 host × 4 chip。原多模态 runtime 是单进程，从不调用 `jax.distributed.initialize`，因此 `jax.devices()` 只看到本地 4 块芯片。

**选择**：SPMD multi-controller（4 个 pod 各跑一个进程，组成 16 芯片 mesh，所有 rank 跑同一个 jitted forward）；Pathways 被否决（目标 GKE 环境无现成支持）。

**核心冲突**：AR stage 用进程内 `QueueBackend`，绕过了标准 pub/sub。若加了 distributed init 却不广播 batch → rank0 的 AR collective 等其它 rank，而它们的进程内 queue 是空的 → 死锁。所以每个 AR batch 都必须从 rank0 lockstep 广播给全部 4 个 rank。

**5 处改动**：
1. `multimodal/entrypoint/http_server.py:launch` —— 透传 `nnodes/node_rank/dist_init_addr`；`node_rank>=1` 只启动 GlobalScheduler 进程。
2. `multimodal/manager/global_scheduler.py` —— 在构造 `DeviceManager()` 前 `jax.distributed.initialize()`（nnodes>1）。
3. `managers/communication.py` —— 新增 `MultiHostQueueBackend`：rank0 PUB 广播（含空 batch），非 rank0 SUB 接收，`send_pyobj` 仅在 rank0。
4. `multimodal/manager/stage.py` —— nnodes>1 时 AR stage 用 `MultiHostQueueBackend`，否则用 `QueueBackend`。
5. embed（CPU）只在 rank0 生效；**stage 启动被串行化**（nnodes>1 时按 stage_id 顺序构建），使并行线程的 `broadcast_one_to_all` 不会在跨 host 间出现不确定顺序 → collective 死锁。

---

## 7. 关键实现细节与约束（在真机上暴露并修复）

- **fused MoE 对齐（硬约束）**：`num_tokens` 必须是 `ep_size * t_packing`（=32，bf16 下 `t_packing=32//dtype_bits=2`）的倍数；所有 precompile 的 bs/token 桶都对齐到 32。`ep_size` 必须同时整除 256 个 expert 与设备总数。
- **explicit sharding 下的复制**：embed stage 用 explicit-sharding mesh；以下张量在 reshape/scatter 前必须复制（`with_sharding_constraint(P())`，各塔很小、embed seq 也短），并用 `getattr(self,"mesh")` 做 mesh 守卫：
  - 每次 audio-tower hidden reshape 之前；
  - `_scatter_modality` 的 `input_embeds` + `modality_embeds`（`.at[pos].set(..., mode="drop")`，`fill_value=seq_len` 以避免越界写到 token 0）；
  - **所有 vision 权重以复制方式加载（`sharding=()`）**：MiMoVL ViT 用裸 `nnx.Linear`（无 sharding 标注），loader 基于名字的默认 sharding 会把 gate/up/down/qkv kernel tensor-shard，使其与复制的激活做 `dot_general` 在 embed mesh 上无法 resolve → 崩溃；强制复制即可修复。
- **vision config 归一化**（`_normalize_vision_config`）：checkpoint 没有 `qk_channels` → 填默认 `64`；`in_chans` → `in_channels`。错误值会在权重加载时被 qkv 形状不匹配捕获（fail-safe）。
- **vision merger 无 bias**：`MiMoVisionPatchMerger` 的 `ln_q`/`mlp_fc1`/`mlp_fc2` 以 `use_bias=False` 构建（checkpoint 没有 merger bias），否则 bias 参数会停留为抽象的 `eval_shape` 占位符 → 使用时崩溃。
- **把 vision 输入 pin 到 CPU mesh（V4 修复）**：`embed_model_runner._prepare_input` 里由 `jnp.asarray` 构造的 `pixel_values` 落在默认 TPU backend 上；embed forward 是 eager、无 `with mesh` 上下文，于是 patch_embed 的 reshape 跑在 TPU 上并分配 TPU HBM（与 AR stage 的芯片争用），长视频时 OOM。修复：把 `pixel_values/pixel_values_videos/image_grid_thw/video_grid_thw` `device_put` 到 embed 的 CPU mesh，让整个 ViT 跑在 CPU/host RAM 上。
- **模型无关的 embed 契约**：模型返回 `EmbedOutput(input_embeds, deepstack=None, pos_mask=None)`，runner 按字段名读取；`get_embed_model_config` 剥掉 fp8 `quantization_config`（embed 以 bf16 加载，AR stage 自建 ModelConfig 并解析 fp8）；`get_total_num_kv_heads` 对裸 HF config 做 hasattr 守卫。host 侧输入校验通过一个可选模型 hook（`validate_embed_inputs`）暴露，使 runner 不含 model-specific 逻辑。
- **AR input_embedding hook**：仅在 extend/draft_extend/mixed forward 模式下注入 stage0 合并结果；decode 用 token embedding；deepstack 是 no-op；位置来自 `forward_batch.positions`。

---

## 8. 文件清单（当前实现）

**新增（`multimodal/models/mimo_v2_5/`）**：`embedding.py`（MiMoV2_5Embedding）、`vision_encoder.py`（MiMoVisionTransformer，基于上游 PR #1302）、`audio_encoder.py`、`audio_codec_processor.py`、`processor.py`（MiMoV25Processor，包装 Qwen2.5-VL + host RVQ codec）、`weights_mapping.py`、`__init__.py`、`static_configs/mimo_v2_5_stage_config.yaml`。

**AR 复用（非新增）**：`models/mimo_v2_pro.py::MiMoV2ForCausalLM` + `mimo_v2_flash.py::MiMoV2Model`。

**修改的入口/执行器/runtime**：`configs/model_config.py`（`is_multimodal_model()`）、`entrypoints/openai/serving_chat.py`（`GenerateOmniReqInput` 路由）、`multimodal/entrypoint/http_server.py`、`managers/communication.py`（`MultiHostQueueBackend`）、`multimodal/manager/{global_scheduler,stage}.py`、`model_executor/embed/embed_model_runner.py`、`io_struct.py`/`schedule_batch.py`/`forward_batch_info.py`（`Req.multimodal_embedding` → `ForwardBatch.input_embedding`）、`multimodal/manager/multimodal_tokenizer.py` + `mm_assembly.py`（通用 mm_items 组装）。

---

## 9. 部署（GKE）

- 集群 `lianfang-v6e-mimo`（us-east5-a），pool 4×4=16 芯片。4-pod Indexed Job + headless Service 做 rank 发现；环境变量 `TPU_WORKER_HOSTNAMES`（4 个 pod DNS）+ `TPU_WORKER_ID` → node_rank。
- 镜像 `jax-ai-image/tpu:jax0.8.1-rev1`（jax 装在 `/opt/venv`；`bash -lc` 会丢掉 venv 的 PATH，所以用显式的 `/opt/venv/bin/python`）。
- gcsfuse CSI 插件被禁用 + 节点磁盘 <293 GiB → 用 **RAM-emptyDir（节点约 708 GiB）+ pod 内 `gcloud storage rsync`** 拉取 294 GB 权重，绕过 gcsfuse。
- 额外依赖（镜像缺失，部署时安装）：`torch torchvision torchaudio`（Qwen2.5-VL AutoProcessor + host codec）、`decord` + `imageio-ffmpeg`（视频预处理）。
- 启动：`/opt/venv/bin/python -m sgl_jax.launch_server --multimodal --model-path /weights/MiMo-V2.5 --trust-remote-code --nnodes 4 --node-rank $TPU_WORKER_ID --dist-init-addr <pod0>:9876 --tp-size 16 --ep-size 16 --moe-backend fused --page-size 64`。

---

## 10. 真机测试结果（v6e-16）

全部在 4-pod multi-host（tp=16 ep=16 fused MoE FP8）上运行，验证的是**内容理解**，而不只是管路打通：

| 输入 | 结果 | 备注 |
|---|---|---|
| text | ✅ | 自我识别为 Xiaomi MiMo，43 tok/s |
| text + 真实语音 | ✅ | MLK "I Have a Dream" 13s —— **正确识别语音并转写出匹配的文本** |
| text + image | ✅ | 两只猫 + 遥控器均正确识别 |
| text + video | ✅ | 指挥中心 / 地图屏 / 人物均正确（8 帧） |
| text + image + video | ✅ | 两个模态分别描述，无串扰 |
| text + image + video + audio | ✅ | 四者全部正确，无跨模态混淆 |

> 真实语音的端到端运行证明了 host RVQ codec（含 mel 前端）产出的结果足以支撑准确的内容理解——这是 review 中标记的最大未验证项（"raw-audio→codes 链路从未对真实 checkpoint 跑过"，R2-3/9/11），现已功能性确认。AR `num_tpus=16` 也在真机上确认可服务（解决了此前 R2-2 的尺寸问题）。

---

## 11. 未解决问题 / 已知限制

### 11.1 OpenAI 多模态入口回归（无需真实权重；影响所有多模态 chat 模型）
- **R3-1** `serving_chat` 构造 `GenerateOmniReqInput` 时不透传 `n` → `n>1` 静默退化为 1。
- **R3-2** `rid` 作为 `list[str]` 被透传；下游把 list 当 dict key → `TypeError`。
- **R3-6** 多模态 `logprobs/top_logprobs` 的 `ValueError` 被包成 HTTP 500；应为 4xx。
- **R3-7** `_send_one_request` 在注册 `rid_to_state` 之前就发送；快速完成的请求其输出可能因 rid 未知被丢弃（race）。

### 11.2 Cache / radix 内容隔离（一旦启用 radix cache 即为正确性 bug）
- **R3-3** `extra_key` 到了 omni request，但 `to_stage_reqs()` 不透传它 → cache-namespace 隔离丢失。
- **R3-4** audio 内容没有进入 AR 的 radix cache key：`cache_input_ids`（含 audio hash）没从 stage0 传给核心 `Req` → 相同 `<audio_pad>` 文本但不同 audio 可能复用错误的 KV prefix。另：`pad_input_tokens` 不递增 image/video/audio 的 idx，导致多段时只用第一个 item 的 pad_value。

### 11.3 模板/降级路径上多模态内容被静默丢弃
- **R3-5** string-format chat template 在 `process_content_for_template_format` 里丢掉 audio/image/video 部分 → 被识别为 string template 的模型静默变成 text-only。
- **R3-8** `resolve_host_processor` 吞掉 import 期异常并 fallback 到裸 HF processor → MiMoV25Processor 里真实的 import bug / 缺依赖会伪装成"无匹配 wrapper"，难以诊断。

### 11.4 数值保真度（功能已证明，golden 测试未做）
- **R3-9 / R2-11** raw audio mel→codes 没有相对 HF/参考 codec 的离线数值 golden；结构参数全部与 config 匹配、真实语音端到端也能正确转写，但 mel 的 scale/window/normalization 细节没有 golden 安全网；`code_lengths` 公式与实际返回的 codes 是两条路径，若差一帧就会错切。建议补一个离线 golden。
- vision 路径同样没有 HF 数值对齐回归（`test_mimo_vision_encoder.py` 测试骨架已在，需 vision-checkpoint 环境才能跑）。

### 11.5 加固（误配置 → 静默错误）
- **R2-8** `get_codebook_sizes()` 在解析失败时 fallback 到统一 `1280`，会放过低 codebook 通道上的越界高 id；应按真实 per-quantizer `[1024,1024,256,128,...]` 逐通道校验。
- **R2-12** AR stage yaml 没有 pin `ep_size`（来自全局 `--ep-size`）；它必须同时整除 256 与设备数，而 fused 在不匹配时只 warn。建议在 stage 里 pin 住。
- **R3-10** 模型路由 `_match_mimo_v25_omni` 在路径中无词边界地排除任何 `pro`/`flash` 子串，所以 `/prod/MiMo-V2.5` 之类会被误路由。

### 11.6 性能
- **embed 未 JIT / 未分桶**：embed forward 目前是 **eager**（`forward_wrapper` 未包 `jax.jit`）。原计划的 JIT + 形状分桶未实现；CPU eager 对小图/短音频可接受，但长视频（178 帧）非常慢。

### 11.7 本轮范围外
- **DP > 1**：每个 stage 的 mesh 把 `data` 轴固定为 1，与核心 `dp_size` 冲突；需要 `ici_parallelism=[dp_size, chips//dp_size]` + 启动断言。本轮为 **DP=1**。
- **batch > 1**：stage0/stage1 `max_batch_size=1`；整序列 input_embedding 替换无法混 batch；需要 token 级 scatter-merge。本轮为 **batch=1**。
- **视频内交错音轨**：`video_audio_interleave_length`；本轮只支持 basic（静音 / 忽略音轨）视频。
- **音频/语音生成（ASR/TTS）**：独立 checkpoint 的 RVQ 生成 / vocoder 不在 omni-理解范围内。

---

## 12. 测试与验收建议

§10 的真机结果验证了功能，但仍缺离线数值 golden（§11.4）。建议补的分层验收：

- **逐塔数值对齐**：text / MiMoVL ViT / audio（host `audio_tokenizer.encode` 的 codes、codes→input_local→projection）各与上游/HF 参考输出对比（bf16/fp8 容差），golden 取 §3 的上游 `MiMoAudioEncoder.get_audio_feature()`。
- **audio tokenizer parity**：固定 wav，host `audio_tokenizer.encode` 的 `audio_codes`（shape / range / 逐段 `code_lengths`）与 HF remote code 对齐；mel 前端另做 parity（~1e-4），盯住 §3.2 的 `audio_token_len` 公式与实际返回 codes 不能差一帧。
- **合并正确性**：每模态"占位符数 == 特征行数"断言（audio 有 hard check `sum(token_lengths)==audio_embeds.shape[0]`）；多图/多段音频各自 `pad_value`（radix cache 可区分，注意 §11.2 的 R3-4）。
- **端到端 golden**：图文 / 视频 / 音频理解各一组，纳入 `test/srt`（vision 的 `test_mimo_vision_encoder.py` 骨架已在，需 vision-checkpoint 环境）。

---

## 13. 后续方向：V1 model-side tokenizer（mel-first）

若要向 §3 的上游形态收敛，把 `mel → audio_codes` 收回 stage0 内部（AUDIO `mm_item` 改为承载 mel，而非 codes）：

```text
V0（当前）：host: raw→mel→audio_tokenizer.encode→codes[T,20]；AUDIO.feature=codes
            stage0: codes → speech_embeddings → local → projection → scatter
V1（方向）：host: raw→mel + length oracle 算 token_lengths；AUDIO.feature=mel（+ mel_lengths/token_lengths/offsets meta）
            stage0: mel → JAX MiMo-V2.5 audio tokenizer → codes[T,20] → …（同上）
```

**scheduler 边界不因 mel/codes 而变**——只要 transport 继续只搬运 `mm_items`、不解释 audio 语义，边界仍干净；V0/V1 真正差异在 **processor 与 stage0 模型合同**。

**V1 前提**：① JAX/TPU 版 MiMo-V2.5 audio tokenizer（移植 `audio_tokenizer/` 的 encoder、动态切段、`get_output_length`、RVQ/codebook search）；② host 侧**精确 placeholder length oracle**——`input_ids` 仍须在 tokenizer 阶段展开正确 `<audio_pad>` 数，用与 stage0 **同源**的纯函数（按 segment_size=6000 切段、conv/stride/avg_pooler/group_size 推导），禁止 host 与 stage0 各写一份易漂的公式；③ length oracle 只用真实 `mel_lengths`，不用 bucket-pad 后的 `T_bucket`；④ 多段逐段算、禁止跨段 grouping；⑤ stage0 运行时一致性断言；⑥ 三层 golden（`mel→codes` / `codes→audio_embeds` / `mel→audio_embeds`）。

**重构冲击**：若已是 `mm_items` 单一真相源，V1 不造成 server/scheduler/AR 大重构（`GlobalScheduler` 不变），工作量集中在 **stage0 模型侧**（新增 JAX audio tokenizer encoder + RVQ + 权重映射 + bucket + parity；processor 从 codes-first 改 mel-first；`mm_assembly`/`_prepare_input` 区分 V2.5-mel 与 Qwen3-Omni continuous）。建议迁移顺序：stage0 同时支持 codes/mel（默认 codes）→ shadow 比对 → 默认切 mel（codes 留测试入口）→ 降级 V0 host codec 主路径。
