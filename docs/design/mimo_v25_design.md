# MiMo-V2.5 omni 接入 · 最终方案设计

> 本文是 MiMo-V2.5 omni 模型在 sglang-jax 上多模态接入的**最终设计**,合并自 step1/step2/implementation-notes/review/vision 等过程文档,反映**当前代码实现**与**真机测试结论**。过程性记录(逐轮 review、增量实现日志、里程碑流水)已剔除。
>
> 状态:**text / audio / image / video → text 全模态已实现并在真机 v6e-16 端到端验证通过**。
> 代码位置:`python/sgl_jax/srt/multimodal/models/mimo_v2_5/`。

---

## 1. 概述

MiMo-V2.5(`XiaomiMiMo/MiMo-V2.5`)是 omni 模型:**多模态输入(text/image/video/audio)、纯 text 输出**,只做理解、不含生成/vocoder。单一 checkpoint、扁平 config(`vision_config`/`audio_config` 直接挂顶层,**无 `thinker_config`**)。

接入采用 **two-stage(staged)** runtime:`embedding` stage 把各模态编码成 embedding 并注入文本序列,`auto_regressive` stage 用 MoE LM 逐 token 生成。两 stage 同进程内以 `queue.Queue` 串联,对外经 ZMQ 接 tokenizer/detokenizer;AR stage 跨 4 host 跑 SPMD。

> 注意区分:`MiMo-V2.5-Pro` 是 **text-only** checkpoint;`MiMo-V2.5`(本文)才是 omni 多模态。模型路由用 exact alias,且 `is_multimodal` 要求 `vision_config`/`audio_config` 非空。

---

## 2. HF checkpoint 核实事实

### 2.1 LLM backbone(AR stage 复用 `MiMoV2ForCausalLM` = mimo_v2_pro fused-QKV MoE)

| 项 | 值 |
|---|---|
| model_type / arch | `mimo_v2` / `MiMoV2ForCausalLM` |
| hidden / vocab / 层数 | 4096 / 152576 / 48(第 0 层 dense,1–47 MoE) |
| MoE | `n_routed_experts=256`、`num_experts_per_tok=8`、无 shared expert、gate sigmoid + noaux_tc、`moe_intermediate_size=2048` |
| 注意力 | `num_attention_heads=64`;KV: GA=4 / SWA=8;`head_dim=192`、`v_head_dim=128`;9 Full + 39 SWA(`sliding_window=128`);sink bias 仅 SWA 层 |
| RoPE | **1-D 标准 RoPE**(`rope_scaling.type=default`,**无 mrope**);`rope_theta`=1e7(GA)/1e4(SWA);`partial_rotary_factor=0.334` |
| 量化 | FP8 e4m3,`weight_block_size=[128,128]`;全部 `o_proj` 排除量化;**omni checkpoint 不加载 MTP** |
| 其它 | `attention_value_scale=0.707`、`attention_projection_layout=fused_qkv`;总量 ~293GiB |

### 2.2 Vision = MiMoVL ViT(`MiMoVisionTransformer`)

| 项 | 值 |
|---|---|
| depth | 28;full-attn 层 `fullatt_block_indexes=[0,9,18,27]`(4 full + 24 窗口) |
| 窗口注意力 | `vit_window_attn_types`∈{-1,0,1}=full/行 1-D 窗口/列 1-D 窗口;`window_size=128`、`visual_token_window_size=64`;`use_sink=true`(attention sink) |
| 维度 | hidden 1280、`num_heads=32`、KV `num_key_value_heads=8`(GQA);**head_dim=64**(来自 `qk_channels` 默认,**非** 1280/32=40) |
| patch | `patch_size=16`、`spatial_merge_size=2`、`temporal_patch_size=2`、`in_chans=3` |
| 输出 | `out_hidden_size=4096`(自带 merger projector,**bias-free**) |

> checkpoint 字段名是 `in_chans`(非 `in_channels`),且**无 `qk_channels`**(真值取默认 64);merger 三层(`ln_q`/`mlp.0`/`mlp.2`)**只有 weight、无 bias**;patch_embed Conv `use_bias=False`、norm1/2 是 RMSNorm。这些在接入时归一/对齐(见 §6)。

### 2.3 Audio 理解 = host RVQ tokenizer + JAX audio tower

离散 RVQ 路径(非连续 mel→projector)。拆两段:

- **host 侧**:raw audio → 24kHz mono → log-mel → 冻结 RVQ `audio_tokenizer.encode` → `audio_codes [T',20]`(20 通道 RVQ)。`audio_tokenizer/` 是 checkpoint 子目录(d_model 1024、24 层、16 heads、`num_quantizers=20`、`codebook_size=[1024,1024,256,128,...]`),**缺失则 hard-fail,禁回落**。mel 前端:`sr=24000`、`n_fft=960`、`hop=240`、`win=960`、`n_mels=128`。
- **JAX 侧(`MiMoV25AudioUnderstandingEncoder`)**:`audio_codes` → 20 通道 `speech_embeddings`(`speech_vocab_size=1280`,group_size=4)→ 6 层 full-attn input_local_transformer(dim 1024/heads 16/head_dim 64/intermediate 4096,非 causal)→ 2 层 projection(`4096→16384→4096`,bias-free,**GELU**)→ `[N,4096]`。

### 2.4 三模态 token id(顶层 config 核实)

`image=151655`、`video=151656`、`vision_start=151652`、`audio=151669`(`processor_config` 子块,需 fallback)。三塔输出均 `[N,4096]`(== LM hidden),按 token_id scatter,无需 adapter。位置全程 **1-D RoPE**。

---

## 3. 两-stage 架构

```
GlobalScheduler  (ZMQ ← tokenizer/detokenizer)
   │  omni request: text + image/video/audio placeholders
   ▼  queue.Queue
Stage-0  embedding        MiMoV2_5Embedding        device_kind=cpu  (num_tpus=1)
   │  text_embed_tokens(input_ids)
   │  _encode_audio  → self.audio_encoder (RVQ codes → [N,4096])
   │  _encode_image  → self.visual (MiMoVL ViT,pixel_values+image_grid_thw)
   │  _encode_video  → self.visual (pixel_values_videos+video_grid_thw)
   │  _scatter_modality: 各模态 embeds 注入对应 token_id 占位
   ▼  input_embeds [seq,4096]  →  mm_inputs["multimodal_embedding"]
Stage-1  auto_regressive  MiMoV2ForCausalLM        16×v6e (tp=16 ep=16 fused MoE FP8)
   │  ForwardBatch.input_embedding hook(extend/mixed mode 注入,decode 用 token embed)
   ▼  48-layer MoE backbone + 1-D RoPE
   text out
```

设计要点:
- **设备分配**:embed stage 跑 **CPU**(模态塔小:audio + 0.68B ViT 共 ~3GB,用 host RAM),把全部 16 个 TPU chip 留给 293GiB 的 AR backbone。AR mesh `[data=1, tensor=16]`,`ep_size=16` 整除 256 experts。
- **模态统一接口**:`MiMoV2_5Embedding.__call__` 对 audio/image/video 走同一 `_encode_<modality>` → `_scatter_modality`;塔命名对齐 HF checkpoint(`self.audio_encoder` / `self.visual`),encoder 与 embedding 同目录(对齐 `qwen3_omni_moe` 约定)。
- **AR 复用**:直接用 `MiMoV2ForCausalLM`(text-only 实现已有),仅新增 input_embedding hook;不建 `*Generation` 包装、不建 mrope。
- **mm_items 单一真相源**:tokenizer 把图/视频/音频统一组成 `mm_items`(`MultimodalDataItem`),`assemble_mm_inputs` 通用拆成 per-modality kwargs;无模型专属逻辑。MiMo 用 1-D RoPE(无 mrope_section)→ tokenizer mrope 分支正确跳过。

### Stage 配置(`models/static_configs/mimo_v2_5_stage_config.yaml`)

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

## 4. 每模态路径

- **Image**:Qwen2.5-VL 风格预处理 → `pixel_values`+`image_grid_thw` → `self.visual`(MiMoVL ViT)→ scatter 到 `image_token_id`。
- **Video**:同一 ViT,`pixel_values_videos`+`video_grid_thw`(`temporal_patch_size=2`)→ scatter 到 `video_token_id`,1-D 时间戳位置。预处理需 `decord`。
- **Audio**:raw audio → host RVQ codec → `audio_codes [T',20]` → `self.audio_encoder`(speech_embeddings → group_size=4 → local transformer → projection)→ `[N,4096]` → scatter 到 `audio_token_id`。`#audio_pad == ceil(T'/4)`,强校验 `sum(token_lengths)==audio_embeds.shape[0]`;多段 audio 逐段独立 pad 到 4 倍数再 concat,禁跨段 grouping;非法 code id 由 host validate。audio 可选,任意子集请求跳过。
- **Text**:`text_embed_tokens(input_ids)`。

---

## 5. multi-host SPMD

**动因**:293GiB FP8 MoE 单 v6e chip(32GiB)装不下,需 16 chip = v6e-16 = 4 host × 4 chip。原 multimodal runtime 单进程,从不调 `jax.distributed.initialize`,`jax.devices()` 只见本机 4 chip。

**路径**:选 SPMD 多-controller(4 pod 各跑一进程,组 16-chip mesh,所有 rank 跑相同 jitted forward);否决 Pathways(GKE 无现成支撑)。

**核心矛盾**:AR stage 走 `QueueBackend` 进程内 queue,旁路了标准 pub/sub。仅加 distributed init 不广播 batch → rank0 AR collective 等其它 rank、非 rank0 queue 永空 → 死锁。∴ 每个 AR batch 必须从 rank0 lockstep 广播到全部 4 rank。

**5 处改动**:
1. `multimodal/entrypoint/http_server.py:launch` — 透传 `nnodes/node_rank/dist_init_addr`;`node_rank>=1` 只起 GlobalScheduler 进程。
2. `multimodal/manager/global_scheduler.py` — 建 `DeviceManager()` 前(nnodes>1)`jax.distributed.initialize()`。
3. `managers/communication.py` — 新增 `MultiHostQueueBackend`:rank0 PUB 广播(含空 batch),非 rank0 SUB 接收,`send_pyobj` 仅 rank0 输出。
4. `multimodal/manager/stage.py` — AR stage 且 nnodes>1 用 `MultiHostQueueBackend`,否则 `QueueBackend`。
5. embed(CPU)仅 rank0 有效;**stage 启动串行化**(nnodes>1 按 stage_id 顺序构建,避免并行线程的 `broadcast_one_to_all` 跨 host 顺序不定 → collective 死锁)。

---

## 6. 关键实现细节与约束(真机暴露并固化)

- **fused MoE 对齐(硬约束)**:`num_tokens` 必须是 `ep_size * t_packing` 倍数(=32,`t_packing=32//dtype_bits=2` for bf16);precompile bs/token bucket 全按 32 对齐。`ep_size` 须同时整除 256 experts 与设备总数。
- **explicit-sharding 下的复制**:embed stage 用 explicit-sharding mesh,以下张量在 reshape/scatter 前必须 `with_sharding_constraint(P())` 全复制(塔小、embed 序列短),并以 `getattr(self,"mesh")` 守卫:
  - audio 塔每个 hidden reshape 前;
  - `_scatter_modality` 的 `input_embeds` + `modality_embeds`(`.at[pos].set(..., mode="drop")`,`fill_value=seq_len` 防越界写 token 0);
  - **vision 权重全部 `sharding=()` 复制加载**:MiMoVL ViT 用裸 `nnx.Linear`(无 sharding 注解),loader 按名会把 gate/up/down/qkv kernel tensor-分片,与复制激活做 `dot_general` 在 embed mesh 上无法解析 → 崩;强制复制修复。
- **vision config 归一**(`_normalize_vision_config`):checkpoint 无 `qk_channels` → 补默认 `64`;`in_chans` → `in_channels`。错值会在 weight load 时被 qkv shape mismatch 抓到(fail-safe)。
- **vision merger bias-free**:`MiMoVisionPatchMerger` 的 `ln_q`/`mlp_fc1`/`mlp_fc2` 建为 `use_bias=False`(checkpoint 无 merger bias),否则 bias 参数停在 eval_shape 抽象态 → 用时崩。
- **vision 输入钉 CPU mesh(V4 修复)**:`embed_model_runner._prepare_input` 用 `jnp.asarray()` 造的 `pixel_values` 默认落 TPU backend;embed forward 是 eager 且无 `with mesh` 上下文,导致 patch_embed reshape 在 TPU 上跑、占 TPU HBM(与 AR stage 抢 chip),长视频 OOM。修复:对 `pixel_values/pixel_values_videos/image_grid_thw/video_grid_thw` `device_put` 到 embed CPU mesh,整条 ViT 全程跑 CPU/host RAM。
- **embed 模型无关契约**:模型返回 `EmbedOutput(input_embeds, deepstack=None, pos_mask=None)`,runner 按字段取用;`get_embed_model_config` 剥离 fp8 `quantization_config`(embed 走 bf16,AR 自建 ModelConfig 解 fp8);`get_total_num_kv_heads` 对裸 HF config 加 hasattr 兜底。
- **AR input_embedding hook**:仅 extend/draft_extend/mixed forward mode 注入 stage0 合并结果,decode 用 token embedding;deepstack no-op,位置走 `forward_batch.positions`。

---

## 7. 文件清单(当前实现)

**新增(`multimodal/models/mimo_v2_5/`)**:`embedding.py`(MiMoV2_5Embedding)、`vision_encoder.py`(MiMoVisionTransformer,基于上游 PR #1302)、`audio_encoder.py`、`audio_codec_processor.py`、`processor.py`(MiMoV25Processor,wraps Qwen2.5-VL + host RVQ codec)、`weights_mapping.py`、`__init__.py`、`static_configs/mimo_v2_5_stage_config.yaml`。

**AR 复用(不新增)**:`models/mimo_v2_pro.py::MiMoV2ForCausalLM` + `mimo_v2_flash.py::MiMoV2Model`。

**改动入口/执行/runtime**:`configs/model_config.py`(`is_multimodal_model()`)、`entrypoints/openai/serving_chat.py`(`GenerateOmniReqInput` 路由)、`multimodal/entrypoint/http_server.py`、`managers/communication.py`(`MultiHostQueueBackend`)、`multimodal/manager/{global_scheduler,stage}.py`、`model_executor/embed/embed_model_runner.py`、`io_struct.py`/`schedule_batch.py`/`forward_batch_info.py`(`Req.multimodal_embedding`→`ForwardBatch.input_embedding`)、`multimodal/manager/multimodal_tokenizer.py` + `mm_assembly.py`(mm_items 通用组装)。

---

## 8. 部署(GKE)

- 集群 `lianfang-v6e-mimo`(us-east5-a),pool 4×4=16 chip。4-pod Indexed Job + headless Service 做 rank 发现;env `TPU_WORKER_HOSTNAMES`(4 pod DNS)+ `TPU_WORKER_ID`→node_rank。
- 镜像 `jax-ai-image/tpu:jax0.8.1-rev1`(jax 装在 `/opt/venv`,`bash -lc` 会丢 venv PATH,须显式 `/opt/venv/bin/python`)。
- gcsfuse CSI addon 未启用 + 节点盘 <293GiB → 用 **RAM-emptyDir(node ~708GiB)+ pod 内 `gcloud storage rsync`** 下载 294G 权重绕过。
- 额外依赖(镜像缺,部署时装):`torch torchvision torchaudio`(Qwen2.5-VL AutoProcessor + host codec)、`decord`+`imageio-ffmpeg`(video 预处理)。
- 启动:`/opt/venv/bin/python -m sgl_jax.launch_server --multimodal --model-path /weights/MiMo-V2.5 --trust-remote-code --nnodes 4 --node-rank $TPU_WORKER_ID --dist-init-addr <pod0>:9876 --tp-size 16 --ep-size 16 --moe-backend fused --page-size 64`。

---

## 9. 真机测试结论(v6e-16)

全部在 4-pod multi-host(tp=16 ep=16 fused MoE FP8)上跑通,且是**内容理解**而非仅链路:

| 输入 | 结果 | 说明 |
|---|---|---|
| text | ✅ | 自我识别为小米 MiMo,43 tok/s |
| text + 真实语音 | ✅ | MLK "I Have a Dream" 13s,**正确识别演讲并转写出与原文一致句子** |
| text + image | ✅ | 两猫+遥控器都正确识别 |
| text + video | ✅ | 任务控制中心/大屏地图/人物正确(8 帧) |
| text + image + video | ✅ | 两模态分别正确、无混淆 |
| text + image + video + audio | ✅ | 四模态全部正确,无跨模态混淆 |

> 真实语音端到端通过,实证了 host RVQ codec(含 mel 前端)的数值输出足以让模型做准确内容识别——这是之前 review 标注"raw-audio→codes 链从未对真实 checkpoint 跑过"(R2-3/9/11)的最大未证项,现已功能性证实。AR `num_tpus=16` 亦经真机证实可服务(原 R2-2 定档存疑已解)。

---

## 10. 未解决问题 / 已知限制

### 10.1 OpenAI 多模态入口回归(不依赖真权重,影响所有多模态 chat 模型)
- **R3-1** `serving_chat` 构造 `GenerateOmniReqInput` 漏传 `n` → `n>1` 静默退化为 1。
- **R3-2** `rid` 为 `list[str]` 时直接透传,下游用 list 作 dict key → `TypeError`。
- **R3-6** 多模态 `logprobs/top_logprobs` 的 `ValueError` 被包成 HTTP 500,应是 4xx。
- **R3-7** `_send_one_request` 先发包后登记 `rid_to_state`,快速完成请求可能输出被 unknown-rid 丢弃(竞态)。

### 10.2 Cache / radix 内容隔离(启用 radix cache 即为 correctness bug)
- **R3-3** `extra_key` 进了 omni request 但 `to_stage_reqs()` 未转发 → cache namespace 隔离丢失。
- **R3-4** audio 内容未进入 AR radix cache key:`cache_input_ids`(含音频 hash)未从 stage0 传给核心 `Req` → 不同音频+相同 `<audio_pad>` 文本可能复用错误 KV 前缀。附:`pad_input_tokens` 里 image/video/audio idx 不递增,多段只用第一个 item 的 pad_value。

### 10.3 多模态内容在模板/降级路径被静默丢弃
- **R3-5** string-format chat template 在 `process_content_for_template_format` 丢弃 audio/image/video parts → 被识别为 string template 的模型静默变 text-only。
- **R3-8** `resolve_host_processor` 吞掉 import 异常后回落裸 HF processor → MiMoV25Processor 的真实 import bug/缺依赖被伪装成"无匹配 wrapper",定位困难。

### 10.4 数值保真(功能已证、golden 未做)
- **R3-9 / R2-11** raw audio mel→codes 未做与 HF/上游 codec 的离线数值 golden parity;结构参数已逐项命中 config,且真实语音端到端已正确转写,但 mel scale/window/normalization 细节无 golden 兜底;`code_lengths` 公式与实际返回 codes 是两条路径,漂 1 帧会切错。建议补离线 golden。
- vision 路径同样未做 HF 数值对齐回归(已有 `test_mimo_vision_encoder.py` 框架,需在装好 vision checkpoint 的环境跑)。

### 10.5 加固项(误配 → 静默错)
- **R2-8** `get_codebook_sizes()` 解析失败回落统一 `1280`,放行低 codebook 通道的非法高位 code;应按真实 per-quantizer `[1024,1024,256,128,...]` 逐 channel 校验。
- **R2-12** AR stage yaml 未 pin `ep_size`(来自全局 `--ep-size`);须同时整除 256 与设备数,fused 不一致仅 warn。建议 stage 显式 pin。
- **R3-10** 模型路由 `_match_mimo_v25_omni` 对 path 上任意 `pro`/`flash` 子串无词边界排除,`/prod/MiMo-V2.5` 等会误判。

### 10.6 性能与依赖
- **embed 未 JIT/分桶**:当前 embed forward 是 **eager**(`forward_wrapper` 未包 `jax.jit`)。原设计的 JIT + 形状分桶未落地;CPU eager 对小图/短音频可接受,但长视频(178 帧)极慢。
- **长视频 OOM 边界**:V4 device 修复后整条 ViT 跑 CPU/host RAM,不再 TPU OOM;但长视频帧数大、CPU eager 慢,生产需限帧/分块策略。
- **依赖未入镜像**:torch/torchvision/torchaudio + decord 目前部署时临时安装,待并入镜像。

### 10.7 超出本轮范围
- **DP > 1**:每 stage 网格 `data` 轴恒 1,与核心 `dp_size` 冲突;需 `ici_parallelism=[dp_size, 芯片数//dp_size]` + 启动断言。本轮 **DP=1**。
- **batch > 1**:stage0/stage1 `max_batch_size=1`,整段替换 input_embedding 无法混跑;需 token-level scatter-merge。本轮 **batch=1**。
- **视频内交错音轨**:`video_audio_interleave_length`,首轮只支持静音/忽略音轨的 basic 视频。
- **音频/语音生成(ASR/TTS)**:独立 checkpoint 的 RVQ 生成 / vocoder,不在 omni 理解范围。
