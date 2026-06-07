# MiMo-V2.5 实现记录与 HF checkpoint 复核结论

更新时间：2026-06-04

## 当前已实现范围

当前分支已为现有 2-stage `embedding -> auto_regressive` 流水线接入一条 MiMo-V2.5 audio-first 路径：

- stage 配置：`python/sgl_jax/srt/multimodal/models/static_configs/mimo_v2_5_stage_config.yaml`
- embed-stage 模型：`python/sgl_jax/srt/multimodal/models/mimo_v2_5/embedding.py`
- AR stage 复用 `MiMoV2ForCausalLM`（`mimo_v2_pro` 的 fused-QKV 变体，匹配真实 config `attention_projection_layout=fused_qkv`），并已支持 `forward_batch.input_embedding`。

本轮仍有意跳过 image/video tower。若传入 image/video tensor，`MiMoV2_5Embedding.__call__` 会抛出 `NotImplementedError`。

## 调研来源与可信度说明

本轮并行 agent 调研中，部分 Web/HF/ModelScope 在线抓取因网关/工具问题失败；但 audio checkpoint / architecture agent 成功基于本地 HF 参考缓存 `/tmp/mimo_hf` 与现有设计文档中的 HF 行号审计给出结论。以下结论分为两类：

- **已确认**：来自 `/tmp/mimo_hf/{config.json,modeling_mimo_v2.py,audio_tokenizer/config.json}`、现有设计文档的 HF 行号审计，或当前仓库源码。
- **仍需最终验证**：需要对真实下载 checkpoint 的 safetensors key list 做一次 `safe_open(...).keys()` dump 才能 100% 确认。

## 已确定的实现决策

1. **音频理解架构**不是连续 mel 直接投影，而是：waveform → log-mel → 冻结 RVQ audio tokenizer encoder → 20-channel discrete codes → 20 个 `speech_embeddings` → 6 层双向 local transformer → 2 层 projection → 4096 维 audio embedding → scatter 到 `audio_token_id=151669`。
2. **`audio_tokenizer/` 是模型 checkpoint 根目录下的子目录**，不是 sglang-jax 源码目录。源码中可复用的是 `python/sgl_jax/srt/multimodal/models/mimo_audio/mimo_audio_tokenizer.py`。HF 代码存在 out-of-band loader（调研缓存：`/tmp/mimo_hf/modeling_mimo_v2.py:1716-1744`），从 `audio_tokenizer/config.json` + `audio_tokenizer/model.safetensors` 或 `pytorch_model.bin` 加载并冻结 tokenizer。
3. **Embed runner 的 config 提取**由模型声明：模型类可提供 `get_embed_model_config`；否则 Qwen3-Omni 继续使用 `thinker_config`。
4. **MiMo-V2.5 generation** 使用 1-D positions，不请求 mrope/deepstack 预编译 flags。现有 HF 审计显示 MiMo-V2.5 forward 使用 1-D `position_ids`，未发现 `get_rope_index` / mrope 函数。

## HF checkpoint / processor 复核结论

### 1. checkpoint 布局：确认有独立 `audio_tokenizer/`

结论：**MiMo-V2.5 checkpoint 预期包含独立的 `audio_tokenizer/` 子目录**。

证据：

- 本轮调研使用的本地 HF 参考缓存中存在 `/tmp/mimo_hf/audio_tokenizer/config.json`。
- HF 代码有独立加载逻辑：`/tmp/mimo_hf/modeling_mimo_v2.py:1716-1744`，加载 `path/config.json` 与 `path/model.safetensors` / `pytorch_model.bin`，并将 tokenizer 置为 eval/frozen。
- 既有设计文档也记录了同一假设：`docs/design/mimo_v25_step2_model_integration.md:59`、`docs/design/mimo_v25_step2_model_integration.md:75`。

仍需最终验证：ModelScope 镜像文件树是否与 HF 完全一致；建议对真实部署 checkpoint 根目录执行文件树检查。

### 2. 主 checkpoint audio 权重名前缀：当前实现部分不匹配

结论：主模型中的 audio understanding 权重预计分布为：

- `speech_embeddings.{i}.weight`，`i=0..19`
- `audio_encoder.input_local_transformer...`
- `audio_encoder.projection...`

原因：HF constructor 创建的是顶层 `self.speech_embeddings = ...` 与 `self.audio_encoder = MiMoAudioEncoder(...)`（调研缓存：`/tmp/mimo_hf/modeling_mimo_v2.py:1706-1711`）。因此 speech embeddings 是顶层前缀，但 local transformer / projection 位于 `audio_encoder.` 下。

当前代码状态：

- `speech_embeddings.{i}.weight` 映射方向基本正确。
- `python/sgl_jax/srt/multimodal/models/mimo_v2_5/embedding.py` 已将 local transformer 源权重前缀修正为 `audio_encoder.input_local_transformer...`。
- projection 映射已修正为 HF 模块名 `audio_encoder.projection.mlp.0.weight` / `audio_encoder.projection.mlp.2.weight`，且不再映射 bias。

当前映射目标应保持类似：

- `audio_encoder.input_local_transformer.layers.{l}.self_attn.{q,k,v,o}_proj.weight`
- `audio_encoder.input_local_transformer.layers.{l}.self_attn.{q,k,v}_proj.bias`
- `audio_encoder.input_local_transformer.layers.{l}.mlp.{gate,up,down}_proj.weight`
- `audio_encoder.input_local_transformer.layers.{l}.{input_layernorm,post_attention_layernorm}.weight`
- `audio_encoder.input_local_transformer.norm.weight`
- `audio_encoder.projection.mlp.0.weight`
- `audio_encoder.projection.mlp.2.weight`

仍需最终验证：对真实主 checkpoint safetensors dump key list，确认是否存在额外前缀或 shard index 变体。

### 3. audio tokenizer encoder/RVQ 权重名：确认是独立 checkpoint 内 `encoder.*`

结论：`audio_tokenizer/` 子 checkpoint 内的 RVQ tokenizer 权重使用 `encoder.*` 前缀。

预期 key 包括：

- `encoder.conv1.weight` / `encoder.conv1.bias`
- `encoder.conv2.weight` / `encoder.conv2.bias`
- `encoder.layers.{l}.self_attn.{q,k,v,out}_proj.{weight,bias}`
- `encoder.layers.{l}.self_attn_layer_norm.{weight,bias}`
- `encoder.layers.{l}.final_layer_norm.{weight,bias}`
- `encoder.layers.{l}.fc1.{weight,bias}`
- `encoder.layers.{l}.fc2.{weight,bias}`
- `encoder.layer_norm.{weight,bias}`
- `encoder.down_sample_layer.0.weight`
- `encoder.down_sample_norm.{weight,bias}`
- `encoder.quantizer.vq.layers.{j}._codebook.embed`

证据：HF tokenizer 结构为 `MiMoAudioTokenizer.encoder = AudioTokenizerEncoder(...)`（调研缓存：`/tmp/mimo_hf/modeling_mimo_v2.py:1374-1423`、`:1500-1515`）；当前 JAX tokenizer mapping 也已有相同命名参考：`python/sgl_jax/srt/multimodal/models/mimo_audio/mimo_audio_tokenizer_weights_mapping.py`，RVQ codebook 手动加载见 `python/sgl_jax/srt/multimodal/models/mimo_audio/mimo_audio_tokenizer.py`。

### 4. projection 结构：确认当前实现不匹配 HF

结论：**HF projection 是 GELU，不是 SiLU；结构也不是两个 4096→4096 线性层。**

HF `AudioProjection` 结构（调研缓存：`/tmp/mimo_hf/modeling_mimo_v2.py:919-929`、`:959-965`）：

```text
reshape [G, 4, 1024] -> [G, 4096]
Linear(4096 -> 16384, bias=False)
GELU
Linear(16384 -> 4096, bias=False)
```

当前 `python/sgl_jax/srt/multimodal/models/mimo_v2_5/embedding.py` 已按 HF 结构修正 audio projection：

- 先将 `[G, 4, 1024]` reshape 为 `[G, 4096]`；
- 使用 bias-free `4096 -> 16384`；
- 层间激活为 GELU；
- 再使用 bias-free `16384 -> 4096`。

仍需通过真实 checkpoint 权重加载与数值对齐验证该实现。

### 5. processor / forward 契约：forward 需要 `audio_codes` 或 `audio_embeds`

结论：**HF 模型 forward 路径接收的是 RVQ `audio_codes` / `audio_embeds`，不是 mel features。** mel 是 host/processor 前处理产物，必须先经 tokenizer encode 变成 codes，或由 helper/wrapper 在 forward 前完成转换。

证据：

- 既有审计记录 HF `_get_multimodal_embeds` 分支检查 `audio_codes` / `audio_embeds`，再调用 `self.audio_encoder(speech_embeddings=self.speech_embeddings, audio_codes=audio_codes, audio_embeds=audio_embeds)`（设计文档 `docs/design/mimo2.5_part2_model_adaptation_plan.md:28`、`:139-143`）。
- 本轮 processor agent 结论：host processor/mel frontend 产出 mel-like `input_features` / `audio_features`；HF model forward 需要 RVQ `audio_codes`，除非 wrapper/helper 先跑 mel → codec encode。

当前 JAX 实现支持两种输入：

- `input_features`：内部调用 audio tokenizer encode；
- `audio_codes`：直接进入 understanding tower。

但从 HF 对齐和 TPU 性能角度看，v1 更推荐 host 侧先跑 mel → RVQ codes，再把 `[T3, 20]` codes 交给 JAX understanding tower，避免在 TPU/JIT 内处理动态 mel 长度、conv、RVQ argmin。

### 6. audio tensor shape 与 token 数：已明确

结论：

- mel feature：每段 audio 为 `[T_mel, 128]`。
- RVQ codes：HF tokenizer encode 得到 `[20, T3]`，转置后作为 `[T3, 20]` 进入 understanding tower。
- grouping：`[T3, 20] -> [G, 4, 20]`，其中 `G = ceil(T3 / 4)`，不足 4 的尾部通过重复最后一帧 padding。
- 20 个 channel embedding 求和后得到 `[G, 4, 1024]`。
- local transformer 保持 `[G, 4, 1024]`。
- projection 输出 `[G, 4096]`。
- tokenizer/chat template 插入的 `<|audio_pad|>` 数必须等于 `G`；只有 `<|audio_pad|>` 被替换，start/end token 保留文本 embedding。

placeholder layout：

```text
<|mimo_audio_start|> 151673
G × <|audio_pad|>   151669
<|mimo_audio_end|>   151674
```

音频 token 数公式，给定 mel 帧数 `L`、`kernel_size=3`、`stride_size=2`、`avg_pooler=2`、`group_size=4`：

```text
t = L + 3 - kernel_size
t = (t + 2 - kernel_size) // stride_size + 1
t = t // avg_pooler + (1 if t % avg_pooler else 0)
G = ceil(t / group_size)
```

当前 JAX `_merge_audio` 只取前 `audio_embeds.shape[0]` 个 audio token 位置进行 scatter，没有 host 侧严格等量断言。后续应在进入 traced/JIT 之前加断言：`#audio_pad == audio_embeds.shape[0]`。

### 7. RVQ tokenizer config：现有 fallback config 不兼容

结论：必须加载 MiMo-V2.5 的 `audio_tokenizer/config.json`，不能 fallback 到现有 `MiMoAudioConfig()` 旧默认值。

V2.5 audio tokenizer config 关键值：

- `d_model=1024`
- `encoder_layers=24`
- `encoder_attention_heads=16`
- `encoder_ffn_dim=4096`
- `num_quantizers=20`
- `codebook_size=[1024,1024,256,128,...]`，第三项是 `256`，不是旧 JAX 默认的 `128`
- `kernel_size=3`
- `stride_size=2`
- `avg_pooler=2`
- `encoder_causal=true`
- `encoder_attn_window_size=[128,0]`
- `hybrid_attention=true`
- `swa_per_block=2`
- mel 参数：24kHz、128 mel bins、nfft 960、hop 240、window 960

当前源码的 `python/sgl_jax/srt/multimodal/configs/mimo_audio/mimo_audio_config.py` 默认是旧 MiMo-Audio：`d_model=1280`、`encoder_layers=32`、`encoder_attention_heads=20`、codebook 第三项为 `128`。因此如果 `audio_tokenizer/config.json` 加载失败，继续 fallback 会导致结构/权重都不匹配。后续代码应改为：MiMo-V2.5 模式下找不到 `audio_tokenizer/` 或 config 加载失败时直接报错，而不是静默使用 `MiMoAudioConfig()`。

### 8. JAX 现有 `mimo_audio` 组件兼容性

结论：只能“部分复用”。

可复用：

- `mimo_audio_tokenizer.py` 的 conv + transformer + RVQ encode 结构可作为 V2.5 codec encoder 移植参考。
- `mimo_audio_tokenizer_weights_mapping.py` 的 encoder/RVQ mapping 可复用一部分。
- `mimo_audio_backbone.py` 的 `MiMoAudioTransformer` 可作为 6 层 local transformer building block 参考。

不兼容 / 需改造：

- 旧 `MiMoAudioConfig` 默认维度不匹配 V2.5。
- 现有 tokenizer class 包含 decoder/vocoder，V2.5 understanding 只需要 encoder/RVQ 半边。
- 现有 tokenizer attention 对每层使用同一 `encoder_attn_window_size`，而 V2.5 codec 需要 `hybrid_attention=true` / `swa_per_block=2` 的 per-layer full/window 切换。
- 当前 `mimo_v2_5/embedding.py` 已补齐 `[T3,20] -> [G,4,20]` grouping，并按 HF 修正 projection 结构；仍需真实 checkpoint 数值对齐。

### 9. 模型 repo id / stage alias

结论：`XiaomiMiMo/MiMo-V2.5-Pro` 是 text-only / 非多模态模型；本轮要接入的是 **MiMo-V2.5 omni 模型**，即带 `vision_config` / `audio_config` / `audio_tokenizer/` 的多模态 checkpoint。二者不能混用，也不应让 text-only Pro 误命中 MiMo-V2.5 omni 的 multimodal stage config。

处理原则：

- `XiaomiMiMo/MiMo-V2.5-Pro` / `MiMo-V2.5-Pro` 走现有 text-only MiMoV2 Pro 路径。
- MiMo-V2.5 omni 走 `mimo_v2_5_stage_config.yaml` 双 stage 路径；stage registry 当前保留明确 alias：`XiaomiMiMo/MiMo-V2.5` / `MiMo-V2.5`。
- 避免使用 `MiMo-V2.5` / `mimo_v2` 这类宽泛 keyword fallback 去匹配 stage config；否则 text-only Pro 也可能误进 multimodal 双 stage。

当前已据此移除 stage/config registry 中宽泛的 MiMo-V2.5 keyword fallback；后续如发现 MiMo-V2.5 omni 的生产 repo id 有额外精确别名，应只添加精确 alias，而不是恢复宽泛匹配。

## 已确认需要修正的代码点

1. `python/sgl_jax/srt/multimodal/models/mimo_v2_5/embedding.py`
   - 已补齐 `[T3,20] -> [G,4,20]` grouping。
   - 已将 projection 改为 bias-free `4096 -> 16384 -> 4096` + GELU。
   - 已将 source mapping 加上 `audio_encoder.` 前缀。
   - 已将 projection source mapping 改为 `audio_encoder.projection.mlp.0.weight` / `audio_encoder.projection.mlp.2.weight`。
   - 已将找不到 `audio_tokenizer/` 的行为改为 hard fail，不再 fallback 到旧 `MiMoAudioConfig()`。
2. `python/sgl_jax/srt/multimodal/models/static_configs/yaml_registry.py`
   - MiMo-V2.5 omni 的精确 alias 已保留为 `XiaomiMiMo/MiMo-V2.5` / `MiMo-V2.5`；不要添加 `XiaomiMiMo/MiMo-V2.5-Pro`，因为它是 text-only / 非多模态模型。
3. `python/sgl_jax/srt/multimodal/configs/config_registry.py`
   - 同上，仅添加真实 omni repo 的精确 audio config alias；避免宽泛 keyword fallback 误匹配 text-only Pro。
4. host/processor 侧
   - 最终应确认 `AutoProcessor` 返回 dict；若其只给 mel features，则推荐在 host 侧完成 mel → RVQ codes，再传入 JAX embed stage。

## 代码 Review 发现（2026-06-04，按修复优先级）

> 以下为基于当前 diff、MiMo-V2.5 HF remote code / config、`mimo_v25_implementation_notes.md` 与 `mimo2.5_part2_model_adaptation_plan.md` 的优先级排序。排序原则：先修会破坏现有模型或导致 MiMo-V2.5 omni 错路径/错结果的问题，再修首轮 audio-first 正确性，最后保留 API 完整性与范围限制。

### P0：必须先修，否则会破坏现有模型或让 MiMo-V2.5 omni 走错路径

1. ~~**三模型边界必须固化：MiMo-Audio、MiMo-V2.5 omni、MiMo-V2.5-Pro 不能混用**~~（已完成，2026-06-04：`is_multimodal_model()` 已收紧为 MiMoV2 仅在真实 `vision_config` / `audio_config` 非空时进入 omni 路径）
   - 文件：`python/sgl_jax/srt/multimodal/models/static_configs/yaml_registry.py`、`python/sgl_jax/srt/multimodal/models/static_configs/mimo_v2_5_stage_config.yaml`、`python/sgl_jax/srt/configs/model_config.py`
   - 现象：当前分支已经避免把 `XiaomiMiMo/MiMo-V2.5-Pro` 加到 omni stage alias，但后续实现仍容易把 Pro 的 text-only 路径、旧 MiMo-Audio 的 audio 路径和 MiMo-V2.5 omni 路径混在一起。
   - HF 结构结论：
     - `XiaomiMiMo/MiMo-V2.5` 是 omni，多模态 checkpoint 带 `audio_config`、`vision_config`、`audio_tokenizer/`，audio understanding 是 20-channel。
     - `XiaomiMiMo/MiMo-V2.5-Pro` 是 text-generation/text-only，同属 MiMoV2 text backbone family，但不能命中 `mimo_v2_5_stage_config.yaml`。
     - `XiaomiMiMo/MiMo-Audio-7B-*` 是 any-to-any audio model，旧 config 为 8-channel、patch encoder/decoder/TTS 语义，不是 MiMo-V2.5 omni 的 audio understanding tower。
   - 修复方向：stage/config registry 只保留真实 omni repo 的精确 alias；`ModelConfig.is_multimodal` 只在真实 `vision_config` / `audio_config` 非空时返回 true；禁止用 `mimo_v2` / `MiMo-V2.5` 宽泛 keyword fallback 自动匹配多模态 stage。
   - 第一轮补充：背景是三个 repo 虽然共享 MiMo 命名，但 serving 路由、stage config 和 audio contract 都不同；本轮已把 omni 判定收紧到真实 `vision_config` / `audio_config` 非空，并继续只保留 `XiaomiMiMo/MiMo-V2.5` 的精确 omni alias。争议/遗留是后续如果要支持 Pro，只应走 text-only 模型注册；如果要支持 MiMo-Audio，也应作为独立 audio model 接入，不能复用 MiMo-V2.5 omni stage。

2. ~~**`EmbedModelRunner` 无条件传入 `audio_codes` 会破坏现有 Qwen3-Omni**~~（已完成，2026-06-04：runner 已按模型声明或 `__call__` 签名过滤 embed kwargs，Qwen3-Omni 不再收到 `audio_codes`）
   - 文件：`python/sgl_jax/srt/multimodal/model_executor/embed/embed_model_runner.py`
   - 现象：通用 runner 现在调用 `model(..., audio_codes=...)`，但 `Qwen3OmniMoeThinkerEmbedding.__call__` 没有 `audio_codes` 参数。
   - 触发：启动/请求现有 Qwen3-Omni embed stage 时直接 `TypeError`。
   - 修复方向：runner 按模型声明的 input spec 过滤 kwargs，例如模型类提供 `get_embed_input_keys()`；MiMo-V2.5 声明 `audio_codes`，Qwen3-Omni 不声明。临时热修可以让 Qwen3-Omni embedding 接收并忽略 `audio_codes=None`，但长期应走 input spec。
   - 第一轮补充：背景是通用 embed runner 不能把某个模型新增的输入字段强塞给所有 embedding class；本轮已在 runner 中按 `get_embed_input_keys()` 或 `__call__` 签名过滤 kwargs，MiMo-V2.5 可以接收 `audio_codes`，Qwen3-Omni 不再收到该字段。争议/遗留是是否要求所有 embedding model 显式声明 input spec；当前签名 introspection 是兼容过渡方案。

3. ~~**`ModelConfig.is_multimodal` 使用 `hasattr` 可能误判 text-only MiMoV2**~~（已完成，2026-06-04：与第 1 项同根，已改为检查真实 `vision_config` / `audio_config` 非空）
   - 文件：`python/sgl_jax/srt/configs/model_config.py`
   - 现象：若 text-only MiMo-V2.5-Pro config 暴露 `vision_config=None` / `audio_config=None`，`hasattr` 仍为 true。
   - 触发：Pro 被误判为 multimodal，OpenAI serving 构造 `GenerateOmniReqInput`，进入错误 stage。
   - 修复方向：改为检查 `getattr(hf_config, "vision_config", None) is not None` 或 `getattr(hf_config, "audio_config", None) is not None`。`audio_token_id` 单独存在不足以证明是 omni，除非同时有真实 processor/audio config 证据。
   - 第一轮补充：背景是部分 text-only config 可能暴露空的 multimodal 字段，`hasattr` 会把“字段存在”误判成“模型支持”；本轮已改成非空配置判定。争议/遗留是 `audio_token_id` 是否可作为辅助信号，当前不采用，因为单 token id 不能证明存在完整 audio tower / processor contract。

5. ~~**MiMoV2 `input_embedding` bypass 未按 forward mode 限制**~~（已完成，2026-06-04：已按 Qwen2/Qwen2.5-VL 形态限制为 extend/draft_extend/mixed 才消费 `input_embedding`）
   - 文件：`python/sgl_jax/srt/models/mimo_v2_flash.py`
   - 现象：只要 `forward_batch.input_embedding is not None` 就绕过 token embedding。
   - 触发：decode/verify batch 若残留或携带非空 `input_embedding`，会用全 prompt embedding 处理单 token decode，导致 shape mismatch 或错误 logits。
   - 修复方向：参考 `Qwen2Model`，仅在 `forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()` 为 true 时使用 `input_embedding`；decode 继续查 token embedding。
   - 第一轮补充：背景是 multimodal embedding 只应在 extend 类 forward 消费，decode 阶段应回到 token embedding；Qwen2.5-VL 能 pass 的原因正是它已有 forward-mode guard，而不是无条件 bypass 本身安全。本轮已把 MiMoV2 改为只在 `extend/draft_extend/mixed` 使用 `input_embedding`。争议/遗留是 Qwen3-Omni 仍存在更宽松写法，但它依赖调度侧只在 extend 阶段注入 embedding；是否统一收敛到 guard 形态可作为后续跨模型清理。

6. ~~**audio scatter 缺少 `#audio_pad == #audio_embeds` 校验**~~（已完成，2026-06-04；复盘修正：该校验已收敛到 MiMo-V2.5 embedding 专属分支，不作为 `EmbedModelRunner` 的通用 audio 规则）
   - 文件：`python/sgl_jax/srt/multimodal/model_executor/embed/embed_model_runner.py`、`python/sgl_jax/srt/multimodal/models/mimo_v2_5/embedding.py`
   - 现象：MiMo-V2.5 `_merge_audio` 使用 `jnp.nonzero(..., size=audio_embeds.shape[0], fill_value=0)` 定位 `<|audio_pad|>`，如果 placeholder 与 audio embedding 行数不一致，会发生静默 pad/truncate。
   - 触发：placeholder 少于 embedding 行数时，多余 audio embedding 可能覆盖 token 0；placeholder 多于 embedding 行数时，剩余 `<|audio_pad|>` 保留文本 embedding。
   - 修复方向：在进入 JAX scatter 前做 host-side 校验，但校验必须绑定 MiMo-V2.5 的 audio contract：`audio_codes` 为 20-channel RVQ ids，`#audio_pad == ceil(T3 / group_size)`。不能把这条规则放成 shared runner 的无条件 audio 校验，因为 Qwen3-Omni 也复用 `EmbedModelRunner`，它走 `audio_features/audio_feature_lengths` contract，不应被 MiMo-V2.5 的 `audio_codes` 规则约束。
   - 第一轮补充：背景是校验思路正确，但最初实现把 `_validate_audio_placeholder_count()` 放在共享 runner 的 `omni_inputs` 路径里，虽然 `audio_codes is None` 时会 return，语义上仍然让 shared runner 带上 MiMo-V2.5 强假设。本轮复盘后已改为 `_validate_mimo_v25_audio_placeholder_count()`，只在 `model_class == MiMoV2_5Embedding` 的分支调用；Qwen3-Omni 即使继续使用同一个 runner，也不会触发 MiMo-V2.5 的 20-channel/group_size 校验。后续已在 `MiMoV25AudioCodecProcessor.attach_offsets_from_input_ids(...)` 中补了 tokenizer 侧 span-aware 校验；runner 侧仍保留总量校验作为最后防线。

### P1：首轮 audio-first 内建议修，避免 MiMo-V2.5 audio 数值或性能不可控

8. ~~**mel → RVQ codes 仍在 embed-stage/JAX 路径中执行**~~（已完成，2026-06-04：omni 请求结构、tokenizer、global scheduler 已透传 `audio_codes`；MiMo-V2.5 embed stage 不再内部执行 mel→codec encode；复盘回滚：共享 tokenizer 内的 MiMo-V2.5 host tokenizer encode 打桩已撤回）
   - 文件：`python/sgl_jax/srt/multimodal/models/mimo_v2_5/embedding.py`
   - 现象：当输入是 `input_features` 时，embed stage 内部跑 conv + 24 层 codec + fp32 RVQ argmin。
   - 触发：processor 只返回 mel features 时，在 TPU/JIT 热路径处理动态长度和 argmin，容易重编译、OOM 或 tracer concretization。
   - 修复方向：按 v1 设计优先在 host/processor 侧完成 mel → RVQ codes，embed stage 只消费 `[T3,20]` `audio_codes`。需要扩展 `GenerateOmniReqInput` / `TokenizedGenerateOmniReqInput` / `Req` 的字段流转，最好使用 MiMo-V2.5 专用 payload，避免复用旧 MiMo-Audio `[8,T]` 语义。
   - 第一轮补充：背景是 codec encode 包含动态长度、argmin 和 tokenizer 权重路径，不适合放在 JAX embed 热路径里；本轮保留 omni request/tokenized request/global scheduler 对 `audio_codes` 的透传，MiMo-V2.5 embed stage 遇到 mel-only `input_features` 会明确报错。复盘后已撤回共享 `MultimodalTokenizer` 中动态加载 MiMo-V2.5 remote code / torch `audio_tokenizer.encode` 的打桩。争议/遗留是 raw audio → `audio_codes` 应放到 MiMo-V2.5 专属 adapter/helper 中实现，而不是塞进共享 tokenizer。

9. ~~**`_codes_from_mels` 在 traced 路径使用 Python `int()`**~~（已完成，2026-06-04：已删除 embed-stage mel encode fallback 及其中的 Python `int()` 路径）
   - 文件：`python/sgl_jax/srt/multimodal/models/mimo_v2_5/embedding.py`
   - 现象：`int(enc.output_lengths[0])` 依赖 JAX runtime value。
   - 触发：JIT 下会抛 `ConcretizationTypeError`。
   - 修复方向：移除 embed-stage 内 mel encode；若保留 eager fallback，需确保不进入 JIT 并仅作为调试路径。
   - 第一轮补充：背景是 `_codes_from_mels` 里对 JAX runtime value 调 Python `int()`，一旦被 JIT trace 会触发 concretization；本轮通过删除 embed-stage mel encode fallback 移除了该路径。争议/遗留是如需调试用 eager encode，应单独放在 host/debug helper，不能回到模型 `__call__`。

10. ~~**MiMo-V2.5 audio path 实例化完整旧 `MiMoAudioTokenizer`，包含 decoder/vocoder**~~（已完成，2026-06-04：MiMo-V2.5 embed 模型已移除完整 tokenizer 实例和 tokenizer 权重加载，只保留 audio understanding tower）
    - 文件：`python/sgl_jax/srt/multimodal/models/mimo_v2_5/embedding.py`、`python/sgl_jax/srt/multimodal/models/mimo_audio/mimo_audio_tokenizer.py`
    - 现象：V2.5 understanding 只需要 encoder/RVQ 半边，但当前构造完整 tokenizer。
    - 触发：真实 V2.5 `audio_tokenizer/` 若只含 encoder/RVQ 权重或缺 decoder/vocoder config，可能初始化/加载失败；即使成功也浪费内存和启动时间。
   - 修复方向：首轮推荐 host 侧直接使用 HF/torch tokenizer encode 输出 `audio_codes`。若必须在 JAX 内支持 tokenizer，则新增 encoder-only tokenizer/codec wrapper，复用 `AudioEncoder` + RVQ 手动加载，丢弃 decoder/vocoder 映射。
   - 第一轮补充：背景是完整旧 tokenizer 会加载 decoder/vocoder 等 V2.5 understanding 不需要的模块，还可能与 V2.5 `audio_tokenizer/` 文件树不兼容；本轮已从 MiMo-V2.5 embed stage 移除完整 `MiMoAudioTokenizer` 实例和 tokenizer 权重加载。复盘后撤回了共享 tokenizer 层动态加载 MiMo-V2.5 remote code 的实现。争议/遗留是 host-side encode 应由 MiMo-V2.5 专属 adapter/helper 完成；如果未来要求 JAX 内完成 codec，需要新增 V2.5 encoder-only tokenizer，而不是恢复旧完整 tokenizer。

11. ~~**复用的 `AudioEncoder` 不支持 V2.5 codec 的 hybrid attention**~~（已完成，2026-06-04：MiMo-V2.5 embed stage 已不再使用 JAX codec/tokenizer；未来若恢复 JAX codec，需另建 encoder-only + hybrid attention 实现）
    - 文件：`python/sgl_jax/srt/multimodal/models/mimo_audio/mimo_audio_tokenizer.py`
    - 现象：现有 encoder 对所有层使用同一 `encoder_attn_window_size`。
    - 触发：V2.5 `audio_tokenizer/config.json` 需要 `hybrid_attention=true` / `swa_per_block=2`，应按层切换 local/full；当前 RVQ codes 会与 HF 不一致。
   - 修复方向：若保留 JAX tokenizer，为 V2.5 codec encoder 增加 per-layer full/window attention 选择；否则 v1 固定 host-side HF tokenizer，JAX embed stage 不负责 codec。
   - 第一轮补充：背景是 V2.5 codec 的 `hybrid_attention=true` / `swa_per_block=2` 与旧 AudioEncoder 的统一窗口注意力不同，会导致 codes 不一致；本轮选择不在 JAX embed stage 跑 codec，因此规避该数值风险。争议/遗留是 host-side HF tokenizer 是当前正确性基线，若后续移植 JAX codec，必须补 per-layer attention 语义并用 HF 输出对齐测试。

12. ~~**audio feature mask / length 处理与 MiMo `[B,T,128]` 契约不一致**~~（已完成，2026-06-04：MiMo-V2.5 runner 路径已只接受 `audio_codes`；共享 tokenizer 内 raw audio/mel → codes 打桩已撤回，不再做 Qwen3-Omni 式压平）
    - 文件：`python/sgl_jax/srt/multimodal/model_executor/embed/embed_model_runner.py`
    - 现象：mask compaction 假设 channel-first，并会把 batched mel flatten 成单个 2D 张量，同时保留 per-sample lengths。
    - 触发：processor 输出 `[B,T,128]` + `feature_attention_mask [B,T]` 时，转置/boolean indexing 形状不匹配或跨样本拼接。
   - 修复方向：MiMo-V2.5 audio 优先传 `audio_codes`；若支持 mel，则保持 `[B,T,128]` 与 `[B]` lengths，不做 Qwen3-Omni 式压平。
   - 第一轮补充：背景是 Qwen3-Omni 的 audio feature compaction 假设与 MiMo-V2.5 `[B,T,128]` / `[T3,20]` contract 不一致；本轮 MiMo-V2.5 runner 只接收 codes，并避免在 embed runner 中做 Qwen 式压平。复盘后撤回共享 tokenizer 内的 MiMo host encode 分支，raw audio / batched mel 的 per-sample lengths 与 span mapping 需要由 MiMo-V2.5 专属 adapter 定义，不能沿用当前通用 compaction。

13. ~~**`MiMoV25AudioTokenizerConfig` 仍是旧默认维度**~~（已完成，2026-06-04：已移除不完整的 V2.5 tokenizer fallback 和 registry 入口，避免误用旧 MiMo-Audio 默认维度）
    - 文件：`python/sgl_jax/srt/multimodal/configs/config_registry.py`
    - 现象：仅覆盖 `num_quantizers=20`，其余仍是旧 MiMo-Audio 默认（`d_model=1280`、32 层、20 heads、第三 codebook=128）。
    - 触发：任何路径使用 registry 获取 V2.5 tokenizer config 都会 shape mismatch 或 RVQ 错误。
   - 修复方向：MiMo-V2.5 模式强制从 `<checkpoint>/audio_tokenizer/config.json` 加载；找不到时 hard fail。若保留 registry fallback，必须完整填入 V2.5 tokenizer config，且不能用宽泛 keyword 匹配。
   - 第一轮补充：背景是不完整 registry fallback 比 hard fail 更危险，因为会构造出看似可用但维度错误的 tokenizer；本轮已移除 `MiMoV25AudioTokenizerConfig` fallback 和对应 registry 入口。争议/遗留是 host-side tokenizer 后续应直接读取 checkpoint 的 `audio_tokenizer/config.json`，是否再提供 registry fallback 需要等真实文件树和调用入口稳定后决定。

14. ~~**AudioConfigRegistry 未注册真实子目录 basename `audio_tokenizer`**~~（已完成，2026-06-04：V2.5 tokenizer 不再走 `AudioConfigRegistry` fallback；`audio_tokenizer/` 若需支持必须直接读 HF config 或加上下文校验）
    - 文件：`python/sgl_jax/srt/multimodal/configs/config_registry.py`
    - 现象：`get_audio_config('/path/to/MiMo-V2.5/audio_tokenizer')` 时 basename 为 `audio_tokenizer`，当前不会命中。
    - 触发：若后续从子目录路径走 registry，会直接 raise。
   - 修复方向：更推荐直接读该目录的 HF config；若保留 registry 路径，增加精确 `audio_tokenizer` + 父目录/上下文校验，避免其它模型的同名子目录误命中。
   - 第一轮补充：背景是 `audio_tokenizer` 是通用子目录名，单靠 basename 注册会误匹配其它模型；本轮不再让 V2.5 tokenizer 走 `AudioConfigRegistry` fallback。争议/遗留是如果后续需要 registry 支持，必须带父 checkpoint 或 model_type 上下文校验，不能只用 basename。

15. ~~**audio RVQ id 被 clip 而非 validate**~~（已完成，2026-06-04：runner 已校验 code range，模型内已移除静默 `jnp.clip`）
    - 文件：`python/sgl_jax/srt/multimodal/models/mimo_v2_5/embedding.py`
    - 现象：非法 code id 被静默映射到 0 或 1279。
    - 触发：processor 输出错误方向/错误 tokenizer codes 时，模型产生“看似正常但错误”的 audio embedding。
   - 修复方向：host 侧校验 `audio_codes` shape、channel 数和 code range；非法 code 直接报错，不要在模型内静默 clip。
   - 第一轮补充：背景是 clip 会把 tokenizer/shape 错误伪装成合法输入，导致难以发现的音频语义错误；本轮已在 runner host 侧校验 code range，并移除模型内 `jnp.clip`。争议/遗留是当前按统一 `speech_vocab_size` 校验所有 codebook，后续如 HF tokenizer 暴露 per-codebook vocab 或特殊 token 规则，可在 host tokenizer 层细化校验。

16. ~~**audio-only 请求缺少 processor guard**~~（已完成，2026-06-04：no-processor guard 已覆盖 `audio_data`；直接传 `audio_codes` 且无原始 audio_data 的路径仍可继续）
    - 文件：`python/sgl_jax/srt/multimodal/manager/multimodal_tokenizer.py`
    - 现象：当前 no-processor guard 只检查 image/video，不检查 audio。
    - 触发：`self.mm_processor is None` 且请求只含 audio 时，后续直接调用 `self.mm_processor(...)` 导致 `TypeError`/`AttributeError`。
   - 修复方向：guard 条件加入 `audio_data`，给出明确错误；或建立 MiMo-V2.5 host-side audio processor。
   - 第一轮补充：背景是 audio-only 请求在没有 processor 时也会进入 tokenizer 路径，之前只检查 image/video 会导致后续空对象调用；本轮 no-processor guard 已覆盖 `audio_data`，但直接传 `audio_codes` 且没有原始 `audio_data` 的路径仍允许继续。争议/遗留是 host-side audio processor/tokenizer 落地后，需要明确 raw audio 与 pre-tokenized audio_codes 两种入口的错误信息和优先级。

### P2：API 完整性与后续范围，首轮可记录但不应混入 P0/P1 修复

17. **OpenAI multimodal route 丢失 logprob/top_logprobs/extra_key 等字段**
    - 文件：`python/sgl_jax/srt/entrypoints/openai/serving_chat.py`
    - 现象：multimodal 分支构造 `GenerateOmniReqInput` 只传 prompt/media/sampling/stream/rid/stop。
    - 触发：用户请求 `logprobs=true` / `top_logprobs>0` 时，下游无法返回 logprobs。
    - 修复方向：扩展 `GenerateOmniReqInput` / tokenized omni 结构；若首轮不支持，应在接口或文档里明确报错，不要静默丢弃。

18. **image/video 暂未接入**
    - 当前仍按 audio-first 范围跳过。若用户传 image/video，模型内 `NotImplementedError` 是预期行为；但对完整 omni 目标来说仍是后续任务。

19. **batch 混跑纯文本仍不支持**
    - 既有设计已说明 batch>1 / mixed pure-text 需要 token-level scatter-merge；当前 whole-segment input_embedding replace 仍不支持混批。

20. **embed-stage JIT/bucketing 仍未完成**
    - 当前仍需要按 shape bucket + mask + 去 pad 的契约实现。若 v1 固定 host-side `audio_codes`，JIT/bucketing 的首要对象应是 `[T3,20]` codes 与 `[G,4096]` audio embedding，而不是动态 mel/codec 路径。

## 仍需最终验证的事项

1. 对真实下载 checkpoint 执行 safetensors key dump，确认主 shard 与 `audio_tokenizer/model.safetensors` 的 key list 与上述推导完全一致。
2. 独立确认 ModelScope 镜像文件树是否与 HF 一致。
3. 在线复核 HF/ModelScope 页面在当前网络环境恢复后确认 `XiaomiMiMo/MiMo-V2.5` 的公开文件树；已知 `XiaomiMiMo/MiMo-V2.5-Pro` 是 text-only，不纳入 multimodal stage alias。
4. 运行 compile/test；此前 `python -m compileall` 曾因 harness 安全分类器临时不可用而未能执行。

## 本轮实现记录：V2.5 audio payload / codec processor 切片

本轮按 `mimo_v25_step2_model_integration.md` 的 4.3.1 设计先落 **codes/payload plumbing**，没有把 raw audio -> RVQ encode 直接塞进共享 `MultimodalTokenizer`。

已完成：

1. `io_struct.py` 新增 `MiMoV25AudioPayload`，并在 `GenerateOmniReqInput` / `TokenizedGenerateOmniReqInput` 上增加 `audio_payload` 字段。
2. 新增 `multimodal/manager/mimo_v25_audio_codec_processor.py`：
   - `build_payload_from_codes(...)`：把显式或 processor 返回的 codes 规范化成 `[T,20]`。
   - 校验 channel 数、code id range、`token_lengths=ceil(T/group_size)`。
   - `validate_placeholder_count(...)`：用 payload 的 `audio_token_id` 校验 `<audio_pad>` 数。
   - `encode(audio_data)` 当前 hard fail，避免误以为 raw audio codec 已接入。
3. `multimodal_tokenizer.py`：
   - MiMo-V2.5 请求若显式传 `audio_codes`，会构造 payload 并写入 `mm_inputs["mimo_v25_audio_payload"]`。
   - HF processor 若返回 `audio_codes` / `audio_code`，也走同一 payload 构造。
   - 若 MiMo-V2.5 raw audio 只得到 `audio_features/input_features` 而没有 codes，提前抛出明确错误，不再等到 embed stage 才失败。
4. `global_scheduler.py`：
   - 保留 `input.audio_codes -> req.audio_codes`。
   - 若 tokenized request 只带 `MiMoV25AudioPayload`，从 `payload.codes` 兜底回填 `Req.audio_codes`。

当前刻意不做的事：

- 不在共享 tokenizer 中动态加载 MiMo-V2.5 remote code / torch `audio_tokenizer.encode`。这条路径之前已经被复盘为容易污染共享 tokenizer，并且会把模型专属 codec 生命周期和错误处理混到通用请求预处理里。
- 不复用 `mimo_audio` 的 JAX tokenizer 作为 fallback。MiMo-Audio 是 8-channel 纯音频合同，MiMo-V2.5 是 20-channel omni audio understanding 合同，fallback 比 hard fail 更危险。
- 不在 `Req` 上新增单独 `audio_payload` 字段。当前 payload 放在 `omni_inputs` 中，stage0 仍通过已有 `Req.audio_codes` 获取模型输入；后续多段 audio/span 需要更强 per-span 校验时，再决定是否把 payload 提升为 `Req` 一等字段。

遗留问题：

1. `MiMoV25AudioCodecProcessor.encode(audio_data)` 在下一节已补 raw/mel -> RVQ codes 的懒加载代码骨架；仍需要在具备 torch/torchaudio/transformers/safetensors 与真实 checkpoint 的环境中跑通，并与 sglang/HF remote code 的 `audio_codes` 做数值/离散码对齐。
2. 当前 payload 只稳定了单段 `[T,20]` codes 的路径；多段 audio / batch audio 仍只做总量校验，需要引入 `offsets` 和 per-span `token_lengths` 后再增强。
3. `multimodal_tokenizer._is_mimo_v25_model()` 当前依赖 `model_type` 或 model path alias；后续应收敛到声明式 model registry/config capability，避免字符串判断扩散。
4. raw audio 请求现在会明确失败，这是预期中间态；完成 Slice 2 后，失败条件应改成 codec 加载失败或 HF parity 校验失败。

验证记录：

- `PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 -m compileall -q ...` 已通过本轮改动的 4 个 Python 文件。
- 本地 `python3` 是 3.9.6。上一轮直接 import `mimo_v25_audio_codec_processor` 时会先 import `io_struct.py`，而 `io_struct.py` 既有 Pydantic model 使用 `str | None` 注解；Pydantic 在 Python 3.9 下要求安装 `eval_type_backport` 或改写注解，因此 runtime smoke 曾被这个环境问题阻断。下一节已将 `MiMoV25AudioPayload` 下沉到 codec helper，codes-only helper import 不再经过 Pydantic；但完整 `multimodal_tokenizer.py` 仍依赖 `io_struct.py`，后续应在项目支持矩阵里明确 Python 3.10+，或统一迁移到 `Optional[...]`/安装 backport。

## 本轮继续实现记录：raw/mel codec helper 接入

在上一节的 payload plumbing 基础上，本轮继续推进 Slice 2 的代码骨架：`MiMoV25AudioCodecProcessor` 不再只是 hard fail，而是具备真实 raw/mel -> RVQ codes 的懒加载路径。

已完成：

1. `MiMoV25AudioPayload` 从 `io_struct.py` 下沉到 `mimo_v25_audio_codec_processor.py`，`io_struct.py` 只导入引用。这样 codec helper 可以不经过 Pydantic-heavy `io_struct` 独立 import 和测试。
2. `MiMoV25AudioCodecProcessor` 新增懒加载能力：
   - 从 `<model_path>/modeling_mimo_v2.py` 本地 remote code 加载 `MiMoAudioTokenizerConfig` / `MiMoAudioTokenizer`；若本地文件不存在，预留 `transformers.dynamic_module_utils.get_class_from_dynamic_module(...)` 路径。
   - 从 `<model_path>/audio_tokenizer/{model.safetensors,pytorch_model.bin}` 加载冻结 tokenizer 权重。
   - `encode_mels(...)` 复刻 HF `tokenize_audio_batch(...)` 的分段、group-by-length、`encoder.encode(return_codes_only=True)`、按 code length split 逻辑。
   - `encode(...)` 支持 waveform tuple、numpy/torch waveform、mel `[T,128]` / `[B,T,128]`、path/URL/base64/bytes 等输入，并用 torchaudio MelSpectrogram 参数对齐 V2.5 codec：24kHz、`n_fft=960`、`hop_length=240`、`win_length=960`、`n_mels=128`、`power=1.0`、`log(clamp(min=1e-7))`。
3. `multimodal_tokenizer.py` 的 MiMo-V2.5 分支改为：
   - 不把 audio 传给 Qwen2.5-VL `mm_processor(audio=...)`。
   - 若 processor 已返回 `audio_codes`，继续走 payload。
   - 若只有 `audio_features/input_features`，调用 `encode_mels(...)`。
   - 若只有 raw `audio_data`，调用 `encode(...)`。
   - 构造 payload 后仍执行 placeholder 数量校验，避免 codec 成功但 prompt 没有 `<audio_pad>` 时 silent 通过。

当前风险/问题：

1. **codec helper 代码已写，但真实 tokenizer encode 未在本机跑通**：当前系统 Python 环境缺少 `transformers`，也未确认 torch/torchaudio/safetensors 组合是否完整；因此只能验证 codes-only helper 路径和 Python 编译，不能声明 HF codec 数值已对齐。
2. **raw audio 端到端还依赖 chat template/audio placeholder**：本地 `/tmp/mimo_hf/preprocessor_config.json` 显示 processor_class 是 `Qwen2_5_VLProcessor`，不一定会为 `{type: audio}` 内容生成 MiMo-V2.5 的 `<audio_start>/<audio_pad>/<audio_end>`。如果 template 没有插入 audio pad，当前 tokenizer 会在 `validate_placeholder_count` 失败，这是预期保护。
3. **dynamic remote module fallback 未实测**：本地路径有 `modeling_mimo_v2.py` 时可以走 `importlib`；HF repo id 场景依赖 transformers dynamic module API，需在有依赖的环境里确认 `get_class_from_dynamic_module("modeling_mimo_v2.MiMoAudioTokenizer", ...)` 对类名解析可用。
4. **多段 span 仍未完整结构化**：`encode(...)` 会把多段 codes 沿时间维 concat，并记录 per-item `token_lengths`，但 `offsets` 仍为空；后续要把 prompt 中每段 audio span 的 `[start,end)` 写入 payload，并做逐段校验。

验证记录：

- `PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 -m compileall -q python/sgl_jax/srt/multimodal/manager/{io_struct.py,mimo_v25_audio_codec_processor.py,multimodal_tokenizer.py,global_scheduler.py}` 通过。
- codes-only smoke 通过：`build_payload_from_codes(np.zeros((20,5)))` 归一到 `(5,20)`，`token_lengths == [2]`，placeholder count `[audio,audio]` 校验通过。
- `git diff --check` 通过本轮改动文件。

## 本轮继续实现记录：audio span offsets 校验

本轮把上一节遗留的“多段 audio 只做总量校验”推进到 span-aware 校验。

已完成：

1. `MiMoV25AudioCodecProcessor.validate_placeholder_count(...)` 改为调用 `attach_offsets_from_input_ids(...)`。
2. `attach_offsets_from_input_ids(...)` 会扫描 `input_ids` 中连续的 `audio_token_id` run：
   - 总 `<audio_pad>` 数必须等于 `sum(payload.token_lengths)`。
   - run 数必须等于 `len(payload.token_lengths)`。
   - 每个 run length 必须逐段等于对应 `token_lengths[i]`。
   - 校验通过后写回 `payload.offsets=[(start,end), ...]`。
3. tokenizer 现有两处 payload 校验调用不用改，都会在校验通过后自动带上 offsets；`mm_inputs["mimo_v25_audio_payload"]` 中因此可以携带 span metadata 给后续 stage/debug 使用。

仍需注意：

- 如果 chat template 把两个 audio 段的 `<audio_pad>` 连成一个连续 run，当前会报 span count mismatch。这是刻意选择：V0 要求每段 audio 有独立 start/end 边界，避免多段 audio 的 codes 和 prompt span 无法追踪。
- `Req` 仍未新增一等 `audio_payload` 字段；offsets 目前随 `omni_inputs` 透传。若后续 stage0 scatter 需要逐段 offsets，而不仅是当前按 token mask 写入，建议把 payload 提升到 `Req` 字段，减少 dict key 依赖。

验证记录：

- codes-only smoke 已覆盖单段 offsets：`[audio,audio] -> offsets=[(1,3)]`。
- 多段 smoke 已覆盖 `token_lengths=[1,2]`：`[audio, text, audio, audio] -> offsets=[(0,1),(2,4)]`。
- `compileall` 和 `git diff --check` 见下一轮/当前命令输出。

## 本轮继续实现记录：audio pad 锚点扩展

本轮确认 `/tmp/mimo_hf/tokenizer_config.json` 中的 MiMo-V2.5 chat template 对每段 audio 只渲染一个 `<|audio_pad|>`：

```text
<|mimo_audio_start|><|audio_pad|><|mimo_audio_end|>
```

而 codec 产出的 `token_lengths[i] = ceil(T_code_i / 4)` 往往大于 1。若只做校验，raw audio 请求会在 tokenizer 阶段必然失败。因此本轮补了“单 pad 锚点扩展”：

1. `MiMoV25AudioCodecProcessor.expand_single_audio_placeholders(...)`
   - 如果 `input_ids` 中每段 audio run 都只有 1 个 `<audio_pad>`，且 run 数等于 `len(token_lengths)`，则把每个 run 扩展成 `token_lengths[i]` 个 `<audio_pad>`。
   - 如果模板已经生成了正确长度，直接保留并写 offsets。
   - 如果模板完全没有 audio pad，或生成了非 1 且不等于期望长度的 run，不自动修正，交给 span 校验报错。
2. `multimodal_tokenizer.py` 在两处 payload finalize 前调用上述扩展，再执行 `validate_placeholder_count(...)`，因此 `payload.offsets` 是基于扩展后的 `input_ids`。

这个选择的边界：

- 它不是 silent trim，也不是任意补 token；只把官方模板的“单 `<audio_pad>` 锚点”扩成 codec 决定的实际长度。
- 多段 audio 仍要求模板为每段 audio 提供独立锚点。如果两个 audio 段在模板里被合并成一个连续 run，仍会报 span count mismatch。
- `prompt` 文本字符串可能仍是扩展前的 decode 结果；真正进入调度的是扩展后的 `input_ids`。若后续日志或缓存需要 prompt/input_ids 完全一致，需要在 tokenizer 扩展后重新 decode 或单独记录 expanded prompt。

验证记录：

- 单段扩展 smoke：`token_lengths=[2]`，`[text,audio,text] -> [text,audio,audio,text]`，offsets 写为 `[(1,3)]`。
- 多段扩展 smoke：`token_lengths=[1,2]`，`[audio,text,audio] -> [audio,text,audio,audio]`，offsets 写为 `[(0,1),(2,4)]`。

## 本轮继续实现记录：MiMo-V2.5 config/token id 读取修正

本轮复核 `/tmp/mimo_hf/config.json` 后发现：真实 MiMo-V2.5 `model_type` 是通用的 `mimo_v2`，`audio_token_id=151669` 位于 `processor_config` 下，而不一定是顶层 attribute。只靠 `model_path` 包含 `mimo-v2.5` 或 `getattr(config, "audio_token_id")` 会导致两个问题：

1. 本地 checkpoint 路径不含 `mimo-v2.5` 时，`_is_mimo_v25_model()` 可能误判为 false，进而跳过 V2.5 专属 codec/payload 路径。
2. `payload.audio_token_id` 可能是 `None`，导致 `validate_placeholder_count(...)` 早退，placeholder/span 校验被绕过。

已完成：

- `multimodal_tokenizer.py` 新增 `_get_mm_config_value(...)`，优先读顶层 config，缺失时读 `config.processor_config`。
- `_is_mimo_v25_model()` 现在支持 `model_type == "mimo_v2"` 且 `audio_config` 符合 V2.5 audio understanding 结构（20 channel、group size 4、speech vocab 1280、audio token id 151669）的本地 checkpoint。
- `MiMoV25AudioCodecProcessor` 初始化、payload 构造、`mm_inputs["audio_token_id"]` 都改用统一 config 读取路径。

验证记录：

- 本轮只做静态/编译验证；真实 `AutoConfig` 实例化仍需在有 `transformers` 的环境中确认 remote config 是否已把 `processor_config.audio_token_id` 提升为 attribute。

## 本轮继续实现记录：audio payload 单元测试

本轮为 MiMo-V2.5 audio payload/helper 增加了纯 numpy/unittest 覆盖，避免后续改动破坏已落地的 host-side contract。

新增测试文件：

- `python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py`

覆盖项：

1. `[20,T]` channel-first codes 规范化为 `[T,20]`，并计算 `token_lengths=ceil(T/4)`。
2. 非法 RVQ code id 直接报错，不允许 silent clip。
3. 单段 audio 的单 `<audio_pad>` 锚点扩展为实际 token length，并写入 `offsets`。
4. 多段 audio 的多个单 pad 锚点分别扩展，并写入逐段 offsets。
5. 模板把多段 audio 合并为一个连续 run 时，span count mismatch 报错。

验证记录：

- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过，5 tests。
- `compileall` 通过本轮相关 Python 文件。
- `git diff --check` 通过。

仍未覆盖：

- 真实 `audio_tokenizer.encoder.encode(return_codes_only=True)` 数值对齐，因为当前本机 Python 环境缺少 `transformers`，且没有完整本地 checkpoint 权重可执行。
- `MultimodalTokenizer._tokenize_one_request(...)` 的 raw OpenAI audio 端到端路径，因为需要真实 tokenizer/mm_processor/chat_template 以及 torch/torchaudio/transformers 依赖。

## 本轮继续实现记录：MiMo-V2.5 tokenizer mrope 与 token id 边界修正

时间：2026-06-04 22:22:45 CST。

本轮继续对齐 `mimo_v25_step2_model_integration.md` 的 4.3.1 调用链，修正 shared `MultimodalTokenizer` 中两个会影响 MiMo-V2.5 audio V0 的边界问题。

已完成：

1. `multimodal_tokenizer.py` 新增 `_get_config_value(config, key, default)`，并让 `_get_mm_config_value(...)` 同时支持：
   - 顶层 `mm_config.<key>`；
   - 顶层 `mm_config.processor_config.<key>`；
   - `mm_config.thinker_config.<key>`；
   - `mm_config.thinker_config.processor_config.<key>`。
2. MiMo-V2.5 请求在 tokenizer 阶段不再计算 Qwen/Qwen3-Omni 的 `mrope_positions`。
   - MiMo-V2.5 设计与 HF 审计均按 1-D RoPE 进入 AR stage。
   - tokenizer 只保留扩展后的 `input_ids` 与 placeholder span，不产出 mrope。
3. `mm_inputs` 中的 `vision_start_token_id` / `vision_end_token_id` / `image_token_id` / `video_token_id` / `audio_token_id` 改为统一通过 `_get_mm_config_value(...)` 读取，避免真实 MiMo-V2.5 config 中 token id 位于 `processor_config` 时读成 `None`。
4. MiMo-V2.5 分支不再把 processor/mel 侧的 `audio_features` 作为 `MultimodalDataItem(AUDIO)` 写入 `mm_items`。
   - V2.5 stage0 的唯一 audio 输入是 `audio_codes` / `MiMoV25AudioPayload`。
   - Qwen3-Omni 等连续 `audio_features` 合同仍保留原行为。

设计影响：

- `audio -> mel -> audio_codes` 仍在 host-side codec helper 完成。
- `audio_codes -> audio embedding` 仍在 `MiMoV2_5Embedding` / audio understanding tower 完成。
- shared tokenizer 只负责 prompt/template、payload 构造、pad span 扩展、token id 读取和契约校验；不把 MiMo-V2.5 audio 请求混入 Qwen mrope 或 Qwen3-Omni audio feature contract。

验证记录：

- `PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 -m compileall -q python/sgl_jax/srt/multimodal/manager/multimodal_tokenizer.py python/sgl_jax/srt/multimodal/manager/io_struct.py python/sgl_jax/srt/multimodal/manager/global_scheduler.py python/sgl_jax/srt/multimodal/manager/mimo_v25_audio_codec_processor.py` 通过。
- 最终复跑时同时覆盖了 `python/sgl_jax/srt/multimodal/models/mimo_v2_5/embedding.py`、`python/sgl_jax/srt/multimodal/model_executor/embed/embed_model_runner.py`、`python/sgl_jax/srt/multimodal/manager/stage.py` 的 `compileall`。
- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过，5 tests。
- `git diff --check -- python/sgl_jax/srt/multimodal/manager/multimodal_tokenizer.py` 通过。

仍未覆盖：

- 真实 `AutoConfig` / `AutoProcessor` 下的 MiMo-V2.5 raw audio request 端到端验证，仍依赖具备 `transformers`、torch/torchaudio/safetensors 和真实 checkpoint 的 Python 3.12+ 环境。
- HF/sglang remote code 的 `audio_tokenizer.encode(return_codes_only=True)` 离散码数值对齐。

## 本轮继续实现记录：audio payload 提升为 Req 一等字段

时间：2026-06-04 22:28:41 CST。

本轮继续推进 4.3.1 的 server/tokenizer/scheduler/stage0 调用链，把 MiMo-V2.5 audio payload 从 `omni_inputs` 内的 debug/metadata 字段提升为 `Req` 的一等字段。

注意：前文历史记录中“`Req` 仍未新增一等 `audio_payload` 字段”的状态已被本节实现更新覆盖；保留旧记录只是为了保留过程脉络。

已完成：

1. `schedule_batch.py::Req` 新增 `audio_payload: MiMoV25AudioPayload | dict | None`。
2. `global_scheduler.py::convert_omni_request(...)`：
   - 将 `TokenizedGenerateOmniReqInput.audio_payload` 规范化为 `MiMoV25AudioPayload`。
   - 写入 `req.audio_payload`。
   - 若 `req.audio_codes` 为空，从 `payload.codes` 回填。
   - 若 `req.omni_inputs` 为空，创建 dict 并写入 `mimo_v25_audio_payload`，保证后续 embed stage 仍能识别这是 omni prefill。
3. `embed_model_runner.py`：
   - MiMo-V2.5 分支优先从 `Req.audio_payload` 取 payload，缺失时再从 `omni_inputs["mimo_v25_audio_payload"]` 取。
   - 若 `Req.audio_codes` 为空但 payload 存在，则用 `payload.codes` 作为 JAX audio tower 输入。
   - payload 存在时，进入 JAX 前调用 `MiMoV25AudioCodecProcessor.attach_offsets_from_input_ids(...)` 做 span-aware 校验。

本轮发现并修正的细节：

- 多段 audio payload 的 `payload.codes` 是拼接后的 `[sum(T_i),20]`，不能用 `ceil(sum(T_i)/4)` 推导 placeholder 数。
- 正确规则是 `sum(payload.token_lengths) == sum(ceil(T_i/4))`，并且每段 offset 长度必须等于对应 `token_lengths[i]`。
- 因此 runner 的 payload 校验现在使用 `payload.token_lengths/offsets` 校验 span 数量和长度；`payload.codes` 只额外做 channel/code range 校验。
- 无 payload 的历史兜底路径仍保留旧行为：直接根据 `audio_codes` shape 推导总 placeholder 数，用于单段或简单 3D batch codes。

验证记录：

- `PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 -m compileall -q python/sgl_jax/srt/multimodal/model_executor/embed/embed_model_runner.py python/sgl_jax/srt/multimodal/manager/schedule_batch.py python/sgl_jax/srt/multimodal/manager/global_scheduler.py` 通过。
- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过，5 tests。
- `git diff --check -- python/sgl_jax/srt/multimodal/manager/schedule_batch.py python/sgl_jax/srt/multimodal/manager/global_scheduler.py python/sgl_jax/srt/multimodal/model_executor/embed/embed_model_runner.py` 通过。

环境限制：

- 当前本机 `python3` 环境缺少 `jax`，直接 import `EmbedModelRunner` 的 runtime smoke 会失败：`ModuleNotFoundError: No module named 'jax'`。因此本轮只能做语法编译和 codec helper 单测；runner runtime 仍需在项目 Python 3.12+ / JAX 环境中验证。

## 本轮继续实现记录：多段 audio codes 按段 padding 后再进入 stage0

时间：2026-06-04 22:32:23 CST。

本轮复核 `MiMoV25AudioPayload` 与 `MiMoV2_5AudioUnderstandingEncoder._group_audio_codes(...)` 的契约时发现一个更细的多段 audio 风险：

- tokenizer/payload 侧的 placeholder 规则是逐段 `token_lengths[i] = ceil(T_i / 4)`。
- 但 stage0 audio encoder 只看到一个连续 `audio_codes` tensor，并按整个时间轴做 `group_size=4` 分组。
- 如果 host 侧直接把多段 raw codes concat 成 `[sum(T_i),20]`，当某段 `T_i % 4 != 0` 时，stage0 会把第一段尾部和下一段开头放进同一个 group，输出行数也会变成 `ceil(sum(T_i)/4)`，不再等于 `sum(ceil(T_i/4))`。

已完成：

1. `MiMoV25AudioCodecProcessor.build_payload_from_codes(...)` 支持单段 codes 和多段 codes list。
2. 每段 codes 先独立规范化为 `[T_i,20]` 并校验 channel / code range。
3. 每段 codes 独立 pad 到 `group_size=4` 的倍数，pad 值重复该段最后一帧 code。
4. `payload.codes` 改为 stage0-ready codes：`concat(pad_to_group_size(segment_i))`。
5. `payload.token_lengths` 仍记录未 pad 前每段的 `ceil(T_i/4)`，用于 placeholder span 校验。
6. `encode_mels(...)` 不再手写 concat + 覆盖 token_lengths，而是直接调用 `build_payload_from_codes(codes_np, ...)`，统一走同一套分段 padding 逻辑。

同步更新：

- `docs/design/mimo_v25_step2_model_integration.md` 的 4.3.1 已明确区分 raw code segments `[T_i,20]` 和 stage0-ready `payload.codes [T_pad,20]`。

新增测试：

- `test_build_payload_pads_each_audio_segment_before_concat`
  - 覆盖两段 codes：`T_1=5`、`T_2=6`。
  - 期望 `payload.token_lengths == [2,2]`。
  - 期望 `payload.codes.shape == (16,20)`，且第一段与第二段分别重复自己的最后一帧做 padding，避免跨段 grouping。

验证记录：

- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过，6 tests。
- `PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 -m compileall -q python/sgl_jax/srt/multimodal/manager/mimo_v25_audio_codec_processor.py python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py python/sgl_jax/srt/multimodal/manager/multimodal_tokenizer.py python/sgl_jax/srt/multimodal/model_executor/embed/embed_model_runner.py` 通过。
- `git diff --check -- python/sgl_jax/srt/multimodal/manager/mimo_v25_audio_codec_processor.py python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过。

仍未覆盖：

- stage0 真实 JAX runtime 中 `payload.codes` 进入 `MiMoV2_5AudioUnderstandingEncoder` 后的 `audio_embeds.shape[0] == sum(payload.token_lengths)`，当前环境缺少 `jax`，只能通过 host-side payload shape 与单元测试间接保证。

## 本轮继续实现记录：预编码 audio_codes 支持 list / 3D batch 输入

时间：2026-06-04 22:36:21 CST。

上一节把 `payload.codes` 改成“每段独立 pad 后再 concat”的 stage0-ready tensor。本轮继续补齐入口形态：显式传入或 processor 返回的 `audio_codes` 不能只支持单段 2D，否则多段 codes 会在进入 payload helper 前被 shared tokenizer 拒绝。

已完成：

1. `MiMoV25AudioCodecProcessor.normalize_code_segments(...)`
   - 支持单段 2D `[T,20]` / `[20,T]`。
   - 支持多段 list / tuple，每个元素可以是 `[T_i,20]` 或 `[20,T_i]`。
   - 支持 3D batch ndarray `[B,T,20]` / `[B,20,T]`，按 batch 维拆成多段 segment。
   - ragged list 在不同 NumPy 版本下可能抛 `ValueError` 或变成 object array，本轮都按 segment list 处理。
2. `multimodal_tokenizer.py::_normalize_audio_codes(...)`
   - 放宽为允许 3D batch codes。
   - ragged list-of-segments 不再提前报错，而是保留为 list[np.ndarray]，交给 MiMo-V2.5 payload helper 做 channel/range/padding 校验。
3. `docs/design/mimo_v25_step2_model_integration.md` 的 Slice 1 输入支持已同步为：2D 单段、list 多段、3D batch 三种预编码形态。

新增测试：

- `test_build_payload_splits_batched_time_major_codes`
  - 覆盖 `[B,T,20]`，确认按 batch 拆段后每段独立 padding。
- `test_build_payload_splits_batched_channel_major_codes`
  - 覆盖 `[B,20,T]`，确认 channel-first batch 也能按段归一到 `[T,20]`。

验证记录：

- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过，8 tests。
- `PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 -m compileall -q python/sgl_jax/srt/multimodal/manager/mimo_v25_audio_codec_processor.py python/sgl_jax/srt/multimodal/manager/multimodal_tokenizer.py python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过。
- `git diff --check -- python/sgl_jax/srt/multimodal/manager/mimo_v25_audio_codec_processor.py python/sgl_jax/srt/multimodal/manager/multimodal_tokenizer.py python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过。

仍未覆盖：

- shared `MultimodalTokenizer._tokenize_one_request(...)` 的 runtime 测试仍依赖完整 tokenizer/mm_processor 初始化环境；本轮只验证了 payload helper 的 host-side shape contract。

## 本轮继续实现记录：tokenizer 预规范化保留 ragged/list audio_codes

时间：2026-06-04 22:39:04 CST。

本轮检查 `audio_payload` 从 tokenizer 到 scheduler/stage0 的 handoff，确认 payload-only 请求会在 `GlobalScheduler.convert_omni_request(...)` 中创建 `req.omni_inputs` 并写入 `mimo_v25_audio_payload`，不会因为空 dict truthiness 直接丢失 omni 身份。

同时发现一个更靠前的入口问题：上一节虽然让 `MiMoV25AudioCodecProcessor.normalize_code_segments(...)` 支持 list / 3D batch，但 shared tokenizer 的 `_normalize_audio_codes(...)` 仍可能在 payload helper 前把 ragged list 或 list-of-tensors 转成 object array 后报错。

已完成：

1. `multimodal_tokenizer.py` 新增 `_normalize_audio_code_segment_list(...)`。
2. `_normalize_audio_codes(...)` 遇到 ragged list 抛 `ValueError` 时，不再直接失败，而是保留为 `list[np.ndarray]`。
3. `_normalize_audio_codes(...)` 遇到 NumPy object array 且原输入是 list/tuple 时，也保留为 `list[np.ndarray]`。
4. 后续 channel 数、code range、按段 padding、token_lengths 仍由 `MiMoV25AudioCodecProcessor.build_payload_from_codes(...)` 统一校验和处理。

设计影响：

- tokenizer 的 pre-normalize 只做“可搬运形态”转换，不抢 MiMo-V2.5 payload helper 的最终 contract 校验。
- 预编码 `audio_codes` 的三种入口保持一致：2D 单段、list 多段、3D batch。

验证记录：

- `PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 -m compileall -q python/sgl_jax/srt/multimodal/manager/multimodal_tokenizer.py python/sgl_jax/srt/multimodal/manager/mimo_v25_audio_codec_processor.py` 通过。
- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过，8 tests。
- `git diff --check -- python/sgl_jax/srt/multimodal/manager/multimodal_tokenizer.py` 通过。

仍未覆盖：

- `_tokenize_one_request(...)` 的真实 runtime 单测仍依赖当前环境缺失的完整 tokenizer/mm_processor/JAX 运行栈；本轮通过语法编译和 payload helper 单测覆盖可验证部分。

## 本轮继续实现记录：直接传 audio_payload 的 stage0-ready 规范化

时间：2026-06-04 22:43:48 CST。

本轮继续检查 `audio_payload` 直传路径。风险点是：如果调用方绕过 `audio_codes` list/3D 入口，直接传 `MiMoV25AudioPayload(codes=..., token_lengths=...)`，旧路径会跳过 `build_payload_from_codes(...)` 的按段 padding，可能让多段 raw-concat codes 再次进入 stage0 并跨段 grouping。

已完成：

1. `MiMoV25AudioCodecProcessor.normalize_payload(...)`
   - 规范化 payload 的 `codes` channel layout 与 code range。
   - 校验 `token_lengths` 必须为正。
   - 单段 payload 若传入未 pad codes，会自动按 `group_size` pad 成 stage0-ready `codes`。
   - 多段 payload 要求 `codes.shape[0] == sum(token_lengths) * group_size`；否则报错，提示改传 list/3D `audio_codes` 或已经按段 pad 后的 `payload.codes`。
2. `multimodal_tokenizer.py`
   - 对 request 里直接带的 `audio_payload` 调用 `_normalize_mimo_v25_audio_payload(...)`，再继续做 placeholder 扩展与 span 校验。
3. `global_scheduler.py`
   - 对 tokenized request 中的 payload 再做一次 `normalize_payload(...)` 兜底，并将规范化结果写入 `req.audio_payload` 和 `omni_inputs["mimo_v25_audio_payload"]`。
4. `embed_model_runner.py`
   - MiMo-V2.5 runner 取到 payload 后再次 normalize，确保 `audio_codes` 传入 JAX 前是 stage0-ready。

新增测试：

- `test_normalize_payload_pads_single_unpadded_payload`
  - 单段 `codes.shape=(5,20)`、`token_lengths=[2]` 自动 pad 到 `(8,20)`。
- `test_normalize_payload_rejects_multi_segment_raw_concat`
  - 多段 `token_lengths=[2,2]` 但 `codes.shape=(11,20)` 直接报错，避免无法恢复边界的 raw concat 进入 stage0。

验证记录：

- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过，10 tests。
- `PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 -m compileall -q python/sgl_jax/srt/multimodal/manager/mimo_v25_audio_codec_processor.py python/sgl_jax/srt/multimodal/manager/multimodal_tokenizer.py python/sgl_jax/srt/multimodal/manager/global_scheduler.py python/sgl_jax/srt/multimodal/model_executor/embed/embed_model_runner.py python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过。
- `git diff --check -- python/sgl_jax/srt/multimodal/manager/mimo_v25_audio_codec_processor.py python/sgl_jax/srt/multimodal/manager/multimodal_tokenizer.py python/sgl_jax/srt/multimodal/manager/global_scheduler.py python/sgl_jax/srt/multimodal/model_executor/embed/embed_model_runner.py python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过。

仍未覆盖：

- `global_scheduler.py` / `embed_model_runner.py` 的 runtime import 仍受当前环境缺少 `jax` 限制；本轮用 compileall 与 host-side helper 单测覆盖可验证部分。

## 本轮继续实现记录：runner/scheduler audio_token_id fallback

时间：2026-06-04 22:46:27 CST。

前面已经修正 tokenizer 从 `processor_config.audio_token_id` 读取真实 MiMo-V2.5 的 audio token id。但继续审查后发现 stage0 侧仍有两个 fallback 缺口：

1. `EmbedModelRunner._validate_mimo_v25_audio_placeholder_count(...)` 只读 `model_config.audio_token_id` 顶层字段；若真实 config 只把 `audio_token_id=151669` 放在 `processor_config` 下，runner 的 placeholder count 校验会早退。
2. `GlobalScheduler.convert_omni_request(...)` 对 tokenized payload 做 `normalize_payload(...)` 兜底时，没有从 `mm_inputs["audio_token_id"]` 补回 payload 的 token id。

已完成：

- `embed_model_runner.py` 新增 `_get_config_value(...)`，读取顺序为顶层 config，缺失时读 `processor_config`。
- MiMo-V2.5 runner 的 placeholder count 校验改用 `_get_config_value(model_config, "audio_token_id")`。
- MiMo-V2.5 runner normalize payload 时会把 fallback 得到的 `audio_token_id` 传入 `MiMoV25AudioCodecProcessor.normalize_payload(...)`。
- `global_scheduler.py` normalize tokenized payload 时，如果 `input.mm_inputs` 是 dict，则优先用 `input.mm_inputs["audio_token_id"]` 填充 payload token id。

设计影响：

- tokenizer、scheduler、runner 三层现在都不会只依赖顶层 `audio_token_id`。
- 即使真实 MiMo-V2.5 config 把 audio token id 放在 `processor_config`，stage0 前置校验也不会因为 token id 缺失而静默跳过。

验证记录：

- `PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 -m compileall -q python/sgl_jax/srt/multimodal/model_executor/embed/embed_model_runner.py python/sgl_jax/srt/multimodal/manager/global_scheduler.py python/sgl_jax/srt/multimodal/manager/mimo_v25_audio_codec_processor.py` 通过。
- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过，10 tests。
- `git diff --check -- python/sgl_jax/srt/multimodal/model_executor/embed/embed_model_runner.py python/sgl_jax/srt/multimodal/manager/global_scheduler.py` 通过。

仍未覆盖：

- 需要在有 `jax` 和真实 `AutoConfig` 的环境中确认 `processor_config.audio_token_id` 通过 HF config 对象暴露方式与本 fallback 兼容。

## 本轮继续实现记录：GlobalScheduler 使用规范化后的 req.omni_inputs

时间：2026-06-04 22:48:27 CST。

本轮继续检查 `GlobalScheduler.convert_omni_request(...)` 的后半段字段填充。前面已经会在 payload-only 请求中创建 `req.omni_inputs = {}` 并写入 `mimo_v25_audio_payload`，但后半段仍以 `input.mm_inputs` 为判定和读取来源。

风险：

- 如果 tokenized request 没有原始 `mm_inputs`，但有 `audio_payload`，前半段会创建规范化后的 `req.omni_inputs`。
- 后半段若继续只读 `input.mm_inputs`，就无法统一处理规范化后的 payload dict、token id、grid/cache 字段。
- 当前没有 `mm_items` 时 `pad_input_tokens(...)` 会返回原始 ids，这是可接受的；问题在于同一个函数内部存在两个不同的 multimodal source of truth。

已完成：

- `global_scheduler.py::convert_omni_request(...)` 在 payload 规范化后引入局部 `mm_inputs = req.omni_inputs if isinstance(req.omni_inputs, dict) else None`。
- 后续 `mm_items`、`image_grid_thw`、`video_grid_thw`、`im_token_id`、`video_token_id`、`audio_token_id` 统一从 `mm_inputs` 读取。
- payload-only 请求现在会沿用前半段创建的 `req.omni_inputs`，而不是因为 `input.mm_inputs is None` 跳过后半段的统一路径。

验证记录：

- `PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 -m compileall -q python/sgl_jax/srt/multimodal/manager/global_scheduler.py` 通过。
- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过，10 tests。
- `git diff --check -- python/sgl_jax/srt/multimodal/manager/global_scheduler.py` 通过。

仍未覆盖：

- 需要在完整 runtime 环境中构造 payload-only `TokenizedGenerateOmniReqInput`，确认 `Req.to_stage_reqs("auto_regressive")` 后 `mm_inputs` 仍携带 `mimo_v25_audio_payload` 与 stage0 生成的 `multimodal_embedding`。

## 本轮继续实现记录：单段 audio_payload 不再静默改写 token_lengths

时间：2026-06-04 22:54:25 CST。

本轮继续收紧直接传 `MiMoV25AudioPayload` 的合同。此前 `normalize_payload(...)` 对单段未 pad `codes` 会重新调用 `build_payload_from_codes(...)`，这能自动 padding，但也会从 `codes.shape[0]` 重新推导 `token_lengths`。如果调用方传入的 `token_lengths` 本身错误，旧逻辑可能静默改写 span 长度合同，后续 tokenizer/server 侧错误更难定位。

已完成：

1. `MiMoV25AudioCodecProcessor.build_payload_from_codes(...)`
   - 显式校验 `group_size > 0`，避免在 `ceil(T / group_size)` 处出现不清晰的除零错误。
2. `MiMoV25AudioCodecProcessor.normalize_payload(...)`
   - 显式校验 `group_size > 0`。
   - 单段未 pad payload 只在 `token_lengths[0] == ceil(codes_rows / group_size)` 时自动 padding。
   - 如果单段 payload 的 `token_lengths` 与 raw `codes` 长度不一致，直接报 `token_lengths mismatch`，不再静默重建 token_lengths。
   - 自动 padding 时保留原 payload 的 `offsets`、`source`、`is_tokenized`、`audio_token_id` 等 metadata。

新增/调整测试：

- `test_normalize_payload_pads_single_unpadded_payload`
  - 覆盖单段未 pad payload 自动 padding，并确认 `offsets/source/is_tokenized` 不丢失。
- `test_normalize_payload_rejects_single_payload_token_length_mismatch`
  - 覆盖 `codes.shape=(5,20)` 但 `token_lengths=[3]` 的错误输入，要求前置失败。

验证记录：

- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过，11 tests。
- `PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 -m compileall -q python/sgl_jax/srt/multimodal/manager/mimo_v25_audio_codec_processor.py python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过。

仍未覆盖：

- 真实 `MiMoV25AudioPayload` 从 OpenAI server / tokenizer / scheduler 一路进入 `EmbedModelRunner` 的 runtime 路径仍依赖完整 JAX 与 serving 环境，本轮只覆盖 host-side payload helper 的可验证合同。

## 本轮继续实现记录：统一 audio_payload dict 反序列化入口

时间：2026-06-04 22:57:43 CST。

本轮继续检查 `MiMoV25AudioPayload` 在 `multimodal_tokenizer -> GlobalScheduler -> EmbedModelRunner` 之间的跨层传递。发现三处都在直接使用 `MiMoV25AudioPayload(**payload_dict)`。这对 Python 内存态对象可用，但对真实跨进程/JSON 传输不够稳：`codes` 可能是 list-of-lists，`offsets` 可能是 list-of-lists，`token_lengths` 和 token/config 字段也可能是字符串或 JSON number；如果每层各自处理，容易形成不一致的 payload contract。

已完成：

1. `MiMoV25AudioPayload.from_obj(...)`
   - 支持 `None`、已有 dataclass、dict 三种输入。
   - dict 里的 `codes` 统一转成 `np.ndarray(dtype=np.int32)`。
   - `token_lengths` 统一转成 `list[int]`。
   - `offsets` 统一转成 `list[tuple[int, int]]`。
   - `audio_token_id/num_channels/codebook_size/group_size` 统一转成 `int`。
2. `multimodal_tokenizer.py`
   - request 直传 `audio_payload` 时改走 `MiMoV25AudioPayload.from_obj(...)`，再进入 `normalize_payload(...)`。
3. `global_scheduler.py`
   - tokenized request 进入 `Req` 前改走同一个 `from_obj(...)`，避免 scheduler 层持有未规范化的 dict payload。
4. `embed_model_runner.py`
   - 从 `batch.audio_payload` 或 `omni_inputs["mimo_v25_audio_payload"]` 取 payload 时也改走 `from_obj(...)`，作为 stage0 前的兜底。

新增测试：

- `test_payload_from_obj_normalizes_json_transport_shape`
  - 覆盖 JSON/dict 形态的 `codes/token_lengths/offsets/audio_token_id/group_size` 被统一恢复，并继续通过 `normalize_payload(...)`。

验证记录：

- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过，12 tests。
- `PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 -m compileall -q python/sgl_jax/srt/multimodal/manager/mimo_v25_audio_codec_processor.py python/sgl_jax/srt/multimodal/manager/multimodal_tokenizer.py python/sgl_jax/srt/multimodal/manager/global_scheduler.py python/sgl_jax/srt/multimodal/model_executor/embed/embed_model_runner.py python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过。
- `git diff --check -- python/sgl_jax/srt/multimodal/manager/mimo_v25_audio_codec_processor.py python/sgl_jax/srt/multimodal/manager/multimodal_tokenizer.py python/sgl_jax/srt/multimodal/manager/global_scheduler.py python/sgl_jax/srt/multimodal/model_executor/embed/embed_model_runner.py python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py docs/design/mimo_v25_implementation_notes.md` 通过。
- `rg -n "MiMoV25AudioPayload\\(\\*\\*" ...` 确认当前 audio payload dict 反序列化点已统一替换为 `from_obj(...)`。

仍未覆盖：

- 真实 ZMQ/HTTP 边界是否会把 `np.ndarray` 自动序列化为 list 仍需在完整 serving runtime 下确认；本轮只把接收侧 contract 收敛到单一 helper。

## 本轮继续实现记录：MiMoV25AudioPayload 成为 audio_codes 的 source of truth

时间：2026-06-04 23:01:01 CST。

本轮继续检查 `audio_payload` 与裸 `audio_codes` 同时存在时的优先级。发现前面虽然已经会把 payload 规范化为 stage0-ready codes，但后续仍可能保留请求中的裸 `audio_codes`：

- `multimodal_tokenizer.py`：如果请求同时带 `audio_payload` 和 `audio_codes`，payload 规范化后不会覆盖已有 `audio_codes`。
- `global_scheduler.py`：`req.audio_codes` 先来自 `input.audio_codes`，只有为空时才从 normalized payload 回填。
- `embed_model_runner.py`：`raw_audio_codes` 优先取 `batch.audio_codes`，payload 只作为 fallback。

风险：

- 裸 `audio_codes` 可能是未按段 pad 的原始 codes，也可能与 payload 的 `token_lengths/offsets` 不一致。
- Stage0 runner 若优先使用裸 codes，就可能绕过 payload helper 已经完成的 stage0-ready contract。
- 多段音频时尤其危险：payload 已保证“每段独立 pad 后 concat”，裸 codes 可能仍是 raw concat，从而重新引入跨段 grouping。

已完成：

1. `multimodal_tokenizer.py`
   - MiMo-V2.5 下，只要存在 normalized `MiMoV25AudioPayload`，就让 `audio_codes = audio_payload.codes`。
   - 保留 `is_mimo_v25` guard，避免非 MiMo-V2.5 请求误带 payload 时影响其它模型。
2. `global_scheduler.py`
   - tokenized request 带 payload 时，normalize 后无条件 `req.audio_codes = req.audio_payload.codes`。
3. `embed_model_runner.py`
   - MiMo-V2.5 payload 存在时，runner 优先使用 `mimo_v25_audio_payload.codes`；只有没有 payload 时才回退到 `batch.audio_codes` 的旧裸字段校验路径。

设计影响：

- `MiMoV25AudioPayload` 正式成为 MiMo-V2.5 audio 链路的跨层 source of truth。
- 裸 `audio_codes` 仍可作为测试/内部入口，但进入 payload 后就必须服从 payload 的 stage0-ready 形态。
- 这与设计文档里“不要复用旧 MiMo-Audio 8-channel `Req.audio_codes` 语义”的边界一致。

验证记录：

- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过，12 tests。
- `PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 -m compileall -q python/sgl_jax/srt/multimodal/manager/multimodal_tokenizer.py python/sgl_jax/srt/multimodal/manager/global_scheduler.py python/sgl_jax/srt/multimodal/model_executor/embed/embed_model_runner.py python/sgl_jax/srt/multimodal/manager/mimo_v25_audio_codec_processor.py` 通过。
- `git diff --check -- python/sgl_jax/srt/multimodal/manager/multimodal_tokenizer.py python/sgl_jax/srt/multimodal/manager/global_scheduler.py python/sgl_jax/srt/multimodal/model_executor/embed/embed_model_runner.py docs/design/mimo_v25_implementation_notes.md` 通过。

仍未覆盖：

- 需要完整 runtime 构造“同时带 payload 和裸 audio_codes 且二者不一致”的 tokenized request，确认 stage0 实际使用 payload codes；当前环境缺少 JAX runtime，只能用代码审查、compileall 和 helper 单测覆盖可验证部分。

## 本轮继续实现记录：OpenAI input_audio 入口接入 audio_data

时间：2026-06-04 23:06:15 CST。

本轮从 server 入口继续检查 audio 数据是否能进入后续 `multimodal_tokenizer -> codec helper` 链路。已有 `audio_url` content part 会被 `process_content_for_template_format(...)` 抽取到 `audio_data`，但 OpenAI 常见的 base64 audio content part 是 `input_audio`，当前 protocol 和模板处理都没有覆盖。这样会导致 base64 audio 请求在进入 `GenerateOmniReqInput.audio_data` 之前就丢失或解析失败。

已完成：

1. `entrypoints/openai/protocol.py`
   - 新增 `ChatCompletionMessageContentInputAudio` 和 `ChatCompletionMessageContentInputAudioPart`。
   - `ChatCompletionMessageContentPart` union 增加 `input_audio`，形态为 `{"type":"input_audio","input_audio":{"data":"...","format":"wav"}}`。
   - 文件顶部补 `from __future__ import annotations`，降低本地 Python 3.9 对现有 `|` 注解的类定义求值问题；但 pydantic 在 Python 3.9 下仍需要 `eval_type_backport` 才能完整运行该 protocol 文件。
2. `jinja_template_utils.py`
   - `process_content_for_template_format(...)` 新增 `input_audio` 分支。
   - 将 `input_audio.data` 追加到 `audio_data`，并把模板内容归一为 `{"type":"audio"}`，复用现有 audio placeholder 模板路径。
3. `multimodal_tokenizer.py`
   - `_load_audio_from_source(...)` 增加 base64 和 data URI 支持。
   - `audio_url`、`input_audio` 进入后都能被解码为 bytes 并交给 `librosa.load(...)`；MiMo-V2.5 路径也可继续由 `MiMoV25AudioCodecProcessor._load_waveform(...)` 处理 base64/data URI。

新增测试：

- `python/sgl_jax/test/test_openai_audio_content_parts.py`
  - `test_template_processing_extracts_input_audio_data` 实际验证 `input_audio` 被抽取到 `audio_data` 并在模板内容中变成 `{"type":"audio"}`。
  - `test_chat_request_accepts_input_audio_part` 验证 protocol 解析；当前本地 Python 3.9 + pydantic 无法 evaluate 项目既有 `str | None` 注解，测试会 skip，并记录需要 Python 3.10+ 或 `eval_type_backport`。

验证记录：

- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/test_openai_audio_content_parts.py` 通过，2 tests，1 skipped。
- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过，12 tests。
- `PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 -m compileall -q python/sgl_jax/srt/entrypoints/openai/protocol.py python/sgl_jax/srt/jinja_template_utils.py python/sgl_jax/srt/multimodal/manager/multimodal_tokenizer.py python/sgl_jax/test/test_openai_audio_content_parts.py` 通过。
- `git diff --check -- python/sgl_jax/srt/entrypoints/openai/protocol.py python/sgl_jax/srt/jinja_template_utils.py python/sgl_jax/srt/multimodal/manager/multimodal_tokenizer.py python/sgl_jax/test/test_openai_audio_content_parts.py docs/design/mimo_v25_implementation_notes.md` 通过。

仍未覆盖：

- 当前没有完整 OpenAI serving runtime e2e 验证 `input_audio` 请求经 HTTP 进入 `GenerateOmniReqInput.audio_data` 再到 MiMo-V2.5 codec helper。
- 本地 Python 3.9 环境缺少 `jax`，也无法完整 import serving/tokenizer runtime；本轮只覆盖可隔离的协议/模板前置路径和语法检查。

## 本轮继续实现记录：stage0 MiMoV2_5Embedding token id 读取对齐 processor_config

时间：2026-06-04 23:08:39 CST。

本轮继续检查 stage0 audio scatter。`EmbedModelRunner` 之前已经改成从顶层 config 或 `processor_config` 读取 `audio_token_id`，用于 host 侧 placeholder 校验；但 `MiMoV2_5Embedding` 自己在 `_merge_audio(...)` 中使用的 `self.audio_token_id` 仍只读顶层 `config.audio_token_id`，否则 fallback 到默认 151669。真实 MiMo-V2.5 token id 可能在 `processor_config` 下，如果模型侧和 runner 侧读取规则不一致，会出现“runner 校验通过，但模型按另一个 token id scatter”的风险。

已完成：

1. `multimodal/models/mimo_v2_5/embedding.py`
   - 新增本地 `_get_config_value(config, key, default)`。
   - 读取顺序与 `EmbedModelRunner` 对齐：先读顶层 config，再读 dict/object 形态的 `processor_config`。
   - `audio_token_id`、`image_token_id`、`video_token_id` 改为通过该 helper 读取。

设计影响：

- stage0 模型侧 scatter token id 与 tokenizer/runner 的 config fallback 保持一致。
- 当前 audio scatter 仍按 `audio_token_id` 的位置顺序写入；payload offsets 已在 host/runner 前置校验，stage0 JAX hot path 不额外接收 offsets。

验证记录：

- `PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 -m compileall -q python/sgl_jax/srt/multimodal/models/mimo_v2_5/embedding.py` 通过。
- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过，12 tests。
- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/test_openai_audio_content_parts.py` 通过，2 tests，1 skipped。
- `git diff --check -- python/sgl_jax/srt/multimodal/models/mimo_v2_5/embedding.py docs/design/mimo_v25_implementation_notes.md` 通过。

仍未覆盖：

- 本地环境缺少 `jax`，不能实际 import/instantiate `MiMoV2_5Embedding` 验证 processor_config fallback 的 runtime 行为；本轮只能通过源码检查和 compileall 验证可覆盖部分。

## 本轮继续实现记录：audio_payload transport 字段进一步规范化

时间：2026-06-04 23:11:30 CST。

本轮继续收紧 `MiMoV25AudioPayload.from_obj(...)` 这个跨层 payload 反序列化入口。前面已经把 dict 形态的 `codes/token_lengths/offsets/audio_token_id/group_size` 做了基础恢复，但还有两个 transport 细节没有覆盖：

- `is_tokenized` 经过 JSON/HTTP/ZMQ 边界后可能是字符串 `"true"` / `"false"`，如果原样保留，后续代码会把非空字符串当 truthy。
- `offsets` 原样 map 成 tuple 时没有校验长度；错误的 `[start, end, extra]` 会被更晚才发现，错误信息也不直接。

已完成：

1. `MiMoV25AudioPayload.from_obj(...)`
   - `offsets` 必须是长度为 2 的 `[start, end]` pair，否则直接报 `MiMo-V2.5 audio payload offsets must be [start, end] pairs`。
   - `is_tokenized` 支持字符串 `"true"/"1"/"yes"` 和 `"false"/"0"/"no"`，统一转成 bool。
   - 其它字符串值直接报错，避免 silent truthy。
2. `test_mimo_v25_audio_codec_processor.py`
   - `test_payload_from_obj_normalizes_json_transport_shape` 改为用 `"false"` 字符串覆盖 bool coercion，并确认 normalize 后为 `False`。
   - 新增 `test_payload_from_obj_rejects_invalid_offset_shape`，覆盖非法 offset 形态前置失败。

验证记录：

- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过，13 tests。
- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/test_openai_audio_content_parts.py` 通过，2 tests，1 skipped。
- `PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 -m compileall -q python/sgl_jax/srt/multimodal/manager/mimo_v25_audio_codec_processor.py python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过。
- `git diff --check -- python/sgl_jax/srt/multimodal/manager/mimo_v25_audio_codec_processor.py python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py docs/design/mimo_v25_implementation_notes.md` 通过。

仍未覆盖：

- 完整 HTTP/ZMQ 传输层是否会把 payload dataclass 自动序列化为预期 dict/list 形态仍需 runtime e2e 验证；本轮只收紧接收侧反序列化 contract。

## 本轮继续实现记录：audio_payload offsets 与 token_lengths 前置一致性校验

时间：2026-06-04 23:13:41 CST。

本轮继续收紧直接传 `MiMoV25AudioPayload` 的 span metadata 合同。前面已经校验了 `offsets` 的基础 transport 形态，但如果 payload 自带 offsets，`normalize_payload(...)` 仍只是原样保留，没有检查 offsets 的段数和每段长度是否与 `token_lengths` 一致。对于 payload-only 或 tokenized request，这会让错误 span metadata 穿过 scheduler/runner，直到 stage0 scatter 或更晚的位置才暴露。

已完成：

1. `MiMoV25AudioCodecProcessor.normalize_offsets(...)`
   - 新增 offsets 语义校验 helper。
   - `offsets is None` 保持允许，表示后续可从 `input_ids` 的 audio pad runs 重新 attach。
   - 若 offsets 存在，要求 `len(offsets) == len(token_lengths)`。
   - 每个 offset 必须是正长度 span，且 `end - start == token_lengths[i]`。
2. `MiMoV25AudioCodecProcessor.normalize_payload(...)`
   - 在 codes normalize 前先规范化并校验 offsets。
   - 单段未 pad 自动 padding 分支和 stage0-ready 分支都会使用校验后的 offsets。
3. `test_mimo_v25_audio_codec_processor.py`
   - 新增 `test_normalize_payload_rejects_offset_count_mismatch`。
   - 新增 `test_normalize_payload_rejects_offset_length_mismatch`。

设计影响：

- payload 自带 offsets 时，`offsets/token_lengths/codes` 三者的合同更早闭合。
- tokenizer 从 `input_ids` attach offsets 的路径不受影响；payload 没有 offsets 时仍会由 `validate_placeholder_count(...)` 根据 audio pad runs 写入。

验证记录：

- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过，15 tests。
- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/test_openai_audio_content_parts.py` 通过，2 tests，1 skipped。
- `PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 -m compileall -q python/sgl_jax/srt/multimodal/manager/mimo_v25_audio_codec_processor.py python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过。
- `git diff --check -- python/sgl_jax/srt/multimodal/manager/mimo_v25_audio_codec_processor.py python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py docs/design/mimo_v25_implementation_notes.md` 通过。

仍未覆盖：

- 需要完整 runtime 验证 tokenized request 自带 offsets 时，scheduler/runner 经过 `normalize_payload(...)` 的错误信息是否能正确返回到 client；本地仍缺 JAX/serving 环境。

## 本轮继续实现记录：mm_inputs 中的 audio_payload 改为 transport dict

时间：2026-06-04 23:16:43 CST。

本轮继续检查 `mimo_v25_audio_payload` 在 `mm_inputs` 中的跨层传递形态。当前 tokenizer 到 global scheduler 使用 `send_pyobj`，dataclass 可以通过 pickle 传递；但 `mm_inputs` 同时也会被日志、stage 间请求和 AR request 继续携带。为了让 `mm_inputs` 的内容更明确地符合可序列化 transport 形态，本轮将 `mm_inputs["mimo_v25_audio_payload"]` 改为 dict，而 top-level `TokenizedGenerateOmniReqInput.audio_payload` / `Req.audio_payload` 仍保留 dataclass 作为 Python 内存态 source of truth。

已完成：

1. `MiMoV25AudioPayload.to_transport_dict(...)`
   - 新增 transport dict 输出。
   - `codes` 转 `list[list[int]]`。
   - `token_lengths`、`offsets`、`audio_token_id`、`num_channels`、`codebook_size`、`group_size`、`is_tokenized` 都转成普通 JSON/pickle 友好类型。
2. `multimodal_tokenizer.py`
   - 写入 `mm_inputs["mimo_v25_audio_payload"]` 时改为 `audio_payload.to_transport_dict()`。
   - top-level `audio_payload` 仍传 dataclass。
3. `global_scheduler.py`
   - normalize 后写回 `req.omni_inputs["mimo_v25_audio_payload"]` 时改为 transport dict。
4. `embed_model_runner.py`
   - runner 兜底 normalize 后写回 `omni_inputs["mimo_v25_audio_payload"]` 时也改为 transport dict。
5. `test_mimo_v25_audio_codec_processor.py`
   - 新增 `test_payload_transport_dict_roundtrip`，覆盖 `payload -> dict -> from_obj -> normalize_payload` 的 round-trip。

设计影响：

- `mm_inputs` 中的 payload 明确是 transport-safe dict，减少后续日志/跨 stage/AR request 携带 dataclass 的隐性假设。
- 接收侧无需改动，因为 `MiMoV25AudioPayload.from_obj(...)` 已统一支持 dict/dataclass。
- 内存态 source of truth 仍是 `audio_payload` 字段；`mm_inputs` 是辅助携带和兜底来源。

验证记录：

- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过，16 tests。
- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/test_openai_audio_content_parts.py` 通过，2 tests，1 skipped。
- `PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 -m compileall -q python/sgl_jax/srt/multimodal/manager/mimo_v25_audio_codec_processor.py python/sgl_jax/srt/multimodal/manager/multimodal_tokenizer.py python/sgl_jax/srt/multimodal/manager/global_scheduler.py python/sgl_jax/srt/multimodal/model_executor/embed/embed_model_runner.py python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过。
- `git diff --check -- python/sgl_jax/srt/multimodal/manager/mimo_v25_audio_codec_processor.py python/sgl_jax/srt/multimodal/manager/multimodal_tokenizer.py python/sgl_jax/srt/multimodal/manager/global_scheduler.py python/sgl_jax/srt/multimodal/model_executor/embed/embed_model_runner.py python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py docs/design/mimo_v25_implementation_notes.md` 通过。

仍未覆盖：

- 完整 runtime 中 `mm_inputs` transport dict 经 stage0 写入、AR stage 继续携带时是否有额外内存/性能压力尚未验证；本轮只保证类型合同和 round-trip。

## 本轮继续实现记录：stage0 消费后清理 mm_inputs 中的 raw audio payload

时间：2026-06-04 23:18:51 CST。

本轮继续检查 stage0 embedding 到 AR stage 的边界。前面为了 transport 友好，把 `mm_inputs["mimo_v25_audio_payload"]` 改成 dict；但该 dict 内仍包含完整 `codes` list。`EmbedModelRunner.forward(...)` 在写入 `mm_inputs["multimodal_embedding"]` 后，会把整个 `mm_inputs` 继续交给 `Req.to_stage_reqs("auto_regressive")`，随后 AR scheduler 只需要 `multimodal_embedding` / deepstack 相关字段，不再需要 raw audio payload。继续携带 payload 会造成不必要的跨 stage 内存和序列化负担。

已完成：

1. `embed_model_runner.py`
   - 新增 `_drop_consumed_mimo_v25_audio_payload(omni_inputs)`。
   - 在 `forward(...)` 写入 `mm_inputs["multimodal_embedding"]` 后，删除 `mm_inputs["mimo_v25_audio_payload"]`。
   - 保留 `batch.audio_payload` 作为 stage0 当前 batch 的内存态 debug/source-of-truth，不把 raw payload 继续塞给 AR stage。

设计影响：

- MiMo-V2.5 audio 的 raw codes payload 只在 tokenizer/scheduler/stage0 runner 前置校验与 audio encoder 输入阶段存在。
- AR stage 只收到已经 scatter 好的 `multimodal_embedding`，符合设计文档“AR stage 不接触 raw audio/mel/codes”的边界。
- 如果后续需要 debug，可以从 stage0 当前 `batch.audio_payload` 或日志中取，不让 AR 请求长期携带 codes list。

验证记录：

- `PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 -m compileall -q python/sgl_jax/srt/multimodal/model_executor/embed/embed_model_runner.py` 通过。
- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过，16 tests。
- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/test_openai_audio_content_parts.py` 通过，2 tests，1 skipped。
- `git diff --check -- python/sgl_jax/srt/multimodal/model_executor/embed/embed_model_runner.py docs/design/mimo_v25_implementation_notes.md` 通过。

仍未覆盖：

- 本地环境缺少 `jax`，不能实际执行 `EmbedModelRunner.forward(...)`，因此尚未 runtime 验证 AR request 中 payload key 被清理；本轮通过源码检查和 compileall 覆盖可验证部分。

## 本轮继续实现记录：tokenizer 到 scheduler 之间避免重复传 audio_codes

时间：2026-06-04 23:20:52 CST。

本轮继续收敛跨进程 payload 大小。前面已经让 `MiMoV25AudioPayload` 成为 MiMo-V2.5 audio 链路的 source of truth，并且 `GlobalScheduler.convert_omni_request(...)` 会在收到 payload 后无条件 `req.audio_codes = req.audio_payload.codes`。但 tokenizer 发出的 `TokenizedGenerateOmniReqInput` 仍同时携带：

- `audio_payload`：dataclass，内部有完整 stage0-ready `codes`。
- `audio_codes`：裸 ndarray/list，同样是完整 codes。

这会在 tokenizer -> scheduler 的 ZMQ `send_pyobj` 边界重复传一份 codes，尤其长音频时不必要。

已完成：

1. `multimodal_tokenizer.py::_create_tokenized_omni_object(...)`
   - 如果存在 `MiMoV25AudioPayload`，则 `TokenizedGenerateOmniReqInput.audio_codes` 置为 `None`。
   - 如果没有 payload，保留原裸 `audio_codes` 行为，兼容其它模型或非 payload 内部入口。
2. `global_scheduler.py`
   - 现有逻辑保持不变：payload 存在时 normalize 并回填 `req.audio_codes = req.audio_payload.codes`。

设计影响：

- tokenizer -> scheduler 只传一份 MiMo-V2.5 codes：通过 `audio_payload`。
- scheduler/runner 看到的内部 `Req.audio_codes` 不变，仍由 payload 回填，stage0 输入合同不变。
- 这与前面“payload 是 source of truth，裸 audio_codes 只是 fallback/internal entry”的方向一致。

验证记录：

- `PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 -m compileall -q python/sgl_jax/srt/multimodal/manager/multimodal_tokenizer.py python/sgl_jax/srt/multimodal/manager/global_scheduler.py` 通过。
- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过，16 tests。
- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/test_openai_audio_content_parts.py` 通过，2 tests，1 skipped。
- `git diff --check -- python/sgl_jax/srt/multimodal/manager/multimodal_tokenizer.py docs/design/mimo_v25_implementation_notes.md` 通过。

仍未覆盖：

- 需要完整 runtime 构造带 `audio_payload` 的 tokenized request，确认 scheduler 端确实从 payload 回填 `Req.audio_codes`；本地缺 JAX/serving 环境，本轮只做源码和语法级验证。

## 本轮继续实现记录：多模态请求日志跳过大 audio payload 字段

时间：2026-06-04 23:24:05 CST。

本轮继续检查长音频请求的实际运行成本。前面已经减少了 tokenizer -> scheduler 的重复 codes 传输，但如果开启 `--log-requests`，`MultimodalTokenizer.generate_request(...)` 会用 `dataclass_to_string_truncated(...)` 打印原始请求对象。基类 `TokenizerManager.get_log_request_metadata(...)` 对 log level 0/1 已跳过 `image_data/audio_data`，但没有覆盖 MiMo-V2.5 新增的 `audio_codes/audio_payload`，也没有跳过 `video_data/input_reference`。如果请求显式携带预编码 codes 或 payload，日志可能展开完整 codes。

已完成：

1. `MultimodalTokenizer.__init__(...)`
   - 在 `super().__init__(...)` 后调用 `_extend_multimodal_log_skip_names()`。
2. `MultimodalTokenizer._extend_multimodal_log_skip_names(...)`
   - 仅在 `self.log_requests` 开启时生效。
   - 对已有 `skip_names/out_skip_names` 增加：
     - `image_data`
     - `video_data`
     - `audio_data`
     - `audio_codes`
     - `audio_payload`
     - `input_reference`
   - 如果 log level 2/3 没有 skip set，则不改变用户显式选择的完整/截断日志策略。

设计影响：

- MiMo-V2.5 预编码 codes/payload 不会在默认 request logging level 下被完整写入日志。
- 修改只作用于 `MultimodalTokenizer` 实例，不影响普通文本 `TokenizerManager`。

验证记录：

- `PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 -m compileall -q python/sgl_jax/srt/multimodal/manager/multimodal_tokenizer.py` 通过。
- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过，16 tests。
- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/test_openai_audio_content_parts.py` 通过，2 tests，1 skipped。
- `git diff --check -- python/sgl_jax/srt/multimodal/manager/multimodal_tokenizer.py docs/design/mimo_v25_implementation_notes.md` 通过。

仍未覆盖：

- 需要完整 runtime 开启 `--log-requests` 验证 receive/finish 日志中确实跳过 `audio_payload/audio_codes`；本地缺完整 serving runtime，本轮只做源码和编译级验证。

## 本轮继续实现记录：日志 skip_names 递归应用到嵌套 payload

时间：2026-06-04 23:29:20 CST。

前一节已经在 `MultimodalTokenizer` 的默认 request logging skip set 中加入 `audio_codes/audio_payload`。继续检查通用日志 formatter 后发现，`dataclass_to_string_truncated(...)` 原先只在当前层过滤字段：

- dataclass 当前层会跳过 `skip_names`；
- dict 当前层会跳过 `skip_names`；
- 但 dataclass/list/dict 的子对象递归调用时没有继续传 `skip_names`，因此嵌套在 `mm_inputs`、transport dict 或 list 内的 `audio_payload/audio_codes` 仍可能被展开。

已完成：

1. `common_utils.py::dataclass_to_string_truncated(...)`
   - 对 list/tuple 元素、dict value、dataclass field value 的递归调用继续传递同一份 `skip_names`。
   - 保留 list/tuple 的容器格式，避免为了 skip 行为改变日志基本形态。
2. `test_common_utils_logging.py`
   - 新增纯 formatter 单测，覆盖：
     - dataclass -> dict -> list 嵌套对象中的 `audio_codes/audio_payload` 都会被跳过；
     - tuple 容器格式在递归格式化后仍保持为 tuple。

设计影响：

- `MultimodalTokenizer._extend_multimodal_log_skip_names(...)` 加入的 MiMo-V2.5 大字段跳过策略可以覆盖嵌套 payload，不只覆盖请求对象顶层字段。
- 这降低了显式传 `audio_codes/audio_payload` 或 `mm_inputs["mimo_v25_audio_payload"]` 时默认日志展开完整 codes 的风险。
- 修改作用于通用 formatter，因此普通文本路径如果传入 `skip_names`，嵌套字段也会被一致跳过；没有传 `skip_names` 的调用行为不变。

验证记录：

- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/test_common_utils_logging.py` 通过，2 tests。
- `PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 -m compileall -q python/sgl_jax/srt/utils/common_utils.py python/sgl_jax/test/test_common_utils_logging.py` 通过。
- 同轮回归：
  - `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过，16 tests。
  - `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/test_openai_audio_content_parts.py` 通过，2 tests，1 skipped。

仍未覆盖：

- 仍需要完整 serving runtime 开启 `--log-requests` 验证实际 receive/finish 日志中不会输出 MiMo-V2.5 audio codes；本地当前 Python 环境缺少完整 runtime 依赖，本轮只覆盖 formatter 行为和语法检查。

## 本轮继续实现记录：MiMo-V2.5 raw audio 入口不再强依赖通用 mm_processor

时间：2026-06-04 23:32:51 CST。

本轮继续检查 `server / multimodal_tokenizer / codec helper` 的完整调用关系。设计文档中 MiMo-V2.5 V0 的 audio 边界是：

- server 入口只抽取 raw audio 或预编码 codes；
- `multimodal_tokenizer.py` 调用 MiMo-V2.5 专用 codec helper；
- raw audio -> mel -> `audio_tokenizer.encode` 这一段不放进通用 `mm_processor`，也不走 Qwen/Omni 连续 `audio_features` 合同。

源码检查发现旧条件仍有一个不一致点：`_tokenize_one_request(...)` 在看到 `audio_data` 时会先用通用逻辑要求 `self.mm_processor` 存在。对 MiMo-V2.5 来说，如果请求已经提供 `input_ids` 或 prompt 中包含 audio placeholder，raw audio 本可以由 `MiMoV25AudioCodecProcessor.encode(...)` 处理；提前要求通用 processor 会把这个 V0 fallback 路径挡掉。

已完成：

1. `multimodal_tokenizer.py::_tokenize_one_request(...)`
   - 新增局部判断：
     - `needs_mm_processor = image_data or video_data or (audio_data and not is_mimo_v25)`；
     - `should_run_mm_processor = image_data or video_data or (audio_data and self.mm_processor is not None)`。
   - 普通 audio 模型仍要求 `mm_processor`，MiMo-V2.5 raw audio 在没有通用 processor 时允许继续走专用 codec helper。
2. MiMo-V2.5 raw audio fallback
   - 在普通 tokenizer 完成 `input_ids` 编码后，如果 `is_mimo_v25`、仍没有 `audio_codes/audio_payload`、但存在 `audio_data`，调用 `MiMoV25AudioCodecProcessor.encode(audio_data)`。
   - 这样 OpenAI/server 提取出来的 raw audio 可以落到同一个 `MiMoV25AudioPayload` 合同上。
3. tokenizer 侧前置 span 错误
   - 如果已经有 `audio_payload`，但 `input_ids` 仍为 `None`，立即报错：
     - MiMo-V2.5 audio 请求必须提供 `input_ids`，或提供能 tokenize 出 `<audio_pad>` placeholder 的 prompt。
   - 避免无 placeholder 的 audio payload 继续传到 stage0 后才失败。

设计影响：

- MiMo-V2.5 raw audio 入口和设计文档中的 “host-side codec helper” 边界对齐，不再把 raw audio encode 错绑到通用 `mm_processor` 是否可用。
- 普通 Qwen/Omni audio 路径不受影响，仍通过 `mm_processor.feature_extractor` 产生连续 `audio_features`。
- 没有 chat template/placeholder 的 raw audio 请求会在 tokenizer 侧以明确错误失败，而不是进入 stage0 后发生 embedding scatter 错误。

验证记录：

- `PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 -m compileall -q python/sgl_jax/srt/multimodal/manager/multimodal_tokenizer.py` 通过。
- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/multimodal/test_mimo_v25_audio_codec_processor.py` 通过，16 tests。
- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/test_openai_audio_content_parts.py` 通过，2 tests，1 skipped。
- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/test_common_utils_logging.py` 通过，2 tests。

仍未覆盖：

- 本地直接 import `multimodal_tokenizer.py` 会被缺失的可选多媒体依赖 `imageio` 挡住，因此本轮没有构造完整 tokenizer 实例单测。
- 仍需在完整依赖环境下验证：
  - `mm_processor=None + input_ids + audio_data` 能走 raw codec fallback；
  - `mm_processor=None + audio_data` 但没有 placeholder/input ids 会在 tokenizer 侧报出明确错误；
  - 有 MiMo-V2.5 HF processor 时仍能通过 processor/chat template 生成 placeholder，再由 codec helper 产出 payload。

## 本轮继续实现记录：补 tokenizer raw audio fallback 轻量单测

时间：2026-06-04 23:37:52 CST。

上一节实现了 MiMo-V2.5 raw audio 在 `mm_processor=None` 时走专用 codec helper 的 fallback，但当时只做了语法检查和 codec helper 回归，没有直接覆盖 tokenizer 分支。本轮继续补这个验证缺口。

已完成：

1. `test_mimo_v25_multimodal_tokenizer_audio.py`
   - 用最小 import stubs 隔离本地缺失的可选 runtime 依赖（`imageio/librosa/PIL/transformers/zmq` 等），保留真实 `MiMoV25AudioPayload` 与 codec helper 合同。
   - 通过 `MultimodalTokenizer.__new__(...)` 构造轻量实例，直接调用 `_tokenize_one_request(...)`，不启动 server/ZMQ。
   - 覆盖三条关键路径：
     - MiMo-V2.5 `mm_processor=None + input_ids + audio_data` 会调用 codec helper，生成 `audio_payload`，把单个 audio placeholder 扩展成 `token_lengths` 个 placeholder，并把 transport dict 写入 `mm_inputs["mimo_v25_audio_payload"]`。
     - MiMo-V2.5 `mm_processor=None + audio_data` 但没有 `input_ids/prompt` 会在 tokenizer 侧报出 `require input_ids`。
     - 非 MiMo-V2.5 audio 在 `mm_processor=None` 时仍按原逻辑报 `processor/config`，没有被 V2.5 fallback 放宽。
2. `multimodal_tokenizer.py`
   - 增加 `from __future__ import annotations`。
   - 这个修复是测试过程中暴露出的真实兼容问题：本地 Python 3.9 在 import 该文件时会立即求值 `MiMoV25AudioPayload | None` 等注解并失败。开启 postponed annotations 后，轻量测试和 Python 3.9 import 都能继续执行。

设计影响：

- raw audio fallback 从“源码检查 + compileall”提升为“tokenizer 分支可执行单测覆盖”。
- 测试明确限制在 tokenizer 合同层，不声称覆盖完整 HF processor、真实 codec 权重加载或 serving runtime。
- `from __future__ import annotations` 对运行时行为无影响，但避免新注解在 Python 3.9 下破坏模块 import。

验证记录：

- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/test_mimo_v25_multimodal_tokenizer_audio.py` 通过，3 tests。

仍未覆盖：

- 完整依赖环境下的真实 `MultimodalTokenizer.__init__`、ZMQ/server 请求路径、真实 MiMo-V2.5 HF processor/chat template、真实 `audio_tokenizer/` 权重加载与 raw audio codec parity 仍需后续验证。

## 本轮继续实现记录：补 scheduler/Req audio payload 传递合同测试

时间：2026-06-04 23:42:04 CST。

本轮继续检查 tokenizer 之后的链路：`TokenizedGenerateOmniReqInput -> GlobalScheduler.convert_omni_request -> Req -> Req.to_stage_reqs("auto_regressive")`。设计文档要求 MiMo-V2.5 audio payload 不只在 tokenizer 侧生成，还必须在 scheduler/Req/stage 边界持续保持以下合同：

- `audio_payload` 是 MiMo-V2.5 codes 的 source of truth；
- scheduler 需要把 payload normalize 后回填 `Req.audio_payload` 与 `Req.audio_codes`；
- `mm_inputs["mimo_v25_audio_payload"]` 作为 transport dict 继续提供给 embedding stage；
- 进入 AR stage 时，普通生成参数如 `sampling_params/stop/stream` 不能因为走 omni/audio payload 路径被丢掉。

已完成：

1. `test_mimo_v25_scheduler_payload.py`
   - 用最小 import stubs 隔离本地缺失的 `jax/zmq/PIL` 等 runtime 依赖。
   - 导入真实 `GlobalScheduler.convert_omni_request(...)` 与真实 `Req.to_stage_reqs(...)`。
   - 构造带 `MiMoV25AudioPayload` transport dict、`sampling_params`、`stop`、`stream` 的 `TokenizedGenerateOmniReqInput`。
   - 验证：
     - `convert_omni_request(...)` 生成 `Req.audio_payload`；
     - `Req.audio_codes` 与 payload codes 对齐；
     - `Req.omni_inputs["mimo_v25_audio_payload"]` 被写成 transport dict；
     - `Req.extra` 保留 `sampling_params/stop/stream`；
     - `Req.to_stage_reqs("auto_regressive")` 生成的 AR tokenized request 继续携带 `mm_inputs` 与 sampling 参数。
2. `schedule_batch.py`
   - 增加 `from __future__ import annotations`。
   - 这是测试过程中暴露出的 Python 3.9 import 兼容问题：该文件有大量 `jax.Array | None`、`DataType | None` 等注解。本地环境没有 JAX，且 Python 3.9 会立即求值 `|` 注解；postponed annotations 可以避免在 import 阶段失败。

设计影响：

- tokenizer 后半段链路从“源码检查”提升为“scheduler/Req 合同单测覆盖”。
- 该测试覆盖的是数据合同与 AR request 构造，不声称覆盖真实 queue、Stage thread、EmbedModelRunner 或 AR scheduler runtime。
- `schedule_batch.py` 的注解兼容修复不改变 dataclass 字段或运行逻辑。

验证记录：

- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/test_mimo_v25_scheduler_payload.py` 通过，1 test。

仍未覆盖：

- 完整 `GlobalScheduler` ZMQ event loop、真实 Stage queue、`EmbedModelRunner.forward(...)` 消费后清理 `mm_inputs["mimo_v25_audio_payload"]`、以及 AR scheduler 真实处理 `multimodal_embedding` 仍需完整 runtime 环境验证。

## 本轮继续实现记录：补 EmbedModelRunner payload 消费合同测试

时间：2026-06-04 23:45:16 CST。

本轮继续检查 stage0 embedding runner 边界。前面已经覆盖了 tokenizer 和 scheduler/Req 层，但 `EmbedModelRunner._prepare_input(...)` 是真正把 `MiMoV25AudioPayload` 转成 stage0 model 输入的位置，需要保证：

- transport dict/dataclass payload 会被统一还原并 normalize；
- 单段未 pad codes 会在 runner 侧补齐到 `group_size`；
- placeholder span 与 `token_lengths` 不一致时会在 runner 侧失败；
- MiMo-V2.5 embed stage 拒绝 mel-only `audio_features`；
- stage0 forward 之后 raw `mimo_v25_audio_payload` 不继续带进 AR `mm_inputs`。

已完成：

1. `embed_model_runner.py`
   - 增加 `from __future__ import annotations`，避免 Python 3.9 在 import 时立即求值 `jax.Array | None` 注解。
2. `test_mimo_v25_embed_model_runner_payload.py`
   - 用最小 import stubs 隔离本地缺失的 JAX/flax/transformers/model loader 依赖。
   - 通过 `EmbedModelRunner.__new__(...)` 构造轻量 runner，只调用 `_prepare_input(...)` 和 `_drop_consumed_mimo_v25_audio_payload(...)`。
   - 覆盖：
     - transport dict payload 被 normalize 成 `batch.audio_payload`，未 pad `[5,20]` codes 补成 `[8,20]`，并写回 `mm_inputs["mimo_v25_audio_payload"]` transport dict；
     - runner 侧 attach 出 `payload.offsets=[(1,3)]`；
     - MiMo-V2.5 mel-only 请求报 `requires host-side RVQ audio_codes`；
     - placeholder 数量不足时报 `placeholder count mismatch`；
     - 消费后清理 helper 只删除 raw payload key，保留 `multimodal_embedding`。

设计影响：

- tokenizer -> scheduler -> runner 的 MiMo-V2.5 audio payload 合同现在有分层单测覆盖。
- 测试覆盖的是 runner host-side input preparation，不覆盖真实 JAX audio understanding tower、`jitted_embedding` 数值、权重加载或 AR scheduler 运行。

验证记录：

- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/test_mimo_v25_embed_model_runner_payload.py` 通过，4 tests。

仍未覆盖：

- `EmbedModelRunner.forward(...)` 中真实 `jitted_embedding(**inputs)` 执行、`MiMoV2_5Embedding.audio_encoder(audio_codes)` 数值输出、以及 AR stage 接收 `multimodal_embedding` 的完整 runtime 行为仍需 JAX/flax/真实 checkpoint 环境验证。

## 本轮继续实现记录：补 MiMo-V2.5 stage config registry 入口测试

时间：2026-06-04 23:49:04 CST。

本轮继续检查服务启动入口。前面已经覆盖了 tokenizer/scheduler/runner 的 payload 合同，但 `GlobalScheduler.__init__(...)` 进入两 stage 流水线前，首先依赖 `get_stage_config_path(server_args.model_path)` 找到静态 YAML。若本地 checkpoint 目录名不是精确 `MiMo-V2.5`，例如带后缀的 snapshot 目录或小写路径，原 registry 只靠精确匹配会更脆弱。

已完成：

1. `yaml_registry.py`
   - 在 `_KEYWORD_PATTERNS` 增加 MiMo-V2.5 fallback：
     - `MiMo-V2.5 -> mimo_v2_5_stage_config.yaml`
     - `mimo-v2.5 -> mimo_v2_5_stage_config.yaml`
   - keyword fallback 改为大小写不敏感匹配，避免本地目录名大小写差异导致无法找到 stage config。
2. `embedding.py`
   - 增加 `from __future__ import annotations`。
   - 这是同类 Python 3.9 import 兼容修复：该文件有 `jax.Array | None` 等注解，本地缺 JAX 且 Python 3.9 会立即求值 `|` 注解；postponed annotations 可避免 import 阶段先失败。
3. `test_mimo_v25_stage_registry.py`
   - 覆盖：
     - HF repo id `XiaomiMiMo/MiMo-V2.5` 命中 `mimo_v2_5_stage_config.yaml`；
     - 本地 basename `/models/checkpoints/MiMo-V2.5` 命中；
     - 带后缀本地目录 `/models/MiMo-V2.5-local-snapshot` 与小写 `/models/mimo-v2.5-local-snapshot` 也能通过 keyword fallback 命中。

设计影响：

- MiMo-V2.5 两 stage 配置发现从“精确名字可用”推进到“常见本地 snapshot 路径也可用”。
- 该测试只覆盖 stage YAML registry，不覆盖 `GlobalScheduler` 真实初始化、Stage thread 启动或模型加载。

验证记录：

- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/test_mimo_v25_stage_registry.py` 通过，3 tests。
- `PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 -m compileall -q python/sgl_jax/srt/multimodal/models/mimo_v2_5/embedding.py python/sgl_jax/srt/multimodal/models/static_configs/yaml_registry.py python/sgl_jax/test/test_mimo_v25_stage_registry.py` 通过。

仍未覆盖：

- 完整 server 启动时从真实 `server_args.model_path` 进入 `GlobalScheduler.__init__`、读取 YAML、构造两个 Stage，并完成 `MiMoV2_5Embedding` / AR 模型加载，仍需完整依赖和 checkpoint 环境验证。

## 本轮继续实现记录：补 OpenAI chat audio_data 到 omni request 入口测试

时间：2026-06-04 23:53:05 CST。

本轮继续检查 server/OpenAI 入口。前面已有 `process_content_for_template_format(...)` 单测，证明 OpenAI `input_audio` content part 会抽取到 `audio_data` list，但还缺少一层证明：`OpenAIServingChat._convert_to_internal_request(...)` 在 multimodal 模式下是否真的把 `processed_messages.audio_data` 放进 `GenerateOmniReqInput.audio_data`，而不是只停留在模板处理结果中。

已完成：

1. `serving_chat.py`
   - 增加 `from __future__ import annotations`，避免 Python 3.9 import 时立即求值 `GenerateReqInput | GenerateOmniReqInput` 等注解。
2. `test_openai_audio_to_omni_request.py`
   - 用最小 import stubs 隔离 FastAPI/Pydantic/protocol/manager 依赖。
   - 通过 `OpenAIServingChat.__new__(...)` 构造轻量实例，并 stub `_process_messages(...)` / `_build_sampling_params(...)`。
   - 直接调用真实 `_convert_to_internal_request(...)`，验证 multimodal=true 时：
     - 返回 `GenerateOmniReqInput`；
     - `prompt` 保留 processed prompt；
     - `audio_data=["UklGRg=="]` 被写入 internal request；
     - `sampling_params`、`stream`、`rid`、`stop` 同步保留。

设计影响：

- OpenAI `input_audio` 到 sglang-jax omni request 的入口链路现在有两层本地覆盖：
  - content part -> `audio_data`；
  - processed message -> `GenerateOmniReqInput.audio_data`。
- 测试只覆盖转换函数，不覆盖真实 FastAPI request、Pydantic model validation、chat template 渲染或 tokenizer/server runtime。

验证记录：

- `PYTHONPATH=python PYTHONPYCACHEPREFIX=/private/tmp/sglang_jax_pycache python3 python/sgl_jax/test/test_openai_audio_to_omni_request.py` 通过，1 test。

仍未覆盖：

- 真实 `/v1/chat/completions` HTTP 请求、Pydantic `ChatCompletionRequest` 在当前 Python 3.9 环境下的完整解析、以及后续 `tokenizer_manager.generate_request(...)` streaming 路径仍需完整依赖环境验证。

## 本轮继续实现记录：同步 payload transport 设计并补 stage0 边界测试

时间：2026-06-04 23:57:15 CST。

本轮重新检查 `docs/design/mimo_v25_step2_model_integration.md` 的 4.3.1 调用图与当前实现，发现一处文档已经落后于代码：前面为了降低 tokenizer -> scheduler 的重复传输，已经改成 `MiMoV25AudioPayload` 作为 source of truth；当 payload 存在时，`TokenizedGenerateOmniReqInput.audio_codes` 会置为 `None`，`GlobalScheduler.convert_omni_request(...)` 再从 payload 回填 `Req.audio_codes` 给 stage0 兼容使用。旧文档还写成 tokenized request 直接携带 `audio_codes=payload.codes`，容易误导后续实现把完整 codes 再传一份。

已完成：

1. `docs/design/mimo_v25_step2_model_integration.md`
   - 更新 4.3.1 的完整调用图：
     - tokenizer 输出 `audio_payload=payload` 与 `mm_inputs["mimo_v25_audio_payload"]=payload.to_transport_dict()`；
     - payload 存在时 `audio_codes=None`，避免重复携带完整 codes；
     - scheduler normalize payload 后回填 `Req.audio_codes = Req.audio_payload.codes`；
     - stage0 forward 写入 `multimodal_embedding` 后删除 `mm_inputs["mimo_v25_audio_payload"]`，AR stage 不再携带 raw audio payload。
   - 更新模块职责表，明确 `Req.audio_codes` 是 scheduler -> stage0 的兼容字段，不是跨进程 source of truth。
2. `test_mimo_v25_embed_model_runner_payload.py`
   - 新增 `test_prepare_input_uses_batch_payload_from_scheduler`。
   - 构造更接近 scheduler 输出的 batch：`batch.audio_payload` 已有 dataclass，`batch.omni_inputs` 里还没有 payload transport dict。
   - 验证 `EmbedModelRunner._prepare_input(...)` 可以从 `batch.audio_payload` 取 codes，写回 `mm_inputs["mimo_v25_audio_payload"]` transport dict，并保留 offsets。

设计影响：

- 当前链路的跨层责任更明确：
  - tokenizer -> scheduler：top-level `audio_payload` + `mm_inputs` transport dict；
  - scheduler -> stage0：`Req.audio_payload` 是主合同，`Req.audio_codes` 是兼容输入字段；
  - stage0 -> AR：只保留 `multimodal_embedding`，不继续传完整 audio codes payload。
- 这个测试补的是 scheduler/stage0 边界，不依赖真实 JAX/flax runtime；真实 stage thread、device array sharding 和 AR scheduler 仍需完整环境验证。

## 本轮继续实现记录：payload-first 请求边界和 runner 兜底恢复

时间：2026-06-05 00:01:55 CST。

本轮继续沿着 payload-first 合同检查文档和 stage0 runner。发现两个细节：

1. `docs/design/mimo_v25_step2_model_integration.md` 的 3.4 小节仍写成 `TokenizedGenerateOmniReqInput.audio_codes -> Req.audio_codes`，和 4.3.1 已经更新的 payload-first 实现不一致。
2. `EmbedModelRunner._prepare_input(...)` 只有在 `batch.omni_inputs` 是 dict 时才会读取 `batch.audio_payload`。当前 scheduler 会在 payload 存在时创建/写回 `req.omni_inputs`，所以主路径不受影响；但从模块边界看，`Req.audio_payload` 才是 scheduler -> stage0 的主合同，runner 不应把 top-level payload 的消费依赖在 `omni_inputs` 已经存在上。

已完成：

1. `docs/design/mimo_v25_step2_model_integration.md`
   - 将 3.4 标题从 `audio_codes-first` 改为 `payload-first`。
   - 明确 `multimodal_tokenizer` 产出 `MiMoV25AudioPayload`；`GlobalScheduler.convert_omni_request(...)` 先接收/normalize payload，再从 payload 回填 `Req.audio_codes`。
   - 明确 stage0 消费后删除 `mm_inputs["mimo_v25_audio_payload"]`，AR stage 只透传 `multimodal_embedding`。
2. `embed_model_runner.py`
   - 将 `is_mimo_v25_embedding` 与 `mimo_v25_audio_payload` 的读取提前到 `omni_inputs` 分支外。
   - 当 `batch.audio_payload` 存在但 `batch.omni_inputs` 不是 dict 时，创建空 dict 并写回 `batch.omni_inputs`，再把 normalized payload 写成 transport dict。
   - `audio_feature_attention_mask` 读取增加 `omni_inputs is not None` 保护，避免 MiMo-V2.5 audio-only payload 分支因缺少 `mm_inputs` 触发 `None.get(...)`。
3. `test_mimo_v25_embed_model_runner_payload.py`
   - 新增 `test_prepare_input_recovers_missing_omni_inputs_from_batch_payload`。
   - 验证 `batch.audio_payload` 存在、`batch.omni_inputs=None` 时，runner 仍能生成 `audio_codes` 输入，并恢复 `batch.omni_inputs["mimo_v25_audio_payload"]` transport dict。

设计影响：

- `MiMoV25AudioPayload` 作为 scheduler -> stage0 主合同更加独立，`mm_inputs["mimo_v25_audio_payload"]` 是 transport/日志/跨层可见形态，不再是 runner 消费 payload 的唯一来源。
- 这进一步降低了“某个中间层只保留 top-level payload，却没有同步 mm_inputs dict”时 stage0 静默丢音频的风险。

仍未覆盖：

- 真实 stage queue 中 `Req` 对象如何被序列化/反序列化、device sharding 后 payload 是否仍按预期保留，以及真实 `EmbedModelRunner.forward(...) -> Req.to_stage_reqs("auto_regressive")` 的端到端 runtime 路径，仍需完整依赖和 checkpoint 环境验证。

## 本轮继续实现记录：文档后半同步 payload-first 并覆盖 scheduler payload-only

时间：2026-06-05 00:04:41 CST。

本轮继续检查设计文档后半部分和 scheduler 入口。发现 3.4 / 4.3.1 已经改成 payload-first 后，后面的关键数据表与新增/改动文件清单仍有几处旧表述：

- `audio_codes [T_pad,20]` 被描述成跨层主字段；
- `multimodal_tokenizer.py` 被描述为把 `audio_codes` 写入 `TokenizedGenerateOmniReqInput`；
- `global_scheduler.py` / `schedule_batch.py` 被描述为只确保 `audio_codes` 字段透传。

这和当前实现不一致。当前合同是：`MiMoV25AudioPayload` 是 source of truth，`mm_inputs["mimo_v25_audio_payload"]` 是 transport/debug 可见形态，`Req.audio_codes` 只是 scheduler 回填给 stage0 的兼容字段。

已完成：

1. `docs/design/mimo_v25_step2_model_integration.md`
   - 更新关键数据表，明确 `audio_codes` 的跨层来源是 `MiMoV25AudioPayload.codes`，`Req.audio_codes` 只是 scheduler -> stage0 兼容字段。
   - 更新 V0 模块边界表：
     - `MiMoV25AudioPayload` 建议在 helper 中定义，`io_struct.py` 引用；
     - `multimodal_tokenizer.py` 输出 top-level `audio_payload`，并在 `mm_inputs` 写 transport dict；
     - `global_scheduler.py` normalize payload，必要时创建空 `omni_inputs`，再写回 transport payload；
     - `schedule_batch.py::Req` 保存 payload/codes 给 stage0，AR stage 只接收 `multimodal_embedding`。
   - 更新新增/改动文件清单，去掉“把完整 `audio_codes` 写入 tokenized request”的旧说法。
2. `test_mimo_v25_scheduler_payload.py`
   - 新增 `test_convert_omni_request_recovers_payload_only_request`。
   - 构造 `TokenizedGenerateOmniReqInput(mm_inputs=None, audio_codes=None, audio_payload=transport_dict)`。
   - 验证 `GlobalScheduler.convert_omni_request(...)` 会：
     - normalize `Req.audio_payload`；
     - 从 payload 回填 `Req.audio_codes`；
     - 创建 `Req.omni_inputs` dict；
     - 写入 `Req.omni_inputs["mimo_v25_audio_payload"]` transport dict；
     - 保留 `sampling_params`。

设计影响：

- payload-only tokenized request 现在有本地测试覆盖，减少了 scheduler 对 `mm_inputs` 预先存在的隐式依赖。
- 文档从入口、模块边界、文件清单三处都统一为 payload-first，后续实现不应再把完整 `audio_codes` 当作 tokenizer -> scheduler 的主传输字段。

仍未覆盖：

- 真实 ZMQ / `send_pyobj` 路径、stage queue 传输、device sharding 后 `Req.audio_payload` 与 `Req.omni_inputs` 的保留情况仍需完整 runtime 环境验证。

## 本轮继续实现记录：澄清 host codec 与 payload-first，并补 payload 优先级测试

时间：2026-06-05 00:07:28 CST。

本轮继续检查 `MiMoV25AudioPayload` 作为 source of truth 的一致性。发现 4.3.3 仍使用 `audio_codes-first` 这个术语，容易把两件事混在一起：

- audio 预处理策略确实是 host-side codec 先生成 `audio_codes`；
- 但 tokenizer -> scheduler -> stage0 的跨层主合同已经不是裸 `audio_codes`，而是 `MiMoV25AudioPayload`。

同时，虽然代码此前已经让 payload 覆盖裸 `audio_codes`，但缺少本地测试证明：请求同时带 `audio_payload` 和冲突的 `audio_codes` 时，tokenizer 必须以 payload 为准，避免调用方传入两个不一致来源后进入不确定状态。

已完成：

1. `docs/design/mimo_v25_step2_model_integration.md`
   - 将 4.3.3 的表述改为 **host-side codec encode + payload-first transport**。
   - 明确 `audio_codes` 是 host codec 产物，但跨层主合同是 `MiMoV25AudioPayload`。
   - 明确 `Req.audio_codes` 在 V0 中只是 scheduler 回填给 stage0 的兼容字段，不是 tokenizer -> scheduler 的主传输合同。
2. `test_mimo_v25_multimodal_tokenizer_audio.py`
   - 新增 `test_mimo_v25_audio_payload_wins_over_conflicting_audio_codes`。
   - 构造请求同时携带：
     - `audio_payload.codes=[8,20]`、`token_lengths=[2]`；
     - 冲突的裸 `audio_codes=[4,20]`。
   - 验证 tokenizer：
     - 不调用 raw audio codec；
     - 使用 payload codes；
     - 将单个 `<audio_pad>` 扩展成 2 个；
     - tokenized request 中 `audio_codes` 仍为 `None`，避免重复传输完整 codes；
     - `mm_inputs["mimo_v25_audio_payload"]` 保留 payload transport dict。

设计影响：

- 当前文档术语更准确：V0 是“host codec 先得到 codes”，不是“裸 codes 作为跨层主字段”。
- payload 与裸 codes 同时出现时，payload 的优先级有测试保护，符合 `MiMoV25AudioPayload` source-of-truth 设计。

仍未覆盖：

- 真实 OpenAI/server 请求中如果未来暴露预编码 `audio_codes` / `audio_payload` API，仍需在公开 schema 层定义二者同时出现时的错误策略或 payload 优先策略；本轮只覆盖内部 tokenizer 行为。

## 本轮继续实现记录：payload 优先级改为真正先于裸 audio_codes 校验

时间：2026-06-05 00:09:55 CST。

本轮继续检查“payload wins”是否只是测试语义，还是代码实际顺序。发现 `MultimodalTokenizer._tokenize_one_request(...)` 旧顺序是：

1. 先读取并 normalize `obj.audio_codes`；
2. 再读取并 normalize `obj.audio_payload`；
3. 如果 payload 存在，再用 payload 覆盖 audio_codes。

这样在“payload 合法，但裸 `audio_codes` 非法”的请求中，tokenizer 会先因为裸 codes shape/range 失败，根本走不到 payload 覆盖逻辑。这和 `MiMoV25AudioPayload` 是 source of truth 的设计不一致。

已完成：

1. `multimodal_tokenizer.py`
   - 调整解析顺序：
     - 先读取/normalize `audio_payload`；
     - MiMo-V2.5 且 payload 存在时，直接使用 `audio_payload.codes`；
     - 只有 payload 不存在时，才 normalize 裸 `audio_codes` 并从 codes 构造 payload。
   - 非 MiMo-V2.5 或无 payload 路径仍保留原来的裸 `audio_codes` normalize 行为。
2. `test_mimo_v25_multimodal_tokenizer_audio.py`
   - 加强 `test_mimo_v25_audio_payload_wins_over_conflicting_audio_codes`：
     - 冲突的裸 `audio_codes` 改成非法形态 `[[99999]]`；
     - 验证合法 payload 仍被使用，tokenized request 不携带重复 `audio_codes`，且 raw codec 不被调用。
3. `docs/design/mimo_v25_step2_model_integration.md`
   - 顶层 stage flow 图从 `Req(..., audio_codes, ...)` 改为 `Req(..., audio_payload, audio_codes(stage0兼容), ...)`，避免继续暗示裸 codes 是 Req 主合同。

设计影响：

- payload 优先级现在不再只是“合法冲突时覆盖”，而是“payload 存在时完全不依赖裸 `audio_codes` 的可解析性”。
- 这让内部 request/schema 在过渡期更稳：即使调用方同时传了旧字段和新 payload，V2.5 链路也不会被旧字段的坏值抢先打断。

仍未覆盖：

- 公开 API 层如果未来允许用户同时传 `audio_payload` 和 `audio_codes`，仍需决定是沿用 payload 优先，还是直接报 400；本轮只固化内部 tokenizer 行为。

## 本轮继续实现记录：audio placeholder 缺 payload/codes 时前置失败

时间：2026-06-05 00:12:46 CST。

本轮继续检查 stage0 runner 与 `MiMoV2_5Embedding` 的责任边界。发现一个实际风险：

- `MiMoV2_5Embedding._merge_audio(...)` 是 JIT-friendly scatter，使用固定 `size=audio_embeds.shape[0]` 的 `jnp.nonzero(...)`，它假设输入已经被 runner/helper 校验过。
- 如果请求的 `input_ids` 里已有 `<audio_pad>`，但既没有 `MiMoV25AudioPayload` / `audio_codes`，也没有 mel `audio_features`，旧 runner 会放行。
- 模型侧 `audio_encoder(audio_codes=None)` 返回 `None`，`_merge_audio(...)` 不做替换，最终 `<audio_pad>` 会保留为普通 text embedding。这比显式失败更危险。

已完成：

1. `embed_model_runner.py`
   - 新增 `_count_mimo_v25_audio_placeholders(...)`。
   - 在 MiMo-V2.5 embedding 分支中，如果没有 payload、没有 raw `audio_codes`，但 `input_ids` 中存在 audio placeholder，则在 JIT 前抛出：
     - `MiMo-V2.5 audio placeholders require host-side RVQ audio_codes or audio_payload.`
   - 仍允许纯文本请求没有 payload/codes；只有出现 audio placeholder 时才失败。
2. `test_mimo_v25_embed_model_runner_payload.py`
   - 新增 `test_prepare_input_rejects_audio_placeholders_without_payload_or_codes`。
   - 验证 `input_ids` 含 audio token、payload/codes 缺失时，runner 前置失败。
3. `docs/design/mimo_v25_step2_model_integration.md`
   - 修正 3.1：动态合同校验由 `MiMoV25AudioCodecProcessor` / `EmbedModelRunner._prepare_input` 在 JIT 前完成；`MiMoV2_5Embedding` 内部 scatter 假设输入已满足合同。

设计影响：

- audio placeholder 不会再被静默当作普通文本 token 送进 AR。
- 文档中的责任边界更准确：模型内保持 JIT-friendly，动态错误留在 host/runner 层处理。

仍未覆盖：

- 真实 stage0 `forward(...)` 的 JIT 编译与设备执行路径仍需完整 JAX/flax/runtime 环境验证；本轮只覆盖 `_prepare_input(...)` 的前置校验。

## 本轮继续实现记录：stage0 forward 消费 payload 后清理 AR 输入

时间：2026-06-05 00:18:47 CST。

本轮继续检查 stage0 payload 的消费边界。此前已有覆盖：

- `_prepare_input(...)` 会从 `Req.audio_payload` 或 `mm_inputs["mimo_v25_audio_payload"]` 取 payload；
- payload 会被 normalize，并把 `audio_codes` 传给 `MiMoV2_5Embedding`；
- `_drop_consumed_mimo_v25_audio_payload(...)` 会删除 raw payload key。

但还缺少对 `EmbedModelRunner.forward(...)` 整体路径的直接测试。这个路径很关键，因为它是 stage0 和 AR stage 的真实交界：

1. `forward(...)` 先调用 `_prepare_input(...)`，把 payload 转成模型输入；
2. `jitted_embedding(...)` 返回合并后的 `input_embeds`；
3. runner 写入 `omni_inputs["multimodal_embedding"]`；
4. runner 删除 `omni_inputs["mimo_v25_audio_payload"]`，避免 AR stage 继续携带 raw audio payload。

已完成：

1. `test_mimo_v25_embed_model_runner_payload.py`
   - 新增 `test_forward_writes_embedding_and_drops_audio_payload`。
   - 使用 fake `jitted_embedding` 验证：
     - `forward(...)` 传入模型的 `audio_codes` shape 为 `(8,20)`；
     - `batch.omni_inputs["multimodal_embedding"]` 被写入；
     - `batch.omni_inputs["mimo_v25_audio_payload"]` 被删除；
     - 无 deepstack 输出时 runner 仍写入默认 `deepstack_visual_embedding` / `deepstack_visual_pos_mask`。

设计影响：

- stage0 的 payload 消费现在有直接单测保护：MiMo-V2.5 payload 只存在于 tokenizer/scheduler/stage0 输入边界，AR stage 看到的应是 `multimodal_embedding`。
- 这与 4.3.1 的调用链设计一致：`audio_data/mel/audio_codes/payload` 不应继续流入 AR model runner。

仍未覆盖：

- 真实 JAX `jitted_embedding`、设备执行、stage queue 传输和完整 `to_stage_reqs("auto_regressive")` 路径仍需后续端到端 runtime 验证。

## 本轮继续实现记录：audio_codes 轴归一避免 T==20 歧义

时间：2026-06-05 00:22:35 CST。

本轮继续检查 stage0 `MiMoV25AudioUnderstandingEncoder` 的输入轴约定。V0 payload 合同要求 `payload.codes` 进入 stage0 时是 `[T,20]`，其中最后一维是 20 个 RVQ channel；模型内部分组前需要转成 `[B,20,T]`。

发现的风险：

- 旧逻辑对 2D codes 先加 batch 维，得到 `[1,T,20]`；
- 只有当 `codes.shape[1] != audio_channels` 且 `codes.shape[-1] == audio_channels` 时才 swap；
- 如果 `T==20`，`[1,20,20]` 同时满足“第二维像 channel”和“最后一维像 channel”，旧逻辑会把 `[T,20]` 误当 `[20,T]`，导致后续 grouping 沿错轴处理。

已完成：

1. `multimodal/models/mimo_v2_5/embedding.py`
   - 新增 `MiMoV25AudioUnderstandingEncoder._ensure_channel_first_audio_codes(...)`。
   - 2D 输入优先解释为 V0 payload 合同 `[T,C]`，即最后一维为 `audio_channels` 时转成 `[1,C,T]`。
   - 只有最后一维不是 channel、第一维是 channel 时，才按 `[C,T]` 兼容入口处理。
   - 3D 输入支持 `[B,T,C]` 与 `[B,C,T]`。
   - 非法 shape 在模型入口显式报错。
2. `test_mimo_v25_audio_encoder_axes.py`
   - 新增轻量 stub 单测，不初始化真实 JAX/Flax 模型。
   - 覆盖 `[T,20]` 且 `T==20` 的歧义场景。
   - 覆盖 `[20,T]`、`[B,T,20]`、`[B,20,T]` 和非法 shape。

设计影响：

- stage0 audio tower 现在与 payload-first 合同一致：`[T,20]` 是优先输入格式，即使 `T` 恰好等于 `audio_channels` 也不会被误判为 channel-first。
- 兼容入口仍保留 `[20,T]` / `[B,20,T]`，但不会压过主合同。

仍未覆盖：

- 真实 `MiMoV25AudioUnderstandingEncoder.__call__` 数值输出还需要在完整 JAX/Flax 环境下与上游 sglang/HF 的 `process_audio + projection` 输出做对齐。

## 本轮继续实现记录：audio_tokenizer 缺权重时 hard fail

时间：2026-06-05 00:27:14 CST。

本轮继续检查 host-side codec helper 的 checkpoint 加载边界。MiMo-V2.5 V0 依赖 `<model_path>/audio_tokenizer/` 子目录完成 raw/mel → `audio_codes`，因此 tokenizer config 与权重必须严格匹配。此前 `_load_audio_tokenizer(...)` 使用：

```python
tokenizer.load_state_dict(state_dict, strict=False)
```

但没有检查返回的 `missing_keys`。这会带来一个危险状态：如果 checkpoint 缺少 encoder/RVQ 权重，或 config 与权重树不匹配，加载仍可能继续，直到后续 encode 才暴露 shape/数值错误。

已完成：

1. `mimo_v25_audio_codec_processor.py`
   - 保留 `strict=False` 以允许 PyTorch 返回 `IncompatibleKeys`，但立即读取 `missing_keys`。
   - 如果存在 missing keys，抛出：
     - `MiMo-V2.5 audio tokenizer weights are incomplete: missing ...`
   - 成功路径仍执行 `to(device,dtype)`、`eval()`、`requires_grad_(False)` 并缓存 tokenizer。
2. `test_mimo_v25_audio_codec_processor.py`
   - 新增 `test_load_audio_tokenizer_rejects_missing_weight_keys`。
   - 新增 `test_load_audio_tokenizer_caches_successful_tokenizer`。
   - 使用 fake torch / fake remote tokenizer，不依赖真实 checkpoint 或真实 torch 权重文件。

设计影响：

- raw audio codec 入口现在更符合设计文档中的 hard-fail 原则：找得到 `audio_tokenizer/` 但权重不完整时不能继续。
- 这能更早暴露 checkpoint 布局或 remote tokenizer config 不匹配问题，避免把坏 codes 继续传到 payload/stage0。

仍未覆盖：

- 真实 MiMo-V2.5 `audio_tokenizer/model.safetensors` 与 remote `MiMoAudioTokenizer` 的完整 key 集合仍需在有 checkpoint 的环境里验证；本轮只覆盖 missing-keys 错误策略。

## 本轮继续实现记录：mel 归一化前置校验与 numpy channel-major 轴交换

时间：2026-06-05 00:30:32 CST。

本轮继续审计 host-side codec helper 的 mel 输入边界。V0 允许 raw audio、processor `audio_features/input_features`、以及测试/内部调用直接传 mel。进入 tokenizer encoder 前，mel 必须统一为 time-major `[T,128]` 且 `T>0`。

发现两个问题：

1. 空 mel `[0,128]` / `[128,0]` 没有前置报错，可能继续进入 `_tokenize_audio_batch(...)`，在那里才因空 segment / empty concat 暴露更难定位的错误。
2. `_ensure_mel_time_major(...)` 对 channel-major numpy mel 使用 `mel.transpose(0,1)`，这对 numpy 是 no-op，不会把 `[128,T]` 转成 `[T,128]`。该写法只对 torch Tensor 的 `transpose(0,1)` 语义正确。

已完成：

1. `mimo_v25_audio_codec_processor.py`
   - `_ensure_mel_time_major(...)` 统一返回 `[T,128]`。
   - torch Tensor 继续使用 `transpose(0,1)`。
   - numpy/array-like channel-major mel 改用 `np.swapaxes(mel, 0, 1)`。
   - 归一化后如果 `T<=0`，立即抛出：
     - `MiMo-V2.5 mel time dimension cannot be empty`
2. `test_mimo_v25_audio_codec_processor.py`
   - 新增 `test_ensure_mel_time_major_transposes_numpy_channel_major_mel`。
   - 新增空 time-major / channel-major mel 的拒绝测试。

设计影响：

- host codec helper 对 mel 输入的主合同更清晰：进入 tokenizer encoder 前必须是非空 `[T,128]`。
- 直接传 numpy channel-major mel 的内部/测试入口现在与文档一致，不会被错误地保留为 `[128,T]`。

仍未覆盖：

- 真实 HF processor 输出的 `audio_features/input_features` shape 需要在完整 processor 环境里再确认，尤其是 batch 维和 time/channel 维的排列。

## 本轮继续实现记录：runner 拒绝空裸 audio_codes

时间：2026-06-05 09:55:27 CST。

本轮继续审计 MiMo-V2.5 stage0 runner 的前置校验。`MiMoV25AudioPayload` / codec helper 已经拒绝空 codes segment，但 `EmbedModelRunner` 仍保留一条兼容入口：没有 payload 时可以从 `batch.audio_codes` 读取裸 RVQ ids。该入口此前会接受空 shape：

- `[0,20]`
- `[0,T,20]`
- `[B,0,20]`

旧逻辑会把 `steps=0` 算成 `expected_placeholders=0`，如果请求里也没有 audio placeholder 就放行；随后 `MiMoV25AudioUnderstandingEncoder._group_audio_codes(...)` 可能在重复最后一帧时才失败，错误位置更晚且不清晰。

已完成：

1. `embed_model_runner.py`
   - 在 `_validate_mimo_v25_audio_codes_shape(...)` 中增加空输入校验。
   - 2D codes 若 `steps<=0`，直接抛出：
     - `MiMo-V2.5 audio_codes cannot be empty`
   - 3D codes 若 `batch<=0` 或 `steps<=0`，同样直接失败。
2. `test_mimo_v25_embed_model_runner_payload.py`
   - 新增 `test_prepare_input_rejects_empty_raw_audio_codes`。
   - 新增 `test_prepare_input_rejects_empty_batched_raw_audio_codes`。

设计影响：

- 裸 `audio_codes` 兼容入口现在和 payload/code helper 的非空合同一致。
- 空 audio 不会进入 JAX audio tower，也不会在 `_group_audio_codes(...)` 内部才暴露尾帧 padding 失败。

仍未覆盖：

- 真实 tokenizer/processor 是否可能产生空 audio span 仍需端到端请求验证；当前策略是 stage0 前置失败，而不是 silent skip。

## 本轮继续实现记录：scheduler 从 mm_inputs transport payload 恢复 Req.audio_payload

时间：2026-06-05 09:58:15 CST。

本轮继续检查 tokenizer -> scheduler 的 payload handoff。主路径中 `TokenizedGenerateOmniReqInput.audio_payload` 会携带 Python dataclass source of truth，同时 `mm_inputs["mimo_v25_audio_payload"]` 会携带 transport dict。但跨进程/序列化/外部构造 tokenized request 时，可能只剩 `mm_inputs` 中的 transport payload，而 top-level `audio_payload` 为 `None`。

发现的缺口：

- `GlobalScheduler.convert_omni_request(...)` 只从 `input.audio_payload` 读取 payload。
- 如果 payload 只在 `input.mm_inputs["mimo_v25_audio_payload"]`，scheduler 不会回填 `Req.audio_payload` / `Req.audio_codes`。
- 后续 `EmbedModelRunner` 虽然还能兜底从 `mm_inputs` 取 payload，但 scheduler 层的合同与文档不一致，也少了一次 normalize/writeback 机会。

已完成：

1. `global_scheduler.py`
   - `input.audio_payload` 为空时，尝试从 `req.omni_inputs["mimo_v25_audio_payload"]` 恢复 payload。
   - 恢复后仍统一调用 `MiMoV25AudioCodecProcessor.normalize_payload(...)`。
   - 从 `mm_inputs["audio_token_id"]` 补齐 payload 的 `audio_token_id`。
   - 回填 `Req.audio_payload`、`Req.audio_codes`，并将规范化 transport dict 写回 `Req.omni_inputs["mimo_v25_audio_payload"]`。
2. `test_mimo_v25_scheduler_payload.py`
   - 新增 `test_convert_omni_request_recovers_payload_from_mm_inputs_transport`。
   - 覆盖 top-level `audio_payload=None`、payload 只在 `mm_inputs` 中的场景。

设计影响：

- `mm_inputs["mimo_v25_audio_payload"]` 作为 transport/debug 可见形态现在也能在 scheduler 层恢复成 dataclass source of truth。
- 这不改变主合同：tokenizer 正常仍应同时写 top-level payload 和 transport dict；本轮只是补强跨层鲁棒性。

仍未覆盖：

- 真实 ZMQ `send_pyobj` / stage queue 中 dataclass 与 dict payload 的实际保留形态仍需完整 runtime 验证。

## 本轮继续实现记录：stage0 audio tower 自身拒绝空 audio_codes

时间：2026-06-05 10:01:45 CST。

本轮继续收紧 MiMo-V2.5 stage0 audio tower 的输入合同。runner 已经拒绝空裸 `audio_codes`，payload/code helper 也拒绝空 codes segment；但 `MiMoV25AudioUnderstandingEncoder` 仍可能被测试、模型单独调用或未来内部旁路直接喂入空 codes。

发现的缺口：

- encoder 负责把 `[T,20]`、`[20,T]`、`[B,T,20]`、`[B,20,T]` 统一归一化成 `[B,20,T]`。
- 归一化后没有统一检查 `B` 和 `T` 是否非空。
- 空 `T` 进入 `_group_audio_codes(...)` 后，尾帧 padding 会尝试读取 `codes[:, :, -1:]`，错误位置晚于合同边界，且错误信息不指向 `audio_codes` 非空要求。

已完成：

1. `embedding.py`
   - 新增 `_require_non_empty_audio_codes(...)`。
   - `_ensure_channel_first_audio_codes(...)` 在每条合法轴归一化路径返回前统一检查归一化后的 `[B,20,T]`。
   - 若 `B<=0` 或 `T<=0`，直接抛出：
     - `MiMo-V2.5 audio_codes cannot be empty`
2. `test_mimo_v25_audio_encoder_axes.py`
   - 新增 time-major 空 codes 拒绝测试：`[0,20]`。
   - 新增 channel-major 空 codes 拒绝测试：`[20,0]`。
   - 新增 batched 空 codes 拒绝测试：`[0,T,20]`、`[B,0,20]`。
   - 新增 `_group_audio_codes(...)` 尾帧 padding 测试，确认 `T` 不能整除 `group_size=4` 时重复最后一帧补齐。

设计影响：

- encoder 自身合同现在与 tokenizer/codec/runner 三层保持一致：进入 audio tower 的 codes 必须是非空 RVQ ids。
- `[T,20]` 在 `T==20` 的歧义场景仍优先按 time-major 处理；非空检查发生在归一化之后，不改变轴判定规则。
- `_group_audio_codes(...)` 的 padding 行为被单测固化：`T=5` 会变成 2 个 group，第二个 group 后 3 帧均重复第 5 帧。

仍未覆盖：

- 真实 JAX 权重加载后的 audio embedding 数值 parity 还未验证；当前覆盖的是 host/shape/contract 层。

## 本轮继续实现记录：scheduler 层固化 payload 优先于裸 audio_codes

时间：2026-06-05 10:04:11 CST。

本轮继续检查 tokenizer -> scheduler 边界。前面已经在 tokenizer 层固化：MiMo-V2.5 请求同时带 `audio_payload` 和裸 `audio_codes` 时，以 payload 为准，`TokenizedGenerateOmniReqInput.audio_codes` 通常置空。scheduler 主逻辑当前也会先写入 `input.audio_codes`，随后如果发现 `audio_payload`，再 normalize payload 并回填 `Req.audio_codes = Req.audio_payload.codes`。

发现的缺口：

- scheduler 的实际行为已经是 payload 覆盖裸 `audio_codes`，但缺少测试固化。
- 未来如果有人把 `req.audio_codes = input.audio_codes` 后的 payload 回填逻辑改弱，可能重新引入“同一个请求两份 codes source of truth”的歧义。

已完成：

1. `test_mimo_v25_scheduler_payload.py`
   - 新增 `test_convert_omni_request_payload_wins_over_conflicting_audio_codes`。
   - 构造 `audio_payload.codes=[8,20]` 与冲突的 `audio_codes=[4,20]`。
   - 验证 `GlobalScheduler.convert_omni_request(...)` 输出：
     - `Req.audio_payload.codes` 来自 payload；
     - `Req.audio_codes` 也被回填为 payload codes；
     - 裸 `audio_codes` 不会成为最终 stage0 输入。

设计影响：

- payload-first 合同现在在 tokenizer 层和 scheduler 层都有测试保护。
- `Req.audio_codes` 在 MiMo-V2.5 V0 中继续保持“stage0 兼容字段”语义，不升级为跨层 source of truth。

仍未覆盖：

- 公开 API 层如果未来直接暴露 `audio_payload` / `audio_codes` 双字段，还需要决定是否沿用 payload 优先，或改成 schema validation 直接拒绝二者同时出现。

## 本轮继续实现记录：codec normalize_codes 直接拒绝空 RVQ codes

时间：2026-06-05 10:05:47 CST。

本轮继续检查 host-side codec helper 的底层 codes 归一化边界。此前 `build_payload_from_codes(...)` 会在 segment 层拒绝空 codes，runner 和 JAX audio tower 也已经拒绝空 codes；但 `MiMoV25AudioCodecProcessor.normalize_codes(...)` 本身如果直接收到 `[0,20]` 或 `[20,0]`，会返回空 `[0,20]`，把错误推迟到上层 token length 或 grouping 校验。

发现的缺口：

- `normalize_codes(...)` 是 payload normalize、build payload、测试/内部预编码 codes 的共同基础入口。
- 空 codes 已经能被识别成 MiMo-V2.5 20-channel RVQ ids，只是时间维为 0；此时错误应该是 `audio_codes cannot be empty`，不应依赖上层间接暴露。

已完成：

1. `mimo_v25_audio_codec_processor.py`
   - 在 `normalize_codes(...)` 完成 `[T,20]` 归一化后增加 `T>0` 校验。
   - 对 `[0,20]` 和 `[20,0]` 统一抛出：
     - `MiMo-V2.5 audio_codes cannot be empty`
2. `test_mimo_v25_audio_codec_processor.py`
   - 新增 `test_normalize_codes_rejects_empty_time_major_codes`。
   - 新增 `test_normalize_codes_rejects_empty_channel_major_codes`。
   - 新增 `test_normalize_payload_rejects_empty_codes`，覆盖 payload dict/dataclass 归一化入口。

设计影响：

- codec、runner、JAX audio tower 三层现在共享同一条非空 RVQ ids 合同。
- 未来若新增公开预编码 `audio_codes` API，可以直接复用 `normalize_codes(...)` 的底层错误语义，而不是每个入口各写一遍空 shape 校验。

仍未覆盖：

- 真实 MiMo-V2.5 remote codec encode 如果输入极短音频是否可能产生空 codes，仍需在 checkpoint 环境验证；本轮只保证一旦空 codes 出现，会在 host-side 合同层清晰失败。

## 本轮继续实现记录：stage0 后 AR 请求只携带 multimodal_embedding

时间：2026-06-05 10:08:00 CST。

本轮继续检查 stage0 -> AR handoff。设计文档要求 MiMo-V2.5 audio payload 只存在于 tokenizer/scheduler/stage0 输入边界；`EmbedModelRunner.forward(...)` 消费后写入 `omni_inputs["multimodal_embedding"]` 并删除 `omni_inputs["mimo_v25_audio_payload"]`，随后 `Req.to_stage_reqs("auto_regressive")` 只把 embedding 传给普通 AR stage。

已有覆盖：

- runner 层已经有测试证明 `forward(...)` 会写入 `multimodal_embedding` 并删除 payload transport key。
- scheduler 层已经有测试证明 `convert_omni_request(...)` 会从 payload 回填 stage0 兼容的 `Req.audio_codes`。

发现的缺口：

- 缺少一条直接覆盖 “stage0 已消费 payload 后，`to_stage_reqs("auto_regressive")` 输出的 tokenized AR request 不再包含 raw audio payload/codes” 的测试。
- 如果后续改动在 AR handoff 前重新写回 `mimo_v25_audio_payload`，普通 AR scheduler 仍可能收到不该感知的 MiMo-V2.5 audio transport 字段。

已完成：

1. `test_mimo_v25_scheduler_payload.py`
   - 新增 `test_auto_regressive_request_after_stage0_carries_only_embedding`。
   - 先通过 `GlobalScheduler.convert_omni_request(...)` 构造真实 `Req`。
   - 模拟 stage0 forward 后状态：写入 `omni_inputs["multimodal_embedding"]`，删除 `mimo_v25_audio_payload`。
   - 调用 `Req.to_stage_reqs("auto_regressive")`，验证：
     - AR request 的 `mm_inputs` 仍引用同一个 `omni_inputs`；
     - `multimodal_embedding` 存在；
     - `mimo_v25_audio_payload` 不存在；
     - AR tokenized request 没有 `audio_payload` / `audio_codes` 字段；
     - sampling params 继续保留。

设计影响：

- `stage0 payload/codes -> AR input_embedding` 的边界有了更贴近完整调用链的本地测试。
- 这进一步固化“不新增 AR audio 分支”的 V0 设计：AR 只消费 embedding，不感知 raw audio、mel、codec 或 payload。

仍未覆盖：

- 真实 AR scheduler 是否把 `mm_inputs["multimodal_embedding"]` 正确落到 `ForwardBatch.input_embedding`，需要在核心 scheduler/model runner 侧结合真实 prefill 路径继续验证。

## 本轮继续实现记录：AR scheduler 合并 input_embedding 的长度前置校验

时间：2026-06-05 10:13:24 CST。

本轮继续推进 stage0 -> AR 的后半段。前面已经验证 stage0 后 `Req.to_stage_reqs("auto_regressive")` 只携带 `mm_inputs["multimodal_embedding"]`；这次继续检查核心 AR scheduler 侧的落点：`scheduler.py::handle_generate_request(...)` 会把 `mm_inputs["multimodal_embedding"]` 写到核心 `Req.multimodal_embedding`，随后 `ScheduleBatch._merge_multimodal(...)` 在 extend/prefill 阶段按 `[prefix_len, seq_len)` window 切片，生成 `ModelWorkerBatch.input_embedding`，最后进入 `ForwardBatch.input_embedding`。

发现的缺口：

- `_merge_multimodal(...)` 之前直接 `chunk = mm_full[start:end]`，如果 `multimodal_embedding` 行数短于当前 extend window，可能在后续 numpy 赋值处才触发 broadcast 错误。
- 这个错误不指向 MiMo-V2.5/stage0 handoff 合同，排查时不容易看出是 stage0 embedding 长度与 prompt window 不一致。
- 本地环境没有真实 JAX，因此此前没有直接覆盖核心 `ScheduleBatch._merge_multimodal(...)` 的轻量单测。

已完成：

1. `schedule_batch.py`
   - 在 `_merge_multimodal(...)` 读取 `req.multimodal_embedding` 后增加前置校验：
     - embedding 必须是 2D `[seq_len, hidden]`；
     - embedding 行数必须覆盖当前 extend window 的 `end`。
   - 失败时抛出明确错误：
     - `multimodal_embedding length mismatch`
2. `test_mimo_v25_ar_input_embedding_merge.py`
   - 新增一个带最小 JAX/依赖 stub 的核心 AR scheduler 单测文件。
   - 覆盖正常路径：`multimodal_embedding[1:4]` 被切入 `input_embedding[:3]`。
   - 覆盖错误路径：embedding 只有 3 行但 extend window 需要到 `end=4`，直接抛出 length mismatch。

设计影响：

- `stage0 multimodal_embedding -> AR input_embedding` 这条链路现在不仅有 request handoff 测试，也有核心 batch 合并测试。
- 如果 MiMo-V2.5 stage0 输出长度和 prompt/audio placeholder 展开长度不一致，错误会在 AR batch 合并阶段以合同错误暴露，而不是低层 shape broadcast。
- 纯文本请求、decode 请求、没有 `multimodal_embedding` 的请求不受该校验影响。

仍未覆盖：

- 真实 `ForwardBatch.init_new(...)` 在有 JAX 环境下把 `ModelWorkerBatch.input_embedding` 转成 bf16 device array 的行为还需要完整 runtime 或带真实 JAX 的单测继续验证。

## 本轮继续实现记录：ForwardBatch.init_new 保留并 cast input_embedding

时间：2026-06-05 10:16:43 CST。

本轮继续补齐 `stage0 multimodal_embedding -> AR input_embedding` 的最后一个本地可验证环节。上轮已经覆盖 `ScheduleBatch._merge_multimodal(...)` 会把核心 `Req.multimodal_embedding` 切到 `ModelWorkerBatch.input_embedding`；这次继续验证 `ForwardBatch.init_new(...)` 是否把 worker batch 上的 `input_embedding` 传到 `ForwardBatch.input_embedding`。

发现的缺口：

- notes 中上一节把入口误写成 `ForwardBatch.from_model_worker_batch(...)`；当前源码实际入口是 `ForwardBatch.init_new(...)`。
- `ForwardBatch.init_new(...)` 已有逻辑：如果 `batch.input_embedding is not None`，通过 `device_array(...)` 放到 mesh sharding 上，再 `astype(jnp.bfloat16)`。
- 但本地缺少真实 JAX，因此之前没有轻量测试证明该字段不会在 worker batch -> forward batch 转换时丢失。

已完成：

1. `test_mimo_v25_forward_batch_input_embedding.py`
   - 新增带最小 JAX/依赖 stub 的 `ForwardBatch.init_new(...)` 测试。
   - `test_init_new_preserves_and_casts_input_embedding` 验证：
     - `batch.input_embedding` 会出现在 `forward_batch.input_embedding`；
     - dtype 会通过 `astype(jnp.bfloat16)` 转换（测试中 stub 为 `np.float32`）。
   - `test_init_new_keeps_missing_input_embedding_none` 验证纯文本/无 embedding 路径仍保持 `None`。

设计影响：

- 从 stage0 到 AR 模型输入的本地覆盖现在形成连续链：
  - stage0 forward 写 `mm_inputs["multimodal_embedding"]` 并删除 payload；
  - `Req.to_stage_reqs("auto_regressive")` 只透传 embedding；
  - 核心 scheduler 读取 `mm_inputs["multimodal_embedding"]` 到 `Req.multimodal_embedding`；
  - `ScheduleBatch._merge_multimodal(...)` 合并到 `ModelWorkerBatch.input_embedding`；
  - `ForwardBatch.init_new(...)` 保留并 cast 到 `ForwardBatch.input_embedding`；
  - MiMoV2 LM 在 extend/prefill 时优先使用 `forward_batch.input_embedding`。

仍未覆盖：

- 测试使用 stub `device_array` 和 `jnp.bfloat16`，只能证明字段传递和 cast 调用语义；真实 TPU/JAX mesh sharding、bf16 device dtype、以及完整 checkpoint prefill parity 仍需在有真实 JAX/runtime 的环境继续验证。

## 本轮继续实现记录：同步设计文档第 5 节与当前 V0 实现形态

时间：2026-06-05 10:23:45 CST。

本轮继续审计 `mimo_v25_step2_model_integration.md` 与当前 worktree 的一致性。前面实现已经推进到 payload、codec、stage0 audio tower、AR input_embedding handoff 的本地可测链路，但第 5 节仍保留了早期理想拆分形态，把若干尚未创建的文件写成“新增”，同时把核心 `_merge_multimodal` 写成“明确无改动”。

发现的缺口：

- 文档写 `multimodal/models/static_configs/mimo_v2.5_stage_config.yaml`，当前实际文件是 `mimo_v2_5_stage_config.yaml`。
- 文档把 `audio_encoder.py`、`generation.py`、`vision_mimovl.py` 都列为当前新增文件；当前 V0 实际只有 `mimo_v2_5/embedding.py`，其中内聚了 `MiMoV25AudioUnderstandingEncoder`。
- 文档仍说 `to_stage_reqs / 核心 _merge_multimodal` “明确无改动”，但当前已经补了：
  - stage0 后 AR request 只携带 `multimodal_embedding` 的测试；
  - `ScheduleBatch._merge_multimodal(...)` 的 input embedding 合并和长度前置校验；
  - `ForwardBatch.init_new(...)` 的 input_embedding 保留/cast 测试。
- 文档残留 `MiMoV2_5Generation` 类名；当前 stage yaml 使用 `MiMoV2ForCausalLM`，AR 侧复用现有 MiMoV2 causal LM/flash 路径。

已完成：

1. `mimo_v25_step2_model_integration.md`
   - 4.3 audio 表格中，将 JAX audio tower 落点改为：
     - `mimo_v2_5/embedding.py::MiMoV25AudioUnderstandingEncoder`
   - 3.3 stage config 文件名改为：
     - `models/static_configs/mimo_v2_5_stage_config.yaml`
   - AR stage 类名改为当前实际使用的：
     - `MiMoV2ForCausalLM`
   - 第 5 节重写为三类：
     - 当前 V0 已新增 / 已落地；
     - 后续计划新增 / 拆分；
     - 当前 V0 已改动 / 明确不新增。
   - 把 `ScheduleBatch._merge_multimodal(...)` 和 `ForwardBatch.init_new(...)` 纳入已改动/已验证 handoff 范围。

设计影响：

- 文档现在区分“当前 V0 实现形态”和“后续理想模块拆分”，不会误导开发者去找尚不存在的 `audio_encoder.py` / `generation.py`。
- `stage0 multimodal_embedding -> AR input_embedding` 被明确写成现有 core scheduler / ForwardBatch handoff，而不是 prework 中已经完全解决的黑盒前提。

仍未覆盖：

- 视觉塔 `vision_mimovl.py`、权重映射拆分、真实 `MiMoV2ForCausalLM`/flash 选择和 checkpoint parity 仍需继续按里程碑推进；本轮只同步文档边界，不新增运行时行为。

## 本轮继续实现记录：stage config 内容校验覆盖 AR input_embedding 合同

时间：2026-06-05 10:26:03 CST。

本轮继续检查 MiMo-V2.5 stage config 与设计文档同步后的配置合同。前面已经修正文档，确认当前 V0 stage yaml 使用 `mimo_v2_5_stage_config.yaml`、stage0 `MiMoV2_5Embedding`、stage1 `MiMoV2ForCausalLM`，并要求 AR stage `precompile_params.input_embedding=True`。

发现的缺口：

- 现有 `test_mimo_v25_stage_registry.py` 只验证 model path 能解析到 `mimo_v2_5_stage_config.yaml`。
- 它没有验证 yaml 内容本身，因此无法防止后续把 AR `model_class` 改回旧名或漏掉 `input_embedding` precompile 参数。

已完成：

1. `test_mimo_v25_stage_registry.py`
   - 新增 `test_mimo_v25_stage_config_uses_embedding_and_ar_input_embedding`。
   - 读取 `get_stage_config_path("XiaomiMiMo/MiMo-V2.5")` 返回的 yaml。
   - 验证：
     - `model_arch == "MiMo-V2.5"`；
     - stage0 scheduler/model class 是 `embedding` / `MiMoV2_5Embedding`；
     - stage1 scheduler/model class 是 `auto_regressive` / `MiMoV2ForCausalLM`；
     - `precompile_params.input_embedding=True`；
     - `deepstack_visual_embedding=False`、`mrope=False`。

设计影响：

- stage config 现在不仅有路径发现测试，也有内容级测试保护。
- 这直接保护 `stage0 multimodal_embedding -> AR ForwardBatch.input_embedding` 的配置入口，避免 AR stage 预编译时漏开 input embedding。

仍未覆盖：

- stage runtime 真正按该 yaml 构造 `Stage` / model runner 并加载真实 `MiMoV2ForCausalLM` 权重的路径仍需完整 runtime 环境验证。

## audio-only V1 整改记录（依据 step2 review）

时间：2026-06-05 CST。

依据 `docs/design/mimo_v25_step2_review.md`（6 维度 + adversarial 复核）对 audio-only 切片做的一轮整改。**本节为当前权威记录，凡与前文 §9 / P0 / P1 旧条目冲突处，以本节为准。** 环境限制：本机仅有 numpy（无 jax/flax/torch/transformers，Python 3.9.6），故 host-codec / 路由 / 配置类改动用 numpy 单测 + `py_compile` 验证；涉及 jax/torch 的前向用 `skipUnless` 守卫，留 CI 跑。

### 已落地改动

1. **per-quantizer codebook 校验（review D4-5）**
   - `mimo_v25_audio_codec_processor.py`：`normalize_codes` 新增 `codebook_sizes` 可选参数，提供时按通道逐列校验 `code < codebook_sizes[c]`，否则回退到旧的标量 `codebook_size` 上界。新增 `_validate_code_range` 辅助。
   - 真实 RVQ 是 per-quantizer codebook（`[1024,1024,256,128,...]`），统一 1280 会放行高位通道非法 code。
   - `codebook_sizes` 经 `normalize_code_segments` / `build_payload_from_codes` / `normalize_payload` 透传，并加到 `MiMoV25AudioPayload` 字段与 transport dict。
   - codec 新增 `get_codebook_sizes()`：best-effort 从 `audio_tokenizer/config.json` 的 `codebook_size` 读取；`encode_mels` 自动带上。gated 不可用时返回 `None`（退回标量校验，行为不变）。

2. **layout 歧义（review D3-7/D4-4）**
   - `MiMoV25AudioPayload` 新增 `codes_layout="time_major"` 字段，明确 host codec 恒输出 `[T, C]`（time-major）。
   - `normalize_codes` 显式处理方阵 `[C, C]`：按 time-major（最后一轴=channel）解释，并加注释说明取舍。
   - `embedding.py::_ensure_channel_first_audio_codes` 同步改为优先「最后一轴==channel」，与 host 契约一致，方阵不再两端分歧。

3. **audio scatter 静默覆盖（review D3-6/D4-6）**
   - `embed_model_runner.py::_prepare_input`：MiMo-V2.5 分支在有 audio（codes/payload）但 `audio_token_id` 无法解析时**硬报错**，不再静默早退（早退会让模型用默认 id scatter，覆盖 token 0）。
   - `_merge_audio` 的 host 侧不变量（`#placeholder == audio_embeds.shape[0]`）由 codec/runner 校验链保证；新增 jax-guarded 测试直接验证 scatter 落位（见下）。

4. **embed runner 模型无关化（review D5-4）**
   - 去掉 `embed_model_runner.py` 里 `model_class.__name__ == "MiMoV2_5Embedding"` 字符串判定，改为读模型类 capability flag `requires_mimo_v25_audio_contract`（在 `MiMoV2_5Embedding` 上声明）。子类/重命名不再失效。

5. **路由脆弱性（review D4-2/D5-1）**
   - `multimodal_tokenizer.py`：`is_mimo_audio` 由「`mimo`+`audio` 两个子串」收紧为相邻 `mimo-audio`，避免 `MiMo-V2.5-omni-audio` 误判；`_is_mimo_v25_model` 删除宽泛 `"mimo-v2.5" in model_path` 分支，仅保留 `model_type` 别名 + 真实 audio_config 能力判定。
   - `yaml_registry.py`：删除宽泛 `("MiMo-V2.5"/"mimo-v2.5", ...)` keyword fallback，改为 `_match_mimo_v25_omni()` —— 命中 `mimo-v2.5` 但排除含 `pro`/`flash` 的 text-only 变体。保留本地 snapshot 可用性，同时让 `MiMo-V2.5-Pro/-Flash` 不再误入 omni stage（已加单测）。

6. **OpenAI 多模态字段丢失（review D5-5）**
   - `io_struct.py::GenerateOmniReqInput` 新增 `return_logprob/logprob_start_len/top_logprobs_num/extra_key` 字段。
   - `serving_chat.py` 多模态分支：填上这些字段（forward-compat），并对 `logprobs/top_logprobs>0` **显式报 ValueError**，把原来的静默丢弃改成明确错误（logprobs 经 omni AR stage 的贯通仍是后续工作）。

7. **stage1 AR 类解析澄清（review D2-4）**
   - 真实 config `attention_projection_layout=fused_qkv`，故 yaml 保留 `MiMoV2ForCausalLM`（解析到 `mimo_v2_pro` 的 fused-QKV 子类）是合理的；在 `mimo_v2_5_stage_config.yaml` 加注释说明该解析与 OPEN 项（需真实 checkpoint 确认 QKV 是否融合；若分离则切 `MiMoV2FlashForCausalLM`，已注册）。**修正前文「复用 flash」表述：当前权威选择是 fused-QKV 的 `MiMoV2ForCausalLM`。**

8. **死代码清理（review D5-2）**
   - 删除 `config_registry.py::MiMoV25AudioBackboneConfig` 及其两条 `AudioBackboneConfigRegistry` 注册项——MiMo-V2.5 omni 无 `audio_backbone` stage，从不走该 registry，属死代码。

9. **设计文档一致性（review D1/D2）**
   - `mimo_v25_step2_model_integration.md`：顶部加 V1 audio-only scope 声明；§1.2 MTP 行改为「omni checkpoint 不加载 MTP」；§1.3/§4.1 视觉 `head_dim` 改 40→**64**（`qk_channels` 默认）；§1.5 token id 标注顶层 vs `processor_config` 层级；§3.2/§3.4/§5/§6.1/§6.3 修正 MTP、`is_multimodal`/`EmbedModelRunner`/`http_server`/`serving_chat` 的「无改动」误述、milestone A 措辞、易错项 head_dim。

### 测试

- 可在本机运行（numpy）：`multimodal/test_mimo_v25_audio_codec_processor.py`（24）、新增 `multimodal/test_mimo_v25_audio_codec_v1.py`（8：per-quantizer 校验 + layout 契约 + transport roundtrip）、`test_mimo_v25_stage_registry.py`（新增 Pro/Flash 排除）、`test_openai_audio_to_omni_request.py`（新增 logprobs 报错）、其余既有 mimo_v25 / openai_audio 用例。全部通过。
- jax/torch 守卫（CI 运行，本机 skip）：新增 `test_mimo_v25_embedding_merge_jax.py` 直接用真实 `jnp` 验证 `_merge_audio` scatter 落位、None 透传、方阵 time-major。

### 仍未覆盖 / 后续

- **mel parity（review D6-5）**：`_waveform_to_mel`（torchaudio `n_fft=960/hop=240/log clamp 1e-7`）与 `_tokenize_audio_batch`（segment 6000 / `get_output_length` / `avg_pooler`）需 torchaudio + 真实 checkpoint 做数值对拍，本机无 torchaudio，未补。
- **RVQ encode 数值对齐（review D4-1）**：raw audio→codes 路径代码已写但从未跑通/与 HF remote code 对齐，仍是 §6.4 开放项；首轮可用入口仍是预编码 `audio_codes` / host tokenizer。
- **真实权重 load parity**：`MiMoV2_5Embedding` 各塔 + AR 类的 safetensors 加载（含 input_local transpose 方向、QKV 融合与否）需完整 runtime + 真实 checkpoint 验证。
- **DP>1 sharding**（review D3-8）：audio 塔各层 sharding 轴在 DP>1 需重核；本轮 DP=1。
- **`test_mimo_v25_audio_encoder_axes.py` 命名**：仍只覆盖 `_ensure_channel_first`/`_group_audio_codes` 纯函数；真实 encoder 前向已由 jax-guarded 测试补足，未改名以减少 churn。

## 非视觉完整方案模块化落地（review §3.1 代码落地）

时间：2026-06-05 CST。

review §3.1「文档 scope 与代码落差」此前只做了文档侧对齐（把 doc 改成描述 audio-only 现状）。本轮按要求把**除 vision 外的完整方案代码真正落地**，让 design doc §5 承诺的模块结构成为现实。**本节为当前权威记录。** 环境仍只有 numpy（无 jax/torch/transformers），故纯逻辑模块（configuration）用 numpy 单测，模型类用 `py_compile` + jax-guarded 测试（CI 跑）。

### 新增 / 改动模块

1. **`models/mimo_v2_5/configuration.py`（新增）**
   - `get_config_value`（processor_config fallback）、`resolve_token_ids`（audio token id `str→int` 强转）、`resolve_audio_contract`（audio 塔结构字段解析，含 `speech_vocab_size/speech_zeroemb_idx` 字符串强转）。
   - `MiMoV2_5Config(PretrainedConfig)`：用 **guarded base import**（无 transformers 时退回 `object`），故模块可在纯 numpy 环境导入/单测；`from_hf_config()` 从扁平 HF config 构造 typed view（token ids + audio_contract + vision_config 透传）。

2. **`models/mimo_v2_5/audio_encoder.py`（新增，从 embedding 拆出）**
   - `MiMoV25AudioUnderstandingEncoder` 移到此处，`__init__` 改用 `resolve_audio_contract` 统一取字段（行为等价）。`embedding.py` re-export 之，兼容既有 `test_mimo_v25_audio_encoder_axes.py` 的 import。

3. **`models/mimo_v2_5/weights_mapping.py`（新增）**
   - text/speech_embeddings/input_local/projection 权重映射构建函数 + `build_embedding_weight_mappings()`；从 embedding 的内联 dict 拆出，映射内容逐字保持。

4. **`models/mimo_v2_5/generation.py`（新增）**
   - `MiMoV2_5Generation(MiMoV2ForCausalLM)`：AR stage 生成模型。复用融合-QKV backbone（匹配真实 config `attention_projection_layout=fused_qkv`）+ `MiMoV2Model` 已有的 input_embedding / 1-D rope / no-deepstack hooks；`__init__` 对 rope_scaling 非 default 时 warn（1-D rope 预期）。`EntryClass = MiMoV2_5Generation`。
   - 接线：`stage.py` 注册 `MiMoV2_5Generation`；`mimo_v2_5_stage_config.yaml` AR `model_class` 由 `MiMoV2ForCausalLM` 改为 `MiMoV2_5Generation`。

5. **`models/mimo_v2_5/embedding.py`（重构）**
   - 改为 import `audio_encoder`/`weights_mapping`/`configuration`；保留 `MiMoV2_5Embedding`（capability flag、scatter、结构化输出），删除内联的 encoder 类与映射方法（`_get_config_value` 改为薄委托保留向后兼容）。

6. **`models/mimo_v2_5/__init__.py`（重构为 lazy）**
   - PEP 562 lazy 导出 `MiMoV2_5Embedding`/`MiMoV25AudioUnderstandingEncoder`/`MiMoV2_5Generation`/`MiMoV2_5Config`，使 `configuration` 等轻模块无需拉起 jax/flax 即可导入/单测。

### 设计决策

- **AR 类选 fused-QKV（Pro）base 而非 Flash**：真实 config `attention_projection_layout=fused_qkv`，故 `MiMoV2_5Generation` 子类化 `mimo_v2_pro.MiMoV2ForCausalLM`。yaml + generation 均留 OPEN 注释：若真实 omni checkpoint 用分离 q/k/v_proj，则把 base 换成已注册的 `MiMoV2FlashForCausalLM`。
- **MiMoV2_5Config 不注册到 AutoConfig**：runtime 仍用 remote-code config（`AutoConfig.from_pretrained`）；`MiMoV2_5Config` 作为 typed view / 工具用，避免与 remote `model_type=mimo_v2` 注册冲突。其底层 helper（`get_config_value`/`resolve_*`）已被 embedding/audio_encoder 实际使用，非死代码。
- **vision 明确不做**：`vision_mimovl.py` 仍是唯一未落地模块；`embedding.py` 对 image/video 维持 `NotImplementedError`。

### 测试

- 新增可本机运行（numpy）：`test_mimo_v25_config.py`（6：fallback / `str→int` / contract / from_hf_config）。
- 新增 jax-guarded（CI）：`test_mimo_v25_generation_registry_jax.py`（subclass / stage 注册 / lazy 导出）。
- 更新 `test_mimo_v25_stage_registry.py`：AR `model_class` 断言改为 `MiMoV2_5Generation`。
- 既有 mimo_v25 / openai_audio / codec 测试全部仍通过（embedding 重构 + re-export 保持兼容）。

### 仍未覆盖

- 真实权重加载 parity（融合-QKV 与否、input_local transpose 方向）需真实 checkpoint + runtime。
- vision 塔（`vision_mimovl.py`）与其权重映射。
- raw audio→codes 的 RVQ encode 数值对齐（§6.4，沿用上一轮记录）。

### 修正：撤销 `MiMoV2_5Generation`（上一小节的 generation.py）

时间：2026-06-05 CST。**本小节修正上一小节关于 `generation.py` 的内容。**

复核后确认：MiMo-V2.5 的 **text / AR backbone 与 `MiMoV2ForCausalLM` 完全相同**，没有任何 omni 专属的 AR 行为——
- input_embedding hook、1-D rope、no-deepstack 全部已在共享的 `MiMoV2Model` 内（按 forward_mode gate）；
- omni 的多模态合并在 stage0 (`MiMoV2_5Embedding`) 完成，AR 模型只读 `forward_batch.input_embedding`，与任何支持该字段的模型一致。

因此 `MiMoV2_5Generation` 只是 `MiMoV2ForCausalLM` 的零行为子类（仅多一行 warning 日志 + `*Generation` 命名），属冗余间接层；这恰是 review D2-7/D5-7 当初指出「文档残留 `MiMoV2_5Generation`、已正确清理为复用 `MiMoV2ForCausalLM`」的那个壳，上一小节把它又造了回来，是对 doc §5「generation.py」的过度解读。

已撤销：
- 删除 `models/mimo_v2_5/generation.py` 与 `test_mimo_v25_generation_registry_jax.py`；
- `stage.py` 移除 `MiMoV2_5Generation` import 与注册项；
- `mimo_v2_5_stage_config.yaml` AR `model_class` 改回 `MiMoV2ForCausalLM`（保留 fused-QKV vs flash 的 OPEN 注释）；
- `__init__.py` lazy map 移除 `MiMoV2_5Generation`；`test_mimo_v25_stage_registry.py` 断言改回 `MiMoV2ForCausalLM`；
- 设计文档 scope/§3.2/§3.3/§5 改为「AR 直接复用 `MiMoV2ForCausalLM`，不新增 `*Generation` 包装类」。

保留不变：`audio_encoder.py` / `weights_mapping.py` / `configuration.py` 的模块化拆分（这些是 embed stage 的 MiMo-V2.5 专属内容或有用重构，非冗余），以及 `MiMoV2_5Embedding`（embed stage 有专属 audio 塔，子类成立）。

### 修正：撤销 `configuration.py` / `MiMoV2_5Config`

时间：2026-06-05 CST。**本小节再次修正上面「保留不变」中关于 `configuration.py` 的说法。**

复核后确认 `configuration.py` 不该存在：
- `MiMoV2_5Config` 是**死代码**——runtime 模型加载走 `AutoConfig.from_pretrained(trust_remote_code=True)` 的 remote-code config，从不构造 `MiMoV2_5Config`；它只被自己的定义、`__init__` 导出和我写的测试引用。
- 它**不符合本仓约定**——`qwen2_5VL` / `qwen3_omni_moe` 等多模态模型都没有独立 config 模块，都是 inline `getattr` 读 config。
- 所谓「集中化」是伪需求——`audio_encoder` 读 `audio_config` 直接字段、`embedding` 读 3 个 token id（需 processor_config fallback），两者几乎无重叠；runner/tokenizer 里各自的 `_get_config_value` 副本我也没动，等于又加了第三份。

已撤销：
- 删除 `configuration.py` 与 `test_mimo_v25_config.py`；
- `audio_encoder.py.__init__` 改回 inline `getattr` + `int()`/`float()` 强转（保留 `speech_vocab_size` 等字符串字段的强转处理）；
- `embedding.py` 恢复自带的 `_get_config_value`（processor_config fallback）并 inline 解析 token id；
- `__init__.py` lazy map 移除 `MiMoV2_5Config`；设计文档 scope/§5/§6.1-B 改为「config 沿用 remote-code `AutoConfig`，inline 读取，不新增 config 类」。

最终 `mimo_v2_5/` 目录：`__init__.py` / `audio_encoder.py`（对标 qwen3_omni_moe）/ `embedding.py` / `weights_mapping.py`（对标 mimo_audio），与本仓约定一致，无异常 config 文件、无死代码。`str→int`/processor_config fallback 这些 MiMo-V2.5 特有的 config 坑改为 inline + 注释保留。

## embed 接口前瞻化（为 image/video 预留）

时间：2026-06-05 CST。

`MiMoV2_5Embedding` 的 `__call__` 改为**模态统一的 encode→scatter 流程**，让后续接 vision 只需「实现塔 + 实例化 self.visual + 加权重映射」，不动接口：

- **塔命名对齐 HF checkpoint**：`self.audio_encoder`（已接）+ `self.visual`（MiMoVL ViT，image/video 共用，本轮 = `None` 占位）。与 MiMo-V2.5 HF（`self.visual`/`self.audio_encoder`）和 qwen3_omni（`self.visual`）一致。
- **统一 encode**：`_encode_audio` / `_encode_image` / `_encode_video` 各返回 `[N, hidden]` 或 `None`；image/video 仅在「输入存在且 `self.visual is None`」时抛 `NotImplementedError`（之前是 `__call__` 顶部一刀切 raise），text+audio 请求正常。
- **统一 scatter**：`_merge_audio` 重命名为通用的 **`_scatter_modality(input_ids, input_embeds, modality_embeds, token_id)`**（static），三模态各 scatter 到各自 token id（位置不相交）。
- **token id 全预留**：`audio_token_id` / `image_token_id` / `video_token_id` 都已解析。
- **EmbedOutput** 保留 `deepstack_embeds`/`deepstack_pos_mask`（MiMo-V2.5 ViT 自带 merger、无 deepstack，恒 None）。
- **runner/`__call__` keyword 面**已是全模态超集（8 个 key），无需改动；vision 落地时在 `weights_mapping.py` 加 ViT 映射、在 runner 加 vision 校验即可。

接 vision 的 checklist（写在 `embedding.py` docstring）：实现 `mimo_v2_5/vision_encoder.py` → `__init__` 里 `self.visual = MiMoVLViT(...)` → `weights_mapping.py` 补 `visual.*` 映射。

附带修正：`test_mimo_v25_embedding_merge_jax.py` 原先把 `_ensure_channel_first_audio_codes` 误挂在 `MiMoV2_5Embedding`（实际在 encoder 上，因 jax 缺失被 skip 掩盖），本轮一并改对（scatter 测 embedding、axes 测 encoder）。

## 任意模态子集（不强假设全模态都在）

时间：2026-06-05 CST。

确认并加固「text/audio/vision 可任意子集输入，单独一种也行」：

- **embedding**：`__call__` 对每个模态走 `_encode_<m>`（输入缺失返回 `None`）→ `_scatter_modality`（`None` 或 token_id 缺失则 no-op）；text 永远 embed。故 text-only / audio-only / 组合均可。image/video 仅在「输入存在且 `self.visual is None`」时抛 `NotImplementedError`（ViT 未接，属 scope），不是"必须全有"。
- **embed_model_runner**：audio 校验/token_id 检查全部 gate 在 `has_audio`（payload/codes 存在）下；无 audio 的请求跳过。`mel-only not supported` 等 raise 只在"提供了 audio 但格式不对"时触发。
- **multimodal_tokenizer**：audio codec / payload 全在 `if audio_*` 下；text-only 直接文本 tokenize，不跑 codec。
- **能力 flag 改名**：`requires_mimo_v25_audio_contract` → **`uses_mimo_v25_audio_contract`**，避免"audio 必填"的误读——它只表示"该模型在收到 audio 时走 MiMo-V2.5 host 校验路径"，audio 本身可选。
- **新增测试**（jax-guarded，CI 跑）：`test_mimo_v25_embedding_merge_jax.py::TestMiMoV25AnyModalitySubset` 覆盖 text-only、audio-only、配了 audio_token_id 但无 audio 输入（no-op）、image/video 未接报错。

## host codec 归位：manager/ → models/mimo_v2_5/

时间：2026-06-05 CST。

`mimo_v25_audio_codec_processor.py` 原放在通用 `multimodal/manager/`（与 `io_struct`/`global_scheduler`/`stage` 等基础设施同级），但它是 **MiMo-V2.5 专属** host codec，与本仓约定不符——其它 MiMo-V2.5 代码都在 `models/mimo_v2_5/`。已归位：

- 文件移动并去前缀：`manager/mimo_v25_audio_codec_processor.py` → **`models/mimo_v2_5/audio_codec_processor.py`**（与 `audio_encoder.py`/`weights_mapping.py` 命名一致）。该模块只依赖 numpy/stdlib，对 `manager/` 无依赖，故 `manager → models` 是无环单向边。
- **解耦 io_struct**（review D5-8）：`io_struct.py` 之前在顶层硬 import 该 codec 并作为 `MiMoV25AudioPayload` 的 re-export 枢纽。改为 `TYPE_CHECKING` import（io_struct/schedule_batch 都有 `from __future__ import annotations`，payload 仅出现在注解里，运行时不需要）。通用 transport/Req 结构不再 runtime 依赖任何具体模型。
- **运行时消费者**（真正调用 codec 的）改为从新路径直接 import：`global_scheduler.py`、`multimodal_tokenizer.py`、`embed_model_runner.py`。
- `models/mimo_v2_5/__init__.py` lazy 导出 `MiMoV25AudioCodecProcessor`/`MiMoV25AudioPayload`。
- 5 个测试文件 import 路径同步更新；设计文档 §2.x/§4.3.x/§5 路径引用同步。

层次结果：`manager/`（infra：io_struct/schedule_batch）对模型零 runtime 依赖；只有 active 的 manager（tokenizer/scheduler）和 executor（embed runner）import 模型 codec——这是 payload-first 设计的必然，且与今天「manager import 一个放在 manager 里的模型专属文件」相比是更干净的方向。`mimo_v2_5/` 目录现为：`__init__.py` / `audio_codec_processor.py` / `audio_encoder.py` / `embedding.py` / `weights_mapping.py`，model-specific 代码全部内聚。

## tokenizer 去 MiMo-V2.5 audio 特判：host-adapter hook

时间：2026-06-05 CST。

**背景**：`multimodal_tokenizer.py`（通用组件）里有 ~25 处 MiMo-V2.5 audio 强绑定（4 个 `_*_mimo_v25_*` 方法 + `_tokenize_one_request` 中 ~15 处 `if is_mimo_v25` 内联分支）。查证真实 HF：MiMo-V2.5 的 processor 是 **`Qwen2_5_VLProcessor`（纯视觉，无 audio）**，audio(mel+RVQ codec)按设计在 processor 之外做——所以 host 侧自定义 audio 预处理是**硬约束**(非可选特例)。但把它**内联**进通用 tokenizer 是错的；其它带 audio 的模型(如 Qwen3-Omni)走通用 `AutoProcessor` 路径,tokenizer 里没有一个 `if is_qwen`。

**重构（hook 方案）**：
- 新增 `manager/audio_host_adapter.py`：`AudioHostAdapter` 接口 + `resolve_audio_host_adapter(mm_config, server_args)`（lazy 注册表，按 `matches()` 选中；选不到返回 `None` = 通用 AutoProcessor 路径）。tokenizer 不 import 任何具体模型。
- 新增 `models/mimo_v2_5/audio_host_adapter.py`：`MiMoV25AudioHostAdapter`，封装原 tokenizer 里所有 mimo audio 逻辑（`matches`/`from_request`/`ensure_payload`/`apply_to_input_ids`/`mm_inputs_entry` + `consumes_audio_data=True`/`uses_mrope=False` + lazy codec）。
- `multimodal_tokenizer.py`：删 4 个 `_*_mimo_v25_*` 方法 + `MiMoV25*` import；`__init__` 解析 `self.host_adapter`；`_tokenize_one_request` 所有 `if is_mimo_v25` → `if adapter`/`consumes_audio`/`uses_mrope` 通用判定；`transport_audio_codes` 由 `isinstance(MiMoV25AudioPayload)` 泛化为 `audio_payload is not None`。**tokenizer 现零 mimo 引用**。
- 顺带清掉死代码 `processor_out.get("audio_codes")`（Qwen 视觉 processor 不产 audio_codes）——并入 adapter.ensure_payload，仅在确有时生效。

**语义保持**：mrope 之前由 `not is_mimo_v25` 守，现由 adapter `uses_mrope`（MiMo=False）守；audio 绕过 AutoProcessor 由 `consumes_audio_data`（MiMo=True）；非 mimo 模型 adapter=None → 行为与重构前完全一致（audio 走 AutoProcessor、mrope 开）。

**验证**：`test_mimo_v25_multimodal_tokenizer_audio.py` 改为注入 `host_adapter`(含 fake codec) 后 4 例全过（raw audio→codec、缺 input_ids 报错、payload 优先于冲突 codes、非 mimo 仍要求 mm_processor）；resolver smoke 确认 mimo→adapter、qwen/None→None；全套 13 测试模块通过。

**扩展性**：再来一个 HF processor 不带 audio 的模型,只需在 `models/<model>/` 写一个 adapter 并加进 `_ADAPTER_SPECS`,tokenizer 不动。

## 2b + P1 落地进度（mm_items 单一真相源）

时间：2026-06-05 CST。按 §4.3.5 方案落地中。**本机无 jax/torch/transformers,纯逻辑用 numpy 测,跨模型部分 py_compile + stub 测,端到端真实回归留到具备依赖的环境(已确认)。**

**已落地 + 已测(numpy)**
1. `manager/mm_assembly.py::assemble_mm_inputs(mm_inputs)`：mm_items→per-modality kwargs(pixel_values/audio_codes 或 audio_features/grids)。audio 按 item 元数据(`is_codes`/`token_lengths`)路由 codes vs 连续。共享一处、模型无关。`test_mm_assembly`(5)过。
2. `models/mimo_v2_5/processor.py::MiMoV25Processor`：HF-processor 同形,组合 Qwen2.5-VL(视觉/文本)+ RVQ codec(audio);产 `audio_codes`+meta、展开 `<audio_pad>`、host 校验(D3-6 搬此)。matches()/from_hf_processor() 供选择。纯 codes/展开逻辑 numpy 可测(注入 fake HF+codec),`test_mimo_v25_processor`(4)过。
3. `manager/host_processor.py::resolve_host_processor`：lazy 注册表,按 capability 选 processor,选不到返回原 HF processor。
4. `multimodal_tokenizer.py`：`__init__` 用 resolve_host_processor 包装;`_tokenize_one_request` **回纯泛型**——删 adapter/`is_mimo_v25`/`consumes_audio`/`uses_mrope`;新增「`processor_out` 有 `audio_codes` → AUDIO mm_item(codes→feature、meta→model_specific_data、span→offsets)」通用规则;**mrope 改 `rope_scaling.mrope_section` 配置驱动**(`_uses_mrope`);audio 加载按 processor 有无 `feature_extractor` 决定(连续预加载 / codes 传原始源)。tokenizer 现零 mimo 引用。`test_mimo_v25_multimodal_tokenizer_audio` 改为断言 mm_items,(4)过。

**待落地(跨模型,本机不可端到端验证)**
5. `embed_model_runner._prepare_input` + `vit_model_runner`：改为 `assemble_mm_inputs(omni_inputs["mm_items"])` 取 kwargs;删 `_validate_mimo_v25_*`/`uses_mimo_v25_audio_contract`/payload normalize;连续 audio `feature_attention_mask` 变换按 meta 判定。
6. `global_scheduler.convert_omni_request`：只透传 mm_inputs(含 mm_items)+编排;删 `mm_items→req.pixel_values_*/audio_features` flatten;删 `mimo_v25_audio_payload` 透传 + `Req.audio_payload`。
7. 删 `audio_host_adapter.py`(两个)、`io_struct.audio_payload`、`Req.audio_payload`、flatten 字段;`MiMoV2_5Embedding` 读组装后的 audio kwargs。

**中间态警告**：tokenizer 现已输出「audio 在 mm_items」且**不再产 payload 侧信道**,而 scheduler/runner(任务 5/6 未做)仍按 payload 读 → **端到端在 5/6/7 完成前不连贯**(本机本就无法端到端跑;各组件 stub 测分别绿)。落地顺序保证每组件测试绿,完整链路待 5/6/7 完成 + 真实环境回归。

### 2b + P1 落地完成（5/6/7 已做）

时间：2026-06-05 CST。任务 5/6/7 已落地,中间态警告解除。

5. **executor 从 mm_items 组装**:`embed_model_runner._prepare_input` 改为 `assemble_mm_inputs(omni_inputs)` 取 kwargs,删 `_validate_mimo_v25_*`/`uses_mimo_v25_audio_contract`/payload normalize/`_get_config_value` 全家 + `np`/`math` import;连续 audio `feature_attention_mask` densify 改由「audio_features 存在且有 mask」通用判定。`vit_model_runner.forward` 改为 `assemble_mm_inputs` 取 `pixel_values`(image+video 合并)+grids。assembler 增加合并 `pixel_values` 字段。
6. **scheduler 变薄**:`global_scheduler.convert_omni_request` 只 `req.omni_inputs = mm_inputs` + 编排 + cache padding 用的 `all_mm_items` 归一;删整段 `mm_items→req.pixel_values_*/audio_features` flatten/concat + 删 mimo payload normalize 块 + 删 MiMoV25 import。
7. **删冗余**:删 `audio_host_adapter.py`(manager+model 两个,无外部 importer);删 `Req.audio_payload` + `Req.audio_features`/`pixel_values_images`/`pixel_values_videos`(全仓无读者,grep 确认;**保留 `Req.audio_codes`**——mimo-audio backbone 路径在用)+ schedule_batch 的 TYPE_CHECKING import;删 `TokenizedGenerateOmniReqInput.audio_codes/audio_payload`(死);`_create_tokenized_omni_object` 去掉 audio_codes/audio_payload 参数。`GenerateOmniReqInput.audio_codes/audio_payload` **保留**(tokenizer 经 getattr 读作请求侧预编码入口)。`MiMoV2_5Embedding` 不变(消费 audio_codes kwarg)。

**最终数据路径**:`请求 → MiMoV25Processor(__init__ 选)→ _tokenize_one_request(纯泛型)→ mm_items[AUDIO=codes] → GlobalScheduler(只透传+编排)→ assemble_mm_inputs(共享一处)→ embed/vit runner kwargs → 模型 forward`。tokenizer/scheduler 零 mimo 特判;mm_items 单一真相源,无 flatten 冗余、无 payload 侧信道、无 adapter。

**验证**:本机 15 个 mimo/multimodal/openai-audio 测试模块全过(numpy/stub);`compileall` 整个 manager+model_executor+mimo_v2_5 通过(无 import 断裂);全仓无 `audio_host_adapter`/`host_adapter` 残留。新增/重写测试:`test_mm_assembly`(5)、`test_mimo_v25_processor`(4)、tokenizer audio(4,改断言 mm_items)、embed runner(4,改测 mm_items 组装)、scheduler(3,改测薄 convert)。

**真实环境验收前置(本机不可端到端验证)**:① MiMo-V2.5 audio 端到端(raw audio→codes→scatter→AR);② Qwen2.5-VL / Qwen3-Omni 图/视频/音 **不回归**(executor 输入契约 + mrope 配置化改动面);③ `MiMoV25Processor` 组合(Qwen 处理器 + codec + 占位符展开 + host 校验)在有 transformers/torch 的环境跑通;④ stage 间 mm_items 的 ZMQ 序列化往返。

## audio encoder backbone 复用核对 + partial_rotary 守卫

时间：2026-06-05 CST。回应"直接复用 mimo-audio backbone 是否调研过"。

**逐行核对结论**：`audio_encoder.py` 复用 `MiMoAudioTransformer(MiMoAudioDecoderLayer)`。HF 真实 input_local 是 stock `Qwen2Model(partial_rotary_factor=1.0)`。逐项比对 `MiMoAudioDecoderLayer`/`MiMoAudioAttention` vs Qwen2：残差/pre-norm 顺序、q/k/v bias + o_proj 无 bias、SwiGLU(silu)、neox 半分 rope(`rotary_dim=head_dim, is_neox_style=True`)、无 qk-norm、非因果(`use_causal_mask=False`)——**当前 V2.5 配置下结构等价**。

**为何不能切到 `use_qwen2_layers=True`**：repo `Qwen2Attention` 只有 RadixAttention 一条路径(`token_to_kv_pool` 必填),而 encoder 是无状态(`token_to_kv_pool=None`)前向；只有 `MiMoAudioAttention` 的 branch-2(标准 softmax)支持。故复用 `MiMoAudioDecoderLayer` 是**必需**,非随意。

**`rotary_dim=head_dim` 硬编码来源 + 回归安全**：`MiMoAudioAttention` 是 mimo-audio 与 mimo-v2.5 **共享**层(mimo-audio 的 patch_encoder/patch_decoder 也用它,主 backbone 走 qwen2 路径)。mimo-audio config(`mimo_audio_backbone_config.py`)**无 `partial_rotary_factor` 字段**——硬编码 full rotary 就是对它的忠实实现。mimo-v2.5 真实 `partial_rotary_factor=1.0`——也是 full。两模型当前都 full,硬编码对两者都对。

**守卫(方案 A,零回归)**：在 `MiMoV25AudioUnderstandingEncoder.__init__` 加 `if partial_rotary_factor != 1.0: raise`，**只在 mimo-v2.5 侧、不动共享的 `MiMoAudioAttention`** → mimo-audio 完全不受影响(`git diff mimo_audio_backbone.py` 为空确认)。效果:未来若出现 partial-rotary 的 V2.5 variant,显式报错而非静默用 full 算错。新增 `test_init_rejects_partial_rotary_factor`(过)。`MiMoAudioAttention` 一字未改。
