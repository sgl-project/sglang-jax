# MiMo-V2.5 第二步接入 · 设计文档 + 代码 Review

> **范围**：审读 `docs/design/mimo_v25_step2_model_integration.md`（下称「设计文档」）与分支 `feat/mimo-v2.5` 的全部代码改动。
> **方法**：6 个维度并行 review（D1 架构数值保真 / D2 文档↔代码落差 / D3 embedding.py 正确性 / D4 audio 链路+复用 / D5 AR/配置/调度接线 / D6 测试有效性），每条 finding 由独立 adversarial verifier 回到源码 + 真实 HF 资料复核。
> **关键数据源**：本轮 review 在 `/tmp/mimo_hf/` 找到 **真实 HF `config.json` + `modeling_mimo_v2.py` + `audio_tokenizer/config.json`**（非 gated 缓存），故 §1 数值绝大多数可一手核验。
> **结论速览**：未发现 critical；**12 条 major（已确认）**；多条 minor / question；**6 条原始 finding 被复核推翻**（说明对应设计/代码其实正确，可放心）。最大问题集中在 **文档 scope 与代码落差**、**stage1 AR 类解析风险**、**OpenAI 多模态请求字段丢失**、**测试有效性虚高**。

---

## 1. Review 结论总表

| ID | 维度 | 严重度* | 一行结论 |
|---|---|---|---|
| D1-1 | 架构 | **major** | 视觉 `head_dim` 真值 **64**（`qk_channels` 默认），文档写 40 且 §6.3 易错项方向写反 |
| D2-1 | 落差 | **major** | §5「明确无改动」称 `is_multimodal` 沿用旧推导，实际新增 `is_multimodal_model()` + mimo_v2 分支 |
| D2-2 | 落差 | **major** | §5 把 `EmbedModelRunner` 归入「无改动」，实际 +261/-28，含多模型分支与整套 payload 校验 |
| D2-3 | 落差 | **major** | `http_server.py` 净删 ~133 行（移除 OpenAI prompt 抽取/路由），文档 §5 完全未提 |
| D2-4 | 落差 | **major** | stage1 `model_class: MiMoV2ForCausalLM` 被 `stage.py` 解析到 **Pro 融合 QKV 子类**，与实现记录「复用 flash」矛盾，存在按错误权重布局加载的风险 |
| D4-2 | audio | **major** | MiMo-V2.5 与旧 `MiMoAudioProcessor` 都靠 `model_path` 子串路由，本地目录名含 `audio` 会误判 |
| D5-5 | 接线 | **major** | `serving_chat` 多模态分支丢弃 `logprob/top_logprobs/extra_key`，`GenerateOmniReqInput` 无对应字段——影响**所有**多模态 chat 模型 |
| D6-1 | 测试 | **major** | `test_..._audio_encoder_axes` 用 `__new__` 只测 2 个纯函数，文件名误导成「encoder 已测」 |
| D6-2 | 测试 | **major** | §6.2 的「逐塔数值对齐 / mel parity / 端到端 golden」三类测试全部缺失 |
| D6-4 | 测试 | **major** | 多个测试用 numpy 桩替换 jnp，`.at[].set` 越界 clamp、dtype 提升、sharding 全被桩掉 |
| D6-5 | 测试 | **major** | host 侧 codec `encode`/`_waveform_to_mel`/`_tokenize_audio_batch` 零执行覆盖 |
| D6-10 | 测试 | **major** | backbone/ViT 数值易错项（KV4-8、head_dim、sink-SWA、partial_rotary、行列窗口、FP8、ep_size）无任何断言测试 |
| D1-2 | 架构 | minor | omni checkpoint 显式丢弃 MTP（`_keys_to_ignore_on_load_unexpected` 含 `model.mtp.*`），文档把「3 层 MTP」当必须对齐字段 |
| D1-4 | 架构 | minor | `audio/video/*_start/end` token id 实际只在 `processor_config` 子块，文档笼统记为已核 |
| D2-5 | 落差 | minor | §1–§4.2 用「原生 omni/三塔」叙事，但 vision 塔在 `embedding.py:345-348` 直接 `NotImplementedError` |
| D2-6 | 落差 | minor | `serving_chat.py` 新增 omni 分支属入口路由改动，§5「不新增 route」需收紧 |
| D3-6 / D4-6 | embed/audio | minor | `_merge_audio` 用 `jnp.nonzero(size=N, fill_value=0)`，`audio_token_id` 为 None 时校验早退 → 静默覆盖 token 0 |
| D3-7 / D4-4 | embed/audio | minor | `[20,20]` 方阵 codes 的 channel/time 轴无法消歧，host 传 channel-first 会被静默转置错位 |
| D3-8 | embed | question | `speech_embeddings`(不分片) / proj / text_embed 的 sharding 轴在 DP>1 需重核（本轮 DP=1 无害） |
| D4-1 | audio | question | raw audio→RVQ codes 路径**代码已写但从未跑通/数值对齐**，首轮可用入口实为预编码 codes |
| D4-5 | audio | question | RVQ codebook 是 per-quantizer（`[1024,1024,256,128,...]`），codec 用统一 1280 range 校验放行非法高位 code |
| D5-1 | 接线 | minor | `yaml_registry` 重新引入宽泛 `mimo-v2.5` 子串 fallback，`MiMo-V2.5-Pro` 会命中（实际触发受 `--multimodal` flag 收窄） |
| D5-2 | 接线 | minor | `config_registry` 新增 `MiMoV25AudioBackboneConfig` 对 V2.5 pipeline 是死代码（不走 `audio_backbone` stage） |
| D5-3 | 接线 | minor | §6.1 里程碑 A 称 flash「逐字段对齐」，实际 diff 仅 +9 行 input_embedding hook，字段是既有实现 |
| D5-4 | 接线 | minor | `embed_model_runner` 用 `__name__ == "MiMoV2_5Embedding"` 字符串硬编码分支，违背「模型无关契约」叙述 |
| D5-6 | 接线 | minor | 文档称 embed stage「读 `config.vision_config/audio_config`」，实际 `get_embed_model_config` 返回顶层扁平 config，由模型内部自取 |
| D5-8 | 接线 | minor | `io_struct` 顶层硬 import host-side codec 模块，耦合层级偏低（无循环 import，可接受） |
| D5-9 | 接线 | question | `global_scheduler` 仍无条件填 `req.audio_features`，但 tokenizer 对 MiMo 不写 AUDIO item，故恒空、无害 |
| D6-3 | 测试 | minor | `_merge_audio` scatter 无任何测试调用 |
| D6-6 | 测试 | minor | forward 测试硬编码 `deepstack_visual_embedding` 形状 `(3,1,6)`，把 no-op 占位当行为冻结 |
| D6-7 | 测试 | minor | 合并正确性只覆盖 audio 单段，未验证 host(token_lengths)↔encoder(真实行数) 等式 |
| D6-9 | 测试 | minor | `test_common_utils_logging` 属 plumbing，不应计入「模型接入测试覆盖」 |

\* 严重度为 adversarial verifier 复核后的修正值（部分较原始 finding 上下调整）。下列「已推翻」项不在表内。

---

## 2. 已推翻 / 已验证正确（可放心，无需改动）

复核中有 6 条 reviewer 提出的疑虑，回到一手 HF config 后被证明**设计/代码本就正确**，特此列出以建立信心：

- **D2-8（数值不可核验？→ 已核验全对）**：`/tmp/mimo_hf/config.json` 是可读的一手文件。§1.2–§1.4 的 hidden4096/48层、`n_routed_experts=256`、`9 Full+39 SWA`、KV `GA=4/SWA=8`、`head_dim=192/v_head_dim=128`、`partial_rotary=0.334`、`rope_theta 1e7/1e4`、`value_scale=0.707` 等**逐字段命中**，无错。
- **D3-2（speech vocab per-channel？→ V2.5 是 uniform）**：上游 `MiMoV2AudioConfig` 默认 `speech_vocab_size="1280"`、`audio_channels=20`，20 通道**共享同一 vocab**；旧 8-channel MiMo-Audio 才是异质列表。当前单标量实现与 V2.5 一致。
- **D3-4（input_local head_dim 16 / partial_rotary 0.334？→ 真值 64 / 1.0）**：真实 `audio_config` 为 `input_local_head_dim=64`、`partial_rotary_factor=1.0`，与 `embedding.py` 硬编码默认及文档 §1.4 一致，full rotary 正确。
- **D3-5（`add_post_norm=False` 会导致加载异常？→ 真值 true）**：真实 `audio_config.add_post_norm=true`，input_local 保留 RMSNorm，权重映射正确。
- **D3-9（audio 字段嵌套/落 processor_config 静默回落默认？→ config 是扁平的）**：真实 `audio_config` 扁平直挂全部结构字段，`getattr` 正常命中，不触发 fallback。
- **D5-7（文档残留 `MiMoV2_5Generation`？→ 设计文档中 0 处）**：该名只在 `implementation_notes.md` 的「已清理」记录里出现，被审文档无残留。

> 残留提醒（非缺陷）：上述虽正确，但 `embedding.py` 对 audio 结构字段普遍 `getattr(..., default)` 且无存在性校验、`partial_rotary` 硬编码 full rotary——与 `implementation_notes §7`「fallback 比 hard fail 更危险」原则方向不一致，可作为加固项（见 §4 建议）。

---

## 3. 详细 Major 问题

### 3.1 文档 scope 与代码落差（D2-1/2/3/5/6，D5-3/4/6）—— 本轮最系统性的问题

设计文档 §1–§4.2 通篇是「原生 omni · 三塔汇入」的**完整愿景**，而 `feat/mimo-v2.5` 实际落地的是 **audio-only V0 切片**：

- `models/mimo_v2_5/` 仅有 `__init__.py` + `embedding.py`；`vision_mimovl.py` / `audio_encoder.py` / `generation.py` / `*_weights_mapping.py` 全不存在（§5 已诚实列为「后续计划」，但 §1–§4.2 正文未在显著位置声明）。
- `embedding.py:345-348` 对 `pixel_values/pixel_values_videos` 直接 `raise NotImplementedError`；`__init__` 根本未构造 `self.visual`，§3.1「持有三塔」契约在代码层不成立（D3-1，已降为 nit）。

**「明确无改动」清单与 git diff 直接矛盾**（最易误导审读者评估回归面）：

| §5 声称「无改动」 | 实际 diff |
|---|---|
| `is_multimodal` 沿用旧推导（D2-1） | `model_config.py` 改 `is_multimodal = is_multimodal_model(...)`，新增整函数 + `model_type=="mimo_v2"` 分支 |
| `EmbedModelRunner` 继续沿用（D2-2） | +261/-28，含 kwargs 签名过滤(P0-2)、`_validate_mimo_v25_audio_*`、`_drop_consumed_*`、重启用 JIT split/merge |
| 入口路由不变（D2-3/D2-6） | `http_server.py` 删 `/v1/chat/completions` 路由（净 -133）；`serving_chat.py` 新增 `if is_multimodal: GenerateOmniReqInput(...)` 分支 |

此外 §6.1 里程碑 A 称 flash「逐字段对齐 V2.5」，实际 `mimo_v2_flash.py` 本轮只 +9 行 input_embedding hook，字段对齐是**既有 text-only 能力**（D5-3）；§3.4 称 embed stage「读 `vision_config/audio_config`」，实际 `get_embed_model_config` 返回**顶层扁平 config**（D5-6）。

**建议**：在 §1/§2 顶部加一行醒目 scope 声明（「本轮 V0 仅落地 audio 理解切片；vision/video/generation 为设计描述，stage0 对 image/video 抛 NotImplementedError」）；把 `is_multimodal_model`、`EmbedModelRunner`、`http_server.py`、`serving_chat.py` 从「无改动」移到「已改动（含回归面）」；§6.1 A 改为「复用既有字段实现 + 新增 input_embedding hook」。

### 3.2 stage1 AR 类解析到 Pro 融合 QKV 子类（D2-4）—— 权重加载风险

`mimo_v2_5_stage_config.yaml:22` 写 `model_class: MiMoV2ForCausalLM`（沿用 HF `architectures` 名）。但 `stage.py:13/259` 把该名 hard-map 到 `mimo_v2_pro.py:22` 的 **`MiMoV2ForCausalLM(MiMoV2FlashForCausalLM)`**——Pro 子类期望**融合 `qkv_proj` key + per-TP-shard FP8 反量化布局**；而 `implementation_notes:11` 与设计文档 §3.2 都说「复用 `MiMoV2FlashForCausalLM`」（读分离 `q/k/v_proj`）。

这是 **HF 架构名与本地 Pro 子类名碰撞** 导致的解析分歧：若 MiMo-V2.5 omni 主 checkpoint 的 QKV 不是 Pro 那种 per-shard 融合布局，走 Pro 类会找不到 `q/k/v_proj`、去找不存在的 `qkv_proj` → 加载失败或缺权重。

**建议**：先用真实 checkpoint 确认 QKV 布局（分离 vs 融合）；若分离则把 yaml 改为 `MiMoV2FlashForCausalLM`（需在 `stage.py` 注册该名），并统一文档/notes 的类名表述。

### 3.3 OpenAI 多模态请求字段丢失（D5-5）—— 影响所有多模态模型

`serving_chat.py` 的 `if is_multimodal:` 分支构造 `GenerateOmniReqInput` 时，相比 `else` 分支缺 `return_logprob=request.logprobs` / `logprob_start_len` / `top_logprobs_num=request.top_logprobs` / `extra_key`；`io_struct.py` 的 `GenerateOmniReqInput` dataclass 也无这些字段。后果：任何走多模态 chat 的请求带 `logprobs=true/top_logprobs>0` 会被**静默丢弃**且不报错（`implementation_notes` 自列为 P2-17 未修）。

**建议**：短期在多模态分支对 `logprobs/top_logprobs>0` 显式报 not-supported；长期给 `GenerateOmniReqInput` 补齐字段并贯通下游。

### 3.4 路由脆弱性（D4-2，关联 D5-1）

`multimodal_tokenizer.py:294` `is_mimo_audio = "mimo" in path and "audio" in path` 与 `_is_mimo_v25_model` 的 `"mimo-v2.5" in model_path` 都是子串匹配：

- 本地目录名如 `MiMo-V2.5-omni-audio` 会让两者**同时为真** → `mm_processor` 被设成旧 `MiMoAudioProcessor` 且 `mm_config=None`，而请求又按 v25 omni 走 codec/payload，processor 与 config 全错。
- `yaml_registry._KEYWORD_PATTERNS` 新增的 `mimo-v2.5` 子串会让 `MiMo-V2.5-Pro` 命中 omni 双 stage（与 `implementation_notes §9/P0-1`「避免宽泛 keyword fallback」自相矛盾；实际触发受 `--multimodal` flag + `is_multimodal_model` 双重收窄，故 D5-1 降为 minor）。

**建议**：路由统一改为 `model_type` + 真实 `vision_config/audio_config` 能力判定；删除宽泛子串 fallback，只保留精确 alias。同步修正 `implementation_notes §9` 与后续记录的互相矛盾。

### 3.5 测试有效性虚高（D6-1/2/4/5/10）—— 覆盖看似充足，实则避开了高风险路径

11 个新增测试**几乎全部集中在传输层**（payload 归一化、request 转换、stage 路由），而 D1–D5 关注的算法/数值/前向路径 **0 覆盖**：

- **D6-1**：`test_..._audio_encoder_axes.py` 用 `MiMoV25AudioUnderstandingEncoder.__new__()` 跳过 `__init__`，只测 `_ensure_channel_first_audio_codes` / `_group_audio_codes` 两个纯 numpy 函数；20 通道查表、`zeroemb` 屏蔽、6 层 transformer、proj→`[N,4096]` 全未执行。文件名却暗示「encoder 已测」。
- **D6-4**：6 个文件 `_install_import_stubs()` 把 `jnp/nnx` 桩成 numpy/object（`bfloat16=np.float32`、`nnx.Module=object`），于是 `.at[].set` 越界 clamp、dtype 提升、sharding、JIT shape 多态——这些**静默正确性 bug 高发区**全被绕过。
- **D6-5**：host codec `encode`→`_waveform_to_mel`(torchaudio `n_fft=960/hop=240`，§4.3.2 parity 关键点)→`_tokenize_audio_batch`(segment 6000 / `get_output_length` / `avg_pooler`) **零执行覆盖**，只测了 fake tokenizer 的权重缺失/缓存路径。
- **D6-2 / D6-10**：§6.2 的逐塔数值对齐 / mel parity / 端到端 golden 全缺；backbone 易错字段（KV4-8、head_dim、sink-SWA、partial_rotary 切片、FP8、ep_size 整除 256）无断言测试。

**建议**（这些大多**不依赖 gated 权重**即可做）：
1. 补 `_waveform_to_mel` 对已知波形的 mel 数值 parity（纯 torchaudio，离线对拍 ~1e-4）。
2. CPU 后端真实 `jnp` 跑一遍 encoder `codes→[N,4096]` + `_merge_audio` scatter，断言 `zeroemb` 屏蔽、reshape、越界不写 token 0。
3. backbone「config→层配置」断言测试（KV heads、`head_dim`、sink 层 mask、`partial_rotary` 切片长度）。
4. host(`token_lengths`)↔encoder(真实行数) 一致性测试，覆盖多段音频。

### 3.6 架构数值：视觉 head_dim（D1-1）

真实 `modeling_mimo_v2.py:791` `head_dim = getattr(config, "qk_channels", 64)`，而 `vision_config` **无 `qk_channels`** → `head_dim=64`（`qkv_dim=(32+2*8)*64=3072`，与 `hidden_size=1280` 解耦）。文档 §1.3 写 `1280/32=40`，§6.3 还把正确的 64 标成「model card 错误记法」——**方向写反**。按 40 实现 ViT 会与 HF 权重 shape 不匹配。

**建议**：§1.3 改 `head_dim=64`，删除 `1280/32` 推导；§6.3 改为「视觉 `head_dim=64`（来自 `qk_channels` 默认）」。

---

## 4. 静默正确性隐患（建议在落地前加护栏）

这些非当前 happy-path bug，但都是「误配置 → 无报错 → 数值悄悄错」的高危模式：

- **audio scatter 静默覆盖（D3-6/D4-6）**：`_merge_audio` 用 `jnp.nonzero(size=audio_embeds.shape[0], fill_value=0)`，当 `config` 缺 `audio_token_id` 时 host 校验早退（runner/codec 均 `if audio_token_id is None: return`），而模型内默认 `151669` 仍 scatter → 多余行写 token 0。**建议**：`audio_token_id` 解析失败 hard-fail；host 侧对 `sum(token_lengths)==audio_embeds.shape[0]` 做最终断言并打印 span。
- **`[20,20]` 方阵歧义（D3-7/D4-4）**：codec 与 embedding 都按「最后一轴==channel」启发式，真实 channel-first `[C=20,T=20]` 会被静默转置、time/channel 互换，且 range 校验（全局 min/max）抓不到。**建议**：host codec 固定输出一种 layout（推荐 `[T,20]`）并在 payload 标注，禁止依赖形状推断。
- **RVQ encode 未跑通（D4-1）**：`encode_mels`→`_tokenize_audio_batch` 调 `encoder.encode(return_codes_only=True)` 的 kwarg 名、返回形态、`get_output_length` 均依 HF remote code 推断编写，**从未与真实 checkpoint 数值对齐**。**建议**：在 §6.1 F 里程碑显式标注 raw-audio→codes 为「代码已写未对齐(question)」，验证前不要宣称 audio 端到端可用。
- **per-quantizer codebook（D4-5）**：真实 `audio_tokenizer/config.json` 的 `codebook_size` 是逐量化器的（`[1024,1024,256,128,...]`），codec 用统一 `1280` range 校验会放行高位 channel 的非法 code。**建议**：用真实 per-quantizer `codebook_size` 做逐 channel 校验。
- **audio 结构字段无存在性校验（§2 残留提醒）**：与 `implementation_notes §7` 的 hard-fail 原则统一，对关键字段（`audio_channels/input_local_*/group_size`）命中失败应 warn/raise，而非静默用默认。

---

## 5. 建议处理顺序

1. **落地前必须澄清/修**（阻塞正确性）：D2-4（AR 类解析与权重布局）、D1-1（ViT head_dim，影响后续 ViT 落地）、D4-2（路由碰撞）、D5-5（多模态 logprob 丢失）。
2. **文档一致性（一轮集中修）**：D2-1/2/3/5/6、D5-1/3/4/6、D1-2/4、§9 自相矛盾——把「无改动」清单、scope 声明、MTP/字段对齐表述一次性对齐代码现状。
3. **护栏 + 测试（落地 audio 端到端前）**：§4 全部静默隐患的护栏；D6-2/4/5/10 的不依赖 gated 权重的测试（mel parity、CPU jnp encoder/scatter、backbone config 断言）。
4. **可延后**：D5-2（死代码清理）、D5-8（解耦）、D6-6/7/9（测试归类与补段）、D3-8（DP>1 sharding，本轮 DP=1）。

---

## 附录 · 数据源与方法

- 设计文档/代码：`feat/mimo-v2.5` @ `b9ce2fcc`，`git diff` + 新增文件 Read。
- HF 真实结构：`/tmp/mimo_hf/{config.json, modeling_mimo_v2.py, audio_tokenizer/config.json}` 一手核验；上游 `sgl-project/sglang` `mimo_v2.py`/`mimo_audio.py` 交叉对照。
- 每条 finding 经独立 adversarial verifier 复核（confirmed / refuted / uncertain + 修正严重度）；本文只采纳 confirmed 与有价值的 uncertain/question，6 条 refuted 见 §2。
- `文件:行号` 为审读时所见，实施前请复核。


初始化阶段
      - 引入 resolve_host_processor，加载 HF AutoProcessor/AutoConfig 后允许按模型能力包装 processor：python/sgl_jax/srt/multimodal/manager/
        multimodal_tokenizer.py:325。
      - MiMo-Audio 识别从 "mimo" and "audio" 改成 "mimo-audio"：python/sgl_jax/srt/multimodal/manager/multimodal_tokenizer.py:287。这是合理的，否则 MiMo-V2.5
        omni 路径里如果目录名带 audio，很容易误判成纯 MiMo-Audio。
      - 日志 skip list 加入 audio_data/audio_codes/audio_payload 等大字段，避免 request logging 打爆日志：python/sgl_jax/srt/multimodal/manager/
        multimodal_tokenizer.py:356。
  2. 新增 helper
      - _get_config_value/_get_mm_config_value 支持从 processor_config、thinker_config 里取 token id：python/sgl_jax/srt/multimodal/manager/
        multimodal_tokenizer.py:406。
      - _uses_mrope 改为只看 rope_scaling.mrope_section，避免 MiMo-V2.5 这种 1D RoPE 模型误走 Qwen mrope：python/sgl_jax/srt/multimodal/manager/
        multimodal_tokenizer.py:427。
  3. 请求处理阶段
      - 新增 audio_payload/audio_codes 支持，允许内部/debug request 直接传预编码 codes：python/sgl_jax/srt/multimodal/manager/multimodal_tokenizer.py:575。
      - needs_mm_processor 从只看 image/video/audio_data，扩展到包含 audio_payload/audio_codes：python/sgl_jax/srt/multimodal/manager/
        multimodal_tokenizer.py:594。
  4. audio routing
      - 如果 processor 有 feature_extractor，继续把 audio source 预加载成 waveform，这是 Qwen3-Omni 这类 continuous audio processor 的路径：python/sgl_jax/srt/
        multimodal/manager/multimodal_tokenizer.py:614。
      - 如果 processor 没有 feature_extractor，直接把 raw audio 或 audio_payload/audio_codes 交给 processor。MiMoV25Processor 故意不暴露 feature_extractor，因
        此会走 host codec/code path：python/sgl_jax/srt/multimodal/manager/multimodal_tokenizer.py:616。
  5. processor 输出转 mm_items
      - 原来只把 audio_features/input_features 当 continuous audio 放进 mm_items。
      - 现在优先识别 processor_out["audio_codes"]，创建 Modality.AUDIO item，并带 is_codes/token_lengths/group_size/codebook_sizes：python/sgl_jax/srt/
        multimodal/manager/multimodal_tokenizer.py:683。
      - 如果没有 audio_codes，仍保留原来的 continuous audio_features/input_features 路径：python/sgl_jax/srt/multimodal/manager/multimodal_tokenizer.py:700。
  6. mm_inputs 传递
      - mm_inputs 现在成为 omni 请求的主要载体，TokenizedGenerateOmniReqInput 创建时直接带上 mm_inputs：python/sgl_jax/srt/multimodal/manager/
        multimodal_tokenizer.py:1044。
      - 普通 MM tokenized object 也会设置 tokenized_obj.mm_inputs = mm_inputs：python/sgl_jax/srt/multimodal/manager/multimodal_tokenizer.py:764。

       1. _create_tokenized_omni_object(1035)仍带 audio_codes/audio_payload 参数 + transport_audio_codes 逻辑,调用方已不传;io_struct.TokenizedGenerateOmniReqInput
  仍有这两个字段——是上一轮"删除"没真正落地的死代码(与 notes 记录不符)。
  2. __init__ 顶部 is_mimo_audio 子串特判仍在(老 mimo-audio 没纳入 processor 机制,与"tokenizer 零模型特判"的目标不完全一致)。
  3. log skip-names 仍列 audio_codes/audio_payload(无害,但也是旧字段残留)。

---

# 6. 二轮全流程复核 · audio+text V0(清理后,2026-06-07)

> **触发**:在删除请求侧 codes/payload 入口、统一 thinker 解析、`mm_inputs` 单点挂载、`_load_audio_from_source` 清理、`_is_qwen_video_processor` 认 wrapper 等一系列整改后,对 **MiMo-V2.5 audio+text(排除 vision 输入)端到端通路** 做的第二轮复核。
> **方法**:4 个并行子代理分片**通读全文**(非 diff)——①请求/传输/调度 ②host 侧 codec/processor ③embed stage/model ④stage 配置/registry/AR 接线;每条承重 finding 由我**回到源码逐条核验**(标注 ✅ 已核 / ⚠️ 未对真实 checkpoint 验证 / ❌ 推翻)。
> **结论速览**:无回归(本轮整改自洽);**3 条 critical / 8 条 major**,几乎全部集中在 **"从未对真实 checkpoint 跑过" 的 audio 编码链路 + AR 类/资源** 上;**1 条上轮疑似问题被推翻**(澄清后可放心)。

## 6.1 结论总表

| ID | 切片 | 严重度 | 核验 | 一行结论 |
|---|---|---|---|---|
| R2-1 | AR 接线 | ~~critical~~ → **推翻** | ✅ 一手核 | config.json `attention_projection_layout="fused_qkv"` + modeling:278 一手确认 checkpoint 是**融合 qkv**;yaml 用 `MiMoV2ForCausalLM`(Pro 融合类)**恰好正确**,文档/notes「复用 Flash/分离 q/k/v」表述说反了。详见 §6.2 RF-2 |
| R2-2 | stage 配置 | **critical** | ✅ | AR `num_tpus` 现已设 16(512GB,权重占比 ~60%);~300B FP8 256-expert 需 `du -sh` 校准,余量紧建议 v6e-32 |
| R2-3 | codec | ~~critical~~ → **降级 minor** | ✅ 一手核 | `encode(input_features=,input_lens=,return_codes_only=True)`(modeling:1477)签名/2-tuple 返回/`get_output_length`(1425)/`avg_pooler`/`down_sample_layer` **全部存在且与 codec 调用对齐**;仅剩权重**数值正确性**需真权重跑(见 §6.2 RF-3) |
| R2-4 | codec | ~~major~~ → **推翻** | ✅ 一手核 | `MiMoAudioTokenizer`/`Config`/`Encoder` **就定义在 `modeling_mimo_v2.py`**(1500/1045/1374),`audio_tokenizer/` 只放 config.json → 从 `modeling_mimo_v2.{symbol}` 加载**正确**(见 §6.2 RF-4) |
| R2-5 | 权重映射 | **major** | ✅ → **已修** | `weights_mapping.py` HF 源 key 前缀**自相不一致**(text `model.` / speech 裸 `speech_embeddings.` / audio 塔 `audio_encoder.`)且未对真实 checkpoint 验证;漏 key 只 warn+skip → audio 塔静默随机初始化 |
| R2-6 | embed model | **major** | ✅ → **已修** | `_scatter_modality` 用 `nonzero(size=rows, fill_value=0)`:placeholder 数 < codes 行数时多余行**静默覆盖 token 0**;唯一护栏 host 端校验在 `audio_token_id is None` 时早退 |
| R2-7 | 路由一致性 | **major** | ✅ → **已修** | `is_multimodal_model` 只认 `model_type=="mimo_v2"`+config(`model_config.py:796`),而 `processor.matches()` 还认 `mimo_v2_5`;真实 config 若是 `mimo_v2_5`,**多模态路由根本不触发** |
| R2-8 | codec | **major** | ✅ → **已修** | `get_codebook_sizes()` 配置缺失/解析失败时回落统一 `1280` → 低 codebook 通道非法高位 code 被放行(=D4-5)。per-quantizer 校验逻辑本就存在,缺口是**静默回落**;一手 `audio_tokenizer/config.json` 已含 `codebook_size=[1024,1024,256,128×17]` |
| R2-9 | codec | ~~major~~ → **部分推翻 / 降级** | ◑ 一手核 | 结构参数 `nfft960/n_mels128/sr24000/hop240/fmin0/fmax null` 与 codec `_waveform_to_mel` **逐项命中**(audio_tokenizer/config.json);仅 power/log/window 细节在 modeling 外无法证伪,保留为 ⚠️ 待真权重(见 §6.2 RF-5) |
| R2-10 | processor | ~~major~~ → **推翻** | ✅ 一手核 | `config.json:33 speech_vocab_size:"1280"`(字符串标量,**非 list**)→ `matches()` 里 `int("1280")` 安全,不会 `int(list)` 崩(见 §6.2 RF-6) |
| R2-11 | codec | **major** | ⚠️ | 切分用的 `code_lengths`(`get_output_length`+pooler 公式)与实际返回 `codes` 是**两条独立路径**,漂 1 帧就 `torch.split` 错位/抛错 |
| R2-12 | AR 接线 | minor | ✅ | `ep_size` 不从 stage yaml 取,来自全局 `--ep-size`;须同时整除 256 与 AR 设备数,fused 不一致仅 warn,EPMoE 不一致会错分片 |
| R2-13 | 调度 | minor | ✅ → **已修** | omni 每请求都算 `cache_input_ids` 但**无人消费**(死计算) |
| R2-14 | 调度 | minor | ✅ → **已修** | `to_stage_reqs` debug 日志在 v2.5 路径上读老 `Req.audio_codes`(恒 None,调试时易误导) |
| R2-15 | codec/embed | minor | ✅ → **已修** | `audio_token_id` 为 None 时:占位展开 / host 校验 / scatter **三处全静默 no-op**——护栏恰好以"会失效的那个值"为条件 |
| R2-16 | codec/encoder | minor | ✅ 一手核 | 方阵 `[20,20]` 轴消歧靠假设——但真实 `_pad_and_group_audio_codes`(modeling:541)确认 time-major `[T,C]` 末轴=channel,**编码器假设被一手证实正确**;`codes_layout` 是 transport 信息字段(编码器收到裸数组、无法读它),保留作文档,非缺陷 |
| R2-17 | yaml_registry | nit | ✅ → **已修** | `pro`/`flash` 子串排除无词边界,含这些子串的路径会误排 |
| R2-18 | 入口 | nit | ✅ → **已修** | `serving_chat` 在 hard-raise 之后仍把 `return_logprob/top_logprobs` 塞进 `GenerateOmniReqInput`(死赋值) |

## 6.2 已推翻(澄清后可放心)

- **RF-1(❌ 推翻)·"codec 发 `[T,20]` 与 `_build_audio_segment` 的 `[C,T]` 布局契约冲突"**:核源确认 `_build_audio_segment` / `_build_backbone_input`(`schedule_batch.py:343-451`)是**老 MiMo-Audio 8 通道 backbone 路径**(`audio_mode in {asr,tts,audio_understanding}`、读 `Req.audio_codes`),**v2.5 通路根本不进**(v2.5 走 `mm_items` + embedding stage)。v2.5 codes 的真实消费者是 `audio_encoder._ensure_channel_first_audio_codes`(time-major、末轴=channel),与 codec emit 的 `[T,20]` 一致。两个 backbone 的字段同名(`audio_codes`)造成了误判。残留只剩 `codes_layout` 写而不读这条 nit(并入 R2-16)。
- **RF-2(❌ 推翻 R2-1,原判 critical)·"AR 类解析到 Pro 融合 qkv,若 checkpoint 分离 q/k/v 则加载崩"**:这条**不需要 gated 权重**即可一手核验——`/tmp/mimo_hf/` 已有真实 `config.json` + `modeling_mimo_v2.py`。核验:`config.json:14 "attention_projection_layout": "fused_qkv"`,配合 `modeling_mimo_v2.py:278 if self.projection_layout == "fused_qkv": self.qkv_proj = nn.Linear(...)`——该 omni checkpoint **确定走融合 qkv** 分支,存的是 `self_attn.qkv_proj.weight`,无分离 q/k/v。故 `stage.py` 把 `MiMoV2ForCausalLM` 解析到 `mimo_v2_pro` 的融合 qkv 子类**恰好正确**;反而是设计文档/notes「复用 `MiMoV2FlashForCausalLM`(分离 q/k/v)」的表述写反了,照那条做会加载失败。**已修**:notes:11、yaml AR 注释(OPEN 收敛为 CONFIRMED)。教训:`configuration_mimo_v2.py:122` 类默认是 `"split"`,只看默认值会判反,**必须看 checkpoint 自带 config.json 的显式覆盖值**。(原 D2-4 同此,一并消解。)
- **RF-3(R2-3 降级 critical→minor)·"raw-audio→codes 编码 API 全是猜测"**:一手 `modeling_mimo_v2.py` 核验:`AudioTokenizerEncoder.encode(input_features, input_lens=None, ..., return_codes_only=False)`(:1477)签名存在;`return_codes_only=True` 时 `return codes, output_length`(2-tuple、codes 在前)与 codec `codes, _ = encoder.encode(...)` 调用对齐;`get_output_length(mel_len)`(:1425)、`down_sample_layer`/`config.avg_pooler`(:1407-1414)均存在;`quantizer.encode` 返回 `torch.stack(all_indices)` = `[n_q, T]`,codec `.transpose(0,1)`→`[T,n_q]` 一致。**API 形状/签名/返回结构全部对齐**,不再是"猜测"。残留仅**权重数值正确性**(codes 是否与官方逐位一致)——那确属 runtime 项,降为 minor。
- **RF-4(❌ 推翻 R2-4)·"tokenizer 类模块路径错"**:`MiMoAudioTokenizer`(:1500)、`MiMoAudioTokenizerConfig`(:1045)、`AudioTokenizerEncoder`(:1374)**都定义在 `modeling_mimo_v2.py`**;`audio_tokenizer/` 子目录只含 `config.json`(无 .py)。codec 用 `get_class_from_dynamic_module("modeling_mimo_v2.MiMoAudioTokenizer", ...)` **正确**。
- **RF-5(R2-9 部分推翻)·"mel 参数未对齐"**:一手 `audio_tokenizer/config.json` 的 `nfft:960 / n_mels:128 / sampling_rate:24000 / hop_length:240 / fmin:0 / fmax:null` 与 codec `_waveform_to_mel` **逐项命中**(此前怀疑 `n_mels=128` 是猜测,实为正确)。剩 `power`/`log` 底数/`mel_scale`/window 类型在 modeling 中无对应特征提取代码,无法用这份文件证伪 → 保留为 ⚠️ 待真权重对拍(降级,不再笼统记"全未对齐")。
- **RF-6(❌ 推翻 R2-10)·"`matches()` 遇 list 型 speech_vocab 崩"**:一手 `config.json:33 "speech_vocab_size": "1280"`(**字符串标量**,非 per-quantizer list;per-quantizer 的是 RVQ codec 的 `codebook_size`,属另一个 vocab)。`_audio_contract` 里 `int("1280")` 安全,`matches()` 不会 `int(list)` 抛 `TypeError`,也不会因此 fall-through。R2-8 的 per-quantizer 风险仍成立(那是 codec RVQ 校验,不是 speech 塔 vocab)。

## 6.3 关键 major/critical 详述

### R2-5 / R2-6 / R2-7 — 已修(2026-06-07,不依赖 gated 权重)
- **R2-7**(`model_config.py:is_multimodal_model`):把 `model_type == "mimo_v2"` 改为归一化后匹配 `{mimo_v2, mimo_v2_5, mimo_v25, mimo2_5}`(与 `processor.matches()` 同一集合,统一 `-`/`.`→`_`),仍以 vision/audio config 存在为门控,text-only Pro/Flash 不受影响。真实 config 无论 model_type 是 `mimo_v2` 还是 `mimo_v2_5`,多模态路由都会触发。
- **R2-6**(scatter 静默覆盖):①`embedding._scatter_modality` 的 `jnp.nonzero` 改 `fill_value=seq_len`(永远越界的行号)+ `.at[].set(..., mode="drop")`,placeholder 不足时多余行**丢弃而非写 token 0**;②`embed_model_runner._validate_audio_placeholder_contract` 在"有 audio codes 但 `audio_token_id` 为 None"时**hard-fail**(原为静默 return),并加回归测试 `test_audio_codes_without_audio_token_id_raises`。
- **R2-5**(audio 塔权重静默随机初始化):`embedding.load_weights` 在 `load_weights_from_safetensors` 前新增 `_assert_audio_tower_weights_present`,用 `loader.has_weight_on_disk` 探测所有 `speech_embeddings.*` / `audio_encoder.*` HF key;任一缺失即 `raise`(列出样例 key),把"漏 key→warn+skip→随机初始化"变成上真机即暴露。**注**:这是护栏,不是对真实 key 名的确认——首次加载真实 checkpoint 若触发,即说明 `weights_mapping.py` 前缀需按真实 tensor 名校正(R2-5 根因仍在,只是不再静默)。

### R2-1 / R2-2 / R2-12 — AR 阶段
- **R2-1(已推翻,见 §6.2 RF-2)**:一手 `config.json:attention_projection_layout="fused_qkv"` + `modeling_mimo_v2.py:278` 确认 checkpoint 融合 qkv,yaml 现用的 `MiMoV2ForCausalLM`(Pro 融合类)正确,无需改;文档/notes 表述已修正。
- **R2-2**:AR `num_tpus` 现已设为 16(4×→16×);~300B FP8 权重 ~300GB,16×v6e=512GB 权重占比 ~60%,KV/activation 余量偏紧,稳妥用 v6e-32(见正文 §TPU 估算)。`du -sh` 真实 safetensors 校准后定档。
- **R2-12**:`config.ep_size` 来自 `model_runner.py:303`= 全局 `--ep-size`,stage yaml 没 `ep_size`;须同时整除 256 与 AR mesh 设备数。fused 路径与 mesh 不一致**仅 warn**(`scheduler.py:304`),EPMoE 路径会真错分片。建议 AR stage 显式 pin ep_size。

### R2-3 / R2-4 / R2-9 / R2-11 — raw-audio→codes:整条链未对真实 checkpoint 跑过
`audio_codec_processor.py:809-844 _tokenize_audio_batch` + `:736-790 _load_audio_tokenizer`:`tokenizer.encoder` 属性名、`encode(input_features=,input_lens=,return_codes_only=True)` 签名与 2-tuple 返回、`codes` 是 `[C,ΣT]` 故 `.transpose(0,1)`→`[T,C]`、`get_output_length`/`down_sample_layer`/`config.avg_pooler` 存在性、以及 **从 `modeling_mimo_v2.py` 取 tokenizer 类(R2-4,实际类大概率在 `audio_tokenizer/` 子模块)**——**全是假设**。叠加 R2-9(mel 参数未对齐)、R2-11(切分长度与实际 codes 双路径易漂),这条链是首测音频最可能崩/最可能"不崩但乱"的地方。**强烈建议真机首测先单独验证 `codec.encode` 对一段已知音频的 codes 形状/数值,再开端到端。**

### R2-5 / R2-6 / R2-7 — 静默正确性陷阱
- **R2-5**:`weights_mapping.py` 三类 HF 源 key 前缀不一致(`:16` `model.embed_tokens` / `:27` 裸 `speech_embeddings.{i}` / `:37,43` `audio_encoder.input_local_transformer.*`);`weight_utils` 对漏 key 仅 `warn + continue` → audio 塔可能整段停在随机初始化而**不报错**。至少 speech_embeddings 与 input_local 的前缀应统一(要么都带 `audio_encoder.`,要么都不带),并对真实 checkpoint key 名核对。
- **R2-6**:`embedding.py _scatter_modality` 的 `jnp.nonzero(mask, size=modality_embeds.shape[0], fill_value=0)`——placeholder 不足时多余行落到 index 0,`.at[0].set` 静默覆盖首 token;`token_id is None` 时整段 no-op(audio 静默丢弃)。唯一护栏 `embed_model_runner._validate_audio_placeholder_contract` 在 `audio_token_id is None` / 非 dict 时早退。**建议**:`audio_token_id` 解析失败应 hard-fail;in-graph 加 `sum==rows` 断言而非依赖可跳过的 host 校验。
- **R2-7**:`model_config.py:796` 只认 `model_type=="mimo_v2"`+(vision/audio config);`processor.py:27,75` 还认 `{mimo_v2_5,mimo_v25,mimo2_5}`。若真实 omni config 的 `model_type` 是 `mimo_v2_5`,`is_multimodal_model` 返 False → 根本不走多模态 tokenizer/scheduler,processor.matches 永不被问。**两处必须对齐到真实 config 的 model_type**。

## 6.4 已核验正确(勿再标问题)

- `mm_inputs` 是单一真相、`convert_omni_request` 按引用透传、**不** 回填 `Req.audio_features`/payload(对齐设计)。
- `mm_items`(numpy + python 原生 meta;torch 已在 codec 内 `.detach().cpu().numpy()`)**可过 ZMQ**;`multimodal_embedding` 走 stage 间 **in-memory queue(非 ZMQ)** 共享 dict 引用,AR 端 shape 校验。
- AR `input_embedding` hook(`mimo_v2_flash.py:471-480`):extend 模式用预算 embedding、跳过 `embed_tokens`,**无重复 embed**;两个 AR 类都继承。
- vision-deferral 守卫以 `pixel_values is not None` 为前提,**audio+text 路径不会误触**。
- `group_size` padding 不虚增 placeholder;单 payload 的 count 数学自洽。
- partial_rotary 守卫作用域正确,**mimo-audio backbone 未受影响**。
- `__init__` lazy PEP562 不会把 jax/torch 拉进 host tokenizer 进程。
- `is_multimodal_model` 不会误判 text-only Pro/Flash;`yaml_registry` omni 命中、Pro/Flash 排除、无 mimo-audio 碰撞(除 R2-17 nit);`config_registry` 无 `MiMoV25AudioBackboneConfig` 死码(已清)。

## 6.5 上真机前处理顺序

> 进度标注:R2-1 已一手核验推翻(yaml 本就正确);R2-5/6/7 已修(§6.3 修复记录);R2-2 num_tpus 已设 16(待 `du -sh` 校准定档)。

1. **阻塞(必须先定)**:R2-2(AR num_tpus 按真实权重大小定 v6e-16/32)。~~R2-1~~(已推翻)、~~R2-7~~(已修)。
2. **音频首测前**:R2-3/4/9/11 先离线单测 `codec.encode`(形状+数值),再开端到端。~~R2-5~~(已加载前 hard-fail 护栏)、~~R2-6~~(已 drop-mode + hard-fail + 测试)。
3. **加固**:R2-8(per-quantizer codebook 校验)、R2-10(`matches()` 容错 list 型 vocab)、R2-15(None token id 全链 hard-fail)。
4. **可延后**:R2-12(pin ep_size)、R2-13/14(死码/误导日志)、R2-16(layout 字段)、R2-17/18(nit)。

---

# 7. 最后一轮全通路 review · text/audio only(2026-06-07)

> **范围**:本轮按用户要求只覆盖 **text + audio** 端到端链路,不把 vision 输入纳入本轮阻塞项;但会检查新增通用多模态改动是否回归现有 Qwen2.5-VL / Qwen3-Omni 等模型。
> **方法**:7 个角度并行扫 diff + 非 diff 全通路(逐 hunk、删除行为、跨文件调用、codec、embed/AR、cleanup/altitude、现有模型回归),再回到本地源码与 `/tmp/mimo_hf/{config.json,modeling_mimo_v2.py,audio_tokenizer/config.json}` 一手核验。
> **结论速览**:无新的 AR/embedding 数值 blocker;上一轮若干高风险项已被代码或 HF 一手资料推翻/收敛。剩余最值得在端到端前处理的是 **OpenAI 多模态请求字段/错误码回归、radix cache 与 audio 内容隔离、raw-audio codec parity 未验证、以及 host processor 失败静默降级**。

## 7.1 新增/仍成立的确认问题

| ID | 切片 | 严重度 | 核验 | 一行结论 |
|---|---|---|---|---|
| R3-1 | OpenAI 多模态入口 | **major** | ✅ | `serving_chat.py:99-112` 构造 `GenerateOmniReqInput` 时漏传 `n=request.n`;现有多模态 chat `n>1` 会退化为 1 个 choice |
| R3-2 | OpenAI 多模态入口 | **major** | ✅ | `request.rid` 仍允许 `list[str]`,但 omni path 直接透传;`rid_to_state[tokenized_obj.rid]` / `req_store[req.rid]` 用 list 作 dict key 会 `TypeError` |
| R3-3 | cache 隔离 | **major** | ✅ | `extra_key` 从 OpenAI 请求进了 `GenerateOmniReqInput`,但 `TokenizedGenerateOmniReqInput` 与 `Req.to_stage_reqs()` 均未转发;多模态 cache namespace 隔离丢失 |
| R3-4 | cache 隔离 | **major** | ✅ | `convert_omni_request()` 计算了带音频 hash 的 `req.cache_input_ids`,但 AR `TokenizedGenerateReqInput` 没接收;不同音频但相同 `<audio_pad>` 文本可能复用错误 KV 前缀 |
| R3-5 | OpenAI 多模态入口 | **major** | ✅ | string-format chat template 会在 `process_content_for_template_format()` 里丢弃 audio/image/video parts;使用 string/custom template 的现有多模态模型会静默变 text-only |
| R3-6 | OpenAI 错误码 | **major** | ✅ | 多模态 `logprobs/top_logprobs` 的显式 `ValueError` 发生在 conversion 阶段,被 `OpenAIServingBase.handle_request()` 包成 HTTP 500,应是 4xx unsupported 参数 |
| R3-7 | tokenizer manager | **major** | ✅ | `_send_one_request()` 先 `send_pyobj()` 再登记 `rid_to_state`;快速完成/报错的 omni 请求可能先回包后建 state,输出被 unknown-rid 路径丢掉 |
| R3-8 | host processor | **major** | ✅ | `resolve_host_processor()` 吞掉任意 import-time Exception 并回落裸 HF processor;MiMo wrapper 缺依赖/导入 bug 会延迟成无关的 audio 预处理失败 |
| R3-9 | raw-audio codec | **major** | ⚠️ | `_waveform_to_mel()` 仍未与 HF/上游 codec 做数值 parity;即使 `encode()` 形状跑通,mel 滤波/归一差异会导致 audio_codes 语义漂移 |
| R3-10 | stage config registry | minor | ✅ | `_match_mimo_v25_omni()` 在完整 path 上排除任意 `pro`/`flash` 子串;`/prod/MiMo-V2.5`、`/mnt/flash-cache/MiMo-V2.5` 会误判找不到 omni yaml |

### R3-1 / R3-2 / R3-3 / R3-6 — OpenAI 多模态入口字段回归

- `ChatCompletionRequest.rid` 类型仍是 `list[str] | str | None`(`protocol.py:451`),但 `GenerateOmniReqInput.__post_init__` 只在 `None` 时生成 uuid(`io_struct.py:161-163`),不处理 list。随后 `_send_one_request()` 用 `tokenized_obj.rid` 作为 dict key(`multimodal_tokenizer.py:1022`),`GlobalScheduler.convert_omni_request()` 也用 `req.rid` 查 `req_store`(`global_scheduler.py:342-344`)。**复现**:OpenAI multimodal chat 带 `rid=["a"]` → tokenized 前后都保持 list → unhashable list 崩溃。
- `serving_chat.py:99-112` 漏 `n=request.n`,而 `_create_tokenized_omni_object()` 只会读取 `obj.n`(`multimodal_tokenizer.py:987-993`)。**复现**:Qwen2.5-VL/Qwen3-Omni/MiMo-V2.5 chat 请求 `n=3` → 下游只看默认 `GenerateOmniReqInput.n=1` → 返回 1 个 choice。
- `extra_key` 只存在于 `GenerateOmniReqInput`(`io_struct.py:153-159`),但 `TokenizedGenerateOmniReqInput` 没字段(`io_struct.py:190-199`),`_create_tokenized_omni_object()` 也不转发(`multimodal_tokenizer.py:987-995`),`Req.to_stage_reqs()` 构造 AR `TokenizedGenerateReqInput` 时仍未填 `extra_key`(`schedule_batch.py:248-254`)。**复现**:两个带不同 `extra_key/cache_salt` 的多模态请求进入同一 AR cache namespace。
- logprobs guard 本身方向正确(不再静默丢),但它在 `_convert_to_internal_request()` 内 `raise ValueError`(`serving_chat.py:91-98`),外层 `OpenAIServingBase.handle_request()` 对 conversion 的所有异常统一返回 `InternalServerError` 500(`serving_base.py:47-52`)。**建议**:在 `_validate_request()` 阶段拒绝,或让 conversion 的可预期 `ValueError` 映射为 4xx。

### R3-4 — audio 内容没有进入 AR radix cache key

`GlobalScheduler.convert_omni_request()` 已按 `mm_items` 计算 `req.cache_input_ids = pad_input_tokens(...)`(`global_scheduler.py:326-336`),其中 audio `MultimodalDataItem.set_pad_value()` 基于 feature hash 生成 pad value(`modality_enum.py:214-228`)。但 stage0 → AR 的转换只把 `input_ids=self.input_ids or self.origin_input_ids` 与 `mm_inputs` 放进 `TokenizedGenerateReqInput`(`schedule_batch.py:248-255`),没有把 `cache_input_ids` 或等价 hash key 传给核心 AR `Req`。结果是两个请求只要文本/placeholder token 相同,即使 audio_codes 不同,AR radix cache 仍可能按相同 `<audio_pad>` prefix 复用 KV。

**建议**:给 AR `TokenizedGenerateReqInput`/核心 `Req` 增加 `cache_input_ids` 透传,并在核心 scheduler 构造 prefix key 时优先使用它。顺手修 `pad_input_tokens()` 里 image/video/audio idx 永不递增的问题(`modality_enum.py:121-147`),否则多段 audio 即使透传也会只用第一个 item 的 pad_value。

### R3-5 — string-format template 丢多模态内容

`process_content_for_template_format()` 在 `content_format == "string"` 时只拼 text chunk,注释也明确忽略 images/audio(`jinja_template_utils.py:95-107`)。`serving_chat._apply_jinja_template()` 对所有消息都按 template 检测结果调用该 helper(`serving_chat.py:187-202`),最后 `GenerateOmniReqInput` 只拿 `processed_messages.audio_data/image_data/video_data`(`serving_chat.py:99-103`)。因此只要某个已支持多模态模型被识别/配置成 string-format template,OpenAI structured `input_audio` / `audio_url` 会在进入 MiMo/Qwen processor 前被丢弃。

**建议**:即使 template content 需要 flatten 成 string,也应先无条件抽取 media 到 side-channel (`audio_data/image_data/video_data`),再决定传给 chat template 的 content 形态;否则这属于「模板风格影响媒体是否存在」的跨模型回归。

### R3-7 — 发送请求与登记 state 的竞态

`_send_one_request()` 当前先 `self.send_to_scheduler.send_pyobj(tokenized_obj)`(`multimodal_tokenizer.py:1008`),再创建并写入 `self.rid_to_state[tokenized_obj.rid]`(`:1013-1022`)。对于很快失败的预处理后请求、短 text-only omni 请求或 cache 命中的请求,detokenizer 回包可能先于 state 登记到达,`_handle_batch_output` 侧只能看到 unknown rid 并丢包,客户端随后等到超时。

**建议**:与 TTS/ASR 路径一致,先创建 state并登记,再 send;若 send 失败再清理 state。

### R3-8 — host processor import 失败静默降级

`resolve_host_processor()` 对候选 processor 的 `importlib.import_module()` 捕获 `Exception` 后直接 `continue`(`host_processor.py:24-28`)。这会把 MiMoV25Processor 的真实 import-time bug(缺依赖、语法/类型错误、间接 import 失败)伪装成「没有匹配 wrapper」,最终返回裸 HF Qwen processor(`:37`)。MiMo audio 请求随后才在 processor 参数或缺少 `audio_codes` 处失败,定位会非常困难。

**建议**:只对 `ModuleNotFoundError` 且 `exc.name` 是候选模块本身时降级;若候选类模块存在但内部 import/初始化失败,对匹配 MiMo config 应 hard-fail 并带 traceback。

### R3-9 — raw audio codec parity 仍是端到端前最大未证项

`MiMoV25AudioCodecProcessor._waveform_to_mel()` 使用 `torchaudio.transforms.MelSpectrogram(... power=1.0, center=True, f_min=0, f_max=None)` 后 `log(clamp)`(`audio_codec_processor.py:709-734`)。HF 一手 `audio_tokenizer/config.json` 只给出 nfft/hop/window/n_mels/fmin/fmax/avg_pooler 等结构参数(`/tmp/mimo_hf/audio_tokenizer/config.json:28-35`),并不证明 torchaudio 默认 mel scale / window / normalization 与官方 processor 完全一致。remote code 中 `tokenize_audio_batch()` 逻辑与本地 `_tokenize_audio_batch()` 基本同构(`/tmp/mimo_hf/modeling_mimo_v2.py:1538-1566` vs `audio_codec_processor.py:809-844`),所以当前最可能「不崩但语义乱」的点已收敛到 waveform→mel parity。

**建议**:端到端前先做离线 golden:固定一段 wav,用 HF/上游真实 processor 导出 mel 与 codes,本地 `_waveform_to_mel()` / `encode()` 分别对齐 shape、长度、前几帧数值与 code 分布。若短期拿不到官方 mel,至少把 raw-audio 入口标记为 experimental,端到端先用预编码 `audio_codes` 验证 stage0+AR。

## 7.2 本轮推翻/更新(不要再按旧结论阻塞)

- **R2-1 AR 类解析**:真实 config `attention_projection_layout="fused_qkv"`(`/tmp/mimo_hf/config.json:14`),HF remote 在 fused 时构造 `self.qkv_proj`(`/tmp/mimo_hf/modeling_mimo_v2.py:277-283`)。因此 yaml 的 `model_class: MiMoV2ForCausalLM` 解析到本地 Pro/fused 类是正确方向(`stage.py:12-13,258-259`;`mimo_v2_5_stage_config.yaml:21-32`),不应再作为 blocker。
- **R2-4 tokenizer symbol 路径**:一手 `modeling_mimo_v2.py` 确实在根模块定义 `MiMoAudioTokenizerConfig` 与 `MiMoAudioTokenizer`(`/tmp/mimo_hf/modeling_mimo_v2.py:1045-1051,1500-1508`),本地 `_load_remote_symbol("modeling_mimo_v2.{symbol}")`(`audio_codec_processor.py:780-790`)对当前 checkpoint 是合理的。
- **R2-5 speech_embeddings 前缀**:HF `MiMoV2ForCausalLM.__init__` 把 `speech_embeddings` 挂在顶层,`audio_encoder` 单独挂载(`/tmp/mimo_hf/modeling_mimo_v2.py:1706-1711`),所以 `weights_mapping.py` 的 `speech_embeddings.{i}.weight` + `audio_encoder.input_local_transformer...` 前缀组合与 remote code 命名一致。保留 `load_weights` 前 hard-fail 护栏即可。
- **R2-6 row0 覆盖**:当前 `_scatter_modality()` 已改成 `fill_value=seq_len` + `.set(..., mode="drop")`(`embedding.py:203-217`),不再把多余 embedding 行写到 token 0。仍建议后续加 `#slots == #rows` 的显式断言,但它已不是 row0 corruption blocker。
- **R2-7 model_type 对齐**:`is_multimodal_model()` 已接受 `mimo_v2 / mimo_v2_5 / mimo_v25 / mimo2_5` 且要求存在 `vision_config/audio_config`(`model_config.py:791-805`),与 `MiMoV25Processor.matches()` 的别名集合一致(`processor.py:27,62-79`)。
- **speech_vocab_size per-channel 疑点**:当前 MiMo-V2.5 主 `audio_config.speech_vocab_size` 是字符串标量 `"1280"`(`/tmp/mimo_hf/config.json:17-35`),HF `_build_speech_embeddings()` 也是把标量扩展到 20 个 embedding(`/tmp/mimo_hf/modeling_mimo_v2.py:521-535`)。`audio_tokenizer/config.json` 的 per-quantizer `codebook_size=[1024,1024,256,128,...]`(`/tmp/mimo_hf/audio_tokenizer/config.json:36-58`)只约束 codec code range,不等价于 speech embedding vocab shape。`processor.matches()` 对 list 型 `speech_vocab_size` 的 `int(list)` 容错仍可作为 future-proof 加固,但不是当前 HF MiMo-V2.5 blocker。
- **AR partial rotary**:本地 `get_rope()` 会在 `partial_rotary_factor < 1.0` 时把 `rotary_dim` 缩小(`embeddings.py:566-598`),`RotaryEmbedding.__call__()` 只旋转 `query/key[..., :rotary_dim]` 并拼回 pass-through 部分(`embeddings.py:232-251`)。因此 `mimo_v2_flash.py:214-223` 传 `partial_rotary_factor=0.334` 会得到与 HF `rope_dim=int(head_dim*factor)`(`/tmp/mimo_hf/modeling_mimo_v2.py:253,305-310`)同向的 partial rotary,不是全头旋转。

## 7.3 端到端测试前建议处理顺序

1. **先修通用入口回归**:R3-1/R3-2/R3-3/R3-6/R3-7。这些不依赖真权重,会影响现有多模态模型,也会让 MiMo 首测结果难解释。
2. **修 cache 内容隔离**:R3-4。audio+text 模型一旦启用 radix cache,不同音频复用同一 placeholder prefix 是 correctness bug,应在性能/长测前处理。
3. **raw-audio 单测先于端到端**:R3-9。建议先单独跑 `MiMoV25AudioCodecProcessor.encode()` 的 golden/parity,再跑完整 OpenAI chat。
4. **启动/配置加固**:R3-8/R3-10 与 R2-12(ep_size pin)。这些不一定阻塞 happy path,但能显著减少首测误判成本。
