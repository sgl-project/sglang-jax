# MiMo-V2.5 Vision 接入(staged 方案,基于 PR #1302)

> workspace: `/Users/lianfang/primatrix/sglang-jax`,branch `feat/mimo-v2.5-vision`(基于 `feat/mimo-v2.5@03db14bf`,含全部 audio 修复)。
> 目标:把 PR #1302 的 `MiMoVisionTransformer` 接进现有 staged embed runtime,实现 image+text→text 端到端。

## 已完成的接入(commits 3f006f12 / 0c2b7b81)

| # | 项 | 实现 |
|---|---|---|
| import | PR #1302 vision 模块 | 取 `mimo_vision/{__init__,vision_encoder}.py` + 对齐测试,run_suite 加 1 行(剔除 PR 夹带的 main 无关改动) |
| 1 | `self.visual` 实例化 | `embedding.py:__init__` 从 `config.vision_config` 建 `MiMoVisionTransformer`(lazy import,避免 audio-only stub 测试拉 jax 重模块) |
| 2 | un-stub encode_image/video | 调 `self.visual(pixel_values, grid_thw)`;grid_thw 归一化为可哈希 tuple;`self.visual is None`(audio-only build)才 NotImplementedError |
| 3 | vision 权重映射 | `load_weights` 在 `self.visual` 存在时 `create_mimo_vision_weight_mappings(source="visual", target="visual.")` 并入 |
| 4 | processor/tokenizer image | **已天然支持**:processor `__call__` 透传 images 给 Qwen2.5-VL 产出 pixel_values/image_grid_thw;tokenizer 已 generic 组 IMAGE mm_item。无需改 |
| 5 | scatter + sharding | 复用既有 generic `_scatter_modality`(已 replicate + drop-mode,R2-6/R2-12 安全),vision embeds 注入 `image_token_id`(151655) |

## 关键决策 / 一手核验

- **vision head_dim = 64**(review D1-1):checkpoint `vision_config` 无 `qk_channels` → `MiMoVisionTransformer` 默认 64;1280/32=40 是错的。`_normalize_vision_config` 显式补 `qk_channels=64`、`in_channels=in_chans`。weight load 时 qkv shape 不符会被 R2-5 风格 hard-fail 抓到。
- **MiMo 用 1-D RoPE,非 mrope**:`rope_scaling.type=default`,无 `mrope_section` → tokenizer mrope 分支正确跳过。vision 只需 pixel_values + scatter。
- **device 预算**:vision 塔仅 0.68B / ~1.26 GiB BF16,+ 现有 embed 塔 ~1.5GB,embed stage 留 **CPU**(host RAM 708GB 充足),AR 仍 16 TPU。**stage yaml 不变**。
- **CPU 兼容**:vision attention 是纯 `jnp.einsum`,无 pallas/TPU-only kernel;embed forward 是 eager(未 jit)→ vision eager 跑 CPU 可行(每图秒级,功能测试可接受)。

## 端到端测试待跑(下一步)

- 部署:复用 `tmp/mimo_v2.5/job_mimo_v25_run.yaml`(RAM 下载 + torch/torchvision/torchaudio + multi-host),分支改为 `feat/mimo-v2.5-vision`。
- 请求:OpenAI `image_url` content part + text(如"描述这张图")。
- 预期风险点(参考 audio 踩坑):
  1. vision embeds 的 reshape/merger 在 embed CPU mesh 上的 sharding(audio 踩过,`_scatter_modality` 已复制,但 ViT 内部 reshape 若在多 host explicit-sharding 下可能需 `_replicate`——CPU 单 device mesh 大概率无事,真机确认);
  2. pixel_values 经 ZMQ 跨进程(mm_items feature)体积;
  3. image_grid_thw 静态性(eager 下无虞)。
- 数据集:image+text 跑通后扩展更多样例。
