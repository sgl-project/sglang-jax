# MiMo-V2.5 omni integration · final design

> This is the **final design** for integrating the MiMo-V2.5 omni model into sglang-jax, consolidated from the step1/step2/implementation-notes/review/vision working docs. It reflects the **current code** and **real-hardware test results**; process material (review rounds, incremental implementation logs, milestone narrative) has been removed.
>
> Status: **text / audio / image / video → text — all modalities implemented and verified end-to-end on real v6e-16 hardware.**
> Code location: `python/sgl_jax/srt/multimodal/models/mimo_v2_5/`.

---

## 1. Overview

MiMo-V2.5 (`XiaomiMiMo/MiMo-V2.5`) is an omni model: **multimodal in (text/image/video/audio), text out**. It only does understanding — no generation/vocoder. Single checkpoint, flat config (`vision_config`/`audio_config` directly at the top level, **no `thinker_config`**).

The integration uses a **two-stage (staged)** runtime: the `embedding` stage encodes each modality into embeddings and injects them into the text sequence; the `auto_regressive` stage generates token-by-token with the MoE LM. The two stages are chained in-process via `queue.Queue` and talk to the tokenizer/detokenizer over ZMQ; the AR stage runs SPMD across 4 hosts.

> Don't confuse: `MiMo-V2.5-Pro` is a **text-only** checkpoint; `MiMo-V2.5` (this doc) is the omni multimodal one. Model routing uses an exact alias, and `is_multimodal` requires a non-empty `vision_config`/`audio_config`.

---

## 2. Verified HF checkpoint facts

### 2.1 LLM backbone (the AR stage reuses `MiMoV2ForCausalLM` = mimo_v2_pro fused-QKV MoE)

| Field | Value |
|---|---|
| model_type / arch | `mimo_v2` / `MiMoV2ForCausalLM` |
| hidden / vocab / layers | 4096 / 152576 / 48 (layer 0 dense, 1–47 MoE) |
| MoE | `n_routed_experts=256`, `num_experts_per_tok=8`, no shared expert, gate sigmoid + noaux_tc, `moe_intermediate_size=2048` |
| Attention | `num_attention_heads=64`; KV: GA=4 / SWA=8; `head_dim=192`, `v_head_dim=128`; 9 Full + 39 SWA (`sliding_window=128`); sink bias on SWA layers only |
| RoPE | **1-D standard RoPE** (`rope_scaling.type=default`, **no mrope**); `rope_theta`=1e7 (GA) / 1e4 (SWA); `partial_rotary_factor=0.334` |
| Quantization | FP8 e4m3, `weight_block_size=[128,128]`; all `o_proj` excluded from quantization; **omni checkpoint does not load MTP** |
| Other | `attention_value_scale=0.707`, `attention_projection_layout=fused_qkv`; total ~293 GiB |

### 2.2 Vision = MiMoVL ViT (`MiMoVisionTransformer`)

| Field | Value |
|---|---|
| depth | 28; full-attn layers `fullatt_block_indexes=[0,9,18,27]` (4 full + 24 windowed) |
| Window attention | `vit_window_attn_types` ∈ {-1,0,1} = full / row 1-D window / col 1-D window; `window_size=128`, `visual_token_window_size=64`; `use_sink=true` (attention sink) |
| Dimensions | hidden 1280, `num_heads=32`, KV `num_key_value_heads=8` (GQA); **head_dim=64** (from the `qk_channels` default, **not** 1280/32=40) |
| Patch | `patch_size=16`, `spatial_merge_size=2`, `temporal_patch_size=2`, `in_chans=3` |
| Output | `out_hidden_size=4096` (built-in merger projector, **bias-free**) |

> The checkpoint uses the field name `in_chans` (not `in_channels`) and has **no `qk_channels`** (real value is the default 64); the merger's three layers (`ln_q`/`mlp.0`/`mlp.2`) have **weight only, no bias**; the patch_embed Conv is `use_bias=False` and norm1/2 are RMSNorm. These are normalized/aligned at integration time (see §6).

### 2.3 Audio understanding = host RVQ tokenizer + JAX audio tower

Discrete RVQ path (not continuous mel→projector). Split in two parts:

- **Host side**: raw audio → 24kHz mono → log-mel → frozen RVQ `audio_tokenizer.encode` → `audio_codes [T',20]` (20-channel RVQ). `audio_tokenizer/` is a checkpoint subdirectory (d_model 1024, 24 layers, 16 heads, `num_quantizers=20`, `codebook_size=[1024,1024,256,128,...]`); **hard-fail if missing, no fallback**. Mel front-end: `sr=24000`, `n_fft=960`, `hop=240`, `win=960`, `n_mels=128`.
- **JAX side (`MiMoV25AudioUnderstandingEncoder`)**: `audio_codes` → 20-channel `speech_embeddings` (`speech_vocab_size=1280`, group_size=4) → 6-layer full-attn input_local_transformer (dim 1024 / heads 16 / head_dim 64 / intermediate 4096, non-causal) → 2-layer projection (`4096→16384→4096`, bias-free, **GELU**) → `[N,4096]`.

### 2.4 Per-modality token ids (verified from top-level config)

`image=151655`, `video=151656`, `vision_start=151652`, `audio=151669` (in the `processor_config` sub-block, needs a fallback lookup). All three towers output `[N,4096]` (== LM hidden), scattered by token id, no adapter needed. Positions are **1-D RoPE** throughout.

---

## 3. Two-stage architecture

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

Key design points:
- **Device split**: the embed stage runs on **CPU** (the towers are small: audio + a 0.68B ViT, ~3 GB, on host RAM), leaving all 16 TPU chips for the 293 GiB AR backbone. AR mesh `[data=1, tensor=16]`, `ep_size=16` divides 256 experts.
- **Uniform modality interface**: `MiMoV2_5Embedding.__call__` runs the same `_encode_<modality>` → `_scatter_modality` for audio/image/video; tower names mirror the HF checkpoint (`self.audio_encoder` / `self.visual`), and the encoders live beside `embedding` in the same directory (matching the `qwen3_omni_moe` layout).
- **AR reuse**: directly reuse `MiMoV2ForCausalLM` (the existing text-only implementation), adding only the input_embedding hook; no `*Generation` wrapper, no mrope.
- **mm_items as single source of truth**: the tokenizer builds image/video/audio into a uniform set of `mm_items` (`MultimodalDataItem`), and `assemble_mm_inputs` splits them into per-modality kwargs with no model-specific logic. MiMo uses 1-D RoPE (no mrope_section), so the tokenizer's mrope branch is correctly skipped.

### Stage config (`models/static_configs/mimo_v2_5_stage_config.yaml`)

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

## 4. Per-modality path

- **Image**: Qwen2.5-VL-style preprocessing → `pixel_values` + `image_grid_thw` → `self.visual` (MiMoVL ViT) → scatter at `image_token_id`.
- **Video**: same ViT, `pixel_values_videos` + `video_grid_thw` (`temporal_patch_size=2`) → scatter at `video_token_id`, 1-D timestamp positions. Preprocessing needs `decord`.
- **Audio**: raw audio → host RVQ codec → `audio_codes [T',20]` → `self.audio_encoder` (speech_embeddings → group_size=4 → local transformer → projection) → `[N,4096]` → scatter at `audio_token_id`. `#audio_pad == ceil(T'/4)`, with a hard check `sum(token_lengths) == audio_embeds.shape[0]`; multi-segment audio is padded to a multiple of 4 per segment before concatenation (no cross-segment grouping); out-of-range code ids are validated host-side. Audio is optional — any-subset requests skip it.
- **Text**: `text_embed_tokens(input_ids)`.

---

## 5. multi-host SPMD

**Motivation**: the 293 GiB FP8 MoE doesn't fit on a single v6e chip (32 GiB), so it needs 16 chips = v6e-16 = 4 hosts × 4 chips. The original multimodal runtime was single-process and never called `jax.distributed.initialize`, so `jax.devices()` only saw the local 4 chips.

**Choice**: SPMD multi-controller (4 pods each run one process, forming a 16-chip mesh, all ranks running the same jitted forward); Pathways was rejected (no ready support in the target GKE env).

**Core conflict**: the AR stage uses an in-process `QueueBackend`, bypassing the standard pub/sub. Adding distributed init without broadcasting the batch → rank0's AR collective waits on the other ranks while their in-process queue stays empty → deadlock. So every AR batch must be lockstep-broadcast from rank0 to all 4 ranks.

**5 changes**:
1. `multimodal/entrypoint/http_server.py:launch` — pass through `nnodes/node_rank/dist_init_addr`; `node_rank>=1` starts only the GlobalScheduler process.
2. `multimodal/manager/global_scheduler.py` — `jax.distributed.initialize()` (nnodes>1) before building `DeviceManager()`.
3. `managers/communication.py` — new `MultiHostQueueBackend`: rank0 PUB-broadcasts (incl. empty batches), non-rank0 SUB-receives, `send_pyobj` only on rank0.
4. `multimodal/manager/stage.py` — the AR stage uses `MultiHostQueueBackend` when nnodes>1, otherwise `QueueBackend`.
5. embed (CPU) is only effective on rank0; **stage startup is serialized** (nnodes>1, built in stage_id order) so that parallel threads' `broadcast_one_to_all` can't be ordered nondeterministically across hosts → collective deadlock.

---

## 6. Key implementation details and constraints (surfaced and fixed on real hardware)

- **fused MoE alignment (hard constraint)**: `num_tokens` must be a multiple of `ep_size * t_packing` (=32, `t_packing=32//dtype_bits=2` for bf16); all precompile bs/token buckets are aligned to 32. `ep_size` must divide both 256 experts and the total device count.
- **Replication under explicit sharding**: the embed stage uses an explicit-sharding mesh; the following tensors must be replicated (`with_sharding_constraint(P())`) before reshape/scatter (the towers are small, embed seq is short), mesh-guarded via `getattr(self,"mesh")`:
  - before every audio-tower hidden reshape;
  - `_scatter_modality`'s `input_embeds` + `modality_embeds` (`.at[pos].set(..., mode="drop")`, `fill_value=seq_len` to avoid an out-of-bounds write to token 0);
  - **all vision weights loaded replicated (`sharding=()`)**: the MiMoVL ViT uses bare `nnx.Linear` (no sharding annotation), so the loader's name-based default sharding would tensor-shard the gate/up/down/qkv kernels, making the `dot_general` against replicated activations unresolvable on the embed mesh → crash; forcing replication fixes it.
- **vision config normalization** (`_normalize_vision_config`): the checkpoint has no `qk_channels` → fill the default `64`; `in_chans` → `in_channels`. A wrong value is caught at weight load by a qkv shape mismatch (fail-safe).
- **vision merger is bias-free**: `MiMoVisionPatchMerger`'s `ln_q`/`mlp_fc1`/`mlp_fc2` are built with `use_bias=False` (the checkpoint has no merger bias), otherwise the bias params stay as abstract `eval_shape` placeholders → crash on use.
- **pin vision inputs to the CPU mesh (V4 fix)**: `pixel_values` built by `jnp.asarray` in `embed_model_runner._prepare_input` lands on the default TPU backend; the embed forward is eager with no `with mesh` context, so the patch_embed reshape ran on TPU and allocated TPU HBM (competing with the AR stage's chips), OOMing on long video. Fix: `device_put` `pixel_values/pixel_values_videos/image_grid_thw/video_grid_thw` onto the embed CPU mesh, so the whole ViT runs on CPU/host RAM.
- **model-agnostic embed contract**: the model returns `EmbedOutput(input_embeds, deepstack=None, pos_mask=None)` and the runner reads fields by name; `get_embed_model_config` strips the fp8 `quantization_config` (embed loads bf16, the AR stage builds its own ModelConfig and resolves fp8); `get_total_num_kv_heads` is hasattr-guarded for the raw HF config. Host-side input validation is exposed via an optional model hook (`validate_embed_inputs`), keeping the runner free of model-specific logic.
- **AR input_embedding hook**: injects the stage0 merged result only in extend/draft_extend/mixed forward modes; decode uses token embedding; deepstack is a no-op; positions come from `forward_batch.positions`.

---

## 7. File list (current implementation)

**New (`multimodal/models/mimo_v2_5/`)**: `embedding.py` (MiMoV2_5Embedding), `vision_encoder.py` (MiMoVisionTransformer, based on upstream PR #1302), `audio_encoder.py`, `audio_codec_processor.py`, `processor.py` (MiMoV25Processor, wraps Qwen2.5-VL + host RVQ codec), `weights_mapping.py`, `__init__.py`, `static_configs/mimo_v2_5_stage_config.yaml`.

**AR reuse (not new)**: `models/mimo_v2_pro.py::MiMoV2ForCausalLM` + `mimo_v2_flash.py::MiMoV2Model`.

**Modified entry/executor/runtime**: `configs/model_config.py` (`is_multimodal_model()`), `entrypoints/openai/serving_chat.py` (`GenerateOmniReqInput` routing), `multimodal/entrypoint/http_server.py`, `managers/communication.py` (`MultiHostQueueBackend`), `multimodal/manager/{global_scheduler,stage}.py`, `model_executor/embed/embed_model_runner.py`, `io_struct.py`/`schedule_batch.py`/`forward_batch_info.py` (`Req.multimodal_embedding` → `ForwardBatch.input_embedding`), `multimodal/manager/multimodal_tokenizer.py` + `mm_assembly.py` (generic mm_items assembly).

---

## 8. Deployment (GKE)

- Cluster `lianfang-v6e-mimo` (us-east5-a), pool 4×4=16 chips. 4-pod Indexed Job + headless Service for rank discovery; env `TPU_WORKER_HOSTNAMES` (4 pod DNS) + `TPU_WORKER_ID` → node_rank.
- Image `jax-ai-image/tpu:jax0.8.1-rev1` (jax is installed in `/opt/venv`; `bash -lc` drops the venv PATH, so use an explicit `/opt/venv/bin/python`).
- gcsfuse CSI addon disabled + node disk <293 GiB → use a **RAM-emptyDir (node ~708 GiB) + in-pod `gcloud storage rsync`** to pull the 294 GB weights, bypassing gcsfuse.
- Extra deps (missing from the image, installed at deploy time): `torch torchvision torchaudio` (Qwen2.5-VL AutoProcessor + host codec), `decord` + `imageio-ffmpeg` (video preprocessing).
- Launch: `/opt/venv/bin/python -m sgl_jax.launch_server --multimodal --model-path /weights/MiMo-V2.5 --trust-remote-code --nnodes 4 --node-rank $TPU_WORKER_ID --dist-init-addr <pod0>:9876 --tp-size 16 --ep-size 16 --moe-backend fused --page-size 64`.

---

## 9. Real-hardware test results (v6e-16)

All run on 4-pod multi-host (tp=16 ep=16 fused MoE FP8), and verify **content understanding**, not just plumbing:

| Input | Result | Notes |
|---|---|---|
| text | ✅ | self-identifies as Xiaomi MiMo, 43 tok/s |
| text + real speech | ✅ | MLK "I Have a Dream" 13s — **correctly identifies the speech and transcribes matching text** |
| text + image | ✅ | both cats + remotes correctly identified |
| text + video | ✅ | mission-control center / map screens / person correct (8 frames) |
| text + image + video | ✅ | both modalities described separately, no cross-talk |
| text + image + video + audio | ✅ | all four correct, no cross-modal confusion |

> The real-speech run end-to-end proves the host RVQ codec (incl. the mel front-end) produces output good enough for accurate content understanding — the biggest previously-unverified item flagged in review ("the raw-audio→codes chain was never run against a real checkpoint", R2-3/9/11), now functionally confirmed. AR `num_tpus=16` is also confirmed serviceable on real hardware (resolving the earlier R2-2 sizing question).

---

## 10. Unresolved issues / known limitations

### 10.1 OpenAI multimodal entry regressions (no real weights needed; affect all multimodal chat models)
- **R3-1** `serving_chat` builds `GenerateOmniReqInput` without forwarding `n` → `n>1` silently degrades to 1.
- **R3-2** `rid` as `list[str]` is passed through; downstream uses the list as a dict key → `TypeError`.
- **R3-6** multimodal `logprobs/top_logprobs` `ValueError` is wrapped as HTTP 500; should be 4xx.
- **R3-7** `_send_one_request` sends before registering `rid_to_state`; a fast-completing request can have its output dropped as an unknown rid (race).

### 10.2 Cache / radix content isolation (a correctness bug once radix cache is enabled)
- **R3-3** `extra_key` reaches the omni request but `to_stage_reqs()` doesn't forward it → cache-namespace isolation is lost.
- **R3-4** audio content does not enter the AR radix cache key: `cache_input_ids` (with the audio hash) is not passed from stage0 to the core `Req` → different audio with the same `<audio_pad>` text can reuse the wrong KV prefix. Also: `pad_input_tokens` doesn't increment the image/video/audio idx, so multi-segment only uses the first item's pad_value.

### 10.3 Multimodal content silently dropped on template/degrade paths
- **R3-5** string-format chat templates drop audio/image/video parts in `process_content_for_template_format` → a model recognized as a string template silently becomes text-only.
- **R3-8** `resolve_host_processor` swallows import-time exceptions and falls back to the bare HF processor → a real import bug / missing dep in MiMoV25Processor masquerades as "no matching wrapper", making it hard to diagnose.

### 10.4 Numerical fidelity (functionally proven, golden test not done)
- **R3-9 / R2-11** raw audio mel→codes has no offline numerical golden vs the HF/reference codec; structural params all match the config and real speech transcribes correctly end-to-end, but mel scale/window/normalization details have no golden safety net; the `code_lengths` formula and the actually-returned codes are two paths and would mis-split if off by one frame. Recommend an offline golden.
- The vision path likewise has no HF numerical-alignment regression (the `test_mimo_vision_encoder.py` harness exists, needs a vision-checkpoint environment to run).

### 10.5 Hardening (misconfig → silent error)
- **R2-8** `get_codebook_sizes()` falls back to a uniform `1280` on parse failure, admitting out-of-range high ids on low-codebook channels; should validate per-channel against the real per-quantizer `[1024,1024,256,128,...]`.
- **R2-12** the AR stage yaml doesn't pin `ep_size` (it comes from the global `--ep-size`); it must divide both 256 and the device count, and fused only warns on mismatch. Recommend pinning it in the stage.
- **R3-10** model routing `_match_mimo_v25_omni` excludes any `pro`/`flash` substring in the path without word boundaries, so `/prod/MiMo-V2.5` etc. would be misrouted.

### 10.6 Performance
- **embed not JIT'd / bucketed**: the embed forward is currently **eager** (`forward_wrapper` is not wrapped in `jax.jit`). The originally-planned JIT + shape bucketing is not implemented; CPU eager is acceptable for small images / short audio, but very slow for long video (178 frames).

### 10.7 Out of scope this round
- **DP > 1**: each stage's mesh has the `data` axis fixed at 1, conflicting with the core `dp_size`; would need `ici_parallelism=[dp_size, chips//dp_size]` + a startup assertion. This round is **DP=1**.
- **batch > 1**: stage0/stage1 `max_batch_size=1`; whole-sequence input_embedding replacement can't mix batches; would need token-level scatter-merge. This round is **batch=1**.
- **interleaved video audio tracks**: `video_audio_interleave_length`; this round only supports basic (silent / ignored-audio-track) video.
- **audio/speech generation (ASR/TTS)**: the separate-checkpoint RVQ generation / vocoder is out of the omni-understanding scope.
