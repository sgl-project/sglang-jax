# sglang-jax Multimodal Subsystem: Current State, Upstream Reference, and Long-term VLM Architecture

> **Nature of this document**: a complete rewrite of the sglang-jax multimodal subsystem, in four parts:
> 1. **Pre-refactor current state (`main` baseline) + the feature gap between LLM and multimodal**;
> 2. **Upstream sglang (PyTorch) VLM design reference, and how it supports the features missing in Part 1** (future reference);
> 3. **sglang-jax long-term multimodal architecture** (understanding plane / VLM detailed, generation/diffusion plane only bounded), including the plan to fill the missing features on VLMs;
> 4. **VLM refactor milestones and implementation status**.
>
> **Baseline & branch conventions**:
> - Part-1 baseline = `main` (pre-refactor, everything multimodal is staged).
> - Refactor = **PR #1350 `refactor/mm-understanding-in-model`** (base `main`, OPEN): migrate the VLM understanding chain from the staged `GlobalScheduler` pipeline onto the standard LLM control plane (in-model); diffusion and TTS stay staged. **After a rollback, the current code lives on `epic/support_mimo_v2.5_vlm` (tip `c107a606`, based on PR #1350 + rollback)**: only MiMo-V2.5 is in-model; Qwen2.5-VL/Qwen3-Omni are reverted to staged. Part 4 is pinned to `c107a606` (diff vs `main` `0ed80f61` = 235 files / +21351 / -5704).
>
> **Evidence convention**: code claims carry `file:line` (with the commit); we distinguish Evidence (directly proven in code) / Inference (structural deduction) / Unknown (undeterminable from code). This document avoids performance/accuracy numbers unless explicitly marked as a relayed pod measurement not re-verified on-site. Upstream claims are based on reading `sgl-project/sglang` main.

---

# Part 1 · Pre-refactor Current State (`main` baseline) and the LLM-vs-Multimodal Feature Gap

> All `file:line` in this part are relative to `python/sgl_jax/srt/` on `main`.

## 1.1 One-line overview and the two runtimes

On `main`, the multimodal subsystem is not a model but **a separate runtime that splits a model into several stages and strings them into a linear pipeline**. `GlobalScheduler` is the pipeline orchestrator; each stage is a scheduler running on its own thread and occupying its own device mesh; stages pass a unified `Req` object via an in-process `queue.Queue`; which stages, which model, and how many chips each takes are decided by a "one YAML per model" config.

The repo actually contains two runtimes, and the core design is to **let the staged runtime "nest" the standard text runtime**:

| | Standard text runtime | Multimodal staged runtime |
|---|---|---|
| Orchestrator | `managers/scheduler.py: Scheduler` | `multimodal/manager/global_scheduler.py: GlobalScheduler` |
| Scheduling unit | continuous-batching `Req` | linear-pipeline stage (one scheduler per stage) |
| Scheduled `Req` | `managers/schedule_batch.py: Req` (lightweight, LLM-oriented) | `multimodal/manager/schedule_batch.py: Req` (giant union dataclass) |
| Entry manager | `TokenizerManager` | `MultimodalTokenizer` (subclass) |
| Cross-component comm | ZMQ | ZMQ (external) + `queue.Queue` (inter-stage) |

**The interleaving point (the essence of the text-out design)**: `scheduler: auto_regressive` in a stage YAML is resolved by `stage.py:11,219-221` into the **standard text `Scheduler` itself** (`from sgl_jax.srt.managers.scheduler import Scheduler as AutoRegressiveScheduler`). That is, **the "generation" of an only-text-out multimodal model runs entirely on the standard LLM control plane** (radix cache / paged attention / sampling / KV pool all reused); the multimodal-specific logic is compressed into the stages before auto-regression — "encode the modality features + splice them into the text embedding sequence".

The standard LLM control plane **already has a built-in entry point to consume multimodal embeddings**: `handle_generate_request` at `managers/scheduler.py:1023-1035` reads `recv_req.mm_inputs["multimodal_embedding"]` / `deepstack_*`, with the fields already defined in `managers/schedule_batch.py`. This is exactly why PR #1350 can make the understanding chain in-model — the downstream LLM has long been able to consume a merged embedding, so the refactor only needs to move the "merge" from an external stage into the LLM control plane / model interior.

## 1.2 Process / thread topology

```
              Main process
 HTTP client ─▶ FastAPI (uvicorn)
              │   /v1/chat/completions            (understanding, GenerateOmniReqInput)
              │   /api/v1/images|videos/generation (generation, GenerateMMReqInput)
              │   /v1/audio/speech | transcriptions
              │      │
              │      ▼  MultimodalTokenizer (TokenizerManager subclass, in-process)
              └──────┬──────────────────────────▲────────────
                ZMQ PUSH (tokenized req)     ZMQ PULL (results)
                     ▼                          │
        subprocess: global_scheduler
          GlobalScheduler.event_loop()
            ├ Stage 0 (thread + mesh_0) ─ queue.Queue ─┐
            ├ Stage 1 (thread + mesh_1) ◀──────────────┘
            └ Stage N (thread + mesh_N) …
                     │ ZMQ PUSH (final BatchTokenIDOut)
                     ▼
        subprocess: multimodal_detokenizer (DetokenizerManager subclass)
```

Key facts (`multimodal/entrypoint/http_server.py`, `global_scheduler.py`, `stage.py`):

- **Process boundaries**: `MultimodalTokenizer` + HTTP server in the main process; `GlobalScheduler + all Stages` inside one subprocess (stages are threads, not processes — `global_scheduler.py:396-398`, one `threading.Thread` per stage); `MultimodalDetokenizer` is a separate subprocess.
- **Wiring**: main ↔ subprocess over ZMQ (`global_scheduler.py:91-98`); **stage ↔ stage over in-process `queue.Queue`** (`global_scheduler.py:135-139`).
- **Communication abstraction**: the stage-internal scheduler talks via `QueueBackend` (`managers/communication.py`), the queue implementation of the standard `Scheduler`'s `CommunicationBackend` abstraction — the same `Scheduler` class switches between ZMQ mode (text) and Queue mode (stage) by injecting a different backend.
- **Multi-host SPMD**: `main`'s `launch()` has no explicit multi-host branch code; multi-host behavior on `main` is **Unknown** (not provable from code).

## 1.3 Core data structures and the 4-object stage stack

**`Modality` / `MultimodalDataItem` / `MultimodalInputs`** (`multimodal/common/modality_enum.py`):

- `Modality` (L151): `IMAGE / MULTI_IMAGES / VIDEO / AUDIO` (one more than upstream: `MULTI_IMAGES`).
- `MultimodalDataItem` (L171): all inputs of one modality packed into one item. Common fields first (`modality / hash / pad_value / offsets`); `feature` (raw processor output) XOR `precomputed_embeddings`; model-specific fields go into the `model_specific_data` dict, transparently accessed via `__getattr__`/`__setitem__`. `set_pad_value()` (L214) sha256s the feature, `pad_value = hash % 2^24` (**no shift, no vocab sanity** — potential token-id collision risk).
- `MultimodalInputs` (L281): a container of `mm_items` + various token ids + `mrope_*`.

**union `Req`** (`multimodal/manager/schedule_batch.py:33`): the giant dataclass passed stage→stage in the pipeline, merging fields of three task types — diffusion (latents/timesteps/prompt_embeds…), Omni understanding (omni_inputs/vision_embeds/input_embeds/cache_input_ids…), audio (audio_codes/mel_input/generated_*…). Two adapter methods: `to_stage_reqs(scheduler_name)` (L219; for `auto_regressive`, builds a standard `TokenizedGenerateReqInput` and attaches `.mm_inputs`), `from_stage()` (L301; reconstructs a Req from a stage's output).

**The 4-object stage stack** (one per stage):

```
Stage (manager/stage.py)        holds device mesh + in/out queue; spawns a thread running the scheduler
  └ XxxScheduler (manager/scheduler/)  thin event loop: recv → worker.forward → send
       └ XxxModelWorker (model_executor/)  ultra-thin shell, delegates to runner
            └ XxxModelRunner (model_executor/)  load_model + jit + forward
```

A Stage carves out a **disjoint** mesh via `DeviceManager.allocate()` based on `runtime.device_kind` (cpu/tpu) + `num_tpus` (`stage.py:88-109`); this is the extra capability staged has over the standard path (per-stage CPU/TPU placement), at the cost of inter-stage embeddings having to host-roundtrip through `queue.Queue`.

## 1.4 Config-driven: stage YAML + the actual stage chains on `main`

`StageConfigRegistry` (`static_configs/yaml_registry.py`) resolves by `model_path` (exact name → full path → keyword fallback). `scheduler` name→class at `stage.py:216` (if/elif chain); `model_class` name→class at `stage.py:239` (hard-coded dict).

**On `main`, `static_configs/` has only 6 YAMLs (no mimo_v2_5)**:

| Model | stage chain (scheduler / model_class / device) | task |
|---|---|---|
| **Qwen2.5-VL** (7B) | `vit`(Qwen2_5_VL_VisionModel,1TPU) → `auto_regressive`(Qwen2_5_VL_Generation,1TPU) | image/video → text |
| **Qwen2.5-VL** (32B/72B) | same, AR tp4 (separate `_tp4` YAML, ViT on CPU) | image/video → text |
| **Qwen3-Omni** (30B-A3B) | `embedding`(Qwen3OmniMoeThinkerEmbedding,**CPU**) → `auto_regressive`(…ThinkerText…,4TPU) | image/video/audio → text |
| **MiMo-Audio** (7B) | `audio_encoder`(MiMoAudioTokenizer) → `audio_backbone`(MiMoAudioForCausalLM) [→ `audio_decoder` commented out] | speech ASR/TTS |
| **FLUX.1-dev** | `text_encoder`(CLIP+T5) → `diffusion`(FluxTransformer2DModel) → `vae`(AutoencoderKL) | text → image |
| **Wan2.1** | `auto_regressive`(UMT5,capture_hidden) → `diffusion`(WanTransformer3DModel) → `vae`(AutoencoderKLWan) | text → video |
| **Wan2.2** | `auto_regressive`(UMT5,**CPU**) → `diffusion`(WanDualTransformer3DModel) → `vae` | text → video |

Two notable reuses: (1) Wan's UMT5 text encoder runs on the `auto_regressive` scheduler, using `capture_hidden_mode=2` to make the standard LLM scheduler emit `last_hidden_state` as the diffusion condition; (2) the AR stage of Qwen2.5-VL / Qwen3-Omni uses `precompile_params.input_embedding=True` so the AR model reads `forward_batch.input_embedding` (the already-merged multimodal embedding) instead of its own embed_tokens.

## 1.5 Two understanding-chain forms + two merge implementations (refactor focus)

On `main`, text-out understanding has two merge forms, both ending in the `auto_regressive` standard Scheduler; they differ in how early stages splice modality embeddings into the text sequence:

**Form A: vit-stage (Qwen2.5-VL)** — merge in the **runner**. `vit_model_runner.py:106-139` `_merge_multimodal_embeddings`: after ViT produces `vision_embeds`, it uses `cumsum(is_multimodal)` + gather + `jnp.where` to splice the vision embedding into `text_embed(input_ids)` (a leading dummy zero row aligns the cumsum index). Writes back to `mm_inputs["multimodal_embedding"]`. No deepstack.

**Form B: embedding-stage (Qwen3-Omni)** — merge inside the **model `__call__`**. `qwen3_omni_thinker_embedding.py:332-400`: in-model `input_embeds.at[mask].set(...)` scatter (once each for audio/image/video); the runner just passes through (`embed_model_runner.py:181-191`). Additionally produces deepstack multi-scale features + a visual mask.

| | Form A vit-stage | Form B embedding-stage |
|---|---|---|
| Where merge happens | runner | model `__call__` |
| Merge operator | `cumsum`+gather+`where` | `.at[mask].set()` scatter |
| deepstack | none | yes |

> **`main` has no shared "merge pure function"** — `mm_assembly.py` / `assemble_mm_inputs` get zero hits on `main` (introduced only by PR #1350). The coexistence of two stylistically inconsistent merge implementations is exactly one of the things the refactor unifies into `mm_core`.

## 1.6 Generation (FLUX/Wan) and audio (MiMo-Audio) — brief

- **Diffusion**: FLUX = `text_encoder`(CLIP+T5) → `diffusion`(denoise loop, abortable between steps) → `vae`; Wan = UMT5 (borrowing the AR scheduler) → DiT (Wan2.2 is a high/low-noise dual-expert MoE) → causal 3D VAE.
- **Audio (MiMo-Audio)**: `audio_encoder`(mel→RVQ codes) → `audio_backbone`. The `audio_backbone` input is aggregated by `_build_backbone_input()` into a `[1, 9, seq]` tensor of "1 text channel + 8 audio channels"; `AudioBackboneScheduler` has **its own auto-regressive loop** (does not reuse the standard Scheduler) — the structural reason it cannot be folded into the understanding plane. `audio_decoder` (codes→waveform) is commented out in the YAML, so TTS currently cannot produce a waveform end-to-end; the usable path is ASR.

## 1.7 srt ↔ multimodal reverse coupling (decoupling prerequisite)

On `main`, `srt/` (non-multimodal) reverse-imports `srt/multimodal/` in **8 places**, which the refactor must untangle first:

| Caller (srt non-mm) | imported content |
|---|---|
| `entrypoints/engine.py` | `MultimodalServerArgs` (lazy import) |
| `entrypoints/openai/serving_chat.py` | `GenerateOmniReqInput` |
| `managers/{tokenizer,detokenizer}_manager.py`, `scheduler.py` | `resolve_tokenizer_subdir` (×3) |
| `managers/io_struct.py` | `flatten_nested_list` |
| `server_args.py` | `MultimodalServerArgs` (×2, lazy) |

The dirtiest are `managers/io_struct.py` and the three `resolve_tokenizer_subdir`s — the core srt control plane imports multimodal utility functions at module top level, making srt unable to load independently of multimodal. The reverse direction (multimodal → srt, e.g. a stage importing `Scheduler`/`QueueBackend`) is the expected nested reuse, not a decoupling item.

## 1.8 [Core] Standard LLM runtime vs. multimodal staged path: the feature gap

First, the boundary: in the staged path **only the `auto_regressive` stage reuses the standard `Scheduler`**; encode (ViT/embed), merge, diffusion, VAE, audio, etc. are all **self-written thin shell schedulers** in `multimodal/manager/scheduler/*.py` and inherit none of the standard Scheduler's scheduling abilities. So the gap analysis must distinguish two sub-segments: the **AR segment** (inherits part of the standard Scheduler) vs. the **encode/merge/external-stage segment** (fully external, zero inheritance).

### Comparison table

| Feature | Standard LLM runtime | Staged multimodal (main) | Structural cause of the gap |
|---|---|---|---|
| **Continuous batching / bs>1** | supported (`waiting_queue`+`running_batch`+`get_next_batch_to_run`, `scheduler.py:384,1475`) | AR segment supports it structurally; **encode segment is forced per-request serial** (`vit_scheduler.py` `for req in reqs: run_vit_step(req)`, runner one Req at a time) | encode is an external shell with no batching logic; one ViT per image. YAML `max_batch_size:1` has no code consumer (not a hard constraint) |
| **chunked prefill** | supported (default 4096, `schedule_policy.py` PrefillAdder truncates) | AR segment nominally inherits it, but **is unsafe for pre-merged `input_embedding`, no guard** (`schedule_policy` is mm-unaware, truncation only touches `fill_ids`) | truncation does not coordinate the batch-level embedding buffer; upstream disables chunked specifically for Transformers-backend mm, jax has no such guard |
| **radix / prefix cache** | on by default (`RadixCache`) | **globally disabled**: `--multimodal` → `disable_radix_cache=True` (`server_args.py:251-254`) → `ChunkCache`, `match_prefix` always empty. **Zero prefix reuse** | explicit blanket disable (comment says UMT5 has no KV cache); the pad_value→radix wiring is dead code on main (see below) |
| **overlap schedule** | on by default | off: `--multimodal` forces `enable_overlap=False` (`scheduler.py:200-202`) + a redundant assert in the AR stage (`stage.py:191-194`) | two gates |
| **data parallel DP** | supported (`dp_size`/`select_dp_for_request`/per-DP running_batch) | AR segment inherits it; encode/external stages have no DP (single-mesh shell) | the encode shell has no DP concept |
| **precompile / shape bucketing** (CUDA-graph equivalent) | full bucketing on the token dim (`PRECOMPILE_DEFAULT_TOKEN_PADDINGS=[64..8192]` + `CompilationManager`) | AR token dim is already bucketed; **ViT/vision has zero bucketing** | `vit_model_runner.py:64` bare jit, `pixel_values` patch dim is traced not static; `_make_dummy_batch` builds no pixel → recompiles per new num_patches |
| **multi-host SPMD** | has a multi-host branch (`nnodes>1` + `jax.distributed.initialize`) | AR segment can inherit it (Inference); GlobalScheduler is single-process + in-process queue, no evidence of cross-host stage orchestration | GlobalScheduler uses in-process queue + local device slicing |
| **CPU/TPU per-stage placement** | single mesh | **supported** (encode often on CPU, AR on TPU) | staged's only "reverse advantage" (`stage.py:88-109` builds mesh by device_kind) |
| **speculative decoding** | supported (EAGLE/NEXTN) | not wired (AR stage has no spec config) | not configured on the staged path |
| **structured output / grammar** | supported | off (`--multimodal` skips grammar init, `scheduler.py:267`) | explicitly skipped |

### Synthesis: why the staged architecture structurally gives up these features

**A. radix is killed globally with one switch (not lost, explicitly disabled).** `server_args.py:251-254` unconditionally sets `disable_radix_cache=True` under `--multimodal`, degrading the whole path to `ChunkCache` (`match_prefix` always empty). Even though `pad_input_tokens()` writes per-image hashes into `cache_input_ids` (`global_scheduler.py:369`) as a half-built thing, it is **neither read by anyone nor enters `RadixKey`** (RadixKey uses `origin_input_ids`/`fill_ids`, never reads `cache_input_ids`) — dead wiring. So multimodal prefix reuse on main is 0.

**B. encode/merge are external shells, naturally outside the standard scheduler's domain.** ViT/embed stages only do `for req in reqs: forward(req)` — no waiting_queue, no continuous batching, no bucketing, no DP, no overlap. So bs>1 collapses to per-request serial in the encode segment (one ViT per image; N small matmuls on TPU ≪ one big matmul); vision shape has no bucketing → recompile per shape (Qwen2.5 ViT is still dense O(T²) attention + distance mask, no flash).

**C. The AR segment inherits part of the standard ability, but the multimodal switch turns off two of them.** The AR stage reuses the standard `Scheduler`, structurally getting continuous batching, token-dim bucketing, the chunked mechanism, DP; but `--multimodal` actively turns off overlap + grammar, and the inherited chunked prefill lacks correctness guarantees for pre-merged embeddings.

**D. Physical mesh isolation across stages is the topological root cause.** `DeviceManager.allocate` gives each stage a disjoint device slice, so stages can only host-roundtrip via `queue.Queue`. This buys per-stage CPU/TPU placement flexibility (staged's only reverse advantage), at the cost that encode and AR are never in the same forward / same mesh, and cannot be brought into the standard Scheduler's continuous-batching / bucketing / radix system.

**One line**: on main only the AR segment stands on the shoulders of the standard LLM runtime (and even there overlap/grammar are off, chunked has no mm guard); the entire encode/merge segment is outside the standard scheduler (continuous batching collapses to serial, vision has no bucketing, no DP/overlap); radix is globally killed and the hash→radix wiring is dead code.

---

# Part 2 · Upstream sglang (PyTorch) VLM Design Reference + Support for the Gap Features

> This part provides an upstream design reference for the refactor and maps each missing feature from Part 1 to its upstream implementation + JAX portability. Upstream paths are relative to `python/sglang/srt/`.

## 2.1 One-line positioning and scope split

Upstream implements the VLM **understanding** chain as "a prefill adapter of an ordinary LLM": media encoding (ViT/audio encoder) is **inlined in the model's own `forward()`**, sharing the same process, mesh, and scheduler as the LLM body, with no stage/process/mesh boundary at all. The merge core is a single function `general_mm_embed_routine`, called once from each VLM's own `forward()`. This is exactly the target form PR #1350 wants the sglang-jax understanding plane to align with.

Upstream splits multimodal into **runtimes with non-overlapping scopes**:

| Scope | Upstream location | Produces media? |
|---|---|---|
| **Understanding** (media→text) | `srt/multimodal/`(processor) + `managers/mm_utils.py`(merge), inside the LLM forward | No |
| **Image/video diffusion generation** | standalone top-level package `python/sglang/multimodal_gen/` (FastVideo fork, peer of srt, `ComposedPipelineBase`+`PipelineStages` inside one engine) | Yes |
| **Text-token diffusion** | `srt/dllm/` (masked-token, outputs text) | No |
| **TTS / speech generation** | **no counterpart at all** | — |

**Implication for the jax plane split**: the jax understanding plane and upstream `srt/` are the same concept (encode→scatter→reuse the standard AR LLM), heavily borrowable — exactly the direction of PR #1350; the diffusion plane (upstream picks "standalone package + composed stages in one engine", jax picks "a single staged GlobalScheduler unifying it") is jax's own trade-off; TTS has nothing upstream to borrow — jax (MiMo-Audio) is entirely its own.

## 2.2 Data structures

All co-located in `managers/schedule_batch.py` (same file as the text `Req`):

- `Modality` **has only three members** `IMAGE/VIDEO/AUDIO` — multi-image is modeled as N independent `Modality.IMAGE` items, no `MULTI_IMAGES`.
- `MultimodalDataItem`: common-fields-first (`modality/hash/pad_value/offsets/format`) + `feature` XOR `precomputed_embeddings` (comment: *one and only one will be empty*) + a `model_specific_data` dict (transparently accessed via `__getattr__`/`set`, no explicit `pixel_values` etc.). The docstring states *each item has its own hash and pad_value, enabling per-image RadixAttention caching*.
- **pad_value collision avoidance**: `MM_PAD_SHIFT_VALUE = 1_000_000`; `pad_value = 1_000_000 + (hash % (1<<30))`, i.e. `∈ [1e6, 1e6+2^30)`, specifically guaranteeing no collision with real text token ids; `sanity_check_mm_pad_shift_value(vocab_size)` raises when `vocab_size > 1e6`. (Contrast: jax main is `hash % 2^24`, no shift, no sanity.)
- The `MultimodalInputFormat` enum (`NORMAL/PROCESSOR_OUTPUT/PRECOMPUTED_EMBEDDING`) erases the implicit "which field got filled" convention.
- `MultimodalProcessorOutput` (typed processor boundary; upstream has migrated from "return a dict" to "return a dataclass") → `MultimodalInputs.from_processor_output` (where hashing + pad happen) → attached to `Req`.

## 2.3 Processor abstraction and architecture-keyed registration

- `BaseMultimodalProcessor` (ABC) has **only one** `@abstractmethod` `process_mm_data_async(...)`; the rest is shared: an HF `AutoProcessor` wrapper, `ATTR_NAME_TO_MODALITY` (output attribute name → Modality).
- **Registration = package scan + a `models` class attribute + match by architecture class name**: `import_processors` scans the package with `pkgutil.iter_modules`, `for arch in cls.models: PROCESSOR_MAPPING[arch] = cls`; `get_mm_processor` matches `cls.__name__` against `hf_config.architectures`. The registration key is the **HF architecture class** (e.g. `Qwen2_5_VLForConditionalGeneration`); a new processor self-registers just by being a file in the `processors/` package declaring the architectures it serves with `models=[...]` (43 files total).
- **Async parallel media loading**: the processor has a built-in `io_executor` (ThreadPool, `SGLANG_IO_WORKERS`) + `cpu_executor` (ProcessPool, fork) + `asyncio.wrap_future` — this is a documented capability gap for jax (jax loads in order / per-source).

## 2.4 The merge core `mm_utils.py`: an in-model prefill adapter

- **Single entry `general_mm_embed_routine(input_ids, forward_batch, language_model, multimodal_model, ...)`**: called once from the model forward. Gate = three ANDs: `not is_decode() ∧ not is_target_verify() ∧ forward_batch.contains_mm_inputs()`; otherwise plain-text `embed_tokens(input_ids)`. It copies `input_embeds` into a pre-allocated `forward_batch.input_embeds` (stable address for CUDA-graph) and calls `language_model(input_ids=None, input_embeds=input_embeds)` — **on an MM prefill step the LLM body is driven by embedding, never by token id**.
- `embed_mm_inputs`: flatten mm_items → resolve the embedder by the naming convention `getattr(model, f"get_{modality}_feature")` → `input_ids.clamp_(0, vocab-1)` (because placeholder rows store `pad_value` > vocab) → scatter.
- **The actual scatter uses `masked_scatter_`** (not the old index assignment; comment: *avoids the cudaStreamSynchronize that torch.where triggers*). The placeholder mask is built by `torch.isin(input_ids, pad_values)`.
- **padding pattern** bakes pad_value into input_ids (`MultiModalityDataPaddingPattern{TokenPairs,MultimodalTokens}`), the pivot for the §2.6 radix prefix hit.
- handoff = in-graph `forward_batch.input_embeds`: produced and consumed within **one** forward, no IPC, no stage boundary — the fundamental contrast with jax's "cross-stage/mesh dict-key handoff".

## 2.5 Model contract: a single `nn.Module` + 4 hooks

A VLM = a single `nn.Module`, whose `__init__` holds both `self.visual` (vision tower) + `self.model` (text LLM); `forward` calls `general_mm_embed_routine(language_model=self.model, multimodal_model=self)`. **Encode + merge + LLM forward all in one object, one process, one forward.**

The 4 hooks a new model must implement:

1. **`get_<modality>_feature(items)`**: concat item feature → `self.visual(pixel_values, grid_thw)`. Resolved by the naming convention `getattr(model, f"get_{modality}_feature")`.
2. **`get_input_embeddings()`**: `return self.model.embed_tokens` (hard-asserted by the routine).
3. **`pad_input_ids(input_ids, mm_inputs)`**: delegates to the padding pattern, expands placeholder tokens and replaces them with the per-item `pad_value` (so RadixAttention can distinguish media).
4. **deepstack (optional, Qwen3-VL/Omni)**: the vision tower returns a wide tensor `hidden*(1+num_deepstack)`; the routine calls `separate_deepstack_embeds` to split it, injects `input_deepstack_embeds` into the LLM, and adds them back per-layer via `post_residual_addition`.

Registration: each model file declares a module-level `EntryClass`; `ModelRegistry` scans with `pkgutil` `model_arch_name_to_cls[cls.__name__]=cls`; the HF arch string is resolved via `resolve_model_cls`.

## 2.6 Scheduler / radix integration

- **Single engine, single scheduler**: ViT is inlined in `model.forward`, same forward and same device batch as the LLM prefill. `mm_inputs` flow along `Req.multimodal_inputs` → `ScheduleBatch.multimodal_inputs` → `ForwardBatch.mm_inputs`.
- **The radix cache itself is mm-agnostic**: `RadixKey = token_ids + extra_key`, no mm hash field. mm distinction works because the mm hash is baked into placeholder token ids (pad_input_tokens) **before** the radix key is formed; those hash-valued ids are only `clamp_`-ed to vocab **after** the mm region is identified inside `embed_mm_inputs`. Same image → same hash → prefix hit.
- **chunked prefill × mm**: `_get_chunked_prefill_embedding` does per-item interval intersection, `MultiModalStaticCache` (a byte-budget LRU keyed by `item.hash`) + CPU feature offload as fallback. **caveat: chunked prefill is disabled only for Transformers-backend mm** (`uses_transformers_backend`, *to avoid partial multimodal chunk mismatches*); native VLMs keep it on — jax has no Transformers backend, so it should not copy the "disable".

## 2.7 Full request flow diagram

```
POST /v1/chat/completions (text + image)
  │ serving_chat: apply_chat_template → placeholder markers into prompt; GenerateReqInput(image_data)
  ▼ TokenizerManager._tokenize_one_request (async)
  │   await process_mm_data_async: io/cpu executor parallel load → HF AutoProcessor
  │   → input_ids(placeholders expanded) + pixel_values + grid; per-image MultimodalDataItem
  ▼ ZMQ → Scheduler.handle_generate_request (per TP rank)
  │   from_processor_output: set_pad_value (1e6 + hash%2^30, > vocab)
  │   pad_input_ids: overwrite placeholder region of origin_input_ids with pad_value (for radix key)
  │                  keep origin_input_ids_unpadded clean copy for detok
  ▼ schedule (mm gets no special treatment, continuous batching; chunked truncates as usual)
  │   RadixKey = token_ids(now contains pad_value) + extra_key  → same image hits / different image forks
  ▼ model.forward(input_ids, positions, forward_batch)  ── single nn.Module, single process, single mesh
  │   general_mm_embed_routine(language_model=self.model, multimodal_model=self)
  │     gate: first PP rank ∧ not decode/verify ∧ contains_mm_inputs()
  │     ├ get_image_feature(items) → self.visual(...)  ── ViT inlined!
  │     │    (chunked: MultiModalStaticCache.get(item.hash) miss→encode→set; feature offload CPU)
  │     ├ mask = isin(input_ids, pad_values); input_ids.clamp_(0,vocab-1) → embed_tokens
  │     └ input_embeds.masked_scatter_(mask, vision_emb)  ── the only merge write
  │   language_model(input_ids=None, input_embeds=input_embeds)  ── LLM driven by embedding
  ▼ decode: mm not involved (mrope delta is a scalar recompute)
  ▼ detokenize: use the unpadded copy, pad_value never reaches decode → streamed text
```

## 2.8 [Core] How upstream supports the features missing in Part 1 + JAX portability

| Missing capability | Upstream mechanism (function/class) | Key trick | Portable to JAX | Not portable / must change |
|---|---|---|---|---|
| **Multi-request vision-encode batching** | `embed_mm_inputs` flatten + `get_image_feature(items)` one `self.visual()` + `masked_scatter_` | concat-encode-split + `items_size` offset re-slice | concat-encode-split semantics are portable | `masked_scatter_`→`at[].set(mode=drop)`; dynamic total length → **total-length bucketing** |
| **mm × chunked prefill** | `_get_chunked_prefill_embedding` + `_get_chunked_embedding_by_item` + `MultiModalStaticCache` + CPU offload | interval intersection + per-item LRU cache + host-raw fallback re-encode | interval-intersection algebra + two-level cache structure | "casually placing a GPU tensor" on device is not portable (KV pool eats all HBM) → explicit pool + G1 budget; dynamic chunk length → padded gather-scatter |
| **radix per-image prefix cache** | pad_value baked into placeholder token id → RadixKey naturally distinguishes images | mm hash enters token id before the radix key is formed | mechanism fully portable | upstream pads into `origin_input_ids`; jax picks **Option B** padding into `cache_input_ids` (so forward never sees a >vocab id) |
| **VLM continuous batching** | single Scheduler, an mm req = a standard `Req` with `multimodal_inputs`; encode dynamically dispatched inside forward | gate `not decode ∧ contains_mm_inputs()` | framework already supports it structurally | upstream runs+cache-skips every step inside forward; jax changes it to host, once before prefill (to avoid in-forward recompile + chunk mismatch) |
| **DP-sharded vision encoding** | `run_dp_sharded_mrope_vision_model` + `get_dp_encoder_lb_assignment` (`multimodal/mm_utils.py`) | LPT greedy (assign by descending size to least-loaded rank) + pad-to-max all_gather + inverse permutation | LPT greedy + pad-to-max idea is portable (pad-to-max is bucket thinking) | all_gather over NCCL → TPU mesh collective + static padding; currently ViT runs fully replicated/redundant |
| **Avoiding recompiles** | PyTorch **dynamic shape**: concat any total length freely, no recompile | dynamic shape is first-class | **not directly portable** (the core porting difficulty) | `jax.jit` caches by aval(shape) → recompile per new shape → must **bucket** (patch/seq/total-length buckets) |

**The core porting difficulty**: upstream compresses all six capabilities into three pillars — "single engine + in-forward dynamic dispatch + dynamic shape". Because of JAX's three hard constraints (**static shape + no CUDA graph + TPU mesh**): pillar 1 (continuous batching) is structurally inherited; pillar 2 (in-forward dispatch) is intentionally changed to host-prefill-once dispatch; **pillar 3 (dynamic shape) cannot be inherited → bucketing must pin the infinite shape space to a finite set** — the unified difficulty and unified solution of all six ports. Directly portable are the "algorithmic semantics" (concat-encode-split, interval intersection, LPT greedy, pad-to-max, pad_value→radix); not portable are the three upstream environment dividends "free shape flexibility", "casual device caching", "single-process with no collective contention", which JAX must compensate with bucketing, an explicit cache pool + G1 budget, and a thread Event barrier + local-placement collectives, respectively.

> Disambiguation: what binds recompiles is **bucketing, not "splitting the jit"** — jit caches by aval, and shape crosses any jit/process boundary; splitting into two jits does not reduce the recompile count, it only decides which graph it lands in. The only lever that actually converges the count is bucketing.

## 2.9 Runtime routing in the CLI shell (is_multimodal semantics)

Upstream used to have `is_multimodal_gen`/`is_image_gen` on `ModelConfig` (always-False placeholders); main has **removed them entirely**, moving runtime-ownership decision up to the **CLI shell**: `cli/serve.py: get_is_diffusion_model(M)` looks at the checkpoint's physical identity (`model_index.json` contains `_diffusers_version`? HF `library_name=="diffusers"`?) to route to the `srt/` understanding+text runtime or the `multimodal_gen/` generation runtime. The three semantic questions each get a home: ① "which runtime to enter" = CLI shell by checkpoint identity (no ModelConfig field — the dead-code lesson); ② "whether to process media input" = `is_multimodal` (single semantic) + tri-state `enable_multimodal`; ③ "what to produce" = absorbed by the package identity. This "remove fields + split packages" is exactly the reference for jax's field decomposition (§3.2.5).

---

# Part 3 · sglang-jax Long-term Multimodal Architecture

> Nature of this part: aimed at the **long-term (to-be) target architecture**. In one line: **dual-plane separation** — the understanding plane (media→text) dissolves into the standard `srt` LLM control plane (in-model), with multimodal merge compressed into one shared `merge()` inside prefill; the generation plane (diffusion/TTS) keeps the staged runtime (only bounded this round). The two planes share a neutral `mm_core` layer (data structures / merge contract / processor registry). This round focuses on the **understanding plane (VLM)**, detailed to a plan; the generation plane is only bounded, details deferred.
>
> **Current landing scope (important)**: the in-model understanding-plane plan below is the **long-term target form**; **the only VLM landed in-model right now is MiMo-V2.5**, while Qwen2.5-VL / Qwen3-Omni stay staged (multi-stage) and migrate later. Unless noted otherwise, §3's in-model plan treats MiMo-V2.5 as the current implementer; landing status is in Part 4.

## 3.0 Design goals, jax constraints, trade-off principles

**Design goals**:
1. Converge main's structural duplication/coupling: §1.5 two merge implementations, §1.4 scattered hard-coded substring dispatch, §1.3 the union `Req` mixing three task types, §1.7 the 8 srt→multimodal reverse imports.
2. Minimize touch points for "adding a new understanding model": reuse the existing arch-keyed self-registration of upstream/srt.
3. Fill the missing features listed in Part 1 on VLMs (radix / chunked / multi-batch+batched-encode / bucketing / overlap / DP / memory).

**jax constraints that must be respected (non-negotiable)**:
- **Multi-host SPMD lockstep**: standard AR runs multi-host SPMD; once the understanding plane reuses the control plane, the VLM's whole forward (including ViT) lands on this multi-host path.
- **`jax.jit` non-uniform**: encode/AR use different jit/sharding/precision; the interface must treat non-uniform jit as a first-class case.
- **Sharded-scatter full-replication constraint**: before a scatter the operands must be resharded to fully replicated; **must use `jax.sharding.reshard`, not `with_sharding_constraint`** (the latter is an assertion under the all-Explicit-axes standard AR mesh, and crashes outright on 'data'-sharded input). Fix it into the merge contract, do not let it drift into models.
- **Replicate ViT, do not shard it**: the vision/audio tower only survives `dot_general` sharding by being forced fully replicated; in-model multi-host runs ViT fully replicated/redundant (each host computes it once, no cross-host collective, no deadlock, same as upstream).
- **JAX static shape + no CUDA graph**: the core porting difficulty of Part 2 §2.8 — must bucket.

**Trade-off principles**:
- **Split scope, share the foundation**: the understanding/generation plane boundary uses **two criteria** — satisfying either puts it in the generation plane (staged): ① media is produced as output; ② the model backbone is not the standard single-channel AR LLM form (cannot reuse the standard Scheduler's KV/sampling semantics, e.g. MiMo-Audio's 9-channel interleaved backbone + in-stage patch_decode). Both planes share `mm_core`.
- **Reuse over reinvent**: understanding-side registration and contracts follow srt's existing mechanisms and the upstream form, no fresh wheels.
- **Share one over write many**: anything solvable with one pure function + an explicit contract (like merge) should not be re-written inside each model's forward (each rewrite inevitably drifts, as §1.5 shows).

## 3.1 Dual-plane overview + why the understanding plane picks in-model

```
        HTTP request (chat / images|videos / audio.speech)
                  │  two-criteria plane split (media-as-output OR non-standard AR backbone → generation)
   ┌──────────────┴───────────────────────────────┐
   ▼ media-out                            text-out ▼
 Generation Plane (deferred this round)   Understanding Plane (focus this round)
 GlobalScheduler + stages          standard srt Scheduler (reuse control plane)
 diffusion(FLUX/Wan) / TTS         VLM = a model in the srt ModelRegistry
 per-stage mesh                    forward: text_embed
                                     → per-modality tower encode (per-model)
                                     → merge() shared pure function (scatter)
                                     → AR LLM (RadixAttn/KV/SPMD)
        │ one-way dependency                │ it IS a model on srt
        └────────────────┬─────────────────┘
                         ▼ shared, one-way
        mm CORE (neutral layer)
          data : MultimodalDataItem / MultimodalInputs (robust pad_value)
          merge: merge() pure fn (encode products → a single mask-scatter)
          proc : ProcessorRegistry (arch-keyed, upstream-style)
                         │ one-way
                         ▼
        existing srt: ModelRegistry / Scheduler / KVPool / RadixCache / loader / layers
```

**Why the understanding plane picks "reuse the control plane" over "staged" (the core reason for the refactor)**:

1. **The staged embed stage is debt, not an asset**: externalizing embed to its own mesh brings the §1.8-D host-roundtrip, the §1.5 two merges, and the topology waste of "embed takes 1 chip → 1+16=17 won't fit in a clean slice". Going in-model makes all three disappear at once: ViT shares the chips AR already holds (no 17th chip), no host-roundtrip, only one merge left.
2. **encode is usually not heavy**: understanding-side tower weights are small (ViT ~1.4 GB class), a minor part of AR's HBM budget, so in-model is feasible (the residual video-activation issue is in §3.3).
3. **Zero-cost reuse is a jax-proven strength**: the "generation" segment of text-out already runs on the standard LLM control plane (§1.1); moving "encode+merge" into the same forward returns the VLM to upstream's mature "single model + prefill adapter" form.

## 3.2 Understanding-plane design (focus this round)

### 3.2.1 Reuse the LLM control plane (VLM = a model on srt)

- The VLM model class declares `EntryClass` and registers into the `ModelRegistry` in `srt/models/registry.py` — same path as any text LLM. Note `import_model_classes` **only scans the top level of `srt.models`, non-recursively**, so a sub-package implementation needs a top-level thin-shell module (`*_mm.py`) to re-export + declare `EntryClass` (already landed in the PR, see Part 4).
- Inside its `forward`: `text_embed(input_ids) → per-modality tower encode → merge() → AR LLM body (reads forward_batch.input_embedding)`. This is exactly the upstream §2.5 "single model + prefill adapter" form, except merge is not welded into the model.
- continuous batching / RadixAttention / KV pool / sampling / multi-host SPMD are **all inherited**. Plain text and multimodal go through the same standard Scheduler, with no separate embed/vit stage, no per-stage mesh, no host-roundtrip.

### 3.2.2 Merge converges into a shared pure function

**Conclusion**: make merge **one shared pure function `mm_core.merge`, called from the model forward** (not bound to self/mesh, all parameters passed in), collapsing §1.5's two implementations + main's `_merge_multimodal` + each model's scatter. Three fixed contract rules:

1. **The only primitive = per-modality mask-scatter**: for each modality, locate its own placeholder rows with `mask = input_ids == tok_id`, `positions = jnp.nonzero(mask, size=feats.shape[0], fill_value=seq_len)`, `fused.at[positions].set(feats, mode="drop")`. **Per-modality**, not a global isin over concatenated features — this way an interleaved prompt `<image>…<audio>…<image>` never bleeds placeholder rows into another modality. Count mismatch is safe (surplus rows map out-of-bounds and `mode="drop"` discards them, never overwriting row 0). cumsum-gather is dropped.
2. **Key by the original placeholder token id (not pad_value)**: under Option B, forward's input_ids are clean (placeholder rows = real in-vocab token ids), so merge keys directly on input_ids; the per-image radix key is decoupled to `Req.cache_input_ids`. Therefore forward input never contains a >vocab id, and clamp is unnecessary.
3. **Fully replicate before scatter (with `jax.sharding.reshard`)**: reshard `text_embed`/`input_ids`/`mod_embeds` to `PartitionSpec()` (fully replicated), otherwise the scatter cannot resolve its output sharding under a sharded mesh. **Must `reshard`, not `with_sharding_constraint`** (constraint in §3.0) — fixed in CORE, especially critical under multi-host.

deepstack as an optional merge side-channel: the encoder returns sparse per-level features + a visual mask; **densify is not in merge** — it is done per-chunk on host by `ScheduleBatch._merge_multimodal` (to avoid a second device-side densify implementation).

### 3.2.3 Model contract (what an understanding VLM must implement)

```
class XxxVLForConditionalGeneration(...):   # register EntryClass
    supported_modalities = ("image", "video", ...)   # explicit declaration (decoupled from method names)
    def encode_<modality>(self, items) -> ModalityEmbed   # per-model: brings its own tower
    def forward(self, input_ids, forward_batch, ...):
        text = self.text_embed(input_ids)
        mods = [self.encode_image(...), ...]              # per-model
        fused = merge(text, mods, placeholder_ids, input_ids)  # shared
        return self.ar_llm(input_embedding=fused, ...)         # reuse srt text model
```

- **encode is per-model** (towers differ), aligned with upstream `get_<modality>_feature`, but resolved by registration (§3.2.5) rather than magic method-name parsing.
- **merge is shared** (§3.2.2).
- **The AR body reuses the srt text model** (e.g. `MiMoV2ForCausalLM`/`Qwen2`), reading only `forward_batch.input_embedding`, never seeing pixels.

**Loading contract G2 (two must-dos for a fused VLM)**: the standard loader/quantizer implicitly assumes "model = a homogeneous TP-sharded LLM"; a fused VLM is the first to stuff a heterogeneous subtree (replicated vision + sharded LLM) into one object, so two global mechanisms that walk blindly by name/regex go out of bounds:
- **G2-a: vision weight mappings must explicitly set `sharding=()`**. The loader's `_infer_default_sharding` guesses sharding by weight name (q/gate/up→column-parallel…); ViT-internal weights with the same names get mis-guessed as TP-sharded; but ViT runs fully replicated, so on a TP>1 mesh "replicated activation × sharded weight" makes `dot_general` unable to resolve the output layout → crash (**triggered on single-host TP>1 in M3**, not a multi-host issue). Fix: CORE provides a `replicate_mappings` helper + a load-time assertion.
- **G2-b: FP8 quantization rules must not hit `visual.*`**. Quantization replaces every `LinearBase` matched by its regex with `QuantizedLinear`; an overly broad regex sweeps in vision layers of the same name (in an fp8 ckpt the vision part is bf16 with no scale → numeric error). Fix: anchor the quant regex to the LLM subtree (`model\.layers\..*`) + add towers to an ignore list + assert after load that there is no `QuantizedLinear` under `visual.*` (first used by M3 MiMo fp8).

### 3.2.4 Data structures and contract convergence

| Current (main) | Target |
|---|---|
| union `Req` (latents+omni+audio mixed) | understanding plane: standard `srt` `Req` + `MultimodalInputs` (aligned with upstream, attached directly to the standard Req); generation plane typed payload (deferred) |
| `pad_value = hash % 2^24` (no defense) | `MM_PAD_SHIFT_VALUE(1e6) + hash % 2^30` + vocab sanity (ported from upstream §2.2, keeping the sha256 cross-device deterministic hash) |
| `Modality.MULTI_IMAGES` | **removed**, multi-image modeled as N `IMAGE` items (aligned with upstream) — unlocks per-image radix (each item has its own hash/pad_value) + one fewer modality branch + merge naturally consumes an N-item list |
| precomputed decided by "which field is filled" | introduce the `MultimodalInputFormat` enum (ported from upstream) |

The `auto_regressive` special-case in understanding-side `to_stage_reqs`/`from_stage` **disappears** — understanding uses the standard entry directly, the merge product goes straight into the standard prefill's `input_embedding`, with no cross-mesh dict key.

### 3.2.5 Registration and dispatch (three-level discriminator fields)

Registration is **split by plane**; the understanding side reuses srt's existing mechanisms, aligned with upstream:

- **Model registration**: a VLM declares `EntryClass` into srt's `ModelRegistry` (understanding VLM model files are physically moved into `srt/models/`, since `import_model_classes` only scans that level).
- **Processor table**: a processor subclass self-registers into `ProcessorRegistry` via `models=[arch]` + package scan, matched by `hf_config.architectures`; base = the upstream `BaseMultimodalProcessor` form (a single abstract method + HF wrapper + input-cap constants).
- **Capability attached to the model class**: `supported_modalities` (explicit set, authoritative, decoupled from method names), `audio_kind` (`"codes"|"features"|None`), `has_deepstack`.

**`is_multimodal` three-level split** (aligned with upstream §2.9's "remove fields + split packages"):

| Level | Semantic question | Expression |
|---|---|---|
| ❶ runtime routing | which plane to enter | **no field** — the launch entry queries `StageConfigRegistry` (a generation model has a stage YAML, ≈ upstream querying `model_index.json`) + a `--runtime` override |
| ❷ input processing | whether to process media input | `model_config.is_multimodal` **keeps the name, swaps the impl, narrows the semantics**: source of truth = capability-derived (arch → ModelRegistry → model class → `supported_modalities`/`encode_*`), the config-key is only a "class unresolvable" fallback |
| ❸ deployment override | run a VLM as plain text | tri-state `enable_multimodal` (None=auto / True / False to save HBM) |

**Tokenizer home: composition not subclassing** (aligned with upstream, which has no separate mm tokenizer class): understanding-related logic (media loading, processor calls, mm_items building, mrope) sinks into an `if is_multimodal:` guarded branch of the standard `TokenizerManager`; the `MultimodalTokenizer` subclass is kept only for generation-plane media-out endpoints.

### 3.2.6 Decouple srt ↔ multimodal (hard prerequisite)

The 8 reverse imports of §1.7 must be untangled before CORE sinks (otherwise they conflict with srt's existing `schedule_batch` naming): `MultimodalServerArgs` fields merge into srt's `ServerArgs`; `resolve_tokenizer_subdir`/`flatten_nested_list` sink to a srt common util; `GenerateOmniReqInput` is removed after the understanding side switches to the standard `Req` entry. Terminal dependency direction: `generation plane → CORE → srt` (one-way); the understanding side = a model inside srt + CORE. Bidirectional entanglement → DAG.

## 3.3 [Core] The plan to fill Part-1's missing features on VLMs

Per-item plans for the in-model understanding plane (implementation status in Part 4):

### 3.3.1 radix per-image prefix cache — Option B (dual-track cache_input_ids)

main's current state: radix globally disabled + dead pad→radix wiring. Two options:

| Option | where pad is written | trade-off |
|---|---|---|
| A (isomorphic to upstream) | pad into `origin_input_ids` | needs an unpadded copy + clamp before embed + merge keying by pad_value |
| **B (jax-native, chosen)** | pad into the reserved field `Req.cache_input_ids`, **only the radix-key construction switches to it** | `origin_input_ids` stays clean throughout → detokenize/embed/merge **need zero change**; deviates from upstream but converges to the single point "which id string RadixKey uses" |

**Pick B**: JAX `Embed.at[].get()` clamps a >vocab id by default and reads a dirty row (no upstream-style explicit clamp step); the existing merge locates by token id, so after padding the mask is all-False. B makes forward **never see a pad_value**, so these problems all vanish. The pad_value formula still uses §3.2.4's 1e6+hash%2^30. Companion: lift the understanding-plane radix disable (narrow the `server_args.py` forced-disable to apply only to the staged generation plane); a radix-key length mismatch must **fail loudly** (not silently fall back to `origin_input_ids`, which would cause wrong KV reuse).

### 3.3.2 chunked prefill in-model — encode whole first chunk + held by req + absolute-position slice

After moving ViT/encode into the main forward, the understanding chain must handle "the media placeholder region cut by a chunk". A three-level evolution path:

| Path | embedding holder | slice | roundtrip |
|---|---|---|---|
| **Phase A (甲, first cut)** | after the first chunk encodes, `device_get` back to host, attach to req | host numpy absolute-position slice (reuse `_merge_multimodal`) | D2H + per-chunk H2D (~1-2%, only the chunked tail) |
| **Phase B (乙, later optimization)** | keep the whole-segment embedding on device, attach to req | device `dynamic_slice` | none |
| **Final (终态)** | per-chunk per-item interval encode + a device LRU cache pool | padded gather-scatter (OOB sentinel) | none; HBM bounded and evictable |

Phase one takes Phase A (host-held, roundtrip cost quantified as noise-level); Phase B removes the 1-2% but keeps the whole-segment embedding resident in HBM (must enter the G1 budget); Final aligns with upstream `_get_chunked_embedding_by_item` interval-intersection + `MultiModalStaticCache`, but with JAX static-shape padded gather-scatter (`at[src_idx].get(mode="fill")` + `at[dst_idx].set(mode="drop")`, indices padded to a fixed bucket, one jit serving all chunks). chunked prefill **stays on** for mm requests (do not copy upstream's disable escape hatch; jax has no Transformers backend).

### 3.3.3 multi-batch + vision-encode batching — concat-encode-split

continuous batching is already inherited by reusing the control plane (multi-batch structurally supported). Vision-encode batching = replacing the per-request loop in `encode_mm_reqs` with "concat the whole batch, encode once → split by row count → merge each", isomorphic to upstream concat-encode-split, with a different landing point (host, not forward):

- **Contract split**: split the unified `embed_mm` into `encode_mm` (batched, model's own tower) + `merge_mm` (per-req).
- **Correctness premise (verified)**: ViT attention is segmented per image (`cu_seqlens`/`window_index` accumulate per image; window blocks and full-attention blocks don't cross images), so cross-request concat = a natural extension of one request with multiple images, no leakage.
- **Recompile control (jax-specific)**: the batch-level total patch count `ΣP` varies wildly with batch composition → needs **total-length bucketing** (tail padding as a separate cu_seqlens segment, merge discards by real row count).
- **OOM protection (jax-specific)**: N concurrent images merged into one encode → ViT dense attn activation ~(Σpatches)² → must split into multiple sequential encodes by a patch budget (cut at image boundaries, bit-identical).
- **Multi-host collective contention (jax-specific)**: the implicit collective launched by batched encode on the host scheduler thread can reorder relative to the forward thread → must use local placement (`make_array_from_process_local_data`) + a thread Event barrier.

### 3.3.4 Shape bucketing (the core porting difficulty) — dual jit + patch/seq buckets

The root constraint of Part 2 §2.8. Design:

- **Dual jit**: vision encode is a separate `encode_jit`, which after producing embeddings goes through the §3.3.2 slice → AR jit (zero change). Do not fold ViT into the LLM whole graph — protect the LLM hot graph's compile/sharding/precision freedom.
- **patch/seq/total-length bucketing**: add `vision_patch_paddings`/`--vision-bucket-size`, pad each image to an integer multiple of the canonical LLM grid (the real size as a traced parameter, ViT masks out padding patches in dense attention, output compacted to valid); input_ids get matching seq bucketing (pad to a multiple of 256); batched encode uses total-length bucketing.
- **Warmup and observability**: `CompilationManager` gains a vision warmup path; deployment enables `JAX_COMPILATION_CACHE_DIR` by default; wire a recompile probe, with a gate asserting "compile count under mixed-resolution traffic ≤ number of buckets".

> Trade-off of splitting ViT|merge in a single process: only worthwhile when seq is not bucketed; once seq is bucketed, splitting only saves a constant, so the current state uses a single encode jit (not split).

### 3.3.5 overlap, DP-sharded encode, multi-host

- **overlap**: an in-model VLM req goes through the standard scheduler, standard overlap (encode runs only at prefill, not at decode step, so it doesn't pollute overlapped decode). Prerequisites: fix the future-map size bug (size by the max decode bs bucket) + the encode-path collective contention (§3.3.3).
- **DP-sharded encode**: port upstream `get_dp_encoder_lb_assignment` (LPT greedy by patch count across ranks) + pad-to-max (already bucket thinking) + inverse permutation to merge back; also solves multi-host redundant compute. Slated for the optimization phase.
- **Multi-host SPMD**: ViT fully replicated (each host computes it once, no cross-host collective, no deadlock, same as upstream). The merge full-replication constraint (§3.2.2 rule 3) is naturally satisfied under multi-host; the replicated `input_embedding` feeds the sharded AR LLM, resharded by AR. Cost = redundant compute + HBM + raw pixel_values broadcast bandwidth across hosts (an M3 measurement item).

### 3.3.6 Memory planning G1 (vision activation reserve) — day-1 must-do

**Mechanism blind spot**: the standard runtime KV pool eats all "HBM remaining after load" in one shot **before** precompile, and the dummy batch has no vision; meanwhile ViT dense-attention activation is O(T²) (at T=16k patches, the scores term alone is ~8.6 GB), and going in-model brings this consumer back to the main table without changing the seating formula → the first big video request OOMs at runtime with no warning.

**Plan**: explicitly add `vision_activation_reserve` to the KV budget formula. Reserve sizing (recommended) = **AOT compile + XLA `memory_analysis`**: order the startup as `load weights → AOT lower the encode-jit @ the largest vision bucket (compile only, no execute) → memory_analysis reads temp_size → init_memory_pool with the reserve → precompile AR`. It auto-tracks bucket/kernel changes, and auto-zeros when the encode-jit is empty for a text-only model. Companion admission guard (`--vision-max-patches` hard-rejects over-limit input at the door). **Synergy with V-1**: after ViT migrates to a segment-packed flash kernel, scores don't land in HBM, O(T²)→O(T), and the G1 reserve shrinks greatly.

### 3.3.7 Video / high-resolution ViT activation — input capping

The only real residual in-model (belongs to the understanding plane). Upstream does not do per-forward activation chunking for a single large input — the main defense is **input-side capping** to squeeze a single item down to what one forward can hold (`smart_resize`/`smart_nframes`, `FPS_MAX_FRAMES`, `VIDEO_MAX_PIXELS` and similar constants into the processor base). This round = input capping + accept the cap (downsample/reject over-limit); full-fidelity huge-video intra-item chunking is deferred.

## 3.4 Generation plane (deferred this round, boundary only)

The generation plane keeps the `GlobalScheduler`-driven staged runtime (diffusion FLUX/Wan + TTS MiMo-Audio), scope-split from the understanding plane and sharing the `mm_core` common kernel (modality/payload/hash/model_specific_data); this round it is **not deepened**, only the boundary is locked:

- **Keep staged**: diffusion/TTS are natural staged scenarios (cross-mesh orchestration + no upstream template).
- **Shared depth = the kernel, not the whole type**: generation input needs first-class concepts meaningless to the understanding plane — "condition role / pairing between items / per-item strength" — and the `feature` XOR `precomputed_embeddings` invariant would be broken (Wan I2V needs the same first frame as both pixels + CLIP). So the generation plane builds its own `ConditionItem` by composition (kernel + role/paired_ref/strength, multiple payload representations coexisting), instead of reusing the understanding plane's `MultimodalDataItem` wholesale.
- **Three known convergence points (recorded, not done this round)**: Wan UMT5 promoted from "borrowing auto_regressive" to the generation plane's own text-encoder; the TTS `audio_backbone` in-stage AR loop placed in a state-encapsulating node; reconnect the commented-out `audio_decode` stage to fix the waveform break.

## 3.5 Summary of decisions made

① Understanding plane in-model reuses the control plane, not staged (long-term goal; **right now only MiMo-V2.5 is landed in-model, Qwen2.5-VL/Qwen3-Omni stay staged and migrate later**); ② merge = a shared pure function (called inside forward) + three contract rules (reshard); ③ registration split by plane (understanding reuses the srt `ModelRegistry` + processor table); ④ generation plane deferred this round, boundary only; ⑤ pad_value collision avoidance, split the union Req, remove `MULTI_IMAGES` (goal; `MULTI_IMAGES` still present currently, see §4.3 L-a); ⑥ video/high-res uses "input capping + accept the cap"; ⑦ processor-centric self-registration + capability + the `is_multimodal` three-level split; ⑧ shared merge is forced by default, override is YAGNI this round; ⑨ chunked takes Phase A (host-held), Phase B/Final slated for optimization; ⑩ G1 vision activation reserve is a day-1 must-do.

---

# Part 4 · VLM Refactor Milestones and Implementation Status (branch epic/support_mimo_v2.5_vlm)

> This part is pinned to the **post-rollback current branch** `epic/support_mimo_v2.5_vlm` (tip `c107a606`, base `main`). **Rollback summary**: only MiMo-V2.5 goes in-model; Qwen2.5-VL / Qwen3-Omni are reverted to staged (multi-stage), to migrate later. `file:line` is relative to that tip. Three status tiers: **landed & verified** (code present + CPU/structurally verifiable) / **landed code present, verification PENDING** (code present, end-to-end/eval are relayed pod reports not re-verified on-site) / **designed, not in branch** (designed, not yet in branch code).
>
> §4.2–4.5 below ("missing-feature backfill / backlog / verification status") all refer to **the sole in-model VLM = MiMo-V2.5**; Qwen2.5-VL/Qwen3-Omni go staged, and their Part-1 gaps remain at main's state (out of scope for in-model backfill this round).

## 4.1 Milestones (post-rollback: MiMo-V2.5 as the only in-model pilot)

This round the understanding plane **only migrates MiMo-V2.5 onto the in-model VLM route**; Qwen2.5-VL / Qwen3-Omni stay staged (multi-stage) — their in-model code was landed then stripped (proven feasible, deferred for stability and scope convergence), to migrate later. The shared foundation (M1 decouple + M2 CORE) is common to both routes; the old "tear down the staged understanding side" goal (formerly numbered M6) is **reversed** this round: the staged understanding side is intentionally kept for Qwen2.5-VL/Qwen3-Omni. (In commit labels MiMo in-model is tagged M4, which differs from the semantic numbering below — the table uses the semantic numbering.)

| M | Goal | Exit gate | Status | Representative commit / evidence |
|---|---|---|---|---|
| **M1** Decouple prerequisite (shared) | break `srt→multimodal` reverse imports, dependency becomes a DAG | no `from sgl_jax.srt.multimodal` in the srt control plane (excluding the `mm_core` lazy import) | **DONE** (structurally verifiable; residual lazy import in §4.3 L-i) | `1192f063` (M1 + mm_core scaffold) |
| **M2** CORE sink + structural convergence (shared) | neutral `mm_core` (`merge()`/pad_value/capability/processor/weights), Option B (pad→radix) | merge() unit tests pass; pad_value scatter+radix consistency checked | **DONE** (used by the in-model path) | `1192f063`, `69de65cd` (Option B: clean input_ids + radix key on `cache_input_ids`), `83824f18` (G2-a `replicate_mappings` + assertions); `mm_core/{merge,capability,pad_value,processor,mm_assembly,weights,audio_processor}.py` present |
| **M3** MiMo-V2.5 in-model (the sole understanding-plane pilot: +vision +RVQ-codes audio +multi-host +fp8) | image/audio end-to-end correct; multi-host replicated-ViT + fully-replicated scatter repro; KV retract-resume does not crash | **DONE (code present) / eval = relayed pod, PENDING** | `3abb63b0` (in-model + V-4 jit-safe ViT), `cc6b537e` (audio RVQ codes), `084b0a8d` (multi-host bring-up + G2-b quant scoping), `54296d41` (concurrency BUG 1/A/B/5/6 fixes), `c107a606` (KV retraction clears fused embedding + encode over full seq). Registration: `srt/models/mimo_v2_5/` + the top-level shell `mimo_v2_5_mm.py` (`EntryClass=[MiMoV2_5ForConditionalGeneration]`), via the standard `Scheduler.encode_mm_reqs` host encode path |
| **M4** Qwen2.5-VL / Qwen3-Omni stay staged (decision this round, migrate later) | the staged understanding chain is restored and usable (vit/embedding → auto_regressive); in-model code stripped | **DONE (reverted to staged)** | `fc861f5a` (strip in-model Qwen2.5-VL + Qwen3-Omni), `dfea3a98` (restore staged Qwen2.5-VL: vit → auto_regressive), `39802e36` (restore staged Qwen3-Omni: embedding → auto_regressive); stage YAML and the vit/embed runner+scheduler are all restored |

**Structural evidence (@ `c107a606`)**:
- **in-model (VLM route)**: only MiMo-V2.5, in `srt/models/mimo_v2_5/` (vision/audio tower + `mimo_v2_5_inmodel.py`) + the top-level shell `srt/models/mimo_v2_5_mm.py`; reuses `mm_core` + the standard `Scheduler`'s `encode_mm_reqs`/`_merge_multimodal`.
- **staged (understanding side still present)**: Qwen2.5-VL = `srt/multimodal/models/qwen2_5VL/` (`qwen2_5_vit.py`/`qwen2_5_vl_generation.py`) + `vit_scheduler`/`vit_model_runner` + `qwen2_5_vl_stage_config.yaml`(+`_tp4`); Qwen3-Omni = `srt/multimodal/models/qwen3_omni_moe/` + `embed_scheduler`/`embed_model_runner` + `qwen3_omni_stage_config.yaml`. `embed_model_runner.py`/`vit_model_runner.py` are **not deleted** (opposite of the pre-rollback state).
- **staged generation plane**: FLUX/Wan (diffusion) + MiMo-Audio (TTS/ASR) unchanged.
- **routing**: has a stage YAML → staged (Qwen2.5-VL/Qwen3-Omni/diffusion/MiMo-Audio); no stage YAML + registered in `ModelRegistry` (EntryClass) → in-model (MiMo-V2.5). `static_configs/` now has 7 YAMLs (no mimo_v2_5).

## 4.2 Missing-feature backfill status (Part-1 gap → in-model path, only MiMo-V2.5 this round)

> All features below are backfilled on the **in-model understanding plane**, with the sole current beneficiary = **MiMo-V2.5** (the only in-model VLM). Qwen2.5-VL/Qwen3-Omni go staged, and their Part-1 gaps remain at main's state.

| Feature | Status | commit | Evidence / notes |
|---|---|---|---|
| **multi-batch (structural)** | landed & verified | in-model path | `scheduler.py` collects `mm_reqs` per batch → `encode_mm_reqs` (`model_runner.py:795`); `_merge_multimodal` single pass over the whole batch; no "mm forced bs=1" limit |
| **chunked-prefill in-model (C-1 Phase A)** | landed & verified | `0a411402` etc. | host-side encode once before prefill, slice by chunk window to feed AR; no encode inside forward |
| **radix per-image (Option B)** | **PARTIAL** | `69de65cd`, `db582b54` (length-mismatch fails loudly) | `Req.cache_input_ids` + `radix_key_ids`/length assert landed, same-image hit / different-image fork works; **but per-image split is an open gap (L-a)** — `MULTI_IMAGES` still in the enum, multi-image folds into one IMAGE item (whole-set hash), so per-image caching does not yet exist |
| **vision-encode batching (L-k)** | landed & verified | `99aa7fdc` | `encode_mm_reqs`→`_encode_mm_batched` (with bucketing off, concat across requests for one ViT, split by placeholder count); currently in-model only MiMo vision is batched, audio per-req (conservative). (The original Qwen2.5-VL batching left the in-model path as it reverted to staged.) |
| **overlap re-enable (K-14)** | landed & verified | `7cc82c80` (depends on `edb1dde8` future-map size fix) + `54296d41` (BUG 1/A encode-path collective fix) | in-model mm follows `--disable-overlap-schedule` like text; the future-map is sized by the max decode bs bucket; the encode/forward collective contention is resolved by the `forward_idle` barrier + `make_array_from_process_local_data` (see §4.4) |
| **device-get batching (Phase A→B partial)** | landed & verified | `651203ed` | one batched D2H + async dispatch of N merge jits; full device-resident = L-m backlog |
| **G1 vision activation reserve** | landed & verified | `b636920b`/`2f5521ef` (AOT memory_analysis)/`47cacb75` (auto-size reserve + V-2 probe) | `_aot_vision_reserve_bytes` (AOT lower reads XLA temp_size + closed-form fallback); `--vision-max-patches` hard-reject at the door (admission guard) |
| **shape bucketing (V-2, grid+seq)** | landed (off by default) / **no in-model consumer this round** | `158c6561` | `encode_bucketed` now exists only on the staged `multimodal/models/qwen2_5VL/qwen2_5_vit.py` (Qwen2.5-VL reverted to staged); the in-model (MiMo) `vision_encoder` has no `encode_bucketed`, so V-2 in-model bucketing is dormant for now; `--vision-bucket-size` still present |
| **is_multimodal capability-first** | landed & verified | `d66d1f8e` | resolve architectures[0]→class→`is_multimodal_arch`; `supported_modalities` explicitly declared |
| **embed_mm unified-signature contract + startup assert** | landed & verified | `e55a6237` | `EMBED_MM_CONTRACT_PARAMS` + `reconcile_mm_capability` startup assertion; eliminates the H-1 drift |
| **K-1 per-modality scatter (interleave fix) / K-2 placeholder-count guard** | landed & verified | `3fe98fe2` / `bee680c7` | per-modality scatter in merge + intake guard |
| **C-3 release after prefill / K-18 CPU gate tests** | landed & verified | `b06f2769` / `1ef0b186` | release hook added; now **13** in-model mm CPU gate tests wired into `test/srt/run_suite.py` (in-model Qwen-related tests removed with `fc861f5a`) |

## 4.3 backlog / deferred (all open, deferred)

| # | Item | Size | Evidence |
|---|---|---|---|
| **L-a** | **P0-1 per-image item split** (the biggest gap) | large | `MULTI_IMAGES` still in the enum; multi-image folds into one IMAGE item (whole-set hash) → per-image radix does not exist. Upstream reference `get_new_expanded_mm_items`, but an upstream issue shows its offset assignment has a bug — do not copy it |
| **L-b** | merge silently drops when placeholder count ≠ feature rows | medium | `nonzero(size=feature_rows)`+`mode="drop"`; a naive assert would wrongly hit V-2 bucketing surplus scatter, so it must be bucketing-aware + infer audio rows per model |
| **L-c** | CLI flag system (`--runtime`/`--mm-understanding-runtime`/tri-state `enable_multimodal`/`--multimodal` alias) | medium | grep=0; currently closed by the "YAML removed → hit = generation" semantics + comments |
| **L-d** | U1 input-cap final state (`MediaInputCaps` vs `--vision-max-patches` reconciliation) | small(decision)+medium | `MediaInputCaps` has zero producers/consumers; what actually takes effect = admission reject |
| **L-e** | audio input has no admission cap (G1 only handles vision) | medium | over-long audio can bypass the door reject; audio reserve uses a proxy estimate |
| **L-f** | merge ordering contract for interleaved multimodal prompts | medium | merge requires mod_embeds in placeholder appearance order, but the in-model model (only MiMo this round) hard-codes per-modality blocks → interleaved prompts silently misalign |
| **L-g** | typed `mm_inputs` (raw dict contract) | medium-large | `_extract_mm_value` tolerates dict/object, a misspelled key silently returns None; the `MultimodalInputs` dataclass is defined but idle |
| **L-h** | video source loading (URL/path/data-URI crashes) | small | no `_load_video`, only pre-decoded frame arrays work. Upstream reference torchcodec→decord, TPU pins CPU decode |
| **L-i** | mm_core neutral layer not fully clean | medium | `mm_assembly.py` has one lazy reverse-import of the staged `modality_enum`; core data structures still live in the staged namespace — "both planes depend one-way on CORE" holds only if you "ignore the lazy import" |
| **L-k remaining** | total-length bucketing for batched encode + audio batching (needs proving the RVQ/mel tower segments per clip first) | large | Phase 1 vision-only (MiMo) landed; remaining sub-items open. (The original Qwen3-Omni deepstack batching left the in-model scope as it reverted to staged.) |
| **L-l** | cross-request encode cache (upstream `MultiModalStaticCache` style) | medium | per-item content-hash cache to skip re-encode; orthogonal to L-k; tip `c107a606` has no such mechanism |
| **L-m** | full device-resident Phase A→B | small-medium | partial already captured the main "de-serialize" win; the `_put` implicit-collective bug is fixed by BUG 1 (`make_array_from_process_local_data`, see §4.4), and the full device-resident version's gain is small (a wash) → still backlog |

## 4.4 in-flight and verification status (key clarification)

**The concurrency bug fixes have landed on the current branch tip `c107a606` (newly added relative to the pre-rollback snapshot `651203ed`; the in-model encode path currently serves only MiMo-V2.5).** A set of concurrency hazards under in-model VLM + continuous batching + overlap + multi-host mesh — their fixes were all introduced by `54296d41` ("land BUG 1/A/B/5/6") and are in place at the tip:

| # | bug | fix | `c107a606` evidence |
|---|---|---|---|
| 1 | `_put` implicit collective (multi-process `device_put` implicit allgather reorders with the forward thread) | switch to `make_array_from_process_local_data` | **applied** `model_runner.py:834` (inside `_put`) |
| A | overlap's second collective (encode-path embed_tokens gather) | add a `forward_idle` thread Event barrier | **applied** `tp_worker_overlap_thread.py:84` (`forward_idle=threading.Event()`) |
| B | sampler sharding mismatch (Explicit mesh) | add `_align_sharding` reshard | **applied** `layers/binary_search.py:30` (`def _align_sharding`, used at :173 for `jnp.where`) |
| 5 | L-k batched encode OOM | `_chunk_by_patch_budget` splits into multiple | **applied** `model_runner.py:109` (called at :1045) |
| 6 | vision reserve AOT probe shape bug | probe dict-safe + closed-form fix | **applied** `model_runner.py:36` (`_vision_probe_geometry` import, used at :1105) |

**=> Relative to the tip `c107a606`, BUG 1/A/B/5/6 are all in place** (introduced by `54296d41`; retraction-resume is additionally covered by the tip `c107a606` "KV retract clears fused embedding + encode over full seq"). The pre-rollback snapshot `651203ed` does not contain these fixes — they are "absent" only in the old narrative pinned to `651203ed`, which no longer applies to this branch.

**Causal chain (closed)**: K-14 re-enabling overlap (`7cc82c80`) + L-k batching (`99aa7fdc`) once introduced the encode-path collective contention (BUG 1/A) and the batched OOM (BUG 5); `54296d41` fixed them accordingly (`forward_idle` barrier + `make_array_from_process_local_data` + `_chunk_by_patch_budget`). The current tip has no known blocker for "must crash under high-concurrency multi-host"; high-concurrency correctness still awaits on-site pod re-verification (see below).

**eval accuracy conclusion (relayed pod measurement, PENDING on-site re-verification)**: per a relayed pod measurement (not re-verified on-site): "with the right config (mem 0.88 + `--vision-max-patches 4096` + probe-fix), CONC=1 full-set n=693 cumulative acc≈0.70, matching the pre-refactor full-set 0.710 → no accuracy regression". probe-fix (BUG 6) is in place at the tip; but without a checkpoint/TPU it cannot be re-verified on-site → **claimed, verification PENDING**; high-CONC (CONC=8) correctness pod validation is in progress.

## 4.5 One-line landing-status summary

- **Architecture trunk landed & structurally verifiable (@ `c107a606`)**: M1 decouple, M2 CORE + Option B, **M3 MiMo-V2.5 in-model** (the only in-model VLM), **M4 Qwen2.5-VL/Qwen3-Omni stay staged** (in-model code stripped, staged chain restored); the seven `mm_core` modules are in place.
- **Most missing features backfilled (code present, beneficiary = MiMo in-model)**: multi-batch, C-1 chunked, L-k batching (only MiMo vision), K-14 overlap (+ concurrency fixes), G1 reserve, is_multimodal capability-first, the three merge contracts, K-1/K-2, 13 CPU gate tests.
- **Three things to make explicit**: ① **per-image radix (L-a) is the biggest opening** — `MULTI_IMAGES` not removed, multi-image whole-set hash, per-image caching does not yet exist; ② **V-2 shape bucketing has no in-model consumer this round** — `encode_bucketed` reverted to staged with Qwen2.5-VL, MiMo has no bucketing; ③ **the concurrency bugs are fixed at the tip `c107a606`** (`54296d41`), but high-concurrency multi-host correctness + eval no-regression are still relayed pod reports awaiting on-site re-verification.
- **The generation plane (diffusion/TTS) roadmap is untouched**, still staged, deferred this round.

---

> **Appendix: evidence convention of this document.** All claims are based on reading three sources — `main` (the pre-refactor baseline, commit `0ed80f61`), the in-model work of PR #1350 `refactor/mm-understanding-in-model` (snapshot `651203ed`), and **the post-rollback current branch `epic/support_mimo_v2.5_vlm` (tip `c107a606`, the Part-4 baseline)** — plus reading `sgl-project/sglang` main upstream. Code claims carry `file:line` with the commit; this is a self-contained final design document, all factual conclusions are inlined in the body and it depends on no external document.
