# DFlash Stage C: Wire Greedy Runtime Path

## Goal

Turn `DFlashWorker` from the current fail-fast stub into a runnable greedy
draft/verify loop on a single TPU v6e chip. Build on stage A (config + dflash
info) and stage B (draft model + spec-info data structures). The end state is:
`sgl_jax.launch_server` starts with `--speculative-algorithm DFLASH`, loads
`Qwen/Qwen3-8B` target + `z-lab/Qwen3-8B-DFlash-b16` draft, and completes a
deterministic greedy generation for a prompt such as `"The capital of France is"`.

This is a port of the algorithmic structure in SGLang PyTorch PR 22077
("Add DFLASH speculative decoding support"), adapted to the sglang-jax scheduler
and worker contract. It is not a line-by-line port: CUDA graph, FlashInfer, and
the Triton fused KV materialize kernel do not apply here.

## Scope

In scope:

- `tp_size=1`, `dp_size=1`, `disable_overlap_schedule=True`
- greedy only: `topk=1`, `speculative_num_steps=1`, `block_size=16`
- `page_size=1`, `--disable-radix-cache` for the first correctness run
- attention backend `fa`

Explicitly out of scope for stage C:

- overlap scheduler, DP attention, `tp_size > 1`
- tree verification, retraction, non-empty `filter_batch` / DP split
- fused (Triton) KV materialization
- non-greedy verification
- radix prefix reuse (draft KV must be materialized for every reused target slot;
  defer until the simple path is correct)
- paged mode (`page_size > 1`)

## Reference Architecture (PR 22077)

The PyTorch implementation uses a single entry point
`DFlashWorker.forward_batch_generation(batch) -> GenerationBatchResult` with two
internal paths (extend / decode-verify). It:

- creates a draft `TpModelWorker(is_draft_worker=True)` that owns its own KV
  buffers but shares the target's `token_to_kv_pool_allocator` and
  `req_to_token_pool`
- delegates unimplemented methods to the target worker via `__getattr__`
- runs the draft non-causal 16-token block by reusing `ForwardMode.TARGET_VERIFY`
- keeps scheduler changes minimal (`validate_dflash_request` + one hook in
  `handle_generate_request`)

## sglang-jax Contract Differences

The porting work centers on these gaps versus PR 22077:

1. **Entry method + batch type.** The sglang-jax scheduler calls
   `draft_worker.forward_batch_speculative_generation(model_worker_batch)` and
   passes an already-flattened, DP-padded, precompile-shaped `ModelWorkerBatch`,
   not a `ScheduleBatch`. Verify-block KV allocation is done by the scheduler via
   `get_spec_model_worker_batch(..., draft_token_num=16)`, so the worker does not
   allocate target-verify KV itself.

2. **EAGLE-gated scheduler.** `_run_speculative_batch`, `GenerationBatchResult`,
   and the output processor are gated on `is_eagle()`, and
   `GenerationBatchResult.next_draft_input` is typed as `EagleDraftInput`. DFlash
   must be threaded through these points with `is_dflash()` branches.

## Decisions

- **Draft KV allocation:** draft `ModelWorker` owns its own 5-layer KV buffers but
  shares the target's `token_to_kv_pool_allocator` and `req_to_token_pool`, so
  committed tokens land at the same `cache_loc` in both pools.
- **Scheduler integration:** minimal `is_dflash()` hooks added alongside existing
  `is_eagle()` gates; reuse the EAGLE seq-len advance and output-processor
  accept-length expansion rather than a separate DFlash chain.

## Worker Design

`python/sgl_jax/srt/speculative/dflash_worker.py` — replace the stub.

Public surface required by the scheduler:

- `speculative_num_draft_tokens: int` (= `block_size` = 16)
- `forward_batch_speculative_generation(model_worker_batch, launch_done=None) -> GenerationBatchResult`
- `run_spec_decode_precompile()`
- `__getattr__(name)` delegating to `target_worker`

Not implemented in stage C (asserted/raised if hit): the three overlap methods,
`_can_use_fused_spec_prefill` (overlap disabled → scheduler short-circuits before
calling it).

### Construction

```text
draft_args = deepcopy(server_args); draft_args.skip_tokenizer_init = True
draft_args.attention_backend = "fa"
draft_worker = ModelWorker(
    server_args=draft_args, is_draft_worker=True, mesh=<same mesh>,
    req_to_token_pool=<target's>, token_to_kv_pool_allocator=<target's>,
)
draft_model = draft_worker.model_runner.model          # DFlashDraftModel (stage B)
draft_model.set_embed_and_head(<target embed>, <target lm_head>)  # borrow, do not load
block_size = server_args.speculative_num_draft_tokens  # 16
mask_token_id = parse_dflash_draft_config(...).mask_token_id
```

### Prefill path (`forward_mode.is_extend()`)

1. Set `capture_hidden_mode = FULL`; run target Qwen3 forward → `logits_output`
   (with concatenated 5-layer aux hidden `[Σextend, 20480]`) + greedy
   `next_token_ids`.
2. Build `DFlashDraftInput(verified_id=next_token_ids, target_hidden,
   ctx_lens=extend_seq_lens, draft_seq_lens)`.
3. `_append_target_hidden_to_draft_kv`: `ctx = hidden_norm(fc(target_hidden))`;
   for each of the 5 draft layers do `k=k_proj(ctx)`, `v=v_proj(ctx)`,
   `k=k_norm(k)`, `k=rope(positions,k)`, then write K/V into the draft KV pool at
   the committed `cache_loc`.
4. Return `GenerationBatchResult(logits_output, next_token_ids,
   next_draft_input=draft_input, accept_lens=None, ...)`.

### Decode / verify path (`forward_mode.is_decode()`)

1. `draft_input = model_worker_batch.spec_info` (the `DFlashDraftInput`).
2. **Draft block.** `block_ids = [verified_id, mask*15]`; `input_embeds` from the
   borrowed target embedding; `positions = seq_len + arange(16)`. Draft allocator
   temporarily `alloc(bs*16)` with `backup_state()/restore_state()` (EAGLE3-style,
   block discarded after use). Run draft model in `ForwardMode.TARGET_VERIFY`
   (non-causal over the 16-token block) → `draft_hidden [bs,16,H]`.
3. **Draft greedy sample.** `argmax(draft_hidden[:,1:] @ target_lm_head.T)` →
   `draft_next [bs,15]`; assemble `draft_token [bs,16] = [verified_id, d1..d15]`.
4. Build `DFlashVerifyInput(draft_token, positions, draft_token_num=16)`;
   `prepare_for_verify` builds target-verify causal metadata.
5. **Target verify.** Run target in `TARGET_VERIFY` over the block → `target_logits`;
   `target_predict = argmax(target_logits)`.
6. **Accept.** Reuse `compute_dflash_accept_len_and_bonus(candidates,
   target_predict)` → `accept_len`, `bonus`; derive `commit_len`,
   `new_verified_id`, `next_target_hidden`.
7. **Commit / free.** Free unaccepted target KV (page_size=1: compact
   `out_cache_loc`); update `req_to_token`, `seq_lens`, `kv_committed_len`;
   `_append_target_hidden_to_draft_kv` for the committed verify tokens.
8. Return `GenerationBatchResult(logits_output, next_token_ids=new_verified_id,
   accept_lens=accept_len, next_draft_input=draft_input(updated))`.

## Spec Info

Stage B already provides `DFlashDraftInput`, `DFlashVerifyInput`, and
`compute_dflash_accept_len_and_bonus`. Stage C adds:

- `DFlashVerifyInput.prepare_for_verify(model_worker_batch, page_size, target_worker)`
  building target-verify attention metadata for a linear 16-token block.
- A `verify(...)`-style helper returning `(new_verified_id, commit_lens,
  next_target_hidden, accept_len)` from `target_logits`.

`filter_batch` with non-empty `target_hidden` stays `NotImplementedError`
(dp_size=1, no retraction).

## Scheduler Adaptation (minimal `is_dflash()` hooks)

- `scheduler.py` draft-worker launch (~L346/L372): DFlash construction exists
  (stage A/B); ensure `run_spec_decode_precompile()` is invoked (L516 path is
  generic).
- `GenerationBatchResult.next_draft_input` type annotation → accept
  `DFlashDraftInput`; the `is_eagle()` gate + `isinstance(EagleDraftInput)`
  assertion at ~L2275/2280 must also accept dflash.
- `_run_speculative_batch` (~L2318): decode uses
  `get_spec_model_worker_batch(draft_token_num=16)`; seq-len advance reuses the
  EAGLE `advance_from_accept_lens` branch.
- `scheduler_output_processor_mixin.py` (~L310/L345): `resolve_spec_decode_token_ids`
  accept-length expansion accepts dflash `accept_lens`.

## Startup Guards

Fail fast unless: `algorithm == DFLASH`, `disable_overlap_schedule`,
`dp_size == 1`, `tp_size == 1`, all-greedy, no grammar/logprob/LoRA, target
exposes `get_embed_and_head()`, and target capture-layer dim matches the draft
`fc.weight` input dim. Per-request validation mirrors PR 22077's
`validate_dflash_request` (reject `return_logprob`, grammar-constrained decoding).

## Testing

- **CPU/JAX unit test** `python/sgl_jax/test/speculative/test_dflash_worker.py`:
  tiny config + mocked target, verify core state advance — prefill produces a
  well-shaped `DFlashDraftInput`; a decode step produces `DFlashVerifyInput`,
  runs accept/commit, and advances `seq_lens` consistently. Accept math is
  already covered by `test_dflash_info.py`.
- **TPU v6e-1 smoke** (recorded as a manual command):

```bash
python3 -m sgl_jax.launch_server \
  --model-path Qwen/Qwen3-8B \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 \
  --speculative-num-draft-tokens 16 \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --disable-overlap-schedule \
  --disable-radix-cache \
  --trust-remote-code \
  --device tpu --tp-size 1 --dp-size 1 \
  --dtype bfloat16 --attention-backend fa \
  --page-size 1 --max-running-requests 1 --max-total-tokens 4096 \
  --skip-server-warmup
```

Then a single greedy OpenAI-API request with prompt `"The capital of France is"`;
confirm deterministic output and no shape recompilation loop.

## Definition of Done

1. Server starts with DFLASH args; no stub `NotImplementedError`.
2. Target Qwen3 forward yields the 5-layer aux hidden capture DFlash needs.
3. Draft model consumes target hidden + embedding and drafts a 16-token block.
4. Verify computes accept length and bonus token.
5. Batch metadata, KV materialization, and token handoff close the loop in the
   current sglang-jax scheduler.
6. One deterministic greedy smoke case passes.
7. A CPU/JAX unit test covers `DFlashWorker` core state advance; the TPU smoke is
   a recorded manual command.

## References

- SGLang PR 22077, "Add DFLASH speculative decoding support"
- `docs/design/dflash_tpu_v6e1.md` (stage B design)
- `z-lab/Qwen3-8B-DFlash-b16` checkpoint config

## Implementation Status (stage C)

Landed:

- `dflash_info.py`: `build_dflash_draft_block`, `dflash_greedy_verify_outputs`,
  `dflash_committed_slices`, and `DFlashVerifyInput.prepare_for_verify` / `verify`.
  Pure logic is CPU-unit-tested (`test_dflash_info.py`).
- `dflash_worker.py`: full `DFlashWorker` (construction, prefill, decode/verify,
  KV materialization, greedy head sampling). Array helpers are CPU-unit-tested
  via `test_dflash_worker.py` (worker built with `__new__`).
- `scheduler.py`: `validate_dflash_request` request guard; widened the
  `next_draft_input` gate/assertion to accept `DFlashDraftInput`.
- `schedule_batch.py`: `_split_spec_info_per_rank` passes `DFlashDraftInput`
  through for dp_size=1.
- `scheduler_output_processor_mixin.py`: decode output-id extension and
  `kv_committed_len` advance now fire for `is_dflash()`.

Adaptation vs. original design: sglang-jax's `ModelWorker.__init__` takes no KV
allocator, so the draft worker is built sharing only `req_to_token_pool` (like
EAGLE) and the target's `token_to_kv_pool_allocator` is aliased onto the draft
`model_runner` after construction.

### TPU bring-up risk areas (validate first on v6e-1)

These paths are structurally in place but only verifiable on device:

1. **Target aux-hidden capture shape.** Prefill assumes `logits_output.hidden_states`
   is the concatenated `[Σextend, K*H]` feature when `capture_hidden_mode=FULL`.
2. **Draft non-causal block attention.** The draft block runs in `TARGET_VERIFY`
   with `get_eagle_forward_metadata`; the `fa` backend may need explicit
   non-causal / linear-block mask support (cf. the QLEN_ONLY mask gap).
3. **KV commit / free accounting.** `kv_committed_len` currently mirrors EAGLE
   (`+= accept-1`); the exact delta, freeing of rejected verify-block slots, and
   `req_to_token` update for committed block tokens need on-device verification.
4. **Draft-block allocator backup/restore** collision-freedom under the aliased
   shared allocator.
5. **Greedy verify only.** Non-greedy requests silently fall back to greedy
   argmax verify (matches PR 22077); no request-level rejection.

