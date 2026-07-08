# DFlash on TPU v6e-1 Design

## Goal

Bring up DFlash speculative decoding for `z-lab/Qwen3-8B-DFlash-b16` on a single
TPU v6e chip. The initial scope is deliberately narrow:

- target model: Qwen3 dense family, validated first with Qwen3-8B target plus
  Qwen3-8B-DFlash-b16 draft
- execution: `tp_size=1`, `dp_size=1`, no overlap scheduler, no DP attention
- decoding: greedy only, `topk=1`, no tree verification
- block size: draft checkpoint `block_size=16`
- attention backend: existing TPU FlashAttention path where possible

This is not a direct port of SGLang PyTorch DFlash. CUDA graph, FlashInfer, and
Triton fused KV materialization do not map to TPU/JAX. The design keeps the
algorithmic data flow from the PyTorch implementation, but reshapes it around
static JAX arrays, fixed compile buckets, and explicit sharding.

## Checkpoint Facts

The draft checkpoint config declares:

- `architectures = ["DFlashDraftModel"]`
- `model_type = "qwen3"`
- `num_hidden_layers = 5`
- `hidden_size = 4096`
- `num_attention_heads = 32`
- `num_key_value_heads = 8`
- `head_dim = 128`
- `block_size = 16`
- `num_target_layers = 36`
- `dflash_config.target_layer_ids = [1, 9, 17, 25, 33]`
- `dflash_config.mask_token_id = 151669`
- `vocab_size = 151936`

The five captured target layer features are concatenated into a
`5 * 4096 = 20480` feature vector and projected by the draft `fc` layer back to
`4096`.

## Existing Stage 1

The current stage already added:

- `SpeculativeAlgorithm.DFLASH`
- `parse_dflash_draft_config()` and mask token resolution helpers
- DFlash server argument validation for the minimal non-overlap path
- target-layer capture setup in `ModelRunner.load_model()`
- scheduler import hook for `sgl_jax.srt.speculative.dflash_worker.DFlashWorker`

The missing pieces are:

- `python/sgl_jax/srt/models/dflash.py`
- `DFlashDraftInput` / `DFlashVerifyInput`
- `python/sgl_jax/srt/speculative/dflash_worker.py`
- target verify attention metadata for a linear block
- draft KV materialization from captured target hidden states
- accept/commit path that updates request state, KV allocation, req-to-token
  mapping, and next draft state

## Runtime Loop

### Prefill

1. Run target Qwen3 prefill with `CaptureHiddenMode.FULL`.
2. Target Qwen3 captures configured aux hidden states at target layers
   `[1, 9, 17, 25, 33]`, using the existing Qwen3 aux-hidden mechanism.
3. The logits processor returns concatenated aux hidden features for all prompt
   tokens.
4. Greedy sample the first target token normally.
5. Create `DFlashDraftInput`:
   - `verified_id`: sampled next token per request
   - `target_hidden`: captured prompt features, shape
     `[sum(extend_seq_lens), 20480]`
   - `ctx_lens`: prompt token count per request
   - `draft_seq_lens`: committed draft prefix length
6. Project `target_hidden` through draft `fc + hidden_norm`.
7. For each draft layer, run K/V projection, K norm, RoPE, then write K/V into
   the draft KV pool at the same cache locations as target committed tokens.
8. Store `DFlashDraftInput` on the running request state.

### Decode Step

1. Append any pending committed target hidden features from the previous verify
   step into the draft KV pool.
2. Build the draft block input per request:
   - token 0: `verified_id`
   - tokens 1..15: `mask_token_id`
   - positions: absolute target positions `seq_len + arange(16)`
3. Run the draft model as a fixed-size non-causal block over 16 tokens.
4. Greedy sample draft proposals from the target LM head using draft hidden states
   at positions 1..15.
5. Build `DFlashVerifyInput`:
   - `draft_token`: flattened `[verified_id, d1, ..., d15]`
   - `positions`: flattened absolute positions
   - `draft_token_num = 16`
   - `capture_hidden_mode = FULL`
6. Run target verify in `ForwardMode.TARGET_VERIFY` over the whole block.
7. Compute greedy accept:
   - `target_predict = argmax(target_logits).reshape(bs, 16)`
   - accept while `draft_token[:, 1:] == target_predict[:, :-1]`
   - commit length is accepted draft tokens plus one bonus token
   - bonus token is `target_predict[:, accept_len]`
8. Commit accepted tokens and bonus to request output.
9. Free uncommitted target KV slots, compact `out_cache_loc`, update
   `req_to_token`, `seq_lens`, `kv_committed_len`, and scheduler accounting.
10. Slice target aux hidden for committed verify tokens and put it back into
    `DFlashDraftInput` for the next step.

## JAX Model Design

Add `python/sgl_jax/srt/models/dflash.py`.

The model should be an NNX module with no owned embedding and no owned LM head.
It consumes `forward_batch.input_embedding`, which is built from the target
embedding by the worker.

Core classes:

- `DFlashAttention`
  - use Qwen3-style q/k/v projections, q/k RMSNorm, RoPE, and `RadixAttention`
  - use non-causal draft-block attention metadata
  - expose `kv_proj_only(hidden_states, positions)` for draft KV
    materialization
- `DFlashMLP`
  - Qwen3 gated MLP with separate `gate_proj`, `up_proj`, `down_proj`
- `DFlashDecoderLayer`
  - Qwen3 pre-norm residual structure
- `DFlashDraftModel`
  - layers, final RMSNorm, `fc`, `hidden_norm`
  - `project_target_hidden(target_hidden)`
  - `set_embed_and_head()` should store borrowed target weights if that is
    simpler for worker code, but the model should not load these weights from the
    draft checkpoint

Weight mapping must support these draft checkpoint keys:

- `model.layers.*.self_attn.{q,k,v,o}_proj.weight`
- `model.layers.*.self_attn.{q,k}_norm.weight`
- `model.layers.*.mlp.{gate,up,down}_proj.weight`
- `model.layers.*.{input,post_attention}_layernorm.weight`
- `model.norm.weight`
- `fc.weight`
- `hidden_norm.weight`

For TPU v6e-1, start with unfused Q/K/V projections. Fusing QKV is not required
to bring up correctness and makes checkpoint mapping more invasive.

## Spec Info Design

Add `python/sgl_jax/srt/speculative/dflash_info.py`.

`DFlashDraftInput` should implement the local `SpecInput` protocol:

- `verified_id: np.ndarray | jax.Array`
- `target_hidden: jax.Array`
- `ctx_lens: np.ndarray`
- `draft_seq_lens: np.ndarray`
- `capture_hidden_mode = CaptureHiddenMode.FULL`

Initial scope can set `filter_batch()` and `merge_batch()` to strict,
well-tested CPU/host operations and reject DP-specific paths. Since `dp_size=1`,
there is no need to implement DP scatter/split in the first runnable version.

`DFlashVerifyInput` should be a pytree because it flows into JIT:

- `draft_token: jax.Array`
- `positions: jax.Array`
- `draft_token_num: int`
- `custom_mask: jax.Array | None`
- `capture_hidden_mode = CaptureHiddenMode.FULL`

It should provide:

- `prepare_for_verify(model_worker_batch, page_size, target_worker)`
- `verify_greedy(logits_output, batch_shape) -> accept_lens, commit_lens, bonus`

For page size 1, commit/free logic is straightforward. For paged mode, keep the
first runnable target on `page_size=1` unless existing JAX paged allocation APIs
are already stable for variable commit lengths.

## Worker Design

Add `python/sgl_jax/srt/speculative/dflash_worker.py`.

The worker should follow the existing composition pattern:

- hold `target_worker`
- create a draft `ModelWorker(..., is_draft_worker=True)` using the same mesh
- share target `ReqToTokenPool` initially
- share or coordinate the token allocator carefully so target and draft cache
  locations match for committed tokens

Minimal methods:

- `forward_batch_speculative_generation(model_worker_batch, launch_done=None)`
- `forward_target_extend(...)`
- `draft_extend_for_prefill(...)`
- `draft(...)`
- `verify(...)`
- `draft_extend_for_decode(...)`

The existing `BaseSpecWorker` is EAGLE-shaped, so DFlash should not inherit it
blindly unless the interface is generalized. The first version can provide the
same external method used by the scheduler while keeping DFlash-specific internals
separate.

## TPU/JAX Constraints

### Static Shapes

DFlash has a natural fixed block size of 16. Use that as the core compile shape:

- draft block input: `[bs * 16]`
- verify input: `[bs * 16]`
- target hidden returned by verify: `[bs * 16, 20480]`
- draft proposal logits are not needed as full vocab tensors if greedy head
  sampling can be done directly from hidden states

Use existing precompile batch paddings for `bs`; do not introduce dynamic
per-request token shapes inside JIT.

### Greedy Head Sampling

On v6e-1, `tp_size=1`, so greedy sampling can compute:

```text
token = argmax(draft_hidden @ target_lm_head.T)
```

for `bs * 15` rows. This is a large `[tokens, vocab]` matmul but acceptable for
bring-up. Later TP support needs a shard-local max plus all-gather of max/value
pairs, matching the PyTorch PR conceptually.

### KV Materialization

Correctness-first path:

```text
ctx = hidden_norm(fc(target_hidden))
for each draft layer:
    k = k_proj(ctx)
    v = v_proj(ctx)
    k = k_norm(k)
    k = rope(positions, k)
    token_to_kv_pool.set_kv_buffer(layer_id, cache_loc, k, v)
```

This loops over five draft layers and writes K/V into the draft pool. On v6e-1
that is acceptable for the first runnable path. A later optimization can batch
layer KV projection with `vmap` or a stacked weight layout, but only after the
simple path is correct.

### Attention Mask

Draft attention is non-causal over the 16-token block plus committed prefix. If
the current `FlashAttention` backend cannot express encoder-only/non-causal
behavior in `TARGET_VERIFY`, introduce DFlash-specific metadata instead of
overloading EAGLE tree mask semantics.

Target verify is causal over prefix plus block. For `page_size=1`, the standard
target verify causal path should be preferred. Only build a custom mask if the
backend requires one.

### Memory

Qwen3-8B bf16 weights are roughly 16 GB. The 5-layer DFlash draft adds roughly
another 2-3 GB of weights plus a second draft KV pool. TPU v6e-1 memory is tight,
so the first launch should use conservative limits:

- `--dtype bfloat16`
- `--mem-fraction-static` below the current Qwen3-only value if allocation fails
- `--max-running-requests 1`
- `--max-total-tokens` small enough for smoke tests
- `--page-size 1` for the first correctness run
- `--disable-overlap-schedule`
- `--disable-radix-cache` if shared target/draft KV correctness is unclear

## Required Guards

At startup, fail fast unless:

- `speculative_algorithm == "DFLASH"`
- `disable_overlap_schedule is True`
- `dp_size == 1`
- `tp_size == 1` for the first version
- all requests are greedy
- no grammar/constrained decoding
- no logprob return
- no LoRA
- target model exposes `get_embed_and_head()`
- target capture layers exactly match the draft `fc.weight` input dimension

## Implementation Order

1. Tighten config validation.
   - Add `dp_size == 1`, `tp_size == 1`, greedy-only, no grammar/logprob guards.
   - Fix the stage1 comment: for this checkpoint `block_size == speculative_num_draft_tokens == 16`, not `num_draft_tokens - 1`.

2. Add the DFlash draft model.
   - Implement model classes and weight mappings.
   - Add a CPU/JAX unit test that instantiates a tiny config and verifies shapes:
     `project_target_hidden`, draft forward, and weight mapping keys.

3. Add DFlash spec info.
   - Implement greedy accept math as a pure JAX function:
     `accept_len = cumprod(candidates[:, 1:] == target_predict[:, :-1]).sum(axis=1)`.
   - Add tests for accept length and bonus token selection.

4. Add DFlash worker prefill path.
   - Capture target hidden.
   - Materialize prompt hidden into draft KV.
   - Store `DFlashDraftInput`.
   - Smoke test one prompt with `max_new_tokens=1`.

5. Add draft block and target verify path.
   - Build `[verified_id, mask * 15]`.
   - Run draft block with `input_embedding`.
   - Greedy sample draft tokens.
   - Run target verify.
   - Commit accepted tokens and bonus.

6. Add end-to-end TPU smoke command.
   - Start server on v6e-1 with one request.
   - Verify deterministic greedy output and no shape recompilation loop.

## First Smoke Command

```bash
python3 -m sgl_jax.launch_server \
  --model-path Qwen/Qwen3-8B \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 \
  --speculative-num-draft-tokens 16 \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --disable-overlap-schedule \
  --trust-remote-code \
  --device tpu \
  --tp-size 1 \
  --dp-size 1 \
  --dtype bfloat16 \
  --attention-backend fa \
  --page-size 1 \
  --max-running-requests 1 \
  --max-total-tokens 4096 \
  --skip-server-warmup
```

After it runs, increase `max_total_tokens`, enable normal warmup, and then test
`page_size=64` only after the page-1 path is stable.

## Known Risks

- `Qwen3ForCausalLM.__call__` currently always passes `aux_hidden_states` to the
  logits processor. It should mirror Llama and clear aux states unless
  `capture_aux_hidden_states` is enabled.
- `set_eagle3_layers_to_capture()` is reused for DFlash. A dedicated
  `set_dflash_layers_to_capture()` would make the intent clearer and allow
  stricter validation.
- Draft non-causal attention may need explicit backend support rather than
  reusing `TARGET_VERIFY`.
- Shared target/draft KV allocation has to be handled carefully. Prefix-cache
  reuse is unsafe unless draft KV has been materialized for every target KV slot.
- Full-vocab greedy sampling from draft hidden is simple but expensive. It is
  acceptable for v6e-1 bring-up, not the final performance path.

## References

- `z-lab/Qwen3-8B-DFlash-b16` Hugging Face checkpoint config
- SGLang PR 22077, "Add DFLASH speculative decoding support"
