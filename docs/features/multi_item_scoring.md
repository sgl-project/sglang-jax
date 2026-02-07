# Multi-Item Scoring (JAX)

## Summary

Multi-item scoring combines `query` and many `items` into one prefill sequence:

`query<delimiter>item1<delimiter>item2<delimiter>...<delimiter>`

For text requests, the server tokenizes `query` and `items` separately and builds this
sequence in token space (inserting delimiter token IDs directly) to avoid retokenization drift.

The implementation uses a custom attention mask so each item only attends to:

- the shared query prefix
- its own item block (causal)

It must not attend to other items.

## Required Flags

Enable with:

- `--multi-item-scoring-delimiter <token_id>`

When enabled, these constraints are enforced at startup:

- `--disable-radix-cache`
- `--chunked-prefill-size -1`
- `--attention-backend fa`
- no speculative decoding (`--speculative-algorithm` unset)

Optional safety bound:

- `--max-multi-item-seq-len` (default: `8192`)
- `--multi-item-scoring-chunk-size` (default: `2`, set `0` to disable chunking)

Smaller chunk sizes improve isolation robustness but reduce peak throughput.

## Current Kernel Compatibility Note

The current fused ragged attention path used by multi-item scoring is validated on
Qwen3 models in this branch, but some Qwen2.5 variants can fail at runtime with:

- `ValueError: ... reshape((2, -1, 8, 128)) ...`

This indicates a KV layout assumption mismatch in the fused kernel path (for example,
KV-head tiling that does not match the current fixed tile shape).

Until the kernel path is generalized, treat Qwen3 runs as the validated rollout target
for this feature gate.

## Request-Side Validation

For multi-item scoring requests:

- query must contain at least one token
- number of items must be <= 128
- delimiter token cannot appear in query or items
- combined sequence length must be <= `max_multi_item_seq_len`

## Logprob Semantics

In multi-item mode, scores come from `input_token_ids_logprobs` at delimiter boundaries:

- each delimiter entry corresponds to the model state at `delimiter_position - 1`
  (the last token before that delimiter)
- first delimiter (query/item1 boundary) is ignored
- one boundary score is used per item

## Position Semantics

Multi-item scoring uses delimiter-reset RoPE positions in extend mode:

- query tokens use standard positions `0..query_len-1`
- each item block resets at its delimiter token
- this removes position drift from earlier-item length changes

`apply_softmax` behavior is unchanged:

- `True`: softmax over provided `label_token_ids`
- `False`: `exp(logprob)` values

## Comparison Plan (JAX vs PyTorch)

Use the same model + tokenizer revision and compare:

1. raw token-id logprobs at each scored boundary
2. final score vectors after softmax/exp conversion
3. isolation correctness by perturbing one item while holding others fixed
4. latency + throughput vs single-item baseline

For parity, use both references:

1. serial semantics: `query + item`
2. multi semantics: `query + delimiter + item`

Multi-item mode intentionally inserts visible delimiters, so query-only parity is expected to differ.

Recommended parity inputs:

- tokenized inputs (avoid text-tokenization drift)
- item counts: 1, 8, 32, 64, 128
- mixed item lengths, including empty-item cases

Reusable scripts:

- `scripts/multi_item/evaluate_score_endpoint.py`
- `scripts/multi_item/combine_multi_item_eval.py`

Regression coverage:

- `test/srt/test_multi_item_regression.py`
