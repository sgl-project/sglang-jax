# Ling3 Tiny HF CPU golden validation

This directory provides the merge gate for Ling3 Tiny numerical accuracy. It
uses the official Hugging Face `modeling_bailing_moe_v3.py` and BF16 checkpoint
as the golden. The official model's GPU/Triton-only FLA primitives are replaced
on CPU by their documented pure-Torch equations; model structure, weights,
routing, MLA, residuals, normalization, and LM head remain official HF code.

The model revision is pinned in `prompts.json`. Tokenization happens once in the
Torch job and the exact `input_ids` are stored in each golden artifact, so the
JAX comparison does not depend on tokenizer or chat-template behavior.

## 1. Generate the Torch CPU golden

Use a Linux machine with at least 48 GiB RAM and enough local/model-cache space
for the 15.8 GB BF16 checkpoint.

```bash
python generate_torch_cpu_golden.py \
  --model-path /models/Ling-3.0-tiny-b61f4338 \
  --output-dir /artifacts/ling3-tiny-hf-cpu
```

The command first runs a reduced random-weight self-test, then writes one NPZ
per case plus `manifest.json`. Each NPZ contains exact input IDs, full first-token
FP32 logits, top-20 logprobs for every greedy step, greedy token IDs, and the
official model's per-layer prefill hidden states.

## 2. Launch JAX with opt-in dumps

Set the dump variables before the server starts. They are disabled by default
and therefore do not alter normal serving executables.

```bash
export SGLANG_JAX_DEBUG_DUMP=1
export SGLANG_JAX_DEBUG_DUMP_DIR=/artifacts/ling3-tiny-jax-dumps
export SGLANG_JAX_DEBUG_DUMP_COMPONENTS=ling3_io,ling3_layer,ling3_model
python -m sgl_jax.launch_server \
  --model-path /models/Ling-3.0-tiny-b61f4338 \
  --tp-size 1 \
  --device tpu \
  --page-size 128 \
  --host 127.0.0.1 \
  --port 30000
```

For the normal gate, `SGLANG_JAX_DEBUG_DUMP_COMPONENTS=ling3_io` is sufficient.
Enable `ling3_layer,ling3_model` only to bisect a mismatch. The implementation
follows the callback-based tensor dump pattern from PR #1062, with a JSONL
manifest added for deterministic artifact lookup.

## 3. Compare

```bash
python compare_jax_server.py \
  --base-url http://127.0.0.1:30000 \
  --golden-dir /artifacts/ling3-tiny-hf-cpu \
  --jax-dump-dir /artifacts/ling3-tiny-jax-dumps \
  --output /artifacts/ling3-tiny-comparison.json
```

The gate checks full first-token logits, top-k logprobs at every step, and exact
greedy token IDs. The checked-in thresholds are initial cross-device BF16
guardrails; the validation report must retain the raw metrics so reviewers can
judge the observed margin rather than seeing only pass/fail.
