# Speculative Decoding (EAGLE / EAGLE3)

SGL-JAX ships a full speculative decoding pipeline that follows the EAGLE family
of algorithms. A lightweight *draft* model proposes several candidate tokens in
parallel, and the *target* model verifies them in larger batches. When tokens
are accepted, we skip several expensive target forward passes and gain higher
throughput without hurting quality.

This document describes the implementation that exists in the repository today,
the runtime flags you must set, and the combinations of models we currently
validate.

## How the pipeline works

| Stage | What happens | Key knobs |
| --- | --- | --- |
| Prefill | Target model runs as usual to build initial KV-cache. | `--max-prefill-tokens`, `--page-size` |
| Draft decode | The draft runner (EAGLE or EAGLE3) grows every request by proposing `speculative_num_steps × speculative_eagle_topk` tokens. | `--speculative-num-steps`, `--speculative-eagle-topk`, `--speculative-num-draft-tokens` |
| Verify | Verified tokens are replayed on the target model with custom attention masks. Tokens that pass are committed, failures fall back to regular decode. | `--speculative-accept-threshold-single`, `--speculative-accept-threshold-acc` |

Two draft flavors are available:

- **EAGLE** – classic draft model that shares the target embedding/head. Use it
  with ordinary causal checkpoints when a distilled EAGLE draft is available.
- **EAGLE3** – draft model with auxiliary hidden state inputs. The repository
  includes inference-only modules for Qwen3 and LLaMA style EAGLE3 checkpoints.

## Enabling speculative decoding

The relevant `launch_server` flags are:

| Flag | Description |
| --- | --- |
| `--speculative-algorithm {EAGLE,EAGLE3,NEXTN,STANDALONE}` | Selects the speculative worker. Use `EAGLE3` for multi-token Qwen3/LLaMA drafts. |
| `--speculative-draft-model-path` | HF repo or local path that contains the draft weights. Required for any speculative run. |
| `--speculative-draft-model-revision` | Optional branch/tag/commit for the draft weights. Useful when draft checkpoints live on a specific commit. |
| `--speculative-num-steps` | Number of draft iterations produced per target step. Higher values increase throughput but require more KV budget. |
| `--speculative-eagle-topk` | Branching factor per iteration (number of candidates sampled from the draft logits). |
| `--speculative-num-draft-tokens` | Maximum candidate tokens kept per request per iteration (limits draft KV allocation). |
| `--page-size` | Size of one KV page. Values >1 require paged attention (`--attention-backend=fa`). |
| `--speculative-accept-threshold-*` | Fine-grained acceptance heuristics when verifying. Usually keep defaults. |
| `--disable-overlap-schedule` | Recommended for EAGLE3 so the verify pass can reuse draft metadata immediately. |

Other options such as `--attention-backend`, `--dtype`, `--tp-size`, and the
KV-cache limits (`--max-prefill-tokens`, `--max-running-requests`) affect both
speculative and regular decoding.

## Quick start command

The following command launches an 8-core TPU VM that serves Qwen3-32B with an
AngelSlim EAGLE3 draft model:

```bash
python3 -u -m sgl_jax.launch_server \
  --model-path Qwen/Qwen3-32B \
  --trust-remote-code \
  --device=tpu \
  --mem-fraction-static=0.8 \
  --max-prefill-tokens=4096 \
  --max-running-requests=16 \
  --decode-log-interval=1 \
  --attention-backend=fa \
  --dtype=bfloat16 \
  --port 30000 --host 0.0.0.0 \
  --disable-overlap-schedule \
  --tp-size 4 \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path AngelSlim/Qwen3-32B_eagle3 \
  --speculative-draft-model-revision 67caf31f9062d7ab64872e0a111d499bc16cd205 \
  --page-size 64 \
  --speculative-eagle-topk 1 \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 4
```

Important notes for the example above:

- `--speculative-algorithm EAGLE3` switches the worker into the EAGLE3 code-path.
- `--speculative-draft-model-path AngelSlim/Qwen3-32B_eagle3` loads the HF
  repository that contains the distilled draft weights. The revision hash ensures reproducibility.
- `--page-size 64` plus `--attention-backend=fa` gives us paged FlashAttention
  metadata that matches the EAGLE3 draft allocation logic.
- `--disable-overlap-schedule` avoids interleaving verify passes with other work
  (EAGLE3 reuses mutable metadata buffers, so overlap adds back-pressure).
- You can still enable `--precompile-*` paddings once the configuration
  stabilizes; the example keeps them disabled while iterating.

## Supported target / draft combinations

| Target model | Draft model | Status |
| --- | --- | --- |
| `Qwen/Qwen3-32B` (or other Qwen3 checkpoints with identical architecture) | `AngelSlim/Qwen3-32B_eagle3` (commit `67caf31f9062d7ab64872e0a111d499bc16cd205`) | Fully validated on TPU with EAGLE3. |
| Custom LLaMA-derived checkpoint converted to the `LlamaEagleModel` format | Bring-your-own EAGLE3 draft weights that follow the `llama_eagle3.py` interfaces | Experimental: the repository provides the model wrapper, but you must supply compatible weights. |

At the time of writing we have not published public drafts for other models. If
you have your own EAGLE/EAGLE3 distilled checkpoint, point
`--speculative-draft-model-path` to that repo and ensure the tokenizer exactly
matches the target model.

## TODO

- **Performance Tuning:**

the performance optimization is needed, some jnp array operations need move to JIT functions

- **Non-greedy sample Kernel:**

some kernels need to be implemented when verify candidate draft tokens

## Operational tips

- **Monitor KV usage:** Each speculative step allocates
  `speculative_eagle_topk × speculative_num_steps` KV locations per request. Keep
  an eye on `--max-prefill-tokens` and server logs (`Prefill batch` lines) to
  avoid OOM.
- **Logging:** `--decode-log-interval=1` prints per-batch cache-miss statistics.
  Useful when tuning `speculative_num_steps`.
- **Fallback path:** If the draft model fails (for example HF download issues),
  the server will start in regular decode mode. Double-check logs before running
  benchmarks.
- **Validation:** Always run a short generation suite after changing speculative
  hyper-parameters. A bad draft/target pairing can degrade quality even when
  throughput looks good.

For more background on the EAGLE algorithm, refer to the original paper
[EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077).
