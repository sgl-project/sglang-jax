# Global JIT Compile

SGL-JAX runs model forward and sampling through JAX JIT by default. The goal is to keep the hot path compiled while controlling shape variation with precompile buckets and runtime padding.

For the broader executor data flow, see [Model Executor](../architecture/04-model-executor.md).

## Runtime Flow

`ModelRunner.init_model_fns()` builds the compiled callables:

| Callable | Purpose |
|---|---|
| `jitted_run_model` | Runs the model forward pass with donated `memory_pools` and static model state structure. |
| `jitted_sampler` | Runs sampling with the sampler state structure and top-p/top-k/min-p sort behavior as static inputs. |
| `jitted_compute_logprobs` | Computes selected log probabilities under JIT. |

The model and sampler are reconstructed inside the JIT boundary with `nnx.merge`. The mutable runtime arrays are carried through `memory_pools`, and model outputs return pool updates that `ModelRunner` writes back after the compiled call.

## Precompile and Padding

Shape misses can trigger expensive XLA compilation during serving. SGL-JAX reduces those misses by precompiling common prefill and decode shapes before the scheduler loop starts, then padding runtime batches into the same bucket scheme.

There are two bucket families:

| Bucket family | Flag | Used for |
|---|---|---|
| Token buckets | `--precompile-token-paddings` | Prefill/extend shapes such as `input_ids`, `positions`, and output cache locations. |
| Batch-size buckets | `--precompile-bs-paddings` | Decode shapes such as request indices and sequence lengths. |

Prefill padding uses a fixed batch size derived from the configured serving limits and pads token-like fields to the selected token bucket. Decode padding normally pads both batch size and token count to the selected batch-size bucket.

## User Knobs

| Flag | Effect |
|---|---|
| `--precompile-token-paddings` | Override the prefill token bucket list. |
| `--precompile-bs-paddings` | Override the decode batch-size bucket list. |
| `--disable-precompile` | Skip startup precompilation. Runtime JIT compilation can still happen when new shapes appear. |
| `--max-running-requests` | Changes the maximum active batch size and therefore the useful bucket range. |
| `--chunked-prefill-size` | Changes the preferred prefill chunk size and therefore the token bucket pressure. |

Choose bucket lists with the workload shape distribution in mind. Too few buckets can cause excessive padding or runtime compilation. Too many buckets increase startup time and compiled executable footprint.

## Logprob Path

`return_logprob` is supported on the JIT path, but it can select a heavier compiled variant because logprob metadata changes the work returned by the model and sampler. For latency-sensitive serving, benchmark the exact logprob configuration rather than assuming it has the same profile as plain generation.

## Implementation Entry Points

- `python/sgl_jax/srt/model_executor/model_runner.py`: compiled model, sampler, and logprob callables.
- `python/sgl_jax/srt/server_args.py`: precompile bucket flags and `--disable-precompile`.
- `docs/architecture/04-model-executor.md`: executor-level architecture and precompile flow.
