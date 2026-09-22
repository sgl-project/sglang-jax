# AOT IR export PoC

`python -m sgl_jax.compile` provides an independent export path inspired by MaxText
`train_compile`: target topology → abstract model/inputs → shared serving forward →
lower → compile → save artifacts. It does not execute the forward function or start
the scheduler, tokenizer, or HTTP server.

## Supported scope

- Qwen3 dense, BF16, complete decode forward, including logits and KV updates.
- Native attention, DP=1; model dimensions must satisfy TP divisibility constraints.
- `head_dim=128`, using the serving `MHATokenToKVPool` layout.
- Built-in tiny model: 2 layers, hidden size 512, intermediate size 1024,
  4 query heads, 2 KV heads, and vocabulary size 256.
- An optional local Qwen3 `config.json`; no checkpoint is loaded.
- TPU topology mappings from MaxText: `v6e-1` and `v6e-4`. The built-in model supports
  TP=1/2; TP=4 requires a model configuration whose KV heads and other partitioned
  dimensions are divisible by 4. TPU device count must match `--tp-size`.

These are native-attention graphs and do not represent FlashAttention/RPA graphs
or performance. Quantization, prefill, MoE, LoRA, MTP, multimodal models, and
executable serialization are outside this PoC.

## CPU checks

From the repository root, with a compatible JAX 0.11.1 / Flax 0.12.9 environment:

```bash
PYTHONPATH=python python -m sgl_jax.compile \
  --target cpu --stage stablehlo --output /tmp/qwen3-stablehlo

PYTHONPATH=python python -m sgl_jax.compile \
  --target cpu --stage compiled --output /tmp/qwen3-cpu-hlo
```

CPU graphs use the CPU KV-update path and validate the export workflow. For TPU
optimization, use StableHLO/HLO/LLO generated with `--target tpu`.

## Cross-compile for TPU on a CPU host

This requires Linux and matching `jax[tpu]` / libtpu packages. A macOS CPU environment
cannot substitute for this check. The command sets `JAX_PLATFORMS=cpu` and describes
the target with compile-only TPU devices; no physical TPU is allocated.

```bash
PYTHONPATH=python python -m sgl_jax.compile \
  --target tpu --topology v6e-1 --tp-size 1 \
  --batch-size 1 --context-length 32 \
  --kv-capacity 128 --page-size 16 \
  --stage compiled --dump-llo --output /tmp/qwen3-tpu-ir
```

Add `--model-config /path/to/config.json` to supply a model configuration.
Compiler options can be passed with repeated `--compiler-option NAME=JSON_VALUE`
arguments, for example `--compiler-option xla_tpu_enable_log_recorder=true`.
Available options depend on the libtpu version.

`--batch-size` is the number of requests in one decode step, with one new token per
request. `--context-length` sets the page-aligned cache-location capacity per request;
sequence lengths and positions remain dynamic inputs. `--kv-capacity` must cover
every request's page-aligned context. Capacity, page size, shapes, and TP are recorded
in the manifest. `--stage` selects the compilation stage, not prefill versus decode;
decode is currently fixed.

The output directory must be new or empty so that stale dumps cannot count as new
artifacts.

## Artifacts and failure behavior

```text
manifest.json          Configuration, versions, source fingerprint, input signatures,
                       stage status, and artifact hashes
stablehlo.mlir         Saved immediately after lowering
optimized_hlo.txt      HLO after compilation for the selected backend
xla_dump/             Backend HLO/proto, buffer assignment, memory reports, etc.
llo/                  Raw libtpu dumps when LLO export is requested
error.txt             Python traceback on failure
```

Persistent compilation caching is disabled to ensure that code generation runs.
Dump flags are configured before importing JAX. Existing dump flags in the environment
are rejected so that artifacts cannot silently go to another directory.

If backend compilation fails, any generated StableHLO is retained, the manifest is
marked `failed`, and the process exits nonzero. Requested LLO output is also required:
successful compilation without nonempty LLO snapshots is an export failure.

libtpu 0.0.46.1 writes LLO pass snapshots as `*-original.txt` and `*-post-*.txt`.
The `llo/` directory retains intermediate and late-pass artifacts; auxiliary memory
reports alone do not count as LLO. Prefer a local output directory and archive the
results before uploading, rather than writing thousands of small files directly to
object storage.

## Implementation and validation

`model_forward.py` shares the forward/JIT configuration with serving, including
donation and backend state preparation. `aot_inputs.py` traces the existing model and
dummy weight loader inside `nnx.eval_shape`, reusing the real weight mappings. Weights
remain dynamic graph inputs; dummy zeros are not embedded as model constants.
The KV pool's `abstract=True` option reuses its normal shape calculations and creates
only abstract arrays.

When changing the shared forward, compare StableHLO against the baseline function
and compare logits/KV outputs with nonzero inputs. Compiler artifacts do not replace
physical TPU execution, numerical validation, or performance measurements.

PoC validation on 2026-09-22:

- CPU StableHLO-only, compiled HLO, and TP=2 exports succeeded.
- A one-off comparison with the original serving JIT produced identical StableHLO
  text. With batch size 2, nonzero random weights/KV, and valid decode inputs, all
  three logits/KV output arrays matched exactly.
- Existing `test_native_attention_paged_decode.py`: 17 tests and 40 subtests passed.
- Linux CPU → v6e-1 cross-compilation with JAX/jaxlib 0.11.1, Flax 0.12.9, and
  libtpu 0.0.46.1 produced 114,044 bytes of StableHLO, 352,637 bytes of optimized HLO,
  and 1,893 recognized LLO pass snapshots.
- Physical TPU execution, larger model configurations, and v6e-4 remain unverified.
  These results cover only the tiny model and native attention described above.

References:

- [MaxText train_compile](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/trainers/pre_train/train_compile.py)
- [JAX AOT](https://docs.jax.dev/en/latest/aot.html)
- [XLA HLO dumps](https://openxla.org/xla/hlo_dumps)
