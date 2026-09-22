# Offline AOT compiler IR export

`python -m sgl_jax.compile` provides an independent export path inspired by MaxText
`train_compile`: target topology → abstract model/inputs → shared serving forward →
lower → compile → save artifacts. It does not execute the forward function or start
the scheduler, tokenizer, or HTTP server.

Use this path to generate compiler artifacts for graph inspection and optimization.
TPU cross-compilation runs on a Linux CPU host with libtpu: it describes the target
device topology and constructs abstract model state and inputs, without requiring
physical TPU hardware, checkpoint weights, or request data.

## Supported scope

The export pipeline shares serving's forward function. The currently supported
model/backend combinations are:

| Model | Precision / workload | Attention | MoE | Constraints |
| --- | --- | --- | --- | --- |
| Qwen3 dense | BF16 decode, including logits and KV updates | Native | None | DP=1, EP=1, `head_dim=128`, no sliding window |
| MiMo-V2-Flash | Synthetic BF16 decode | FA / RPA v3 | `fused_v2` | Hybrid full/SWA attention, attention sinks, separate KV pools, EP=total devices |

- Model dimensions must satisfy the selected parallelism's divisibility constraints.
- Built-in tiny model: 2 layers, hidden size 512, intermediate size 1024,
  4 query heads, 2 KV heads, and vocabulary size 256.
- Supply a local `config.json` for a different Qwen3 configuration or MiMo-V2-Flash;
  no checkpoint is loaded.
- Compile-only TPU topologies: `v6e-1/4/8/16/32/64` and `v7x-8/16/32/64`.
  Following MaxText's topology/host-bounds approach, the suffix counts JAX-visible
  devices. v6e has one device per chip; v7x has two. The built-in model supports
  TP=1/2; TP=4 requires a model configuration whose KV heads and other partitioned
  dimensions are divisible by 4. TPU device count must match `--tp-size`.

Quantization, prefill, LoRA, MTP, multimodal models, and executable serialization
are not currently supported. These are synthetic BF16 graphs: the exporter does not
infer per-tensor checkpoint dtypes or reproduce checkpoint-specific post-load transforms.
In particular, this is not the official MiMo FP8 checkpoint graph.

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

## MiMo-V2-Flash with FA and fused MoE v2

Use a local copy of the official model's
[`config.json`](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash/blob/2f5a22fe08d2c3ecad8fcaf119d47c8fc848bcd1/config.json).
Its quantization metadata requires an
explicit `--bf16-model` override; both the original config and effective graph config
are saved. This changes the weight format, not the architecture or number of layers.

```bash
PYTHONPATH=python python -m sgl_jax.compile \
  --model-config /path/to/MiMo-V2-Flash/config.json --bf16-model \
  --target tpu --topology v6e-32 --tp-size 32 --dp-size 8 --ep-size 32 \
  --attention-backend fa --moe-backend fused_v2 \
  --batch-size 64 --context-length 1024 --kv-capacity 65536 --page-size 128 \
  --stage compiled --dump-llo --output /tmp/mimo-v6e32-ir
```

For a 64-device target, change to `--topology v6e-64 --tp-size 64 --dp-size 16
--ep-size 64`. Both examples use attention TP=4. As in serving, `--tp-size` counts
all devices; `--dp-size` partitions attention requests and KV pages. For offline
export, fused MoE v2 uses all devices for EP, so `ep_size=tp_size`, expert count must divide
evenly across EP, and batch size must be divisible by EP. KV capacity is global and
must be divisible by `dp_size * page_size`.

For v7x, select `--topology v7x-32` with the same 32-device command, or
`--topology v7x-64 --tp-size 64 --dp-size 16 --ep-size 64` for 64 devices.
The exporter targets TPU7x while running on the CPU host. These names map to
`TPU7x:2x2x4` (16 chips / 32 devices) and `TPU7x:2x4x4` (32 chips / 64 devices),
respectively. The smaller `v7x-8` and `v7x-16` targets use `2x2x1` and `2x2x2`.
To preserve attention TP=4, use `--dp-size 2 --tp-size 8 --ep-size 8` or
`--dp-size 4 --tp-size 16 --ep-size 16`. Target size alone does not establish that
the compiled graph will fit in the target's HBM or run correctly on real hardware.

Both KV pools have the specified token capacity; SWA uses its own page-table input.
This is not an automatic HBM-budget allocator.
FA cumulative lengths have `batch_size + dp_size` entries, and distribution has
`3 * dp_size` entries, matching serving's per-DP decode metadata. Values are dynamic:
the graph contains the backend's dynamic attention branches, not a constant-folded
all-decode distribution. No actual requests or checkpoint tensors are allocated.

For a reduced-depth smoke check, prepare a separate config with fewer layers and
truncate both `hybrid_layer_pattern` and `moe_layer_freq` to the same count. Preserve
the full configuration for a subsequent full-model export; reduced-depth results
must not be reported as full-model compilation.

## Artifacts and failure behavior

```text
manifest.json          Configuration, versions, source fingerprint, input signatures,
                       stage status, and artifact hashes
source_config.json     Original local model configuration, when supplied
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
Parameter shardings are bound separately from those mappings: nested dummy-loader
JITs can report replicated tracer outputs even when their compiled outputs are sharded.
The KV pool's `abstract=True` option reuses its normal shape calculations and creates
only abstract arrays.

When changing the shared forward, compare StableHLO against the baseline function
and compare logits/KV outputs with nonzero inputs. Compiler artifacts do not replace
physical TPU execution, numerical validation, or performance measurements.

Validation on 2026-09-22:

- CPU StableHLO-only, compiled HLO, and TP=2 exports succeeded.
- A one-off comparison with the original serving JIT produced identical StableHLO
  text. With batch size 2, nonzero random weights/KV, and valid decode inputs, all
  three logits/KV output arrays matched exactly.
- Existing `test_native_attention_paged_decode.py`: 17 tests and 40 subtests passed.
- The Qwen3 exporter at `476553fb` passed Linux CPU → v6e-1 cross-compilation
  with JAX/jaxlib 0.11.1, Flax 0.12.9, and libtpu 0.0.46.1. It produced
  114,044 bytes of StableHLO, 352,637 bytes of optimized HLO,
  and 1,893 recognized LLO pass snapshots.
- MiMo FA metadata shapes/shardings matched serving's decode metadata builder.
  Weight specs matched the existing model mappings; implicit replacement of the
  official config's FP8 metadata was rejected.
- CPU → TPU MiMo compilation with FA/RPA v3, fused MoE v2, synthetic BF16 weights,
  batch 64, context capacity 1024, page size 128, and KV capacity 65536 succeeded:

  | Model scope | Target | Attention DP / TP | EP | StableHLO / optimized HLO / LLO | LLO snapshots |
  | --- | --- | --- | --- | --- | --- |
  | First 2 layers, original dimensions | v6e-8 | 2 / 4 | 8 | Complete | 3477 |
  | Full 48 layers | v6e-32 | 8 / 4 | 32 | Complete | 3585 |
  | Full 48 layers | v6e-64 | 16 / 4 | 64 | Complete | 3513 |
  | First 2 layers, original dimensions | v7x-8 | 2 / 4 | 8 | Complete | 3383 |
  | First 2 layers, original dimensions | v7x-16 | 4 / 4 | 16 | Complete | 3523 |
  | Full 48 layers | v7x-32 | 8 / 4 | 32 | Complete | 3640 |
  | Full 48 layers | v7x-64 | 16 / 4 | 64 | Complete | 3636 |

  The full configuration retains 9 full-attention layers, 39 SWA layers, 47 MoE
  layers with 256 experts each, and original hidden/intermediate/vocabulary sizes.
  Compilation ran on a Linux CPU worker, without loading or executing the roughly
  617.7 GB of abstract BF16 weights. The native-attention regression suite and
  Qwen3 CPU TP=2 export also passed after adding the independent sharding binding.
- v7x validation used JAX/jaxlib 0.11.1, Flax 0.12.9, and libtpu 0.0.46.1.
  The four targets produced 94/98/101/101 nonempty final LLO bundles, respectively,
  including FA/RPA and fused MoE v2. Selected StableHLO, optimized HLO, source config,
  final LLO, and static memory-report files were read back and verified against
  manifest SHA256 hashes. The compiled repository Python source matched `c81ce8e97`;
  installation added only the generated `_version.py` file.
- Compile-only mesh/device-kind resolution and an all-reduce graph passed for
  v6e-4 and all four v7x targets. HLO partition counts matched the requested device
  counts, and hardware helpers restored CPU behavior outside the target mesh.
- Physical TPU execution, FP8 checkpoint fidelity, and runtime performance remain
  unverified. v6e-4 has only the collective check above, and v6e-16 has not been
  compilation validated.

References:

- [MaxText train_compile](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/trainers/pre_train/train_compile.py)
- [JAX AOT](https://docs.jax.dev/en/latest/aot.html)
- [XLA HLO dumps](https://openxla.org/xla/hlo_dumps)
