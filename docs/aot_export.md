# Offline AOT compiler IR export

Use `python -m sgl_jax.compile` to save StableHLO, optimized HLO, and TPU LLO for
graph inspection and optimization. TPU compilation runs on a Linux CPU host with
libtpu, without physical TPUs, checkpoint weights, request data, or a running
inference server.

## Prepare the environment

From the repository root, install the Python package with TPU dependencies in a
Python 3.12 environment:

```bash
python -m pip install -e "python[tpu]"
PYTHONPATH=python python -m sgl_jax.compile --help
```

Use compatible JAX/jaxlib and libtpu versions. The examples below were checked with
JAX/jaxlib 0.11.1, Flax 0.12.9, and libtpu 0.0.46.1. The compilation target defaults
to `tpu`. The command sets `JAX_PLATFORMS=cpu` for the host and constructs a
compile-only TPU mesh from the requested topology. Model tracing, kernel selection,
and compilation use that TPU target; running on a CPU host does not select CPU
kernels.

For CPU-target exports, install `python[cpu]` instead and use the CPU command below.
TPU cross-compilation requires Linux; installing CPU JAX on macOS does not provide
the TPU compiler.

## Export TPU IR

This command uses the built-in tiny Qwen3 configuration and writes all three IR
stages to a new directory:

```bash
PYTHONPATH=python python -m sgl_jax.compile \
  --target tpu --topology v6e-1 --tp-size 1 \
  --batch-size 1 --context-length 32 --kv-capacity 128 --page-size 16 \
  --stage compiled --dump-llo --output /tmp/qwen3-tpu-ir
```

Choose a new or empty `--output` directory for every run. The command prints a JSON
summary with `status`, `output`, and `stages` when export finishes.

Choose the amount of compilation to perform:

| Arguments | Output |
| --- | --- |
| `--stage stablehlo` | StableHLO after lowering |
| `--stage compiled` | StableHLO, optimized HLO, and XLA dumps |
| `--stage compiled --dump-llo` | All of the above plus TPU LLO |

For StableHLO only, replace `--stage compiled --dump-llo` in the command with
`--stage stablehlo`. Keep `--target tpu` when generating IR for TPU optimization.

To explicitly generate CPU IR for a local smoke check:

```bash
PYTHONPATH=python python -m sgl_jax.compile \
  --target cpu --stage compiled --output /tmp/qwen3-cpu-ir
```

Only the explicit `--target cpu` command above selects CPU kernel paths. It does
not validate TPU kernels or produce TPU LLO. Omit the TPU topology options and
`--dump-llo` for this command. For TPU graph analysis, use the default TPU target
and inspect `options.target`, `target_topology`, and `custom_calls` in the manifest.

## Use a model configuration

Pass `--model-config /path/to/config.json` to select a local model configuration.
Its `architectures` field selects the model through the serving registry and
loader; checkpoint tensors are not read. Choose backends, parallelism, and cache
capacities as you would for serving:

```bash
PYTHONPATH=python python -m sgl_jax.compile \
  --model-config /path/to/config.json \
  --target tpu --topology v6e-4 --tp-size 4 \
  --attention-backend fa --batch-size 4 \
  --context-length 128 --kv-capacity 512 --page-size 128 \
  --stage compiled --dump-llo --output /tmp/model-ir
```

The exporter currently constructs synthetic BF16 parameters. If the config
contains quantization metadata, add `--bf16-model` only when you want a BF16
variant: it removes that metadata from the effective config, without reading or
converting checkpoint tensors. Quantized weight and scale construction is not
yet wired into this abstract loader; this is independent of the compilation
target. An FP8 graph needs its quantized parameter structure and kernel path, so
BF16 IR cannot represent its compute or memory requirements. The original config
is saved as `source_config.json`; the effective config and synthetic weight format
are recorded in `manifest.json`.

## Choose a TPU topology and parallelism

`--topology` accepts `v6e-1`, `v6e-4`, `v6e-8`, `v6e-16`, `v6e-32`, `v6e-64`,
`v7x-8`, `v7x-16`, `v7x-32`, and `v7x-64` as presets. The suffix counts JAX-visible
devices: v6e has one device per chip, and v7x has two. For example, `v7x-32`
describes 16 chips with 32 devices.

The topology presets passed to libtpu are:

| `--topology` | libtpu topology | Chips | Logical hosts |
| --- | --- | --- | --- |
| `v6e-1` | `v6e:1x1` | 1 | 1 |
| `v6e-4` | `v6e:2x2` | 4 | 1 |
| `v6e-8` | `v6e:2x4` | 8 | 2 |
| `v6e-16` | `v6e:4x4` | 16 | 4 |
| `v6e-32` | `v6e:4x8` | 32 | 8 |
| `v6e-64` | `v6e:8x8` | 64 | 16 |
| `v7x-8` | `TPU7x:2x2x1` | 4 | 1 |
| `v7x-16` | `TPU7x:2x2x2` | 8 | 2 |
| `v7x-32` | `TPU7x:2x2x4` | 16 | 4 |
| `v7x-64` | `TPU7x:2x4x4` | 32 | 8 |

To specify a topology directly, replace `--topology` with `--topology-name` and
`--host-bounds X Y Z`. For example, the explicit equivalent of `--topology v7x-8` is:

```bash
PYTHONPATH=python python -m sgl_jax.compile \
  --target tpu --topology-name TPU7x:2x2x1 --host-bounds 2 2 1 --tp-size 8 \
  --dp-size 2 --batch-size 2 \
  --stage compiled --dump-llo --output /tmp/custom-topology-ir
```

The name is passed directly to libtpu, so its supported topologies do not need an
exporter preset. `--topology` and `--topology-name` are mutually exclusive.
`--host-bounds` specifies positive chip counts along the three physical axes per
host; it is required with `--topology-name` and can also override a preset's bounds.
For example, `--topology v6e-8 --host-bounds 1 1 1 --tp-size 8` describes eight
logical hosts with one chip each. libtpu validates the topology and host layout.

The default chips-per-host bounds are `(2, 2, 1)` for every preset except `v6e-1`,
which uses `(1, 1, 1)`. TPU targets use one slice and no wraparound. These hosts
describe the target topology; the export itself runs in one CPU process. The
logical mesh axis order is `(data, tensor)` with shape `(dp-size, tp-size / dp-size)`,
using JAX's topology-aware mapping with physical-axis splitting allowed. Check `mesh`, `mesh_device_ids`
(flattened in mesh axis order), and `target_topology` in the manifest when comparing
collectives: device count alone does not identify the physical mapping.

- `--tp-size`: total device count; it must match the devices returned by libtpu
  (the topology suffix when using a preset).
- `--dp-size`: attention data parallelism. Attention TP is `tp-size / dp-size`.
- `--ep-size`: expert parallelism. With `--moe-backend fused_v2`, set it equal
  to `--tp-size`.

For fused MoE v2, both expert count and input token count must be divisible by EP.
The input count is batch size for decode, or batch size times verification width
for target verify.

## Set the workload shape

`--workload` selects the forward to export. The default `decode` has one new token
per request. `--stage` independently selects how far compilation proceeds.

| `--workload` | Model inputs and outputs |
| --- | --- |
| `decode` | One token per request, target model |
| `prefill` | A token chunk shared by the requests on each DP rank, target model |
| `mtp-draft` | One token and one hidden-state row per request, one MTP block |
| `mtp-draft-extend` | A token/hidden-state block per request plus accepted lengths |
| `target-verify` | A candidate-token block per request; logits and hidden states for every row |

| Argument | Meaning |
| --- | --- |
| `--batch-size` | Global request count for this forward, divisible by DP |
| `--chunked-prefill-size` | Prefill token budget **per DP rank**, shared by that rank's requests; a positive multiple of page size |
| `--num-tokens` | Global prefill token shape; defaults to `chunked-prefill-size * dp-size` |
| `--draft-token-num` | Verify/draft-extend tokens per request, including the seed; at least 2 |
| `--mtp-layer-idx` | MTP weight-set index, starting at 0 |
| `--context-length` | Total KV context capacity per request, including the current token block; rounded up to a page boundary |
| `--kv-capacity` | Global KV token capacity, excluding padding; applied to each full/SWA pool |
| `--page-size` | Tokens per KV page |
| `--recurrent-capacity` | Valid recurrent-state slots for linear attention; defaults to batch size |

For prefill, select the chunk budget independently of the request count:

```bash
PYTHONPATH=python python -m sgl_jax.compile \
  --model-config /path/to/config.json \
  --target tpu --topology v6e-4 --tp-size 4 --attention-backend fa \
  --workload prefill --batch-size 4 --chunked-prefill-size 4096 \
  --context-length 8192 --kv-capacity 32768 --page-size 128 \
  --stage compiled --dump-llo --output /tmp/prefill-ir
```

With DP=1, these four requests share 4096 input tokens. With DP=2, each rank has
two request slots and a 4096-token budget, so the default global input shape is
8192 tokens. Individual extend lengths, prefix lengths, and last-token logits
indices remain dynamic inputs; requests need not contribute equal token counts.
The context capacity includes the cached prefix and current chunk, so it can be
larger than the chunk budget.

Use `--num-tokens` to export a smaller prefill bucket. It selects the exact global
compiled token shape, must be divisible by DP, and must lie between `batch-size`
and both `chunked-prefill-size * dp-size` and `batch-size * context-length`.
Choose the matching serving token bucket when comparing IR; this command does
not simulate scheduler packing or silently round to another bucket. The manifest
records the original options and resolved global/per-DP request and token shapes.

For draft/verify exports, use the target model config and `--attention-backend fa`.
`--draft-token-num` includes the seed token; for example, batch 16 with width 4
has 64 input positions. These are NEXTN causal-chain forwards (`topk=1`); sampling
and token acceptance happen outside the exported graph. Export each desired MTP
weight set separately with `--mtp-layer-idx`.

For example, `--workload target-verify --batch-size 32 --draft-token-num 8` has
256 input tokens. At DP=4, each rank has eight requests and 64 input tokens.
Both request count and verification width are retained in the input metadata;
another workload with the same total token count can have a different graph.

For linear attention, `--recurrent-capacity` must cover the batch and be divisible
by DP. Recurrent states default to FP32 and convolution states to BF16, following
serving. Override them with `SGLANG_JAX_RECURRENT_STATE_DTYPE` and
`SGLANG_JAX_CONV_STATE_DTYPE` (`float32`, `bfloat16`, or `float16`).

Sequence lengths, token IDs, positions, and page mappings remain abstract runtime
inputs. You do not need to supply a dataset or prompt file.

Set `kv-capacity` to at least
`batch-size * ceil(context-length / page-size) * page-size`, and make it divisible
by `dp-size * page-size`. Batch size must also be divisible by DP. For example,
batch 64 with context capacity 1024 needs at least 65536 KV slots. KV capacity is
specified directly; the tool does not choose it from an HBM budget.

## Pass compiler options

Append repeated `--compiler-option NAME=JSON_VALUE` arguments to an export command,
for example:

```text
--compiler-option xla_tpu_enable_log_recorder=true
```

Values must be JSON scalars, such as `true`, `4`, or a quoted string. Option names
must start with `xla_`; their availability depends on the compiler version. Dump
options are managed by the tool. Remove existing dump flags from `XLA_FLAGS` and
`LIBTPU_INIT_ARGS` before running it, and invoke the CLI in a fresh Python process.

To match serving decode's SparseCore gather workaround, prefix the command with
`SGLANG_JAX_DECODE_DISABLE_SC_GATHER_OFFLOAD=1`. The exporter reuses serving's
per-forward options: both gather-offload passes are disabled for `decode` and
`mtp-draft`, while verify and draft-extend retain their defaults. Serving uses this
workaround through its `SGLANG_JAX_AOT_DISPATCH` path; the exporter already compiles
explicitly and needs no dispatch setting. `SGLANG_JAX_ENABLE_KERNEL_LOG_RECORDER=1`
and attention-backend compiler options are also honored. Explicit
`--compiler-option` values take precedence; inspect `compiler_options` in the
manifest for the effective values.

Leave `PALLAS_INTERPRET` unset when inspecting TPU kernels. Explicit interpret
settings still select their debug implementations in kernels that honor them.

## Find and inspect the output

| Path under `--output` | What to inspect |
| --- | --- |
| `manifest.json` | Export status, actual forward mode, workload, effective config, input/output signatures, versions, and file hashes |
| `source_config.json` | Original model config, when supplied |
| `stablehlo.mlir` | Graph after lowering |
| `optimized_hlo.txt` | HLO compiled for the selected target |
| `xla_dump/` | Backend HLO/proto, buffer assignment, and static memory reports |
| `llo/` | Raw TPU compiler dumps when `--dump-llo` is set |
| `error.txt` | Traceback if export fails after initialization |

Start with `manifest.json` and check for `status: complete` and the requested stages.
For LLO inspection, look for `*-final_bundles.txt`; intermediate pass snapshots are
also retained. Static memory reports describe compiler allocations, not measured
runtime memory peaks. The output contains compiler IR, not a serialized executable.

Inspect `custom_calls` in the manifest to check which kernels actually reached
optimized HLO. Each entry records a custom-call target, HLO instruction, and source
`op_name`. For example, an EPMoE GMM v2 export should contain a `tpu_custom_call`
whose instruction name or `op_name` includes `gmm_v2-`; finding `gmm` elsewhere in
HLO does not establish that the TPU kernel was compiled. Compile-only artifacts
support graph, layout, and static-allocation analysis; measure performance on the
target hardware.

Prefer a local output directory, then archive it for transfer:

```bash
tar -czf /tmp/model-ir.tar.gz -C /tmp model-ir
```

## Troubleshooting

- **Output directory is not empty:** select a new directory for the next run.
- **Device count or divisibility error:** check topology, TP/DP/EP, model dimensions,
  batch size, and KV capacity together.
- **Model configuration is rejected:** check `architectures` in the local JSON,
  the installed serving model implementation, and the backend/parallelism settings
  reported in `error.txt`. No model name needs to be added to the exporter.
- **Backend compilation fails:** inspect `error.txt` and `manifest.json`. StableHLO
  is retained if lowering completed, even when later stages fail.
- **Requested LLO is missing:** check the libtpu version and recorded dump flags in
  the manifest. Compilation caching is disabled so code generation runs on each
  invocation; missing requested LLO makes the export fail.
