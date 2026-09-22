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
JAX/jaxlib 0.11.1, Flax 0.12.9, and libtpu 0.0.46.1. The command selects a CPU host
backend automatically; `--target tpu` selects the compilation target.

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

To try the workflow with a CPU target:

```bash
PYTHONPATH=python python -m sgl_jax.compile \
  --target cpu --stage compiled --output /tmp/qwen3-cpu-ir
```

CPU-target IR uses CPU kernel paths. Omit `--topology` and `--dump-llo` for this
command.

## Use a model configuration

Add `--model-config /path/to/config.json` to select a local model configuration.
The file describes the architecture; checkpoint tensors are not read. Without this
argument, the tool uses a tiny two-layer Qwen3 model suitable for the first export.

For a Qwen3 configuration, use native attention with `--dp-size 1`, `--ep-size 1`,
and `head_dim=128`. Partitioned model dimensions, including KV heads, must be
divisible by `--tp-size`. The built-in tiny model can use TP=1 or TP=2.

### MiMo-V2-Flash with FA and fused MoE v2

Save the model's
[`config.json`](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash/blob/2f5a22fe08d2c3ecad8fcaf119d47c8fc848bcd1/config.json)
locally, then run:

```bash
PYTHONPATH=python python -m sgl_jax.compile \
  --model-config /path/to/MiMo-V2-Flash/config.json --bf16-model \
  --target tpu --topology v7x-32 --tp-size 32 --dp-size 8 --ep-size 32 \
  --attention-backend fa --moe-backend fused_v2 \
  --batch-size 64 --context-length 1024 --kv-capacity 65536 --page-size 128 \
  --stage compiled --dump-llo --output /tmp/mimo-v7x32-ir
```

The official config includes quantization metadata. `--bf16-model` explicitly
replaces that metadata for a synthetic BF16 export; it preserves the architecture
and layer count, but does not reproduce FP8 checkpoint computation. The original
config is saved as `source_config.json`, and the effective config is recorded in
`manifest.json`.

To compile fewer layers, make a separate config and change `num_hidden_layers`,
`hybrid_layer_pattern`, and `moe_layer_freq` together. Use the original config to
export the full model.

### MiMo MTP draft and target verify

Use the same MiMo-V2-Flash config for both commands. To export one MTP draft
forward with an abstract hidden-state input:

```bash
PYTHONPATH=python python -m sgl_jax.compile \
  --model-config /path/to/MiMo-V2-Flash/config.json --bf16-model \
  --workload mtp-draft --mtp-layer-idx 0 \
  --target tpu --topology v7x-8 --tp-size 8 --dp-size 2 \
  --attention-backend fa \
  --batch-size 16 --context-length 1024 --kv-capacity 16384 --page-size 128 \
  --stage compiled --dump-llo --output /tmp/mimo-mtp-draft-ir
```

Each MTP runner contains one SWA attention block and a dense MLP. Omit
`--moe-backend` and keep `--ep-size 1` for draft workloads. Embedding and LM-head
parameters have the same layouts as the arrays shared from the target in serving.
`--mtp-layer-idx` selects a weight set; export each desired MTP layer separately.
Without checkpoint loading, this index does not verify that the weight set exists.

To export the draft-extend forward after target verification, replace
`--workload mtp-draft` with `--workload mtp-draft-extend --draft-token-num 4` and
choose a new output directory. This adds the hidden-state block and accepted-length
inputs used to update draft KV and select the next logits.

To export the full target verifying eight tokens per request:

```bash
PYTHONPATH=python python -m sgl_jax.compile \
  --model-config /path/to/MiMo-V2-Flash/config.json --bf16-model \
  --workload target-verify --draft-token-num 8 \
  --target tpu --topology v7x-32 --tp-size 32 --dp-size 8 --ep-size 32 \
  --attention-backend fa --moe-backend fused_v2 \
  --batch-size 16 --context-length 1024 --kv-capacity 16384 --page-size 128 \
  --stage compiled --dump-llo --output /tmp/mimo-target-verify-ir
```

`--batch-size` counts requests. This verify example has `16 * 8 = 128` input
tokens and returns logits and hidden states for all 128 positions. The width
includes the seed token, so eight positions represent the seed plus seven
candidates. These exports use the NEXTN causal-chain model forwards (`topk=1`);
sampling, token acceptance, and the scheduler run outside these graphs.

### Kimi Linear with KDA and MLA

Save the model's
[`config.json`](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/e1df551a447157d4658b573f9a695d57658590e9/config.json)
locally, then run:

```bash
PYTHONPATH=python python -m sgl_jax.compile \
  --model-config /path/to/Kimi-Linear-48B-A3B-Instruct/config.json \
  --target tpu --topology v7x-16 --tp-size 16 --dp-size 4 --ep-size 16 \
  --attention-backend fa --moe-backend epmoe \
  --batch-size 32 --context-length 1024 --kv-capacity 32768 --page-size 256 \
  --recurrent-capacity 128 \
  --stage compiled --dump-llo --output /tmp/kimi-v7x16-ir
```

The model configuration selects KDA for linear-attention layers and absorbed MLA
for full-attention layers. This command uses the model's EPMoE implementation,
with `ep-size=tp-size`. `--recurrent-capacity` sets the global number of valid
recurrent-state slots, independently of the token-based `--kv-capacity`. It defaults
to batch size and must cover the batch and be divisible by DP. Each DP rank also
gets a dummy state slot.

Recurrent states use FP32 and convolution states use BF16 by default, following
serving. `SGLANG_JAX_RECURRENT_STATE_DTYPE` and `SGLANG_JAX_CONV_STATE_DTYPE` select
`float32`, `bfloat16`, or `float16`; inspect the manifest's input signatures for the
resulting shapes and dtypes.

For a four-layer export containing both KDA and MLA, set `num_hidden_layers=4` in
a separate config and retain only IDs 1 through 4 in `linear_attn_config.kda_layers`
and `linear_attn_config.full_attn_layers`. These lists use one-based layer IDs.

## Choose a TPU topology and parallelism

`--topology` accepts `v6e-1`, `v6e-4`, `v6e-8`, `v6e-16`, `v6e-32`, `v6e-64`,
`v7x-8`, `v7x-16`, `v7x-32`, and `v7x-64`. The suffix counts JAX-visible devices:
v6e has one device per chip, and v7x has two. For example, `v7x-32` describes
16 chips with 32 devices.

- `--tp-size`: total device count; it must match the topology suffix.
- `--dp-size`: attention data parallelism. Attention TP is `tp-size / dp-size`.
- `--ep-size`: expert parallelism. For the MiMo `fused_v2` command, set it equal
  to the total device count.

To keep attention TP=4 in the MiMo example, replace its parallelism arguments with
one of these combinations and select a new output directory:

| `--topology` | `--tp-size` | `--dp-size` | `--ep-size` |
| --- | --- | --- | --- |
| `v6e-8` or `v7x-8` | 8 | 2 | 8 |
| `v6e-16` or `v7x-16` | 16 | 4 | 16 |
| `v6e-32` or `v7x-32` | 32 | 8 | 32 |
| `v6e-64` or `v7x-64` | 64 | 16 | 64 |

For fused MoE v2, both expert count and input token count must be divisible by EP.
The input count is batch size for decode, or batch size times verification width
for target verify.

## Set the workload shape

`--workload` selects the forward to export. The default `decode` has one new token
per request. `--stage` independently selects how far compilation proceeds.

| `--workload` | Model inputs and outputs |
| --- | --- |
| `decode` | One token per request, target model |
| `mtp-draft` | One token and one hidden-state row per request, one MTP block |
| `mtp-draft-extend` | A token/hidden-state block per request plus accepted lengths |
| `target-verify` | A candidate-token block per request; logits and hidden states for every row |

| Argument | Meaning |
| --- | --- |
| `--batch-size` | Number of requests |
| `--draft-token-num` | Verify/draft-extend tokens per request, including the seed; at least 2 |
| `--mtp-layer-idx` | MTP weight-set index, starting at 0 |
| `--context-length` | Total KV context capacity per request, including the current token block; rounded up to a page boundary |
| `--kv-capacity` | Global KV token capacity, excluding padding; applied to each full/SWA pool |
| `--page-size` | Tokens per KV page |
| `--recurrent-capacity` | Valid recurrent-state slots for linear attention; defaults to batch size |

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

Prefer a local output directory, then archive it for transfer:

```bash
tar -czf /tmp/mimo-v7x32-ir.tar.gz -C /tmp mimo-v7x32-ir
```

## Troubleshooting

- **Output directory is not empty:** select a new directory for the next run.
- **Device count or divisibility error:** check topology, TP/DP/EP, model dimensions,
  batch size, and KV capacity together.
- **Model configuration is rejected:** the current input builder selects Qwen3,
  MiMo-V2-Flash, or Kimi Linear. Additional architectures need their model/input
  construction wired into `aot_inputs.py` before they can be selected from the CLI.
- **Backend compilation fails:** inspect `error.txt` and `manifest.json`. StableHLO
  is retained if lowering completed, even when later stages fail.
- **Requested LLO is missing:** check the libtpu version and recorded dump flags in
  the manifest. Compilation caching is disabled so code generation runs on each
  invocation; missing requested LLO makes the export fail.
