# GLM-5.2 delivery scripts

This directory contains the scheduler-independent serve, benchmark, profile,
and accuracy-evaluation entry points for the GLM-5.2 FP8 delivery.

## Layout

```text
benchmark/glm52/delivery/
├── README.md
├── serve/
│   ├── common.sh
│   ├── blockwise_8chip.sh
│   ├── blockwise_16chip.sh
│   ├── channelwise_8chip.sh
│   └── channelwise_16chip.sh
├── benchmark/
│   ├── common.sh
│   ├── run_8chip.sh
│   └── run_16chip.sh
├── eval/
│   └── run.sh
├── validation/
│   └── validate_delivery_config.py
└── falcon/
    ├── runner.sh
    ├── blockwise/
    │   ├── 8chip/{benchmark,profile,eval}.yaml
    │   └── 16chip/{benchmark,profile,eval}.yaml
    └── channelwise/
        ├── 8chip/{benchmark,profile,eval}.yaml
        └── 16chip/{benchmark,profile,eval}.yaml
```

The public scripts under `serve/`, `benchmark/`, and `eval/` do not
require Falcon or another particular scheduler.

## Supported deployment matrix

| Serve entry point | Physical chips / JAX devices | TP/DP/EP | Checkpoint policy | Benchmark workload |
| --- | ---: | ---: | --- | --- |
| `serve/blockwise_8chip.sh` | 8 / 16 | 16/16/16 | legacy block-wise FP8 checkpoint config | C32, shared 128K prefix |
| `serve/channelwise_8chip.sh` | 8 / 16 | 16/16/16 | per-channel weights; MoE W8A8, other Linear W8A16 | C32, shared 128K prefix |
| `serve/blockwise_16chip.sh` | 16 / 32 | 32/32/32 | legacy block-wise FP8 checkpoint config | C64, 64 unique 128K prefixes |
| `serve/channelwise_16chip.sh` | 16 / 32 | 32/32/32 | per-channel weights; MoE W8A8, other Linear W8A16 | C64, 64 unique 128K prefixes |

The channel-wise entry points intentionally use
`fp8_glm52_static_per_channel_moe_w8a8_linear_w8a16.yaml`. They are not
aliases for the older all-W8A16 configuration.

## Environment

From the repository root, install the local project and its TPU optional
dependencies on every host:

```bash
python3 -m pip install -e './python[tpu]'
```

This command does use `python/pyproject.toml`: `./python` selects that local
Python project, `[tpu]` selects its `tpu` optional-dependency group, and
`-e` keeps the checkout editable. The equivalent form is:

```bash
cd python
python3 -m pip install -e '.[tpu]'
```

Every host must use the same checkout and be able to read the same complete
checkpoint directory.

## Serve

The 8-physical-chip layout uses two v7x-8 hosts. Run the following on both
hosts, changing `RANK` from 0 to 1:

```bash
WORLD=2 RANK=0 MASTER_ADDR=<rank-0-host> \
MODEL_PATH=/models/GLM5.2-fp8-channel-wise \
  benchmark/glm52/delivery/serve/channelwise_8chip.sh
```

The 16-physical-chip layout uses four v7x-8 hosts. Run it with ranks 0 through
3:

```bash
WORLD=4 RANK=0 MASTER_ADDR=<rank-0-host> \
MODEL_PATH=/models/GLM5.2-fp8-channel-wise \
  benchmark/glm52/delivery/serve/channelwise_16chip.sh
```

For the block-wise checkpoint, select the matching `blockwise_*.sh` entry
point and set `MODEL_PATH=/models/GLM-5.2-FP8`.

The launchers fail before model compilation if the host count, model index,
quantization policy, or exact fused-MoE v2 hot-shape tune entries do not match
the selected topology. They write a rank-local log under `/tmp` by default.
Set `GLM52_SERVER_LOG` to change the path or to an empty string to disable the
built-in `tee`.

## Benchmark and profile

Run a benchmark only on rank 0 after `/get_server_info` is healthy:

```bash
QUANTIZATION=channelwise \
  benchmark/glm52/delivery/benchmark/run_8chip.sh

QUANTIZATION=channelwise \
  benchmark/glm52/delivery/benchmark/run_16chip.sh
```

The workload is deliberately topology-specific: 8 chips always uses
C32/shared-prefix, while 16 chips always uses C64/64 unique prefixes. The
benchmark validates cache-hit length, per-DP placement, one measured extend
batch, full decode concurrency, output length, and server token capacity before
accepting the result.

To capture a lightweight stage profile, set trace limits before starting the
server and add `PROFILE_OUTPUT_DIR` to the same benchmark:

```bash
export SGLANG_PROFILE_MAX_HOSTS=1
export SGLANG_PROFILE_NUM_CHIPS_PER_TASK=1
export SGLANG_PROFILE_NUM_SPARSE_CORES_TO_TRACE=1
export SGLANG_PROFILE_NUM_SPARSE_CORE_TILES_TO_TRACE=1

PROFILE_OUTPUT_DIR=$PWD/artifacts/profiles/glm52-channelwise-8chip \
QUANTIZATION=channelwise \
  benchmark/glm52/delivery/benchmark/run_8chip.sh
```

The profile captures the validated cache-hit extend stage and three decode
steps.

## Eval

Pin the evaluator used for the reported scores:

```bash
SGL_EVAL_COMMIT=32fa49229575e433629c37379821b5a589a2e422
python3 -m pip install \
  "sgl-eval @ git+https://github.com/sgl-project/sgl-eval.git@$SGL_EVAL_COMMIT"
```

GSM8K defaults to a quick deterministic 200-example smoke evaluation:

```bash
MODEL_PATH=/models/GLM5.2-fp8-channel-wise \
  benchmark/glm52/delivery/eval/run.sh gsm8k
```

A full 1,319-example GSM8K evaluation must be requested explicitly:

```bash
EVAL_SCOPE=full \
MODEL_PATH=/models/GLM5.2-fp8-channel-wise \
  benchmark/glm52/delivery/eval/run.sh gsm8k
```

AIME26 remains a 30-example run:

```bash
MODEL_PATH=/models/GLM5.2-fp8-channel-wise \
  benchmark/glm52/delivery/eval/run.sh aime26
```

`NUM_EXAMPLES` can explicitly override either scope. `BASE_URL`,
`OUT_ROOT`, `NUM_THREADS`, and sampling variables are also configurable,
but changing sampling parameters makes results no longer directly comparable
with the documented baseline.

## Internal Falcon manifests

`falcon/` contains a complete 4 × 3 matrix:

- deployment axis: block-wise/channel-wise × 8/16 physical chips;
- scenario axis: benchmark/profile/eval.

Each of the 12 manifests is standalone: it records source revision, topology,
model mount, quantization policy, runtime, and scenario parameters, then calls
only checked-in files through `falcon/runner.sh`. No manifest copies scripts,
configs, or patches from `/tmp`. The eval manifests intentionally select the
quick GSM8K 200-example scope; set `EVAL_SCOPE=full` for a full run.
