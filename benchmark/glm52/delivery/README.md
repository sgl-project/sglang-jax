# GLM-5.2 delivery scripts

This directory contains the scheduler-independent serve, benchmark, profile,
and accuracy-evaluation entry points for the GLM-5.2 FP8 delivery.

## Layout

```text
benchmark/glm52/delivery/
├── README.md
├── convert/
│   ├── convert_channelwise_fp8.py
│   └── run.sh
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
├── evalscope/
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

The public scripts under `convert/`, `serve/`, `benchmark/`, and `eval/` do
not require Falcon or another particular scheduler.

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

## Convert the BF16 checkpoint

External deployments that do not already have the static channel-wise
checkpoint can generate it from the complete GLM-5.2 BF16 checkpoint. This is
a CPU and shared-filesystem workflow; it does not require TPU devices. Run it
once from the repository root on a machine that can read and write the model
directories:

```bash
SOURCE_MODEL=/models/GLM-5.2 \
TARGET_MODEL=/models/GLM5.2-fp8-channel-wise \
WORKERS=16 \
  benchmark/glm52/delivery/convert/run.sh
```

The regular `python[tpu]` environment already provides NumPy and `ml_dtypes`.
For a conversion-only environment, install just those dependencies first:

```bash
python3 -m pip install numpy ml-dtypes
```

The converter applies FP8 E4M3FN per-output-channel weight quantization to the
attention, indexer, routed/shared expert, and dense MLP projection matrices.
Each `[out, in]` weight gets a FP32 `[out]` `weight_scale_inv` sidecar. Embedding,
normalization, router gate, `indexer.weights_proj`, and non-matrix tensors are
copied unchanged. The serve wrapper then uses the checked-in YAML to select
dynamic per-token FP8 activation for MoE and BF16 activation for other Linear
layers.

The default wrapper pins the validated GLM-5.2 source revision by requiring
282 shards, 59,044 converted tensors, and 118,629 final index keys. It converts
one bounded row chunk at a time, resumes valid shards from
`${TARGET_MODEL}.staging-v1`, validates checksums and index/header consistency,
and writes `${TARGET_MODEL}/_DOWNLOAD_COMPLETE` only after publication passes.
Rerunning the same command resumes an interrupted staging directory or exits
successfully after revalidating an already complete target. It refuses to
overwrite a non-empty incomplete target directory.

Keep source, staging, and target on a shared writable filesystem. Local scratch
under `/tmp/glm52-fp8-channelwise` only needs room for one output shard per
worker. The validated output is about 756.3 GB; because staging and final are
kept separately for atomic publication, reserve about 1.6 TB in addition to the
source checkpoint during conversion. After the published checkpoint passes
serve/eval validation, the staging directory can be removed according to the
deployment's retention policy. For an intentionally different compatible checkpoint revision, adjust
`EXPECTED_SHARDS`, `EXPECTED_SELECTED_TENSORS`, and
`EXPECTED_WEIGHT_MAP_COUNT` explicitly after reviewing the tensor policy.

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

For the stage-aware migrated backend, select `fused_rs` with the same
channel-wise checkpoint and launcher. It uses fused-RS for prefill-family
forward modes and keeps fused-v2 for decode and target verification:

```bash
GLM52_MOE_BACKEND=fused_rs WORLD=4 RANK=0 MASTER_ADDR=<rank-0-host> \
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

Set `MIN_SCORE` to turn the deterministic run into a CI/PR gate. The gate also
requires the requested number of examples to complete and rejects partial runs:

```bash
MIN_SCORE=0.90 \
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

### EvalScope agent smoke

The EvalScope path is separate from the classic `sgl-eval` path above. It
validates OpenAI-compatible structured tool calling and the native EvalScope
agent loop with the default `officeqa_pro` subset. A smoke run uses 16
samples, 16 concurrent agents (one per DP rank), and a 15-step cap. The
Falcon runner overrides the throughput-oriented round-robin scheduler with
`cache_aware` for this scenario so each multi-turn agent returns to the DP
rank holding its prefix. The API request timeout defaults to 3,600 seconds so
long-tail 4,096-token generations do not get recomputed by the OpenAI client
under 16-way concurrency. The 16-case smoke has observed individual agent
generations beyond 1,800 seconds. It records the complete
EvalScope work directory, including debug logs, predictions, reviews, reports,
per-sample `agent_trace` events, progress, and request performance metrics.

Initialize the pinned submodule and install it into an isolated evaluator
environment before invoking this entry point. The OfficeQA dataset directory
defaults to `/models/evalscope/officeqa`; Falcon mounts that path from a
writable, persistent GCS prefix so the approximately 460 MB corpus downloaded
on the first run is reused by subsequent runs.

```bash
git submodule update --init --recursive third_party/evalscope
python3 -m pip install -e third_party/evalscope

EVALSCOPE_DATASET_DIR=/models/evalscope/officeqa \
MODEL_PATH=/models/GLM5.2-fp8-channel-wise \
  benchmark/glm52/delivery/evalscope/run.sh officeqa
```

The runner first verifies that thinking output is separated into non-empty
`reasoning_content` and final `content` fields without literal thinking tags.
It then sends a named-function request with thinking requested and requires the
server's grammar path to suppress reasoning while returning a structured
`tool_calls` response. Finally, it repeats a unique long-prefix request and
requires the second OpenAI usage record to report cached tokens. After
evaluation it audits the server-log window for real agent cache hits and every
review row for a function-calling trace, structured
reasoning, no thinking-tag leakage, matching tool call/result IDs, and a
terminal submit or error event. Expected agent-quality outcomes such as
exhausting the step limit or a non-zero tool command exit are retained as
`quality_issues` rather than infrastructure failures. Accuracy is retained in
the report but is deliberately not a pass/fail criterion for the 16-sample
smoke. The submodule pins an unmodified upstream EvalScope revision, so bash
observations, model-request retries, the agent state machine, and scoring all
retain upstream behavior. Delivery-specific trace and cache checks run only
after evaluation and do not alter prompts, observations, retries, or scores.

## Internal Falcon manifests

`falcon/` contains a complete 4 × 3 matrix:

- deployment axis: block-wise/channel-wise × 8/16 physical chips;
- scenario axis: benchmark/profile/eval.

Each of the 12 manifests is standalone: it records source revision, topology,
model mount, quantization policy, runtime, and scenario parameters, then calls
only checked-in files through `falcon/runner.sh`. No manifest copies scripts,
configs, or patches from `/tmp`. The eval manifests intentionally select the
quick GSM8K 200-example scope; set `EVAL_SCOPE=full` for a full run.
