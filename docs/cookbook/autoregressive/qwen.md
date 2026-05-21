# Qwen-7B-Chat on SGL-JAX

> First-generation Qwen recipe. For Qwen3-8B / Qwen3-32B see [`qwen3.md`](qwen3.md).

## 1. Model Introduction

[**Qwen/Qwen-7B-Chat**](https://huggingface.co/Qwen/Qwen-7B-Chat) is Alibaba's first-generation Qwen 7B chat model — a 7B-parameter dense decoder LLM that fits comfortably on a single TPU v6e-4 host. SGL-JAX serves it with tensor parallelism for low-latency chat workloads.

**Key Features**:

- **Compact dense model**: 7B parameters, ~14 GB BF16 weights — comfortable fit on a single v6e-4 host.
- **Chat-tuned**: Instruction-following baseline for first-gen Qwen evaluations.
- **8K context** (extends to 32K with rope scaling on supported builds).

**Recommended Generation Parameters**: `temperature=0.7`, `top_p=0.95`, `max_tokens=512`.

**License**: see the [HuggingFace model card](https://huggingface.co/Qwen/Qwen-7B-Chat) for the authoritative license terms.

## 2. Deployment

### 2.1 Hardware Matrix

| TPU | Topology | Chips | `--tp-size` | Notes |
|---|---|---|---|---|
| **v6e-4** (minimum and recommended) | 2x2 | 4 | 4 | Single host; v6e is 1:1 chip↔device |

See [`../base/tpu-topology-reference.md`](../base/tpu-topology-reference.md) for the TPU generation reference.

### 2.2 Environment

Install per [`../../get_started/install.md`](../../get_started/install.md) and use [`../deployment/single-host-docker.md`](../deployment/single-host-docker.md) for the container setup. The required JAX TPU container image:

| Hardware Platform               | Docker Image                                                       |
|---|---|
| TPU v5e / v5p / v6e (Trillium)  | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| TPU v7x (Ironwood)              | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

Extra pip for accuracy benchmarking only:

```bash
pip install evalscope
```

### 2.3 Launch

#### Single-host (Docker) — TPU v6e-4

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server \
  --model-path Qwen/Qwen-7B-Chat \
  --trust-remote-code \
  --tp-size 4 \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.8 \
  --max-prefill-tokens 8192 \
  --download-dir /tmp \
  --dist-init-addr 0.0.0.0:10011 --nnodes 1 --node-rank 0 \
  --random-seed 3 \
  --skip-server-warmup \
  --host 0.0.0.0 --port 30000
```

> The `--dist-init-addr` / `--nnodes 1` / `--node-rank 0` trio is required even on a single-host launch — SGL-JAX always initializes JAX distributed.

### 2.4 Configuration Tips

**Memory Management:**
- `--mem-fraction-static 0.8` is conservative for 7B + dedicated KV cache. Raise to `0.9` for higher concurrency / batch sizes if the host is dedicated.
- `--max-prefill-tokens 8192` caps prefill batch tokens. Raise for longer prompts; lower if prefill-time OOM.

**Throughput Tuning:**
- `--page-size 16` (vs default `1`) reduces page-table overhead for longer sequences and can increase throughput at high concurrency. Default `1` is more flexible for low-concurrency mixed traffic.
- `--attention-backend fa` is the default (FlashAttention on Pallas) — no need to set explicitly unless overriding.

**Compilation Cache Hygiene:**
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` is mandatory — without it, first request blocks ~4 min while XLA/Pallas re-compiles.
- The cache keys on full kernel shape: changing `--page-size`, `--tp-size`, or context length invalidates cached entries.

For full flag definitions and defaults see [`../base/launch-flags-reference.md`](../base/launch-flags-reference.md).

## 3. Invocation

### 3.1 Basic Chat Completion

```bash
curl -X POST http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen-7B-Chat",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Python OpenAI client equivalent:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="Qwen-7B-Chat",
    messages=[{"role": "user", "content": "Hello"}],
)
print(resp.choices[0].message.content)
```

> Qwen-7B-Chat is a first-generation chat model without hybrid reasoning or native tool-calling formats. For reasoning / tool-call workloads use [Qwen3](qwen3.md) or a later Qwen series.

## 4. Benchmark

> Benchmark data below is a snapshot pinned to the `Tested build` listed in each Test Environment; not refreshed on every release.

### 4.1 Accuracy — GSM8K

**Test Environment**

| Field | Value |
|---|---|
| Hardware | TPU v6e-4 (single host, 4 chips) |
| Model | Qwen/Qwen-7B-Chat (BF16) |
| Tensor Parallelism | 4 |
| Tested build | _Pending_ (run pre-dates pin convention) |

**Deployment Command** — same as [§2.3 Single-host](#single-host-docker--tpu-v6e-4).

**Benchmark Command**

```bash
evalscope eval \
  --model Qwen-7B-Chat \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 8 \
  --limit 500
```

**Test Results**

| Model | Dataset | Metric | Subset | Num | Score |
|:---|:---|:---|:---|:---|:---|
| Qwen-7B-Chat | gsm8k | AverageAccuracy | main | 500 | 0.504 |

### 4.2 Speed — single workload (low-concurrency latency baseline)

**Test Environment** — same as §4.1.

**Deployment Command** — same as [§2.3 Single-host](#single-host-docker--tpu-v6e-4).

**Benchmark Command**

```bash
python -m sgl_jax.bench_serving \
  --backend sgl-jax \
  --dataset-name random \
  --num-prompts 100 \
  --random-input 512 \
  --random-output 128 \
  --max-concurrency 8 \
  --random-range-ratio 1 \
  --warmup-requests 0
```

**Test Results** — _Pending. Run the command above and PR back the full `============ Serving Benchmark Result ============` block from `bench_serving`._

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| OOM at startup | Weights + KV cache exceed budget | Lower `--max-prefill-tokens` to 4096, or `--mem-fraction-static` to 0.75. Verify `--tp-size 4` matches v6e-4 chip count. |
| First request takes ~4 min | JIT cache empty | Persist `JAX_COMPILATION_CACHE_DIR` across restarts (host volume mount in Docker). |
| Low throughput at high concurrency | `--page-size 1` overhead dominates | Raise `--page-size` to 16 or 32 for long-sequence high-concurrency workloads. |

## Additional Resources

- [Qwen Model Cards](https://huggingface.co/Qwen)
- [`qwen3.md`](qwen3.md) — newer Qwen3 8B/32B recipe with framework comparison numbers.
- [JAX Scaling Book](https://jax-ml.github.io/scaling-book/)
- [`../base/launch-flags-reference.md`](../base/launch-flags-reference.md)
- [`../troubleshooting.md`](../troubleshooting.md) — cross-recipe generic issues.
