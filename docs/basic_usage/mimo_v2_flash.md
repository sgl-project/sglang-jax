# MiMo-V2-Flash on SGL-JAX

MiMo-V2-Flash is Xiaomi's 256-expert MoE model with hybrid attention (full attention + sliding window attention), optimized for long-context reasoning tasks. SGL-JAX supports it on TPU v6e/v7x with FP8 quantization, tensor parallelism, and expert parallelism.

## Quick Start

### Multi-node (TPU v6e-16, 4 nodes)

Launch on each node with the appropriate `--node-rank` (0-3):

#### Fused MoE (Recommended)

Uses the fused Pallas kernel which combines expert computation and all-to-all communication into a single optimized operation:

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache uv run python -u -m sgl_jax.launch_server \
    --model-path XiaomiMiMo/MiMo-V2-Flash \
    --trust-remote-code \
    --tp-size 16 --dp-size 4 --ep-size 16 \
    --moe-backend fused \
    --nnodes 4 --node-rank $RANK \
    --dist-init-addr $MASTER_IP:30000 \
    --host 0.0.0.0 --port 30271 \
    --page-size 256 --context-length 262144 \
    --chunked-prefill-size 2048 \
    --dtype bfloat16 --mem-fraction-static 0.95 \
    --swa-full-tokens-ratio 0.2 --skip-server-warmup \
    --max-running-requests 128 \
    --attention-backend fa
```

#### EP MoE (Alternative)

Uses GMM-based expert-parallel dispatch with separate all-to-all communication:

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache uv run python -u -m sgl_jax.launch_server \
    --model-path XiaomiMiMo/MiMo-V2-Flash \
    --trust-remote-code \
    --tp-size 16 --dp-size 4 --ep-size 16 \
    --moe-backend epmoe \
    --nnodes 4 --node-rank $RANK \
    --dist-init-addr $MASTER_IP:30000 \
    --host 0.0.0.0 --port 30271 \
    --page-size 256 --context-length 262144 \
    --chunked-prefill-size 2048 \
    --dtype bfloat16 --mem-fraction-static 0.95 \
    --swa-full-tokens-ratio 0.2 --skip-server-warmup \
    --max-running-requests 128 \
    --attention-backend fa
```

Key flags:
- `--tp-size / --ep-size`: Match your total TPU chip count across all nodes
- `--dp-size`: Enables data parallelism for the attention path. The attention TP degree becomes `tp_size / dp_size` (e.g. `16 / 4 = 4`). MoE layers still run with full `ep_size`.
- `--moe-backend fused|epmoe`: `fused` uses an optimized Pallas kernel; `epmoe` uses GMM-based expert dispatch
- `--swa-full-tokens-ratio 0.2`: Allocates 20% of KV cache pool to full-attention layers, 80% to SWA layers
- `--page-size 256`: Recommended page size for SWA eviction efficiency

### Single-node (TPU v7x-8)

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache uv run python -u -m sgl_jax.launch_server \
    --model-path XiaomiMiMo/MiMo-V2-Flash \
    --trust-remote-code \
    --tp-size 8 --dp-size 2 --ep-size 8 \
    --moe-backend epmoe \
    --host 0.0.0.0 --port 30271 \
    --page-size 256 --context-length 262144 \
    --chunked-prefill-size 4096 \
    --dtype bfloat16 --mem-fraction-static 0.95 \
    --swa-full-tokens-ratio 0.25 --skip-server-warmup \
    --max-running-requests 128 \
    --attention-backend fa
```

> **MoE backend on single-host v7x**: at EP=8 (32 experts/device), `epmoe` consistently outperforms `fused` by ~18-26% on long-context throughput. See [MoE Backend Selection](#moe-backend-selection) below. The `fused` backend remains the recommendation for multi-host slices (EP≥16).

## Configuration Tips

### Memory Management
- **mem-fraction-static**: Use `0.9`-`0.95` for dedicated serving. MiMo-V2-Flash weights are ~20 GB/chip in FP8.
- **swa-full-tokens-ratio**: Controls the split between full-attention (9 layers) and SWA (39 layers) KV cache pools. `0.2` works well on v6e-16; `0.25` is slightly better on v7x-8.
- **max-running-requests**: Limit concurrent decoding requests to prevent OOM. `128` is a good starting point for v6e-16.

### Model-Specific Features
- **Hybrid Attention**: 9 full-attention layers + 39 sliding-window (window=128) layers. SWA layers use a separate KV cache pool with automatic eviction.
- **v_head_dim**: K head_dim=192, V head_dim=128. Handled transparently by the attention backend.
- **Attention Sink**: SWA layers use a phantom token in the softmax denominator for numerical stability.
- **FP8 Quantization**: Weights are natively quantized in FP8. No additional quantization flags needed.

## Benchmarking

### Throughput Testing
```bash
uv run python -m sgl_jax.bench_serving \
    --backend sgl-jax \
    --dataset-name random \
    --num-prompts 256 \
    --random-input 16384 \
    --random-output 1024 \
    --max-concurrency 64 \
    --random-range-ratio 1 \
    --warmup-requests 0 \
    --tokenizer XiaomiMiMo/MiMo-V2-Flash
```

### Accuracy Evaluation
Using EvalScope for GSM8K accuracy:
```bash
pip install evalscope==0.17.1

evalscope eval \
    --model XiaomiMiMo/MiMo-V2-Flash \
    --api-url http://127.0.0.1:30271/v1/chat/completions \
    --api-key EMPTY \
    --eval-type service \
    --datasets gsm8k \
    --eval-batch-size 32 \
    --generation-config '{"temperature": 0.8, "top_p": 0.95, "max_tokens": 32768}'
```

| Model | Dataset | Metric | Subset | Num | Score |
|:------|:--------|:-------|:-------|:----|:------|
| MiMo-V2-Flash | gsm8k | AverageAccuracy | main | 1319 | 0.9401 |

## Performance Tuning

### TPU Configuration Guide

| TPU Type | Nodes | TP Size | DP Size | EP Size | moe-backend | chunked-prefill-size | mem-fraction-static | max-running-requests |
|----------|-------|---------|---------|---------|-------------|----------------------|--------------------|-----------------------|
| v6e-16   | 4     | 16      | 4       | 16      | fused       | 2048                 | 0.95               | 128                   |
| v7x-8    | 1     | 8       | 2       | 8       | **epmoe**   | 4096                 | 0.95               | 128                   |

### MoE Backend Selection

The optimal `--moe-backend` is **scale-dependent**. On single-host v7x-8 (EP=8, 32 experts/device), `epmoe` outperforms `fused`; on larger slices (EP≥16), `fused` is recommended.

Measured on v7x-8 @ `b787fdef` (dp=1), 256 prompts × 16K input / 1K output, concurrency 64:

| moe-backend | chunked-prefill-size | swa-full-tokens-ratio | mem-fraction-static | Output tok/s | Median ITL |
|-------------|---------------------|-----------------------|--------------------|--------------|-----------|
| **epmoe**   | 4096                | 0.25                  | 0.95               | **480.04**   | 37.84 ms  |
| epmoe       | 2048                | 0.2                   | 0.90               | 467.32       | 36.90 ms  |
| fused       | 2048                | 0.2                   | 0.90               | 396.75       | 38.19 ms  |
| fused       | 4096                | 0.25                  | 0.95               | 382.32       | 42.45 ms  |

The fused MoE tuned config table covers the EP=8 shapes (server logs show `Using tuned block config` for the precompiled buckets), so the gap is not a tuner-coverage issue. See discussion in [#931](https://github.com/sgl-project/sglang-jax/pull/931).

### SWA Pool Tuning
- Monitor SWA pool usage via server logs: `swa token usage` and `full token usage`
- If SWA OOM occurs, reduce `--max-running-requests` or increase `--swa-full-tokens-ratio`
- For long-output workloads, lower `--swa-full-tokens-ratio` to give more capacity to SWA layers

## Troubleshooting

- **OOM Errors**: Reduce `--max-running-requests` or `--mem-fraction-static`. Check both SWA and full-attention pool usage in logs.
- **SWA Pool Exhaustion**: The SWA pool evicts tokens outside the sliding window automatically. If requests still fail, reduce concurrency.
- **Compilation Timeout**: Set `JAX_COMPILATION_CACHE_DIR` to persist JIT compilation cache across restarts.
- **Multi-node Connectivity**: Ensure `--dist-init-addr` points to the rank-0 node's IP and the port is accessible from all nodes.

## Additional Resources

- [MiMo-V2-Flash Model Card](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash)
- [SGL-JAX SWA Documentation](../basic_usage/features/)
