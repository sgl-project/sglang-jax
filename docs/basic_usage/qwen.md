# Qwen Models on SGL-JAX

Qwen is Alibaba's family of large language models optimized for diverse applications, now with full TPU support through SGL-JAX.

## Quick Start

Launch a Qwen-7B-Chat server on TPU:

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python3 -u -m sgl_jax.launch_server \
    --model-path Qwen/Qwen-7B-Chat \
    --trust-remote-code \
    --dist-init-addr=0.0.0.0:10011 \
    --nnodes=1 \
    --tp-size=4 \
    --device=tpu \
    --random-seed=3 \
    --node-rank=0 \
    --mem-fraction-static=0.2 \
    --max-prefill-tokens=8192 \
    --download-dir=/tmp \
    --jax-precompile-prefill-token-paddings 16384 \
    --dtype=bfloat16 \
    --skip-server-warmup \
    --attention-backend=fa
```

## Configuration Tips

### Memory Management
Set `--mem-fraction-static 0.2` for optimal TPU memory utilization. For larger models or batch sizes, adjust `--max-prefill-tokens` accordingly.

### TPU Optimization
- **FlashAttention Backend**: Use `--attention-backend fa` for TPU-optimized attention
- **JIT Compilation Cache**: Set `JAX_COMPILATION_CACHE_DIR` to accelerate startup
- **Tensor Parallelism**: Match `--tp-size` to your TPU core count (typically 1, 4 or 8)

### Advanced Features
#### Paged Attention
**Configuration:**
```bash
# Set page size (default: 1, recommended: 1, 4, 8, or 16)
--page-size 1

# For larger models or longer sequences, increase page size
--page-size 16  # Reduces page table overhead for long sequences
```

**Implementation Details:**
- **Page Size Impact**:
  - Size 1: Maximum flexibility, higher overhead
  - Size 16: Better memory locality, reduced page table size
- **Auto-tuning**: The kernel automatically selects optimal block sizes based on TPU version
- **Mixed Batching**: Supports simultaneous prefill and decode operations


## Benchmarking

### Throughput Testing
```bash
python3 -m sgl_jax.bench_serving \
    --backend sgl-jax \
    --dataset-name random \
    --num-prompts 100 \
    --random-input 512 \
    --random-output 128 \
    --max-concurrency 8 \
    --random-range-ratio 1 \
    --warmup-requests 0
```

### Accuracy Evaluation
Using EvalScope for model accuracy assessment:
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
| Model        | Dataset | Metric          | Subset | Num | Score | Cat.0   |
| :----------- | :------ | :-------------- | :----- | :-- | :---- | :------ |
| Qwen-7B-Chat | gsm8k   | AverageAccuracy | main   | 500 | 0.504 | default |

## Performance Tuning

### TPU Configuration Guide

| TPU Type | TP Size | mem-fraction-static | max-prefill-tokens |
|----------|---------|--------------------|--------------------|
| v6e-4    | 4       | 0.2                | 8192               |


## Troubleshooting

- **OOM Errors**: Reduce `--max-prefill-tokens` or `--mem-fraction-static`
- **Compilation Timeout**: Ensure `JAX_COMPILATION_CACHE_DIR` is properly configured
- **Low Throughput**: Verify `--tp-size` matches TPU configuration


## Additional Resources

- [Qwen Model Cards](https://huggingface.co/Qwen)
- [Jax Scaling Book](https://jax-ml.github.io/scaling-book/)
