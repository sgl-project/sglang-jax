# Benchmark Progress Log

## Environment

- Platform: TPU v6e (2x2, 4 chips, single-host)
- Pod: `fp8-benchmark` on GKE cluster `tpuv6e-256-node`
- Node pool: `tpu-v6e-4-fp8`
- Docker image: `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1`
- Model config: MiMo-V2-Flash

## Test Matrix

- Projections: q_proj, k_proj(full), k_proj(SWA), v_proj(full), v_proj(SWA), o_proj
- Quantization: FP8 block-128 (weight only) vs BF16 baseline
- Tokens: 1, 128, 256, 512, 1024, 2048, 4096
- TP: 1, 2, 4 (k_proj full 跳过 TP=4, 768/4=192 不整除 128)
- Total: 17 configs × 7 tokens = 119 test points

## Run Log

### Run 1: q_proj [4096, 12288] column-parallel

- Start: 2026-04-02
- Command: `python -m benchmark.quantization.bench_quantized_linear --hidden-size 4096 --intermediate-size 12288 --tokens 1 128 256 512 1024 2048 4096 --weight-dtype fp8 --act-dtype none --block-size 128 --tp-size 1 2 4 --parallel-mode column --warmup 3 --tries 10`
- Status: **完成**
- Issues:
  1. TP>1 AxisType.Auto 错误 → 改用 `create_device_mesh` 创建 Explicit axes mesh 解决
  2. 所有 FP8 配置均报 "Couldn't find tuned sizes" → `tuned_block_sizes.py` 仅有 INT8 条目，FP8 on v6 无预调优
- 结论: FP8 block-128 在 TPU v6e 上性能远低于 BF16 (0.03x-0.73x)，根本原因是缺少 tuned block sizes

### Run 2: k_proj Full Attention [4096, 768] column-parallel

- Start:
- Command: `python -m benchmark.quantization.bench_quantized_linear --hidden-size 4096 --intermediate-size 768 --tokens 1 128 256 512 1024 2048 4096 --weight-dtype fp8 --act-dtype none --block-size 128 --tp-size 1 2 --parallel-mode column --warmup 3 --tries 10`
- Status:
- Issues:

### Run 3: k_proj SWA [4096, 1536] column-parallel

- Start:
- Command: `python -m benchmark.quantization.bench_quantized_linear --hidden-size 4096 --intermediate-size 1536 --tokens 1 128 256 512 1024 2048 4096 --weight-dtype fp8 --act-dtype none --block-size 128 --tp-size 1 2 4 --parallel-mode column --warmup 3 --tries 10`
- Status:
- Issues:

### Run 4: v_proj Full Attention [4096, 512] column-parallel

- Start:
- Command: `python -m benchmark.quantization.bench_quantized_linear --hidden-size 4096 --intermediate-size 512 --tokens 1 128 256 512 1024 2048 4096 --weight-dtype fp8 --act-dtype none --block-size 128 --tp-size 1 2 4 --parallel-mode column --warmup 3 --tries 10`
- Status:
- Issues:

### Run 5: v_proj SWA [4096, 1024] column-parallel

- Start:
- Command: `python -m benchmark.quantization.bench_quantized_linear --hidden-size 4096 --intermediate-size 1024 --tokens 1 128 256 512 1024 2048 4096 --weight-dtype fp8 --act-dtype none --block-size 128 --tp-size 1 2 4 --parallel-mode column --warmup 3 --tries 10`
- Status:
- Issues:

### Run 6: o_proj [8192, 4096] row-parallel

- Start:
- Command: `python -m benchmark.quantization.bench_quantized_linear --hidden-size 8192 --intermediate-size 4096 --tokens 1 128 256 512 1024 2048 4096 --weight-dtype fp8 --act-dtype none --block-size 128 --tp-size 1 2 4 --parallel-mode row --warmup 3 --tries 10`
- Status:
- Issues:

## Issues & Solutions

### 1. TP>1 AxisType.Auto 错误
- **现象**: `ValueError: PartitionSpec passed to normal cannot contain axis names that are of type Auto or Manual`
- **原因**: `Mesh(np.array(devices), ("tensor",))` 创建的是 AxisType.Auto，JAX 0.8.1 的 PartitionSpec 要求 AxisType.Explicit
- **解决**: 改用 `create_device_mesh(ici_parallelism=[1, tp_size], dcn_parallelism=[1, 1], devices=devices, mesh_axes=("data", "tensor"))` 创建 Explicit axes mesh

### 2. FP8 block-128 Pallas kernel 无预调优 block sizes
- **现象**: 所有 FP8 配置均报 `Couldn't find tuned sizes for the quantized matmul kernel with TunedKey(tpu_version=6, ..., w_q_dtype='float8_e4m3fn')`
- **原因**: `tuned_block_sizes.py` 中 TPU v6 仅有 INT8 的调优条目，没有 FP8 (float8_e4m3fn) 的条目
- **影响**: Pallas blockwise kernel 使用未调优的默认 block sizes，性能比 BF16 慢 3-30 倍
- **状态**: 已知限制，需要为 FP8 on v6 运行 autotuning 生成调优参数
