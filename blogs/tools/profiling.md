# TPU Profiling with XProf

## 环境依赖

### 安装 libtpu 0.0.38.dev (nightly)

libtpu 是 JAX 在 TPU 上运行的底层库，profiling 功能依赖较新版本。

```bash
# 从 Google Cloud Storage 安装 nightly 版本（根据 Python 版本选择对应 wheel）
# Python 3.11
pip install https://storage.googleapis.com/libtpu-nightly-releases/wheels/libtpu/libtpu-0.0.38.dev20260318+nightly-cp311-cp311-manylinux_2_31_x86_64.whl

# Python 3.12
pip install https://storage.googleapis.com/libtpu-nightly-releases/wheels/libtpu/libtpu-0.0.38.dev20260318+nightly-cp312-cp312-manylinux_2_31_x86_64.whl

# Python 3.13
pip install https://storage.googleapis.com/libtpu-nightly-releases/wheels/libtpu/libtpu-0.0.38.dev20260318+nightly-cp313-cp313-manylinux_2_31_x86_64.whl
```

所有可用版本索引：https://storage.googleapis.com/jax-releases/libtpu_releases.html

验证安装：

```bash
pip show libtpu
# Name: libtpu
# Version: 0.0.38.dev20260318+nightly
```

## Custom Call Profiling

参考文档：[xprof custom_call_profiling](https://github.com/openxla/xprof/blob/master/docs/custom_call_profiling.md)

### XLA Flags

要在 Trace Viewer 中看到 custom call（如 Pallas/Mosaic kernel）的详细性能数据，需要设置以下两个 XLA flag：

| Flag | 作用 |
|------|------|
| `--xla_enable_custom_call_region_trace=true` | 启用 custom call 区域追踪 |
| `--xla_xprof_register_llo_debug_info=true` | 注册 LLO 调试信息，展示硬件资源利用率 |

### 使用方式

通过 `LIBTPU_INIT_ARGS` 环境变量传入：

```bash
LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true" \
  python your_jax_workload.py
```

### LLO Utilization

启用后，Trace Viewer 中每个 TPU core 会出现 **LLO utilization** 行，可视化 custom call 执行期间的硬件资源使用情况。这对定位 Pallas kernel 的性能瓶颈非常有用。

### 注意事项

- 这些 flag 会增大 profile 数据量并可能轻微影响运行性能，仅在调试优化时启用
- 如果启用 flag 后未看到 LLO utilization 行，需确认编译器后端支持对应 custom call 实现的 LLO debug info 注册

## Profiling Fused MoE Kernel

`benchmark/moe/bench_fused_moe.py` 支持 `--profile` 参数，可以将 kernel 执行的 profiling trace dump 到磁盘，供 TensorBoard 分析。

### 硬件环境

- 单机 4 卡 TPU v7（每 chip 2 cores，共 8 cores）
- EP（Expert Parallelism）= 8

### Ring-1T-FP8 模型参数

从 [inclusionAI/Ring-1T-FP8](https://huggingface.co/inclusionAI/Ring-1T-FP8) config 中读取：

| 参数 | 值 |
|------|-----|
| `num_experts` | 256 |
| `moe_intermediate_size` | 2048 |
| `hidden_size` | 5120 |
| `num_experts_per_tok` (top_k) | 8 |

### 运行命令

```bash
# 结合 XLA custom call profiling flags 和 bench_fused_moe --profile
LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true" \
python -m benchmark.moe.bench_fused_moe \
    --profile \
    --profile-dir ./profile_ring1t_moe \
    --num-experts 256 \
    --top-k 8 \
    --hidden-size 5120 \
    --intermediate-size 2048 \
    --iters 3
```

### 查看 Profiling 结果

```bash
# 使用 TensorBoard 加载 trace
tensorboard --logdir=./profile_ring1t_moe

# 或使用 Perfetto UI（https://ui.perfetto.dev/）直接打开 trace 文件
```

Trace Viewer 中可以看到：
- 每个 TPU core 的 kernel 执行时间线
- 启用 XLA flags 后的 **LLO utilization** 行（Pallas kernel 硬件利用率）
- `StepTraceAnnotation` 标记的每次迭代边界
