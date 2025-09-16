# Megablox GMM Auto-Tuning System

This directory contains an auto-tuning system for optimizing tiling parameters in Megablox GMM operations used by MoE layers.

## Overview

The system automatically tunes the `tiling` parameter for GMM operations to achieve optimal performance on your specific hardware and model configuration. Instead of using hardcoded tile sizes, the system:

1. **Benchmarks** different tiling configurations for your typical workload shapes
2. **Caches** the optimal results for fast lookup
3. **Automatically applies** the best tiling parameters during inference

## Files

- `bench_megablox_gmm.py` - Basic benchmark for testing different GMM configurations
- `auto_tune_tiling.py` - Core auto-tuning system that finds optimal tiling parameters
- `../../../python/sgl_jax/srt/layers/gmm/tiling_manager.py` - Runtime tiling parameter manager
- `../../../python/sgl_jax/srt/auto_tune_startup.py` - Startup integration for auto-tuning

## Quick Start

### 1. Run Auto-Tuning for Your Model

For a Llama-7B style model:
```bash
python python/sgl_jax/srt/auto_tune_startup.py --model llama-7b
```

For a Mixtral-8x7B style model:
```bash
python python/sgl_jax/srt/auto_tune_startup.py --model mixtral-8x7b
```

For custom configurations:
```bash
python python/sgl_jax/srt/auto_tune_startup.py --model llama-13b --batch-sizes 1 2 4 --seq-lengths 1024 2048 4096
```

### 2. Integration in Your Service

Add this to your service startup code:

```python
from sgl_jax.srt.auto_tune_startup import auto_tune_for_startup, initialize_tiling_manager

# Run auto-tuning on startup (only tunes missing configurations)
auto_tune_for_startup(model_name="llama-7b", verbose=True)

# Initialize the tiling manager
initialize_tiling_manager(verbose=True)

# Your MoE layers will now automatically use optimal tiling!
```

### 3. Manual Benchmarking

To run the basic benchmark:
```bash
python benchmark/kernels/megablox_gmm/bench_megablox_gmm.py
```

To run auto-tuning for specific shapes:
```bash
python benchmark/kernels/megablox_gmm/auto_tune_tiling.py
```

## How It Works

### 1. Auto-Tuning Process

The auto-tuner:
- Generates candidate tiling configurations (e.g., `(128, 256, 512)`, `(256, 512, 1024)`)
- Benchmarks each configuration with your actual workload shapes
- Finds the configuration with the lowest latency
- Caches the result for future use

### 2. Runtime Integration

The MoE layer now uses `get_optimal_tiling_for_gmm()` instead of hardcoded values:

```python
# Before (hardcoded):
default_tile_size = (512, 1024, 1024)
tiling = (min(default_tile_size[0], m), min(default_tile_size[1], k), min(default_tile_size[2], n))

# After (auto-tuned):
optimal_tiling = get_optimal_tiling_for_gmm(m, k, n, num_groups=self.num_experts)
tiling = (min(optimal_tiling[0], m), min(optimal_tiling[1], k), min(optimal_tiling[2], n))
```

### 3. Caching System

Results are cached in JSON files like:
```
tuning_cache/
├── m1024_k4096_n11008_g8.json
├── m2048_k4096_n11008_g8.json
└── ...
```

Each cache file contains:
```json
{
  "optimal_tiling": [128, 512, 1024],
  "best_time_ms": 0.85,
  "timestamp": 1704067200.0
}
```

## Command Line Options

### auto_tune_startup.py

```bash
python python/sgl_jax/srt/auto_tune_startup.py [OPTIONS]

Options:
  --model MODEL              Model name (llama-7b, mixtral-8x7b, etc.)
  --cache-dir DIR            Cache directory (default: tuning_cache)
  --force-retune            Force re-tuning even if cache exists
  --batch-sizes 1 2 4 8     Batch sizes to tune for
  --seq-lengths 512 1024    Sequence lengths to tune for
  --quiet                   Suppress verbose output
```

## Supported Models

Pre-configured models:
- `llama-7b`, `llama-13b`, `llama-30b`, `llama-65b`
- `mixtral-8x7b`, `mixtral-8x22b`
- `deepseek-v3`

You can also use custom configurations by specifying shapes manually.

## Performance Tips

1. **Run tuning on your target hardware** - Optimal tiling varies by TPU/GPU type
2. **Tune for your actual workload** - Use realistic batch sizes and sequence lengths
3. **Cache is persistent** - Tuning only needs to be done once per model/hardware combination
4. **Force retune after hardware changes** - Use `--force-retune` if you change hardware

## Example Output

```
SGL-JAX MoE Auto-Tuner
========================================
Model: llama-7b
Cache directory: tuning_cache

Tuning 1/32: m=512, k=4096, n=11008, groups=8
  Testing 24 tiling candidates...
   1/24: (64, 64, 64) -> 2.35 ms
   2/24: (64, 64, 128) -> 2.12 ms
   ...
   18/24: (256, 512, 1024) -> 0.85 ms
   ...
  Best tiling: (256, 512, 1024) (0.85 ms)

...

Tuning completed in 45.2 seconds
Tuned 32 configurations
Initialized tiling manager with 32 cached configurations
```

Now your MoE layers will automatically use the optimal tiling parameters!
