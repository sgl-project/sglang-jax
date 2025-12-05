# Fused MoE vs EP MoE Benchmark

Comprehensive layer-level benchmark comparing **FusedEPMoE** (Pallas TPU kernel) vs **EPMoE** (GMM kernel) implementations.

## Features

- ✅ **Layer-level testing** with synthetic weights
- ✅ **Controlled token distribution** scenarios (random, balanced, imbalanced)
- ✅ **Load imbalance metrics** (max expert load / average load)
- ✅ **Distributed configurations** (EP, TP) with multi-node support
- ✅ **JAX profiling** support via `jax.profiler`
- ✅ **HuggingFace config loading** or manual configuration
- ✅ **Multiple output formats** (CSV for plotting, Markdown for viewing)

## Quick Start

### 1. Simple Test (Manual Config)

```bash
python benchmark/fused_moe/bench_fused_vs_epmoe.py \
    --num-experts 8 \
    --num-experts-per-tok 2 \
    --hidden-size 1024 \
    --intermediate-size 4096 \
    --num-tokens 512 \
    --scenarios random
```

### 2. Using HuggingFace Model Config

```bash
python benchmark/fused_moe/bench_fused_vs_epmoe.py \
    --model-path Qwen/Qwen2.5-MoE-A2.7B \
    --ep-size 8 \
    --num-tokens 1024 2048 4096 \
    --scenarios random balanced imbalanced
```

### 3. With Profiling

```bash
python benchmark/fused_moe/bench_fused_vs_epmoe.py \
    --model-path Qwen/Qwen2.5-MoE-A2.7B \
    --ep-size 8 \
    --num-tokens 1024 2048 \
    --scenarios random balanced imbalanced \
    --profile \
    --profile-dir ./profiles/qwen_benchmark
```

### 4. Multi-Node Setup

**Node 0:**
```bash
python benchmark/fused_moe/bench_fused_vs_epmoe.py \
    --model-path Qwen/Qwen2.5-MoE-A2.7B \
    --dist-init-addr 10.0.0.1:12345 \
    --nnodes 2 \
    --node-rank 0 \
    --ep-size 16
```

**Node 1:**
```bash
python benchmark/fused_moe/bench_fused_vs_epmoe.py \
    --model-path Qwen/Qwen2.5-MoE-A2.7B \
    --dist-init-addr 10.0.0.1:12345 \
    --nnodes 2 \
    --node-rank 1 \
    --ep-size 16
```

## Command-Line Arguments

### Model Configuration

Choose **one** of:

- `--model-path PATH`: Load config from HuggingFace model (downloads or loads from local)
- `--manual-config`: Use manual configuration (requires additional args below)

**Manual configuration options** (required if `--manual-config` is used):
- `--num-experts INT`: Number of experts
- `--num-experts-per-tok INT`: Top-k value
- `--hidden-size INT`: Hidden dimension
- `--intermediate-size INT`: Intermediate dimension
- `--activation {silu,gelu,swigluoai}`: Activation function (default: silu)

### Distributed Configuration

- `--ep-size INT`: Expert parallel size (default: 1)
- `--tp-size INT`: Total number of devices to use (default: 1)
  - The actual tensor parallel size is computed as `tp_size // ep_size` in MoE layers
- `--dist-init-addr ADDR`: Distributed init address (e.g., `10.0.0.1:12345`)
- `--nnodes INT`: Number of nodes (default: 1)
- `--node-rank INT`: Current node rank (default: 0)

### Benchmark Parameters

- `--num-tokens INT [INT ...]`: List of token counts to test (default: 512 1024 2048)
- `--scenarios {random,balanced,imbalanced} [...]`: Scenarios to test (default: all)
- `--imbalance-factor FLOAT`: Target imbalance for "imbalanced" scenario (default: 3.0)
  - **Definition**: `max_expert_load / avg_expert_load`
  - Balanced scenario always targets ~1.0 (perfect balance)
  - Imbalanced scenario uses this factor (e.g., 3.0 = busiest expert gets 3x average)
- `--warmup-iters INT`: Warmup iterations (default: 1, only need one for JAX JIT)
- `--benchmark-iters INT`: Benchmark iterations (default: 10)

### Profiling

- `--profile`: Enable JAX profiler
- `--profile-dir PATH`: Profile output directory (default: ./profiles)

### Output

- `--output-format {csv,markdown,both}`: Output format (default: both)
- `--output-file PATH`: Output file base path (default: ./benchmark_results)
- `--verbose`: Enable verbose logging

## Scenarios

### 1. Random

Uniform random router logits from N(0, 1). Results in natural imbalance ~1.2-1.5x.

**Use case**: Realistic scenario simulating natural token distribution.

### 2. Balanced

Engineered logits using round-robin assignment to ensure equal expert distribution. Target imbalance: ~1.0x (perfect balance).

**Use case**: Best-case scenario for MoE performance.

### 3. Imbalanced

Exponential distribution favoring first few experts. Controlled by `--imbalance-factor` (default: 3.0).

**Use case**: Worst-case scenario to test robustness under load imbalance.

## Output Formats

### CSV Format

```csv
implementation,scenario,num_tokens,ep_size,tp_size,num_experts,num_experts_per_tok,
latency_mean_ms,latency_std_ms,latency_p50_ms,latency_p95_ms,latency_p99_ms,
max_load,min_load,avg_load,max_imbalance,throughput_tok_per_sec
fused,random,1024,8,1,60,8,2.3456,0.1234,2.3000,2.5000,2.6000,150,130,140.5,1.07,436543.21
epmoe,random,1024,8,1,60,8,3.1234,0.2345,3.1000,3.4000,3.5000,150,130,140.5,1.07,327891.23
```

**Columns:**
- `implementation`: "fused" or "epmoe"
- `scenario`: "random", "balanced", or "imbalanced"
- `num_tokens`: Number of tokens in the test
- `ep_size`: Expert parallel size
- `tp_size`: Total number of devices (actual tensor parallel = tp_size // ep_size)
- `num_experts`, `num_experts_per_tok`: MoE configuration
- `latency_*_ms`: Latency statistics in milliseconds
- `max_load`, `min_load`, `avg_load`: Expert load distribution
- `max_imbalance`: Maximum imbalance ratio (max_load / avg_load)
- `throughput_tok_per_sec`: Throughput in tokens per second

### Markdown Format

```markdown
# MoE Benchmark Results
**Configuration:** 60 experts, top-8, EP=8, TP=1

## Scenario: balanced, Tokens: 1024

| Metric | Fused MoE | EP MoE | Speedup |
|--------|-----------|--------|---------|
| Mean Latency (ms) | 2.3456 | 3.1234 | 1.33x |
| P95 Latency (ms) | 2.5678 | 3.4567 | - |
| Throughput (tok/s) | 436.5 | 327.8 | - |
| Max Imbalance | 1.05x | 1.05x | - |

## Scenario: imbalanced, Tokens: 1024

| Metric | Fused MoE | EP MoE | Speedup |
|--------|-----------|--------|---------|
| Mean Latency (ms) | 2.8901 | 3.6789 | 1.27x |
| Max Imbalance | 3.12x | 3.12x | - |
```

## Imbalance Metrics

The benchmark reports **imbalance factor** defined as:

```
imbalance_factor = max_expert_load / avg_expert_load
```

**Examples:**
- `1.0x`: Perfect balance (all experts receive equal tokens)
- `1.5x`: Mild imbalance (busiest expert gets 50% more than average)
- `3.0x`: High imbalance (busiest expert gets 3x more than average)

**Reported metrics:**
- `max_load`: Maximum tokens assigned to any single expert
- `min_load`: Minimum tokens assigned to any single expert
- `avg_load`: Average tokens per expert
- `max_imbalance`: Imbalance factor (max_load / avg_load)

## Profiling

When `--profile` is enabled, JAX profiler traces are saved for each scenario/implementation combination.

### View Traces

Trace files can be loaded and visualized from:

1. **Perfetto UI**: https://ui.perfetto.dev/ (any browser)
2. **Chrome Tracing**: chrome://tracing (Chrome browser only)

Open the trace file from `<profile-dir>/<scenario>_tokens<N>_<impl>/plugins/profile/*/trace.json.gz`

If browser cannot open trace file due to its large size, reduce `--num-tokens` or `--benchmark-iters` to generate smaller traces.

### View Traces with Tensorboard

```bash
tensorboard --logdir=<profile-dir>
# Open the displayed URL in browser
```

### View Traces with XProf

[XProf](https://github.com/openxla/xprof) includes a suite of tools for JAX, TensorFlow, and PyTorch/XLA.

```bash
# Install XProf (nightly version)
pip install xprof-nightly

# Without TensorBoard:
xprof --logdir=<profile-dir> --port=6006

# With TensorBoard:
tensorboard --logdir=<profile-dir>
```

## Implementation Details

### Weight Equivalence

The benchmark ensures mathematical equivalence between FusedEPMoE and EPMoE:

```python
# EPMoE format
wi_0: (num_experts, hidden_size, intermediate_size)  # gate projection
wi_1: (num_experts, hidden_size, intermediate_size)  # up projection
wo: (num_experts, intermediate_size, hidden_size)    # down projection

# FusedEPMoE format (transposed!)
w1: (num_experts, 2, intermediate_size, hidden_size)  # [gate, up] fused
w2: (num_experts, intermediate_size, hidden_size)     # down projection

# Transformation
w1[:, 0, :, :] = wi_0.transpose(0, 2, 1)  # gate
w1[:, 1, :, :] = wi_1.transpose(0, 2, 1)  # up
w2 = wo
```

### Router Logits

Both implementations receive the same router logits, but:
- **FusedEPMoE**: Handles top-k selection internally
- **EPMoE**: Requires explicit `TopK` module call first

## File Structure

```
benchmark/fused_moe/
├── bench_fused_vs_epmoe.py    # Main benchmark script
├── config_utils.py             # Configuration loading and validation
├── synthetic_data.py           # Synthetic data generation
├── benchmark_runner.py         # Core benchmark execution
├── output_formatter.py         # CSV and Markdown formatting
└── README.md                   # This file
```

## Troubleshooting

### Error: "num_experts must be divisible by ep_size"

Ensure `num_experts % ep_size == 0`. For example, if you have 60 experts, valid `ep_size` values are: 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60.

### Error: "tp_size exceeds device count"

`--tp-size` should equal the total number of devices you want to use. Check available devices with:

```python
import jax
print(f"Available devices: {jax.device_count()}")
```

Example: For 4 devices with EP=4, use `--tp-size 4 --ep-size 4` (tp_actual will be 1).

### Multi-node setup not working

- Ensure `--dist-init-addr` is accessible from all nodes
- Verify firewall rules allow communication on the specified port
- Check that `--nnodes` and `--node-rank` are correct for each node

## Benchmark Test Cases

### Case 1: Qwen3-Coder-30B-A3B-Instruct (4x TPU v6e)

**Configuration:**
- Model: `Qwen/Qwen3-Coder-30B-A3B-Instruct`
- Hardware: 4x TPU v6e chips
- Token counts: 1024, 2048, 4096, 8192, 16384
- Scenarios: random, balanced, imbalanced

**Note**: 4 chips cannot support `ep_size=8`. Recommended comparison: `ep_size=4, tp_size=4` (full EP, tp_actual=1) vs `ep_size=1, tp_size=4` (full TP, tp_actual=4).

**Test commands:**

```bash
# Test 1: 4 devices, EP=4, tp_actual=1 (Expert Parallelism)
python benchmark/fused_moe/bench_fused_vs_epmoe.py \
    --model-path Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --tp-size 4 \
    --ep-size 4 \
    --num-tokens 1024 2048 4096 8192 16384 \
    --scenarios random balanced imbalanced \
    --warmup-iters 1 \
    --benchmark-iters 10 \
    --output-file ./results/qwen3_ep4_tp1

# Test 2: 4 devices, EP=1, tp_actual=4 (Tensor Parallelism only)
python benchmark/fused_moe/bench_fused_vs_epmoe.py \
    --model-path Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --tp-size 4 \
    --ep-size 1 \
    --num-tokens 1024 2048 4096 8192 16384 \
    --scenarios random balanced imbalanced \
    --warmup-iters 1 \
    --benchmark-iters 10 \
    --output-file ./results/qwen3_ep1_tp4
```

### Case 2: Grok2 (32 chips, 8 machines)

**Configuration:**
- Model: Grok2
- Hardware: 32 chips across 8 machines (4 chips per machine)
- Token counts: 1024, 2048, 4096, 8192, 16384
- Scenarios: random, balanced, imbalanced

**Test commands:**

Run on each machine with different `--node-rank` (0-7):

```bash
# Test 1: 32 devices, EP=8, tp_actual=4 (Expert Parallelism)
# Machine 0 (rank 0):
python benchmark/fused_moe/bench_fused_vs_epmoe.py \
    --model-path /path/to/grok2 \
    --tp-size 32 \
    --ep-size 8 \
    --dist-init-addr <MASTER_IP>:12345 \
    --nnodes 8 \
    --node-rank 0 \
    --num-tokens 1024 2048 4096 8192 16384 \
    --scenarios random balanced imbalanced \
    --warmup-iters 1 \
    --benchmark-iters 10 \
    --output-file ./results/grok2_ep8_tp4

# Machines 1-7: Same command but change --node-rank to 1, 2, ..., 7

# Test 2: 32 devices, EP=1, tp_actual=32 (Tensor Parallelism only)
# Machine 0 (rank 0):
python benchmark/fused_moe/bench_fused_vs_epmoe.py \
    --model-path /path/to/grok2 \
    --tp-size 32 \
    --ep-size 1 \
    --dist-init-addr <MASTER_IP>:12345 \
    --nnodes 8 \
    --node-rank 0 \
    --num-tokens 1024 2048 4096 8192 16384 \
    --scenarios random balanced imbalanced \
    --warmup-iters 1 \
    --benchmark-iters 10 \
    --output-file ./results/grok2_ep1_tp32

# Machines 1-7: Same command but change --node-rank to 1, 2, ..., 7
```

**Multi-machine launcher script:**

```bash
#!/bin/bash
# run_grok2_bench.sh
MASTER_IP="10.0.0.1"  # Replace with actual master IP
NODE_RANK=${1:-0}

python benchmark/fused_moe/bench_fused_vs_epmoe.py \
    --model-path /path/to/grok2 \
    --tp-size 32 \
    --ep-size 8 \
    --dist-init-addr ${MASTER_IP}:12345 \
    --nnodes 8 \
    --node-rank $NODE_RANK \
    --num-tokens 1024 2048 4096 8192 16384 \
    --scenarios random balanced imbalanced \
    --warmup-iters 1 \
    --benchmark-iters 10 \
    --output-file ./results/grok2_ep8_tp4

# Run on each machine: bash run_grok2_bench.sh 0, bash run_grok2_bench.sh 1, ...
```

## Contributing

To extend this benchmark:

1. **Add new scenarios**: Edit `generate_router_logits()` in `synthetic_data.py`
2. **Add new metrics**: Modify `BenchmarkResult` in `benchmark_runner.py`
3. **Change output format**: Edit `output_formatter.py`

## References

- FusedEPMoE implementation: `sgl_jax/srt/layers/fused_moe.py`
- EPMoE implementation: `sgl_jax/srt/layers/moe.py`
- Main benchmark pattern: `sgl_jax/bench_one_batch.py`
