#!/usr/bin/env bash
# Run all scatter strategy benchmarks on a single pod.
# Usage: run_all_scatter_bench.sh [NUM_TOKENS...]
# Example: run_all_scatter_bench.sh 512 2048 8192

set -euo pipefail
cd /root/sglang-jax

NUM_EXPERTS=384
TOP_K=8
HIDDEN_SIZE=6144
INTERMEDIATE_SIZE=2048
ITERS=5
TOKENS="${@:-512 2048 8192}"

echo "=== BASELINE ==="
python -m benchmark.moe.bench_fused_moe \
  --num-experts $NUM_EXPERTS --top-k $TOP_K \
  --hidden-size $HIDDEN_SIZE --intermediate-size $INTERMEDIATE_SIZE \
  --num-tokens $TOKENS --iters $ITERS

echo ""
echo "=== VMEM PERMUTE SCATTER ==="
FUSED_MOE_BENCHMARK_USE_VMEM_PERMUTE_SCATTER=1 python -m benchmark.moe.bench_fused_moe \
  --num-experts $NUM_EXPERTS --top-k $TOP_K \
  --hidden-size $HIDDEN_SIZE --intermediate-size $INTERMEDIATE_SIZE \
  --num-tokens $TOKENS --iters $ITERS

echo ""
echo "=== OVERLAP SCATTER ==="
FUSED_MOE_BENCHMARK_USE_OVERLAP_SCATTER=1 python -m benchmark.moe.bench_fused_moe \
  --num-experts $NUM_EXPERTS --top-k $TOP_K \
  --hidden-size $HIDDEN_SIZE --intermediate-size $INTERMEDIATE_SIZE \
  --num-tokens $TOKENS --iters $ITERS

echo ""
echo "=== ALL DONE ==="
