#!/bin/bash
# Diagnostic script for persistent compilation cache miss
# Usage: bash scripts/diag_cache_miss.sh [dp_size]
# Run on pod niu-v6e4-sleep

set -euo pipefail

DP_SIZE="${1:-1}"
MODEL="/models/meta-llama/Llama-3.1-8B-Instruct"
PORT=30000
LOG="/tmp/server_log_dp${DP_SIZE}.txt"

echo "=== Cache miss diagnostic: dp=$DP_SIZE ==="

# 1. Kill existing server
pkill -9 -f "sgl_jax.launch_server" || true
sleep 3

# 2. Clear all caches
rm -rf /tmp/jax_cache /tmp/xla_dump /tmp/jit_cache ~/.cache/jax /tmp/hlo_dump
mkdir -p /tmp/jax_cache /tmp/hlo_dump

# 3. Start server with full diagnostics
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
export JAX_DEBUG_LOG_MODULES=jax._src.compiler
export JAX_EXPLAIN_CACHE_MISSES=1
export SGLANG_JAX_DUMP_RUN_MODEL_HLO_DIR=/tmp/hlo_dump
export SGLANG_JAX_DUMP_RUN_MODEL_HLO_LIMIT=200

echo "Starting server with dp=$DP_SIZE ..."
python -m sgl_jax.launch_server \
  --model-path "$MODEL" \
  --port "$PORT" \
  --dp-size "$DP_SIZE" \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path "$MODEL" \
  --speculative-eagle-topk 1 \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 4 \
  --precompile-bs-paddings 1 4 8 16 \
  --precompile-token-paddings 256 1024 2048 4096 \
  2>&1 | tee "$LOG" &

SERVER_PID=$!

# 4. Wait for fired up
echo "Waiting for server to fire up..."
while ! grep -q "fired up" "$LOG" 2>/dev/null; do
  sleep 5
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "ERROR: Server process died"
    exit 1
  fi
done

FIRED_UP_LINE=$(grep -n "fired up" "$LOG" | head -1 | cut -d: -f1)
echo "Server fired up at log line $FIRED_UP_LINE"

# 5. Count cache entries after precompile
PRECOMPILE_COUNT=$(find /tmp/jax_cache -type f 2>/dev/null | wc -l)
echo "Persistent cache entries after precompile: $PRECOMPILE_COUNT"

# 6. Run evalscope
echo "Running evalscope..."
evalscope eval \
  --model "$MODEL" \
  --api-url "http://127.0.0.1:${PORT}/v1/chat/completions" \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --eval-batch-size 8 \
  --timeout 600000 \
  --generation-config '{"temperature": 0, "max_tokens": 2048}' \
  2>&1 | tee "/tmp/evalscope_dp${DP_SIZE}.txt" || true

sleep 5

# 7. Count cache entries after evalscope
RUNTIME_COUNT=$(find /tmp/jax_cache -type f 2>/dev/null | wc -l)
echo ""
echo "============================================"
echo "RESULTS for dp=$DP_SIZE"
echo "============================================"
echo "Persistent cache entries after precompile: $PRECOMPILE_COUNT"
echo "Persistent cache entries after evalscope:  $RUNTIME_COUNT"
echo "New persistent entries (runtime):          $((RUNTIME_COUNT - PRECOMPILE_COUNT))"
echo ""

# 8. Parse log for PERSISTENT COMPILATION CACHE MISS after fired up
echo "--- PERSISTENT COMPILATION CACHE MISS after fired up ---"
tail -n "+$FIRED_UP_LINE" "$LOG" | grep -c "PERSISTENT COMPILATION CACHE MISS" || echo "0"
echo ""
echo "Top functions with persistent cache miss (after fired up):"
tail -n "+$FIRED_UP_LINE" "$LOG" | grep "PERSISTENT COMPILATION CACHE MISS" | \
  sed "s/.*MISS for '\([^']*\)'.*/\1/" | sort | uniq -c | sort -rn | head -20
echo ""

# 9. Parse log for TRACING CACHE MISS (JAX_EXPLAIN)
echo "--- TRACING CACHE MISS (JAX_EXPLAIN) after fired up ---"
tail -n "+$FIRED_UP_LINE" "$LOG" | grep -c "TRACING CACHE MISS" || echo "0"
echo ""

# 10. Parse log for our custom trace miss logging
echo "--- Custom TRACE CACHE MISS in jitted_run_model (after fired up) ---"
tail -n "+$FIRED_UP_LINE" "$LOG" | grep "TRACE CACHE MISS in jitted_run_model" || echo "(none)"
echo ""

echo "--- SPEC_EXTEND/SPEC_DECODE extra trace miss (after fired up) ---"
tail -n "+$FIRED_UP_LINE" "$LOG" | grep "extra trace miss" || echo "(none)"
echo ""

# 11. HLO dump summary
echo "--- HLO dump files ---"
ls -la /tmp/hlo_dump/*.json 2>/dev/null | wc -l
echo "files dumped"
echo ""
echo "Precompile HLOs:"
ls /tmp/hlo_dump/ 2>/dev/null | grep "precompile\|spec_precompile" | head -10
echo ""
echo "Runtime HLOs:"
ls /tmp/hlo_dump/ 2>/dev/null | grep "runtime" | head -10

# Kill server
kill $SERVER_PID 2>/dev/null || true

echo ""
echo "Full log saved to $LOG"
echo "HLO dumps saved to /tmp/hlo_dump/"
echo "Done."
