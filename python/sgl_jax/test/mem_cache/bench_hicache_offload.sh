#!/usr/bin/env bash
# HiCache L2 offload A/B benchmark.
#
# Same warm-cache workload run twice against an identical sgl-jax server:
#   [off] --hicache-storage disable  -> no host offload
#   [on]  --hicache-storage none     -> L1(HBM)<->L2(host pinned) offload enabled
#
# The device KV pool is capped small via --max-total-tokens so the shared system
# prompts overflow HBM. With offload OFF an evicted prefix is gone and must be
# re-prefilled on its next hit; with offload ON it is demoted to host and loaded
# back (fast H2D), so cache-hit-rate climbs and TTFT drops. The slow D2H backup
# runs async off the critical path.
#
# Run on a TPU pod (single host, tp=4). Override knobs via env vars.
set -euo pipefail

MODEL=${MODEL:-/models/Qwen3-8B}
PORT=${PORT:-30000}
PAGE_SIZE=${PAGE_SIZE:-128}
TP=${TP:-4}
# Cap device KV so prefixes overflow HBM and eviction/offload actually happens.
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-32768}
HICACHE_RATIO=${HICACHE_RATIO:-4.0}

# Workload: GSP_GROUPS unique shared prefixes, each touched GSP_PER_GROUP times.
# Total prefix tokens = GSP_GROUPS * GSP_SYS_LEN. Size it >> MAX_TOTAL_TOKENS but
# <= MAX_TOTAL_TOKENS * (1 + HICACHE_RATIO) so it fits device+host together.
GSP_GROUPS=${GSP_GROUPS:-24}
GSP_PER_GROUP=${GSP_PER_GROUP:-8}
GSP_SYS_LEN=${GSP_SYS_LEN:-4096}
GSP_Q_LEN=${GSP_Q_LEN:-128}
GSP_OUT_LEN=${GSP_OUT_LEN:-64}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-16}
NUM_PROMPTS=$((GSP_GROUPS * GSP_PER_GROUP))

LOGDIR=${LOGDIR:-/tmp/hicache_bench}
mkdir -p "$LOGDIR"

wait_ready () {
  local tag=$1 spid=$2
  for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/get_model_info" >/dev/null 2>&1; then
      echo "[$tag] server ready after ~$((i*5))s"; return 0
    fi
    if ! kill -0 "$spid" 2>/dev/null; then
      echo "[$tag] SERVER DIED during startup:"; tail -60 "$LOGDIR/server_$tag.log"; return 1
    fi
    sleep 5
  done
  echo "[$tag] server did not become ready in time"; tail -60 "$LOGDIR/server_$tag.log"; return 1
}

launch_and_bench () {
  local tag=$1 storage=$2
  echo "=================== RUN [$tag] hicache_storage=$storage ==================="
  python -m sgl_jax.launch_server \
    --model-path "$MODEL" \
    --trust-remote-code \
    --tp-size "$TP" \
    --page-size "$PAGE_SIZE" \
    --max-total-tokens "$MAX_TOTAL_TOKENS" \
    --hicache-storage "$storage" \
    --hicache-ratio "$HICACHE_RATIO" \
    --host 0.0.0.0 --port "$PORT" \
    > "$LOGDIR/server_$tag.log" 2>&1 &
  local spid=$!
  wait_ready "$tag" "$spid" || { kill "$spid" 2>/dev/null || true; return 1; }

  python -m sgl_jax.bench_serving \
    --backend sgl-jax \
    --model "$MODEL" \
    --port "$PORT" \
    --dataset-name generated-shared-prefix \
    --gsp-num-groups "$GSP_GROUPS" \
    --gsp-prompts-per-group "$GSP_PER_GROUP" \
    --gsp-system-prompt-len "$GSP_SYS_LEN" \
    --gsp-question-len "$GSP_Q_LEN" \
    --gsp-output-len "$GSP_OUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --output-file "$LOGDIR/bench_$tag.jsonl" \
    2>&1 | tee "$LOGDIR/bench_$tag.log"

  kill "$spid" 2>/dev/null || true
  wait "$spid" 2>/dev/null || true
  sleep 8
}

launch_and_bench off disable
launch_and_bench on  none

echo ""
echo "############## A/B SUMMARY (offload off vs on) ##############"
for tag in off on; do
  echo "----- [$tag] -----"
  grep -E "Successful requests:|cache hit rate|Total cached tokens:|Mean TTFT|Median TTFT|P99 TTFT|Output token throughput|Total token throughput" \
    "$LOGDIR/bench_$tag.log" || echo "  (no metrics parsed; see $LOGDIR/bench_$tag.log)"
done
