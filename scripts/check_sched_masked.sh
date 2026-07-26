#!/usr/bin/env bash
# Regression guard for issue #293: verify the scheduler's per-forward-step HOST
# work (recv + schedule + build=get_model_worker_batch + dispatch) stays MASKED
# under the TPU forward, so overlap hides it and the scheduler is not the
# throughput bottleneck.
#
# Requires the env-gated SGLANG_SCHED_PHASE_TIME instrumentation in the server
# (scheduler.event_loop_overlap logs "[sched/fwd-step ...]"). Run on a TPU host
# (e.g. v6e-4). Exits 0 if masked (with margin), 1 if the scheduler is the
# bottleneck (regression), 2 on parse/setup failure.
#
#   MARGIN=0.85 bash scripts/check_sched_masked.sh
set -uo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
# Pass if HOST < MARGIN * forward  (forward = HOST + result). 0.85 => require
# >=15% headroom below the point where the scheduler stops being masked.
MARGIN="${MARGIN:-0.85}"
PORT="${PORT:-30000}"
LOG="${LOG:-/tmp/sched_masked_server.log}"

SGLANG_SCHED_PHASE_TIME=1 SGLANG_SCHED_PHASE_INTERVAL=100 \
python -m sgl_jax.launch_server --model-path "$MODEL" --trust-remote-code \
  --device tpu --dtype bfloat16 --tp-size 4 --mem-fraction-static 0.8 \
  --page-size 64 --attention-backend fa --chunked-prefill-size 8192 \
  --download-dir /dev/shm/ --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 &
SRV=$!
trap 'kill "$SRV" 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 5
done

_bench() {  # in out nprompts conc
  python -m sgl_jax.bench_serving --backend sgl-jax --model "$MODEL" \
    --base-url "http://127.0.0.1:$PORT" --dataset-name random --num-prompts "$3" \
    --random-input-len "$1" --random-output-len "$2" --random-range-ratio 1 \
    --max-concurrency "$4" --warmup-requests 0 >/dev/null 2>&1
}
_bench 1024 64   8   8    # warmup / precompile
_bench 1024 1024 200 16   # steady-state decode at the #293 config

# Use a steady-state line (drop the last, which may catch tail/drain).
line=$(grep "sched/fwd-step" "$LOG" | tail -5 | head -1)
host=$(echo "$line" | sed -n 's/.*HOST(recv+sched+build)=\([0-9.]*\)ms.*/\1/p')
res=$(echo "$line"  | sed -n 's/.*result(fwd-wait+D2H)=\([0-9.]*\)ms.*/\1/p')

awk -v h="$host" -v r="$res" -v m="$MARGIN" 'BEGIN{
  if (h=="" || r==""){ print "PARSE FAIL: no [sched/fwd-step] line (instrumentation missing?)"; exit 2 }
  fwd = h + r;                 # forward = host overlap + remaining wait
  ratio = h / fwd;
  printf "sched HOST=%.2fms  forward=%.2fms  host/forward=%.2f  (margin=%.2f)\n", h, fwd, ratio, m;
  if (ratio < m){ print "PASS: scheduler masked under the forward"; exit 0 }
  else { print "FAIL: scheduler host >= "m"x forward -- #293 regression (not masked)"; exit 1 }
}'
