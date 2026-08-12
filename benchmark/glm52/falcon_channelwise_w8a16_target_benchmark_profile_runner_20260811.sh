#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

export PYTHONUNBUFFERED=1
export TMPDIR=/tmp/tpu_logs/tmp
export PIP_CACHE_DIR=/tmp/tpu_logs/pip-cache
export UV_CACHE_DIR=/tmp/tpu_logs/uv-cache
export JAX_COMPILATION_CACHE_DIR=/tmp/tpu_logs/jax-compilation-cache
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:+$LIBTPU_INIT_ARGS }--xla_tpu_dvfs_p_state=7"

: "${RUN_MODE:?RUN_MODE must be benchmark or profile}"
case "$RUN_MODE" in
  benchmark|profile) ;;
  *) printf 'unsupported RUN_MODE=%s\n' "$RUN_MODE" >&2; exit 2 ;;
esac

OUT="${ARTIFACT_LOCAL_DIR:-/tmp/glm52-channelwise-w8a16-${RUN_MODE}-artifact}"
RANK="${FALCON_RANK:-${FALCON_JAX_PROCESS_ID:-${JOB_COMPLETION_INDEX:-0}}}"
WORLD="${FALCON_WORLD_SIZE:-2}"
POD_HOST="$(hostname)"
JOB_BASE="${POD_HOST%-*}"
MASTER_ADDR="$JOB_BASE-0.$JOB_BASE"
RANK_OUT="$OUT/rank-$RANK"
LOCAL_ROOT="/tmp/tpu_logs/glm52-channelwise-w8a16-${RUN_MODE}-rank-$RANK"
SERVER_LOG="$LOCAL_ROOT/server.log"
PATCH=/tmp/glm52-static-channelwise-w8a8.patch
TUNED_CONFIG=/tmp/glm52-w8a16-tuned-block-configs.py
QUANT_CONFIG=/tmp/fp8_glm52_static_per_channel_w8a16.yaml
BENCH_SCRIPT=/tmp/bench_dsa_cache_hit.py
MODEL_PATH=/models/GLM5.2-fp8-channel-wise
SERVER_PID=0
TAIL_PID=0

mkdir -p "$RANK_OUT" "$RANK_OUT/compiler/llo" "$OUT/workload" "$OUT/profiling" \
  "$LOCAL_ROOT" "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" \
  "$JAX_COMPILATION_CACHE_DIR"

if [ "$RUN_MODE" = "profile" ]; then
  export SGLANG_PROFILE_MAX_HOSTS=1
  export SGLANG_PROFILE_NUM_CHIPS_PER_TASK=1
  if [ "${SGLANG_PROFILE_COMPILER_METADATA:-1}" = "1" ]; then
    export LIBTPU_INIT_ARGS="$LIBTPU_INIT_ARGS --xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true --xla_mosaic_dump_to=$RANK_OUT/compiler/llo"
  else
    printf 'GLM52_CHANNELWISE_W8A16_LIGHTWEIGHT_XPROF compiler_metadata=disabled\n'
  fi
fi

for required in "$PATCH" "$TUNED_CONFIG" "$QUANT_CONFIG" "$BENCH_SCRIPT"; do
  test -s "$required"
done
printf '%s  %s\n' 4370536c37860eb0d8b5ccb6c2d9e4baf7117b8f8c403d0f2de5c37b986cf858 "$PATCH" | sha256sum -c -
printf '%s  %s\n' "$TUNED_CONFIG_SHA256" "$TUNED_CONFIG" | sha256sum -c -
printf '%s  %s\n' "$QUANT_CONFIG_SHA256" "$QUANT_CONFIG" | sha256sum -c -
printf '%s  %s\n' "$BENCH_SCRIPT_SHA256" "$BENCH_SCRIPT" | sha256sum -c -

cd /workspace/sglang-jax
test "$(git rev-parse HEAD)" = 3a83b2b3c56f1e6e1f3e018cdaf503bfa669f427
git apply --check "$PATCH"
git apply "$PATCH"
install -m 0644 "$TUNED_CONFIG" \
  python/sgl_jax/srt/kernels/fused_moe/v2/tuned_block_configs.py
install -m 0644 "$QUANT_CONFIG" \
  python/sgl_jax/srt/utils/quantization/configs/fp8_glm52_static_per_channel_w8a16.yaml
install -m 0644 "$BENCH_SCRIPT" benchmark/glm52/bench_dsa_cache_hit.py
git diff --check
git rev-parse HEAD > "$RANK_OUT/base-revision.txt"

python3 -m pip install -q --upgrade pip uv
uv venv --python 3.12 --seed "/tmp/tpu_logs/sglang-w8a16-${RUN_MODE}-venv"
. "/tmp/tpu_logs/sglang-w8a16-${RUN_MODE}-venv/bin/activate"
uv pip install -q -e 'python[tpu]'
export PYTHONPATH=/workspace/sglang-jax/python:${PYTHONPATH:-}

python3 - "$QUANT_CONFIG" <<'PY'
import pathlib
import sys

from sgl_jax.srt.configs.quantization_config import QuantizationConfig

config = QuantizationConfig.from_yaml(str(pathlib.Path(sys.argv[1])))
assert config.is_static_checkpoint
assert config.weight_block_size is None
assert config.get_moe_activation_dtype() is None
assert all(rule.get("activation_dtype") is None for rule in config.get_linear_rules())
print("GLM52_CHANNELWISE_W8A16_TARGET_CONFIG_OK")
PY

python3 - <<'PY'
import jax.numpy as jnp

from sgl_jax.srt.kernels.fused_moe.v2 import tuned_block_configs as tuned
from sgl_jax.srt.kernels.fused_moe.v2.kernel import FusedMoEBlockConfig

tuned.get_device_name = lambda: "TPU v7"

def lookup(tokens):
    return tuned.get_tuned_fused_moe_v2_block_config(
        num_tokens=tokens,
        num_experts=256,
        top_k=8,
        hidden_size=6144,
        intermediate_size=2048,
        dtype=jnp.bfloat16,
        weight_dtype=jnp.float8_e4m3fn,
        ep_size=16,
        use_shared_expert=True,
        use_grouped_topk=False,
        enable_act_quant=False,
        quant_mode="per_channel",
    )

assert lookup(32) == FusedMoEBlockConfig(bt=8, bf=256, btc=8, bse=256, bts=8)
assert lookup(32768) == FusedMoEBlockConfig(bt=128, bf=1024, btc=32, bse=1024, bts=128)
print("GLM52_CHANNELWISE_W8A16_TUNED_BUCKETS_OK")
PY

python3 - "$MODEL_PATH" <<'PY' > "$RANK_OUT/model-check.json"
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
required = [root / "config.json", root / "model.safetensors.index.json", root / "_DOWNLOAD_COMPLETE"]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"incomplete channel-wise checkpoint: {missing}")
index = json.loads(required[1].read_text())
shards = sorted(set(index["weight_map"].values()))
missing_shards = [name for name in shards if not (root / name).is_file()]
if missing_shards:
    raise SystemExit(f"missing checkpoint shards: {missing_shards[:10]}")
print(json.dumps({"root": str(root), "shards": len(shards)}, sort_keys=True))
PY

stop_process() {
  local pid="$1"
  if [ "$pid" -le 0 ]; then
    return
  fi
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
      return
    fi
    sleep 1
  done
  kill -9 "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

cleanup() {
  set +e
  touch "$OUT/ALL_DONE"
  stop_process "$SERVER_PID"
  stop_process "$TAIL_PID"
  cp "$SERVER_LOG" "$RANK_OUT/server.log" 2>/dev/null || true
}
trap cleanup EXIT

LAUNCH_START=$(date +%s)
printf 'GLM52_CHANNELWISE_W8A16_SERVER_LAUNCH mode=%s rank=%s world=%s host=%s master=%s\n' \
  "$RUN_MODE" "$RANK" "$WORLD" "$POD_HOST" "$MASTER_ADDR"
python3 -m sgl_jax.launch_server \
  --model-path "$MODEL_PATH" \
  --trust-remote-code \
  --device tpu \
  --dtype bfloat16 \
  --kv-cache-dtype bf16 \
  --quantization-config-path "$QUANT_CONFIG" \
  --attention-backend dsa_sparse \
  --dsa-sparse-impl exact \
  --dsa-topk-impl radix \
  --dsa-use-pallas \
  --page-size 64 \
  --chunked-prefill-size 2048 \
  --max-prefill-tokens 32768 \
  --context-length 135168 \
  --tp-size 16 \
  --dp-size 16 \
  --dp-schedule-policy round_robin \
  --ep-size 16 \
  --moe-backend fused_v2 \
  --mem-fraction-static 0.90 \
  --max-running-requests 32 \
  --precompile-bs-paddings 32 \
  --precompile-token-paddings 32768 \
  --skip-server-warmup \
  --random-seed 3 \
  --stream-output \
  --stream-interval 1 \
  --nnodes "$WORLD" \
  --node-rank "$RANK" \
  --dist-init-addr "$MASTER_ADDR:25000" \
  --host 0.0.0.0 \
  --port 30000 \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
tail -n +1 -F "$SERVER_LOG" &
TAIL_PID=$!

if [ "$RANK" != "0" ]; then
  while [ ! -f "$OUT/ALL_DONE" ]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      printf 'GLM52_CHANNELWISE_W8A16_SERVER_DIED mode=%s rank=%s\n' "$RUN_MODE" "$RANK"
      exit 1
    fi
    sleep 15
  done
  printf 'GLM52_CHANNELWISE_W8A16_RANK_COMPLETE mode=%s rank=%s\n' "$RUN_MODE" "$RANK"
  exit 0
fi

READY=0
for _ in $(seq 1 1080); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    printf 'GLM52_CHANNELWISE_W8A16_SERVER_DIED_BEFORE_READY mode=%s\n' "$RUN_MODE"
    exit 1
  fi
  if curl -sf http://localhost:30000/get_server_info > "$OUT/server-info-ready.json"; then
    READY=1
    break
  fi
  sleep 10
done
if [ "$READY" != "1" ]; then
  printf 'GLM52_CHANNELWISE_W8A16_SERVER_READY_TIMEOUT mode=%s seconds=10800\n' "$RUN_MODE"
  exit 1
fi
printf 'GLM52_CHANNELWISE_W8A16_SERVER_READY mode=%s elapsed_seconds=%s\n' \
  "$RUN_MODE" "$(($(date +%s) - LAUNCH_START))"

PROFILE_ARGS=()
METRICS_NAME=benchmark-metrics.jsonl
if [ "$RUN_MODE" = "profile" ]; then
  METRICS_NAME=profile-metrics.jsonl
  PROFILE_ARGS=(
    --profile-output-dir "$OUT/profiling/glm52-channelwise-w8a16-target-c32-128k-stage"
    --profile-host-tracer-level 0
    --profile-python-tracer-level 0
    --profile-num-steps 3
    --profile-by-stage
    --profile-stages prefill decode
  )
fi

python3 benchmark/glm52/bench_dsa_cache_hit.py \
  --base-url http://localhost:30000 \
  --server-log "$SERVER_LOG" \
  --concurrency 32 \
  --dp-size 16 \
  --expected-requests-per-dp 2 \
  --prefix-mode shared \
  --prefix-len 131072 \
  --extend-len 1024 \
  --output-len 1024 \
  --random-seed 3 \
  --variant "channelwise_w8a16_target_${RUN_MODE}_c32_128k_1k_1k" \
  --cache-hit-tolerance 64 \
  "${PROFILE_ARGS[@]}" \
  --output "$OUT/workload/$METRICS_NAME" \
  2>&1 | tee "$OUT/workload/${RUN_MODE}.log"

curl -sf http://localhost:30000/get_server_info > "$OUT/server-info-final.json"
grep -Ei 'memory|hbm|available_kv_cache|max_total_num_tokens|weights loaded|precompile finished|LOOKUP MISS' \
  "$SERVER_LOG" > "$RANK_OUT/memory-load-compile-kernel-lines.txt" || true

if [ "$RUN_MODE" = "profile" ]; then
  TRACE_FILES=$(find "$OUT/profiling" -type f \( -name '*.xplane.pb' -o -name '*.trace.json.gz' \) | wc -l)
  test "$TRACE_FILES" -gt 0
  printf 'GLM52_CHANNELWISE_W8A16_PROFILE_TRACE_FILES count=%s\n' "$TRACE_FILES"
fi

printf 'GLM52_CHANNELWISE_W8A16_%s_RESULT %s\n' \
  "$(printf '%s' "$RUN_MODE" | tr '[:lower:]' '[:upper:]')" \
  "$(cat "$OUT/workload/$METRICS_NAME")"
printf 'GLM52_CHANNELWISE_W8A16_%s_OK\n' \
  "$(printf '%s' "$RUN_MODE" | tr '[:lower:]' '[:upper:]')"
