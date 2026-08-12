#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

export PYTHONUNBUFFERED=1
export TMPDIR=/tmp/tpu_logs/tmp
export PIP_CACHE_DIR=/tmp/tpu_logs/pip-cache
export UV_CACHE_DIR=/tmp/tpu_logs/uv-cache
export JAX_COMPILATION_CACHE_DIR=/tmp/tpu_logs/jax-compilation-cache
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:+$LIBTPU_INIT_ARGS }--xla_tpu_dvfs_p_state=7"

: "${RUN_MODE:?RUN_MODE must be smoke, benchmark, or profile}"
case "$RUN_MODE" in
  smoke|benchmark|profile) ;;
  *) printf 'unsupported RUN_MODE=%s\n' "$RUN_MODE" >&2; exit 2 ;;
esac

OUT="${ARTIFACT_LOCAL_DIR:-/tmp/glm52-channelwise-w8a16-ep32-${RUN_MODE}-artifact}"
RANK="${FALCON_JAX_PROCESS_ID:-${FALCON_RANK:-${JOB_COMPLETION_INDEX:-0}}}"
WORLD="${FALCON_WORLD_SIZE:-4}"
POD_HOST="$(hostname)"
JOB_BASE="${POD_HOST%-*}"
MASTER_ADDR="$JOB_BASE-0.$JOB_BASE"
RANK_OUT="$OUT/rank-$RANK"
LOCAL_ROOT="/tmp/tpu_logs/glm52-channelwise-w8a16-ep32-${RUN_MODE}-rank-$RANK"
SERVER_LOG="$LOCAL_ROOT/server.log"
PATCH=/tmp/glm52-static-channelwise-w8a8.patch
TUNED_CONFIG=/tmp/glm52-w8a16-ep32-tuned-block-configs.py
QUANT_CONFIG=/tmp/fp8_glm52_static_per_channel_w8a16.yaml
BENCH_SCRIPT=/tmp/bench_dsa_cache_hit_grouped.py
MODEL_PATH=/models/GLM5.2-fp8-channel-wise
SERVER_PID=0
TAIL_PID=0

mkdir -p "$RANK_OUT" "$OUT/workload" "$OUT/profiling" \
  "$LOCAL_ROOT" "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" \
  "$JAX_COMPILATION_CACHE_DIR"

if [ "$RUN_MODE" = "profile" ]; then
  export SGLANG_PROFILE_MAX_HOSTS=1
  export SGLANG_PROFILE_NUM_CHIPS_PER_TASK=1
  export SGLANG_PROFILE_NUM_SPARSE_CORES_TO_TRACE=1
  export SGLANG_PROFILE_NUM_SPARSE_CORE_TILES_TO_TRACE=1
  printf 'GLM52_CHANNELWISE_W8A16_EP32_LIGHTWEIGHT_STAGE_PROFILE compiler_metadata=disabled hosts=1 chips_per_host=1 sparse_cores=1 sparse_core_tiles=1\n'
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
uv venv --python 3.12 --seed "/tmp/tpu_logs/sglang-w8a16-ep32-${RUN_MODE}-venv"
. "/tmp/tpu_logs/sglang-w8a16-ep32-${RUN_MODE}-venv/bin/activate"
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
print("GLM52_CHANNELWISE_W8A16_EP32_TARGET_CONFIG_OK")
PY

python3 - <<'PY'
import jax.numpy as jnp

from sgl_jax.srt.kernels.fused_moe.v2 import tuned_block_configs as tuned

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
        ep_size=32,
        use_shared_expert=True,
        use_grouped_topk=False,
        enable_act_quant=False,
        quant_mode="per_channel",
    )

configs = {tokens: lookup(tokens) for tokens in (64, 65536)}
assert all(config is not None for config in configs.values()), configs
print("GLM52_CHANNELWISE_W8A16_EP32_TUNED_BUCKETS_OK", configs)
PY

python3 - <<'PY'
import importlib.util
import pathlib

path = pathlib.Path("benchmark/glm52/bench_dsa_cache_hit.py")
spec = importlib.util.spec_from_file_location("bench_dsa_cache_hit", path)
assert spec is not None and spec.loader is not None
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)

prefixes, _ = benchmark._make_inputs(
    64, 128, 8, prefix_mode="grouped", prefix_group_count=2
)
layout = benchmark._prefix_layout(prefixes)
assert prefixes[:32] == [prefixes[0]] * 32
assert prefixes[32:] == [prefixes[32]] * 32
assert prefixes[0] != prefixes[32]
assert layout["unique_prefixes"] == 2
assert sorted(group["requests"] for group in layout["prefix_groups"]) == [32, 32]
print("GLM52_CHANNELWISE_W8A16_EP32_GROUPED_PREFIX_TEST_OK", layout)
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
printf 'GLM52_CHANNELWISE_W8A16_EP32_SERVER_LAUNCH mode=%s rank=%s world=%s host=%s master=%s\n' \
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
  --max-prefill-tokens 65536 \
  --context-length 135168 \
  --tp-size 32 \
  --dp-size 32 \
  --dp-schedule-policy round_robin \
  --ep-size 32 \
  --moe-backend fused_v2 \
  --mem-fraction-static 0.88 \
  --max-running-requests 64 \
  --precompile-bs-paddings 64 \
  --precompile-token-paddings 65536 \
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
      printf 'GLM52_CHANNELWISE_W8A16_EP32_SERVER_DIED mode=%s rank=%s\n' "$RUN_MODE" "$RANK"
      exit 1
    fi
    sleep 15
  done
  printf 'GLM52_CHANNELWISE_W8A16_EP32_RANK_COMPLETE mode=%s rank=%s\n' "$RUN_MODE" "$RANK"
  exit 0
fi

READY=0
for _ in $(seq 1 1080); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    printf 'GLM52_CHANNELWISE_W8A16_EP32_SERVER_DIED_BEFORE_READY mode=%s\n' "$RUN_MODE"
    exit 1
  fi
  if curl -sf http://localhost:30000/get_server_info > "$OUT/server-info-ready.json"; then
    READY=1
    break
  fi
  sleep 10
done
if [ "$READY" != "1" ]; then
  printf 'GLM52_CHANNELWISE_W8A16_EP32_SERVER_READY_TIMEOUT mode=%s seconds=10800\n' "$RUN_MODE"
  exit 1
fi
printf 'GLM52_CHANNELWISE_W8A16_EP32_SERVER_READY mode=%s elapsed_seconds=%s\n' \
  "$RUN_MODE" "$(($(date +%s) - LAUNCH_START))"

python3 - "$OUT/server-info-ready.json" "$OUT/workload/capacity-gate.json" <<'PY'
import json
import pathlib
import sys

info_path = pathlib.Path(sys.argv[1])
output_path = pathlib.Path(sys.argv[2])
info = json.loads(info_path.read_text())
states = info.get("internal_states", [])
raw_capacities = [
    int(state.get("memory_usage", {}).get("token_capacity", -1))
    for state in states
]
configured_dp_size = int(info.get("dp_size", -1))
if len(states) == 32:
    capacity_layout = "per_dp_scheduler_states"
    capacities = raw_capacities
elif len(states) == 1 and configured_dp_size == 32:
    capacity_layout = "single_global_scheduler_state"
    capacities = [raw_capacities[0] // configured_dp_size] * configured_dp_size
else:
    raise SystemExit(
        "unexpected scheduler-state layout: "
        f"states={len(states)}, configured_dp_size={configured_dp_size}, "
        f"raw_capacities={raw_capacities}"
    )
result = {
    "expected_dp_size": 32,
    "observed_dp_states": len(states),
    "configured_dp_size": configured_dp_size,
    "capacity_layout": capacity_layout,
    "raw_scheduler_token_capacity": raw_capacities,
    "minimum_required_per_dp_token_capacity": 300000,
    "per_dp_token_capacity": capacities,
    "min_per_dp_token_capacity": min(capacities) if capacities else -1,
    "max_per_dp_token_capacity": max(capacities) if capacities else -1,
}
output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print("GLM52_CHANNELWISE_W8A16_EP32_CAPACITY_GATE", json.dumps(result, sort_keys=True))
if min(capacities) < 300000:
    raise SystemExit(
        f"insufficient per-DP token capacity: min={min(capacities)}, required=300000"
    )
PY

python3 - "$SERVER_LOG" "$OUT/workload/kernel-lookup-audit.json" <<'PY'
import json
import pathlib
import sys

log_path = pathlib.Path(sys.argv[1])
output_path = pathlib.Path(sys.argv[2])
known_zero_grid_key = (
    "('decode', 'bfloat16', 'bfloat16', 64, 512, 64, 64, 2048)"
)
misses = [
    line for line in log_path.read_text(errors="replace").splitlines()
    if "LOOKUP MISS" in line
]
unexpected = [line for line in misses if known_zero_grid_key not in line]
result = {
    "lookup_miss_count": len(misses),
    "known_zero_grid_decode_mnt2048_count": len(misses) - len(unexpected),
    "unexpected_lookup_misses": unexpected,
    "known_zero_grid_status": "pending_runtime_trace_confirmation",
}
output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print("GLM52_CHANNELWISE_W8A16_EP32_KERNEL_LOOKUP_AUDIT", json.dumps(result, sort_keys=True))
if unexpected:
    raise SystemExit(f"unexpected kernel lookup misses: {unexpected}")
PY

grep -Ei 'memory|hbm|available_kv_cache|max_total_num_tokens|weights loaded|precompile finished|LOOKUP MISS|Using v2 tuned block config' \
  "$SERVER_LOG" > "$RANK_OUT/memory-load-compile-kernel-lines.txt" || true

if [ "$RUN_MODE" = "smoke" ]; then
  printf 'GLM52_CHANNELWISE_W8A16_EP32_SMOKE_OK\n'
  exit 0
fi

PROFILE_ARGS=()
METRICS_NAME=benchmark-metrics.jsonl
if [ "$RUN_MODE" = "profile" ]; then
  METRICS_NAME=profile-metrics.jsonl
  PROFILE_ARGS=(
    --profile-output-dir "$OUT/profiling/glm52-channelwise-w8a16-ep32-c64-two-prefix-stage"
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
  --concurrency 64 \
  --dp-size 32 \
  --expected-requests-per-dp 2 \
  --prefix-mode grouped \
  --prefix-group-count 2 \
  --prefix-len 131072 \
  --extend-len 1024 \
  --output-len 1024 \
  --random-seed 3 \
  --variant "channelwise_w8a16_ep32_${RUN_MODE}_c64_two_prefix_128k_1k_1k" \
  --cache-hit-tolerance 64 \
  "${PROFILE_ARGS[@]}" \
  --output "$OUT/workload/$METRICS_NAME" \
  2>&1 | tee "$OUT/workload/${RUN_MODE}.log"

curl -sf http://localhost:30000/get_server_info > "$OUT/server-info-final.json"
python3 - "$SERVER_LOG" "$OUT/workload/concurrency-evidence.json" <<'PY'
import ast
import json
import pathlib
import re
import sys

log_path = pathlib.Path(sys.argv[1])
output_path = pathlib.Path(sys.argv[2])
lines = log_path.read_text(errors="replace").splitlines()

def extract_scalar(line, label):
    match = re.search(re.escape(label) + r"\s*(\d+)", line)
    if not match:
        raise SystemExit(f"missing {label!r} in: {line}")
    return int(match.group(1))

def extract_per_dp(line, label):
    match = re.search(re.escape(label) + r"\s*(\[[^\]]+\])", line)
    if not match:
        raise SystemExit(f"missing {label!r} in: {line}")
    return ast.literal_eval(match.group(1))

prefill = [
    line for line in lines
    if "Prefill batch." in line
    and extract_scalar(line, "#cached-token:") > 0
]
decode = [
    line for line in lines
    if "Decode batch." in line and "#running-req: 64" in line
]
if len(prefill) != 1:
    raise SystemExit(
        "measured cache-hit prefill must be one batch: "
        f"observed_count={len(prefill)}, lines={prefill}"
    )
if not decode:
    raise SystemExit("missing measured C64 decode evidence")

prefill_totals = {
    "new_seq": sum(extract_scalar(line, "#new-seq:") for line in prefill),
    "new_token": sum(extract_scalar(line, "#new-token:") for line in prefill),
    "cached_token": sum(extract_scalar(line, "#cached-token:") for line in prefill),
}
prefill_layouts = [extract_per_dp(line, "#prefill per DP:") for line in prefill]
if any(len(layout) != 32 for layout in prefill_layouts):
    raise SystemExit(f"unexpected measured prefill DP layouts: {prefill_layouts}")
prefill_per_dp = [sum(layout[i] for layout in prefill_layouts) for i in range(32)]
decode_per_dp = extract_per_dp(decode[-1], "#running-req per DP:")
if prefill_totals != {"new_seq": 64, "new_token": 65536, "cached_token": 8388608}:
    raise SystemExit(f"unexpected measured prefill totals: {prefill_totals}")
if prefill_per_dp != [2] * 32:
    raise SystemExit(f"unexpected C64 prefill DP layout: {prefill_per_dp}")
if len(decode_per_dp) != 32 or decode_per_dp != [2] * 32:
    raise SystemExit(f"unexpected C64 decode DP layout: {decode_per_dp}")
evidence = {
    "concurrency": 64,
    "dp_size": 32,
    "requests_per_dp": 2,
    "prefill_batch_count": len(prefill),
    "prefill_totals": prefill_totals,
    "prefill_per_dp": prefill_per_dp,
    "decode_per_dp": decode_per_dp,
    "prefill_log_lines": prefill,
    "decode_log_line": decode[-1],
}
output_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
print("GLM52_CHANNELWISE_W8A16_EP32_C64_LOG_EVIDENCE", json.dumps(evidence, sort_keys=True))
PY

if [ "$RUN_MODE" = "profile" ]; then
  TRACE_FILES=$(find "$OUT/profiling" -type f \( -name '*.xplane.pb' -o -name '*.trace.json.gz' \) | wc -l)
  test "$TRACE_FILES" -gt 0
  printf 'GLM52_CHANNELWISE_W8A16_EP32_PROFILE_TRACE_FILES count=%s\n' "$TRACE_FILES"
fi

printf 'GLM52_CHANNELWISE_W8A16_EP32_%s_RESULT %s\n' \
  "$(printf '%s' "$RUN_MODE" | tr '[:lower:]' '[:upper:]')" \
  "$(cat "$OUT/workload/$METRICS_NAME")"
printf 'GLM52_CHANNELWISE_W8A16_EP32_%s_OK\n' \
  "$(printf '%s' "$RUN_MODE" | tr '[:lower:]' '[:upper:]')"
