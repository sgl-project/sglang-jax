#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
SOURCE_MODEL="${SOURCE_MODEL:-/models/GLM-5.2}"
TARGET_MODEL="${TARGET_MODEL:-/models/GLM5.2-fp8-channel-wise}"
STAGING_MODEL="${STAGING_MODEL:-${TARGET_MODEL}.staging-v1}"
WORKERS="${WORKERS:-16}"
LOCAL_ROOT="${LOCAL_ROOT:-/tmp/glm52-fp8-channelwise}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${LOCAL_ROOT}/artifacts}"
CHUNK_ELEMENTS="${CHUNK_ELEMENTS:-4194304}"
BARRIER_TIMEOUT="${BARRIER_TIMEOUT:-172800}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-282}"
EXPECTED_SELECTED_TENSORS="${EXPECTED_SELECTED_TENSORS:-59044}"
EXPECTED_WEIGHT_MAP_COUNT="${EXPECTED_WEIGHT_MAP_COUNT:-118629}"

if ! [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'WORKERS must be a positive integer, got %s\n' "$WORKERS" >&2
  exit 2
fi

mkdir -p "$LOCAL_ROOT" "$ARTIFACT_ROOT"
WRAPPER_STATUS_DIR="$LOCAL_ROOT/wrapper-$RUN_ID"
if [[ -e "$WRAPPER_STATUS_DIR" ]]; then
  printf 'wrapper status path already exists; choose a new RUN_ID: %s\n' \
    "$WRAPPER_STATUS_DIR" >&2
  exit 2
fi
mkdir -p "$WRAPPER_STATUS_DIR"

pids=()
terminate_workers() {
  local pid
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap terminate_workers INT TERM

for ((rank = 0; rank < WORKERS; rank++)); do
  worker_local="$LOCAL_ROOT/rank-$rank"
  worker_artifact="$ARTIFACT_ROOT/rank-$rank"
  mkdir -p "$worker_local" "$worker_artifact"
  (
    worker_pid=""
    terminate_python() {
      if [[ -n "$worker_pid" ]]; then
        kill "$worker_pid" 2>/dev/null || true
      fi
    }
    trap terminate_python INT TERM
    "$PYTHON" "$SCRIPT_DIR/convert_channelwise_fp8.py" \
      --source "$SOURCE_MODEL" \
      --staging "$STAGING_MODEL" \
      --final "$TARGET_MODEL" \
      --rank "$rank" \
      --world "$WORKERS" \
      --run-id "$RUN_ID" \
      --local-dir "$worker_local" \
      --artifact-dir "$worker_artifact" \
      --chunk-elements "$CHUNK_ELEMENTS" \
      --barrier-timeout "$BARRIER_TIMEOUT" \
      --expected-shards "$EXPECTED_SHARDS" \
      --expected-selected-tensors "$EXPECTED_SELECTED_TENSORS" \
      --expected-weight-map-count "$EXPECTED_WEIGHT_MAP_COUNT" &
    worker_pid=$!
    if wait "$worker_pid"; then
      printf 'complete\n' >"$WRAPPER_STATUS_DIR/rank-$rank.complete"
    else
      status=$?
      printf '%s\n' "$status" >"$WRAPPER_STATUS_DIR/rank-$rank.failed"
      exit "$status"
    fi
    trap - INT TERM
  ) &
  pids+=("$!")
done

while true; do
  finished=0
  failed_rank=""
  for ((rank = 0; rank < WORKERS; rank++)); do
    if [[ -f "$WRAPPER_STATUS_DIR/rank-$rank.complete" ]]; then
      ((finished += 1))
    elif [[ -f "$WRAPPER_STATUS_DIR/rank-$rank.failed" ]]; then
      failed_rank="$rank"
      break
    fi
  done
  if [[ -n "$failed_rank" ]]; then
    terminate_workers
    for pid in "${pids[@]}"; do
      wait "$pid" 2>/dev/null || true
    done
    printf 'GLM-5.2 checkpoint conversion worker %s failed; peers were stopped\n' \
      "$failed_rank" >&2
    exit 1
  fi
  if ((finished == WORKERS)); then
    break
  fi
  sleep 1
done

for pid in "${pids[@]}"; do
  wait "$pid"
done
trap - INT TERM

test -s "$TARGET_MODEL/_DOWNLOAD_COMPLETE"
printf 'GLM52_CHANNELWISE_CONVERSION_WRAPPER_COMPLETE target=%s workers=%s run_id=%s\n' \
  "$TARGET_MODEL" "$WORKERS" "$RUN_ID"
