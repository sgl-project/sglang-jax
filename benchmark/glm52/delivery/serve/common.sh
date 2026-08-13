#!/usr/bin/env bash
set -euo pipefail

# Shared launcher for the four public GLM-5.2 delivery entry points. Call one
# of {blockwise,channelwise}_{8,16}chip.sh instead of invoking this file.

: "${GLM52_PHYSICAL_CHIPS:?set by a delivery serve wrapper}"
: "${GLM52_QUANTIZATION:?set by a delivery serve wrapper}"
: "${WORLD:?number of serving hosts is required}"
: "${RANK:?zero-based host rank is required}"
: "${MASTER_ADDR:?rank-0 hostname or IP is required}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"

case "$GLM52_PHYSICAL_CHIPS" in
  8)
    EXPECTED_WORLD=2
    PARALLEL_SIZE=16
    CONCURRENCY=32
    MAX_PREFILL_TOKENS=32768
    ;;
  16)
    EXPECTED_WORLD=4
    PARALLEL_SIZE=32
    CONCURRENCY=64
    MAX_PREFILL_TOKENS=65536
    ;;
  *)
    printf 'unsupported physical chip count: %s (expected 8 or 16)\n' \
      "$GLM52_PHYSICAL_CHIPS" >&2
    exit 2
    ;;
esac

if [[ "$WORLD" != "$EXPECTED_WORLD" ]]; then
  printf 'WORLD=%s does not match %s physical chips; expected WORLD=%s v7x-8 hosts\n' \
    "$WORLD" "$GLM52_PHYSICAL_CHIPS" "$EXPECTED_WORLD" >&2
  exit 2
fi
if (( RANK < 0 || RANK >= WORLD )); then
  printf 'RANK=%s is outside [0, %s)\n' "$RANK" "$WORLD" >&2
  exit 2
fi

QUANT_CONFIG=""
case "$GLM52_QUANTIZATION" in
  blockwise)
    MODEL_PATH="${MODEL_PATH:-/models/GLM-5.2-FP8}"
    MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.83}"
    ;;
  channelwise)
    MODEL_PATH="${MODEL_PATH:-/models/GLM5.2-fp8-channel-wise}"
    MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.88}"
    QUANT_CONFIG="${QUANT_CONFIG:-$REPO_ROOT/python/sgl_jax/srt/utils/quantization/configs/fp8_glm52_static_per_channel_moe_w8a8_linear_w8a16.yaml}"
    if [[ ! -s "$QUANT_CONFIG" ]]; then
      printf 'channel-wise quantization config is missing: %s\n' "$QUANT_CONFIG" >&2
      exit 2
    fi
    ;;
  *)
    printf 'unsupported quantization: %s (expected blockwise or channelwise)\n' \
      "$GLM52_QUANTIZATION" >&2
    exit 2
    ;;
esac

for required_file in "$MODEL_PATH/config.json" "$MODEL_PATH/model.safetensors.index.json"; do
  if [[ ! -s "$required_file" ]]; then
    printf 'model checkpoint is incomplete; missing %s\n' "$required_file" >&2
    exit 2
  fi
done

if [[ "${GLM52_SKIP_TUNE_VALIDATION:-0}" != "1" ]]; then
  PYTHONPATH="$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}" \
    python3 "$REPO_ROOT/benchmark/glm52/delivery/validation/validate_delivery_config.py" \
      --physical-chips "$GLM52_PHYSICAL_CHIPS" \
      --quantization "$GLM52_QUANTIZATION"
fi

export PYTHONUNBUFFERED=1
if [[ "${GLM52_DVFS_P_STATE:-7}" != "off" ]] && \
   [[ "${LIBTPU_INIT_ARGS:-}" != *"--xla_tpu_dvfs_p_state="* ]]; then
  export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:+$LIBTPU_INIT_ARGS }--xla_tpu_dvfs_p_state=${GLM52_DVFS_P_STATE:-7}"
fi

SERVER_LOG="${GLM52_SERVER_LOG-/tmp/glm52-${GLM52_QUANTIZATION}-${GLM52_PHYSICAL_CHIPS}chip-rank${RANK}.log}"
if [[ -n "$SERVER_LOG" ]]; then
  if [[ "$SERVER_LOG" == "/dev/stdout" || "$SERVER_LOG" == "/dev/stderr" ]]; then
    exec >> "$SERVER_LOG" 2>&1
  else
    mkdir -p "$(dirname -- "$SERVER_LOG")"
    exec > >(tee -a "$SERVER_LOG") 2>&1
  fi
fi

printf 'GLM52_DELIVERY_SERVER quantization=%s physical_chips=%s jax_devices=%s rank=%s world=%s model=%s log=%s\n' \
  "$GLM52_QUANTIZATION" "$GLM52_PHYSICAL_CHIPS" "$PARALLEL_SIZE" \
  "$RANK" "$WORLD" "$MODEL_PATH" "${SERVER_LOG:-stdout-only}"

LAUNCH_ARGS=(
  -m sgl_jax.launch_server
  --model-path "$MODEL_PATH" \
  --trust-remote-code \
  --device tpu \
  --dtype bfloat16 \
  --kv-cache-dtype bf16 \
  --attention-backend dsa_sparse \
  --dsa-sparse-impl exact \
  --dsa-topk-impl radix \
  --dsa-use-pallas \
  --page-size 64 \
  --chunked-prefill-size 2048 \
  --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
  --context-length 135168 \
  --tp-size "$PARALLEL_SIZE" \
  --dp-size "$PARALLEL_SIZE" \
  --dp-schedule-policy round_robin \
  --ep-size "$PARALLEL_SIZE" \
  --moe-backend fused_v2 \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --max-running-requests "$CONCURRENCY" \
  --precompile-bs-paddings "$CONCURRENCY" \
  --precompile-token-paddings "$MAX_PREFILL_TOKENS" \
  --skip-server-warmup \
  --random-seed 3 \
  --stream-output \
  --stream-interval 1 \
  --nnodes "$WORLD" \
  --node-rank "$RANK" \
  --dist-init-addr "$MASTER_ADDR:${DIST_PORT:-25000}" \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-30000}"
)
if [[ -n "$QUANT_CONFIG" ]]; then
  LAUNCH_ARGS+=(--quantization-config-path "$QUANT_CONFIG")
fi
exec python3 "${LAUNCH_ARGS[@]}" "$@"
