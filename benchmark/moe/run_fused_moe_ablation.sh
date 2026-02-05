#!/usr/bin/env bash
set -euo pipefail

# Measure fused_moe stage costs by toggling benchmark env flags.
#
# Notes:
# - Some stages have dependencies in the kernel:
#   - A2A requires ALL_REDUCE_METADATA + SYNC_BARRIER (and usually TOPK for realistic routing).
# - For "one-enabled" stage profiling, this script uses a baseline where
#   TOPK + ALL_REDUCE_METADATA + SYNC_BARRIER stay enabled ("control_only"),
#   and toggles other stages one by one. It also supports benchmarking those
#   control stages themselves as dedicated one-enabled cases.

# "Control" stages that we usually want enabled as a baseline for ablations.
CONTROL_VARS=(
  "FUSED_MOE_BENCHMARK_DISABLE_TOPK"                   # TOPK
  "FUSED_MOE_BENCHMARK_DISABLE_ALL_REDUCE_METADATA"    # ALL_REDUCE_METADATA
  "FUSED_MOE_BENCHMARK_DISABLE_SYNC_BARRIER"           # SYNC_BARRIER
)

# Stages we want to benchmark in "one-enabled" mode (all other stages disabled,
# while CONTROL_VARS remain enabled).
STAGE_VARS=(
  "FUSED_MOE_BENCHMARK_DISABLE_A2A"                    # A2A
  "FUSED_MOE_BENCHMARK_DISABLE_DYNAMIC_FFN1"           # DYNAMIC_FFN1
  "FUSED_MOE_BENCHMARK_DISABLE_DYNAMIC_FFN2"           # DYNAMIC_FFN2
  "FUSED_MOE_BENCHMARK_DISABLE_WEIGHT_LOAD"            # WEIGHT_LOAD
  "FUSED_MOE_BENCHMARK_DISABLE_A2A_S_TILE_READ"        # A2A_S_TILE_READ
  "FUSED_MOE_BENCHMARK_DISABLE_A2A_S_ACC_TILE_WRITE"   # A2A_S_ACC_TILE_WRITE
  "FUSED_MOE_BENCHMARK_DISABLE_SHARED_EXPERT"          # SHARED_EXPERT
)

ONE_ENABLED_VARS=(
  "${STAGE_VARS[@]}"
  "${CONTROL_VARS[@]}"
)

COL_NAMES=(
  "all_enable"
  "all_disable"
  "control_only"
  "a2a"
  "dynamic_ffn1"
  "dynamic_ffn2"
  "weight_load"
  "a2a_s_tile_read"
  "a2a_s_acc_tile_write"
  "shared_expert"
  "topk"
  "all_reduce_metadata"
  "sync_barrier"
)

COUNTS=(8 16 32 64 128 256)

NUM_TOKENS=${1:-4096}
RANGE=${2:-":"}

START=0
END=${#ONE_ENABLED_VARS[@]}

if [[ "$RANGE" == *":"* ]]; then
  START=${RANGE%%:*}
  END_PART=${RANGE#*:}
  [[ -n "$START" ]] || START=0
  [[ -n "$END_PART" ]] && END=$END_PART
else
  START=$RANGE
  END=$((RANGE + 1))
fi

COMMON_ARGS="--use-shared-expert --num-experts 256 --topk 8 --hidden-size 8192 --intermediate-size 2048 --num-expert-group 8 --topk-group 4 --weight-dtype float8_e4m3fn --num-tokens ${NUM_TOKENS} --imbalance-mode sparse_hotspot --hotspot-ratio 1"

declare -A RESULTS

TMPFILE=$(mktemp)
trap "rm -f $TMPFILE" EXIT

parse_time() {
  # Extract the first occurrence of:
  #   fused_moe[default]: <number> ms
  #
  # Use grep -F to avoid regex portability issues across environments.
  local line val
  line="$(grep -F 'fused_moe[default]:' | head -1 || true)"
  val="$(printf '%s\n' "${line}" | sed -nE 's/.*:[[:space:]]*([0-9]+([.][0-9]+)?)[[:space:]]*ms.*/\1/p')"
  if [[ -n "${val}" ]]; then echo "${val}"; else echo "N/A"; fi
}

is_number() {
  # Accepts integers or decimals like "0.123" (no exponent support).
  [[ "${1}" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

should_subtract_control_only() {
  # Display as: val(val-control_only) for these columns.
  # Keep control stages (topk / metadata / barrier) as raw timings.
  case "${1}" in
    a2a | dynamic_ffn1 | dynamic_ffn2 | weight_load | a2a_s_tile_read | a2a_s_acc_tile_write | shared_expert)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

format_with_control_only() {
  local val="${1}"
  local ctrl="${2}"
  local diff
  if is_number "${val}" && is_number "${ctrl}"; then
    diff="$(awk -v a="${val}" -v b="${ctrl}" 'BEGIN{printf "%.3f", a-b}')"
    printf "%s(%s)" "${val}" "${diff}"
  else
    printf "%s" "${val}"
  fi
}

build_env_all_enable() {
  local env_str="FUSED_MOE_BENCHMARK_ALL_DISABLE=False "
  local var
  for var in "${CONTROL_VARS[@]}"; do
    env_str+="${var}=False "
  done
  local var
  for var in "${STAGE_VARS[@]}"; do
    env_str+="${var}=False "
  done
  echo "$env_str"
}

build_env_all_disable() {
  # Disable everything (including TOPK / ALL_REDUCE_METADATA / SYNC_BARRIER).
  # Be explicit so this baseline is not affected by any pre-set env vars.
  local env_str="FUSED_MOE_BENCHMARK_ALL_DISABLE=True "
  local var
  for var in "${CONTROL_VARS[@]}"; do
    env_str+="${var}=True "
  done
  local var
  for var in "${STAGE_VARS[@]}"; do
    env_str+="${var}=True "
  done
  echo "$env_str"
}

# Baseline that keeps CONTROL_VARS enabled, but disables all other stages.
# This is the reference to subtract from "one-enabled" timings.
build_env_control_only() {
  local env_str="FUSED_MOE_BENCHMARK_ALL_DISABLE=True "
  local var
  for var in "${CONTROL_VARS[@]}"; do
    env_str+="${var}=False "
  done
  for var in "${STAGE_VARS[@]}"; do
    env_str+="${var}=True "
  done
  echo "$env_str"
}

# Build env string that disables everything except one stage, while keeping
# required dependencies enabled so the benchmark can run.
build_env_with_one_enabled() {
  local enabled_var="$1"

  # Disable everything by default, then selectively enable:
  # - If benchmarking a STAGE_VARS stage: keep CONTROL_VARS enabled as baseline.
  # - If benchmarking a CONTROL_VARS stage: enable only that control stage.
  local env_str="FUSED_MOE_BENCHMARK_ALL_DISABLE=True "

  local controls_default="False"
  case "${enabled_var}" in
    FUSED_MOE_BENCHMARK_DISABLE_TOPK|FUSED_MOE_BENCHMARK_DISABLE_ALL_REDUCE_METADATA|FUSED_MOE_BENCHMARK_DISABLE_SYNC_BARRIER)
      controls_default="True"
      ;;
  esac

  local var
  for var in "${CONTROL_VARS[@]}"; do
    env_str+="${var}=${controls_default} "
  done
  for var in "${STAGE_VARS[@]}"; do
    env_str+="${var}=True "
  done

  env_str+="${enabled_var}=False "

  # Dependency enforcement: ALL_REDUCE_METADATA depends on SYNC_BARRIER.
  if [[ "${enabled_var}" == "FUSED_MOE_BENCHMARK_DISABLE_ALL_REDUCE_METADATA" ]]; then
    env_str+="FUSED_MOE_BENCHMARK_DISABLE_SYNC_BARRIER=False "
  fi

  echo "$env_str"
}

TOTAL_CASES=$((3 + END - START))

echo "num_tokens: ${NUM_TOKENS}, range: [${START}, ${END})"
echo "Running benchmarks..."
echo ""

echo ">>> [1/${TOTAL_CASES}] ALL ENABLE"
for i in "${!COUNTS[@]}"; do
  c=${COUNTS[$i]}
  echo -n "  hotspot_count=$c ... "
  env $(build_env_all_enable) python -m benchmark.moe.bench_fused_moe ${COMMON_ARGS} --hotspot-count "$c" 2>&1 | tee "$TMPFILE"
  time_val=$(parse_time < "$TMPFILE")
  RESULTS["$i,0"]=$time_val
  echo "  -> ${time_val} ms"
done

echo ">>> [2/${TOTAL_CASES}] ALL DISABLE"
for i in "${!COUNTS[@]}"; do
  c=${COUNTS[$i]}
  echo -n "  hotspot_count=$c ... "
  env $(build_env_all_disable) python -m benchmark.moe.bench_fused_moe ${COMMON_ARGS} --hotspot-count "$c" 2>&1 | tee "$TMPFILE"
  time_val=$(parse_time < "$TMPFILE")
  RESULTS["$i,1"]=$time_val
  echo "  -> ${time_val} ms"
done

echo ">>> [3/${TOTAL_CASES}] CONTROL ONLY (TOPK + METADATA + BARRIER)"
for i in "${!COUNTS[@]}"; do
  c=${COUNTS[$i]}
  echo -n "  hotspot_count=$c ... "
  env $(build_env_control_only) python -m benchmark.moe.bench_fused_moe ${COMMON_ARGS} --hotspot-count "$c" 2>&1 | tee "$TMPFILE"
  time_val=$(parse_time < "$TMPFILE")
  RESULTS["$i,2"]=$time_val
  echo "  -> ${time_val} ms"
done

for idx in $(seq $START $((END - 1))); do
  col=$((idx + 3))
  enabled_var=${ONE_ENABLED_VARS[$idx]}
  env_str=$(build_env_with_one_enabled "$enabled_var")
  case_no=$((4 + idx - START))
  echo ">>> [${case_no}/${TOTAL_CASES}] ENABLE: ${enabled_var#FUSED_MOE_BENCHMARK_DISABLE_}"

  for i in "${!COUNTS[@]}"; do
    c=${COUNTS[$i]}
    echo -n "  hotspot_count=$c ... "
    # shellcheck disable=SC2086
    env $env_str python -m benchmark.moe.bench_fused_moe ${COMMON_ARGS} --hotspot-count "$c" 2>&1 | tee "$TMPFILE"
    time_val=$(parse_time < "$TMPFILE")
    RESULTS["$i,$col"]=$time_val
    echo "  -> ${time_val} ms"
  done
done

echo ""
echo "============================================================"
echo "Results (TSV - paste into Excel):"
echo "============================================================"

header="hotspot_count"
for j in "${!COL_NAMES[@]}"; do
  if [[ $j -lt 3 ]] || [[ $((j - 3)) -ge $START && $((j - 3)) -lt $END ]]; then
    header+="\t${COL_NAMES[$j]}"
  fi
done
echo -e "$header"

for i in "${!COUNTS[@]}"; do
  c=${COUNTS[$i]}
  row="$c"
  control_only_val=${RESULTS["$i,2"]:-"-"}
  for j in "${!COL_NAMES[@]}"; do
    if [[ $j -lt 3 ]] || [[ $((j - 3)) -ge $START && $((j - 3)) -lt $END ]]; then
      val=${RESULTS["$i,$j"]:-"-"}
      col_name=${COL_NAMES[$j]}
      if should_subtract_control_only "${col_name}"; then
        val="$(format_with_control_only "${val}" "${control_only_val}")"
      fi
      row+="\t$val"
    fi
  done
  echo -e "$row"
done

echo ""
echo "Done!"
