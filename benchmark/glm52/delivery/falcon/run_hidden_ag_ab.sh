#!/usr/bin/env bash
set -euo pipefail

: "${ARTIFACT_LOCAL_DIR:?Falcon must provide ARTIFACT_LOCAL_DIR}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AB_ROOT="$ARTIFACT_LOCAL_DIR"

run_variant() {
  local variant="$1"
  local fp8_hidden="$2"

  printf 'GLM52_HIDDEN_AG_AB_VARIANT_START variant=%s\n' "$variant"
  (
    export ARTIFACT_LOCAL_DIR="$AB_ROOT/$variant"
    export GLM52_DELIVERY_RUN_TAG="$variant"
    if [[ "$fp8_hidden" == "1" ]]; then
      export GLM52_FUSED_RS_FP8_HIDDEN_ALL_GATHER=1
      export GLM52_FUSED_RS_FP8_HIDDEN_ROW_SCALE=1
    else
      unset GLM52_FUSED_RS_FP8_HIDDEN_ALL_GATHER
      unset GLM52_FUSED_RS_FP8_HIDDEN_ROW_SCALE
    fi
    bash "$SCRIPT_DIR/runner.sh"
  )
  printf 'GLM52_HIDDEN_AG_AB_VARIANT_OK variant=%s\n' "$variant"
}

run_variant bf16 0
run_variant fp8-row-scale 1
printf 'GLM52_HIDDEN_AG_AB_OK variants=bf16,fp8-row-scale\n'
