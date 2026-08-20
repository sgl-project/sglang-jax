#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export GLM52_PHYSICAL_CHIPS=16
export GLM52_QUANTIZATION=channelwise
# fused_rs is stage-aware: RS for prefill-family forward modes and the existing
# fused-v2 implementation for decode/target-verify/idle compilations.
export GLM52_MOE_BACKEND=fused_rs
exec "$SCRIPT_DIR/common.sh" "$@"
