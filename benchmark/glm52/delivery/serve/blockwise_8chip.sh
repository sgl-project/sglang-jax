#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export GLM52_PHYSICAL_CHIPS=8
export GLM52_QUANTIZATION=blockwise
exec "$SCRIPT_DIR/common.sh" "$@"
