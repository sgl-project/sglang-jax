#!/bin/bash
set -euxo pipefail

# Install TPU Info
uv pip install tpu-info
tpu-info
# Clean SGLang processes
pgrep -f 'sglang::|sglang-jax::|sglang\.launch_server|sglang\.bench|sglang\.data_parallel|sglang\.srt|sgl_jax\.launch_server|sgl_jax\.srt|sgl_jax\.bench|sgl_jax\.data_parallel' | xargs -r kill -9 || true

# Clean all GPU processes if any argument is provided
if [ $# -gt 0 ]; then
    # Kill TPU processes
    if command -v tpu-info >/dev/null 2>&1; then
        TPU_PIDS=$(tpu-info -p | grep "/dev/vfio" | awk '{print $4}' | grep -v "None" | sort -u)
        if [ ! -z "$TPU_PIDS" ]; then
            echo "Killing TPU processes: $TPU_PIDS"
            for pid in $TPU_PIDS; do
                kill -9 $pid 2>/dev/null
            done
        else
            echo "No active TPU processes found"
        fi
    fi
fi

# Show End Time TPU info
tpu-info
