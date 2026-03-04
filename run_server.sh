#!/bin/bash
# Enable JAX compilation cache to speed up subsequent runs
export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache
# Enable JAX logging
export JAX_LOG_COMPILES=1
# CRITICAL: Prevent JAX from pre-allocating 90% of system RAM
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "=== System Memory Before Start ==="
vm_stat | head -n 10
echo "=================================="

# Activate the virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Launch the server with optimized CPU settings and verbose python logging
# Using -u for unbuffered output to see logs immediately
python -u -m sgl_jax.launch_server \
    --model-path /Users/jiongxuan/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B/ \
    --device cpu \
    --trust-remote-code \
    --port 30001 \
    --host 127.0.0.1 \
    --tp-size 1 \
    --mem-fraction-static 0.2 \
    --max-prefill-tokens 256 \
    --disable-radix-cache \
    --skip-server-warmup \
    --disable-precompile \
    --enable-single-process \
    --log-level debug

EXIT_CODE=$?
echo "Server exited with code: $EXIT_CODE"

if [ $EXIT_CODE -ne 0 ]; then
    echo "=== System Memory After Failure ==="
    vm_stat | head -n 10
fi
