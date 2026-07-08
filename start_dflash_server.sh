#!/bin/bash
pkill -9 -f "sgl_jax.launch_server" 2>/dev/null
sleep 2

export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache

HF_HUB_CACHE=/models HF_HOME=/models PYTHONPATH=python .venv/bin/python -m sgl_jax.launch_server \
    --model-path /models/Qwen/Qwen3-8B \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path /models/z-lab/Qwen3-8B-DFlash-b16 \
    --speculative-num-steps 1 --speculative-eagle-topk 1 \
    --disable-overlap-schedule --tp-size 1 --dtype bfloat16 \
    --attention-backend fa --mem-fraction-static 0.75 \
    --trust-remote-code --grammar-backend none \
    --host 0.0.0.0 --port 30000
