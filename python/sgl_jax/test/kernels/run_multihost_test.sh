#!/bin/bash
set -e

echo "Running multi-host test..."
python3 python/sgl_jax/test/kernels/test_kimi_int4_loading.py
