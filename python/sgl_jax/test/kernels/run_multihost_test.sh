#!/bin/bash
set -e

echo "Running multi-host test..."
export JAX_COORDINATOR_ADDRESS="gke-tpu-4a99f854-2zmz:9915"
export NUM_LAYERS=2
python3 python/sgl_jax/test/kernels/test_kimi_int4_loading.py
