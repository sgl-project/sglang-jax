#!/bin/bash
# Run on EVERY host (rank = TPU worker number). Logs to ~/v41_server.log
source ~/sgj/bin/activate
export RANK=$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number)
export JAX_LOG_COMPILES=1 NPROC=8 HEAD_IP=${HEAD_IP:-127.0.0.1} MAX_SEQ=${MAX_SEQ:-16384} JAX_COMPILATION_CACHE_DIR=~/jit_cache
cd ~/v41_jax && exec python -u server.py /mnt/models/DeepSeek-V4.1-Flash-1017 ${PORT:-8000}
