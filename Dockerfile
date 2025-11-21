FROM python:3.12

WORKDIR /app

# Copy the project files
COPY . .

# Install the sglang-jax package with TPU support
RUN cd python && pip install -e .[tpu]

# Set environment variable for JAX compilation cache
ENV JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache

# Create directories that may need to be mounted from host
RUN mkdir -p /tmp/jit_cache /tmp/models

# Set entrypoint to launch server with default TPU arguments
# Note: TPU devices are automatically detected by JAX when running on TPU VM
# User must provide --model-path at minimum
# These defaults match the TPU configuration from install.md
# Arguments in ENTRYPOINT will always be applied, user args will be appended
#
# IMPORTANT: To run on TPU VM, you MUST use:
#   docker run --privileged --network=host sglang-jax --model-path <MODEL> --trust-remote-code
#
# TPU access requires:
#   - --privileged: For necessary system permissions
#   - --network=host: TPU is accessed via network (gRPC), not device files
#   - No device mounting needed: TPU is network-based, not /dev-based
#
# Note: TPU devices cannot be mounted in Dockerfile (they're network-accessed),
#       so runtime flags are required for TPU access.
ENTRYPOINT ["python", "-u", "-m", "sgl_jax.launch_server", "--host", "0.0.0.0", "--port", "30000", "--device=tpu", "--dist-init-addr=0.0.0.0:10011", "--nnodes=1", "--tp-size=4", "--node-rank=0", "--mem-fraction-static=0.8", "--max-prefill-tokens=8192", "--download-dir=/tmp", "--dtype=bfloat16", "--skip-server-warmup", "--enable-single-process", "--grammar-backend=none"]
