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

# Note: TPU devices are automatically detected by JAX when running on TPU VM
# User must provide --model-path at minimum
# These defaults match the TPU configuration from install.md
# Arguments in ENTRYPOINT will always be applied, user args will be appended
#
# IMPORTANT: To run on TPU VM, you MUST use:
#   docker run --privileged --network=host sglang-jax --model-path <MODEL> --trust-remote-code
#
# Required Docker flags:
#   - --privileged: Grants access to all devices including VFIO (/dev/vfio/0,1,2,3)
#                   and necessary system permissions for TPU operations
#   - --network=host: Enables gRPC network communication between JAX and TPU Runtime
CMD ["/bin/bash"]
