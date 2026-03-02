FROM python:3.12-slim

WORKDIR /app

# Copy the project files
COPY . .

# Install the sglang-jax package with TPU support
RUN cd python && pip install -e .[tpu] --no-cache-dir

# Set environment variable for JAX compilation cache
ENV JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache

# Create directories that may need to be mounted from host
RUN mkdir -p /tmp/jit_cache /tmp/models