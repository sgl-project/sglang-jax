FROM python:3.12-slim

WORKDIR /app

# Copy the project files
COPY . .

# Version injection for setuptools-scm (release builds pass this; dev builds fall back to git)
ARG SETUPTOOLS_SCM_PRETEND_VERSION=""
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION}

# Install the sglang-jax package with TPU support
RUN pip install --no-cache-dir ./python[tpu]

# Set environment variable for JAX compilation cache
ENV JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache

# Create directories that may need to be mounted from host
RUN mkdir -p /tmp/jit_cache /tmp/models
