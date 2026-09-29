# Docker Images for sglang-jax

This directory contains Dockerfiles for building different container images tailored for specific use cases.

## Dockerfiles Overview

| Dockerfile | Purpose | Base Image | Usage / Audience |
| --- | --- | --- | --- |
| `Dockerfile.runtime` | Minimal production image optimized for inference serving | `python:3.12-slim` | Production deployment & release containers |
| `Dockerfile.dev` | Development image containing debug tools and build utilities | `python:3.12` | Active development, debugging, and testing |

---

## Building the Images

> **Note:** Always run `docker build` commands from the repository **root directory** so that the full source code context is included in the build.

### 1. Runtime / Production Image (`Dockerfile.runtime`)

Build a lightweight release image for serving:

```bash
docker build -f docker/Dockerfile.runtime -t sglang-jax:runtime .
```

### 2. Development Image (`Dockerfile.dev`)

Build a development image with build tools and editable installation:

```bash
docker build -f docker/Dockerfile.dev -t sglang-jax:dev .
```

Run an interactive shell in the development container:

```bash
docker run --rm -it \
  --privileged \
  --network=host \
  -v $(pwd):/app \
  sglang-jax:dev bash
```
