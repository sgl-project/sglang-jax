# Install SGLang-Jax

You can install SGLang-Jax using one of the methods below.

This page is mainly applicable to TPU devices running through JAX.

## Method 1: With pip

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install "sglang-jax[tpu]"
```

## Method 2: From source

```bash
# Use the main branch
git clone https://github.com/sgl-project/sglang-jax
cd sglang-jax

# Install the python packages
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e "python[all]"
```

## Run service after pip or source install

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server \
  --model-path Qwen/Qwen-7B-Chat \
  --trust-remote-code \
  --tp-size=4 \
  --device=tpu \
  --dtype=bfloat16 \
  --mem-fraction-static=0.8 \
  --host=0.0.0.0 \
  --port=30000
```

## Method 3: Using docker
Docker images are published at <https://hub.docker.com/r/lmsysorg/sglang-jax>.

Pull image
```bash
docker pull lmsysorg/sglang-jax:<TAG>
```

Run service on a TPU VM

```bash
docker run --rm -it \
  --name sglang-jax \
  --privileged \
  --network=host \
  -e JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache \
  -e HF_HOME=/tmp/models \
  -v /tmp/jit_cache:/tmp/jit_cache \
  lmsysorg/sglang-jax:<TAG> \
  python3 -u -m sgl_jax.launch_server \
  --model-path Qwen/Qwen-7B-Chat \
  --trust-remote-code \
  --tp-size=4 \
  --device=tpu \
  --dtype=bfloat16 \
  --mem-fraction-static=0.8 \
  --download-dir=/tmp/models \
  --host=0.0.0.0 \
  --port=30000
```

## Verify the service

```bash
curl -X POST http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen-7B-Chat",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

## Method 4: Using Kubernetes

🚧 **Under Construction** 🚧

## Method 5: Run on Cloud TPU with SkyPilot

<details>
<summary>More</summary>

To deploy on Google’s Cloud TPU, you can use [SkyPilot](https://github.com/skypilot-org/skypilot).

1. Install SkyPilot and set up cloud access: see [SkyPilot's documentation](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html) and [Cloud TPU — SkyPilot documentation](https://docs.skypilot.co/en/latest/reference/tpu.html)
2. Deploy on your own infra with a single command and get the HTTP API endpoint:
<details>
<summary>SkyPilot YAML: <code>sglang-jax.sky.yaml</code></summary>

```yaml
# sglang-jax.sky.yaml
resources:
   accelerators: tpu-v6e-4
   accelerator_args:
      tpu_vm: True
      runtime_version: v2-alpha-tpuv6e
run: |
  git clone https://github.com/sgl-project/sglang-jax.git
  cd sglang-jax && git fetch origin $REF:$REF && git checkout $REF
  uv venv --python 3.12
  source .venv/bin/activate
  uv pip install -e "python[all]"
```

</details>

```bash
sky launch -c sglang-jax sglang.sky.yaml --infra=gcp

```
- For debugging and testing purposes, you can use spot instances to reduce costs by adding the `--use-spot` flag to your SkyPilot commands:
  ```bash
  sky launch -c sglang-jax sglang.sky.yaml --infra=gcp --use-spot
  ```

</details>
