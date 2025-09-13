# Install SGLang-Jax

You can install SGLang-Jax using one of the methods below.

This page is mainly applicable to TPU devices running through JAX.

## Method 1: With pip or uv

🚧 **Under Construction** 🚧

## Method 2: From source

```bash
# Use the main branch
git clone https://github.com/sgl-project/sglang-jax
cd sglang-jax

# Install the python packages
pip install --upgrade pip setuptools packaging
pip install -e "python[all]"

# Run Qwen-7B Model
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python3 -u -m sgl_jax.launch_server --model-path Qwen/Qwen-7B-Chat --trust-remote-code  --dist-init-addr=0.0.0.0:10011 --nnodes=1  --tp-size=4 --device=tpu --random-seed=3 --node-rank=0 --mem-fraction-static=0.8 --max-prefill-tokens=8192 --download-dir=/tmp --dtype=bfloat16  --skip-server-warmup --host 0.0.0.0 --port 30000
```

## Method 3: Using docker

🚧 **Under Construction** 🚧

## Method 4: Using Kubernetes

🚧 **Under Construction** 🚧

## Method 5: Using docker compose

🚧 **Under Construction** 🚧

## Method 6: Run on Cloud TPU with SkyPilot

<details>
<summary>More</summary>

To deploy on Google’s Cloud TPU, you can use [SkyPilot](https://github.com/skypilot-org/skypilot).

1. Install SkyPilot and set up cloud access: see [SkyPilot's documentation](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html) and [Cloud TPU — SkyPilot documentation](https://docs.skypilot.co/en/latest/reference/tpu.html)
2. Deploy on your own infra with a single command and get the HTTP API endpoint:
<details>
<summary>SkyPilot YAML: <code>sglang-jax.yaml</code></summary>

```yaml
# sglang-jax.yaml
resources:
   accelerators: tpu-v6e-4
   accelerator_args:
      tpu_vm: True
      runtime_version: v2-alpha-tpuv6e
file_mounts:
  ~/.ssh/id_rsa: ~/.ssh/id_rsa
setup: |
  chmod 600 ~/.ssh/id_rsa
  rm ~/.ssh/config
  GIT_SSH_COMMAND="ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" git clone https://github.com/sgl-project/sglang-jax
run: |
  cd sglang-jax
  pip install -e "python[all]"
  JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python3 -u -m sgl_jax.launch_server --model-path Qwen/Qwen-7B-Chat --trust-remote-code  --dist-init-addr=0.0.0.0:10011 --nnodes=1  --tp-size=4 --device=tpu --random-seed=3 --node-rank=0 --mem-fraction-static=0.8 --max-prefill-tokens=8192 --download-dir=/tmp --dtype=bfloat16  --skip-server-warmup --attention-backend=fa --host 0.0.0.0 --port 30000
```

</details>

```bash
sky launch -c sglang-jax sglang.yaml --infra=gcp

# Get the HTTP API endpoint
sky status --endpoint 30000 sglang-jax
```
- For debugging and testing purposes, you can use spot instances to reduce costs by adding the `--use-spot` flag to your SkyPilot commands:
  ```bash
  sky launch -c sglang-jax sglang.yaml --infra=gcp --use-spot
  ```

</details>
