# Install SGLang-Jax on GPU

You can install SGLang-Jax on GPU using the methods below.

This page is mainly applicable to GPU devices running through JAX.


## Method : From source

```bash
# Use the main branch
git clone https://github.com/sgl-project/sglang-jax
cd sglang-jax

# Install the python packages
pip install --upgrade pip setuptools packaging
pip install -e "python[gpu]"

# Run Qwen-7B Model
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python3 -u -m sgl_jax.launch_server --model-path Qwen/Qwen-7B-Chat --trust-remote-code  --dist-init-addr=0.0.0.0:10011 --nnodes=1  --tp-size=1 --xla-backend=native --attention-backend=native --device=cuda --random-seed=3 --node-rank=0 --mem-fraction-static=0.8 --max-prefill-tokens=64 --max-running-requests=1 --download-dir=/tmp --dtype=bfloat16  --skip-server-warmup --host 0.0.0.0 --port 30000
```

## Bencmark

Running Benchmark using sglang-jax on gpu

```bash
python3 -m sgl_jax.bench_serving --backend sgl-jax --dataset-name random --num-prompts 20 --random-input 32 --random-output 32 --max-concurrency 1 --random-range-ratio 1 --warmup-requests 1
```
