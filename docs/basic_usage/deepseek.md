

```zsh
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache uv run python -u -m sgl_jax.launch_server --model-path deepseek-ai/DeepSeek-V2-Lite --trust-remote-code  --dist-init-addr=0.0.0.0:10011 --nnodes=1  --tp-size=1 --device=tpu --random-seed=3 --node-rank=0 --mem-fraction-static=0.8 --max-prefill-tokens=8192 --download-dir=/tmp --dtype=bfloat16  --skip-server-warmup --host 0.0.0.0 --port 30000 2>&1 | tee /home/gcpuser/sky_workdir/sglang-jax/docs/basic_usage/dbug.txt
```