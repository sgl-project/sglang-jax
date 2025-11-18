## Grok-2 Model Server Launch on SkyPilot
### Hardware Requirements
#### Minimum TPU Setup (v6e-32) topology:
- Total nodes: 8
- TPU chips per node: 4 (v6e)
- Total TPU chips: 32

### Launch a TPU v6e-32 Cluster with SkyPilot
From the root of the sglang-jax workspace:
```bash
cd ${WORKSPACE_DIR}/sglang-jax
bash scripts/launch_tpu.sh tpu-v6e-32 main
```
This provisions an 8-node TPU v6e-32 cluster using your SkyPilot configuration.

### Start the sglang-jax Server with Grok-2
After the cluster is running, launch the Grok-2 model server:
```bash
sky exec <CLUSTER_NAME> -- "cd sglang-jax && source .venv/bin/activate && \
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache uv run python -u -m sgl_jax.launch_server \
--model-path /models/xai-grok-2 --trust-remote-code --tp-size=32 --device=tpu \
--random-seed=3 --mem-fraction-static=0.9 --chunked-prefill-size=2048 \
--download-dir=/dev/shm --dtype=bfloat16 --max-running-requests=256 \
--skip-server-warmup --page-size=128 --tokenizer-path alvarobartt/grok-2-tokenizer \
--dist-init-addr=<NODE_0_IP_ADDRESS>:<PORT> \
--nnodes=8 --node-rank=\${SKYPILOT_NODE_RANK}"
```
- `-d`: with `-d` flag, as soon as a job is submitted, return from this call and do not stream execution logs.

### Example of sending a request to the Grok-2 Server
```bash
curl -X POST http://0.0.0.0:30000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "xai-org/grok-2",
"prompt": "The capital city of France is",
"max_tokens": 8,
"temperature": 0,
"top_k": 1}
```

### Example command using evalscope to evaluate grok-2
```bash
evalscope eval  \
--model /models/xai-grok-2 \
--api-url http://127.0.0.1:30000/v1/chat/completions \
--api-key EMPTY \
--eval-type openai_api \
--datasets gsm8k \
--eval-batch-size 64 \
--generation-config '{"temperature": 0.7,"top_p":0.8,"top_k":20,"min_p":0.0,"presence_penalty":0.5}'
```

Test gpqa_diamond datasets:
```bash
evalscope eval \
--model /models/xai-grok-2 \
--api-url http://127.0.0.1:30000/v1/chat/completions \
--api-key EMPTY  --eval-type openai_api --datasets gpqa_diamond \
--eval-batch-size 198 --dataset-args '{"gpqa_diamond": {"few_shot_num": 4}}'  \
--generation-config '{"temperature": 0.5,"top_p":0.8,"top_k":40, "max_tokens": 4096}'
```

### Useful SkyPilot Commands
```bash
sky -h # show help
sky status # show all clusters status
sky queue # show all jobs for cluster(s)
sky logs <CLUSTER_NAME> # show the last job logs
```
