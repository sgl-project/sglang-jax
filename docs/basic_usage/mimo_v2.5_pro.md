# MiMo-V2.5-Pro on SGL-JAX

MiMo-V2.5-Pro is Xiaomi's large-scale MoE model with hybrid attention (full attention + sliding window attention) and FP8-quantized weights, optimized for long-context reasoning. SGL-JAX supports it on TPU v6e and v7x with FP8 dequantization, tensor parallelism, and expert parallelism.

This cookbook walks through setting up a multi-node TPU slice and serving MiMo-V2.5-Pro end-to-end. The primary reference configuration is **TPU v7x-16** (`2x2x4`, 4 nodes); a TPU v6e-64 (`4x4x4`, 16 nodes) launch command is also provided.

## Hardware Requirements

| TPU Type | Topology | Chips per node | Nodes | Total chips |
|----------|----------|----------------|-------|-------------|
| v7x      | 2x2x4    | 4              | 4     | 16          |
| v6e      | 4x4x4    | 4              | 16    | 64          |

All nodes must be in the same TPU slice and able to reach each other on the JAX init port (`5000` by default below) and the TPU process port (`8471`).

## Environment Setup

### 1. Start a JAX TPU container on each node

Use the official JAX 0.8.1 TPU image:

```bash
docker run -it --privileged \
  --shm-size=32g \
  --ipc=host \
  --network=host \
  -v /dev:/dev \
  us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1 bash
```

> The same image is used by SGL-JAX's GKE / SkyPilot launchers; pinning to `jax0.8.1-rev1` keeps the JAX runtime in lockstep with the SGL-JAX `[tpu]` extras.

### 2. Clone and install SGL-JAX

```bash
git clone https://github.com/sgl-project/sglang-jax.git
cd sglang-jax
git fetch origin
pip install -e "python[tpu]"
```

This installs `sgl-jax` together with its TPU-specific dependencies (matching `jax==0.8.1`).

### 3. (Optional) Install evalscope for accuracy evaluation

```bash
pip install evalscope==0.17.1
```

## Launching the Server

Run the **same** command on every node, only varying `${NODE_RANK}` and pointing all nodes at the rank-0 host via `${MASTER_ADDR}` (e.g. `node0.cluster.local:5000`).

### TPU v7x (16 chips, 4 nodes, `2x2x4`)

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m sgl_jax.launch_server \
    --model-path XiaomiMiMo/MiMo-V2.5-Pro \
    --trust-remote-code \
    --tp-size 32 --dp-size 4 --ep-size 32 \
    --moe-backend fused \
    --host 0.0.0.0 --port 30271 \
    --page-size 256 --context-length 262144 \
    --chunked-prefill-size 4096 \
    --dtype bfloat16 --mem-fraction-static 0.95 \
    --swa-full-tokens-ratio 0.25 \
    --log-level info --max-running-requests 512 \
    --nnodes 4 --node-rank ${NODE_RANK} \
    --dist-init-addr ${MASTER_ADDR}
```

`${NODE_RANK}` ranges from `0` to `3`.

### TPU v6e (64 chips, 16 nodes, `4x4x4`)

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m sgl_jax.launch_server \
    --model-path XiaomiMiMo/MiMo-V2.5-Pro \
    --trust-remote-code \
    --port 30271 \
    --tp-size 64 --dp-size 8 --ep-size 64 \
    --context-length 262144 --max-seq-len 4096 \
    --chunked-prefill-size 4096 --max-prefill-tokens 16384 \
    --page-size 256 \
    --mem-fraction-static 0.92 \
    --max-running-requests 512 \
    --swa-full-tokens-ratio 0.15 \
    --attention-backend fa --moe-backend fused \
    --nnodes 16 --node-rank ${NODE_RANK} \
    --dist-init-addr ${MASTER_ADDR}
```

`${NODE_RANK}` ranges from `0` to `15`. Compared to the v7x recipe, the v6e command lowers `mem-fraction-static` to `0.92` and tightens `swa-full-tokens-ratio` to `0.15` because v6e has less HBM per chip.

### Key flags

- `--tp-size / --ep-size`: Match the total JAX device count across all nodes. v7x exposes 2 logical devices per chip (32 = 16 chips × 2); v6e exposes 1 (64 = 64 chips × 1).
- `--dp-size`: Enables data parallelism for the attention path. The attention TP degree becomes `tp_size / dp_size` (8 for both recipes here). MoE layers still run with full `ep_size`.
- `--moe-backend fused`: Uses the fused Pallas MoE kernel (recommended). `epmoe` is also supported but slower at this scale.
- `--page-size 256`: Required page size for the SWA pool's eviction logic. Smaller page sizes are not supported with MiMo-V2.5-Pro.
- `--context-length 262144`: 256K context. Match this to your workload's max prompt + output length.
- `--chunked-prefill-size 4096`: Splits long prefills into 4K-token chunks to bound peak HBM during prefill.
- `--swa-full-tokens-ratio`: Fraction of the KV cache pool reserved for full-attention layers. `0.25` for v7x, `0.15` for v6e.
- `--mem-fraction-static`: Fraction of HBM reserved for weights + KV cache. Lower if the host shares the TPU.
- `--max-running-requests 512`: Upper bound on concurrent decoding requests.
- `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache`: Persists XLA / Pallas compilation cache so subsequent restarts skip the ~4-minute precompile step.

## Running on GKE (Indexed Job)

For Kubernetes / GKE deployments, the four nodes are launched as an Indexed Job + headless Service so that pod `${index}` resolves to a stable DNS name. Below is a minimal manifest for a TPU v7x `2x2x4` slice. Adjust `nodeSelector`, `claimName`, and `image` for your cluster.

```yaml
---
apiVersion: v1
kind: Service
metadata:
  name: mimo-v2-5-pro-headless-svc
spec:
  clusterIP: None
  selector:
    job-name: mimo-v2-5-pro
  ports:
  - name: dist-init
    port: 5000
  - name: tpu-process
    port: 8471
---
apiVersion: batch/v1
kind: Job
metadata:
  name: mimo-v2-5-pro
spec:
  backoffLimit: 0
  completionMode: Indexed
  parallelism: 4
  completions: 4
  template:
    metadata:
      annotations:
        gke-gcsfuse/volumes: "true"
    spec:
      subdomain: mimo-v2-5-pro-headless-svc
      restartPolicy: Never
      serviceAccountName: gcs-account
      nodeSelector:
        cloud.google.com/gke-tpu-accelerator: tpu7x
        cloud.google.com/gke-tpu-topology: 2x2x4
      containers:
      - name: mimo-v2-5-pro
        image: us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1
        command: ["bash", "-lc"]
        args:
        - |
          set -euxo pipefail

          # --- 1. Clone & install ---
          REPO_DIR=/tmp/sglang-jax
          if [ ! -d "$REPO_DIR/.git" ]; then
            git clone https://github.com/sgl-project/sglang-jax.git "$REPO_DIR"
          fi
          cd "$REPO_DIR" && git fetch origin && pip install -e "python[tpu]"

          # --- 2. Launch the server ---
          export NODE_RANK=${JOB_COMPLETION_INDEX}
          export MASTER_ADDR=mimo-v2-5-pro-0.mimo-v2-5-pro-headless-svc:5000
          JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m sgl_jax.launch_server \
              --model-path /models/MiMo-V2.5-Pro \
              --trust-remote-code \
              --tp-size 32 --dp-size 4 --ep-size 32 \
              --moe-backend fused \
              --host 0.0.0.0 --port 30271 \
              --page-size 256 --context-length 262144 \
              --chunked-prefill-size 4096 \
              --dtype bfloat16 --mem-fraction-static 0.95 \
              --swa-full-tokens-ratio 0.25 \
              --log-level info --max-running-requests 512 \
              --nnodes 4 --node-rank ${NODE_RANK} \
              --dist-init-addr ${MASTER_ADDR}
        env:
        - name: TPU_PROCESS_ADDRESSES
          value: mimo-v2-5-pro-0.mimo-v2-5-pro-headless-svc:8471,mimo-v2-5-pro-1.mimo-v2-5-pro-headless-svc:8471,mimo-v2-5-pro-2.mimo-v2-5-pro-headless-svc:8471,mimo-v2-5-pro-3.mimo-v2-5-pro-headless-svc:8471
        - name: TPU_WORKER_HOSTNAMES
          value: mimo-v2-5-pro-0.mimo-v2-5-pro-headless-svc,mimo-v2-5-pro-1.mimo-v2-5-pro-headless-svc,mimo-v2-5-pro-2.mimo-v2-5-pro-headless-svc,mimo-v2-5-pro-3.mimo-v2-5-pro-headless-svc
        - name: TPU_PROCESS_PORT
          value: "8471"
        - name: JOB_COMPLETION_INDEX
          valueFrom:
            fieldRef:
              fieldPath: metadata.labels['batch.kubernetes.io/job-completion-index']
        - name: TPU_WORKER_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.labels['batch.kubernetes.io/job-completion-index']
        ports:
        - containerPort: 30271
          name: http
        - containerPort: 5000
          name: dist-init
        resources:
          requests:
            google.com/tpu: "4"
          limits:
            google.com/tpu: "4"
        volumeMounts:
        - mountPath: /models
          name: model-storage
        - mountPath: /dev/shm
          name: dev-shm
      volumes:
      - name: dev-shm
        emptyDir:
          medium: Memory
      - name: gke-gcsfuse-cache
        emptyDir:
          medium: Memory
      - name: model-storage
        persistentVolumeClaim:
          claimName: <your-model-pvc>
```

Apply with:

```bash
kubectl apply -f mimo-v2-5-pro.yaml
kubectl wait --for=condition=Ready pod -l job-name=mimo-v2-5-pro --timeout=600s
```

The server is ready once `mimo-v2-5-pro-0` logs `Uvicorn running on http://0.0.0.0:30271`.

## Sending a Request

```bash
curl -X POST http://<rank0-ip>:30271/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "XiaomiMiMo/MiMo-V2.5-Pro",
    "messages": [{"role": "user", "content": "Prove that sqrt(2) is irrational."}],
    "temperature": 1, "top_p": 0.95, "max_tokens": 4096
  }'
```

## Accuracy Evaluation

### AIME 2025 (with thinking enabled)

```bash
evalscope eval \
    --model XiaomiMiMo/MiMo-V2.5-Pro \
    --api-url http://127.0.0.1:30271/v1/chat/completions \
    --api-key EMPTY \
    --eval-type service \
    --datasets aime25 \
    --eval-batch-size 16 \
    --timeout 6000000 \
    --generation-config '{"temperature":1,"top_p":0.95,"max_tokens":131072,"chat_template_kwargs":{"enable_thinking":true}}'
```

Reference numbers measured on TPU v7x-16 (`2x2x4`, `tp=32 ep=32`):

| Model         | Dataset | Metric        | Subset      | Num | Score   |
|:--------------|:--------|:--------------|:------------|:----|:--------|
| MiMo-V2.5-Pro   | aime25  | AveragePass@1 | AIME2025-I  | 15  | 0.8667  |
| MiMo-V2.5-Pro   | aime25  | AveragePass@1 | AIME2025-II | 15  | 1.0000  |
| MiMo-V2.5-Pro   | aime25  | AveragePass@1 | OVERALL     | 30  | 0.9334  |

## Additional Resources

- [MiMo-V2.5-Pro Model Card](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro)
