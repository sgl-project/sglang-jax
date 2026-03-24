#!/bin/bash
# Full setup: create workload, install deps, clone repo on a GKE TPU v7x pod.
#
# Usage:
#   bash scripts/gke_tpu7x/setup_pod.sh [WORKLOAD_NAME] [IMAGE_TAG]
#
# Defaults:
#   WORKLOAD_NAME = test-workload
#   IMAGE_TAG     = jax0.8.1-rev1  (must match pyproject.toml JAX version)

set -euo pipefail

export PATH="${HOME}/.local/bin:/opt/homebrew/bin:/opt/homebrew/share/google-cloud-sdk/bin:/usr/bin:$PATH"

WORKLOAD="${1:-test-workload}"
IMAGE_TAG="${2:-jax0.8.1-rev1}"
PROJECT="tpu-service-473302"
CLUSTER="xpk-cluster"
ZONE="us-central1-c"
TPU_TYPE="tpu7x-8"
IMAGE="us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:${IMAGE_TAG}"

echo "=== Creating workload: $WORKLOAD (image: $IMAGE_TAG) ==="
xpk workload create \
    --workload "$WORKLOAD" \
    --num-slices=1 \
    --tpu-type="$TPU_TYPE" \
    --cluster="$CLUSTER" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --docker-name="$WORKLOAD" \
    --docker-image="$IMAGE" \
    --command="sleep infinity"

echo ""
echo "=== Waiting for pod ==="
POD=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name="$WORKLOAD" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
RETRIES=0
while [ -z "$POD" ] && [ $RETRIES -lt 30 ]; do
    sleep 5
    POD=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name="$WORKLOAD" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    RETRIES=$((RETRIES + 1))
done

if [ -z "$POD" ]; then
    echo "ERROR: Could not find pod for workload $WORKLOAD"
    exit 1
fi

echo "  Pod: $POD"
kubectl wait --for=condition=Ready "pod/$POD" --timeout=300s

echo ""
echo "=== Installing sglang-jax + deps on both containers ==="
for C in "${WORKLOAD}-1" "${WORKLOAD}-2"; do
    echo "  -> $C"
    kubectl exec "$POD" -c "$C" -- bash -c '
        cd /tmp && git clone --depth 1 https://github.com/sgl-project/sglang-jax.git 2>&1
        cd sglang-jax/python && pip install --no-deps -e . 2>&1
        pip install pyzmq fastapi orjson uvicorn jinja2 pydantic python-multipart \
            huggingface-hub safetensors transformers tiktoken \
            setproctitle psutil pandas httpx openai aiohttp \
            pybase64 partial_json_parser omegaconf \
            msgpack-python requests typing-extensions 2>&1 | tail -3
    ' 2>&1
    echo ""
done

echo "=== Setup complete ==="
echo ""
echo "Pod name: $POD"
echo "Workload: $WORKLOAD"
echo ""
echo "Run scripts with:"
echo "  bash scripts/gke_tpu7x/run_on_pod.sh $POD $WORKLOAD scripts/gke_tpu7x/smoke_test.py"
echo ""
echo "  bash scripts/gke_tpu7x/run_on_pod.sh $POD $WORKLOAD \\"
echo "    benchmark/moe/bench_fused_moe.py \\"
echo "    --num-experts 8 --top-k 2 --hidden-size 2048 --intermediate-size 512 \\"
echo "    --num-tokens 64 128 --iters 3 --warmup-iters 1"
