#!/bin/bash
set -ex

# Variables
IMAGE_NAME="samos123/sglang-jax"
MANIFEST="k8s/sglang-lws.yaml"
LWS_NAME="sglang-stoelinga"

# Optional: Build and push image
if [ "$1" == "--build" ]; then
    echo "Building Docker image..."
    docker build -t $IMAGE_NAME .
    docker push $IMAGE_NAME
fi

# Apply LWS manifest
echo "Applying LWS manifest..."
kubectl apply -f $MANIFEST

# Apply LWS manifest
echo "Applying LWS manifest..."
kubectl apply -f $MANIFEST

# Wait for leader pod to be created
echo "Waiting for leader pod sglang-stoelinga-0 to be created..."
while ! kubectl get pod/sglang-stoelinga-0 &> /dev/null; do sleep 1; done

echo "Waiting for leader pod to be Running..."
kubectl wait --for=jsonpath='{.status.phase}'=Running pod/sglang-stoelinga-0 --timeout=300s || {
    echo "Pod failed to reach Running state. Fetching status..."
    kubectl get pod/sglang-stoelinga-0
    kubectl logs sglang-stoelinga-0 -c sglang-leader --previous || true
    exit 1
}

echo "Pod is Running. Waiting for Ready condition..."
if ! kubectl wait --for=condition=Ready pod/sglang-stoelinga-0 --timeout=600s; then
    echo "Pod did not become ready. Logs:"
    kubectl logs sglang-stoelinga-0 -c sglang-leader
    exit 1
fi

echo "Leader pod is ready. Fetching active logs..."
kubectl logs sglang-stoelinga-0 -c sglang-leader | head -n 20

echo "Starting port-forward in background..."
kubectl port-forward svc/sglang-leader-stoelinga 8000:8000 &
PF_PID=$!

# Wait for port-forward to be established
sleep 5

echo "Sending test request..."
curl -v http://localhost:8000/v1/models

echo "Sending generation request..."
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The capital of France is",
    "sampling_params": {
      "max_new_tokens": 16,
      "temperature": 0
    }
  }'

echo ""
echo "Cleaning up..."
kill $PF_PID
# Optional: kubectl delete -f $MANIFEST
