# Getting Started with SGLang-Jax on GKE TPU

This guide provides step-by-step instructions for deploying and serving Large Language Models (LLMs) using **SGLang-Jax** on **Google Kubernetes Engine (GKE)** equipped with **Cloud TPUs**. JAX's unique JIT compilation model requires specific configurations to avoid common pitfalls like startup timeouts, deadlocks, etc. so this guide has the correct configurations for a good getting started experience.

> [!IMPORTANT]
> This guide is designed to help you get familiar with SGLang-Jax on GKE TPU in a development environment. The configurations presented here (such as disabling JIT precompilation and using lightweight `/health` checks) are optimized for fast feedback and quick testing, and are **not recommended for production deployments**.

## Prerequisites

1.  **Google Cloud Project**: An active GCP project (eg: `your-project-id`).
2.  **TPU Quota**: Ensure you have quota for Cloud TPUs in your chosen region. *(Example: Cloud TPU v5e `ct5lp-hightpu-8t` machine type in `us-west1-c`).*
3.  **gcloud CLI & kubectl**: Installed and authenticated.


## Step 1: Create a GKE Cluster and TPU Node Pool

First, set up your GKE cluster and provision a dedicated Cloud TPU v5e node pool.

### 1. Define Environment Variables

```bash
export PROJECT_ID="your-project-id"
export ZONE="us-west1-c" # Replace with your chosen region/zone (eg: us-west1-c)
export CLUSTER_NAME="sglang-tpu-cluster"
export MACHINE_TYPE="ct5lp-hightpu-8t" # Replace with your allocated TPU machine type (eg: ct5lp-hightpu-8t for TPU v5e, 8 chips)
export NODE_POOL_NAME="tpu-node-pool"
```

### 2. Create the Cluster

Create a standard GKE cluster with the necessary addons (RayOperator and GcsFuseCsiDriver are optional but highly recommended for LLM workloads):

```bash
gcloud container clusters create $CLUSTER_NAME \
  --addons=RayOperator,GcsFuseCsiDriver \
  --workload-pool=${PROJECT_ID}.svc.id.goog \
  --machine-type=n2-standard-16 \
  --location=$ZONE \
  --enable-image-streaming \
  --project=$PROJECT_ID
```

### 3. Connect Kubectl (get cluster credentials and set as kubeconfig)

```bash
gcloud container clusters get-credentials $CLUSTER_NAME --location $ZONE --project $PROJECT_ID
```

### 4. Create the TPU Node Pool

Provision a single-host Spot TPU node pool using your allocated machine type (the example below creates a Spot node pool for cost savings):

```bash
gcloud container node-pools create $NODE_POOL_NAME \
  --location=$ZONE \
  --cluster=$CLUSTER_NAME \
  --spot \
  --num-nodes=1 \
  --reservation-affinity=none \
  --machine-type=$MACHINE_TYPE \
  --project=$PROJECT_ID
```

## Step 2: Set Up Artifact Registry and IAM Permissions

GKE nodes need explicit permission to pull your custom Docker image.

### 1. Create Artifact Registry

Create a Docker repository in the same region as your cluster.

**NOTE**: the example command below assumes the location is us-west1, you will need to modify us-west1 to the region of your cluster manually for your usage.

```bash
gcloud artifacts repositories create sglang-jax-repo \
    --repository-format=docker \
    --location=us-west1 \
    --description="Docker repository for sglang-jax" \
    --project=$PROJECT_ID
```

### 2. Configure IAM Policy (Critical)

GKE nodes typically run under the default Compute Engine service account. You must grant this service account the `Artifact Registry Reader` role, otherwise the deployment will fail with `ImagePullBackOff` (403 Forbidden).

Retrieve your project number from `gcloud projects describe $PROJECT_ID` and run:

```bash
export PROJECT_NUMBER="your-project-number"
export SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/artifactregistry.reader"
```

## Step 3: Build and Push the Docker Image

*(Note: The sgl-jax team is actively working on an official release pipeline. Once it lands, you will be able to pull the official image directly from Docker Hub and skip this build step).*

Use the provided `Dockerfile` in the root of the `sglang-jax` repository to build and push the image.

```bash
export REGISTRY_URL="us-west1-docker.pkg.dev/$PROJECT_ID/sglang-jax-repo"

# Configure docker auth (use --quiet to prevent interactive prompt hangs in scripts)
gcloud auth configure-docker us-west1-docker.pkg.dev --project=$PROJECT_ID --quiet

# Build the image (run from the root of the sglang-jax repository)
docker build -t $REGISTRY_URL/sglang-jax:latest -f Dockerfile .

# Push to Artifact Registry
docker push $REGISTRY_URL/sglang-jax:latest
```

## Step 4: Create and Deploy the Kubernetes Manifest

Save the following manifest as `sglang-tpu-deployment.yaml`. This manifest includes critical JAX/TPU optimizations explained in the **Friction Points & Key Insights** section below.

**NOTE**: the example YAML below has some sections that you will need to update based on the image you pushed, your machine-type, and node pool capacity.  Look for the "NOTE:" comments in the YAML below and update as needed.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sglang-tpu
  labels:
    app: sglang-tpu
spec:
  replicas: 1
  # CRITICAL: Must use Recreate strategy for single-node TPU upgrades to avoid allocation deadlock
  strategy:
    type: Recreate
  selector:
    matchLabels:
      app: sglang-tpu
  template:
    metadata:
      labels:
        app: sglang-tpu
    spec:
      # NOTE: The nodeSelector below is configured for TPU v5e (2x4 topology). 
      # Adjust these labels to match your allocated TPU type and topology (eg: v4, v6e).
      nodeSelector:
        cloud.google.com/gke-tpu-accelerator: tpu-v5-lite-podslice
        cloud.google.com/gke-tpu-topology: 2x4
        cloud.google.com/gke-spot: "true"
      tolerations:
      - key: "google.com/tpu"
        operator: "Exists"
        effect: "NoSchedule"
      containers:
      - name: sglang-container
        # NOTE: Adjust the image as needed
        image: us-west1-docker.pkg.dev/your-project-id/sglang-jax-repo/sglang-jax:latest
        imagePullPolicy: Always
        command:
        - python3
        - -u # CRITICAL: Disable python stdout buffering to stream logs immediately
        - -m
        - sgl_jax.launch_server
        - --model-path
        - Qwen/Qwen2.5-7B-Instruct
        - --tp-size
        - "8" # GOTCHA: tp-size here actually represents total devices (world size).
        - --dp-size
        - "2" # Actual TP = tp_size // dp_size = 8 // 2 = 4 (required for Qwen 28 heads)
        - --device
        - tpu
        - --host
        - 0.0.0.0
        - --port
        - "30000"
        - --mem-fraction-static
        - "0.8"
        - --disable-precompile # RECOMMENDED for Quickstart: bypasses 30min startup JIT
        - --watchdog-timeout
        - "1200" # CRITICAL: Bypasses 5min watchdog crash during long JIT compilation phases
        env:
        - name: JAX_COMPILATION_CACHE_DIR
          value: "/tmp/jit_cache"
        # CRITICAL TIMEOUTS: Prevent internal JIT warmup crashes
        - name: SGLANG_WAIT_TIMEOUT
          value: "600"
        - name: SGLANG_HEALTH_CHECK_TIMEOUT
          value: "600"
        ports:
        - containerPort: 30000
          name: api-port
        # PROBE STRATEGY: Use /health for fast, non-blocking readiness checks to avoid JIT cancellations
        readinessProbe:
          httpGet:
            path: /health
            port: 30000
          initialDelaySeconds: 300 # Give warmup JIT 5 minutes of quiet time
          periodSeconds: 15
          timeoutSeconds: 10
          failureThreshold: 30
        livenessProbe:
          httpGet:
            path: /health
            port: 30000
          initialDelaySeconds: 600
          periodSeconds: 30
        # NOTE: Adjust TPU resource requests to match your allocated node pool capacity.
        resources:
          requests:
            cpu: "10"
            memory: "128Gi"
            google.com/tpu: "8" # Match this to your node pool capacity
          limits:
            cpu: "10"
            memory: "128Gi"
            google.com/tpu: "8"
        volumeMounts:
        - mountPath: /tmp
          name: tmp-volume
      volumes:
      - name: tmp-volume
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: sglang-tpu-service
spec:
  selector:
    app: sglang-tpu
  ports:
  - protocol: TCP
    port: 30000
    targetPort: 30000
  type: ClusterIP
```

Apply the manifest:

```bash
kubectl apply -f sglang-tpu-deployment.yaml
```

## Step 5: Verification

Because `--disable-precompile` is set, the server will start up quickly (~2-3 minutes to load weights), but it will immediately execute an internal warmup request that JIT compiles the core model graphs on-the-fly.

### 1. Monitor Startup Logs

Do not send requests until you see JIT compilation finish. Stream the logs:

```bash
kubectl logs -f -l app=sglang-tpu
```

You will see JAX JIT compilation messages (`get_default_block_sizes`). After ~7-8 minutes (depending on network and compiler speed), you should see:

```text
[2026-05-19 00:31:12] INFO:     127.0.0.1:41826 - "POST /generate HTTP/1.1" 200 OK
[2026-05-19 00:31:12] The server is fired up and ready to roll!
```

### 2. Port Forward to Local Environment

Expose the service port locally:

```bash
kubectl port-forward svc/sglang-tpu-service 30000:30000
```

### 3. Send a Test Request

Send a chat completion request to the Qwen model (see example below):

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Hello, tell me a joke."}]
  }'
```

You should receive an instant response:

```json
{
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Why don't scientists trust atoms?\n\nBecause they make up everything!"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 36,
    "completion_tokens": 24,
    "total_tokens": 60
  }
}
```

## Cleaning Up

To avoid ongoing charges on your Google Cloud account, ensure you clean up all resources when you are finished.

### 1. Delete the Kubernetes Workload

Delete the deployment and service in one step by running:

```bash
kubectl delete -f sglang-tpu-deployment.yaml
```

### 2. Delete the GKE Cluster

Deleting the GKE cluster will automatically delete the TPU node pool and all associated VM resources:

```bash
gcloud container clusters delete $CLUSTER_NAME \
  --location $ZONE \
  --project $PROJECT_ID
```
*(You will be prompted to confirm the deletion).*

## Friction Points & Key Insights for GKE TPU Serving

### 1. TPU Resource Deadlock during Rolling Updates

*   **The Issue**: By default, Kubernetes Deployments use the `RollingUpdate` strategy, creating a new pod *before* terminating the old one. On a single-host TPU cluster, the new pod will remain `Pending` forever with `Insufficient google.com/tpu` because the old pod is still holding the TPU device.
*   **The Fix**: Always set `strategy: type: Recreate` in your deployment. This terminates the old pod first, fully releasing the TPU hardware, before starting the new one.

### 2. JAX JIT Cancellation Crashes (Readiness Probes)

*   **The Issue**: JAX JIT compilation (especially FlashAttention Pallas kernels) is CPU-heavy and sequential. If you configure a strict readiness probe (eg: `/health_generate` running a 1-token generation) with a short timeout (default 1s), Kubernetes will timeout and disconnect the request while it's JIT compiling.
*   FastAPI propagates this client disconnect as an `asyncio.exceptions.CancelledError`. JAX JIT operations are not safe to cancel mid-way; cancelling an active compilation corrupts the JAX/XLA compiler state, throwing `CancelledError` across the entire process and crashing the server.
*   **The Fix**:
    *   Use the fast `/health` endpoint (which does not JIT) for readiness probes during development.
    *   If using `/health_generate` for production-grade readiness, ensure you set a large `timeoutSeconds: 120` (2 minutes) and a high `initialDelaySeconds: 300` (5 minutes) to prevent Kubernetes from disconnecting active compiler tasks.

### 3. Naming Gotcha: --tp-size is actually World Size

*   **The Issue**: In `sglang-jax`, the `--tp-size` CLI argument represents the total number of TPU devices (world size) you wish to allocate to the job, *not* the final Tensor Parallelism size.
*   **The Math**: The actual Tensor Parallel size is calculated internally as `tp_size // dp_size`.
*   **The Divisibility Rule**: The model's number of attention heads must be divisible by the *actual* Tensor Parallel size. For `Qwen2.5-7B` (28 heads), an actual TP size of 8 is invalid (28 % 8 != 0), causing an assertion crash.
*   **The Fix**: As an example, to utilize all 8 chips of `ct5lp-hightpu-8t` for Qwen 7B, configure `--tp-size 8 --dp-size 2`. This correctly yields `dp_size = 2` and `actual_tp_size = 4` (which divides 28 perfectly).

### 4. Silent Python Buffering

*   **The Issue**: By default, Python buffers standard output in Docker containers. When JAX is performing heavy initialization or downloading weights, `kubectl logs` will show absolutely nothing, giving the impression the container is hung.
*   **The Fix**: Always launch the python server with the `-u` flag (`python3 -u -m sgl_jax.launch_server`) to disable buffering and stream logs in real-time.

### 5. Watchdog Timeout Crash

*   **The Issue**: SGLang has an internal `--watchdog-timeout` (default 300s / 5 minutes). If JIT compilation of the warmup request (prefill + decode phases) takes longer than 5 minutes total, the watchdog assumes the process is hung and aggressively kills the server.
*   **The Fix**: Configure `--watchdog-timeout 1200` (20 minutes) to allow JIT compilation to complete safely.
