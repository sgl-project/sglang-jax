# Tutorial: Deploying sglang-jax Multi-Host on GKE with LWS

This tutorial guides you through deploying `sglang-jax` on a multi-host TPU v7x-16 cluster using Kubernetes LeaderWorkerSet (LWS).

## Prerequisites

- **GKE Cluster**: A cluster with a TPU v7x-16 nodepool (2x2x2 topology).
- **LWS Installed**: The [LeaderWorkerSet](https://github.com/kubernetes-sigs/lws) controller must be installed in your cluster.
- **Docker Registry**: Access to a registry to push the custom `sglang-jax` image.
- **Kubectl**: Configured to access your cluster.
- **Hugging Face Token**: A Kubernetes secret named `hf-secret` containing your HF token.
    ```bash
    kubectl create secret generic hf-secret --from-literal=hf_api_token=YOUR_HF_TOKEN
    ```

## Architecture Overview

On a **TPU 7x-16** system:
- **Topology**: 2x2x2 (8 chips total).
- **Cores**: 16 cores (2 per chip).
- **Nodes**: With standard GKE setup, this is split into **2 nodes**, each with **4 chips** (8 cores).

Note that TPU 7x is the first generation where each TPU core is exposed as a separate Jax device.

We use **LeaderWorkerSet (LWS)** to orchestrate this as a group of pods that scale together.
- **Size**: 2 replicas (1 Leader + 1 Worker) per group.
- **Leader**: Handling coordination and entrypoint (Rank 0).
- **Worker**: Joining the distributed mesh (Rank 1).

## Step 1: Prepare the Docker Image

You need an image that contains `sglang-jax` and the necessary dependencies.

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/sgl-project/sglang.git
    cd sglang
    ```

2.  **Define Image Name and Build**:
    Set your intented image registry and name.
    ```bash
    # Replace with your actual registry/image name
    export IMAGE_NAME="us-central1-docker.pkg.dev/your-project/your-repo/sglang-jax:latest"
    
    docker build -f docker/Dockerfile.tpu -t $IMAGE_NAME .
    docker push $IMAGE_NAME
    ```

## Step 2: Configure and Deploy the LWS

We will generate the `sglang-lws.yaml` manifest using the `IMAGE_NAME` variable defined in the previous step.

### Key Configurations
*   **Topology**: `cloud.google.com/gke-tpu-topology: 2x2x2`
*   **Accelerator**: `cloud.google.com/gke-tpu-accelerator: tpu7x`
*   **TP Size**: Set to **8** for `Qwen2.5-Coder-32B` (40 heads) to ensure divisibility. For other models, use 16 if heads % 16 == 0.

### Generate and Apply Manifest
Run the following command to create the manifest file and apply it.

```bash
cat <<EOF | kubectl apply -f -
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: sglang-tpu-cluster
  annotations:
    leaderworkerset.sigs.k8s.io/exclusive-topology: cloud.google.com/gke-nodepool
spec:
  replicas: 1
  leaderWorkerTemplate:
    size: 2
    restartPolicy: RecreateGroupOnPodRestart
    leaderTemplate:
      metadata:
        labels:
          role: leader
      spec:
        nodeSelector:
          cloud.google.com/gke-tpu-accelerator: tpu7x
          cloud.google.com/gke-tpu-topology: 2x2x2
        containers:
          - name: sglang-leader
            image: $IMAGE_NAME
            env:
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-secret
                  key: hf_api_token
            command:
              - bash
              - -c
              - |
                set -x
                export NODE_RANK=0
                # Use LWS provided leader address or hostname
                if [ -z "\$LWS_LEADER_ADDRESS" ]; then
                  DIST_HOST=\$(hostname -i)
                else
                  DIST_HOST=\$LWS_LEADER_ADDRESS
                fi
                
                python3 -m sgl_jax.launch_server \\
                  --model-path "Qwen/Qwen2.5-Coder-32B-Instruct" \\
                  --tp 8 \\
                  --nnodes 2 \\
                  --node-rank 0 \\
                  --dist-init-addr "\${DIST_HOST}:20000" \\
                  --host 0.0.0.0 \\
                  --port 8000 \\
                  --trust-remote-code
            resources:
              limits:
                google.com/tpu: "4"
            ports:
              - containerPort: 8000
            readinessProbe:
              tcpSocket:
                port: 8000
              initialDelaySeconds: 15
              periodSeconds: 10
            volumeMounts:
              - mountPath: /dev/shm
                name: dshm
              - mountPath: /root/.cache/huggingface
                name: cache-volume
        volumes:
        - name: dshm
          emptyDir:
            medium: Memory
        - name: cache-volume
          ephemeral:
            volumeClaimTemplate:
              spec:
                accessModes: [ "ReadWriteOnce" ]
                storageClassName: "premium-rwo"
                resources:
                  requests:
                    storage: 200Gi
    workerTemplate:
      spec:
        nodeSelector:
          cloud.google.com/gke-tpu-accelerator: tpu7x
          cloud.google.com/gke-tpu-topology: 2x2x2
        containers:
          - name: sglang-worker
            image: $IMAGE_NAME
            env:
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-secret
                  key: hf_api_token
            command:
              - bash
              - -c
              - |
                set -x
                # Derive Rank from hostname suffix
                HOSTNAME_SUFFIX=\${HOSTNAME##*-}
                if [[ "\$HOSTNAME_SUFFIX" =~ ^[0-9]+$ ]]; then
                  NODE_RANK=\$HOSTNAME_SUFFIX
                else
                  NODE_RANK=1
                fi
                
                if [ -z "\$LWS_LEADER_ADDRESS" ]; then
                   echo "Error: LWS_LEADER_ADDRESS missing"
                   exit 1
                fi
                
                python3 -m sgl_jax.launch_server \\
                  --model-path "Qwen/Qwen2.5-Coder-32B-Instruct" \\
                  --tp 8 \\
                  --nnodes 2 \\
                  --node-rank \$NODE_RANK \\
                  --dist-init-addr "\${LWS_LEADER_ADDRESS}:20000" \\
                  --host 0.0.0.0 \\
                  --port 8000 \\
                  --trust-remote-code
            resources:
              limits:
                google.com/tpu: "4"
            volumeMounts:
              - mountPath: /dev/shm
                name: dshm
              - mountPath: /root/.cache/huggingface
                name: cache-volume
        volumes:
        - name: dshm
          emptyDir:
            medium: Memory
        - name: cache-volume
          ephemeral:
            volumeClaimTemplate:
              spec:
                accessModes: [ "ReadWriteOnce" ]
                storageClassName: "premium-rwo"
                resources:
                  requests:
                    storage: 200Gi
---
apiVersion: v1
kind: Service
metadata:
  name: sglang-leader-tpu
spec:
  ports:
    - name: http
      port: 8000
      targetPort: 8000
  selector:
    leaderworkerset.sigs.k8s.io/name: sglang-tpu-cluster
    role: leader
  type: ClusterIP
EOF
```

## Step 3: Verification

### 1. Check Pod Status
Wait for pods to be running.
```bash
kubectl get pods -l leaderworkerset.sigs.k8s.io/name=sglang-tpu-cluster -w
```

### 2. View Logs
Confirm JAX initialization and model loading.
```bash
kubectl logs -f -l leaderworkerset.sigs.k8s.io/name=sglang-tpu-cluster -c sglang-leader
```

### 3. Test Inference
```bash
kubectl port-forward svc/sglang-leader-tpu 8000:8000 &
PID=$!
sleep 5

# Send request
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "sampling_params": {"max_new_tokens": 10}}'

# Cleanup
kill $PID
```

## Troubleshooting

-   **Pod stuck in `Pending`**: Check if your nodepool has enough resources or if the `nodeSelector` matches your TPU nodes.
-   **JAX Init Timeout**: Check logs of the worker pod. Ensure it can reach the leader's address. Verify `LWS_LEADER_ADDRESS` in env vars.
-   **AssertionError: Number of attention heads...**: This means your `tp` size is incompatible with the model. Adjust `tp` in the manifest (e.g., use 8 instead of 16 for 40-head models).
