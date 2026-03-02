# Run in Pathways

**Note**:
Before you execute the following script, please ensure your google account has been a member of PROJECT. Contact with @Prayer3th.

## Prerequisites

Ensure the following tools are installed and configured before proceeding.

### Core Tools

* **Python 3.10+**: Ensure `pip` and `venv` are included.
    * *Check:* `python3 --version`
* **Google Cloud SDK (gcloud)**: [Install from here](https://cloud.google.com/sdk/docs/install).
    * Run `gcloud init`
    * [Authenticate](https://cloud.google.com/sdk/gcloud/reference/auth/application-default/login) to Google Cloud.
    * *Check:* `gcloud auth list`
* **Kubectl**: [Install from here](https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_kubectl).
    * Install the auth plugin: `gke-gcloud-auth-plugin` ([Guide](https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_plugin)).
    * *Check:* `kubectl version --client`
* **Docker**: [Install from here](https://docs.docker.com/engine/install/).
    * *Linux users:* [Configure sudoless docker](https://docs.docker.com/engine/install/linux-postinstall/).
    * Run `gcloud auth configure-docker` to enable image uploads to the registry.
* **Xpk**: [Install from here](https://github.com/AI-Hypercomputer/xpk/blob/main/docs/installation.md)
    * *Check:* `xpk --help`

## Create Cluster

```zsh
export PROJECT_ID=tpu-service-473302
export CLUSTER_NAME=rl-pathway
export TPU_TYPE=v6e-16
export NUM_SLICES=1
export ZONE=asia-northeast1-b

xpk cluster create-pathways \
--cluster $CLUSTER_NAME \
--num-slices=$NUM_SLICES \
--tpu-type=$TPU_TYPE \
--zone=$ZONE \
--spot 2>&1 | tee .xpk_creation.log
```

## Build Docker images and push to remote registry

```zsh
make push
```

The command will print the full image path at the end, for example:

```
Pushed image: asia-northeast1-docker.pkg.dev/tpu-service-473302/sglang-project/sglang-jax:20260302-152303-hongmao
```

Copy the image tag (the part after `:`) and use it in the next step.

## Create workload

> **Note**: The `--command` value must be a single line with no line breaks. Shell line breaks will truncate the command and cause the workload to fail.

Replace `<IMAGE_TAG>` with the tag printed by `make push`.

```zsh
export CLUSTER_NAME=rl-pathway
export TPU_TYPE=v6e-16
export NUM_SLICES=1
export ZONE=asia-northeast1-b

xpk workload create-pathways \
  --workload rl-pathway \
  --num-slices=$NUM_SLICES \
  --tpu-type=$TPU_TYPE \
  --cluster=$CLUSTER_NAME \
  --zone=$ZONE \
  --docker-name='rl-pathway-workload' \
  --docker-image="asia-northeast1-docker.pkg.dev/tpu-service-473302/sglang-project/sglang-jax:<IMAGE_TAG>" \
  --command="JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 JAX_USE_SHARDY_PARTITIONER=0 python3 -u -m sgl_jax.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --trust-remote-code --tp-size=4 --mem-fraction-static=0.8 --chunked-prefill-size=2048 --download-dir=/tmp --dtype=bfloat16 --max-running-requests 8 --skip-server-warmup --page-size=64 --max-total-tokens=257536 --random-seed=27 --precompile-token-paddings=2048 --precompile-bs-paddings=8 --enable-single-process"
```

To resubmit, delete the existing workload first (requires the environment variables above to be set):

```zsh
export CLUSTER_NAME=rl-pathway
export ZONE=asia-northeast1-b

xpk workload delete --workload rl-pathway --cluster $CLUSTER_NAME --zone $ZONE
```

## Verify the Deployment

### 1. Check pod status

```zsh
kubectl get pods
```

Expected output:

```
NAME                                 READY   STATUS    RESTARTS   AGE
rl-pathway-pathways-head-0-0-rqx6k   3/3     Running   0          93s
rl-pathway-worker-0-0-2rfbr          1/1     Running   0          89s
rl-pathway-worker-0-1-2f94j          1/1     Running   0          88s
rl-pathway-worker-0-2-c8qw5          1/1     Running   0          88s
rl-pathway-worker-0-3-8vkrb          1/1     Running   0          88s
```

### 2. Forward the server port

Replace `<HEAD_POD_NAME>` with the actual pod name from the output above (the one with `3/3` ready containers).

```zsh
kubectl port-forward pod/<HEAD_POD_NAME> 30000:30000
```

Expected output:

```
Forwarding from 127.0.0.1:30000 -> 30000
Forwarding from [::1]:30000 -> 30000
```

### 3. Send a test request

```zsh
curl -X POST 'http://127.0.0.1:30000/generate' \
  -H 'Content-Type: application/json' \
  -d '{"text": "the capital of France is", "sampling_params": {"max_new_tokens": 10, "temperature": 0.6, "top_k": 10, "top_p": 0.9, "min_p": 0.6}}' \
  | jq
```

Expected output:

```json
{
  "text": " the capital of the capital of France.\n\nIs this",
  "output_ids": [279, 6722, 315, 279, 6722, 315, 9625, 382, 3872, 419],
  "meta_info": {
    "id": "1cf58ed7733c4a688c63fcb9e5ed7559",
    "finish_reason": {
      "type": "length",
      "length": 10
    },
    "prompt_tokens": 6,
    "completion_tokens": 10,
    "cached_tokens": 0,
    "routed_experts": null,
    "cache_miss_count": 0,
    "e2e_latency": 0.1346907615661621
  }
}
```