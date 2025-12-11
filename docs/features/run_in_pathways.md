# Run in Pathways

## Single Host

Note:
1. Before you execute the following script, please ensure your google account has been a member of PROJECT. Contact with @Prayer3th.
2. You can execute the `Setup Pathways` on CPU VM, and you will omit to configure on your personal computer.

### Setup Pathways

**Step1: Install gcloud and configure it**

Host: your personal computer.

- Install gcloud: https://docs.cloud.google.com/sdk/docs/install-sdk?hl=zh-cn#deb
- Login with your google account: `gcloud auth login`

**Step2: Setup GKE**

Host: your personal computer.

```bash
CLUSTER=pathways
PROJECT=tpu-service-473302
ZONE=us-east5-a
REGION=us-east5
CLUSTER_VERSION="1.33.5-gke.1308000"
PW_CPU_MACHINE_TYPE="n2-standard-64" # use `gcloud alpha compute machine-types` to get more types
CLUSTER_NODEPOOL_COUNT=1
TPU_MACHINE_TYPE="ct6e-standard-4t"
WORKERS_PER_SLICE=1
TOPOLOGY="2x2"
NUM_CPU_NODES=2
RESERVATION_ID="reservation-20251114-082511"

gcloud beta container clusters create ${CLUSTER} \
--project=${PROJECT} \
--zone=${ZONE} \
--cluster-version=${CLUSTER_VERSION} \
--scopes=storage-full,gke-default,cloud-platform \
--machine-type ${PW_CPU_MACHINE_TYPE}


for i in $(seq 1 ${CLUSTER_NODEPOOL_COUNT}); do
gcloud container node-pools create "tpu-np-${i}" \
--project=${PROJECT} \
--zone=${ZONE} \
--cluster=${CLUSTER} \
--machine-type=${TPU_MACHINE_TYPE} \
--num-nodes=${WORKERS_PER_SLICE} \
--placement-type=COMPACT \
--tpu-topology=${TOPOLOGY} \
--scopes=storage-full,gke-default,cloud-platform \
--workload-metadata=GCE_METADATA \
--reservation=$RESERVATION_ID \
--reservation-affinity=specific
done



gcloud container node-pools create "cpu-pathways-np" \
--project ${PROJECT} \
--zone ${ZONE} \
--cluster ${CLUSTER} \
--machine-type ${PW_CPU_MACHINE_TYPE} \
--num-nodes ${NUM_CPU_NODES} \
--scopes=storage-full,gke-default,cloud-platform \
--workload-metadata=GCE_METADATA



gcloud container clusters get-credentials ${CLUSTER} \
--zone=${ZONE} \
--project=${PROJECT} && kubectl config set-context --current --namespace=default



kubectl apply --server-side -f https://github.com/kubernetes-sigs/jobset/releases/download/v0.8.0/manifests.yaml
kubectl apply --server-side -f https://github.com/google/pathways-job/releases/download/v0.1.2/install.yaml
```

**Step3: Apply PathwaysJob**

Host: your personal computer.

Execute `kubectl apply -f pathwaysjob.yaml`

```yaml
# pathwaysjob.yaml
apiVersion: pathways-job.pathways.domain/v1
kind: PathwaysJob
metadata:
  name: pathways-rl  # USERNAME
spec:
  maxRestarts: 3 # MAX_RESTARTS
  workers:
    - type: ct6e-standard-4t # TPU_MACHINE_TYPE
      topology: 2x2 # TOPOLOGY, refer to https://docs.cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus?hl=zh-cn#autopilot_2
      numSlices: 1 # WORKLOAD_NODEPOOL_COUNT
  pathwaysDir: "gs://pathways_rl_tmp" # BUCKET_NAME
  controller:
    deploymentMode: default
```


### Run SGLangJax in Interactive Mode

Note: Refer to [Run an interactive workload with Pathways](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-interactive-mode).

**Step1: Launch a CPU VM in the same region**

```yaml
name: pathways-cpu

resources:
  cloud: gcp
  instance_type: e2-standard-16 # 16core x 64GB
  region: us-east5

setup: |
  sudo apt-get update -y
  sudo apt-get install -y apt-transport-https ca-certificates gnupg curl
  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
  sudo apt-get update -y && sudo apt-get install -y google-cloud-cli google-cloud-sdk-gke-gcloud-auth-plugin
  curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && chmod +x kubectl && mv ./kubectl ~/.local/bin/kubectl

```

**Step2: Configure for gcloud and kubectl**

Execute `gcloud init`. You are required to configure some information like account, region and so on.

Execute `gcloud container clusters get-credentials ${CLUSTER} --zone=${ZONE}` to get kube config for kubectl.

Check the configution: `kubectl get pods`, the successful results are similar to the following output:
```bash
NAME                                        READY   STATUS    RESTARTS   AGE
pathways-aolemila-pathways-head-0-0-s5zjl   2/2     Running   0          11m
pathways-aolemila-worker-0-0-ccfhs          1/1     Running   0          11m
```

**Step3: Launch SGLangJax**

```bash
JAX_PLATFORMS=proxy \
JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 \
JAX_USE_SHARDY_PARTITIONER=0 \
python3 -u -m sgl_jax.launch_server \
--model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
--trust-remote-code  \
--tp-size=4  \
--mem-fraction-static=0.8 \
--chunked-prefill-size=2048 \
--download-dir=/tmp \
--dtype=bfloat16 \
--max-running-requests 8 \
--skip-server-warmup \
--page-size=64  \
--max-total-tokens=257536 \
--random-seed=27 \
--precompile-token-paddings=2048 \
--precompile-bs-paddings=8 \
--enable-single-process
```

```bash
# curl
curl -X POST 'http://127.0.0.1:30000/generate' -d '{"sampling_params": {"max_new_tokens": 10,"temperature":0.6, "top_k":10, "top_p":0.9, "min_p":0.6}, "text": "the capital of France is"}' -H 'Content-Type: application/json'

## curl result
curl -X POST 'http://127.0.0.1:30000/generate' -d '{"sampling_params": {"max_new_tokens": 10,"temperature":0.6, "top_k":10, "top_p":0.9, "min_p":0.6}, "text": "the capital of France is"}' -H 'Content-Type: application/json'
{"text":" the capital of the capital of France.\n\nIs this","output_ids":[279,6722,315,279,6722,315,9625,382,3872,419],"meta_info":{"id":"1b990e673c494324b1a2b6a9e5e3cc0a","finish_reason":{"type":"length","length":10},"prompt_tokens":6,"completion_tokens":10,"cached_tokens":0,"cache_miss_count":0,"e2e_latency":2.1345818042755127}}
```

## Multi Hosts in the Pod

Replace **Setup Pathways.Step2: Setup GKE** with the following script.

```bash
CLUSTER=pathways
PROJECT=tpu-service-473302
ZONE=us-east5-a
REGION=us-east5
CLUSTER_VERSION="1.33.5-gke.1308000"
PW_CPU_MACHINE_TYPE="n2-standard-64" # use `gcloud alpha compute machine-types` to get more types
CLUSTER_NODEPOOL_COUNT=1
TPU_MACHINE_TYPE="ct6e-standard-4t"
WORKERS_PER_SLICE=4
TOPOLOGY="4x4"
NUM_CPU_NODES=1
RESERVATION_ID="reservation-20251114-082511"

gcloud beta container clusters create ${CLUSTER} \
--project=${PROJECT} \
--zone=${ZONE} \
--cluster-version=${CLUSTER_VERSION} \
--scopes=storage-full,gke-default,cloud-platform \
--machine-type ${PW_CPU_MACHINE_TYPE}


for i in $(seq 1 ${CLUSTER_NODEPOOL_COUNT}); do
gcloud container node-pools create "tpu-np-${i}" \
--project=${PROJECT} \
--zone=${ZONE} \
--cluster=${CLUSTER} \
--machine-type=${TPU_MACHINE_TYPE} \
--num-nodes=${WORKERS_PER_SLICE} \
--placement-type=COMPACT \
--tpu-topology=${TOPOLOGY} \
--scopes=storage-full,gke-default,cloud-platform \
--workload-metadata=GCE_METADATA \
--reservation=$RESERVATION_ID \
--reservation-affinity=specific
done



gcloud container node-pools create "cpu-pathways-np" \
--project ${PROJECT} \
--zone ${ZONE} \
--cluster ${CLUSTER} \
--machine-type ${PW_CPU_MACHINE_TYPE} \
--num-nodes ${NUM_CPU_NODES} \
--scopes=storage-full,gke-default,cloud-platform \
--workload-metadata=GCE_METADATA



gcloud container clusters get-credentials ${CLUSTER} \
--zone=${ZONE} \
--project=${PROJECT} && kubectl config set-context --current --namespace=default



kubectl apply --server-side -f https://github.com/kubernetes-sigs/jobset/releases/download/v0.8.0/manifests.yaml
kubectl apply --server-side -f https://github.com/google/pathways-job/releases/download/v0.1.2/install.yaml
```

Replace **Setup Pathways.Step3: Apply PathwaysJob** with the following yaml.

Execute `kubectl apply -f pathwaysjob.yaml`

```yaml
# pathwaysjob.yaml
apiVersion: pathways-job.pathways.domain/v1
kind: PathwaysJob
metadata:
  name: pathways-rl  # USERNAME
spec:
  maxRestarts: 3 # MAX_RESTARTS
  workers:
    - type: ct6e-standard-4t # TPU_MACHINE_TYPE
      topology: 4x4 # TOPOLOGY, refer to https://docs.cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus?hl=zh-cn#autopilot_2
      numSlices: 1 # WORKLOAD_NODEPOOL_COUNT
  pathwaysDir: "gs://pathways_rl_tmp" # BUCKET_NAME
  controller:
    deploymentMode: default
```
