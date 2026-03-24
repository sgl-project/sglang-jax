# Run on GKE with TPU v7x

> **Note:** Before running the following commands, make sure your Google account has been added to the project. Contact @Prayer3th for access.

## Prerequisites

Ensure the following tools are installed and configured before proceeding.

### Core Tools

* **Google Cloud SDK (gcloud)**: [Install from here](https://cloud.google.com/sdk/docs/install).
    * Run `gcloud init`
    * [Authenticate](https://cloud.google.com/sdk/gcloud/reference/auth/application-default/login) to Google Cloud.
    * *Check:* `gcloud auth list`
* **Kubectl**: [Install from here](https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_kubectl).
    * Install the auth plugin: `gke-gcloud-auth-plugin` ([Guide](https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_plugin)).
    * *Check:* `kubectl version --client`
* **Xpk**: [Install from here](https://github.com/AI-Hypercomputer/xpk/blob/main/docs/installation.md)
    * *Check:* `xpk --help`

## Create Cluster (Node Pools)

```zsh
export PROJECT_ID=tpu-service-473302
export CLUSTER_NAME=xpk-cluster
export TPU_TYPE=tpu7x-8
export NUM_SLICES=1
export ZONE=us-central1-c

xpk cluster create-pathways \
  --cluster $CLUSTER_NAME \
  --num-slices=$NUM_SLICES \
  --tpu-type=$TPU_TYPE \
  --zone=$ZONE \
  --spot 2>&1 | tee .xpk_creation.log
```

## Create Workload

> **Note**: The `--command` value must be a single line with no line breaks. Shell line breaks will truncate the command and cause the workload to fail.

```zsh
export PROJECT_ID=tpu-service-473302
export CLUSTER_NAME=xpk-cluster
export TPU_TYPE=tpu7x-8
export NUM_SLICES=1
export ZONE=us-central1-c

xpk workload create \
    --workload test-workload \
    --num-slices=$NUM_SLICES \
    --tpu-type=$TPU_TYPE \
    --cluster=$CLUSTER_NAME \
    --zone=$ZONE \
    --docker-name='test-workload' \
    --docker-image="us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu@sha256:33fd74d1ac4a45de18855cfec6c41644bf59d5b4e76a343d32f52b6553f0e804" \
    --command="sleep infinity"
```

## Verify the Deployment

Check pod status:

```zsh
kubectl get pods
```

Expected output:

```
NAME                                READY   STATUS    RESTARTS   AGE
test-workload-slice-job-0-0-26m45   2/2     Running   0          6m47s
```

Attach to the pod:

```zsh
kubectl exec -it test-workload-slice-job-0-0-26m45 -- bash
```
