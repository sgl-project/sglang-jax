# How to Run SGLang-Jax Tests on GKE TPU via Agent

## Introduction

During SGLang-Jax development, you need to run unit tests and end-to-end (E2E) tests on TPUs to verify code correctness. Manually managing GKE clusters, deploying environments, and executing test commands one by one is tedious and error-prone.

This guide walks you through using the `exec-remote` skill plugin to automate the entire workflow — from GKE TPU cluster creation to test execution — via an AI Agent (such as Claude Code). After completing this guide, you will be able to:

- Install and configure the `exec-remote` skill plugin
- Automatically create GKE TPU clusters via the Agent
- Run unit tests and E2E tests on remote TPUs
- Manage and clean up cluster resources

## Prerequisites

Before you begin, make sure you have the following:

- A configured GCP account with access to the project `tpu-service-473302`.
- [Google Cloud SDK (`gcloud`)](https://cloud.google.com/sdk/docs/install) installed locally and authenticated (`gcloud auth login`).
- [kubectl](https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl) installed locally. Verify with:

```bash
kubectl version --client
```

- [xpk](https://github.com/AI-Hypercomputer/xpk) installed locally, used for creating GKE TPU clusters. Verify with:

```bash
xpk --help
```

- [SkyPilot](https://skypilot.readthedocs.io/) installed and configured locally. Verify with:

```bash
sky --help
```

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) or another AI Agent that supports skill plugins installed locally.
- The SGLang-Jax repository cloned locally with the development environment set up. See the [Contribution Guide](../contribution_guide.md#setup-environment-for-sgl-jax) for details.

## Step 1 — Install the exec-remote Skill Plugin

In this step, you will install the `exec-remote` skill plugin. This plugin gives the Agent the ability to execute code on remote GPU/TPU clusters.

Open a terminal in your project directory and run the following commands to register `primatrix/skills` as a plugin marketplace and install `exec-remote`:

```bash
# Add the community plugin marketplace
/plugin marketplace add primatrix/skills

# Install the exec-remote plugin (project scope)
/plugin install exec-remote@primatrix-skills --scope project
```

After installation, verify the plugin is loaded correctly by running the following in a Claude Code session:

```bash
/exec-remote
```

If the plugin was installed successfully, the Agent will recognize the skill and use it automatically in subsequent tasks.

With the plugin installed, you next need to understand the full test execution pipeline.

## Step 2 — Understand the GKE TPU Test Pipeline

Before running tests, you need to understand the three-stage pipeline that `exec-remote` uses on GKE:

```
apply-resource   →  Create a GKE cluster with TPU node pools via xpk
                     Poll until cluster status becomes RUNNING
deploy-cluster   →  Configure and launch SkyPilot on GKE
                     Write the .cluster_name_tpu file
exec-remote      →  Sync local directory and run test scripts on the remote TPU
```

**The `apply-resource` stage** uses `xpk cluster create-pathways --spot` to create a GKE cluster on GCP and automatically provisions TPU node pools. The Agent continuously polls `gcloud container clusters list` until the cluster status changes from `RECONCILING` or `PROVISIONING` to `RUNNING`.

> **Note:** Do not proceed to subsequent steps while the cluster is in `RECONCILING` or `PROVISIONING` status, as this will cause SSL connection errors. The Agent handles this waiting logic automatically.

**The `deploy-cluster` stage** deploys SkyPilot on the ready GKE cluster and writes the cluster name to the `.cluster_name_tpu` file. This file serves as the integration point between stages.

**The `exec-remote` stage** uses `sky exec` to sync the local working directory to the remote cluster and execute the specified test script.

Now that you understand the overall pipeline, you will use a structured prompt to have the Agent execute the tests.

## Step 3 — Run Unit Tests and E2E Tests

### Available Test Suites

`test/srt/run_suite.py` defines several test suites. The commonly used unit and E2E suites include:

| Suite Name | TPU Type | Description |
|------------|----------|-------------|
| `unit-test-tpu-v6e-1` | v6e-1 | Unit tests: flash attention, MoE, sampler, KV cache, LoRA, etc. |
| `unit-test-tpu-v6e-4` | v6e-4 | Unit tests: mesh-related |
| `e2e-test-tpu-v6e-1` | v6e-1 | E2E tests: OpenAI protocol, SRT engine, LoRA, etc. |
| `e2e-test-tpu-v6e-4` | v6e-4 | E2E tests: tool calling, chunked prefill, quantization, multimodal, etc. |

### Run a Test Suite via Prompt

You can use the following structured prompt to have the Agent run a complete test suite on a remote TPU. The prompt includes config template selection — the Agent will automatically use the correct `~/.sky/config.yaml` based on the TPU type:

- **v6e-1 (single chip)**: Uses [`config_v6e_1.yaml`](./config_v6e_1.yaml), without `JAX_COMPILATION_CACHE_DIR`, no Pod Affinity
- **v6e-4 and above (multi-chip)**: Uses [`config.yaml`](./config.yaml), with `JAX_COMPILATION_CACHE_DIR`, Pod Affinity, and multi-node Worker Hostname derivation

> **[Context]**
> I'm developing SGLang-JAX and need to run tests on a remote TPU. The GCP project is `tpu-service-473302`, zone is `asia-northeast1-b`. The test suite I want to run is `unit-test-tpu-v6e-1`, which requires a v6e-1 TPU. The SkyPilot pod config template to use is `docs/developer_guide/remote-testing/config_v6e_1.yaml` (for v6e-1) or `docs/developer_guide/remote-testing/config.yaml` (for v6e-4+). The test command is:
> ```
> python test/srt/run_suite.py --suite unit-test-tpu-v6e-1
> ```
>
> **[Objective]**
> Create a GKE TPU cluster (name: sglang-jax-agent-tests, if one doesn't already exist), deploy the environment, and run the specified test suite on a remote v6e-1 TPU. Use the `exec-remote` skill to handle the full pipeline: `apply-resource` → `deploy-cluster` → `exec-remote`.
>
> **[Style]**
> Step-by-step automated execution. After `xpk` creates the GKE cluster, poll `gcloud container clusters list` until status is `RUNNING` — do NOT proceed while `RECONCILING` or `PROVISIONING` (deploying in these states causes SSL errors).
>
> If a **new** cluster was created, enable GCS model storage support before deploying SkyPilot (skip if the cluster already has these configured):
> 1. Enable Workload Identity on the cluster: `gcloud container clusters update <CLUSTER> --region=<REGION> --project=tpu-service-473302 --workload-pool=tpu-service-473302.svc.id.goog`
> 2. Enable GCSFuse CSI driver: `gcloud container clusters update <CLUSTER> --region=<REGION> --project=tpu-service-473302 --update-addons=GcsFuseCsiDriver=ENABLED`
> 3. Enable Workload Identity on each **new** TPU node pool: `gcloud container node-pools update <POOL> --cluster=<CLUSTER> --region=<REGION> --project=tpu-service-473302 --workload-metadata=GKE_METADATA`
> 4. Bind KSA to GSA (only needed once per cluster): `gcloud iam service-accounts add-iam-policy-binding 785128357837-compute@developer.gserviceaccount.com --role roles/iam.workloadIdentityUser --member "serviceAccount:tpu-service-473302.svc.id.goog[default/skypilot-service-account]" --project=tpu-service-473302` and `kubectl annotate serviceaccount skypilot-service-account --overwrite iam.gke.io/gcp-service-account=785128357837-compute@developer.gserviceaccount.com`
>
> Then deploy the cluster and execute the test.
>
> **[Tone]**
> Proactive — execute each step automatically and report progress.
>
> **[Audience]**
> Developer running tests to validate code changes before submitting a PR.
>
> **[Response]**
> After each stage, briefly report the result and verify success before proceeding. When the test finishes, report the pass/fail summary. If any step fails, diagnose the error and suggest a fix before continuing.

After sending the prompt, the Agent will automatically complete the entire workflow of cluster creation, environment deployment, and test execution. Replace the `--suite` parameter with the suite name you need to run different tests.

> **Tip:** If you only need to run a single test file instead of an entire suite, replace the test command with `python <test_file_path>`, for example `python python/sgl_jax/test/test_flashattention.py`.

### Run a Single Test File via Prompt

If you only modified a specific module, you can run just the corresponding test file:

> **[Context]**
> I'm developing SGLang-JAX and need to run a single test file on a remote TPU. The GCP project is `tpu-service-473302`, zone is `asia-northeast1-b`. The test file is `python/sgl_jax/test/test_flashattention.py`, which requires a v6e-1 TPU. The SkyPilot pod config template to use is `docs/developer_guide/remote-testing/config_v6e_1.yaml`.
>
> **[Objective]**
> Run the specified test file on a remote v6e-1 TPU using the `exec-remote` skill. Create the cluster(name: sglang-jax-agent-tests) if it doesn't exist.
>
> **[Style]**
> Step-by-step automated execution. Poll cluster status until `RUNNING` before proceeding. If a **new** cluster or node pool was created, enable GCS model storage support before deploying SkyPilot (skip if already configured):
> 1. Enable Workload Identity on the cluster: `gcloud container clusters update <CLUSTER> --region=<REGION> --project=tpu-service-473302 --workload-pool=tpu-service-473302.svc.id.goog`
> 2. Enable GCSFuse CSI driver: `gcloud container clusters update <CLUSTER> --region=<REGION> --project=tpu-service-473302 --update-addons=GcsFuseCsiDriver=ENABLED`
> 3. Enable Workload Identity on each **new** TPU node pool: `gcloud container node-pools update <POOL> --cluster=<CLUSTER> --region=<REGION> --project=tpu-service-473302 --workload-metadata=GKE_METADATA`
> 4. Bind KSA to GSA (only needed once per cluster): `gcloud iam service-accounts add-iam-policy-binding 785128357837-compute@developer.gserviceaccount.com --role roles/iam.workloadIdentityUser --member "serviceAccount:tpu-service-473302.svc.id.goog[default/skypilot-service-account]" --project=tpu-service-473302` and `kubectl annotate serviceaccount skypilot-service-account --overwrite iam.gke.io/gcp-service-account=785128357837-compute@developer.gserviceaccount.com`
>
> **[Tone]**
> Proactive — execute each step automatically and report progress.
>
> **[Audience]**
> Developer validating a specific code change.
>
> **[Response]**
> Report the test pass/fail result. If the test fails, show the relevant error output.

## Step 4 — View Test Logs

When the Agent runs tests via `sky exec`, logs are streamed in real time to the terminal. However, if your Agent session has ended, or you want to view logs independently in another terminal, SkyPilot provides several options.

### View the Task Queue

First, use `sky queue` to see the status and Job IDs of all tasks on the cluster:

```bash
sky queue <cluster_name>
```

Example output:

```
Job queue of current user on cluster sglang-unit-test
ID  NAME     SUBMITTED    STARTED      DURATION  RESOURCES   STATUS     LOG
2   sky-cmd  13 mins ago  13 mins ago  9m 47s    1x[CPU:1+]  SUCCEEDED  ~/sky_logs/sky-2026-03-11-16-13-31-101180
1   sky-cmd  14 mins ago  14 mins ago  1s        1x[CPU:1+]  SUCCEEDED  ~/sky_logs/sky-2026-03-11-16-12-44-239736
```

`STATUS` column meanings:
- `RUNNING` — Task is currently executing
- `SUCCEEDED` — Task completed successfully
- `FAILED` — Task execution failed

### Stream Live Logs

For running tasks, use `sky logs` to follow the output in real time (similar to `tail -f`):

```bash
# Follow the latest task's logs
sky logs <cluster_name>

# Follow logs for a specific Job ID
sky logs <cluster_name> <job_id>
```

Press `Ctrl+C` to exit the log stream without terminating the remote task.

### Download Historical Logs

For completed tasks, use `--sync-down` to download logs to the local `~/sky_logs/` directory:

```bash
sky logs --sync-down <cluster_name> <job_id>
```

After downloading, you can view the full logs with any text editor or `less`.

### SSH into the Remote Cluster

For deeper debugging (e.g., inspecting the remote filesystem, checking TPU status), you can SSH directly:

```bash
ssh <cluster_name>
```

Once logged in, the working directory is `~/sky_workdir/`, which is synced with the local project.

## Step 5 — Clean Up Resources

After testing is complete, make sure to clean up cluster resources to avoid unnecessary charges. You can clean up in two ways:

**Option 1: Let the Agent clean up**

Simply tell the Agent: "Please delete the GKE cluster that was just created." The Agent will automatically perform the cleanup.

**Option 2: Manual cleanup**

If you need to manually delete the cluster, run the following commands:

```bash
# Delete the SkyPilot cluster
sky down <cluster_name>

# Delete the GKE cluster (if no longer needed)
xpk cluster delete --cluster <cluster_name> --project tpu-service-473302
```

Replace `<cluster_name>` with your cluster name.

> **Warning:** If you used Spot instances (`--spot`), the cluster may be automatically reclaimed by GCP. However, charges will still accrue until it is reclaimed. Make sure to clean up resources promptly after testing.

## Summary

In this guide, you completed the following:

1. Installed the `exec-remote` skill plugin, giving the Agent remote execution capabilities.
2. Understood the three-stage GKE TPU test pipeline.
3. Ran unit tests and E2E tests on remote TPUs.
4. Used `sky queue`, `sky logs`, and `ssh` to view and follow test logs.
5. Cleaned up cluster resources to avoid unnecessary charges.

---

## Appendix A — SkyPilot Pod Config Templates

This appendix describes the Pod config templates used by the `deploy-cluster` stage to generate `~/.sky/config.yaml`. During normal usage, the Agent automatically selects the correct template based on TPU type; the information below is helpful when you need to manually configure or troubleshoot.

### Config Template Files

This directory provides two SkyPilot Pod config templates, selected based on TPU type:

| Template File | Applicable TPU Types | Key Differences |
|---------------|---------------------|-----------------|
| [`config.yaml`](./config.yaml) | v6e-4 and above (multi-chip) | Includes `JAX_COMPILATION_CACHE_DIR`, Pod Affinity, multi-node Worker Hostname derivation |
| [`config_v6e_1.yaml`](./config_v6e_1.yaml) | v6e-1 (single chip) | No `JAX_COMPILATION_CACHE_DIR`, no Pod Affinity |

Both templates share the following common configurations:
- **GCSFuse volume mount**: Mounts the GCS bucket `model-storage-sglang` to `/models/`, used by tests that need to load model weights
- **TPU environment variable injection**: Writes TPU environment variables to `/etc/profile.d/tpu-env.sh` via a `postStart` lifecycle hook

Placeholders in the templates (`<ACCELERATOR_TYPE>`, `<TPU_TOPOLOGY>`, etc.) are automatically replaced by `deploy.py` based on TPU type. See the table below for placeholder values:

| TPU Type | ACCELERATOR_TYPE | TPU_TOPOLOGY | CHIPS_PER_HOST | CPU_REQUEST | MEMORY_REQUEST |
|----------|-----------------|-------------|----------------|-------------|---------------|
| v6e-1 | tpu-v6e-slice | 1x1 | 1 | 30 | 150Gi |
| v6e-4 | tpu-v6e-slice | 2x2 | 4 | 36 | 500Gi |
| v6e-8 | tpu-v6e-slice | 2x4 | 4 | 36 | 500Gi |
| v6e-16 | tpu-v6e-slice | 4x4 | 4 | 36 | 500Gi |

### GCS Model Storage Mount

The `/models/` directory on CI runners points to a GCS bucket with pre-downloaded model weights. Some tests (e.g., `test_wan_vae_precision.py`, `test_vae_scheduler.py`) depend on this path to load models.

Both config templates mount the GCS bucket to `/models/` via the [GCSFuse CSI driver](https://cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/cloud-storage-fuse-csi-driver). This requires the following GKE cluster-level configuration (only needed once for new clusters):

**1. Enable Workload Identity**

```bash
# Cluster level
gcloud container clusters update <CLUSTER_NAME> \
  --region=<REGION> \
  --project=<PROJECT> \
  --workload-pool=<PROJECT>.svc.id.goog

# Node pool level (must be enabled for each node pool that needs the mount)
gcloud container node-pools update <NODE_POOL_NAME> \
  --cluster=<CLUSTER_NAME> \
  --region=<REGION> \
  --project=<PROJECT> \
  --workload-metadata=GKE_METADATA
```

**2. Enable GCSFuse CSI Driver**

```bash
gcloud container clusters update <CLUSTER_NAME> \
  --region=<REGION> \
  --project=<PROJECT> \
  --update-addons=GcsFuseCsiDriver=ENABLED
```

**3. Bind Kubernetes Service Account to GCP Service Account**

SkyPilot uses the `skypilot-service-account` KSA. It needs to be bound to a GSA that has GCS read permissions:

```bash
# Grant the KSA permission to impersonate the GSA
gcloud iam service-accounts add-iam-policy-binding <GSA_EMAIL> \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:<PROJECT>.svc.id.goog[default/skypilot-service-account]" \
  --project=<PROJECT>

# Annotate the KSA
kubectl annotate serviceaccount skypilot-service-account \
  --overwrite \
  iam.gke.io/gcp-service-account=<GSA_EMAIL>
```

## Appendix B — Troubleshooting

### SSL Connection Errors

**Symptom:** `sky launch` reports `SSL: UNEXPECTED_EOF_WHILE_READING` error.

**Cause:** After a GKE cluster is deleted and recreated, the SkyPilot API server caches stale TLS state.

**Solution:**

```bash
sky api stop
sky api start
```

### Insufficient Pod Resources

**Symptom:** `sky launch` reports `Insufficient google.com/tpu` or insufficient memory.

**Cause:** Resource requests in `~/.sky/config.yaml` exceed the node's allocatable capacity. v6e-1 nodes (`ct6e-standard-1t`) have only about 44 vCPUs and 163 GB of allocatable memory.

**Solution:** Verify that `deploy-cluster` is using the correct config template. v6e-1 should use `config_v6e_1.yaml` (cpu: 30, memory: 150Gi), not `config.yaml` (cpu: 36, memory: 500Gi).

### GCSFuse Mount Failure

**Symptom:** Pod status is `Init:0/1`, events show `FailedMount`.

**Common causes and solutions:**

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `PermissionDenied: storage.objects.list` | KSA is not bound to a GSA with GCS permissions | Follow Appendix A to configure Workload Identity binding |
| `Workload Identity Federation is not enabled on node` | Node pool does not have Workload Identity enabled | Run `gcloud container node-pools update ... --workload-metadata=GKE_METADATA` |
| `fusermount3: exit status 1` | Running gcsfuse directly inside the container (no fuse device) | Use the GCSFuse CSI driver instead of direct mounting |

### `/models/` Path Not Found in Tests

**Symptom:** `HFValidationError: Repo id must be in the form 'repo_name'...`

**Cause:** Test code has hardcoded `/models/Wan-AI/...` paths. This directory is pre-mounted on CI runners, but is not available on SkyPilot nodes by default.

**Solution:** Ensure `~/.sky/config.yaml` uses a config template that includes the GCSFuse volume mount ([`config.yaml`](./config.yaml) or [`config_v6e_1.yaml`](./config_v6e_1.yaml)), which mounts the GCS bucket to `/models/`.
