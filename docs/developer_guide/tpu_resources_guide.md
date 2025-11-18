# Guide

## Skypilot Installation/Login

```bash
# Refer to https://docs.skypilot.co/en/latest/getting-started/installation.html
pip install skypilot

sky api login
```

Enter your SkyPilot API server endpoint: Please join the [SGL-JAX Slack](https://sgl-fru7574.slack.com/archives/C09EBE5HT5X) and gain it from maintainers.

## Generate a Yaml

```yaml
Example Yaml
resources:
   accelerators: tpu-v6e-4 #
   accelerator_args:
      tpu_vm: True
      runtime_version: v2-alpha-tpuv6e  # optional
file_mounts:
  ~/.ssh/id_rsa: ~/.ssh/id_rsa
setup: |
  chmod 600 ~/.ssh/id_rsa
  rm ~/.ssh/config
  GIT_SSH_COMMAND="ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" git clone git@github.com:sgl-project/sglang-jax.git
  sudo mount -t tmpfs -o size=200G tmpfs /tmp # Suggestion: This configuration is suitable to model whose parameters are less than 100B. And this directory will be used to store model which downloads from HuggingFace.
```

Note: If you want to test model whose parameters are bigger than 100B, we recommend you to use GCS. Please contact us with [Slack](https://sgl-fru7574.slack.com/archives/C09EBE5HT5X).

## Scenarios

### Start the developer computer
- `--use-spot` Use preemptible instances
- `-i 15`: Automatically stops after 15 minutes of inactivity, can be modified by yourself, but is required to be filled in
- `--down`: Directly destroy the server when stopped

If an error occurs during the creation process, you can directly check the status by viewing the developer computer below. Usually, it has already been created. If you encounter the issue of duplicate creation, you need to manage it manually.

If you log in to the developer computer via ssh, then the developer computer will not be idle, nor will it automatically shut down

```bash
sky launch test.yaml -y --use-spot --infra=gcp -i 5 --down

YAML to run: test.yaml
Running on cluster: sky-6f63-xl
⚙︎ Uploading files to API server
✓ Files uploaded  View logs: ~/sky_logs/file_uploads/sky-2025-08-05-18-09-53-882143-c194e4ef.log
Launching a spot job that does not automatically recover from preemptions. To get automatic recovery, use managed job instead: sky jobs launch or sky.jobs.launch().
Considered resources (1 node):
------------------------------------------------------------------------------------------
 INFRA                 INSTANCE       vCPUs   Mem(GB)   GPUS          COST ($)   CHOSEN
------------------------------------------------------------------------------------------
 GCP (us-central1-b)   TPU-VM[Spot]   -       -         tpu-v6e-4:1   2.26          ✔
------------------------------------------------------------------------------------------
⚙︎ Launching on GCP us-central1 (us-central1-b).
⠏ Launching  View logs: sky api logs -l sky-2025-08-05-10-09-55-434972/provision.log
```

### View Developer Computer
```bash
# View the current status of the developer computer
sky queue

Fetching and parsing job queue...
Fetching job queue for: sky-6f63-xl

Job queue of current user on cluster sky-6f63-xl
ID  NAME  USER  SUBMITTED  STARTED    DURATION  RESOURCES   STATUS     LOG                                        GIT COMMIT
1   -     xl    1 min ago  1 min ago  < 1s      1x[CPU:1+]  SUCCEEDED  ~/sky_logs/sky-2025-08-05-10-09-55-434972  -
```

### Log in to the developer computer

```bash
ssh ${your_cluster_name}
```

### Synchronize local code

```bash
rsync -Pavz python ${your_cluster_name}:/home/gcpuser/sky_workdir/sglang-jax/python
```

### Start the test cluster

- Keep the startup method unchanged, just modify the specifications in the yaml file

### Submit a test task

```bash
#Execute the following instruction, and exec will execute the code under run in job.yaml on all nodes
sky exec ${your_cluster_name} job.yaml
```

### Stop the test task

```bash
sky cancel ${your_cluster_name}
```

### Delete Server

```bash
sky down ${your_cluster_name}
```

# Q & A
