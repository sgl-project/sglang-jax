---
title: "SkyPilot"
description: "Generic launcher template for multi-node TPU serving via SkyPilot, with provisioning + sky exec pattern."
---

# SkyPilot Multi-Node TPU Launcher

Generic SkyPilot recipe for provisioning a multi-node TPU cluster and launching `sgl_jax.launch_server` on it. Cookbook recipes that use SkyPilot ([Grok-2](../autoregressive/Grok/Grok2.md), and any future multi-node dense / MoE recipe) reference this page instead of duplicating the workflow.

## Prerequisites

- SkyPilot installed locally and authenticated against your GCP account (`sky check`).
- GCP TPU quota in the region you target.
- A working SGL-JAX checkout on the local machine — the launcher script lives at `scripts/launch_tpu.sh`.

For the broader SkyPilot development workflow (clone, sync, destroy) see [`../../developer_guide/tpu_resources_guide.md`](../../developer_guide/tpu_resources_guide.md).

## The launcher script

`scripts/launch_tpu.sh` is a thin wrapper around `sky launch` that:

1. Reads the SkyPilot template `scripts/tpu_resource.sky.yaml`.
2. Substitutes `$ACCELERATOR` and `$REF` (git ref) into a rendered copy.
3. Generates a unique cluster name based on the ref + a random suffix.
4. Calls `sky launch --infra=gcp -i 10 --down -y` (auto-stop after 10 min idle, auto-destroy on stop).
5. Polls `sky status --refresh` until the cluster is `UP` (10-minute timeout).

Usage:

```bash
bash scripts/launch_tpu.sh <accelerator> <ref> [test_type]
```

| Arg | Meaning | Examples |
|---|---|---|
| `<accelerator>` | SkyPilot resource name. **Only v6e is currently templated** — see "Generation support" below. | `tpu-v6e-1`, `tpu-v6e-4`, `tpu-v6e-32` |
| `<ref>` | Git branch/tag/SHA to check out on the cluster | `main`, `release-0.1`, `1a2b3c4` |
| `[test_type]` | Optional cluster-name suffix for CI bucketing | `e2e`, `performance` |

The resolved cluster name is written to `.cluster_name` at the repo root, which downstream scripts (`scripts/cleanup_cluster.sh`, etc.) read.

### Generation support

`scripts/tpu_resource.sky.yaml` pins `runtime_version: v2-alpha-tpuv6e`, which is **v6e-only**. To use v5e / v5p / v7x you have to fork the template (or pass an alternative resource block via `sky launch` directly) and pick the matching TPU runtime version. The v7x recipes in the cookbook use the GKE path instead — see [MiMo-V2.5-Pro §4.4](../autoregressive/Xiaomi/MiMo-V2.5-Pro.md#44-gke-indexed-job--headless-service).

## The cluster template

`scripts/tpu_resource.sky.yaml`:

```yaml
resources:
  accelerators: $ACCELERATOR
  accelerator_args:
      tpu_vm: True
      runtime_version: v2-alpha-tpuv6e
  use_spot: True

setup: |
  if [ -d "sglang-jax" ]; then pip uninstall -y sgl-jax || true && rm -rf sglang-jax; fi
  git clone https://github.com/sgl-project/sglang-jax.git
  cd sglang-jax && git fetch origin $REF:$REF && git checkout $REF
  uv venv --python 3.12
  source .venv/bin/activate
  uv pip install -e "python[all]"
  bash scripts/killall_sglang.sh
  uv tool install evalscope
```

**Key points**
- `use_spot: True` is the default — flip to `False` (or pass `--no-use-spot`) for production.
- `setup` runs once per node when the cluster comes up. It installs SGL-JAX + evalscope into a uv-managed venv at `~/.venv` per node.
- After setup completes, the cluster sits idle; you launch the model server via a separate `sky exec` (see below).

## Launching a server on the cluster

The full pattern:

```bash
# 1. Provision the cluster (blocks until UP)
cd ${WORKSPACE_DIR}/sglang-jax
bash scripts/launch_tpu.sh <accelerator> <git-ref>

# 2. Read the cluster name the launcher wrote
CLUSTER_NAME=$(cat .cluster_name)

# 3. Launch the model server across all nodes
sky exec ${CLUSTER_NAME} -- "cd sglang-jax && source .venv/bin/activate && \
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache uv run python -u -m sgl_jax.launch_server \
  --model-path <MODEL> --trust-remote-code \
  --tp-size=<N> --device=tpu \
  --dist-init-addr=<NODE_0_IP_ADDRESS>:<PORT> \
  --nnodes=<NODES> --node-rank=\${SKYPILOT_NODE_RANK} \
    # ... rest of recipe-specific flags
"
```

**Important substitutions**

| Placeholder | What to fill in |
|---|---|
| `<MODEL>` | HuggingFace repo id or absolute path on the node (e.g. `/models/xai-grok-2`). |
| `<N>` | `--tp-size` = total JAX devices across all nodes. See [`../base/tpu-topology-reference.md`](../base/tpu-topology-reference.md). |
| `<NODE_0_IP_ADDRESS>:<PORT>` | Rank-0 node's internal IP and an unused TCP port. SkyPilot does not auto-expose this — `sky status -a <cluster>` shows per-node IPs. |
| `<NODES>` | Node count. Matches the SkyPilot accelerator name's chip count divided by chips-per-node (4 for v6e). E.g. `tpu-v6e-32` → 8 nodes. |
| `${SKYPILOT_NODE_RANK}` | Provided automatically by SkyPilot in the remote shell. Escape the `$` (`\$`) so the local shell does not expand it. |

## Useful SkyPilot commands

```bash
sky -h                    # show help
sky status                # all clusters
sky status --refresh      # poll GCP for true state
sky status -a <cluster>   # detailed view incl. per-node IPs
sky queue                 # all jobs across clusters
sky logs <cluster>        # last job's logs (add -f to follow)
sky cancel <cluster> <job-id>
sky down <cluster>        # destroy the cluster
```

The launcher passes `-i 10 --down`, so an idle cluster auto-destroys after 10 minutes — `sky down` is mainly for explicit cleanup.

## Worked examples

| Recipe | Accelerator | Nodes | `--tp-size` |
|---|---|---|---|
| [Grok-2](../autoregressive/Grok/Grok2.md) | `tpu-v6e-32` | 8 | 32 |

Single-host recipes ([Qwen-7B-Chat](../autoregressive/Qwen/Qwen.md), [Qwen3](../autoregressive/Qwen/Qwen3.md)) can also run via SkyPilot by passing `tpu-v6e-4`, but Docker on a single host is usually simpler — see the recipes themselves for the direct launch form.

## Related docs

- [`../../developer_guide/tpu_resources_guide.md`](../../developer_guide/tpu_resources_guide.md) — broader SkyPilot dev workflow (sync code, destroy clusters).
- [`../base/tpu-topology-reference.md`](../base/tpu-topology-reference.md) — TPU/device/chip table for picking `--tp-size`.
- [`../base/launch-flags-reference.md`](../base/launch-flags-reference.md) — full launch flag reference.
