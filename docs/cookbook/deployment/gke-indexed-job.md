---
title: "GKE Indexed Job"
---

# GKE Indexed Job Launcher

Generic GKE template for launching a multi-host TPU slice serving SGL-JAX. This page is the **single source of truth** for the Indexed Job + headless Service pattern — recipes reference it instead of pasting a full manifest, and fill in only their model-specific fields (`<JOB>`, `<ACCELERATOR>`, `<TOPOLOGY>`, `<LAUNCH_FLAGS>`).

## Why Indexed Job + headless Service

Multi-host JAX needs:
1. **A stable rank-0 hostname** for `--dist-init-addr`. Indexed Job gives every pod the index `${JOB_COMPLETION_INDEX}`; combined with a headless Service the pod `0` becomes `<job>-0.<svc>` permanently.
2. **TPU process address fanout**. The JAX TPU runtime expects `TPU_PROCESS_ADDRESSES` listing every node's `host:8471`. Headless DNS makes those names enumerable.
3. **No restarts on crash**. `restartPolicy: Never` + `backoffLimit: 0` — if one pod dies, the whole job fails fast rather than partial restart that would desync ranks.

## Manifest template

Fill in:
- `<JOB>` — pick a short name; used everywhere as a prefix.
- `<ACCELERATOR>` — the GKE accelerator label value. **Asymmetric naming**: v7x is `tpu7x`, v6e is `tpu-v6e-slice` (per GKE docs). Don't normalise these to a common form.
- `<TOPOLOGY>` — e.g. `2x2x4` for v7x-16, `4x4x4` for v6e-64.
- `<N>` — number of nodes in the slice.
- `<CHIPS_PER_NODE>` — `4` for both v6e and v7x.
- `<MODEL_PATH>` — absolute path inside the container.
- `<LAUNCH_FLAGS>` — flags from the recipe's `--tp-size`/`--dp-size`/etc. block.
- `<HTTP_PORT>` — server port (cookbook recipes use 30000 or 30271).
- `<MODEL_PVC>` — your PersistentVolumeClaim name with the model weights.

```yaml
---
apiVersion: v1
kind: Service
metadata:
  name: <JOB>-headless-svc
spec:
  clusterIP: None
  selector:
    job-name: <JOB>
  ports:
  - name: dist-init
    port: 5000
  - name: tpu-process
    port: 8471
---
apiVersion: batch/v1
kind: Job
metadata:
  name: <JOB>
spec:
  backoffLimit: 0
  completionMode: Indexed
  parallelism: <N>
  completions: <N>
  template:
    metadata:
      annotations:
        gke-gcsfuse/volumes: "true"
    spec:
      subdomain: <JOB>-headless-svc
      restartPolicy: Never
      serviceAccountName: gcs-account
      nodeSelector:
        cloud.google.com/gke-tpu-accelerator: <ACCELERATOR>
        cloud.google.com/gke-tpu-topology: <TOPOLOGY>
      containers:
      - name: <JOB>
        image: us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1
        command: ["bash", "-lc"]
        args:
        - |
          set -euxo pipefail

          REPO_DIR=/tmp/sglang-jax
          if [ ! -d "$REPO_DIR/.git" ]; then
            git clone https://github.com/sgl-project/sglang-jax.git "$REPO_DIR"
          fi
          cd "$REPO_DIR" && git fetch origin && pip install -e "python[tpu]"

          export NODE_RANK=${JOB_COMPLETION_INDEX}
          export MASTER_ADDR=<JOB>-0.<JOB>-headless-svc:5000
          JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m sgl_jax.launch_server \
              --model-path <MODEL_PATH> \
              --trust-remote-code \
              <LAUNCH_FLAGS> \
              --host 0.0.0.0 --port <HTTP_PORT> \
              --nnodes <N> --node-rank ${NODE_RANK} \
              --dist-init-addr ${MASTER_ADDR}
        env:
        - name: TPU_PROCESS_ADDRESSES
          value: <JOB>-0.<JOB>-headless-svc:8471,<JOB>-1.<JOB>-headless-svc:8471,...,<JOB>-(N-1).<JOB>-headless-svc:8471
        - name: TPU_WORKER_HOSTNAMES
          value: <JOB>-0.<JOB>-headless-svc,<JOB>-1.<JOB>-headless-svc,...,<JOB>-(N-1).<JOB>-headless-svc
        - name: TPU_PROCESS_PORT
          value: "8471"
        - name: JOB_COMPLETION_INDEX
          valueFrom:
            fieldRef:
              fieldPath: metadata.labels['batch.kubernetes.io/job-completion-index']
        - name: TPU_WORKER_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.labels['batch.kubernetes.io/job-completion-index']
        ports:
        - containerPort: <HTTP_PORT>
          name: http
        - containerPort: 5000
          name: dist-init
        resources:
          requests:
            google.com/tpu: "<CHIPS_PER_NODE>"
          limits:
            google.com/tpu: "<CHIPS_PER_NODE>"
        volumeMounts:
        - mountPath: /models
          name: model-storage
        - mountPath: /dev/shm
          name: dev-shm
      volumes:
      - name: dev-shm
        emptyDir:
          medium: Memory
      - name: model-storage
        persistentVolumeClaim:
          claimName: <MODEL_PVC>
```

> **`TPU_PROCESS_ADDRESSES` and `TPU_WORKER_HOSTNAMES` cannot use shell expansion** — Kubernetes env values are static strings. For a 4-node job, enumerate all 4 explicitly (`<JOB>-0`, `<JOB>-1`, `<JOB>-2`, `<JOB>-3`). For very large slices (16+ nodes) consider templating the manifest with Helm/Kustomize.

## Apply and watch

```bash
kubectl apply -f <job>.yaml
kubectl wait --for=condition=Ready pod -l job-name=<JOB> --timeout=600s
kubectl logs -l job-name=<JOB> -c <JOB> --tail=200 -f
```

The server is ready once `<JOB>-0` logs `Uvicorn running on http://0.0.0.0:<HTTP_PORT>`. Exposing it externally is out of scope for this template — typical patterns are NodePort, LoadBalancer, or an Ingress in front of pod `0`.

## What this template intentionally does NOT cover

- GCSFuse mount for model weights — add a `gke-gcsfuse/volumes` annotation with the appropriate `volumeMount` if you serve weights via GCS.
- Multi-cluster / multi-region.
- Autoscaling — TPU pools are typically pre-provisioned at slice granularity.
- TLS / authentication on the HTTP port.

## Related

- [MiMo-V2.5-Pro §4.4](../autoregressive/Xiaomi/MiMo-V2.5-Pro.md#44-gke-indexed-job--headless-service) — a fully filled-in instance of this template.
- [`single-host-docker.md`](single-host-docker.md) — when you only need one node.
- [`skypilot.md`](skypilot.md) — SkyPilot alternative (v6e-only today).
