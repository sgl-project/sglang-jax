# MiMo-V2-Flash TPU Debug Commands

This note records the current commands used to launch and benchmark MiMo-V2-Flash on a `tpu-v6e-16` SkyPilot cluster.

## Launch A Cheap Spot Cluster

```bash
sky launch scripts/tpu_v6e16_mimo.sky.yaml -n sky-69ce-jiongxuan -y --use-spot -i 120
printf '%s\n' sky-69ce-jiongxuan > .cluster_name_tpu
```

## Start Service With Local Model Path

The script now prefers `/models/MiMo-V2-Flash` by default and automatically falls back to `XiaomiMiMo/MiMo-V2-Flash` through ModelScope on nodes where the local model is missing.

```bash
sky exec $(cat .cluster_name_tpu) --workdir . \
  'WARMUPS="" \
   DISABLE_PRECOMPILE=1 \
   SKIP_SERVER_WARMUP=1 \
   CONTEXT_LENGTH=65536 \
   MAX_RUNNING_REQUESTS=64 \
   bash scripts/run_remote_mimo_debug.sh'
```

## Force ModelScope On All Nodes

Use this only when you explicitly want every node to download from ModelScope.

```bash
sky exec $(cat .cluster_name_tpu) --workdir . \
  'export SGLANG_USE_MODELSCOPE=true; \
   WARMUPS="" \
   DISABLE_PRECOMPILE=1 \
   SKIP_SERVER_WARMUP=1 \
   MODEL_PATH=XiaomiMiMo/MiMo-V2-Flash \
   CONTEXT_LENGTH=65536 \
   MAX_RUNNING_REQUESTS=64 \
   bash scripts/run_remote_mimo_debug.sh'
```

## Wait For Readiness

```bash
sky exec $(cat .cluster_name_tpu) --workdir . \
  'if [ "${SKYPILOT_NODE_RANK:-0}" != "0" ]; then exit 0; fi; \
   python - <<'"'"'PY'"'"'
import json
import sys
import time
import urllib.request

url = "http://127.0.0.1:30271/get_model_info"
for i in range(1, 181):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            payload = json.loads(r.read().decode())
        print({"attempt": i, "is_ready": payload.get("is_ready")}, flush=True)
        if payload.get("is_ready"):
            sys.exit(0)
    except Exception as e:
        print({"attempt": i, "err": str(e)}, flush=True)
    time.sleep(5)
raise SystemExit(1)
PY'
```

## One-Batch Benchmark

```bash
sky exec $(cat .cluster_name_tpu) --workdir . \
  'if [ "${SKYPILOT_NODE_RANK:-0}" != "0" ]; then exit 0; fi; \
   source ~/sky_workdir/sgl-jax/.venv/bin/activate; \
   cd ~/sky_workdir/sgl-jax; \
   export PYTHONPATH=~/sky_workdir/sgl-jax/python; \
   python -m sgl_jax.bench_one_batch_server \
     --model-path /models/MiMo-V2-Flash \
     --base-url http://127.0.0.1:30271 \
     --run-name mimo_v2_flash_prod_cfg \
     --batch-size 1 2 4 8 \
     --input-len 472 \
     --output-len 128 \
     --skip-server-info \
     --skip-flush-cache'
```

If the service was started with ModelScope fallback, `--model-path` above can stay as `/models/MiMo-V2-Flash` because the benchmark talks to the HTTP server, not directly to the model loader.

## Useful Log Checks

```bash
sky exec $(cat .cluster_name_tpu) \
  'python - <<'"'"'PY'"'"'
from pathlib import Path
for path in sorted(Path("/home/gcpuser").glob("mimo_sink_debug_node*.log")):
    print(f"===== {path} =====")
    lines = path.read_text(errors="ignore").splitlines()
    for line in lines[-40:]:
        print(line)
PY'
```

```bash
sky exec $(cat .cluster_name_tpu) \
  'python - <<'"'"'PY'"'"'
import os
print(os.path.exists("/models/MiMo-V2-Flash"), os.path.isdir("/models/MiMo-V2-Flash"))
print(os.path.exists("/models/MiMo-V2-Flash/config.json"))
PY'
```

## Current Practical Notes

- `CONTEXT_LENGTH=262144` is too aggressive for the current `tpu-v6e-16`, `tp=16`, `ep=16`, `page_size=1` setup. Startup can fail with `max running requests: 0`.
- `CONTEXT_LENGTH=65536` is the current production-like setting that can fit this hardware.
- Spot TPU nodes may have inconsistent local `/models` contents. The startup script now auto-detects this and only falls back to ModelScope on the affected node.
- Forcing `SGLANG_USE_MODELSCOPE=true` on all nodes is slower because every node downloads model shards.
