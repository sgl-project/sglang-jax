# MiMo-V2-Flash TPU Debug Workflow

This file is the single handoff document for bringing up a `tpu-v6e-16` cluster, starting the MiMo-V2-Flash service, reproducing known issues, and running benchmarks/evals.

## 1. Scope

This workflow covers:

- launching a cheap spot TPU cluster with SkyPilot
- starting the current MiMo-V2-Flash service on `sgl-jax`
- handling inconsistent `/models` placement across TPU workers
- checking readiness and logs
- running one-batch benchmarks
- running small eval/debug commands
- reproducing and understanding the known multinode TPU scheduling bug that was fixed

The current source of truth for the remote start logic is:

- [scripts/run_remote_mimo_debug.sh](/Users/jiongxuan/workspace/sgl-jax/scripts/run_remote_mimo_debug.sh)

## 2. Current Service Defaults

The debug launcher now uses these practical defaults:

- `context-length=65536`
- `mem-fraction-static=0.75`
- `max-running-requests=64`
- `chunked-prefill-size=16384`
- `page-size=1`
- `attention-backend=fa`
- `reasoning-parser=qwen3`
- `--disable-precompile`
- `--skip-server-warmup`

Important notes:

- `context-length=262144` is too aggressive for the current `tpu-v6e-16`, `tp=16`, `ep=16`, `page_size=1` setup. Startup can fail with `max running requests: 0`.
- `65536` is the current production-like setting that actually fits.
- Spot TPU nodes may have inconsistent local `/models` contents.
- The launcher now prefers local `/models/MiMo-V2-Flash`, and only falls back to ModelScope on the node that is missing the local model.

## 3. Launch A Cheap Spot TPU

```bash
sky launch scripts/tpu_v6e16_mimo.sky.yaml -n <cluster_name> -y --use-spot -i 120
printf '%s\n' <cluster_name> > .cluster_name_tpu
```

Typical example:

```bash
sky launch scripts/tpu_v6e16_mimo.sky.yaml -n sky-69ce-jiongxuan -y --use-spot -i 120
printf '%s\n' sky-69ce-jiongxuan > .cluster_name_tpu
```

Check cluster state:

```bash
sky status $(cat .cluster_name_tpu)
```

## 4. Start The Service

### 4.1 Normal start

This is the default and should be used first.

```bash
sky exec $(cat .cluster_name_tpu) --workdir . \
  'WARMUPS="" \
   DISABLE_PRECOMPILE=1 \
   SKIP_SERVER_WARMUP=1 \
   CONTEXT_LENGTH=65536 \
   MAX_RUNNING_REQUESTS=64 \
   bash scripts/run_remote_mimo_debug.sh'
```

Behavior:

- if a node has `/models/MiMo-V2-Flash/config.json`, it uses the local model path
- if a node is missing that local path, only that node falls back to:
  - `SGLANG_USE_MODELSCOPE=true`
  - `XiaomiMiMo/MiMo-V2-Flash`

### 4.2 Force ModelScope on all nodes

Only use this if you explicitly want every worker to download from ModelScope.

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

This is slower because every node downloads model shards.

## 5. Wait For Readiness

Use `/get_model_info` instead of just checking whether the port is open.

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

Quick direct check:

```bash
sky exec $(cat .cluster_name_tpu) \
  'if [ "${SKYPILOT_NODE_RANK:-0}" != "0" ]; then exit 0; fi; \
   python - <<'"'"'PY'"'"'
import json, urllib.request
url = "http://127.0.0.1:30271/get_model_info"
try:
    with urllib.request.urlopen(url, timeout=10) as r:
        print(r.read().decode())
except Exception as e:
    print(f"ERR: {e}")
PY'
```

## 6. Useful Log Checks

Tail all per-node logs:

```bash
sky exec $(cat .cluster_name_tpu) \
  "python - <<'PY'
from pathlib import Path
for path in sorted(Path('/home/gcpuser').glob('mimo_sink_debug_node*.log')):
    print(f'===== {path.name} =====')
    lines = path.read_text(errors='ignore').splitlines()
    for line in lines[-40:]:
        print(line)
PY"
```

Check which nodes have the local model:

```bash
sky exec $(cat .cluster_name_tpu) \
  'python - <<'"'"'PY'"'"'
import os
print(os.path.exists("/models/MiMo-V2-Flash"), os.path.isdir("/models/MiMo-V2-Flash"))
print(os.path.exists("/models/MiMo-V2-Flash/config.json"))
PY'
```

Check current remote jobs:

```bash
sky queue $(cat .cluster_name_tpu)
```

Cancel a stuck job:

```bash
sky cancel $(cat .cluster_name_tpu) <job_id> -y
```

## 7. One-Batch Benchmark

This is the most reliable benchmark path right now. `bench_serving.py` was not stable enough on this service path.

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

Notes:

- This script performs targeted warmup before formal timing.
- If the service was started via ModelScope fallback, the benchmark command can still keep `--model-path /models/MiMo-V2-Flash` because the benchmark talks to the HTTP server.

Known historical results:

- older hot-ish result on the fixed service path:
  - `batch=1`: `latency 49.72s`, `TTFT 1.01s`, `output throughput 2.63 tok/s`
  - `batch=4`: `latency 54.42s`, `TTFT 3.96s`, `output throughput 10.15 tok/s`
- cold production-like run before spot preemption:
  - `batch=1`: `latency 134.40s`, `TTFT 41.06s`
  - `batch=2`: `latency 143.26s`, `TTFT 48.83s`
  - `batch=4`: `latency 122.73s`, `TTFT 62.03s`

Do not mix warmup/cold numbers with final hot numbers.

## 8. Eval Commands

### 8.1 GPQA Diamond, 10 examples

```bash
sky exec $(cat .cluster_name_tpu) --workdir . \
  'if [ "${SKYPILOT_NODE_RANK:-0}" != "0" ]; then exit 0; fi; \
   source ~/sky_workdir/sgl-jax/.venv/bin/activate; \
   cd ~/sky_workdir/sgl-jax/test/srt; \
   python run_eval.py \
     --base-url http://127.0.0.1:30271 \
     --model /models/MiMo-V2-Flash \
     --eval-name gpqa \
     --num-examples 10 \
     --num-threads 1 \
     --temperature 0.0 \
     --top-p 1.0 \
     --max-tokens 128 \
     --request-timeout 600 \
     --max-retries 2'
```

Historical result after the multinode TPU fix:

- `Score: 0.400`
- `Total latency: 91.168s`

### 8.2 GSM8K

```bash
sky exec $(cat .cluster_name_tpu) --workdir . \
  'if [ "${SKYPILOT_NODE_RANK:-0}" != "0" ]; then exit 0; fi; \
   source ~/sky_workdir/sgl-jax/.venv/bin/activate; \
   cd ~/sky_workdir/sgl-jax/test/srt; \
   python run_eval.py \
     --base-url http://127.0.0.1:30271 \
     --model /models/MiMo-V2-Flash \
     --eval-name gsm8k \
     --num-examples 10 \
     --num-threads 1 \
     --temperature 0.8 \
     --top-p 0.95 \
     --max-tokens 256 \
     --request-timeout 600 \
     --max-retries 2'
```

Practical note:

- full GSM8K is still hours-scale under the current throughput
- treat full GSM8K as a separate scaling task

## 9. Known Bug That Was Fixed

This is the main logic bug that was narrowed down and fixed during this debug round.

### 9.1 Symptom

The hang was triggered by:

- multinode TPU
- radix cache hit
- only 1 token left to extend

This showed up in:

- synthetic repeated-prompt repro
- `GPQA sample_2 -> sample_9`

### 9.2 Root cause

Relevant file:

- [python/sgl_jax/srt/managers/schedule_batch.py](/Users/jiongxuan/workspace/sgl-jax/python/sgl_jax/srt/managers/schedule_batch.py)

When `extend_input_len == 1`, the scheduler can route the request from `EXTEND` into `DECODE fastpath for extend`.

That optimization is fine in general, but it was unstable for:

- multinode TPU
- radix-cache hit
- single-token extend

### 9.3 Fix

The fix is narrow:

- on multinode TPU
- if there is a prefix-cache hit
- and only 1 token remains to extend

then bypass that prefix hit for this round:

- `prefix_indices = []`
- `last_node = tree_cache.root_node`
- `last_host_node = tree_cache.root_node`
- `host_hit_length = 0`

This forces the request back onto the normal `EXTEND` path.

Relevant files:

- [python/sgl_jax/srt/managers/schedule_batch.py](/Users/jiongxuan/workspace/sgl-jax/python/sgl_jax/srt/managers/schedule_batch.py)
- [python/sgl_jax/test/test_schedule_batch.py](/Users/jiongxuan/workspace/sgl-jax/python/sgl_jax/test/test_schedule_batch.py)

## 10. Minimal Repros For The Fixed Bug

### 10.1 Synthetic repeated prompt

```bash
sky exec $(cat .cluster_name_tpu) \
  'if [ "${SKYPILOT_NODE_RANK:-0}" != "0" ]; then exit 0; fi; \
   source ~/sky_workdir/sgl-jax/.venv/bin/activate; \
   cd ~/sky_workdir/sgl-jax; \
   export PYTHONPATH=~/sky_workdir/sgl-jax/test/srt:~/sky_workdir/sgl-jax; \
   python scripts/debug_same_prompt_repro.py \
     --base-url http://127.0.0.1:30271/v1/chat/completions \
     --model /models/MiMo-V2-Flash \
     --timeout 120'
```

Expected healthy behavior:

- first request succeeds, usually much slower
- second request also succeeds, usually around 1-2 seconds

### 10.2 Real GPQA subset repro

```bash
sky exec $(cat .cluster_name_tpu) \
  'if [ "${SKYPILOT_NODE_RANK:-0}" != "0" ]; then exit 0; fi; \
   source ~/sky_workdir/sgl-jax/.venv/bin/activate; \
   cd ~/sky_workdir/sgl-jax; \
   export PYTHONPATH=~/sky_workdir/sgl-jax/test/srt:~/sky_workdir/sgl-jax; \
   python scripts/debug_gpqa_hang_repro.py \
     --mode run-gpqa-subset \
     --base-url http://127.0.0.1:30271/v1 \
     --model /models/MiMo-V2-Flash \
     --indices 2,9 \
     --max-tokens 1 \
     --timeout 180 \
     --temperature 0.0 \
     --top-p 1.0'
```

Expected healthy behavior:

- `sample_2` returns
- `sample_9` also returns

## 11. Practical Debug Notes

- Do not use `--disable-radix-cache` when reproducing the single-token extend bug. That hides the failure mode.
- Use `WARMUPS=""` for clean repro.
- `bench_serving.py` was not the reliable path for this setup. Prefer `bench_one_batch_server.py`.
- If you see `Connection refused` for a long time, check logs before assuming the model is still loading. In one failure mode, a single worker had no local model directory and the whole JAX distributed startup stalled.
- If one spot worker is missing `/models/MiMo-V2-Flash`, the current launcher will let only that node fall back to ModelScope.
- If you force ModelScope on all nodes, budget extra time and network for multi-node shard download.

## 12. Teardown

Stop the current cluster:

```bash
sky down $(cat .cluster_name_tpu) -y
```

Or stop without deleting:

```bash
sky stop $(cat .cluster_name_tpu)
```

## 13. Related Files

- [scripts/run_remote_mimo_debug.sh](/Users/jiongxuan/workspace/sgl-jax/scripts/run_remote_mimo_debug.sh)
- [docs/developer_guide/mimo_v2_flash_tpu_debug.md](/Users/jiongxuan/workspace/sgl-jax/docs/developer_guide/mimo_v2_flash_tpu_debug.md)
- [MIMO_MULTINODE_TPU_DEBUG_HANDOFF.md](/Users/jiongxuan/workspace/sgl-jax/MIMO_MULTINODE_TPU_DEBUG_HANDOFF.md)
- [python/sgl_jax/srt/managers/schedule_batch.py](/Users/jiongxuan/workspace/sgl-jax/python/sgl_jax/srt/managers/schedule_batch.py)
- [python/sgl_jax/test/test_schedule_batch.py](/Users/jiongxuan/workspace/sgl-jax/python/sgl_jax/test/test_schedule_batch.py)
- [scripts/debug_same_prompt_repro.py](/Users/jiongxuan/workspace/sgl-jax/scripts/debug_same_prompt_repro.py)
- [scripts/debug_gpqa_hang_repro.py](/Users/jiongxuan/workspace/sgl-jax/scripts/debug_gpqa_hang_repro.py)
