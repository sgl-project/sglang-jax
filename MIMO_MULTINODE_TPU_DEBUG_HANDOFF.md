# MiMo-V2-Flash Multinode TPU Debug Handoff

## Summary

This round narrowed down and fixed the request hang that showed up in GPQA and in repeated long-prompt requests on multinode TPU.

The key result is:

- The hang is not caused by `attention sink`.
- The hang is not caused by `ragged_paged_attention_split.py` input/output aliasing.
- The hang is caused by a very specific scheduling/cache path:
  - multinode TPU
  - radix cache hit
  - only 1 token left to extend

The fix is now implemented in:

- [python/sgl_jax/srt/managers/schedule_batch.py](/Users/jiongxuan/workspace/sgl-jax/python/sgl_jax/srt/managers/schedule_batch.py)

and covered by regression tests in:

- [python/sgl_jax/test/test_schedule_batch.py](/Users/jiongxuan/workspace/sgl-jax/python/sgl_jax/test/test_schedule_batch.py)


## What Was Broken

The broken scenario was:

1. A request warms prefix cache.
2. A later request hits that prefix cache.
3. After the prefix hit, only 1 token remains in the current extend round.
4. On multinode TPU, that request was routed into the `decode fastpath for extend`.
5. That path could hang and never return an HTTP response.

This showed up most clearly in two reproducible cases:

- Synthetic case: the same `659-token` prompt sent twice.
- Real case: `GPQA sample_2 -> sample_9`.


## Root Cause

The relevant logic lives in:

- [python/sgl_jax/srt/managers/schedule_batch.py#L378](/Users/jiongxuan/workspace/sgl-jax/python/sgl_jax/srt/managers/schedule_batch.py#L378)
- [python/sgl_jax/srt/managers/schedule_batch.py#L1222](/Users/jiongxuan/workspace/sgl-jax/python/sgl_jax/srt/managers/schedule_batch.py#L1222)

Each request round starts by:

- building `fill_ids = origin_input_ids + output_ids`
- matching prefix cache
- computing:
  - `extend_input_len = len(fill_ids) - len(prefix_indices)`

If `extend_input_len == 1`, the scheduler can convert that request from `EXTEND` into `DECODE` fastpath.

That optimization is normally fine, but on multinode TPU with a radix-cache hit it became unstable and could hang.


## Fix

The fix is intentionally narrow.

In [python/sgl_jax/srt/managers/schedule_batch.py#L389](/Users/jiongxuan/workspace/sgl-jax/python/sgl_jax/srt/managers/schedule_batch.py#L389), when all of the following are true:

- multinode TPU runtime
- prefix cache hit exists
- only 1 token remains to extend

we bypass that prefix hit for this round:

- `prefix_indices = []`
- `last_node = tree_cache.root_node`
- `last_host_node = tree_cache.root_node`
- `host_hit_length = 0`

This makes the request behave like a normal extend request again, which avoids the broken multinode TPU fastpath.

Important detail:

- Clearing only `prefix_indices` is not enough.
- `last_node`, `last_host_node`, and `host_hit_length` must also be reset, otherwise later lock/ref logic sees an inconsistent partially-hit cache state.


## Regression Tests

Regression tests were added in:

- [python/sgl_jax/test/test_schedule_batch.py](/Users/jiongxuan/workspace/sgl-jax/python/sgl_jax/test/test_schedule_batch.py)

They cover:

- single-token extend on multinode TPU should bypass prefix cache
- multi-token extend should not bypass prefix cache

Remote result:

```text
..
----------------------------------------------------------------------
Ran 2 tests in 0.001s

OK
```


## Repro Commands

### 1. Launch cheap spot TPU v6e-16

```bash
sky launch scripts/tpu_v6e16_mimo.sky.yaml -n <cluster_name> -y --use-spot -i 120
```


### 2. Start remote debug service

This uses current local code, syncs it to the cluster, and launches the service with no startup warmups.

```bash
sky exec <cluster_name> --workdir . 'WARMUPS="" EXTRA_SERVER_ARGS="--chunked-prefill-size 256 --watchdog-timeout 3600" bash scripts/run_remote_mimo_debug.sh'
```


### 3. Wait for service ready

```bash
sky exec <cluster_name> 'if [ "${SKYPILOT_NODE_RANK:-0}" != "0" ]; then exit 0; fi; python - <<'"'"'PY'"'"'
import json, time, urllib.request
url = "http://127.0.0.1:30271/get_model_info"
for i in range(1, 181):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            payload = json.loads(r.read().decode())
        print({"attempt": i, "is_ready": payload.get("is_ready")}, flush=True)
        if payload.get("is_ready"):
            break
    except Exception as e:
        print({"attempt": i, "err": str(e)}, flush=True)
    time.sleep(2)
PY'
```


### 4. Synthetic minimal repro

This reproduces the old bug very reliably.

```bash
sky exec <cluster_name> 'if [ "${SKYPILOT_NODE_RANK:-0}" != "0" ]; then exit 0; fi; source ~/sky_workdir/sgl-jax/.venv/bin/activate; cd ~/sky_workdir/sgl-jax; export PYTHONPATH=~/sky_workdir/sgl-jax/test/srt:~/sky_workdir/sgl-jax; python scripts/debug_same_prompt_repro.py --base-url http://127.0.0.1:30271/v1/chat/completions --model /models/MiMo-V2-Flash --timeout 120'
```

Expected healthy behavior after the fix:

- first request: succeeds, tens of seconds
- second request: succeeds, around 1-2 seconds


### 5. Real GPQA minimal repro

```bash
sky exec <cluster_name> 'if [ "${SKYPILOT_NODE_RANK:-0}" != "0" ]; then exit 0; fi; source ~/sky_workdir/sgl-jax/.venv/bin/activate; cd ~/sky_workdir/sgl-jax; export PYTHONPATH=~/sky_workdir/sgl-jax/test/srt:~/sky_workdir/sgl-jax; python scripts/debug_gpqa_hang_repro.py --mode run-gpqa-subset --base-url http://127.0.0.1:30271/v1 --model /models/MiMo-V2-Flash --indices 2,9 --max-tokens 1 --timeout 180 --temperature 0.0 --top-p 1.0'
```

Expected healthy behavior after the fix:

- `sample_2` returns
- `sample_9` also returns


## Validation Results

### Synthetic repeated prompt

Observed healthy output after the fix:

- first request: `200`, about `40.455s`
- second request: `200`, about `1.412s`


### GPQA subset repro

Observed healthy output after the fix:

- `sample_2`: `0.523s`
- `sample_9`: `1.030s`


### GPQA 10-sample eval

Observed result:

- `Score: 0.400`
- `Total latency: 91.168 s`

This is important because the old bug previously tended to surface near the tail of GPQA evaluation, especially around later examples.


### Benchmark

Command used:

```bash
sky exec <cluster_name> --workdir . 'if [ "${SKYPILOT_NODE_RANK:-0}" != "0" ]; then exit 0; fi; source ~/sky_workdir/sgl-jax/.venv/bin/activate; cd ~/sky_workdir/sgl-jax; export PYTHONPATH=~/sky_workdir/sgl-jax/python; python -m sgl_jax.bench_one_batch_server --model-path /models/MiMo-V2-Flash --base-url http://127.0.0.1:30271 --run-name mimo_v2_flash_fix_validation --batch-size 1 4 --input-len 472 --output-len 128 --skip-server-info --skip-flush-cache'
```

Observed results:

- `batch=1, 472->128`
  - `latency 49.73s`
  - `TTFT 1.02s`
  - `output throughput 2.63 tok/s`
- `batch=4, 472->128`
  - `latency 54.48s`
  - `TTFT 4.02s`
  - `output throughput 10.15 tok/s`


## Notes About GSM8K

`GSM8K full` was not completed in this round.

Reason:

- with the current service throughput, full GSM8K is still hours-scale
- `num_threads=4` was too slow
- `num_threads=16` made the first samples even slower, so it was not a good tradeoff

Practical takeaway:

- if you need full GSM8K next time, treat it as a separate throughput/eval-scaling problem
- do not assume the current service config is efficient enough for full-set evaluation


## Practical Debug Tips

- For this bug, do not use `--disable-radix-cache`, because that hides the failure mode.
- For clean reproduction, use `WARMUPS=""`.
- Readiness should be checked with `/get_model_info` and `is_ready`, not just “port is open”.
- The most useful head-node log check is:

```bash
sky exec <cluster_name> 'if [ "${SKYPILOT_NODE_RANK:-0}" != "0" ]; then exit 0; fi; tail -n 80 ~/mimo_sink_debug_node0.log'
```

- If you only want to validate this fix, always prefer:
  - synthetic repeated-prompt repro
  - GPQA `2,9` subset repro

These two are much cheaper and more diagnostic than re-running full evals.


## Current Files Touched For This Bug

- [python/sgl_jax/srt/managers/schedule_batch.py](/Users/jiongxuan/workspace/sgl-jax/python/sgl_jax/srt/managers/schedule_batch.py)
- [python/sgl_jax/test/test_schedule_batch.py](/Users/jiongxuan/workspace/sgl-jax/python/sgl_jax/test/test_schedule_batch.py)
- [scripts/debug_same_prompt_repro.py](/Users/jiongxuan/workspace/sgl-jax/scripts/debug_same_prompt_repro.py)
- [scripts/debug_gpqa_hang_repro.py](/Users/jiongxuan/workspace/sgl-jax/scripts/debug_gpqa_hang_repro.py)
- [scripts/run_remote_mimo_debug.sh](/Users/jiongxuan/workspace/sgl-jax/scripts/run_remote_mimo_debug.sh)


## Next Recommended Step

If continuing this work later, the best next sequence is:

1. relaunch a cheap spot TPU v6e-16
2. start debug service with `WARMUPS=""`
3. run the synthetic repeated-prompt repro
4. run the GPQA `2,9` subset repro
5. only then run broader benchmark/eval jobs

This keeps the debug loop cheap and avoids burning TPU time on full datasets before basic correctness is re-confirmed.
