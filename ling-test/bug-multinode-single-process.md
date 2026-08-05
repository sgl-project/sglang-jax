# sglang-jax Bug: `enable_single_process=True` + multi-node (`nnodes>1`) crash

## Summary

When using `enable_single_process=True` with `nnodes>1`, non-rank-0 workers crash with `UnboundLocalError` on `scheduler_pipe_readers`. This blocks multi-node `dp>1` deployments (e.g. `dp=4, tp=4` on a 4x4 TPU pod).

## Reproduction

```python
Engine(
    model_path=MODEL, tp_size=16, dp_size=4,
    nnodes=4, node_rank=1,  # any node_rank >= 1
    dist_init_addr="worker-0:29500",
    enable_single_process=True,
    ...
)
```

## Error

```
File "sgl_jax/srt/entrypoints/engine.py", line 684, in _launch_threads
    for reader in scheduler_pipe_readers:
                  ^^^^^^^^^^^^^^^^^^^^^^
UnboundLocalError: cannot access local variable 'scheduler_pipe_readers'
    where it is not associated with a value
```

## Root Cause

In `_launch_threads()` (engine.py:651), the `node_rank >= 1` branch (line 674) references `scheduler_pipe_readers`, but this variable is only initialized in the `_launch_subprocesses()` path — it is never assigned in `_launch_threads()`.

```python
# engine.py:668-684
def _launch_threads(...):
    scheduler_threads = []       # ← defined
    scheduler_infos = []         # ← defined
    scheduler_pipe_readers = []  # ← NOT defined here
    scheduler_info = run_scheduler_loop_thread_after_create(...)
    scheduler_infos.append(scheduler_info)

    if server_args.node_rank >= 1:
        for reader in scheduler_pipe_readers:  # ← UnboundLocalError
            ...
```

## Impact

- **dp>1 on multi-node is unusable** with `enable_single_process=True`
- Workaround: use `tp_size=total_chips` with `dp_size=1` (pure TP), which works but is suboptimal for small models (cross-node TP communication overhead)
- `nnodes=1` (single-node) is unaffected

## Environment

- sglang-jax 0.0.2
- GKE TPU v6e, 4x4 topology (4 nodes × 4 chips)
- JAX 0.8.1, Python 3.12
