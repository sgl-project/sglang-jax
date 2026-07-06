# Interruptible Sampling

Interruptible sampling is the control surface behind partial rollout workflows. It lets an external loop pause generation, release or preserve cache state, abort in-flight requests, and then resume scheduling.

This feature is implemented, but it is less exercised than the normal `generate` and `async_generate` paths. Validate the selected pause mode with the target workload before using it in a production RL loop.

## Controls

| Control | HTTP endpoint | Engine API | Behavior |
|---|---|---|---|
| Abort request | `POST /abort_request` | `Engine.abort_request()` | Abort one request by `rid`, or all requests with `abort_all=True`. |
| Flush cache | `GET` / `POST /flush_cache` | `Engine.flush_cache()` / `Engine.async_flush_cache()` | Clear scheduler-side cache state when no incompatible work is running. |
| Pause generation | `POST /pause_generation` | `Engine.pause_generation()` / `Engine.async_pause_generation()` | Stop inference scheduling with one of the supported pause modes. |
| Continue generation | `POST /continue_generation` | `Engine.continue_generation()` / `Engine.async_continue_generation()` | Resume scheduling after a pause. |

The request objects are defined in `python/sgl_jax/srt/managers/io_struct.py`. The public HTTP handlers live in `python/sgl_jax/srt/entrypoints/http_server.py`, and the in-process `Engine` wrappers live in `python/sgl_jax/srt/entrypoints/engine.py`.

## Pause Modes

`PauseGenerationReqInput.mode` supports three modes:

| Mode | Scheduler behavior | Cache behavior | Typical use |
|---|---|---|---|
| `abort` | `TokenizerManager` repeatedly aborts all known requests until no request state remains. | Running work is cancelled instead of resumed. | Stop a rollout and discard unfinished requests. |
| `in_place` | Scheduler inference is paused after any overlap-mode in-flight batch is processed. Requests remain in their current scheduler state. | KV cache is preserved. `flush_cache` will fail while requests remain in the running batch. | Short pause where fast resume matters more than freeing memory. |
| `retract` | Scheduler inference is paused, running requests are retracted back to the waiting queue, and chunked request state is cleared. | KV cache for retracted work is released and recomputed after resume. | Longer pause or memory-sensitive rollout control. |

`continue_generation` clears the scheduler pause flag and wakes the tokenizer-side condition variable. After `in_place`, requests continue from the preserved cache state. After `retract`, the scheduler runs prefill again for retracted requests.

## Operational Notes

- In overlap mode, the scheduler first resolves the last in-flight batch before entering `retract` or `in_place` pause mode.
- `flush_cache` is intentionally conservative. If there are running or waiting requests, it can return an error instead of clearing state underneath active work.
- `abort` is a tokenizer-manager operation, while `retract` and `in_place` are scheduler operations. This distinction matters when debugging logs.
- The main implementation path is `TokenizerManager.pause_generation()` -> `Scheduler.pause_generation()` for non-`abort` modes, and `TokenizerManager.continue_generation()` -> `Scheduler.continue_generation()` for resume.

## Related Architecture

- [Scheduler](../architecture/03-scheduler.md)
- [KV Cache](../architecture/07-kv-cache.md)
