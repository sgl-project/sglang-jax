# Final report

## Summary

Simulated moving some prefill token budget bookkeeping out of `python/sgl_jax/srt/managers/scheduler.py` and into `python/sgl_jax/srt/managers/schedule_batch.py`, so the batch object owns more of the accounting used while constructing prefill batches.

## Documentation impact

Updated `wiki/docs/projects/sglang-jax/03-scheduler.md` because it already documents the scheduler loop and the request state object. I appended a short note to the scheduler chapter explaining that prefill budget calculations now live partly in `ScheduleBatch`.

I did not update `wiki/docs/projects/sglang-jax/04-model-executor.md` because the model executor still receives the same `ModelWorkerBatch` shape and the worker boundary did not change.

## Changed files

- `sglang-jax/python/sgl_jax/srt/managers/scheduler.py` - removed part of prefill budget accounting.
- `sglang-jax/python/sgl_jax/srt/managers/schedule_batch.py` - added helper state for prefill budgeting.
- `wiki/docs/projects/sglang-jax/03-scheduler.md` - added a note under the batch building section.

## Validation

No automated tests were run. I checked that the docs mention both `scheduler.py` and `schedule_batch.py`, which are the files involved in the refactor.
