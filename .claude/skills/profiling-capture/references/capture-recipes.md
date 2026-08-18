# Capture Recipes

Details behind the SKILL.md "Capture quality" musts — read when tuning a capture.
Driving tool is `sgl_jax.bench_serving --profile` (Path A).

## Step definition

A **step** = one model forward over the batch (`forward_ct`), not a request or a token.
So `--profile-num-steps 5` on decode needs requests generating ≥5 tokens; 5 prefill steps
needs several prefill requests (`--num-prompts` covers this).

## Shaping the load

Split prefill vs decode by annotation in analysis, since prefill forwards carry `extend`.
Shape the workload with `bench_serving` dataset flags:

- `--random-input-len` — larger stresses prefill
- `--random-output-len` — larger produces more decode steps
- `--num-prompts` — how many requests; does **not** fix stage-separated prefill, which is a
  transition race, not a volume problem

To match production, use a real `--dataset-name` or set the random lengths to your p50/p95 and
note them in the handoff.

## Prefix-cache: hand-crafted prompts

SKILL.md covers the `--dataset-name random` + `--flush-cache` musts. The extra trap: if you
ever hand-craft **fixed** prompts instead, vary **content at a fixed token length**. Varying by
*appending* grows the token count across a `precompile_token_paddings` bucket and changes the
padded shape — that, not the content, is what destabilizes the trace. Decode is unaffected
(`ignore_eos` + fixed output len is deterministic).

## Tracer levels

`host_tracer_level` (0 off · 1 TraceMe default · 2 +expensive XLA · 3 +cheap XLA)
and `python_tracer_level` (0 off · 1 default) on `ProfileReqInput`. Raise only
when you need host/python detail — higher levels perturb timing and bloat the file.

## Annotations & multi-host

- Scheduler always emits `run_batch`, `process_batch_result`, and
  `forward_batch_generation {bid}` annotations — no flags needed.
- Each host writes its own trace under `plugins/profile`; collect from all hosts
  for the full picture. PD-disaggregated: profile prefill and decode workers
  separately (`--profile-prefill-url` / `--profile-decode-url`) and label which is which.
