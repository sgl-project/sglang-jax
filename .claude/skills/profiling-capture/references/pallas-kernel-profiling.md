# Pallas Kernel Profiling

Collect profiling evidence for a **Pallas/Mosaic kernel** (Path C). Start with a plain XProf
trace, then add only the evidence modes required by the profiling question.

## Four composable capture modes

| Mode | What it provides | How to enable it | Best suited for |
| --- | --- | --- | --- |
| Plain trace | JAX/TPU runtime timeline, kernel duration, named scopes | `jax.profiler.trace(...)` | Which kernel is slow, and how long does it take? |
| LLO utilization | LLO bundle utilization aligned to a Pallas custom call | Add two custom-call flags to `LIBTPU_INIT_ARGS` | How are MXU, DMA, and wait time distributed inside the kernel? |
| Compiler dump | Compile-time HLO, MLIR, and Mosaic LLO files | Set `--xla_mosaic_dump_to` and `--xla_dump_to` | What instructions and layouts did the compiler generate? |
| Periodic counters | Fine-grained, time-series hardware counters | Set `ProfileOptions.advanced_configuration` | Why is the MXU idle? Are spills or hardware bottlenecks occurring? |

The modes are composable, but none of the three advanced modes is a default. Select each
additional mode explicitly. Enabling all modes can substantially increase compilation time,
profile size, and runtime collection overhead.

Having an LLO or HLO dump does **not** mean you captured a runtime trace: a failed compile can
leave compiler artifacts but no valid `.xplane.pb`.

## Prerequisites

- Warm up before tracing so compilation and autotuning stay outside the runtime trace.
- Set `LIBTPU_INIT_ARGS` / `XLA_FLAGS` before `import jax` when enabling LLO utilization or
  compiler dumps. Setting them after JAX/libtpu initialization is a silent no-op.
- Use **libtpu ≥ 0.0.39** (or a pinned recent nightly) for LLO utilization. Older versions may
  ignore or reject the custom-call flags with `Unknown command line flag '<flag>'`.
- Use periodic counters only on **Ironwood (TPU v7x) or later** with **JAX ≥0.9** and a
  compatible runtime/libtpu. This repository pins JAX 0.8.1 by default, so counter mode needs
  a separate compatible environment.

## Mode 1 — plain trace (default)

Use no dump flags, custom-call profiling flags, or `ProfileOptions` for the first pass. Name
each phase so Trace Viewer identifies the work, warm up, then capture enough iterations to make
short kernels visible:

```python
import jax

OUT = "/tmp/pallas-profile"


@jax.jit
def run_kernel():
    with jax.named_scope("w1_load"):
        w1 = load_weights()
    with jax.named_scope("gemm"):
        acc = matmul(x, w1)
    with jax.named_scope("store_output"):
        store(acc)
    return acc


for _ in range(3):
    jax.block_until_ready(run_kernel())

with jax.profiler.trace(f"{OUT}/xprof"):
    for _ in range(100):
        result = run_kernel()
    jax.block_until_ready(result)
```

Keep the final `block_until_ready` inside the trace block. Output lands under
`$OUT/xprof/plugins/profile/<ts>/<host>.xplane.pb` plus `.trace.json.gz`.

## Mode 2 — add LLO utilization

Use this layer only when you need custom-call region visibility or custom-call ↔ LLO
utilization correlation. Set both flags before `import jax`, then run the same plain trace:

```bash
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:+$LIBTPU_INIT_ARGS }\
--xla_enable_custom_call_region_trace=true \
--xla_xprof_register_llo_debug_info=true"
```

- `--xla_enable_custom_call_region_trace=true` exposes custom-call regions in the trace.
- `--xla_xprof_register_llo_debug_info=true` lets XProf align a custom call with LLO bundles.

These flags increase captured profile size and collection overhead. Leave them unset for
routine timing traces. See the OpenXLA
[custom-call profiling guide](https://openxla.org/xprof/custom_call_profiling).

## Mode 3 — add compiler dumps

Use this layer only when investigating lowering, layouts, or generated instructions. Set the
flags before `import jax`; they take effect when the kernel compiles:

```bash
OUT=/tmp/pallas-profile
mkdir -p "$OUT/compiler/llo" "$OUT/compiler/hlo"
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:+$LIBTPU_INIT_ARGS }\
--xla_mosaic_dump_to=$OUT/compiler/llo"
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }\
--xla_dump_to=$OUT/compiler/hlo --xla_dump_hlo_as_text --xla_dump_hlo_as_proto"
```

- Mosaic IR/LLO artifacts land in `compiler/llo/`.
- Human-readable HLO and HLO proto files land in `compiler/hlo/`.
- A persistent compilation-cache hit may skip compilation and produce no fresh dumps. Use a
  fresh `JAX_COMPILATION_CACHE_DIR` or clear it when new artifacts are required.

Compiler dumps add compile-time work and disk output; they do not replace the runtime trace.
See the OpenXLA [HLO dump guide](https://openxla.org/xla/hlo_dumps).

## Mode 4 — add periodic counters (Ironwood/v7x+)

Use this layer only for a focused hardware-bottleneck question in a compatible JAX ≥0.9
environment. Periodic counters are independent of the LLO utilization and compiler-dump flags.
This official TC example samples five counter indices; replace the set with counters selected
for the question in XProf's Perf Counters tool:

```python
opts = jax.profiler.ProfileOptions()
opts.advanced_configuration = {
    "tpu_enable_periodic_counter_sampling": True,
    "tpu_tc_perf_counter_sampling_options": (
        "interval_us:1 scaling:0 counter_size_bits:1 "
        "indices:10 indices:11 indices:56 indices:57 indices:58"
    ),
}

with jax.profiler.trace(f"{OUT}/xprof", profiler_options=opts):
    for _ in range(100):
        result = run_kernel()
    jax.block_until_ready(result)
```

The example uses the minimum 1 μs interval and 16-bit packed counters (`counter_size_bits:1`),
so treat it as an aggressive illustrative configuration. For another TPU component, change
both the `tpu_<component>_perf_counter_sampling_options` key and its component-specific indices.
Sample only the components, indices, and frequency needed: more counters and shorter intervals
increase payload size and runtime collection overhead and can cause trace drops. See the
[XProf Kernel guide](https://openxla.org/xprof/kernel-profiling).

## Verify the selected evidence

```bash
OUT=/tmp/pallas-profile
find "$OUT/xprof" -name '*.xplane.pb' -o -name '*.trace.json.gz'
find "$OUT/compiler" -type f 2>/dev/null | head
```

- **Plain trace:** confirm `*.xplane.pb` exists, then check kernel durations and named scopes.
- **LLO utilization:** also confirm the LLO Utilization line, `bundle.*`, and custom-call regions.
- **Compiler dumps:** also confirm fresh files exist under `compiler/llo/` and `compiler/hlo/`.
- **Periodic counters:** also confirm the selected counter lines contain time-series points.

Do not expect LLO utilization, compiler files, or periodic-counter lines from a plain trace.

## Delivery layout

Keep the full `plugins/profile/<ts>/` tree; `xprof --logdir` and TensorBoard consume the logdir
that contains `plugins/profile`.

```text
$OUT/
├── xprof/plugins/profile/<ts>/<host>.xplane.pb
├── compiler/llo/       # only with compiler-dump mode
└── compiler/hlo/       # only with compiler-dump mode
```

For multi-host captures, write each rank to its own `rank-N/` tree. Never let two ranks share a
compiler-dump or trace directory.

## Common pitfalls

| Symptom | Cause → fix |
| --- | --- |
| no runtime trace | kernel did not run, or `block_until_ready` was outside the trace block |
| no LLO utilization | mode not selected, old libtpu, or flags set after `import jax` |
| no compiler dumps | mode not selected, flags set late, or persistent cache skipped compilation |
| LLO/HLO but no xplane | compiler dumps are compile-time evidence, not a runtime capture |
| no periodic counter lines | unsupported TPU generation or incomplete counter configuration |
| trace is unexpectedly large | too many modes, counters, or short sampling intervals enabled |
| VMEM OOM | shrink the tile or set `pltpu.CompilerParams(vmem_limit_bytes=...)` |

## References

- [JAX `profiler.trace`](https://docs.jax.dev/en/latest/_autosummary/jax.profiler.trace.html)
- [OpenXLA custom-call profiling](https://openxla.org/xprof/custom_call_profiling)
- [OpenXLA HLO dumps](https://openxla.org/xla/hlo_dumps)
- [XProf Kernel guide](https://openxla.org/xprof/kernel-profiling)
