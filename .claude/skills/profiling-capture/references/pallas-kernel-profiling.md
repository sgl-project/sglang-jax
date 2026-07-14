# Pallas Kernel Profiling

Producing LLO + HLO + XProf artifacts for a **Pallas/Mosaic kernel** (Path C).

## Three artifact types

| Artifact | When | Path |
| --- | --- | --- |
| Mosaic **LLO** dump | compile time | `compiler/llo/` |
| HLO / MLIR dump | compile time | `compiler/hlo/` |
| **XProf** trace / xplane | run time | `plugins/profile/<ts>/` |

Having an LLO dump does **not** mean you captured a runtime trace — a failed
compile can still leave LLO/HLO but no valid `.xplane.pb`.

## Prerequisites

- **libtpu ≥ 0.0.39** (or a pinned recent nightly). On older libtpu the LLO debug
  flags may not exist: some are silently ignored (flags set, no LLO utilization in
  the trace), others are rejected outright — libtpu aborts at startup with
  `Unknown command line flag '<flag>'` (seen with `--xla_enable_custom_call_region_trace`
  on libtpu 0.0.30). Pin a recent libtpu before setting them.
- **Set flags before `import jax`.** `LIBTPU_INIT_ARGS` / `XLA_FLAGS` must be
  exported in the shell or set via `os.environ` at the very top, before any JAX /
  libtpu init. Setting them late = silent no-op.
- Periodic perf counters are mainly **v7/Ironwood+**; on v6e expect custom-call /
  LLO region info but not the fine-grained counters.

## Dump flags

```bash
OUT=/tmp/pallas-profile      # any writable dir
mkdir -p "$OUT/compiler/llo" "$OUT/compiler/hlo"
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:+$LIBTPU_INIT_ARGS }\
--xla_enable_custom_call_region_trace=true \
--xla_xprof_register_llo_debug_info=true \
--xla_mosaic_dump_to=$OUT/compiler/llo"
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }\
--xla_dump_to=$OUT/compiler/hlo --xla_dump_hlo_as_text --xla_dump_hlo_as_proto"
```

- `--xla_mosaic_dump_to` → Pallas/Mosaic LLO
- `--xla_enable_custom_call_region_trace=true` → custom-call regions enter the trace
- `--xla_xprof_register_llo_debug_info=true` → XProf can align custom-call ↔ LLO
- `--xla_dump_hlo_as_text` → human-readable `*.txt` HLO; `--xla_dump_hlo_as_proto` → `*.pb` for tooling
- LLO **must** land in `compiler/llo/` (or `rank-*/compiler/llo/`) — analysis
  tooling scans that path; dumping elsewhere reads as `NO_LLO`.

## Capture skeleton

Illustrative skeleton (placeholders — substitute your kernel). Name each phase with
`jax.named_scope` so Trace Viewer shows business-stage names, then warm up and record:

```python
# Run with the LIBTPU_INIT_ARGS / XLA_FLAGS from "Dump flags" already exported,
# so they take effect before the first `import jax` below.
import jax

OUT = "tmp/pallas-profile"

@jax.jit
def run_kernel():
    # wrap each phase in its own named_scope context manager
    with jax.named_scope("w1_load"):
        w1 = load_weights()
    with jax.named_scope("gemm"):
        acc = matmul(x, w1)
    with jax.named_scope("store_output"):
        store(acc)
    return acc

opts = jax.profiler.ProfileOptions()          # add counter config only for counters mode
for _ in range(3):                            # warm up so compile/autotune is outside the trace
    jax.block_until_ready(run_kernel())
with jax.profiler.trace(f"{OUT}/xprof", profiler_options=opts):
    for _ in range(100):                      # a ns-scale kernel needs many iters or the window is ~empty
        r = run_kernel()
    jax.block_until_ready(r)                  # block_until_ready MUST be inside the block
```

Output lands under the logdir passed to `jax.profiler.trace` — here
`$OUT/xprof/plugins/profile/<ts>/<host>.xplane.pb` (+ `.trace.json.gz`).

## Verify the capture is valid

```bash
find "$OUT" -name '*.xplane.pb' -o -name '*.trace.json.gz'
find "$OUT" -name '*.xplane.pb' -print0 | xargs -0 \
  grep -aE 'MXU|tpu_custom_call|bundle\.|gemm|matmul' | head
```
Good signals: `MXU`, `Vector Load/Store/Fills/Spills`, `Scalar ALU`, `XLU`,
`bundle.*`, `tpu_custom_call`, your `named_scope` names.

## Delivery layout

Keep the full `plugins/profile/<ts>/` tree — `xprof --logdir` / TensorBoard need it.
Multi-host: each rank writes its own `rank-N/...` — never let two ranks write the
same `compiler/llo/` or trace dir.

## Common pitfalls

| Symptom | Cause → fix |
| --- | --- |
| no LLO utilization | old libtpu / flags set late → pin libtpu, set flags before `import jax` |
| `NO_LLO` | wrong dump path → `compiler/llo/` or `rank-*/compiler/llo/` |
| job SUCCEEDED, no trace | benchmark swallowed a compile/runtime error → check logs + `.xplane.pb` |
| LLO/HLO but no xplane | kernel didn't run / `block_until_ready` outside trace block |
| stale or empty `compiler/llo/` on a fresh `$OUT` | a persistent compile cache skipped recompile → no new Mosaic dump; unset/clear `JAX_COMPILATION_CACHE_DIR` (or use a fresh cache dir) so the kernel recompiles |
| VMEM OOM | shrink tile / `pltpu.CompilerParams(vmem_limit_bytes=...)` |

## References

OpenXLA custom-call profiling: https://openxla.org/xprof/custom_call_profiling ·
XProf kernel guide: https://openxla.org/xprof/kernel-profiling ·
Mosaic dump errors: https://openxla.org/xla/errors/error_3000
