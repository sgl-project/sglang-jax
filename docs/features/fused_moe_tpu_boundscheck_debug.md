# Fused MoE TPU debug (boundscheck / debug_print)

`pl.debug_print` inside TPU Mosaic/Pallas kernels requires enabling TPU log recording at compile time, otherwise prints may not show up.

## Recommended workflow

- Enable fused MoE kernel prints:
  - `SGL_FUSED_MOE_DEBUG=1`
- Enable TPU log recorder for JIT compilation:
  - `SGL_TPU_ENABLE_LOG_RECORDER=1`

This repo wires the log-recorder flag into:
- End-to-end model JIT (`jitted_run_model` in `python/sgl_jax/srt/model_executor/model_runner.py`)
- Fused MoE benchmark (`benchmark/moe/bench_fused_moe.py`)

## Standalone example

If you are compiling a kernel manually, pass the compiler option during compilation:

```python
kernel = pl.pallas_call(...)
compiled = jax.jit(kernel).lower(x).compile(
    compiler_options={"xla_tpu_enable_log_recorder": "true"}
)
out = compiled(x)
```

## Notes

- Turning on log recording can increase compile time and log volume; keep it off for normal benchmarking.
