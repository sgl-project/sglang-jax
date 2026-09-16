# CSA compressor validation

Single-device TPU v6e operator only; no model integration or TPU-Inference baseline.

## Correctness

Run the independent NumPy suite before benchmarking:

```sh
PYTHONPATH=python:. python -m pytest -q test/srt/kernels/csa_compressor/test_compressor.py
```

BF16 inputs use FP32 reference arithmetic. Pre-quantization values and finite
state elements use `rtol=2e-2, atol=1e-2`; empty scores retain their `-inf` sentinel.
Decoded cache values must fall within the arithmetic budget plus the format's
rounding cell, with separate scale validation. Encoding the same FP32 values
must match byte-for-byte; independently computed caches need not. Untouched
cache slots must remain exact. Downstream attention probes use a propagated
cache-error bound, not a model-quality guarantee.

## Performance

```sh
PYTHONPATH=python:. python benchmark/kernels/csa_compressor/benchmark_compressor.py \
  --batch 4 --sequence 2048 --hidden 4096 --pattern decode prefill ragged \
  --cache-pages 2048 --cache-layout mixed --output compressor.json
```

`--cache-pages 0` allocates the minimum for the input; explicit capacities must
fit all requests. Each page contains 128 records. Layouts are `contiguous`,
`paged` (shuffled physical pages), `mixed` (shuffled rows in every fourth occupied
page), `holes` (one suppressed write in those pages), and `cross_page` (shifted
destinations). Small inputs may not contain enough occupied pages to be mixed.

Report `mean_ms`: complete device-module time, including computation, layout
checks and state/cache writes. Four input buffers rotate; 3 warmups + 10 timed
calls exclude compilation, input generation and host-to-device resets.
`host_mean_ms` includes resets and is not operator latency. Benchmark checks
are finite-value smoke checks, not a substitute for the NumPy suite.

Results apply to the recorded device, software, shape, capacity and layout.
Mixed-block DMA is not universally faster: small-cache and ordinary-layout
regressions must also be reported. The historical 112-case before/after sweep
checked equivalence to the previous implementation, not 112 independent NumPy
oracles; D=7168/B=32/S=8192 and other TPU generations were not covered.
