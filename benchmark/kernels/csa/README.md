# Single-device CSA

`csa_attention` composes the compressor, existing DSA Indexer/Top-K and joint
attention. It accepts hidden activations, projected attention/index queries,
new uncompressed KV, model parameters, caller-owned metadata and `CSACache`.
It returns attention output and the six updated cache/state buffers; reuse
those returned buffers because the inputs are donated.

Retrieval is bypassed when all completed compressed entries fit in Top-K.
Decode uses SparseCore packed gather through temporary HBM, followed by
TensorCore joint attention and row-wise SWA writeback. Prefill/ragged share
page reads across query blocks with per-query causal masks. Persistent cache
formats are unchanged. Query/KV projections, output projection and model-serving
registration are outside this operator.

`prepare_csa_metadata` accepts host-side request lengths, prefixes, state slots
and fixed-stride page tables. It does not allocate storage. Active requests own
distinct pages/state slots, page zero is reserved, and token padding is trailing.
The decode schedule requires at most one query per request.

## Test and benchmark

```bash
PYTHONPATH=python:. python -m pytest -q \
  test/srt/kernels/csa test/srt/kernels/csa_compressor test/srt/kernels/csa_attention

PYTHONPATH=python:. python benchmark/kernels/csa/bench_csa.py \
  --pattern decode_emit --batch 4 --sequence 4096 --hidden 4096 --heads 64
```

Patterns are `decode_emit`, `decode_update`, `prefill` and `ragged`. Emit finishes
a four-token compression group; update is the following non-emitting step.
Prefill has S queries per request. Ragged uses `[1, S/4, S/2, S]` repeatedly,
or S/2 queries for a single request, with prefixes ending at context S.

Latency is the complete CSA device module, including compression, retrieval,
temporary HBM staging, attention and persistent writes. Four independent inputs
rotate, with three warmups and ten timed calls (arithmetic mean). Input/cache
restoration and compilation are outside device timing. Host time is reported
separately. The validated platform is a single TPU v6e.

## Measurements

TPU v6e-1, JAX/JAXLIB 0.11.1, hidden size 4096, 64 attention heads (D=512),
64 index heads (D=128), BF16 inputs and default dot precision. Decode includes
compression output. All tests passed before measurement.

### Decode (µs)

| Batch / context length | 1024 | 2048 | 4096 | 8192 |
|---:|---:|---:|---:|---:|
| 1 | 86.85 | 87.87 | 107.25 | 114.18 |
| 4 | 72.81 | 75.89 | 104.38 | 114.88 |
| 8 | 81.40 | 93.56 | 124.36 | 139.98 |
| 16 | 109.65 | 142.77 | 172.82 | 197.57 |
| 32 | 180.24 | 225.69 | 273.41 | 325.17 |

### Prefill (ms)

| Batch / context length | 1024 | 2048 | 4096 | 8192 |
|---:|---:|---:|---:|---:|
| 1 | 0.52 | 0.96 | 3.39 | 10.86 |
| 4 | 1.65 | 3.84 | 13.47 | 44.67 |
| 8 | 3.38 | 8.02 | 27.78 | 93.28 |
| 16 | 7.11 | 16.14 | 59.40 | 200.28 |
| 32 | 14.31 | 35.67 | 131.47 | OOM |
