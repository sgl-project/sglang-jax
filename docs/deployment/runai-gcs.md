# Loading GCS checkpoints with RunAI

The optional `runai_streamer` loader reads safetensors byte ranges directly
from GCS. It downloads configuration and tokenizer files into a local cache,
then uses RunAI Model Streamer for weight I/O. This avoids staging complete
weight files on local disk and bypasses the GCSFuse full-file warmup.

Install the extra on each Linux serving host, alongside the usual TPU dependencies:

```bash
pip install -e 'python[runai]'
```

Use a GCS directory containing the model's configuration, tokenizer, and
safetensors checkpoint. The pod or host must already have GCS read/list access
through Application Default Credentials, for example GKE Workload Identity.
No GCSFuse mount is needed for this loading path.

```bash
python -m sgl_jax.launch_server \
  --model-path gs://YOUR_BUCKET/models/YOUR_MODEL \
  --load-format runai_streamer \
  --model-loader-extra-config '{"concurrency":16,"memory_limit":268435456}'
```

`--load-format auto` also selects this loader for a `gs://` model path. Local
directories and Hugging Face model IDs keep their existing behavior under
`auto`; explicitly selecting `runai_streamer` also works with local safetensors.
For a Hugging Face ID, files are downloaded through the existing HF cache first.

| Option | Meaning |
|---|---|
| `concurrency` | Positive integer; sets the SDK's `RUNAI_STREAMER_CONCURRENCY`. |
| `memory_limit` | Positive byte count; sets the SDK's `RUNAI_STREAMER_MEMORY_LIMIT`. This limits SDK staging buffers, not total process memory. |
| `distributed` | Only `false` is supported. JAX processes independently read their addressable shards; the PyTorch distributed streamer is not used. |
| `--download-dir` | Root for the metadata cache. Defaults to `HF_HOME`, or `~/.cache/huggingface`. |

Unset SDK options retain the environment/SDK defaults. Settings are process-wide.
The loader copies each received chunk into an owned host buffer before asking
the SDK for another chunk. Allow memory for those host shards and any JAX
transfers in addition to the SDK staging limit.

The cache lives in `sglang-jax-runai/<URI hash>` beneath the cache root. The loader
locks metadata downloads and marks completed downloads for reuse. Listing and
metadata downloads use object read/list permissions without fetching bucket metadata.
Treat the
GCS prefix as immutable: use a new prefix for a new checkpoint, or remove its
metadata cache directory while serving processes are stopped. Primary and draft
model URIs use separate cache entries. Metadata is cached separately on each host.

## Supported scope

- Text models using the shared JAX `WeightLoader`, including its transpose,
  QKV split, expert stacking, dtype conversion and sharding paths.
- Safetensors checkpoints, optionally with `model.safetensors.index.json`.
  When an index is present, only its referenced shard files are read.
- GCS and local files. Other object-storage schemes and encrypted checkpoints
  are not supported by this integration.
- Multimodal, Gemma4 and Qwen3.5 models currently have additional local-file
  loading paths and are rejected by this loader. Use local checkpoints with
  `--load-format auto` for these models.

The dependency is pinned to RunAI Model Streamer 0.16.1 because its `FileChunks`
interface is version-specific. This integration uses `FileStreamer` byte-range
requests rather than a sequential PyTorch tensor iterator, so existing JAX
weight mappings can request local shards in their own order.

## Validation and performance

CPU tests compare final values, dtypes and shardings against local safetensors
loading, including QKV and cross-file MoE mappings. Run them with:

```bash
JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=4 PYTHONPATH=python \
  python -m pytest -q python/sgl_jax/test/test_runai_loader.py
```

These checks do not establish a GCS speedup. For a loading-time comparison,
use the same checkpoint revision, host topology and dtype, record cache state,
and measure weight-loading wall time and peak host RSS. Compare the existing
GCSFuse path with this direct GCS path; separate metadata download, weight I/O
and model compilation times.
