# sglang-jax Ling-mini-2.0 Benchmark Suite

Ling-mini-2.0 (16B MoE, 256 experts) inference benchmark for sglang-jax on TPU v6e.

## Files

| File | Description |
|------|-------------|
| `sglang_bench.py` | Benchmark script: quality check (5 prompts) + throughput matrix (24 data points) |
| `sglang-2x2-bench.yaml` | K8s Job for single-node 2x2 (4 chips, tp=4) |
| `sglang-4x4-bench-jobset.yaml` | K8s JobSet for multi-node 4x4 (16 chips, tp=16, 4 nodes) |

## Quick Start

### 2x2 (single node, 4 chips)

```bash
kubectl apply -f sglang-2x2-bench.yaml
kubectl logs -f job/sglang-2x2-bench
```

### 4x4 (multi-node, 16 chips)

Requires a 4x4 TPU node pool (4 nodes x 4 chips) and JobSet CRD (GKE >= 1.35 built-in).

```bash
kubectl apply -f sglang-4x4-bench-jobset.yaml
# Watch worker 0 logs:
kubectl logs -f $(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=sglang-4x4-bench \
    -o jsonpath='{.items[0].metadata.name}')
```

## Configuration

All parameters are environment variables — override in YAML `env` section or `export` before running the script directly.

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `/tmp/models/inclusionAI/Ling-mini-2.0` | Model checkpoint path |
| `MODEL_GCS_PATH` | `gs://wenqiao-ainfer-test/models/inclusionAI/Ling-mini-2.0` | GCS path to download model from |
| `SGLANG_GCS_PATH` | `gs://wenqiao-ainfer-test/sglang-jax.tar.gz` | GCS path to sglang-jax source tarball |
| `TP_SIZE` | `4` (2x2) / `16` (4x4) | Tensor parallel size |
| `NNODES` | `1` (2x2) / `4` (4x4) | Number of nodes |
| `DIST_INIT_ADDR` | (auto from JobSet DNS) | Coordinator address for multi-node |
| `BS_LIST` | `1,2,4,8,16,32` | Batch sizes to benchmark |
| `TOKEN_LIST` | `128,256,512,1024` | Max token lengths to benchmark |
| `WARMUP_ROUNDS` | `1` | Warmup rounds before timed runs |
| `TIMED_ROUNDS` | `2` | Timed rounds for averaging |
| `JAX_COMPILATION_CACHE_DIR` | (set in YAML) | GCS path for XLA compilation cache |

## Output Format

Quality check prints human-readable Q&A pairs with `QUALITY_CHECK: PASS/FAIL`.

Throughput results are JSON lines (one per data point), parseable with `jq`:

```json
{"max_tokens": 128, "bs": 1, "total_tokens": 128, "elapsed_s": 0.773, "tok_per_s": 165.6}
```

```bash
# Extract throughput table from logs:
kubectl logs job/sglang-2x2-bench | grep tok_per_s | python3 -m json.tool
```

## Known Limitations

- **dp>1 + multi-node**: `enable_single_process=True` + `nnodes>1` + `dp_size>1` has a bug
  in sglang-jax (`scheduler_pipe_readers` unbound for `node_rank>=1`). Multi-node currently
  uses pure TP only.
- **tp=16 suboptimal for small hidden_dim**: Ling-mini-2.0 (hidden=2048) gets ~18% slower at
  bs=1 with tp=16 vs tp=4 due to cross-node communication overhead. `dp=4,tp=4` would be
  ideal but requires the bug fix above.
- **Multi-node exit**: Worker 0 uses `os._exit(0)` to avoid JAX shutdown barrier timeout.
  Workers 1-3 will show `Failed` status — this is expected.
