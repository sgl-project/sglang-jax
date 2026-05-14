# Fused EP MoE V2 Kernel

Double-buffer expert FFN with direct scatter to VMEM (skip HBM intermediate).

## Files

- `kernel.py` — kernel implementation + `ref_moe_simple` reference
- `test_multi.py` — multi-device correctness test

## Run Tests

Upload `kernel.py` and `test_multi.py` to the **same directory** on the pod, then:

```bash
# ep=8 (single pod, e.g. bench-4):
python test_multi.py              # small config (default)
python test_multi.py mimo-v2-pro  # MiMo-V2-Pro config

# ep=32 (4 pods, e.g. ablation-16):
for pod in ablation-16-0-SUFFIX ablation-16-1-SUFFIX ablation-16-2-SUFFIX ablation-16-3-SUFFIX; do
  kubectl exec $pod -c bench -- python /path/to/test_multi.py small &
done
wait
```

## Available Configs

| Name | d | f | E | top_k | bt | bf |
|------|---|---|---|-------|----|----|
| `small` | 768 | 256 | 64 | 2 | 16 | 256 |
| `mimo-v2-pro` | 6144 | 2048 | 128 | 8 | 16 | 256 |

## Test Results

| Config | ep | rel_err | Status |
|--------|----|---------|--------|
| small | 8 | 0.005494 | PASS |
| small | 32 | 0.004237 | PASS |
| mimo-v2-pro | 32 | — | VMEM OOM |
