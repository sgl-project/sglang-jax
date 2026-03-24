# Task 2: Roofline Analysis of Fused MoE Kernel

**Status**: Complete (theoretical analysis). Profiling (2b) and new experiments (2c) require TPU cluster access.
**Output**: `blogs/docs/FusedMoE-Roofline-Analysis.md`

## Deliverables

### 2a. Theoretical Roofline Calculation — Complete
- [x] TPU v7 specs (peak FLOPS, HBM BW, ICI BW, critical AI)
- [x] Per-stage arithmetic intensity at Ring-1T dimensions
- [x] Compute-bound vs memory-bound analysis at various token counts (1–4096)
- [x] Bottleneck identification: HBM bandwidth (weight loading) is primary bottleneck

### 2b. Profiling — Requires TPU Cluster
- [ ] Run xprof profiling on TPU v7 cluster
- [ ] Collect per-stage time breakdown
- [ ] Compare measured vs theoretical predictions
- [ ] Identify gap between actual and roofline ceiling

### 2c. Ablation Study — Partially Complete
- [x] Organized existing ablation results from `FusedMoE_benchmark.md`
- [x] Connected optimizations to roofline analysis
- [ ] New experiments: impact of expert load imbalance at different levels
- [ ] New experiments: weight prefetch overlap effectiveness measurement
- [ ] Quantitative gap analysis with actual profiling data

## Key Findings (Updated with official TPU v7 Ironwood specs)

1. **Hardware**: BF16 2307 TFLOPS, FP8 4614 TFLOPS, HBM 7380 GB/s, ICI 1200 GBps bidirectional
2. **Critical AI**: 312.6 FLOP/Byte (BF16 vs HBM), 625.2 (FP8 vs HBM)
3. **FFN is deeply memory-bound** at decode: BS=512 → AI=33.6, far below 312.6 threshold
4. **Compute-bound requires**: BS≈4768 (BF16 compute + FP8 weights) or BS≈9536 (FP8 compute + FP8 weights)
5. **v7 shifts bottleneck**: HBM BW 4.5x increase → weight loading (0.052ms at EP=32/FP8) now comparable to A2A comm (~0.3ms)
6. **FP8 value**: Halves weight loading in bandwidth-bound regime, ~2x speedup
7. **Shared expert**: Already compute-bound at BS=512 (AI=1056 >> 312.6)
8. **Weight prefetch** remains most impactful optimization for sparse expert FFN
9. **Model config**: BailingMoeV2 — hidden=8192, intermediate=2048, 256 experts, top_k=8, FP8 W8A8
