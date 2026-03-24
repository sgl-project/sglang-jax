# Task 1: Fused MoE Kernel Flow Document

**Status**: Complete
**Output**: `blogs/docs/FusedMoE-Kernel-Flow.md`
**Source files used**:
- `blogs/tpu-inference/fused_moe/v1/kernel.py`
- `blogs/tpu-inference/fused_moe/v1/tuned_block_sizes.py`
- `blogs/docs/FusedMoE kernel 走读.md`

## Deliverables

- [x] Overview — what the kernel does
- [x] Pipeline stages (routing, scatter, FFN1, FFN2, gather, accumulation, writeback)
- [x] Key design patterns (3 levels of double buffering, overlap strategy, token packing, dynamic token count)
- [x] Tuning parameters table (bt, bf, bd1, bd2, btc, bfc, bd1c, bd2c)
- [x] Pipeline diagram showing overlap
- [x] Ring-1T specific configuration
