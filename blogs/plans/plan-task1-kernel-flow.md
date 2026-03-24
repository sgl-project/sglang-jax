# Plan: Task 1 — Fused MoE Kernel Flow Document

## Goal
Create a standalone document explaining how the fused EP MoE kernel works, to support blog section 3.2.

## Approach
1. Read the full kernel source (`kernel.py`, ~1699 lines) to trace execution flow
2. Cross-reference with existing walkthrough notes (`FusedMoE kernel 走读.md`)
3. Document each pipeline stage with code locations
4. Identify and explain the three levels of double buffering
5. Create ASCII pipeline diagram showing overlap pattern
6. Document tuning parameters and their trade-offs

## Status: Complete
All sections written and saved to `blogs/docs/FusedMoE-Kernel-Flow.md`.
