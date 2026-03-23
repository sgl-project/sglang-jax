# Ling-1T support

## Motivation

SGLang-JAX now supports the Ling-2.0 architecture and has been performance-optimized for small models such as inclusionAI/Ling-mini-2.0, but it still faces some challenges on larger models. Now aim to optimize the framework based on the Ling-1T model to achieve strong performance across metrics, and we expect these general-purpose optimizations to also apply to other large models.

## Goals

*   Achieve 80% of SGLang’s performance on GPU.


## Design

Current potential risks: 

*   MoE: 

    *   Dispatch uses allgather instead of a2a.

    *   EP load imbalance. 

*   Attention: 

    *   Attention heads are too small (=64); v6e requires deployment on 128 GPUs, causing redundant computation (can be avoided with DP-ATTN, which we currently don’t support).

    *   Ling-1T supports up to 128K context length; the FA kernel is limited by SMEM size. With smaller page\_size (higher KV cache hit rate), batch size can’t be scaled up.


Baseline selection: 

*   Performance of Ling-1T + SGLang on GPU.

*   Theoretical compute performance limit on TPU.


Other potential improvements:  

*   Quantization


## TODO

1.  Download model (2TB size, cost > 12h).

2.  Baseline testing(contains accuracy testing, baseline performance testing, and profiling)

    1.  SGLang-JAX on TPU

    2.  SGLang on GPU

3.  Performance analysis

    1.  Visualize an accurate computation graph based on profiling or HLO (with automatic parallelism, communication is mostly inserted implicitly).

    2.  Evaluate the arithmetic intensity of core modules (such as FlashAttention，MegaBlox Gemm) and identify optimization headroom.
