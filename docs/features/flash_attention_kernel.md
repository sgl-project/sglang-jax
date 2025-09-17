
# Flash Attention Kernel

## Goals

The Flash Attention Kernel in SGLang-JAX provides a highly optimized, TPU-friendly implementation of ragged paged attention specifically designed for large language model inference. The core objectives are:

- **TPU Optimization**: Leverage TPU architecture with Pallas kernels for maximum performance on Google Cloud TPU infrastructure
- **Memory Efficiency**: Implement paged attention with intelligent KV cache management to handle long sequences efficiently
- **Mixed Workload Support**: Enable simultaneous processing of prefill and decode requests in a single batch for optimal throughput
- **Production Scalability**: Support high-throughput inference scenarios with automatic performance tuning and robust memory management

## Design

The Flash Attention Kernel implements a sophisticated multi-layer architecture that combines algorithmic optimizations with hardware-specific tuning for TPU environments.

### Core Architecture

The kernel follows a layered design with distinct functional domains:

```
Input Validation → Data Preparation → TPU Kernel Execution → Output Processing
```

#### Key Architectural Components

1. **Ragged Paged Attention Algorithm**: Handles variable-length sequences efficiently through paged memory management
2. **Double Buffering System**: Overlaps computation and memory transfer for optimal TPU utilization
3. **Auto-Tuned Block Sizes**: Dynamic performance optimization based on model configuration and hardware capabilities
4. **Fused KV Cache Management**: Interleaved key-value storage format optimized for TPU memory patterns
5. **Flash Attention**: Implementation based on the IO-aware attention algorithm from Dao et al. [[1]](#references) that uses tiling to reduce memory reads/writes
### Memory Management Design

#### Paged Memory System
- **Page-Based Allocation**: Configurable page sizes (64, 128, 256 tokens) for flexible memory management
- **Fused KV Format**: Interleaved K1,V1,K2,V2... layout reduces memory bandwidth requirements
- **Automatic Alignment**: Hardware-optimized data alignment (128-byte boundaries) for maximum throughput

#### Cache Architecture
```python
kv_cache_fused: [total_num_pages, page_size, num_kv_heads * 2, head_dim]
# Head interleaving format: [K1, V1, K2, V2, ...] for optimal memory access
```

### TPU-Specific Optimizations

#### Pallas Kernel Integration
The kernel leverages JAX Pallas for low-level TPU optimization:
- **Custom Memory Management**: Direct VMEM (Vector Memory) control for optimal data placement
- **Asynchronous Operations**: Overlapped computation and data transfer using TPU semaphores
- **Vectorized Operations**: Hardware-accelerated attention computation with optimal data types

### Advanced Features

#### Multi-Mode Processing
- **Decode Processing**: Single-token generation wit h static q_len=1 for maximum efficiency
- **Prefill Processing**: Chunked processing with configurable chunk sizes for long sequences
- **Mixed Processing**: Simultaneous prefill and decode requests in unified batches

#### Attention Enhancements
- **Sliding Window Attention**: Configurable local attention windows for long-range efficiency
- **Soft Capping**: Logit soft capping for improved training stability
- **Quantization Support**: Native support for quantized KV caches with scaling factors

### Characteristics

The Flash Attention implementation provides significant advantages over native attention:

| Aspect | Flash Attention | Native Attention |
|--------|-----------------|------------------|
| Memory Complexity | **O(N)** | O(N²) |
| Algorithm Approach | **Tiled computation with IO-awareness** | Full matrix computation |
| Computation Strategy | **Incremental softmax with rescaling** | Standard softmax over full matrix |
| TPU Optimization | **Hardware-specific tuning** | Generic operations |
| Paged Attention | ✅ **Supported** | ❌ Not supported |
| Mixed Prefill/Decode | ✅ **Optimized** | ❌ Separate processing |
| Sliding Window | ✅ **Configurable** | ❌ Not supported |

## Usage

### Basic Configuration

The Flash Attention Kernel integrates seamlessly with SGLang-JAX's attention backend system:

```bash
# Launch server with Flash Attention backend
python -m sgl_jax.launch_server \
    --model-path Qwen/Qwen-7B-Chat \
    --attention-backend=fa \
    --device=tpu \
    --trust-remote-code
```

### Parameters

The following parameters configure the ragged paged attention with fused KV cache:

| Parameter | Description |
|-----------|-------------|
| `queries` | Concatenated all sequences' queries |
| `keys` | Concatenated all sequences' keys (quantized) |
| `values` | Concatenated all sequences' values (quantized) |
| `kv_cache_fused` | Paged KV cache with head interleaving format [K1,V1,K2,V2,...] |
| `kv_lens` | Padded kv lengths. Only the first num_seqs values are valid |
| `page_indices` | Flattened page indices look-up table |
| `cu_q_lens` | Cumulative sum of effective query lengths. Only first num_seqs+1 values are valid |
| `distribution` | (i, j, k) where sequences[0:i] are decode-only, sequences[i:j] are chunked-prefill-only, sequences[j:k] are mixed |
| `actual_head_dim` | Actual head size of attention. Assumes k and v have same actual head size |
| `sm_scale` | Softmax scale applied to Q@K^T |
| `sliding_window` | Sliding window size for attention |
| `soft_cap` | Logit soft cap for attention |
| `mask_value` | Mask value for causal mask |
| `k_scale` | Scale for key cache |
| `v_scale` | Scale for value cache |
| `num_kv_pages_per_block` | Number of kv pages processed in one flash attention block |
| `num_queries_per_block` | Number of queries processed in one flash attention block |
| `vmem_limit_bytes` | Vmem limit for pallas kernel |


## References

[1] Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. Advances in Neural Information Processing Systems (NeurIPS). arXiv:2205.14135