import jax
import jax.numpy as jnp
import numpy as np
import time
from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention

def benchmark_metadata_jit():
    # Setup
    delimiter_id = 12345
    q_len = 2000 + 500 * 20 # 12000 tokens
    # Generate tokens: Query <D> Item1 <D> Item2 ...
    tokens = np.zeros(q_len, dtype=np.int32)
    tokens[2000] = delimiter_id
    for i in range(1, 501):
        tokens[2000 + i * 21] = delimiter_id
    
    tokens_jax = jnp.array(tokens)
    
    print(f"Benchmarking metadata calculation for {q_len} tokens, 500 items...")
    
    # 1. Benchmark NumPy (Current)
    start_cpu = time.perf_counter()
    for _ in range(100):
        # We manually call the logic from _build_multi_item_segment_layout
        delimiter_indices = np.flatnonzero(tokens == delimiter_id)
        prefix_end = int(delimiter_indices[0]) + 1
        row_seg_starts = np.zeros((q_len,), dtype=np.int32)
        for i in range(delimiter_indices.size - 1):
            seg_start = int(delimiter_indices[i]) + 1
            seg_end_delim = int(delimiter_indices[i + 1])
            row_seg_starts[seg_start : seg_end_delim + 1] = seg_start
    end_cpu = time.perf_counter()
    cpu_time = (end_cpu - start_cpu) / 100
    print(f"NumPy (CPU) Average Time: {cpu_time * 1000:.4f} ms")

    # 2. Benchmark JIT (Proposed)
    # Warmup
    FlashAttention._calculate_row_seg_starts_jit(tokens_jax, delimiter_id)
    
    start_jit = time.perf_counter()
    for _ in range(100):
        res = FlashAttention._calculate_row_seg_starts_jit(tokens_jax, delimiter_id)
        res[0].block_until_ready()
        res[1].block_until_ready()
    end_jit = time.perf_counter()
    jit_time = (end_jit - start_jit) / 100
    print(f"JAX JIT Average Time: {jit_time * 1000:.4f} ms")
    
    print(f"Speedup: {cpu_time / jit_time:.2f}x")

if __name__ == "__main__":
    benchmark_metadata_jit()
