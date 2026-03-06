import functools
from math import inf

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.utils.perf import multiple_iteration_timeit_from_trace
from sgl_jax.srt.multimodal.kernels.flash_attention import (
    BlockSizes,
    SegmentIds,
    flash_attention,
)


def create_qkv_data(batch_size, head_num, q_len, kv_len, head_dim):
    key = jax.random.PRNGKey(1)
    key1, key2 = jax.random.split(key, num=2)
    query = jax.random.normal(key, (batch_size, head_num, q_len, head_dim), dtype=jnp.float32)
    key = jax.random.normal(key1, (batch_size, head_num, kv_len, head_dim), dtype=jnp.float32)
    value = jax.random.normal(key2, (batch_size, head_num, kv_len, head_dim), dtype=jnp.bfloat16)
    seg_q = jnp.ones((batch_size, q_len))
    seg_kv = jnp.ones((batch_size, kv_len))
    segment_ids = SegmentIds(q=seg_q, kv=seg_kv)
    return query, key, value, segment_ids


def benchmark_backend(
    batch,
    head_num,
    q_len,
    kv_len,
    head_dim,
    num_queries_per_block,
):
    scale = head_dim**-0.5
    q, k, v, segment_ids = create_qkv_data(batch, head_num, q_len, kv_len, head_dim)
    block_sizes = BlockSizes(
        block_q=num_queries_per_block,
        block_b=1,
        block_k_major=kv_len,
        block_k=kv_len,
    )
    attn = functools.partial(
        flash_attention, q, k, v, segment_ids=segment_ids, sm_scale=scale, block_sizes=block_sizes
    )
    # Warmup
    output = attn()
    jax.block_until_ready(output)

    # Benchmark
    times = multiple_iteration_timeit_from_trace(
        compute_func=lambda: attn(),
        data_generator=lambda: (),
        task=f"block_q_{num_queries_per_block}",
        tries=3,
    )
    avg_time = float(np.mean(times)) if times else float("nan")

    return (
        avg_time,
        q.dtype,
        k.dtype,
        v.dtype,
    )


def main():
    print("JAX devices:", jax.devices())
    print("Device count:", jax.device_count())
    print()

    seq_len_list = 256 * np.array(range(66, 69))
    head_dim_list = [128]
    batch_size_list = [2]
    head_num_list = [12]
    all_combinations = []
    for batch_size in batch_size_list:
        for head_num in head_num_list:
            for q_len in seq_len_list:
                for kv_len in seq_len_list:
                    for head_dim in head_dim_list:
                        all_combinations.append(
                            (
                                batch_size,
                                head_num,
                                q_len,
                                kv_len,
                                head_dim,
                            )
                        )

    num_queries_per_block_list = [128, 256, 512, 1024, 2048, 4096]

    for i, (
        batch_size,
        head_num,
        q_len,
        kv_len,
        head_dim,
    ) in enumerate(all_combinations):
        best_output = inf
        best_config = None
        for i, num_queries_per_block in enumerate(num_queries_per_block_list):
            try:
                (
                    flash_time,
                    q_dtype,
                    k_dtype,
                    v_dtype,
                ) = benchmark_backend(
                    batch_size,
                    head_num,
                    q_len,
                    kv_len,
                    head_dim,
                    num_queries_per_block,
                )
                if flash_time < best_output:
                    best_output = flash_time
                    best_config = (num_queries_per_block,)
            except Exception:
                pass
        if best_config:
            print(
                f"('{q_dtype}', '{k_dtype}', '{v_dtype}', {batch_size}, {head_num}, {q_len}, {kv_len}, {head_dim}): ({best_config[0]},),"
            )


if __name__ == "__main__":
    main()
