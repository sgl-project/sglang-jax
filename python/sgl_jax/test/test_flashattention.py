import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.flash_attn_kernel.flash_attention import (
    ref_ragged_paged_attention,
)
from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention
from sgl_jax.srt.layers.attention.native_backend import NativeAttention
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.test.test_utils import CustomTestCase

mesh = create_device_mesh(ici_parallelism=[1, -1, 1, 1], dcn_parallelism=[1, 1, 1, 1])
jax.sharding.set_mesh(mesh)


def unique_in_original_order(arr: jax.Array) -> jax.Array:
    unique_info = jnp.unique_all(arr)
    unique_values = unique_info.values
    original_indices = unique_info.indices

    # Sort the original indices to get the correct order
    sorted_order = jnp.argsort(original_indices)

    # Reorder the unique values based on the sorted indices
    unique_in_original_order = unique_values[sorted_order]
    return unique_in_original_order


def create_qkv_cache(
    lens,
    num_heads,
    head_dim,
    num_kv_heads,
    page_size=1,
    dtype=jnp.bfloat16,
):
    batched_q_len = sum([q_len for q_len, _ in lens])
    batched_kv_len = sum([kv_len for _, kv_len in lens])

    # Calculate aligned batched_kv_len
    seq_lens = jnp.array([kv_len for _, kv_len in lens], dtype=jnp.int32)
    aligned_seq_lens = ((seq_lens + page_size - 1) // page_size) * page_size
    batched_aligned_kv_len = jnp.sum(aligned_seq_lens).item()

    key = jax.random.PRNGKey(42)
    # Use realistic scale factor to simulate RMSNorm output (typically in range [-3, 3])
    scale_factor = 1  # More realistic for post-RMSNorm values
    q = (
        jax.random.normal(key, (batched_q_len, num_heads, head_dim), dtype=dtype)
        * scale_factor
    )

    # Create k,v with proper alignment gaps between sequences
    k = jnp.zeros((batched_aligned_kv_len, num_kv_heads, head_dim), dtype=dtype)
    v = jnp.zeros((batched_aligned_kv_len, num_kv_heads, head_dim), dtype=dtype)

    # Fill in the actual data for each sequence with proper alignment
    actual_pos = 0
    aligned_pos = 0
    for seq_len in [kv_len for _, kv_len in lens]:
        aligned_len = ((seq_len + page_size - 1) // page_size) * page_size

        # Generate data for this sequence
        seq_k = (
            jax.random.normal(
                jax.random.split(key, len(lens) * 2)[actual_pos],
                (seq_len, num_kv_heads, head_dim),
                dtype=dtype,
            )
            * scale_factor
        )
        seq_v = (
            jax.random.normal(
                jax.random.split(key, len(lens) * 2)[actual_pos + len(lens)],
                (seq_len, num_kv_heads, head_dim),
                dtype=dtype,
            )
            * scale_factor
        )

        # Place data at aligned positions
        k = k.at[aligned_pos : aligned_pos + seq_len].set(seq_k)
        v = v.at[aligned_pos : aligned_pos + seq_len].set(seq_v)

        actual_pos += 1
        aligned_pos += aligned_len

    return q, k, v


def write_prefix_tokens_for_kv(forward_batch, lens, k, v):
    page_size = forward_batch.attn_backend.page_size
    # Use aligned positions for k/v indexing since k/v arrays are created with alignment gaps
    aligned_seq_lens = (
        (forward_batch.seq_lens + page_size - 1) // page_size
    ) * page_size
    aligned_cache_loc_idx = jnp.concatenate(
        [jnp.array([0], dtype=jnp.int32), jnp.cumsum(aligned_seq_lens)]
    )

    extend_k = []
    extend_v = []
    for i, (q_len, kv_len) in enumerate(lens):
        start = aligned_cache_loc_idx[i]
        prefix_end = start + (kv_len - q_len)
        extend_start = prefix_end
        extend_end = start + kv_len

        print(
            f"start: {start}, prefix_end: {prefix_end}, extend_start: {extend_start}, extend_end: {extend_end}"
        )

        if kv_len > q_len:
            # write prefix token
            prefix_cache_loc = forward_batch.cache_loc[start:prefix_end]
            prefix_k = k[start:prefix_end]
            prefix_v = v[start:prefix_end]
            forward_batch.token_to_kv_pool.set_kv_buffer(
                0, prefix_cache_loc, prefix_k, prefix_v
            )

        extend_k.append(k[extend_start:extend_end])
        extend_v.append(v[extend_start:extend_end])

    return jnp.concatenate(extend_k), jnp.concatenate(extend_v)


def create_test_data(
    mode,
    lens,  # [(q_len, kv_len)], kv_len includes q_len
    num_heads,
    head_dim,
    num_kv_heads,
    page_size,
    input_ids=None,
    model_config=None,
    max_total_token_size=200000,
    dtype=jnp.bfloat16,
):
    """Create a real ForwardBatch for testing."""
    assert mode in ["prefill", "decode"]
    batch_size = len(lens)
    # Create sequence lengths array
    seq_lens = jnp.array([kv_len for _, kv_len in lens], dtype=jnp.int32)
    total_q_lens = sum([q_len for q_len, _ in lens])

    # Align seq_lens to page_size for cache allocation
    aligned_seq_lens = ((seq_lens + page_size - 1) // page_size) * page_size
    total_aligned_tokens = jnp.sum(aligned_seq_lens).item()

    # Create dummy input_ids if not provided
    if input_ids is None:
        input_ids = jnp.arange(total_q_lens, dtype=jnp.int32)

    # Create fake positions, not used in attention
    positions = jnp.arange(total_aligned_tokens, dtype=jnp.int32)
    # Create fake extend_start_loc, not used in attention
    extend_start_loc = jnp.ones((batch_size,), dtype=jnp.int32)
    # fake req_pool_indices, not used in attention
    req_pool_indices = jnp.arange(batch_size, dtype=jnp.int32)

    current_kv_cache = MHATokenToKVPool(
        size=max_total_token_size,
        page_size=page_size,
        dtype=jnp.bfloat16 if model_config["bf16"] else jnp.float32,
        head_num=model_config["num_kv_heads"],
        head_dim=model_config["head_dim"],
        layer_num=model_config["num_hidden_layers"],
        mesh=mesh,
    )
    # create q, k v
    q, k, v = create_qkv_cache(
        lens, num_heads, head_dim, num_kv_heads, page_size, dtype
    )

    # cache loc - match schedule_batch.py logic with align_to_size
    def align_to_size(l, size, value=0):
        align_len = (len(l) + size - 1) // size * size
        return l + [value] * (align_len - len(l))

    cache_loc_flat = []
    current_aligned_pos = 0  # Track aligned position in k/v cache

    for i, (_, kv_len) in enumerate(lens):
        # Create token indices for this sequence based on actual k/v storage position
        seq_token_indices = list(
            range(current_aligned_pos, current_aligned_pos + kv_len)
        )
        # Apply alignment padding to this sequence
        aligned_seq_indices = align_to_size(seq_token_indices, page_size, 0)
        cache_loc_flat.extend(aligned_seq_indices)
        # Move to next aligned position (matches k/v storage)
        aligned_len = ((kv_len + page_size - 1) // page_size) * page_size
        current_aligned_pos += aligned_len

    cache_loc = jnp.array(cache_loc_flat, dtype=jnp.int32)
    if mode == "prefill":
        # out_cache_loc - use aligned seq_lens for cache indexing
        cache_loc_idx = jnp.concatenate(
            [jnp.array([0], dtype=jnp.int32), jnp.cumsum(aligned_seq_lens)]
        )
        out_cache_loc = []
        extend_prefix_lens = []
        extend_seq_lens = []
        for i, (q_len, kv_len) in enumerate(lens):
            start = cache_loc_idx[i]
            # Use actual seq_len for the sequence, not aligned
            actual_end = start + seq_lens[i]
            extend_prefix_len = kv_len - q_len
            out_start = start + extend_prefix_len

            out_cache_loc.append(cache_loc[out_start:actual_end])
            extend_prefix_lens.append(jnp.array([extend_prefix_len], dtype=jnp.int32))
            extend_seq_lens.append(jnp.array([q_len], dtype=jnp.int32))

        out_cache_loc = jnp.concatenate(out_cache_loc, dtype=jnp.int32)
        extend_prefix_lens = jnp.concatenate(extend_prefix_lens, dtype=jnp.int32)
        extend_seq_lens = jnp.concatenate(extend_seq_lens, dtype=jnp.int32)
    else:
        # out_cache_loc - use aligned seq_lens for cache indexing
        cache_start_loc = jnp.concatenate(
            [jnp.array([0], dtype=jnp.int32), jnp.cumsum(aligned_seq_lens)]
        )
        out_cache_loc = []
        for i, (q_len, kv_len) in enumerate(lens):
            start = cache_start_loc[i]
            # Use actual seq_len for the sequence end
            actual_end = start + seq_lens[i]
            out_start = actual_end - 1
            out_cache_loc.append(cache_loc[out_start:actual_end])

        out_cache_loc = jnp.concatenate(out_cache_loc, dtype=jnp.int32)
        # extend_prefix_len
        extend_prefix_lens = None
        extend_seq_lens = None

    # init attention backend
    attention_backend = FlashAttention(
        num_heads, num_kv_heads, head_dim, page_size=page_size
    )
    forward_mode = ForwardMode.EXTEND if mode == "prefill" else ForwardMode.DECODE

    fb = ForwardBatch(
        forward_mode=forward_mode,
        batch_size=batch_size,
        input_ids=input_ids,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        out_cache_loc=out_cache_loc,
        positions=positions,
        extend_start_loc=extend_start_loc,
        token_to_kv_pool=current_kv_cache,
        attn_backend=attention_backend,
        cache_loc=cache_loc,
        extend_prefix_lens=extend_prefix_lens,
        extend_seq_lens=extend_seq_lens,
    )
    attention_backend.init_forward_metadata(fb)
    return fb, q, k, v


class TestAttention(CustomTestCase):
    """Test cases for the Attention layer."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

        # Initialize random seeds for reproducible results
        self.rng_key = jax.random.PRNGKey(42)
        np.random.seed(42)

    def run_test(self, mode, lens, mode_args):
        # Create mock forward_batch
        num_heads, head_dim, num_kv_heads, page_size, dtype = mode_args

        if dtype == jnp.bfloat16:
            is_bf16 = True
        else:
            is_bf16 = False

        forward_batch, q, k, v = create_test_data(
            mode,
            lens,
            num_heads,
            head_dim,
            num_kv_heads,
            page_size,
            model_config={
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "num_hidden_layers": 1,
                "bf16": is_bf16,
            },
        )

        # Debug cache mapping
        print(f"=== Cache Mapping Debug ===")
        print(f"lens: {lens}")
        print(f"seq_lens: {forward_batch.seq_lens}")
        print(f"cu_q_lens: {forward_batch.attn_backend.forward_metadata.cu_q_lens}")
        print(f"cu_kv_lens: {forward_batch.attn_backend.forward_metadata.cu_kv_lens}")
        print(f"cache_loc: {forward_batch.cache_loc[:100]}")
        print(f"cache_loc[100:200]: {forward_batch.cache_loc[100:200]}")
        print(f"out_cache_loc: {forward_batch.out_cache_loc[:100]}")
        print()

        # Create test data
        shading = jax.sharding.NamedSharding(mesh, P(None, "tensor"))
        q_shard = jax.device_put(q.copy(), shading).reshape(q.shape[0], -1)
        k_cache_shard = jax.device_put(k.copy(), shading)
        v_cache_shard = jax.device_put(v.copy(), shading)

        # write prefix tokens
        extend_k, extend_v = write_prefix_tokens_for_kv(
            forward_batch, lens, k_cache_shard, v_cache_shard
        )
        extend_k = extend_k.reshape(extend_k.shape[0], -1)
        extend_v = extend_v.reshape(extend_v.shape[0], -1)

        # JAX attention
        attn = RadixAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            scaling=head_dim**-0.5,
            num_kv_heads=num_kv_heads,
            layer_id=0,
        )

        padding_size = 4096
        cache_loc_list = []

        aligned_seq_lens = (
            (forward_batch.seq_lens + page_size - 1) // page_size
        ) * page_size
        cache_start_loc = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(aligned_seq_lens)]
        )
        for i in range(forward_batch.batch_size):
            start = cache_start_loc[i]
            end = start + forward_batch.seq_lens[i]
            cache_loc = forward_batch.cache_loc[start:end]
            page_indices_for_seq = cache_loc // page_size
            page_indices_unique = unique_in_original_order(page_indices_for_seq)
            padded_page_indices = jnp.pad(
                jnp.array(page_indices_unique, dtype=jnp.int32),
                (0, padding_size - len(page_indices_unique)),
                constant_values=0,
            )
            cache_loc_list.append(padded_page_indices)
        page_table = jnp.stack(cache_loc_list)

        expected = ref_ragged_paged_attention(
            q.reshape(q.shape[0], num_heads, head_dim),
            k.reshape(k.shape[0] // page_size, page_size, num_kv_heads, head_dim),
            v.reshape(v.shape[0] // page_size, page_size, num_kv_heads, head_dim),
            forward_batch.seq_lens,
            page_table,
            forward_batch.attn_backend.forward_metadata.cu_q_lens,
            forward_batch.attn_backend.forward_metadata.num_seqs,
            sm_scale=head_dim**-0.5,
        )
        jax.block_until_ready(expected)

        @jax.jit
        def jit_attn(q, k, v, forward_batch):
            out = attn(q, k, v, forward_batch)
            return out

        # run
        jax_output, _, _ = jit_attn(q_shard, extend_k, extend_v, forward_batch)
        jax.block_until_ready(jax_output)

        rtol = 2e-2  # Relative tolerance
        atol = 1e-2  # Absolute tolerance
        jax_flat = np.asarray(jax_output)
        expected_flat = np.asarray(expected.reshape(expected.shape[0], -1))
        diff = np.abs(jax_flat - expected_flat)
        max_diff = np.max(diff)

        print(f"=== Detailed Analysis ===")
        print(f"JAX output shape: {jax_flat.shape}")
        print(f"Expected shape: {expected_flat.shape}")
        print(f"Max difference: {max_diff}")

        # Analyze by token dimension (rows) - show only first 5 tokens
        print(f"\n=== Token-wise Analysis (first 20 tokens) ===")
        num_tokens = jax_flat.shape[0]
        for i in range(min(num_tokens, 20)):
            jax_row = np.asarray(jax_flat[i])
            expected_row = np.asarray(expected_flat[i])
            row_diff = np.abs(jax_row - expected_row)
            jax_mean = np.mean(jax_row)
            expected_mean = np.mean(expected_row)
            jax_std = np.std(jax_row)
            expected_std = np.std(expected_row)

            print(
                f"Token {i}: max_diff={float(np.max(row_diff)):.6f}, jax_mean={float(jax_mean):.6f}, expected_mean={float(expected_mean):.6f}, jax_std={float(jax_std):.6f}, expected_std={float(expected_std):.6f}"
            )
            print()

        # Overall statistics
        print(f"=== Overall Statistics ===")
        print(
            f"JAX output:      mean={float(np.mean(jax_flat)):.6f}, std={float(np.std(jax_flat)):.6f}"
        )
        print(
            f"Expected output: mean={float(np.mean(expected_flat)):.6f}, std={float(np.std(expected_flat)):.6f}"
        )
        print(
            f"Absolute diff:   mean={float(np.mean(diff)):.6f}, std={float(np.std(diff)):.6f}, max={float(np.max(diff)):.6f}"
        )

        # Check how many tokens have large differences
        large_diff_tokens = int(
            np.sum(np.max(diff.reshape(num_tokens, -1), axis=1) > 0.1)
        )
        print(f"Tokens with max diff > 0.1: {large_diff_tokens}/{num_tokens}")

        are_close = np.allclose(
            jax_flat,
            expected_flat,
            rtol=rtol,
            atol=atol,
        )
        self.assertTrue(
            are_close,
            f"JAX output and expected output are not close, max diff: {max_diff}",
        )

    def run_mixed_test(self, mode, mixed_lens, test_req_idx, mode_args):
        # Create mock forward_batch
        num_heads, head_dim, num_kv_heads, page_size, dtype = mode_args

        if dtype == jnp.bfloat16:
            is_bf16 = True
        else:
            is_bf16 = False

        forward_batch, q, k, v = create_test_data(
            mode,
            mixed_lens,
            num_heads,
            head_dim,
            num_kv_heads,
            page_size,
            model_config={
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "num_hidden_layers": 1,
                "bf16": is_bf16,
            },
            dtype=dtype,
        )
        aligned_seq_lens = (
            (forward_batch.seq_lens + page_size - 1) // page_size
        ) * page_size
        cache_start_loc = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(aligned_seq_lens)]
        )
        cu_q_lens = jnp.concatenate(
            [
                jnp.array([0], dtype=jnp.int32),
                jnp.cumsum(forward_batch.extend_seq_lens),
            ]
        )
        forward_batch.traced_cache_indices = forward_batch.cache_loc[
            cache_start_loc[test_req_idx] : cache_start_loc[test_req_idx]
            + forward_batch.seq_lens[test_req_idx]
        ]

        # Debug cache mapping
        print(f"=== Cache Mapping Debug ===")
        print(f"lens: {mixed_lens}")
        print(f"test_req_idx: {test_req_idx}")
        print(f"seq_lens: {forward_batch.seq_lens}")
        # print(f"cu_q_lens: {forward_batch.attn_backend.forward_metadata.cu_q_lens}")
        # print(f"cu_kv_lens: {forward_batch.attn_backend.forward_metadata.cu_kv_lens}")
        print(f"cache_loc: {forward_batch.cache_loc}")
        print(f"out_cache_loc: {forward_batch.out_cache_loc}")
        print()

        print(f"=== Attention Input ===")
        print(f"q min: {jnp.min(q)}")
        print(f"q max: {jnp.max(q)}")
        print(f"q mean: {jnp.mean(q)}")
        print(f"q std: {jnp.std(q)}")
        print(f"q shape: {q.shape}")

        # Create test data
        shading = jax.sharding.NamedSharding(mesh, P(None, "tensor"))
        q_shard = jax.device_put(q.copy(), shading).reshape(q.shape[0], -1)
        k_cache_shard = jax.device_put(k.copy(), shading)
        v_cache_shard = jax.device_put(v.copy(), shading)

        print(f"=== Cache Index Debug ===")
        print(
            f"traced_cache_indices length: {len(forward_batch.traced_cache_indices) if forward_batch.traced_cache_indices is not None else 'None'}"
        )
        print(
            f"traced_cache_indices first 10: {forward_batch.traced_cache_indices[:10] if forward_batch.traced_cache_indices is not None else 'None'}"
        )
        print(f"cache_loc first 10: {forward_batch.cache_loc[:10]}")
        print(
            f"seq_lens for test_req_idx {test_req_idx}: {forward_batch.seq_lens[test_req_idx]}"
        )

        # 检查数据源的具体值
        if forward_batch.traced_cache_indices is not None:
            k_buffer, v_buffer = k_cache_shard, v_cache_shard
            flash_k_sample = k_buffer[forward_batch.traced_cache_indices[:5]].flatten()
            flash_v_sample = v_buffer[forward_batch.traced_cache_indices[:5]].flatten()
            print(
                f"=== FlashAttention K sample (first 20 values): {flash_k_sample[:20]}"
            )
            print(
                f"=== FlashAttention V sample (first 20 values): {flash_v_sample[:20]}"
            )

        # 对比 reference 实现中的数据
        ref_indices = (
            forward_batch.traced_cache_indices[:5]
            if forward_batch.traced_cache_indices is not None
            else None
        )
        if ref_indices is not None:
            # 模拟 reference 实现的数据访问方式
            page_indices = ref_indices // 1  # page_size = 1 in this test
            ref_k = k_cache_shard.reshape(-1, num_kv_heads, head_dim)[
                ref_indices
            ].flatten()
            ref_v = v_cache_shard.reshape(-1, num_kv_heads, head_dim)[
                ref_indices
            ].flatten()
            print(f"=== Reference K sample (first 20 values): {ref_k[:20]}")
            print(f"=== Reference V sample (first 20 values): {ref_v[:20]}")
            print(
                f"=== Sample difference K: {jnp.max(jnp.abs(flash_k_sample[:20] - ref_k[:20]))}"
            )
            print(
                f"=== Sample difference V: {jnp.max(jnp.abs(flash_v_sample[:20] - ref_v[:20]))}"
            )

            # 测试完整数据的统计差异
            flash_k_full = k_buffer[forward_batch.traced_cache_indices].flatten()
            flash_v_full = v_buffer[forward_batch.traced_cache_indices].flatten()
            ref_k_full = k_cache_shard.reshape(-1, num_kv_heads, head_dim)[
                forward_batch.traced_cache_indices
            ].flatten()
            ref_v_full = v_cache_shard.reshape(-1, num_kv_heads, head_dim)[
                forward_batch.traced_cache_indices
            ].flatten()

            print(f"=== Full Data Mean Comparison ===")
            print(f"FlashAttention K mean: {jnp.mean(flash_k_full)}")
            print(f"Reference K mean: {jnp.mean(ref_k_full)}")
            print(
                f"K mean difference: {jnp.abs(jnp.mean(flash_k_full) - jnp.mean(ref_k_full))}"
            )
            print(f"FlashAttention V mean: {jnp.mean(flash_v_full)}")
            print(f"Reference V mean: {jnp.mean(ref_v_full)}")
            print(
                f"V mean difference: {jnp.abs(jnp.mean(flash_v_full) - jnp.mean(ref_v_full))}"
            )

            # 检查数据是否逐元素相同
            print(f"=== Element-wise Comparison ===")
            k_diff = jnp.max(jnp.abs(flash_k_full - ref_k_full))
            v_diff = jnp.max(jnp.abs(flash_v_full - ref_v_full))
            print(f"Max K difference: {k_diff}")
            print(f"Max V difference: {v_diff}")
            print(
                f"Are K arrays identical: {jnp.allclose(flash_k_full, ref_k_full, rtol=0, atol=0)}"
            )
            print(
                f"Are V arrays identical: {jnp.allclose(flash_v_full, ref_v_full, rtol=0, atol=0)}"
            )

        # write prefix tokens
        extend_k, extend_v = write_prefix_tokens_for_kv(
            forward_batch, mixed_lens, k_cache_shard, v_cache_shard
        )
        extend_k = extend_k.reshape(extend_k.shape[0], -1)
        extend_v = extend_v.reshape(extend_v.shape[0], -1)

        # JAX attention
        attn = RadixAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            scaling=head_dim**-0.5,
            num_kv_heads=num_kv_heads,
            layer_id=0,
        )

        @jax.jit
        def jit_attn(q, k, v, forward_batch):
            out = attn(q, k, v, forward_batch, kv_partition_axis="None")
            return out

        # run
        jax_output, _, _ = jit_attn(q_shard, extend_k, extend_v, forward_batch)
        jax.block_until_ready(jax_output)

        expected = ref_ragged_paged_attention(
            q.reshape(-1, num_heads, head_dim),
            k.reshape(k.shape[0] // page_size, page_size, num_kv_heads, head_dim),
            v.reshape(v.shape[0] // page_size, page_size, num_kv_heads, head_dim),
            jnp.array([mixed_lens[test_req_idx][1]]),
            [forward_batch.traced_cache_indices],
            jnp.array([cu_q_lens[test_req_idx], cu_q_lens[test_req_idx + 1]]),
            jnp.array([1]),
            sm_scale=head_dim**-0.5,
        )
        jax.block_until_ready(expected)

        print(
            f"{cu_q_lens[test_req_idx]=} \n{cu_q_lens[test_req_idx+1]=} \n{cu_q_lens=} \n{q.shape=} \n{k.shape=} \n{v.shape=} \n{forward_batch.traced_cache_indices=}"
        )

        rtol = 5e-3  # Tighter relative tolerance for better accuracy
        atol = 5e-3  # Tighter absolute tolerance
        jax_flat = np.asarray(
            jax_output[cu_q_lens[test_req_idx] : cu_q_lens[test_req_idx + 1]]
        )
        expected_flat = np.asarray(expected.reshape(expected.shape[0], -1))
        diff = np.abs(jax_flat - expected_flat)
        max_diff = np.max(diff)

        print(f"{diff=}")

        print(f"=== Detailed Analysis ===")
        print(f"JAX output shape: {jax_flat.shape}")
        print(f"Expected shape: {expected_flat.shape}")
        print(f"Max difference: {max_diff}")

        # Analyze by token dimension (rows) - show only first 5 tokens
        print(f"\n=== Token-wise Analysis (first 20 tokens) ===")
        num_tokens = jax_flat.shape[0]
        for i in range(min(num_tokens, 20)):
            jax_row = np.asarray(jax_flat[i])
            expected_row = np.asarray(expected_flat[i])
            row_diff = np.abs(jax_row - expected_row)
            jax_min = np.min(jax_row)
            expected_min = np.min(expected_row)
            jax_max = np.max(jax_row)
            expected_max = np.max(expected_row)
            jax_mean = np.mean(jax_row)
            expected_mean = np.mean(expected_row)
            jax_std = np.std(jax_row)
            expected_std = np.std(expected_row)

            print(
                f"Token {i}: max_diff={float(np.max(row_diff)):.6f}, jax_min={float(jax_min):.6f}, expected_min={float(expected_min):.6f}, jax_max={float(jax_max):.6f}, expected_max={float(expected_max):.6f}, jax_mean={float(jax_mean):.6f}, expected_mean={float(expected_mean):.6f}, jax_std={float(jax_std):.6f}, expected_std={float(expected_std):.6f}"
            )
            print()

        # Overall statistics
        print(f"=== Overall Statistics ===")
        print(
            f"JAX output:      min={float(np.min(jax_flat)):.6f} max={float(np.max(jax_flat)):.6f} mean={float(np.mean(jax_flat)):.6f}, std={float(np.std(jax_flat)):.6f}"
        )
        print(
            f"Expected output: min={float(np.min(expected_flat)):.6f} max={float(np.max(expected_flat)):.6f} mean={float(np.mean(expected_flat)):.6f}, std={float(np.std(expected_flat)):.6f}"
        )
        print(
            f"Absolute diff:   min={float(np.min(diff)):.6f} max={float(np.max(diff)):.6f} mean={float(np.mean(diff)):.6f}, std={float(np.std(diff)):.6f}, "
        )

        # Check how many tokens have large differences
        large_diff_tokens = int(
            np.sum(np.max(diff.reshape(num_tokens, -1), axis=1) > 0.1)
        )
        print(f"Tokens with max diff > 0.1: {large_diff_tokens}/{num_tokens}")

        are_close = np.allclose(
            jax_flat,
            expected_flat,
            rtol=rtol,
            atol=atol,
        )
        self.assertTrue(
            are_close,
            f"JAX output and expected output are not close, max diff: {max_diff}",
        )

    def test_mha_prefill_accuracy_page_size_1(self):
        """Test JAX attention accuracy against PyTorch reference"""
        # Parameters
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128
        lens = [
            (1, 128),
            (125, 125),
            (1024, 1024),
            (123, 522),
            (1, 511),
            (512, 1024),
        ]

        self.run_test(
            "prefill", lens, (num_heads, head_dim, num_kv_heads, 1, jnp.bfloat16)
        )

    def test_mha_decode_accuracy_page_size_1(self):
        """Test JAX attention accuracy against native fa"""
        # Parameters
        num_heads = 32
        num_kv_heads = 32
        head_dim = 128
        lens = [
            (1, 119),
            (1, 127),
            (1, 128),
            (1, 129),
            (1, 133),
            (1, 1001),
            (1, 1023),
            (1, 1024),
            (1, 1025),
        ]

        self.run_test(
            "decode", lens, (num_heads, head_dim, num_kv_heads, 1, jnp.bfloat16)
        )

    def test_mha_prefill_accuracy_page_size_8(self):
        """
        Test JAX attention accuracy against PyTorch reference
        This test case will failed when batch size > 2, the second batch tokens will has wrong value, the first and third batch tokens are correct.
        """
        # Parameters
        num_heads = 32
        num_kv_heads = 32
        head_dim = 128
        lens = [
            (5, 17),
            (5, 33),
            (5, 5),
        ]
        self.run_test(
            "prefill", lens, (num_heads, head_dim, num_kv_heads, 8, jnp.bfloat16)
        )

    def test_mha_decode_accuracy_page_size_8(self):
        """Test JAX attention accuracy against native fa"""
        # Parameters
        num_heads = 32
        num_kv_heads = 32
        head_dim = 128
        lens = [
            (1, 17),
            (1, 6),
            (1, 5),
        ]
        self.run_test(
            "decode", lens, (num_heads, head_dim, num_kv_heads, 8, jnp.bfloat16)
        )

    def test_mha_prefill_accuracy_page_size_64(self):
        """Test JAX attention accuracy against PyTorch reference"""
        # Parameters
        num_heads = 32
        num_kv_heads = 32
        head_dim = 128
        lens = [
            (1, 128),
            (3, 20),
            (64, 64),
            (20, 20),
            (125, 125),
            (1024, 1024),
            (123, 522),
            (1, 511),
        ]
        self.run_test(
            "prefill", lens, (num_heads, head_dim, num_kv_heads, 64, jnp.bfloat16)
        )

    def test_mha_decode_accuracy_page_size_64(self):
        """Test JAX attention accuracy against native fa"""
        # Parameters
        num_heads = 32
        num_kv_heads = 32
        head_dim = 128
        lens = [
            (1, 20),
            (1, 64),
            (1, 30),
            (1, 129),
            (1, 133),
            (1, 256),
            (1, 1001),
            (1, 1024),
            (1, 1025),
        ]
        self.run_test(
            "decode", lens, (num_heads, head_dim, num_kv_heads, 64, jnp.bfloat16)
        )

    def test_gqa_prefill_accuracy_page_size_64(self):
        """Test JAX attention accuracy against PyTorch reference"""
        # Parameters
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128
        lens = [
            (1, 128),
            (3, 20),
            (64, 64),
            (20, 20),
            (125, 125),
            (1024, 1024),
            (123, 522),
            (1, 511),
        ]
        self.run_test(
            "prefill", lens, (num_heads, head_dim, num_kv_heads, 64, jnp.bfloat16)
        )

    def test_gqa_decode_accuracy_page_size_64(self):
        """Test JAX attention accuracy against native fa"""
        # Parameters
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128
        lens = [
            (1, 119),
            (1, 127),
            (1, 128),
            (1, 129),
            (1, 133),
            (1, 1001),
            (1, 1023),
            (1, 1024),
            (1, 1025),
        ]

        self.run_test(
            "decode", lens, (num_heads, head_dim, num_kv_heads, 64, jnp.bfloat16)
        )

    def test_mixed_chunked_prefill_accuracy(self):
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128
        lens = [
            (511, 1040),
            (1, 509),
            # (20, 20),
            # (125, 125),
            # (1024, 1024),
            # (123, 522),
        ]

        self.run_mixed_test(
            "prefill", lens, 1, (num_heads, head_dim, num_kv_heads, 1, jnp.bfloat16)
        )

    def test_mixed_vs_pure_batch_precision_differences(self):
        """
        Test to verify JAX compilation path precision differences between mixed batches and pure batches.
        This test aims to reproduce the precision issue where mixed prefill scenarios show different
        numerical precision compared to pure prefill/decode batches.
        """
        print("\n=== Testing Mixed vs Pure Batch Precision Differences ===")

        # Fixed configuration to ensure reproducibility
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128
        page_size = 1
        dtype = jnp.bfloat16
        fixed_seed = 42

        # Test scenarios: mixed batch vs equivalent pure batches
        # Scenario 1: Mixed batch with prefill + decode
        mixed_lens = [
            (81, 512),  # target_prefill - large prefill
            (427, 936),  # req1_prefill - large prefill
            (1, 100),  # req1_decode - decode
            (1, 200),  # req2_decode - decode
            (1, 150),  # req3_decode - decode
            (1, 300),  # req4_decode - decode
        ]

        # Scenario 2: Pure prefill batch (equivalent to prefill requests from mixed)
        pure_prefill_lens = [
            (81, 512),  # same as target_prefill
            (427, 936),  # same as req1_prefill
        ]

        # Scenario 3: Pure decode batch (equivalent to decode requests from mixed)
        pure_decode_lens = [
            (1, 100),  # same as req1_decode
            (1, 200),  # same as req2_decode
            (1, 150),  # same as req3_decode
            (1, 300),  # same as req4_decode
        ]

        def create_deterministic_test_data(
            lens_config, mode_suffix="", custom_seed=None
        ):
            """Create test data with fixed seed for reproducibility"""
            # Use custom seed if provided, otherwise use fixed_seed
            seed_to_use = custom_seed if custom_seed is not None else fixed_seed

            forward_batch, q, k, v = create_test_data(
                "prefill",
                lens_config,
                num_heads,
                head_dim,
                num_kv_heads,
                page_size,
                model_config={
                    "num_kv_heads": num_kv_heads,
                    "head_dim": head_dim,
                    "num_hidden_layers": 1,
                    "bf16": True,
                },
                dtype=dtype,
            )

            print(f"\n=== {mode_suffix} Configuration ===")
            print(f"forward_mode: {forward_batch.forward_mode}")
            print(f"extend_seq_lens: {forward_batch.extend_seq_lens}")
            print(f"seq_lens: {forward_batch.seq_lens}")
            print(f"batch_size: {forward_batch.batch_size}")

            return forward_batch, q, k, v

        def run_attention_with_tracing(forward_batch, q, k, v, test_indices, mode_name):
            """Run attention with numerical tracing for specific request indices"""
            # Set up tracing for specific requests
            if test_indices:
                traced_cache_indices_list = []
                for idx in test_indices:
                    aligned_seq_lens = (
                        (forward_batch.seq_lens + page_size - 1) // page_size
                    ) * page_size
                    cache_start_loc = jnp.concatenate(
                        [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(aligned_seq_lens)]
                    )
                    cache_indices = forward_batch.cache_loc[
                        cache_start_loc[idx] : cache_start_loc[idx]
                        + forward_batch.seq_lens[idx]
                    ]
                    traced_cache_indices_list.extend(cache_indices.tolist())
                forward_batch.traced_cache_indices = jnp.array(
                    traced_cache_indices_list, dtype=jnp.int32
                )
                forward_batch.traced_req_indices = jnp.array(
                    test_indices, dtype=jnp.int32
                )

            # Prepare sharded data
            sharding = jax.sharding.NamedSharding(mesh, P(None, "tensor"))
            q_shard = jax.device_put(q.copy(), sharding).reshape(q.shape[0], -1)
            k_cache_shard = jax.device_put(k.copy(), sharding)
            v_cache_shard = jax.device_put(v.copy(), sharding)

            # Write prefix tokens
            extend_k, extend_v = write_prefix_tokens_for_kv(
                forward_batch,
                [
                    (q_len, kv_len)
                    for q_len, kv_len in zip(
                        jax.device_get(forward_batch.extend_seq_lens).tolist(),
                        jax.device_get(forward_batch.seq_lens).tolist(),
                    )
                ],
                k_cache_shard,
                v_cache_shard,
            )
            extend_k = extend_k.reshape(extend_k.shape[0], -1)
            extend_v = extend_v.reshape(extend_v.shape[0], -1)

            # Create attention layer
            attn = RadixAttention(
                num_heads=num_heads,
                head_dim=head_dim,
                scaling=head_dim**-0.5,
                num_kv_heads=num_kv_heads,
                layer_id=0,
            )

            # Initialize KV cache
            forward_batch.token_to_kv_pool = MHATokenToKVPool(
                size=10000,
                page_size=page_size,
                dtype=dtype,
                head_num=num_kv_heads,
                head_dim=head_dim,
                layer_num=1,
                mesh=mesh,
            )

            # Initialize attention backend forward metadata
            metadata = forward_batch.attn_backend.init_forward_metadata(forward_batch)
            forward_batch.attn_backend.forward_metadata = metadata

            # Run forward pass
            print(f"\n=== Running {mode_name} Forward Pass ===")
            output, _, _ = attn(q_shard, extend_k, extend_v, forward_batch)

            # Extract traced outputs for analysis
            traced_output = None
            if test_indices and forward_batch.traced_cache_indices is not None:
                # Calculate output indices corresponding to traced requests
                cu_q_lens = jnp.concatenate(
                    [
                        jnp.array([0], dtype=jnp.int32),
                        jnp.cumsum(forward_batch.extend_seq_lens),
                    ]
                )
                output_indices = []
                for idx in test_indices:
                    start_idx = cu_q_lens[idx]
                    end_idx = cu_q_lens[idx + 1]
                    output_indices.extend(range(start_idx, end_idx))
                traced_output = output[jnp.array(output_indices)].flatten()

                print(f"Traced output stats for {mode_name}:")
                print(f"  Mean: {float(jnp.mean(traced_output)):.10f}")
                print(f"  Std:  {float(jnp.std(traced_output)):.10f}")
                print(f"  Min:  {float(jnp.min(traced_output)):.10f}")
                print(f"  Max:  {float(jnp.max(traced_output)):.10f}")
                print(f"  First 10 values: {traced_output[:10]}")

            return output, traced_output

        # Test 1: Mixed batch
        print("\n" + "=" * 50)
        print("TEST 1: Mixed Batch (Prefill + Decode)")
        print("=" * 50)
        mixed_batch, mixed_q, mixed_k, mixed_v = create_deterministic_test_data(
            mixed_lens, "Mixed Batch"
        )
        mixed_output, mixed_traced = run_attention_with_tracing(
            mixed_batch, mixed_q, mixed_k, mixed_v, [0, 1], "Mixed Batch"
        )

        # Test 1b: Mixed batch - extract decode parts
        print("\n" + "=" * 50)
        print("TEST 1b: Mixed Batch (Prefill + Decode) - Extract Decode Parts")
        print("=" * 50)
        mixed_batch_decode, mixed_q_decode, mixed_k_decode, mixed_v_decode = (
            create_deterministic_test_data(mixed_lens, "Mixed Batch (for decode)")
        )
        mixed_output_decode, mixed_decode_traced = run_attention_with_tracing(
            mixed_batch_decode,
            mixed_q_decode,
            mixed_k_decode,
            mixed_v_decode,
            [2, 3, 4, 5],
            "Mixed Batch Decode Parts",
        )

        # Test 2: Pure prefill batch (reusing data from mixed batch)
        print("\n" + "=" * 50)
        print(
            "TEST 2: Pure Prefill Batch (using same data as mixed batch prefill part)"
        )
        print("=" * 50)
        # Extract prefill requests from mixed batch data to ensure identical input
        prefill_q_len = sum([q_len for q_len, _ in pure_prefill_lens])
        prefill_kv_len = sum([kv_len for _, kv_len in pure_prefill_lens])
        pure_prefill_q = mixed_q[:prefill_q_len].copy()
        pure_prefill_k = mixed_k[:prefill_kv_len].copy()
        pure_prefill_v = mixed_v[:prefill_kv_len].copy()

        pure_prefill_batch, _, _, _ = create_deterministic_test_data(
            pure_prefill_lens, "Pure Prefill Batch"
        )
        pure_prefill_output, pure_prefill_traced = run_attention_with_tracing(
            pure_prefill_batch,
            pure_prefill_q,
            pure_prefill_k,
            pure_prefill_v,
            [0, 1],
            "Pure Prefill Batch",
        )

        # Test 3: Pure decode batch (reusing data from mixed batch)
        print("\n" + "=" * 50)
        print("TEST 3: Pure Decode Batch (using same data as mixed batch decode part)")
        print("=" * 50)
        # Extract decode requests from mixed batch data to ensure identical input
        # Decode requests start after prefill requests in the mixed batch
        mixed_prefill_q_len = sum([q_len for q_len, _ in pure_prefill_lens])
        mixed_prefill_kv_len = sum([kv_len for _, kv_len in pure_prefill_lens])
        decode_q_len = sum([q_len for q_len, _ in pure_decode_lens])
        decode_kv_len = sum([kv_len for _, kv_len in pure_decode_lens])

        # Extract decode part from mixed batch (assuming decode requests follow prefill requests)
        pure_decode_q = mixed_q_decode[
            mixed_prefill_q_len : mixed_prefill_q_len + decode_q_len
        ].copy()
        pure_decode_k = mixed_k_decode[
            mixed_prefill_kv_len : mixed_prefill_kv_len + decode_kv_len
        ].copy()
        pure_decode_v = mixed_v_decode[
            mixed_prefill_kv_len : mixed_prefill_kv_len + decode_kv_len
        ].copy()

        pure_decode_batch, _, _, _ = create_deterministic_test_data(
            pure_decode_lens, "Pure Decode Batch"
        )
        pure_decode_output, pure_decode_traced = run_attention_with_tracing(
            pure_decode_batch,
            pure_decode_q,
            pure_decode_k,
            pure_decode_v,
            [0, 1, 2, 3],
            "Pure Decode Batch",
        )

        # Numerical comparison - Prefill: Mixed vs Pure
        print("\n" + "=" * 50)
        print("PRECISION COMPARISON RESULTS - PREFILL")
        print("=" * 50)

        if mixed_traced is not None and pure_prefill_traced is not None:
            # Compare the prefill outputs: mixed batch prefill vs pure prefill batch
            diff = jnp.abs(mixed_traced - pure_prefill_traced)
            max_diff = jnp.max(diff)
            mean_diff = jnp.mean(diff)

            print(f"\nPrefill Precision Differences (Mixed vs Pure Prefill):")
            print(f"  Max absolute difference:  {float(max_diff):.2e}")
            print(f"  Mean absolute difference: {float(mean_diff):.2e}")
            print(
                f"  Relative error:          {float(mean_diff / jnp.mean(jnp.abs(mixed_traced))):.2e}"
            )

            # Check if differences exceed expected precision thresholds
            # For bfloat16, we expect differences to be minimal for identical computations
            bfloat16_eps = 1e-3  # Approximate precision for bfloat16
            significant_diff_threshold = 10 * bfloat16_eps

            if max_diff > significant_diff_threshold:
                print(f"\n⚠️  SIGNIFICANT PRECISION DIFFERENCE DETECTED IN PREFILL!")
                print(
                    f"   Max difference ({float(max_diff):.2e}) exceeds threshold ({significant_diff_threshold:.2e})"
                )
                print(
                    f"   This indicates different JAX compilation paths for mixed vs pure prefill batches"
                )

                # Show detailed differences
                large_diff_indices = jnp.where(diff > significant_diff_threshold)[0]
                if len(large_diff_indices) > 0:
                    print(
                        f"   Number of elements with large differences: {len(large_diff_indices)}"
                    )
                    print(
                        f"   Indices with large differences (first 10): {large_diff_indices[:10]}"
                    )

            else:
                print(f"\n✅ Prefill precision differences within expected range")
                print(
                    f"   Max difference ({float(max_diff):.2e}) is below threshold ({significant_diff_threshold:.2e})"
                )

        # Numerical comparison - Decode: Mixed vs Pure
        print("\n" + "=" * 50)
        print("PRECISION COMPARISON RESULTS - DECODE")
        print("=" * 50)

        if mixed_decode_traced is not None and pure_decode_traced is not None:
            # Compare the decode outputs: mixed batch decode vs pure decode batch
            diff_decode = jnp.abs(mixed_decode_traced - pure_decode_traced)
            max_diff_decode = jnp.max(diff_decode)
            mean_diff_decode = jnp.mean(diff_decode)

            print(f"\nDecode Precision Differences (Mixed vs Pure Decode):")
            print(f"  Max absolute difference:  {float(max_diff_decode):.2e}")
            print(f"  Mean absolute difference: {float(mean_diff_decode):.2e}")
            print(
                f"  Relative error:          {float(mean_diff_decode / jnp.mean(jnp.abs(mixed_decode_traced))):.2e}"
            )

            if max_diff_decode > significant_diff_threshold:
                print(f"\n⚠️  SIGNIFICANT PRECISION DIFFERENCE DETECTED IN DECODE!")
                print(
                    f"   Max difference ({float(max_diff_decode):.2e}) exceeds threshold ({significant_diff_threshold:.2e})"
                )
                print(
                    f"   This indicates different JAX compilation paths for mixed vs pure decode batches"
                )

                # Show detailed differences
                large_diff_indices_decode = jnp.where(
                    diff_decode > significant_diff_threshold
                )[0]
                if len(large_diff_indices_decode) > 0:
                    print(
                        f"   Number of elements with large differences: {len(large_diff_indices_decode)}"
                    )
                    print(
                        f"   Indices with large differences (first 10): {large_diff_indices_decode[:10]}"
                    )
            else:
                print(f"\n✅ Decode precision differences within expected range")
                print(
                    f"   Max difference ({float(max_diff_decode):.2e}) is below threshold ({significant_diff_threshold:.2e})"
                )

        # Additional test: Multiple runs with same data to check compilation consistency
        print(f"\n" + "=" * 50)
        print("COMPILATION CONSISTENCY TEST")
        print("=" * 50)

        # Run mixed batch multiple times to check for compilation-dependent variations
        print("Running mixed batch 3 times to check compilation consistency...")
        mixed_outputs = []
        for run_id in range(3):
            print(f"\nRun {run_id + 1}:")
            # Create fresh batch data each time to trigger recompilation
            batch_copy, q_copy, k_copy, v_copy = create_deterministic_test_data(
                mixed_lens, f"Mixed Run {run_id + 1}"
            )
            output, traced = run_attention_with_tracing(
                batch_copy, q_copy, k_copy, v_copy, [0, 1], f"Mixed Run {run_id + 1}"
            )
            if traced is not None:
                mixed_outputs.append(traced)
                print(f"  Run {run_id + 1} mean: {float(jnp.mean(traced)):.10f}")

        # Check consistency across runs
        if len(mixed_outputs) >= 2:
            run_diffs = []
            for i in range(1, len(mixed_outputs)):
                diff = jnp.max(jnp.abs(mixed_outputs[0] - mixed_outputs[i]))
                run_diffs.append(diff)
                print(f"Max difference between run 1 and run {i+1}: {float(diff):.2e}")

            max_run_diff = max(run_diffs) if run_diffs else 0
            if max_run_diff > bfloat16_eps:
                print(f"\n⚠️  COMPILATION INCONSISTENCY DETECTED!")
                print(
                    f"   Multiple runs of identical data show differences up to {float(max_run_diff):.2e}"
                )
            else:
                print(f"\n✅ Compilation consistency verified")
                print(
                    f"   Multiple runs show differences within precision limit ({float(max_run_diff):.2e})"
                )

        print(f"\n" + "=" * 50)
        print("TEST COMPLETE")
        print("=" * 50)


if __name__ == "__main__":
    unittest.main()
