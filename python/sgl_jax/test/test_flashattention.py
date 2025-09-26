import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.flash_attn_kernel.flash_attention import (
    ref_ragged_paged_attention,
)
from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.model_executor.model_runner import ModelRunner
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.test.test_utils import CustomTestCase

mesh = create_device_mesh(ici_parallelism=[1, -1, 1], dcn_parallelism=[1, 1, 1])
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
):
    batched_q_len = sum([q_len for q_len, _ in lens])
    batched_kv_len = sum([kv_len for _, kv_len in lens])

    # Calculate aligned batched_kv_len
    seq_lens = jnp.array([kv_len for _, kv_len in lens], dtype=jnp.int32)
    aligned_seq_lens = ((seq_lens + page_size - 1) // page_size) * page_size
    batched_aligned_kv_len = jnp.sum(aligned_seq_lens).item()

    key = jax.random.PRNGKey(42)
    q = jax.random.normal(key, (batched_q_len, num_heads, head_dim), dtype=jnp.bfloat16)

    # Create k,v with proper alignment gaps between sequences
    k = jnp.zeros((batched_aligned_kv_len, num_kv_heads, head_dim), dtype=jnp.bfloat16)
    v = jnp.zeros((batched_aligned_kv_len, num_kv_heads, head_dim), dtype=jnp.bfloat16)

    # Fill in the actual data for each sequence with proper alignment
    actual_pos = 0
    aligned_pos = 0
    for seq_len in [kv_len for _, kv_len in lens]:
        aligned_len = ((seq_len + page_size - 1) // page_size) * page_size

        # Generate data for this sequence
        seq_k = jax.random.normal(
            jax.random.split(key, len(lens) * 2)[actual_pos],
            (seq_len, num_kv_heads, head_dim),
            dtype=jnp.bfloat16,
        )
        seq_v = jax.random.normal(
            jax.random.split(key, len(lens) * 2)[actual_pos + len(lens)],
            (seq_len, num_kv_heads, head_dim),
            dtype=jnp.bfloat16,
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
    max_total_token_size=710016,
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
    q, k, v = create_qkv_cache(lens, num_heads, head_dim, num_kv_heads, page_size)

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
        num_heads,
        num_kv_heads,
        head_dim,
        page_size=page_size,
        mesh=mesh,
    )
    forward_mode = ForwardMode.EXTEND if mode == "prefill" else ForwardMode.DECODE

    mwb = ModelWorkerBatch(
        bid=1,
        forward_mode=forward_mode,
        input_ids=np.asarray(input_ids),
        real_input_ids_len=input_ids.shape[0],
        seq_lens=np.asarray(seq_lens),
        out_cache_loc=np.asarray(out_cache_loc),
        req_pool_indices=np.asarray(req_pool_indices),
        sampling_info=None,
        positions=np.asarray(positions),
        extend_start_loc=np.asarray(extend_start_loc),
        cache_loc=np.asarray(cache_loc),
        extend_seq_lens=np.asarray(extend_seq_lens),
        extend_prefix_lens=np.asarray(extend_prefix_lens),
        return_logprob=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        extend_logprob_start_lens=None,
        extend_input_logprob_token_ids=None,
        real_bs=seq_lens.shape[0],
    )

    fb = ForwardBatch(
        bid=1,
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
    fb.attn_backend.forward_metadata = attention_backend.get_forward_metadata(mwb)
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

        # Create test data
        shading = jax.sharding.NamedSharding(mesh, P(None, "tensor"))
        q_shard = jax.device_put(q.copy(), shading)
        k_cache_shard = jax.device_put(k.copy(), shading)
        v_cache_shard = jax.device_put(v.copy(), shading)

        # write prefix tokens
        extend_k, extend_v = write_prefix_tokens_for_kv(
            forward_batch, lens, k_cache_shard, v_cache_shard
        )

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
            # forward_batch.attn_backend.forward_metadata.cu_kv_lens,
            forward_batch.attn_backend.forward_metadata.num_seqs,
            sm_scale=head_dim**-0.5,
        )
        jax.block_until_ready(expected)

        @jax.jit
        def jit_attn(q, k, v, forward_batch):
            out = attn(q, k, v, forward_batch)
            return out

        # run
        jax_output, _ = jit_attn(q_shard, extend_k, extend_v, forward_batch)
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


if __name__ == "__main__":
    unittest.main()
