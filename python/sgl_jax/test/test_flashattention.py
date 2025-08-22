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
from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.test.test_utils import CustomTestCase

mesh = create_device_mesh(ici_parallelism=[1, -1, 1, 1], dcn_parallelism=[1, 1, 1, 1])
jax.sharding.set_mesh(mesh)


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

    # Create k,v with aligned length and pad with zeros
    k_unpadded = jax.random.normal(
        jax.random.split(key)[0],
        (batched_kv_len, num_kv_heads, head_dim),
        dtype=jnp.bfloat16,
    )
    v_unpadded = jax.random.normal(
        jax.random.split(key)[1],
        (batched_kv_len, num_kv_heads, head_dim),
        dtype=jnp.bfloat16,
    )

    # Pad k,v to aligned length
    padding_size = batched_aligned_kv_len - batched_kv_len
    k = jnp.pad(k_unpadded, ((0, padding_size), (0, 0), (0, 0)), constant_values=0)
    v = jnp.pad(v_unpadded, ((0, padding_size), (0, 0), (0, 0)), constant_values=0)

    return q, k, v


def write_prefix_tokens_for_kv(forward_batch, lens, k, v):
    cache_loc_idx = jnp.concatenate(
        [jnp.array([0], dtype=jnp.int32), jnp.cumsum(forward_batch.seq_lens)]
    )
    extend_k = []
    extend_v = []
    for i, (q_len, kv_len) in enumerate(lens):
        start = cache_loc_idx[i]
        prefix_end = start + (kv_len - q_len)
        extend_start = prefix_end
        extend_end = start + kv_len
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

    # cache loc - aligned to page_size
    cache_loc = jnp.arange(total_aligned_tokens, dtype=jnp.int32)
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
                "num_kv_heads": num_heads,
                "head_dim": head_dim,
                "num_hidden_layers": 1,
                "bf16": is_bf16,
            },
        )

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
        # Use aligned cache locations for start computation
        aligned_seq_lens = (
            forward_batch.attn_backend.forward_metadata.cu_kv_lens[1:]
            - forward_batch.attn_backend.forward_metadata.cu_kv_lens[:-1]
        )
        cache_start_loc = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(aligned_seq_lens)]
        )
        for i in range(forward_batch.batch_size):
            start = cache_start_loc[i]
            # Only use actual sequence length for the valid portion
            actual_seq_len = forward_batch.seq_lens[i]
            cache_loc = forward_batch.cache_loc[start : start + actual_seq_len]
            padded_size = padding_size - actual_seq_len
            padded_cache_loc = jnp.pad(cache_loc, (0, padded_size), constant_values=0)
            cache_loc_list.append(padded_cache_loc)
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
        are_close = np.allclose(
            np.asarray(jax_output),
            np.asarray(expected.reshape(expected.shape[0], -1)),
            rtol=rtol,
            atol=atol,
        )
        self.assertTrue(are_close, f"JAX output and expected output are not close")

    def test_mha_prefill_accuracy_page_size_1(self):
        """Test JAX attention accuracy against PyTorch reference"""
        # Parameters
        num_heads = 32
        num_kv_heads = 32
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

    def test_mha_prefill_accuracy_page_size_4(self):
        """Test JAX attention accuracy against PyTorch reference"""
        # Parameters
        num_heads = 32
        num_kv_heads = 32
        head_dim = 128
        lens = [
            (1, 128),
        ]
        self.run_test(
            "prefill", lens, (num_heads, head_dim, num_kv_heads, 4, jnp.bfloat16)
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


if __name__ == "__main__":
    unittest.main()
