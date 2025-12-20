import unittest

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.paged_attention.paged_attention import (
    paged_attention,
    paged_attention_reference,
)
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.test.test_utils import CustomTestCase

mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
jax.sharding.set_mesh(mesh)


class TestPagedAttention(CustomTestCase):
    def setUp(self):
        self.devices = jax.devices()
        self.num_devices = len(self.devices)
        self.mesh = mesh

        self.head_dim = 128
        self.dtype = jnp.bfloat16

    def _prepare_paged_data(self, batch_size, num_heads, num_kv_heads, seq_lens, page_size):
        """
        Helper to construct data in PagedAttention format.
        Logic:
        1. Reserve Physical Page 0 as a 'safe zero page'.
        2. Generate contiguous Q, K, V.
        3. Map sequence blocks to physical pages [1, total_pages].
        4. Pad the block_tables with 0 (pointing to the safe page).
        """
        key = jax.random.PRNGKey(42)

        # 1. Generate Query [Batch, Num_Heads, Head_Dim]
        q = jax.random.normal(key, (batch_size, num_heads, self.head_dim), dtype=self.dtype)

        # 2. Generate contiguous Reference K, V [Batch, Max_Len, Num_KV_Heads, Head_Dim]
        max_seq_len = max(seq_lens)
        k_ref = jax.random.normal(
            key, (batch_size, max_seq_len, num_kv_heads, self.head_dim), dtype=self.dtype
        )
        v_ref = jax.random.normal(
            key, (batch_size, max_seq_len, num_kv_heads, self.head_dim), dtype=self.dtype
        )

        # 3. Construct Page Table and Paged K/V Buffers
        actual_pages_needed_per_seq = [
            (seq_len + page_size - 1) // page_size for seq_len in seq_lens
        ]
        max_actual_pages = max(actual_pages_needed_per_seq)

        # Pad width of block_tables to next power of two
        block_table_width = 1 << (max_actual_pages - 1).bit_length() if max_actual_pages > 0 else 1

        # total_pages excluding the safety page 0
        total_data_pages = batch_size * max_actual_pages

        # Assign physical IDs starting from 1 to total_data_pages
        physical_ids_pool = np.arange(1, total_data_pages + 1)
        np.random.shuffle(physical_ids_pool)

        # block_tables initialized with 0 (Safety Page)
        block_tables = np.zeros((batch_size, block_table_width), dtype=np.int32)

        # Paged Buffer Shape: [num_kv_heads, total_num_pages, page_size, head_dim]
        # total_num_pages = total_data_pages + 1 (for page 0)
        total_buffer_pages = total_data_pages + 1
        paged_k = np.zeros(
            (num_kv_heads, total_buffer_pages, page_size, self.head_dim), dtype=np.float32
        )
        paged_v = np.zeros(
            (num_kv_heads, total_buffer_pages, page_size, self.head_dim), dtype=np.float32
        )

        pool_ptr = 0
        for b in range(batch_size):
            num_pages = actual_pages_needed_per_seq[b]
            for p_idx in range(num_pages):
                # Take a physical ID from the pool (starts from 1)
                phys_id = physical_ids_pool[pool_ptr]
                block_tables[b, p_idx] = phys_id
                pool_ptr += 1

                start_tok = p_idx * page_size
                valid_len = min(page_size, seq_lens[b] - start_tok)

                # Fill the paged buffer with actual data
                for h in range(num_kv_heads):
                    paged_k[h, phys_id, :valid_len, :] = k_ref[
                        b, start_tok : start_tok + valid_len, h, :
                    ]
                    paged_v[h, phys_id, :valid_len, :] = v_ref[
                        b, start_tok : start_tok + valid_len, h, :
                    ]

        return (
            q,
            jnp.array(paged_k, dtype=self.dtype),
            jnp.array(paged_v, dtype=self.dtype),
            jnp.array(block_tables),
            jnp.array(seq_lens, dtype=jnp.int32),
            k_ref,
            v_ref,
        )

    def run_test(
        self,
        batch_size,
        num_heads,
        num_kv_heads,
        seq_lens,
        page_size,
        use_sharding=False,
        soft_cap=None,
    ):
        # Prepare inputs with specific page_size and zero-padded block table
        q, k_pages, v_pages, block_tables, lengths, k_ref, v_ref = self._prepare_paged_data(
            batch_size, num_heads, num_kv_heads, seq_lens, page_size
        )

        # Compute Reference Output (Ground Truth)
        sm_scale = self.head_dim**-0.5
        expected_out = paged_attention_reference(
            q, k_ref, v_ref, lengths, sm_scale=sm_scale, attn_logits_soft_cap=soft_cap
        )

        # Compute Paged Attention Kernel Output
        current_mesh = self.mesh if use_sharding else None
        # Not exceeding 128 tokens per compute block
        pages_per_compute_block = max(1, 128 // page_size)
        k_splits = min(16, pages_per_compute_block, page_size) if use_sharding else 1

        actual_out = paged_attention(
            q,
            k_pages,
            v_pages,
            block_tables,
            lengths,
            sm_scale=sm_scale,
            attn_logits_soft_cap=soft_cap,
            mesh=current_mesh,
            k_splits=k_splits,
            pages_per_compute_block=pages_per_compute_block,
        )

        # Precision Validation
        rtol, atol = 2e-2, 1e-2

        np.testing.assert_allclose(
            np.array(actual_out),
            np.array(expected_out),
            rtol=rtol,
            atol=atol,
            err_msg=f"Mismatch found for B={batch_size}, PS={page_size}, H={num_heads}, KV_H={num_kv_heads}",
        )
        print(
            f"PASS: PageSize={page_size}, B={batch_size}, H={num_heads}, KV_H={num_kv_heads}, Sharding={use_sharding}, Cap={soft_cap}"
        )

    def test_mha_standard(self):
        """Test standard Multi-Head Attention (MHA)."""
        self.run_test(
            batch_size=4, num_heads=32, num_kv_heads=32, seq_lens=[128, 64, 512, 200], page_size=16
        )

    def test_various_page_sizes(self):
        """Iterate through page sizes to ensure correct safe-page and padding logic."""
        for ps in [1, 2, 4, 8, 16, 32]:
            with self.subTest(page_size=ps):
                self.run_test(
                    batch_size=2, num_heads=16, num_kv_heads=16, seq_lens=[111, 222], page_size=ps
                )

    def test_gqa_standard(self):
        """Test Grouped-Query Attention (GQA)."""
        self.run_test(
            batch_size=2, num_heads=32, num_kv_heads=8, seq_lens=[1024, 513], page_size=16
        )

    def test_soft_cap(self):
        """Test Logit Soft Capping."""
        self.run_test(
            batch_size=1, num_heads=8, num_kv_heads=8, seq_lens=[256], page_size=16, soft_cap=30.0
        )

    def test_spmd_sharding(self):
        """Test Distributed Inference using jax.shard_map."""
        if self.num_devices < 1:
            self.skipTest("At least 1 GPU/TPU device is required.")

        self.run_test(
            batch_size=4,
            num_heads=16,
            num_kv_heads=4,
            seq_lens=[
                128,
                256,
                2048,
                8192,
            ],
            page_size=16,
            use_sharding=True,
        )

    def test_irregular_lengths(self):
        """Test lengths that are not multiples of page_size."""
        self.run_test(
            batch_size=8,
            num_heads=4,
            num_kv_heads=4,
            seq_lens=[1, 2, 15, 16, 17, 31, 32, 33],
            page_size=8,
        )

    def test_long_sequences(self):
        self.run_test(batch_size=1, num_heads=8, num_kv_heads=8, seq_lens=[16384], page_size=16)


if __name__ == "__main__":
    unittest.main()
