import unittest

import jax.numpy as jnp
import numpy as np

from sgl_jax.test.flashattention_common import AttentionTestBase


class TestFlashAttentionMHA(AttentionTestBase):
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
            "prefill",
            lens,
            (num_heads, head_dim, num_kv_heads, 1, jnp.bfloat16),
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
            "decode",
            lens,
            (num_heads, head_dim, num_kv_heads, 1, jnp.bfloat16),
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
            "prefill",
            lens,
            (num_heads, head_dim, num_kv_heads, 8, jnp.bfloat16),
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
            "decode",
            lens,
            (num_heads, head_dim, num_kv_heads, 8, jnp.bfloat16),
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
            "prefill",
            lens,
            (num_heads, head_dim, num_kv_heads, 64, jnp.bfloat16),
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
            "decode",
            lens,
            (num_heads, head_dim, num_kv_heads, 64, jnp.bfloat16),
        )

    def test_mha_prefill_with_custom_mask(self):
        """Test JAX attention accuracy against PyTorch reference"""
        # Parameters
        num_heads = 8
        num_kv_heads = [8, 4]
        head_dim = 128
        lens = [(32, 32), (42, 66), (128, 256)]
        page_size = [
            1,
        ]
        causal_mask = False
        for size in page_size:
            for num_kv_head in num_kv_heads:
                self.run_test(
                    "prefill",
                    lens,
                    (num_heads, head_dim, num_kv_head, size, jnp.bfloat16, causal_mask),
                )

    def test_mha_decode_with_custom_mask(self):
        pass

    def test_mha_attention_sink_decode_accuracy(self):
        """Test attention sink accuracy in decode mode with MHA (num_heads == num_kv_heads)"""
        num_heads = 32
        num_kv_heads = 32
        head_dim = 128

        lens = [
            (1, 256),
            (1, 512),
            (1, 1024),
        ]

        rng = np.random.RandomState(789)
        attention_sink = jnp.array(rng.randn(num_heads).astype(np.float32))

        self.run_test(
            "decode",
            lens,
            (num_heads, head_dim, num_kv_heads, 1, jnp.bfloat16),
            max_total_token_size=200000,
            attention_sink=attention_sink,
        )

    def test_mha_attention_sink_prefill_accuracy(self):
        """Test attention sink accuracy in prefill mode with MHA (num_heads == num_kv_heads)"""
        num_heads = 32
        num_kv_heads = 32
        head_dim = 128

        lens = [
            (1, 128),
            (64, 64),
            (128, 256),
        ]

        rng = np.random.RandomState(101)
        attention_sink = jnp.array(rng.randn(num_heads).astype(np.float32))

        self.run_test(
            "prefill",
            lens,
            (num_heads, head_dim, num_kv_heads, 1, jnp.bfloat16),
            max_total_token_size=200000,
            attention_sink=attention_sink,
        )


if __name__ == "__main__":
    unittest.main()
