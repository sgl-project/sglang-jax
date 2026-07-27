import unittest
from unittest import mock

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.ragged_paged_attention import (
    ragged_paged_attention_v3 as rpa_v3,
)
from sgl_jax.test.flashattention_common import AttentionTestBase


class TestFlashAttentionGQA(AttentionTestBase):
    def test_v6_custom_mask_caps_query_tiles(self):
        kwargs = dict(
            q_dtype=jnp.bfloat16,
            kv_dtype=jnp.bfloat16,
            actual_num_q_heads=128,
            actual_num_kv_heads=8,
            head_dim=128,
            page_size=16,
            max_num_tokens=256,
            max_num_seqs=3,
            pages_per_seq=16,
            case=rpa_v3.RpaCase.MIXED,
        )

        with mock.patch.object(rpa_v3, "get_tpu_version", return_value=6):
            without_mask = rpa_v3.get_default_block_sizes(**kwargs, use_custom_mask=False)
            with_mask = rpa_v3.get_default_block_sizes(**kwargs, use_custom_mask=True)

        self.assertEqual(without_mask["bq_sz"], 32)
        self.assertEqual(without_mask["bq_csz"], 32)
        self.assertEqual(with_mask["bq_sz"], 16)
        self.assertEqual(with_mask["bq_csz"], 16)

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
        self.run_test("prefill", lens, (num_heads, head_dim, num_kv_heads, 64, jnp.bfloat16))

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

        self.run_test("decode", lens, (num_heads, head_dim, num_kv_heads, 64, jnp.bfloat16))

    def test_gqa_prefill_accuracy_page_size_64_temperature(self):
        """Test JAX attention accuracy against PyTorch reference
        Testcase (1024, 1024) fails on token 607, possible precision issue?
        Token 607: max_diff=0.023438, jax_mean=-0.011597, expected_mean=-0.011597, jax_std=0.048096, expected_std=0.047607
        """
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
            (123, 522),
            (1, 511),
            (1024, 1024),
        ]
        self.run_test(
            "prefill",
            lens,
            (num_heads, head_dim, num_kv_heads, 64, jnp.bfloat16),
            xai_temperature_len=512,
        )

    def test_gqa_decode_accuracy_page_size_64_temperature(self):
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
            "decode",
            lens,
            (num_heads, head_dim, num_kv_heads, 64, jnp.bfloat16),
            xai_temperature_len=512,
        )

    def test_gqa_prefill_accuracy_page_size_1_temperature(self):
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
            "prefill",
            lens,
            (num_heads, head_dim, num_kv_heads, 1, jnp.bfloat16),
            xai_temperature_len=512,
        )

    def test_gqa_decode_accuracy_page_size_1_temperature(self):
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
            "decode",
            lens,
            (num_heads, head_dim, num_kv_heads, 1, jnp.bfloat16),
            xai_temperature_len=512,
        )

    def test_gqa_prefill_with_custom_mask(self):
        """Test JAX attention accuracy against PyTorch reference"""
        # Parameters
        num_heads = 128
        num_kv_heads = 8
        head_dim = 128
        lens = [(32, 32), (42, 66), (128, 256)]
        page_size = [
            16,
        ]
        causal_mask = False
        for size in page_size:
            self.run_test(
                "prefill",
                lens,
                (num_heads, head_dim, num_kv_heads, size, jnp.bfloat16, causal_mask),
            )

    def test_gqa_decode_with_custom_mask(self):
        pass

    def test_gqa_4q1kv_attention_sink_decode_accuracy(self):
        """Test attention sink with GQA (4 q_heads, 1 kv_head) in decode mode"""
        num_heads = 4
        num_kv_heads = 1
        head_dim = 128

        lens = [
            (1, 256),
            (1, 512),
            (1, 1024),
        ]

        rng = np.random.RandomState(202)
        attention_sink = jnp.array(rng.randn(num_heads).astype(np.float32))

        self.run_test_on_single_device(
            "decode",
            lens,
            (num_heads, head_dim, num_kv_heads, 1, jnp.bfloat16),
            max_total_token_size=200000,
            attention_sink=attention_sink,
        )

    def test_gqa_4q1kv_attention_sink_prefill_accuracy(self):
        """Test attention sink with GQA (4 q_heads, 1 kv_head) in prefill mode"""
        num_heads = 4
        num_kv_heads = 1
        head_dim = 128

        lens = [
            (1, 128),
            (64, 64),
            (128, 256),
        ]

        rng = np.random.RandomState(303)
        attention_sink = jnp.array(rng.randn(num_heads).astype(np.float32))

        self.run_test_on_single_device(
            "prefill",
            lens,
            (num_heads, head_dim, num_kv_heads, 1, jnp.bfloat16),
            max_total_token_size=200000,
            attention_sink=attention_sink,
        )


if __name__ == "__main__":
    unittest.main()
