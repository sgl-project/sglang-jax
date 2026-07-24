import unittest

import jax.numpy as jnp
import numpy as np

from sgl_jax.test.flashattention_common import AttentionTestBase


class TestFlashAttentionMisc(AttentionTestBase):
    def test_sliding_window_and_soft_cap_prefill_accuracy(self):
        """Test combined sliding window and soft cap attention accuracy in prefill mode"""
        # Parameters
        num_heads = 32
        num_kv_heads = 32
        head_dim = 256
        sliding_window_size = 512
        logit_cap = 20.0

        lens = [
            (1, 128),
            (64, 64),
            (128, 256),
            (100, 300),
            (1, 400),
        ]

        self.run_test(
            "prefill",
            lens,
            (num_heads, head_dim, num_kv_heads, 1, jnp.bfloat16),
            sliding_window=sliding_window_size,
            logit_cap=logit_cap,
        )

    def test_sliding_window_and_soft_cap_decode_accuracy(self):
        """Test combined sliding window and soft cap attention accuracy in decode mode"""
        # Parameters
        num_heads = 32
        num_kv_heads = 32
        head_dim = 128
        sliding_window_size = 512
        logit_cap = 20.0

        lens = [
            (1, 256),
            (1, 400),
            (1, 512),
            (1, 1024),
        ]

        self.run_test(
            "decode",
            lens,
            (num_heads, head_dim, num_kv_heads, 1, jnp.bfloat16),
            sliding_window=sliding_window_size,
            logit_cap=logit_cap,
        )

    def test_attention_sink_decode_accuracy(self):
        """Test attention sink accuracy in decode mode with per-head sink logits"""
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128

        lens = [
            (1, 256),
            (1, 512),
            (1, 1024),
        ]

        # Per-head sink logits
        rng = np.random.RandomState(123)
        attention_sink = jnp.array(rng.randn(num_heads).astype(np.float32))

        self.run_test(
            "decode",
            lens,
            (num_heads, head_dim, num_kv_heads, 1, jnp.bfloat16),
            attention_sink=attention_sink,
        )

    def test_attention_sink_prefill_accuracy(self):
        """Test attention sink accuracy in prefill mode with per-head sink logits"""
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128

        lens = [
            (1, 128),
            (64, 64),
            (128, 256),
        ]

        # Per-head sink logits
        rng = np.random.RandomState(456)
        attention_sink = jnp.array(rng.randn(num_heads).astype(np.float32))

        self.run_test(
            "prefill",
            lens,
            (num_heads, head_dim, num_kv_heads, 1, jnp.bfloat16),
            attention_sink=attention_sink,
        )

    def test_single_head_attention_sink_decode_accuracy(self):
        """Test attention sink with single head MHA (1 q_head, 1 kv_head) in decode mode"""
        num_heads = 1
        num_kv_heads = 1
        head_dim = 128

        lens = [
            (1, 256),
            (1, 512),
            (1, 1024),
        ]

        rng = np.random.RandomState(404)
        attention_sink = jnp.array(rng.randn(num_heads).astype(np.float32))

        self.run_test_on_single_device(
            "decode",
            lens,
            (num_heads, head_dim, num_kv_heads, 1, jnp.bfloat16),
            max_total_token_size=200000,
            attention_sink=attention_sink,
        )

    def test_single_head_attention_sink_prefill_accuracy(self):
        """Test attention sink with single head MHA (1 q_head, 1 kv_head) in prefill mode"""
        num_heads = 1
        num_kv_heads = 1
        head_dim = 128

        lens = [
            (1, 128),
            (64, 64),
            (128, 256),
        ]

        rng = np.random.RandomState(505)
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
