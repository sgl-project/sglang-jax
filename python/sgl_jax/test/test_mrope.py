import jax
import jax.numpy as jnp
import numpy as np
import unittest

from sgl_jax.srt.layers.embeddings import MRotaryEmbedding


class TestMRotaryEmbedding(unittest.TestCase):

    def setUp(self):
        # Basic configuration
        self.head_size = 128
        self.rotary_dim = 128
        self.max_pos = 1024
        self.base = 10000
        self.batch_size = 2
        self.seq_len = 10
        self.dtype = jnp.float32

        # Typical Qwen2-VL split configuration [16, 24, 24] -> Sum 64 (head_size // 2)
        self.mrope_section = [16, 24, 24]

        # Layer for Chunked mode (default)
        self.mrope_layer_chunked = MRotaryEmbedding(
            head_size=self.head_size,
            rotary_dim=self.rotary_dim,
            max_position_embeddings=self.max_pos,
            base=self.base,
            is_neox_style=True,
            dtype=self.dtype,
            mrope_section=self.mrope_section,
            mrope_interleaved=False
        )

        # Layer for Interleaved mode
        self.mrope_layer_interleaved = MRotaryEmbedding(
            head_size=self.head_size,
            rotary_dim=self.rotary_dim,
            max_position_embeddings=self.max_pos,
            base=self.base,
            is_neox_style=True,
            dtype=self.dtype,
            mrope_section=self.mrope_section,
            mrope_interleaved=True
        )

    def _reference_rotate_half(self, x):
        """Reference implementation of rotate_half"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return np.concatenate((-x2, x1), axis=-1)

    def _reference_apply_rope_1d(self, x, pos, inv_freq):
        """
        Apply standard 1D RoPE using numpy logic
        x: [N, dim]
        pos: [N]
        inv_freq: [dim/2]
        """
        # 1. Compute frequencies: theta = pos * inv_freq
        # inv_freq: [dim/2] -> [1] or broadcast
        # pos: [N]

        # If in Interleaved mode, inv_freq might already be a constructed complex tensor
        # So we handle it more generally here
        if inv_freq.ndim == 1:
            freqs = np.outer(pos, inv_freq)
        else:
            # In this case, inv_freq is already the frequency table of shape [N, dim/2]
            freqs = inv_freq

        # 2. Expand to [N, dim] to match x (NeoX style: cat(cos, cos))
        emb = np.concatenate((freqs, freqs), axis=-1)
        cos = np.cos(emb)
        sin = np.sin(emb)

        # 3. Apply rotation
        return x * cos + self._reference_rotate_half(x) * sin

    def reference_mrope(self, q, k, positions, interleaved=False):
        """
        Reference implementation for mRoPE (Numpy based).
        Handles both Chunked and Interleaved modes for NeoX Style RoPE.
        """
        q_np = np.array(q)
        k_np = np.array(k)
        pos_np = np.array(positions)  # [3, N]

        dim = self.rotary_dim
        # Base inv_freq [64]
        inv_freq_all = 1.0 / (self.base ** (np.arange(0, dim, 2).astype(np.float32) / dim))

        # Generate base frequency table [3, N, 64]
        # Row 0: Time Freqs
        # Row 1: Height Freqs
        # Row 2: Width Freqs
        freqs_all = np.einsum("cn,d->cnd", pos_np, inv_freq_all)

        # Target frequency table [N, 64]
        target_freqs = np.zeros((pos_np.shape[1], dim // 2), dtype=self.dtype)

        if not interleaved:
            # --- Chunked Mode (Legacy Logic) ---
            split_lens = self.mrope_section
            # Manually fill sections
            # Section 0 (Time): 0 ~ 16
            # Section 1 (Height): 16 ~ 40
            # Section 2 (Width): 40 ~ 64
            start = 0
            for i, length in enumerate(split_lens):
                end = start + length
                target_freqs[:, start:end] = freqs_all[i, :, start:end]
                start = end
        else:
            # --- Interleaved Mode (New Logic) ---
            # 1. Fill everything with Time frequencies first
            target_freqs[:] = freqs_all[0]  # Copy Time freqs everywhere

            # 2. Overwrite Height: indices 1, 4, 7...
            # Corresponds to slice x[..., 1:end:3]
            h_len = self.mrope_section[1]
            h_slice = slice(1, h_len * 3, 3)
            target_freqs[:, h_slice] = freqs_all[1, :, h_slice]

            # 3. Overwrite Width: indices 2, 5, 8...
            w_len = self.mrope_section[2]
            w_slice = slice(2, w_len * 3, 3)
            target_freqs[:, w_slice] = freqs_all[2, :, w_slice]

        # Now we have the correct target_freqs [N, 64], apply RoPE directly.
        # Since it is NeoX Style, we treat the two halves of q/k separately.
        # The tricky part is that target_freqs is for rotary_dim // 2.
        # Inside _reference_apply_rope_1d, it concatenates it to apply to the full dim.
        # This aligns perfectly with our logic: frequencies are aligned.

        q_out = self._reference_apply_rope_1d(q_np, None, target_freqs)
        k_out = self._reference_apply_rope_1d(k_np, None, target_freqs)

        return q_out, k_out

    def _run_comparison(self, layer, interleaved):
        num_tokens = self.batch_size * self.seq_len
        rng = np.random.default_rng(42)

        q_input = jnp.array(rng.standard_normal((num_tokens, self.head_size)), dtype=self.dtype)
        k_input = jnp.array(rng.standard_normal((num_tokens, self.head_size)), dtype=self.dtype)

        pos_t = rng.integers(0, 10, (num_tokens,))
        pos_h = rng.integers(0, 100, (num_tokens,))
        pos_w = rng.integers(0, 100, (num_tokens,))
        positions = jnp.array(np.stack([pos_t, pos_h, pos_w]), dtype=jnp.int32)

        # JAX Output
        q_out_jax, k_out_jax = layer(positions, q_input, k_input)

        # Reference Output
        q_out_ref, k_out_ref = self.reference_mrope(q_input, k_input, positions, interleaved=interleaved)

        max_diff_q = np.max(np.abs(q_out_jax - q_out_ref))

        print(f"Max Diff Q ({'Interleaved' if interleaved else 'Chunked'}): {max_diff_q}")

        np.testing.assert_allclose(q_out_jax, q_out_ref, rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(k_out_jax, k_out_ref, rtol=1e-5, atol=1e-4)

    def test_correctness_chunked(self):
        print("\n=== Testing Chunked Mode ===")
        self._run_comparison(self.mrope_layer_chunked, interleaved=False)
        print(">> Passed!")

    def test_correctness_interleaved(self):
        print("\n=== Testing Interleaved Mode ===")
        self._run_comparison(self.mrope_layer_interleaved, interleaved=True)
        print(">> Passed!")

    def test_fallback_1d(self):
        print("\n=== Testing 1D Fallback ===")
        num_tokens = 10
        positions_1d = jnp.arange(num_tokens, dtype=jnp.int32)
        q_input = jnp.ones((num_tokens, self.head_size), dtype=self.dtype)
        k_input = jnp.ones((num_tokens, self.head_size), dtype=self.dtype)

        # Just ensure it doesn't crash
        q_out, _ = self.mrope_layer_chunked(positions_1d, q_input, k_input)
        self.assertEqual(q_out.shape, q_input.shape)
        print(">> Passed!")


if __name__ == "__main__":
    unittest.main()