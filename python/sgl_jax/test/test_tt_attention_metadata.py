import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

# Trigger JAX plugin discovery before importing the TT-only MLIR lowerings.
jax.devices()

from sgl_jax.srt.layers.attention.tt_backend import TTAttention


class TTAttentionMetadataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        devices = np.asarray(jax.devices()).reshape(1, 1)
        cls.mesh = Mesh(
            devices,
            ("data", "tensor"),
            axis_types=(jax.sharding.AxisType.Explicit,) * 2,
        )

    def setUp(self):
        self.backend = TTAttention(
            num_attn_heads=8,
            num_kv_heads=2,
            head_dim=128,
            page_size=32,
            mesh=self.mesh,
        )

    @staticmethod
    def _batch(
        *,
        prefix_tokens: int | list[int],
        chunk_tokens: int | list[int],
        batch_size: int = 1,
    ) -> SimpleNamespace:
        if isinstance(prefix_tokens, int):
            prefix_tokens = [prefix_tokens] * batch_size
        if isinstance(chunk_tokens, int):
            chunk_tokens = [chunk_tokens] * batch_size
        if len(prefix_tokens) != len(chunk_tokens):
            raise ValueError("prefix and chunk length lists must match")
        batch_size = len(chunk_tokens)

        input_bucket = 1 << max(6, (sum(chunk_tokens) - 1).bit_length())
        input_ids = np.zeros(input_bucket, dtype=np.int32)
        out_cache_loc = np.full(input_bucket, -1, dtype=np.int32)

        cache_locations = []
        chunk_locations = []
        start = 32
        for prefix_length, chunk_length in zip(prefix_tokens, chunk_tokens):
            total_tokens = prefix_length + chunk_length
            cache_capacity = ((total_tokens + 31) // 32) * 32
            user_locations = np.arange(
                start, start + cache_capacity, dtype=np.int32
            )
            cache_locations.append(user_locations)
            chunk_locations.append(
                user_locations[prefix_length : prefix_length + chunk_length]
            )
            start += cache_capacity
        out_cache_loc[: sum(chunk_tokens)] = np.concatenate(chunk_locations)

        return SimpleNamespace(
            input_ids=input_ids,
            seq_lens=np.add(prefix_tokens, chunk_tokens, dtype=np.int32),
            out_cache_loc=out_cache_loc,
            cache_loc=np.concatenate(cache_locations),
            extend_seq_lens=np.asarray(chunk_tokens, dtype=np.int32),
            extend_prefix_lens=np.asarray(prefix_tokens, dtype=np.int32),
            real_bs=batch_size,
            real_bs_per_dp=[batch_size],
            logits_indices_selector=np.arange(batch_size, dtype=np.int32),
            dp_size=1,
            per_dp_bs_size=batch_size,
        )

    def test_first_scheduler_chunk_uses_bucketed_attention_shape(self):
        metadata = self.backend._prefill_metadata(
            self._batch(prefix_tokens=0, chunk_tokens=100)
        )

        self.assertEqual(metadata.prefill_input_indices.shape, (1, 512))
        np.testing.assert_array_equal(
            np.asarray(metadata.page_table)[0],
            np.arange(1, 5, dtype=np.int32),
        )
        np.testing.assert_array_equal(
            np.asarray(metadata.prefill_chunk_start),
            np.asarray([0], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            np.asarray(metadata.fill_page_table)[0, :4],
            np.asarray([1, 2, 3, 4], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            np.asarray(metadata.prefill_input_indices)[0, :100],
            np.arange(100, dtype=np.int32),
        )

    def test_later_scheduler_chunk_uses_full_page_table(self):
        metadata = self.backend._prefill_metadata(
            self._batch(prefix_tokens=256, chunk_tokens=100)
        )

        self.assertEqual(metadata.prefill_input_indices.shape, (1, 512))
        np.testing.assert_array_equal(
            np.asarray(metadata.page_table)[0],
            np.arange(1, 13, dtype=np.int32),
        )
        np.testing.assert_array_equal(
            np.asarray(metadata.prefill_chunk_start),
            np.asarray([256], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            np.asarray(metadata.fill_page_table)[0, :4],
            np.arange(9, 13, dtype=np.int32),
        )

    def test_scheduler_chunk_requires_page_aligned_prefix(self):
        with self.assertRaisesRegex(ValueError, "prefixes must be page-aligned"):
            self.backend._prefill_metadata(
                self._batch(prefix_tokens=16, chunk_tokens=100)
            )

    def test_attention_backend_does_not_cap_scheduler_chunks(self):
        metadata = self.backend._prefill_metadata(
            self._batch(prefix_tokens=0, chunk_tokens=1024)
        )

        self.assertEqual(metadata.prefill_input_indices.shape, (1, 1024))

    def test_prefill_uses_one_batched_attention_call(self):
        metadata = self.backend._prefill_metadata(
            self._batch(
                prefix_tokens=[0, 512],
                chunk_tokens=[512, 512],
            )
        )
        np.testing.assert_array_equal(
            np.asarray(metadata.prefill_chunk_start),
            np.asarray([0, 512], dtype=np.int32),
        )
        self.backend.forward_metadata = metadata
        cache = jnp.zeros((96, 2, 32, 128), dtype=jnp.bfloat16)
        pool = SimpleNamespace(get_kv_buffer=lambda _layer_id: (cache, cache))
        layer = SimpleNamespace(layer_id=0, scaling=128**-0.5)

        graph = jax.make_jaxpr(
            lambda q, k, v: self.backend._prefill(q, k, v, layer, pool)
        )(
            jnp.zeros((1024, 8, 128), dtype=jnp.bfloat16),
            jnp.zeros((1024, 2, 128), dtype=jnp.bfloat16),
            jnp.zeros((1024, 2, 128), dtype=jnp.bfloat16),
        )

        calls = [
            equation
            for equation in graph.jaxpr.eqns
            if equation.primitive.name
            == "tt_chunked_scaled_dot_product_attention"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].invars[0].aval.shape[0], 2)
        self.assertEqual(calls[0].invars[3].aval.shape[0], 2)
        self.assertEqual(calls[0].invars[4].aval.shape, (2,))

    def test_ragged_scheduler_batch_is_repacked_per_request(self):
        metadata = self.backend._prefill_metadata(
            self._batch(
                prefix_tokens=[0, 512, 256],
                chunk_tokens=[100, 64, 200],
            )
        )

        self.assertEqual(metadata.prefill_input_indices.shape, (3, 512))
        np.testing.assert_array_equal(
            np.asarray(metadata.prefill_chunk_start),
            np.asarray([0, 512, 256], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            np.asarray(metadata.prefill_input_indices)[0, :100],
            np.arange(100, dtype=np.int32),
        )
        np.testing.assert_array_equal(
            np.asarray(metadata.prefill_input_indices)[1, :64],
            np.arange(100, 164, dtype=np.int32),
        )
        np.testing.assert_array_equal(
            np.asarray(metadata.prefill_input_indices)[2, :200],
            np.arange(164, 364, dtype=np.int32),
        )
        expected_output_indices = np.zeros(512, dtype=np.int32)
        expected_output_indices[:100] = np.arange(100, dtype=np.int32)
        expected_output_indices[100:164] = 512 + np.arange(64, dtype=np.int32)
        expected_output_indices[164:364] = 1024 + np.arange(200, dtype=np.int32)
        np.testing.assert_array_equal(
            np.asarray(metadata.prefill_output_indices), expected_output_indices
        )

    def test_ragged_scheduler_batch_preserves_dp_segments(self):
        rank_0_locations = np.arange(32, 32 + 256, dtype=np.int32)
        rank_1_locations = np.pad(
            np.arange(320, 320 + 128, dtype=np.int32), (0, 128)
        )
        batch = SimpleNamespace(
            input_ids=np.zeros(512, dtype=np.int32),
            seq_lens=np.asarray([100, 128, 0, 112, 0, 0], dtype=np.int32),
            out_cache_loc=None,
            cache_loc=np.concatenate((rank_0_locations, rank_1_locations)),
            extend_seq_lens=np.asarray([100, 64, 0, 80, 0, 0], dtype=np.int32),
            extend_prefix_lens=np.asarray([0, 64, 0, 32, 0, 0], dtype=np.int32),
            real_bs=3,
            real_bs_per_dp=[2, 1],
            logits_indices_selector=np.asarray([0, 1, 3], dtype=np.int32),
            dp_size=2,
            per_dp_bs_size=3,
        )

        metadata = self.backend._prefill_metadata(batch)

        input_indices = np.asarray(metadata.prefill_input_indices)
        np.testing.assert_array_equal(input_indices[0, :100], np.arange(100))
        np.testing.assert_array_equal(input_indices[1, :64], np.arange(100, 164))
        np.testing.assert_array_equal(input_indices[2, :80], np.arange(256, 336))

        fill_page_table = np.asarray(metadata.fill_page_table)
        np.testing.assert_array_equal(fill_page_table[0, :4], np.arange(1, 5))
        np.testing.assert_array_equal(fill_page_table[1, :2], np.arange(7, 9))
        np.testing.assert_array_equal(fill_page_table[2, :3], np.arange(11, 14))

    def test_scheduler_chunk_accepts_noncontiguous_physical_pages(self):
        batch = self._batch(prefix_tokens=256, chunk_tokens=256)
        page_starts = np.arange(32, 32 + 16 * 64, 64, dtype=np.int32)
        cache_pages = [
            start + np.arange(32, dtype=np.int32) for start in page_starts
        ]
        batch.cache_loc = np.concatenate(cache_pages)
        batch.out_cache_loc = np.concatenate(cache_pages[8:])

        metadata = self.backend._prefill_metadata(batch)

        np.testing.assert_array_equal(
            np.asarray(metadata.page_table)[0],
            page_starts // 32,
        )
        np.testing.assert_array_equal(
            np.asarray(metadata.fill_page_table)[0, :8],
            page_starts[8:] // 32,
        )


if __name__ == "__main__":
    unittest.main()
