import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from sgl_jax.srt.hardware_backend.tt.attention.tt_backend import TTAttention


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
        self.backend = TTAttention(page_size=32, mesh=self.mesh)

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

        cache_locations = []
        start = 32
        for prefix_length, chunk_length in zip(prefix_tokens, chunk_tokens):
            total_tokens = prefix_length + chunk_length
            cache_capacity = ((total_tokens + 31) // 32) * 32
            user_locations = np.arange(start, start + cache_capacity, dtype=np.int32)
            cache_locations.append(user_locations)
            start += cache_capacity

        return SimpleNamespace(
            input_ids=input_ids,
            seq_lens=np.add(prefix_tokens, chunk_tokens, dtype=np.int32),
            cache_loc=np.concatenate(cache_locations),
            extend_seq_lens=np.asarray(chunk_tokens, dtype=np.int32),
            extend_prefix_lens=np.asarray(prefix_tokens, dtype=np.int32),
            real_bs=batch_size,
            logits_indices_selector=np.arange(batch_size, dtype=np.int32),
        )

    def test_first_scheduler_chunk_uses_bucketed_attention_shape(self):
        metadata = self.backend._prefill_metadata(self._batch(prefix_tokens=0, chunk_tokens=100))

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
        metadata = self.backend._prefill_metadata(self._batch(prefix_tokens=256, chunk_tokens=100))

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
            self.backend._prefill_metadata(self._batch(prefix_tokens=16, chunk_tokens=100))

    def test_attention_backend_does_not_cap_scheduler_chunks(self):
        metadata = self.backend._prefill_metadata(self._batch(prefix_tokens=0, chunk_tokens=1024))

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

        graph = jax.make_jaxpr(lambda q, k, v: self.backend._prefill(q, k, v, layer, pool))(
            jnp.zeros((1024, 8, 128), dtype=jnp.bfloat16),
            jnp.zeros((1024, 2, 128), dtype=jnp.bfloat16),
            jnp.zeros((1024, 2, 128), dtype=jnp.bfloat16),
        )

        calls = [
            equation
            for equation in graph.jaxpr.eqns
            if equation.primitive.name == "ffi_call"
            and equation.params["target_name"] == "tt.chunked_scaled_dot_product_attention"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].params["custom_call_api_version"], 4)
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

    def test_data_parallelism_is_rejected(self):
        mesh = SimpleNamespace(shape={"data": 2})
        with self.assertRaisesRegex(NotImplementedError, "dp_size=1"):
            TTAttention(page_size=32, mesh=mesh)

    def test_decode_metadata_pads_after_live_requests(self):
        batch = SimpleNamespace(
            input_ids=np.zeros(4, dtype=np.int32),
            seq_lens=np.asarray([32, 64, 0, 0], dtype=np.int32),
            cache_loc=np.arange(32, 160, dtype=np.int32),
            real_bs=2,
        )

        metadata = self.backend._decode_metadata(batch)

        np.testing.assert_array_equal(np.asarray(metadata.page_table)[:, 0], [1, 2, 0, 0])
        np.testing.assert_array_equal(np.asarray(metadata.positions), [31, 63, -1, -1])

    def test_scheduler_chunk_accepts_noncontiguous_physical_pages(self):
        batch = self._batch(prefix_tokens=256, chunk_tokens=256)
        page_starts = np.arange(32, 32 + 16 * 64, 64, dtype=np.int32)
        cache_pages = [start + np.arange(32, dtype=np.int32) for start in page_starts]
        batch.cache_loc = np.concatenate(cache_pages)

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
