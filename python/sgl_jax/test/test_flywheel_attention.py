"""--attention-backend flywheel against the FlashAttention (RPA v3) backend.

Each case runs both backends on the same batch and KV pool and checks that
the attention outputs and the updated KV caches agree. Needs flywheel-tpu.
"""

import types
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.flywheel_backend import (
    FlywheelAttention,
    build_block_table,
    build_kv_write_slices,
    extend_kernels,
    kv_write_slice_bound,
    select_extend_kernel,
)
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.test.flashattention_common import (
    create_test_data,
    mesh,
    write_prefix_tokens_for_kv,
)
from sgl_jax.test.test_utils import CustomTestCase

PAGE_SIZE = 128
CONTEXT_LEN = 8192


def _attention_layer(num_heads, num_kv_heads, head_dim):
    return RadixAttention(
        num_heads=num_heads,
        head_dim=head_dim,
        scaling=head_dim**-0.5,
        num_kv_heads=num_kv_heads,
        layer_id=0,
    )


def _host_batch(forward_batch, num_pad_tokens=0, num_pad_seqs=0):
    """The ModelWorkerBatch fields the flywheel backend reads, with the token
    and request axes padded as the scheduler's buckets pad them."""
    seq_lens = np.asarray(forward_batch.seq_lens)
    extend_seq_lens = (
        None
        if forward_batch.extend_seq_lens is None
        else np.pad(np.asarray(forward_batch.extend_seq_lens), (0, num_pad_seqs))
    )
    return types.SimpleNamespace(
        dp_size=1,
        forward_mode=forward_batch.forward_mode,
        seq_lens=np.pad(seq_lens, (0, num_pad_seqs)),
        input_ids=np.pad(np.asarray(forward_batch.input_ids), (0, num_pad_tokens)),
        cache_loc=np.asarray(forward_batch.cache_loc),
        extend_seq_lens=extend_seq_lens,
    )


class TestFlywheelHostMetadata(unittest.TestCase):
    def test_block_table_unpacks_pages(self):
        # Two requests of 130 and 128 tokens: 2 pages and 1 page, packed.
        table = build_block_table(
            np.array([7, 3, 9, 0], dtype=np.int32),
            np.array([130, 128, 0], dtype=np.int32),
            page_size=128,
            pages_per_seq=4,
        )
        np.testing.assert_array_equal(table, [[7, 3, 0, 0], [9, 0, 0, 0], [0, 0, 0, 0]])

    def test_block_table_rejects_overflow(self):
        with self.assertRaises(ValueError):
            build_block_table(
                np.arange(3, dtype=np.int32),
                np.array([300], dtype=np.int32),
                page_size=128,
                pages_per_seq=2,
            )

    def test_kv_write_slices_split_at_pages(self):
        # 100 cached tokens + 200 new on pages [5, 6, 7]: the new rows span
        # positions 100..299, cut at the 128 and 256 page boundaries.
        cu_q_lens = np.array([0, 200], dtype=np.int32)
        out = np.zeros((3, kv_write_slice_bound(256, 1, 128)), dtype=np.int32)
        n = build_kv_write_slices(
            cu_q_lens,
            np.array([300], dtype=np.int32),
            np.array([[5, 6, 7]], dtype=np.int32),
            page_size=128,
            out=out,
        )
        self.assertEqual(n, 3)
        np.testing.assert_array_equal(
            out[:, :n],
            [[5 * 128 + 100, 6 * 128, 7 * 128], [0, 28, 156], [28, 128, 44]],
        )

    def test_extend_kernel_selection(self):
        # A cached prefix goes to the paged kernel; a fresh prefill does not.
        self.assertEqual(
            select_extend_kernel(np.array([10, 1]), np.array([10, 500]), 512), "extend"
        )
        kernel = select_extend_kernel(np.array([1000]), np.array([1000]), 1024)
        self.assertIn(kernel, extend_kernels(1024))
        self.assertNotEqual(kernel, "extend")


class TestFlywheelAttention(CustomTestCase):
    def _run(self, mode, lens, num_heads, num_kv_heads, head_dim, num_pad_tokens=0, num_pad_seqs=0):
        forward_batch, pool, q, k, v = create_test_data(
            mode,
            lens,
            num_heads,
            head_dim,
            num_kv_heads,
            PAGE_SIZE,
            model_config={
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "num_hidden_layers": 1,
                "bf16": True,
            },
            # A whole number of pages.
            max_total_token_size=1600 * PAGE_SIZE,
        )
        extend_k, extend_v = write_prefix_tokens_for_kv(forward_batch, pool, lens, k, v)
        layer = _attention_layer(num_heads, num_kv_heads, head_dim)
        sharding = NamedSharding(mesh, P("data", "tensor"))
        q = q.reshape(q.shape[0], -1)

        @jax.jit
        def run(q, k, v, forward_batch, pool):
            return layer(q, k, v, forward_batch, pool)

        # Reference: the FlashAttention backend create_test_data installed.
        ref_out, ref_cache = run(
            jax.device_put(q, sharding),
            jax.device_put(extend_k, sharding),
            jax.device_put(extend_v, sharding),
            forward_batch,
            pool,
        )

        backend = FlywheelAttention(
            num_heads,
            num_kv_heads,
            head_dim,
            page_size=PAGE_SIZE,
            context_len=CONTEXT_LEN,
            mesh=mesh,
        )
        backend.forward_metadata = backend.get_forward_metadata(
            _host_batch(forward_batch, num_pad_tokens, num_pad_seqs)
        )
        forward_batch.attn_backend = backend

        def pad(x):
            return jax.device_put(
                jnp.pad(x, ((0, num_pad_tokens),) + ((0, 0),) * (x.ndim - 1)), sharding
            )

        out, cache = run(pad(q), pad(extend_k), pad(extend_v), forward_batch, pool)

        num_tokens = q.shape[0]
        out = np.asarray(out, dtype=np.float32)
        np.testing.assert_allclose(
            out[:num_tokens], np.asarray(ref_out, dtype=np.float32), rtol=2e-2, atol=1e-2
        )
        np.testing.assert_array_equal(out[num_tokens:], 0)
        np.testing.assert_array_equal(
            np.asarray(cache, dtype=np.float32), np.asarray(ref_cache, dtype=np.float32)
        )

    # Qwen3-8B's attention shape (32 query heads, 8 KV heads, head_dim 128).
    def test_prefill_one_request(self):
        self._run("prefill", [(1024, 1024)], 32, 8, 128)

    def test_prefill_ragged(self):
        self._run("prefill", [(300, 300), (1000, 1000), (57, 57)], 32, 8, 128)

    def test_prefill_padded_bucket(self):
        self._run(
            "prefill", [(700, 700), (100, 100)], 32, 8, 128, num_pad_tokens=224, num_pad_seqs=2
        )

    def test_extend_with_prefix(self):
        self._run("prefill", [(200, 1500), (64, 64), (130, 260)], 32, 8, 128)

    def test_mixed_extend_and_decode_rows(self):
        self._run("prefill", [(1, 800), (1, 129), (300, 300), (77, 2000)], 32, 8, 128)

    def test_decode(self):
        self._run("decode", [(1, 130), (1, 1), (1, 2048), (1, 513)], 32, 8, 128)

    def test_decode_padded_batch(self):
        self._run("decode", [(1, 300), (1, 4000)], 32, 8, 128, num_pad_tokens=6, num_pad_seqs=6)

    def test_head_dim_256_mqa(self):
        self._run("prefill", [(256, 512), (128, 128)], 16, 1, 256)
        self._run("decode", [(1, 512), (1, 3)], 16, 1, 256)

    def test_rejects_unsupported_layer(self):
        backend = FlywheelAttention(
            32, 8, 128, page_size=PAGE_SIZE, context_len=CONTEXT_LEN, mesh=mesh
        )
        layer = RadixAttention(
            num_heads=32,
            head_dim=128,
            scaling=128**-0.5,
            num_kv_heads=8,
            layer_id=0,
            sliding_window_size=1024,
        )
        forward_batch = types.SimpleNamespace(forward_mode=ForwardMode.DECODE)
        q = jnp.zeros((1, 32 * 128), jnp.bfloat16)
        with self.assertRaisesRegex(NotImplementedError, "sliding window"):
            backend(q, q, q, layer, forward_batch, None)

    def test_rejects_unsupported_config(self):
        with self.assertRaisesRegex(NotImplementedError, "head_dim"):
            FlywheelAttention(32, 8, 64, page_size=PAGE_SIZE, context_len=CONTEXT_LEN, mesh=mesh)
        with self.assertRaisesRegex(ValueError, "page size"):
            FlywheelAttention(32, 8, 128, page_size=64, context_len=CONTEXT_LEN, mesh=mesh)


if __name__ == "__main__":
    unittest.main()
