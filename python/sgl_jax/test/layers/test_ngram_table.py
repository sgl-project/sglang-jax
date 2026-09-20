"""Loading and gathering the PLE table.

Built against synthetic safetensors shards with the released checkpoint's
naming and split, so it runs on a CPU runner without the 95.4 GiB table.
"""

import tempfile
import unittest
from pathlib import Path

import ml_dtypes
import numpy as np

from sgl_jax.srt.layers.ngram_embedding import build_hash_params
from sgl_jax.srt.layers.ngram_table import NGramTable, shard_placements
from sgl_jax.test.test_utils import CustomTestCase

PREFIX = "model.language_model.layers.1.ple.ple_embedding"
NGRAM_SIZE = 3
HEADS_PER_NGRAM = 2
HEADS = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM  # 4
DIM = 5
PLE_EMBED_DIM = HEADS * DIM  # 20
VOCAB_BASE = 1000
SPLIT_PARTS = 8
EOS = 7


class _Config:
    ngram_size = NGRAM_SIZE
    heads_per_ngram = HEADS_PER_NGRAM
    vocab_size = 128
    ngram_vocab_size_base = VOCAB_BASE
    make_ngram_vocab_size_divisible_by = 128
    split_ngram_parts = SPLIT_PARTS
    ple_embed_dim = PLE_EMBED_DIM
    eos_token_id = EOS


def _params():
    return build_hash_params(
        ngram_size=NGRAM_SIZE,
        heads_per_ngram=HEADS_PER_NGRAM,
        vocab_size=_Config.vocab_size,
        ngram_vocab_size_base=VOCAB_BASE,
        eos_token_id=EOS,
    )


def _write_checkpoint(directory, params, *, corrupt=None, drop_buffer=False):
    """Synthetic shards laid out exactly as the real checkpoint names them.

    Row r gets the constant value r % 65535 in every column, so a gather's
    correctness is readable straight off the returned bits.
    """
    from safetensors.numpy import save_file

    # Shard the padded height, as the exporter does.
    padded = ((params.total_vocab_size + 127) // 128) * 128
    placements = shard_placements(padded, SPLIT_PARTS)
    weight_files = {}
    for p in placements:
        rows = (np.arange(p.start, p.start + p.rows) % 65535).astype(np.uint16)
        tensor = np.repeat(rows[:, None], DIM, axis=1)
        name = f"{PREFIX}.ngram_embedding.shard_{p.index}.weight"
        path = Path(directory) / f"shard_{p.index}.safetensors"
        save_file({name: tensor}, str(path))
        weight_files[name] = str(path)

    buffers = {
        "layer_multipliers": params.multipliers.copy(),
        "ngram_heads_vocab_sizes": params.sizes.copy(),
        "ngram_heads_offsets": params.offsets.copy(),
    }
    if corrupt is not None:
        buffers[corrupt][0] += 2
    if not drop_buffer:
        path = Path(directory) / "buffers.safetensors"
        save_file({f"{PREFIX}.{k}": v for k, v in buffers.items()}, str(path))
        for k in buffers:
            weight_files[f"{PREFIX}.{k}"] = str(path)
    return weight_files


class TestShardPlacement(CustomTestCase):
    def test_released_checkpoint_split(self):
        """All 128 shards are 2,500,012 rows -- checked against the real headers.

        The exporter shards the padded height (320,001,536), not the sum of
        the 16 primes (320,001,446). Sharding the unpadded number gives a
        short last shard and misplaces every boundary after the first, which
        would load the table 90 rows out of alignment at the tail.
        """
        padded = 320_001_536
        places = shard_placements(padded, 128)
        self.assertEqual(len(places), 128)
        self.assertEqual(places[0].start, 0)
        self.assertTrue(all(p.rows == 2_500_012 for p in places))
        self.assertEqual(places[-1].start + places[-1].rows, padded)
        # The 90 padding rows sit past the last addressable id.
        self.assertGreater(padded, 320_001_446)

    def test_unpadded_split_would_be_wrong(self):
        """Pin the trap the line above avoids."""
        wrong = shard_placements(320_001_446, 128)
        self.assertNotEqual(wrong[-1].rows, 2_500_012)

    def test_placements_tile_without_gaps_or_overlap(self):
        for total, parts in ((1000, 8), (1024, 8), (7, 8), (320_001_446, 128)):
            places = shard_placements(total, parts)
            covered = 0
            for p in places:
                self.assertEqual(p.start, covered)
                covered += p.rows
            self.assertEqual(covered, total)

    def test_more_parts_than_rows_drops_empty_shards(self):
        places = shard_placements(3, 8)
        self.assertEqual(sum(p.rows for p in places), 3)
        self.assertTrue(all(p.rows > 0 for p in places))


class TestLoadAndGather(CustomTestCase):
    def test_round_trip_through_safetensors(self):
        params = _params()
        with tempfile.TemporaryDirectory() as d:
            files = _write_checkpoint(d, params)
            table = NGramTable.from_safetensors(files, _Config())

            self.assertEqual(table.rows, params.total_vocab_size)
            self.assertEqual(table.dim, DIM)
            # padded up to make_ngram_vocab_size_divisible_by
            self.assertEqual(table.padded_rows % 128, 0)
            self.assertGreaterEqual(table.padded_rows, table.rows)

            rng = np.random.default_rng(0)
            ids = rng.integers(0, table.rows, size=(6, HEADS)).astype(np.int32)
            got = table.gather(ids)
            # gather hands JAX bfloat16; compare through the raw bits, since
            # some of the synthetic patterns are NaN and NaN != NaN.
            self.assertEqual(got.dtype, ml_dtypes.bfloat16)
            bits = got.view(np.uint16).reshape(6, HEADS, DIM)
            want = np.repeat((ids % 65535).astype(np.uint16)[:, :, None], DIM, axis=2)
            np.testing.assert_array_equal(bits, want)

    def test_gather_is_the_same_single_and_multi_threaded(self):
        """The row-count threshold picks the thread count; both paths must agree."""
        from sgl_jax.srt.layers import ngram_table as mod

        params = _params()
        with tempfile.TemporaryDirectory() as d:
            table = NGramTable.from_safetensors(_write_checkpoint(d, params), _Config())
            rng = np.random.default_rng(1)
            # Enough rows that _thread_count returns > 1.
            ids = rng.integers(0, table.rows, size=(4096, HEADS)).astype(np.int32)
            self.assertGreater(mod._thread_count(ids.size), 1)
            threaded = table.gather(ids).view(np.uint16).copy()
            saved, mod._ROWS_PER_THREAD = mod._ROWS_PER_THREAD, 1 << 40
            try:
                self.assertEqual(mod._thread_count(ids.size), 1)
                single = table.gather(ids).view(np.uint16).copy()
            finally:
                mod._ROWS_PER_THREAD = saved
            np.testing.assert_array_equal(threaded, single)

    # -- ported from sglang test/registered/kernels/ops/embeddings/
    #    test_qwen4_ple_offload.py; their backend-specific cases (pinned vs
    #    file-backed mmap, prefetch, RSS trim) are CUDA host-pointer machinery
    #    with no TPU equivalent, so only the gather-level ones carry over.

    def test_gather_over_odd_embedding_dims(self):
        """sglang parametrizes embedding_dim over 7 / 64 / 257 and the id dtype
        over int32 / int64. A dim that is not a nice multiple catches a stride
        bug that a power of two hides."""
        for dim in (7, 64, 257):
            for dtype in (np.int32, np.int64):
                with self.subTest(dim=dim, dtype=np.dtype(dtype).name):
                    params = _params()
                    table = NGramTable(params, dim)
                    rng = np.random.default_rng(0)
                    table.data[:] = rng.integers(
                        0, 1 << 16, size=table.data.shape, dtype=np.uint16
                    )
                    ids = np.array([[0, 7, 3, 1], [4, 1, 6, 2]], dtype=dtype)[:, :HEADS]
                    got = table.gather(ids)
                    self.assertEqual(got.shape, (ids.shape[0], HEADS * dim))
                    want = table.data[ids.reshape(-1).astype(np.int64)].reshape(
                        ids.shape[0], HEADS * dim
                    )
                    np.testing.assert_array_equal(got.view(np.uint16), want)

    def test_gather_reuses_the_out_buffer(self):
        """sglang asserts the returned tensor aliases the caller's ``out``."""
        params = _params()
        table = NGramTable(params, DIM)
        rng = np.random.default_rng(1)
        table.data[:] = rng.integers(0, 1 << 16, size=table.data.shape, dtype=np.uint16)
        ids = rng.integers(0, table.rows, size=(6, HEADS)).astype(np.int32)

        out = np.full((6, HEADS, DIM), 0xDEAD, np.uint16)
        got = table.gather(ids, out=out)
        self.assertTrue(np.shares_memory(got, out))
        np.testing.assert_array_equal(got.view(np.uint16), table.gather(ids).view(np.uint16))

    def test_gather_on_empty_input(self):
        """sglang's empty-input case: zero rows in, zero rows out, no crash."""
        params = _params()
        table = NGramTable(params, DIM)
        got = table.gather(np.empty((0, HEADS), np.int32))
        self.assertEqual(got.shape, (0, HEADS * DIM))
        self.assertEqual(got.size, 0)

    def test_gather_rejects_the_wrong_head_count(self):
        params = _params()
        with tempfile.TemporaryDirectory() as d:
            table = NGramTable.from_safetensors(_write_checkpoint(d, params), _Config())
            with self.assertRaises(ValueError):
                table.gather(np.zeros((4, HEADS + 1), np.int32))

    def test_missing_shard_is_an_error(self):
        params = _params()
        with tempfile.TemporaryDirectory() as d:
            files = _write_checkpoint(d, params)
            del files[f"{PREFIX}.ngram_embedding.shard_3.weight"]
            with self.assertRaises(KeyError):
                NGramTable.from_safetensors(files, _Config())


class TestHashVerification(CustomTestCase):
    """The checkpoint's own hash buffers vs our derivation.

    A drift here produces valid-but-wrong row ids. Nothing downstream errors,
    so this check is the only thing between a bad derivation and a model that
    quietly reads the wrong memory.
    """

    def test_matching_buffers_pass(self):
        params = _params()
        with tempfile.TemporaryDirectory() as d:
            NGramTable.from_safetensors(_write_checkpoint(d, params), _Config())

    def test_a_single_wrong_multiplier_is_caught(self):
        params = _params()
        with tempfile.TemporaryDirectory() as d:
            files = _write_checkpoint(d, params, corrupt="layer_multipliers")
            with self.assertRaises(ValueError) as cm:
                NGramTable.from_safetensors(files, _Config())
            self.assertIn("layer_multipliers", str(cm.exception))

    def test_a_single_wrong_prime_is_caught(self):
        params = _params()
        with tempfile.TemporaryDirectory() as d:
            files = _write_checkpoint(d, params, corrupt="ngram_heads_vocab_sizes")
            with self.assertRaises(ValueError):
                NGramTable.from_safetensors(files, _Config())

    def test_absent_buffers_fall_back_to_the_derivation(self):
        params = _params()
        with tempfile.TemporaryDirectory() as d:
            files = _write_checkpoint(d, params, drop_buffer=True)
            table = NGramTable.from_safetensors(files, _Config())
            np.testing.assert_array_equal(table.params.multipliers, params.multipliers)


class TestLocalCache(CustomTestCase):
    def test_save_then_restore_matches(self):
        params = _params()
        with tempfile.TemporaryDirectory() as d:
            table = NGramTable.from_safetensors(_write_checkpoint(d, params), _Config())
            cache = Path(d) / "cache"
            table.save_cache(cache)
            back = NGramTable.from_cache(cache)

            self.assertEqual(
                (back.rows, back.padded_rows, back.dim), (table.rows, table.padded_rows, table.dim)
            )
            np.testing.assert_array_equal(back.params.multipliers, table.params.multipliers)
            np.testing.assert_array_equal(back.params.sizes, table.params.sizes)
            np.testing.assert_array_equal(back.params.offsets, table.params.offsets)

            ids = np.random.default_rng(2).integers(0, table.rows, size=(32, HEADS))
            np.testing.assert_array_equal(
                back.gather(ids.astype(np.int32)).view(np.uint16),
                table.gather(ids.astype(np.int32)).view(np.uint16),
            )

    def test_truncated_cache_is_rejected(self):
        params = _params()
        with tempfile.TemporaryDirectory() as d:
            table = NGramTable.from_safetensors(_write_checkpoint(d, params), _Config())
            cache = Path(d) / "cache"
            table.save_cache(cache)
            path = cache / "ngram_table.bin"
            with open(path, "r+b") as f:
                f.truncate(path.stat().st_size - 2 * DIM)
            with self.assertRaises(ValueError):
                NGramTable.from_cache(cache)

    def test_stream_round_trip_without_touching_disk(self):
        """RAM <-> object store is the only path on a box with < 95 GiB free."""
        import io

        params = _params()
        with tempfile.TemporaryDirectory() as d:
            table = NGramTable.from_safetensors(_write_checkpoint(d, params), _Config())
            buf = io.BytesIO()
            written = table.write_to(buf)
            self.assertEqual(written, table.padded_rows * table.dim * 2)

            buf.seek(0)
            back = NGramTable.from_metadata(table.metadata())
            back.read_into(buf)
            np.testing.assert_array_equal(back.data, table.data)

    def test_short_stream_is_rejected(self):
        import io

        params = _params()
        with tempfile.TemporaryDirectory() as d:
            table = NGramTable.from_safetensors(_write_checkpoint(d, params), _Config())
            buf = io.BytesIO()
            table.write_to(buf)
            truncated = io.BytesIO(buf.getvalue()[:-64])
            with self.assertRaises(ValueError):
                NGramTable.from_metadata(table.metadata()).read_into(truncated)

    def test_version_mismatch_is_rejected(self):
        import json

        params = _params()
        with tempfile.TemporaryDirectory() as d:
            table = NGramTable.from_safetensors(_write_checkpoint(d, params), _Config())
            cache = Path(d) / "cache"
            table.save_cache(cache)
            meta_path = cache / "ngram_table.json"
            meta = json.loads(meta_path.read_text())
            meta["version"] += 1
            meta_path.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                NGramTable.from_cache(cache)


if __name__ == "__main__":
    unittest.main()
