"""Loading and gathering the PLE table.

Built against synthetic safetensors shards with the released checkpoint's
naming and split, so it runs on a CPU runner without the 95.4 GiB table.
"""

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
    def test_shards_tile_the_padded_table(self):
        for total, parts in ((1000, 8), (1024, 8), (3, 8), (320_001_536, 128)):
            with self.subTest(total=total, parts=parts):
                places, covered = shard_placements(total, parts), 0
                for placement in places:
                    self.assertEqual(placement.start, covered)
                    self.assertGreater(placement.rows, 0)
                    covered += placement.rows
                self.assertEqual(covered, total)
                if total == 320_001_536:  # released checkpoint, including 90 padding rows
                    self.assertEqual(len(places), 128)
                    self.assertTrue(all(p.rows == 2_500_012 for p in places))


class TestLoadAndGather(CustomTestCase):
    def test_checkpoint_and_cache_round_trip(self):
        params = _params()
        for drop_buffer in (False, True):
            with self.subTest(drop_buffer=drop_buffer), tempfile.TemporaryDirectory() as directory:
                files = _write_checkpoint(directory, params, drop_buffer=drop_buffer)
                table = NGramTable.from_safetensors(files, _Config())
                self.assertEqual((table.rows, table.dim), (params.total_vocab_size, DIM))
                self.assertEqual(table.padded_rows % 128, 0)
                cache = Path(directory) / "cache"
                table.save_cache(cache)
                restored = NGramTable.from_cache(cache)
                stream = io.BytesIO()
                self.assertEqual(table.write_to(stream), table.padded_rows * DIM * 2)
                stream.seek(0)
                streamed = NGramTable.from_metadata(table.metadata())
                streamed.read_into(stream)
                ids = np.random.default_rng(0).integers(
                    0, table.rows, size=(6, HEADS), dtype=np.int32
                )
                want = np.repeat((ids % 65535).astype(np.uint16)[:, :, None], DIM, axis=2)
                for candidate in (table, restored, streamed):
                    for field in ("multipliers", "sizes", "offsets"):
                        np.testing.assert_array_equal(
                            getattr(candidate.params, field), getattr(params, field)
                        )
                    got = candidate.gather(ids)
                    self.assertEqual(got.dtype, ml_dtypes.bfloat16)
                    np.testing.assert_array_equal(got.view(np.uint16).reshape(6, HEADS, DIM), want)

    def test_gather_shapes_dtypes_and_output_reuse(self):
        for dim in (7, 64, 257):
            table = NGramTable(_params(), dim)
            table.data[:] = np.random.default_rng(0).integers(
                0, 1 << 16, size=table.data.shape, dtype=np.uint16
            )
            for dtype in (np.int32, np.int64):
                for count in (0, 6):
                    with self.subTest(dim=dim, dtype=dtype, count=count):
                        ids = np.arange(count * HEADS, dtype=dtype).reshape(count, HEADS)
                        out = np.empty((count, HEADS, dim), np.uint16)
                        got = table.gather(ids, out=out)
                        self.assertEqual(got.shape, (count, HEADS * dim))
                        if count:
                            self.assertTrue(np.shares_memory(got, out))
                        np.testing.assert_array_equal(
                            got.view(np.uint16), table.data[ids].reshape(count, HEADS * dim)
                        )
            with self.assertRaises(ValueError):
                table.gather(np.zeros((4, HEADS + 1), np.int32))

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

    def test_checkpoint_and_cache_validation(self):
        params = _params()
        for bad in (
            "missing_shard",
            "layer_multipliers",
            "ngram_heads_vocab_sizes",
            "ngram_heads_offsets",
        ):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                files = _write_checkpoint(
                    directory, params, corrupt=None if bad == "missing_shard" else bad
                )
                if bad == "missing_shard":
                    del files[f"{PREFIX}.ngram_embedding.shard_3.weight"]
                with self.assertRaises((KeyError, ValueError)):
                    NGramTable.from_safetensors(files, _Config())
        for bad in ("truncated_cache", "short_stream", "version"):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                table = NGramTable.from_safetensors(_write_checkpoint(directory, params), _Config())
                cache = Path(directory) / "cache"
                table.save_cache(cache)
                if bad == "short_stream":
                    stream = io.BytesIO()
                    table.write_to(stream)
                    with self.assertRaises(ValueError):
                        NGramTable.from_metadata(table.metadata()).read_into(
                            io.BytesIO(stream.getvalue()[:-64])
                        )
                    continue
                if bad == "truncated_cache":
                    path = cache / "ngram_table.bin"
                    with path.open("r+b") as file:
                        file.truncate(path.stat().st_size - 2 * DIM)
                else:
                    path = cache / "ngram_table.json"
                    metadata = json.loads(path.read_text())
                    metadata["version"] += 1
                    path.write_text(json.dumps(metadata))
                with self.assertRaises(ValueError):
                    NGramTable.from_cache(cache)


class TestSchedulerHandoff(CustomTestCase):
    def _schedule(self, mode, ranks):
        from sgl_jax.srt.managers.schedule_batch import ScheduleBatch, ScheduleReqsInfo

        batch = ScheduleBatch.__new__(ScheduleBatch)
        batch.dp_size, batch.forward_mode, batch.spec_algorithm = len(ranks), mode, None
        batch.reqs_info = []
        for rank in ranks:
            batch.reqs_info.append(
                ScheduleReqsInfo(
                    reqs=[SimpleNamespace(origin_input_ids=p, output_ids=o) for p, o, _, _ in rank],
                    seq_lens=np.array([start + length for _, _, start, length in rank], np.int32),
                    prefix_lens=[start for _, _, start, _ in rank],
                )
            )
        return batch

    def _handoff(self, batch, ids, embeddings):
        import jax

        from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
        from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
        from sgl_jax.test.layers.test_ngram_embedding import _make_mesh

        mode = batch.forward_mode
        counts = [len(info.reqs) for info in batch.reqs_info]
        per_dp_bs = max(counts)
        seq_lens = np.concatenate(
            [
                np.pad(info.seq_lens, (0, per_dp_bs - count))
                for info, count in zip(batch.reqs_info, counts, strict=True)
            ]
        )
        prefixes = np.concatenate(
            [
                np.pad(np.asarray(info.prefix_lens, np.int32), (0, per_dp_bs - count))
                for info, count in zip(batch.reqs_info, counts, strict=True)
            ]
        )
        worker = ModelWorkerBatch(
            bid=0,
            forward_mode=mode,
            input_ids=ids,
            real_input_ids_len=len(ids),
            seq_lens=seq_lens,
            out_cache_loc=np.arange(len(ids), dtype=np.int32),
            req_pool_indices=np.arange(len(seq_lens), dtype=np.int32),
            sampling_info=None,
            positions=np.arange(len(ids), dtype=np.int32),
            cache_loc=None,
            return_logprob=False,
            return_output_logprob_only=False,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            extend_seq_lens=seq_lens - prefixes if mode.is_extend() else None,
            extend_prefix_lens=prefixes if mode.is_extend() else None,
            extend_logprob_start_lens=None,
            extend_input_logprob_token_ids=None,
            logits_indices=None,
            real_bs=sum(counts),
            real_bs_per_dp=counts,
            dp_size=batch.dp_size,
            per_dp_bs_size=per_dp_bs,
            ple_embeddings=embeddings,
        )
        runner = SimpleNamespace(
            mesh=_make_mesh(),
            attn_backend=None,
            model_config=SimpleNamespace(
                is_embedding=False, hf_config=SimpleNamespace(architectures=[])
            ),
        )
        forward = ForwardBatch.init_new(worker, runner)
        back = jax.jit(lambda fb: fb)(forward)  # exercises the actual pytree/JIT handoff
        if embeddings is None:
            self.assertIsNone(back.ple_embeddings)
        else:
            self.assertEqual(back.ple_embeddings.dtype, embeddings.dtype)
            np.testing.assert_array_equal(
                np.asarray(back.ple_embeddings).view(np.uint16), embeddings.view(np.uint16)
            )
            self.assertEqual(back.ple_embeddings.sharding.spec[0], "data")

    def test_dp_padding_context_and_forward_batch(self):
        from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
        from sgl_jax.test.layers.test_ngram_embedding import _ref_ngram_ids

        cases = (
            (
                ForwardMode.EXTEND,
                [
                    [
                        ([10, 11, EOS, 13, 14], [15, 16], 4, 2),
                        ([21, 22], [], 0, 2),
                        ([31, 32], [], 2, 0),
                    ],
                    [],  # an empty middle DP rank must still advance the output offset
                    [([41, 42], [43, 44, 45], 3, 2)],
                ],
            ),
            (
                ForwardMode.DECODE,
                [
                    [([10, 11, EOS, 13, 14], [15, 16], 6, 1), ([EOS, 22], [23], 2, 1)],
                    [],
                    [([41, 42], [43, 44, 45], 4, 1)],
                ],
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            table = NGramTable.from_safetensors(_write_checkpoint(directory, _params()), _Config())
            for mode, ranks in cases:
                with self.subTest(mode=mode):
                    per_dp = 5
                    ids = np.zeros(per_dp * len(ranks), np.int32)
                    expected_ids = np.zeros((len(ids), HEADS), np.int64)
                    for rank, requests in enumerate(ranks):
                        offset = rank * per_dp
                        for prompt, output, start, length in requests:
                            stream = prompt + output
                            ids[offset : offset + length] = stream[start : start + length]
                            whole = _ref_ngram_ids(
                                stream, [0, len(stream)], [[EOS] * (NGRAM_SIZE - 1)], table.params
                            )
                            expected_ids[offset : offset + length] = whole[start : start + length]
                            offset += length
                    batch = self._schedule(mode, ranks)
                    with patch(
                        "sgl_jax.srt.managers.schedule_batch.get_ngram_table", return_value=table
                    ):
                        embeddings = batch._merge_ngram_ple(per_dp, len(ids), ids)
                    want = table.data[expected_ids].reshape(len(ids), PLE_EMBED_DIM)
                    np.testing.assert_array_equal(embeddings.view(np.uint16), want)
                    self._handoff(batch, ids, embeddings)

    def test_absent_table_and_unsupported_scheduling(self):
        from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

        batch = self._schedule(ForwardMode.DECODE, [[([10, 11], [12], 2, 1)]])
        ids = np.array([12], np.int32)
        with patch("sgl_jax.srt.managers.schedule_batch.get_ngram_table", return_value=None):
            self.assertIsNone(batch._merge_ngram_ple(1, 1, ids))
        self._handoff(batch, ids, None)
        table = NGramTable(_params(), DIM)
        with patch("sgl_jax.srt.managers.schedule_batch.get_ngram_table", return_value=table):
            with self.assertRaisesRegex(RuntimeError, "future tokens"):
                batch._merge_ngram_ple(1, 1, np.array([-1], np.int32))
            batch.spec_algorithm = SimpleNamespace(is_none=lambda: False)
            with self.assertRaisesRegex(NotImplementedError, "speculative"):
                batch._merge_ngram_ple(1, 1, ids)
            batch.spec_algorithm = None
            batch.reqs_info[0].seq_lens[0] = 5
            with self.assertRaisesRegex(ValueError, "stale"):
                batch._merge_ngram_ple(1, 1, ids)


if __name__ == "__main__":
    unittest.main()
