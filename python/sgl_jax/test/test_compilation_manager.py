import unittest
from unittest.mock import MagicMock

from sgl_jax.srt.model_executor.compilation_manager import CompilationManager
from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sgl_jax.srt.utils.common_utils import pad_to_bucket


def _make_server_args(**overrides):
    args = MagicMock()
    args.precompile_token_paddings = None
    args.precompile_bs_paddings = None
    args.moe_backend = "none"
    args.enable_static_lora = False
    args.multimodal = False
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


class TestBucketComputation(unittest.TestCase):
    def test_token_buckets_default(self):
        cm = CompilationManager(
            server_args=_make_server_args(),
            max_padded_batch_size=128,
            max_padded_num_tokens=2048,
            dp_size=1,
            tp_size=4,
            page_size=128,
            max_req_len=4096,
            vocab_size=32000,
        )
        for b in cm.token_buckets:
            assert b >= 128, f"bucket {b} < max_padded_batch_size 128"
            assert b <= 2048, f"bucket {b} > max_padded_num_tokens 2048"
        assert cm.token_buckets[-1] == 2048

    def test_token_buckets_dp_size_filter(self):
        cm = CompilationManager(
            server_args=_make_server_args(),
            max_padded_batch_size=64,
            max_padded_num_tokens=2048,
            dp_size=4,
            tp_size=4,
            page_size=128,
            max_req_len=4096,
            vocab_size=32000,
        )
        for b in cm.token_buckets:
            assert b % 4 == 0, f"bucket {b} not divisible by dp_size=4"

    def test_bs_buckets_fused_moe_minimum(self):
        cm = CompilationManager(
            server_args=_make_server_args(moe_backend="fused"),
            max_padded_batch_size=128,
            max_padded_num_tokens=2048,
            dp_size=1,
            tp_size=4,
            page_size=128,
            max_req_len=4096,
            vocab_size=32000,
        )
        for b in cm.bs_buckets:
            assert b >= 8, f"bucket {b} < tp_size*2=8 for fused moe"

    def test_bs_buckets_max_included(self):
        cm = CompilationManager(
            server_args=_make_server_args(),
            max_padded_batch_size=200,
            max_padded_num_tokens=2048,
            dp_size=1,
            tp_size=4,
            page_size=128,
            max_req_len=4096,
            vocab_size=32000,
        )
        assert cm.bs_buckets[-1] == 200

    def test_cache_loc_buckets_length_matches_bs(self):
        cm = CompilationManager(
            server_args=_make_server_args(),
            max_padded_batch_size=128,
            max_padded_num_tokens=2048,
            dp_size=1,
            tp_size=4,
            page_size=128,
            max_req_len=4096,
            vocab_size=32000,
        )
        assert len(cm.cache_loc_buckets) == len(cm.bs_buckets)

    def test_pad_to_bucket(self):
        buckets = [64, 128, 256, 512]
        assert pad_to_bucket(1, buckets) == (64, 0)
        assert pad_to_bucket(64, buckets) == (64, 0)
        assert pad_to_bucket(65, buckets) == (128, 1)
        assert pad_to_bucket(500, buckets) == (512, 3)
        with self.assertRaises(ValueError):
            pad_to_bucket(999, buckets)

    def test_user_specified_paddings(self):
        cm = CompilationManager(
            server_args=_make_server_args(
                precompile_token_paddings=[256, 512, 1024],
                precompile_bs_paddings=[4, 16, 64],
            ),
            max_padded_batch_size=128,
            max_padded_num_tokens=2048,
            dp_size=1,
            tp_size=4,
            page_size=128,
            max_req_len=4096,
            vocab_size=32000,
        )
        assert 256 in cm.token_buckets
        assert 512 in cm.token_buckets


class TestLazyCompilation(unittest.TestCase):
    def test_register_variant_if_new_first_time(self):
        cm = CompilationManager(
            server_args=_make_server_args(),
            max_padded_batch_size=128,
            max_padded_num_tokens=2048,
            dp_size=1,
            tp_size=4,
            page_size=128,
            max_req_len=4096,
            vocab_size=32000,
        )
        key = ("DECODE", 128, 128, True)
        assert cm.register_variant_if_new(key) is True
        assert cm.register_variant_if_new(key) is False

    def test_different_variants_are_independent(self):
        cm = CompilationManager(
            server_args=_make_server_args(),
            max_padded_batch_size=128,
            max_padded_num_tokens=2048,
            dp_size=1,
            tp_size=4,
            page_size=128,
            max_req_len=4096,
            vocab_size=32000,
        )
        key1 = ("DECODE", 128, 128, False)
        key2 = ("DECODE", 128, 128, True)
        assert cm.register_variant_if_new(key1) is True
        assert cm.register_variant_if_new(key2) is True
        assert cm.register_variant_if_new(key1) is False


class TestDummyBatch(unittest.TestCase):
    """Verify _make_dummy_batch produces correct shapes and metadata."""

    def setUp(self):
        self.cm = CompilationManager(
            server_args=_make_server_args(),
            max_padded_batch_size=128,
            max_padded_num_tokens=2048,
            dp_size=1,
            tp_size=4,
            page_size=128,
            max_req_len=4096,
            vocab_size=32000,
        )

    def test_extend_batch_shapes(self):
        bs, num_tokens = 32, 256
        cache_loc_size = 1024
        batch = self.cm._make_dummy_batch(bs, num_tokens, ForwardMode.EXTEND, cache_loc_size)
        assert batch.forward_mode == ForwardMode.EXTEND
        assert batch.real_bs == bs
        assert batch.real_input_ids_len == bs
        assert batch.input_ids.shape == (num_tokens,)
        assert batch.out_cache_loc.shape == (num_tokens,)
        assert batch.positions.shape == (num_tokens,)
        assert batch.cache_loc.shape == (cache_loc_size,)
        assert batch.req_pool_indices.shape == (bs,)
        assert batch.seq_lens.shape == (bs,)
        assert batch.extend_seq_lens is not None
        assert batch.extend_seq_lens.shape == (bs,)
        assert batch.extend_prefix_lens.shape == (bs,)
        assert batch.logits_indices.shape == (bs,)
        assert batch.capture_hidden_mode == CaptureHiddenMode.NULL

    def test_decode_batch_shapes(self):
        bs = 64
        cache_loc_size = 2048
        batch = self.cm._make_dummy_batch(bs, bs, ForwardMode.DECODE, cache_loc_size)
        assert batch.forward_mode == ForwardMode.DECODE
        assert batch.real_bs == bs
        assert batch.input_ids.shape == (bs,)
        assert batch.out_cache_loc.shape == (bs,)
        assert batch.positions.shape == (bs,)
        assert batch.cache_loc.shape == (cache_loc_size,)
        assert batch.extend_seq_lens is None
        assert batch.extend_prefix_lens is None
        assert batch.logits_indices is None

    def test_dp_metadata(self):
        cm = CompilationManager(
            server_args=_make_server_args(),
            max_padded_batch_size=64,
            max_padded_num_tokens=1024,
            dp_size=4,
            tp_size=4,
            page_size=128,
            max_req_len=4096,
            vocab_size=32000,
        )
        bs = 64
        batch = cm._make_dummy_batch(
            bs,
            bs,
            ForwardMode.DECODE,
            2048,
            dp_size=4,
            per_dp_bs_size=16,
        )
        assert batch.dp_size == 4
        assert batch.per_dp_bs_size == 16
        assert batch.real_bs_per_dp == [64, 64, 64, 64]

    def test_multimodal_capture_hidden(self):
        cm = CompilationManager(
            server_args=_make_server_args(),
            max_padded_batch_size=32,
            max_padded_num_tokens=512,
            dp_size=1,
            tp_size=4,
            page_size=128,
            max_req_len=4096,
            vocab_size=32000,
            multimodal=True,
        )
        batch = cm._make_dummy_batch(32, 128, ForwardMode.EXTEND, 512)
        assert batch.capture_hidden_mode == CaptureHiddenMode.FULL

    def test_invalid_cache_loc_raises(self):
        with self.assertRaises(ValueError):
            self.cm._make_dummy_batch(64, 128, ForwardMode.EXTEND, 32)


if __name__ == "__main__":
    unittest.main()
