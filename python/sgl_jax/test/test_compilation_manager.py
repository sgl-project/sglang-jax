import unittest
from unittest.mock import MagicMock

from sgl_jax.srt.model_executor.compilation_manager import CompilationManager
from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sgl_jax.srt.utils.common_utils import (
    SAFE_EXTEND_PER_DP_BS,
    pad_to_bucket,
    projected_per_dp_bucket,
    selected_extend_per_dp_bs,
)


def _make_server_args(**overrides):
    args = MagicMock()
    args.precompile_token_paddings = None
    args.precompile_bs_paddings = None
    args.moe_backend = "none"
    args.enable_static_lora = False
    args.multimodal = False
    args.nnodes = 1
    args.speculative_algorithm = None
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
        assert batch.real_bs_per_dp == [16, 16, 16, 16]

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


def _make_recurrent_cm(**overrides):
    """CompilationManager on the validated affected config (tp16/dp4/fused/page1
    recurrent multi-host) unless overridden."""
    kwargs = dict(
        max_padded_batch_size=64,
        max_padded_num_tokens=2048,
        dp_size=4,
        tp_size=16,
        page_size=1,
        max_req_len=4096,
        vocab_size=32000,
        has_recurrent_state=True,
    )
    sa_overrides = {}
    for k in ("nnodes", "speculative_algorithm", "moe_backend", "precompile_bs_paddings"):
        if k in overrides:
            sa_overrides[k] = overrides.pop(k)
    kwargs.update(overrides)
    sa_overrides.setdefault("nnodes", 4)
    sa_overrides.setdefault("moe_backend", "fused")
    return CompilationManager(server_args=_make_server_args(**sa_overrides), **kwargs)


class TestRecurrentExtendSafeBucket(unittest.TestCase):
    """Affected recurrent multi-host EXTEND must always have a runtime-reachable
    bucket with per_dp_bs <= SAFE_EXTEND_PER_DP_BS."""

    def test_safe_bucket_inserted_when_user_paddings_only_large(self):
        cm = _make_recurrent_cm(precompile_bs_paddings=[64])
        safe_total = SAFE_EXTEND_PER_DP_BS * 4
        self.assertIn(safe_total, cm.bs_buckets)
        self.assertTrue(any(b // 4 <= SAFE_EXTEND_PER_DP_BS for b in cm.bs_buckets))

    def test_no_insert_when_max_already_safe(self):
        # max_padded_batch_size == SAFE*dp_size -> largest bucket already safe.
        cm = _make_recurrent_cm(max_padded_batch_size=SAFE_EXTEND_PER_DP_BS * 4)
        self.assertTrue(all(b // 4 <= SAFE_EXTEND_PER_DP_BS for b in cm.bs_buckets))

    def test_infeasible_safe_bucket_raises(self):
        # fused-moe tp_size*2 = 64 > SAFE*dp_size = 32 -> no feasible safe bucket.
        with self.assertRaises(ValueError):
            _make_recurrent_cm(tp_size=32, max_padded_batch_size=128, precompile_bs_paddings=[128])

    def test_single_host_not_affected_no_safe_bucket(self):
        cm = _make_recurrent_cm(nnodes=1, precompile_bs_paddings=[64])
        self.assertNotIn(SAFE_EXTEND_PER_DP_BS * 4, cm.bs_buckets)


class TestExtendPrecompileBsPairs(unittest.TestCase):
    def test_affected_filters_unsafe_buckets(self):
        cm = _make_recurrent_cm(precompile_bs_paddings=[32, 64])
        pairs = cm._extend_precompile_bs_pairs()
        self.assertTrue(all(bs // 4 <= SAFE_EXTEND_PER_DP_BS for _, bs in pairs))
        self.assertNotIn(64, [bs for _, bs in pairs])
        # the matching cache_loc bucket index is preserved
        for i, bs in pairs:
            self.assertEqual(cm.bs_buckets[i], bs)

    def test_not_affected_uses_largest_only(self):
        cm = _make_recurrent_cm(nnodes=1, precompile_bs_paddings=[32, 64])
        pairs = cm._extend_precompile_bs_pairs()
        self.assertEqual(pairs, [(len(cm.bs_buckets) - 1, cm.max_padded_batch_size)])


class TestProjectedPerDpBucket(unittest.TestCase):
    def test_small_active_maps_to_smallest_bucket(self):
        # smallest bucket 32 -> per_dp 8 even for a single active req
        self.assertEqual(projected_per_dp_bucket(1, 4, [32, 64]), 8)
        self.assertEqual(projected_per_dp_bucket(8, 4, [32, 64]), 8)

    def test_over_safe_maps_to_next_bucket(self):
        self.assertEqual(projected_per_dp_bucket(9, 4, [32, 64]), 16)

    def test_beyond_largest_bucket_does_not_raise(self):
        self.assertEqual(projected_per_dp_bucket(20, 4, [32, 64]), 20)

    def test_fine_grained_buckets(self):
        self.assertEqual(projected_per_dp_bucket(1, 1, [4, 8, 16]), 4)
        self.assertEqual(projected_per_dp_bucket(5, 1, [4, 8, 16]), 8)


class TestSelectedExtendPerDpBs(unittest.TestCase):
    """Global selected bucket is keyed to the max active count across dp ranks."""

    def test_keyed_to_global_max_not_single_rank(self):
        # dp1 has 16 (e.g. running decode that would mix in), dp0 has 1: the
        # selected bucket must reflect the global max (16), not dp0's 1.
        self.assertEqual(selected_extend_per_dp_bs([1, 16, 3, 2], 4, [32, 64, 128]), 16)

    def test_all_within_safe(self):
        self.assertEqual(selected_extend_per_dp_bs([8, 8, 8, 8], 4, [32, 64]), 8)
        self.assertEqual(selected_extend_per_dp_bs([1, 0, 0, 0], 4, [32, 64]), 8)

    def test_empty(self):
        self.assertEqual(selected_extend_per_dp_bs([], 4, [32, 64]), 8)


if __name__ == "__main__":
    unittest.main()
