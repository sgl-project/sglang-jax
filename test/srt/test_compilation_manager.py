import unittest
from types import SimpleNamespace

from sgl_jax.srt.model_executor.compilation_manager import CompilationManager
from sgl_jax.test.test_utils import CustomTestCase


def _make_cm(moe_backend: str, tp_size: int, bs_paddings=None, max_bs=64):
    sa = SimpleNamespace(
        moe_backend=moe_backend,
        enable_static_lora=False,
        precompile_token_paddings=[64, 256],
        precompile_bs_paddings=bs_paddings,
    )
    return CompilationManager(
        sa,
        max_padded_batch_size=max_bs,
        max_padded_num_tokens=2048,
        dp_size=1,
        tp_size=tp_size,
        page_size=128,
        max_req_len=2048,
        vocab_size=100,
    )


class TestComputeBsBuckets(CustomTestCase):
    """fused_ep_moe (used by epmoe/fused/fused_v2 backends) requires
    num_tokens % (ep_size * t_packing) == 0, where ep_size = dp*tp and
    t_packing=2 for bf16. Decode num_tokens == bs, so bs_buckets must be
    >= tp_size*2 for these backends."""

    def test_epmoe_filters_small_bs(self):
        cm = _make_cm("epmoe", tp_size=16)
        self.assertTrue(all(bs >= 32 for bs in cm.bs_buckets), cm.bs_buckets)
        self.assertIn(32, cm.bs_buckets)
        self.assertIn(64, cm.bs_buckets)

    def test_epmoe_user_paddings_also_filtered(self):
        cm = _make_cm("epmoe", tp_size=16, bs_paddings=[1, 8, 16, 32, 64])
        self.assertEqual(cm.bs_buckets, [32, 64])

    def test_fused_v2_still_filtered(self):
        cm = _make_cm("fused_v2", tp_size=16)
        self.assertTrue(all(bs >= 32 for bs in cm.bs_buckets), cm.bs_buckets)

    def test_auto_backend_unchanged(self):
        cm = _make_cm("auto", tp_size=16)
        self.assertIn(1, cm.bs_buckets)


if __name__ == "__main__":
    unittest.main()
