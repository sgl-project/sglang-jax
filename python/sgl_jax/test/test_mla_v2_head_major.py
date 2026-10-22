"""head-major q/o contract of the MLA v2 kernel must be bit-exact
with the packed token-major contract (CPU emulation of the Pallas kernel,
same harness as test_mla_v2_bitexact.py)."""

import unittest

import jax.numpy as jnp
import numpy as np

from sgl_jax.test.test_mla_v2_bitexact import (
    _cpu_execution_context,
    extract_valid_kv_tokens,
    make_mla_v2_parity_inputs,
    pre1733_reference_mla_v2,
    run_v2_kernel_with_capture,
)

# bf16 packing 2 -> head-major query block must be a multiple of 16 rows.
_BQ = 16


class TestMLAV2HeadMajorBitExact(unittest.TestCase):
    # (name, lens[(q, kv)], page_size, num_heads, extra_kw, bq); sum(q) % 16 == 0
    CASES = [
        ("prefill_aligned_multi_block", [(16, 32), (32, 48), (16, 16)], 16, 8, {}, 16),
        ("prefill_unaligned_shared_tiles", [(5, 21), (11, 30), (7, 40), (9, 17)], 16, 8, {}, 16),
        ("prefill_extend_q1_tails", [(8, 8), (5, 21), (1, 19), (18, 40)], 16, 8, {}, 16),
        (
            "prefill_sliding_window",
            [(12, 32), (8, 25), (28, 60)],
            16,
            8,
            {"sliding_window": 16},
            16,
        ),
        (
            "prefill_soft_cap_scales",
            [(8, 24), (6, 18), (18, 30)],
            16,
            8,
            {"soft_cap": 30.0, "q_scale": 0.125},
            16,
        ),
        ("prefill_2heads_bq32", [(16, 32), (9, 17), (39, 50)], 16, 2, {}, 32),
        ("prefill_bq32_cross_blocks", [(45, 70), (3, 9), (16, 33)], 16, 8, {}, 32),
    ]

    def _run_pair(self, lens, page_size, num_heads, extra_kw, bq=_BQ):
        with _cpu_execution_context():
            inputs = make_mla_v2_parity_inputs(
                lens, page_size=page_size, num_heads=num_heads, dtype=jnp.bfloat16, seed=42
            )
            common_kw = dict(
                sm_scale=(512 + 64) ** -0.5,
                num_kv_pages_per_block=(1, 1, 1),
                num_queries_per_block=(1, 1, bq),
                decode_batch_size=1,
                **extra_kw,
            )
            out_tm, kv_tm, l_tm, acc_tm = run_v2_kernel_with_capture(*inputs, **common_kw)
            out_hm, kv_hm, l_hm, acc_hm = run_v2_kernel_with_capture(
                *inputs, skip_decode_cases=True, q_head_major=True, **common_kw
            )
            ref_kw = {k: v for k, v in common_kw.items() if k != "mixed_static_q_len"}
            out_ref, _, _, _ = pre1733_reference_mla_v2(*inputs, **ref_kw)
            vkv_tm = extract_valid_kv_tokens(
                kv_tm, lens, page_size, inputs[8], inputs[6], jnp.bfloat16
            )
            vkv_hm = extract_valid_kv_tokens(
                kv_hm, lens, page_size, inputs[8], inputs[6], jnp.bfloat16
            )
        return (out_tm, vkv_tm, l_tm, acc_tm), (out_hm, vkv_hm, l_hm, acc_hm), out_ref

    def test_head_major_bitexact_vs_token_major(self):
        for name, lens, page_size, num_heads, extra_kw, bq in self.CASES:
            with self.subTest(name=name):
                self.assertEqual(sum(q for q, _ in lens) % 16, 0, name)
                (out_tm, kv_tm, l_tm, acc_tm), (out_hm, kv_hm, l_hm, acc_hm), out_ref = (
                    self._run_pair(lens, page_size, num_heads, extra_kw, bq)
                )
                self.assertEqual(out_tm.shape, out_hm.shape)
                np.testing.assert_array_equal(out_hm, out_tm, err_msg=f"{name}: output")
                np.testing.assert_array_equal(kv_hm, kv_tm, err_msg=f"{name}: updated kv")
                np.testing.assert_array_equal(out_hm, out_ref, err_msg=f"{name}: vs reference")
                # (l / acc snapshots are not row-comparable: the head-major grid
                # carries junk rows outside [q_start, q_end); the written output is.)

    def test_head_major_requires_skip_decode_cases(self):
        with _cpu_execution_context():
            inputs = make_mla_v2_parity_inputs(
                [(16, 32)], page_size=16, num_heads=8, dtype=jnp.bfloat16, seed=1
            )
            with self.assertRaises(ValueError):
                run_v2_kernel_with_capture(
                    *inputs,
                    sm_scale=1.0,
                    num_kv_pages_per_block=(1, 1, 1),
                    num_queries_per_block=(1, 1, _BQ),
                    q_head_major=True,
                )

    def test_head_major_rejects_subtile_block(self):
        with _cpu_execution_context():
            inputs = make_mla_v2_parity_inputs(
                [(16, 32)], page_size=16, num_heads=8, dtype=jnp.bfloat16, seed=1
            )
            with self.assertRaises(ValueError):
                run_v2_kernel_with_capture(
                    *inputs,
                    sm_scale=1.0,
                    num_kv_pages_per_block=(1, 1, 1),
                    num_queries_per_block=(1, 1, 8),
                    skip_decode_cases=True,
                    q_head_major=True,
                )


if __name__ == "__main__":
    unittest.main()
