import unittest

from benchmark.moe.fused_rs_fp8_hidden_ag_contract import (
    DEFAULT_REL_L2_THRESHOLD,
    evaluate_fp8_hidden_ag_contract,
)


class FusedRsFp8HiddenAgContractTest(unittest.TestCase):
    def test_accepts_finite_close_outputs_and_exact_invalid_padding(self):
        result = evaluate_fp8_hidden_ag_contract(
            full_all_finite=True,
            full_rel_l2=0.006,
            padded_all_finite=True,
            padded_rel_l2=0.007,
            padding_invariance_rel_l2=0.004,
            invalid_padding_max_abs=0.0,
        )

        self.assertEqual(DEFAULT_REL_L2_THRESHOLD, 0.01)
        self.assertTrue(result["contract_ok"])

    def test_rejects_accuracy_or_padding_violation(self):
        result = evaluate_fp8_hidden_ag_contract(
            full_all_finite=True,
            full_rel_l2=DEFAULT_REL_L2_THRESHOLD + 1e-4,
            padded_all_finite=True,
            padded_rel_l2=0.0,
            padding_invariance_rel_l2=0.0,
            invalid_padding_max_abs=1e-7,
        )

        self.assertFalse(result["contract_ok"])
        self.assertFalse(result["full_output_ok"])
        self.assertFalse(result["padding_invariance_ok"])


if __name__ == "__main__":
    unittest.main()
