import unittest

from benchmark.moe.fused_rs_preop_contract import (
    DEFAULT_FINAL_REL_L2_THRESHOLD,
    evaluate_preop_variant_contract,
)


class FusedRsPreopContractTest(unittest.TestCase):
    def test_accepts_expected_low_precision_drift_when_gather_is_exact(self):
        result = evaluate_preop_variant_contract(
            hidden_gather_all_finite=True,
            hidden_gather_max_abs=0.0,
            full_all_finite=True,
            full_rel_l2=0.0032,
            padded_all_finite=True,
            padded_rel_l2=0.0021,
            invalid_padding_max_abs=0.0,
        )

        self.assertEqual(DEFAULT_FINAL_REL_L2_THRESHOLD, 0.01)
        self.assertTrue(result["hidden_gather_exact"])
        self.assertTrue(result["final_output_ok"])
        self.assertTrue(result["contract_ok"])

    def test_rejects_non_exact_hidden_gather_even_if_final_output_is_close(self):
        result = evaluate_preop_variant_contract(
            hidden_gather_all_finite=True,
            hidden_gather_max_abs=1.0 / 128.0,
            full_all_finite=True,
            full_rel_l2=0.0032,
            padded_all_finite=True,
            padded_rel_l2=0.0021,
            invalid_padding_max_abs=0.0,
        )

        self.assertFalse(result["hidden_gather_exact"])
        self.assertFalse(result["contract_ok"])

    def test_rejects_padding_or_final_output_contract_violation(self):
        result = evaluate_preop_variant_contract(
            hidden_gather_all_finite=True,
            hidden_gather_max_abs=0.0,
            full_all_finite=True,
            full_rel_l2=DEFAULT_FINAL_REL_L2_THRESHOLD + 1e-4,
            padded_all_finite=True,
            padded_rel_l2=0.0,
            invalid_padding_max_abs=1e-7,
        )

        self.assertTrue(result["hidden_gather_exact"])
        self.assertFalse(result["final_output_ok"])
        self.assertFalse(result["contract_ok"])


if __name__ == "__main__":
    unittest.main()
