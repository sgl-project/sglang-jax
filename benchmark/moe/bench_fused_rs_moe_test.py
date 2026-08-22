import unittest

from benchmark.moe.fused_rs_tuning import (
    GLM52_RS_REFERENCE_CONFIG,
    GLM52_RS_VMEM_LIMIT_BYTES,
    analyze_rs_config,
    generate_rs_tuning_configs,
)


class FusedRsTuningTest(unittest.TestCase):
    def test_candidates_follow_full_k_pipeline_and_vmem_contract(self):
        configs = generate_rs_tuning_configs((128, 256, 384))

        self.assertEqual(len(configs), 4)
        self.assertEqual(len(set(configs)), len(configs))
        self.assertIn(GLM52_RS_REFERENCE_CONFIG, configs)
        self.assertTrue(all(config[0] == 128 for config in configs))

        for config in configs:
            contract = analyze_rs_config(config)
            self.assertTrue(contract["full_k"])
            self.assertTrue(contract["pipeline_contract_valid"])
            self.assertTrue(contract["padding_contract_valid"])
            self.assertTrue(contract["vmem_contract_valid"])
            self.assertTrue(contract["eligible_for_tuning"])
            self.assertLessEqual(
                contract["estimated_vmem_with_headroom_bytes"],
                GLM52_RS_VMEM_LIMIT_BYTES,
            )

    def test_contract_rejects_multi_step_single_buffer_config(self):
        contract = analyze_rs_config((384, 6144, 1024, 2048, 3072, 1, 1))

        self.assertFalse(contract["pipeline_contract_valid"])
        self.assertFalse(contract["eligible_for_tuning"])
        self.assertEqual(contract["w1_buffer_mode"], "invalid")
        self.assertEqual(contract["w2_buffer_mode"], "invalid")

    def test_streaming_weight_candidates_are_outside_padding_contract(self):
        for config in (
            (256, 6144, 1024, 2048, 1024, 2, 2),
            (256, 6144, 512, 2048, 6144, 2, 1),
            (256, 6144, 512, 2048, 2048, 2, 2),
            (384, 6144, 512, 2048, 1024, 2, 2),
        ):
            contract = analyze_rs_config(config)
            self.assertTrue(contract["pipeline_contract_valid"])
            self.assertFalse(contract["padding_contract_valid"])
            self.assertFalse(contract["eligible_for_tuning"])

        with self.assertRaisesRegex(ValueError, "no full-K fused-RS candidates"):
            generate_rs_tuning_configs((256, 384))

    def test_full_resident_m256_is_pruned_by_declared_vmem(self):
        contract = analyze_rs_config((256, 6144, 256, 2048, 256, 8, 24))

        self.assertTrue(contract["pipeline_contract_valid"])
        self.assertTrue(contract["can_cache_w1"])
        self.assertTrue(contract["can_cache_w2"])
        self.assertTrue(contract["padding_contract_valid"])
        self.assertFalse(contract["vmem_contract_valid"])
        self.assertFalse(contract["eligible_for_tuning"])

    def test_rejects_k_splitting_even_when_shapes_divide(self):
        contract = analyze_rs_config((128, 3072, 1024, 1024, 3072, 4, 4))

        self.assertFalse(contract["full_k"])
        self.assertFalse(contract["eligible_for_tuning"])


if __name__ == "__main__":
    unittest.main()
