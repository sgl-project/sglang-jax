import unittest

from benchmark.moe.fused_rs_tuning import (
    GLM52_RS_REFERENCE_CONFIG,
    analyze_rs_config,
    generate_rs_tuning_configs,
)


class FusedRsTuningTest(unittest.TestCase):
    def test_candidates_follow_author_weight_cache_contract(self):
        configs = generate_rs_tuning_configs((128, 256, 384))

        self.assertEqual(len(configs), 12)
        self.assertEqual(len(set(configs)), len(configs))
        self.assertIn(GLM52_RS_REFERENCE_CONFIG, configs)
        self.assertIn((384, 6144, 1024, 2048, 3072, 2, 2), configs)
        self.assertNotIn((384, 6144, 1024, 2048, 3072, 1, 1), configs)

        for config in configs:
            contract = analyze_rs_config(config)
            self.assertTrue(contract["buffer_contract_valid"])
            self.assertGreaterEqual(contract["num_w1_bufs"], contract["w1_steps"])
            self.assertGreaterEqual(contract["num_w2_bufs"], contract["w2_steps"])

    def test_contract_rejects_multi_step_single_buffer_config(self):
        contract = analyze_rs_config((384, 6144, 1024, 2048, 3072, 1, 1))

        self.assertEqual(
            contract,
            {
                "tile_m": 384,
                "tile_k1": 6144,
                "tile_n1": 1024,
                "tile_k2": 2048,
                "tile_n2": 3072,
                "num_w1_bufs": 1,
                "num_w2_bufs": 1,
                "w1_steps": 2,
                "w2_steps": 2,
                "can_cache_w1": False,
                "can_cache_w2": False,
                "buffer_contract_valid": False,
            },
        )

    def test_candidates_include_independent_split_n_probes(self):
        configs = set(generate_rs_tuning_configs((256,)))

        self.assertEqual(
            configs,
            {
                (256, 6144, 2048, 2048, 6144, 1, 1),
                (256, 6144, 1024, 2048, 6144, 2, 1),
                (256, 6144, 2048, 2048, 3072, 1, 2),
                (256, 6144, 1024, 2048, 3072, 2, 2),
            },
        )


if __name__ == "__main__":
    unittest.main()
