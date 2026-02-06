import os
import sys
import unittest

import numpy as np

# Add python directory to sys.path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))

from sgl_jax.srt.eplb.eplb_algorithms import EplbAlgorithm, deepseek, rebalance_experts


class TestEPLB(unittest.TestCase):
    def test_deepseek_rebalance_basic(self):
        num_layers = 2
        num_logical_experts = 4
        num_replicas = 4
        num_gpus = 2
        num_nodes = 1
        num_groups = 1  # Not used for non-hierarchical effectively

        weight = np.random.rand(num_layers, num_logical_experts)

        # Test direct call to deepseek implementation (non-hierarchical)
        phy2log, log2phy, logcnt = deepseek.rebalance_experts(
            weight,
            num_replicas=num_replicas,
            num_groups=num_groups,
            num_nodes=num_nodes,
            num_gpus=num_gpus,
            enable_hierarchical=False,
        )

        self.assertEqual(phy2log.shape, (num_layers, num_replicas))
        self.assertEqual(logcnt.shape, (num_layers, num_logical_experts))
        # log2phy shape is dynamic on last dim, but here maxlogcnt should be >= 1
        self.assertEqual(log2phy.shape[0], num_layers)
        self.assertEqual(log2phy.shape[1], num_logical_experts)

        self.assertTrue(isinstance(phy2log, np.ndarray))
        self.assertTrue(isinstance(log2phy, np.ndarray))
        self.assertTrue(isinstance(logcnt, np.ndarray))

    def test_deepseek_rebalance_hierarchical(self):
        num_layers = 2
        num_logical_experts = 8
        num_replicas = 8
        num_gpus = 4
        num_nodes = 2
        num_groups = 4

        # specific constraints for hierarchical:
        # num_logical_experts % num_groups == 0 (8 % 4 == 0)
        # num_groups % num_nodes == 0 (4 % 2 == 0)
        # num_gpus % num_nodes == 0 (4 % 2 == 0)
        # num_physical_experts % num_gpus == 0 (8 % 4 == 0)

        weight = np.random.rand(num_layers, num_logical_experts)

        phy2log, log2phy, logcnt = deepseek.rebalance_experts(
            weight,
            num_replicas=num_replicas,
            num_groups=num_groups,
            num_nodes=num_nodes,
            num_gpus=num_gpus,
            enable_hierarchical=True,
        )

        self.assertEqual(phy2log.shape, (num_layers, num_replicas))
        self.assertEqual(logcnt.shape, (num_layers, num_logical_experts))

    def test_rebalance_experts_wrapper(self):
        # Test the wrapper in __init__.py
        num_layers = 2
        num_logical_experts = 4
        num_physical_experts = 4
        num_local_physical_experts = 2  # implies 2 GPUs if num_physical_experts=4?
        # In wrapper: num_gpus = num_physical_experts // num_local_physical_experts
        # so 4 // 2 = 2 GPUs.

        num_nodes = 1
        num_groups = 1

        tokens_per_expert = np.random.rand(num_layers, num_logical_experts)

        algorithm = EplbAlgorithm.deepseek

        res = rebalance_experts(
            tokens_per_expert=tokens_per_expert,
            num_physical_experts=num_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
            num_groups=num_groups,
            num_nodes=num_nodes,
            algorithm=algorithm,
        )

        # deepseek returns (phy2log, log2phy, logcnt)
        self.assertEqual(len(res), 3)


if __name__ == "__main__":
    unittest.main()
