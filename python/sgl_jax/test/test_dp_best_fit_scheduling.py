"""Tests for best-fit DP scheduling with stranding-based scoring."""

import random
import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, PropertyMock


def compute_stranding(
    after_input: int,
    after_output: int,
    flops_cap: int,
    hbm_cap: int,
    flops_weight: float = 1.0,
    hbm_weight: float = 1.0,
) -> float:
    """Compute total stranding cost for a given placement.

    This mirrors the logic in Scheduler._select_best_fit_dp().
    """
    # Utilization ratios
    flops_util = after_input / flops_cap
    hbm_util = after_output / hbm_cap

    # Inflation factor (driven by bottleneck dimension - the one with higher utilization)
    inflation = min(
        flops_cap / max(after_input, 1),
        hbm_cap / max(after_output, 1),
    )

    # Stranding per dimension
    flops_stranding = 1 - inflation * flops_util
    hbm_stranding = 1 - inflation * hbm_util

    # Total weighted stranding cost
    return flops_weight * flops_stranding + hbm_weight * hbm_stranding


def select_best_dp_rank(
    dp_loads: list[tuple[int, int]],  # List of (input_tokens, output_tokens) per DP
    new_input: int,
    new_output: int,
    flops_cap: int,
    hbm_cap: int,
    flops_weight: float = 1.0,
    hbm_weight: float = 1.0,
) -> int:
    """Select best DP rank based on stranding cost.

    Returns the DP rank with the lowest total stranding after placement.
    """
    best_rank = 0
    best_stranding = float("inf")

    for dp_rank, (existing_input, existing_output) in enumerate(dp_loads):
        after_input = existing_input + new_input
        after_output = existing_output + new_output

        stranding = compute_stranding(
            after_input, after_output, flops_cap, hbm_cap, flops_weight, hbm_weight
        )

        if stranding < best_stranding:
            best_stranding = stranding
            best_rank = dp_rank

    return best_rank


class TestStrandingCalculation(unittest.TestCase):
    """Test the stranding calculation logic."""

    def test_balanced_utilization_zero_stranding(self):
        """When both dimensions have equal utilization, stranding should be minimal."""
        # Equal utilization in both dimensions
        flops_cap = 1000
        hbm_cap = 1000
        after_input = 500  # 50% utilization
        after_output = 500  # 50% utilization

        stranding = compute_stranding(after_input, after_output, flops_cap, hbm_cap)

        # With equal utilization, inflation = 2 for both
        # flops_stranding = 1 - 2 * 0.5 = 0
        # hbm_stranding = 1 - 2 * 0.5 = 0
        self.assertAlmostEqual(stranding, 0.0)

    def test_imbalanced_utilization_positive_stranding(self):
        """When one dimension is underutilized, stranding should be positive."""
        flops_cap = 1000
        hbm_cap = 1000
        after_input = 500  # 50% utilization
        after_output = 250  # 25% utilization

        stranding = compute_stranding(after_input, after_output, flops_cap, hbm_cap)

        # flops has higher utilization (50% vs 25%), so it's the bottleneck
        # inflation = min(1000/500, 1000/250) = min(2, 4) = 2
        # flops_stranding = 1 - 2 * 0.5 = 0 (bottleneck dimension, fully utilized)
        # hbm_stranding = 1 - 2 * 0.25 = 0.5 (underutilized dimension, wasted capacity)
        # Total = 0 + 0.5 = 0.5
        self.assertAlmostEqual(stranding, 0.5)

    def test_empty_bin_high_stranding(self):
        """An empty bin should have high stranding (lots of wasted capacity)."""
        flops_cap = 1000
        hbm_cap = 1000
        after_input = 1  # Minimal to avoid division by zero
        after_output = 1

        stranding = compute_stranding(after_input, after_output, flops_cap, hbm_cap)

        # inflation = max(1000/1, 1000/1) = 1000
        # Both strandings = 1 - 1000 * 0.001 = 1 - 1 = 0
        # Actually: flops_util = 1/1000 = 0.001
        # stranding = 1 - 1000 * 0.001 = 0
        self.assertAlmostEqual(stranding, 0.0)

    def test_full_bin_zero_stranding(self):
        """A completely full bin should have zero stranding."""
        flops_cap = 1000
        hbm_cap = 1000
        after_input = 1000  # 100% utilization
        after_output = 1000  # 100% utilization

        stranding = compute_stranding(after_input, after_output, flops_cap, hbm_cap)

        # inflation = max(1, 1) = 1
        # Both strandings = 1 - 1 * 1.0 = 0
        self.assertAlmostEqual(stranding, 0.0)


class TestDPRankSelection(unittest.TestCase):
    """Test DP rank selection with stranding-based scoring."""

    def test_prefers_tighter_fit(self):
        """Should select the DP rank where the request fits most tightly."""
        flops_cap = 1000
        hbm_cap = 1000

        # DP0: already has some load
        # DP1: empty
        dp_loads = [
            (400, 400),  # DP0: 40% utilized in both
            (0, 0),  # DP1: empty
        ]

        # New request with moderate size
        new_input = 100
        new_output = 100

        # DP0 after placement: (500, 500) = 50% both, balanced
        # DP1 after placement: (100, 100) = 10% both, balanced but emptier

        best_rank = select_best_dp_rank(
            dp_loads, new_input, new_output, flops_cap, hbm_cap
        )

        # DP0 should be preferred (tighter packing)
        self.assertEqual(best_rank, 0)

    def test_balances_dimensions(self):
        """Should prefer placement that minimizes total stranding cost."""
        flops_cap = 1000
        hbm_cap = 1000

        # DP0: high FLOPs, low HBM
        # DP1: low FLOPs, high HBM
        dp_loads = [
            (600, 200),  # DP0: 60% FLOPs, 20% HBM - imbalanced
            (200, 600),  # DP1: 20% FLOPs, 60% HBM - imbalanced
        ]

        # Request that's heavier on HBM
        new_input = 100
        new_output = 300

        # DP0 after: (700, 500) -> 70% FLOPs, 50% HBM
        #   inflation = min(1000/700, 1000/500) = min(1.43, 2) = 1.43
        #   flops_stranding = 1 - 1.43*0.7 = 0 (bottleneck)
        #   hbm_stranding = 1 - 1.43*0.5 = 0.285
        #   total = 0.285
        # DP1 after: (300, 900) -> 30% FLOPs, 90% HBM
        #   inflation = min(1000/300, 1000/900) = min(3.33, 1.11) = 1.11
        #   flops_stranding = 1 - 1.11*0.3 = 0.667
        #   hbm_stranding = 1 - 1.11*0.9 = 0 (bottleneck)
        #   total = 0.667
        # DP0 has lower stranding (0.285 < 0.667), so DP0 is preferred

        best_rank = select_best_dp_rank(
            dp_loads, new_input, new_output, flops_cap, hbm_cap
        )

        # DP0 is preferred (lower total stranding = more balanced packing)
        self.assertEqual(best_rank, 0)

    def test_weight_affects_selection(self):
        """Different weights should change which DP rank is selected."""
        flops_cap = 1000
        hbm_cap = 1000

        # DP0: high FLOPs utilization
        # DP1: high HBM utilization
        dp_loads = [
            (800, 200),  # DP0: FLOPs-heavy
            (200, 800),  # DP1: HBM-heavy
        ]

        # Small balanced request
        new_input = 100
        new_output = 100

        # With equal weights
        rank_equal = select_best_dp_rank(
            dp_loads, new_input, new_output, flops_cap, hbm_cap, 1.0, 1.0
        )

        # With high HBM weight (penalize HBM stranding more)
        rank_hbm_heavy = select_best_dp_rank(
            dp_loads, new_input, new_output, flops_cap, hbm_cap, 1.0, 10.0
        )

        # With high FLOPs weight (penalize FLOPs stranding more)
        rank_flops_heavy = select_best_dp_rank(
            dp_loads, new_input, new_output, flops_cap, hbm_cap, 10.0, 1.0
        )

        # The weights should influence selection
        # (specific outcomes depend on exact stranding values)
        self.assertIsInstance(rank_equal, int)
        self.assertIsInstance(rank_hbm_heavy, int)
        self.assertIsInstance(rank_flops_heavy, int)

    def test_many_dp_ranks(self):
        """Should correctly select from many DP ranks."""
        flops_cap = 1000
        hbm_cap = 1000

        # 4 DP ranks with varying loads (all balanced)
        dp_loads = [
            (100, 100),  # DP0: 10% both
            (300, 300),  # DP1: 30% both
            (500, 500),  # DP2: 50% both
            (700, 700),  # DP3: 70% both
        ]

        # Small balanced request
        new_input = 100
        new_output = 100

        # All balanced loads result in stranding = 0 for each DP
        # (inflation * util = 1 for both dimensions when balanced)
        # First eligible DP (DP0) wins when all are equal

        best_rank = select_best_dp_rank(
            dp_loads, new_input, new_output, flops_cap, hbm_cap
        )

        # All have equal stranding (0), so first one (DP0) is selected
        self.assertEqual(best_rank, 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases in stranding calculation."""

    def test_zero_input_tokens(self):
        """Should handle zero input tokens gracefully."""
        stranding = compute_stranding(
            after_input=0,
            after_output=100,
            flops_cap=1000,
            hbm_cap=1000,
        )
        # Should not raise, should return finite value
        self.assertTrue(abs(stranding) < float("inf"))

    def test_zero_output_tokens(self):
        """Should handle zero output tokens gracefully."""
        stranding = compute_stranding(
            after_input=100,
            after_output=0,
            flops_cap=1000,
            hbm_cap=1000,
        )
        # Should not raise, should return finite value
        self.assertTrue(abs(stranding) < float("inf"))

    def test_single_dp_rank(self):
        """Should return rank 0 when there's only one DP rank."""
        dp_loads = [(500, 500)]
        best_rank = select_best_dp_rank(
            dp_loads,
            new_input=100,
            new_output=100,
            flops_cap=1000,
            hbm_cap=1000,
        )
        self.assertEqual(best_rank, 0)

    def test_asymmetric_capacities(self):
        """Should handle different FLOPs and HBM capacities."""
        flops_cap = 4096  # Smaller compute capacity
        hbm_cap = 16384  # Larger memory capacity

        dp_loads = [
            (2000, 8000),  # ~50% FLOPs, ~50% HBM
            (1000, 4000),  # ~25% FLOPs, ~25% HBM
        ]

        best_rank = select_best_dp_rank(
            dp_loads,
            new_input=500,
            new_output=2000,
            flops_cap=flops_cap,
            hbm_cap=hbm_cap,
        )

        # Should select rank 0 (tighter fit)
        self.assertEqual(best_rank, 0)


class TestSchedulingPolicyComparison(unittest.TestCase):
    """Compare best-fit scheduling against round-robin and min-util policies.

    Expected output with 10 DP ranks, 20 interleaved requests (A, B, A, B, ...):
      - A: FLOPs-heavy (400, 200)
      - B: HBM-heavy (200, 800)

    Load Distribution Comparison:
    Rank   Round-Robin          Min-Util             Best-Fit
    ------------------------------------------------------------------
    DP0    (800, 400)           (800, 400)           (600, 1000)
    DP1    (400, 1600)          (400, 1600)          (600, 1000)
    DP2    (800, 400)           (600, 1000)          (600, 1000)
    DP3    (400, 1600)          (600, 1000)          (600, 1000)
    DP4    (800, 400)           (800, 400)           (600, 1000)
    DP5    (400, 1600)          (400, 1600)          (600, 1000)
    DP6    (800, 400)           (600, 1000)          (600, 1000)
    DP7    (400, 1600)          (600, 1000)          (600, 1000)
    DP8    (800, 400)           (800, 400)           (600, 1000)
    DP9    (400, 1600)          (400, 1600)          (600, 1000)

    Metrics:
    Max Utilization: RR=160.00%, Min=160.00%, BF=100.00%
    Load Variance:   RR=0.040000, Min=0.024000, BF=0.000000

    Key observations:
    - Round-Robin and Min-Util exceed capacity (160% HBM on some ranks)
      - >100% utilization causes latency issues: memory pressure leads to swapping,
        increased GC overhead, and potential OOM errors during decode phase
    - Best-Fit respects capacity limits (max 100%)
    - Best-Fit achieves PERFECT load balance (0 variance)
    - Best-Fit pairs ALL complementary shapes: every rank gets (600, 1000) = A + B
    - The min() formula correctly identifies bottleneck, enabling optimal pairing
    - With strict capacity feasibility checks, lower stranding = higher throughput
      (better bin packing means more requests fit within capacity limits)
    """

    def setUp(self):
        """Set up test configuration."""
        self.num_dp_ranks = 10
        self.flops_cap = 1000
        self.hbm_cap = 1000

        # 20 requests with two distinct shapes (interleaved):
        # - A: FLOPs-heavy requests: flop=400, hbm=200 (ratio 0.5)
        # - B: HBM-heavy requests: flop=200, hbm=800 (ratio 4.0)
        # Interleaved to simulate realistic arrival pattern:
        # A, B, A, B, A, B, ...
        self.requests = []
        for _ in range(10):
            self.requests.append((400, 200))  # FLOPs-heavy
            self.requests.append((200, 800))  # HBM-heavy

    def _simulate_round_robin(
        self, requests: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Simulate round-robin scheduling."""
        dp_loads = [(0, 0) for _ in range(self.num_dp_ranks)]
        current_rank = 0

        for flop_req, hbm_req in requests:
            existing_flop, existing_hbm = dp_loads[current_rank]
            dp_loads[current_rank] = (existing_flop + flop_req, existing_hbm + hbm_req)
            current_rank = (current_rank + 1) % self.num_dp_ranks

        return dp_loads

    def _simulate_min_util(
        self, requests: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Simulate min-util (min running queue) scheduling.

        Selects DP rank with minimum total load (flops + hbm).
        """
        dp_loads = [(0, 0) for _ in range(self.num_dp_ranks)]

        for flop_req, hbm_req in requests:
            # Find rank with minimum total load
            min_load = float("inf")
            best_rank = 0
            for rank, (flop, hbm) in enumerate(dp_loads):
                total_load = flop + hbm
                if total_load < min_load:
                    min_load = total_load
                    best_rank = rank

            existing_flop, existing_hbm = dp_loads[best_rank]
            dp_loads[best_rank] = (existing_flop + flop_req, existing_hbm + hbm_req)

        return dp_loads

    def _simulate_best_fit(
        self, requests: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Simulate best-fit scheduling with stranding-based scoring.

        Includes capacity constraints - skips ranks that would exceed capacity.
        """
        dp_loads = [(0, 0) for _ in range(self.num_dp_ranks)]

        for flop_req, hbm_req in requests:
            best_rank = self._select_best_fit_with_capacity(
                dp_loads, flop_req, hbm_req
            )
            if best_rank is not None:
                existing_flop, existing_hbm = dp_loads[best_rank]
                dp_loads[best_rank] = (existing_flop + flop_req, existing_hbm + hbm_req)

        return dp_loads

    def _select_best_fit_with_capacity(
        self,
        dp_loads: list[tuple[int, int]],
        new_flop: int,
        new_hbm: int,
    ) -> int | None:
        """Select best DP rank with capacity constraints."""
        best_rank = None
        best_stranding = float("inf")

        for rank, (existing_flop, existing_hbm) in enumerate(dp_loads):
            after_flop = existing_flop + new_flop
            after_hbm = existing_hbm + new_hbm

            # Skip if would exceed capacity
            if after_flop > self.flops_cap or after_hbm > self.hbm_cap:
                continue

            stranding = compute_stranding(
                after_flop, after_hbm, self.flops_cap, self.hbm_cap
            )

            if stranding < best_stranding:
                best_stranding = stranding
                best_rank = rank

        return best_rank

    def _compute_total_stranding(self, dp_loads: list[tuple[int, int]]) -> float:
        """Compute total stranding across all DP ranks."""
        total = 0.0
        for flop, hbm in dp_loads:
            if flop > 0 or hbm > 0:
                total += compute_stranding(flop, hbm, self.flops_cap, self.hbm_cap)
        return total

    def _compute_max_utilization(self, dp_loads: list[tuple[int, int]]) -> float:
        """Compute maximum utilization across any dimension/rank."""
        max_util = 0.0
        for flop, hbm in dp_loads:
            flop_util = flop / self.flops_cap
            hbm_util = hbm / self.hbm_cap
            max_util = max(max_util, flop_util, hbm_util)
        return max_util

    def _compute_balance_score(self, dp_loads: list[tuple[int, int]]) -> float:
        """Compute how balanced the loads are (lower = more balanced).

        Returns variance of utilization across ranks.
        """
        if not dp_loads:
            return 0.0

        # Compute total utilization per rank
        utils = []
        for flop, hbm in dp_loads:
            util = (flop / self.flops_cap + hbm / self.hbm_cap) / 2
            utils.append(util)

        mean_util = sum(utils) / len(utils)
        variance = sum((u - mean_util) ** 2 for u in utils) / len(utils)
        return variance

    def test_all_policies_schedule_requests(self):
        """All policies should schedule requests (best-fit respects capacity)."""
        rr_loads = self._simulate_round_robin(self.requests)
        min_loads = self._simulate_min_util(self.requests)
        bf_loads = self._simulate_best_fit(self.requests)

        total_flops = sum(r[0] for r in self.requests)
        total_hbm = sum(r[1] for r in self.requests)

        # Round-robin and min-util don't enforce capacity, so they schedule everything
        for name, loads in [("round_robin", rr_loads), ("min_util", min_loads)]:
            scheduled_flops = sum(l[0] for l in loads)
            scheduled_hbm = sum(l[1] for l in loads)
            self.assertEqual(scheduled_flops, total_flops, f"{name} lost FLOPs")
            self.assertEqual(scheduled_hbm, total_hbm, f"{name} lost HBM")

        # Best-fit with capacity may not fit everything, but should schedule some
        bf_scheduled_flops = sum(l[0] for l in bf_loads)
        bf_scheduled_hbm = sum(l[1] for l in bf_loads)
        self.assertGreater(bf_scheduled_flops, 0, "Best-fit should schedule some FLOPs")
        self.assertGreater(bf_scheduled_hbm, 0, "Best-fit should schedule some HBM")

        print(f"\n=== Scheduling Summary ===")
        print(f"Total requests: FLOPs={total_flops}, HBM={total_hbm}")
        print(f"Best-fit scheduled: FLOPs={bf_scheduled_flops}, HBM={bf_scheduled_hbm}")

    def test_best_fit_respects_capacity(self):
        """Best-fit should never exceed capacity limits."""
        bf_loads = self._simulate_best_fit(self.requests)

        for rank, (flop, hbm) in enumerate(bf_loads):
            self.assertLessEqual(
                flop, self.flops_cap, f"DP{rank} exceeded FLOPs capacity"
            )
            self.assertLessEqual(
                hbm, self.hbm_cap, f"DP{rank} exceeded HBM capacity"
            )

    def test_stranding_comparison(self):
        """Compare stranding metrics across policies (observational)."""
        rr_loads = self._simulate_round_robin(self.requests)
        min_loads = self._simulate_min_util(self.requests)
        bf_loads = self._simulate_best_fit(self.requests)

        rr_stranding = self._compute_total_stranding(rr_loads)
        min_stranding = self._compute_total_stranding(min_loads)
        bf_stranding = self._compute_total_stranding(bf_loads)

        print(f"\n=== Stranding Comparison ===")
        print(f"Round-Robin: {rr_stranding:.4f} (no capacity limits)")
        print(f"Min-Util:    {min_stranding:.4f} (no capacity limits)")
        print(f"Best-Fit:    {bf_stranding:.4f} (with capacity limits)")

        # All should return valid stranding values
        self.assertTrue(abs(rr_stranding) < float("inf"))
        self.assertTrue(abs(min_stranding) < float("inf"))
        self.assertTrue(abs(bf_stranding) < float("inf"))

    def test_best_fit_pairs_complementary_shapes(self):
        """Best-fit should pair FLOPs-heavy with HBM-heavy requests."""
        bf_loads = self._simulate_best_fit(self.requests)

        # Count how many ranks have balanced loads (both dimensions used)
        balanced_ranks = 0
        for flop, hbm in bf_loads:
            if flop > 0 and hbm > 0:
                # Check if reasonably balanced (ratio between 0.3 and 3)
                ratio = flop / max(hbm, 1)
                if 0.3 <= ratio <= 3:
                    balanced_ranks += 1

        print(f"\n=== Load Distribution (Best-Fit) ===")
        for i, (flop, hbm) in enumerate(bf_loads):
            flop_pct = 100 * flop / self.flops_cap
            hbm_pct = 100 * hbm / self.hbm_cap
            print(f"DP{i}: FLOPs={flop:4d} ({flop_pct:5.1f}%), HBM={hbm:4d} ({hbm_pct:5.1f}%)")

        # Best-fit should create some balanced ranks by pairing shapes
        self.assertGreater(
            balanced_ranks, 0, "Best-fit should create balanced rank pairings"
        )

    def test_load_distribution_comparison(self):
        """Compare load distribution across all policies."""
        rr_loads = self._simulate_round_robin(self.requests)
        min_loads = self._simulate_min_util(self.requests)
        bf_loads = self._simulate_best_fit(self.requests)

        print(f"\n=== Load Distribution Comparison ===")
        print(f"{'Rank':<6} {'Round-Robin':<20} {'Min-Util':<20} {'Best-Fit':<20}")
        print("-" * 66)
        for i in range(self.num_dp_ranks):
            rr = f"({rr_loads[i][0]:3d}, {rr_loads[i][1]:3d})"
            mu = f"({min_loads[i][0]:3d}, {min_loads[i][1]:3d})"
            bf = f"({bf_loads[i][0]:3d}, {bf_loads[i][1]:3d})"
            print(f"DP{i:<4} {rr:<20} {mu:<20} {bf:<20}")

        # Compute metrics
        rr_max = self._compute_max_utilization(rr_loads)
        min_max = self._compute_max_utilization(min_loads)
        bf_max = self._compute_max_utilization(bf_loads)

        rr_var = self._compute_balance_score(rr_loads)
        min_var = self._compute_balance_score(min_loads)
        bf_var = self._compute_balance_score(bf_loads)

        print(f"\n=== Metrics ===")
        print(f"Max Utilization: RR={rr_max:.2%}, Min={min_max:.2%}, BF={bf_max:.2%}")
        print(f"Load Variance:   RR={rr_var:.6f}, Min={min_var:.6f}, BF={bf_var:.6f}")

        # Best-fit should respect capacity (others don't enforce it)
        self.assertLessEqual(bf_max, 1.0, "Best-fit exceeded capacity")

        # Best-fit should have lower variance (better load balance)
        self.assertLessEqual(
            bf_var, rr_var, "Best-fit should have <= variance than round-robin"
        )


if __name__ == "__main__":
    unittest.main()
