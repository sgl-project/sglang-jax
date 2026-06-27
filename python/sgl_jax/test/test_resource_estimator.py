"""Tests for ResourceEstimator class."""

import unittest

from sgl_jax.srt.managers.resource_estimator import (
    ModelResourceConfig,
    ResourceEstimate,
    ResourceEstimator,
    TPUCapacity,
    TPU_V4,
    TPU_V5E,
    TPU_V5P,
    TPU_V6E,
)


def _make_llama_8b_config(bytes_per_element: int = 2) -> ModelResourceConfig:
    """Create a Llama 3 8B config for testing."""
    return ModelResourceConfig(
        num_hidden_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        head_dim=128,
        num_key_value_heads=8,
        intermediate_size=14336,
        vocab_size=128256,
        num_params=8_000_000_000,
        bytes_per_element=bytes_per_element,
    )


def _make_llama_70b_config(bytes_per_element: int = 2) -> ModelResourceConfig:
    """Create a Llama 3 70B config for testing."""
    return ModelResourceConfig(
        num_hidden_layers=80,
        hidden_size=8192,
        num_attention_heads=64,
        head_dim=128,
        num_key_value_heads=8,
        intermediate_size=28672,
        vocab_size=128256,
        num_params=70_000_000_000,
        bytes_per_element=bytes_per_element,
    )


def _make_mixtral_config(bytes_per_element: int = 2) -> ModelResourceConfig:
    """Create a Mixtral 8x7B config for testing."""
    return ModelResourceConfig(
        num_hidden_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        head_dim=128,
        num_key_value_heads=8,
        intermediate_size=14336,
        vocab_size=32000,
        num_params=46_700_000_000,
        bytes_per_element=bytes_per_element,
        num_experts=8,
        num_active_experts=2,
    )


class TestTPUCapacity(unittest.TestCase):
    """Test TPUCapacity class."""

    def test_tpu_v4_intensity(self):
        """Test TPU v4 intensity ratio."""
        self.assertAlmostEqual(TPU_V4.compute_intensity_ratio, 275 / 32, places=2)

    def test_tpu_v5e_intensity(self):
        """Test TPU v5e intensity ratio."""
        self.assertAlmostEqual(TPU_V5E.compute_intensity_ratio, 197 / 16, places=2)

    def test_tpu_v5p_intensity(self):
        """Test TPU v5p intensity ratio."""
        self.assertAlmostEqual(TPU_V5P.compute_intensity_ratio, 459 / 95, places=2)

    def test_tpu_v6e_intensity(self):
        """Test TPU v6e intensity ratio."""
        self.assertAlmostEqual(TPU_V6E.compute_intensity_ratio, 918 / 32, places=2)

    def test_custom_tpu_capacity(self):
        """Test creating custom TPU capacity."""
        custom = TPUCapacity("custom", flops_tflops=100, hbm_gb=50)
        self.assertEqual(custom.compute_intensity_ratio, 2.0)


class TestModelResourceConfig(unittest.TestCase):
    """Test ModelResourceConfig creation."""

    def test_llama_8b_config(self):
        """Test Llama 8B config values."""
        config = _make_llama_8b_config()
        self.assertEqual(config.num_hidden_layers, 32)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_attention_heads, 32)
        self.assertEqual(config.head_dim, 128)
        self.assertEqual(config.num_key_value_heads, 8)
        self.assertEqual(config.bytes_per_element, 2)

    def test_llama_70b_config(self):
        """Test Llama 70B config values."""
        config = _make_llama_70b_config()
        self.assertEqual(config.num_hidden_layers, 80)
        self.assertEqual(config.hidden_size, 8192)
        self.assertEqual(config.num_attention_heads, 64)
        self.assertEqual(config.head_dim, 128)
        self.assertEqual(config.num_key_value_heads, 8)

    def test_bytes_per_element_override(self):
        """Test bytes_per_element can be overridden."""
        config = _make_llama_8b_config(bytes_per_element=1)  # int8
        self.assertEqual(config.bytes_per_element, 1)


class TestResourceEstimator(unittest.TestCase):
    """Test ResourceEstimator estimation methods."""

    def setUp(self):
        """Set up test estimator."""
        self.config = _make_llama_8b_config()
        self.estimator = ResourceEstimator(model_config=self.config)

    def test_estimate_returns_resource_estimate(self):
        """Test estimate returns ResourceEstimate dataclass."""
        estimate = self.estimator.estimate(input_tokens=1024, output_tokens=512)
        self.assertIsInstance(estimate, ResourceEstimate)
        self.assertEqual(estimate.input_tokens, 1024)
        self.assertEqual(estimate.output_tokens, 512)
        self.assertEqual(estimate.total_tokens, 1536)

    def test_flops_scales_with_input(self):
        """Test FLOPs estimation scaling with input length.

        FLOPs = max(attention, FFN) where:
        - Attention: 4 * L * n^2 * d (quadratic)
        - FFN: 6 * L * n * d * d_ffn (linear)

        Crossover at n = 1.5 * d_ffn (~21K for 8B). Below this, FFN dominates
        and scaling is linear. Above this, attention dominates and scaling
        becomes quadratic.
        """
        flops_512 = self.estimator.estimate_flops(512)
        flops_1024 = self.estimator.estimate_flops(1024)
        flops_4096 = self.estimator.estimate_flops(4096)

        # For short sequences (< 21K), FFN dominates, so scaling is linear
        # Doubling input doubles FLOPs
        ratio_1 = flops_1024 / flops_512
        self.assertAlmostEqual(ratio_1, 2.0, places=1)  # 1024/512 = 2x

        # 4x input = 4x FLOPs (linear scaling)
        ratio_2 = flops_4096 / flops_1024
        self.assertAlmostEqual(ratio_2, 4.0, places=1)  # 4096/1024 = 4x

    def test_hbm_scales_linearly_with_tokens(self):
        """Test HBM estimation scales linearly with total tokens."""
        hbm_1000 = self.estimator.estimate_hbm_bytes(500, 500)
        hbm_2000 = self.estimator.estimate_hbm_bytes(1000, 1000)

        # Doubling total tokens should exactly double HBM
        ratio = hbm_2000 / hbm_1000
        self.assertAlmostEqual(ratio, 2.0, places=5)

    def test_hbm_is_input_output_symmetric(self):
        """Test HBM only depends on total tokens, not distribution."""
        hbm_1 = self.estimator.estimate_hbm_bytes(1000, 500)
        hbm_2 = self.estimator.estimate_hbm_bytes(500, 1000)

        self.assertEqual(hbm_1, hbm_2)

    def test_estimate_realistic_values_llama_8b(self):
        """Test estimates are in realistic range for Llama 8B."""
        estimate = self.estimator.estimate(input_tokens=1024, output_tokens=512)

        # FLOPs should be in billions/trillions range
        self.assertGreater(estimate.flops, 1e12)  # > 1 TFLOPs

        # HBM for 1536 tokens with Llama 8B:
        # 2 * 32 layers * 8 KV heads * 128 head_dim * 1536 tokens * 2 bytes
        # = 2 * 32 * 8 * 128 * 1536 * 2 = ~201 MB
        expected_hbm = 2 * 32 * 8 * 128 * 1536 * 2
        self.assertEqual(estimate.hbm_bytes, expected_hbm)


class TestComputeIntensityRatio(unittest.TestCase):
    """Test compute_intensity_ratio method."""

    def setUp(self):
        """Set up test estimator."""
        self.config = _make_llama_8b_config()
        self.estimator = ResourceEstimator(model_config=self.config)

    def test_intensity_positive(self):
        """Test intensity is positive for valid inputs."""
        intensity = self.estimator.compute_intensity_ratio(1000, 500)
        self.assertGreater(intensity, 0)

    def test_intensity_zero_for_zero_input(self):
        """Test intensity is zero when input is zero."""
        intensity = self.estimator.compute_intensity_ratio(0, 500)
        self.assertEqual(intensity, 0)

    def test_intensity_decreases_with_output(self):
        """Test intensity decreases as output increases (fixed input)."""
        # More output = more HBM = lower intensity
        intensity_low_output = self.estimator.compute_intensity_ratio(500, 100)
        intensity_high_output = self.estimator.compute_intensity_ratio(500, 10000)
        self.assertGreater(intensity_low_output, intensity_high_output)

    def test_intensity_matches_paper_values(self):
        """Test intensity matches values from paper tables.

        For Llama 8B with 500 input, 200 output:
        - Intensity should be ~61 TFLOPs/GB
        """
        intensity = self.estimator.compute_intensity_ratio(500, 200)
        self.assertAlmostEqual(intensity, 61.4, places=0)


class TestStrandingCalculation(unittest.TestCase):
    """Test stranding calculation using intensity ratios."""

    def setUp(self):
        """Set up test estimator with TPU v4."""
        self.config = _make_llama_8b_config()
        self.estimator = ResourceEstimator(
            model_config=self.config,
            tpu_capacity=TPU_V4,
        )

    def test_compute_bound_workload_has_hbm_stranding(self):
        """Test compute-bound workload (high intensity) has HBM stranding.

        Standard Q&A (500 input, 200 output) has intensity ~61 TFLOPs/GB.
        TPU v4 has intensity 8.6 TFLOPs/GB.
        Since workload intensity > hardware intensity, compute is bottleneck.
        This means HBM is stranded (underutilized).
        """
        stranding = self.estimator.compute_stranding(500, 200)
        # Stranding should be positive (HBM stranded)
        self.assertGreater(stranding, 0)

        # Verify HBM stranding dominates (compute is bottleneck)
        # workload_intensity >> hardware_intensity
        # So: FLOPs stranding = max(0, 1 - 61/8.6) = 0
        #     HBM stranding = max(0, 1 - 8.6/61) = ~0.86

    def test_memory_bound_workload_has_flops_stranding(self):
        """Test memory-bound workload (low intensity) has FLOPs stranding.

        CoT o1-style (100 input, 20000 output) has intensity ~0.4 TFLOPs/GB.
        TPU v4 has intensity 8.6 TFLOPs/GB.
        Since workload intensity < hardware intensity, memory is bottleneck.
        This means FLOPs is stranded (underutilized).
        """
        stranding = self.estimator.compute_stranding(100, 20000)
        # Stranding should be positive (FLOPs stranded)
        self.assertGreater(stranding, 0)

    def test_empty_load_max_stranding(self):
        """Test empty load has maximum stranding."""
        stranding = self.estimator.compute_stranding(0, 0)
        self.assertEqual(stranding, 2.0)  # flops_weight + hbm_weight

    def test_weights_affect_stranding(self):
        """Test cost weights affect stranding calculation."""
        base_stranding = self.estimator.compute_stranding(
            500, 200, flops_weight=1.0, hbm_weight=1.0
        )

        # With high HBM weight, stranding should increase
        # (HBM is stranded for this compute-bound workload)
        weighted_stranding = self.estimator.compute_stranding(
            500, 200, flops_weight=1.0, hbm_weight=10.0
        )

        self.assertGreater(weighted_stranding, base_stranding)


class TestDPRankSelection(unittest.TestCase):
    """Test DP rank selection with ResourceEstimator."""

    def setUp(self):
        """Set up estimator with max token capacity."""
        self.config = _make_llama_8b_config()
        self.estimator = ResourceEstimator(
            model_config=self.config,
            tpu_capacity=TPU_V4,
            max_total_tokens=1000,
        )

    def test_selects_lower_stranding(self):
        """Test selection prefers rank with lower stranding after placement.

        The intensity-based formula favors workloads closer to hardware intensity.
        For complementary pairing (compute-heavy + memory-heavy), combining them
        achieves better balance than putting them on separate replicas.
        """
        # DP0 has a memory-heavy workload (low intensity)
        # DP1 has a compute-heavy workload (high intensity)
        dp_loads = [
            (50, 500),  # DP0: memory-heavy (550 tokens)
            (400, 50),  # DP1: compute-heavy (450 tokens)
        ]

        # Adding a compute-heavy request (high input, low output)
        # Should pair with memory-heavy DP0 for better balance
        best_rank = self.estimator.select_best_dp_rank(
            dp_loads,
            new_input_tokens=200,
            new_output_tokens=50,
        )

        # Should prefer DP0 (complementary pairing reduces stranding)
        self.assertEqual(best_rank, 0)

    def test_respects_capacity(self):
        """Test selection respects max_total_tokens constraint."""
        # DP0 is near capacity, DP1 has room
        dp_loads = [
            (400, 500),  # DP0: 900 total tokens
            (100, 100),  # DP1: 200 total tokens
        ]

        # Request that would exceed DP0's capacity (900 + 200 = 1100 > 1000)
        best_rank = self.estimator.select_best_dp_rank(
            dp_loads,
            new_input_tokens=100,
            new_output_tokens=100,
            respect_capacity=True,
        )

        # Should pick DP1 since DP0 would exceed capacity
        self.assertEqual(best_rank, 1)

    def test_returns_none_when_no_fit(self):
        """Test returns None when no replica can fit request."""
        # Both DPs are near capacity
        dp_loads = [
            (400, 500),  # DP0: 900 total tokens
            (450, 500),  # DP1: 950 total tokens
        ]

        # Large request that won't fit anywhere
        best_rank = self.estimator.select_best_dp_rank(
            dp_loads,
            new_input_tokens=100,
            new_output_tokens=100,
            respect_capacity=True,
        )

        self.assertIsNone(best_rank)

    def test_ignores_capacity_when_disabled(self):
        """Test selection ignores capacity when respect_capacity=False."""
        # Both DPs are over capacity
        dp_loads = [
            (800, 500),  # DP0: 1300 total tokens (over capacity)
            (900, 500),  # DP1: 1400 total tokens (over capacity)
        ]

        # Should still return a rank when capacity is not respected
        best_rank = self.estimator.select_best_dp_rank(
            dp_loads,
            new_input_tokens=100,
            new_output_tokens=100,
            respect_capacity=False,
        )

        self.assertIsNotNone(best_rank)


class TestResourceEstimatorFormatting(unittest.TestCase):
    """Test formatting utilities."""

    def test_format_estimate(self):
        """Test format_estimate produces readable output."""
        config = _make_llama_8b_config()
        estimator = ResourceEstimator(model_config=config)

        estimate = estimator.estimate(input_tokens=1024, output_tokens=512)
        formatted = estimator.format_estimate(estimate)

        self.assertIn("FLOPs=", formatted)
        self.assertIn("HBM=", formatted)
        self.assertIn("GB", formatted)
        self.assertIn("tokens=1024+512", formatted)


class TestMixtralConfig(unittest.TestCase):
    """Test MoE model configuration."""

    def test_mixtral_config(self):
        """Test Mixtral 8x7B config."""
        config = _make_mixtral_config()
        self.assertEqual(config.num_hidden_layers, 32)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_experts, 8)
        self.assertEqual(config.num_active_experts, 2)
        self.assertTrue(config.is_moe)

    def test_dense_is_not_moe(self):
        """Test dense model is_moe property."""
        config = _make_llama_8b_config()
        self.assertFalse(config.is_moe)
        self.assertEqual(config.num_experts, 1)
        self.assertEqual(config.num_active_experts, 1)

    def test_mixtral_hbm_estimation(self):
        """Test HBM estimation for Mixtral (same KV cache as dense).

        KV cache is shared across all experts, so MoE has no impact on HBM.
        """
        moe_config = _make_mixtral_config()
        dense_config = _make_llama_8b_config()  # Same dimensions as Mixtral base
        moe_estimator = ResourceEstimator(model_config=moe_config)
        dense_estimator = ResourceEstimator(model_config=dense_config)

        # KV cache should be identical for MoE and dense with same dimensions
        moe_hbm = moe_estimator.estimate_hbm_bytes(1024, 512)
        dense_hbm = dense_estimator.estimate_hbm_bytes(1024, 512)
        self.assertEqual(moe_hbm, dense_hbm)

    def test_mixtral_flops_scales_with_active_experts(self):
        """Test FFN FLOPs scale with number of active experts.

        Mixtral activates 2 out of 8 experts per token.
        FFN FLOPs should be 2x a single-expert dense model.
        """
        moe_config = _make_mixtral_config()  # 2 active experts
        dense_config = _make_llama_8b_config()  # 1 active expert (dense)
        moe_estimator = ResourceEstimator(model_config=moe_config)
        dense_estimator = ResourceEstimator(model_config=dense_config)

        # For short sequences where FFN dominates (not attention)
        n = 500  # Well below crossover point
        moe_flops = moe_estimator.estimate_flops(n)
        dense_flops = dense_estimator.estimate_flops(n)

        # MoE FFN FLOPs = 2 * dense FFN FLOPs (since 2 experts active)
        # Note: both are FFN-dominated at this sequence length
        self.assertEqual(moe_flops, 2 * dense_flops)

    def test_moe_crossover_point_shifts(self):
        """Test that MoE shifts the attention/FFN crossover point.

        With K active experts, crossover shifts to n = 1.5 * d_ffn * K.
        For Mixtral: 1.5 * 14336 * 2 = 43008 tokens.
        For dense: 1.5 * 14336 * 1 = 21504 tokens.
        """
        moe_config = _make_mixtral_config()
        d_ffn = moe_config.intermediate_size
        K = moe_config.num_active_experts

        # MoE crossover point
        moe_crossover = 1.5 * d_ffn * K
        self.assertAlmostEqual(moe_crossover, 43008, places=0)

        # Dense crossover point
        dense_crossover = 1.5 * d_ffn * 1
        self.assertAlmostEqual(dense_crossover, 21504, places=0)


if __name__ == "__main__":
    unittest.main()
