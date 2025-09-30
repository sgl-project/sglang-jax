import unittest

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.sampler import multinomial_with_seed


class TestMultinomialWithSeed(unittest.TestCase):

    def test_deterministic_sampling_with_same_seed(self):
        """Test that same (inputs, seed) pair always yields the same sample."""
        # Setup test data
        batch_size = 4
        vocab_size = 10

        # Create logits that simulate different temperature scenarios
        flatter_distribution = jnp.array(
            [
                [1.0, 1.1, 0.9, 1.2, 0.8, 1.3, 0.7, 1.4, 0.6, 1.5],
                [2.0, 2.1, 1.9, 2.2, 1.8, 2.3, 1.7, 2.4, 1.6, 2.5],
                [0.5, 0.6, 0.4, 0.7, 0.3, 0.8, 0.2, 0.9, 0.1, 1.0],
                [3.0, 3.1, 2.9, 3.2, 2.8, 3.3, 2.7, 3.4, 2.6, 3.5],
            ],
            dtype=jnp.bfloat16,
        )

        flatter_distribution_processed = jax.nn.softmax(flatter_distribution, axis=-1)

        shaper_distribution = jnp.array(
            [
                [1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 8.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                [0.5, 0.5, 0.5, 7.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [3.0, 3.0, 3.0, 3.0, 9.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            ],
            dtype=jnp.bfloat16,
        )

        shaper_distribution_processed = jax.nn.softmax(shaper_distribution, axis=-1)

        seeds = jnp.array([12345, 67890, 54321, 98765])
        positions = jnp.array([0, 1, 2, 3])

        test_cases = [
            ("flatter_distribution", flatter_distribution_processed),
            ("shaper_distribution", shaper_distribution_processed),
        ]

        for test_name, inputs in test_cases:
            with self.subTest(test_name=test_name):
                # Sample multiple times with the same inputs and seeds
                samples = []
                for _ in range(10):  # Run 10 times
                    sample = multinomial_with_seed((inputs, seeds, positions, None))
                    samples.append(sample)

                # All samples should be identical
                first_sample = samples[0]
                for i, sample in enumerate(samples[1:], 1):
                    np.testing.assert_array_equal(
                        first_sample,
                        sample,
                        f"Sample {i} differs from first sample for {test_name}",
                    )

    def test_different_seeds_produce_different_samples(self):
        """Test that different seeds produce different samples (with high probability)."""
        batch_size = 1
        vocab_size = 10

        inputs = jnp.ones((batch_size, vocab_size), dtype=jnp.bfloat16) * 0.1
        inputs = jax.nn.softmax(inputs, axis=-1)
        positions = jnp.array([0])

        seeds = [jnp.array([1]), jnp.array([2]), jnp.array([12345]), jnp.array([98765])]

        samples = []
        for seed in seeds:
            sample = multinomial_with_seed((inputs, seed, positions, None))
            samples.append(sample)

        original_len = len(samples)
        unique_samples = set(tuple(sample.flatten().tolist()) for sample in samples)
        self.assertEqual(original_len, len(unique_samples))

    def test_output_shape_and_range(self):
        """Test that output has correct shape and values are in valid range."""
        batch_size = 3
        vocab_size = 7

        inputs = jnp.ones((batch_size, vocab_size), dtype=jnp.bfloat16)
        inputs = jax.nn.softmax(inputs, axis=-1)
        seeds = jnp.array([1, 2, 3])
        positions = jnp.array([0, 1, 2])

        sample = multinomial_with_seed((inputs, seeds, positions, None))

        expected_shape = (batch_size, 1)  # Function returns keepdims=True
        self.assertEqual(sample.shape, expected_shape)

        self.assertTrue(jnp.all(sample >= 0))
        self.assertTrue(jnp.all(sample < vocab_size))
        self.assertTrue(sample.dtype in [jnp.int32, jnp.int64])


if __name__ == "__main__":
    unittest.main()
