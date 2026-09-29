import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers import sampler as sampler_mod
from sgl_jax.srt.layers.sampler import multinomial_with_seed


class TestMultinomialWithSeed(unittest.TestCase):
    def test_deterministic_sampling_with_same_seed(self):
        """Test that same (inputs, seed) pair always yields the same sample."""
        # Setup test data
        # batch_size = 4
        # vocab_size = 10

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
                    sample = multinomial_with_seed((inputs, seeds, positions, None, True))
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
            sample = multinomial_with_seed((inputs, seed, positions, None, True))
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

        sample = multinomial_with_seed((inputs, seeds, positions, None, True))

        expected_shape = (batch_size, 1)  # Function returns keepdims=True
        self.assertEqual(sample.shape, expected_shape)

        self.assertTrue(jnp.all(sample >= 0))
        self.assertTrue(jnp.all(sample < vocab_size))
        self.assertTrue(sample.dtype in [jnp.int32, jnp.int64])


# ---------------------------------------------------------------------------
# top-k / top-p / min-p filtering. Each cutoff is measured on the temperature-
# scaled distribution *before* any masking, and the three are intersected.
#
# `_reference_probs` below spells that out. It used to be spelled out by the
# `jnp.sort` implementation these tests compared against; that path is gone, so
# the reference lives here instead and the guards survive its removal.
# ---------------------------------------------------------------------------

_VOCAB = 4096
_BATCH = 4
# The masks reject with `_MASK_FILL_VALUE`; anything above half of it is a real
# logit that survived.
_KEPT_LOGIT_FLOOR = sampler_mod._MASK_FILL_VALUE / 2
# How far down the fixture's sorted distribution the probabilities are known to
# be tie-free. See `test_fixture_ties_sit_below_every_cutoff`.
_TIE_FREE_PREFIX = 512


def _make_logits(seed: int = 0) -> jax.Array:
    """Logits with a realistic shape: a broad tail plus a band of peaks."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    logits = jax.random.normal(k1, (_BATCH, _VOCAB))
    peaks = jax.random.randint(k2, (_BATCH, 200), 0, _VOCAB)
    return logits.at[jnp.arange(_BATCH)[:, None], peaks].add(3.0).astype(jnp.float32)


def _make_args(logits, temperature, top_k, top_p, min_p):
    """Build the 10-tuple `args` that `Sampler._regular_sampling` passes down.

    `sampling_seeds=None` selects the plain `multinomial`, which is the function
    `_capture` intercepts.
    """
    temperatures = jnp.full((_BATCH, 1), temperature, dtype=jnp.float32)
    probs = jax.nn.softmax(jnp.divide(logits, temperatures), axis=-1)
    return (
        logits,
        probs,
        jnp.full((_BATCH,), top_k, dtype=jnp.int32),
        jnp.full((_BATCH,), top_p, dtype=jnp.float32),
        jnp.full((_BATCH,), min_p, dtype=jnp.float32),
        jnp.arange(_BATCH, dtype=jnp.int32),
        temperatures,
        None,
        jnp.asarray(min_p > 0.0),
        jax.random.PRNGKey(0),
    )


def _capture(args):
    """Run the sampler and return the tensor it was about to sample from."""
    captured = {}

    def spy(operands):
        captured["inputs"] = operands[0]
        return jnp.zeros((_BATCH, 1), dtype=jnp.int32)

    with mock.patch.object(sampler_mod, "multinomial", new=spy):
        sampler_mod.top_k_top_p_min_p_sampling_from_probs_jax_with_mask(args)
    return captured["inputs"]


def _sampled_probs(args):
    """Vocab-ordered distribution the sampler draws from."""
    return np.asarray(jax.nn.softmax(_capture(args), axis=-1))


def _reference_probs(logits, temperature, top_k, top_p, min_p):
    """The semantics under test, written out plainly in numpy.

    Every cutoff is taken from the same temperature-scaled, *unmasked*
    distribution, and the three are then intersected. This is what SGLang GPU
    does, and what the removed `jnp.sort` path did.
    """
    probs = np.asarray(jax.nn.softmax(np.asarray(logits) / temperature, axis=-1))
    order = np.argsort(probs, axis=-1)[:, ::-1]
    ordered = np.take_along_axis(probs, order, axis=-1)

    within_top_k = np.arange(_VOCAB)[None, :] < top_k
    within_top_p = (np.cumsum(ordered, axis=-1) - ordered) <= top_p
    kept = np.where(within_top_k & within_top_p, ordered, 0.0)
    if min_p > 0.0:
        kept = np.where(kept < kept.max(axis=-1, keepdims=True) * min_p, 0.0, kept)

    dense = np.zeros_like(probs)
    np.put_along_axis(dense, order, kept, axis=-1)
    return dense / dense.sum(axis=-1, keepdims=True)


def _total_variation(a, b):
    return float(np.abs(a - b).sum(axis=-1).max() / 2)


# (name, temperature, top_k, top_p, min_p)
_CONFIGS = [
    ("top_p only", 0.6, _VOCAB, 0.95, 0.0),
    ("top_k only", 0.7, 20, 1.0, 0.0),
    ("top_k + top_p", 0.7, 20, 0.80, 0.0),
    ("top_k + top_p + min_p", 0.8, 50, 0.90, 0.10),
    ("min_p only", 1.0, _VOCAB, 1.0, 0.05),
    ("temperature 1.0", 1.0, 20, 0.80, 0.02),
    ("top_p tighter than top_k", 0.7, 200, 0.30, 0.0),
]


class TestTopKTopPMinPSemantics(unittest.TestCase):
    """The sampler must filter to the distribution `_reference_probs` describes.

    Runs on whatever the default backend is. The sharding pitfall recorded in
    test_sampler_deterministic_cond.py needs an explicit mesh, which the
    unsharded fixture below never builds.
    """

    def test_fixture_ties_sit_below_every_cutoff(self):
        """top-k is a rank cutoff, so a tie straddling position k is ambiguous.

        float32 does tie a handful of probabilities far down the 4096-wide tail.
        That is harmless as long as the ties sit below everything the configs
        keep: the leading `_TIE_FREE_PREFIX` probabilities are distinct, and
        `test_matches_reference` pins that no config keeps more than that many.
        """
        probs = np.asarray(_make_args(_make_logits(), 0.6, _VOCAB, 1.0, 0.0)[1])
        for row in probs:
            head = np.sort(row)[::-1][:_TIE_FREE_PREFIX]
            self.assertEqual(len(np.unique(head)), _TIE_FREE_PREFIX)

    def test_matches_reference(self):
        logits = _make_logits()
        for name, temperature, top_k, top_p, min_p in _CONFIGS:
            with self.subTest(config=name):
                got = _sampled_probs(_make_args(logits, temperature, top_k, top_p, min_p))
                self.assertLessEqual(int((got > 0.0).sum(axis=-1).max()), _TIE_FREE_PREFIX)
                tv = _total_variation(
                    got, _reference_probs(logits, temperature, top_k, top_p, min_p)
                )
                self.assertLess(
                    tv, 1e-5, f"sampler disagrees with the reference on {name!r}: TV {tv:.3e}"
                )

    def test_keeps_the_tokens_the_reference_keeps(self):
        """A distribution match could in principle hide a swapped tail; pin the support too."""
        logits = _make_logits()
        for name, temperature, top_k, top_p, min_p in _CONFIGS:
            with self.subTest(config=name):
                got = _sampled_probs(_make_args(logits, temperature, top_k, top_p, min_p)) > 0.0
                want = _reference_probs(logits, temperature, top_k, top_p, min_p) > 0.0
                np.testing.assert_array_equal(got, want, f"support differs on {name!r}")

    def test_temperature_is_applied_before_top_p(self):
        """Temperature reshapes the distribution, so it has to move the nucleus.

        Applying it after the masks instead makes the kept set identical at
        every temperature, which is what this asserts against.
        """
        logits = _make_logits()
        kept = {
            t: _capture(_make_args(logits, t, _VOCAB, 0.90, 0.0)) > _KEPT_LOGIT_FLOOR
            for t in (0.5, 1.0, 2.0)
        }
        self.assertLess(int(kept[0.5].sum()), int(kept[1.0].sum()))
        self.assertLess(int(kept[1.0].sum()), int(kept[2.0].sum()))

    def test_top_p_measures_the_unmasked_distribution(self):
        """top-p has to see the real distribution, not the top-k-renormalized one.

        `topp_mask` softmaxes its input, so running it after `topk_mask` would
        inflate every prefix mass by 1/mass(top-k): the top_p cutoff is reached
        earlier and the nucleus comes out strictly narrower -- for this fixture,
        7-38 tokens per row instead of 75-200.
        """
        logits = _make_logits()
        got = _sampled_probs(_make_args(logits, 1.0, 200, 0.6, 0.0)) > 0.0
        want = _reference_probs(logits, 1.0, 200, 0.6, 0.0) > 0.0
        np.testing.assert_array_equal(got, want)
        self.assertGreater(int(got.sum(axis=-1).min()), 38)

    def test_min_p_rejects_in_logit_space(self):
        """min_p runs on logits here, so it must use the sentinel, not 0.0.

        0.0 is an ordinary logit and sits far above the -1e12 the top-k/top-p
        masks write, so filling with it resurrects rejected tokens.
        """
        args = _make_args(_make_logits(), 1.0, _VOCAB, 1.0, 0.30)
        rejected = np.asarray(_capture(args) <= _KEPT_LOGIT_FLOOR)
        self.assertTrue(rejected.any(), "min_p=0.30 should reject something")
        # Rejected entries carry exactly zero probability, not merely a small one.
        self.assertTrue(np.all(_sampled_probs(args)[rejected] == 0.0))

    def test_min_p_zero_rejects_nothing(self):
        """min_p=0 is the disabled encoding; log(0) = -inf must not mask or NaN."""
        logits = _make_logits()
        filtered = sampler_mod._apply_min_p_filter((logits, jnp.zeros((_BATCH,), jnp.float32)))
        self.assertFalse(bool(jnp.isnan(filtered).any()))
        np.testing.assert_array_equal(np.asarray(filtered), np.asarray(logits))


if __name__ == "__main__":
    unittest.main()
