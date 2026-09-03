import unittest
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers import sampler as sampler_mod
from sgl_jax.srt.layers.sampler import multinomial_with_seed


class TestGreedySamplingLogprobs(unittest.TestCase):
    def setUp(self):
        self.sampler = sampler_mod.Sampler()
        self.logits = jnp.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]])

    def test_skips_logprobs_when_not_requested(self):
        metadata = SimpleNamespace(return_logprob=False)
        with mock.patch.object(
            jax.nn, "log_softmax", side_effect=AssertionError("log_softmax must not run")
        ):
            token_ids, logprobs = self.sampler._greedy_sampling((self.logits, metadata, None))

        np.testing.assert_array_equal(token_ids, jnp.array([1, 2]))
        self.assertIsNone(logprobs)

    def test_computes_logprobs_when_requested(self):
        metadata = SimpleNamespace(return_logprob=True)
        token_ids, logprobs = self.sampler._greedy_sampling((self.logits, metadata, None))

        np.testing.assert_array_equal(token_ids, jnp.array([1, 2]))
        np.testing.assert_allclose(logprobs, jax.nn.log_softmax(self.logits, axis=-1))


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
# `--use-sort-for-toppk-minp` selects between two implementations of the same
# top-k / top-p / min-p semantics, and defaults to False -- so the mask path is
# what production runs. The two have to sample from the same distribution.
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

    `sampling_seeds=None` selects the plain `multinomial` in both paths, which is
    the function the tests below intercept.
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


def _capture(path_fn, args):
    """Run one sampling path and return the tensor it was about to sample from."""
    captured = {}

    def spy(operands):
        captured["inputs"] = operands[0]
        return jnp.zeros((_BATCH, 1), dtype=jnp.int32)

    with mock.patch.object(sampler_mod, "multinomial", new=spy):
        path_fn(args)
    return captured["inputs"]


def _mask_path_probs(args):
    """Vocab-ordered distribution the mask path samples from."""
    return np.asarray(
        jax.nn.softmax(
            _capture(sampler_mod.top_k_top_p_min_p_sampling_from_probs_jax_with_mask, args), axis=-1
        )
    )


def _sort_path_probs(args):
    """Vocab-ordered distribution the sort path samples from.

    The sort path hands `multinomial` a *descending-sorted* weight vector, so the
    capture has to be scattered back through the sort permutation, which is
    recomputed here. See `test_fixture_ties_sit_below_every_cutoff` for why that
    reconstruction is exact for this fixture.
    """
    probs = args[1]
    kept = _capture(sampler_mod.top_k_top_p_min_p_sampling_from_probs_jax_with_sort, args)
    order = jnp.argsort(probs, axis=-1)[:, ::-1]
    dense = jnp.zeros_like(probs).at[jnp.arange(_BATCH)[:, None], order].set(kept)
    return np.asarray(dense / dense.sum(axis=-1, keepdims=True))


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


class TestMaskPathMatchesSortPath(unittest.TestCase):
    """The two sampling paths must filter to the same distribution.

    Runs on whatever the default backend is. The sharding pitfall recorded in
    test_sampler_deterministic_cond.py needs an explicit mesh, which the
    unsharded fixture below never builds.
    """

    def test_fixture_ties_sit_below_every_cutoff(self):
        """Precondition for the sort-permutation reconstruction in `_sort_path_probs`.

        float32 does tie a handful of probabilities far down the 4096-wide tail,
        so the permutation is ambiguous there. That is harmless as long as the
        ties sit below everything the configs keep: the leading
        `_TIE_FREE_PREFIX` probabilities are distinct, and `test_paths_agree`
        pins that no config keeps more than that many tokens.
        """
        probs = np.asarray(_make_args(_make_logits(), 0.6, _VOCAB, 1.0, 0.0)[1])
        for row in probs:
            head = np.sort(row)[::-1][:_TIE_FREE_PREFIX]
            self.assertEqual(len(np.unique(head)), _TIE_FREE_PREFIX)

    def test_paths_agree(self):
        logits = _make_logits()
        for name, temperature, top_k, top_p, min_p in _CONFIGS:
            with self.subTest(config=name):
                args = _make_args(logits, temperature, top_k, top_p, min_p)
                mask_probs = _mask_path_probs(args)
                self.assertLessEqual(int((mask_probs > 0.0).sum(axis=-1).max()), _TIE_FREE_PREFIX)
                tv = _total_variation(mask_probs, _sort_path_probs(args))
                self.assertLess(
                    tv,
                    1e-5,
                    f"mask and sort paths disagree on {name!r}: total variation {tv:.3e}",
                )

    def test_paths_keep_the_same_tokens(self):
        """A distribution match could in principle hide a swapped tail; pin the support too."""
        logits = _make_logits()
        for name, temperature, top_k, top_p, min_p in _CONFIGS:
            with self.subTest(config=name):
                args = _make_args(logits, temperature, top_k, top_p, min_p)
                mask_kept = _mask_path_probs(args) > 0.0
                sort_kept = _sort_path_probs(args) > 0.0
                np.testing.assert_array_equal(mask_kept, sort_kept, f"support differs on {name!r}")

    def test_temperature_is_applied_before_top_p(self):
        """Temperature reshapes the distribution, so it has to move the nucleus.

        Applying it after the masks instead makes the kept set identical at
        every temperature, which is what this asserts against.
        """
        logits = _make_logits()
        kept = {
            t: _capture(
                sampler_mod.top_k_top_p_min_p_sampling_from_probs_jax_with_mask,
                _make_args(logits, t, _VOCAB, 0.90, 0.0),
            )
            > _KEPT_LOGIT_FLOOR
            for t in (0.5, 1.0, 2.0)
        }
        self.assertLess(int(kept[0.5].sum()), int(kept[1.0].sum()))
        self.assertLess(int(kept[1.0].sum()), int(kept[2.0].sum()))

    def test_top_p_measures_the_unmasked_distribution(self):
        """top-p has to see the real distribution, not the top-k-renormalized one.

        Checked against an explicit reference rather than against the sort path,
        so the two production paths agreeing on something wrong would still fail
        here. Running top-k first inflates every prefix mass by 1/mass(top-k), so
        the top_p cutoff is reached earlier and the nucleus comes out strictly
        narrower -- for this fixture, 75-200 tokens per row instead of 7-38.
        """
        top_k, top_p = 200, 0.6
        args = _make_args(_make_logits(), 1.0, top_k, top_p, 0.0)

        probs = np.asarray(args[1])
        order = np.argsort(probs, axis=-1)[:, ::-1]
        ordered = np.take_along_axis(probs, order, axis=-1)
        keep = (np.cumsum(ordered, axis=-1) - ordered) <= top_p  # top-p, full probs
        keep &= np.arange(_VOCAB)[None, :] < top_k  # intersected with top-k
        expected = np.zeros_like(keep)
        np.put_along_axis(expected, order, keep, axis=-1)

        np.testing.assert_array_equal(_mask_path_probs(args) > 0.0, expected)

    def test_min_p_rejects_in_logit_space(self):
        """min_p on the mask path must use the sentinel, not 0.0.

        0.0 is an ordinary logit and sits far above the -1e12 the top-k/top-p
        masks write, so filling with it resurrects rejected tokens.
        """
        logits = _make_logits()
        args = _make_args(logits, 1.0, _VOCAB, 1.0, 0.30)
        captured = _capture(sampler_mod.top_k_top_p_min_p_sampling_from_probs_jax_with_mask, args)
        rejected = np.asarray(captured <= _KEPT_LOGIT_FLOOR)
        self.assertTrue(rejected.any(), "min_p=0.30 should reject something")
        # Rejected entries carry exactly zero probability, not merely a small one.
        self.assertTrue(np.all(_mask_path_probs(args)[rejected] == 0.0))

    def test_min_p_zero_rejects_nothing(self):
        """min_p=0 is the disabled encoding; log(0) = -inf must not mask or NaN."""
        logits = _make_logits()
        filtered = sampler_mod._apply_min_p_filter(
            (logits, jnp.zeros((_BATCH,), dtype=jnp.float32), False)
        )
        self.assertFalse(bool(jnp.isnan(filtered).any()))
        np.testing.assert_array_equal(np.asarray(filtered), np.asarray(logits))


if __name__ == "__main__":
    unittest.main()
