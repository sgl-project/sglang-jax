"""TPU arithmetic and ragged/EOS parity for the experimental limb hash."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from benchmark.kernels.ngram.bench_ngram_hash_limb import (
    _mul_words,
    _remainder,
    make_hash,
)
from sgl_jax.srt.layers.ngram_embedding import build_hash_params, compute_ngram_ids

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="TPU Pallas lowering")


def _params():
    return build_hash_params(
        ngram_size=3,
        heads_per_ngram=8,
        vocab_size=248320,
        ngram_vocab_size_base=20_000_000,
        eos_token_id=2,
    )


@pytest.mark.parametrize("multiplier", list(map(int, _params().multipliers)))
def test_wide_multiply(multiplier):
    rng = np.random.default_rng(10)
    tokens = rng.integers(0, 248320, (8, 128), np.uint32)
    tokens[0, :7] = [0, 1, 2, 65535, 65536, 65537, 248319]

    def kernel(x, lo, hi):
        lo[...], hi[...] = _mul_words(x[...], multiplier)

    output = pl.pallas_call(
        kernel, out_shape=(jax.ShapeDtypeStruct(tokens.shape, jnp.uint32),) * 2
    )(jnp.asarray(tokens))
    expected = tokens.astype(np.uint64) * np.uint64(multiplier)
    np.testing.assert_array_equal(
        np.asarray(output[0]), (expected & np.uint64(2**32 - 1)).astype(np.uint32)
    )
    np.testing.assert_array_equal(
        np.asarray(output[1]), (expected >> np.uint64(32)).astype(np.uint32)
    )


def test_remainder_at_multiples():
    primes = np.array(
        [3, 1009, 65521, 2**24 + 43, 20000003, 20000171, 2**25 - 3, 2**25 - 1],
        np.uint32,
    )
    divisors = np.broadcast_to(primes[:, None], (8, 128)).copy()
    numbers = np.empty((8, 128), np.uint32)
    for row, prime in enumerate(primes):
        p = int(prime)
        edge = [
            min(max(q * p + delta, 0), 128 * p - 1)
            for q in (0, 1, 2, 63, 64, 126, 127, 128)
            for delta in (-2, -1, 0, 1, 2)
        ]
        numbers[row] = np.resize(np.array(edge, np.uint32), 128)

    def kernel(n, p, result):
        result[...] = _remainder(n[...], p[...])

    got = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(numbers.shape, jnp.uint32))(
        jnp.asarray(numbers), jnp.asarray(divisors)
    )
    np.testing.assert_array_equal(np.asarray(got), numbers % divisors)


@pytest.mark.parametrize(
    "lengths,decode",
    [
        ([1], True),
        ([1] * 9, True),
        ([0, 1, 3, 0], False),
        ([1, 127, 0, 128, 1], False),
        ([2049, 2048, 2048, 2048], False),
    ],
)
def test_hash_ragged_eos_and_tail(lengths, decode):
    params = _params()
    rng = np.random.default_rng(11)
    count = sum(lengths)
    ids = rng.integers(0, 248320, count, np.int32)
    context = rng.integers(0, 248320, (len(lengths), 2), np.int32)
    ids[::3] = params.eos_token_id
    ids[-1] = 248319
    context[::2, 1] = params.eos_token_id
    cu = np.r_[0, np.cumsum(lengths)].astype(np.int32)
    want = compute_ngram_ids(ids, cu, context, params)
    got = make_hash(params, decode=decode)(*jax.device_put((ids, cu, context)))
    np.testing.assert_array_equal(np.asarray(got), want)
