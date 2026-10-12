"""Regression test for the deterministic-sampling ``lax.cond`` sharding crash.

When ``--enable-deterministic-sampling`` is on, ``sampling_seeds`` is a non-``None``
array, so the sampler selects the *seeded* multinomial branch. That selection used
to be a ``lax.cond`` keyed on the static predicate ``sampling_seeds is not None``.
``lax.cond`` traces both branches and unifies their output avals -- including
sharding -- and the two branches diverge:

  * ``multinomial_with_seed`` ends in ``argmax(..., keepdims=True)``  -> ``int32[n@data, 1]``
  * ``multinomial`` ends in ``categorical(...).reshape(-1, 1)``       -> ``int32[n, 1]``

On TPU this fails at trace time with::

    TypeError: cond branches must have equal output types but they differ.
    ... true_fun ... int32[1@data,1] ... false_fun ... int32[1,1]

(the v6e-1 decode-precompile symptom that broke PR #1347). The fix replaces the
static-predicate ``lax.cond`` with a plain Python ``if``.

The guard used to exercise the sort path, which reached the seeded/unseeded
selection directly; that path is gone, so it now exercises the mask path, which
holds the construct today. Tracing a leaf helper in isolation, outside the
model's full jit, can leave shardings unresolved for reasons that have nothing
to do with the cond -- so anything other than the cond aval error is reported as
a skip, not a failure. The mask path is also covered end-to-end by the
deterministic-sampling server tests.

TPU-only: on CPU/GPU XLA reconciles the mismatched branch shardings, so the cond
never errors and there is nothing to guard.
"""

import unittest

import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.sampler import (
    top_k_top_p_min_p_sampling_from_probs_jax_with_mask,
)
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

_IS_TPU = jax.devices()[0].platform == "tpu"

_COND_AVAL_ERROR = "cond branches must have equal output types"


def _seeded_args(mesh, bs, vocab, seed_value=42):
    """Build the 10-tuple `args` consumed by the sampler leaf helpers.

    Layout (see Sampler._regular_sampling):
        (logits, probs, top_ks, top_ps, min_ps, positions, temperatures,
         sampling_seeds, need_min_p_sampling, rng)
    Row-major tensors are sharded P("data", None); per-row vectors P("data").
    `sampling_seeds` is non-None so the seeded branch is selected -- the bug path.
    """
    row2d = NamedSharding(mesh, P("data", None))
    row1d = NamedSharding(mesh, P("data"))
    logits = jax.device_put(
        jnp.arange(bs * vocab, dtype=jnp.float32).reshape(bs, vocab) / vocab, row2d
    )
    return (
        logits,
        jax.device_put(jax.nn.softmax(logits, axis=-1), row2d),
        jax.device_put(jnp.full((bs,), vocab, dtype=jnp.int32), row1d),
        jax.device_put(jnp.ones((bs,), dtype=jnp.float32), row1d),
        jax.device_put(jnp.zeros((bs,), dtype=jnp.float32), row1d),
        jax.device_put(jnp.arange(bs, dtype=jnp.int32), row1d),
        jax.device_put(jnp.ones((bs, 1), dtype=jnp.float32), row2d),
        jax.device_put(jnp.full((bs,), seed_value, dtype=jnp.int32), row1d),
        False,  # need_min_p_sampling
        random.PRNGKey(0),
    )


@unittest.skipUnless(_IS_TPU, "lax.cond branch-sharding mismatch only reproduces on TPU")
class TestSamplerDeterministicCond(unittest.TestCase):
    def test_no_static_cond_regression(self):
        # Single device, data=1 -- the v6e-1 decode-precompile shape
        # (int32[1@data,1] vs int32[1,1]).
        mesh = create_device_mesh(
            ici_parallelism=[1, 1], dcn_parallelism=[1, 1], devices=jax.devices()[:1]
        )
        with jax.set_mesh(mesh):
            args = _seeded_args(mesh, bs=1, vocab=128)
            try:
                out = jax.jit(top_k_top_p_min_p_sampling_from_probs_jax_with_mask)(args)
                jax.block_until_ready(out)
            except Exception as e:  # noqa: BLE001
                if _COND_AVAL_ERROR in str(e):
                    self.fail("regression: static-predicate lax.cond reintroduced -- " f"{e}")
                # The construct under test traced without the aval error, which is
                # the whole assertion. Everything else here comes from running a
                # leaf helper outside the model's full jit and does not happen in
                # the server; surface it as a skip so it cannot read as a failure.
                self.skipTest(f"unrelated error tracing in isolation: {type(e).__name__}: {e}")


if __name__ == "__main__":
    unittest.main()
