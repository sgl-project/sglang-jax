"""Regression check for mutable grammar masks with cached sampler inputs."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata


def test_changed_grammar_mask_is_uploaded():
    mesh = Mesh(np.array(jax.devices()), ("data",))
    mask = np.zeros((4, 3), dtype=np.int32)
    metadata = SamplingMetadata(
        return_logprob=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        temperatures=jnp.ones((4, 1)),
        top_ps=None,
        top_ks=None,
        min_ps=None,
        sampling_seeds=None,
        positions=None,
    )
    metadata.update_vocab_mask(mask, mesh, 65)
    np.testing.assert_array_equal(np.asarray(metadata.vocab_mask), mask)
    mask[:] = 17
    metadata.update_vocab_mask(mask, mesh, 65)
    np.testing.assert_array_equal(np.asarray(metadata.vocab_mask), mask)
    assert bool(metadata.apply_vocab_mask)
    metadata.update_vocab_mask(None, mesh, 65)
    assert not bool(metadata.apply_vocab_mask)
    np.testing.assert_array_equal(np.asarray(metadata.vocab_mask), np.zeros((4, 3)))
