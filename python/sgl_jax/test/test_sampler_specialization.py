import itertools
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
from sgl_jax.srt.layers.sampler import Sampler
from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata


class TestSamplerSpecialization(unittest.TestCase):
    def setUp(self):
        devices = np.asarray(jax.devices()[:1]).reshape(1, 1)
        self.mesh = Mesh(
            devices,
            ("data", "tensor"),
            axis_types=(jax.sharding.AxisType.Explicit,) * 2,
        )
        self.data = NamedSharding(self.mesh, P("data"))
        self.data_vocab = NamedSharding(self.mesh, P("data", "tensor"))
        self.data_mask = NamedSharding(self.mesh, P("data", None))
        self.vocab_size = 32

        logits = jnp.arange(self.vocab_size, dtype=jnp.float32).reshape(1, -1)
        self.logits_output = LogitsProcessorOutput(
            next_token_logits=jax.device_put(logits, self.data_vocab)
        )
        sampler = Sampler(rngs=nnx.Rngs(0), mesh=self.mesh)
        rng = jax.random.PRNGKey(123)

        @jax.jit
        def run(metadata):
            return sampler(
                self.logits_output,
                metadata,
                use_sort_for_toppk_minp=True,
                rng_override=rng,
            )

        self.run = run

    def _metadata(self, *, greedy, penalty, grammar, logprob):
        linear_penalty = np.zeros((1, self.vocab_size), dtype=np.float32)
        linear_penalty[0, 5] = -100
        linear_penalty[0, 31] = -100
        vocab_mask = np.array([[(1 << 4) | (1 << 5)]], dtype=np.int32)
        return SamplingMetadata(
            return_logprob=logprob,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            temperatures=jax.device_put(np.ones((1, 1), np.float32), self.data),
            top_ps=jax.device_put(np.ones(1, np.float32), self.data),
            top_ks=jax.device_put(np.ones(1, np.int32), self.data),
            min_ps=jax.device_put(np.zeros(1, np.float32), self.data),
            sampling_seeds=None,
            positions=jax.device_put(np.zeros(1, np.int32), self.data),
            is_all_greedy=greedy,
            need_min_p_sampling=False,
            do_penalties=penalty,
            linear_penalty=jax.device_put(linear_penalty, self.data_vocab),
            vocab_mask=jax.device_put(vocab_mask, self.data_mask),
            apply_vocab_mask=grammar,
        )

    def test_runtime_matrix_uses_four_graphs_and_preserves_results(self):
        for greedy, penalty, grammar, logprob in itertools.product((False, True), repeat=4):
            metadata = self._metadata(
                greedy=greedy,
                penalty=penalty,
                grammar=grammar,
                logprob=logprob,
            )
            token_ids, token_logprobs, logits_output = self.run(metadata)
            jax.block_until_ready(token_ids)

            expected_token = 4 if grammar and penalty else 5 if grammar else 30 if penalty else 31
            np.testing.assert_array_equal(np.asarray(token_ids), [expected_token])
            self.assertEqual(token_logprobs is not None, logprob)
            self.assertEqual(logits_output is not None, logprob)

        # Greedy and logprob select materially different output programs and
        # remain static. Penalty and grammar predicates stay inside those
        # programs, so their four combinations do not multiply the cache.
        self.assertEqual(self.run._cache_size(), 4)


if __name__ == "__main__":
    unittest.main()
