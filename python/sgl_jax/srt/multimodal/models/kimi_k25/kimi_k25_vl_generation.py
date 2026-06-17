import logging

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class KimiK25ForConditionalGeneration(nnx.Module):
    """Dummy Kimi K2.5 generation model for e2e pipeline testing."""

    def __init__(self, config=None, dtype=None, mesh=None):
        super().__init__()
        self.config = config
        self.mesh = mesh
        self.dtype = dtype or jnp.bfloat16
        self.vocab_size = getattr(config, "vocab_size", 163840)

    def load_weights(self, model_config):
        pass  # no weights to load

    def __call__(
        self,
        forward_batch: ForwardBatch,
        kv_cache: KVCache,
        logits_metadata: LogitsMetadata,
    ):
        num_seqs = forward_batch.seq_lens.shape[0]
        hidden_size = 1
        dummy_hidden = jax.sharding.reshard(
            jnp.zeros((num_seqs, hidden_size), dtype=jnp.bfloat16),
            NamedSharding(self.mesh, P("data", None)),
        )
        dummy_embed = jax.sharding.reshard(
            jnp.zeros((self.vocab_size, hidden_size), dtype=jnp.bfloat16),
            NamedSharding(self.mesh, P("tensor", None)),
        )
        dummy_logits = jnp.dot(
            dummy_hidden, dummy_embed.T, out_sharding=NamedSharding(self.mesh, P("data", "tensor"))
        )
        output = LogitsProcessorOutput(next_token_logits=dummy_logits)

        pool_updates = list(kv_cache.kv_buffer)

        return output, pool_updates, [], None
