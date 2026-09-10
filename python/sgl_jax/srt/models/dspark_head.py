from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.linear import LinearBase


class VanillaMarkovHead(nnx.Module):
    """Low-rank token-to-token correction used by DSpark stage1."""

    def __init__(
        self,
        *,
        vocab_size: int,
        markov_rank: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        if int(markov_rank) <= 0:
            raise ValueError(f"VanillaMarkovHead requires markov_rank > 0, got {markov_rank}.")
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        self.mesh = mesh
        self.markov_w1 = nnx.Param(
            jax.random.normal(
                jax.random.PRNGKey(0),
                (self.vocab_size, self.markov_rank),
                dtype=dtype,
                out_sharding=P(None, None),
            )
        )
        self.markov_w2 = LinearBase(
            input_size=self.markov_rank,
            output_size=self.vocab_size,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="markov_w2",
        )

    def get_prev_embeddings(self, token_ids: jax.Array) -> jax.Array:
        sharding = NamedSharding(self.mesh, P("data", None))
        return self.markov_w1.value.at[token_ids.astype(jnp.int32)].get(out_sharding=sharding)

    def apply_step_logits(
        self,
        base_logits: jax.Array,
        token_ids: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        markov_embedding = self.get_prev_embeddings(token_ids)
        bias, _ = self.markov_w2(markov_embedding)
        return base_logits + bias, markov_embedding

    def sample_block_tokens(
        self, base_logits: jax.Array, first_prev_tokens: jax.Array
    ) -> jax.Array:
        """Greedily sample each correction using the preceding sampled token."""
        if base_logits.ndim != 3 or base_logits.shape[-1] != self.vocab_size:
            raise ValueError("Markov logits must have shape [batch, proposals, vocab_size].")
        if first_prev_tokens.shape != base_logits.shape[:1]:
            raise ValueError("Markov seed tokens must have shape [batch].")
        prev_tokens = first_prev_tokens.astype(jnp.int32)
        tokens = []
        for step in range(base_logits.shape[1]):
            logits, _ = self.apply_step_logits(base_logits[:, step, :], prev_tokens)
            prev_tokens = jnp.argmax(logits, axis=-1).astype(jnp.int32)
            tokens.append(prev_tokens)
        return jnp.stack(tokens, axis=1)
