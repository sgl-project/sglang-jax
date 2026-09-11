from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.models.dflash import DFlashDraftModel
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping


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


class DSparkDraftModel(DFlashDraftModel):
    """DSpark stage1: DFlash backbone with a token-conditioned Markov head."""

    def __init__(self, config, mesh, dtype=jnp.bfloat16):
        rank = int(getattr(config, "markov_rank", 0))
        if rank <= 0 or getattr(config, "markov_head_type", None) != "vanilla":
            raise ValueError("DSPARK stage1 requires markov_rank > 0 and a vanilla Markov head.")
        super().__init__(config=config, mesh=mesh, dtype=dtype)
        self.markov_head = VanillaMarkovHead(
            vocab_size=int(config.vocab_size), markov_rank=rank, mesh=mesh, dtype=dtype
        )

    def sample_block_tokens(self, base_logits, first_prev_tokens):
        return self.markov_head.sample_block_tokens(base_logits, first_prev_tokens)

    def _create_weight_mappings(self):
        mappings = DFlashDraftModel._create_weight_mappings(self)
        mappings.update(
            {
                "markov_head.markov_w1.weight": WeightMapping(
                    target_path="markov_head.markov_w1", sharding=(None, None), transpose=False
                ),
                "markov_head.markov_w2.weight": WeightMapping(
                    target_path="markov_head.markov_w2.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                ),
            }
        )
        return mappings

    def load_weights(self, model_config):
        import logging

        loader = WeightLoader(
            model=self, model_config=model_config, mesh=self.mesh, dtype=self.dtype
        )
        mappings = self._create_weight_mappings()
        if not loader.dummy_mode:
            keys = set(loader._scan_weight_info())
            missing = set(mappings) - keys
            if missing:
                raise ValueError(
                    f"DSPARK checkpoint is missing required weights: {sorted(missing)}."
                )
            if any(key.startswith("confidence_head.") for key in keys):
                logging.getLogger(__name__).info(
                    "DSPARK stage1 ignores confidence head weights; verifies all proposals."
                )
        loader.load_weights_from_safetensors(mappings)


class Qwen3DSparkModel(DSparkDraftModel):
    pass


EntryClass = Qwen3DSparkModel
