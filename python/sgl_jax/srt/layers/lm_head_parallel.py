"""LM-head vocabulary parallelism independent of attention's data axis."""

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


def weight_spec(enable_dp_lm_head: bool) -> P:
    return P("tensor", None) if enable_dp_lm_head else P(("data", "tensor"), None)


def prepare_weight(weight: jax.Array, mesh: jax.sharding.Mesh, enable_dp_lm_head: bool):
    """Pad vocabulary if necessary, then place weights in their persistent layout."""
    partitions = mesh.shape["tensor"]
    if not enable_dp_lm_head:
        partitions *= mesh.shape["data"]
    padding = -weight.shape[0] % partitions
    if padding:
        # Padding a sharded dimension can change shard boundaries.
        weight = jax.sharding.reshard(weight, NamedSharding(mesh, P(None, None)))
        weight = jnp.pad(weight, ((0, padding), (0, 0)))
    return jax.sharding.reshard(weight, NamedSharding(mesh, weight_spec(enable_dp_lm_head)))


def compute_lm_head_logits(
    hidden_states, weight, mesh, vocab_size, enable_dp_lm_head, *, preserve_vocab_sharding=False
):
    """Project hidden states, normally restoring the caller's DP token layout.

    Global TP gathers token rows, computes a vocabulary shard on each device,
    then redistributes vocabulary shards to their token owners. The compiler
    lowers this resharding to collectives; no host round-trip is involved.
    Fused greedy callers may keep the global vocabulary layout until argmax.
    """
    weight = prepare_weight(weight, mesh, enable_dp_lm_head)
    hidden_spec = P("data", None) if enable_dp_lm_head else P(None, None)
    logits_spec = P("data", "tensor") if enable_dp_lm_head else P(None, ("data", "tensor"))
    hidden_states = jax.sharding.reshard(hidden_states, NamedSharding(mesh, hidden_spec))
    logits = jnp.dot(hidden_states, weight.T, out_sharding=NamedSharding(mesh, logits_spec))
    if (
        preserve_vocab_sharding
        and not enable_dp_lm_head
        and vocab_size % (mesh.shape["data"] * mesh.shape["tensor"]) == 0
    ):
        # Greedy callers can reduce the vocabulary before redistributing rows.
        # Keep padded vocabularies on the existing trim-before-sampling path.
        return logits[:, :vocab_size]
    # Restore the caller's DP layout before trimming. A vocabulary that cannot
    # be split over attention TP stays replicated within each DP group.
    output_spec = P("data", "tensor") if vocab_size % mesh.shape["tensor"] == 0 else P("data", None)
    logits = jax.sharding.reshard(logits, NamedSharding(mesh, output_spec))
    return logits[:, :vocab_size]


def argmax_with_dp_sharding(logits):
    """Reduce vocabulary shards first, then redistribute only the token IDs."""
    token_ids = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    sharding = jax.typeof(logits).sharding
    if isinstance(sharding, NamedSharding) and "data" in sharding.mesh.axis_names:
        token_ids = jax.sharding.reshard(token_ids, NamedSharding(sharding.mesh, P("data")))
    return token_ids
