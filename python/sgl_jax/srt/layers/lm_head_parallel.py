"""LM-head vocabulary parallelism independent of attention's data axis."""

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P

from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead


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


def _standalone_heads(model):
    modules = list(nnx.iter_modules(model))
    tied = {
        id(m.embedding)
        for _, m in modules
        if isinstance(m, Embed) and not isinstance(m, ParallelLMHead)
    }
    return [
        (path, module)
        for path, module in modules
        if isinstance(module, ParallelLMHead) and id(module.embedding) not in tied
    ]


def lm_head_load_shardings(model, mesh, enable_dp_lm_head):
    """Load divisible standalone heads directly into vocabulary shards.

    Non-divisible heads load replicated and are padded once after loading.
    Tied embeddings keep their original placement for embedding lookup.
    """
    heads = _standalone_heads(model)
    if not heads:
        return {}
    partitions = mesh.shape["tensor"] * (1 if enable_dp_lm_head else mesh.shape["data"])
    return {
        ".".join(map(str, (*path, "embedding"))): (
            tuple(weight_spec(enable_dp_lm_head))
            if head.embedding.value.shape[0] % partitions == 0
            else (None, None)
        )
        for path, head in heads
    }


def configure_lm_heads(model, mesh, enable_dp_lm_head):
    # Local import avoids a cycle with LogitsProcessor's compute helper.
    from sgl_jax.srt.layers.logits_processor import LogitsProcessor

    for _, module in nnx.iter_modules(model):
        if isinstance(module, LogitsProcessor):
            module.enable_dp_lm_head = enable_dp_lm_head
    for _, head in _standalone_heads(model):
        # Draft heads supplied later by the target may still be abstract here.
        if isinstance(head.embedding.value, jax.ShapeDtypeStruct):
            continue
        head.embedding.value = prepare_weight(head.embedding.value, mesh, enable_dp_lm_head)
        head.kernel_axes = tuple(weight_spec(enable_dp_lm_head))


def compute_lm_head_logits(hidden_states, weight, mesh, vocab_size, enable_dp_lm_head):
    """Return token-sharded logits after a DP-local or global-TP projection.

    Global TP gathers token rows, computes a vocabulary shard on each device,
    then redistributes vocabulary shards to their token owners. The compiler
    lowers this resharding to collectives; no host round-trip is involved.
    """
    weight = prepare_weight(weight, mesh, enable_dp_lm_head)
    hidden_spec = P("data", None) if enable_dp_lm_head else P(None, None)
    logits_spec = P("data", "tensor") if enable_dp_lm_head else P(None, ("data", "tensor"))
    hidden_states = jax.sharding.reshard(hidden_states, NamedSharding(mesh, hidden_spec))
    logits = jnp.dot(hidden_states, weight.T, out_sharding=NamedSharding(mesh, logits_spec))
    # Restore the caller's DP layout before trimming. A vocabulary that cannot
    # be split over attention TP stays replicated within each DP group.
    output_spec = P("data", "tensor") if vocab_size % mesh.shape["tensor"] == 0 else P("data", None)
    logits = jax.sharding.reshard(logits, NamedSharding(mesh, output_spec))
    return logits[:, :vocab_size]
