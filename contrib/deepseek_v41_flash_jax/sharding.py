"""Sharding layout for the V4.1 JAX engine (GSPMD annotations; mesh axes: dp = batch, tp = heads / FFN width).

Experts are sharded over BOTH axes (expert parallelism over all dp*tp devices). Attention heads and the grouped output
projection shard over tp (o_groups must be divisible by tp); caches shard over dp (batch).
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def make_mesh(devices, dp, tp):
    return Mesh(np.array(devices).reshape(dp, tp), ("dp", "tp"))


def param_spec(name: str) -> P:
    n = re.sub(r"^layers\.\d+\.", "", name)
    if n in (
        "attn.wq_b.weight",
        "attn.wo_a.weight",
        "ffn.shared_experts.w1.weight",
        "ffn.shared_experts.w3.weight",
    ):
        return P("tp", None)
    if n == "attn.attn_sink":
        return P("tp")
    if n in ("attn.wo_b.weight", "ffn.shared_experts.w2.weight"):
        return P(None, "tp")
    if n in (
        "ffn.experts.w1",
        "ffn.experts.w2",
        "ffn.experts.w3",
        "ffn.experts.w1_scale",
        "ffn.experts.w2_scale",
        "ffn.experts.w3_scale",
    ):
        return P(("dp", "tp"), None, None)
    if n in ("ffn.experts.w1_rowscale", "ffn.experts.w2_rowscale", "ffn.experts.w3_rowscale"):
        return P(("dp", "tp"), None)
    return P()


def shard_params(params: dict, mesh: Mesh) -> dict:
    return {k: jax.device_put(v, NamedSharding(mesh, param_spec(k))) for k, v in params.items()}


def shard_states(states, mesh: Mesh):
    spec = NamedSharding(mesh, P("dp", None, None))
    return jax.tree_util.tree_map(lambda x: jax.device_put(x, spec), states)


def to_kernel_layout(params: dict, mesh: Mesh, n_layers: int) -> dict:
    """One-time on-device conversion of the routed experts to the SGL-JAX fused_ep_moe_v2 layout (kernel mode):
    w1,w3 [E,F,D] -> [E,D,F]; w2 [E,D,F] -> [E,F,D]; row scales [E,X] -> [E,1,1,X]. Layer by layer, freeing the old arrays.
    """
    ep = ("dp", "tp")
    s3 = NamedSharding(mesh, P(ep, None, None))
    s4 = NamedSharding(mesh, P(ep, None, None, None))
    sw = jax.jit(lambda a: jnp.swapaxes(a, 1, 2), out_shardings=s3, donate_argnums=0)
    rs = jax.jit(lambda a: a[:, None, None, :], out_shardings=s4, donate_argnums=0)
    for l in range(n_layers):
        for w in ("w1", "w2", "w3"):
            params[f"layers.{l}.ffn.experts.{w}"] = sw(params[f"layers.{l}.ffn.experts.{w}"])
            params[f"layers.{l}.ffn.experts.{w}_rowscale"] = rs(
                params[f"layers.{l}.ffn.experts.{w}_rowscale"]
            )
    jax.block_until_ready(params)
    return params


def init_states_sharded(init_fn, mesh: Mesh):
    """Build the cache pytree directly in its sharded layout (jit with out_shardings): no single-device staging copy,
    which would not fit next to the weights at 1M context. init_fn() -> states pytree (called under jit).
    """
    spec = NamedSharding(mesh, P("dp", None, None))
    shapes = jax.eval_shape(init_fn)
    return jax.jit(init_fn, out_shardings=jax.tree_util.tree_map(lambda _: spec, shapes))()


def stack_experts(params: dict, n_layers: int, n_experts: int) -> dict:
    """Replace per-expert tensors 'layers.L.ffn.experts.E.wX.weight' by stacked 'layers.L.ffn.experts.wX' [E,N,K]."""
    import jax.numpy as jnp

    out = {k: v for k, v in params.items() if ".ffn.experts." not in k}
    for l in range(n_layers):
        for w in ("w1", "w2", "w3"):
            out[f"layers.{l}.ffn.experts.{w}"] = jnp.stack(
                [params[f"layers.{l}.ffn.experts.{e}.{w}.weight"] for e in range(n_experts)]
            )
    return out
