"""A1 -- DeepSeek-V4's grouped output projection (``wo_a`` / ``wo_b``).

V4's attention output does not go through a single ``o_proj``. It goes through an
**inverse RoPE, then a per-group LoRA down-projection, then one shared
up-projection**:

    x   : [T, n_heads, head_dim]        attention output
    x   = inverse_rope(x, positions)    # trailing rope_head_dim, GPT-J interleaved
    x   = x.reshape(T, G, H * head_dim) # G groups of H heads
    y   = einsum("tgd,gdr->tgr", x, wo_a)   # per-group, no cross-group mixing
    out = y.reshape(T, G * R) @ wo_b.T      # [T, hidden]

Shapes for Flash 0731, read off the checkpoint rather than inferred::

    wo_a.weight  [8192, 4096]  = [G*R, H*head_dim] = [8*1024, 8*512]
    wo_b.weight  [4096, 8192]  = [hidden, G*R]

so ``o_lora_rank`` is 1024 **per group**, not 1024 in total, and ``o_groups=8``
partitions the 64 heads into 8 groups of 8. There is no cross-group mixing until
``wo_b``.

Sources: upstream vLLM `vllm/models/deepseek_v4/attention.py` builds ``wo_a`` with
``is_bmm=True`` and ``bmm_batch_size = n_local_groups``, i.e. a batched per-group
matmul, and comments the step as "Inverse-RoPE + wo_a + wo_b output projection".
tpu-inference's `kernels/experimental/deepseek_v4/o_projection.py` states the same
contraction as ``einsum("tgd,dgr->tgr", ...)`` and fuses the inverse RoPE into it.

DeepSeek-V3's plain ``o_proj`` cannot be reused for any of this, which is why
nothing in this repo covered it.

This module prepares grouped weights and wraps the fused inverse-RoPE + wo_a
kernel. The model owns the final wo_b projection.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

__all__ = ["group_wo_a"]


def group_wo_a(wo_a, *, num_groups: int, out_sharding=None):
    """Reshape checkpoint ``wo_a`` ``[G*R, D]`` into ``[G, D, R]``.

    The checkpoint's output index is ``g * R + r`` -- group-major -- which is what
    lets the reshape be a plain view plus one transpose. Getting this backwards
    (``[R, G, D]``) silently mixes groups and still produces the right *shape*, so it
    is worth having in one named place with a test.
    """
    wo_a = jnp.asarray(wo_a)
    if wo_a.ndim != 2:
        raise ValueError(f"wo_a must be 2-D [G*R, D], got {wo_a.shape}")
    out_features, reduction = wo_a.shape
    if out_features % num_groups:
        raise ValueError(
            f"wo_a output width {out_features} is not divisible by o_groups {num_groups}"
        )
    lora_rank = out_features // num_groups
    grouped = jax.lax.reshape(wo_a, (num_groups, lora_rank, reduction), out_sharding=out_sharding)
    return jnp.transpose(grouped, (0, 2, 1))


def use_fused_wo_a() -> bool:
    """``DSV4_FUSED_WO_A=1``: inverse RoPE + grouped wo_a as one Pallas kernel per layer."""
    return os.environ.get("DSV4_FUSED_WO_A", "1") == "1"


def fuse_wo_a_weights(grouped, *, mesh=None):
    """``[G, D, R]`` grouped wo_a -> the kernel's ``[D, G*R]`` (column ``g*R + r``)."""
    grouped = jnp.asarray(grouped)
    num_groups, reduction, lora_rank = grouped.shape
    kwargs = {}
    if mesh is not None:
        kwargs["out_sharding"] = NamedSharding(mesh, P(None, "tensor"))
    return jax.lax.reshape(
        jnp.transpose(grouped, (1, 0, 2)), (reduction, num_groups * lora_rank), **kwargs
    )


def fused_wo_a_projection(
    attn_out, cos, sin, wo_a_fused, *, mesh, rope_head_dim: int, dtype, interpret: bool = False
):
    """``[T, H, D]`` attention output -> ``[T, G*R]``: inverse partial RoPE and the grouped
    wo_a contraction in one kernel per device (heads and wo_a columns follow the
    ``tensor`` axis; each device owns whole groups of eight heads)."""
    from sgl_jax.srt.kernels.dsv4.wo_a_projection import widen_cos_sin, wo_a_projection

    cos_sin = widen_cos_sin(cos, sin, rope_head_dim=rope_head_dim, inverse=True)

    def local(x_, w_, cs_):
        return wo_a_projection(x_, w_, cs_, out_dtype=dtype, interpret=interpret)

    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(P("data", "tensor", None), P(None, "tensor"), P("data", None)),
        out_specs=P("data", "tensor"),
        check_vma=False,
    )(jnp.asarray(attn_out).astype(jnp.bfloat16), wo_a_fused, cos_sin)
