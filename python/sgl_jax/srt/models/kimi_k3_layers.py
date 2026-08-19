"""K3-specific layers that Kimi-Linear does not have: SITU activation and Attention Residuals.

Ported from the PyTorch reference at
``vllm_torchtpu/models/vllm/kimi_k3/layers.py`` (SituAndMul, AttentionResidual).

Kimi-Linear (already in ``models/kimi_linear.py``) supplies KDA, MLA and the MoE routing that K3
also uses. These two modules are the architectural delta:

* **SITU** — K3's gated activation, replacing SiLU in the MLP.
* **AttnRes** — K3's attention-residual: a *learned softmax-weighted sum* over the per-block
  residuals plus the running prefix sum, rather than a plain additive residual. This is the
  "improve how information flows across model depth" half of the K3 architecture.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

# NOTE: sgl_jax.srt.layers.* transitively imports the serving stack (zmq et al). The pure
# numerics below must stay importable without it so they can be unit-tested anywhere, so those
# imports are deferred into AttentionResidual.__init__.


def situ_and_mul(
    x: jax.Array,
    beta: float,
    linear_beta: float | None = None,
) -> jax.Array:
    """K3's SITU gated activation.

    Reference (PyTorch)::

        gate, up = x.chunk(2, dim=-1)
        gate = beta * tanh(gate / beta) * sigmoid(gate)
        if linear_beta is not None:
            up = linear_beta * tanh(up / linear_beta)
        return gate * up

    The ``beta * tanh(x / beta)`` term is a smooth soft-clip to ±beta; multiplying by
    ``sigmoid(gate)`` keeps the SiLU-like gating shape while bounding the magnitude. When
    ``linear_beta`` is set the *up* branch is soft-clipped too, which is what distinguishes SITU
    from a plain bounded-SiLU.
    """
    gate, up = jnp.split(x, 2, axis=-1)
    # compute the nonlinearity in fp32 regardless of input dtype: tanh/sigmoid of a bf16
    # argument loses enough mantissa to shift the product visibly at K3's hidden width.
    g32 = gate.astype(jnp.float32)
    gate_out = beta * jnp.tanh(g32 / beta) * jax.nn.sigmoid(g32)
    if linear_beta is not None:
        u32 = up.astype(jnp.float32)
        up_out = linear_beta * jnp.tanh(u32 / linear_beta)
    else:
        up_out = up.astype(jnp.float32)
    return (gate_out * up_out).astype(x.dtype)


class SituAndMul(nnx.Module):
    """Module wrapper around :func:`situ_and_mul`, mirroring the PyTorch layer's signature."""

    def __init__(self, beta: float, linear_beta: float | None = None):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def __call__(self, x: jax.Array) -> jax.Array:
        return situ_and_mul(x, self.beta, self.linear_beta)


def mla_output_gate(attn_output: jax.Array, gate: jax.Array) -> jax.Array:
    """K3's MLA output gate, applied to the attention output BEFORE ``o_proj``.

    Reference (``modeling_kimi_linear.py``, ``KimiMLAAttention.forward``)::

        g = self.g_proj(hidden_states).sigmoid()
        attn_output = attn_output * g

    ``o_proj`` is linear, but the gate is elementwise on its INPUT, so there is no equivalent
    place to apply it afterwards -- ordering here is load-bearing, not stylistic.

    The sigmoid is taken in fp32: it saturates, and a bf16 argument near the tails rounds to
    exactly 0 or 1, which silently drops or passes a whole head's contribution.
    """
    g = jax.nn.sigmoid(gate.astype(jnp.float32))
    return (attn_output.astype(jnp.float32) * g).astype(attn_output.dtype)


def attention_residual_apply(
    prefix_sum: jax.Array,
    block_residuals: jax.Array,
    norm_scale: jax.Array,
    proj_kernel: jax.Array,
    eps: float,
) -> jax.Array:
    """Pure-functional core of :class:`AttentionResidual` (weights passed explicitly).

    Kept separate so the numerics can be tested without constructing an nnx module or a device
    mesh, and so a parity oracle can drive exactly the same code path the module uses.
    """
    values = jnp.concatenate(
        (block_residuals, jnp.expand_dims(prefix_sum, axis=-2)), axis=-2
    )
    v32 = values.astype(jnp.float32)
    var = jnp.mean(jnp.square(v32), axis=-1, keepdims=True)
    normed = v32 * jax.lax.rsqrt(var + eps) * norm_scale.astype(jnp.float32)
    # HIGHEST precision is REQUIRED here, not an optimization. TPU's default einsum precision
    # is a bf16 multiply; these scores feed a softmax, so a ~1e-3 absolute error in the score
    # becomes a large *relative* error in the mixing weights. Measured on v7x: default precision
    # gives max_rel_err 3.7e-1 (37%) against the fp32 oracle, HIGHEST gives 1.9e-7. The
    # projection is hidden->1, so HIGHEST costs essentially nothing.
    scores = jnp.einsum(
        "...h,ho->...o",
        normed.astype(jnp.float32),
        proj_kernel.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
    )
    probabilities = jax.nn.softmax(scores.astype(jnp.float32), axis=-2)
    return jnp.sum(probabilities * v32, axis=-2).astype(values.dtype)


class AttentionResidual(nnx.Module):
    """K3's attention-residual weighted sum.

    Given ``block_residuals`` ``[..., n_blocks, hidden]`` and the running ``prefix_sum``
    ``[..., hidden]``, concatenates them into ``n_blocks + 1`` candidate vectors, scores each with
    ``proj(norm(v))`` (a learned hidden->1 projection), softmaxes **over the candidate axis**, and
    returns the weighted sum.

    Reference (PyTorch)::

        values = cat((block_residuals, prefix_sum.unsqueeze(-2)), dim=-2)
        scores, _ = self.proj(self.norm(values))
        probabilities = scores.float().softmax(dim=-2)
        return (probabilities * values.float()).sum(dim=-2).to(values.dtype)

    The softmax and the weighted sum run in fp32 in the reference; that is reproduced here
    exactly, because the sum is over a short axis of same-magnitude terms where bf16 rounding is
    the dominant error.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        scope_name: str = "attn_residual",
    ):
        super().__init__()
        self.eps = eps
        from sgl_jax.srt.layers.layernorm import RMSNorm
        from sgl_jax.srt.layers.linear import LinearBase

        self.norm = RMSNorm(
            hidden_size,
            epsilon=eps,
            param_dtype=jnp.float32,
            scope_name=f"{scope_name}.norm",
        )
        # hidden -> 1 scorer. Replicated, not sharded: the output is a single scalar per
        # candidate, so a tensor-parallel split would need an all-reduce for one number.
        self.proj = LinearBase(
            input_size=hidden_size,
            output_size=1,
            kernel_axes=(None, None),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name=f"{scope_name}.proj",
        )

    def __call__(
        self,
        prefix_sum: jax.Array,
        block_residuals: jax.Array,
    ) -> jax.Array:
        # Delegate to the pure core so the module and the tested path have identical numerics --
        # in particular the HIGHEST-precision scoring einsum, which self.proj() would not give.
        return attention_residual_apply(
            prefix_sum,
            block_residuals,
            self.norm.scale.value,
            self.proj.weight.value,   # LinearBase names its param `weight`, not `kernel`
            self.eps,
        )
