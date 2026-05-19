"""Qwen3-5 Gated DeltaNet linear-attention layer.

Shape-correct port of HuggingFace's ``Qwen3_5GatedDeltaNet``. The HF
checkpoint has four separate projections (a key difference from
Qwen3-Next, which fuses everything into ``in_proj_qkvz``)::

    in_proj_qkv:   [hidden, 2*key_dim + value_dim]   ([Q | K | V] block-concat)
    in_proj_z:     [hidden, value_dim]
    in_proj_b:     [hidden, num_v_heads]
    in_proj_a:     [hidden, num_v_heads]
    conv1d.weight: [conv_dim, kernel_size]
    A_log:         [num_v_heads]
    dt_bias:       [num_v_heads]
    norm.weight:   [head_v_dim]
    out_proj:      [value_dim, hidden]

We use :class:`MergedColumnParallelLinear` for ``in_proj_qkv`` so the
single GEMM uses TPU's MXU better than three smaller ones. Q/K/V are
declared as three components with sizes ``[key_dim, key_dim, value_dim]``;
each component shards on its own head dim, and the per-device output is
already laid out as block-concat ``[q_tp | k_tp | v_tp]`` of size
``conv_dim/TP`` — exactly what the conv1d + recurrence pipeline want.
``z`` / ``b`` / ``a`` stay on their own ``LinearBase``s (small projections
where fusing wouldn't help much, and HF stores them separately anyway).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sgl_jax.srt.layers.linear import LinearBase, MergedColumnParallelLinear

if TYPE_CHECKING:
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch


class Qwen3_5GatedDeltaNet(nnx.Module):
    """Qwen3-5 Gated DeltaNet linear-attention layer.

    ``config`` exposes ``hidden_size``, ``linear_num_value_heads``,
    ``linear_num_key_heads``, ``linear_key_head_dim``,
    ``linear_value_head_dim``, ``linear_conv_kernel_dim``, and
    ``rms_norm_eps`` (matches HF's ``Qwen3_5TextConfig``).
    """

    def __init__(
        self,
        config: Any,
        layer_id: int,
        mamba_layer_id: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.mamba_layer_id = mamba_layer_id
        self.mesh = mesh
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.eps = config.rms_norm_eps

        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_dim = 2 * self.key_dim + self.value_dim

        # Fused Q/K/V projection: one big GEMM with per-shard block-concat
        # layout `[q_tp | k_tp | v_tp]` of size `conv_dim/TP` columns per
        # device. Components shard independently on their own head dim, so
        # GQA (where Q/K and V have different per-head sizes) doesn't cut
        # a shard mid-component.
        self.in_proj_qkv = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[self.key_dim, self.key_dim, self.value_dim],
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="in_proj_qkv",
        )
        # z / b / a remain separate: small projections, and HF stores them
        # as independent tensors. Each is column-parallel on its own head dim.
        self.in_proj_z = LinearBase(
            input_size=self.hidden_size,
            output_size=self.value_dim,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="in_proj_z",
        )
        self.in_proj_b = LinearBase(
            input_size=self.hidden_size,
            output_size=self.num_v_heads,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="in_proj_b",
        )
        self.in_proj_a = LinearBase(
            input_size=self.hidden_size,
            output_size=self.num_v_heads,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="in_proj_a",
        )

        # Backend owns conv1d_weight + A_log + dt_bias and runs the
        # conv+recurrence under shard_map.
        #
        # TODO(post-qwen3.5+kimi-linear): wrap this through
        # ``RadixLinearAttention`` so all linear-attention layers in the
        # repo share the same dispatch shape (matches the sglang attention
        # backend pattern). The interface contract today is shaped around
        # KDA's split-stream layout (three separate
        # ``q_conv1d`` / ``k_conv1d`` / ``v_conv1d`` ``LinearBase``
        # containers, ``(q, k, v, a, b)`` call signature), while GDN runs
        # one fused conv1d on ``mixed_qkv``. The reviewer flagged that
        # the unified shape should be designed after both qwen3.5 and
        # kimi-linear land, with both use cases visible — keeping the
        # fused-vs-split decision open until then.
        self.attention = GDNAttnBackend(
            num_k_heads=self.num_k_heads,
            num_v_heads=self.num_v_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_kernel_size=self.conv_kernel_size,
            mesh=mesh,
            dtype=dtype,
        )

        # Gated GemmaRMSNorm per-head along head_v_dim.
        self.rms_scale = nnx.Param(jnp.ones((self.head_v_dim,), dtype=jnp.float32))

        # Row-parallel output projection (all-reduce across "tensor").
        self.out_proj = LinearBase(
            input_size=self.value_dim,
            output_size=self.hidden_size,
            use_bias=False,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="out_proj",
        )

    # ----- helpers ----------------------------------------------------------

    def _rms_gate(self, core_attn_out: jax.Array, z: jax.Array) -> jax.Array:
        """``Qwen3NextRMSNormGated``: ``rmsnorm(core) · γ * silu(z)``.

        Both inputs are ``[T, num_v_heads, head_v_dim]``. Order matches HF's
        :class:`Qwen3NextRMSNormGated.forward` exactly so numerical-equivalence
        tests against the HF reference reproduce bit-for-bit::

            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(float32)
            variance      = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * rsqrt(variance + eps)         # fp32
            hidden_states = self.weight * hidden_states.to(input_dtype)   # round + promote
            hidden_states = hidden_states * F.silu(gate.to(float32))      # fp32
            return hidden_states.to(input_dtype)

        The non-obvious step is the round-trip cast to ``input_dtype``
        between the norm and the gamma-multiply: HF quantizes the normalized
        activations to bf16 first, then the ``fp32 * bf16`` multiply with
        ``γ`` (stored fp32) promotes back to fp32. Skipping that
        intermediate cast — what the previous version did — kept everything
        in fp32 and produced slightly more accurate values that didn't
        match HF in the last bf16 ulp.
        """
        input_dtype = core_attn_out.dtype
        x = core_attn_out.astype(jnp.float32)
        rms = jnp.sqrt((x * x).mean(axis=-1, keepdims=True) + self.eps)
        x = x / rms
        # Round to input_dtype before gamma-multiply; promote the product
        # back to fp32 via the fp32-stored weight so the silu-gate step
        # runs in fp32 (matches HF's `self.weight * hidden_states.to(input_dtype)`).
        gamma_f32 = self.rms_scale.value.astype(jnp.float32)
        x = gamma_f32 * x.astype(input_dtype).astype(jnp.float32)
        gated = x * jax.nn.silu(z.astype(jnp.float32))
        return gated.astype(input_dtype)

    # ----- forward ----------------------------------------------------------

    def __call__(
        self,
        hidden_states: jax.Array,  # [T, hidden]
        forward_batch: ForwardBatch,
        recurrent_state_pool,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Returns ``(output [T, hidden], new_conv [B, conv_dim, K-1], new_rec [B, H, K, V])``.

        The backend fetches this layer's ``(recurrent_state, conv_state)``
        from ``recurrent_state_pool`` via its base class's
        :meth:`get_layer_cache` helper (keyed on ``self.layer_id``) and
        returns per-request new states ready for
        ``RecurrentStatePool.write_layer``.

        Donation contract: ``recurrent_state_pool`` is read inside the
        backend and then only the per-request new slices are emitted. The
        outer jitted forward step should mark the pool buffers as
        ``donate_argnames=`` (or ``donate_argnums=``) on its ``jax.jit``
        so XLA can reuse their HBM for the next step's pool. Per-layer
        state copies across dozens of GDN layers are the dominant per-step
        HBM traffic on large models; without donation each layer pays a
        full pool copy. The caller must guarantee it does not read the
        donated pool buffers after the forward step returns, or JAX will
        raise ``Donated buffer has been deleted``.
        """
        T = hidden_states.shape[0]

        # Fused Q/K/V via a single GEMM. Per-device output is per-shard
        # block-concat `[q_tp | k_tp | v_tp]`, exactly what conv1d wants.
        mixed_qkv, _ = self.in_proj_qkv(hidden_states)  # [T, conv_dim]
        z, _ = self.in_proj_z(hidden_states)  # [T, value_dim]
        b, _ = self.in_proj_b(hidden_states)  # [T, num_v_heads]
        a, _ = self.in_proj_a(hidden_states)  # [T, num_v_heads]

        # Reshape z for the post-recurrence gate; sharding stays on n_v.
        z = jax.lax.reshape(
            z,
            (T, self.num_v_heads, self.head_v_dim),
            out_sharding=NamedSharding(self.mesh, P(None, "tensor", None)),
        )

        core_attn_out, new_conv, new_rec = self.attention(
            forward_batch,
            mixed_qkv,
            b,
            a,
            recurrent_state_pool,
            self.layer_id,
        )
        # core_attn_out: [T, num_v_heads, head_v_dim]

        gated = self._rms_gate(core_attn_out, z)
        gated_flat = jax.lax.reshape(
            gated,
            (T, self.num_v_heads * self.head_v_dim),
            out_sharding=NamedSharding(self.mesh, P(None, "tensor")),
        )
        output, _ = self.out_proj(gated_flat)
        return output, new_conv, new_rec
