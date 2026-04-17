"""BailingMoeV2_5LinearAttention: linear attention layer for BailingMoeV2.5.

Implements decode (fused_recurrent_simple_gla) and prefill (simple_gla_fwd)
forward passes with ALiBi slopes, optional QK RMSNorm, partial RoPE, and
sigmoid gating with GroupRMSNorm.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.fla.group_rmsnorm import GroupRMSNorm
from sgl_jax.srt.layers.attention.fla.linear_attention_backend import (
    LinearAttentionBackend,
    gather_from_packed,
    scatter_to_packed,
)
from sgl_jax.srt.layers.embeddings import get_rope
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

try:
    from sgl_jax.srt.kernels.simple_gla.simple_gla import (
        fused_recurrent_simple_gla,
        simple_gla_fwd,
    )
except ModuleNotFoundError:
    simple_gla_fwd = None
    fused_recurrent_simple_gla = None


class BailingMoeV2_5LinearAttention(nnx.Module):
    """Linear attention layer for BailingMoeV2.5.

    Implements the fused-QKV linear attention variant with:
    - ALiBi slopes (per-layer decay)
    - Optional QK RMSNorm
    - Partial rotary positional embeddings
    - GroupRMSNorm on the gate output
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        mesh,
        backend: LinearAttentionBackend,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_hidden_layers = config.num_hidden_layers
        self.mesh = mesh
        self.backend = backend

        # Fused QKV projection (column-parallel)
        self.qkv_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=3 * self.num_heads * self.head_dim,
            use_bias=getattr(config, "use_qkv_bias", False),
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="query_key_value",
        )

        # Gate projection (column-parallel, bias always False per HF reference)
        self.g_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.num_heads * self.head_dim,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="g_proj",
        )

        # Output projection (row-parallel)
        self.dense = LinearBase(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            use_bias=getattr(config, "use_bias", False),
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="dense",
        )

        # Optional Q/K RMSNorm
        if config.use_qk_norm:
            self.q_norm = RMSNorm(
                self.head_dim,
                epsilon=config.rms_norm_eps,
                param_dtype=dtype,
                scope_name="query_layernorm",
            )
            self.k_norm = RMSNorm(
                self.head_dim,
                epsilon=config.rms_norm_eps,
                param_dtype=dtype,
                scope_name="key_layernorm",
            )
        else:
            self.q_norm = None
            self.k_norm = None

        # Partial rotary positional embeddings
        rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=rotary_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            is_neox_style=True,
            dtype=dtype,
        )

        # GroupRMSNorm on gate output
        # param_dtype intentionally omitted (default float32): matches HF
        # BailingMoeV2_5GroupRMSNorm which stores weight as float32.
        self.g_norm = GroupRMSNorm(
            hidden_size=self.num_heads * self.head_dim,
            num_groups=getattr(config, "group_norm_size", 8),
            epsilon=config.rms_norm_eps,
            scope_name="g_norm",
        )

        # ALiBi slopes (stored as a JAX constant, not a trainable parameter)
        self.slope = self._compute_slope()

    def _compute_slope(self) -> jnp.ndarray:
        """Compute per-head ALiBi slopes with per-layer decay."""
        base_slopes = np.array(self.build_slope_tensor(self.num_heads), dtype=np.float32)
        slope = -base_slopes * (1 - (self.layer_idx - 1) / (self.num_hidden_layers - 1) + 1e-5)
        return jnp.array(slope, dtype=jnp.float32)

    @staticmethod
    def build_slope_tensor(num_heads: int) -> list[float]:
        """Compute ALiBi base slopes for the given number of heads.

        Matches the HuggingFace reference implementation exactly.
        """

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(num_heads).is_integer():
            return get_slopes_power_of_2(num_heads)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + BailingMoeV2_5LinearAttention.build_slope_tensor(2 * closest_power_of_2)[0::2][
                    : num_heads - closest_power_of_2
                ]
            )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch,
        recurrent_state: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Forward pass.

        Returns:
            output: [T, hidden_size] in input dtype.
            new_state: [N, H, K, V] always float32 (from kernel scan carry),
                regardless of input dtype.
        """
        T = hidden_states.shape[0]

        # 1. QKV projection
        qkv, _ = self.qkv_proj(hidden_states)
        qkv = jax.lax.reshape(
            qkv,
            (T, 3, self.num_heads, self.head_dim),
            out_sharding=NamedSharding(self.mesh, P(None, None, "tensor", None)),
        )
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each [T, H, K]

        # 2. Q/K RMSNorm (V skipped)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # 3. Partial RoPE
        q, k = self.rotary_emb(positions, q, k)

        # Cast recurrent state to float32 for numerical stability
        recurrent_state = recurrent_state.astype(jnp.float32)

        # 4. Kernel dispatch
        if forward_batch.forward_mode.is_decode():
            if fused_recurrent_simple_gla is None:
                raise ImportError("simple_gla kernel is required for linear attention decode")
            # Reshard recurrent_state along H so scan carry matches q/k/v sharding.
            recurrent_state = jax.sharding.reshard(
                recurrent_state,
                jax.sharding.NamedSharding(self.mesh, P(None, "tensor", None, None)),
            )
            # Decode: [T, H, K] -> [T, 1, H, K]
            q_d = q[:, None, :, :]
            k_d = k[:, None, :, :]
            v_d = v[:, None, :, :]
            output_d, new_state = fused_recurrent_simple_gla(
                q_d,
                k_d,
                v_d,
                g_gamma=self.slope,
                initial_state=recurrent_state,
                output_final_state=True,
                scale=None,
            )
            attn_output = output_d[:, 0, :, :]  # [T, H, V]

        elif forward_batch.forward_mode == ForwardMode.EXTEND:
            # Exact match: MIXED is converted to EXTEND by the scheduler before
            # reaching this point; DRAFT_EXTEND/TARGET_VERIFY (spec decoding) are
            # not supported for linear attention models.
            if simple_gla_fwd is None:
                raise ImportError("simple_gla kernel is required for linear attention prefill")
            # Prefill: scatter to chunk-aligned layout
            T_pb = self.backend.T_packed_bucket
            scatter_idx = forward_batch.linear_attn_metadata.scatter_idx
            cu_seqlens = forward_batch.linear_attn_metadata.cu_seqlens_dev

            # Reshard slope and h0 onto the mesh before shard_map.
            # self.slope is a plain jnp.array (not NNX, so not pre-sharded);
            # recurrent_state comes from the caller and may be replicated.
            slope_sm = jax.sharding.reshard(
                self.slope, jax.sharding.NamedSharding(self.mesh, P("tensor"))
            )
            h0_sm = jax.sharding.reshard(
                recurrent_state,
                jax.sharding.NamedSharding(self.mesh, P(None, "tensor", None, None)),
            )

            # Scatter is done inside shard_map so each device handles its local H
            # partition. GSPMD cannot auto-partition Mosaic kernels; shard_map
            # gives explicit per-device control.
            # cu_seqlens is passed as a replicated shard_map argument (P()).
            # All Python-level operations on cu_seqlens_dev inside the kernel chain
            # are ShardMapTracer-safe: identity checks (is None/is not None),
            # .shape / len() access, JAX ops (jnp.searchsorted, slice indexing),
            # and finally passing as a pallas_call input. No Python value extraction.
            def _prefill_fn(q_local, k_local, v_local, gamma, h0, scatter_idx_p, cu_seqlens_p):
                q_p = scatter_to_packed(q_local, scatter_idx_p, T_pb)
                k_p = scatter_to_packed(k_local, scatter_idx_p, T_pb)
                v_p = scatter_to_packed(v_local, scatter_idx_p, T_pb)
                return simple_gla_fwd(
                    q_p,
                    k_p,
                    v_p,
                    g_gamma=gamma,
                    h0=h0,
                    cu_seqlens_dev=cu_seqlens_p,
                    scale=None,
                    use_ht=True,
                    chunk_size=64,
                )

            output_packed, new_state = shard_map(
                _prefill_fn,
                mesh=self.mesh,
                in_specs=(
                    P(None, "tensor", None),  # q:     [T, H_local, K]
                    P(None, "tensor", None),  # k
                    P(None, "tensor", None),  # v
                    P("tensor"),  # slope: [H_local]
                    P(None, "tensor", None, None),  # h0:    [N, H_local, K, V]
                    P(),  # scatter_idx (replicated)
                    P(),  # cu_seqlens (replicated)
                ),
                out_specs=(
                    P(None, None, "tensor", None),  # output_packed [1, T_pb, H_local, V]
                    P(None, "tensor", None, None),  # new_state     [N, H_local, K, V]
                ),
                check_vma=False,
            )(q, k, v, slope_sm, h0_sm, scatter_idx, cu_seqlens)
            attn_output = gather_from_packed(output_packed, scatter_idx)  # [T, H, V]

        else:
            raise NotImplementedError(f"Unsupported forward mode: {forward_batch.forward_mode}")

        # 5. Reshape to [T, H*V]
        attn_output = attn_output.reshape(T, -1)

        # 6. Gating: GroupRMSNorm(attn_output) * sigmoid(g_proj(hidden_states))
        g, _ = self.g_proj(hidden_states)
        gate = jax.nn.sigmoid(g)
        attn_output = self.g_norm(attn_output) * gate

        # 7. Dense projection
        output, _ = self.dense(attn_output)

        return output, new_state
