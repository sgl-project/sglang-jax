"""BailingMoeV2_5LinearAttention: linear attention layer for BailingMoeV2.5.

Pure model layer responsible for projections, norms, and gating. Attention
dispatch (GLA kernel) is delegated to ``LightningAttnBackend`` via the
``RadixLightningAttention`` Radix wrapper, mirroring how ``KimiDeltaAttention``
delegates to ``KDAAttnBackend`` via ``RadixLinearAttention``.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.fla.group_rmsnorm import GroupRMSNorm
from sgl_jax.srt.layers.embeddings import get_rope
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.radix_lightning_attention import RadixLightningAttention


class BailingMoeV2_5LinearAttention(nnx.Module):
    """Linear attention layer for BailingMoeV2.5.

    Implements the fused-QKV linear attention variant with:
    - ALiBi slopes (per-layer decay) — owned by the backend, indexed by layer_id
    - Optional QK RMSNorm
    - Partial rotary positional embeddings
    - GroupRMSNorm on the gate output

    Attention dispatch goes through ``self.attn = RadixLightningAttention(...)``,
    which forwards to ``forward_batch.attn_backend`` (HybridLinearAttnBackend ->
    LightningAttnBackend).
    """

    def __init__(
        self,
        config,
        layer_id: int,
        mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.mesh = mesh

        self.qkv_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=3 * self.num_heads * self.head_dim,
            use_bias=getattr(config, "use_qkv_bias", False),
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="query_key_value",
        )

        self.g_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.num_heads * self.head_dim,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="g_proj",
        )

        self.dense = LinearBase(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            use_bias=getattr(config, "use_bias", False),
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="dense",
        )

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

        rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=rotary_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            is_neox_style=True,
            dtype=dtype,
        )

        self.g_norm = GroupRMSNorm(
            hidden_size=self.num_heads * self.head_dim,
            num_groups=getattr(config, "group_norm_size", 8),
            epsilon=config.rms_norm_eps,
            scope_name="g_norm",
        )

        self.attn = RadixLightningAttention(
            layer_id=layer_id,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch,
        recurrent_state_pool,
    ) -> tuple[jax.Array, tuple]:
        """Forward pass.

        Returns:
            output: [T, hidden_size] in input dtype.
            pool_updates: tuple of updated pool buffers for MemoryPools.replace_all.
        """
        T = hidden_states.shape[0]

        # 1. QKV projection (cast to float32 for norm/RoPE/kernel precision,
        #    matching GPU bailing_moe_linear.py:530)
        qkv, _ = self.qkv_proj(hidden_states)
        qkv = qkv.astype(jnp.float32)
        qkv = jax.lax.reshape(
            qkv,
            (T, 3, self.num_heads, self.head_dim),
            out_sharding=NamedSharding(self.mesh, P("data", None, "tensor", None)),
        )
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # 2. Q/K RMSNorm (V skipped)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # 3. Partial RoPE
        q, k = self.rotary_emb(positions, q, k)

        # 4. Delegate attention to backend via RadixLightningAttention dispatcher
        attn_output, pool_updates = self.attn(forward_batch, q, k, v, recurrent_state_pool)
        attn_output = attn_output.astype(hidden_states.dtype)

        # 5. Gating: GroupRMSNorm(attn_output) * sigmoid(g_proj(hidden_states))
        g, _ = self.g_proj(hidden_states)
        gate = jax.nn.sigmoid(g)
        attn_output = self.g_norm(attn_output) * gate

        # 6. Dense projection
        output, _ = self.dense(attn_output)

        return output, pool_updates
