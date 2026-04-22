"""Multi-head Latent Attention (MLA) layer — factored absorb path.

Implements RFC §3.9: runs the absorb path as two factored matmuls on each
side of the core attention kernel, so no matmul ever sees the wider fused
product (FLOP savings from §3.3). The weight shape on disk stays compatible
with upstream DeepSeek (``kv_b_proj`` is one matrix); we split its rows
inside this module into the per-head ``W_UK`` / ``W_UV`` factors and never
run it fused at inference.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.layers.embeddings import get_rope
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch


class MLAAttention(nnx.Module):
    """MLA attention (absorb path).

    Forward data flow (RFC §3.9):
      Q path:  hidden -> q_a_proj -> q_a_layernorm -> q_b_proj -> (q_nope, q_pe)
               q_nope -[W_UK]-> ql_nope   (Dk bottleneck, §3.3)
               q_pe -> RoPE
      K path:  hidden -> kv_a_proj -> split(c_kv_raw, k_pe_raw)
               c_kv_raw -> kv_a_layernorm -> c_kv
               k_pe_raw -> RoPE -> k_pe
      Kernel:  mla_backend(ql_nope, q_pe, c_kv, k_pe) -> o_latent [T, N, R]
      O path:  o_latent -[W_UV]-> o_v [T, N, Dv]  (R bottleneck, §3.3)
               o_v -[o_proj]-> out [T, hidden]

    ``kv_b_proj`` from the original HF checkpoint is kept as one parameter so
    weight loading stays unchanged; at forward time its rows are reshaped to
    the per-head ``W_UK`` / ``W_UV`` factors (there is no matmul through the
    full ``kv_b_proj`` — the split is free).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, Any] | None = None,
        rope_interleave: bool = True,
        max_position_embeddings: int = 163840,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        super().__init__()

        self.mesh = mesh
        self.layer_id = layer_id
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        # Attention scaling — caller can post-multiply by YaRN mscale**2.
        self.scaling = self.qk_head_dim**-0.5

        if q_lora_rank is None:
            self.q_proj = LinearBase(
                hidden_size,
                num_heads * self.qk_head_dim,
                mesh,
                use_bias=False,
                params_dtype=dtype,
                kernel_axes=(None, "tensor"),
                scope_name="q_proj",
            )
        else:
            self.q_a_proj = LinearBase(
                hidden_size,
                q_lora_rank,
                mesh,
                use_bias=False,
                params_dtype=dtype,
                kernel_axes=(None, None),
                scope_name="q_a_proj",
            )
            self.q_a_layernorm = RMSNorm(q_lora_rank, param_dtype=jnp.float32, dtype=dtype)
            self.q_b_proj = LinearBase(
                q_lora_rank,
                num_heads * self.qk_head_dim,
                mesh,
                use_bias=False,
                params_dtype=dtype,
                kernel_axes=(None, "tensor"),
                scope_name="q_b_proj",
            )

        self.kv_a_proj = LinearBase(
            hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, None),
            scope_name="kv_a_proj",
        )
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, param_dtype=jnp.float32, dtype=dtype)
        # kv_b_proj is kept for weight loading compatibility; at runtime we
        # view its rows as W_UK and W_UV per head (no matmul through it).
        self.kv_b_proj = LinearBase(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="kv_b_proj",
        )

        self.o_proj = LinearBase(
            num_heads * v_head_dim,
            hidden_size,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=("tensor", None),
            scope_name="o_proj",
        )

        self.rotary_emb = get_rope(
            head_size=qk_rope_head_dim,
            rotary_dim=qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=int(rope_theta),
            is_neox_style=not rope_interleave,
            rope_scaling=rope_scaling,
            dtype=dtype,
        )

    def _get_w_uk_w_uv(self) -> tuple[jax.Array, jax.Array]:
        """View ``kv_b_proj`` weight as separate per-head ``W_UK`` / ``W_UV``.

        ``kv_b_proj`` is logically ``[kv_lora_rank, N*(Dk + Dv)]`` so reshaping
        to ``[R, N, Dk+Dv]`` and slicing the last axis gives the factored
        per-head weights — no matmul through the full fused form (RFC §3.3).

        Under static FP8 quantization ``kv_b_proj`` is a ``QuantizedLinear``
        that stores ``weight_q`` transposed (``[out, in]``) alongside a
        block-wise ``weight_scale``; dequantize on the fly so the two factored
        einsums run in the layer's compute dtype.
        """
        compute_dtype = self.kv_b_proj.params_dtype
        if hasattr(self.kv_b_proj, "weight"):
            # LinearBase: [in, out] = [R, N*(Dk+Dv)]
            w = self.kv_b_proj.weight.value.astype(compute_dtype)
        else:
            # QuantizedLinear: weight_q [out, in], weight_scale one of
            #   [in_blocks, 1, out] (block-wise) or [out] (per-channel).
            w_q = self.kv_b_proj.weight_q.value.astype(compute_dtype)
            scale = self.kv_b_proj.weight_scale.value.astype(compute_dtype)
            block = self.kv_b_proj.weight_block_size
            if scale.ndim == 3:
                # [in_blocks, 1, out]; expand to [in, out] and multiply.
                in_blocks, _, n_out = scale.shape
                block_k = block[1] if block is not None else max(1, self.kv_lora_rank // in_blocks)
                expanded = jnp.repeat(scale[:, 0, :], block_k, axis=0)[: self.kv_lora_rank, :]
                # w_q is [out, in]; transpose to [in, out] before elementwise multiply.
                w = w_q.T * expanded
            else:
                # Per-channel scale [out]: broadcast on last axis.
                w = w_q.T * scale[None, :]
        w = w.reshape(self.kv_lora_rank, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        w_uk = w[:, :, : self.qk_nope_head_dim]  # [R, N, Dk]
        w_uv = w[:, :, self.qk_nope_head_dim :]  # [R, N, Dv]
        return w_uk, w_uv

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, jax.Array]:
        # ----- Q path -----
        if self.q_lora_rank is None:
            q_full, _ = self.q_proj(hidden_states)
        else:
            q_compressed, _ = self.q_a_proj(hidden_states)
            q_compressed = self.q_a_layernorm(q_compressed)
            q_full, _ = self.q_b_proj(q_compressed)
        q_full = q_full.reshape(-1, self.num_heads, self.qk_head_dim)
        q_nope = q_full[:, :, : self.qk_nope_head_dim]  # [T, N, Dk]
        q_pe = q_full[:, :, self.qk_nope_head_dim :]  # [T, N, Dr_raw]

        # ----- KV path -----
        kv_a_out, _ = self.kv_a_proj(hidden_states)
        c_kv = self.kv_a_layernorm(kv_a_out[:, : self.kv_lora_rank])  # [T, R]
        k_pe_raw = kv_a_out[:, self.kv_lora_rank :]  # [T, Dr_raw]

        # ----- RoPE (on q_pe and k_pe, not on c_kv / q_nope) -----
        # rotary_emb takes q: [..., H, D], k: [..., 1, D] and returns them rotated.
        k_pe_for_rope = k_pe_raw[:, None, :]  # [T, 1, Dr_raw]
        q_pe_rot, k_pe_rot = self.rotary_emb(positions, q_pe, k_pe_for_rope)
        k_pe = k_pe_rot.reshape(-1, self.qk_rope_head_dim)  # [T, Dr_raw]

        # ----- Absorb W_UK onto Q: q_nope -> ql_nope -----
        # ql_nope[t, n, r] = sum_d q_nope[t, n, d] * W_UK[r, n, d]
        w_uk, w_uv = self._get_w_uk_w_uv()
        ql_nope = jnp.einsum("tnd,rnd->tnr", q_nope, w_uk, preferred_element_type=q_nope.dtype)

        # ----- Core MLA attention -----
        o_latent, kv_fused = forward_batch.attn_backend(
            ql_nope,
            q_pe_rot,
            c_kv,
            k_pe,
            layer=self,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
            sm_scale=self.scaling,
        )  # o_latent: [T, N, R]

        # ----- Absorb W_UV onto output: o_latent -> o_v -----
        o_v = jnp.einsum("tnr,rnv->tnv", o_latent, w_uv, preferred_element_type=o_latent.dtype)
        attn_output = o_v.reshape(-1, self.num_heads * self.v_head_dim)

        output, _ = self.o_proj(attn_output)
        return output, kv_fused
